"""Regression tests for board #31 — DB heartbeat during the cold-cache first load.

The bug: a mass-search row sat status='queued' for ~42 min while the worker
thread was ALIVE, loading 365-day 30Sec data. Two facts combined —

  (a) the fresh-start path only ever set 'running' in memory, never in the DB
      (the resume path in api/routers/mass_builder.py already flips it), and
  (b) the only periodic DB write lives inside _progress, gated on a 60s wall
      clock — but a cold-cache load fires ZERO progress callbacks for its whole
      duration, so that gate never ran.

Net: /progress (in-memory) looked healthy while the DB row and every external
monitor read a live worker as stuck at 'queued'.

The fix is gated by RORT_MASS_SEARCH_HEARTBEAT (default OFF). OFF is
byte-identical to the legacy path: no up-front flip, no heartbeat thread. ON
adds (1) an immediate status-only 'running' flip and (2) a daemon thread that
flushes the in-memory snapshot on a wall clock, sharing last_db_flush with
_progress so the periodic write budget is unchanged.

These tests stub run_mass_search and db.update_mass_search, so they run in CI
without a DB and without loading any market data. _HB_POLL_SECS/_HB_FLUSH_SECS
are shrunk to keep the suite sub-second.

Run: cd src && ../.venv/bin/python test_mass_search_heartbeat_31.py
"""

from __future__ import annotations

import os
import threading
import time
import unittest.mock as mock


def _setup_modules():
    """Import db + mass_builder with stubbed Supabase env so module-level
    constants don't blow up on missing keys."""
    os.environ.setdefault('SUPABASE_URL', 'http://stub.local')
    os.environ.setdefault('SUPABASE_SERVICE_ROLE_KEY', 'stub-service-role-key')
    os.environ.setdefault('SUPABASE_ANON_KEY', 'stub-anon-key')
    import db as db_mod
    import mass_builder as mb_mod
    return db_mod, mb_mod


DB, MB = _setup_modules()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class _Recorder:
    """Captures every update_mass_search(search_id, payload) call, in order."""

    def __init__(self, write_delay: float = 0.0):
        self.writes = []                      # [(search_id, payload), ...]
        self._lock = threading.Lock()
        self._write_delay = write_delay

    @staticmethod
    def _is_snapshot_flush(payload):
        # A heartbeat / _progress flush. NOT the terminal 'completed' write,
        # which also carries a (cleared) progress key.
        return payload.get('status') == 'running' and 'progress' in payload

    def __call__(self, search_id, payload):
        # Simulate DB round-trip latency on snapshot flushes only, so an
        # in-flight heartbeat write racing a terminal write is real.
        if self._write_delay and self._is_snapshot_flush(payload):
            time.sleep(self._write_delay)
        with self._lock:
            self.writes.append((str(search_id), dict(payload)))

    def snapshot(self):
        with self._lock:
            return list(self.writes)

    def statuses(self):
        return [p.get('status') for _, p in self.snapshot()]

    def snapshot_flushes(self):
        """Writes that carry a progress payload (heartbeat or _progress)."""
        return [p for _, p in self.snapshot()
                if p.get('status') == 'running' and 'progress' in p]

    def status_only_writes(self):
        return [p for _, p in self.snapshot()
                if set(p.keys()) == {'status'}]


class _Harness:
    """Drives start_mass_search_async with a stubbed run_mass_search."""

    def __init__(self, flag_on: bool, load_fn, poll=0.02, flush=0.08,
                 write_delay=0.0):
        self.flag_on = flag_on
        self.load_fn = load_fn          # (progress_callback) -> results list
        self.poll = poll
        self.flush = flush
        self.rec = _Recorder(write_delay=write_delay)
        self.progress_cb = None
        self._cb_ready = threading.Event()

    def _run_mass_search(self, config, progress_callback=None,
                         resume_checkpoint=None, checkpoint_callback=None,
                         partial_callback=None):
        self.progress_cb = progress_callback
        self._cb_ready.set()
        return (self.load_fn(progress_callback), {})

    def run(self, search_id, join_timeout=8.0):
        env = {'RORT_MASS_SEARCH_HEARTBEAT': '1' if self.flag_on else '0'}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(MB, 'run_mass_search', self._run_mass_search), \
                mock.patch.object(MB, '_HB_POLL_SECS', self.poll), \
                mock.patch.object(MB, '_HB_FLUSH_SECS', self.flush), \
                mock.patch.object(DB, 'update_mass_search', self.rec), \
                mock.patch.object(DB, 'get_current_user_id', lambda: None):
            thread = MB.start_mass_search_async(search_id, {'name': 'stub'})
            thread.join(timeout=join_timeout)
            assert not thread.is_alive(), "worker thread did not finish in time"
            # Give the heartbeat one poll interval to observe _hb_stop.
            time.sleep(self.poll * 3 + 0.05)
        return self.rec


def _hb_threads(search_id):
    return [t for t in threading.enumerate()
            if t.name == f"mass_hb_{search_id}" and t.is_alive()]


def _cleanup(search_id):
    with MB._search_lock:
        MB._active_searches.pop(str(search_id), None)


def _blocking_load(duration):
    """A cold-cache load: blocks, fires ZERO progress callbacks."""
    def _load(_cb):
        time.sleep(duration)
        return []
    return _load


# ---------------------------------------------------------------------------
# Flag OFF — legacy behavior preserved exactly (this is the bug, on purpose)
# ---------------------------------------------------------------------------
def test_flag_off_no_db_write_during_callbackless_load():
    sid = 'hb-off-1'
    rec = _Harness(flag_on=False, load_fn=_blocking_load(0.4)).run(sid)
    # Legacy: the ONLY write is the terminal 'completed' one. Nothing marks the
    # row 'running' during the whole load — that is exactly the #31 symptom.
    assert rec.statuses() == ['completed'], rec.snapshot()
    assert rec.status_only_writes() == [], rec.snapshot()
    _cleanup(sid)


def test_flag_off_starts_no_heartbeat_thread():
    sid = 'hb-off-2'
    _Harness(flag_on=False, load_fn=_blocking_load(0.3)).run(sid)
    assert _hb_threads(sid) == [], "flag OFF must not spawn a heartbeat thread"
    _cleanup(sid)


# ---------------------------------------------------------------------------
# Flag ON — part 1: immediate status flip
# ---------------------------------------------------------------------------
def test_flag_on_flips_row_to_running_up_front():
    sid = 'hb-on-1'
    rec = _Harness(flag_on=True, load_fn=_blocking_load(0.4)).run(sid)
    writes = rec.snapshot()
    assert writes, "expected at least the up-front flip"
    first = writes[0][1]
    # Status-column-only: no JSONB key, so db.update_mass_search skips the
    # config_data SELECT+merge. Cheapest possible liveness write.
    assert first == {'status': 'running'}, first
    _cleanup(sid)


def test_flip_lands_while_the_load_is_still_blocking():
    """The whole point: the row reads 'running' DURING the load, not after."""
    sid = 'hb-on-2'
    seen_during_load = {}

    def _load(_cb):
        time.sleep(0.25)
        seen_during_load['writes'] = len(h.rec.snapshot())
        time.sleep(0.25)
        return []

    h = _Harness(flag_on=True, load_fn=_load)
    h.run(sid)
    assert seen_during_load['writes'] >= 1, \
        "no DB write landed while the load was still running"
    _cleanup(sid)


# ---------------------------------------------------------------------------
# Flag ON — part 2: wall-clock heartbeat with no callbacks at all
# ---------------------------------------------------------------------------
def test_heartbeat_flushes_snapshot_during_callbackless_load():
    sid = 'hb-on-3'
    rec = _Harness(flag_on=True, load_fn=_blocking_load(0.5),
                   poll=0.02, flush=0.08).run(sid)
    flushes = rec.snapshot_flushes()
    assert len(flushes) >= 1, \
        f"heartbeat produced no snapshot flush during the load: {rec.snapshot()}"
    p = flushes[0]['progress']
    assert set(p) == {'current_step', 'total_steps', 'current_label',
                      'phase', 'phase_detail'}, p
    _cleanup(sid)


def test_heartbeat_carries_last_known_phase_from_memory():
    """One callback fires at the top of the load ('Loading SPY 30Sec data...'),
    then nothing for 42 min. The heartbeat re-publishes that last known phase,
    so the row says WHAT the worker is stuck on, not just that it is alive."""
    sid = 'hb-on-4'

    def _load(cb):
        cb(0, 10, 'Loading SPY 30Sec data...', phase='load',
           phase_detail='SPY 30Sec')
        time.sleep(0.5)
        return []

    rec = _Harness(flag_on=True, load_fn=_load, poll=0.02, flush=0.08).run(sid)
    flushes = rec.snapshot_flushes()
    assert flushes, f"no heartbeat flush: {rec.snapshot()}"
    p = flushes[0]['progress']
    assert p['phase'] == 'load', p
    assert p['phase_detail'] == 'SPY 30Sec', p
    assert p['current_label'] == 'Loading SPY 30Sec data...', p
    _cleanup(sid)


def test_heartbeat_respects_the_shared_flush_window():
    """Write budget is unchanged: with the window wider than the load, the
    heartbeat must NOT write at all beyond the single up-front flip."""
    sid = 'hb-on-5'
    rec = _Harness(flag_on=True, load_fn=_blocking_load(0.4),
                   poll=0.02, flush=30.0).run(sid)
    assert rec.snapshot_flushes() == [], \
        f"heartbeat wrote inside the throttle window: {rec.snapshot()}"
    assert rec.statuses() == ['running', 'completed'], rec.snapshot()
    _cleanup(sid)


# ---------------------------------------------------------------------------
# Lifecycle — the heartbeat must never outlive or outrun the worker
# ---------------------------------------------------------------------------
def test_heartbeat_thread_exits_after_worker_finishes():
    sid = 'hb-life-1'
    _Harness(flag_on=True, load_fn=_blocking_load(0.3), poll=0.02,
             flush=0.08).run(sid)
    assert _hb_threads(sid) == [], "heartbeat thread leaked past worker exit"
    _cleanup(sid)


def test_completed_is_the_last_write():
    sid = 'hb-life-2'
    rec = _Harness(flag_on=True, load_fn=_blocking_load(0.4), poll=0.02,
                   flush=0.08).run(sid)
    assert rec.statuses()[-1] == 'completed', rec.snapshot()
    _cleanup(sid)


def test_in_flight_heartbeat_cannot_resurrect_a_completed_row():
    """The race the write-lock exists for: a heartbeat write already in flight
    when the search finishes must not land AFTER the terminal status — a row
    stuck back at 'running' would be reaped or auto-resumed by the orphan
    sweeper, re-running a finished search."""
    sid = 'hb-race-1'
    # Snapshot flushes take 0.3s; the load ends 0.05s after one is issued, so
    # the heartbeat write is guaranteed to be in flight at completion time.
    h = _Harness(flag_on=True, load_fn=_blocking_load(0.35), poll=0.02,
                 flush=0.08, write_delay=0.3)
    rec = h.run(sid, join_timeout=15.0)
    assert rec.snapshot_flushes(), \
        f"race not exercised — no heartbeat flush was issued: {rec.snapshot()}"
    assert rec.statuses()[-1] == 'completed', \
        f"a heartbeat write landed after the terminal status: {rec.statuses()}"
    _cleanup(sid)


def test_cancelled_is_the_last_write():
    sid = 'hb-life-3'

    def _load(cb):
        # Poll-style load that keeps calling back, so the cancel is observed.
        for i in range(200):
            cb(i, 200, f'step {i}', phase='load')
            time.sleep(0.01)
        return []

    h = _Harness(flag_on=True, load_fn=_load, poll=0.02, flush=0.08)
    stopper = threading.Timer(0.3, lambda: MB.cancel_search(sid))
    stopper.start()
    rec = h.run(sid)
    stopper.cancel()
    assert rec.statuses()[-1] == 'cancelled', rec.statuses()
    assert _hb_threads(sid) == [], "heartbeat leaked past cancel"
    _cleanup(sid)


def test_failed_is_the_last_write():
    sid = 'hb-life-4'

    def _load(_cb):
        time.sleep(0.3)
        raise RuntimeError("boom")

    rec = _Harness(flag_on=True, load_fn=_load, poll=0.02, flush=0.08).run(sid)
    assert rec.statuses()[-1] == 'failed', rec.statuses()
    assert _hb_threads(sid) == [], "heartbeat leaked past failure"
    _cleanup(sid)


def test_heartbeat_stops_when_memory_status_leaves_running():
    """Defence in depth: even without _hb_stop, an in-memory terminal status
    ends the loop before the next write."""
    sid = 'hb-life-5'

    def _load(_cb):
        time.sleep(0.1)
        with MB._search_lock:
            MB._active_searches[sid]['status'] = 'completed'
        time.sleep(0.4)
        return []

    rec = _Harness(flag_on=True, load_fn=_load, poll=0.02, flush=0.05).run(sid)
    # Only the up-front flip + whatever landed in the first 0.1s; nothing after.
    flushes = rec.snapshot_flushes()
    assert len(flushes) <= 2, \
        f"heartbeat kept writing after the in-memory status left 'running': {flushes}"
    _cleanup(sid)


if __name__ == '__main__':
    tests = [
        test_flag_off_no_db_write_during_callbackless_load,
        test_flag_off_starts_no_heartbeat_thread,
        test_flag_on_flips_row_to_running_up_front,
        test_flip_lands_while_the_load_is_still_blocking,
        test_heartbeat_flushes_snapshot_during_callbackless_load,
        test_heartbeat_carries_last_known_phase_from_memory,
        test_heartbeat_respects_the_shared_flush_window,
        test_heartbeat_thread_exits_after_worker_finishes,
        test_completed_is_the_last_write,
        test_in_flight_heartbeat_cannot_resurrect_a_completed_row,
        test_cancelled_is_the_last_write,
        test_failed_is_the_last_write,
        test_heartbeat_stops_when_memory_status_leaves_running,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}\n  {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}\n  {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
