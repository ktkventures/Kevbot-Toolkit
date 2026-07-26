"""Regression tests for board #32 — recover 'queued' mass searches after an
API restart.

The bug: cleanup_orphaned_mass_searches only swept status='running' rows. If
the API OOM-restarted during the load phase — while the row was still 'queued'
(the worker hadn't flushed 'running' or written its first checkpoint yet) — the
job was invisible to the sweep and stuck 'queued' forever, never picked up.

The fix is gated by RORT_MASS_RECOVER_QUEUED (default OFF). With the flag OFF
behavior is byte-identical to the legacy sweep; with it ON, 'queued' rows are
reclaimed (resumed from scratch, bounded by _MAX_AUTO_RESUMES), and searches
still live in THIS process are never reaped (protects the manual
/cleanup-orphans trigger).

These tests use a mocked Supabase client and a stubbed start_mass_search_async
so they run in CI without a DB and without spawning worker threads.

Run: cd src && ../.venv/bin/python test_mass_recover_queued_32.py
"""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# Fake Supabase client — enough of the PostgREST builder chain to drive the
# select/in_/eq/update/execute calls cleanup_orphaned_mass_searches makes.
# ---------------------------------------------------------------------------
class _FakeQuery:
    def __init__(self, rows, updates, calls):
        self._rows = rows            # shared dataset (list of row dicts)
        self._updates = updates      # shared capture: [(id, payload), ...]
        self._calls = calls          # shared capture of filter methods used
        self._mode = None            # 'select' | 'update'
        self._payload = None
        self._filters = []           # list of ('eq'|'in', col, val)

    def select(self, *_a, **_k):
        self._mode = 'select'
        return self

    def update(self, payload):
        self._mode = 'update'
        self._payload = payload
        return self

    def eq(self, col, val):
        self._calls.append(('eq', col, val))
        self._filters.append(('eq', col, val))
        return self

    def in_(self, col, vals):
        self._calls.append(('in', col, list(vals)))
        self._filters.append(('in', col, list(vals)))
        return self

    def _match(self, row):
        for kind, col, val in self._filters:
            if kind == 'eq' and row.get(col) != val:
                return False
            if kind == 'in' and row.get(col) not in val:
                return False
        return True

    def execute(self):
        if self._mode == 'update':
            # The id filter is applied via .eq('id', sid) before execute.
            target_id = None
            for kind, col, val in self._filters:
                if col == 'id':
                    target_id = val
            self._updates.append((target_id, self._payload))
            # Mutate the in-memory row so a subsequent select reflects it.
            for r in self._rows:
                if r.get('id') == target_id:
                    r.update(self._payload)
            return mock.MagicMock(data=[])
        # select
        matched = [dict(r) for r in self._rows if self._match(r)]
        return mock.MagicMock(data=matched)


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.updates = []     # [(id, payload), ...]
        self.calls = []       # filter methods observed

    def table(self, _name):
        return _FakeQuery(self.rows, self.updates, self.calls)


class _Harness:
    """Runs cleanup_orphaned_mass_searches against a fake dataset, capturing
    both DB updates and start_mass_search_async relaunch calls."""
    def __init__(self, rows, flag_on):
        self.rows = rows
        self.flag_on = flag_on
        self.relaunches = []  # [(sid, config, resume_checkpoint, user_id)]

    def run(self):
        db, mb = _setup_modules()
        fake_client = _FakeClient(self.rows)

        def _fake_start(sid, config, resume_checkpoint=None,
                        user_id_override=None):
            self.relaunches.append(
                (sid, config, resume_checkpoint, user_id_override))

        prev_flag = os.environ.get('RORT_MASS_RECOVER_QUEUED')
        if self.flag_on:
            os.environ['RORT_MASS_RECOVER_QUEUED'] = '1'
        else:
            os.environ.pop('RORT_MASS_RECOVER_QUEUED', None)
        try:
            with mock.patch.object(db, 'USE_DB', True), \
                 mock.patch.object(db, 'get_admin_client',
                                   return_value=fake_client), \
                 mock.patch.object(mb, 'start_mass_search_async', _fake_start):
                summary = mb.cleanup_orphaned_mass_searches()
        finally:
            if prev_flag is None:
                os.environ.pop('RORT_MASS_RECOVER_QUEUED', None)
            else:
                os.environ['RORT_MASS_RECOVER_QUEUED'] = prev_flag
        self.summary = summary
        self.updates = fake_client.updates
        self.calls = fake_client.calls
        return summary

    # ---- assertion helpers ------------------------------------------------
    def update_for(self, sid):
        """Last committed update payload for a given id (or None)."""
        found = [p for (i, p) in self.updates if i == sid]
        return found[-1] if found else None

    def relaunch_for(self, sid):
        found = [r for r in self.relaunches if r[0] == sid]
        return found[-1] if found else None


def _queued_row(sid=101, config=None):
    return {
        'id': sid,
        'status': 'queued',
        'user_id': 'user-uuid',
        'config_data': {'config': config if config is not None
                        else {'tickers': ['AAPL'], 'timeframes': ['1Min']}},
    }


def _running_row(sid, completed=None, resume_count=0, config=None,
                 has_config=True):
    checkpoint = {}
    if completed is not None:
        checkpoint['completed_symbol_tfs'] = completed
    if resume_count:
        checkpoint['resume_count'] = resume_count
    cfg_data = {}
    if checkpoint:
        cfg_data['checkpoint'] = checkpoint
    if has_config:
        cfg_data['config'] = (config if config is not None
                              else {'tickers': ['AAPL'], 'timeframes': ['1Min']})
    return {
        'id': sid,
        'status': 'running',
        'user_id': 'user-uuid',
        'config_data': cfg_data,
    }


# ===========================================================================
# Flag ON — the new behavior
# ===========================================================================
def test_queued_row_recovered_from_scratch():
    """A stranded 'queued' row is relaunched from scratch, flipped to
    'running', its resume_count seeded to 1, and counted as queued_recovered."""
    h = _Harness([_queued_row(sid=101)], flag_on=True)
    s = h.run()

    assert s['resumed'] == 1, s
    assert s['queued_recovered'] == 1, s
    assert s['orphaned'] == 0, s

    r = h.relaunch_for(101)
    assert r is not None, "queued row was never relaunched"
    # From scratch → no checkpoint handed to the worker.
    assert r[2] is None, f"expected resume_checkpoint=None, got {r[2]}"
    assert r[3] == 'user-uuid', f"user_id not forwarded: {r[3]}"

    upd = h.update_for(101)
    assert upd is not None, "queued row was never updated in DB"
    assert upd.get('status') == 'running', f"status not flipped: {upd}"
    ckpt = upd['config_data'].get('checkpoint', {})
    assert ckpt.get('resume_count') == 1, f"resume_count not seeded: {ckpt}"


def test_running_with_checkpoint_resumes_from_checkpoint():
    """A 'running' row with a real checkpoint still resumes from that
    checkpoint (legacy path preserved), NOT counted as queued_recovered."""
    h = _Harness(
        [_running_row(202, completed=[['AAPL', '1Min']], resume_count=0)],
        flag_on=True)
    s = h.run()

    assert s['resumed'] == 1, s
    assert s['queued_recovered'] == 0, s
    r = h.relaunch_for(202)
    assert r is not None and r[2] is not None, \
        "checkpoint row should resume WITH a checkpoint"
    assert r[2].get('resume_count') == 1, f"resume_count not bumped: {r[2]}"


def test_running_no_checkpoint_still_orphaned_flag_on():
    """No-regression guard: a first-run 'running' row that died before its
    first checkpoint stays ORPHANED even with the flag ON (resume_count==0,
    not queued, no completed groups)."""
    h = _Harness([_running_row(303, completed=None, resume_count=0)],
                 flag_on=True)
    s = h.run()

    assert s['resumed'] == 0, s
    assert s['orphaned'] == 1, s
    assert h.relaunch_for(303) is None, "must not relaunch a no-checkpoint run"
    upd = h.update_for(303)
    assert upd and upd.get('status') == 'orphaned', f"not orphaned: {upd}"


def test_from_scratch_retry_chain_respects_cap():
    """A queued-origin row already at the retry cap (resume_count ==
    _MAX_AUTO_RESUMES, no completed groups) is orphaned, not relaunched —
    crash-loop protection for from-scratch resumes."""
    _, mb = _setup_modules()
    cap = mb._MAX_AUTO_RESUMES
    h = _Harness([_running_row(404, completed=[], resume_count=cap)],
                 flag_on=True)
    s = h.run()

    assert s['resumed'] == 0, s
    assert s['orphaned'] == 1, s
    assert h.relaunch_for(404) is None


def test_queued_without_config_orphaned():
    """A 'queued' row with no saved config can't be relaunched → orphaned."""
    h = _Harness([_queued_row(sid=505, config={})], flag_on=True)
    s = h.run()

    assert s['resumed'] == 0, s
    assert s['orphaned'] == 1, s
    assert h.relaunch_for(505) is None


def test_live_process_search_not_reaped():
    """A row whose worker is still alive in THIS process must be skipped by
    the sweep (protects the manual /cleanup-orphans trigger)."""
    _, mb = _setup_modules()
    mb._active_searches['606'] = {'status': 'running', 'cancelled': False}
    try:
        h = _Harness([_queued_row(sid=606)], flag_on=True)
        s = h.run()
        assert s['resumed'] == 0, s
        assert s['orphaned'] == 0, s
        assert h.relaunch_for(606) is None, "live search was reaped!"
        assert h.update_for(606) is None, "live search row was mutated!"
    finally:
        mb._active_searches.pop('606', None)


# ===========================================================================
# Flag OFF — must be byte-identical to the legacy sweep
# ===========================================================================
def test_queued_ignored_when_flag_off():
    """With the flag OFF, 'queued' rows are invisible: the sweep filters on
    status=='running' only and never touches the queued row."""
    h = _Harness([_queued_row(sid=707)], flag_on=False)
    s = h.run()

    assert s['resumed'] == 0, s
    assert s['orphaned'] == 0, s
    assert s['queued_recovered'] == 0, s
    assert h.relaunch_for(707) is None
    assert h.update_for(707) is None
    # The query must be the legacy .eq('status','running'), never .in_().
    assert ('eq', 'status', 'running') in h.calls, h.calls
    assert not any(k == 'in' for (k, *_ ) in h.calls), \
        f"OFF path must not use .in_(): {h.calls}"


def test_running_checkpoint_resumes_flag_off_no_status_write():
    """Flag OFF legacy path: a 'running' row with a checkpoint resumes from
    that checkpoint and the DB update carries NO 'status' field (byte-identical
    to pre-#32 behavior)."""
    h = _Harness(
        [_running_row(808, completed=[['AAPL', '1Min']], resume_count=1)],
        flag_on=False)
    s = h.run()

    assert s['resumed'] == 1, s
    r = h.relaunch_for(808)
    assert r is not None and r[2] is not None, "should resume with checkpoint"
    assert r[2].get('resume_count') == 2, f"resume_count not bumped: {r[2]}"
    upd = h.update_for(808)
    assert upd is not None
    assert 'status' not in upd, \
        f"OFF path must not write status (would diverge from legacy): {upd}"


def test_running_no_checkpoint_orphaned_flag_off():
    """Flag OFF: 'running' with no checkpoint → orphaned (unchanged)."""
    h = _Harness([_running_row(909, completed=None)], flag_on=False)
    s = h.run()
    assert s['resumed'] == 0 and s['orphaned'] == 1, s
    upd = h.update_for(909)
    assert upd and upd.get('status') == 'orphaned', upd


if __name__ == '__main__':
    tests = [
        test_queued_row_recovered_from_scratch,
        test_running_with_checkpoint_resumes_from_checkpoint,
        test_running_no_checkpoint_still_orphaned_flag_on,
        test_from_scratch_retry_chain_respects_cap,
        test_queued_without_config_orphaned,
        test_live_process_search_not_reaped,
        test_queued_ignored_when_flag_off,
        test_running_checkpoint_resumes_flag_off_no_status_write,
        test_running_no_checkpoint_orphaned_flag_off,
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
