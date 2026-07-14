"""Lock the engine liveness watchdogs (2026-07-14).

Context: on 2026-07-14 a periodic in-process rebuild (PRIMARY-RESYNC) starved the
live engine for ~9 minutes at a stretch — alerts dispatched 5-10 MINUTES after
their bar closed, bar closes inside the stall were skipped entirely (a real live
entry was lost), and NOTHING in the system said a word: the Docker healthcheck
only tested that a file EXISTED, and that file is touched by the worker MANAGER
thread, which keeps running while the engine is starved.

These lock the three watchdogs that make a stall impossible to miss:
1. engine_health.check() ages out the ENGINE heartbeat (not the manager's)
2. it still tolerates a cold boot (engine heartbeat absent during warmup)
3. _fire_alert warns when a dispatch trails its bar close by minutes

Run:  cd src && ../.venv/bin/pytest test_engine_liveness.py -v
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, '.')

import engine_health  # noqa: E402


# ─────────────────────────── healthcheck ────────────────────────────────

def _touch(path, age_s: float, now: float):
    path.write_text("x")
    os.utime(path, (now - age_s, now - age_s))
    return str(path)


def test_healthy_when_engine_heartbeat_is_fresh(tmp_path):
    now = 1_000_000.0
    eng = _touch(tmp_path / "engine_alive", 3, now)
    mgr = _touch(tmp_path / "worker_alive", 1, now)
    ok, why = engine_health.check(now=now, engine_file=eng, manager_file=mgr,
                                  max_age_s=300)
    assert ok, why


def test_unhealthy_when_engine_stalls_even_if_manager_is_fresh(tmp_path):
    """THE regression this exists for: manager thread alive, engine starved."""
    now = 1_000_000.0
    eng = _touch(tmp_path / "engine_alive", 600, now)   # engine stalled 10 min
    mgr = _touch(tmp_path / "worker_alive", 1, now)     # manager still ticking
    ok, why = engine_health.check(now=now, engine_file=eng, manager_file=mgr,
                                  max_age_s=300)
    assert not ok, "a starved engine must FAIL the healthcheck"
    assert "STALLED" in why


def test_boot_grace_when_engine_heartbeat_absent(tmp_path):
    """Cold boot: warmup runs minutes before the periodic loop starts."""
    now = 1_000_000.0
    mgr = _touch(tmp_path / "worker_alive", 20, now)
    ok, why = engine_health.check(
        now=now, engine_file=str(tmp_path / "nope"), manager_file=mgr,
        max_age_s=300)
    assert ok and "boot grace" in why


def test_unhealthy_when_engine_never_starts(tmp_path):
    """Boot grace is bounded — 'engine never came up' still fails."""
    now = 1_000_000.0
    mgr = _touch(tmp_path / "worker_alive", 900, now)
    ok, _ = engine_health.check(
        now=now, engine_file=str(tmp_path / "nope"), manager_file=mgr,
        max_age_s=300)
    assert not ok


def test_unhealthy_when_no_files(tmp_path):
    ok, _ = engine_health.check(
        now=1_000_000.0, engine_file=str(tmp_path / "a"),
        manager_file=str(tmp_path / "b"), max_age_s=300)
    assert not ok


# ─────────────────────────── alert-lag tripwire ─────────────────────────

class _FakeMonitor:
    strat_id = 333
    tf_seconds = 30
    strategy = {'id': 333, 'name': 'fake', 'symbol': 'TSLA',
                'timeframe': '30Sec'}
    user_id = 'test-user'
    session = 'RTH'
    live_model = 'ws'

    class _Ind:
        class _State:
            current = {}
        state = _State()
    indicators = _Ind()


def _fire(sig, caplog):
    import ralph_engine
    hub = ralph_engine.SymbolHub('TSLA')
    fired = []
    with caplog.at_level(logging.WARNING, logger='ralph'):
        hub._fire_alert(sig, _FakeMonitor(), {}, lambda *a, **k: fired.append(1))
    return fired, caplog.text


def test_alert_lag_warns_when_dispatch_trails_its_bar(caplog):
    """The 07-14 signature: bar closed minutes ago, alert only firing now."""
    bar_start = datetime.now(timezone.utc) - timedelta(minutes=6)
    fired, text = _fire(
        {'type': 'entry', 'bar_time': bar_start.isoformat(), 'price': 100.0},
        caplog)
    assert fired, "tripwire must not block the dispatch"
    assert "[ALERT-LAG]" in text
    assert "sid=333" in text


def test_alert_lag_silent_on_a_normal_dispatch(caplog):
    """Bar just closed — normal path must stay quiet."""
    bar_start = datetime.now(timezone.utc) - timedelta(
        seconds=_FakeMonitor.tf_seconds + 2)
    fired, text = _fire(
        {'type': 'entry', 'bar_time': bar_start.isoformat(), 'price': 100.0},
        caplog)
    assert fired
    assert "[ALERT-LAG]" not in text


def test_alert_lag_tripwire_never_breaks_dispatch_on_bad_input(caplog):
    fired, text = _fire(
        {'type': 'entry', 'bar_time': 'not-a-timestamp', 'price': 100.0},
        caplog)
    assert fired, "a malformed bar_time must never suppress the alert"
