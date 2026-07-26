"""Tests for the gap_healer skip-when-closed gate (board #140 / #15 / #51).

Confirms: (1) the gate is OFF by default (OFF == pre-#140 behavior);
(2) when armed AND the market is closed, both the sweeper and the
detect-path chokepoint (queue_heal) short-circuit WITHOUT scanning or
queueing; (3) when armed but the market is OPEN, nothing changes;
(4) a calendar error fails safe to OPEN. Run from src/:
    python -m pytest test_gap_heal_skip_closed.py -q
"""
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, '.')

import gap_healer  # noqa: E402

TF = 10


class _NoopExecutor:
    """Stand-in for the heal thread pool: records submits, runs nothing."""
    def __init__(self):
        self.submits = 0

    def submit(self, *a, **k):
        self.submits += 1
        return None


def _recent_bar_starts(n=3):
    """Bar starts inside queue_heal's accept window (older than 2*tf,
    newer than the lookback floor) so they'd queue when the market's open."""
    now = datetime.now(timezone.utc)
    return [now - timedelta(seconds=300 + i * TF) for i in range(n)]


def _reset(monkeypatch):
    gap_healer._heal_state.clear()
    gap_healer._closed_log_last["mono"] = 0.0
    # A stub insert callback so queue_heal passes its _insert_callback guard,
    # and a no-op executor so accepted windows don't spawn real REST heals.
    monkeypatch.setattr(gap_healer, '_insert_callback',
                        lambda *a, **k: 'inserted')
    ex = _NoopExecutor()
    monkeypatch.setattr(gap_healer, '_get_executor', lambda: ex)
    return ex


# --- flag default ---------------------------------------------------------

def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("GAP_HEAL_SKIP_WHEN_CLOSED", raising=False)
    assert gap_healer.skip_when_closed_enabled() is False
    # Even if the calendar says closed, the gate is inert when the flag is off.
    monkeypatch.setattr("market_calendar.is_market_closed", lambda now=None: True)
    assert gap_healer._market_is_closed() is False


def test_flag_on_reads_env(monkeypatch):
    monkeypatch.setenv("GAP_HEAL_SKIP_WHEN_CLOSED", "1")
    assert gap_healer.skip_when_closed_enabled() is True


# --- queue_heal (detect chokepoint) --------------------------------------

def test_queue_heal_skips_when_closed(monkeypatch):
    ex = _reset(monkeypatch)
    monkeypatch.setenv("GAP_HEAL_SKIP_WHEN_CLOSED", "1")
    monkeypatch.setattr("market_calendar.is_market_closed", lambda now=None: True)
    n = gap_healer.queue_heal("TEST", TF, _recent_bar_starts(), "sweep")
    assert n == 0
    assert ex.submits == 0  # nothing queued for a heal run


def test_queue_heal_runs_when_open(monkeypatch):
    ex = _reset(monkeypatch)
    monkeypatch.setenv("GAP_HEAL_SKIP_WHEN_CLOSED", "1")
    monkeypatch.setattr("market_calendar.is_market_closed", lambda now=None: False)
    n = gap_healer.queue_heal("TEST", TF, _recent_bar_starts(), "sweep")
    assert n > 0
    assert ex.submits == 1


def test_queue_heal_runs_when_closed_but_flag_off(monkeypatch):
    # Market closed but flag OFF → pre-#140 behavior (still queues).
    ex = _reset(monkeypatch)
    monkeypatch.delenv("GAP_HEAL_SKIP_WHEN_CLOSED", raising=False)
    monkeypatch.setattr("market_calendar.is_market_closed", lambda now=None: True)
    n = gap_healer.queue_heal("TEST", TF, _recent_bar_starts(), "sweep")
    assert n > 0
    assert ex.submits == 1


# --- sweep_gaps (skips the expensive scan) -------------------------------

def test_sweep_skips_scan_when_closed(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("GAP_HEAL_SKIP_WHEN_CLOSED", "1")
    monkeypatch.setattr("market_calendar.is_market_closed", lambda now=None: True)
    calls = {"n": 0}

    def _scan():
        calls["n"] += 1
        return [("TEST", TF, _recent_bar_starts())]

    monkeypatch.setattr(gap_healer, '_scan_provider', _scan)
    assert gap_healer.sweep_gaps() == 0
    assert calls["n"] == 0  # scan provider never invoked


def test_sweep_scans_when_open(monkeypatch):
    ex = _reset(monkeypatch)
    monkeypatch.setenv("GAP_HEAL_SKIP_WHEN_CLOSED", "1")
    monkeypatch.setattr("market_calendar.is_market_closed", lambda now=None: False)
    calls = {"n": 0}

    def _scan():
        calls["n"] += 1
        return [("TEST", TF, _recent_bar_starts())]

    monkeypatch.setattr(gap_healer, '_scan_provider', _scan)
    queued = gap_healer.sweep_gaps()
    assert calls["n"] == 1
    assert queued > 0
    assert ex.submits >= 1


# --- fail-safe ------------------------------------------------------------

def test_calendar_error_fails_safe_open(monkeypatch):
    monkeypatch.setenv("GAP_HEAL_SKIP_WHEN_CLOSED", "1")

    def _boom(now=None):
        raise RuntimeError("calendar blew up")

    monkeypatch.setattr("market_calendar.is_market_closed", _boom)
    # Error → treated as OPEN so the healer keeps its current behavior.
    assert gap_healer._market_is_closed() is False
