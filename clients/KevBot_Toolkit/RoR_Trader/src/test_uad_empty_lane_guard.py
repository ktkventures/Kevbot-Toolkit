"""Lock the full-UAD empty-lane guard (board #12 / #16).

A full-UAD recompute REPLACES a lane (DELETE + INSERT). When the bar fetch
transiently fails (Polygon miss / empty window) the engine produces ZERO
trades, and the backtest persist path then runs an UNFILTERED
`stored_trades=[]` redirect that wipes BOTH the backtest AND algo lanes and
inserts nothing — silently zeroing accumulated history.

The guard refuses that destructive empty-write when the recompute is empty AND
the bar fetch is unconfirmed/empty (the transient-failure signature), while
still allowing a GENUINE no-signal recompute (bars loaded, engine produced
nothing) to persist an empty lane normally.

`services.get_strategy_trades` stamps `source_bar_count` on `.attrs` so the
guard can tell the two apart:
  - source_bar_count > 0  → bars loaded  → genuine no-signal → allow empty write
  - source_bar_count == 0 → bars did NOT load → transient failure → refuse
  - source_bar_count None → unstamped → unconfirmed → refuse (fail loud)

Run:  cd src && ../.venv/bin/python -m pytest test_uad_empty_lane_guard.py -v
"""
import sys
sys.path.insert(0, '.')

from api.services.forward_test_service import (
    _should_refuse_empty_lane,
    _uad_empty_lane_guard_enabled,
)


# --- the transient-failure signature: refuse (this is the bug we're fixing) ---

def test_fetch_failure_populated_lane_refuses():
    """0 trades + bars did NOT load + lane has data → refuse the wipe."""
    assert _should_refuse_empty_lane(0, True, 0) is True


def test_fetch_unconfirmed_populated_lane_refuses():
    """source_bar_count unstamped (None) is treated as unconfirmed → refuse."""
    assert _should_refuse_empty_lane(0, True, None) is True


def test_fetch_failure_unknown_existence_refuses():
    """Existence query failed (None) → err toward refusing (fail loud), never
    risk a silent wipe when we can't confirm there's nothing to lose."""
    assert _should_refuse_empty_lane(0, None, 0) is True
    assert _should_refuse_empty_lane(0, None, None) is True


# --- genuine no-signal recompute: allow (bars DID load) ---

def test_genuine_no_signal_populated_lane_allowed():
    """Bars loaded (source_bar_count > 0) but engine produced nothing — a real
    result. Must NOT refuse even when the lane currently has trades."""
    assert _should_refuse_empty_lane(0, True, 5000) is False


def test_genuine_no_signal_empty_lane_allowed():
    assert _should_refuse_empty_lane(0, False, 5000) is False


# --- nothing to lose: allow even on a fetch failure ---

def test_fetch_failure_empty_lane_allowed():
    """First run / no prior rows → nothing to destroy → allow the empty write."""
    assert _should_refuse_empty_lane(0, False, 0) is False
    assert _should_refuse_empty_lane(0, False, None) is False


# --- non-empty recompute: never a zeroing risk ---

def test_nonempty_recompute_never_refuses():
    assert _should_refuse_empty_lane(42, True, 0) is False
    assert _should_refuse_empty_lane(42, True, None) is False
    assert _should_refuse_empty_lane(1, None, 0) is False


# --- flag / kill switch ---

def test_guard_default_on(monkeypatch):
    monkeypatch.delenv('RORT_UAD_GUARD_EMPTY_LANE', raising=False)
    assert _uad_empty_lane_guard_enabled() is True


def test_guard_kill_switch_off(monkeypatch):
    for val in ('0', 'false', 'no', 'off', 'OFF', 'False'):
        monkeypatch.setenv('RORT_UAD_GUARD_EMPTY_LANE', val)
        assert _uad_empty_lane_guard_enabled() is False, val


def test_guard_explicit_on(monkeypatch):
    for val in ('1', 'true', 'yes', 'on'):
        monkeypatch.setenv('RORT_UAD_GUARD_EMPTY_LANE', val)
        assert _uad_empty_lane_guard_enabled() is True, val
