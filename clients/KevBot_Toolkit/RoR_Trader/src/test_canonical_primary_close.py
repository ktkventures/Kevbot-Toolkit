"""RORT_CANONICAL_PRIMARY_CLOSE (Brandon audit P0 + P1-routing) — flag-gated
regression gate. Locks BOTH states so the switch stays reversible:
  * flag ON  → the fix (Brandon's contracts hold; default model is dispatched)
  * flag OFF → legacy behavior preserved byte-for-byte

Companion to test_current_dev_candle_parity_audit.py (which documents the
default-off failure). Run:
    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_canonical_primary_close.py
"""
from __future__ import annotations

import importlib
from datetime import timedelta

import pytest

import ralph_engine as RE
# Reuse the audit's test doubles (same monitor/bar contracts the engine expects).
from test_current_dev_candle_parity_audit import (  # noqa: E402
    _RecordingMonitor,
    _second_bar,
    RTH_START,
)

FLAG = "RORT_CANONICAL_PRIMARY_CLOSE"


# ── _elig_ws_agg: single-sourced capability set, flag-gated ──────────────
@pytest.mark.parametrize("model,off,on", [
    ("ws_agg_locked", True, True),
    ("ws_agg_with_rest_backfill", True, True),
    ("ws_rest_spliced", False, True),      # the DEFAULT — Brandon P1
    ("ws_agg_reconciled", False, True),    # spec: consumes ws_agg, was omitted
    ("ws_with_corrections", False, False),  # AM-consumer, never ws_agg
    (None, False, False),                   # unset — preserve legacy (no dispatch)
])
def test_elig_ws_agg_flag_gated(monkeypatch, model, off, on):
    monkeypatch.delenv(FLAG, raising=False)
    assert RE._elig_ws_agg(model) is off, f"{model} flag-OFF should be {off}"
    monkeypatch.setenv(FLAG, "1")
    assert RE._elig_ws_agg(model) is on, f"{model} flag-ON should be {on}"


def test_flag_helper_reads_runtime(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert RE._canonical_primary_close() is False
    monkeypatch.setenv(FLAG, "1")
    assert RE._canonical_primary_close() is True
    monkeypatch.setenv(FLAG, "0")
    assert RE._canonical_primary_close() is False


# ── P0: flush must not drive a strategy close on an incomplete >=60s partial ──
def _flush_partial_close_calls(monitor_tf: int) -> int:
    """Build a >=60s (or sub-minute) monitor, feed 55 per-second chart partials,
    force a wall-clock flush at +61s, return how many times on_bar_close ran."""
    hub = RE.SymbolHub("SPY")
    monitor = _RecordingMonitor(monitor_tf, "ws_rest_spliced")
    builder = RE.BarBuilder(monitor_tf)
    hub.monitors = {monitor.strat_id: monitor}
    hub.builders = {monitor_tf: builder}
    for second in range(55):
        builder.accept_second_bar(
            _second_bar(RTH_START, second), close_on_boundary=False,
            skip_volume=True)
    hub.flush_stale_bars(RTH_START + timedelta(seconds=61))
    return len(monitor.close_calls)


def test_p0_flag_on_no_partial_close_for_60s_primary(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    assert _flush_partial_close_calls(60) == 0, (
        "flag ON: flush must NOT drive a >=60s primary close on the incomplete "
        "chart partial — canonical fan-out owns it")


def test_p0_flag_off_preserves_legacy_flush_close(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert _flush_partial_close_calls(60) >= 1, (
        "flag OFF: legacy behavior — flush still drives the >=60s close "
        "(reversible switch)")


def test_p0_subminute_primary_unaffected_by_flag(monkeypatch):
    # The flush guard is `tf_seconds >= 60` only — sub-minute primaries (which
    # legitimately rely on flush-close) must behave IDENTICALLY in both states.
    monkeypatch.setenv(FLAG, "1")
    on = _flush_partial_close_calls(30)
    monkeypatch.delenv(FLAG, raising=False)
    off = _flush_partial_close_calls(30)
    assert on == off, (
        f"sub-minute (30s) flush-close must be flag-invariant (on={on} off={off})")
