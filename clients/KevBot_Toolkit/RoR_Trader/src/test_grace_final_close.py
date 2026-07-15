"""RORT_GRACE_FINAL_CLOSE_ELIGIBLE (Brandon audit P1 — no-signal-at-grace) —
flag-gated regression gate locking BOTH states:
  * flag ON  → grace that emits NO signal does NOT mark the bucket fired
               (final-close signal stays eligible)
  * flag OFF → legacy: grace marks the bucket fired unconditionally
A grace that DOES emit a signal marks the bucket fired in BOTH states.

    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_grace_final_close.py
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import ralph_engine as RE
from test_current_dev_candle_parity_audit import RTH_START  # noqa: E402

FLAG = "RORT_GRACE_FINAL_CLOSE_ELIGIBLE"


def _monitor(signals):
    """A grace-eligible monitor whose fire_on_partial_bucket returns `signals`."""
    class _M:
        tf_seconds = 10
        live_model = "ws_rest_spliced"
        grace_seconds = 4
        _fired_bucket = None
        strat_id = 1

        def fire_on_partial_bucket(self, *_a, **_k):
            return list(signals), {}
    return _M()


def _run_grace(monitor):
    hub = RE.SymbolHub("SPY")
    hub.monitors = {monitor.strat_id: monitor}
    hub.last_tick_time = RTH_START + timedelta(seconds=14)  # past due_at (10+4)
    builder = SimpleNamespace(
        _bar_count=10,
        _partial=SimpleNamespace(bar_start=RTH_START, open=100.0, high=101.0,
                                 low=99.0, close=100.0, volume=10.0))
    hub._check_strategy_grace_fires(10, builder, None, {})
    return monitor._fired_bucket


def test_flag_helper_runtime(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert RE._grace_final_close_eligible() is False
    monkeypatch.setenv(FLAG, "1")
    assert RE._grace_final_close_eligible() is True


def test_no_signal_flag_on_leaves_bucket_unfired(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    assert _run_grace(_monitor([])) is None, (
        "flag ON: a grace that emitted no signal must NOT mark the bucket fired")


def test_no_signal_flag_off_marks_bucket_legacy(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert _run_grace(_monitor([])) is not None, (
        "flag OFF: legacy — grace marks the bucket fired even with no signal")


def test_real_signal_marks_bucket_both_states(monkeypatch):
    sig = [{"type": "entry", "side": "long"}]
    monkeypatch.setenv(FLAG, "1")
    assert _run_grace(_monitor(sig)) is not None, (
        "flag ON: a grace that DID dispatch a signal marks the bucket fired")
    monkeypatch.delenv(FLAG, raising=False)
    assert _run_grace(_monitor(sig)) is not None, (
        "flag OFF: a real grace fire marks the bucket fired")
