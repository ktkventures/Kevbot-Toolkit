"""RORT_CANONICAL_FINE_TF_STATE (M-RS5b) — flag-gated regression gate:
  * flag ON  → fine RTH shadow state is REBUILT from the canonical resample of
               the hub's own 1Min history at every close (one path; refresher
               retires); derive failure = LOUD error + previous records held
               (never a silent incremental fallback — Kevin ruling 07-16)
  * flag OFF → legacy incremental on_bar_close publish, byte-identical

    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_canonical_fine_tf_state.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import ralph_engine as RE

FLAG = "RORT_CANONICAL_FINE_TF_STATE"
T0 = pd.Timestamp("2026-07-16 14:30:00+00:00")


def _m1_history(n_minutes=10):
    idx = pd.date_range(T0, periods=n_minutes, freq="1min")
    px = [100.0 + i for i in range(n_minutes)]
    return pd.DataFrame({"open": px, "high": [p + 0.5 for p in px],
                         "low": [p - 0.5 for p in px], "close": px,
                         "volume": [10.0] * n_minutes}, index=idx)


class _Shadow:
    def __init__(self):
        self.calls = []
        self.indicators = SimpleNamespace(_initialized=True)

    def on_bar_close(self, bar):
        self.calls.append(("incremental", bar))
        return {"INC_STATE"}

    def recompute_confluence(self, df):
        self.calls.append(("canonical", len(df)))
        return {"CANON_STATE"}


def _hub(m1=None):
    hub = RE.SymbolHub("SPY")
    if m1 is not None:
        hub.builders[60] = SimpleNamespace(history=m1)
    return hub


def _completed_5m_bar(start=T0):
    return {"timestamp": start.isoformat(), "open": 100.0, "high": 101.0,
            "low": 99.5, "close": 100.5, "volume": 50.0}


def test_flag_helper_runtime(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert RE._canonical_fine_tf_state() is False
    monkeypatch.setenv(FLAG, "1")
    assert RE._canonical_fine_tf_state() is True


def test_flag_off_legacy_incremental(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    hub, shadow = _hub(_m1_history()), _Shadow()
    hub._close_shadow_with_bar(300, "RTH", shadow, _completed_5m_bar())
    assert shadow.calls and shadow.calls[0][0] == "incremental"
    assert hub._mtf_confluence.get((300, "RTH")) == {"INC_STATE"}


def test_flag_on_publishes_canonical_state(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    hub, shadow = _hub(_m1_history(10)), _Shadow()
    # completed 5Min bucket starting 14:30 → close_ts 14:35; constituents
    # 14:30-14:34 are all closed by then → exactly ONE canonical bucket.
    hub._close_shadow_with_bar(300, "RTH", shadow, _completed_5m_bar())
    assert shadow.calls == [("canonical", 1)], shadow.calls
    assert hub._mtf_confluence.get((300, "RTH")) == {"CANON_STATE"}


def test_flag_on_derive_failure_is_loud_and_holds_previous(monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "1")
    hub, shadow = _hub(m1=None), _Shadow()          # no 1Min history at all
    hub._mtf_confluence[(300, "RTH")] = {"PREVIOUS"}
    with caplog.at_level("ERROR"):
        hub._close_shadow_with_bar(300, "RTH", shadow, _completed_5m_bar())
    assert shadow.calls == [], "no incremental fallback allowed (fail loud)"
    assert hub._mtf_confluence[(300, "RTH")] == {"PREVIOUS"}, "hold previous"
    assert any("CANONICAL-FINE-STATE" in r.message for r in caplog.records), (
        "derive failure must be a LOUD error")


def test_flag_on_coarse_tf_unaffected(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    hub, shadow = _hub(_m1_history()), _Shadow()
    coarse = RE.COARSE_SECONDARY_SECONDS  # first coarse TF
    hub._close_shadow_with_bar(coarse, "RTH", shadow,
                               {"timestamp": T0.isoformat(), "open": 1,
                                "high": 1, "low": 1, "close": 1, "volume": 1})
    # Coarse path is NOT the canonical-fine derive: either legacy incremental
    # or the coarse-RTH-reload path — never the fine canonical recompute.
    assert not any(c[0] == "canonical" for c in shadow.calls)


def test_canonical_fine_df_closed_buckets_only():
    hub = _hub(_m1_history(12))  # 14:30..14:41
    close_ts = T0 + pd.Timedelta(minutes=10)  # 14:40
    cdf = hub._canonical_fine_df(300, close_ts)
    # Buckets 14:30 (closes 14:35) and 14:35 (closes 14:40) are fully closed;
    # the forming 14:40 bucket must be excluded.
    assert list(cdf.index) == [T0, T0 + pd.Timedelta(minutes=5)]
