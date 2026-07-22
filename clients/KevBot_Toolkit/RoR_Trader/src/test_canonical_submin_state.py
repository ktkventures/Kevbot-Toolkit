"""RORT_CANONICAL_SUBMIN_STATE (Phase 2b, Plan_SubMinute_Canonical_Source) —
flag-gated regression gate for the SUB-MINUTE canonical state derive:
  * flag ON  → sub-minute shadow records are REBUILT by a full clean replay
               over the builder's own closed history, SESSION-FILTERED to the
               shadow's session and tail-bounded (RORT_SUBMIN_DERIVE_BARS);
               derive failure = LOUD error + previous records held (never a
               silent incremental fallback — mirrors the M-RS5b ruling)
  * flag OFF → legacy incremental on_bar_close publish, byte-identical
Also gates the store-side flag plumbing (default_submin_targets empty when
OFF) and the offline swap's None-when-off contract.

    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_canonical_submin_state.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import ralph_engine as RE

FLAG = "RORT_CANONICAL_SUBMIN_STATE"
# 14:30Z = 10:30 ET — inside RTH (EDT).
T0 = pd.Timestamp("2026-07-16 14:30:00+00:00")
# 08:00Z = 04:00 ET — premarket, OUTSIDE RTH.
T_PRE = pd.Timestamp("2026-07-16 08:00:00+00:00")


def _h10(start, n):
    idx = pd.date_range(start, periods=n, freq="10s")
    px = [300.0 + 0.01 * i for i in range(n)]
    return pd.DataFrame({"open": px, "high": [p + 0.05 for p in px],
                         "low": [p - 0.05 for p in px], "close": px,
                         "volume": [10.0] * n}, index=idx)


class _Shadow:
    def __init__(self):
        self.calls = []
        self.indicators = SimpleNamespace(
            _initialized=True,
            recompute_from_history=lambda df, **kw: self.calls.append(
                ("history_recompute", len(df))))

    def on_bar_close(self, bar):
        self.calls.append(("incremental", bar))
        return {"INC_STATE"}

    def recompute_confluence(self, df):
        self.calls.append(("canonical", len(df)))
        return {"CANON_STATE"}


def _hub_with_shadow(history, session="RTH"):
    hub = RE.SymbolHub("TSLA")
    shadow = _Shadow()
    hub._shadow_engines[(10, session)] = shadow
    builder = SimpleNamespace(history=history)
    # The canonical derive reads self.builders (the single source of truth in
    # BOTH close sites); register the builder as production does.
    hub.builders[10] = builder
    return hub, shadow, builder


def _completed_10s_bar(start=T0):
    return {"timestamp": start.isoformat(), "open": 300.0, "high": 300.1,
            "low": 299.9, "close": 300.05, "volume": 25.0}


def test_flag_helper_runtime(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert RE._canonical_submin_state() is False
    monkeypatch.setenv(FLAG, "1")
    assert RE._canonical_submin_state() is True


def test_flag_off_legacy_incremental(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    hub, shadow, builder = _hub_with_shadow(_h10(T0, 30))
    hub._close_subminute_shadows(10, builder, _completed_10s_bar(),
                                 was_duplicate=False, was_correction=False)
    assert shadow.calls and shadow.calls[0][0] == "incremental"
    assert hub._mtf_confluence.get((10, "RTH")) == {"INC_STATE"}


def test_flag_on_publishes_canonical_state(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    hub, shadow, builder = _hub_with_shadow(_h10(T0, 30))
    hub._close_subminute_shadows(10, builder, _completed_10s_bar(),
                                 was_duplicate=False, was_correction=False)
    # All 30 bars are inside RTH → the whole history replays canonically.
    assert shadow.calls == [("canonical", 30)], shadow.calls
    assert hub._mtf_confluence.get((10, "RTH")) == {"CANON_STATE"}


def test_flag_on_session_filter_drops_premarket(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    # 20 premarket bars + 30 RTH bars: the RTH shadow's canonical replay must
    # see ONLY the RTH bars (the offline lane's ribbon never sees premarket —
    # an incremental state advanced on premarket bars is itself divergence).
    hist = pd.concat([_h10(T_PRE, 20), _h10(T0, 30)])
    hub, shadow, builder = _hub_with_shadow(hist)
    hub._close_subminute_shadows(10, builder, _completed_10s_bar(),
                                 was_duplicate=False, was_correction=False)
    assert shadow.calls == [("canonical", 30)], shadow.calls


def test_flag_on_tail_bound(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("RORT_SUBMIN_DERIVE_BARS", "600")
    hub, shadow, builder = _hub_with_shadow(_h10(T0, 900))
    hub._close_subminute_shadows(10, builder, _completed_10s_bar(),
                                 was_duplicate=False, was_correction=False)
    assert shadow.calls == [("canonical", 600)], shadow.calls


def test_flag_on_derive_failure_is_loud_and_holds_previous(monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "1")
    # RTH shadow but ONLY premarket history → session-filtered cdf is EMPTY.
    hub, shadow, builder = _hub_with_shadow(_h10(T_PRE, 20))
    hub._mtf_confluence[(10, "RTH")] = {"PREVIOUS"}
    with caplog.at_level("ERROR"):
        hub._close_subminute_shadows(10, builder, _completed_10s_bar(T_PRE),
                                     was_duplicate=False, was_correction=False)
    assert shadow.calls == [], "no incremental fallback allowed (fail loud)"
    assert hub._mtf_confluence[(10, "RTH")] == {"PREVIOUS"}, "hold previous"
    assert any("CANONICAL-SUBMIN-STATE" in r.message for r in caplog.records)


def test_flag_on_correction_rebroadcast_skips_legacy_recompute(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    hub, shadow, builder = _hub_with_shadow(_h10(T0, 30))
    hub._close_subminute_shadows(10, builder, _completed_10s_bar(),
                                 was_duplicate=True, was_correction=True)
    # Canonical mode picks the corrected history up at the next close; the
    # legacy history-recompute must NOT run twice over the same bars.
    assert shadow.calls == [], shadow.calls


def test_flag_off_correction_rebroadcast_keeps_legacy(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    hub, shadow, builder = _hub_with_shadow(_h10(T0, 30))
    hub._close_subminute_shadows(10, builder, _completed_10s_bar(),
                                 was_duplicate=True, was_correction=True)
    assert shadow.calls == [("history_recompute", 30)], shadow.calls


def _hub_with_shadow_and_builder(history, session="RTH"):
    """Variant registering the builder on the hub — the primary-TF topology
    (_close_shadow_with_bar reads self.builders, not a passed-in builder)."""
    hub, shadow, builder = _hub_with_shadow(history, session)
    hub.builders[10] = builder
    return hub, shadow, builder


def test_close_shadow_with_bar_submin_flag_on_canonical(monkeypatch):
    """The MULTI-MONITOR topology (found live 2026-07-21): when the sub-minute
    TF is ALSO a primary on the hub (267/338 real 10Sec monitors), the shadow
    closes via _close_shadow_with_bar — the canonical derive must own that
    path too, or live silently stays incremental while the harness's
    single-monitor hub shows canonical (the label-drift-class trap)."""
    monkeypatch.setenv(FLAG, "1")
    hub, shadow, _ = _hub_with_shadow_and_builder(_h10(T0, 30))
    hub._close_shadow_with_bar(10, "RTH", shadow, _completed_10s_bar())
    assert shadow.calls == [("canonical", 30)], shadow.calls
    assert hub._mtf_confluence.get((10, "RTH")) == {"CANON_STATE"}


def test_close_shadow_with_bar_submin_flag_off_legacy(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    hub, shadow, _ = _hub_with_shadow_and_builder(_h10(T0, 30))
    hub._close_shadow_with_bar(10, "RTH", shadow, _completed_10s_bar())
    assert shadow.calls and shadow.calls[0][0] == "incremental"
    assert hub._mtf_confluence.get((10, "RTH")) == {"INC_STATE"}


def test_close_shadow_with_bar_submin_no_builder_is_loud_hold(monkeypatch,
                                                              caplog):
    monkeypatch.setenv(FLAG, "1")
    hub = RE.SymbolHub("TSLA")
    shadow = _Shadow()
    hub._shadow_engines[(10, "RTH")] = shadow
    hub._mtf_confluence[(10, "RTH")] = {"PREVIOUS"}
    with caplog.at_level("ERROR"):
        hub._close_shadow_with_bar(10, "RTH", shadow, _completed_10s_bar())
    assert shadow.calls == [], "no incremental fallback allowed (fail loud)"
    assert hub._mtf_confluence[(10, "RTH")] == {"PREVIOUS"}
    assert any("CANONICAL-SUBMIN-STATE" in r.message for r in caplog.records)


def _pack_monitor(sid, tf, sess, conf, entry='utv4_bull_flip',
                  exit_='utv4_bear_flip'):
    strat = {'id': sid, 'symbol': 'TSLA', 'timeframe': tf,
             'trading_session': sess, 'direction': 'long',
             'confluence': conf, 'general_confluences': [], 'triggers': [],
             'entry_trigger': entry, 'exit_trigger': exit_, 'config': {}}
    return RE.StrategyMonitor(strat, None, general_packs=[])


def test_finalize_creates_submin_shadow_despite_covering_monitor(monkeypatch):
    """Live finding #2 (2026-07-21): 267/338 are utv4-TRIGGERED 10Sec monitors,
    so the interp-aware suppression said 'real monitor covers UT_BOT_V4' and
    never created the (10,'RTH') shadow — 339's gate consumed 267's
    INCREMENTAL own-records, bypassing the canonical derive entirely. Under
    the flag, a sub-minute key must ALWAYS get its dedicated shadow (the
    canonical owner); OFF keeps legacy suppression."""
    monkeypatch.setenv("USE_DB", "false")
    import pack_registry
    pack_registry.scan_and_load_all()

    def _mk_hub():
        hub = RE.SymbolHub("TSLA")
        hub.add_monitor(_pack_monitor(267, '10Sec', 'RTH',
                                      ['2m-SWING_123-NEUTRAL']))
        hub.add_monitor(_pack_monitor(
            339, '30Sec', 'RTH',
            ['30m-BOLLINGER_BANDS-SQUEEZE_LOWER',
             '10s-UT_BOT_V4-BULL_TREND']))
        hub.finalize_shadow_engines()
        return hub

    monkeypatch.setenv(FLAG, "1")
    assert (10, 'RTH') in _mk_hub()._shadow_engines, (
        "flag ON: sub-minute key must get a dedicated canonical-owner shadow "
        "even when a real monitor's pack covers the interp")

    monkeypatch.delenv(FLAG, raising=False)
    assert (10, 'RTH') not in _mk_hub()._shadow_engines, (
        "flag OFF: legacy suppression byte-identical (real monitor covers)")


def test_store_targets_empty_when_flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    import resampled_bar_store as rbs
    assert rbs.default_submin_targets() == []
    assert rbs.is_submin_store_tf("10Sec") is False


def test_store_targets_flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("RORT_SUBMIN_STORE_SYMBOLS", "TSLA")
    import resampled_bar_store as rbs
    assert rbs.is_submin_store_tf("10Sec") is True
    assert rbs.is_submin_store_tf("15Sec") is False, "not in SUBMIN_STORE_TFS"
    assert rbs.is_store_tf("10Sec") is False, (
        "coarse consumers must NEVER admit a sub-minute TF")


def test_store_build_byte_identical_from_synthetic_1sec(monkeypatch):
    """build_window (per-UTC-day WINDOW-REPLACE units) must equal the
    whole-window canonical oracle for the sub-minute TFs — the same proven
    property the coarse store relies on, one octave down."""
    import resampled_bar_store as rbs
    idx = pd.date_range("2026-07-15 08:00:00+00:00", periods=2 * 86400,
                        freq="1s")[:120000]
    px = 300.0 + (pd.Series(range(len(idx))) % 977) * 0.001
    one_sec = pd.DataFrame({"open": px.values, "high": px.values + 0.01,
                            "low": px.values - 0.01, "close": px.values,
                            "volume": [7.3] * len(idx)}, index=idx)
    start, end = idx[0], idx[-1]
    for tf in rbs.SUBMIN_STORE_TFS:
        for sess in ("RTH", "Extended Hours"):
            r = rbs.build_and_compare("TSLA", tf, sess, start, end,
                                      one_min_df=one_sec)
            assert r["match"], (tf, sess, r.get("note"), r["cell_diffs"][:3])


def test_offline_swap_none_when_flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    import services
    df = _h10(T0, 30)
    strat = {"id": 339, "symbol": "TSLA", "timeframe": "30Sec"}
    assert services._submin_secondary_canonical(strat, "10Sec", df, "RTH") is None


def test_offline_swap_ignores_fine_tfs(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    import services
    df = _h10(T0, 30)
    strat = {"id": 339, "symbol": "TSLA", "timeframe": "30Sec"}
    assert services._submin_secondary_canonical(strat, "5Min", df, "RTH") is None
