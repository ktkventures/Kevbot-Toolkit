"""RORT_HOTRELOAD_BOOT_PARITY (Brandon audit P1-hot-reload) — flag-gated
regression gate locking BOTH states:
  * flag ON  → hot-reload warms like a clean boot (TF-scaled _load_warmup_df)
               and a config-EDIT finalizes + warms NEWLY-added gate shadows
  * flag OFF → legacy behavior byte-identical (flat days=7; edit leaves a new
               gate shadow-less until restart)

    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_hotreload_boot_parity.py
"""
from __future__ import annotations

from types import SimpleNamespace

import worker as W

FLAG = "RORT_HOTRELOAD_BOOT_PARITY"


def test_flag_helper_runtime(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert W._hotreload_boot_parity() is False
    monkeypatch.setenv(FLAG, "1")
    assert W._hotreload_boot_parity() is True


def test_warmup_df_flag_off_uses_legacy_flat_load(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    calls = {}

    def fake_lmd(sym, days=None, timeframe=None, feed=None, session=None):
        calls.update(sym=sym, days=days, timeframe=timeframe, session=session)
        return "LEGACY_DF"

    import data_loader
    monkeypatch.setattr(data_loader, "load_market_data", fake_lmd)
    out = W._hotreload_warmup_df("SPY", 86400, "1Day", "RTH", "sip")
    assert out == "LEGACY_DF"
    assert calls == {"sym": "SPY", "days": 7, "timeframe": "1Day",
                     "session": "RTH"}, "flag OFF must be the flat days=7 load"


def test_warmup_df_flag_on_uses_boot_loader(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    calls = {}

    def fake_boot_loader(sym, tf_seconds, session):
        calls.update(sym=sym, tf_seconds=tf_seconds, session=session)
        return "BOOT_DF"

    import ralph_engine
    monkeypatch.setattr(ralph_engine, "_load_warmup_df", fake_boot_loader)
    out = W._hotreload_warmup_df("SPY", 86400, "1Day", "RTH", "sip")
    assert out == "BOOT_DF"
    assert calls == {"sym": "SPY", "tf_seconds": 86400, "session": "RTH"}, (
        "flag ON must use the SAME TF-scaled loader boot uses")


def _stub_hub_and_monitor():
    events = []

    def shadow(initialized):
        s = SimpleNamespace(indicators=SimpleNamespace(_initialized=initialized))
        s.warmup = lambda df, _s=s: events.append(("warmup", _s))
        return s

    uninit = shadow(False)
    init = shadow(True)
    hub = SimpleNamespace(
        _shadow_engines={(86400, "RTH"): uninit, (900, "RTH"): init})
    hub.finalize_shadow_engines = lambda: events.append(("finalize",))
    hub.seed_mtf_from_shadow = lambda tf, sess: events.append(("seed", tf, sess))
    monitor = SimpleNamespace(session="RTH")
    return hub, monitor, events, uninit, init


def test_edit_finalize_flag_off_is_noop(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    hub, monitor, events, *_ = _stub_hub_and_monitor()
    assert W._hotreload_finalize_new_shadows(hub, monitor, "SPY") == 0
    assert events == [], "flag OFF: edit path must not touch shadows (legacy)"


def test_edit_finalize_flag_on_warms_only_new_shadows(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    import ralph_engine
    monkeypatch.setattr(ralph_engine, "_load_warmup_df",
                        lambda sym, tf, sess: [1, 2, 3])
    hub, monitor, events, uninit, init = _stub_hub_and_monitor()
    warmed = W._hotreload_finalize_new_shadows(hub, monitor, "SPY")
    assert warmed == 1
    assert ("finalize",) in events, "must finalize (idempotent) so an edit-added gate gets a shadow"
    assert ("warmup", uninit) in events, "the NEW (uninitialized) shadow must be warmed"
    assert ("warmup", init) not in events, "already-initialized shadows untouched (bounded work)"
    assert ("seed", 86400, "RTH") in events, "warmed state must publish into the mtf gate dict"


def test_edit_finalize_none_hub_safe(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    assert W._hotreload_finalize_new_shadows(None, SimpleNamespace(session="RTH"), "SPY") == 0
