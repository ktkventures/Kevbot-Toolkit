"""RORT_MTF_PB_PREV_EPOCH (board #121) — epoch-scoped PB-defer prev.

Replays the recorded 2026-07-24 19:00Z NVDA sequence (sids 341/343/344)
through the REAL SymbolHub._publish_mtf on a duck-typed hub:

  15:00:16  1h close publish   SML   (epoch eff=16:00... any prior epoch)
  19:00:11  flush force-close  MSL   (epoch eff=19:00)  prev:=SML
  19:00:25  fan-out correction MSL'  (epoch eff=19:00 — SAME logical bar)
            legacy: prev:=MSL  -> defer serves MSL -> live missed the entry
            flag ON: prev stays SML -> defer serves SML -> entry fires
  19:00:25+ 1Min monitors evaluate the 18:59 bar (cutoff 18:59 < eff 19:00
            -> defer serves prev)

Run: cd src && ../.venv/bin/python -m pytest test_mtf_pb_prev_epoch.py -q
"""
import types

import pandas as pd
import pytest

import ralph_engine
from ralph_engine import SymbolHub

SML = {'1H-EMA_STACK_V2-SML', '1H-RSI_ZONES_2-NEUTRAL_BULLISH',
       '1H-SUPERTREND-BEAR_TRENDING'}
MSL_V1 = {'1H-EMA_STACK_V2-MSL', '1H-RSI_ZONES_2-NEUTRAL_BEARISH',
          '1H-SUPERTREND-BEAR_TRENDING'}
# corrected republish after the late 18:59 fold-in (same records here, as
# on 07-24 — but use a distinct object to prove no aliasing assumptions)
MSL_V2 = set(MSL_V1)

KEY = (3600, 'RTH')
BAR_17 = pd.Timestamp('2026-07-24 17:00:00+00:00')   # closes 18:00
BAR_18 = pd.Timestamp('2026-07-24 18:00:00+00:00')   # closes 19:00
BAR_19 = pd.Timestamp('2026-07-24 19:00:00+00:00')   # closes 20:00


def _hub():
    """Duck-typed hub: _publish_mtf touches only the three dicts."""
    h = types.SimpleNamespace()
    h._mtf_confluence = {}
    h._mtf_confluence_prev = {}
    h._mtf_confluence_effective_from = {}
    return h


def _publish(h, records, bar_start):
    SymbolHub._publish_mtf(h, KEY, set(records),
                           closed_bar_start_ts=bar_start)


def _eff(ts_bar_start):
    return ts_bar_start.timestamp() + 3600


def _defer_selected(h, primary_bar_start):
    """The step-3c selection rule from StrategyMonitor.on_bar_close:
    current records merge iff effective_from <= primary bar start,
    else the key's prev records merge."""
    cutoff = primary_bar_start.timestamp()
    if h._mtf_confluence_effective_from.get(KEY, 0.0) > cutoff:
        return h._mtf_confluence_prev.get(KEY, set())
    return h._mtf_confluence[KEY]


@pytest.fixture(autouse=True)
def _defer_on(monkeypatch):
    monkeypatch.setattr(ralph_engine, 'MTF_PB_DEFER', True)


def _replay_0724(h):
    _publish(h, SML, BAR_17)       # steady state since 15:00
    _publish(h, MSL_V1, BAR_18)    # 19:00:11 flush force-close (no 18:59)
    _publish(h, MSL_V2, BAR_18)    # 19:00:25 fan-out correction republish


def test_legacy_clobber_reproduces_0724_miss(monkeypatch):
    monkeypatch.setattr(ralph_engine, 'MTF_PB_PREV_EPOCH', False)
    h = _hub()
    _replay_0724(h)
    # legacy: the same-epoch republish rotated prev to the flipped state
    assert h._mtf_confluence_prev[KEY] == MSL_V1
    # ...so the delayed 18:59 primary bar merges MSL: the recorded miss
    sel = _defer_selected(h, pd.Timestamp('2026-07-24 18:59:00+00:00'))
    assert '1H-EMA_STACK_V2-SML' not in sel


def test_epoch_prev_serves_prior_epoch_to_delayed_bar(monkeypatch):
    monkeypatch.setattr(ralph_engine, 'MTF_PB_PREV_EPOCH', True)
    h = _hub()
    _replay_0724(h)
    # same-epoch republish refreshed current but left prev intact
    assert h._mtf_confluence[KEY] == MSL_V2
    assert h._mtf_confluence_prev[KEY] == SML
    assert h._mtf_confluence_effective_from[KEY] == _eff(BAR_18)
    # the delayed 18:59 primary bar now merges the prior epoch (SML):
    # gate opens, entry fires — matches backtest/algo
    sel = _defer_selected(h, pd.Timestamp('2026-07-24 18:59:00+00:00'))
    assert '1H-EMA_STACK_V2-SML' in sel
    # a 19:00 primary bar merges the current epoch (MSL) — unchanged
    sel = _defer_selected(h, BAR_19)
    assert '1H-EMA_STACK_V2-MSL' in sel


def test_epoch_advance_still_rotates(monkeypatch):
    monkeypatch.setattr(ralph_engine, 'MTF_PB_PREV_EPOCH', True)
    h = _hub()
    _replay_0724(h)
    nxt = {'1H-EMA_STACK_V2-MLS'}
    _publish(h, nxt, BAR_19)       # 20:00 close: genuine epoch advance
    assert h._mtf_confluence_prev[KEY] == MSL_V2
    assert h._mtf_confluence[KEY] == nxt
    assert h._mtf_confluence_effective_from[KEY] == _eff(BAR_19)


def test_first_publish_rotates_empty_prev(monkeypatch):
    monkeypatch.setattr(ralph_engine, 'MTF_PB_PREV_EPOCH', True)
    h = _hub()
    _publish(h, SML, BAR_17)
    assert h._mtf_confluence_prev[KEY] == set()
    assert h._mtf_confluence[KEY] == SML


def test_flag_combo_matrix_no_ts_publish(monkeypatch):
    """No-ts publishes (warmup seed) keep eff=0.0 in BOTH modes; a later
    real-close publish rotates in both modes (0.0 -> real eff)."""
    for epoch_flag in (False, True):
        monkeypatch.setattr(ralph_engine, 'MTF_PB_PREV_EPOCH', epoch_flag)
        h = _hub()
        SymbolHub._publish_mtf(h, KEY, set(SML))          # seed, no ts
        assert h._mtf_confluence_effective_from[KEY] == 0.0
        _publish(h, MSL_V1, BAR_18)
        assert h._mtf_confluence_prev[KEY] == SML
        assert h._mtf_confluence_effective_from[KEY] == _eff(BAR_18)
