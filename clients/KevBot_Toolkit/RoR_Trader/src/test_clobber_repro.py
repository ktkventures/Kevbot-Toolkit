"""Faithful production repro for the 2m clobber + the 247afec freeze.

Drives SymbolHub._fanout_to_secondary_builders exactly as production
does: seed_history first (warmup), then DUAL delivery per minute —
ws_agg first, AM ~seconds later — across many minutes.

On CURRENT (reverted) code this documents the CLOBBER: closed 2m bars
end up with single-minute volume (~0.50x) whenever the late AM minute
hits the just-closed period. When the 247afec guard is re-applied, this
repro must (a) show closes STILL HAPPEN (the freeze bug) and (b) volume
correct — the re-land gate.

Run:  cd src && ../.venv/bin/python -m pytest test_clobber_repro.py -v
"""
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from ralph_engine import BarBuilder, SymbolHub, _ShadowIndicatorEngine

PARAMS = {'ema_periods': [9, 21], 'macd_fast_period': 12,
          'macd_slow_period': 26, 'macd_signal_period': 9, 'atr_period': 14}

T0 = pd.Timestamp('2026-06-12T14:00:00+00:00')


def _minute(i, vol=1000.0):
    ts = T0 + timedelta(minutes=i)
    px = 100.0 + i * 0.05
    return {'timestamp': ts.isoformat(), 'open': px, 'high': px + 0.3,
            'low': px - 0.3, 'close': px + 0.1, 'volume': vol}


def _seed_df(n_minutes=30):
    idx = pd.date_range(T0 - timedelta(minutes=n_minutes), periods=n_minutes,
                        freq='1min', tz='UTC')
    px = pd.Series(99.0 + np.arange(n_minutes) * 0.02, index=idx)
    return pd.DataFrame({'open': px, 'high': px + 0.2, 'low': px - 0.2,
                         'close': px + 0.05,
                         'volume': [1000.0] * n_minutes}, index=idx)


def _resample_2m(df):
    return df.resample('120s').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close': 'last', 'volume': 'sum'}).dropna(subset=['open'])


def build_prod_like_hub():
    hub = SymbolHub('TEST')
    b = BarBuilder(tf_seconds=120)
    hub.builders[120] = b
    # production wiring: the fan-out only feeds TFs with a shadow engine
    shadow = _ShadowIndicatorEngine(120, {'ema', 'macd', 'atr'},
                                    {'MACD_LINE'}, dict(PARAMS))
    seed = _resample_2m(_seed_df())
    shadow.warmup(seed)
    hub._shadow_engines[120] = shadow
    # production: warmup seeds the secondary builder's history
    hub.seed_history(120, seed)
    return hub, b


def run_dual_feed(hub, minutes=12):
    """PRODUCTION order: AM lags ~5s behind ws_agg, so at each minute
    boundary the ws_agg path delivers minute i FIRST (possibly closing a
    2m bar), and AM's delivery of minute i-1 lands AFTER that close —
    hitting the closed-period rebroadcast branch. This lag pattern is
    what triggers the clobber in prod."""
    for i in range(minutes):
        hub._fanout_to_secondary_builders(_minute(i), source_label='ws_agg')
        if i >= 1:
            hub._fanout_to_secondary_builders(_minute(i - 1),
                                              source_label='ws')


def test_closes_keep_happening_under_dual_feed():
    """THE FREEZE GATE: 12 minutes of dual-fed 1Min bars must close
    five 2m bars into history (the 6th is forming). 247afec failed this
    in production — any re-land must pass it."""
    hub, b = build_prod_like_hub()
    hist_before = len(b.history)
    run_dual_feed(hub, minutes=12)
    closed = len(b.history) - hist_before
    assert closed == 5, (
        f"expected 5 new closed 2m bars, got {closed} — "
        "secondary closes are broken (the 247afec freeze)")


def test_clobber_documented_or_fixed():
    """Volume of closed 2m bars: correct = 2000 (two minutes). The
    pre-fix clobber leaves some bars at 1000 (single minute). This test
    PASSES either way but PRINTS the state — flip the assert when the
    clobber fix re-lands."""
    hub, b = build_prod_like_hub()
    n0 = len(b.history)
    run_dual_feed(hub, minutes=12)
    vols = [float(v) for v in b.history['volume'].iloc[n0:]]
    print('closed 2m volumes:', vols)
    clobbered = [v for v in vols if v < 1999]
    print(f'clobbered bars: {len(clobbered)}/{len(vols)}')
    # Re-land gate (currently documents the bug; enable with the fix):
    # assert not clobbered
