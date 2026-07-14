"""Lock RORT_SHADOW_RETRUE_FORCE_FULL (2026-07-14).

`_ShadowIndicatorEngine.recompute_confluence` — the primitive under the
MTF state refresher, the non-RTH session reloads, and the coarse-RTH
CLASS-1a reload — calls `recompute_from_history(df)`, whose default path
takes the O(1) snapshot fast-path whenever a pre-bar snapshot exists,
WITHOUT checking that the snapshot aligns with the reload df. On a warmed
live shadow that means every "clean re-true" actually restores the LIVE
lineage and re-applies only the df's last bar — drifted states survive
re-true after re-true (observed live 2026-07-14: TSLA 5m SWING held a
wrong state for hours across 120s refresher cycles while the same df
full-replays to the canonical state).

Locks:
- flag OFF: with a snapshot present, recompute_confluence PRESERVES the
  live lineage (kill-switch byte-identity — documents the legacy bug)
- flag ON: recompute_confluence == a fresh engine's full replay of the
  same df, regardless of lineage/snapshots
- flag ON: cold shadow (no snapshot) unchanged — full replay either way

Run:  cd src && ../.venv/bin/pytest test_shadow_retrue_force_full.py -v
"""

import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from ralph_engine import _ShadowIndicatorEngine  # noqa: E402

PARAMS = {
    'ema_periods': [9, 21],
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'atr_period': 14,
}
REQUIRED = {'ema', 'macd', 'atr'}
TF = 300
T0 = pd.Timestamp("2026-04-14T13:30:00+00:00")


@contextmanager
def flag(value: str):
    old = os.environ.get('RORT_SHADOW_RETRUE_FORCE_FULL')
    os.environ['RORT_SHADOW_RETRUE_FORCE_FULL'] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('RORT_SHADOW_RETRUE_FORCE_FULL', None)
        else:
            os.environ['RORT_SHADOW_RETRUE_FORCE_FULL'] = old


def make_bars(n: int, base: float = 100.0, seed: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = base + np.cumsum(rng.normal(0, 0.05, n))
    idx = pd.date_range(T0, periods=n, freq=f'{TF}s', tz='UTC')
    return pd.DataFrame({
        'open': close - 0.02, 'high': close + 0.05,
        'low': close - 0.05, 'close': close,
        'volume': rng.randint(1000, 5000, n).astype(float),
    }, index=idx)


def bar_dict(df: pd.DataFrame, i: int) -> dict:
    row = df.iloc[i]
    return {
        'timestamp': df.index[i].isoformat(),
        'open': float(row['open']), 'high': float(row['high']),
        'low': float(row['low']), 'close': float(row['close']),
        'volume': float(row['volume']),
    }


def warmed_live_shadow(df_lineage: pd.DataFrame) -> _ShadowIndicatorEngine:
    """A shadow with LIVE lineage: warmup on most bars, then the last two
    consumed via on_bar_close so `_pre_bar_snapshot` exists (mirrors the
    production shadow after any incremental close)."""
    sh = _ShadowIndicatorEngine(TF, set(REQUIRED), {'MACD_LINE'}, dict(PARAMS))
    sh.warmup(df_lineage.iloc[:-2])
    sh.indicators._snapshot_enabled = True
    sh.on_bar_close(bar_dict(df_lineage, len(df_lineage) - 2))
    sh.on_bar_close(bar_dict(df_lineage, len(df_lineage) - 1))
    assert sh.indicators._pre_bar_snapshot is not None
    return sh


def fresh_full_replay_values(df: pd.DataFrame) -> dict:
    sh = _ShadowIndicatorEngine(TF, set(REQUIRED), {'MACD_LINE'}, dict(PARAMS))
    sh.recompute_confluence(df)  # cold: no snapshot → full replay either way
    return dict(sh.indicators.get_values())


def test_flag_off_retrue_preserves_lineage():
    """Legacy lock: on a warmed live shadow, recompute_confluence over a
    DIFFERENT df leaves the state near the lineage, not the df — the
    snapshot fast-path wins (this is the bug, locked for the kill-switch)."""
    df_lineage = make_bars(120, base=100.0)
    # A materially different reload df (shifted +5.0) sharing the same
    # index — a full replay would land ~5.0 higher on every EMA/MACD.
    df_reload = df_lineage.copy()
    for c in ('open', 'high', 'low', 'close'):
        df_reload[c] = df_reload[c] + 5.0

    with flag('0'):
        sh = warmed_live_shadow(df_lineage)
        sh.recompute_confluence(df_reload)
        got = sh.indicators.get_values()

    want_full = fresh_full_replay_values(df_reload)
    # Lineage-preserved EMA sits ~5.0 below the full-replay EMA (only the
    # last +5.0 bar was applied onto the restored lineage state).
    assert abs(got['ema_21'] - want_full['ema_21']) > 1.0, (
        "flag OFF: re-true unexpectedly matched the full replay — the "
        "legacy fast-path behavior changed (kill-switch no longer "
        "byte-identical)")


def test_flag_on_retrue_equals_fresh_full_replay():
    """Fix lock: flag ON, the same re-true equals a fresh engine's full
    replay of the reload df — lineage and snapshots are discarded."""
    df_lineage = make_bars(120, base=100.0)
    df_reload = df_lineage.copy()
    for c in ('open', 'high', 'low', 'close'):
        df_reload[c] = df_reload[c] + 5.0

    with flag('1'):
        sh = warmed_live_shadow(df_lineage)
        recs = sh.recompute_confluence(df_reload)
        got = sh.indicators.get_values()

    want = fresh_full_replay_values(df_reload)
    for k, wv in want.items():
        gv = got.get(k)
        if isinstance(wv, float) and isinstance(gv, float):
            assert abs(gv - wv) < 1e-9, f"{k}: retrue={gv!r} fresh={wv!r}"
    # And the records derive from the clean state
    fresh_sh = _ShadowIndicatorEngine(
        TF, set(REQUIRED), {'MACD_LINE'}, dict(PARAMS))
    assert recs == fresh_sh.recompute_confluence(df_reload)


def test_flag_on_cold_shadow_unchanged():
    """Cold shadow (no snapshot): flag ON and OFF agree (full replay was
    already the only path)."""
    df = make_bars(80)
    with flag('0'):
        a = _ShadowIndicatorEngine(TF, set(REQUIRED), {'MACD_LINE'},
                                   dict(PARAMS))
        recs_off = a.recompute_confluence(df)
        vals_off = dict(a.indicators.get_values())
    with flag('1'):
        b = _ShadowIndicatorEngine(TF, set(REQUIRED), {'MACD_LINE'},
                                   dict(PARAMS))
        recs_on = b.recompute_confluence(df)
        vals_on = dict(b.indicators.get_values())
    assert recs_off == recs_on
    for k, v in vals_off.items():
        if isinstance(v, float):
            assert abs(vals_on[k] - v) < 1e-12
