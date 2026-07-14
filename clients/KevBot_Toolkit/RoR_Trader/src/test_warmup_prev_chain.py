"""Lock RORT_WARMUP_PREV_CHAIN (2026-07-14).

IncrementalIndicatorEngine.warmup() (and therefore the
recompute_from_history full-replay path) never populated the prev-value
chain (`state.prev_values` / `prev_macd_hist` / `prev2_macd_hist`) — only
update_bar() maintains it. Every NON-incremental record derivation
(`_derive_confluence_records` after a warmup seed, the non-RTH session
reload that runs on EVERY close, the RORT_MTF_STATE_REFRESH refresher,
rebroadcast recompute, gap-heal) therefore handed prev={} to the user-pack
interpreter dispatch, which built a 1-row DataFrame → shift(1) = NaN →
shift-based interpreters (MACD_HISTOGRAM_V2) silently emitted NO record.

Observed live 2026-07-13: sid 285's '2m-MACD_HISTOGRAM_V2-H+up' Extended
gate record absent in 204/204 live_gate telemetry events; sid 136's 15m
key EMPTY for 5-20 min after every deploy (ungated fires under fail-open,
silent blocks under RORT_GATE_FAIL_CLOSED).

Locks:
- flag OFF reproduces the legacy behavior exactly (empty prev chain, no
  shift-based record from derive) — the kill-switch byte-identity side
- flag ON: a warmed engine's full state (current + prev chain) equals
  sequential update_bar ingestion of the same bars
- flag ON: _derive_confluence_records() == the incremental on_bar_close
  record set at the same bar (derive ≡ incremental — the fix property)
- flag ON: recompute_from_history full replay re-derives the record

Run:  cd src && ../.venv/bin/pytest test_warmup_prev_chain.py -v
"""

import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import pack_registry  # noqa: E402
pack_registry.scan_and_load_all()

from ralph_engine import _ShadowIndicatorEngine  # noqa: E402
from unified_engine import IncrementalIndicatorEngine  # noqa: E402


PARAMS = {
    'ema_periods': [9, 21],
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'atr_period': 14,
    # macd_histogram_v2 pack params (manifest defaults, explicit)
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9,
}
SHADOW_IND = {'atr', '_user_pack_macd_histogram_v2'}
SHADOW_INTERP = {'MACD_HISTOGRAM_V2'}
TF = 120
T0 = pd.Timestamp("2026-04-14T13:30:00+00:00")
HIST_REC_PREFIX = '2M-MACD_HISTOGRAM_V2-'


@contextmanager
def flag(value: str):
    old = os.environ.get('RORT_WARMUP_PREV_CHAIN')
    os.environ['RORT_WARMUP_PREV_CHAIN'] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('RORT_WARMUP_PREV_CHAIN', None)
        else:
            os.environ['RORT_WARMUP_PREV_CHAIN'] = old


def make_bars(n: int, tf: int = TF, start: pd.Timestamp = T0,
              seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.normal(0.01, 0.08, n))
    idx = pd.date_range(start, periods=n, freq=f'{tf}s', tz='UTC')
    return pd.DataFrame({
        'open': close - 0.02,
        'high': close + 0.05,
        'low': close - 0.05,
        'close': close,
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


def make_shadow() -> _ShadowIndicatorEngine:
    return _ShadowIndicatorEngine(
        TF, set(SHADOW_IND), set(SHADOW_INTERP), dict(PARAMS))


def hist_records(records) -> set:
    return {r for r in records if r.startswith(HIST_REC_PREFIX)}


def test_flag_off_reproduces_missing_record():
    """Legacy (flag OFF): derive-after-warmup drops the shift-based record
    — the live bug — and the prev chain stays empty (byte-identity lock
    for the kill-switch state)."""
    df = make_bars(60)
    with flag('0'):
        shadow = make_shadow()
        shadow.warmup(df)
        assert shadow.indicators.get_prev_values() == {}
        assert shadow.indicators.state.prev_macd_hist == 0.0
        assert shadow.indicators.state.prev2_macd_hist == 0.0
        recs = shadow._derive_confluence_records()
    assert hist_records(recs) == set(), (
        "legacy derive unexpectedly produced a histogram record — the "
        "flag-OFF path is no longer byte-identical")


def test_flag_on_state_equals_sequential_update_bar():
    """Flag ON: warmup(df) leaves the exact state (current values + prev
    chain) of feeding every bar through update_bar on a fresh engine."""
    df = make_bars(60)
    with flag('1'):
        warmed = IncrementalIndicatorEngine(set(SHADOW_IND), dict(PARAMS))
        warmed.warmup(df)

    ref = IncrementalIndicatorEngine(set(SHADOW_IND), dict(PARAMS))
    for i in range(len(df)):
        ref.update_bar({
            'open': float(df['open'].iloc[i]),
            'high': float(df['high'].iloc[i]),
            'low': float(df['low'].iloc[i]),
            'close': float(df['close'].iloc[i]),
            'volume': float(df['volume'].iloc[i]),
            'timestamp': df.index[i],
        })

    for name, got, want in (
        ('current', warmed.get_values(), ref.get_values()),
        ('prev_values', warmed.get_prev_values(), ref.get_prev_values()),
    ):
        assert set(got) == set(want), f"{name}: key sets differ"
        for k, wv in want.items():
            gv = got[k]
            if isinstance(wv, float) and isinstance(gv, float):
                assert gv == wv or abs(gv - wv) < 1e-12, (
                    f"{name}[{k}]: warmed={gv!r} ref={wv!r}")
            else:
                assert gv == wv, f"{name}[{k}]: warmed={gv!r} ref={wv!r}"
    assert warmed.state.prev_macd_hist == ref.state.prev_macd_hist
    assert warmed.state.prev2_macd_hist == ref.state.prev2_macd_hist


def test_flag_on_derive_matches_incremental():
    """Flag ON: the derive path emits exactly the record set the
    incremental on_bar_close path produces at the same bar (derive ≡
    incremental — what the seed/reload/refresher paths need)."""
    df = make_bars(60)

    # Incremental truth: warm on all-but-last (legacy path is fine — the
    # first update_bar populates prev from the warmed current), then close
    # the last bar live.
    with flag('0'):
        live = make_shadow()
        live.warmup(df.iloc[:-1])
        live_recs = live.on_bar_close(bar_dict(df, len(df) - 1))

    with flag('1'):
        derived = make_shadow()
        derived.warmup(df)
        derived_recs = derived._derive_confluence_records()

    assert hist_records(live_recs), (
        "incremental path produced no histogram record — test inputs bad")
    assert derived_recs == live_recs, (
        f"derive != incremental: {sorted(derived_recs)} vs "
        f"{sorted(live_recs)}")


def test_flag_on_recompute_full_replay_rederives_record():
    """Flag ON: the session-reload/refresher path (recompute_confluence →
    recompute_from_history full replay) produces the record that was
    absent live in 204/204 events on sid 285's key."""
    df = make_bars(60)
    with flag('1'):
        shadow = make_shadow()
        shadow.warmup(df.iloc[:10])  # boot-era state, then a full reload
        recs = shadow.recompute_confluence(df)
    assert hist_records(recs), (
        "recompute_confluence produced no histogram record with the flag "
        "on — the reload/refresher path is still dropping it")


def test_flag_off_recompute_still_drops_record():
    """Kill-switch lock: with the flag OFF the reload path behaves exactly
    as before the fix (record absent)."""
    df = make_bars(60)
    with flag('0'):
        shadow = make_shadow()
        shadow.warmup(df.iloc[:10])
        recs = shadow.recompute_confluence(df)
    assert hist_records(recs) == set()
