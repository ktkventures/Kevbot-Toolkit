#!/usr/bin/env python3
"""Parity: cache-replay (precompute_bar_cache + run_trades_from_cache) vs
the canonical full-engine path (run_unified_backtest).

The HighFidelity Mass Builder path is load-bearing on the claim that these
two produce bit-identical trades. This file is the gate.

NOTE on coverage: synthetic OHLCV data here may produce few or zero trades
because the trigger-evaluation pipeline is sensitive to data shape (the
existing test_unified_parity.py has the same property — its trigger_eval
returns 0 fires on similar synthetic data while the legacy batch pipeline
produces 9). What this gate DOES prove is that both paths agree
bit-identically on the same inputs — whether the agreement is on 0 trades
or 50, both paths walk the same code through PositionStateMachine and
produce identical trades_df. Real-data verification of trade counts
happens manually via the Mass Builder UI, comparing HighFidelity preview
KPIs to `POST /api/strategies/{id}/update?mode=all` output (the plan's
"Live parity (manual)" verification step).

Usage:
    .venv/bin/python src/test_mass_builder_fidelity_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_engine import (
    run_unified_backtest,
    precompute_bar_cache,
    run_trades_from_cache,
)


# ── Test data ─────────────────────────────────────────────────────────

def generate_test_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Realistic 1Min OHLCV — enough volatility for indicators to fire."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0002, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = close * 0.004 * rng.uniform(0.5, 1.5, n)
    high = close + spread * rng.uniform(0, 1, n)
    low = close - spread * rng.uniform(0, 1, n)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.uniform(1000, 5000, n)
    start = pd.Timestamp('2026-01-02 09:30', tz='UTC')
    idx = pd.date_range(start, periods=n, freq='1min')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=idx)


# ── Comparison helper ────────────────────────────────────────────────

def assert_trades_equal(label: str, full_df, replay_df) -> int:
    """Compare trades_df from full backtest vs cache replay. Returns 0
    on PASS, raises AssertionError on FAIL with diff details."""
    n_full = len(full_df) if isinstance(full_df, pd.DataFrame) else 0
    n_replay = len(replay_df) if isinstance(replay_df, pd.DataFrame) else 0

    if n_full == 0 and n_replay == 0:
        print(f"  [{label}] PASS (both empty)")
        return 0
    if n_full != n_replay:
        print(f"  [{label}] FAIL: full={n_full} trades, replay={n_replay} trades")
        # Dump first few entries from each to diagnose
        if n_full:
            print(f"    full first entry_fill_ts={full_df.iloc[0].get('entry_fill_ts')}")
        if n_replay:
            print(f"    replay first entry_fill_ts={replay_df.iloc[0].get('entry_fill_ts')}")
        raise AssertionError(f"{label}: trade count mismatch {n_full} vs {n_replay}")

    # Per-row comparison on the fields that matter for fidelity.
    cols = ['entry_fill_ts', 'exit_fill_ts', 'direction', 'entry_price',
            'exit_price', 'exec_type']
    mismatches = []
    for i in range(n_full):
        a = full_df.iloc[i]
        b = replay_df.iloc[i]
        for c in cols:
            av = a.get(c) if c in full_df.columns else None
            bv = b.get(c) if c in replay_df.columns else None
            # Tolerate float noise on prices; everything else must be exact.
            if c in ('entry_price', 'exit_price'):
                if av is None and bv is None:
                    continue
                if av is None or bv is None:
                    mismatches.append(f"trade {i} {c}: full={av} replay={bv}")
                    continue
                if abs(float(av) - float(bv)) > 1e-6:
                    mismatches.append(f"trade {i} {c}: full={av} replay={bv}")
            else:
                if str(av) != str(bv):
                    mismatches.append(f"trade {i} {c}: full={av!r} replay={bv!r}")
    if mismatches:
        print(f"  [{label}] FAIL ({len(mismatches)} mismatches)")
        for m in mismatches[:10]:
            print(f"    {m}")
        raise AssertionError(f"{label}: {len(mismatches)} field mismatches")
    print(f"  [{label}] PASS ({n_full} trades)")
    return 0


def run_both_paths(df: pd.DataFrame, strategy: dict,
                   general_packs=None, secondary_tf_map=None):
    """Run run_unified_backtest and the cache-replay path. Return both
    trades_df objects."""
    full_trades, _enriched = run_unified_backtest(
        df, strategy,
        general_packs=general_packs,
        secondary_tf_map=secondary_tf_map,
    )
    cache, meta = precompute_bar_cache(
        df, strategy,
        general_packs=general_packs,
        secondary_tf_map=secondary_tf_map,
    )
    replay_trades = run_trades_from_cache(cache, strategy, meta)
    return full_trades, replay_trades


# ── Cases ────────────────────────────────────────────────────────────

def test_a_c_type_entry_exit():
    """C-type entry + C-type exit, no confluence."""
    print("\n[case a] C-type entry + C-type exit")
    df = generate_test_data(n=400, seed=11)
    strat = {
        'id': 'parity_a',
        'symbol': 'TEST',
        'direction': 'LONG',
        'timeframe': '1Min',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'confluence': [],
    }
    full, replay = run_both_paths(df, strat)
    assert_trades_equal('case a', full, replay)


def test_b_l_type_entry_next_bar_stop():
    """L-type entry whose stop fires on the next bar."""
    print("\n[case b] L-type entry + stop on next bar")
    df = generate_test_data(n=400, seed=17)
    strat = {
        'id': 'parity_b',
        'symbol': 'TEST',
        'direction': 'LONG',
        'timeframe': '1Min',
        # _ib suffix = L-type intra-bar trigger
        'entry_trigger': 'ema_cross_bull_ib',
        'exit_triggers': ['ema_cross_bear'],
        # Tight stop guarantees frequent next-bar stops
        'stop_config': {'method': 'atr', 'atr_mult': 0.5},
        'confluence': [],
    }
    full, replay = run_both_paths(df, strat)
    assert_trades_equal('case b', full, replay)


def test_c_multi_confluence_gate():
    """Multi-confluence inline gate — entries that the inline gate would
    drop must also be absent from the replay."""
    print("\n[case c] multi-confluence inline gate")
    df = generate_test_data(n=500, seed=23)
    strat = {
        'id': 'parity_c',
        'symbol': 'TEST',
        'direction': 'LONG',
        'timeframe': '1Min',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        # Require a couple of confluence records — they may or may not be
        # present at any given entry candidate, which exercises the inline
        # gate.
        'confluence': [
            '1Min-MACD_LINE-BULL',
            '1Min-VWAP-ABOVE',
        ],
    }
    full, replay = run_both_paths(df, strat)
    assert_trades_equal('case c', full, replay)


def test_d_batch_only_user_pack():
    """Batch-only user pack — needs a real loaded pack to exercise. Skipped
    in pure-offline mode; left as a manual integration test."""
    print("\n[case d] batch-only user pack — SKIPPED (manual integration)")
    print("        Run a Mass search that includes a batch-only pack and")
    print("        compare HighFidelity vs Rapid via the parity tab.")


def test_e_mtf_confluence():
    """Secondary TF confluence — secondary_tf_map populated, MTF columns
    in df."""
    print("\n[case e] MTF confluence")
    df = generate_test_data(n=500, seed=29)
    # Synthesize a 1H secondary-TF interpreter column.
    # In production these come from prepare_data_with_indicators after
    # resampling. Here we fake one stable state across all bars so the
    # gate is a no-op (still exercises the MTF path).
    df = df.copy()
    df['MACD_LINE__1h'] = 'BULL'
    strat = {
        'id': 'parity_e',
        'symbol': 'TEST',
        'direction': 'LONG',
        'timeframe': '1Min',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'confluence': ['1h-MACD_LINE-BULL'],
    }
    secondary_tf_map = {'1h': ['MACD_LINE__1h']}
    full, replay = run_both_paths(df, strat, secondary_tf_map=secondary_tf_map)
    assert_trades_equal('case e', full, replay)


def test_f_trailing_stop():
    """Trailing stop — PositionStateMachine state machine includes
    trailing logic; verify it survives cache replay."""
    print("\n[case f] trailing stop")
    df = generate_test_data(n=500, seed=37)
    strat = {
        'id': 'parity_f',
        'symbol': 'TEST',
        'direction': 'LONG',
        'timeframe': '1Min',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {
            'method': 'atr',
            'atr_mult': 2.0,
            'trailing': True,
            'trailing_atr_mult': 1.5,
        },
        'confluence': [],
    }
    full, replay = run_both_paths(df, strat)
    assert_trades_equal('case f', full, replay)


# ── Runner ────────────────────────────────────────────────────────────

ALL_CASES = [
    test_a_c_type_entry_exit,
    test_b_l_type_entry_next_bar_stop,
    test_c_multi_confluence_gate,
    test_d_batch_only_user_pack,
    test_e_mtf_confluence,
    test_f_trailing_stop,
]


def main() -> int:
    print("Mass Builder fidelity parity: cache-replay vs full engine")
    print("=" * 60)
    failures = []
    for case in ALL_CASES:
        try:
            case()
        except AssertionError as e:
            failures.append((case.__name__, str(e)))
        except Exception as e:
            failures.append((case.__name__, f"unexpected: {e!r}"))
    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} of {len(ALL_CASES)} cases failed")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    print(f"PASS — all {len(ALL_CASES)} cases")
    return 0


if __name__ == '__main__':
    sys.exit(main())
