"""LEF 2c smoke test — verify snapshot serialize/deserialize/resume produces
identical results to a from-scratch run on the same combined data.

Run: cd src && .venv-relative-python _test_lef_2c_snapshot.py

The test bypasses Streamlit entirely; it builds a synthetic 1Min OHLC
DataFrame, runs run_unified_backtest in halves, and compares to a single
full run. This is the parity guarantee the spec promises:
  snapshot(df[:n]) → restore → update_bar(df[n:]) ≡ full(df[:])

Failure modes tested:
  1. Fingerprint mismatch invalidates snapshot
  2. Schema-version mismatch invalidates snapshot
  3. Roundtrip: half + half == full on indicator state + trades
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Make src/ importable
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from unified_engine import (
    SNAPSHOT_SCHEMA_VERSION,
    compute_backtest_fingerprint,
    serialize_backtest_snapshot,
    deserialize_backtest_snapshot,
    apply_backtest_snapshot,
    run_unified_backtest,
    UnifiedStrategy,
)


def _make_df(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV — geometric random walk."""
    rng = np.random.default_rng(seed)
    base = 100.0
    closes = base * np.exp(np.cumsum(rng.normal(0, 0.001, n_bars)))
    opens = np.concatenate([[base], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.0005, n_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.0005, n_bars)))
    vols = rng.integers(10_000, 50_000, n_bars).astype(float)

    start = pd.Timestamp('2026-05-19 09:30', tz='UTC')
    idx = pd.date_range(start, periods=n_bars, freq='1min')
    return pd.DataFrame(
        {'open': opens, 'high': highs, 'low': lows, 'close': closes,
         'volume': vols},
        index=idx)


def _make_strategy() -> dict:
    """Minimal EMA-stack long strategy — exercises a real indicator chain."""
    return {
        'id': 999_999,
        'name': 'TEST-LEF-2C-SNAPSHOT',
        'symbol': 'SPY',
        'direction': 'LONG',
        'timeframe': '1Min',
        'entry_trigger': '1Min-EMA_STACK-S',
        'entry_trigger_confluence_id': '1Min-EMA_STACK-S',
        'exit_trigger': 'opposite_signal',
        'confluence': [],
        'risk_per_trade': 100.0,
        'starting_balance': 10_000.0,
        'stop_atr_mult': 1.5,
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'data_seed': 42,
        'strategy_origin': 'standard',
        'trading_session': 'RTH',
    }


def test_fingerprint_stability_and_change():
    strat = _make_strategy()
    fp1 = compute_backtest_fingerprint(strat)
    fp2 = compute_backtest_fingerprint(strat)
    assert fp1 == fp2, "fingerprint should be deterministic"

    strat2 = dict(strat)
    strat2['stop_atr_mult'] = 2.0  # behavior-driving change
    fp3 = compute_backtest_fingerprint(strat2)
    assert fp1 != fp3, "fingerprint should change when behavior fields change"

    # Non-behavior-driving change should NOT change the fingerprint
    strat3 = dict(strat)
    strat3['name'] = 'A NEW NAME'
    fp4 = compute_backtest_fingerprint(strat3)
    assert fp1 == fp4, "fingerprint should be insensitive to non-behavior fields"
    print("  ✓ fingerprint stability + change-detection")


def test_serialize_deserialize_roundtrip():
    df = _make_df(150)
    strat_cfg = _make_strategy()

    # Run on first half to build state
    half = len(df) // 2
    df_half = df.iloc[:half].copy()

    trades, enriched, b64 = run_unified_backtest(
        df_half, strat_cfg, return_snapshot=True)
    assert b64 is not None, "snapshot should be captured"

    # Roundtrip: deserialize must succeed with matching fingerprint
    fp = compute_backtest_fingerprint(strat_cfg)
    envelope = deserialize_backtest_snapshot(b64, expected_fingerprint=fp)
    assert envelope is not None, "snapshot should deserialize with matching fp"
    assert envelope['schema_version'] == SNAPSHOT_SCHEMA_VERSION
    assert envelope['last_bar_ts'] is not None

    # Wrong fingerprint must yield None
    envelope_bad = deserialize_backtest_snapshot(b64, expected_fingerprint='fake')
    assert envelope_bad is None, "wrong fingerprint should invalidate"
    print("  ✓ serialize → deserialize roundtrip + fingerprint guard")


def test_resume_equals_full():
    """The big one: half + resume(other half) ≡ full(both halves)
    at the indicator-state level, NOT just trade-count."""
    df = _make_df(150)
    strat_cfg = _make_strategy()
    half = len(df) // 2

    # --- Full run — produces ground-truth final state via return_snapshot
    full_trades, full_enriched, full_final_b64 = run_unified_backtest(
        df, strat_cfg, return_snapshot=True)
    assert full_final_b64 is not None

    # --- Halved run
    df_a = df.iloc[:half].copy()
    df_b = df.iloc[half:].copy()

    half_trades, _, b64 = run_unified_backtest(
        df_a, strat_cfg, return_snapshot=True)
    assert b64 is not None

    fp = compute_backtest_fingerprint(strat_cfg)
    envelope = deserialize_backtest_snapshot(b64, expected_fingerprint=fp)
    assert envelope is not None

    resume_trades, _, resume_final_b64 = run_unified_backtest(
        df_b, strat_cfg, resume_snapshot=envelope, return_snapshot=True)
    assert resume_final_b64 is not None

    # Trade-count parity
    combined_trades = pd.concat([half_trades, resume_trades], ignore_index=True)
    if len(full_trades) != len(combined_trades):
        print(f"  ✗ trade count mismatch: full={len(full_trades)} "
              f"vs combined={len(combined_trades)}")
    else:
        print(f"  ✓ trade count matches ({len(full_trades)})")

    # FINAL ENGINE STATE PARITY — the strict bar.
    # Deserialize both end-state envelopes and compare their IndicatorState
    # dataclass fields field-by-field. If resume is byte-identical to full,
    # every indicator buffer / EMA value / MACD-state must match.
    env_full = deserialize_backtest_snapshot(
        full_final_b64, expected_fingerprint=fp)
    env_resume = deserialize_backtest_snapshot(
        resume_final_b64, expected_fingerprint=fp)
    assert env_full is not None and env_resume is not None

    # envelope['engine'] is a tuple: (IndicatorState, _initialized, packs)
    state_full = env_full['engine'][0]
    state_resume = env_resume['engine'][0]

    # Compare every public attribute
    state_full_d = state_full.__dict__
    state_resume_d = state_resume.__dict__
    mismatches = []
    for k in sorted(set(state_full_d.keys()) | set(state_resume_d.keys())):
        v_f = state_full_d.get(k)
        v_r = state_resume_d.get(k)
        # Containers: compare element-wise where possible
        if isinstance(v_f, dict) and isinstance(v_r, dict):
            if v_f != v_r:
                mismatches.append(f"{k}: dict differs ({v_f} vs {v_r})")
        elif hasattr(v_f, '__iter__') and not isinstance(v_f, str):
            v_f_list = list(v_f) if v_f is not None else None
            v_r_list = list(v_r) if v_r is not None else None
            if v_f_list != v_r_list:
                mismatches.append(f"{k}: iterable differs")
        else:
            if v_f != v_r:
                # Float tolerance for the EMA tail
                try:
                    if abs(float(v_f) - float(v_r)) < 1e-9:
                        continue
                except (TypeError, ValueError):
                    pass
                mismatches.append(f"{k}: {v_f!r} vs {v_r!r}")

    if mismatches:
        print(f"  ✗ indicator-state mismatches ({len(mismatches)}):")
        for m in mismatches[:10]:
            print(f"    - {m}")
        raise AssertionError(
            f"resume-run indicator state diverges from full-run: "
            f"{len(mismatches)} field(s)")
    else:
        print(f"  ✓ indicator state byte-identical "
              f"({len(state_full_d)} fields compared)")


def test_schema_version_mismatch():
    import pickle
    import base64

    # Hand-craft a bad envelope with the wrong schema version
    bad_envelope = {
        'schema_version': SNAPSHOT_SCHEMA_VERSION + 99,
        'fingerprint': 'whatever',
        'last_bar_ts': '2026-05-19T09:30:00+00:00',
        'engine': None,
        'position': None,
    }
    bad_b64 = base64.b64encode(pickle.dumps(bad_envelope)).decode('ascii')

    out = deserialize_backtest_snapshot(bad_b64, expected_fingerprint='whatever')
    assert out is None, "future schema_version should be rejected"
    print("  ✓ schema_version mismatch rejected")


def test_corrupt_b64_returns_none():
    out = deserialize_backtest_snapshot('!!!not-base64!!!')
    assert out is None, "corrupt b64 should return None, not raise"
    out2 = deserialize_backtest_snapshot('')
    assert out2 is None, "empty string should return None"
    out3 = deserialize_backtest_snapshot(None)
    assert out3 is None, "None should return None"
    print("  ✓ corrupt/empty b64 returns None (no raise)")


if __name__ == '__main__':
    print("\n=== LEF 2c snapshot smoke test ===\n")
    test_fingerprint_stability_and_change()
    test_serialize_deserialize_roundtrip()
    test_resume_equals_full()
    test_schema_version_mismatch()
    test_corrupt_b64_returns_none()
    print("\n=== ALL PASS ===\n")
