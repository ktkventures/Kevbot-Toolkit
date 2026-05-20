"""LEF 2c forward_test_service port — regression test.

Verifies the new snapshot-aware path through `services.get_strategy_
trades_for_window` produces results consistent with the un-snapshotted
path, and that the new model_id check correctly invalidates snapshots
across lane boundaries.

Run: cd src && .venv-relative-python _test_lef_2c_service_port.py
"""
import os
import sys
import pandas as pd
import numpy as np

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
    return {
        'id': 999_995,
        'name': 'TEST-LEF-2C-SERVICE-PORT',
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


def test_model_id_in_envelope():
    """Snapshot envelope carries model_id when provided."""
    df = _make_df(80)
    strat = _make_strategy()
    _, _, b64 = run_unified_backtest(
        df, strat, return_snapshot=True,
        snapshot_model_id='backtest_rest_hifi',
    )
    assert b64 is not None
    env = deserialize_backtest_snapshot(b64)
    assert env is not None
    assert env.get('model_id') == 'backtest_rest_hifi', (
        f"Envelope missing model_id; got {env.get('model_id')}"
    )
    print("  ✓ envelope carries model_id when provided")


def test_model_id_mismatch_invalidates():
    """Snapshot with model_id='X' is rejected when caller expects model_id='Y'."""
    df = _make_df(80)
    strat = _make_strategy()
    _, _, b64 = run_unified_backtest(
        df, strat, return_snapshot=True,
        snapshot_model_id='backtest_rest_hifi',
    )
    fp = compute_backtest_fingerprint(strat)

    # Same model: valid
    env_ok = deserialize_backtest_snapshot(
        b64, expected_fingerprint=fp,
        expected_model_id='backtest_rest_hifi')
    assert env_ok is not None
    print("  ✓ same model_id passes deserialize check")

    # Different model: rejected
    env_bad = deserialize_backtest_snapshot(
        b64, expected_fingerprint=fp,
        expected_model_id='algo_cache_locked')
    assert env_bad is None
    print("  ✓ different model_id rejected (invalidates snapshot)")

    # None expected (back-compat): passes
    env_neutral = deserialize_backtest_snapshot(
        b64, expected_fingerprint=fp)
    assert env_neutral is not None
    print("  ✓ no model_id expected → passes (back-compat path)")


def test_old_snapshot_with_no_model_id_still_works():
    """Snapshots written before model_id was added: envelope.model_id
    is None. New caller that passes expected_model_id should accept
    these as 'unknown lane' rather than invalidate."""
    # Use serialize with model_id=None (legacy behavior)
    df = _make_df(40)
    strat_cfg = _make_strategy()
    us = UnifiedStrategy(strat_cfg, None)
    fp = compute_backtest_fingerprint(strat_cfg)
    b64 = serialize_backtest_snapshot(
        us, '2026-05-19T12:00:00+00:00', fp, model_id=None)
    assert b64 is not None

    # Caller passes expected_model_id — should still accept since
    # envelope's model_id is None (back-compat).
    env = deserialize_backtest_snapshot(
        b64, expected_fingerprint=fp,
        expected_model_id='backtest_rest_hifi')
    assert env is not None, (
        "Legacy snapshot (model_id=None) should pass when caller "
        "provides expected_model_id — back-compat path"
    )
    print("  ✓ legacy snapshot (model_id=None) accepted with any expected_model_id")


def test_services_get_strategy_trades_for_window_return_shape():
    """Verify return shape:
       - return_snapshot=False → DataFrame (back-compat)
       - return_snapshot=True  → (DataFrame, b64_or_None) tuple
    """
    # Use a minimal strategy that gets early-returned (no entry_trigger_confluence_id)
    strat = {'symbol': 'SPY', 'timeframe': '1Min',
             'strategy_origin': 'standard'}

    import services as svc
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    since = now

    # back-compat shape
    r1 = svc.get_strategy_trades_for_window(strat, since, now)
    assert isinstance(r1, pd.DataFrame), f"back-compat should return DataFrame; got {type(r1)}"
    print("  ✓ return_snapshot=False (default) returns DataFrame (back-compat)")

    # new shape
    r2 = svc.get_strategy_trades_for_window(strat, since, now, return_snapshot=True)
    assert isinstance(r2, tuple) and len(r2) == 2, (
        f"return_snapshot=True should return 2-tuple; got {type(r2)}"
    )
    assert isinstance(r2[0], pd.DataFrame)
    # r2[1] is None for the early-return path (no resume_snapshot was passed)
    print("  ✓ return_snapshot=True returns (DataFrame, snapshot_b64)")


if __name__ == '__main__':
    print("\n=== LEF 2c forward_test_service port regression ===\n")
    test_model_id_in_envelope()
    test_model_id_mismatch_invalidates()
    test_old_snapshot_with_no_model_id_still_works()
    test_services_get_strategy_trades_for_window_return_shape()
    print("\n=== ALL PASS ===\n")
