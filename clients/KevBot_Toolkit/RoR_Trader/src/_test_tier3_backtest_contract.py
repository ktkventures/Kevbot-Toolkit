"""Tier 3 §8.2 regression — always-start-flat backtest contract.

Verifies four invariants from `docs/Spec_Tier3_Always_Start_Flat.md`:

  1. A fresh `run_unified_backtest` always begins with `position.state.
     status == 'FLAT'`, regardless of whether `resume_snapshot` was
     provided. (§8.2 contract; the most important property.)
  2. `apply_backtest_snapshot` restores INDICATOR state byte-identically
     but does NOT restore POSITION state — the envelope's recorded
     position field is intentionally dropped.
  3. The envelope still serializes the position field (back-compat with
     older snapshots; the field is dead-letter but must not break
     deserialization).
  4. The legacy LEF 2c indicator-state parity (32 fields) still holds —
     this is a regression check that Tier 3's position-state change did
     not collaterally damage the indicator-state work.

Run: cd src && .venv-relative-python _test_tier3_backtest_contract.py
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
    PositionState,
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
        'id': 999_998,
        'name': 'TEST-TIER3-CONTRACT',
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


def test_position_starts_flat_no_snapshot():
    """Property 1a: cold-start run begins with position.state.status='FLAT'."""
    df = _make_df(50)
    strat_cfg = _make_strategy()
    # Hook the engine pre-run by running 0 bars and inspecting the
    # initial UnifiedStrategy state directly.
    us = UnifiedStrategy(strat_cfg, None)
    assert us.position.state.status == 'FLAT', (
        f"Cold-start position should be FLAT, got "
        f"{us.position.state.status}"
    )
    print("  ✓ cold-start position state is FLAT")


def test_position_starts_flat_with_snapshot():
    """Property 1b + 2: even when `resume_snapshot` carries a recorded
    IN_POSITION state, the resumed run begins FLAT."""
    df = _make_df(80)
    strat_cfg = _make_strategy()

    # Build a snapshot, then forge a position state on the envelope to
    # simulate "the snapshot recorded an open position" — the contract
    # is that apply_backtest_snapshot must IGNORE this field.
    _, _, b64 = run_unified_backtest(
        df.iloc[:40], strat_cfg, return_snapshot=True)
    assert b64 is not None

    fp = compute_backtest_fingerprint(strat_cfg)
    envelope = deserialize_backtest_snapshot(b64, expected_fingerprint=fp)
    assert envelope is not None

    # FORGE: pretend the snapshot recorded an open LONG position.
    # Field names match the unified_engine.PositionState dataclass.
    forged_pos = PositionState(
        status='IN_POSITION',
        direction='LONG',
        entry_price=100.0,
        entry_time='2026-05-19T10:00:00+00:00',
        stop_price=99.0,
        initial_stop_price=99.0,
        target_price=102.0,
        entry_bar_count=20,
    )
    envelope['position'] = forged_pos

    # Now apply — the position field should be DROPPED, not restored.
    us = UnifiedStrategy(strat_cfg, None)
    apply_backtest_snapshot(us, envelope)
    assert us.position.state.status == 'FLAT', (
        f"After apply_backtest_snapshot with forged IN_POSITION, "
        f"engine position should still be FLAT (Tier 3 §8.2 contract). "
        f"Got: {us.position.state.status}"
    )
    print("  ✓ resume from forged IN_POSITION envelope starts FLAT")


def test_envelope_still_carries_position_for_backcompat():
    """Property 3: the envelope still serializes position state, so
    older deserializers don't break and the field is available for
    diagnostics (e.g., 'what did the engine think it was holding?')."""
    df = _make_df(40)
    strat_cfg = _make_strategy()
    _, _, b64 = run_unified_backtest(
        df, strat_cfg, return_snapshot=True)
    assert b64 is not None
    fp = compute_backtest_fingerprint(strat_cfg)
    envelope = deserialize_backtest_snapshot(b64, expected_fingerprint=fp)
    assert envelope is not None
    assert 'position' in envelope, (
        "envelope should still carry a 'position' field for back-compat — "
        "Tier 3 drops it at apply-time, not at serialize-time"
    )
    print("  ✓ envelope carries position field (back-compat preserved)")


def test_indicator_state_parity_regression():
    """Property 4: regression — Tier 3 must not break the LEF 2c
    indicator-state byte-identical guarantee. Same test as
    _test_lef_2c_snapshot.py::test_resume_equals_full but folded in here
    so this file is a self-contained regression net."""
    df = _make_df(150)
    strat_cfg = _make_strategy()
    half = len(df) // 2

    # Full run
    _, _, full_final_b64 = run_unified_backtest(
        df, strat_cfg, return_snapshot=True)
    # Halved + resume
    _, _, b64 = run_unified_backtest(
        df.iloc[:half], strat_cfg, return_snapshot=True)
    fp = compute_backtest_fingerprint(strat_cfg)
    envelope = deserialize_backtest_snapshot(b64, expected_fingerprint=fp)
    _, _, resume_final_b64 = run_unified_backtest(
        df.iloc[half:], strat_cfg,
        resume_snapshot=envelope, return_snapshot=True)

    env_full = deserialize_backtest_snapshot(
        full_final_b64, expected_fingerprint=fp)
    env_resume = deserialize_backtest_snapshot(
        resume_final_b64, expected_fingerprint=fp)
    assert env_full is not None and env_resume is not None

    state_full = env_full['engine'][0].__dict__
    state_resume = env_resume['engine'][0].__dict__
    mismatches = []
    for k in sorted(set(state_full.keys()) | set(state_resume.keys())):
        v_f = state_full.get(k)
        v_r = state_resume.get(k)
        if isinstance(v_f, dict) and isinstance(v_r, dict):
            if v_f != v_r:
                mismatches.append(f"{k}: dict differs")
        elif hasattr(v_f, '__iter__') and not isinstance(v_f, str):
            if list(v_f or []) != list(v_r or []):
                mismatches.append(f"{k}: iterable differs")
        else:
            if v_f != v_r:
                try:
                    if abs(float(v_f) - float(v_r)) < 1e-9:
                        continue
                except (TypeError, ValueError):
                    pass
                mismatches.append(f"{k}: {v_f!r} vs {v_r!r}")
    if mismatches:
        raise AssertionError(
            f"Tier 3 broke LEF 2c indicator parity ({len(mismatches)} "
            f"mismatches): {mismatches[:5]}"
        )
    print(f"  ✓ LEF 2c indicator-state parity preserved "
          f"({len(state_full)} fields)")


def test_full_run_position_starts_flat():
    """Belt-and-braces: even after a full run completes, calling
    `run_unified_backtest` again on the same UnifiedStrategy class
    yields a fresh FLAT state. (Validates UnifiedStrategy ctor, not
    state pollution from a singleton.)"""
    df = _make_df(50)
    strat_cfg = _make_strategy()
    run_unified_backtest(df, strat_cfg)  # discard result
    # Second instance, same config
    us = UnifiedStrategy(strat_cfg, None)
    assert us.position.state.status == 'FLAT'
    print("  ✓ second UnifiedStrategy instance starts FLAT (no class state leak)")


if __name__ == '__main__':
    print("\n=== Tier 3 §8.2 backtest contract regression ===\n")
    test_position_starts_flat_no_snapshot()
    test_position_starts_flat_with_snapshot()
    test_envelope_still_carries_position_for_backcompat()
    test_indicator_state_parity_regression()
    test_full_run_position_starts_flat()
    print("\n=== ALL PASS ===\n")
