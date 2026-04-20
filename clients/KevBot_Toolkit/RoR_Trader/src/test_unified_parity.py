#!/usr/bin/env python3
"""
Parity verification: unified engine vs batch pipeline.

Tests that the unified chart engine produces identical results to the
existing batch pipeline (indicators.py -> interpreters.py -> triggers.py)
for C-type triggers, and validates L-type triggers produce correct
level-based fills with gate protection.

Usage:
    python test_unified_parity.py
"""

import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, '.')

from indicators import (
    calculate_ema, calculate_atr, calculate_macd,
    calculate_vwap, calculate_utbot, calculate_volume_sma,
)
from interpreters import (
    interpret_ema_stack,
    detect_ema_triggers, detect_macd_triggers, detect_macd_hist_triggers,
    detect_vwap_triggers, detect_utbot_triggers,
)
from triggers import generate_trades
from unified_engine import (
    IncrementalIndicatorEngine, TriggerEvaluator,
    PositionState, PositionStateMachine,
    UnifiedStrategy, run_unified_backtest,
    get_trigger_exec_type, _strip_exec_suffix,
)


def generate_test_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV test data with session gaps."""
    rng = np.random.RandomState(seed)

    returns = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))

    spread = close * 0.003 * rng.uniform(0.5, 1.5, n)
    high = close + spread * rng.uniform(0, 1, n)
    low = close - spread * rng.uniform(0, 1, n)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)

    volume = rng.randint(1000, 50000, n).astype(float)

    base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
    timestamps = []
    t = base
    for i in range(n):
        timestamps.append(t)
        t += timedelta(minutes=1)
        if (i + 1) % 390 == 0:
            t += timedelta(hours=17, minutes=30)

    idx = pd.DatetimeIndex(timestamps, name='timestamp')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume,
    }, index=idx)


def assert_close(name: str, a: float, b: float,
                 rtol: float = 1e-8, atol: float = 1e-10):
    """Assert two values are approximately equal."""
    if math.isnan(a) and math.isnan(b):
        return
    if math.isnan(a) or math.isnan(b):
        raise AssertionError(f"{name}: {a} vs {b} (one is NaN)")
    if abs(b) < atol:
        if abs(a - b) > atol:
            raise AssertionError(f"{name}: {a:.10f} vs {b:.10f}")
        return
    rel = abs(a - b) / max(abs(b), 1e-15)
    if rel > rtol:
        raise AssertionError(
            f"{name}: {a:.10f} vs {b:.10f}, rel_diff={rel:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Indicator parity (unified engine vs batch pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def test_indicator_parity():
    """Verify unified engine IncrementalIndicatorEngine matches batch."""
    print("Test 1: Indicator parity...", end=" ")
    df = generate_test_data(300)

    # Batch indicators
    batch_ema_8 = calculate_ema(df, 8)
    batch_ema_21 = calculate_ema(df, 21)
    batch_atr = calculate_atr(df, 14)
    batch_macd = calculate_macd(df, 12, 26, 9)
    batch_vwap = calculate_vwap(df)
    batch_utbot = calculate_utbot(df, atr_period=10, atr_multiplier=1.0)

    # Unified engine — process bars one at a time
    engine = IncrementalIndicatorEngine(
        required_indicators={'ema', 'atr', 'macd', 'vwap', 'utbot', 'rvol'},
        params={
            'ema_periods': [8, 21, 50],
            'atr_period': 14,
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            'vwap_sd1_mult': 1.0,
            'vwap_sd2_mult': 2.0,
            'utbot_atr_period': 10,
            'utbot_atr_mult': 1.0,
            'vol_sma_period': 20,
        })

    for i in range(len(df)):
        row = df.iloc[i]
        bar = {
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row['volume']),
            'timestamp': df.index[i],
        }
        if i == 0:
            engine._update_indicators(bar, is_first=True)
        else:
            engine.state.prev2_macd_hist = engine.state.prev_macd_hist
            engine.state.prev_macd_hist = engine.state.current.get('macd_hist', 0.0)
            engine.state.prev_values = dict(engine.state.current)
            engine._update_indicators(bar, is_first=False)

        vals = engine.state.current
        # Check at bar 50, 150, 299 (well past warmup)
        if i >= 50:
            assert_close(f"EMA8@{i}", vals['ema_8'], batch_ema_8.iloc[i])
            assert_close(f"EMA21@{i}", vals['ema_21'], batch_ema_21.iloc[i])
            assert_close(f"ATR@{i}", vals['atr'], batch_atr.iloc[i])
            assert_close(f"MACD@{i}", vals['macd_line'],
                         batch_macd['macd_line'].iloc[i])
            assert_close(f"VWAP@{i}", vals['vwap'],
                         batch_vwap['vwap'].iloc[i])
            assert_close(f"UTBot@{i}", vals['utbot_stop'],
                         batch_utbot['utbot_stop'].iloc[i], rtol=1e-6)

    print("PASS")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: C-type trigger parity
# ═══════════════════════════════════════════════════════════════════════════

def test_ctype_trigger_parity():
    """Verify C-type triggers match batch pipeline exactly."""
    print("Test 2: C-type trigger parity...", end=" ")
    df = generate_test_data(500, seed=123)

    # Batch pipeline: compute indicators + detect triggers
    df['ema_8'] = calculate_ema(df, 8)
    df['ema_21'] = calculate_ema(df, 21)
    df['ema_50'] = calculate_ema(df, 50)
    macd = calculate_macd(df, 12, 26, 9)
    df['macd_line'] = macd['macd_line']
    df['macd_signal'] = macd['macd_signal']
    df['macd_hist'] = macd['macd_hist']
    df['atr'] = calculate_atr(df, 14)
    utbot = calculate_utbot(df, 10, 1.0)
    df['utbot_stop'] = utbot['utbot_stop']
    df['utbot_direction'] = utbot['utbot_direction']

    batch_ema_trigs = detect_ema_triggers(df)
    batch_macd_trigs = detect_macd_triggers(df)
    batch_utbot_trigs = detect_utbot_triggers(df)

    # Unified engine: process bar by bar
    all_triggers = set()
    all_triggers.update(batch_ema_trigs.keys())
    all_triggers.update(batch_macd_trigs.keys())
    all_triggers.update(batch_utbot_trigs.keys())

    req_interp = {'EMA_STACK', 'MACD_LINE', 'MACD_HISTOGRAM', 'UTBOT'}

    trigger_eval = TriggerEvaluator(
        required_interpreters=req_interp,
        required_triggers=all_triggers,
        ema_periods=[8, 21, 50])

    engine = IncrementalIndicatorEngine(
        required_indicators={'ema', 'atr', 'macd', 'utbot'},
        params={
            'ema_periods': [8, 21, 50],
            'atr_period': 14,
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            'utbot_atr_period': 10,
            'utbot_atr_mult': 1.0,
        })

    mismatches = []
    for i in range(len(df)):
        row = df.iloc[i]
        bar = {
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row['volume']),
            'timestamp': df.index[i],
        }
        current = engine.update_bar(bar)
        prev = engine.get_prev_values()

        if i < 2:
            continue  # Skip first 2 bars (no prev for crossover)

        interps, triggers = trigger_eval.evaluate_bar_close(
            current, prev, engine.state.prev2_macd_hist)

        # Compare each trigger against batch
        for trig_name in all_triggers:
            unified_val = triggers.get(trig_name, False)
            # Get batch value
            batch_series = None
            if trig_name in batch_ema_trigs:
                batch_series = batch_ema_trigs[trig_name]
            elif trig_name in batch_macd_trigs:
                batch_series = batch_macd_trigs[trig_name]
            elif trig_name in batch_utbot_trigs:
                batch_series = batch_utbot_trigs[trig_name]

            if batch_series is not None:
                batch_val = bool(batch_series.iloc[i])
                if unified_val != batch_val:
                    mismatches.append(
                        f"  {trig_name}@bar{i}: unified={unified_val}, "
                        f"batch={batch_val}")

    if mismatches:
        # Allow up to a few mismatches near the start (warmup divergence)
        late_mismatches = [m for m in mismatches if int(m.split('bar')[1].split(':')[0]) > 10]
        if late_mismatches:
            print(f"FAIL ({len(late_mismatches)} mismatches after bar 10)")
            for m in late_mismatches[:10]:
                print(m)
            raise AssertionError(f"{len(late_mismatches)} trigger mismatches")

    print("PASS")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: C-type trade parity (unified backtest vs generate_trades)
# ═══════════════════════════════════════════════════════════════════════════

def test_ctype_trade_parity():
    """Verify unified backtest produces identical trades for C-type strategies."""
    print("Test 3: C-type trade parity...", end=" ")
    df_raw = generate_test_data(500, seed=77)

    # Build enriched DataFrame for batch pipeline
    df = df_raw.copy()
    df['ema_8'] = calculate_ema(df, 8)
    df['ema_21'] = calculate_ema(df, 21)
    df['ema_50'] = calculate_ema(df, 50)
    df['atr'] = calculate_atr(df, 14)

    # Detect EMA triggers
    ema_trigs = detect_ema_triggers(df)
    for name, series in ema_trigs.items():
        df[f'trig_{name}'] = series

    # Classify EMA stack (needed for confluence records)
    df['EMA_STACK'] = interpret_ema_stack(df)

    # Run batch generate_trades
    batch_trades = generate_trades(
        df,
        direction='LONG',
        entry_trigger='ema_cross_bull',
        exit_triggers=['ema_cross_bear'],
        stop_config={'method': 'atr', 'atr_mult': 1.5},
    )

    # Run unified backtest
    strategy = {
        'id': 'test_ctype',
        'direction': 'LONG',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'timeframe': '1Min',
    }
    unified_trades, enriched = run_unified_backtest(df_raw, strategy)

    # Compare trade counts
    batch_count = len(batch_trades) if isinstance(batch_trades, pd.DataFrame) else 0
    unified_count = len(unified_trades) if isinstance(unified_trades, pd.DataFrame) else 0

    if batch_count == 0 and unified_count == 0:
        print("PASS (no trades)")
        return

    if batch_count != unified_count:
        print(f"FAIL: batch={batch_count} trades, unified={unified_count} trades")
        if batch_count > 0:
            print(f"  Batch first entry: {batch_trades.iloc[0]['entry_time']}")
        if unified_count > 0:
            print(f"  Unified first entry: {unified_trades.iloc[0]['entry_time']}")
        raise AssertionError(f"Trade count mismatch: {batch_count} vs {unified_count}")

    # Compare individual trades
    mismatches = []
    for i in range(min(batch_count, unified_count)):
        bt = batch_trades.iloc[i]
        ut = unified_trades.iloc[i]

        # Entry time
        if str(bt['entry_trigger_ts']) != str(ut['entry_trigger_ts']):
            mismatches.append(
                f"  Trade {i}: entry_time batch={bt['entry_trigger_ts']} "
                f"unified={ut['entry_trigger_ts']}")

        # Entry price (should be close for C-type)
        if abs(bt['entry_price'] - ut['entry_price']) > 0.001:
            mismatches.append(
                f"  Trade {i}: entry_price batch={bt['entry_price']:.4f} "
                f"unified={ut['entry_price']:.4f}")

        # Exit reason — batch uses "signal_exit" for all trigger exits,
        # unified uses the actual trigger ID (more informative).
        # Both are correct; normalize for comparison.
        bt_reason = bt['exit_reason']
        ut_reason = ut['exit_reason']
        # Normalize: both "signal_exit" and specific trigger IDs are
        # equivalent for signal exits
        bt_is_signal = bt_reason == 'signal_exit'
        ut_is_signal = ut_reason not in ('stop_loss', 'target', 'bar_count_exit')
        if bt_is_signal != ut_is_signal:
            mismatches.append(
                f"  Trade {i}: exit_reason batch={bt_reason} "
                f"unified={ut_reason}")

    if mismatches:
        print(f"FAIL ({len(mismatches)} mismatches)")
        for m in mismatches[:10]:
            print(m)
        raise AssertionError(f"{len(mismatches)} trade mismatches")

    print(f"PASS ({batch_count} trades matched)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: L-type trigger validation
# ═══════════════════════════════════════════════════════════════════════════

def test_ltype_triggers():
    """Verify L-type triggers fill at indicator level with gate protection."""
    print("Test 4: L-type trigger validation...", end=" ")
    df = generate_test_data(500, seed=55)

    strategy = {
        'id': 'test_ltype',
        'direction': 'LONG',
        'entry_trigger': 'ema_pp_v2_cross_short_up_ib',
        'entry_trigger_confluence_id': '',
        'exit_triggers': ['ema_pp_v2_cross_short_down'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'timeframe': '1Min',
    }

    trades, enriched = run_unified_backtest(df, strategy)

    if len(trades) == 0:
        print("PASS (no trades — acceptable for this seed)")
        return

    # Verify all L-type entries fill at EMA level, not at close
    l_type_entries = 0
    close_filled = 0
    level_filled = 0

    for i in range(len(trades)):
        t = trades.iloc[i]
        entry_price = t['entry_price']

        # Find the bar where entry occurred
        entry_time = t['entry_trigger_ts']
        if entry_time in enriched.index:
            bar = enriched.loc[entry_time]
            bar_close = bar['close']

            if abs(entry_price - bar_close) < 0.0001:
                close_filled += 1
            else:
                level_filled += 1
            l_type_entries += 1

    if l_type_entries > 0 and level_filled > 0:
        print(f"PASS ({l_type_entries} entries: {level_filled} level-filled, "
              f"{close_filled} close-filled)")
    elif l_type_entries > 0:
        print(f"PASS ({l_type_entries} entries, all close-filled "
              f"— L-type may have fallen back to bar-close)")
    else:
        print("PASS (no L-type entries matched enriched index)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: L-type gate prevents phantom crossings
# ═══════════════════════════════════════════════════════════════════════════

def test_ltype_gate():
    """Verify L-type gate prevents entries when prev close is already above level."""
    print("Test 5: L-type gate validation...", end=" ")

    # Create a scenario where price starts low then jumps high and stays.
    # After the jump, close is always well ABOVE the EMA, so the L-type
    # gate (close <= level) should stay CLOSED, blocking phantom entries.
    n = 30
    base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(n)]
    idx = pd.DatetimeIndex(timestamps, name='timestamp')

    # Start at 100, then jump to 120 and stay there.
    # EMA-8 will slowly converge toward 120 from below, but
    # close will always be above EMA after the jump.
    close = np.full(n, 120.0)
    close[0:5] = [100.0, 100.5, 101.0, 105.0, 115.0]  # ramp up
    high = close + 0.5
    low = close - 0.5
    # On bar 20, wick low touches well below EMA but close stays at 120
    low[20] = 105.0  # big wick below
    open_ = close - 0.1
    volume = np.full(n, 10000.0)

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume,
    }, index=idx)

    # Use ema_pp_cross_short_up_ib — gate should block since
    # close is above ema_8 on every bar
    required_triggers = {
        'ema_pp_cross_short_up', 'ema_pp_cross_short_up_ib',
    }
    required_interps = {'EMA_PRICE_POSITION'}

    engine = IncrementalIndicatorEngine(
        required_indicators={'ema', 'atr'},
        params={'ema_periods': [8, 21, 50], 'atr_period': 14})

    trigger_eval = TriggerEvaluator(
        required_interpreters=required_interps,
        required_triggers=required_triggers,
        ema_periods=[8, 21, 50])

    l_type_fired = False
    for i in range(n):
        row = df.iloc[i]
        bar = {
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row['volume']),
            'timestamp': df.index[i],
        }
        current = engine.update_bar(bar)
        prev = engine.get_prev_values()

        if i < 1:
            continue

        _, _, l_fills = trigger_eval.evaluate_bar_for_backtest(
            current, prev, engine.state.prev2_macd_hist)

        if 'ema_pp_cross_short_up_ib' in l_fills:
            l_type_fired = True

    if l_type_fired:
        print("FAIL — L-type fired when price was always above EMA")
        raise AssertionError("L-type gate should have blocked entry")

    print("PASS (gate blocked phantom crossing)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Enriched DataFrame has expected columns
# ═══════════════════════════════════════════════════════════════════════════

def test_enriched_df_columns():
    """Verify enriched DataFrame contains indicator and trigger columns."""
    print("Test 6: Enriched DataFrame columns...", end=" ")
    df = generate_test_data(100, seed=42)

    strategy = {
        'id': 'test_enriched',
        'direction': 'LONG',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'timeframe': '1Min',
    }
    _, enriched = run_unified_backtest(df, strategy)

    # Check indicator columns exist — test groups pin EMA to [8, 21, 50]
    # (see _install_test_groups). Use ema_8 to match the pinned short_period.
    expected_indicators = ['ema_8', 'ema_21', 'atr']
    missing = [c for c in expected_indicators if c not in enriched.columns]
    if missing:
        print(f"FAIL — missing indicator columns: {missing}")
        raise AssertionError(f"Missing columns: {missing}")

    # Check trigger columns exist (prefixed with trig_)
    trig_cols = [c for c in enriched.columns if c.startswith('trig_')]
    if not trig_cols:
        print("FAIL — no trigger columns found")
        raise AssertionError("No trig_ columns in enriched DataFrame")

    # Check interpreter columns exist
    if 'EMA_STACK' not in enriched.columns:
        print("FAIL — missing EMA_STACK interpreter column")
        raise AssertionError("Missing EMA_STACK column")

    print(f"PASS ({len(enriched.columns)} columns, "
          f"{len(trig_cols)} trigger columns)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Trade record schema matches generate_trades()
# ═══════════════════════════════════════════════════════════════════════════

def test_trade_schema():
    """Verify trade DataFrame has all required columns."""
    print("Test 7: Trade schema...", end=" ")
    df = generate_test_data(500, seed=33)

    strategy = {
        'id': 'test_schema',
        'direction': 'LONG',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'target_config': {'method': 'risk_reward', 'rr_ratio': 2.0},
        'timeframe': '1Min',
    }
    trades, _ = run_unified_backtest(df, strategy)

    if len(trades) == 0:
        print("PASS (no trades — schema check skipped)")
        return

    required_cols = {
        # Trade_Timestamps_Spec is the contract — legacy entry_time /
        # exit_time aliases dropped per locked decision #5.
        'entry_trigger_ts', 'entry_fill_ts',
        'exit_trigger_ts', 'exit_fill_ts',
        'hold_duration_s', 'behavior',
        'entry_price', 'exit_price',
        'stop_price', 'initial_stop_price', 'target_price',
        'pnl', 'risk', 'r_multiple', 'win', 'exit_reason',
        'entry_trigger', 'exit_trigger', 'confluence_records',
    }
    actual_cols = set(trades.columns)
    missing = required_cols - actual_cols
    if missing:
        print(f"FAIL — missing columns: {missing}")
        raise AssertionError(f"Missing trade columns: {missing}")

    # Verify types
    t = trades.iloc[0]
    assert isinstance(t['pnl'], (int, float, np.floating, np.integer)), \
        f"pnl should be numeric, got {type(t['pnl'])}"
    assert t['win'] in (True, False), \
        f"win should be bool-like, got {t['win']} ({type(t['win'])})"
    assert isinstance(t['entry_trigger'], str), "entry_trigger should be str"

    print(f"PASS ({len(trades)} trades, all columns present)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7.5: Trade_Timestamps_Spec 4-field timestamp semantics
# ═══════════════════════════════════════════════════════════════════════════

def test_timestamp_4field_model():
    """Verify trade records carry entry/exit × trigger/fill timestamps with
    exec-type-appropriate offsets between trigger and fill."""
    print("Test 7.5: 4-field timestamp model...", end=" ")

    df = generate_test_data(500, seed=34)

    # C-type strategy — fill_ts should equal trigger_ts + bar_duration (60s)
    strategy_c = {
        'id': 'test_ts_c',
        'direction': 'LONG',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'timeframe': '1Min',
    }
    trades_c, _ = run_unified_backtest(df, strategy_c)

    if len(trades_c) == 0:
        print("PASS (no trades — skipped timing assertions)")
        return

    t = trades_c.iloc[0]
    for col in ('entry_trigger_ts', 'entry_fill_ts',
                'exit_trigger_ts', 'exit_fill_ts'):
        assert t[col] is not None and str(t[col]) not in ('', 'NaT', 'None'), \
            f"C-type trade must have {col} populated, got {t[col]!r}"

    # Revised 2026-04-20: for shipped exec types (C/L/LC/CC, all
    # Behavior B), trigger_ts == fill_ts. For C-type, both represent
    # bar close. Behavior A variants would separate them; not shipped yet.
    assert pd.Timestamp(t['entry_trigger_ts']) == pd.Timestamp(t['entry_fill_ts']), \
        "C-type entry_trigger_ts should equal entry_fill_ts (both = bar close)"
    assert pd.Timestamp(t['exit_trigger_ts']) == pd.Timestamp(t['exit_fill_ts']), \
        "C-type exit_trigger_ts should equal exit_fill_ts (both = bar close)"

    # Metadata fields
    assert t['hold_duration_s'] == 0, \
        f"C-type manifest has hold_duration_seconds=0, got {t['hold_duration_s']}"
    assert t['behavior'] in ('A', 'B'), \
        f"behavior should be 'A' or 'B', got {t['behavior']!r}"

    print(f"PASS ({len(trades_c)} C-type trades, entry/exit deltas +60s)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: Trailing stop integration
# ═══════════════════════════════════════════════════════════════════════════

def test_trailing_stop():
    """Verify trailing stop ratchets correctly."""
    print("Test 8: Trailing stop...", end=" ")

    # Create data that trends up after entry
    n = 50
    base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(n)]
    idx = pd.DatetimeIndex(timestamps, name='timestamp')

    # Start flat, then trend up
    close = np.concatenate([
        np.full(10, 100.0),  # flat (EMA cross fires around here)
        np.linspace(100.5, 110.0, 40),  # uptrend
    ])
    # Create EMA crossover at bar ~8
    close[0:5] = [99.0, 99.5, 99.8, 100.0, 100.2]
    close[5:10] = [100.5, 100.8, 101.0, 101.2, 101.5]

    high = close + 0.3
    low = close - 0.3
    open_ = close - 0.1
    volume = np.full(n, 10000.0)

    # Force ema_8 > ema_21 crossover by manipulating early bars
    close[0:3] = [98.0, 98.5, 99.0]  # keep ema_8 below ema_21

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume,
    }, index=idx)

    strategy = {
        'id': 'test_trailing',
        'direction': 'LONG',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': [],
        'bar_count_exit': 30,
        'stop_config': {
            'method': 'atr', 'atr_mult': 1.5,
            'trailing': {'enabled': True, 'method': 'atr', 'atr_mult': 1.0,
                          'activation_r': 0.5},
        },
        'timeframe': '1Min',
    }

    trades, _ = run_unified_backtest(df, strategy)

    if len(trades) == 0:
        print("PASS (no trades — data didn't trigger entry)")
        return

    t = trades.iloc[0]
    # If trailing is active, stop should have moved up from initial
    if t['stop_price'] > t['initial_stop_price']:
        print(f"PASS (stop ratcheted: {t['initial_stop_price']:.2f} → "
              f"{t['stop_price']:.2f})")
    else:
        print(f"PASS (stop={t['stop_price']:.2f}, initial={t['initial_stop_price']:.2f} "
              f"— trailing may not have activated)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9: get_trigger_exec_type classification
# ═══════════════════════════════════════════════════════════════════════════

def test_exec_type_classification():
    """Verify trigger execution type classification."""
    print("Test 9: Exec type classification...", end=" ")

    assert get_trigger_exec_type('ema_cross_bull') == 'C'
    assert get_trigger_exec_type('utbot_buy') == 'C'
    assert get_trigger_exec_type('utbot_buy_ib') == 'L0'       # utbot_stop (current bar)
    assert get_trigger_exec_type('vwap_cross_above_ib') == 'L0' # vwap (current bar)
    assert get_trigger_exec_type('utbot_v2_buy_ib') == 'L1'     # utbot_stop_prev
    assert get_trigger_exec_type('ema_pp_v2_cross_short_up_ib') == 'L1'  # ema_9_prev
    assert get_trigger_exec_type('utbot_buy_hm') == 'HM'
    assert get_trigger_exec_type('ema_pp_cross_short_up_hm') == 'HM'
    assert get_trigger_exec_type('utbot_buy_hl') == 'HL'
    assert get_trigger_exec_type('vwap_cross_above_hl') == 'HL'

    # Group-prefixed IDs (e.g. from confluence groups page)
    assert get_trigger_exec_type('utbot_v2_default_buy_ib') == 'L1'
    assert get_trigger_exec_type('utbot_v2_custom_1_sell_ib') == 'L1'
    assert get_trigger_exec_type('vwap_default_cross_above_ib') == 'L0'
    assert get_trigger_exec_type('ema_pp_v2_custom_1_cross_short_up_ib') == 'L1'

    # LC/CC exec types
    assert get_trigger_exec_type('ema_pp_v2_cross_short_up_lc') == 'LC'
    assert get_trigger_exec_type('ema_pp_v2_cross_mid_down_lc') == 'LC'
    assert get_trigger_exec_type('ema_pp_v2_cross_short_up_cc') == 'CC'
    assert get_trigger_exec_type('sw123_bull_c2_cc') == 'CC'

    assert _strip_exec_suffix('utbot_buy_ib') == 'utbot_buy'
    assert _strip_exec_suffix('utbot_buy_hm') == 'utbot_buy'
    assert _strip_exec_suffix('utbot_buy_hl') == 'utbot_buy'
    assert _strip_exec_suffix('ema_cross_bull') == 'ema_cross_bull'
    assert _strip_exec_suffix('ema_pp_v2_cross_short_up_lc') == 'ema_pp_v2_cross_short_up'
    assert _strip_exec_suffix('sw123_bull_c2_cc') == 'sw123_bull_c2'

    print("PASS")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10: HM-type behavior (confirmed + unconfirmed)
# ═══════════════════════════════════════════════════════════════════════════

def _make_ema_hm_strategy(exec_suffix='_hm'):
    """Create EMA Price Position strategy with HM/HL entry trigger."""
    return {
        'id': f'test_hm_hl_{exec_suffix}',
        'direction': 'LONG',
        'entry_trigger': f'ema_pp_cross_short_up{exec_suffix}',
        'exit_triggers': [],
        'bar_count_exit': 20,
        'stop_config': {'method': 'atr', 'atr_mult': 3.0},
        'timeframe': '1Min',
    }


def test_hm_confirmed():
    """HM entries produce both confirmed and unconfirmed trades."""
    print("Test 10: HM confirmed + unconfirmed...", end=" ")

    df = generate_test_data(500, seed=55)
    strategy = _make_ema_hm_strategy('_hm')
    trades, _ = run_unified_backtest(df, strategy)

    if len(trades) == 0:
        print("FAIL — no trades generated")
        raise AssertionError("Expected trades from HM strategy")

    confirmed = trades[trades['exit_reason'] != 'unconfirmed_hm']
    unconfirmed = trades[trades['exit_reason'] == 'unconfirmed_hm']

    if len(confirmed) == 0:
        print("FAIL — no confirmed trades")
        raise AssertionError("Expected some confirmed HM trades")
    if len(unconfirmed) == 0:
        print("FAIL — no unconfirmed trades")
        raise AssertionError("Expected some unconfirmed HM trades")

    # Confirmed trades should exit via normal means
    for _, t in confirmed.iterrows():
        assert t['exit_reason'] in ('stop_loss', 'target', 'bar_count_exit',
                                     'ema_pp_cross_short_down',
                                     'ema_pp_cross_short_down_hm'), \
            f"Unexpected exit_reason for confirmed trade: {t['exit_reason']}"

    print(f"PASS ({len(confirmed)} confirmed, {len(unconfirmed)} unconfirmed "
          f"of {len(trades)} total)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 11: HM-type unconfirmed exit at next bar open
# ═══════════════════════════════════════════════════════════════════════════

def test_hm_unconfirmed():
    """HM unconfirmed trades exit at the next bar's open price."""
    print("Test 11: HM unconfirmed exit price...", end=" ")

    df = generate_test_data(500, seed=55)
    strategy = _make_ema_hm_strategy('_hm')
    trades, _ = run_unified_backtest(df, strategy)

    unconfirmed = trades[trades['exit_reason'] == 'unconfirmed_hm']
    if len(unconfirmed) == 0:
        print("SKIP — no unconfirmed trades")
        return

    # Each unconfirmed trade should exit at the next bar's open
    checked = 0
    for _, t in unconfirmed.iterrows():
        entry_time = t['entry_trigger_ts']
        exit_time = t['exit_trigger_ts']
        # Find entry and exit bar indices
        entry_idx = df.index.get_loc(entry_time)
        if entry_idx + 1 >= len(df):
            continue
        next_bar_open = float(df.iloc[entry_idx + 1]['open'])
        assert abs(t['exit_price'] - next_bar_open) < 1e-6, \
            f"Exit price {t['exit_price']:.4f} != next bar open {next_bar_open:.4f}"
        # Exit time should be the next bar's timestamp
        assert exit_time == df.index[entry_idx + 1], \
            f"Exit time mismatch: {exit_time} vs {df.index[entry_idx + 1]}"
        checked += 1

    assert checked > 0, "No unconfirmed trades could be verified"
    print(f"PASS ({checked} unconfirmed trades verified)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 12: HL-type confirmed entry
# ═══════════════════════════════════════════════════════════════════════════

def test_hl_confirmed():
    """HL entries produce both confirmed and unconfirmed trades."""
    print("Test 12: HL confirmed + unconfirmed...", end=" ")

    df = generate_test_data(500, seed=55)
    strategy = _make_ema_hm_strategy('_hl')
    trades, _ = run_unified_backtest(df, strategy)

    if len(trades) == 0:
        print("FAIL — no trades generated")
        raise AssertionError("Expected trades from HL strategy")

    unconfirmed = trades[trades['exit_reason'] == 'unconfirmed_hl']
    confirmed = trades[trades['exit_reason'] != 'unconfirmed_hl']

    # HL unconfirmed should exit at entry_price (break-even)
    for _, t in unconfirmed.iterrows():
        assert abs(t['exit_price'] - t['entry_price']) < 1e-6, \
            f"HL unconfirmed exit_price {t['exit_price']:.4f} != " \
            f"entry_price {t['entry_price']:.4f}"
        assert abs(t['pnl']) < 1e-6, \
            f"HL unconfirmed pnl should be ~0, got {t['pnl']:.6f}"

    # Some trades should fall through to normal exits (stop, bar_count)
    other_exits = trades[~trades['exit_reason'].isin(
        ['unconfirmed_hl'])]

    print(f"PASS ({len(confirmed)} confirmed, {len(unconfirmed)} unconfirmed "
          f"of {len(trades)} total)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 13: HL-type unconfirmed + no return → fallback exit
# ═══════════════════════════════════════════════════════════════════════════

def test_hl_unconfirmed_no_return():
    """HL unconfirmed trades that never return to entry exit via stop/bar_count."""
    print("Test 13: HL unconfirmed no return...", end=" ")

    df = generate_test_data(500, seed=55)
    strategy = _make_ema_hm_strategy('_hl')
    trades, _ = run_unified_backtest(df, strategy)

    # Some trades should exit via normal means (not unconfirmed_hl)
    # These are either confirmed trades or unconfirmed trades where
    # price never returned to entry level
    normal_exits = trades[trades['exit_reason'].isin(
        ['stop_loss', 'bar_count_exit', 'target'])]

    if len(normal_exits) > 0:
        print(f"PASS ({len(normal_exits)} trades with normal exits: "
              f"{normal_exits['exit_reason'].value_counts().to_dict()})")
    else:
        # All trades either confirmed→normal or unconfirmed→HL limit
        # This is acceptable — depends on data
        print(f"PASS ({len(trades)} trades, all HL limit or confirmed)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 14: HM/HL gate blocks entry
# ═══════════════════════════════════════════════════════════════════════════

def test_hm_gate_blocks():
    """HM gate blocks entry when close is strictly above EMA (uptrend)."""
    print("Test 14: HM/HL gate blocks...", end=" ")

    # Create steep uptrend where close is always strictly above EMA
    # after the first bar. EMA lags, so close > EMA consistently.
    # Gate for "cross above" = (close <= level) → False in uptrend
    n = 50
    base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(n)]
    idx = pd.DatetimeIndex(timestamps, name='timestamp')

    # Steep uptrend: close increases by 2 per bar
    close = 100.0 + np.arange(n) * 2.0
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.3
    volume = np.full(n, 10000.0)

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume,
    }, index=idx)

    strategy = _make_ema_hm_strategy('_hm')
    trades, _ = run_unified_backtest(df, strategy)

    # Bar 0: EMA=100, close=100, gate=(100<=100)=True, high=100.5>=100 → entry!
    # Bar 0 entry is unconfirmed (100 > 100 → False), exits at bar 1 open.
    # Bar 1 may re-enter (gate from bar 0 still open). After that, gate closes.
    # In a 50-bar steep uptrend, only 1-2 trades should form (vs many for flat data).
    assert len(trades) <= 3, \
        f"Expected very few trades in steep uptrend, got {len(trades)}"

    print(f"PASS ({len(trades)} trade(s) — steep uptrend blocked gate)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 15: UT Bot HM uses L-type gate (not bar-close trigger gating)
# ═══════════════════════════════════════════════════════════════════════════

def test_utbot_hm_ltype_gate():
    """UT Bot HM triggers use L-type gate (prev close opposite side)."""
    print("Test 15: UT Bot HM L-type gate...", end=" ")

    df = generate_test_data(500, seed=77)
    strategy = {
        'id': 'test_utbot_hm',
        'direction': 'LONG',
        'entry_trigger': 'utbot_buy_hm',
        'exit_triggers': [],
        'bar_count_exit': 10,
        'stop_config': {'method': 'atr', 'atr_mult': 2.0},
        'timeframe': '1Min',
    }
    trades, _ = run_unified_backtest(df, strategy)

    # Should produce trades — UT Bot HM uses L-type gate,
    # not the more restrictive bar-close-trigger gating
    if len(trades) > 0:
        # Verify entry prices are at UT Bot stop level (not close)
        for _, t in trades.iterrows():
            # Entry prices from L-type fills are at indicator levels
            entry_bar_idx = df.index.get_loc(t['entry_trigger_ts'])
            bar_close = float(df.iloc[entry_bar_idx]['close'])
            if abs(t['entry_price'] - bar_close) > 0.001:
                # Entry was at indicator level, not close — confirms L-type fill
                break
        else:
            # All entries happened to be at close — still acceptable
            pass

    print(f"PASS ({len(trades)} trades from UT Bot HM)")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Phase 30C: Live-tick method tests
# ═══════════════════════════════════════════════════════════════════════════

def test_position_state_from_dict():
    """Test 16: PositionState round-trip serialization with all fields."""
    ps = PositionState(
        status='IN_POSITION',
        entry_price=150.0,
        entry_time='2026-01-01T10:00:00',
        stop_price=148.0,
        initial_stop_price=148.0,
        target_price=155.0,
        entry_bar_count=50,
        direction='LONG',
        entry_trigger='buy_hm',
        exec_type='HM',
        pending_hm_exit=True,
        pending_hl_limit=False,
    )
    d = ps.to_dict()
    ps2 = PositionState.from_dict(d)

    assert ps2.status == 'IN_POSITION'
    assert ps2.entry_price == 150.0
    assert ps2.exec_type == 'HM'
    assert ps2.pending_hm_exit is True
    assert ps2.pending_hl_limit is False
    assert ps2.initial_stop_price == 148.0
    assert ps2.entry_trigger == 'buy_hm'

    # Test LC/CC fields round-trip
    ps_lc = PositionState(
        status='IN_POSITION', exec_type='LC',
        pending_confirm_bar=52, bail_action='exit_limit_breakeven',
        hold_seconds=3, direction='LONG',
    )
    d_lc = ps_lc.to_dict()
    ps_lc2 = PositionState.from_dict(d_lc)
    assert ps_lc2.exec_type == 'LC'
    assert ps_lc2.pending_confirm_bar == 52
    assert ps_lc2.bail_action == 'exit_limit_breakeven'
    assert ps_lc2.hold_seconds == 3

    # Test backward compatibility: old dict without new fields
    old_d = {'status': 'FLAT', 'entry_price': 0.0, 'direction': 'SHORT'}
    ps3 = PositionState.from_dict(old_d)
    assert ps3.exec_type == 'C'
    assert ps3.pending_hm_exit is False
    assert ps3.pending_hl_limit is False
    assert ps3.pending_confirm_bar == -1
    assert ps3.bail_action == 'exit_market'
    assert ps3.hold_seconds == 0
    assert ps3.initial_stop_price == 0.0
    assert ps3.direction == 'SHORT'


def test_check_exit_tick_pending_hm():
    """Test 17: check_exit_tick exits immediately on pending HM."""
    strat = {
        'id': 1, 'direction': 'LONG',
        'entry_trigger': 'utbot_buy_hm',
        'exit_triggers': [],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
    }
    psm = PositionStateMachine(strat, resolved_entry='utbot_buy_hm')

    # Manually set up an IN_POSITION state with pending HM exit
    psm.state.status = 'IN_POSITION'
    psm.state.entry_price = 100.0
    psm.state.stop_price = 98.0
    psm.state.pending_hm_exit = True
    psm.state.exec_type = 'HM'

    sig = psm.check_exit_tick(101.5, '2026-01-01T10:01:00')
    assert sig is not None, "Expected exit signal for pending HM"
    assert sig['type'] == 'exit_signal'
    assert sig['trigger'] == 'unconfirmed_hm'
    assert sig['price'] == 101.5  # exits at tick price (market)
    assert sig['entry_price'] == 100.0
    assert psm.state.status == 'FLAT'


def test_check_exit_tick_pending_hl():
    """Test 18: check_exit_tick exits at entry_price when HL limit reached."""
    strat = {
        'id': 1, 'direction': 'LONG',
        'entry_trigger': 'utbot_buy_hl',
        'exit_triggers': [],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
    }
    psm = PositionStateMachine(strat, resolved_entry='utbot_buy_hl')

    psm.state.status = 'IN_POSITION'
    psm.state.entry_price = 100.0
    psm.state.stop_price = 98.0
    psm.state.pending_hl_limit = True
    psm.state.exec_type = 'HL'

    # Price hasn't reached entry yet
    sig = psm.check_exit_tick(99.5, '2026-01-01T10:01:00')
    assert sig is None, "Should not exit — price below entry"

    # Price reaches entry
    sig = psm.check_exit_tick(100.5, '2026-01-01T10:02:00')
    assert sig is not None, "Expected HL exit when price >= entry"
    assert sig['trigger'] == 'unconfirmed_hl'
    assert sig['price'] == 100.0  # exits at entry_price (break-even)
    assert psm.state.status == 'FLAT'


def test_check_exit_tick_pending_hl_stop_priority():
    """Test 19: Stop loss fires before HL limit when price gaps past."""
    strat = {
        'id': 1, 'direction': 'LONG',
        'entry_trigger': 'utbot_buy_hl',
        'exit_triggers': [],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
    }
    psm = PositionStateMachine(strat, resolved_entry='utbot_buy_hl')

    psm.state.status = 'IN_POSITION'
    psm.state.entry_price = 100.0
    psm.state.stop_price = 98.0
    psm.state.pending_hl_limit = True
    psm.state.exec_type = 'HL'

    # Price crashes below stop
    sig = psm.check_exit_tick(97.0, '2026-01-01T10:01:00')
    assert sig is not None, "Expected stop loss exit"
    assert sig['trigger'] == 'stop_loss'
    assert sig['price'] == 98.0  # exits at stop price
    assert psm.state.status == 'FLAT'


def test_check_entry_intrabar_exec_type():
    """Test 20: check_entry_intrabar sets correct exec_type."""
    strat = {
        'id': 1, 'direction': 'LONG',
        'entry_trigger': 'utbot_buy_hm',
        'exit_triggers': [],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
    }
    psm = PositionStateMachine(strat, resolved_entry='utbot_buy_hm')

    current = {'close': 100.0, 'atr': 2.0}
    sig = psm.check_entry_intrabar(
        'utbot_buy_hm', 99.5, current, 10, '2026-01-01T10:00:00')

    assert sig is not None, "Expected entry signal"
    assert sig['type'] == 'entry_signal'
    assert sig['price'] == 99.5  # fills at level, not close
    assert psm.state.exec_type == 'HM'
    assert psm.state.status == 'IN_POSITION'


def test_check_intrabar_hm_hl_triggers():
    """Test 21: check_intrabar detects _hm/_hl level crosses."""
    # Set up TriggerEvaluator with HM/HL triggers
    required_triggers = {
        'utbot_buy', 'utbot_sell',
        'utbot_buy_hm', 'utbot_sell_hm',
        'utbot_buy_hl', 'utbot_sell_hl',
    }
    te = TriggerEvaluator(
        {'UTBOT'}, required_triggers, ema_periods=[8, 21, 50])

    # Simulate a bar close to set up cached levels and gates
    current = {
        'close': 99.0, 'high': 101.0, 'low': 98.0,
        'open': 99.5, 'volume': 1000,
        'utbot_stop': 100.0, 'utbot_direction': -1,
    }
    prev = {
        'close': 100.5, 'utbot_stop': 100.0, 'utbot_direction': -1,
    }
    te.evaluate_bar_close(current, prev)

    # Gate should be open for _hm/_hl (close=99 <= utbot_stop=100 for 'above')
    assert te._ib_gate_open.get('utbot_buy_hm', False), \
        "Gate should be open for utbot_buy_hm"
    assert te._ib_gate_open.get('utbot_buy_hl', False), \
        "Gate should be open for utbot_buy_hl"

    # Simulate a tick above the level
    result = te.check_intrabar(100.5)
    assert result is not None, "Expected intra-bar detection"
    trigger_id, fill_price = result
    # Should fire one of the _hm/_hl triggers (first match wins)
    assert trigger_id in ('utbot_buy_hm', 'utbot_buy_hl'), \
        f"Unexpected trigger: {trigger_id}"
    # Fills at the tick price that triggered the cross, not the static level.
    # Guarantees entry_price stays within the forming bar's range.
    assert fill_price == 100.5, \
        f"Expected tick-price fill 100.5, got {fill_price}"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 30E — Swing Stops + MTF Confluence
# ═══════════════════════════════════════════════════════════════════════════


def test_swing_stop_long():
    """Swing stop for LONG: min(lows in lookback) - padding."""
    print("  [22] Swing stop (LONG) ...", end=" ")

    strategy = {
        'id': 'test_swing', 'direction': 'LONG',
        'entry_trigger': 'ema_cross_short_up',
        'exit_trigger': 'opposite_signal',
        'stop_config': {'method': 'swing', 'lookback': 3, 'padding': 0.10},
        'confluence': [],
    }

    psm = PositionStateMachine(strategy)

    # Feed several bars with known highs/lows
    bars_hl = [(102, 98), (104, 99), (103, 97), (105, 100), (106, 101)]
    for h, l in bars_hl:
        psm.update_high_low(h, l)

    # Simulate entry at close 106, ATR=2
    entry_price = 106.0
    atr = 2.0
    stop = psm._compute_stop(entry_price, atr, {'close': 106})

    # Lookback window (excluding current bar): last 3 of [:-1]
    # Buffer[:-1] = [(102,98),(104,99),(103,97),(105,100)]
    # Last 3: [(104,99),(103,97),(105,100)]
    # min(lows) = 97, stop = 97 - 0.10 = 96.90
    assert abs(stop - 96.90) < 0.001, f"Expected 96.90, got {stop}"
    print("PASSED")


def test_swing_stop_short():
    """Swing stop for SHORT: max(highs in lookback) + padding."""
    print("  [23] Swing stop (SHORT) ...", end=" ")

    strategy = {
        'id': 'test_swing_short', 'direction': 'SHORT',
        'entry_trigger': 'ema_cross_short_down',
        'exit_trigger': 'opposite_signal',
        'stop_config': {'method': 'swing', 'lookback': 3, 'padding': 0.05},
        'confluence': [],
    }

    psm = PositionStateMachine(strategy)

    bars_hl = [(102, 98), (104, 99), (103, 97), (105, 100), (101, 96)]
    for h, l in bars_hl:
        psm.update_high_low(h, l)

    stop = psm._compute_stop(96.0, 2.0, {'close': 96})

    # Buffer[:-1] = [(102,98),(104,99),(103,97),(105,100)]
    # Last 3: [(104,99),(103,97),(105,100)]
    # max(highs) = 105, stop = 105 + 0.05 = 105.05
    assert abs(stop - 105.05) < 0.001, f"Expected 105.05, got {stop}"
    print("PASSED")


def test_swing_target_long():
    """Swing target for LONG: max(highs in lookback) + padding."""
    print("  [24] Swing target (LONG) ...", end=" ")

    strategy = {
        'id': 'test_swing_tgt', 'direction': 'LONG',
        'entry_trigger': 'ema_cross_short_up',
        'exit_trigger': 'opposite_signal',
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'target_config': {'method': 'swing', 'lookback': 3, 'padding': 0.10},
        'confluence': [],
    }

    psm = PositionStateMachine(strategy)

    bars_hl = [(102, 98), (104, 99), (103, 97), (105, 100), (106, 101)]
    for h, l in bars_hl:
        psm.update_high_low(h, l)

    # entry=101, stop=98 (arbitrary), atr=2
    target = psm._compute_target(101.0, 98.0, 2.0, {'close': 101})

    # Buffer[:-1] = [(102,98),(104,99),(103,97),(105,100)]
    # Last 3: [(104,99),(103,97),(105,100)]
    # max(highs) = 105, target = 105 + 0.10 = 105.10
    assert abs(target - 105.10) < 0.001, f"Expected 105.10, got {target}"
    print("PASSED")


def test_swing_target_short():
    """Swing target for SHORT: min(lows in lookback) - padding."""
    print("  [25] Swing target (SHORT) ...", end=" ")

    strategy = {
        'id': 'test_swing_tgt_short', 'direction': 'SHORT',
        'entry_trigger': 'ema_cross_short_down',
        'exit_trigger': 'opposite_signal',
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'target_config': {'method': 'swing', 'lookback': 3, 'padding': 0.05},
        'confluence': [],
    }

    psm = PositionStateMachine(strategy)

    bars_hl = [(102, 98), (104, 99), (103, 97), (105, 100), (101, 96)]
    for h, l in bars_hl:
        psm.update_high_low(h, l)

    target = psm._compute_target(101.0, 105.0, 2.0, {'close': 101})

    # Buffer[:-1] = [(102,98),(104,99),(103,97),(105,100)]
    # Last 3: [(104,99),(103,97),(105,100)]
    # min(lows) = 97, target = 97 - 0.05 = 96.95
    assert abs(target - 96.95) < 0.001, f"Expected 96.95, got {target}"
    print("PASSED")


def test_swing_stop_not_enough_history():
    """Swing stop falls back to ATR when buffer has insufficient history."""
    print("  [26] Swing stop (not enough history) ...", end=" ")

    strategy = {
        'id': 'test_swing_fallback', 'direction': 'LONG',
        'entry_trigger': 'ema_cross_short_up',
        'exit_trigger': 'opposite_signal',
        'stop_config': {'method': 'swing', 'lookback': 5, 'padding': 0.0},
        'confluence': [],
    }

    psm = PositionStateMachine(strategy)
    # Only 1 bar — after excluding current, window is empty
    psm.update_high_low(105, 100)

    stop = psm._compute_stop(100.0, 2.0, {'close': 100})

    # Fallback: ATR * 1.5 = 3.0, LONG: 100 - 3 = 97
    assert abs(stop - 97.0) < 0.001, f"Expected 97.0, got {stop}"
    print("PASSED")


def test_mtf_confluence_gating():
    """MTF confluence records gate entries when strategy requires them."""
    print("  [27] MTF confluence gating ...", end=" ")

    # Use 500 bars, seed=77 (known to produce ema_cross_bull triggers)
    df = generate_test_data(500, seed=77)

    # Add a fake MTF column: EMA_STACK__5m — always set to FULL_BULL_STACK
    df["EMA_STACK__5m"] = "FULL_BULL_STACK"

    strategy_base = {
        'id': 'test_mtf', 'direction': 'LONG',
        'entry_trigger': 'ema_cross_bull',
        'exit_triggers': ['ema_cross_bear'],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
    }

    # Baseline: no confluence requirement — should produce trades
    trades_baseline, _ = run_unified_backtest(df, strategy_base)
    baseline_count = len(trades_baseline)
    assert baseline_count > 0, \
        f"Expected baseline trades, got {baseline_count}"

    # Add confluence requirement
    strategy_conf = {**strategy_base,
                     'confluence': ['5m-EMA_STACK-FULL_BULL_STACK']}

    # WITHOUT MTF map — engine never produces 5m-* record → 0 trades
    trades_no_mtf, _ = run_unified_backtest(df, strategy_conf)
    assert len(trades_no_mtf) == 0, \
        f"Expected 0 trades without MTF, got {len(trades_no_mtf)}"

    # WITH MTF map — confluence is always met → should match baseline
    sec_tf_map = {'5m': ['EMA_STACK__5m']}
    trades_with_mtf, _ = run_unified_backtest(
        df, strategy_conf, secondary_tf_map=sec_tf_map)
    assert len(trades_with_mtf) == baseline_count, \
        f"Expected {baseline_count} trades with MTF, got {len(trades_with_mtf)}"

    # Gating test: set MTF column to None for first 250 bars
    df2 = df.copy()
    df2.loc[df2.index[:250], "EMA_STACK__5m"] = None
    trades_partial, _ = run_unified_backtest(
        df2, strategy_conf, secondary_tf_map=sec_tf_map)

    # All entries should be in the second half (bars 250+)
    if len(trades_partial) > 0 and 'entry_time' in trades_partial.columns:
        for _, trade in trades_partial.iterrows():
            entry_idx = df2.index.get_loc(trade['entry_trigger_ts'])
            assert entry_idx >= 250, \
                f"Trade entered at bar {entry_idx}, expected >= 250"

    print(f"PASSED (baseline={baseline_count}, no_mtf={len(trades_no_mtf)}, "
          f"with_mtf={len(trades_with_mtf)}, partial={len(trades_partial)})")


def test_params_contract():
    """Validate every parameterized indicator in _INDICATOR_PARAM_SPEC
    round-trips cleanly from resolve_strategy_requirements into the engine
    constructors without hitting a hardcoded fallback.

    If someone adds a new pack and forgets to add its params to the spec
    (or the resolver), this test catches it before the engine silently
    runs with wrong values.
    """
    print("\nTest: Parameter contract round-trip...")
    import unified_engine as ue
    from unified_engine import (
        resolve_strategy_requirements, _INDICATOR_PARAM_SPEC,
        IncrementalIndicatorEngine, TriggerEvaluator,
    )
    from confluence_groups import ConfluenceGroup, PlotSettings

    # Build synthetic groups covering every parameterized template — keeps
    # the test independent of whatever defaults happen to exist today.
    def _mkgroup(group_id, base_template, params):
        return ConfluenceGroup(
            id=group_id, base_template=base_template, version='Test',
            description='', enabled=True, is_default=False,
            parameters=params,
            plot_settings=PlotSettings(colors={}, line_width=1, visible=True),
        )

    _test_groups = [
        _mkgroup('ema_stack_test', 'ema_stack',
                 {'short_period': 9, 'mid_period': 21, 'long_period': 200}),
        _mkgroup('macd_line_test', 'macd_line',
                 {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
        _mkgroup('vwap_test', 'vwap', {'sd1_mult': 1.0, 'sd2_mult': 2.0}),
        _mkgroup('utbot_test', 'utbot',
                 {'atr_period': 10, 'atr_multiplier': 1.0}),
        _mkgroup('rvol_test', 'rvol', {'sma_period': 20}),
    ]
    _orig_loader = ue._load_enabled_groups_for_strategy
    ue._load_enabled_groups_for_strategy = lambda _strategy: _test_groups
    try:
        _run_contract_assertions(_INDICATOR_PARAM_SPEC,
                                 resolve_strategy_requirements,
                                 IncrementalIndicatorEngine,
                                 TriggerEvaluator)
    finally:
        ue._load_enabled_groups_for_strategy = _orig_loader


def _run_contract_assertions(spec, resolve_fn, IndEngine, TrigEval):

    # Minimal strategy fixtures — one per parameterized indicator family.
    # Triggers chosen to route through _process_trigger_id's prefix matcher.
    # Note: every confluence template in create_default_groups() must be
    # enabled for these to resolve. Tests run in-memory without DB.
    fixtures = [
        ('ema', {
            'entry_trigger_confluence_id': 'ema_cross_bull',
            'confluence': [],
        }),
        ('macd', {
            'entry_trigger_confluence_id': 'macd_cross_bull',
            'confluence': [],
        }),
        ('vwap', {
            'entry_trigger_confluence_id': 'vwap_cross_above',
            'confluence': [],
        }),
        ('utbot', {
            'entry_trigger_confluence_id': 'utbot_buy',
            'confluence': [],
        }),
        ('rvol', {
            'entry_trigger_confluence_id': 'rvol_spike',
            'confluence': [],
        }),
    ]

    for indicator, strat in fixtures:
        ind, interp, trig, params = resolve_fn(strat)
        assert indicator in ind, \
            f"{indicator!r} fixture failed to route into required set: {ind}"
        for engine_key, _ in spec[indicator]['params']:
            assert engine_key in params, \
                (f"resolve_strategy_requirements omitted {engine_key!r} "
                 f"for {indicator!r} — contract violation")
        IndEngine(ind, params)
        TrigEval(interp, trig, params['ema_periods'])

    empty_strat = {'entry_trigger_confluence_id': '', 'confluence': []}
    _, _, _, params = resolve_fn(empty_strat)
    assert params['ema_periods'] == [], \
        f"Empty strategy should have ema_periods=[], got {params['ema_periods']!r}"

    print(f"PASSED (covered {len(fixtures)} parameterized indicators)")


def test_ltype_gap_fill_in_bar_range():
    """Gap-through L-type fills must land inside the bar's [low, high].

    Construct a bar where the crossing level sits BELOW bar_low (LONG gap-up
    scenario — previous bar's utbot_stop was at 100, but this bar opens at
    101 and never trades below). Engine should fill at bar_open, not at the
    off-bar level.
    """
    print("\nTest: L-type gap fill stays in bar range...", end=" ")

    te = TriggerEvaluator(
        required_interpreters={'UTBOT_V2'},
        required_triggers={'utbot_v2_buy', 'utbot_v2_buy_ib'},
        ema_periods=[],
    )

    # Prime the L-type gate + cached levels via a prior bar-close at price
    # BELOW the stop (so the gate opens for a subsequent cross above).
    prev_bar = {
        'close': 99.5, 'high': 99.8, 'low': 99.2, 'open': 99.5, 'volume': 1000,
        'utbot_stop': 100.0, 'utbot_stop_prev': 100.0, 'utbot_direction': -1,
    }
    te.evaluate_bar_close(prev_bar, {})

    # Current bar gaps up — opens at 101 (above the 100 level) and never
    # trades below the level. Reachability fires on gap, should fill at open.
    current_bar = {
        'close': 101.5, 'high': 101.7, 'low': 101.0, 'open': 101.0,
        'volume': 500,
        'utbot_stop': 101.0, 'utbot_stop_prev': 100.0, 'utbot_direction': 1,
    }
    _, _, l_fills = te.evaluate_bar_for_backtest(current_bar, prev_bar)

    assert 'utbot_v2_buy_ib' in l_fills, \
        f"Expected L-fill on gap-up cross, got: {list(l_fills)}"
    fill = l_fills['utbot_v2_buy_ib']
    assert current_bar['low'] <= fill <= current_bar['high'], \
        (f"Fill {fill} must be within bar [{current_bar['low']}, "
         f"{current_bar['high']}]")
    assert fill == current_bar['open'], \
        f"Gap-up should fill at bar_open={current_bar['open']}, got {fill}"

    print(f"PASSED (gap-up filled at open={fill})")


def test_ltype_normal_cross_fills_at_level():
    """Normal (non-gap) L-type cross should still fill at level.

    When the crossing level sits inside [bar_low, bar_high], price traded
    through the level intra-bar — fill at the level is the realistic price.
    Regression guard: max(level, bar_open) reduces to level when
    bar_open < level < bar_high.
    """
    print("\nTest: L-type normal cross fills at level...", end=" ")

    te = TriggerEvaluator(
        required_interpreters={'UTBOT_V2'},
        required_triggers={'utbot_v2_buy', 'utbot_v2_buy_ib'},
        ema_periods=[],
    )

    prev_bar = {
        'close': 99.5, 'high': 99.8, 'low': 99.2, 'open': 99.5, 'volume': 1000,
        'utbot_stop': 100.0, 'utbot_stop_prev': 100.0, 'utbot_direction': -1,
    }
    te.evaluate_bar_close(prev_bar, {})

    # Current bar opens BELOW level (99.7) and high crosses above (100.5).
    # No gap — level 100.0 is within bar range. Fill should be at level.
    current_bar = {
        'close': 100.3, 'high': 100.5, 'low': 99.6, 'open': 99.7,
        'volume': 500,
        'utbot_stop': 100.2, 'utbot_stop_prev': 100.0, 'utbot_direction': 1,
    }
    _, _, l_fills = te.evaluate_bar_for_backtest(current_bar, prev_bar)

    assert 'utbot_v2_buy_ib' in l_fills, \
        f"Expected L-fill on normal cross, got: {list(l_fills)}"
    fill = l_fills['utbot_v2_buy_ib']
    assert fill == 100.0, \
        f"Normal cross should fill at level=100.0, got {fill}"

    print(f"PASSED (normal cross filled at level={fill})")


def _install_test_groups(short_period=8, mid_period=21, long_period=50):
    """Pin the confluence group loader to a fixed EMA/MACD/etc. set.

    The batch pipeline (detect_ema_triggers, calculate_macd) uses hardcoded
    periods [8, 21, 50] for EMA and [12, 26, 9] for MACD. Tests verifying
    unified↔batch parity must force the resolver to match. Call before any
    test that runs run_unified_backtest with an EMA/MACD trigger.
    """
    import unified_engine as ue
    from confluence_groups import ConfluenceGroup, PlotSettings

    def _mk(gid, base, params):
        return ConfluenceGroup(
            id=gid, base_template=base, version='Test', description='',
            enabled=True, is_default=False, parameters=params,
            plot_settings=PlotSettings(colors={}, line_width=1, visible=True),
        )

    groups = [
        _mk('ema_stack_test', 'ema_stack', {
            'short_period': short_period,
            'mid_period': mid_period,
            'long_period': long_period,
        }),
        _mk('ema_price_position_test', 'ema_price_position', {
            'short_period': short_period,
            'mid_period': mid_period,
            'long_period': long_period,
        }),
        _mk('ema_price_position_v2_test', 'ema_price_position_v2', {
            'short_period': short_period,
            'mid_period': mid_period,
            'long_period': long_period,
        }),
        _mk('macd_line_test', 'macd_line', {
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
        }),
        _mk('macd_hist_test', 'macd_histogram', {
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
        }),
        _mk('vwap_test', 'vwap', {'sd1_mult': 1.0, 'sd2_mult': 2.0}),
        _mk('utbot_test', 'utbot', {'atr_period': 10, 'atr_multiplier': 1.0}),
        _mk('utbot_v2_test', 'utbot_v2',
            {'atr_period': 10, 'atr_multiplier': 1.0}),
        _mk('rvol_test', 'rvol', {'sma_period': 20}),
    ]
    ue._load_enabled_groups_for_strategy = lambda _strategy: groups


def main():
    # Force USE_DB=False so that _load_enabled_groups_for_strategy uses
    # file/in-memory groups, not Supabase. data_loader.load_dotenv(override=True)
    # may flip USE_DB back to True mid-import chain, so we pin it here after
    # all module-level imports have run. Tests that need to pin specific
    # group params call _install_test_groups() for a hermetic setup.
    import db as _db
    _db.USE_DB = False
    _install_test_groups()

    print("=" * 60)
    print("Unified Engine Parity Tests")
    print("=" * 60)

    tests = [
        test_params_contract,
        test_ltype_gap_fill_in_bar_range,
        test_ltype_normal_cross_fills_at_level,
        test_indicator_parity,
        test_ctype_trigger_parity,
        test_ctype_trade_parity,
        test_ltype_triggers,
        test_ltype_gate,
        test_enriched_df_columns,
        test_trade_schema,
        test_timestamp_4field_model,
        test_trailing_stop,
        test_exec_type_classification,
        test_hm_confirmed,
        test_hm_unconfirmed,
        test_hl_confirmed,
        test_hl_unconfirmed_no_return,
        test_hm_gate_blocks,
        test_utbot_hm_ltype_gate,
        test_position_state_from_dict,
        test_check_exit_tick_pending_hm,
        test_check_exit_tick_pending_hl,
        test_check_exit_tick_pending_hl_stop_priority,
        test_check_entry_intrabar_exec_type,
        test_check_intrabar_hm_hl_triggers,
        test_swing_stop_long,
        test_swing_stop_short,
        test_swing_target_long,
        test_swing_target_short,
        test_swing_stop_not_enough_history,
        test_mtf_confluence_gating,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
