"""
Triggers and Trade Generation for RoR Trader
==============================================

This module handles:
1. Trade generation based on entry/exit triggers
2. Position management (state machine)
3. Risk/reward calculation
4. Configurable stop loss and take profit methods

Replaces the mock trade generation with real trigger-based logic.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from interpreters import get_confluence_records, get_mtf_confluence_records, INTERPRETERS

logger = logging.getLogger(__name__)


@dataclass
class TradeConfig:
    """Configuration for trade generation."""
    direction: str              # "LONG" or "SHORT"
    entry_trigger: str          # Trigger ID for entry
    exit_trigger: str           # Trigger ID or exit type
    risk_per_trade: float       # Dollar risk per trade
    default_stop_atr: float     # ATR multiplier for stop if not specified


# =============================================================================
# EXIT TYPES (legacy — kept for backward compatibility)
# =============================================================================

EXIT_TYPES = {
    "opposite_signal": "Opposite Entry Signal",
    "fixed_r_2": "Fixed R Target (2R)",
    "fixed_r_3": "Fixed R Target (3R)",
    "trailing_stop": "Trailing Stop (1 ATR)",
    "time_exit_50": "Time Exit (50 bars)",
}


# =============================================================================
# STOP LOSS CALCULATION
# =============================================================================

def calculate_stop_price(
    entry_price: float,
    direction: str,
    row: pd.Series,
    df: pd.DataFrame,
    bar_index: int,
    stop_config: dict,
) -> float:
    """
    Calculate stop loss price based on stop configuration.

    Args:
        entry_price: The price at which the position was entered
        direction: "LONG" or "SHORT"
        row: The current bar's data
        df: Full DataFrame (needed for swing lookback)
        bar_index: Integer position of current bar in df (for slicing)
        stop_config: Dict with "method" and method-specific params

    Returns:
        The calculated stop price
    """
    method = stop_config.get("method", "atr")

    if method == "atr":
        atr = row.get('atr', entry_price * 0.01)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01
            logger.debug("ATR NaN/invalid at bar %d, using fallback: %.4f", bar_index, atr)
        mult = stop_config.get("atr_mult", 1.5)
        distance = atr * mult

    elif method == "fixed_dollar":
        distance = stop_config.get("dollar_amount", 1.0)

    elif method == "percentage":
        pct = stop_config.get("percentage", 0.5)
        distance = entry_price * (pct / 100.0)

    elif method == "swing":
        lookback = stop_config.get("lookback", 5)
        padding = stop_config.get("padding", 0.0)
        start_idx = max(0, bar_index - lookback)
        lookback_slice = df.iloc[start_idx:bar_index]  # Exclude current bar
        if len(lookback_slice) == 0:
            # Not enough history — fall back to ATR
            atr = row.get('atr', entry_price * 0.01)
            if pd.isna(atr) or atr <= 0:
                atr = entry_price * 0.01
            distance = atr * 1.5
        elif direction == "LONG":
            swing_level = lookback_slice['low'].min()
            return swing_level - padding
        else:
            swing_level = lookback_slice['high'].max()
            return swing_level + padding

    else:
        # Fallback to ATR with default multiplier
        atr = row.get('atr', entry_price * 0.01)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01
        distance = atr * 1.5

    if direction == "LONG":
        return entry_price - distance
    else:
        return entry_price + distance


# =============================================================================
# TARGET PRICE CALCULATION
# =============================================================================

def calculate_target_price(
    entry_price: float,
    stop_price: float,
    direction: str,
    row: pd.Series,
    df: pd.DataFrame,
    bar_index: int,
    target_config: Optional[dict],
) -> Optional[float]:
    """
    Calculate target price based on target configuration.

    Args:
        entry_price: The price at which the position was entered
        stop_price: The calculated stop price (needed for risk:reward)
        direction: "LONG" or "SHORT"
        row: The current bar's data
        df: Full DataFrame (needed for swing lookback)
        bar_index: Integer position of current bar in df (for slicing)
        target_config: Dict with "method" and params, or None for no target

    Returns:
        The calculated target price, or None if no target
    """
    if target_config is None:
        return None

    method = target_config.get("method")
    if method is None:
        return None

    risk = abs(entry_price - stop_price)

    if method == "risk_reward":
        rr = target_config.get("rr_ratio", 2.0)
        distance = risk * rr

    elif method == "atr":
        atr = row.get('atr', entry_price * 0.01)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01
        mult = target_config.get("atr_mult", 2.0)
        distance = atr * mult

    elif method == "fixed_dollar":
        distance = target_config.get("dollar_amount", 2.0)

    elif method == "percentage":
        pct = target_config.get("percentage", 1.0)
        distance = entry_price * (pct / 100.0)

    elif method == "swing":
        lookback = target_config.get("lookback", 5)
        padding = target_config.get("padding", 0.0)
        start_idx = max(0, bar_index - lookback)
        lookback_slice = df.iloc[start_idx:bar_index]  # Exclude current bar
        if len(lookback_slice) == 0:
            return None  # Not enough history for swing target
        if direction == "LONG":
            swing_level = lookback_slice['high'].max()
            return swing_level + padding
        else:
            swing_level = lookback_slice['low'].min()
            return swing_level - padding

    else:
        return None

    if direction == "LONG":
        return entry_price + distance
    else:
        return entry_price - distance


# =============================================================================
# TRAILING / BREAKEVEN STOP UPDATES
# =============================================================================

def update_stop_price(
    current_stop: float,
    entry_price: float,
    direction: str,
    row: pd.Series,
    stop_config: dict,
) -> float:
    """Update stop price for trailing and/or breakeven stops.

    Called on each bar while in a position.  Stop can only ratchet in the
    favorable direction (LONG: up, SHORT: down).

    Modifier keys on stop_config:
      trailing:  {"enabled": True, "method": "atr"|"fixed_dollar"|"percentage",
                  "atr_mult": float, "dollar_amount": float, "percentage": float,
                  "activation_r": float}
      breakeven: {"enabled": True, "activation_r": float, "offset": float}
    """
    new_stop = current_stop
    risk = abs(entry_price - current_stop)
    if risk <= 0:
        risk = entry_price * 0.01

    # 1. Breakeven activation
    be = stop_config.get("breakeven")
    if be and be.get("enabled"):
        if direction == "LONG":
            unrealized_r = (row['close'] - entry_price) / risk
        else:
            unrealized_r = (entry_price - row['close']) / risk
        if unrealized_r >= be.get("activation_r", 1.0):
            be_offset = be.get("offset", 0.0)
            if direction == "LONG":
                be_level = entry_price + be_offset
                new_stop = max(new_stop, be_level)
            else:
                be_level = entry_price - be_offset
                new_stop = min(new_stop, be_level)

    # 2. Trailing stop
    trail = stop_config.get("trailing")
    if trail and trail.get("enabled"):
        if direction == "LONG":
            unrealized_r = (row['close'] - entry_price) / risk
        else:
            unrealized_r = (entry_price - row['close']) / risk

        if unrealized_r >= trail.get("activation_r", 0.0):
            trail_method = trail.get("method", "atr")
            if trail_method == "atr":
                atr = row.get('atr', entry_price * 0.01)
                if pd.isna(atr) or atr <= 0:
                    atr = entry_price * 0.01
                trail_distance = atr * trail.get("atr_mult", 1.0)
            elif trail_method == "fixed_dollar":
                trail_distance = trail.get("dollar_amount", 1.0)
            elif trail_method == "percentage":
                trail_distance = entry_price * (trail.get("percentage", 0.5) / 100.0)
            else:
                trail_distance = None

            if trail_distance is not None:
                if direction == "LONG":
                    trail_level = row['close'] - trail_distance
                    new_stop = max(new_stop, trail_level)
                else:
                    trail_level = row['close'] + trail_distance
                    new_stop = min(new_stop, trail_level)

    return new_stop


# =============================================================================
# TRADE GENERATION ENGINE
# =============================================================================

def generate_trades(
    df: pd.DataFrame,
    direction: str,
    entry_trigger: str,
    exit_trigger: str = None,
    exit_triggers: Optional[List[str]] = None,
    confluence_required: Optional[Set[str]] = None,
    risk_per_trade: float = 100.0,
    stop_atr_mult: float = 1.5,
    stop_config: Optional[dict] = None,
    target_config: Optional[dict] = None,
    bar_count_exit: Optional[int] = None,
    general_columns: Optional[List[str]] = None,
    enabled_interpreter_keys: Optional[List[str]] = None,
    secondary_tf_map: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Generate trades based on real trigger logic.

    Args:
        df: DataFrame with OHLCV, indicators, interpretations, and trigger columns
        direction: "LONG" or "SHORT"
        entry_trigger: Trigger ID (e.g., "ema_cross_bull")
        exit_trigger: Single exit trigger ID (legacy — use exit_triggers instead)
        exit_triggers: List of up to 3 exit trigger IDs (any fires → exit)
        confluence_required: Set of confluence records required for entry
        risk_per_trade: Dollar amount risked per trade
        stop_atr_mult: ATR multiplier for stop loss (legacy — use stop_config instead)
        stop_config: Stop loss configuration dict (takes precedence over stop_atr_mult)
        target_config: Take profit target configuration dict (optional)

    Returns:
        DataFrame with trade records
    """
    trades = []
    entry_col = f"trig_{entry_trigger}"

    # _ib triggers share the boolean column with their bar-close base
    if entry_col not in df.columns and entry_trigger.endswith('_ib'):
        entry_col = f"trig_{entry_trigger[:-3]}"

    # Verify trigger column exists
    if entry_col not in df.columns:
        print(f"Warning: Trigger column {entry_col} not found")
        return pd.DataFrame()

    # Build effective stop config (backward compat)
    effective_stop = stop_config if stop_config else {"method": "atr", "atr_mult": stop_atr_mult}

    # Build effective target config (backward compat for fixed_r exits)
    effective_target = target_config
    if effective_target is None and exit_trigger in ("fixed_r_2", "fixed_r_3"):
        r_val = 2.0 if exit_trigger == "fixed_r_2" else 3.0
        effective_target = {"method": "risk_reward", "rr_ratio": r_val}

    # Build effective exit triggers list (backward compat)
    effective_exit_triggers = []
    if exit_triggers is not None:
        effective_exit_triggers = list(exit_triggers)
    elif exit_trigger is not None:
        # Legacy single exit_trigger — wrap in list if it's a signal trigger
        if exit_trigger not in EXIT_TYPES:
            effective_exit_triggers = [exit_trigger]
        elif exit_trigger == "opposite_signal":
            # Handled as special case below
            effective_exit_triggers = []
        # fixed_r_2/fixed_r_3 converted to target_config above
        # time_exit_50 and trailing_stop handled as special cases

    # Determine if legacy special-case exit types are active
    use_opposite_signal = (exit_trigger == "opposite_signal" and exit_triggers is None)
    use_time_exit = (exit_trigger == "time_exit_50" and exit_triggers is None)

    # Get interpreter list for confluence records (only enabled groups if specified)
    interpreter_list = enabled_interpreter_keys if enabled_interpreter_keys is not None else list(INTERPRETERS.keys())

    # Resolve intra-bar level-fill for entry trigger
    _entry_is_ib = entry_trigger.endswith('_ib')
    _entry_level_spec = None
    if _entry_is_ib:
        from realtime_engine import INTRABAR_LEVEL_MAP
        _entry_base = entry_trigger.removesuffix('_ib')
        # Strip the trigger prefix to get the map key (e.g., "vwap_cross_above")
        _entry_level_spec = INTRABAR_LEVEL_MAP.get(_entry_base)

    # Resolve intra-bar level-fill for exit triggers
    _exit_ib_specs: Dict[str, dict] = {}  # exit_trigger → level_spec
    if _entry_is_ib or any(et.endswith('_ib') for et in effective_exit_triggers):
        from realtime_engine import INTRABAR_LEVEL_MAP as _ILM
        for et in effective_exit_triggers:
            if et.endswith('_ib'):
                _exit_base = et.removesuffix('_ib')
                spec = _ILM.get(_exit_base)
                if spec:
                    _exit_ib_specs[et] = spec

    # State machine
    in_position = False
    entry_idx = None
    entry_price = None
    entry_row = None
    stop_price = None
    initial_stop = None
    target_price = None
    entry_bar_i = None

    for i, (idx, row) in enumerate(df.iterrows()):
        if not in_position:
            # Check for entry signal
            if row.get(entry_col, False):
                # Check confluence if required
                if confluence_required and len(confluence_required) > 0:
                    if secondary_tf_map:
                        current_confluence = get_mtf_confluence_records(
                            row, interpreter_list, secondary_tf_map, general_columns)
                    else:
                        current_confluence = get_confluence_records(
                            row, "1M", interpreter_list, general_columns=general_columns)
                    if not isinstance(current_confluence, set):
                        current_confluence = set()
                    if not confluence_required.issubset(current_confluence):
                        continue  # Confluence not met, skip entry

                # Determine entry price: level-fill for [I] triggers, close for [C]
                if _entry_level_spec is not None:
                    level_col = _entry_level_spec['column']
                    level_val = row.get(level_col)
                    if level_val is not None and not pd.isna(level_val):
                        level_val = float(level_val)
                        cross_dir = _entry_level_spec['cross']
                        # Verify bar high/low reaches the level
                        if cross_dir == 'above' and row['high'] >= level_val:
                            fill_price = level_val
                        elif cross_dir == 'below' and row['low'] <= level_val:
                            fill_price = level_val
                        else:
                            continue  # Level not reached within bar
                    else:
                        fill_price = row['close']
                else:
                    fill_price = row['close']

                # Enter position
                in_position = True
                entry_idx = idx
                entry_price = fill_price
                entry_row = row
                entry_bar_i = i

                # Calculate stop price using configured method
                stop_price = calculate_stop_price(
                    entry_price, direction, row, df, i, effective_stop
                )

                # Calculate target price using configured method
                target_price = calculate_target_price(
                    entry_price, stop_price, direction, row, df, i, effective_target
                )
                initial_stop = stop_price  # Store for trade record

        else:
            # Update trailing / breakeven stop before checking exit conditions
            if effective_stop and (effective_stop.get("trailing") or effective_stop.get("breakeven")):
                stop_price = update_stop_price(
                    stop_price, entry_price, direction, row, effective_stop,
                )

            # Check for exit conditions
            # Priority: stop > target > signal exit triggers
            # Same-bar conflict resolution: stop is checked first, so if both
            # stop and target are breached within the same bar, the worse
            # outcome (stop) is assumed. This keeps backtests pessimistic.
            exit_triggered = False
            exit_reason = None
            exit_price = row['close']

            # 1. Check stop loss (highest priority)
            # Gap-aware fill: if the bar opens past the stop (overnight gap,
            # flash crash), fill at the open — not the stop level.
            if direction == "LONG" and row['low'] <= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = min(stop_price, row['open'])
            elif direction == "SHORT" and row['high'] >= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = max(stop_price, row['open'])

            # 2. Check target (second priority)
            # Gap-aware fill: if the bar opens past the target (gap in your
            # favor), fill at the open — you get the windfall.
            if not exit_triggered and target_price is not None:
                if direction == "LONG" and row['high'] >= target_price:
                    exit_triggered = True
                    exit_reason = "target"
                    exit_price = max(target_price, row['open'])
                elif direction == "SHORT" and row['low'] <= target_price:
                    exit_triggered = True
                    exit_reason = "target"
                    exit_price = min(target_price, row['open'])

            # 3. Check bar count exit
            if not exit_triggered and bar_count_exit is not None:
                if i - entry_bar_i >= bar_count_exit:
                    exit_triggered = True
                    exit_reason = "bar_count_exit"

            # 4. Check opposite signal exit (legacy backward compat)
            if not exit_triggered and use_opposite_signal:
                opposite_trigger = get_opposite_trigger(entry_trigger)
                if opposite_trigger and row.get(f"trig_{opposite_trigger}", False):
                    exit_triggered = True
                    exit_reason = "opposite_signal"

            # 5. Check time exit (legacy backward compat)
            if not exit_triggered and use_time_exit:
                bars_in_trade = i - list(df.index).index(entry_idx)
                if bars_in_trade >= 50:
                    exit_triggered = True
                    exit_reason = "time_exit"

            # 6. Check signal-based exit triggers (any of up to 3)
            if not exit_triggered and len(effective_exit_triggers) > 0:
                for et in effective_exit_triggers:
                    exit_col = f"trig_{et}"
                    # _ib triggers share the boolean column with their bar-close base
                    if exit_col not in df.columns and et.endswith('_ib'):
                        exit_col = f"trig_{et[:-3]}"
                    if exit_col in df.columns and row.get(exit_col, False):
                        # Level-fill for [I] exit triggers
                        et_spec = _exit_ib_specs.get(et)
                        if et_spec is not None:
                            lv_col = et_spec['column']
                            lv_val = row.get(lv_col)
                            if lv_val is not None and not pd.isna(lv_val):
                                lv_val = float(lv_val)
                                c_dir = et_spec['cross']
                                if c_dir == 'above' and row['high'] >= lv_val:
                                    exit_price = lv_val
                                elif c_dir == 'below' and row['low'] <= lv_val:
                                    exit_price = lv_val
                                else:
                                    continue  # Level not reached
                        exit_triggered = True
                        exit_reason = "signal_exit"
                        break

            # Process exit
            if exit_triggered:
                # Calculate P&L
                if direction == "LONG":
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price

                risk = abs(entry_price - initial_stop) if initial_stop else abs(entry_price - stop_price)
                if risk <= 0:
                    risk = entry_price * 0.01  # Fallback

                r_multiple = pnl / risk

                # Get confluence at entry
                if secondary_tf_map:
                    confluence = get_mtf_confluence_records(
                        entry_row, interpreter_list, secondary_tf_map, general_columns)
                else:
                    confluence = get_confluence_records(
                        entry_row, "1M", interpreter_list, general_columns=general_columns)
                if not isinstance(confluence, set):
                    confluence = set()

                trades.append({
                    'entry_time': entry_idx,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_price': stop_price,
                    'initial_stop_price': initial_stop,
                    'target_price': target_price,
                    'pnl': pnl,
                    'risk': risk,
                    'r_multiple': r_multiple,
                    'win': pnl > 0,
                    'exit_reason': exit_reason,
                    'entry_trigger': entry_trigger,
                    'exit_trigger': exit_trigger or (effective_exit_triggers[0] if effective_exit_triggers else None),
                    'confluence_records': confluence,
                })

                # Reset state
                in_position = False
                entry_idx = None
                entry_price = None
                entry_row = None
                stop_price = None
                initial_stop = None
                target_price = None
                entry_bar_i = None

    return pd.DataFrame(trades)


def get_opposite_trigger(trigger_id: str) -> Optional[str]:
    """Get the opposite trigger for exit on opposite signal."""
    opposites = {
        "ema_cross_bull": "ema_cross_bear",
        "ema_cross_bear": "ema_cross_bull",
        "ema_mid_cross_bull": "ema_mid_cross_bear",
        "ema_mid_cross_bear": "ema_mid_cross_bull",
        "macd_cross_bull": "macd_cross_bear",
        "macd_cross_bear": "macd_cross_bull",
        "macd_hist_flip_pos": "macd_hist_flip_neg",
        "macd_hist_flip_neg": "macd_hist_flip_pos",
        "vwap_cross_above": "vwap_cross_below",
        "vwap_cross_below": "vwap_cross_above",
    }

    result = opposites.get(trigger_id)
    if result:
        return result

    # Generic opposite detection for user packs (suffix-based)
    suffix_pairs = [
        ("_bull", "_bear"), ("_bear", "_bull"),
        ("_up", "_down"), ("_down", "_up"),
        ("_pos", "_neg"), ("_neg", "_pos"),
        ("_buy", "_sell"), ("_sell", "_buy"),
        ("_long", "_short"), ("_short", "_long"),
    ]
    for suffix, opposite_suffix in suffix_pairs:
        if trigger_id.endswith(suffix):
            return trigger_id[:-len(suffix)] + opposite_suffix

    return None


# =============================================================================
# TRIGGER DEFINITIONS FOR UI
# =============================================================================

def get_entry_triggers_for_direction(direction: str) -> Dict[str, str]:
    """
    Get entry triggers appropriate for a given direction.

    Returns dict mapping trigger_id -> display_name
    """
    if direction == "LONG":
        return {
            "ema_cross_bull": "EMA 8 > 21 Cross",
            "ema_mid_cross_bull": "EMA 21 > 50 Cross",
            "macd_cross_bull": "MACD Bullish Cross",
            "macd_hist_flip_pos": "MACD Histogram Flip Positive",
            "macd_zero_cross_up": "MACD Zero Cross Up",
            "vwap_cross_above": "VWAP Cross Above",
            "rvol_spike": "Volume Spike",
        }
    else:  # SHORT
        return {
            "ema_cross_bear": "EMA 8 < 21 Cross",
            "ema_mid_cross_bear": "EMA 21 < 50 Cross",
            "macd_cross_bear": "MACD Bearish Cross",
            "macd_hist_flip_neg": "MACD Histogram Flip Negative",
            "macd_zero_cross_down": "MACD Zero Cross Down",
            "vwap_cross_below": "VWAP Cross Below",
            "rvol_spike": "Volume Spike",
        }


def get_exit_triggers() -> Dict[str, str]:
    """Get available exit triggers."""
    return {
        "opposite_signal": "Opposite Signal",
        "fixed_r_2": "Fixed R Target (2R)",
        "fixed_r_3": "Fixed R Target (3R)",
        "trailing_stop": "Trailing Stop",
        "time_exit_50": "Time Exit (50 bars)",
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from mock_data import generate_mock_bars
    from indicators import run_all_indicators
    from interpreters import run_all_interpreters, detect_all_triggers
    from datetime import datetime, timedelta

    # Generate test data
    end = datetime.now().replace(hour=16, minute=0)
    start = end - timedelta(days=30)

    print("Generating mock data...")
    bars = generate_mock_bars(["SPY"], start, end, "1Min")
    spy_bars = bars.loc["SPY"]

    print("Running indicators...")
    df = run_all_indicators(spy_bars)

    print("Running interpreters...")
    df = run_all_interpreters(df)

    print("Detecting triggers...")
    df = detect_all_triggers(df)

    # Test 1: Legacy single exit trigger (backward compat)
    print("\n--- Test 1: Legacy opposite signal exit ---")
    trades = generate_trades(
        df,
        direction="LONG",
        entry_trigger="ema_cross_bull",
        exit_trigger="opposite_signal"
    )
    print(f"Generated {len(trades)} trades")
    if len(trades) > 0:
        print(f"  Win rate: {trades['win'].mean() * 100:.1f}%")
        print(f"  Avg R: {trades['r_multiple'].mean():.2f}")
        print(f"  Total R: {trades['r_multiple'].sum():.2f}")

    # Test 2: New stop config (percentage stop)
    print("\n--- Test 2: Percentage stop + R:R target ---")
    trades2 = generate_trades(
        df,
        direction="LONG",
        entry_trigger="ema_cross_bull",
        exit_triggers=["ema_cross_bear"],
        stop_config={"method": "percentage", "percentage": 0.5},
        target_config={"method": "risk_reward", "rr_ratio": 2.0},
    )
    print(f"Generated {len(trades2)} trades")
    if len(trades2) > 0:
        print(f"  Win rate: {trades2['win'].mean() * 100:.1f}%")
        print(f"  Avg R: {trades2['r_multiple'].mean():.2f}")
        print(f"  Total R: {trades2['r_multiple'].sum():.2f}")

    # Test 3: Multi-exit triggers
    print("\n--- Test 3: Multi-exit triggers ---")
    trades3 = generate_trades(
        df,
        direction="LONG",
        entry_trigger="ema_cross_bull",
        exit_triggers=["ema_cross_bear", "macd_cross_bear"],
        stop_config={"method": "atr", "atr_mult": 1.5},
    )
    print(f"Generated {len(trades3)} trades")
    if len(trades3) > 0:
        print(f"  Win rate: {trades3['win'].mean() * 100:.1f}%")
        print(f"  Avg R: {trades3['r_multiple'].mean():.2f}")
        print(f"  Total R: {trades3['r_multiple'].sum():.2f}")
        print(f"\n  Exit reasons: {trades3['exit_reason'].value_counts().to_dict()}")
