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

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from interpreters import get_confluence_records, INTERPRETERS


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
        lookback_slice = df.iloc[start_idx:bar_index + 1]
        if direction == "LONG":
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
        lookback_slice = df.iloc[start_idx:bar_index + 1]
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

    # State machine
    in_position = False
    entry_idx = None
    entry_price = None
    entry_row = None
    stop_price = None
    target_price = None
    entry_bar_i = None

    for i, (idx, row) in enumerate(df.iterrows()):
        if not in_position:
            # Check for entry signal
            if row.get(entry_col, False):
                # Check confluence if required
                if confluence_required and len(confluence_required) > 0:
                    current_confluence = get_confluence_records(row, "1M", interpreter_list, general_columns=general_columns)
                    if not isinstance(current_confluence, set):
                        current_confluence = set()
                    if not confluence_required.issubset(current_confluence):
                        continue  # Confluence not met, skip entry

                # Enter position
                in_position = True
                entry_idx = idx
                entry_price = row['close']
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

        else:
            # Check for exit conditions
            # Priority: stop > target > signal exit triggers
            # Same-bar conflict resolution: stop is checked first, so if both
            # stop and target are breached within the same bar, the worse
            # outcome (stop) is assumed. This keeps backtests pessimistic.
            exit_triggered = False
            exit_reason = None
            exit_price = row['close']

            # 1. Check stop loss (highest priority)
            if direction == "LONG" and row['low'] <= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = stop_price
            elif direction == "SHORT" and row['high'] >= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = stop_price

            # 2. Check target (second priority)
            if not exit_triggered and target_price is not None:
                if direction == "LONG" and row['high'] >= target_price:
                    exit_triggered = True
                    exit_reason = "target"
                    exit_price = target_price
                elif direction == "SHORT" and row['low'] <= target_price:
                    exit_triggered = True
                    exit_reason = "target"
                    exit_price = target_price

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
                    if exit_col in df.columns and row.get(exit_col, False):
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

                risk = abs(entry_price - stop_price)
                if risk <= 0:
                    risk = entry_price * 0.01  # Fallback

                r_multiple = pnl / risk

                # Get confluence at entry
                confluence = get_confluence_records(entry_row, "1M", interpreter_list, general_columns=general_columns)
                if not isinstance(confluence, set):
                    confluence = set()

                trades.append({
                    'entry_time': entry_idx,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_price': stop_price,
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
    return opposites.get(trigger_id)


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
