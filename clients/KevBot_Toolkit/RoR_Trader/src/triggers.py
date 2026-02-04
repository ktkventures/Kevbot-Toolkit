"""
Triggers and Trade Generation for RoR Trader
==============================================

This module handles:
1. Trade generation based on entry/exit triggers
2. Position management (state machine)
3. Risk/reward calculation

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
# EXIT TYPES
# =============================================================================

EXIT_TYPES = {
    "opposite_signal": "Opposite Entry Signal",
    "fixed_r_2": "Fixed R Target (2R)",
    "fixed_r_3": "Fixed R Target (3R)",
    "trailing_stop": "Trailing Stop (1 ATR)",
    "time_exit_50": "Time Exit (50 bars)",
}


# =============================================================================
# TRADE GENERATION ENGINE
# =============================================================================

def generate_trades(
    df: pd.DataFrame,
    direction: str,
    entry_trigger: str,
    exit_trigger: str,
    confluence_required: Optional[Set[str]] = None,
    risk_per_trade: float = 100.0,
    stop_atr_mult: float = 1.5
) -> pd.DataFrame:
    """
    Generate trades based on real trigger logic.

    Args:
        df: DataFrame with OHLCV, indicators, interpretations, and trigger columns
        direction: "LONG" or "SHORT"
        entry_trigger: Trigger ID (e.g., "ema_cross_bull")
        exit_trigger: Exit type (e.g., "opposite_signal", "fixed_r_2")
        confluence_required: Set of confluence records required for entry
        risk_per_trade: Dollar amount risked per trade
        stop_atr_mult: ATR multiplier for stop loss calculation

    Returns:
        DataFrame with trade records
    """
    trades = []
    entry_col = f"trig_{entry_trigger}"

    # Verify trigger column exists
    if entry_col not in df.columns:
        print(f"Warning: Trigger column {entry_col} not found")
        return pd.DataFrame()

    # Get interpreter list for confluence records
    interpreter_list = list(INTERPRETERS.keys())

    # State machine
    in_position = False
    entry_idx = None
    entry_price = None
    entry_row = None
    stop_price = None
    target_price = None

    for i, (idx, row) in enumerate(df.iterrows()):
        if not in_position:
            # Check for entry signal
            if row.get(entry_col, False):
                # Check confluence if required
                if confluence_required and len(confluence_required) > 0:
                    current_confluence = get_confluence_records(row, "1M", interpreter_list)
                    if not confluence_required.issubset(current_confluence):
                        continue  # Confluence not met, skip entry

                # Enter position
                in_position = True
                entry_idx = idx
                entry_price = row['close']
                entry_row = row

                # Calculate stop and target based on ATR
                atr = row.get('atr', entry_price * 0.01)  # Default to 1% if no ATR
                if pd.isna(atr) or atr <= 0:
                    atr = entry_price * 0.01

                if direction == "LONG":
                    stop_price = entry_price - (atr * stop_atr_mult)
                    if exit_trigger == "fixed_r_2":
                        target_price = entry_price + (atr * stop_atr_mult * 2)
                    elif exit_trigger == "fixed_r_3":
                        target_price = entry_price + (atr * stop_atr_mult * 3)
                    else:
                        target_price = None
                else:  # SHORT
                    stop_price = entry_price + (atr * stop_atr_mult)
                    if exit_trigger == "fixed_r_2":
                        target_price = entry_price - (atr * stop_atr_mult * 2)
                    elif exit_trigger == "fixed_r_3":
                        target_price = entry_price - (atr * stop_atr_mult * 3)
                    else:
                        target_price = None

        else:
            # Check for exit conditions
            exit_triggered = False
            exit_reason = None
            exit_price = row['close']

            # Check stop loss
            if direction == "LONG" and row['low'] <= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = stop_price
            elif direction == "SHORT" and row['high'] >= stop_price:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = stop_price

            # Check target
            if not exit_triggered and target_price is not None:
                if direction == "LONG" and row['high'] >= target_price:
                    exit_triggered = True
                    exit_reason = "target"
                    exit_price = target_price
                elif direction == "SHORT" and row['low'] <= target_price:
                    exit_triggered = True
                    exit_reason = "target"
                    exit_price = target_price

            # Check opposite signal exit
            if not exit_triggered and exit_trigger == "opposite_signal":
                # Look for the opposite entry trigger
                opposite_trigger = get_opposite_trigger(entry_trigger)
                if opposite_trigger and row.get(f"trig_{opposite_trigger}", False):
                    exit_triggered = True
                    exit_reason = "opposite_signal"

            # Check time exit
            if not exit_triggered and exit_trigger == "time_exit_50":
                bars_in_trade = i - list(df.index).index(entry_idx)
                if bars_in_trade >= 50:
                    exit_triggered = True
                    exit_reason = "time_exit"

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
                confluence = get_confluence_records(entry_row, "1M", interpreter_list)

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
                    'exit_trigger': exit_trigger,
                    'confluence_records': confluence,
                })

                # Reset state
                in_position = False
                entry_idx = None
                entry_price = None
                entry_row = None
                stop_price = None
                target_price = None

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

    print("\nGenerating trades (EMA cross bull, opposite signal exit)...")
    trades = generate_trades(
        df,
        direction="LONG",
        entry_trigger="ema_cross_bull",
        exit_trigger="opposite_signal"
    )

    print(f"\nGenerated {len(trades)} trades")
    if len(trades) > 0:
        print("\nTrade summary:")
        print(f"  Win rate: {trades['win'].mean() * 100:.1f}%")
        print(f"  Avg R: {trades['r_multiple'].mean():.2f}")
        print(f"  Total R: {trades['r_multiple'].sum():.2f}")

        print("\nSample trades:")
        print(trades[['entry_time', 'exit_time', 'r_multiple', 'win', 'exit_reason']].head(10))
