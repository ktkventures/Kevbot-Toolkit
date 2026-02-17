"""
RSI Zones — Interpreter and Trigger Detection
===============================================

Classifies RSI values into discrete zones and detects trigger events
(overbought/oversold entries/exits, midline crosses).
"""

import pandas as pd
import numpy as np


def interpret_rsi_zones(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify RSI into zones.

    Args:
        df: DataFrame with 'rsi' column present
        **params: Parameters from the confluence group
            - overbought (float): Overbought threshold (default 70)
            - oversold (float): Oversold threshold (default 30)

    Returns:
        Series of categorical state strings:
        OVERBOUGHT, BULLISH, NEUTRAL, BEARISH, OVERSOLD
    """
    overbought = params.get("overbought", 70.0)
    oversold = params.get("oversold", 30.0)

    # Neutral zone is the middle 10% of the 50 line
    neutral_upper = 55.0
    neutral_lower = 45.0

    def classify(row):
        rsi = row.get("rsi", np.nan)
        if pd.isna(rsi):
            return None
        if rsi >= overbought:
            return "OVERBOUGHT"
        elif rsi > neutral_upper:
            return "BULLISH"
        elif rsi >= neutral_lower:
            return "NEUTRAL"
        elif rsi > oversold:
            return "BEARISH"
        else:
            return "OVERSOLD"

    return df.apply(classify, axis=1)


def detect_rsi_zones_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect RSI-based triggers.

    Args:
        df: DataFrame with 'rsi' column present
        **params: Parameters from the confluence group
            - overbought (float): Overbought threshold (default 70)
            - oversold (float): Oversold threshold (default 30)

    Returns:
        Dict mapping trigger_id -> boolean Series (True = trigger fired)
    """
    overbought = params.get("overbought", 70.0)
    oversold = params.get("oversold", 30.0)

    triggers = {}

    rsi = df["rsi"]
    rsi_prev = rsi.shift(1)

    # Enter overbought (RSI crosses above overbought)
    triggers["rsi_enter_overbought"] = (rsi >= overbought) & (rsi_prev < overbought)

    # Exit overbought (RSI crosses below overbought)
    triggers["rsi_exit_overbought"] = (rsi < overbought) & (rsi_prev >= overbought)

    # Enter oversold (RSI crosses below oversold)
    triggers["rsi_enter_oversold"] = (rsi <= oversold) & (rsi_prev > oversold)

    # Exit oversold (RSI crosses above oversold)
    triggers["rsi_exit_oversold"] = (rsi > oversold) & (rsi_prev <= oversold)

    # Cross above 50 (bullish midline cross)
    triggers["rsi_cross_above_50"] = (rsi > 50) & (rsi_prev <= 50)

    # Cross below 50 (bearish midline cross)
    triggers["rsi_cross_below_50"] = (rsi < 50) & (rsi_prev >= 50)

    return triggers
