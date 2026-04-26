"""Swing 1-2-3 Test — interpreter (state classification + trigger forwarding).

State classification reads the boolean indicator columns produced by
the incremental class (via `indicator.py`) and applies the C3 > C2 >
NEUTRAL priority. Trigger function forwards the cached `__s123t_*`
booleans without recomputation — the incremental class is the single
source of truth for both the indicator booleans and the derived
trigger booleans.
"""

import pandas as pd
import numpy as np


def interpret_swing_123_test(df: pd.DataFrame, **params) -> pd.Series:
    required_cols = [
        "sw123t_bull_c2",
        "sw123t_bear_c2",
        "sw123t_bull_c3",
        "sw123t_bear_c3",
    ]
    for col in required_cols:
        if col not in df.columns:
            return pd.Series([None] * len(df), index=df.index)

    lookback = params.get("lookback_bars", 1)

    bull_c2 = df["sw123t_bull_c2"].astype(bool)
    bear_c2 = df["sw123t_bear_c2"].astype(bool)
    bull_c3 = df["sw123t_bull_c3"].astype(bool)
    bear_c3 = df["sw123t_bear_c3"].astype(bool)

    conditions = [bull_c3, bear_c3, bull_c2, bear_c2]
    choices = ["BULLISH_C3", "BEARISH_C3", "BULLISH_C2", "BEARISH_C2"]
    states = np.select(conditions, choices, default="NEUTRAL")
    result = pd.Series(states, index=df.index, dtype=str)

    # Bars before lookback+1 don't have valid C1 reference → None
    min_bars_needed = lookback + 1
    result.iloc[:min_bars_needed] = None
    return result


def detect_swing_123_test_triggers(df: pd.DataFrame, **params) -> dict:
    return {
        "s123t_bullish_c2_detected":  df["__s123t_bullish_c2_detected"].fillna(False).astype(bool),
        "s123t_bullish_c3_confirmed": df["__s123t_bullish_c3_confirmed"].fillna(False).astype(bool),
        "s123t_bearish_c2_detected":  df["__s123t_bearish_c2_detected"].fillna(False).astype(bool),
        "s123t_bearish_c3_confirmed": df["__s123t_bearish_c3_confirmed"].fillna(False).astype(bool),
        "s123t_pattern_reset":        df["__s123t_pattern_reset"].fillna(False).astype(bool),
    }
