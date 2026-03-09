import pandas as pd
import numpy as np


def interpret_swing_123(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar into a swing 1-2-3 pattern state.

    States:
        BULL_C3: Bullish continuation confirmed (C3)
        BULL_C2: Bullish reversal candidate (C2)
        BEAR_C3: Bearish continuation confirmed (C3)
        BEAR_C2: Bearish reversal candidate (C2)
        NEUTRAL: No pattern
    """
    pattern_map = {
        2: "BULL_C3",
        1: "BULL_C2",
        -2: "BEAR_C3",
        -1: "BEAR_C2",
        0: "NEUTRAL",
    }

    def classify(row):
        p = row.get("sw123_pattern", 0)
        if pd.isna(p):
            return None
        return pattern_map.get(int(p), "NEUTRAL")

    return df.apply(classify, axis=1)


def detect_swing_123_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect swing 1-2-3 pattern triggers.

    Each pattern detection IS the trigger (C2 and C3 are discrete events).
    """
    prefix = "sw123"
    triggers = {}

    pattern = df["sw123_pattern"]

    triggers[f"{prefix}_bull_c2"] = pattern == 1
    triggers[f"{prefix}_bull_c3"] = pattern == 2
    triggers[f"{prefix}_bear_c2"] = pattern == -1
    triggers[f"{prefix}_bear_c3"] = pattern == -2

    return triggers
