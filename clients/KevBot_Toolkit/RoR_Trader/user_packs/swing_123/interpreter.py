"""Swing 1-2-3 — interpreter (state classification + trigger forwarding).

State classification reads the integer `sw123_pattern` column produced
by the incremental class and maps it to a state name. Trigger function
forwards the cached booleans without recomputation.
"""

import pandas as pd


_PATTERN_TO_STATE = {
    2: "BULL_C3",
    1: "BULL_C2",
    -2: "BEAR_C3",
    -1: "BEAR_C2",
    0: "NEUTRAL",
}


def interpret_swing_123(df: pd.DataFrame, **params) -> pd.Series:
    def classify(row):
        p = row.get("sw123_pattern", 0)
        if pd.isna(p):
            return None
        return _PATTERN_TO_STATE.get(int(p), "NEUTRAL")

    return df.apply(classify, axis=1)


def detect_swing_123_triggers(df: pd.DataFrame, **params) -> dict:
    return {
        "sw123_bull_c2": df["__sw123_bull_c2"].fillna(False).astype(bool),
        "sw123_bull_c3": df["__sw123_bull_c3"].fillna(False).astype(bool),
        "sw123_bear_c2": df["__sw123_bear_c2"].fillna(False).astype(bool),
        "sw123_bear_c3": df["__sw123_bear_c3"].fillna(False).astype(bool),
    }
