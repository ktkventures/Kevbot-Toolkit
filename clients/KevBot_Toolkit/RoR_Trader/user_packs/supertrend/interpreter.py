"""SuperTrend — interpreter (state classification + trigger forwarding).

Reads the indicator + trigger columns produced by indicator.py /
indicator_incremental.py and forwards them under the engine's expected
shapes. No logic re-computation here; the incremental class is the
single source of truth.
"""

import pandas as pd
import numpy as np


def interpret_supertrend(df: pd.DataFrame, **params) -> pd.Series:
    """Classify each bar's position relative to the trailing line.

    States (mutually exclusive): BULL_TRENDING / BULL_NEAR_STOP /
    BEAR_TRENDING / BEAR_NEAR_STOP. None for warmup bars where ATR
    isn't established yet.
    """
    def classify(row):
        direction = row.get("st_direction", np.nan)
        st_line = row.get("st_line", np.nan)
        atr_val = row.get("st_atr", np.nan)
        close = row.get("close", np.nan)
        if pd.isna(direction) or pd.isna(st_line) or pd.isna(atr_val):
            return None
        if atr_val == 0:
            return None
        threshold = 0.5 * atr_val
        if direction == 1:
            if close - st_line <= threshold:
                return "BULL_NEAR_STOP"
            return "BULL_TRENDING"
        else:
            if st_line - close <= threshold:
                return "BEAR_NEAR_STOP"
            return "BEAR_TRENDING"

    return df.apply(classify, axis=1)


def detect_supertrend_triggers(df: pd.DataFrame, **params) -> dict:
    """Return one boolean Series per logical trigger.

    Trigger booleans are computed inside the incremental class and
    cached as `__st_*` columns by the batch indicator wrapper. Here
    we just forward them under the engine-expected `{prefix}_{base}`
    keys.
    """
    return {
        "st_bull_flip": df["__st_bull_flip"].fillna(False).astype(bool),
        "st_bear_flip": df["__st_bear_flip"].fillna(False).astype(bool),
        "st_near_stop_bull": df["__st_near_stop_bull"].fillna(False).astype(bool),
        "st_near_stop_bear": df["__st_near_stop_bear"].fillna(False).astype(bool),
    }
