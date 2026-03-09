import pandas as pd
import numpy as np


def interpret_supertrend(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar's position relative to SuperTrend.

    States:
        BULL_TRENDING: Price above ST line, not near stop
        BULL_NEAR_STOP: Price above ST but within 0.5x ATR of stop
        BEAR_TRENDING: Price below ST line, not near stop
        BEAR_NEAR_STOP: Price below ST but within 0.5x ATR of stop
    """
    def classify(row):
        direction = row.get("st_direction", np.nan)
        st_line = row.get("st_line", np.nan)
        atr_val = row.get("st_atr", np.nan)
        close = row.get("close", np.nan)

        if pd.isna(direction) or pd.isna(st_line) or pd.isna(atr_val):
            return None

        near_threshold = 0.5 * atr_val

        if direction == 1:
            if close - st_line <= near_threshold:
                return "BULL_NEAR_STOP"
            return "BULL_TRENDING"
        else:
            if st_line - close <= near_threshold:
                return "BEAR_NEAR_STOP"
            return "BEAR_TRENDING"

    return df.apply(classify, axis=1)


def detect_supertrend_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect SuperTrend trigger events.

    Triggers:
        st_bull_flip: Trend changes from bear to bull
        st_bear_flip: Trend changes from bull to bear
        st_near_stop_bull: Price approaches bull stop (within 0.5x ATR)
        st_near_stop_bear: Price approaches bear stop (within 0.5x ATR)
    """
    prefix = "st"
    triggers = {}

    direction = df["st_direction"]
    direction_prev = direction.shift(1)

    # Trend flips
    triggers[f"{prefix}_bull_flip"] = (direction == 1) & (direction_prev == -1)
    triggers[f"{prefix}_bear_flip"] = (direction == -1) & (direction_prev == 1)

    # Near stop triggers
    close = df["close"]
    st_line = df["st_line"]
    atr_val = df["st_atr"]
    threshold = 0.5 * atr_val

    close_prev = close.shift(1)
    st_line_prev = st_line.shift(1)
    threshold_prev = 0.5 * atr_val.shift(1)

    # Was not near, now near (bull)
    was_far_bull = (close_prev - st_line_prev) > threshold_prev
    now_near_bull = ((close - st_line) <= threshold) & (direction == 1)
    triggers[f"{prefix}_near_stop_bull"] = was_far_bull & now_near_bull

    # Was not near, now near (bear)
    was_far_bear = (st_line_prev - close_prev) > threshold_prev
    now_near_bear = ((st_line - close) <= threshold) & (direction == -1)
    triggers[f"{prefix}_near_stop_bear"] = was_far_bear & now_near_bear

    return triggers
