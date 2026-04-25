"""UT Bot V4 — interpreter (state classification + trigger detection).

States: BULL_TREND / BEAR_TREND / BULL_FLIP / BEAR_FLIP
Triggers: utv4_bull_flip / utv4_bear_flip — produced by indicator.py
and forwarded here.
"""

import pandas as pd
import numpy as np


def interpret_ut_bot_v4(df: pd.DataFrame, **params) -> pd.Series:
    """Classify each bar into one mutually exclusive state.

    BULL_FLIP / BEAR_FLIP take precedence on flip bars; otherwise the
    bar reflects whichever side of the trailing stop the close lies on.
    """
    close = df["close"]
    stop = df["utv4_trailing_stop"]
    bull = df.get("__utv4_bull_flip")
    bear = df.get("__utv4_bear_flip")

    states = pd.Series(index=df.index, dtype=object)
    for i in range(len(df)):
        c = close.iloc[i]
        s = stop.iloc[i]
        if pd.isna(c) or pd.isna(s) or s == 0:
            states.iloc[i] = None
            continue
        if bull is not None and bool(bull.iloc[i]):
            states.iloc[i] = "BULL_FLIP"
        elif bear is not None and bool(bear.iloc[i]):
            states.iloc[i] = "BEAR_FLIP"
        elif c > s:
            states.iloc[i] = "BULL_TREND"
        else:
            states.iloc[i] = "BEAR_TREND"
    return states


def detect_ut_bot_v4_triggers(df: pd.DataFrame, **params) -> dict:
    """Return one boolean Series per logical trigger.

    The booleans are pre-computed in indicator.py (so the batch path and
    the incremental path share a single source of truth). Here we just
    forward them under the prefixed keys the engine expects.
    """
    return {
        "utv4_bull_flip": df["__utv4_bull_flip"].fillna(False).astype(bool),
        "utv4_bear_flip": df["__utv4_bear_flip"].fillna(False).astype(bool),
    }
