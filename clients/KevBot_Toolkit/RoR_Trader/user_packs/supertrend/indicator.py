"""SuperTrend — batch indicator (thin wrapper over the incremental class).

Replays the DataFrame through SupertrendIncremental.update_bar so the
two paths share a single source of truth. The parity simulator's
backtest path reads the columns this function writes; the live path
calls the incremental class directly. Both produce the same numeric
values bar-for-bar.
"""

import pandas as pd
from indicator_incremental import SupertrendIncremental


def calculate_supertrend(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = SupertrendIncremental(**params)

    st_line = []
    st_direction = []
    st_atr = []
    st_line_prev = []
    bull_flips = []
    bear_flips = []
    near_stop_bulls = []
    near_stop_bears = []

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        st_line.append(out["st_line"])
        st_direction.append(out["st_direction"])
        st_atr.append(out["st_atr"])
        st_line_prev.append(out["st_line_prev"])
        bull_flips.append(out["st_bull_flip"])
        bear_flips.append(out["st_bear_flip"])
        near_stop_bulls.append(out["st_near_stop_bull"])
        near_stop_bears.append(out["st_near_stop_bear"])

    result = df.copy()
    result["st_line"] = st_line
    result["st_direction"] = st_direction
    result["st_atr"] = st_atr
    result["st_line_prev"] = st_line_prev
    # Trigger booleans cached for interpreter.py to read back without
    # recomputing — keeps the source of truth in one place.
    result["__st_bull_flip"] = bull_flips
    result["__st_bear_flip"] = bear_flips
    result["__st_near_stop_bull"] = near_stop_bulls
    result["__st_near_stop_bear"] = near_stop_bears
    return result
