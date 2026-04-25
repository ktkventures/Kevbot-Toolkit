"""UT Bot V4 — batch indicator.

Thin wrapper that replays the DataFrame through the incremental class
(indicator_incremental.UtBotV4Incremental). This guarantees the batch
path produces the exact same numeric values as the live-mode path, so
the parity simulator can verify true PASS rather than catching drift
between two parallel implementations.
"""

import pandas as pd
from indicator_incremental import UtBotV4Incremental


def calculate_ut_bot_v4(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = UtBotV4Incremental(**params)

    trail_stops = []
    trail_stops_prev = []
    bull_flips = []
    bear_flips = []

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        trail_stops.append(out["utv4_trailing_stop"])
        trail_stops_prev.append(out["utv4_trailing_stop_prev"])
        bull_flips.append(out["utv4_bull_flip"])
        bear_flips.append(out["utv4_bear_flip"])

    result = df.copy()
    result["utv4_trailing_stop"] = trail_stops
    result["utv4_trailing_stop_prev"] = trail_stops_prev
    # Triggers are produced by the trigger function in interpreter.py;
    # we stash the booleans here so detect_*_triggers() can read them
    # back without re-computing the cross logic.
    result["__utv4_bull_flip"] = bull_flips
    result["__utv4_bear_flip"] = bear_flips
    return result
