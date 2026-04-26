"""Swing 1-2-3 — batch indicator (thin wrapper over the incremental class).

Replays each row through `Swing123Incremental.update_bar`. Both batch
and live paths share the same code, so they cannot drift apart — the
parity simulator verifies bit-equivalence.
"""

import pandas as pd
from indicator_incremental import Swing123Incremental


def calculate_swing_123(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = Swing123Incremental(**params)

    pattern: list = []
    candle_color: list = []
    bull_c2: list = []
    bull_c3: list = []
    bear_c2: list = []
    bear_c3: list = []

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        pattern.append(out["sw123_pattern"])
        candle_color.append(out["sw123_candle_color"])
        bull_c2.append(out["sw123_bull_c2"])
        bull_c3.append(out["sw123_bull_c3"])
        bear_c2.append(out["sw123_bear_c2"])
        bear_c3.append(out["sw123_bear_c3"])

    result = df.copy()
    result["sw123_pattern"] = pattern
    result["sw123_candle_color"] = candle_color
    # Trigger booleans cached for interpreter.py to forward.
    result["__sw123_bull_c2"] = bull_c2
    result["__sw123_bull_c3"] = bull_c3
    result["__sw123_bear_c2"] = bear_c2
    result["__sw123_bear_c3"] = bear_c3
    return result
