"""Swing 1-2-3 Test — batch indicator (thin wrapper over the incremental class).

Replays each row through `Swing123TestIncremental.update_bar`. Both
the batch and live paths run the same code, so they cannot drift —
the parity simulator verifies bit-equivalence.
"""

import pandas as pd
from indicator_incremental import Swing123TestIncremental


def calculate_swing_123_test(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = Swing123TestIncremental(**params)

    bull_c2: list = []
    bear_c2: list = []
    bull_c3: list = []
    bear_c3: list = []
    candle_color: list = []
    bull_entry_level_prev: list = []
    bear_entry_level_prev: list = []
    c2_detected_bull: list = []
    c3_confirmed_bull: list = []
    c2_detected_bear: list = []
    c3_confirmed_bear: list = []
    pattern_reset: list = []

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        bull_c2.append(out["sw123t_bull_c2"])
        bear_c2.append(out["sw123t_bear_c2"])
        bull_c3.append(out["sw123t_bull_c3"])
        bear_c3.append(out["sw123t_bear_c3"])
        candle_color.append(out["sw123t_candle_color"])
        bull_entry_level_prev.append(out["sw123t_bull_entry_level_prev"])
        bear_entry_level_prev.append(out["sw123t_bear_entry_level_prev"])
        c2_detected_bull.append(out["s123t_bullish_c2_detected"])
        c3_confirmed_bull.append(out["s123t_bullish_c3_confirmed"])
        c2_detected_bear.append(out["s123t_bearish_c2_detected"])
        c3_confirmed_bear.append(out["s123t_bearish_c3_confirmed"])
        pattern_reset.append(out["s123t_pattern_reset"])

    result = df.copy()
    result["sw123t_bull_c2"] = bull_c2
    result["sw123t_bear_c2"] = bear_c2
    result["sw123t_bull_c3"] = bull_c3
    result["sw123t_bear_c3"] = bear_c3
    result["sw123t_candle_color"] = candle_color
    result["sw123t_bull_entry_level_prev"] = bull_entry_level_prev
    result["sw123t_bear_entry_level_prev"] = bear_entry_level_prev
    # Trigger booleans cached for interpreter.py to forward without
    # recomputation — single source of truth.
    result["__s123t_bullish_c2_detected"] = c2_detected_bull
    result["__s123t_bullish_c3_confirmed"] = c3_confirmed_bull
    result["__s123t_bearish_c2_detected"] = c2_detected_bear
    result["__s123t_bearish_c3_confirmed"] = c3_confirmed_bear
    result["__s123t_pattern_reset"] = pattern_reset
    return result
