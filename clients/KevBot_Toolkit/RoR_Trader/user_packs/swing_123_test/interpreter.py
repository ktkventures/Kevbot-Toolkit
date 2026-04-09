import pandas as pd
import numpy as np


def interpret_swing_123_test(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar into a mutually exclusive Swing 1-2-3 state.

    Priority (highest to lowest):
      1. BULLISH_C3  — full bullish pattern confirmed
      2. BEARISH_C3  — full bearish pattern confirmed
      3. BULLISH_C2  — bullish setup candle
      4. BEARISH_C2  — bearish setup candle
      5. NEUTRAL     — none of the above

    C3 takes priority because a bar can simultaneously satisfy C2 conditions
    (e.g., lower low + close above prior close) while also closing above the
    prior bar's high to confirm a C3. The completed pattern is the more
    significant event.

    Args:
        df: DataFrame with sw123t_bull_c2, sw123t_bear_c2,
            sw123t_bull_c3, sw123t_bear_c3 columns present
        **params: Parameters (not used in classification logic here)

    Returns:
        Series of state label strings from manifest outputs list.
        Returns None for bars with insufficient data.
    """
    required_cols = [
        "sw123t_bull_c2",
        "sw123t_bear_c2",
        "sw123t_bull_c3",
        "sw123t_bear_c3",
    ]

    # Check required columns exist
    for col in required_cols:
        if col not in df.columns:
            return pd.Series([None] * len(df), index=df.index)

    lookback = params.get("lookback_bars", 1)
    # Bars within the first `lookback` rows don't have valid C1 reference
    min_valid_idx = lookback

    bull_c2 = df["sw123t_bull_c2"]
    bear_c2 = df["sw123t_bear_c2"]
    bull_c3 = df["sw123t_bull_c3"]
    bear_c3 = df["sw123t_bear_c3"]

    # Build state series with vectorized np.select — priority order matters
    # Higher index in conditions list = lower priority (select takes first True)
    conditions = [
        bull_c3,
        bear_c3,
        bull_c2,
        bear_c2,
    ]
    choices = [
        "BULLISH_C3",
        "BEARISH_C3",
        "BULLISH_C2",
        "BEARISH_C2",
    ]

    states = np.select(conditions, choices, default="NEUTRAL")
    result = pd.Series(states, index=df.index, dtype=str)

    # Mark early bars as None (insufficient lookback data)
    # C3 needs lookback + 1 bars of history, C2 needs lookback bars
    min_bars_needed = lookback + 1
    result.iloc[:min_bars_needed] = None

    return result


def detect_swing_123_test_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect Swing 1-2-3 trigger events from pattern state transitions.

    Trigger keys (prefix = 's123t'):
      s123t_bullish_c2_detected      — bar just classified as BULLISH_C2
      s123t_bullish_c2_detected_ib   — intra-bar cross above bull entry level
      s123t_bullish_c3_confirmed     — bar just classified as BULLISH_C3
      s123t_bearish_c2_detected      — bar just classified as BEARISH_C2
      s123t_bearish_c2_detected_ib   — intra-bar cross below bear entry level
      s123t_bearish_c3_confirmed     — bar just classified as BEARISH_C3
      s123t_pattern_reset            — bar returned to NEUTRAL after any active state

    Args:
        df: DataFrame with sw123t_* columns present
        **params: Parameters (lookback_bars used for None guard)

    Returns:
        Dict mapping trigger_id -> boolean Series
    """
    required_cols = [
        "sw123t_bull_c2",
        "sw123t_bear_c2",
        "sw123t_bull_c3",
        "sw123t_bear_c3",
    ]
    prefix = "s123t"
    triggers = {}

    # If columns are missing return all-False triggers
    for col in required_cols:
        if col not in df.columns:
            false_series = pd.Series(False, index=df.index)
            triggers[f"{prefix}_bullish_c2_detected"] = false_series
            triggers[f"{prefix}_bullish_c2_detected_ib"] = false_series.copy()
            triggers[f"{prefix}_bullish_c3_confirmed"] = false_series.copy()
            triggers[f"{prefix}_bearish_c2_detected"] = false_series.copy()
            triggers[f"{prefix}_bearish_c2_detected_ib"] = false_series.copy()
            triggers[f"{prefix}_bearish_c3_confirmed"] = false_series.copy()
            triggers[f"{prefix}_pattern_reset"] = false_series.copy()
            return triggers

    lookback = params.get("lookback_bars", 1)
    min_bars_needed = lookback + 1

    bull_c2 = df["sw123t_bull_c2"].astype(bool)
    bear_c2 = df["sw123t_bear_c2"].astype(bool)
    bull_c3 = df["sw123t_bull_c3"].astype(bool)
    bear_c3 = df["sw123t_bear_c3"].astype(bool)

    # Derive current state (same priority as interpreter)
    conditions = [bull_c3, bear_c3, bull_c2, bear_c2]
    choices = ["BULLISH_C3", "BEARISH_C3", "BULLISH_C2", "BEARISH_C2"]
    current_state = pd.Series(
        np.select(conditions, choices, default="NEUTRAL"),
        index=df.index,
        dtype=str
    )
    prev_state = current_state.shift(1)

    # Mark early bars as None-equivalent (use empty string as sentinel)
    current_state.iloc[:min_bars_needed] = ""
    prev_state.iloc[:min_bars_needed] = ""

    # Active = any non-neutral, non-empty state
    active_states = {"BULLISH_C2", "BULLISH_C3", "BEARISH_C2", "BEARISH_C3"}

    # Fires when current bar IS that state (pattern just detected on this bar close)
    triggers[f"{prefix}_bullish_c2_detected"] = (current_state == "BULLISH_C2")
    triggers[f"{prefix}_bullish_c3_confirmed"] = (current_state == "BULLISH_C3")
    triggers[f"{prefix}_bearish_c2_detected"] = (current_state == "BEARISH_C2")
    triggers[f"{prefix}_bearish_c3_confirmed"] = (current_state == "BEARISH_C3")

    # Intra-bar triggers: entry level cross — emitted on C2 bars so the engine
    # can watch price cross the level during the bar. These are boolean Series
    # matching the C2 detection (the engine uses level_column for the actual
    # price cross evaluation; here we simply flag the bars where the level is active).
    triggers[f"{prefix}_bullish_c2_detected_ib"] = (current_state == "BULLISH_C2")
    triggers[f"{prefix}_bearish_c2_detected_ib"] = (current_state == "BEARISH_C2")

    # Pattern reset: transitioned FROM an active state TO NEUTRAL
    prev_was_active = prev_state.isin(active_states)
    curr_is_neutral = (current_state == "NEUTRAL")
    triggers[f"{prefix}_pattern_reset"] = prev_was_active & curr_is_neutral

    return triggers