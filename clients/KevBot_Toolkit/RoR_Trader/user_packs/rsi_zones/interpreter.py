import pandas as pd
import numpy as np


def interpret_rsi_zones(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar into one of six mutually exclusive RSI zone states.
    """
    extreme_ob = params.get("extreme_overbought_level", 80.0)
    ob = params.get("overbought_level", 70.0)
    mid = params.get("midline", 50.0)
    os_ = params.get("oversold_level", 30.0)
    extreme_os = params.get("extreme_oversold_level", 20.0)

    rsi = df["rsi_value"]

    states = pd.Series(index=df.index, dtype=object)
    valid = rsi.notna()

    states[valid & (rsi >= extreme_ob)] = "EXTREME_OVERBOUGHT"
    states[valid & (rsi >= ob) & (rsi < extreme_ob)] = "OVERBOUGHT"
    states[valid & (rsi >= mid) & (rsi < ob)] = "BULLISH_NEUTRAL"
    states[valid & (rsi < mid) & (rsi > os_)] = "BEARISH_NEUTRAL"
    states[valid & (rsi <= os_) & (rsi > extreme_os)] = "OVERSOLD"
    states[valid & (rsi <= extreme_os)] = "EXTREME_OVERSOLD"

    states = states.where(states.notna(), other=None)
    return states


def detect_rsi_zones_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect RSI zone-crossing trigger events.

    Trigger keys are {trigger_prefix}_{base}: rsi_cross_into_overbought, etc.
    The trigger_prefix "rsi" is NOT repeated in the base name.
    """
    extreme_ob = params.get("extreme_overbought_level", 80.0)
    ob = params.get("overbought_level", 70.0)
    mid = params.get("midline", 50.0)
    os_ = params.get("oversold_level", 30.0)
    extreme_os = params.get("extreme_oversold_level", 20.0)

    prefix = "rsi"

    rsi = df["rsi_value"]
    rsi_prev = rsi.shift(1)
    both_valid = rsi.notna() & rsi_prev.notna()

    triggers = {}

    # Entry triggers — zone crossings
    triggers[f"{prefix}_cross_into_overbought"] = (
        both_valid & (rsi >= ob) & (rsi_prev < ob)
    )
    triggers[f"{prefix}_cross_into_oversold"] = (
        both_valid & (rsi <= os_) & (rsi_prev > os_)
    )
    triggers[f"{prefix}_exit_overbought"] = (
        both_valid & (rsi < ob) & (rsi_prev >= ob)
    )
    triggers[f"{prefix}_exit_oversold"] = (
        both_valid & (rsi > os_) & (rsi_prev <= os_)
    )
    triggers[f"{prefix}_cross_into_extreme_overbought"] = (
        both_valid & (rsi >= extreme_ob) & (rsi_prev < extreme_ob)
    )
    triggers[f"{prefix}_cross_into_extreme_oversold"] = (
        both_valid & (rsi <= extreme_os) & (rsi_prev > extreme_os)
    )

    # Midline crosses
    triggers[f"{prefix}_cross_above_midline"] = (
        both_valid & (rsi > mid) & (rsi_prev <= mid)
    )
    triggers[f"{prefix}_cross_below_midline"] = (
        both_valid & (rsi < mid) & (rsi_prev >= mid)
    )

    # Exit triggers
    triggers[f"{prefix}_exit_long_overbought"] = (
        both_valid & (rsi >= ob) & (rsi_prev < ob)
    )
    triggers[f"{prefix}_exit_short_oversold"] = (
        both_valid & (rsi <= os_) & (rsi_prev > os_)
    )
    triggers[f"{prefix}_exit_long_midline_down"] = (
        both_valid & (rsi < mid) & (rsi_prev >= mid)
    )
    triggers[f"{prefix}_exit_short_midline_up"] = (
        both_valid & (rsi > mid) & (rsi_prev <= mid)
    )

    return triggers
