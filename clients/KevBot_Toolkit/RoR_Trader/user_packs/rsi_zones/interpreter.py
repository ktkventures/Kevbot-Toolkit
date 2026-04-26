"""RSI Zones — interpreter (state classification + trigger forwarding)."""

import pandas as pd


def interpret_rsi_zones(df: pd.DataFrame, **params) -> pd.Series:
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
    keys = (
        "rsi_cross_into_overbought",
        "rsi_cross_into_oversold",
        "rsi_exit_overbought",
        "rsi_exit_oversold",
        "rsi_cross_into_extreme_overbought",
        "rsi_cross_into_extreme_oversold",
        "rsi_cross_above_midline",
        "rsi_cross_below_midline",
        "rsi_exit_long_overbought",
        "rsi_exit_short_oversold",
        "rsi_exit_long_midline_down",
        "rsi_exit_short_midline_up",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
