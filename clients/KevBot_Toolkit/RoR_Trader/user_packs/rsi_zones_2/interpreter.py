"""RSI Zones 2 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_rsi_zones_2(df: pd.DataFrame, **params) -> pd.Series:
    ob = params.get("overbought_level", 70.0)
    extreme_ob = params.get("extreme_overbought_level", 80.0)
    os_ = params.get("oversold_level", 30.0)
    extreme_os = params.get("extreme_oversold_level", 20.0)

    rsi = df["rsi2_rsi"]
    conditions = [
        rsi >= extreme_ob,
        (rsi >= ob) & (rsi < extreme_ob),
        (rsi >= 50) & (rsi < ob),
        (rsi > os_) & (rsi < 50),
        (rsi > extreme_os) & (rsi <= os_),
        rsi <= extreme_os,
    ]
    choices = [
        "EXTREME_OVERBOUGHT",
        "OVERBOUGHT",
        "NEUTRAL_BULLISH",
        "NEUTRAL_BEARISH",
        "OVERSOLD",
        "EXTREME_OVERSOLD",
    ]
    result = np.select(conditions, choices, default=None)
    state_series = pd.Series(result, index=df.index, dtype=object)
    state_series = state_series.where(rsi.notna(), other=None)
    return state_series


def detect_rsi_zones_2_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "rsi2_cross_into_overbought",
        "rsi2_cross_into_extreme_overbought",
        "rsi2_cross_into_oversold",
        "rsi2_cross_into_extreme_oversold",
        "rsi2_exit_overbought_zone",
        "rsi2_exit_oversold_zone",
        "rsi2_cross_above_midline",
        "rsi2_cross_below_midline",
        "rsi2_exit_long_overbought",
        "rsi2_exit_short_oversold",
        "rsi2_exit_long_midline_cross_down",
        "rsi2_exit_short_midline_cross_up",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
