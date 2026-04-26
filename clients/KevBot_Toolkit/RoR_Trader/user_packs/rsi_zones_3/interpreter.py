"""RSI Zones 3 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_rsi_zones_3(df: pd.DataFrame, **params) -> pd.Series:
    overbought = float(params.get("overbought_level", 70.0))
    oversold = float(params.get("oversold_level", 30.0))

    signal = df["rsi_z3_signal"].values
    states = pd.Series(index=df.index, dtype=object)
    states[:] = None

    valid = ~np.isnan(signal)
    states[valid & (signal >= overbought)] = "OVERBOUGHT"
    states[valid & (signal <= oversold)] = "OVERSOLD"
    states[valid & (signal > oversold) & (signal < overbought)] = "NEUTRAL"
    return states


def detect_rsi_zones_3_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "rsi3_cross_into_overbought",
        "rsi3_cross_into_oversold",
        "rsi3_exit_overbought_zone",
        "rsi3_exit_oversold_zone",
        "rsi3_overbought_long_exit",
        "rsi3_oversold_short_exit",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
