"""Stochastic Oscillator — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_stochastic_oscillator(df: pd.DataFrame, **params) -> pd.Series:
    overbought = params.get("overbought", 80.0)
    oversold = params.get("oversold", 20.0)
    midline = params.get("midline", 50.0)

    def classify(row):
        k = row.get("stoch_k", np.nan)
        d = row.get("stoch_d", np.nan)
        if pd.isna(k) or pd.isna(d):
            return None
        if k >= overbought:
            return "OVERBOUGHT_BEARISH" if k < d else "OVERBOUGHT_BULLISH"
        if k <= oversold:
            return "OVERSOLD_BULLISH" if k > d else "OVERSOLD_BEARISH"
        return "BULLISH_MIDRANGE" if k >= midline else "BEARISH_MIDRANGE"

    return df.apply(classify, axis=1)


def detect_stochastic_oscillator_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "sto_k_cross_above_d_oversold",
        "sto_k_cross_below_d_overbought",
        "sto_k_cross_above_d_midrange",
        "sto_k_cross_below_d_midrange",
        "sto_enter_overbought",
        "sto_exit_overbought",
        "sto_enter_oversold",
        "sto_exit_oversold",
        "sto_k_cross_above_midline",
        "sto_k_cross_below_midline",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
