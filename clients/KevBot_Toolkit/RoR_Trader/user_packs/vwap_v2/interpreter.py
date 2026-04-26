"""VWAP v2 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_vwap_v2(df: pd.DataFrame, **params) -> pd.Series:
    def classify(row):
        vwap = row.get("vwapv2_value", np.nan)
        price = row.get("close", np.nan)
        sd1u = row.get("vwapv2_sd1_upper", np.nan)
        sd1l = row.get("vwapv2_sd1_lower", np.nan)
        sd2u = row.get("vwapv2_sd2_upper", np.nan)
        sd2l = row.get("vwapv2_sd2_lower", np.nan)
        if pd.isna(vwap) or vwap == 0 or pd.isna(price) or pd.isna(sd1u):
            return None
        half_sd = (sd1u - vwap) * 0.5
        at_upper = vwap + half_sd
        at_lower = vwap - half_sd
        if price > sd2u: return ">+2σ"
        if price > sd1u: return ">+1σ"
        if price > at_upper: return ">V"
        if price >= at_lower: return "@V"
        if price >= sd1l: return "<V"
        if price >= sd2l: return "<-1σ"
        return "<-2σ"

    return df.apply(classify, axis=1)


def detect_vwap_v2_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "vwapv2_cross_above",
        "vwapv2_cross_below",
        "vwapv2_enter_upper_extreme",
        "vwapv2_enter_lower_extreme",
        "vwapv2_return_to_vwap",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
