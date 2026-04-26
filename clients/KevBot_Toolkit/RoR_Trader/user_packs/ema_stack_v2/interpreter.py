"""EMA Stack v2 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_ema_stack_v2(df: pd.DataFrame, **params) -> pd.Series:
    """Six mutually exclusive states based on the relative ordering of
    short/mid/long EMAs."""
    def classify(row):
        s = row.get("esv2_short", np.nan)
        m = row.get("esv2_mid", np.nan)
        l_ = row.get("esv2_long", np.nan)
        if pd.isna(s) or pd.isna(m) or pd.isna(l_):
            return None
        if s > m and m > l_: return "SML"
        if s > l_ and l_ > m: return "SLM"
        if m > s and s > l_: return "MSL"
        if m > l_ and l_ > s: return "MLS"
        if l_ > s and s > m: return "LSM"
        if l_ > m and m > s: return "LMS"
        return "SML"

    return df.apply(classify, axis=1)


def detect_ema_stack_v2_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "esv2_cross_bull",
        "esv2_cross_bear",
        "esv2_mid_cross_bull",
        "esv2_mid_cross_bear",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
