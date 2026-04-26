"""RVOL v2 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_rvol_v2(df: pd.DataFrame, **params) -> pd.Series:
    def classify(row):
        rvol = row.get("rv2_rvol", np.nan)
        if pd.isna(rvol) or rvol == 0:
            return None
        if rvol > 3.0:
            return "EXTREME"
        if rvol > 1.5:
            return "HIGH"
        if rvol > 0.75:
            return "NORMAL"
        if rvol > 0.5:
            return "LOW"
        return "MINIMAL"

    return df.apply(classify, axis=1)


def detect_rvol_v2_triggers(df: pd.DataFrame, **params) -> dict:
    return {
        "rv2_spike":   df["__rv2_spike"].fillna(False).astype(bool),
        "rv2_extreme": df["__rv2_extreme"].fillna(False).astype(bool),
        "rv2_fade":    df["__rv2_fade"].fillna(False).astype(bool),
    }
