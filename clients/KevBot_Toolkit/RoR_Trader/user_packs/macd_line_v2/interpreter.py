"""MACD Line v2 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_macd_line_v2(df: pd.DataFrame, **params) -> pd.Series:
    """Four-state classification: M>S+ / M>S- / M<S+ / M<S- (matches built-in macd_line)."""
    def classify(row):
        ml = row.get("mlv2_line", np.nan)
        ms = row.get("mlv2_signal", np.nan)
        if pd.isna(ml) or pd.isna(ms):
            return None
        if ml > ms:
            return "M>S+" if ml > 0 else "M>S-"
        return "M<S-" if ml <= 0 else "M<S+"

    return df.apply(classify, axis=1)


def detect_macd_line_v2_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "mlv2_cross_bull",
        "mlv2_cross_bear",
        "mlv2_zero_cross_up",
        "mlv2_zero_cross_down",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
