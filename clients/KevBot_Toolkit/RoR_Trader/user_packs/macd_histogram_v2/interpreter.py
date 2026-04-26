"""MACD Histogram v2 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_macd_histogram_v2(df: pd.DataFrame, **params) -> pd.Series:
    """Four-state classification based on histogram sign and direction:
    H+up / H+dn / H-up / H-dn (matching built-in macd_histogram)."""
    def classify(row):
        mh = row.get("mhv2_hist", np.nan)
        if pd.isna(mh):
            return None
        return None  # state determined per-bar by movement, not just sign

    # Pass through using current+prev as the original interpreter does
    result = pd.Series(index=df.index, dtype=object)
    hist = df["mhv2_hist"]
    prev = hist.shift(1)
    valid = hist.notna() & prev.notna()

    # H+up: positive AND rising
    result[valid & (hist > 0) & (hist > prev)] = "H+up"
    # H+dn: positive AND falling
    result[valid & (hist > 0) & (hist <= prev)] = "H+dn"
    # H-up: negative AND rising (less negative)
    result[valid & (hist <= 0) & (hist > prev)] = "H-up"
    # H-dn: negative AND falling (more negative)
    result[valid & (hist <= 0) & (hist <= prev)] = "H-dn"
    return result


def detect_macd_histogram_v2_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "mhv2_flip_pos",
        "mhv2_flip_neg",
        "mhv2_momentum_shift_up",
        "mhv2_momentum_shift_down",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
