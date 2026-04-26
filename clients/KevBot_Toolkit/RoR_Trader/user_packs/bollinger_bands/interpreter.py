"""Bollinger Bands — interpreter (state classification + trigger forwarding).

State classification is unchanged from the legacy implementation —
six mutually exclusive zones based on price position vs basis and
whether bandwidth is below the squeeze threshold.

Triggers are pre-computed by the incremental class and cached as
`__bb_*` columns by the batch wrapper; this function just forwards
them under the engine-expected `{prefix}_{base}` keys. No
recomputation here keeps the source of truth in one place.
"""

import pandas as pd
import numpy as np


def interpret_bollinger_bands(df: pd.DataFrame, **params) -> pd.Series:
    squeeze_threshold = params.get("squeeze_threshold", 0.04)

    def classify(row):
        upper = row.get("bb_upper", np.nan)
        basis = row.get("bb_basis", np.nan)
        lower = row.get("bb_lower", np.nan)
        bw = row.get("bb_bandwidth", np.nan)
        close = row.get("close", np.nan)

        if pd.isna(upper) or pd.isna(basis) or pd.isna(lower) or pd.isna(bw) or pd.isna(close):
            return None

        band_width = upper - lower
        mid_zone_half = band_width * 0.25  # 25% of total width from basis = mid zone
        is_squeeze = bw < squeeze_threshold

        if close >= basis + mid_zone_half:
            return "SQUEEZE_UPPER" if is_squeeze else "UPPER_ZONE"
        elif close <= basis - mid_zone_half:
            return "SQUEEZE_LOWER" if is_squeeze else "LOWER_ZONE"
        else:
            return "SQUEEZE_MID" if is_squeeze else "MID_ZONE"

    return df.apply(classify, axis=1)


def detect_bollinger_bands_triggers(df: pd.DataFrame, **params) -> dict:
    return {
        "bb_cross_upper":      df["__bb_cross_upper"].fillna(False).astype(bool),
        "bb_cross_lower":      df["__bb_cross_lower"].fillna(False).astype(bool),
        "bb_cross_basis_up":   df["__bb_cross_basis_up"].fillna(False).astype(bool),
        "bb_cross_basis_down": df["__bb_cross_basis_down"].fillna(False).astype(bool),
        "bb_squeeze_on":       df["__bb_squeeze_on"].fillna(False).astype(bool),
        "bb_squeeze_off":      df["__bb_squeeze_off"].fillna(False).astype(bool),
    }
