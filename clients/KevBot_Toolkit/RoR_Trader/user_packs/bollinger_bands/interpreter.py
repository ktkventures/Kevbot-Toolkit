import pandas as pd
import numpy as np

def interpret_bollinger_bands(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar into a mutually exclusive Bollinger Band zone.

    Squeeze states take priority when bandwidth is below threshold.
    Zone is determined by price position relative to basis and band width.

    Returns states: SQUEEZE_UPPER, SQUEEZE_MID, SQUEEZE_LOWER,
                    UPPER_ZONE, MID_ZONE, LOWER_ZONE
    """
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
    """
    Detect Bollinger Band trigger events: band crosses and squeeze transitions.

    Returns dict of trigger_id -> boolean Series.
    """
    squeeze_threshold = params.get("squeeze_threshold", 0.04)
    prefix = "bb"

    triggers = {}

    close = df["close"]
    close_prev = close.shift(1)
    upper = df["bb_upper"]
    upper_prev = upper.shift(1)
    lower = df["bb_lower"]
    lower_prev = lower.shift(1)
    basis = df["bb_basis"]
    basis_prev = basis.shift(1)
    bw = df["bb_bandwidth"]
    bw_prev = bw.shift(1)

    # Cross above upper band
    triggers[f"{prefix}_cross_upper"] = (close > upper) & (close_prev <= upper_prev)

    # Cross below lower band
    triggers[f"{prefix}_cross_lower"] = (close < lower) & (close_prev >= lower_prev)

    # Cross above basis
    triggers[f"{prefix}_cross_basis_up"] = (close > basis) & (close_prev <= basis_prev)

    # Cross below basis
    triggers[f"{prefix}_cross_basis_down"] = (close < basis) & (close_prev >= basis_prev)

    # Squeeze on: bandwidth drops below threshold
    triggers[f"{prefix}_squeeze_on"] = (bw < squeeze_threshold) & (bw_prev >= squeeze_threshold)

    # Squeeze off: bandwidth rises above threshold
    triggers[f"{prefix}_squeeze_off"] = (bw >= squeeze_threshold) & (bw_prev < squeeze_threshold)

    return triggers