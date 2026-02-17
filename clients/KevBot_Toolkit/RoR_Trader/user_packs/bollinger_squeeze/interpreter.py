import pandas as pd
import numpy as np

def interpret_bollinger_squeeze(df: pd.DataFrame, **params) -> pd.Series:
    squeeze_pct = params.get("squeeze_pct", 4.0)

    def classify(row):
        close = row.get("close", np.nan)
        upper = row.get("bb_upper", np.nan)
        lower = row.get("bb_lower", np.nan)
        basis = row.get("bb_basis", np.nan)
        bw = row.get("bb_bandwidth", np.nan)

        if pd.isna(close) or pd.isna(upper) or pd.isna(basis):
            return None

        if bw <= squeeze_pct:
            return "SQUEEZE"

        if close > upper:
            return "ABOVE_UPPER"
        elif close < lower:
            return "BELOW_LOWER"
        else:
            mid_zone_width = (upper - lower) * 0.25
            if abs(close - basis) <= mid_zone_width:
                return "MID_ZONE"
            elif close > basis:
                return "UPPER_ZONE"
            else:
                return "LOWER_ZONE"

    return df.apply(classify, axis=1)


def detect_bollinger_squeeze_triggers(df: pd.DataFrame, **params) -> dict:
    squeeze_pct = params.get("squeeze_pct", 4.0)
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

    triggers[f"{prefix}_cross_upper"] = (close > upper) & (close_prev <= upper_prev)
    triggers[f"{prefix}_cross_lower"] = (close < lower) & (close_prev >= lower_prev)
    triggers[f"{prefix}_cross_above_basis"] = (close > basis) & (close_prev <= basis_prev)
    triggers[f"{prefix}_cross_below_basis"] = (close < basis) & (close_prev >= basis_prev)
    triggers[f"{prefix}_squeeze_on"] = (bw <= squeeze_pct) & (bw_prev > squeeze_pct)
    triggers[f"{prefix}_squeeze_off"] = (bw > squeeze_pct) & (bw_prev <= squeeze_pct)

    return triggers