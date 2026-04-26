"""S/R Channels — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_sr_channels(df: pd.DataFrame, **params) -> pd.Series:
    def classify(row):
        top = row.get("src_nearest_top", np.nan)
        bot = row.get("src_nearest_bot", np.nan)
        in_ch = row.get("src_in_channel", np.nan)
        n_ch = row.get("src_num_channels", np.nan)
        close = row.get("close", np.nan)

        if pd.isna(top) or pd.isna(bot) or pd.isna(close) or n_ch == 0:
            return None

        mid = (top + bot) / 2.0
        if in_ch == 1.0:
            return "IN_RESISTANCE" if close >= mid else "IN_SUPPORT"
        if close > top:
            return "ABOVE_RESISTANCE"
        if close < bot:
            return "BELOW_SUPPORT"
        return "BETWEEN_LEVELS"

    return df.apply(classify, axis=1)


def detect_sr_channels_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "src_resistance_broken",
        "src_support_broken",
        "src_enter_sr_zone",
        "src_exit_sr_zone",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
