import pandas as pd
import numpy as np


def interpret_sr_channels(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar's position relative to the nearest S/R channel.
    
    States:
    - ABOVE_RESISTANCE: Price above all S/R channels
    - IN_RESISTANCE: Price inside a channel acting as resistance
    - BETWEEN_LEVELS: Price between channels, not inside any
    - IN_SUPPORT: Price inside a channel acting as support
    - BELOW_SUPPORT: Price below all S/R channels
    """

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
            # Inside a channel — classify as support or resistance
            if close >= mid:
                return "IN_RESISTANCE"
            else:
                return "IN_SUPPORT"
        else:
            # Outside all channels
            if close > top:
                return "ABOVE_RESISTANCE"
            elif close < bot:
                return "BELOW_SUPPORT"
            else:
                return "BETWEEN_LEVELS"

    return df.apply(classify, axis=1)


def detect_sr_channels_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect S/R channel trigger events:
    - resistance_broken: close breaks above a resistance channel top
    - support_broken: close breaks below a support channel bottom
    - enter_sr_zone: price enters any S/R channel
    - exit_sr_zone: price exits any S/R channel
    """
    prefix = "src"
    triggers = {}

    res_broken = df["src_res_broken"]
    sup_broken = df["src_sup_broken"]
    in_ch = df["src_in_channel"]
    in_ch_prev = in_ch.shift(1)

    triggers[f"{prefix}_resistance_broken"] = res_broken == 1.0
    triggers[f"{prefix}_support_broken"] = sup_broken == 1.0
    triggers[f"{prefix}_enter_sr_zone"] = (in_ch == 1.0) & (in_ch_prev == 0.0)
    triggers[f"{prefix}_exit_sr_zone"] = (in_ch == 0.0) & (in_ch_prev == 1.0)

    return triggers