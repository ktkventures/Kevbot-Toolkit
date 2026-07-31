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


# ---------------------------------------------------------------------
# Faithful Pine port — board #73, audit §2.6. Selected by the manifest
# `flag_variants` entry for RORT_SR_CHANNELS_PINE (default OFF).
#
# Two corrections against the Pine's OWN colour rule (PINE L158):
#   * a channel is resistance when BOTH bounds are above close, support
#     when BOTH are below, and NEITHER when price sits inside it — the
#     grey `inch_col` case. The legacy interpreter split price-inside-a-
#     channel into IN_RESISTANCE / IN_SUPPORT on the midpoint, asserting
#     a directional bias exactly where the Pine declines to. That case
#     is now the honest third state, IN_CHANNEL.
#   * ABOVE_RESISTANCE / BELOW_SUPPORT were computed against the NEAREST
#     channel only, so price was labelled ABOVE_RESISTANCE while sitting
#     beneath five stronger channels. They now mean what they read as:
#     above ALL channels / below ALL channels.
# ---------------------------------------------------------------------


def interpret_sr_channels_pine(df: pd.DataFrame, **params) -> pd.Series:
    def classify(row):
        n_ch = row.get("src_num_channels", np.nan)
        in_ch = row.get("src_in_channel", np.nan)
        above = row.get("src_ch_above", np.nan)
        below = row.get("src_ch_below", np.nan)

        if pd.isna(n_ch) or n_ch == 0:
            return None
        if in_ch == 1.0:
            return "IN_CHANNEL"
        if pd.isna(above) or pd.isna(below):
            return None
        if below == n_ch:
            return "ABOVE_RESISTANCE"
        if above == n_ch:
            return "BELOW_SUPPORT"
        return "BETWEEN_LEVELS"

    return df.apply(classify, axis=1)
