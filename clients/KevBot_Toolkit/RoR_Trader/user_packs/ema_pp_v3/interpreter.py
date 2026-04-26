"""EMA Price Position v3 — interpreter (state classification + trigger forwarding)."""

import pandas as pd
import numpy as np


def interpret_ema_pp_v3(df: pd.DataFrame, **params) -> pd.Series:
    """Classify the position of price relative to the three EMAs by ordering.

    Sorts (Price, Short, Mid, Long) descending by value, joins their
    initials. Result is one of 24 permutations (4!) — same scheme as
    the built-in ema_price_position interpreter.
    """
    def classify(row):
        s = row.get("eppv3_short", np.nan)
        m = row.get("eppv3_mid", np.nan)
        l_ = row.get("eppv3_long", np.nan)
        c = row.get("close", np.nan)
        if pd.isna(s) or pd.isna(m) or pd.isna(l_) or pd.isna(c):
            return None
        items = [("P", c), ("S", s), ("M", m), ("L", l_)]
        items.sort(key=lambda x: -x[1])
        return "".join(x[0] for x in items)

    return df.apply(classify, axis=1)


def detect_ema_pp_v3_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "eppv3_cross_short_up",
        "eppv3_cross_short_down",
        "eppv3_cross_mid_up",
        "eppv3_cross_mid_down",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
