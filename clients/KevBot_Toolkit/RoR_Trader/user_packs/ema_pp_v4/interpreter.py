"""EMA Price Position v4 — interpreter (state classification + trigger forwarding).

Same classification scheme as v3 (24 P/S/M/L permutations), reading
the `_prev`-suffixed indicator columns.
"""

import pandas as pd
import numpy as np


def interpret_ema_pp_v4(df: pd.DataFrame, **params) -> pd.Series:
    def classify(row):
        s = row.get("eppv4_short_prev", np.nan)
        m = row.get("eppv4_mid_prev", np.nan)
        l_ = row.get("eppv4_long_prev", np.nan)
        c = row.get("close", np.nan)
        if pd.isna(s) or pd.isna(m) or pd.isna(l_) or pd.isna(c):
            return None
        items = [("P", c), ("S", s), ("M", m), ("L", l_)]
        items.sort(key=lambda x: -x[1])
        return "".join(x[0] for x in items)

    return df.apply(classify, axis=1)


def detect_ema_pp_v4_triggers(df: pd.DataFrame, **params) -> dict:
    keys = (
        "eppv4_cross_short_up",
        "eppv4_cross_short_down",
        "eppv4_cross_mid_up",
        "eppv4_cross_mid_down",
    )
    return {k: df[f"__{k}"].fillna(False).astype(bool) for k in keys}
