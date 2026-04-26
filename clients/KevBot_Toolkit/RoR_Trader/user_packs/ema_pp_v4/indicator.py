"""EMA Price Position v4 — batch indicator (thin wrapper)."""

import pandas as pd
from indicator_incremental import EmaPpV4Incremental


_INDICATOR_KEYS = ("eppv4_short_prev", "eppv4_mid_prev", "eppv4_long_prev")
_TRIGGER_KEYS = (
    "eppv4_cross_short_up",
    "eppv4_cross_short_down",
    "eppv4_cross_mid_up",
    "eppv4_cross_mid_down",
)


def calculate_ema_pp_v4(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = EmaPpV4Incremental(**params)
    ind: dict = {k: [] for k in _INDICATOR_KEYS}
    trig: dict = {k: [] for k in _TRIGGER_KEYS}
    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        for k in _INDICATOR_KEYS:
            ind[k].append(out[k])
        for k in _TRIGGER_KEYS:
            trig[k].append(out[k])
    result = df.copy()
    for k in _INDICATOR_KEYS:
        result[k] = ind[k]
    for k in _TRIGGER_KEYS:
        result[f"__{k}"] = trig[k]
    return result
