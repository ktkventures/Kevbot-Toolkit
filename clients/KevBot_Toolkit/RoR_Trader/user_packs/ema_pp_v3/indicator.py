"""EMA Price Position v3 — batch indicator (thin wrapper)."""

import pandas as pd
from indicator_incremental import EmaPpV3Incremental


_INDICATOR_KEYS = ("eppv3_short", "eppv3_mid", "eppv3_long")
_TRIGGER_KEYS = (
    "eppv3_cross_short_up",
    "eppv3_cross_short_down",
    "eppv3_cross_mid_up",
    "eppv3_cross_mid_down",
)


def calculate_ema_pp_v3(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = EmaPpV3Incremental(**params)
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
