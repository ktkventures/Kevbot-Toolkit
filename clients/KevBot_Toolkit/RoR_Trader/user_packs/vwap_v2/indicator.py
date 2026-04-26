"""VWAP v2 — batch indicator (thin wrapper)."""

import pandas as pd
from indicator_incremental import VwapV2Incremental


_INDICATOR_KEYS = (
    "vwapv2_value", "vwapv2_sd1_upper", "vwapv2_sd1_lower",
    "vwapv2_sd2_upper", "vwapv2_sd2_lower",
)
_TRIGGER_KEYS = (
    "vwapv2_cross_above",
    "vwapv2_cross_below",
    "vwapv2_enter_upper_extreme",
    "vwapv2_enter_lower_extreme",
    "vwapv2_return_to_vwap",
)


def calculate_vwap_v2(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = VwapV2Incremental(**params)
    ind: dict = {k: [] for k in _INDICATOR_KEYS}
    trig: dict = {k: [] for k in _TRIGGER_KEYS}

    for ts, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
            "timestamp": ts,
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
