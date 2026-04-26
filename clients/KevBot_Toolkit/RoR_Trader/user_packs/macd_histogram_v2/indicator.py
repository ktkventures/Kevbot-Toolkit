"""MACD Histogram v2 — batch indicator (thin wrapper)."""

import pandas as pd
from indicator_incremental import MacdHistogramV2Incremental


_INDICATOR_KEYS = ("mhv2_line", "mhv2_signal", "mhv2_hist")
_TRIGGER_KEYS = (
    "mhv2_flip_pos",
    "mhv2_flip_neg",
    "mhv2_momentum_shift_up",
    "mhv2_momentum_shift_down",
)


def calculate_macd_histogram_v2(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = MacdHistogramV2Incremental(**params)
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
