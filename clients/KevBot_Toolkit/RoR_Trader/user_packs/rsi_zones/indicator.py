"""RSI Zones — batch indicator (thin wrapper over the incremental class)."""

import pandas as pd
from indicator_incremental import RsiZonesIncremental


_TRIGGER_KEYS = (
    "rsi_cross_into_overbought",
    "rsi_cross_into_oversold",
    "rsi_exit_overbought",
    "rsi_exit_oversold",
    "rsi_cross_into_extreme_overbought",
    "rsi_cross_into_extreme_oversold",
    "rsi_cross_above_midline",
    "rsi_cross_below_midline",
    "rsi_exit_long_overbought",
    "rsi_exit_short_oversold",
    "rsi_exit_long_midline_down",
    "rsi_exit_short_midline_up",
)


def calculate_rsi_zones(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = RsiZonesIncremental(**params)
    rsi_series: list = []
    trig_lists: dict = {k: [] for k in _TRIGGER_KEYS}

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        rsi_series.append(out["rsi_value"])
        for k in _TRIGGER_KEYS:
            trig_lists[k].append(out[k])

    result = df.copy()
    result["rsi_value"] = rsi_series
    for k in _TRIGGER_KEYS:
        result[f"__{k}"] = trig_lists[k]
    return result
