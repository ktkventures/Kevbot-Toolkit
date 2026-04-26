"""RSI Zones 2 — batch indicator (thin wrapper over the incremental class)."""

import pandas as pd
from indicator_incremental import RsiZones2Incremental


_TRIGGER_KEYS = (
    "rsi2_cross_into_overbought",
    "rsi2_cross_into_extreme_overbought",
    "rsi2_cross_into_oversold",
    "rsi2_cross_into_extreme_oversold",
    "rsi2_exit_overbought_zone",
    "rsi2_exit_oversold_zone",
    "rsi2_cross_above_midline",
    "rsi2_cross_below_midline",
    "rsi2_exit_long_overbought",
    "rsi2_exit_short_oversold",
    "rsi2_exit_long_midline_cross_down",
    "rsi2_exit_short_midline_cross_up",
)


def calculate_rsi_zones_2(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = RsiZones2Incremental(**params)
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
        rsi_series.append(out["rsi2_rsi"])
        for k in _TRIGGER_KEYS:
            trig_lists[k].append(out[k])

    result = df.copy()
    result["rsi2_rsi"] = rsi_series
    for k in _TRIGGER_KEYS:
        result[f"__{k}"] = trig_lists[k]
    return result
