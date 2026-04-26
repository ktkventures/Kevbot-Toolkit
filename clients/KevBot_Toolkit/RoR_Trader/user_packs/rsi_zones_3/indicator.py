"""RSI Zones 3 — batch indicator (thin wrapper over the incremental class)."""

import pandas as pd
from indicator_incremental import RsiZones3Incremental


_TRIGGER_KEYS = (
    "rsi3_cross_into_overbought",
    "rsi3_cross_into_oversold",
    "rsi3_exit_overbought_zone",
    "rsi3_exit_oversold_zone",
    "rsi3_overbought_long_exit",
    "rsi3_oversold_short_exit",
)


def calculate_rsi_zones_3(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = RsiZones3Incremental(**params)

    rsi_series: list = []
    signal_series: list = []
    trig_lists: dict = {k: [] for k in _TRIGGER_KEYS}

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        rsi_series.append(out["rsi_z3"])
        signal_series.append(out["rsi_z3_signal"])
        for k in _TRIGGER_KEYS:
            trig_lists[k].append(out[k])

    result = df.copy()
    result["rsi_z3"] = rsi_series
    result["rsi_z3_signal"] = signal_series
    for k in _TRIGGER_KEYS:
        result[f"__{k}"] = trig_lists[k]
    return result
