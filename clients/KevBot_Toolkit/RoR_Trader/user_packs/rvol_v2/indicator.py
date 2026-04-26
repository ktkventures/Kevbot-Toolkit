"""RVOL v2 — batch indicator (thin wrapper over the incremental class)."""

import pandas as pd
from indicator_incremental import RvolV2Incremental


def calculate_rvol_v2(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = RvolV2Incremental(**params)
    rvol_series: list = []
    sma_series: list = []
    spike: list = []
    extreme: list = []
    fade: list = []
    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        rvol_series.append(out["rv2_rvol"])
        sma_series.append(out["rv2_vol_sma"])
        spike.append(out["rv2_spike"])
        extreme.append(out["rv2_extreme"])
        fade.append(out["rv2_fade"])

    result = df.copy()
    result["rv2_rvol"] = rvol_series
    result["rv2_vol_sma"] = sma_series
    result["__rv2_spike"] = spike
    result["__rv2_extreme"] = extreme
    result["__rv2_fade"] = fade
    return result
