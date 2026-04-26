"""Stochastic Oscillator — batch indicator (thin wrapper)."""

import pandas as pd
from indicator_incremental import StochasticOscillatorIncremental


_TRIGGER_KEYS = (
    "sto_k_cross_above_d_oversold",
    "sto_k_cross_below_d_overbought",
    "sto_k_cross_above_d_midrange",
    "sto_k_cross_below_d_midrange",
    "sto_enter_overbought",
    "sto_exit_overbought",
    "sto_enter_oversold",
    "sto_exit_oversold",
    "sto_k_cross_above_midline",
    "sto_k_cross_below_midline",
)


def calculate_stochastic_oscillator(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = StochasticOscillatorIncremental(**params)
    k_series: list = []
    d_series: list = []
    trig_lists: dict = {k: [] for k in _TRIGGER_KEYS}

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        k_series.append(out["stoch_k"])
        d_series.append(out["stoch_d"])
        for k in _TRIGGER_KEYS:
            trig_lists[k].append(out[k])

    result = df.copy()
    result["stoch_k"] = k_series
    result["stoch_d"] = d_series
    for k in _TRIGGER_KEYS:
        result[f"__{k}"] = trig_lists[k]
    return result
