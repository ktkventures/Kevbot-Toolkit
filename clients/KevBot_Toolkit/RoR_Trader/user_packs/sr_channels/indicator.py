"""S/R Channels — batch indicator (thin wrapper over the incremental class)."""

import pandas as pd
from indicator_incremental import SrChannelsIncremental


_TRIGGER_KEYS = (
    "src_resistance_broken",
    "src_support_broken",
    "src_enter_sr_zone",
    "src_exit_sr_zone",
)

_INDICATOR_KEYS = (
    "src_nearest_top",
    "src_nearest_bot",
    "src_num_channels",
    "src_in_channel",
    "src_res_broken",
    "src_sup_broken",
    "src_nearest_top_prev",
    "src_nearest_bot_prev",
)


def calculate_sr_channels(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = SrChannelsIncremental(**params)
    ind_lists: dict = {k: [] for k in _INDICATOR_KEYS}
    trig_lists: dict = {k: [] for k in _TRIGGER_KEYS}

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        for k in _INDICATOR_KEYS:
            ind_lists[k].append(out[k])
        for k in _TRIGGER_KEYS:
            trig_lists[k].append(out[k])

    result = df.copy()
    for k in _INDICATOR_KEYS:
        result[k] = ind_lists[k]
    for k in _TRIGGER_KEYS:
        result[f"__{k}"] = trig_lists[k]
    return result
