"""Bollinger Bands — batch indicator (thin wrapper over the incremental class).

Replays each row through `BollingerBandsIncremental.update_bar`. Both
the batch and live paths run the same code, guaranteeing bit-equivalent
output and letting the parity simulator confirm PASS without managing
two parallel implementations.
"""

import pandas as pd
from indicator_incremental import BollingerBandsIncremental


def calculate_bollinger_bands(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = BollingerBandsIncremental(**params)

    bb_upper = []
    bb_basis = []
    bb_lower = []
    bb_bandwidth = []
    bb_upper_prev = []
    bb_basis_prev = []
    bb_lower_prev = []
    cross_upper = []
    cross_lower = []
    cross_basis_up = []
    cross_basis_down = []
    squeeze_on = []
    squeeze_off = []

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        bb_upper.append(out["bb_upper"])
        bb_basis.append(out["bb_basis"])
        bb_lower.append(out["bb_lower"])
        bb_bandwidth.append(out["bb_bandwidth"])
        bb_upper_prev.append(out["bb_upper_prev"])
        bb_basis_prev.append(out["bb_basis_prev"])
        bb_lower_prev.append(out["bb_lower_prev"])
        cross_upper.append(out["bb_cross_upper"])
        cross_lower.append(out["bb_cross_lower"])
        cross_basis_up.append(out["bb_cross_basis_up"])
        cross_basis_down.append(out["bb_cross_basis_down"])
        squeeze_on.append(out["bb_squeeze_on"])
        squeeze_off.append(out["bb_squeeze_off"])

    result = df.copy()
    result["bb_upper"] = bb_upper
    result["bb_basis"] = bb_basis
    result["bb_lower"] = bb_lower
    result["bb_bandwidth"] = bb_bandwidth
    result["bb_upper_prev"] = bb_upper_prev
    result["bb_basis_prev"] = bb_basis_prev
    result["bb_lower_prev"] = bb_lower_prev
    # Trigger booleans cached for interpreter.py to forward without
    # recomputing — single source of truth.
    result["__bb_cross_upper"] = cross_upper
    result["__bb_cross_lower"] = cross_lower
    result["__bb_cross_basis_up"] = cross_basis_up
    result["__bb_cross_basis_down"] = cross_basis_down
    result["__bb_squeeze_on"] = squeeze_on
    result["__bb_squeeze_off"] = squeeze_off
    return result
