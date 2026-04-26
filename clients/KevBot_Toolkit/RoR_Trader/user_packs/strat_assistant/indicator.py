"""Strat Assistant — batch indicator (thin wrapper over the incremental class)."""

import pandas as pd
from indicator_incremental import StratAssistantIncremental


def calculate_strat_assistant(df: pd.DataFrame, **params) -> pd.DataFrame:
    engine = StratAssistantIncremental(**params)

    bar_type: list = []
    combo: list = []
    actionable: list = []
    candle_color: list = []
    bull_c2: list = []
    bear_c2: list = []
    outside_bar: list = []
    inside_bar: list = []
    shooter: list = []
    hammer: list = []

    for _, row in df.iterrows():
        out = engine.update_bar({
            "open": row.get("open", 0.0),
            "high": row.get("high", 0.0),
            "low": row.get("low", 0.0),
            "close": row.get("close", 0.0),
            "volume": row.get("volume", 0.0),
        })
        bar_type.append(out["strat_bar_type"])
        combo.append(out["strat_combo"])
        actionable.append(out["strat_actionable"])
        candle_color.append(out["strat_candle_color"])
        bull_c2.append(out["strat_bull_c2"])
        bear_c2.append(out["strat_bear_c2"])
        outside_bar.append(out["strat_outside_bar"])
        inside_bar.append(out["strat_inside_bar"])
        shooter.append(out["strat_shooter"])
        hammer.append(out["strat_hammer"])

    result = df.copy()
    result["strat_bar_type"] = bar_type
    result["strat_combo"] = combo
    result["strat_actionable"] = actionable
    result["strat_candle_color"] = candle_color
    # Trigger booleans cached for interpreter forwarding.
    result["__strat_bull_c2"] = bull_c2
    result["__strat_bear_c2"] = bear_c2
    result["__strat_outside_bar"] = outside_bar
    result["__strat_inside_bar"] = inside_bar
    result["__strat_shooter"] = shooter
    result["__strat_hammer"] = hammer
    return result
