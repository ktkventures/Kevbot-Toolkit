import pandas as pd
import numpy as np


def interpret_strat_assistant(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar's Strat pattern type.

    States (based on bar type):
        INSIDE: 1-bar (consolidation)
        TWO_UP: 2-up bar (bullish expansion)
        TWO_DOWN: 2-down bar (bearish expansion)
        OUTSIDE: 3-bar (outside/engulfing)
    """
    type_map = {
        1: "INSIDE",
        2: "TWO_UP",
        -2: "TWO_DOWN",
        3: "OUTSIDE",
    }

    def classify(row):
        bt = row.get("strat_bar_type", 0)
        if pd.isna(bt) or bt == 0:
            return None
        return type_map.get(int(bt), None)

    return df.apply(classify, axis=1)


def detect_strat_assistant_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect Strat pattern triggers.

    Triggers:
        strat_bull_c2: 2-Up bar detected
        strat_bear_c2: 2-Down bar detected
        strat_outside_bar: 3-bar (outside) detected
        strat_inside_bar: 1-bar (inside) detected
        strat_shooter: Shooter signal
        strat_hammer: Hammer signal
    """
    prefix = "strat"
    triggers = {}

    bar_type = df["strat_bar_type"]
    actionable = df["strat_actionable"]

    triggers[f"{prefix}_bull_c2"] = bar_type == 2
    triggers[f"{prefix}_bear_c2"] = bar_type == -2
    triggers[f"{prefix}_outside_bar"] = bar_type == 3
    triggers[f"{prefix}_inside_bar"] = bar_type == 1
    triggers[f"{prefix}_shooter"] = actionable == "SHOOTER"
    triggers[f"{prefix}_hammer"] = actionable == "HAMMER"

    return triggers
