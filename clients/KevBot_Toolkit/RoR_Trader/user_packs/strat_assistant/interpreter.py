"""Strat Assistant — interpreter (state classification + trigger forwarding)."""

import pandas as pd


_TYPE_TO_STATE = {
    1: "INSIDE",
    2: "TWO_UP",
    -2: "TWO_DOWN",
    3: "OUTSIDE",
}


def interpret_strat_assistant(df: pd.DataFrame, **params) -> pd.Series:
    def classify(row):
        bt = row.get("strat_bar_type", 0)
        if pd.isna(bt) or bt == 0:
            return None
        return _TYPE_TO_STATE.get(int(bt), None)

    return df.apply(classify, axis=1)


def detect_strat_assistant_triggers(df: pd.DataFrame, **params) -> dict:
    return {
        "strat_bull_c2":     df["__strat_bull_c2"].fillna(False).astype(bool),
        "strat_bear_c2":     df["__strat_bear_c2"].fillna(False).astype(bool),
        "strat_outside_bar": df["__strat_outside_bar"].fillna(False).astype(bool),
        "strat_inside_bar":  df["__strat_inside_bar"].fillna(False).astype(bool),
        "strat_shooter":     df["__strat_shooter"].fillna(False).astype(bool),
        "strat_hammer":      df["__strat_hammer"].fillna(False).astype(bool),
    }
