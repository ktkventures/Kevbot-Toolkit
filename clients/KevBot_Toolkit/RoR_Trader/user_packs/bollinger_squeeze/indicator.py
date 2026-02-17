import pandas as pd
import numpy as np

def calculate_bollinger_squeeze(df: pd.DataFrame, **params) -> pd.DataFrame:
    period = params.get("bb_period", 20)
    mult = params.get("bb_mult", 2.0)
    result = df.copy()

    basis = result["close"].rolling(window=period).mean()
    std = result["close"].rolling(window=period).std()

    result["bb_basis"] = basis
    result["bb_upper"] = basis + mult * std
    result["bb_lower"] = basis - mult * std
    result["bb_bandwidth"] = ((result["bb_upper"] - result["bb_lower"]) / basis) * 100

    return result