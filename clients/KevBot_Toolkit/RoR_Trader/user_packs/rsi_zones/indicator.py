"""
RSI Zones — Indicator Calculation
==================================

Calculates the Relative Strength Index (RSI) and adds it as a column
to the DataFrame.
"""

import pandas as pd
import numpy as np


def calculate_rsi_zones(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate RSI indicator values.

    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        **params: Parameters from the confluence group's parameters dict
            - rsi_period (int): RSI lookback period (default 14)

    Returns:
        DataFrame with 'rsi' column added
    """
    period = params.get("rsi_period", 14)

    result = df.copy()

    delta = result["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result["rsi"] = 100 - (100 / (1 + rs))

    return result
