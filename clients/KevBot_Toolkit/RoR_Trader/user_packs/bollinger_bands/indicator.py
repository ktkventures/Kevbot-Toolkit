import pandas as pd
import numpy as np

def calculate_bollinger_bands(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate Bollinger Bands: upper, basis (SMA), lower, and bandwidth.

    Args:
        df: DataFrame with OHLCV columns
        **params: bb_period, bb_mult

    Returns:
        Copy of DataFrame with bb_upper, bb_basis, bb_lower, bb_bandwidth columns.
    """
    period = params.get("bb_period", 20)
    mult = params.get("bb_mult", 2.0)

    result = df.copy()

    basis = result["close"].rolling(window=period).mean()
    std = result["close"].rolling(window=period).std()

    result["bb_basis"] = basis
    result["bb_upper"] = basis + mult * std
    result["bb_lower"] = basis - mult * std
    # Bandwidth as percentage of basis
    result["bb_bandwidth"] = np.where(basis > 0, (result["bb_upper"] - result["bb_lower"]) / basis, np.nan)

    return result