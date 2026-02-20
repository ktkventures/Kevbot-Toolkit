import pandas as pd
import numpy as np


def calculate_supertrend(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate SuperTrend indicator.

    SuperTrend uses ATR to create a trailing stop that ratchets in the
    direction of the trend. When price crosses the stop, trend flips.

    Outputs:
        st_line: The active SuperTrend level (support in uptrend, resistance in downtrend)
        st_direction: 1 for bullish, -1 for bearish
        st_atr: Current ATR value (for proximity calculations)
    """
    atr_period = params.get("atr_period", 10)
    multiplier = params.get("atr_multiplier", 3.0)

    result = df.copy()
    high = result["high"].values
    low = result["low"].values
    close = result["close"].values
    n = len(close)

    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )

    # ATR via Wilder smoothing (alpha = 1/period), matching Pine v4 atr()
    atr = np.zeros(n)
    atr[0] = tr[0]
    alpha = 1.0 / atr_period
    for i in range(1, n):
        atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])

    # Source: hl2
    src = (high + low) / 2.0

    # Basic bands
    basic_up = src - multiplier * atr
    basic_dn = src + multiplier * atr

    # Ratcheted bands and trend
    up = np.zeros(n)
    dn = np.zeros(n)
    trend = np.ones(n, dtype=int)
    st_line = np.zeros(n)

    up[0] = basic_up[0]
    dn[0] = basic_dn[0]

    for i in range(1, n):
        # Ratchet up band (can only rise in uptrend)
        up[i] = max(basic_up[i], up[i - 1]) if close[i - 1] > up[i - 1] else basic_up[i]
        # Ratchet dn band (can only fall in downtrend)
        dn[i] = min(basic_dn[i], dn[i - 1]) if close[i - 1] < dn[i - 1] else basic_dn[i]

        # Trend direction
        if trend[i - 1] == -1 and close[i] > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    # SuperTrend line: up when bull, dn when bear
    for i in range(n):
        st_line[i] = up[i] if trend[i] == 1 else dn[i]

    result["st_line"] = st_line
    result["st_direction"] = trend
    result["st_atr"] = atr

    return result
