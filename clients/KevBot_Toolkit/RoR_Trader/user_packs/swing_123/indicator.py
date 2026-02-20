import pandas as pd
import numpy as np


def calculate_swing_123(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Detect Swing 1-2-3 candle patterns.

    Candle 2 (C2): Reversal candidate
        Bull C2: Makes lower low than prior bar AND closes above prior close
        Bear C2: Makes higher high than prior bar AND closes below prior close

    Candle 3 (C3): Continuation confirmation
        Bull C3: Prior bar was Bull C2 AND current close > prior high
        Bear C3: Prior bar was Bear C2 AND current close < prior low

    Outputs:
        sw123_pattern: Integer code (0=neutral, 1=bull_c2, 2=bull_c3, -1=bear_c2, -2=bear_c3)
        sw123_candle_color: Hex color string for bar coloring (or empty for default)
    """
    result = df.copy()
    close = result["close"].values
    high = result["high"].values
    low = result["low"].values
    n = len(close)

    pattern = np.zeros(n, dtype=int)
    colors = [""] * n

    # Default colors
    bull_c2_color = "#FFD11A"
    bull_c3_color = "#FFFF00"
    bear_c2_color = "#FF66B3"
    bear_c3_color = "#FF33CC"

    for i in range(1, n):
        # Candle 2 conditions
        bull_c2 = low[i] < low[i - 1] and close[i] > close[i - 1]
        bear_c2 = high[i] > high[i - 1] and close[i] < close[i - 1]

        # Candle 3 conditions (check if prior bar was C2)
        bull_c3 = (pattern[i - 1] == 1) and close[i] > high[i - 1]
        bear_c3 = (pattern[i - 1] == -1) and close[i] < low[i - 1]

        # Priority: C3 > C2 (matching Pine Script)
        if bull_c3:
            pattern[i] = 2
            colors[i] = bull_c3_color
        elif bear_c3:
            pattern[i] = -2
            colors[i] = bear_c3_color
        elif bull_c2:
            pattern[i] = 1
            colors[i] = bull_c2_color
        elif bear_c2:
            pattern[i] = -1
            colors[i] = bear_c2_color

    result["sw123_pattern"] = pattern
    result["sw123_candle_color"] = colors

    return result
