import pandas as pd
import numpy as np


def calculate_swing_123_test(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate Swing 1-2-3 candle pattern boolean flags and candle colors.

    C2 Bullish: low < low[lookback] AND close > close[lookback]
                AND (if min_c2_wick_pct > 0) lower wick below prior close
                divided by prior close >= min_c2_wick_pct
    C2 Bearish: high > high[lookback] AND close < close[lookback]
                AND (if min_c2_wick_pct > 0) upper wick above prior close
                divided by prior close >= min_c2_wick_pct
    C3 Bullish: prior bar was bull_c2 AND close > high[1]
    C3 Bearish: prior bar was bear_c2 AND close < low[1]

    Entry levels (for intra-bar triggers):
      sw123t_bull_entry_level_prev: current candle's close (every bar; engine adds 1-bar lag automatically)
      sw123t_bear_entry_level_prev: current candle's close (every bar; engine adds 1-bar lag automatically)

    Args:
        df: DataFrame with OHLCV columns
        **params: min_c2_wick_pct, lookback_bars

    Returns:
        Copy of DataFrame with sw123t_bull_c2, sw123t_bear_c2,
        sw123t_bull_c3, sw123t_bear_c3, sw123t_candle_color,
        sw123t_bull_entry_level_prev, sw123t_bear_entry_level_prev columns added.
    """
    min_c2_wick_pct = params.get("min_c2_wick_pct", 0.0)
    lookback = params.get("lookback_bars", 1)

    result = df.copy()

    close = result["close"]
    high = result["high"]
    low = result["low"]

    # Shifted reference candle (C1) at lookback distance
    close_lb = close.shift(lookback)
    high_lb = high.shift(lookback)
    low_lb = low.shift(lookback)

    # --- Bullish C2 ---
    # Lower low AND reclaimed prior close
    bull_c2_base = (low < low_lb) & (close > close_lb)

    # Optional wick filter: the portion of the wick that dips below prior close
    # relative to prior close must be >= min_c2_wick_pct
    if min_c2_wick_pct > 0:
        # How far the low poked below the prior close, as fraction of prior close
        bull_wick_ext = (close_lb - low) / close_lb.replace(0, np.nan)
        bull_c2_base = bull_c2_base & (bull_wick_ext >= min_c2_wick_pct)

    # --- Bearish C2 ---
    # Higher high AND dropped below prior close
    bear_c2_base = (high > high_lb) & (close < close_lb)

    # Optional wick filter: portion of wick above prior close relative to prior close
    if min_c2_wick_pct > 0:
        bear_wick_ext = (high - close_lb) / close_lb.replace(0, np.nan)
        bear_c2_base = bear_c2_base & (bear_wick_ext >= min_c2_wick_pct)

    # Store raw booleans (fill NaN as False)
    result["sw123t_bull_c2"] = bull_c2_base.fillna(False)
    result["sw123t_bear_c2"] = bear_c2_base.fillna(False)

    # --- Bullish C3 ---
    # Previous bar was bull_c2 AND current close > previous high
    bull_c2_prev = result["sw123t_bull_c2"].shift(1).fillna(False)
    high_prev = high.shift(1)
    result["sw123t_bull_c3"] = (bull_c2_prev & (close > high_prev)).fillna(False)

    # --- Bearish C3 ---
    # Previous bar was bear_c2 AND current close < previous low
    bear_c2_prev = result["sw123t_bear_c2"].shift(1).fillna(False)
    low_prev = low.shift(1)
    result["sw123t_bear_c3"] = (bear_c2_prev & (close < low_prev)).fillna(False)

    # --- Entry level columns (for intra-bar triggers) ---
    # Use current bar's close on EVERY bar — the engine automatically adds a 1-bar lag
    # (caches value on C2 bar, uses it on the next bar for intra-bar cross detection).
    # Do NOT use .shift(1) here as that would cause a 2-bar lag.
    # Do NOT filter with .where() — the level must be valid on every bar.
    result["sw123t_bull_entry_level_prev"] = close
    result["sw123t_bear_entry_level_prev"] = close

    # --- Candle color column ---
    # Priority: C3 > C2 > no color (empty string)
    candle_color = pd.Series("", index=result.index, dtype=str)

    # Apply in reverse priority so higher-priority overwrites
    candle_color = candle_color.where(~result["sw123t_bull_c2"], "#FFD11A")  # Bullish C2: med yellow
    candle_color = candle_color.where(~result["sw123t_bear_c2"], "#FF66B3")  # Bearish C2: med pink
    candle_color = candle_color.where(~result["sw123t_bull_c3"], "#FFFF00")  # Bullish C3: neon yellow (overrides C2)
    candle_color = candle_color.where(~result["sw123t_bear_c3"], "#FF33CC")  # Bearish C3: neon pink (overrides C2)

    result["sw123t_candle_color"] = candle_color

    return result