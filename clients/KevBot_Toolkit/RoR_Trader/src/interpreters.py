"""
Interpreters for RoR Trader
============================

Interpreters analyze indicators and price action to output clear,
mutually exclusive condition states (interpretations).

Each interpreter:
1. Takes raw bar data as input
2. Calculates any necessary indicators
3. Outputs a categorical interpretation for each bar
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class InterpreterConfig:
    """Configuration for an interpreter."""
    name: str
    description: str
    category: str
    outputs: List[str]
    parameters: Dict


# =============================================================================
# INTERPRETER DEFINITIONS
# =============================================================================

INTERPRETERS = {
    "EMA_STACK": InterpreterConfig(
        name="EMA Stack",
        description="Analyzes EMA alignment (8, 21, 50) relative to price",
        category="Moving Averages",
        outputs=["SML", "SLM", "MSL", "MLS", "LSM", "LMS"],
        parameters={"short": 8, "mid": 21, "long": 50}
    ),
    "MACD_SIMPLE": InterpreterConfig(
        name="MACD Simple",
        description="MACD line vs Signal line with momentum direction",
        category="MACD",
        outputs=["M>S↑", "M>S↓", "M<S↓", "M<S↑"],
        parameters={"fast": 12, "slow": 26, "signal": 9}
    ),
    "VWAP": InterpreterConfig(
        name="VWAP Position",
        description="Price position relative to VWAP",
        category="Volume",
        outputs=["ABOVE", "AT", "BELOW"],
        parameters={"tolerance_pct": 0.1}
    ),
    "RVOL": InterpreterConfig(
        name="Relative Volume",
        description="Current volume vs historical average",
        category="Volume",
        outputs=["EXTREME", "HIGH", "NORMAL", "LOW", "MINIMAL"],
        parameters={"lookback": 20}
    ),
}


# =============================================================================
# EMA STACK INTERPRETER
# =============================================================================

def calculate_ema_stack(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate EMA Stack interpretation.

    Outputs:
    - SML: Price > Short > Mid > Long (Full Bull Stack)
    - SLM: Short > Price > Mid > Long
    - MSL: Short > Mid > Price > Long
    - MLS: Short > Mid > Long > Price
    - LSM: Long > Price (various bear configurations)
    - LMS: Price < Short < Mid < Long (Full Bear Stack)
    """
    params = params or INTERPRETERS["EMA_STACK"].parameters
    short, mid, long = params["short"], params["mid"], params["long"]

    result = df.copy()

    # Calculate EMAs
    result['ema_short'] = result['close'].ewm(span=short, adjust=False).mean()
    result['ema_mid'] = result['close'].ewm(span=mid, adjust=False).mean()
    result['ema_long'] = result['close'].ewm(span=long, adjust=False).mean()

    def interpret(row):
        p = row['close']
        s, m, l = row['ema_short'], row['ema_mid'], row['ema_long']

        # Full bull stack: Price > Short > Mid > Long
        if p > s > m > l:
            return "SML"
        # Bull but price below short EMA
        elif s > p > m > l:
            return "SLM"
        # Price between mid and long
        elif s > m > p > l:
            return "MSL"
        # Price below all EMAs but EMAs still bullish order
        elif s > m > l > p:
            return "MLS"
        # Full bear stack: Price < Short < Mid < Long
        elif p < s < m < l:
            return "LMS"
        # Various transitional states
        else:
            return "LSM"

    result['EMA_STACK'] = result.apply(interpret, axis=1)

    return result[['EMA_STACK', 'ema_short', 'ema_mid', 'ema_long']]


# =============================================================================
# MACD SIMPLE INTERPRETER
# =============================================================================

def calculate_macd_simple(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate MACD Simple interpretation.

    Outputs:
    - M>S↑: MACD above signal and rising (strengthening bullish)
    - M>S↓: MACD above signal but falling (weakening bullish)
    - M<S↓: MACD below signal and falling (strengthening bearish)
    - M<S↑: MACD below signal but rising (weakening bearish)
    """
    params = params or INTERPRETERS["MACD_SIMPLE"].parameters
    fast, slow, signal = params["fast"], params["slow"], params["signal"]

    result = df.copy()

    # Calculate MACD
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    result['macd_line'] = ema_fast - ema_slow
    result['macd_signal'] = result['macd_line'].ewm(span=signal, adjust=False).mean()
    result['macd_hist'] = result['macd_line'] - result['macd_signal']

    # Calculate momentum (is histogram growing or shrinking?)
    result['macd_hist_prev'] = result['macd_hist'].shift(1)

    def interpret(row):
        if pd.isna(row['macd_hist_prev']):
            return None

        above_signal = row['macd_line'] > row['macd_signal']
        hist_rising = row['macd_hist'] > row['macd_hist_prev']

        if above_signal and hist_rising:
            return "M>S↑"
        elif above_signal and not hist_rising:
            return "M>S↓"
        elif not above_signal and not hist_rising:
            return "M<S↓"
        else:  # below signal, rising
            return "M<S↑"

    result['MACD_SIMPLE'] = result.apply(interpret, axis=1)

    return result[['MACD_SIMPLE', 'macd_line', 'macd_signal', 'macd_hist']]


# =============================================================================
# VWAP INTERPRETER
# =============================================================================

def calculate_vwap(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate VWAP Position interpretation.

    Uses the VWAP provided in Alpaca data (or calculates if not present).

    Outputs:
    - ABOVE: Price significantly above VWAP
    - AT: Price near VWAP (within tolerance)
    - BELOW: Price significantly below VWAP
    """
    params = params or INTERPRETERS["VWAP"].parameters
    tolerance = params["tolerance_pct"] / 100

    result = df.copy()

    # Use provided VWAP or calculate
    if 'vwap' not in result.columns:
        # Simple VWAP calculation (cumulative for the day)
        result['vwap'] = (result['close'] * result['volume']).cumsum() / result['volume'].cumsum()

    def interpret(row):
        if pd.isna(row['vwap']) or row['vwap'] == 0:
            return None

        pct_diff = (row['close'] - row['vwap']) / row['vwap']

        if pct_diff > tolerance:
            return "ABOVE"
        elif pct_diff < -tolerance:
            return "BELOW"
        else:
            return "AT"

    result['VWAP'] = result.apply(interpret, axis=1)

    return result[['VWAP', 'vwap']]


# =============================================================================
# RVOL INTERPRETER
# =============================================================================

def calculate_rvol(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate Relative Volume interpretation.

    Compares current volume to rolling average.

    Outputs:
    - EXTREME: > 200% of average
    - HIGH: > 150% of average
    - NORMAL: 75-150% of average
    - LOW: 50-75% of average
    - MINIMAL: < 50% of average
    """
    params = params or INTERPRETERS["RVOL"].parameters
    lookback = params["lookback"]

    result = df.copy()

    # Calculate rolling average volume
    result['vol_avg'] = result['volume'].rolling(window=lookback, min_periods=5).mean()
    result['rvol_ratio'] = result['volume'] / result['vol_avg']

    def interpret(row):
        if pd.isna(row['rvol_ratio']):
            return None

        ratio = row['rvol_ratio']

        if ratio > 2.0:
            return "EXTREME"
        elif ratio > 1.5:
            return "HIGH"
        elif ratio > 0.75:
            return "NORMAL"
        elif ratio > 0.5:
            return "LOW"
        else:
            return "MINIMAL"

    result['RVOL'] = result.apply(interpret, axis=1)

    return result[['RVOL', 'rvol_ratio', 'vol_avg']]


# =============================================================================
# MAIN INTERPRETER ENGINE
# =============================================================================

def run_all_interpreters(
    df: pd.DataFrame,
    enabled_interpreters: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run all enabled interpreters on the data.

    Args:
        df: DataFrame with OHLCV data (single symbol, timestamp index)
        enabled_interpreters: List of interpreter keys to run (default: all)

    Returns:
        DataFrame with original data plus interpretation columns
    """
    if enabled_interpreters is None:
        enabled_interpreters = list(INTERPRETERS.keys())

    result = df.copy()

    interpreter_funcs = {
        "EMA_STACK": calculate_ema_stack,
        "MACD_SIMPLE": calculate_macd_simple,
        "VWAP": calculate_vwap,
        "RVOL": calculate_rvol,
    }

    for interp_key in enabled_interpreters:
        if interp_key in interpreter_funcs:
            interp_result = interpreter_funcs[interp_key](df)
            # Merge interpretation column (the one matching the key name)
            result[interp_key] = interp_result[interp_key]

    return result


def get_confluence_records(row: pd.Series, timeframe: str, interpreters: List[str]) -> set:
    """
    Get all confluence records for a single bar.

    Returns a set of strings like {"1M-EMA_STACK-SML", "1M-MACD_SIMPLE-M>S↑"}
    """
    records = set()

    for interp in interpreters:
        if interp in row.index and pd.notna(row[interp]):
            record = f"{timeframe}-{interp}-{row[interp]}"
            records.add(record)

    return records


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from mock_data import generate_mock_bars
    from datetime import datetime, timedelta

    # Generate test data
    end = datetime.now().replace(hour=16, minute=0)
    start = end - timedelta(days=5)

    bars = generate_mock_bars(["SPY"], start, end, "1Min")
    spy_bars = bars.loc["SPY"]

    print("Running interpreters on SPY data...")
    print(f"Input shape: {spy_bars.shape}")
    print()

    # Run all interpreters
    result = run_all_interpreters(spy_bars)

    print("Output columns:", result.columns.tolist())
    print()

    print("Sample output (last 10 bars):")
    print(result[['close', 'EMA_STACK', 'MACD_SIMPLE', 'VWAP', 'RVOL']].tail(10))
    print()

    # Show interpretation distributions
    for interp in ['EMA_STACK', 'MACD_SIMPLE', 'VWAP', 'RVOL']:
        print(f"\n{interp} distribution:")
        print(result[interp].value_counts())
