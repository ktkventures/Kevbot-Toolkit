"""
Indicators for RoR Trader
==========================

Indicators calculate numeric values from price/volume data.
They are separate from interpreters which classify states.

Each indicator:
1. Takes raw bar data as input
2. Calculates technical indicator values
3. Adds columns to the DataFrame
4. Provides chart overlay configuration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IndicatorConfig:
    """Configuration for an indicator."""
    id: str                    # e.g., "ema_8"
    name: str                  # e.g., "EMA 8"
    category: str              # e.g., "Moving Averages"
    parameters: Dict           # e.g., {"period": 8}
    overlay_type: str          # "line", "area", "histogram", "none"
    color: str                 # Chart color
    columns: List[str]         # Columns this indicator adds


# =============================================================================
# INDICATOR COLOR PALETTE
# =============================================================================

INDICATOR_COLORS = {
    "ema_8": "#FF6B6B",         # Red - short term
    "ema_21": "#4ECDC4",        # Teal - medium term
    "ema_50": "#95E1D3",        # Light teal - long term
    "macd_line": "#FFD93D",     # Yellow
    "macd_signal": "#6BCB77",   # Green
    "macd_hist": "#4D96FF",     # Blue
    "vwap": "#A8E6CF",          # Light green
    "vwap_upper": "#A8E6CF",    # Light green (lighter)
    "vwap_lower": "#A8E6CF",    # Light green (lighter)
    "atr": "#FFB6B9",           # Pink
    "vol_sma": "#DDA0DD",       # Plum
}


# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

INDICATORS: Dict[str, IndicatorConfig] = {
    "ema_8": IndicatorConfig(
        id="ema_8",
        name="EMA 8",
        category="Moving Averages",
        parameters={"period": 8},
        overlay_type="line",
        color=INDICATOR_COLORS["ema_8"],
        columns=["ema_8"]
    ),
    "ema_21": IndicatorConfig(
        id="ema_21",
        name="EMA 21",
        category="Moving Averages",
        parameters={"period": 21},
        overlay_type="line",
        color=INDICATOR_COLORS["ema_21"],
        columns=["ema_21"]
    ),
    "ema_50": IndicatorConfig(
        id="ema_50",
        name="EMA 50",
        category="Moving Averages",
        parameters={"period": 50},
        overlay_type="line",
        color=INDICATOR_COLORS["ema_50"],
        columns=["ema_50"]
    ),
    "macd": IndicatorConfig(
        id="macd",
        name="MACD",
        category="Momentum",
        parameters={"fast": 12, "slow": 26, "signal": 9},
        overlay_type="none",  # MACD typically shown in separate pane
        color=INDICATOR_COLORS["macd_line"],
        columns=["macd_line", "macd_signal", "macd_hist"]
    ),
    "vwap": IndicatorConfig(
        id="vwap",
        name="VWAP",
        category="Volume",
        parameters={"std_dev": 2.0},
        overlay_type="line",
        color=INDICATOR_COLORS["vwap"],
        columns=["vwap", "vwap_upper", "vwap_lower"]
    ),
    "atr": IndicatorConfig(
        id="atr",
        name="ATR",
        category="Volatility",
        parameters={"period": 14},
        overlay_type="none",  # ATR not shown on price chart
        color=INDICATOR_COLORS["atr"],
        columns=["atr"]
    ),
    "vol_sma": IndicatorConfig(
        id="vol_sma",
        name="Volume SMA",
        category="Volume",
        parameters={"period": 20},
        overlay_type="none",  # Volume shown separately
        color=INDICATOR_COLORS["vol_sma"],
        columns=["vol_sma", "rvol"]
    ),
}


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Calculate Exponential Moving Average."""
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Calculate Simple Moving Average."""
    return df[column].rolling(window=period, min_periods=1).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is EMA of True Range
    return tr.ewm(span=period, adjust=False).mean()


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Dict[str, pd.Series]:
    """
    Calculate MACD indicator.

    Returns dict with:
    - macd_line: MACD line (fast EMA - slow EMA)
    - macd_signal: Signal line (EMA of MACD line)
    - macd_hist: Histogram (MACD line - Signal line)
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return {
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist
    }


def calculate_vwap(df: pd.DataFrame, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate VWAP with standard deviation bands.

    If VWAP already exists in df, use it. Otherwise calculate.

    Returns dict with:
    - vwap: Volume Weighted Average Price
    - vwap_upper: Upper band (VWAP + std_dev * rolling std)
    - vwap_lower: Lower band (VWAP - std_dev * rolling std)
    """
    if 'vwap' in df.columns and df['vwap'].notna().any():
        vwap = df['vwap']
    else:
        # Calculate VWAP (cumulative for the session)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    # Calculate bands using rolling standard deviation
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    rolling_std = typical_price.rolling(window=20, min_periods=5).std()

    return {
        "vwap": vwap,
        "vwap_upper": vwap + (std_dev * rolling_std),
        "vwap_lower": vwap - (std_dev * rolling_std)
    }


def calculate_volume_sma(df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate Volume SMA and Relative Volume.

    Returns dict with:
    - vol_sma: Simple moving average of volume
    - rvol: Relative volume (current volume / SMA)
    """
    vol_sma = df['volume'].rolling(window=period, min_periods=5).mean()
    rvol = df['volume'] / vol_sma

    return {
        "vol_sma": vol_sma,
        "rvol": rvol
    }


# =============================================================================
# MAIN INDICATOR ENGINE
# =============================================================================

def run_all_indicators(
    df: pd.DataFrame,
    enabled_indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run all enabled indicators on the data.

    Args:
        df: DataFrame with OHLCV data (single symbol, timestamp index)
        enabled_indicators: List of indicator IDs to run (default: all)

    Returns:
        DataFrame with original data plus indicator columns
    """
    if enabled_indicators is None:
        enabled_indicators = list(INDICATORS.keys())

    result = df.copy()

    for ind_id in enabled_indicators:
        if ind_id not in INDICATORS:
            continue

        config = INDICATORS[ind_id]

        # EMA indicators
        if ind_id.startswith("ema_"):
            period = config.parameters["period"]
            result[ind_id] = calculate_ema(df, period)

        # MACD
        elif ind_id == "macd":
            macd_result = calculate_macd(
                df,
                config.parameters["fast"],
                config.parameters["slow"],
                config.parameters["signal"]
            )
            for col, values in macd_result.items():
                result[col] = values

        # VWAP
        elif ind_id == "vwap":
            vwap_result = calculate_vwap(df, config.parameters["std_dev"])
            for col, values in vwap_result.items():
                result[col] = values

        # ATR
        elif ind_id == "atr":
            result["atr"] = calculate_atr(df, config.parameters["period"])

        # Volume SMA
        elif ind_id == "vol_sma":
            vol_result = calculate_volume_sma(df, config.parameters["period"])
            for col, values in vol_result.items():
                result[col] = values

    return result


def get_indicator_overlay_config(
    df: pd.DataFrame,
    indicator_id: str
) -> Optional[Dict]:
    """
    Get chart overlay configuration for an indicator.

    Returns a dict compatible with lightweight-charts series config,
    or None if the indicator doesn't have a visible overlay.
    """
    if indicator_id not in INDICATORS:
        return None

    config = INDICATORS[indicator_id]

    if config.overlay_type == "none":
        return None

    if indicator_id not in df.columns:
        return None

    # Prepare data for lightweight-charts
    data = []
    for idx, row in df.iterrows():
        if pd.notna(row.get(indicator_id)):
            timestamp = int(pd.to_datetime(idx).timestamp())
            data.append({
                "time": timestamp,
                "value": float(row[indicator_id])
            })

    if not data:
        return None

    return {
        "type": "Line",
        "data": data,
        "options": {
            "color": config.color,
            "lineWidth": 2,
            "priceLineVisible": False,
            "crosshairMarkerVisible": True,
            "title": config.name
        }
    }


def get_available_overlay_indicators() -> List[str]:
    """Get list of indicator IDs that can be shown on the price chart."""
    return [
        ind_id for ind_id, config in INDICATORS.items()
        if config.overlay_type != "none"
    ]


def run_indicators_for_group(df: pd.DataFrame, group) -> pd.DataFrame:
    """
    Run indicators for a specific confluence group using its parameters.

    Ensures that custom-parameterized groups (e.g., EMA stack with non-default
    periods) have their indicator columns present in the DataFrame.

    Args:
        df: DataFrame with OHLCV data (may already have standard indicators)
        group: ConfluenceGroup instance with parameters

    Returns:
        DataFrame with additional indicator columns for this group's parameters
    """
    result = df

    if group.base_template == "ema_stack":
        for period_key in ["short_period", "mid_period", "long_period"]:
            period = group.parameters.get(period_key)
            if period:
                col_name = f"ema_{period}"
                if col_name not in result.columns:
                    result = result.copy() if result is df else result
                    result[col_name] = calculate_ema(result, period)

    elif group.base_template == "macd":
        if "macd_line" not in result.columns:
            fast = group.parameters.get("fast_period", 12)
            slow = group.parameters.get("slow_period", 26)
            signal = group.parameters.get("signal_period", 9)
            result = result.copy() if result is df else result
            macd_result = calculate_macd(result, fast, slow, signal)
            for col, values in macd_result.items():
                result[col] = values

    elif group.base_template == "vwap":
        if "vwap" not in result.columns:
            std_dev = group.parameters.get("std_dev", 2.0)
            result = result.copy() if result is df else result
            vwap_result = calculate_vwap(result, std_dev)
            for col, values in vwap_result.items():
                result[col] = values

    elif group.base_template == "rvol":
        if "vol_sma" not in result.columns:
            period = group.parameters.get("sma_period", 20)
            result = result.copy() if result is df else result
            vol_result = calculate_volume_sma(result, period)
            for col, values in vol_result.items():
                result[col] = values

    return result


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

    print("Running indicators on SPY data...")
    print(f"Input shape: {spy_bars.shape}")
    print()

    # Run all indicators
    result = run_all_indicators(spy_bars)

    print("Output columns:", result.columns.tolist())
    print()

    print("Sample output (last 5 bars):")
    indicator_cols = ['close', 'ema_8', 'ema_21', 'ema_50', 'macd_line', 'vwap', 'rvol']
    available_cols = [c for c in indicator_cols if c in result.columns]
    print(result[available_cols].tail(5))
