"""
Mock Data Generator for RoR Trader
===================================

Generates mock market data that matches Alpaca's data structure.
This allows development and testing without an API connection.

When Alpaca is connected, this module can be swapped out for real data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional


def generate_mock_bars(
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: str = "1Min",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate mock OHLCV data matching Alpaca's response structure.

    Returns a multi-index DataFrame (symbol, timestamp) with columns:
    - open, high, low, close, volume, trade_count, vwap

    This matches the structure returned by:
    alpaca.data.historical.StockHistoricalDataClient.get_stock_bars()
    """
    np.random.seed(seed)

    # Parse timeframe to minutes
    tf_minutes = _parse_timeframe(timeframe)

    # Generate timestamps based on market hours (9:30 AM - 4:00 PM ET)
    timestamps = _generate_market_timestamps(start, end, tf_minutes)

    if len(timestamps) == 0:
        return pd.DataFrame()

    all_data = []

    for symbol in symbols:
        # Each symbol gets slightly different characteristics
        symbol_seed = seed + hash(symbol) % 1000
        np.random.seed(symbol_seed)

        # Base price varies by symbol
        base_prices = {
            "SPY": 480.0,
            "AAPL": 185.0,
            "QQQ": 420.0,
            "TSLA": 250.0,
            "NVDA": 720.0,
            "MSFT": 410.0,
            "AMD": 160.0,
            "META": 480.0,
        }
        base_price = base_prices.get(symbol, 100.0 + hash(symbol) % 400)

        # Generate price series with realistic movement
        n_bars = len(timestamps)

        # Random walk with drift and volatility
        volatility = 0.0008  # Per-bar volatility
        drift = 0.00002  # Slight upward drift

        returns = np.random.normal(drift, volatility, n_bars)
        prices = base_price * np.cumprod(1 + returns)

        # Generate OHLC from close prices
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            # Intrabar volatility
            intrabar_range = close * np.random.uniform(0.0005, 0.002)

            high = close + np.random.uniform(0, intrabar_range)
            low = close - np.random.uniform(0, intrabar_range)

            # Open is close of previous bar (with small gap)
            if i == 0:
                open_price = close * (1 + np.random.uniform(-0.001, 0.001))
            else:
                open_price = data[-1]['close'] * (1 + np.random.uniform(-0.0002, 0.0002))

            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Volume varies throughout the day (higher at open/close)
            hour = ts.hour
            if hour == 9 or hour == 15:  # Open and close hours
                vol_mult = np.random.uniform(1.5, 2.5)
            elif hour == 12:  # Lunch
                vol_mult = np.random.uniform(0.5, 0.8)
            else:
                vol_mult = np.random.uniform(0.8, 1.2)

            base_volume = 50000 if symbol in ["SPY", "QQQ"] else 20000
            volume = int(base_volume * vol_mult * np.random.uniform(0.5, 1.5))

            # Trade count correlates with volume
            trade_count = int(volume / np.random.uniform(30, 80))

            # VWAP is close to the midpoint, slightly weighted
            vwap = (high + low + close + close) / 4

            data.append({
                'symbol': symbol,
                'timestamp': ts,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
                'trade_count': trade_count,
                'vwap': round(vwap, 2)
            })

        all_data.extend(data)

    # Create DataFrame with multi-index like Alpaca returns
    df = pd.DataFrame(all_data)
    df = df.set_index(['symbol', 'timestamp'])
    df = df.sort_index()

    return df


def _parse_timeframe(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    tf_map = {
        "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
        "1Hour": 60, "4Hour": 240,
        "1Day": 390,  # Full trading day
    }
    return tf_map.get(timeframe, 1)


def _generate_market_timestamps(
    start: datetime,
    end: datetime,
    tf_minutes: int
) -> List[datetime]:
    """Generate timestamps during market hours only."""
    timestamps = []

    current = start.replace(hour=9, minute=30, second=0, microsecond=0)

    while current <= end:
        # Skip weekends
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            # Market hours: 9:30 AM - 4:00 PM
            market_open = current.replace(hour=9, minute=30)
            market_close = current.replace(hour=16, minute=0)

            bar_time = market_open
            while bar_time < market_close:
                if bar_time >= start and bar_time <= end:
                    timestamps.append(bar_time)
                bar_time += timedelta(minutes=tf_minutes)

        # Move to next day
        current += timedelta(days=1)
        current = current.replace(hour=9, minute=30)

    return timestamps


def resample_bars(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample bars to a higher timeframe.

    E.g., convert 1-minute bars to 5-minute bars.
    """
    tf_map = {
        "1Min": "1min", "5Min": "5min", "15Min": "15min", "30Min": "30min",
        "1Hour": "1h", "4Hour": "4h", "1Day": "1D",
    }

    resample_rule = tf_map.get(target_timeframe, "1min")

    resampled_data = []

    for symbol in df.index.get_level_values('symbol').unique():
        symbol_df = df.loc[symbol].copy()

        resampled = symbol_df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'trade_count': 'sum',
            'vwap': 'mean'
        }).dropna()

        resampled['symbol'] = symbol
        resampled = resampled.reset_index()
        resampled_data.append(resampled)

    result = pd.concat(resampled_data, ignore_index=True)
    result = result.set_index(['symbol', 'timestamp'])

    return result


# Quick test
if __name__ == "__main__":
    # Generate 5 days of 1-minute data for SPY
    end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=5)

    df = generate_mock_bars(["SPY", "AAPL"], start, end, "1Min")

    print("Generated mock data:")
    print(f"Shape: {df.shape}")
    print(f"Symbols: {df.index.get_level_values('symbol').unique().tolist()}")
    print()
    print("Sample (SPY, first 5 bars):")
    print(df.loc["SPY"].head())
    print()
    print("Sample (SPY, last 5 bars):")
    print(df.loc["SPY"].tail())
