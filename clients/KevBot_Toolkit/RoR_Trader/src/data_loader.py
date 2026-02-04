"""
Data Loader for RoR Trader
===========================

Handles loading market data from:
1. Alpaca API (if credentials configured)
2. Mock data (fallback for development)

Usage:
    from data_loader import load_market_data, is_alpaca_configured

    df = load_market_data("SPY", days=30)
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for Alpaca credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


def is_alpaca_configured() -> bool:
    """Check if Alpaca API credentials are configured."""
    return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY and
                not ALPACA_API_KEY.startswith("your_"))


def load_from_alpaca(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min"
) -> Optional[pd.DataFrame]:
    """
    Load historical bar data from Alpaca API.

    Args:
        symbol: Stock symbol (e.g., "SPY")
        days: Number of days of history to fetch
        timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")

    Returns:
        DataFrame with OHLCV data, or None if fetch fails
    """
    if not is_alpaca_configured():
        return None

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        # Initialize client
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        # Parse timeframe
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Minute)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Fetch bars
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start_date,
            end=end_date
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if len(df) == 0:
            return None

        # Extract single symbol (remove multi-index)
        if symbol in df.index.get_level_values(0):
            df = df.loc[symbol]

        return df

    except Exception as e:
        print(f"Alpaca fetch failed: {e}")
        return None


def load_from_mock(
    symbol: str,
    days: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load mock/simulated bar data.

    Args:
        symbol: Stock symbol (used for seed variation)
        days: Number of days of history
        seed: Random seed for reproducibility

    Returns:
        DataFrame with simulated OHLCV data
    """
    from mock_data import generate_mock_bars

    end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    # Vary seed by symbol for different price patterns
    symbol_seed = seed + hash(symbol) % 1000

    bars = generate_mock_bars([symbol], start, end, "1Min", seed=symbol_seed)

    if symbol in bars.index.get_level_values(0):
        return bars.loc[symbol]

    return pd.DataFrame()


def load_market_data(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min",
    seed: int = 42,
    force_mock: bool = False
) -> pd.DataFrame:
    """
    Load market data, preferring Alpaca if configured.

    Args:
        symbol: Stock symbol (e.g., "SPY")
        days: Number of days of history
        timeframe: Bar timeframe (only affects Alpaca, mock always uses 1Min)
        seed: Random seed for mock data
        force_mock: If True, skip Alpaca and use mock data

    Returns:
        DataFrame with OHLCV data (columns: open, high, low, close, volume, trade_count, vwap)
    """
    # Try Alpaca first (unless forced to use mock)
    if not force_mock and is_alpaca_configured():
        df = load_from_alpaca(symbol, days, timeframe)
        if df is not None and len(df) > 0:
            return df

    # Fall back to mock data
    return load_from_mock(symbol, days, seed)


def get_data_source() -> str:
    """Get the current data source being used."""
    if is_alpaca_configured():
        return "Alpaca API"
    return "Mock Data"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print(f"Data source: {get_data_source()}")
    print(f"Alpaca configured: {is_alpaca_configured()}")
    print()

    # Test loading data
    print("Loading SPY data...")
    df = load_market_data("SPY", days=5)
    print(f"Loaded {len(df)} bars")
    print()

    if len(df) > 0:
        print("Sample data (last 5 bars):")
        print(df.tail())
        print()
        print(f"Columns: {list(df.columns)}")
