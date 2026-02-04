"""
Alpaca API Exploration Script
=============================

This script demonstrates how to fetch historical stock data from Alpaca
and shows the data structure we'll work with in RoR Trader.

Setup:
1. Create a free Alpaca account at https://alpaca.markets/
2. Get your API keys from the dashboard
3. Create a .env file in the RoR_Trader folder with:
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
4. Run: python src/explore_alpaca.py

"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API keys
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("=" * 60)
    print("ALPACA API KEYS NOT FOUND")
    print("=" * 60)
    print()
    print("To use this script, you need to:")
    print()
    print("1. Create a free Alpaca account:")
    print("   https://alpaca.markets/")
    print()
    print("2. Get your API keys from the Alpaca dashboard")
    print()
    print("3. Create a .env file in the RoR_Trader folder with:")
    print("   ALPACA_API_KEY=your_key_here")
    print("   ALPACA_SECRET_KEY=your_secret_here")
    print()
    print("=" * 60)
    print()
    print("For now, here's what the data structure looks like:")
    print()

    # Show mock example of what the data looks like
    mock_data = """
    DataFrame structure from Alpaca (multi-index: symbol, timestamp):

                                        open      high       low     close    volume  trade_count      vwap
    symbol timestamp
    SPY    2026-01-02 09:30:00+00:00  476.12    476.55    475.89    476.32    125432        1523    476.21
           2026-01-02 09:31:00+00:00  476.32    476.78    476.15    476.65    98234         1245    476.45
           2026-01-02 09:32:00+00:00  476.65    477.02    476.42    476.89    112543        1389    476.72
           ...
    AAPL   2026-01-02 09:30:00+00:00  185.23    185.67    185.01    185.45    234521        2341    185.34
           ...

    Key fields:
    - open, high, low, close: Price data
    - volume: Number of shares traded
    - trade_count: Number of individual trades
    - vwap: Volume-weighted average price (useful for VWAP interpreter!)
    """
    print(mock_data)
    exit(0)


# If we have keys, import Alpaca and fetch real data
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    import pandas as pd
except ImportError:
    print("Please install required packages:")
    print("pip install alpaca-py pandas")
    exit(1)


def explore_historical_data():
    """Fetch and explore historical bar data."""

    print("=" * 60)
    print("ALPACA API EXPLORATION")
    print("=" * 60)
    print()

    # Initialize client
    print("Initializing Alpaca client...")
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    print("✓ Client initialized")
    print()

    # Define what we want to fetch
    symbols = ["SPY", "AAPL", "QQQ"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)  # Last 5 days

    print(f"Fetching data for: {symbols}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print()

    # Fetch 1-minute bars
    print("Fetching 1-minute bars...")
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    print(f"✓ Received {len(df)} total bars")
    print()

    # Show DataFrame structure
    print("=" * 60)
    print("DATA STRUCTURE")
    print("=" * 60)
    print()
    print("DataFrame Info:")
    print(f"  Index: {df.index.names}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print()

    # Show sample data for one symbol
    print("Sample data for SPY (first 5 bars):")
    print("-" * 60)
    spy_data = df.loc["SPY"].head()
    print(spy_data.to_string())
    print()

    # Show data types
    print("Data types:")
    print("-" * 60)
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print()

    # Show statistics
    print("=" * 60)
    print("DATA STATISTICS (SPY)")
    print("=" * 60)
    spy_df = df.loc["SPY"]
    print()
    print(f"Total bars: {len(spy_df)}")
    print(f"Date range: {spy_df.index.min()} to {spy_df.index.max()}")
    print()
    print("Price statistics:")
    print(f"  Close - Min: ${spy_df['close'].min():.2f}, Max: ${spy_df['close'].max():.2f}")
    print(f"  Average volume: {spy_df['volume'].mean():,.0f}")
    print(f"  Average trade count: {spy_df['trade_count'].mean():,.0f}")
    print()

    # Demonstrate fetching different timeframes
    print("=" * 60)
    print("TESTING DIFFERENT TIMEFRAMES")
    print("=" * 60)
    print()

    timeframes = [
        ("1 Minute", TimeFrame.Minute),
        ("5 Minutes", TimeFrame(5, TimeFrameUnit.Minute)),
        ("15 Minutes", TimeFrame(15, TimeFrameUnit.Minute)),
        ("1 Hour", TimeFrame.Hour),
        ("1 Day", TimeFrame.Day),
    ]

    for name, tf in timeframes:
        request = StockBarsRequest(
            symbol_or_symbols=["SPY"],
            timeframe=tf,
            start=start_date,
            end=end_date
        )
        bars = client.get_stock_bars(request)
        count = len(bars.df)
        print(f"  {name}: {count} bars")

    print()
    print("=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print()
    print("Key takeaways for RoR Trader:")
    print("  1. Data comes as multi-index DataFrame (symbol, timestamp)")
    print("  2. VWAP is included in the response (useful for VWAP interpreter)")
    print("  3. Trade count can be used for additional volume analysis")
    print("  4. Multiple timeframes available from 1-min to monthly")
    print()

    return df


def explore_indicator_calculation(df):
    """Show how we'd calculate indicators from the raw data."""

    print("=" * 60)
    print("INDICATOR CALCULATION EXAMPLE")
    print("=" * 60)
    print()

    # Get SPY data
    spy = df.loc["SPY"].copy()

    # Calculate EMAs
    spy['ema_8'] = spy['close'].ewm(span=8, adjust=False).mean()
    spy['ema_21'] = spy['close'].ewm(span=21, adjust=False).mean()
    spy['ema_50'] = spy['close'].ewm(span=50, adjust=False).mean()

    # EMA Stack interpretation
    def interpret_ema_stack(row):
        price = row['close']
        e8, e21, e50 = row['ema_8'], row['ema_21'], row['ema_50']

        # Check all permutations
        if price > e8 > e21 > e50:
            return "SML"  # Full Bull Stack (Small > Med > Large)
        elif e8 > price > e21 > e50:
            return "SLM"  # Bull below short
        elif price < e8 < e21 < e50:
            return "LMS"  # Full Bear Stack
        else:
            return "OTHER"

    spy['ema_interpretation'] = spy.apply(interpret_ema_stack, axis=1)

    print("EMA Stack Interpretation (last 10 bars):")
    print("-" * 60)
    cols = ['close', 'ema_8', 'ema_21', 'ema_50', 'ema_interpretation']
    print(spy[cols].tail(10).to_string())
    print()

    # Show interpretation distribution
    print("Interpretation Distribution:")
    print(spy['ema_interpretation'].value_counts())
    print()


if __name__ == "__main__":
    df = explore_historical_data()

    if df is not None and len(df) > 0:
        explore_indicator_calculation(df)
