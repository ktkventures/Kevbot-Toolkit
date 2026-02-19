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


def _filter_session(df: pd.DataFrame, session: str = "RTH") -> pd.DataFrame:
    """Filter DataFrame to the specified trading session window.

    Args:
        df: OHLCV DataFrame with DatetimeIndex (tz-aware or naive/UTC).
        session: One of TRADING_SESSIONS. "RTH" is the default.

    Returns filtered DataFrame. If session is unrecognized, returns df unchanged.
    """
    if len(df) == 0:
        return df

    window = SESSION_HOURS.get(session)
    if window is None:
        return df

    sh, sm, eh, em = window

    try:
        import pytz
        import datetime as dt_mod
        et = pytz.timezone("America/New_York")
        idx = df.index
        if idx.tz is not None:
            et_times = idx.tz_convert(et)
        else:
            et_times = idx.tz_localize("UTC").tz_convert(et)

        time_only = et_times.time
        session_start = dt_mod.time(sh, sm)
        session_end = dt_mod.time(eh, em)
        mask = [(t >= session_start and t < session_end) for t in time_only]
        return df.loc[mask]
    except Exception:
        return df


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy alias — filters to Regular Trading Hours."""
    return _filter_session(df, "RTH")


def load_from_alpaca(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    feed: str = "sip",
    session: str = "RTH",
) -> Optional[pd.DataFrame]:
    """
    Load historical bar data from Alpaca API.

    Args:
        symbol: Stock symbol (e.g., "SPY")
        days: Number of days of history to fetch (used if start_date/end_date not provided)
        timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)
        feed: Data feed — "sip" (consolidated, all exchanges) or "iex" (IEX only)
        session: Trading session filter — "RTH", "Pre-Market", "After Hours", "Extended Hours"

    Returns:
        DataFrame with OHLCV data, or None if fetch fails
    """
    if not is_alpaca_configured():
        return None

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import DataFeed
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        # Initialize client
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        # Parse timeframe
        tf_map = {
            "1Min": TimeFrame.Minute,
            "2Min": TimeFrame(2, TimeFrameUnit.Minute),
            "3Min": TimeFrame(3, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "10Min": TimeFrame(10, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "2Hour": TimeFrame(2, TimeFrameUnit.Hour),
            "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
            "1Day": TimeFrame.Day,
            "1Week": TimeFrame.Week,
            "1Month": TimeFrame.Month,
        }
        tf = tf_map.get(timeframe, TimeFrame.Minute)

        # Calculate date range (must be timezone-aware UTC for Alpaca)
        if start_date is None or end_date is None:
            from datetime import timezone
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

        # Select data feed
        data_feed = DataFeed.SIP if feed == "sip" else DataFeed.IEX

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start_date,
            end=end_date,
            feed=data_feed,
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if len(df) == 0:
            return None

        # Extract single symbol (remove multi-index)
        if symbol in df.index.get_level_values(0):
            df = df.loc[symbol]

        # Filter to the requested trading session.
        # Alpaca returns extended-hours bars by default; session filtering
        # keeps only bars within the strategy's active session window.
        df = _filter_session(df, session)

        return df if len(df) > 0 else None

    except Exception as e:
        print(f"Alpaca fetch failed: {e}")
        return None


def load_from_mock(
    symbol: str,
    days: int = 30,
    seed: int = 42,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    timeframe: str = "1Min",
) -> pd.DataFrame:
    """
    Load mock/simulated bar data.

    Args:
        symbol: Stock symbol (used for seed variation)
        days: Number of days of history (used if start_date/end_date not provided)
        seed: Random seed for reproducibility
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)

    Returns:
        DataFrame with simulated OHLCV data
    """
    from mock_data import generate_mock_bars

    if start_date is not None and end_date is not None:
        start = start_date
        end = end_date
    else:
        end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=days)

    # Vary seed by symbol for different price patterns
    symbol_seed = seed + hash(symbol) % 1000

    bars = generate_mock_bars([symbol], start, end, timeframe, seed=symbol_seed)

    if symbol in bars.index.get_level_values(0):
        return bars.loc[symbol]

    return pd.DataFrame()


# Tracks what source was actually used on the last load_market_data call
_last_actual_source: str = "Unknown"


def load_market_data(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min",
    seed: int = 42,
    force_mock: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    feed: str = "sip",
    session: str = "RTH",
) -> pd.DataFrame:
    """
    Load market data, preferring Alpaca if configured.

    Args:
        symbol: Stock symbol (e.g., "SPY")
        days: Number of days of history (used if start_date/end_date not provided)
        timeframe: Bar timeframe (only affects Alpaca, mock always uses 1Min)
        seed: Random seed for mock data
        force_mock: If True, skip Alpaca and use mock data
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)
        feed: Data feed — "sip" (consolidated) or "iex" (IEX only)
        session: Trading session filter — "RTH", "Pre-Market", "After Hours", "Extended Hours"

    Returns:
        DataFrame with OHLCV data (columns: open, high, low, close, volume, trade_count, vwap)
    """
    global _last_actual_source

    # Try Alpaca first (unless forced to use mock)
    if not force_mock and is_alpaca_configured():
        df = load_from_alpaca(symbol, days, timeframe, start_date=start_date,
                              end_date=end_date, feed=feed, session=session)
        if df is not None and len(df) > 0:
            feed_label = "SIP" if feed == "sip" else "IEX"
            _last_actual_source = f"Alpaca {feed_label}"
            return df

    # Fall back to mock data
    _last_actual_source = "Mock Data"
    return load_from_mock(symbol, days, seed, start_date=start_date, end_date=end_date, timeframe=timeframe)


def get_data_source(feed: str = "sip") -> str:
    """Get the actual data source used on the last load."""
    return _last_actual_source


# =============================================================================
# BAR ESTIMATION HELPERS
# =============================================================================

# =============================================================================
# TRADING SESSION DEFINITIONS
# =============================================================================

TRADING_SESSIONS = ["RTH", "Pre-Market", "After Hours", "Extended Hours"]

SESSION_HOURS = {
    "RTH":            (9, 30, 16, 0),    # 6.5 hours
    "Pre-Market":     (4, 0,  9, 30),    # 5.5 hours
    "After Hours":    (16, 0, 20, 0),    # 4.0 hours
    "Extended Hours": (4, 0,  20, 0),    # 16.0 hours
}

SESSION_MARKET_HOURS = {
    "RTH": 6.5, "Pre-Market": 5.5, "After Hours": 4.0, "Extended Hours": 16.0,
}

# Approximate trading bars per day by timeframe (6.5 RTH market hours)
BARS_PER_DAY = {
    "10Sec": 2340, "30Sec": 780,  # Sub-minute (streaming engine only)
    "1Min": 390, "2Min": 195, "3Min": 130, "5Min": 78,
    "10Min": 39, "15Min": 26, "30Min": 13,
    "1Hour": 7, "2Hour": 4, "4Hour": 2,
    "1Day": 1, "1Week": 0.2, "1Month": 1 / 21,
}


def _bars_per_day(timeframe: str, session: str = "RTH") -> float:
    """Return approximate trading bars per day for a timeframe and session."""
    rth_bpd = BARS_PER_DAY.get(timeframe, 390)
    if session == "RTH":
        return rth_bpd
    session_hours = SESSION_MARKET_HOURS.get(session, 6.5)
    return rth_bpd * (session_hours / 6.5)


def estimate_bar_count(days: int, timeframe: str, session: str = "RTH") -> int:
    """Estimate total bar count for a given number of calendar days and timeframe.
    Assumes ~252 trading days per 365 calendar days (~69%).
    """
    trading_days = int(days * 252 / 365)
    return max(1, int(trading_days * _bars_per_day(timeframe, session)))


def days_from_bar_count(bars: int, timeframe: str, session: str = "RTH") -> int:
    """Convert a desired bar count to approximate calendar days."""
    import math
    bpd = _bars_per_day(timeframe, session)
    trading_days = math.ceil(bars / bpd)
    return max(1, int(math.ceil(trading_days * 365 / 252)))


# =============================================================================
# CONVENIENCE HELPERS
# =============================================================================

def load_latest_bars(
    symbol: str,
    bars: int = 200,
    timeframe: str = "1Min",
    seed: int = 42,
    feed: str = "sip",
    session: str = "RTH",
) -> pd.DataFrame:
    """
    Load the most recent N bars for a symbol.

    Calculates the minimum number of days needed to cover `bars` rows
    based on the timeframe's bars-per-day rate and trading session.
    """
    import math
    bpd = _bars_per_day(timeframe, session)
    days = max(1, math.ceil(bars / bpd) + 1)  # +1 for safety margin
    return load_market_data(symbol, days=days, timeframe=timeframe, seed=seed,
                            feed=feed, session=session)


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
