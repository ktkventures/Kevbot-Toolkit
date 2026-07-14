"""
Data Loader for RoR Trader
===========================

Handles loading market data from:
1. Polygon.io / Massive (preferred, if POLYGON_API_KEY configured)
2. Alpaca API (legacy fallback)
3. Mock data (development fallback)

Usage:
    from data_loader import load_market_data, is_alpaca_configured

    df = load_market_data("SPY", days=30)
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Check for Alpaca credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Polygon.io / Massive credentials
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "polygon")  # "polygon", "alpaca", "auto"


def is_data_configured() -> bool:
    """Check if any market data provider is configured (Polygon or Alpaca)."""
    return is_polygon_configured() or is_alpaca_configured()


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
        import logging
        logging.getLogger("data_loader").warning("Alpaca fetch failed for %s: %s", symbol, e)
        return None


def is_crypto(symbol: str) -> bool:
    """Check if a symbol is crypto (uses BASE/QUOTE format, e.g. BTC/USD)."""
    return "/" in symbol


def load_from_alpaca_crypto(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """
    Load historical bar data from Alpaca Crypto API.

    Args:
        symbol: Crypto symbol (e.g., "BTC/USD")
        days: Number of days of history to fetch
        timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)

    Returns:
        DataFrame with OHLCV data, or None if fetch fails
    """
    if not is_alpaca_configured():
        return None

    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

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

        if start_date is None or end_date is None:
            from datetime import timezone
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

        request = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start_date,
            end=end_date,
        )

        bars = client.get_crypto_bars(request)
        df = bars.df

        if len(df) == 0:
            return None

        if symbol in df.index.get_level_values(0):
            df = df.loc[symbol]

        # No session filtering for crypto (24/7 market)
        return df if len(df) > 0 else None

    except Exception as e:
        import logging
        logging.getLogger("data_loader").warning("Alpaca crypto fetch failed for %s: %s", symbol, e)
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
        end = datetime.now(timezone.utc).replace(hour=23, minute=59, second=0, microsecond=0)
        start = end - timedelta(days=days)

    # Vary seed by symbol for different price patterns
    symbol_seed = seed + hash(symbol) % 1000

    bars = generate_mock_bars([symbol], start, end, timeframe, seed=symbol_seed)

    if symbol in bars.index.get_level_values(0):
        return bars.loc[symbol]

    return pd.DataFrame()


# =============================================================================
# POLYGON.IO / MASSIVE — REST API (Phase 31A)
# =============================================================================

POLYGON_BASE_URL = "https://api.polygon.io"


def is_polygon_configured() -> bool:
    """Check if Polygon.io API key is configured."""
    return bool(POLYGON_API_KEY and not POLYGON_API_KEY.startswith("YOUR_"))


def _to_polygon_ticker(symbol: str) -> str:
    """Convert app symbol format to Polygon ticker format.

    'BTC/USD' → 'X:BTCUSD', 'SPY' → 'SPY'
    """
    if "/" in symbol:
        return "X:" + symbol.replace("/", "")
    return symbol


def _from_polygon_ticker(ticker: str) -> str:
    """Convert Polygon ticker to app symbol format.

    'X:BTCUSD' → 'BTC/USD', 'SPY' → 'SPY'
    """
    if ticker.startswith("X:"):
        base = ticker[2:]
        # Common crypto pairs: assume 3-letter base + USD
        if base.endswith("USD"):
            return base[:-3] + "/USD"
        return base
    return ticker


def _polygon_timespan(timeframe: str) -> tuple:
    """Convert app timeframe string to Polygon (multiplier, timespan).

    '1Min' → (1, 'minute'), '5Sec' → (5, 'second'), '1Hour' → (1, 'hour')
    """
    _map = {
        "5Sec": (5, "second"), "10Sec": (10, "second"),
        "15Sec": (15, "second"), "30Sec": (30, "second"),
        "1Min": (1, "minute"), "2Min": (2, "minute"),
        "3Min": (3, "minute"), "5Min": (5, "minute"),
        "10Min": (10, "minute"), "15Min": (15, "minute"),
        "30Min": (30, "minute"),
        "1Hour": (1, "hour"), "2Hour": (2, "hour"), "4Hour": (4, "hour"),
        "1Day": (1, "day"), "1Week": (1, "week"), "1Month": (1, "month"),
    }
    return _map.get(timeframe, (1, "minute"))


def _polygon_bars_to_df(results: list) -> pd.DataFrame:
    """Convert Polygon API bar results to our standard DataFrame format.

    Polygon fields: v(volume), vw(vwap), o(open), c(close), h(high), l(low),
                    t(timestamp ms), n(trade_count)
    """
    if not results:
        return pd.DataFrame()

    records = []
    for bar in results:
        ts = pd.Timestamp(bar['t'], unit='ms', tz='UTC')
        records.append({
            'open': bar.get('o', 0),
            'high': bar.get('h', 0),
            'low': bar.get('l', 0),
            'close': bar.get('c', 0),
            'volume': bar.get('v', 0),
            'vwap': bar.get('vw', 0),
            'trade_count': bar.get('n', 0),
        })

    df = pd.DataFrame(records)
    timestamps = [pd.Timestamp(bar['t'], unit='ms', tz='UTC') for bar in results]
    df.index = pd.DatetimeIndex(timestamps, name='timestamp')
    df.sort_index(inplace=True)
    return df


def _polygon_fetch_bars(ticker: str, multiplier: int, timespan: str,
                         from_date: str, to_date: str) -> list:
    """Fetch bars from Polygon REST API with pagination.

    Returns list of bar dicts. Handles next_url pagination automatically.
    """
    import requests

    all_results = []
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        'apiKey': POLYGON_API_KEY,
        'limit': 50000,
        'sort': 'asc',
        'adjusted': 'true',
    }

    while url:
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                # Rate limited — wait and retry
                import time
                time.sleep(1)
                continue
            if resp.status_code != 200:
                logger.warning(f"Polygon API error {resp.status_code}: {resp.text[:200]}")
                return all_results

            data = resp.json()
            results = data.get('results', [])
            all_results.extend(results)

            # Follow pagination
            next_url = data.get('next_url')
            if next_url:
                # next_url is a full URL; just append apiKey
                url = next_url
                params = {'apiKey': POLYGON_API_KEY}
            else:
                url = None
        except Exception as e:
            logger.warning(f"Polygon fetch error: {e}")
            break

    return all_results


def load_from_polygon(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: str = "RTH",
) -> Optional[pd.DataFrame]:
    """Load historical bars from Polygon.io REST API for stocks.

    Returns DataFrame with standard columns (open, high, low, close, volume,
    vwap, trade_count) and UTC DatetimeIndex, or None on failure.
    """
    ticker = _to_polygon_ticker(symbol)
    multiplier, timespan = _polygon_timespan(timeframe)

    if end_date:
        to_dt = end_date
    else:
        to_dt = datetime.now(timezone.utc)

    if start_date:
        from_dt = start_date
    else:
        from_dt = to_dt - timedelta(days=days)

    from_str = from_dt.strftime('%Y-%m-%d')
    to_str = to_dt.strftime('%Y-%m-%d')

    results = _polygon_fetch_bars(ticker, multiplier, timespan, from_str, to_str)
    if not results:
        return None

    df = _polygon_bars_to_df(results)
    if len(df) == 0:
        return None

    # Apply session filtering (same as Alpaca path)
    if session and session != "24/7":
        df = _filter_session(df, session)

    return df if len(df) > 0 else None


def load_from_polygon_crypto(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """Load historical bars from Polygon.io REST API for crypto.

    Crypto tickers use X:BTCUSD format. No session filtering (24/7).
    """
    ticker = _to_polygon_ticker(symbol)
    multiplier, timespan = _polygon_timespan(timeframe)

    if end_date:
        to_dt = end_date
    else:
        to_dt = datetime.now(timezone.utc)

    if start_date:
        from_dt = start_date
    else:
        from_dt = to_dt - timedelta(days=days)

    from_str = from_dt.strftime('%Y-%m-%d')
    to_str = to_dt.strftime('%Y-%m-%d')

    results = _polygon_fetch_bars(ticker, multiplier, timespan, from_str, to_str)
    if not results:
        return None

    df = _polygon_bars_to_df(results)
    return df if len(df) > 0 else None


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
        symbol: Stock symbol (e.g., "SPY") or crypto (e.g., "BTC/USD")
        days: Number of days of history (used if start_date/end_date not provided)
        timeframe: Bar timeframe (only affects Alpaca, mock always uses 1Min)
        seed: Random seed for mock data
        force_mock: If True, skip Alpaca and use mock data
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)
        feed: Data feed — "sip" (consolidated) or "iex" (IEX only). Ignored for crypto.
        session: Trading session filter — "RTH", "Pre-Market", etc. Ignored for crypto.

    Returns:
        DataFrame with OHLCV data (columns: open, high, low, close, volume, trade_count, vwap)
    """
    global _last_actual_source

    _use_polygon = DATA_PROVIDER in ("polygon", "auto") and is_polygon_configured()
    _use_alpaca = DATA_PROVIDER in ("alpaca", "auto") and is_alpaca_configured()

    # Try Polygon first (unless forced to mock or provider is alpaca-only)
    if not force_mock and _use_polygon:
        if is_crypto(symbol):
            df = load_from_polygon_crypto(symbol, days, timeframe,
                                           start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                _last_actual_source = "Polygon Crypto"
                return df
        else:
            df = load_from_polygon(symbol, days, timeframe, start_date=start_date,
                                    end_date=end_date, session=session)
            if df is not None and len(df) > 0:
                _last_actual_source = "Polygon"
                return df

    # Alpaca fallback
    if not force_mock and _use_alpaca:
        if is_crypto(symbol):
            df = load_from_alpaca_crypto(symbol, days, timeframe,
                                         start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                _last_actual_source = "Alpaca Crypto"
                return df
        else:
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
# TIMEFRAME LABEL MAPPING (for Multi-Timeframe Confluence)
# =============================================================================

TF_LABELS = {
    "5Sec": "5s", "10Sec": "10s", "15Sec": "15s", "30Sec": "30s",
    "1Min": "1m", "2Min": "2m", "3Min": "3m", "5Min": "5m",
    "10Min": "10m", "15Min": "15m", "30Min": "30m",
    "1Hour": "1h", "2Hour": "2h", "4Hour": "4h",
    "1Day": "1d", "1Week": "1w", "1Month": "1mo",
}
TF_FROM_LABEL = {v: k for k, v in TF_LABELS.items()}

# All timeframes available for MTF confluence selection
MTF_AVAILABLE_TIMEFRAMES = [
    "5Sec", "10Sec", "15Sec", "30Sec",
    "1Min", "2Min", "3Min", "5Min", "10Min", "15Min", "30Min",
    "1Hour", "2Hour", "4Hour", "1Day",
]

# Sub-minute timeframes — Polygon REST supports these natively;
# Alpaca REST does not (streaming only). Empty set when Polygon configured.
SUB_MINUTE_TIMEFRAMES = set() if is_polygon_configured() else {"5Sec", "10Sec", "15Sec", "30Sec"}


def get_tf_label(timeframe: str) -> str:
    """Convert canonical timeframe (e.g. '5Min') to short label (e.g. '5m')."""
    return TF_LABELS.get(timeframe, timeframe.lower())


def get_tf_from_label(label: str) -> str:
    """Convert short label (e.g. '5m') to canonical timeframe (e.g. '5Min')."""
    return TF_FROM_LABEL.get(label, label)


def get_required_tfs_from_confluence(confluence_records) -> set:
    """
    Extract unique secondary TF labels from confluence record prefixes.

    Records like '5m-EMA_STACK_DEFAULT-SML' → {'5m'}.
    Ignores 'GEN-' prefixed records and '1M' (primary TF).
    """
    tfs = set()
    for record in (confluence_records or []):
        if record.startswith("GEN-"):
            continue
        tf_label = record.split("-")[0]
        if tf_label != "1M":
            tfs.add(tf_label)
    return tfs


# =============================================================================
# RESAMPLING (Multi-Timeframe)
# =============================================================================

# Mapping from canonical timeframe strings to pandas resample rules
_RESAMPLE_RULES = {
    "5Sec": "5s", "10Sec": "10s", "15Sec": "15s", "30Sec": "30s",
    "2Min": "2min", "3Min": "3min", "5Min": "5min",
    "10Min": "10min", "15Min": "15min", "30Min": "30min",
    "1Hour": "1h", "2Hour": "2h", "4Hour": "4h",
    "1Day": "1D",
}


def resample_to_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample OHLCV data from a finer timeframe to a coarser one.

    Args:
        df: OHLCV DataFrame with DatetimeIndex (must be finer than target_tf).
        target_tf: Canonical target timeframe (e.g., "5Min", "15Min", "1Hour").

    Returns:
        Resampled DataFrame with OHLCV columns. Rows where open is NaN are dropped.
    """
    rule = _RESAMPLE_RULES.get(target_tf)
    if rule is None:
        raise ValueError(f"Cannot resample to timeframe '{target_tf}'. "
                         f"Supported: {list(_RESAMPLE_RULES.keys())}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("OHLCV data must use a DatetimeIndex")

    source = df.copy()
    source.index = pd.to_datetime(source.index, utc=True)
    source = source.sort_index()
    source = source[~source.index.duplicated(keep="last")]

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "trade_count" in df.columns:
        agg["trade_count"] = "sum"
    if "vwap" in df.columns:
        # VWAP must be recomputed on the resampled data — skip here
        pass

    # The live BarBuilder aligns fixed-duration bars to Unix epoch boundaries.
    # Make that contract explicit here so local timezone or pandas defaults
    # cannot shift historical candles relative to live candles.
    resampled = source.resample(
        rule, origin="epoch", label="left", closed="left").agg(agg)
    return resampled.dropna(subset=["open"])


# =============================================================================
# BAR ESTIMATION HELPERS
# =============================================================================

# =============================================================================
# TRADING SESSION DEFINITIONS
# =============================================================================

TRADING_SESSIONS = ["RTH", "Pre-Market", "After Hours", "Extended Hours", "24/7"]

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
    "5Sec": 4680, "10Sec": 2340, "15Sec": 1560, "30Sec": 780,  # Sub-minute (streaming engine only)
    "1Min": 390, "2Min": 195, "3Min": 130, "5Min": 78,
    "10Min": 39, "15Min": 26, "30Min": 13,
    "1Hour": 7, "2Hour": 4, "4Hour": 2,
    "1Day": 1, "1Week": 0.2, "1Month": 1 / 21,
}

# Crypto: 24 hours/day, 7 days/week (no session concept)
CRYPTO_BARS_PER_DAY = {
    "5Sec": 17280, "10Sec": 8640, "15Sec": 5760, "30Sec": 2880,
    "1Min": 1440, "2Min": 720, "3Min": 480, "5Min": 288,
    "10Min": 144, "15Min": 96, "30Min": 48,
    "1Hour": 24, "2Hour": 12, "4Hour": 6,
    "1Day": 1, "1Week": 1 / 7, "1Month": 1 / 30,
}


def _bars_per_day(timeframe: str, session: str = "RTH", asset_type: str = "equity") -> float:
    """Return approximate trading bars per day for a timeframe and session."""
    if asset_type == "crypto":
        return CRYPTO_BARS_PER_DAY.get(timeframe, 1440)
    rth_bpd = BARS_PER_DAY.get(timeframe, 390)
    if session == "RTH":
        return rth_bpd
    session_hours = SESSION_MARKET_HOURS.get(session, 6.5)
    return rth_bpd * (session_hours / 6.5)


def estimate_bar_count(days: int, timeframe: str, session: str = "RTH",
                       asset_type: str = "equity") -> int:
    """Estimate total bar count for a given number of calendar days and timeframe.
    Assumes ~252 trading days per 365 calendar days (~69%) for equities.
    Crypto trades every day (365/365).
    """
    if asset_type == "crypto":
        trading_days = days
    else:
        trading_days = int(days * 252 / 365)
    return max(1, int(trading_days * _bars_per_day(timeframe, session, asset_type)))


def days_from_bar_count(bars: int, timeframe: str, session: str = "RTH",
                        asset_type: str = "equity") -> int:
    """Convert a desired bar count to approximate calendar days."""
    import math
    bpd = _bars_per_day(timeframe, session, asset_type)
    trading_days = math.ceil(bars / bpd)
    if asset_type == "crypto":
        return max(1, trading_days)
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
    Automatically detects crypto symbols (containing '/') and adjusts.
    """
    import math
    asset_type = "crypto" if is_crypto(symbol) else "equity"
    bpd = _bars_per_day(timeframe, session, asset_type)
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
