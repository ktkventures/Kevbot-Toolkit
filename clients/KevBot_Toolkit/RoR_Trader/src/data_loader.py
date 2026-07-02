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
        "1Sec": (1, "second"),
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
    import time

    all_results = []
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        'apiKey': POLYGON_API_KEY,
        'limit': 50000,
        'sort': 'asc',
        'adjusted': 'true',
    }

    # CRITICAL (2026-06-18): never silently return PARTIAL data. With sort=asc,
    # a mid-pagination failure drops the most-RECENT bars, leaving a backtest
    # lane that can't pair recent live alerts (→ all-phantom). Long sub-minute
    # loads span many pages, so transient page errors are common. Retry each
    # page with backoff; if a page can't be fetched after MAX_RETRIES, RAISE
    # loudly so the caller fails visibly instead of persisting a truncated lane.
    MAX_RETRIES = 6
    page_attempts = 0

    while url:
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500 or resp.status_code != 200:
                page_attempts += 1
                if page_attempts >= MAX_RETRIES:
                    raise RuntimeError(
                        f"Polygon API {resp.status_code} for {ticker} "
                        f"{multiplier}/{timespan} after {MAX_RETRIES} retries "
                        f"(have {len(all_results)} bars, INCOMPLETE): "
                        f"{resp.text[:160]}")
                time.sleep(min(2 ** page_attempts, 30))
                continue
            page_attempts = 0  # reset on a successful page

            data = resp.json()
            all_results.extend(data.get('results', []))

            # Follow pagination (next_url is a full URL; just append apiKey)
            next_url = data.get('next_url')
            if next_url:
                url = next_url
                params = {'apiKey': POLYGON_API_KEY}
            else:
                url = None
        except requests.RequestException as e:
            page_attempts += 1
            if page_attempts >= MAX_RETRIES:
                raise RuntimeError(
                    f"Polygon fetch failed for {ticker} {multiplier}/{timespan} "
                    f"after {MAX_RETRIES} retries (have {len(all_results)} bars, "
                    f"INCOMPLETE — refusing to return partial): {e}")
            time.sleep(min(2 ** page_attempts, 30))
            continue

    return all_results


# =============================================================================
# HI-FI: 1-SECOND BAR FETCHING + DAY-LEVEL CACHING
# =============================================================================

_1s_cache: dict = {}  # In-memory cache: {ticker_YYYY-MM-DD: pd.DataFrame}


def fetch_1s_bars_for_window(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    padding_seconds: int = 30,
) -> pd.DataFrame:
    """Fetch 1-second OHLCV from Polygon for a specific time window + padding.

    Used for Hi-Fi backtest resolution and trade drill-down visualization.
    Uses day-level caching: fetches full trading day on first request,
    reuses for subsequent requests on the same ticker+date.

    Args:
        ticker: Stock symbol (e.g., "SPY")
        start_dt: Start of the window (UTC datetime)
        end_dt: End of the window (UTC datetime)
        padding_seconds: Extra seconds before/after the window

    Returns:
        DataFrame with 1-second OHLCV bars, indexed by timestamp.
    """
    import pandas as pd

    padded_start = start_dt - timedelta(seconds=padding_seconds)
    padded_end = end_dt + timedelta(seconds=padding_seconds)

    # M-RS4 Phase 1 — serve 1Sec from the REST Bar Cache (bar_cache) FIRST so no
    # consumer hits Polygon directly. prime_1s_cache_from_rest_bars is gated on
    # BAR_CACHE_ENABLED + a DSN, never clobbers a fresher in-process day, and a
    # miss/error returns 0 → the per-day Polygon loop below runs unchanged (so
    # this is a pure speed+unification win with an automatic Polygon fallback).
    # REST Bars 1Sec is byte-identical to a fresh Polygon pull on settled bars
    # (parity-gated); the live tip re-fetches via the staleness eviction below.
    # Kill-switch: RORT_FETCH1S_FROM_CACHE=0.
    if os.environ.get("RORT_FETCH1S_FROM_CACHE", "1").strip().lower() in (
            "1", "true", "yes", "on"):
        try:
            prime_1s_cache_from_rest_bars(
                ticker, start_dt, end_dt, padding_seconds=padding_seconds)
        except Exception as _prime_err:  # noqa: BLE001 — never block the fetch
            logger.warning("[HIFI] inline prime failed for %s (%s) — per-day Polygon",
                           ticker, _prime_err)

    # Collect all needed dates
    dates_needed = set()
    cursor = padded_start.date()
    while cursor <= padded_end.date():
        dates_needed.add(cursor)
        cursor += timedelta(days=1)

    # Fetch any missing dates
    poly_ticker = _to_polygon_ticker(ticker)
    for d in dates_needed:
        cache_key = f"{poly_ticker}_{d.isoformat()}"
        # Staleness check: Polygon's 1s aggregates settle with ~3-5min
        # lag on recent bars. If the cache was populated mid-aggregation
        # earlier in this container's lifetime, it can be missing bars
        # that have since become available. Symptom (observed
        # 2026-06-05): bulk UADs spanning >5 min leave the second-half
        # strategies' Hi-Fi Pass 2 silently failing because the cache
        # populated by the first strategy's fetch stops short of the
        # later strategies' requested windows. Evict + re-fetch when
        # we detect this.
        if cache_key in _1s_cache:
            cached = _1s_cache[cache_key]
            if cached is not None and len(cached) > 0:
                cached_max = cached.index.max()
                needed_end_ts = pd.Timestamp(padded_end)
                if needed_end_ts.tzinfo is None:
                    needed_end_ts = needed_end_ts.tz_localize('UTC')
                # Only treat a short cache as STALE when the request reaches into
                # the still-settling recent window (~revision horizon). For a fully
                # SETTLED past window, bar_cache/Polygon are complete, and a last
                # bar before padded_end just means "no trade in the tail" — evicting
                # there would needlessly re-fetch identical data from Polygon and
                # defeat the bar_cache routing (M-RS4 Phase 1). The original
                # eviction (2026-06-05) targeted today's mid-aggregation cache, so
                # scoping it to the settling edge preserves that fix.
                _settling_edge = datetime.now(timezone.utc) - timedelta(minutes=45)
                if needed_end_ts > cached_max and needed_end_ts >= _settling_edge:
                    logger.info(
                        "[HIFI] 1s cache stale for %s on %s "
                        "(cached through %s, requested through %s) "
                        "— evicting + re-fetching",
                        ticker, d, cached_max, needed_end_ts)
                    del _1s_cache[cache_key]
        if cache_key not in _1s_cache:
            from_str = d.isoformat()
            to_str = d.isoformat()
            logger.info("[HIFI] Fetching 1-second bars for %s on %s", ticker, from_str)
            results = _polygon_fetch_bars(poly_ticker, 1, 'second', from_str, to_str)
            if results:
                _1s_cache[cache_key] = _polygon_bars_to_df(results)
            else:
                _1s_cache[cache_key] = pd.DataFrame()

    # Combine cached data for all needed dates
    frames = []
    for d in sorted(dates_needed):
        cache_key = f"{poly_ticker}_{d.isoformat()}"
        df = _1s_cache.get(cache_key)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined.sort_index(inplace=True)

    # Filter to padded window (handle both tz-aware and tz-naive datetimes)
    def _to_utc_ts(dt):
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')

    mask = (combined.index >= _to_utc_ts(padded_start)) & \
           (combined.index <= _to_utc_ts(padded_end))
    return combined.loc[mask]


# Circuit breaker for the REST Bars 1-second prime (2026-07-02 incident):
# a 90-day whole-span read exceeded the DB statement timeout, and because
# the prime is invoked per Hi-Fi candidate (up to ~1500 per mass search),
# every candidate re-issued the same doomed multi-million-row query —
# saturating shared Supabase Postgres (Cloudflare 522; live Worker alert
# writes failed). After any read failure, the prime disables itself for
# a cooldown and callers fall back to per-day Polygon (which never touches
# the shared DB).
_prime_1s_breaker_until: float = 0.0

# Days a successful REST Bars read proved bar-less (weekends/holidays), keyed
# "{poly_ticker}_{date}". They can never enter _1s_cache (no rows → no day
# frame), so without this memo every prime call would re-query the windows
# containing them. Deliberately NOT stored in _1s_cache: fetch_1s_bars_for_
# window's per-day Polygon fallback must keep behaving exactly as before for
# these days (e.g. symbol-not-stocked ambiguity). Only settled-past days are
# memoized — a still-filling recent day may gain bars later.
_prime_1s_empty_days: set = set()


def prime_1s_cache_from_rest_bars(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    padding_seconds: int = 60,
) -> int:
    """Load-once: pull the [start, end] 1-second window from REST Bars
    (`bar_cache`) via direct-Postgres reads and populate the per-day
    ``_1s_cache`` that ``fetch_1s_bars_for_window`` reads from.

    This is the M-RS2 "pull big swaths once" speedup for Hi-Fi Pass 2:
    instead of N per-day Polygon fetches (one per distinct trade date), read
    from REST Bars over the full period. REST Bars 1-second is byte-identical
    to a fresh Polygon REST pull for settled days (Two_Bar_Caches_DEFINITIONS.md).

    2026-07-02 hardening (post-incident):
      - Days already warm in ``_1s_cache`` are skipped BEFORE any query, so
        repeated calls over the same span (the per-candidate Hi-Fi loop) cost
        zero queries after the first successful prime.
      - Cold days are read in bounded chunks (``RORT_PRIME1S_CHUNK_DAYS``,
        default 7) instead of one whole-span statement, so no single query
        can approach the DB statement timeout.
      - Any read failure trips a process-wide circuit breaker
        (``RORT_PRIME1S_BREAKER_COOLDOWN_S``, default 600): remaining chunks
        are abandoned and subsequent calls return 0 immediately until the
        cooldown lapses — the DB is never hammered with retries.
      - Kill-switch ``RORT_PRIME1S_CHUNKED=0`` restores the legacy
        single-statement whole-span read (no breaker).

    Safe + transparent: gated on BAR_CACHE_ENABLED + a DSN. Returns the number
    of day-frames primed; returns 0 on any miss (cache off, no DSN, no rows,
    error, breaker open) so the caller's existing fetch_1s_bars_for_window
    calls fall straight back to per-day Polygon. Never clobbers a day already
    in _1s_cache (a fresher Polygon fetch wins). The most-recent day, if REST
    Bars lags the requested end, is re-fetched from Polygon by
    fetch_1s_bars_for_window's staleness eviction — so the live tip stays
    correct.
    """
    global _prime_1s_breaker_until
    # ALWAYS emit one [HIFI] load-once line summarizing the outcome, so we can
    # tell from the logs whether the REST Bars cache actually served the 1-second
    # (vs falling back to per-day Polygon, or finding days already warm in the
    # long-lived process's _1s_cache) WITHOUT having to log-dive. 2026-06-26.
    import time as _time_mod
    if _time_mod.monotonic() < _prime_1s_breaker_until:
        logger.debug("[HIFI] load-once breaker OPEN for %.0fs more — per-day "
                     "Polygon", _prime_1s_breaker_until - _time_mod.monotonic())
        return 0
    chunked = os.environ.get("RORT_PRIME1S_CHUNKED", "1").strip().lower() in (
        "1", "true", "yes", "on")
    try:
        import bar_cache as _bc
        _enabled, _dsn = _bc.is_enabled(), _bc.direct_pg_available()
        if not (_enabled and _dsn):
            logger.info("[HIFI] load-once skipped for %s — BAR_CACHE_ENABLED=%s "
                        "direct_pg=%s → per-day Polygon", ticker, _enabled, _dsn)
            return 0
        padded_start = start_dt - timedelta(seconds=padding_seconds)
        padded_end = end_dt + timedelta(seconds=padding_seconds)
    except Exception as e:
        logger.warning("[HIFI] load-once prime FAILED for %s: %s → per-day Polygon",
                       ticker, e)
        return 0

    poly_ticker = _to_polygon_ticker(ticker)

    if not chunked:
        # Legacy path (kill-switch): one whole-span statement, no breaker.
        try:
            df = _bc.read_bars(ticker, "1Sec", padded_start, padded_end)
        except Exception as e:
            logger.warning("[HIFI] load-once prime FAILED for %s: %s → per-day "
                           "Polygon", ticker, e)
            return 0
        if df is None or len(df) == 0:
            logger.info("[HIFI] load-once: REST Bars returned 0 rows for %s "
                        "[%s..%s] → per-day Polygon", ticker,
                        start_dt.isoformat()[:19], end_dt.isoformat()[:19])
            return 0
        primed = 0
        warm = 0
        for d, day_df in df.groupby(df.index.date):
            cache_key = f"{poly_ticker}_{d.isoformat()}"
            if cache_key not in _1s_cache:  # don't clobber a fresher Polygon fetch
                _1s_cache[cache_key] = day_df
                primed += 1
            else:
                warm += 1
        logger.info("[HIFI] load-once for %s: read %d rows from REST Bars "
                    "(direct-PG); primed %d new day(s), %d already warm "
                    "in-process", ticker, len(df), primed, warm)
        return primed

    # ── Chunked path (default) ──
    # 1. Cold-day set: only days NOT already in _1s_cache get queried.
    all_days: list = []
    cursor = padded_start.date()
    while cursor <= padded_end.date():
        all_days.append(cursor)
        cursor += timedelta(days=1)
    cold_days = [d for d in all_days
                 if f"{poly_ticker}_{d.isoformat()}" not in _1s_cache
                 and f"{poly_ticker}_{d.isoformat()}" not in _prime_1s_empty_days]
    warm = len(all_days) - len(cold_days)
    if not cold_days:
        logger.info("[HIFI] load-once for %s: all %d day(s) already warm "
                    "in-process — 0 queries", ticker, warm)
        return 0

    # 2. Group consecutive cold days into runs, split runs into ≤CHUNK-day
    #    windows so each statement stays far below the DB statement timeout.
    try:
        chunk_days = max(1, int(os.environ.get("RORT_PRIME1S_CHUNK_DAYS", "7")))
    except ValueError:
        chunk_days = 7
    runs: list = []
    run_start = cold_days[0]
    prev = cold_days[0]
    for d in cold_days[1:]:
        if (d - prev).days > 1:
            runs.append((run_start, prev))
            run_start = d
        prev = d
    runs.append((run_start, prev))
    windows: list = []
    for r_start, r_end in runs:
        w_start = r_start
        while w_start <= r_end:
            w_end = min(r_end, w_start + timedelta(days=chunk_days - 1))
            windows.append((w_start, w_end))
            w_start = w_end + timedelta(days=1)

    # 3. Read each window; first failure trips the breaker and abandons the
    #    rest — callers fall back to per-day Polygon for anything unprimed.
    primed = 0
    rows_total = 0
    for w_start, w_end in windows:
        q_start = max(padded_start,
                      datetime(w_start.year, w_start.month, w_start.day,
                               tzinfo=timezone.utc))
        q_end = min(padded_end,
                    datetime(w_end.year, w_end.month, w_end.day,
                             tzinfo=timezone.utc) + timedelta(days=1))
        try:
            df = _bc.read_bars(ticker, "1Sec", q_start, q_end)
        except Exception as e:
            try:
                cooldown = float(os.environ.get(
                    "RORT_PRIME1S_BREAKER_COOLDOWN_S", "600"))
            except ValueError:
                cooldown = 600.0
            _prime_1s_breaker_until = _time_mod.monotonic() + cooldown
            logger.warning(
                "[HIFI] load-once chunk FAILED for %s [%s..%s]: %s — breaker "
                "OPEN for %.0fs, abandoning %d remaining chunk(s) → per-day "
                "Polygon", ticker, w_start.isoformat(), w_end.isoformat(), e,
                cooldown, len(windows) - windows.index((w_start, w_end)) - 1)
            break
        got_days = set()
        if df is not None and len(df) > 0:
            rows_total += len(df)
            for d, day_df in df.groupby(df.index.date):
                got_days.add(d)
                cache_key = f"{poly_ticker}_{d.isoformat()}"
                if cache_key not in _1s_cache:  # don't clobber a fresher fetch
                    _1s_cache[cache_key] = day_df
                    primed += 1
        # Memoize settled-past days this read proved bar-less so later calls
        # skip them (see _prime_1s_empty_days). The still-settling edge is
        # excluded — those days may gain bars.
        _settled_cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).date()
        w_cursor = w_start
        while w_cursor <= w_end:
            if w_cursor not in got_days and w_cursor < _settled_cutoff:
                _prime_1s_empty_days.add(f"{poly_ticker}_{w_cursor.isoformat()}")
            w_cursor += timedelta(days=1)
    logger.info("[HIFI] load-once for %s: read %d rows from REST Bars "
                "(direct-PG, %d chunk(s) of ≤%dd); primed %d new day(s), "
                "%d already warm in-process",
                ticker, rows_total, len(windows), chunk_days, primed, warm)
    return primed


def clear_1s_cache():
    """Clear the in-memory 1-second bar cache (and the empty-day memo)."""
    global _1s_cache
    _1s_cache = {}
    _prime_1s_empty_days.clear()


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
    no_backfill: bool = False,
) -> pd.DataFrame:
    """
    Load market data, preferring Alpaca if configured.

    `no_backfill` (M-RS4 Phase 3): read-only cache access — never fetch Polygon
    or write the cache, and do NOT fall through to a direct Polygon call when the
    cache is short. For read-only consumers (shadow-worker). Returns whatever is
    cached (possibly empty); caller tolerates a short range.

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

    # Persistent bar cache (Layer 1 — opt-in via BAR_CACHE_ENABLED env var).
    # Wraps the Polygon stocks path; routes through Supabase Postgres bar_cache
    # table with delta-fetch from Polygon for any gap. Crypto / Alpaca / mock
    # paths bypass and run through the legacy code below. See bar_cache.py.
    if not force_mock and _use_polygon and not is_crypto(symbol):
        try:
            from bar_cache import cached_load_market_data, is_enabled
            if is_enabled():
                df = cached_load_market_data(
                    symbol=symbol, days=days, timeframe=timeframe,
                    start_date=start_date, end_date=end_date, session=session,
                    no_backfill=no_backfill)
                if df is not None and len(df) > 0:
                    _last_actual_source = "Polygon (cached)"
                    return df
                if no_backfill:
                    # Read-only: do NOT fall through to direct Polygon — return
                    # whatever the cache had (possibly empty). Caller tolerates it.
                    _last_actual_source = "Polygon (cached, read-only)"
                    import pandas as _pd
                    return df if df is not None else _pd.DataFrame()
                # Cache returned None (e.g. cold + Polygon failed) — fall
                # through to direct Polygon call as a safety net
        except Exception as _cache_err:  # noqa: F841
            # Never let a cache bug block a market-data call. Log and fall
            # through to the legacy path.
            import logging as _l
            _l.getLogger(__name__).warning(
                "bar_cache: lookup failed, falling back to direct Polygon "
                "(symbol=%s, tf=%s): %s", symbol, timeframe, _cache_err)

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

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "trade_count" in df.columns:
        agg["trade_count"] = "sum"
    if "vwap" in df.columns:
        # VWAP must be recomputed on the resampled data — skip here
        pass

    resampled = df.resample(rule).agg(agg)
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


# Map timeframe-string → seconds. Mirrors the table in strategies.py
# (kept in sync) but lives here so cache-backed callers can resolve
# without importing from api/routers.
TF_TO_SECONDS = {
    "5Sec": 5, "10Sec": 10, "15Sec": 15, "30Sec": 30,
    "1Min": 60, "2Min": 120, "3Min": 180, "5Min": 300,
    "10Min": 600, "15Min": 900, "30Min": 1800,
    "1Hour": 3600, "2Hour": 7200, "4Hour": 14400,
    "1Day": 86400, "1Week": 604800,
}


# Sources the LIVE ENGINE actually consumed into its in-memory bar
# history (2026-06-11). 'rest_correction' rows are WS bars whose row was
# upserted by apply_rest_correction AFTER the engine spliced the REST
# values into builder.history — the engine consumed the corrected bar,
# so faithful "what the engine saw" views must include them (the upsert
# overwrites source, so filtering to ws/ws_agg silently DROPS every
# corrected bar). 'rest_insert' rows are WS-missed bars healed by
# gap_healer and inserted into builder.history. 'warmup_seed' rows are
# warmup REST bars the engine's startup seed consumed but the cache
# missed (worker restart/deploy holes — see live_bars_writer.
# reconcile_seeded_history). 'rest_backfill' stays excluded everywhere:
# cosmetic cache patching the engine never consumes. first_*
# decision-time columns survive upserts via the preserve-trigger in
# migrations/live_bars_first_values.sql.
ENGINE_CONSUMED_SOURCES = [
    'ws', 'ws_agg', 'rest_correction', 'rest_insert', 'warmup_seed']


def fetch_cache_as_df(
    symbol: str,
    tf_seconds: int,
    start_dt,
    end_dt,
    value_type: str = 'latest',
    sources: list = None,
):
    """Pull `live_bars` rows for (symbol, tf_seconds, time range) and return
    a DataFrame with UTC DatetimeIndex and OHLCV columns.

    M8.7 (2026-05-02): used by chart-data-cache endpoint (Lab tab) to
    inject WS-aggregated bars into the indicator pipeline. Picks
    `first_*` columns when value_type='first' (decision-time view);
    falls back to `*` cols if first_* is NULL (pre-migration rows).

    M8.7 (2026-05-06, Phase E preview): added `sources` parameter for
    `cache_locked` backtest model dispatch. When `sources` is a list
    (e.g. ['ws', 'ws_agg']), filters rows to those sources only —
    excludes `rest_backfill` rows so the result reflects what the
    LIVE engine actually saw. Default None = all sources (used by
    `cache_corrected` once Phase D ships, and the Lab tab via the
    explicit ['ws', 'ws_agg'] filter).

    Returns empty DataFrame if no rows in window.
    """
    import pandas as _pd
    from db import get_admin_client

    c = get_admin_client()
    cols = ("bar_start,open,high,low,close,volume,"
            "first_open,first_high,first_low,first_close,first_volume,source")
    rows = []
    page_size = 1000
    offset = 0
    while True:
        try:
            q = c.table('live_bars').select(cols) \
                .eq('symbol', symbol) \
                .eq('timeframe_seconds', tf_seconds) \
                .gte('bar_start', start_dt.isoformat()) \
                .lte('bar_start', end_dt.isoformat())
            if sources:
                q = q.in_('source', list(sources))
            r = q.order('bar_start') \
                 .range(offset, offset + page_size - 1) \
                 .execute()
        except Exception as e:
            logger.warning("fetch_cache_as_df: fetch failed offset=%d: %s",
                           offset, e)
            break
        batch = r.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not rows:
        return _pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume'])

    use_first = value_type == 'first'
    out_rows = []
    timestamps = []
    for row in rows:
        if use_first and row.get('first_close') is not None:
            o = row.get('first_open')
            h = row.get('first_high')
            l = row.get('first_low')
            cl = row.get('first_close')
            v = row.get('first_volume')
        else:
            o = row.get('open')
            h = row.get('high')
            l = row.get('low')
            cl = row.get('close')
            v = row.get('volume')
        timestamps.append(_pd.Timestamp(row['bar_start']))
        out_rows.append({
            'open': float(o or 0), 'high': float(h or 0),
            'low': float(l or 0), 'close': float(cl or 0),
            'volume': float(v or 0),
        })

    df = _pd.DataFrame(out_rows, index=_pd.DatetimeIndex(
        timestamps, name='timestamp'))
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


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
