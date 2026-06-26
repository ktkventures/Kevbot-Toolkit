""""REST Bars" — persistent REST-canonical bar cache (bar_cache table).

================================================================================
THIS IS THE *REST BARS* CACHE — one of two bar caches. Do not confuse it with
"Live Bars" (`live_bars` table, written by live_bars_writer.py).

  - REST Bars (this, `bar_cache`): Polygon REST aggregates. REVISABLE on purpose
    — maintain_symbol/revision-refresh re-pull and overwrite so it tracks current
    Polygon REST (Polygon revises its own 1s/1min for days). Used for backtest
    speed + fidelity. Byte-identical to a fresh Polygon REST pull is the GOAL.
  - Live Bars (`live_bars`): what the WebSocket saw at decision time. IMMUTABLE —
    a seen ws_agg bar is NEVER post-corrected. (REST only gap-fills *missing*
    minutes there, insert-only, never overwriting a seen bar.)

RED LINE: the overwrite-to-match-Polygon logic in this module touches ONLY
`bar_cache`. It must NEVER be pointed at `live_bars`.
Canonical reference: docs/_active/Two_Bar_Caches_DEFINITIONS.md
================================================================================

Wraps load_market_data so identical (symbol, timeframe, days, session)
combos within a session don't re-fetch from Polygon. Also speeds up
backtests, refreshes, and worker warmup by serving most bars from the
cache and only delta-fetching the gap from Polygon.

Architecture
------------
- Cache stores RAW bars (extended hours included) per (symbol, timeframe, ts).
  Session filter (RTH, etc.) is applied on read, not on store. This avoids
  duplicating the same bar across (RTH vs Pre-Market vs Extended) views.
- Mandatory delta-fetch on every read: always check `now - last_cached_ts`
  and fetch the gap from Polygon. Costs one trivial network call per cache
  hit but ensures the worker warmup path never sees stale bars.
- Bulk upsert via supabase-py with on_conflict to handle duplicates cheaply.
- Feature flag BAR_CACHE_ENABLED — default off. Enable per-environment via
  env var. Backward compatible: when off, callers go straight to Polygon.

Why Supabase Postgres (not parquet on disk)
-------------------------------------------
- No infra setup (already provisioned)
- Cross-replica when we go multi-replica
- Automatic backups
- ~30ms range query latency vs Polygon's 500ms+ — 10-50x speedup

Constraints
-----------
- Worker warmup correctness: indicator state depends on bars being
  current. Mandatory delta-fetch on every read is the safety guarantee.
- Session filter must run AFTER cache read: cache stores all hours, caller
  asked for RTH → we filter the slice to RTH timestamps before returning.
- Resampling stays caller-side: cache stores 1Min (and 1Sec for L3),
  caller resamples to 5Min/15Min/etc. on demand. Matches CLAUDE.md
  convention of always starting from 1Min bars.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Feature flag
# =============================================================================

def is_enabled() -> bool:
    """Cache is opt-in via env var. Default off so existing behaviour is
    preserved unless explicitly enabled."""
    val = os.environ.get("BAR_CACHE_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _revision_horizon_min() -> int:
    """M-RS2 fidelity guard. Bars within this many minutes of *now* are
    treated as UNSETTLED: Polygon revises recent bars (late prints; "TV
    revises closed 1Min within ~5 min" — project_tv_stability_2026-05-04),
    so they are ALWAYS re-fetched from Polygon REST and overwritten on read,
    regardless of what's cached. Bars older than the horizon are SETTLED →
    immutable → served from cache, never re-fetched. Default 45 min pads
    Kevin's ~15-min typical settle observation for safety; tune via env."""
    try:
        return max(0, int(os.environ.get("BAR_CACHE_REVISION_HORIZON_MIN", "45")))
    except ValueError:
        return 45


# =============================================================================
# Cache I/O
# =============================================================================

def _tf_seconds(timeframe: str) -> int:
    """Canonical TF string -> seconds. '10Sec'->10, '1Min'->60, '1Hour'->3600,
    '1Day'->86400. Unknown -> 60 (1-minute) as a safe default. Used to enforce
    the never-upsample invariant in cached_load_market_data."""
    import re
    m = re.match(r"^\s*(\d+)\s*(Sec|Min|Hour|Day|Week)", str(timeframe), re.I)
    if not m:
        return 60
    n = int(m.group(1))
    unit = m.group(2).lower()
    return n * {"sec": 1, "min": 60, "hour": 3600,
                "day": 86400, "week": 604800}[unit]


def _max_cached_ts(symbol: str, timeframe: str) -> Optional[datetime]:
    """Return the latest ts in the cache for (symbol, timeframe), or None
    if nothing is cached yet. Uses ORDER BY + LIMIT 1 — should be cheap
    given the composite primary key index."""
    return _edge_cached_ts(symbol, timeframe, desc=True)


def _min_cached_ts(symbol: str, timeframe: str) -> Optional[datetime]:
    """Return the earliest ts in the cache for (symbol, timeframe), or
    None. Used to detect missing historical-tail coverage so the cache
    can backfill when callers ask for a range that extends below
    `first_cached`."""
    return _edge_cached_ts(symbol, timeframe, desc=False)


def _edge_cached_ts(symbol: str, timeframe: str, *,
                    desc: bool) -> Optional[datetime]:
    from db import get_admin_client
    client = get_admin_client()
    try:
        r = client.table("bar_cache") \
            .select("ts") \
            .eq("symbol", symbol) \
            .eq("timeframe", timeframe) \
            .order("ts", desc=desc) \
            .limit(1) \
            .execute()
    except Exception as e:
        logger.warning("bar_cache: edge_ts(desc=%s) failed for %s/%s: %s",
                       desc, symbol, timeframe, e)
        return None
    if not r.data:
        return None
    ts_str = r.data[0]["ts"]
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _select_range(symbol: str, timeframe: str,
                  start: datetime, end: datetime) -> pd.DataFrame:
    """Read [start, end] inclusive from the cache. Returns a DataFrame
    with the same columns load_market_data produces (open, high, low,
    close, volume) and a UTC DatetimeIndex."""
    from db import get_admin_client
    client = get_admin_client()

    # Pagination. PostgREST's default cap is 1000 rows per request unless
    # the Supabase project's "Max Rows" setting (Project → Settings →
    # API → Max Rows) is bumped. Use the configured cap, autodetected by
    # observing the first response. To unlock the cache's full speed for
    # large queries (180-day 1Min charts), bump Max Rows to 50000+ in the
    # Supabase dashboard.
    PAGE = int(os.environ.get("BAR_CACHE_PAGE_SIZE", "1000"))
    offset = 0
    rows: list[dict] = []
    while True:
        try:
            r = client.table("bar_cache") \
                .select("ts,open,high,low,close,volume") \
                .eq("symbol", symbol) \
                .eq("timeframe", timeframe) \
                .gte("ts", start.isoformat()) \
                .lte("ts", end.isoformat()) \
                .order("ts") \
                .range(offset, offset + PAGE - 1) \
                .execute()
        except Exception as e:
            logger.warning("bar_cache: range select failed at offset %d: %s",
                           offset, e)
            break
        batch = r.data or []
        rows.extend(batch)
        # If the server returned fewer rows than asked, we're at the end
        # OR PostgREST capped us. Detect cap on first iteration: if PAGE
        # is bigger than configured max, the server returns max anyway.
        # Adapt PAGE to the actual returned size for subsequent iterations.
        if len(batch) < PAGE:
            if offset == 0 and len(batch) == 1000 and PAGE > 1000:
                # Configured cap is 1000; adapt and continue paging
                PAGE = 1000
                offset += PAGE
                continue
            break
        offset += PAGE

    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    # Parse ts → DatetimeIndex
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df.index.name = None  # match load_market_data convention
    return df[["open", "high", "low", "close", "volume"]]


def _bulk_upsert(symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Upsert all rows in df to the cache. Returns the row count submitted.

    Uses on_conflict='symbol,timeframe,ts' so re-fetched bars (e.g. when
    Polygon revises a recent bar) are silently overwritten — but the
    common case is INSERT NEW, since delta-fetch only requests bars
    after last_cached_ts.
    """
    if df is None or len(df) == 0:
        return 0
    from db import get_admin_client
    client = get_admin_client()

    # Build payload. Convert ts (DatetimeIndex) to ISO strings.
    payload = []
    for ts, row in df.iterrows():
        if not hasattr(ts, "isoformat"):
            continue
        payload.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "ts": ts.isoformat() if ts.tzinfo
                  else ts.tz_localize("UTC").isoformat(),
            "open": float(row.get("open", 0) or 0),
            "high": float(row.get("high", 0) or 0),
            "low": float(row.get("low", 0) or 0),
            "close": float(row.get("close", 0) or 0),
            "volume": float(row.get("volume", 0) or 0),
        })
    if not payload:
        return 0

    # Chunk to avoid request-size limits (Supabase default 1MB).
    # 1k rows × ~150 bytes = ~150KB per chunk — comfortable.
    CHUNK = 1000
    inserted = 0
    for i in range(0, len(payload), CHUNK):
        chunk = payload[i:i + CHUNK]
        try:
            client.table("bar_cache").upsert(
                chunk, on_conflict="symbol,timeframe,ts").execute()
            inserted += len(chunk)
        except Exception as e:
            logger.warning(
                "bar_cache: upsert failed at chunk offset %d "
                "(symbol=%s tf=%s rows=%d): %s",
                i, symbol, timeframe, len(chunk), e)
    return inserted


# =============================================================================
# Public API
# =============================================================================

def cached_load_market_data(
    symbol: str,
    days: int = 30,
    timeframe: str = "1Min",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: str = "RTH",
) -> Optional[pd.DataFrame]:
    """Drop-in cache wrapper around load_from_polygon for stocks.

    Reads the cache for [start, end], delta-fetches any gap from
    Polygon, upserts the gap into the cache, and returns the requested
    range with the session filter applied.

    Crypto / Alpaca / mock paths bypass this and route through the
    original load_market_data — caching them is out of scope for V1.

    Returns the same DataFrame shape as load_from_polygon (UTC index,
    OHLCV columns).
    """
    from data_loader import (
        is_polygon_configured, _filter_session, load_from_polygon,
    )

    # Bypass cache for non-Polygon paths or when disabled
    if not is_enabled() or not is_polygon_configured():
        return None  # caller falls through to legacy path

    # Resolve range
    if end_date:
        end_dt = end_date
    else:
        end_dt = datetime.now(timezone.utc)
    if start_date:
        start_dt = start_date
    else:
        start_dt = end_dt - timedelta(days=days)

    # Ensure both are UTC-aware
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    # M-RS2 Phase 2: choose which cache layer to read. CRITICAL INVARIANT: the
    # chosen layer must be FINER-OR-EQUAL to the requested TF — NEVER coarser, or
    # we'd have to UPSAMPLE (e.g. 1Min→10Sec), which produces sparse/garbage bars
    # and silently wrecks sub-minute backtests. (Fixed 2026-06-26; the prior
    # `else "1Min"` fallback upsampled every uncached sub-minute strategy —
    # 10Sec/15Sec dropped ~6x/~4x of their trades. 30Sec survived only because
    # TSLA 30Sec happened to be captured natively.)
    # There are exactly TWO native SOURCE layers — **1Sec** and **1Min** —
    # and EVERYTHING else is DERIVED by downsampling from one of them:
    #   - 1Sec / 1Min          → read the source layer directly.
    #   - sub-minute (<60s)     → downsample from native **1Sec** (validated
    #                             byte-identical to Polygon native 10/15/30Sec
    #                             INCLUDING volume, 06-12/16/22).
    #   - 1Min-or-coarser       → downsample from native **1Min** (NOT 1Sec —
    #                             Polygon's native 1Min volume diverges from
    #                             summed 1Sec: 822/944 bars, maxΔ 7300 sh).
    # We deliberately do NOT trust any natively-cached INTERMEDIATE layer (e.g. a
    # stray 30Sec): those are separately maintained and drift on late-trade
    # revisions (2026-06-26: an orphaned TSLA 30Sec layer was 1 bar stale vs
    # Polygon while the 1Sec-derived 30Sec matched exactly). One source of truth
    # per resolution = no drift. The chosen cache_tf is ALWAYS finer-or-equal to
    # the request — never coarser (upsampling 1Min→10Sec was the bug that wrecked
    # sub-minute backtests; the guard below enforces it).
    target_s = _tf_seconds(timeframe)
    if timeframe in ("1Sec", "1Min"):
        cache_tf = timeframe          # the only two native source layers
    elif target_s < 60:
        cache_tf = "1Sec"             # sub-minute → derive from native 1Sec
    else:
        cache_tf = "1Min"             # 1Min-or-coarser → derive from native 1Min
    # Hard guard: the source layer must be present + finer-or-equal (no upsample,
    # no reading a layer that isn't there) → else fall back to Polygon.
    if _max_cached_ts(symbol, cache_tf) is None or _tf_seconds(cache_tf) > target_s:
        return None

    # 1. Find cached coverage
    last_cached = _max_cached_ts(symbol, cache_tf)
    first_cached = _min_cached_ts(symbol, cache_tf)

    # 2. Determine fetch ranges. Two-sided gap detection:
    #    - Backfill historical tail when start_dt < first_cached
    #    - Forward-fill trailing edge when last_cached < end_dt
    # Skipping either side silently corrupts the result for callers asking
    # for a range that extends past current cache coverage.
    fetch_ranges: list[tuple[datetime, datetime]] = []

    if first_cached is None:
        # Cold start — fetch the full requested range
        fetch_ranges.append((start_dt, end_dt))
    else:
        # Historical-tail backfill
        if start_dt < first_cached:
            fetch_ranges.append(
                (start_dt, first_cached - timedelta(minutes=1)))
        # Trailing-edge forward-fill, but skip the network round-trip if
        # the cache is already fresher than 60s. The bar-builder won't
        # have a newer bar within the last minute, and chart loads
        # tolerate up to a minute of staleness on the most-recent bar
        # (live ticks come via Realtime WS separately).
        if last_cached is not None and last_cached < end_dt:
            gap_seconds = (end_dt - last_cached).total_seconds()
            if gap_seconds > 60:
                fetch_ranges.append(
                    (last_cached + timedelta(minutes=1), end_dt))

    # 2b. REVISION-HORIZON GUARD (M-RS2 fidelity — the core correctness fix).
    # The settled/unsettled split: any bar within `now - horizon` may still be
    # revised by Polygon (late prints), so ALWAYS re-fetch + overwrite that
    # window on read — even if it's already cached (the trailing-edge skip
    # above would otherwise serve a stale pre-revision bar). Settled bars
    # (older than the horizon) are never re-fetched. The upsert at step 3
    # (ON CONFLICT DO UPDATE) overwrites the refreshed tail. For reads of
    # purely-historical ranges (end_dt older than the horizon) this is inert.
    horizon_min = _revision_horizon_min()
    if horizon_min > 0:
        actual_now = datetime.now(timezone.utc)
        unsettled_start = actual_now - timedelta(minutes=horizon_min)
        if end_dt > unsettled_start:
            rev_from = max(start_dt, unsettled_start)
            if rev_from < end_dt:
                fetch_ranges.append((rev_from, end_dt))

    # 3. Execute fetches + upsert
    fetch_failed_cold = False
    for f_from, f_to in fetch_ranges:
        if f_from >= f_to:
            continue
        try:
            delta_df = load_from_polygon(
                symbol=symbol, days=0, timeframe=cache_tf,
                start_date=f_from, end_date=f_to,
                session="24/7",  # raw fetch — no filter, store all hours
            )
            if delta_df is not None and len(delta_df) > 0:
                inserted = _bulk_upsert(symbol, cache_tf, delta_df)
                logger.info(
                    "bar_cache: fetched %d %s/%s bars (range %s → %s)",
                    inserted, symbol, cache_tf,
                    f_from.isoformat(), f_to.isoformat())
        except Exception as e:
            logger.warning(
                "bar_cache: fetch failed for %s %s → %s: %s",
                symbol, f_from, f_to, e)
            if first_cached is None:
                # Cold cache + Polygon failed — caller falls through
                fetch_failed_cold = True

    if fetch_failed_cold:
        return None

    # 4. Range select from cache — direct Postgres (fast) when a DSN is
    # configured, else PostgREST. Both byte-identical (validated 2026-06-24).
    if direct_pg_available():
        df = read_bars(symbol, cache_tf, start_dt, end_dt)
        if df is None:  # no rows via direct PG
            df = _select_range(symbol, cache_tf, start_dt, end_dt)
    else:
        df = _select_range(symbol, cache_tf, start_dt, end_dt)
    if df is None or len(df) == 0:
        # Nothing cached, nothing fetched — return None to let caller
        # fall back to Polygon (matches load_market_data semantics)
        return None

    # 5. Session filter (matches load_from_polygon behaviour)
    if session and session != "24/7":
        df = _filter_session(df, session)

    if len(df) == 0:
        return None

    # 6. Resample if caller requested coarser TF
    if timeframe != cache_tf:
        try:
            from data_loader import resample_to_timeframe
            df = resample_to_timeframe(df, timeframe)
        except Exception as e:
            logger.warning(
                "bar_cache: resample %s → %s failed: %s",
                cache_tf, timeframe, e)
            return None

    return df if len(df) > 0 else None


# =============================================================================
# M-RS2 Phase 2 — fast read path (direct Postgres, not PostgREST)
# =============================================================================

def direct_pg_available() -> bool:
    """True if a direct Postgres DSN is configured for fast bulk reads."""
    return bool(os.environ.get("SUPABASE_CONNECTION_STRING"))


def read_bars(symbol: str, timeframe: str,
              start, end) -> Optional["pd.DataFrame"]:
    """Read cached native bars for (symbol, timeframe) in [start, end] over a
    DIRECT Postgres connection (psycopg + SUPABASE_CONNECTION_STRING) — the
    fast read path validated 2026-06-24 (binary protocol, one query, no
    PostgREST/JSON). Returns a DataFrame shaped like load_from_polygon (UTC
    DatetimeIndex, columns open/high/low/close/volume), or None if no DSN /
    no rows. Raw cursor + bulk DataFrame build (not pandas.read_sql, which is
    slow over psycopg3).

    NOTE: returns RAW (all-hours) rows; the caller applies any session filter,
    exactly like _select_range / load_from_polygon(session-on-read).
    """
    dsn = os.environ.get("SUPABASE_CONNECTION_STRING")
    if not dsn:
        return None
    try:
        import psycopg
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        logger.warning("bar_cache.read_bars: psycopg/pandas unavailable: %s", e)
        return None
    if hasattr(start, "isoformat"):
        start = start.isoformat()
    if hasattr(end, "isoformat"):
        end = end.isoformat()
    sql = ("select ts, open, high, low, close, volume from bar_cache "
           "where symbol=%s and timeframe=%s and ts between %s and %s "
           "order by ts")
    try:
        # prepare_threshold=None disables psycopg3 auto-prepared-statements so
        # this works on Supabase's TRANSACTION pooler (port 6543), which doesn't
        # support prepared statements. The transaction pooler multiplexes many
        # short per-call connections (read_bars opens one per call) far better
        # than the SESSION pooler (5432), whose per-client connection cap we were
        # exhausting → "Server disconnected" on both direct-PG AND PostgREST
        # (2026-06-26). Harmless on 5432 too.
        with psycopg.connect(dsn, connect_timeout=15,
                             prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol, timeframe, start, end))
                rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001
        logger.warning("bar_cache.read_bars: direct-PG read failed %s/%s: %s",
                       symbol, timeframe, e)
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df.index.name = None
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =============================================================================
# M-RS2 Phase 1 — proactive supply capture (native bars per (symbol, TF))
# =============================================================================

# Chunk sizing per resolution to bound memory + Polygon pagination on a single
# fetch. 1Sec is ~23k bars/RTH-day (~5.9M/yr) so it must be chunked tight.
_BACKFILL_CHUNK_DAYS = {
    "1Sec": 7, "5Sec": 14, "10Sec": 21, "15Sec": 30, "30Sec": 45,
    "1Min": 120, "1Hour": 365, "1Day": 365,
}


def backfill_symbol(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    *,
    chunk_days: Optional[int] = None,
) -> int:
    """Proactively fetch + cache NATIVE Polygon bars for [start_date, end_date]
    under their own `timeframe` key (M-RS2 Phase 1 — the data supply). Stores
    RAW (session='24/7', all hours included); the session filter is applied on
    read. Chunked so high-volume resolutions (1Sec) don't blow memory or hit a
    single huge paginated fetch. Idempotent via the upsert (re-running overwrites
    overlapping bars). Returns the total rows upserted.

    This is the PROACTIVE primer — distinct from `cached_load_market_data`'s
    lazy fetch-on-read. Caches native data only; deriving/serving other TFs from
    it (Phase 2) is matched 1:1 to the current backtest per-feature, validated.
    """
    from data_loader import load_from_polygon, is_polygon_configured
    if not is_polygon_configured():
        logger.warning("bar_cache.backfill_symbol: Polygon not configured")
        return 0
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Coarse layers (>=1Hour) are DERIVED from the cached 1Min (split-safe) —
    # NOT fetched native (native daily has split-adjustment bugs). See
    # materialize_derived. Requires the 1Min layer to be backfilled first.
    if _tf_seconds(timeframe) >= 3600:
        return materialize_derived(symbol, timeframe, start_date, end_date)

    step = chunk_days or _BACKFILL_CHUNK_DAYS.get(timeframe, 30)
    total = 0
    cur = start_date
    while cur < end_date:
        chunk_end = min(cur + timedelta(days=step), end_date)
        try:
            df = load_from_polygon(
                symbol=symbol, days=0, timeframe=timeframe,
                start_date=cur, end_date=chunk_end, session="24/7")
        except Exception as e:
            logger.warning(
                "bar_cache.backfill_symbol: fetch failed %s/%s [%s..%s]: %s",
                symbol, timeframe, cur, chunk_end, e)
            cur = chunk_end
            continue
        if df is not None and len(df) > 0:
            n = _bulk_upsert(symbol, timeframe, df)
            total += n
            logger.info(
                "bar_cache.backfill_symbol: %s/%s [%s..%s] +%d rows",
                symbol, timeframe, cur.date(), chunk_end.date(), n)
        cur = chunk_end
    return total


def maintain_symbol(symbol: str, timeframe: str) -> int:
    """Keep a captured (symbol, timeframe) layer up to date (M-RS2 Phase 1).
    In ONE fetch it both (a) extends the trailing edge from the last cached bar
    to now, and (b) re-fetches the unsettled window (`now - revision_horizon`)
    to overwrite any bars Polygon has since revised. Fetches from
    `min(last_cached, unsettled_start)` → now so both are covered without a
    second round-trip. No-op (returns 0) if nothing is cached yet — seed with
    `backfill_symbol` first. Returns rows upserted."""
    from data_loader import load_from_polygon, is_polygon_configured
    if not is_polygon_configured():
        return 0
    last = _max_cached_ts(symbol, timeframe)
    if last is None:
        return 0
    now = datetime.now(timezone.utc)
    # Coarse layers: re-materialize a recent window from the (current) 1Min —
    # covers the forming coarse bar + late-revision days (daily bars revise all
    # day). Cheap (a few days of 1Min). The 1Min layer's own maintain keeps it
    # fresh, so re-resampling the recent slice keeps the coarse layer current.
    if _tf_seconds(timeframe) >= 3600:
        return materialize_derived(symbol, timeframe,
                                   min(last, now - timedelta(days=3)), now)
    unsettled_start = now - timedelta(minutes=_revision_horizon_min())
    fetch_from = min(last, unsettled_start)
    try:
        df = load_from_polygon(
            symbol=symbol, days=0, timeframe=timeframe,
            start_date=fetch_from, end_date=now, session="24/7")
    except Exception as e:
        logger.warning("bar_cache.maintain_symbol: fetch failed %s/%s: %s",
                       symbol, timeframe, e)
        return 0
    if df is None or len(df) == 0:
        return 0
    n = _bulk_upsert(symbol, timeframe, df)
    logger.info("bar_cache.maintain_symbol: %s/%s extended from %s, +%d rows",
                symbol, timeframe, fetch_from.isoformat(), n)
    return n


def materialize_derived(symbol: str, timeframe: str,
                        start_date: datetime, end_date: datetime) -> int:
    """Build a COARSE (>=1Hour) cache layer by resampling the cached **1Min** —
    NOT by fetching native Polygon. Native Polygon daily/hourly carries
    split-adjustment bugs (CLAUDE.md: native daily returned pre-split $4,700
    NVDA), which is exactly why the backtest builds coarse TFs from 1Min. Doing
    the same here keeps the cache byte-consistent with the backtest's own coarse
    construction (cache == backtest), and lets coarse-TF live warmup / backtest
    loads read ~246 daily rows instead of re-squishing ~140k 1-minute bars.

    Reads the cached 1Min for [start,end] (must already be backfilled), resamples
    to `timeframe`, upserts under the target TF key. Returns rows upserted."""
    from data_loader import resample_to_timeframe
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    df1 = read_bars(symbol, "1Min", start_date, end_date)
    if df1 is None or len(df1) == 0:
        logger.warning("bar_cache.materialize_derived: no cached 1Min for %s "
                       "[%s..%s] — backfill the 1Min layer first",
                       symbol, start_date.date(), end_date.date())
        return 0
    try:
        out = resample_to_timeframe(
            df1[["open", "high", "low", "close", "volume"]].copy(), timeframe)
    except Exception as e:  # noqa: BLE001
        logger.warning("bar_cache.materialize_derived: resample %s %s<-1Min "
                       "failed: %s", symbol, timeframe, e)
        return 0
    if out is None or len(out) == 0:
        return 0
    n = _bulk_upsert(symbol, timeframe, out)
    logger.info("bar_cache.materialize_derived: %s/%s <- 1Min [%s..%s]: "
                "%d bars upserted (split-safe resample)",
                symbol, timeframe, start_date.date(), end_date.date(), n)
    return n


# =============================================================================
# M-RS2 Phase 1 — capture config (admin control surface) + maintain-all loop
# =============================================================================

def get_capture_targets(enabled_only: bool = True) -> list[dict]:
    """Return the configured (symbol, timeframe) capture targets from
    bar_cache_config. Drives the maintenance loop + the admin page."""
    from db import get_admin_client
    client = get_admin_client()
    try:
        q = client.table("bar_cache_config").select("*")
        if enabled_only:
            q = q.eq("enabled", True)
        return (q.order("symbol").order("timeframe").execute().data) or []
    except Exception as e:
        logger.warning("bar_cache.get_capture_targets failed: %s", e)
        return []


def set_capture_target(symbol: str, timeframe: str, *, enabled: bool = True,
                       capture_days: Optional[int] = None) -> bool:
    """Upsert a capture target (admin page CRUD). Does NOT trigger a backfill —
    the caller runs backfill_symbol once, then the maintain loop keeps it
    current. Returns True on success."""
    from db import get_admin_client
    client = get_admin_client()
    row = {"symbol": symbol, "timeframe": timeframe, "enabled": enabled}
    if capture_days is not None:
        row["capture_days"] = capture_days
    try:
        client.table("bar_cache_config").upsert(
            row, on_conflict="symbol,timeframe").execute()
        return True
    except Exception as e:
        logger.warning("bar_cache.set_capture_target failed %s/%s: %s",
                       symbol, timeframe, e)
        return False


def maintain_all_enabled() -> dict:
    """Keep every enabled capture target current (cron entrypoint). Calls
    maintain_symbol() per target, stamps last_maintained_at. Returns a summary
    {targets, rows, errors}."""
    from db import get_admin_client
    client = get_admin_client()
    targets = get_capture_targets(enabled_only=True)
    rows_total = 0
    errors = 0
    for t in targets:
        sym, tf = t.get("symbol"), t.get("timeframe")
        try:
            rows_total += maintain_symbol(sym, tf)
            client.table("bar_cache_config").update(
                {"last_maintained_at": datetime.now(timezone.utc).isoformat()}
            ).eq("symbol", sym).eq("timeframe", tf).execute()
        except Exception as e:
            errors += 1
            logger.warning("bar_cache.maintain_all_enabled %s/%s: %s",
                           sym, tf, e)
    return {"targets": len(targets), "rows": rows_total, "errors": errors}


# =============================================================================
# Diagnostics
# =============================================================================

def cache_stats() -> dict:
    """Return cache size + per-symbol counts. Useful for ops dashboards."""
    from db import get_admin_client
    client = get_admin_client()
    try:
        r = client.table("bar_cache").select(
            "symbol,timeframe", count="exact").execute()
        total = r.count
    except Exception as e:
        return {"error": str(e)}
    # Per-symbol counts would need a GROUP BY, which postgrest doesn't
    # expose directly. Use the SQL editor for ad-hoc. V1 just returns total.
    return {"total_rows": total, "enabled": is_enabled()}
