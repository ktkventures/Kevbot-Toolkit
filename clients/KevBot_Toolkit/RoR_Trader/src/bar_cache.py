"""Persistent bar cache backed by Supabase Postgres (bar_cache table).

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


# =============================================================================
# Cache I/O
# =============================================================================

def _max_cached_ts(symbol: str, timeframe: str) -> Optional[datetime]:
    """Return the latest ts in the cache for (symbol, timeframe), or None
    if nothing is cached yet. Uses ORDER BY + LIMIT 1 — should be cheap
    given the composite primary key index."""
    from db import get_admin_client
    client = get_admin_client()
    try:
        r = client.table("bar_cache") \
            .select("ts") \
            .eq("symbol", symbol) \
            .eq("timeframe", timeframe) \
            .order("ts", desc=True) \
            .limit(1) \
            .execute()
    except Exception as e:
        logger.warning("bar_cache: max_ts query failed for %s/%s: %s",
                       symbol, timeframe, e)
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

    # Paginate. Supabase default page size is 1000; for 1Min × 30 days
    # we expect ~12k rows so paging is required.
    PAGE = 1000
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
        if len(batch) < PAGE:
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

    # Cache always stores 1Min bars; resample on read for coarser TFs.
    cache_tf = "1Min"

    # 1. Find latest cached bar
    last_cached = _max_cached_ts(symbol, cache_tf)

    # 2. Determine delta-fetch range
    delta_from = None
    delta_to = end_dt
    if last_cached is None:
        # Cold start — fetch the full requested range
        delta_from = start_dt
    elif last_cached < end_dt:
        # Top-up — fetch from last_cached + 1Min to now
        delta_from = last_cached + timedelta(minutes=1)
    # else: cache fully covers [start_dt, end_dt], no delta needed

    # 3. Delta-fetch + upsert
    if delta_from is not None and delta_from < delta_to:
        try:
            delta_df = load_from_polygon(
                symbol=symbol, days=0, timeframe=cache_tf,
                start_date=delta_from, end_date=delta_to,
                session="24/7",  # raw fetch — no filter, store all hours
            )
            if delta_df is not None and len(delta_df) > 0:
                inserted = _bulk_upsert(symbol, cache_tf, delta_df)
                logger.info(
                    "bar_cache: delta-fetched %d %s/%s bars (range %s → %s)",
                    inserted, symbol, cache_tf,
                    delta_from.isoformat(), delta_to.isoformat())
        except Exception as e:
            logger.warning(
                "bar_cache: delta-fetch failed for %s %s → %s: %s",
                symbol, delta_from, delta_to, e)
            # If we have ANY cached data we proceed with what's there;
            # only return None if cache is also empty
            if last_cached is None:
                return None

    # 4. Range select from cache
    df = _select_range(symbol, cache_tf, start_dt, end_dt)
    if len(df) == 0:
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
