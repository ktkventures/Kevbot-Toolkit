"""Fire-and-forget writer for the live_bars cache (Milestone 8.7).

The worker calls write_bar() each time a BarBuilder finalizes a bar
(primary AM bars from on_polygon_bar, sub-minute aggregations from
on_second_bar, and secondary-TF aggregations). Writes happen on a
small background thread pool so Supabase latency never blocks the
bar-processing path.

Tonight's scope: write-only. The read path (data_loader integration)
ships in a later phase. See docs/Plan_Live_Bar_Cache_2026-04-30.md.

Feature flag: LIVE_BAR_CACHE_WRITE_ENABLED (default off).
- Flip on in the Railway worker env to start recording
- Flip off to stop recording with zero code change

Schema: see src/migrations/live_bars_table.sql
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    val = os.environ.get("LIVE_BAR_CACHE_WRITE_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# Small pool — Supabase round-trips are ~30-100ms; one or two workers
# is plenty for ~50-200 writes/min during RTH. Lazy init so we never
# build the pool when the flag is off.
_executor: Optional[ThreadPoolExecutor] = None
_warned_disabled = False


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="live_bars_writer")
    return _executor


def _ensure_iso_utc(ts) -> str:
    if isinstance(ts, str):
        return ts
    if hasattr(ts, "isoformat"):
        if getattr(ts, "tzinfo", None) is None:
            try:
                ts = ts.tz_localize("UTC")
            except (AttributeError, TypeError):
                ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return str(ts)


def _write_bar_sync(symbol: str, tf_seconds: int, bar_dict: dict,
                    source: str) -> None:
    """Insert one row into live_bars. Idempotent via PK on conflict."""
    try:
        from db import get_admin_client
    except Exception as e:
        logger.warning("live_bars: db import failed: %s", e)
        return

    bar_start = bar_dict.get("timestamp")
    if bar_start is None:
        return
    payload = {
        "symbol": symbol,
        "timeframe_seconds": int(tf_seconds),
        "bar_start": _ensure_iso_utc(bar_start),
        "open": float(bar_dict.get("open", 0) or 0),
        "high": float(bar_dict.get("high", 0) or 0),
        "low": float(bar_dict.get("low", 0) or 0),
        "close": float(bar_dict.get("close", 0) or 0),
        "volume": float(bar_dict.get("volume", 0) or 0),
        "source": source,
    }
    tc = bar_dict.get("trade_count")
    if tc is not None:
        try:
            payload["trade_count"] = int(tc)
        except (ValueError, TypeError):
            pass
    vw = bar_dict.get("vwap")
    if vw is not None:
        try:
            payload["vwap"] = float(vw)
        except (ValueError, TypeError):
            pass

    try:
        client = get_admin_client()
        # upsert with ignore_duplicates so worker rebroadcasts of a
        # corrected bar (Polygon does this within the 15-min FINRA
        # window) overwrite cleanly. on_conflict matches the PK.
        client.table("live_bars").upsert(
            payload,
            on_conflict="symbol,timeframe_seconds,bar_start",
        ).execute()
    except Exception as e:
        # Never let a Supabase hiccup crash the worker. Log at warning
        # so it shows up in Railway logs but stays out of error budget.
        logger.warning(
            "live_bars: write failed sym=%s tf=%ss bar=%s: %s",
            symbol, tf_seconds, payload.get("bar_start"), e)


def write_bar(symbol: str, tf_seconds: int, bar_dict: dict,
              source: str = "ws") -> None:
    """Submit a bar write to the background pool. No-op when disabled."""
    global _warned_disabled
    if not is_enabled():
        if not _warned_disabled:
            logger.info(
                "live_bars: writer disabled "
                "(LIVE_BAR_CACHE_WRITE_ENABLED unset)")
            _warned_disabled = True
        return
    if not symbol or not tf_seconds or not bar_dict:
        return
    try:
        _get_executor().submit(
            _write_bar_sync, symbol, tf_seconds, bar_dict, source)
    except Exception as e:
        # Pool submission shouldn't fail, but if it does we don't
        # want to take down the worker.
        logger.warning("live_bars: submit failed: %s", e)


def shutdown(wait: bool = True) -> None:
    """Drain the pool on worker shutdown so in-flight writes finish."""
    global _executor
    if _executor is None:
        return
    try:
        _executor.shutdown(wait=wait)
    except Exception as e:
        logger.warning("live_bars: shutdown error: %s", e)
    _executor = None
