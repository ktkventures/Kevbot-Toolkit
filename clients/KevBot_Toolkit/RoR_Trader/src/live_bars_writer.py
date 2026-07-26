"""Fire-and-forget writer for the "Live Bars" cache (live_bars, M8.7).

================================================================================
THIS IS THE *LIVE BARS* CACHE — one of two bar caches. Do not confuse it with
"REST Bars" (`bar_cache` table, written by bar_cache.py).

  - Live Bars (this, `live_bars`): what the WebSocket saw at decision time
    (source='ws_agg'). IMMUTABLE — a seen bar is NEVER post-corrected with
    revised REST. Its value (replay, phantom/mistrade WS-visibility forensics)
    depends on it staying *what we saw*. REST may gap-fill *missing* minutes
    only (live_bars_rest_backfill.py, insert-only, source='rest_backfill').
  - REST Bars (`bar_cache`): Polygon REST, revisable to track Polygon; backtest
    speed/fidelity.

RED LINE: never point the REST revision/refresh (overwrite-to-match-Polygon)
logic at `live_bars`. Overwriting a seen bar here destroys the forensic record.
Canonical reference: Two_Bar_Caches_DEFINITIONS.md
================================================================================

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


def _reconcile_seed_sync(symbol: str, tf_seconds: int, rows: list,
                         start_iso: str, end_iso: str) -> int:
    """Insert seed bars missing from live_bars in [start_iso, end_iso).

    `rows` = [(bar_start_iso, o, h, l, c, v)] candidates from the seeded
    (warmup REST) history. Existence-checked + ignore_duplicates so a
    real ws/ws_agg row is NEVER overwritten — this only fills holes the
    live writer couldn't cover (worker restart/deploy windows). Returns
    number of rows inserted.
    """
    try:
        from db import get_admin_client
        client = get_admin_client()
        existing = set()
        page, off = 1000, 0
        while True:
            r = (client.table("live_bars").select("bar_start")
                 .eq("symbol", symbol)
                 .eq("timeframe_seconds", int(tf_seconds))
                 .gte("bar_start", start_iso).lt("bar_start", end_iso)
                 .order("bar_start").range(off, off + page - 1)
                 .execute()).data or []
            existing.update(x["bar_start"][:19] for x in r)
            if len(r) < page:
                break
            off += page
        payload = []
        for ts_iso, o, h, l, c, v in rows:
            if ts_iso[:19] in existing:
                continue
            payload.append({
                "symbol": symbol,
                "timeframe_seconds": int(tf_seconds),
                "bar_start": ts_iso,
                "open": float(o), "high": float(h),
                "low": float(l), "close": float(c),
                "volume": float(v),
                "source": "warmup_seed",
            })
        if not payload:
            return 0
        # ignore_duplicates → ON CONFLICT DO NOTHING: a live ws/ws_agg
        # row racing in between our existence check and this write wins
        # (same pattern as live_bars_rest_backfill._write).
        client.table("live_bars").upsert(
            payload,
            on_conflict="symbol,timeframe_seconds,bar_start",
            ignore_duplicates=True,
        ).execute()
        logger.info(
            "live_bars: warmup_seed reconciled %d missing bar(s) "
            "sym=%s tf=%ss window=[%s..%s] (deploy/restart cache hole)",
            len(payload), symbol, tf_seconds,
            start_iso[:19], end_iso[:19])
        return len(payload)
    except Exception as e:
        logger.warning(
            "live_bars: warmup_seed reconcile failed sym=%s tf=%ss: %s",
            symbol, tf_seconds, e)
        return 0


def reconcile_seeded_history(symbol: str, tf_seconds: int, df,
                             lookback_seconds: int = 7200) -> None:
    """Fire-and-forget: backfill live_bars with seeded (warmup REST)
    bars missing from the cache in the recent window (2026-06-11).

    Worker restarts (every deploy) leave a ~1-3 min hole in live_bars —
    the engine itself is whole (warmup seeds its in-memory history from
    REST, including the downtime window) but the cache, and therefore
    the alert lens / Per-Bar Parity Drift ribbon, shows a gap and the
    lens-side indicator recompute drifts around it. The engine genuinely
    CONSUMED these REST bars (its indicator state is built from them),
    so writing them as source='warmup_seed' (in ENGINE_CONSUMED_SOURCES)
    is faithful, not cosmetic — unlike 'rest_backfill', which the engine
    never consumes and stays excluded. Each startup also heals the
    PREVIOUS deploy's hole within the lookback. No-op when the writer is
    disabled.
    """
    if not is_enabled():
        return
    try:
        if df is None or len(df) == 0:
            return
        from datetime import datetime as _dt, timedelta as _td
        now = _dt.now(timezone.utc)
        start = now - _td(seconds=lookback_seconds)
        # Exclude the forming period: only bars fully closed by now.
        end = now - _td(seconds=int(tf_seconds))
        rows = []
        for ts, row in df.iterrows():
            t = ts
            if getattr(t, "tzinfo", None) is None:
                try:
                    t = t.tz_localize("UTC")
                except Exception:
                    continue
            py = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
            if py < start or py >= end:
                continue
            rows.append((
                _ensure_iso_utc(t),
                row.get("open", 0), row.get("high", 0),
                row.get("low", 0), row.get("close", 0),
                row.get("volume", 0),
            ))
        if not rows:
            return
        _get_executor().submit(
            _reconcile_seed_sync, symbol, int(tf_seconds), rows,
            start.isoformat(), end.isoformat())
    except Exception as e:
        logger.warning(
            "live_bars: reconcile submit failed sym=%s tf=%ss: %s",
            symbol, tf_seconds, e)


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
