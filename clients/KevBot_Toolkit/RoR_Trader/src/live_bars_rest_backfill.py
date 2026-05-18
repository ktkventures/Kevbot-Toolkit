"""REST backfill for live_bars cache gaps (Stage B, 2026-05-18).

When the ws_agg path misses a 1-minute bar (a per-second WebSocket gap),
that minute gets no live_bars row at all. This module fills genuine gaps
with Polygon REST canonical 1Min values, written with
``source='rest_backfill'``.

Cosmetic only: rest_backfill rows exist for charts/backtests — the live
engine treats them as consume-only and never fires alerts on them.

Safety invariants:
  - Only INSERTs minutes absent from live_bars (existence-checked) and
    writes with ``ignore_duplicates=True`` (ON CONFLICT DO NOTHING), so an
    existing ws_agg row is never overwritten — its close / first_* stay
    intact. Do NOT route through live_bars_writer.write_bar: its upsert
    lacks ignore_duplicates and would clobber the row.
  - Only fills minutes older than LAG_MINUTES, so it never races the live
    ws_agg writer (which locks a bar ~1s after minute close).
  - Env-gated (LIVE_BARS_BACKFILL_ENABLED, default OFF).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("live_bars_rest_backfill")

TF_SECONDS = 60  # 1Min canonical — >=60s and sub-minute bars are derived
_PAGE = 1000


def is_enabled() -> bool:
    val = os.environ.get("LIVE_BARS_BACKFILL_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def interval_seconds() -> int:
    return int(os.environ.get("LIVE_BARS_BACKFILL_INTERVAL_SECONDS", "900"))


def _window_hours() -> int:
    return int(os.environ.get("LIVE_BARS_BACKFILL_WINDOW_HOURS", "6"))


def _lag_minutes() -> int:
    return int(os.environ.get("LIVE_BARS_BACKFILL_LAG_MINUTES", "5"))


def _epoch(ts) -> int:
    """Normalize an ISO string or datetime/Timestamp to epoch seconds (UTC)."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return int(ts.timestamp())


def _active_symbols(sb, window_start_iso: str) -> list[str]:
    """Stock symbols with recent tf=60 live_bars rows — the cached set.

    Crypto (symbols containing '/') is excluded — backfill is stocks-only.
    """
    rows = (sb.table("live_bars")
            .select("symbol")
            .eq("timeframe_seconds", TF_SECONDS)
            .gte("bar_start", window_start_iso)
            .limit(20000)
            .execute().data or [])
    return sorted({
        r["symbol"] for r in rows
        if r.get("symbol") and "/" not in r["symbol"]
    })


def _existing_bar_starts(sb, symbol: str, window_start_iso: str,
                         cutoff_iso: str) -> set[int]:
    """Epoch-second set of bar_starts already in live_bars for (symbol, 60)."""
    out: set[int] = set()
    offset = 0
    while True:
        page = (sb.table("live_bars")
                .select("bar_start")
                .eq("symbol", symbol)
                .eq("timeframe_seconds", TF_SECONDS)
                .gte("bar_start", window_start_iso)
                .lte("bar_start", cutoff_iso)
                .order("bar_start")
                .range(offset, offset + _PAGE - 1)
                .execute().data or [])
        out.update(_epoch(r["bar_start"]) for r in page)
        if len(page) < _PAGE:
            return out
        offset += _PAGE


def _backfill_symbol(sb, symbol: str, window_start: datetime,
                     cutoff: datetime) -> int:
    """Fill missing tf=60 minutes for one symbol. Returns rows inserted."""
    from data_loader import load_from_polygon

    window_start_iso = window_start.isoformat()
    cutoff_iso = cutoff.isoformat()
    window_start_epoch = int(window_start.timestamp())
    cutoff_epoch = int(cutoff.timestamp())

    existing = _existing_bar_starts(sb, symbol, window_start_iso, cutoff_iso)

    # Polygon REST fetches whole days; we filter to the window below.
    df = load_from_polygon(
        symbol, timeframe="1Min",
        start_date=window_start, end_date=cutoff, session="RTH")
    if df is None or len(df) == 0:
        return 0

    payloads = []
    for ts, row in df.iterrows():
        sec = _epoch(ts)
        if sec < window_start_epoch or sec > cutoff_epoch:
            continue  # outside the rolling window / too recent
        if sec in existing:
            continue  # already cached — never overwrite a ws_agg row
        payloads.append({
            "symbol": symbol,
            "timeframe_seconds": TF_SECONDS,
            "bar_start": datetime.fromtimestamp(
                sec, tz=timezone.utc).isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0) or 0),
            "source": "rest_backfill",
        })
    if not payloads:
        return 0

    # ignore_duplicates=True → ON CONFLICT DO NOTHING. A ws_agg row that
    # appeared between the existence check and this write is left untouched.
    sb.table("live_bars").upsert(
        payloads,
        on_conflict="symbol,timeframe_seconds,bar_start",
        ignore_duplicates=True,
    ).execute()
    logger.info("rest_backfill: %s — filled %d gap bar(s)",
                symbol, len(payloads))
    return len(payloads)


def run_backfill_cycle() -> None:
    """One backfill pass over all cached stock symbols.

    Safe no-op when the feature flag is off.
    """
    if not is_enabled():
        return
    try:
        from db import get_admin_client
        sb = get_admin_client()
    except Exception as e:
        logger.warning("rest_backfill: db unavailable — %s", e)
        return

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=_lag_minutes())
    window_start = now - timedelta(hours=_window_hours())
    window_start_iso = window_start.isoformat()

    symbols = _active_symbols(sb, window_start_iso)
    if not symbols:
        logger.debug("rest_backfill: no cached symbols in window")
        return

    total = 0
    for symbol in symbols:
        try:
            total += _backfill_symbol(sb, symbol, window_start, cutoff)
        except Exception as e:
            logger.warning("rest_backfill: %s failed — %s", symbol, e)
    logger.info(
        "rest_backfill cycle done — %d symbol(s), %d gap bar(s) filled",
        len(symbols), total)
