"""REST backfill for live_bars cache gaps (Stage B, 2026-05-18).

When the ws_agg path misses a 1-minute bar (a per-second WebSocket gap),
that minute gets no live_bars row. This module fills genuine gaps with
Polygon REST canonical values, written with ``source='rest_backfill'``.

It fills two layers:
  1. 1Min (tf=60) gaps — directly from Polygon REST 1Min aggregates.
  2. Derived ≥120s gaps (5Min, 15Min, …) — those bars are built live by
     the ws_agg fan-out from 1Min closes, so a 1Min gap also leaves a
     hole in the ≥60s bar built from it, and backfilling the 1Min row
     alone does NOT rebuild the ≥60s bar. After step 1 makes the 1Min
     series whole, step 2 rolls it up to each ≥60s timeframe the engine
     already builds for the symbol and fills the missing ≥60s buckets.

Cosmetic only: rest_backfill rows exist for charts/backtests — the live
engine treats them as consume-only and never fires alerts on them.

Safety invariants:
  - Only INSERTs minutes/buckets absent from live_bars (existence-checked)
    and writes with ``ignore_duplicates=True`` (ON CONFLICT DO NOTHING),
    so an existing ws_agg row is never overwritten. Do NOT route through
    live_bars_writer.write_bar — its upsert lacks ignore_duplicates.
  - Only fills periods older than LAG_MINUTES, so it never races the live
    writer.
  - A ≥60s bucket is filled only when ALL its constituent 1Min bars are
    present — never a partial roll-up.
  - Env-gated (LIVE_BARS_BACKFILL_ENABLED, default OFF).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("live_bars_rest_backfill")

TF_SECONDS = 60  # 1Min canonical — ≥120s bars are rolled up from it
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


def _fetch_bars(sb, symbol: str, tf: int, w0_iso: str,
                w1_iso: str) -> dict[int, dict]:
    """epoch-second → OHLCV dict for (symbol, tf) rows within the window."""
    out: dict[int, dict] = {}
    offset = 0
    while True:
        page = (sb.table("live_bars")
                .select("bar_start, open, high, low, close, volume")
                .eq("symbol", symbol).eq("timeframe_seconds", tf)
                .gte("bar_start", w0_iso).lte("bar_start", w1_iso)
                .order("bar_start").range(offset, offset + _PAGE - 1)
                .execute().data or [])
        for r in page:
            out[_epoch(r["bar_start"])] = {
                "open": r["open"], "high": r["high"], "low": r["low"],
                "close": r["close"], "volume": r["volume"],
            }
        if len(page) < _PAGE:
            return out
        offset += _PAGE


def _derived_tfs(sb, symbol: str, w0_iso: str) -> list[int]:
    """≥120s timeframes the engine already builds for this symbol — the
    only ones with derived ≥60s bars worth gap-filling."""
    rows = (sb.table("live_bars").select("timeframe_seconds")
            .eq("symbol", symbol).gt("timeframe_seconds", TF_SECONDS)
            .gte("bar_start", w0_iso).limit(20000).execute().data or [])
    return sorted({int(r["timeframe_seconds"]) for r in rows})


def _rollup(minutes: dict[int, dict], tf: int) -> dict[int, dict]:
    """Roll a 1Min series up to `tf`. Emits ONLY buckets that have their
    full complement of 1Min bars — never a partial roll-up."""
    need = tf // 60
    buckets: dict[int, list] = {}
    for sec in sorted(minutes):
        buckets.setdefault((sec // tf) * tf, []).append(minutes[sec])
    out: dict[int, dict] = {}
    for b, bars in buckets.items():
        if len(bars) != need:
            continue
        out[b] = {
            "open": bars[0]["open"],
            "high": max(x["high"] for x in bars),
            "low": min(x["low"] for x in bars),
            "close": bars[-1]["close"],
            "volume": sum((x["volume"] or 0) for x in bars),
        }
    return out


def _payload(symbol: str, tf: int, epoch_sec: int, bar: dict) -> dict:
    return {
        "symbol": symbol,
        "timeframe_seconds": tf,
        "bar_start": datetime.fromtimestamp(
            epoch_sec, tz=timezone.utc).isoformat(),
        "open": float(bar["open"]), "high": float(bar["high"]),
        "low": float(bar["low"]), "close": float(bar["close"]),
        "volume": float(bar.get("volume", 0) or 0),
        "source": "rest_backfill",
    }


def _write(sb, payloads: list[dict]) -> None:
    """Insert backfill rows; ignore_duplicates → ON CONFLICT DO NOTHING,
    so an existing ws_agg row is never overwritten."""
    sb.table("live_bars").upsert(
        payloads,
        on_conflict="symbol,timeframe_seconds,bar_start",
        ignore_duplicates=True,
    ).execute()


def _backfill_symbol(sb, symbol: str, window_start: datetime,
                     cutoff: datetime) -> tuple[int, int]:
    """Fill 1Min then derived ≥120s gaps. Returns (1min_filled, derived_filled)."""
    from data_loader import load_from_polygon

    w0_iso, w1_iso = window_start.isoformat(), cutoff.isoformat()
    w0, wn = int(window_start.timestamp()), int(cutoff.timestamp())

    # --- Layer 1: 1Min gap fill ---
    existing_1m = _fetch_bars(sb, symbol, TF_SECONDS, w0_iso, w1_iso)
    df = load_from_polygon(
        symbol, timeframe="1Min",
        start_date=window_start, end_date=cutoff, session="RTH")
    rest_1m: dict[int, dict] = {}
    if df is not None and len(df) > 0:
        for ts, row in df.iterrows():
            sec = _epoch(ts)
            if w0 <= sec <= wn:
                rest_1m[sec] = {
                    "open": float(row["open"]), "high": float(row["high"]),
                    "low": float(row["low"]), "close": float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                }
    gap_payloads = [_payload(symbol, TF_SECONDS, s, rest_1m[s])
                    for s in sorted(rest_1m) if s not in existing_1m]
    if gap_payloads:
        _write(sb, gap_payloads)
        logger.info("rest_backfill: %s — filled %d 1Min gap(s)",
                    symbol, len(gap_payloads))

    # Complete 1Min series = existing (ws_agg / prior backfill) ∪ REST fills.
    complete_1m = dict(existing_1m)
    for s, bar in rest_1m.items():
        complete_1m.setdefault(s, bar)

    # --- Layer 2: derived ≥120s timeframes, rolled up from the now-whole 1Min ---
    derived_filled = 0
    for tf in _derived_tfs(sb, symbol, w0_iso):
        existing_tf = set(_fetch_bars(sb, symbol, tf, w0_iso, w1_iso))
        rolled = _rollup(complete_1m, tf)
        tf_payloads = [_payload(symbol, tf, b, rolled[b])
                       for b in sorted(rolled)
                       if w0 <= b <= wn and b not in existing_tf]
        if tf_payloads:
            _write(sb, tf_payloads)
            derived_filled += len(tf_payloads)
            logger.info("rest_backfill: %s — filled %d %ds gap(s)",
                        symbol, len(tf_payloads), tf)

    return len(gap_payloads), derived_filled


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

    total_1m = total_derived = 0
    for symbol in symbols:
        try:
            filled_1m, filled_d = _backfill_symbol(
                sb, symbol, window_start, cutoff)
            total_1m += filled_1m
            total_derived += filled_d
        except Exception as e:
            logger.warning("rest_backfill: %s failed — %s", symbol, e)
    logger.info(
        "rest_backfill cycle done — %d symbol(s), %d 1Min + %d derived "
        "gap bar(s) filled", len(symbols), total_1m, total_derived)
