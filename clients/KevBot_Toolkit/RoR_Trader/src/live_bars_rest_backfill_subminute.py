"""REST 1Sec → sub-minute backfill for live_bars cache gaps (2026-05-20).

Sibling to `live_bars_rest_backfill.py` (which handles tf=60 + derived
≥60s). When the live A-channel path misses sub-minute bars (5Sec / 10Sec
/ 15Sec / 30Sec — anywhere a Worker outage or WebSocket gap left a hole),
this module fetches Polygon REST 1-second aggregates, rolls them up to
each sub-minute timeframe the engine builds for the symbol, and writes
the missing buckets with ``source='rest_backfill'``.

Why 1Sec → roll (vs native N-second aggs):
  - One fetch shape covers every sub-minute TF (5/10/15/30) — future
    TFs come along for free.
  - REST 1Sec-rolled with `(sec // tf) * tf` is unconditionally
    absolute-aligned (matches the live BarBuilder's `_align_to_period`
    and the parity validator). Native N-sec aggs align to the query
    `from` param, which is a latent fragility.
  - Validated 2026-05-19 (D3 sweep): native-N-sec (date `from`)
    == 1Sec-rolled at 100% on SPY/TSLA.

Cosmetic only — same rule as the 1Min sibling: rest_backfill rows are
for charts/backtests; the live engine never fires alerts on them.

Safety invariants (mirrored from the 1Min sibling):
  - INSERT only — `ignore_duplicates=True` (ON CONFLICT DO NOTHING),
    so an existing ws row is never overwritten.
  - Only fills periods older than LAG_MINUTES — never races the live
    writer.
  - A sub-minute bucket is emitted only when REST 1Sec has ≥1 bar in
    that bucket. Polygon 1Sec aggregates are *not* emitted for seconds
    with zero trades (unlike 1Min), so a partial-second bucket is
    legitimate — but a completely empty bucket means "no trades, no
    data" and we leave it alone rather than synthesize a flat bar.
  - Env-gated (LIVE_BARS_BACKFILL_SUBMINUTE_ENABLED, default OFF).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("live_bars_rest_backfill_subminute")

_PAGE = 1000


def is_enabled() -> bool:
    val = os.environ.get(
        "LIVE_BARS_BACKFILL_SUBMINUTE_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def interval_seconds() -> int:
    """Cron cadence — default 10 min. Sub-minute gaps don't change much
    cycle-to-cycle, but a tighter cadence than the 1Min cron (default
    15 min) helps recover from outages faster on the buckets that
    matter most for live alerts."""
    return int(os.environ.get(
        "LIVE_BARS_BACKFILL_SUBMINUTE_INTERVAL_SECONDS", "600"))


def _window_hours() -> int:
    """How far back to look per cycle. Default 1h — keeps each Polygon
    1Sec fetch small (~3600 bars per symbol per cycle vs ~21600 at 6h),
    sidestepping the full-day JSON-truncation issue we saw with very
    large 1Sec responses. Longer outages drain across multiple cycles."""
    return int(os.environ.get(
        "LIVE_BARS_BACKFILL_SUBMINUTE_WINDOW_HOURS", "1"))


def _lag_minutes() -> int:
    """Skip the most-recent N minutes to avoid racing live writes."""
    return int(os.environ.get(
        "LIVE_BARS_BACKFILL_SUBMINUTE_LAG_MINUTES", "5"))


def _epoch(ts) -> int:
    """ISO string / datetime → epoch seconds UTC."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return int(ts.timestamp())


def _active_subminute_symbols(sb, window_start_iso: str) -> list[str]:
    """Stock symbols with recent sub-minute (tf<60) live_bars rows.

    Crypto (symbols containing '/') excluded — stocks-only, same as the
    1Min sibling.
    """
    rows = (sb.table("live_bars")
            .select("symbol")
            .lt("timeframe_seconds", 60)
            .gte("bar_start", window_start_iso)
            .limit(20000)
            .execute().data or [])
    return sorted({
        r["symbol"] for r in rows
        if r.get("symbol") and "/" not in r["symbol"]
    })


def _subminute_tfs(sb, symbol: str, w0_iso: str) -> list[int]:
    """Sub-minute TFs the engine actively builds for this symbol — the
    only ones with sub-minute bars worth gap-filling."""
    rows = (sb.table("live_bars").select("timeframe_seconds")
            .eq("symbol", symbol).lt("timeframe_seconds", 60)
            .gte("bar_start", w0_iso).limit(20000).execute().data or [])
    return sorted({int(r["timeframe_seconds"]) for r in rows})


def _fetch_existing(sb, symbol: str, tf: int, w0_iso: str,
                    w1_iso: str) -> set[int]:
    """Epoch-second keys of existing live_bars rows for (symbol, tf)
    in the window. Set-only — we just need 'is this bucket filled?'"""
    out: set[int] = set()
    offset = 0
    while True:
        page = (sb.table("live_bars")
                .select("bar_start")
                .eq("symbol", symbol).eq("timeframe_seconds", tf)
                .gte("bar_start", w0_iso).lte("bar_start", w1_iso)
                .order("bar_start").range(offset, offset + _PAGE - 1)
                .execute().data or [])
        for r in page:
            out.add(_epoch(r["bar_start"]))
        if len(page) < _PAGE:
            return out
        offset += _PAGE


def _fetch_rest_1s(symbol: str, window_start: datetime,
                   cutoff: datetime) -> dict[int, dict]:
    """REST 1Sec aggs keyed by epoch second.

    Uses ms-epoch range (`from`/`to` as millisecond strings) to control
    the exact window — the full-day-via-date-string path can truncate
    the JSON response on very large 1Sec days. Polygon's pagination
    in `_polygon_fetch_bars` handles this fetch cleanly for ≤1h windows.
    """
    from data_loader import _polygon_fetch_bars, _to_polygon_ticker
    from_ms = str(int(window_start.timestamp() * 1000))
    to_ms = str(int(cutoff.timestamp() * 1000))
    raw = _polygon_fetch_bars(
        _to_polygon_ticker(symbol), 1, "second", from_ms, to_ms) or []
    out: dict[int, dict] = {}
    for r in raw:
        sec = int(r.get("t", 0) / 1000)
        out[sec] = {
            "open": r.get("o"), "high": r.get("h"),
            "low": r.get("l"), "close": r.get("c"),
            "volume": r.get("v", 0) or 0,
        }
    return out


def _rollup_from_1s(seconds: dict[int, dict],
                    tf: int) -> dict[int, dict]:
    """Roll a 1Sec OHLCV series up to `tf`-second buckets.

    Emits any bucket that has ≥1 1Sec bar — unlike the 1Min sibling
    which requires the full complement. Polygon 1Sec aggs are only
    emitted when there's at least one trade in the second; partial
    buckets are legitimate. (Buckets with zero 1Sec bars are skipped
    — we don't synthesize a flat carry-forward bar.)
    """
    buckets: dict[int, list] = {}
    for sec in sorted(seconds):
        buckets.setdefault((sec // tf) * tf, []).append(seconds[sec])
    out: dict[int, dict] = {}
    for b, bars in buckets.items():
        if not bars:
            continue
        out[b] = {
            "open": bars[0]["open"],
            "high": max(x["high"] for x in bars
                        if x["high"] is not None),
            "low": min(x["low"] for x in bars
                       if x["low"] is not None),
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
    so an existing ws row is never overwritten."""
    sb.table("live_bars").upsert(
        payloads,
        on_conflict="symbol,timeframe_seconds,bar_start",
        ignore_duplicates=True,
    ).execute()


def _backfill_symbol(sb, symbol: str, window_start: datetime,
                     cutoff: datetime) -> dict[int, int]:
    """Fill sub-minute gaps for every sub-minute TF the symbol builds.

    Returns {tf_seconds: rows_filled} so the caller can log per-TF totals.
    """
    w0_iso, w1_iso = window_start.isoformat(), cutoff.isoformat()
    w0, wn = int(window_start.timestamp()), int(cutoff.timestamp())

    tfs = _subminute_tfs(sb, symbol, w0_iso)
    if not tfs:
        return {}

    # ONE 1Sec fetch covers all sub-minute TFs for this symbol.
    seconds = _fetch_rest_1s(symbol, window_start, cutoff)
    if not seconds:
        return {tf: 0 for tf in tfs}

    filled: dict[int, int] = {}
    for tf in tfs:
        existing = _fetch_existing(sb, symbol, tf, w0_iso, w1_iso)
        rolled = _rollup_from_1s(seconds, tf)
        payloads = [_payload(symbol, tf, b, rolled[b])
                    for b in sorted(rolled)
                    if w0 <= b <= wn and b not in existing]
        filled[tf] = len(payloads)
        if payloads:
            _write(sb, payloads)
            logger.info(
                "rest_backfill (subm): %s — filled %d %ds gap(s)",
                symbol, len(payloads), tf)
    return filled


def run_backfill_cycle() -> None:
    """One sub-minute backfill pass over all active stock symbols.

    Safe no-op when LIVE_BARS_BACKFILL_SUBMINUTE_ENABLED is off.
    """
    if not is_enabled():
        return
    try:
        from db import get_admin_client
        sb = get_admin_client()
    except Exception as e:
        logger.warning("rest_backfill (subm): db unavailable — %s", e)
        return

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=_lag_minutes())
    window_start = now - timedelta(hours=_window_hours())
    window_start_iso = window_start.isoformat()

    symbols = _active_subminute_symbols(sb, window_start_iso)
    if not symbols:
        logger.debug("rest_backfill (subm): no cached symbols in window")
        return

    totals: dict[int, int] = {}
    for symbol in symbols:
        try:
            filled = _backfill_symbol(sb, symbol, window_start, cutoff)
            for tf, n in filled.items():
                totals[tf] = totals.get(tf, 0) + n
        except Exception as e:
            logger.warning(
                "rest_backfill (subm): %s failed — %s", symbol, e)
    by_tf = ", ".join(f"{n} {tf}s" for tf, n in sorted(totals.items()))
    logger.info(
        "rest_backfill (subm) cycle done — %d symbol(s); filled %s",
        len(symbols), by_tf or "(no gaps)")
