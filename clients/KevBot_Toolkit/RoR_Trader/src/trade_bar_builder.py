"""Phase H.1 (2026-05-15) — streaming trade-channel OHLCV bar builder.

Polygon's AM/A WebSocket channels emit PRE-AGGREGATED bars. The first
emission tends to omit late-arriving trades and gets rebroadcast 20-100s
later with corrections — the engine's "decision-time" view of the
market therefore disagrees with REST aggregates ~16-20% of the time at
bar close (Phase G measurements).

This module builds bars from Polygon's T (individual trades) channel
instead, with a configurable wait window per bucket boundary. Three
variants run in parallel during the shadow phase (0ms / 200ms / 500ms)
so we can pick the best latency-vs-accuracy tradeoff empirically.

Shadow-only: writes to `live_bars_trades`, NOT `live_bars`. Production
alerts are unaffected. After 1-2 days of data, Phase H.3 picks a
winning variant and promotes it.

Eligibility filtering reuses `polygon_conditions._classify_eligibility`
— the exact same canonical Polygon rules that flat_file_ingestion uses
for observable-bar generation. Parity by construction.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from polygon_conditions import _classify_eligibility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag + scope
# ---------------------------------------------------------------------------

_DEFAULT_SYMBOLS = "SPY,TSLA"
_DEFAULT_TFS = "10,60"
_DEFAULT_WAIT_VARIANTS = "0,200,500"


def is_enabled() -> bool:
    val = os.environ.get(
        "POLYGON_TRADE_SHADOW_ENABLED", "true").strip().lower()
    return val in ("1", "true", "yes", "on")


def configured_symbols() -> list[str]:
    raw = os.environ.get("POLYGON_TRADE_SYMBOLS", _DEFAULT_SYMBOLS)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def configured_tfs_seconds() -> list[int]:
    raw = os.environ.get("POLYGON_TRADE_TFS_SECONDS", _DEFAULT_TFS)
    out: list[int] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            tf = int(s)
            if tf > 0:
                out.append(tf)
        except ValueError:
            continue
    return out


def configured_wait_variants_ms() -> list[int]:
    raw = os.environ.get("POLYGON_TRADE_WAIT_VARIANTS", _DEFAULT_WAIT_VARIANTS)
    out: list[int] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            w = int(s)
            if w >= 0:
                out.append(w)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Writer pool — separate from live_bars_writer to avoid contention
# ---------------------------------------------------------------------------

_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="trade_bar_writer")
    return _executor


def shutdown(wait: bool = True) -> None:
    """Drain the pool on engine shutdown so in-flight writes finish."""
    global _executor
    if _executor is None:
        return
    try:
        _executor.shutdown(wait=wait)
    except Exception as e:
        logger.warning("trade_bar_builder: shutdown error: %s", e)
    _executor = None


def _write_bar_sync(payload: dict) -> None:
    try:
        from db import get_admin_client
    except Exception as e:
        logger.warning("trade_bar_builder: db import failed: %s", e)
        return
    try:
        client = get_admin_client()
        client.table("live_bars_trades").upsert(
            payload,
            on_conflict="symbol,timeframe_seconds,bar_start,source",
        ).execute()
    except Exception as e:
        logger.warning(
            "trade_bar_builder: write failed sym=%s tf=%ss src=%s bar=%s: %s",
            payload.get("symbol"), payload.get("timeframe_seconds"),
            payload.get("source"), payload.get("bar_start"), e)


# ---------------------------------------------------------------------------
# Streaming bar accumulator
# ---------------------------------------------------------------------------

@dataclass
class _StreamBar:
    """One open bucket. OHLCV semantics mirror flat_file_ingestion._Bar.

    Extended-hours note (Phase H.1 hotfix, 2026-05-15): during ETH ALL
    trades on the consolidated tape carry condition 12 (Form T), which
    the canonical filter classifies as OHLC-ineligible. To avoid
    silently dropping every ETH bar, we also track a parallel vol-only
    OHLC series. If `seeded` is False at emit time but `vol_seeded` is
    True, we promote the vol-only series to the bar's OHLC so ETH
    coverage isn't lost. trade_count stays at 0 in that case — a
    useful built-in marker for "this bar was synthesized from vol-only
    trades" during downstream analysis.
    """
    bucket_start_sec: int          # epoch seconds (UTC) of bar_start
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    trade_count: int = 0           # OHLC-eligible trades that contributed
    filtered_trades: int = 0       # Trades dropped by canonical filter
    seeded: bool = False           # True once we've taken a first OHLC trade
    # Parallel vol-only OHLC tracker (ETH fallback).
    vol_seeded: bool = False
    vol_open: float = 0.0
    vol_high: float = 0.0
    vol_low: float = 0.0
    vol_close: float = 0.0

    def update(
        self,
        price: float,
        size: int,
        update_ohlc: bool,
        update_volume: bool,
    ) -> None:
        if update_ohlc:
            if not self.seeded:
                self.open = self.high = self.low = self.close = price
                self.seeded = True
            else:
                if price > self.high:
                    self.high = price
                if price < self.low:
                    self.low = price
                self.close = price
            self.trade_count += 1
        if update_volume:
            self.volume += size
            if not update_ohlc:
                # Vol-only trade: track in parallel series for ETH fallback.
                if not self.vol_seeded:
                    self.vol_open = price
                    self.vol_high = price
                    self.vol_low = price
                    self.vol_close = price
                    self.vol_seeded = True
                else:
                    if price > self.vol_high:
                        self.vol_high = price
                    if price < self.vol_low:
                        self.vol_low = price
                    self.vol_close = price


class TradeBarBuilder:
    """One builder = one (symbol, tf_seconds, wait_ms) combination.

    Maintains a dict of in-flight buckets keyed by bucket_start_sec.
    A bucket closes when wall-clock exceeds `bucket_end + wait_ms/1000`.
    Closed bars are submitted to the writer pool (no blocking on the WS
    consumer thread).

    Out-of-bucket trades (sip_ts in a bucket that's already been closed)
    are dropped silently — that's the whole point of the wait_ms knob:
    we explicitly trade off late-arrival completeness vs decision latency.
    """

    def __init__(self, symbol: str, tf_seconds: int, wait_ms: int):
        self.symbol = symbol
        self.tf_seconds = tf_seconds
        self.wait_ms = wait_ms
        self.source = f"t_wait{wait_ms}"
        self._open: dict[int, _StreamBar] = {}
        # Track the most recently closed bucket so we know to discard
        # late arrivals deterministically (vs. silently creating a new
        # bucket for a sip_ts older than what we already closed).
        self._last_closed_bucket_sec: Optional[int] = None
        self._dropped_late = 0
        # Phase H diagnostic counters (cumulative). The ratios between
        # these expose where trades go — see TradeBarShadowManager stats log.
        self._dropped_small = 0
        self._dropped_both_ineligible = 0
        self._accepted = 0
        self._bars_emitted = 0

    def stats(self) -> dict:
        """Cumulative ingest breakdown for this builder."""
        return {
            "src": self.source,
            "tf": self.tf_seconds,
            "accepted": self._accepted,
            "late": self._dropped_late,
            "small": self._dropped_small,
            "ineligible": self._dropped_both_ineligible,
            "emitted": self._bars_emitted,
            "open": len(self._open),
        }

    # -- Public API ----------------------------------------------------

    def accept_trade(
        self,
        price: float,
        size: int,
        sip_ms: int,
        conditions,
        wall_clock_sec: float,
    ) -> None:
        """Ingest one trade event.

        - `sip_ms`: Polygon WS SIP timestamp in MILLISECONDS — the format
          stocks T events use. (Flat-file CSV uses nanoseconds; WS uses
          milliseconds. Distinct units, distinct sources — don't confuse.)
        - `conditions`: raw conditions field from the T event (list[int]
          or string — `_classify_eligibility` normalizes both).
        - `wall_clock_sec`: time.time() at the moment the event was
          received. Drives bucket closure for the wait_ms timer.
        """
        # Skip fractional shares — settlement micro-prints, not market activity.
        if size < 1:
            self._dropped_small += 1
            return

        ohlc_ok, vol_ok = _classify_eligibility(conditions)
        if not ohlc_ok and not vol_ok:
            self._dropped_both_ineligible += 1
            return  # ineligible for both → no contribution at all

        sec_ts = sip_ms // 1000
        bucket_start = (sec_ts // self.tf_seconds) * self.tf_seconds

        # Flush any due buckets BEFORE inserting — that way a trade with
        # sip_ts in the next bucket triggers closure of the prior bucket
        # at wait_ms=0 (next-bucket-trade is the close signal).
        self._flush_due(wall_clock_sec, hint_bucket=bucket_start)

        # If this trade's bucket is already closed (late arrival), drop it.
        if (self._last_closed_bucket_sec is not None
                and bucket_start <= self._last_closed_bucket_sec):
            self._dropped_late += 1
            # Still note it on the most recent open bar's filter counter
            # if one happens to be open in the same vicinity — but we
            # don't have a sensible bucket for "late dropped". Track at
            # builder-level via _dropped_late instead.
            return

        bar = self._open.get(bucket_start)
        if bar is None:
            bar = _StreamBar(bucket_start_sec=bucket_start)
            self._open[bucket_start] = bar

        if not ohlc_ok and vol_ok:
            # Vol-only trade: count for volume only.
            bar.update(price, size, update_ohlc=False, update_volume=True)
        elif ohlc_ok and not vol_ok:
            # Rare OHLC-only (e.g., MC official open/close).
            bar.update(price, size, update_ohlc=True, update_volume=False)
        else:
            bar.update(price, size, update_ohlc=True, update_volume=True)
        self._accepted += 1

    def flush_due(self, wall_clock_sec: float) -> None:
        """Public flush hook for the periodic timer in the WS loop."""
        self._flush_due(wall_clock_sec, hint_bucket=None)

    # -- Internals -----------------------------------------------------

    def _flush_due(
        self, wall_clock_sec: float, hint_bucket: Optional[int],
    ) -> None:
        """Close any open buckets whose wait deadline has passed.

        wait_ms=0 means deadline == bucket_end, so as soon as wall-clock
        crosses the boundary we emit. For wait_ms>0 we hold until then.

        `hint_bucket` lets callers force-close prior buckets when a
        trade arrives with sip_ts in a strictly-later bucket — this
        handles wait_ms=0 cleanly even when wall-clock and sip-clock
        drift slightly.
        """
        if not self._open:
            return
        wait_sec = self.wait_ms / 1000.0
        to_close: list[int] = []
        for bucket_start in self._open:
            bucket_end = bucket_start + self.tf_seconds
            deadline = bucket_end + wait_sec
            wall_due = wall_clock_sec >= deadline
            hint_due = (
                hint_bucket is not None
                and hint_bucket > bucket_start
                and self.wait_ms == 0
            )
            if wall_due or hint_due:
                to_close.append(bucket_start)
        for bucket_start in to_close:
            self._close_bucket(bucket_start)

    def _close_bucket(self, bucket_start: int) -> None:
        bar = self._open.pop(bucket_start, None)
        if bar is None:
            return
        if not bar.seeded:
            if bar.vol_seeded:
                # ETH fallback: every trade in this bucket was vol-only
                # (Form T flagged). Promote the parallel vol-only series
                # so the bar still emits — preserves extended-hours
                # coverage that would otherwise vanish.
                bar.open = bar.vol_open
                bar.high = bar.vol_high
                bar.low = bar.vol_low
                bar.close = bar.vol_close
                bar.seeded = True
            else:
                # Truly empty bucket — no qualifying trades at all.
                return
        ts_iso = datetime.fromtimestamp(
            bucket_start, tz=timezone.utc).isoformat()
        payload = {
            "symbol": self.symbol,
            "timeframe_seconds": int(self.tf_seconds),
            "bar_start": ts_iso,
            "source": self.source,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "trade_count": int(bar.trade_count),
            "filtered_trades": int(bar.filtered_trades),
        }
        try:
            _get_executor().submit(_write_bar_sync, payload)
        except Exception as e:
            logger.warning(
                "trade_bar_builder: submit failed sym=%s tf=%ss src=%s: %s",
                self.symbol, self.tf_seconds, self.source, e)
        self._last_closed_bucket_sec = bucket_start
        self._bars_emitted += 1


# ---------------------------------------------------------------------------
# Manager — owns one builder per (symbol, tf, wait_ms) for the engine
# ---------------------------------------------------------------------------

class TradeBarShadowManager:
    """Engine-side facade. Owns builders for the configured scope and
    dispatches T events to them. Safe no-op when feature flag is off.
    """

    def __init__(self):
        self.enabled = is_enabled()
        self.symbols = set(configured_symbols())
        self.tfs = configured_tfs_seconds()
        self.waits = configured_wait_variants_ms()
        self._builders: dict[tuple[str, int, int], TradeBarBuilder] = {}
        # Phase H diagnostic counters — receive-side. `_events_seen` vs the
        # REST trade count tells us whether the WS is delivering trades at
        # all; the lag window tells us whether bars close before their
        # trades finish arriving.
        self._events_seen = 0
        self._events_wrong_sym = 0
        self._events_parse_fail = 0
        self._lag_n = 0
        self._lag_sum = 0.0
        self._lag_min = float("inf")
        self._lag_max = 0.0
        self._last_stats_log = 0.0
        if self.enabled and self.symbols and self.tfs and self.waits:
            for sym in self.symbols:
                for tf in self.tfs:
                    for w in self.waits:
                        self._builders[(sym, tf, w)] = TradeBarBuilder(
                            sym, tf, w)
            logger.info(
                "trade_bar_shadow: enabled — symbols=%s tfs=%s waits=%s "
                "(%d builders)",
                sorted(self.symbols), self.tfs, self.waits,
                len(self._builders))
        else:
            logger.info(
                "trade_bar_shadow: disabled "
                "(enabled=%s symbols=%s tfs=%s waits=%s)",
                self.enabled, sorted(self.symbols), self.tfs, self.waits)

    def trade_channels(self) -> list[str]:
        """Polygon WS channel names to subscribe to."""
        if not self.enabled:
            return []
        return [f"T.{sym}" for sym in sorted(self.symbols)]

    def on_trade_event(self, ev: dict) -> None:
        """Handle one Polygon T event. Cheap no-op if shadow is off."""
        if not self.enabled:
            return
        self._events_seen += 1
        sym = ev.get("sym")
        if not sym or sym not in self.symbols:
            self._events_wrong_sym += 1
            return
        # Polygon stocks WS T event:
        #   p = price (float)
        #   s = size (int, shares)
        #   t = SIP timestamp in MILLISECONDS (NOT nanoseconds — that's
        #       only the flat-file CSV format)
        #   c = conditions (list[int])
        try:
            price = float(ev["p"])
            size = int(ev["s"])
            sip_ms = int(ev["t"])
        except (KeyError, TypeError, ValueError):
            self._events_parse_fail += 1
            return
        conditions = ev.get("c")
        now = time.time()
        lag = now - sip_ms / 1000.0
        self._lag_n += 1
        self._lag_sum += lag
        if lag < self._lag_min:
            self._lag_min = lag
        if lag > self._lag_max:
            self._lag_max = lag
        for tf in self.tfs:
            for w in self.waits:
                b = self._builders.get((sym, tf, w))
                if b is not None:
                    b.accept_trade(price, size, sip_ms, conditions, now)

    def flush_due(self) -> None:
        """Periodic timer-driven close — call from the engine heartbeat."""
        if not self.enabled or not self._builders:
            return
        now = time.time()
        for b in self._builders.values():
            b.flush_due(now)
        self._maybe_log_stats(now)

    def _maybe_log_stats(self, now: float) -> None:
        """Emit a diagnostic stats line at most every 30s.

        Lag stats are windowed (reset each line) so each line reflects
        the last 30s; counters are cumulative — read the deltas.
        """
        if now - self._last_stats_log < 30.0:
            return
        self._last_stats_log = now
        avg_lag = (self._lag_sum / self._lag_n) if self._lag_n else 0.0
        lag_min = self._lag_min if self._lag_min != float("inf") else 0.0
        logger.info(
            "trade_shadow STATS: events_seen=%d wrong_sym=%d parse_fail=%d | "
            "lag(s) n=%d avg=%.1f min=%.1f max=%.1f | builders=%s",
            self._events_seen, self._events_wrong_sym, self._events_parse_fail,
            self._lag_n, avg_lag, lag_min, self._lag_max,
            [b.stats() for b in self._builders.values()],
        )
        self._lag_n = 0
        self._lag_sum = 0.0
        self._lag_min = float("inf")
        self._lag_max = 0.0
