"""Grace-window shadow for sub-minute bars (2026-05-18).

Background — the sub-minute (10Sec) bar builder force-closes a bucket on
wall-clock at bar_end. Polygon's per-second `A` channel arrives with a
measured ~3-second structural delay, so the bucket closes ~2s before its
last second's data lands → flat empty bars ("Bug B"). The fix is a grace
window: hold the bucket open N seconds past bar_end before force-closing.

This module builds 10Sec bars at three grace windows (2s / 3s / 4s past
bar_end) IN PARALLEL from the same `A` per-second stream, writing to the
`live_bars_trades` shadow table with source='grace_2s' / 'grace_3s' /
'grace_4s'. The Bars Comparison UI can then A/B which grace window best
matches REST 1Sec. Shadow-only — production alerts are unaffected.

Reuses `trade_bar_builder`'s writer pool (same `live_bars_trades` table,
PK includes `source` so the grace variants and any trade-channel rows
coexist cleanly).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from trade_bar_builder import _get_executor, _write_bar_sync

logger = logging.getLogger(__name__)

TF_SECONDS = 10  # grace shadow is sub-minute only — 10Sec strategies
_DEFAULT_SYMBOLS = "SPY,TSLA"
_DEFAULT_GRACE = "2,2.5,3,4"  # seconds past bar_end (fractional allowed)


def _grace_label(g: float) -> str:
    """Format a grace value for the `source` tag: 3.0 -> '3', 2.5 -> '2.5'."""
    return str(int(g)) if g == int(g) else ("%g" % g)


def is_enabled() -> bool:
    val = os.environ.get("GRACE_SHADOW_ENABLED", "true").strip().lower()
    return val in ("1", "true", "yes", "on")


def configured_symbols() -> list[str]:
    raw = os.environ.get("GRACE_SHADOW_SYMBOLS", _DEFAULT_SYMBOLS)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def configured_grace_variants() -> list[float]:
    raw = os.environ.get("GRACE_SHADOW_VARIANTS", _DEFAULT_GRACE)
    out: list[float] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            g = float(s)
            if g >= 0:
                out.append(g)
        except ValueError:
            continue
    return out


@dataclass
class _GraceBar:
    """One open 10s bucket. OHLCV aggregated from per-second `A` bars."""
    bucket_start_sec: int
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    sec_count: int = 0       # per-second bars that contributed
    seeded: bool = False

    def update(self, o: float, h: float, l: float, c: float, v: float) -> None:
        if not self.seeded:
            self.open, self.high, self.low, self.close = o, h, l, c
            self.seeded = True
        else:
            if h > self.high:
                self.high = h
            if l < self.low:
                self.low = l
            self.close = c
        self.volume += v
        self.sec_count += 1


class GraceBuilder:
    """Builds 10Sec bars for one (symbol, grace) variant.

    A bucket closes when wall-clock passes `bar_end + grace_sec`. A
    per-second bar for an already-closed bucket is dropped (late) — never
    misattributed to a later bucket.
    """

    def __init__(self, symbol: str, grace_sec: float):
        self.symbol = symbol
        self.grace_sec = grace_sec
        self.source = f"grace_{_grace_label(grace_sec)}s"
        self._open: dict[int, _GraceBar] = {}
        self._last_closed_sec: Optional[int] = None

    def accept_second_bar(self, ts_sec: int, o: float, h: float, l: float,
                          c: float, v: float, wall_clock: float) -> None:
        bucket = (ts_sec // TF_SECONDS) * TF_SECONDS
        self._flush_due(wall_clock)
        if (self._last_closed_sec is not None
                and bucket <= self._last_closed_sec):
            return  # late — this bucket already closed
        bar = self._open.get(bucket)
        if bar is None:
            bar = _GraceBar(bucket)
            self._open[bucket] = bar
        bar.update(o, h, l, c, v)

    def flush_due(self, wall_clock: float) -> None:
        self._flush_due(wall_clock)

    def _flush_due(self, wall_clock: float) -> None:
        if not self._open:
            return
        to_close = [
            b for b in self._open
            if wall_clock >= b + TF_SECONDS + self.grace_sec
        ]
        for b in to_close:
            self._close_bucket(b)

    def _close_bucket(self, bucket: int) -> None:
        bar = self._open.pop(bucket, None)
        if bar is None:
            return
        self._last_closed_sec = bucket
        if not bar.seeded:
            return  # genuinely no data — emit nothing
        payload = {
            "symbol": self.symbol,
            "timeframe_seconds": TF_SECONDS,
            "bar_start": datetime.fromtimestamp(
                bucket, tz=timezone.utc).isoformat(),
            "source": self.source,
            "open": float(bar.open), "high": float(bar.high),
            "low": float(bar.low), "close": float(bar.close),
            # live_bars_trades.volume is bigint — must be an int.
            "volume": int(round(bar.volume)),
            "trade_count": int(bar.sec_count),
        }
        try:
            _get_executor().submit(_write_bar_sync, payload)
        except Exception as e:
            logger.warning("grace_shadow: submit failed sym=%s src=%s: %s",
                            self.symbol, self.source, e)


class GraceShadowManager:
    """Engine-side facade — owns one builder per (symbol, grace) and
    dispatches per-second `A` bars to them. Safe no-op when disabled."""

    def __init__(self):
        self.enabled = is_enabled()
        self.symbols = set(configured_symbols())
        self.grace_variants = configured_grace_variants()
        self._builders: dict[tuple[str, float], GraceBuilder] = {}
        if self.enabled and self.symbols and self.grace_variants:
            for sym in self.symbols:
                for g in self.grace_variants:
                    self._builders[(sym, g)] = GraceBuilder(sym, g)
            logger.info(
                "grace_shadow: enabled — symbols=%s grace=%ss (%d builders)",
                sorted(self.symbols), self.grace_variants,
                len(self._builders))
        else:
            logger.info(
                "grace_shadow: disabled (enabled=%s symbols=%s grace=%s)",
                self.enabled, sorted(self.symbols), self.grace_variants)

    def on_second_bar(self, symbol: str, bar_dict: dict,
                      wall_clock: Optional[float] = None) -> None:
        """Handle one per-second `A` bar. Cheap no-op when out of scope."""
        if not self.enabled or symbol not in self.symbols:
            return
        ts_raw = bar_dict.get("timestamp")
        try:
            if isinstance(ts_raw, str):
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            else:
                ts = ts_raw
            ts_sec = int(ts.timestamp())
            o = float(bar_dict.get("open", 0) or 0)
            h = float(bar_dict.get("high", 0) or 0)
            l = float(bar_dict.get("low", 0) or 0)
            c = float(bar_dict.get("close", 0) or 0)
            v = float(bar_dict.get("volume", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            return
        if o <= 0 or c <= 0:
            return
        if wall_clock is None:
            wall_clock = time.time()
        for g in self.grace_variants:
            b = self._builders.get((symbol, g))
            if b is not None:
                b.accept_second_bar(ts_sec, o, h, l, c, v, wall_clock)

    def flush_due(self) -> None:
        """Periodic timer-driven close — call from the engine heartbeat."""
        if not self.enabled or not self._builders:
            return
        now = time.time()
        for b in self._builders.values():
            b.flush_due(now)
