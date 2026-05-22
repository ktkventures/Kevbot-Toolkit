"""Polygon REST ingest for the data-worker — provisional + reconciliation.

Two cycle functions, driven by daemon loops in data_worker.py:

- `run_ingest_cycle` — every ~1s, fetch the trailing ~25s of 1-second
  bars and upsert them. A bar is committed ~6-7s after it closes
  (spec §10.2 provisional commit).
- `run_recon_cycle` — every ~60s, re-fetch the last ~15 min of 1s bars
  and upsert the whole window. Because the store is last-write-wins,
  any bar that gained late prints is overwritten with the canonical
  REST value — the lane converges to bit-identical-with-REST.

Mirrors the live_bars_rest_backfill_subminute.py fetch-REST-and-reconcile
shape, minus the DB write. Polygon REST 1-second aggregates are queried
by a tight millisecond-epoch range via data_loader._polygon_fetch_bars
(NOT fetch_1s_bars_for_window — that has a day-level cache that would
freeze the series).
"""
from __future__ import annotations

import logging
import os
import threading
import time as _time
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger("data_worker")

# Window the provisional loop fetches each tick: [now-LOOKBACK, now-LAG].
# LAG is how long after a bar closes before it's eligible — the per-second
# loop cadence makes the effective provisional commit ~6-7s (spec §10.2).
INGEST_LOOKBACK_SECONDS = int(os.getenv("DATA_WORKER_INGEST_LOOKBACK_SECONDS", "30"))
INGEST_COMMIT_LAG_SECONDS = int(os.getenv("DATA_WORKER_INGEST_COMMIT_LAG_SECONDS", "5"))


class IngestMetrics:
    """Thread-safe counters shared by the ingest / recon / metrics loops."""

    def __init__(self):
        self._lock = threading.Lock()
        self.last_ingest_latency_s = None
        self._latencies: list = []
        self._cycle_durations: list = []
        self.ingest_cycles = 0
        self.recon_cycles = 0
        self.fetch_errors = 0
        self.provisional_appended_total = 0
        self.recon_revised_total = 0
        # --- Phase 2.5 coarse (Tier-2) ingest counters ---
        self.coarse_cycles = 0
        self.coarse_appended_total = 0
        self.coarse_revised_total = 0
        self.coarse_fetch_errors = 0

    def record_ingest(self, latency, duration, result):
        with self._lock:
            self.ingest_cycles += 1
            if latency is not None:
                self.last_ingest_latency_s = latency
                self._latencies.append(latency)
                if len(self._latencies) > 200:
                    self._latencies.pop(0)
            self._cycle_durations.append(duration)
            if len(self._cycle_durations) > 200:
                self._cycle_durations.pop(0)
            self.provisional_appended_total += result.get('appended', 0)

    def record_recon(self, result):
        with self._lock:
            self.recon_cycles += 1
            self.recon_revised_total += result.get('revised', 0)

    def record_fetch_error(self):
        with self._lock:
            self.fetch_errors += 1

    def record_coarse(self, result):
        with self._lock:
            self.coarse_cycles += 1
            self.coarse_appended_total += result.get('appended', 0)
            self.coarse_revised_total += result.get('revised', 0)

    def record_coarse_fetch_error(self):
        with self._lock:
            self.coarse_fetch_errors += 1

    @staticmethod
    def _p95(values):
        if not values:
            return None
        s = sorted(values)
        return round(s[min(len(s) - 1, int(len(s) * 0.95))], 2)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'last_ingest_latency_s': self.last_ingest_latency_s,
                'ingest_latency_p95_s': self._p95(self._latencies),
                'cycle_dur_p95_s': self._p95(self._cycle_durations),
                'ingest_cycles': self.ingest_cycles,
                'recon_cycles': self.recon_cycles,
                'fetch_errors': self.fetch_errors,
                'provisional_appended_total': self.provisional_appended_total,
                'recon_revised_total': self.recon_revised_total,
                'coarse_cycles': self.coarse_cycles,
                'coarse_appended_total': self.coarse_appended_total,
                'coarse_revised_total': self.coarse_revised_total,
                'coarse_fetch_errors': self.coarse_fetch_errors,
            }


def is_market_window(now: datetime = None) -> bool:
    """True during a weekday Extended-Hours window (4:00-20:00 ET).

    Covers every trading session a strategy might use. Outside it,
    Polygon returns no fresh 1s bars — the loops idle instead of
    pointlessly polling.
    """
    from zoneinfo import ZoneInfo
    now = now or datetime.now(timezone.utc)
    et = now.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:  # Saturday / Sunday
        return False
    minutes = et.hour * 60 + et.minute
    return 4 * 60 <= minutes < 20 * 60


def _polygon_1s(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch Polygon 1-second bars for [start_dt, end_dt] (UTC-aware)."""
    from data_loader import (
        _to_polygon_ticker, _polygon_fetch_bars, _polygon_bars_to_df,
    )
    ticker = _to_polygon_ticker(symbol)
    from_ms = int(start_dt.timestamp() * 1000)
    to_ms = int(end_dt.timestamp() * 1000)
    results = _polygon_fetch_bars(ticker, 1, 'second', str(from_ms), str(to_ms))
    df = _polygon_bars_to_df(results)
    if len(df) > 0:
        lo = pd.Timestamp(start_dt)
        hi = pd.Timestamp(end_dt)
        df = df[(df.index >= lo) & (df.index <= hi)]
    return df


def run_ingest_cycle(store, metrics: IngestMetrics) -> None:
    """One provisional-ingest cycle — fetch the trailing window, upsert."""
    t0 = _time.monotonic()
    now = datetime.now(timezone.utc)
    start = now - timedelta(seconds=INGEST_LOOKBACK_SECONDS)
    end = now - timedelta(seconds=INGEST_COMMIT_LAG_SECONDS)
    try:
        df = _polygon_1s(store.symbol, start, end)
    except Exception as e:
        metrics.record_fetch_error()
        logger.warning("[data-worker] ingest fetch failed %s: %s",
                        store.symbol, e)
        return
    result = store.upsert_bars(df)
    latency = None
    if len(df) > 0:
        newest = df.index[-1].to_pydatetime()
        latency = (now - newest).total_seconds()
    metrics.record_ingest(latency, _time.monotonic() - t0, result)


def prime_store(store, window_minutes: int) -> dict:
    """One-time cold-start backfill of the full rolling window.

    The provisional loop only fetches the trailing ~25s, so the store
    would otherwise fill *forward* over `window_minutes` before it can
    serve a Phase 2 streaming engine's resume window. Priming pulls the
    whole window in one REST call so streaming can start immediately.
    Idempotent — last-write-wins; safe to call again.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=window_minutes)
    end = now - timedelta(seconds=INGEST_COMMIT_LAG_SECONDS)
    try:
        df = _polygon_1s(store.symbol, start, end)
    except Exception as e:
        logger.warning("[data-worker] prime_store fetch failed %s: %s",
                        store.symbol, e)
        return {'appended': 0, 'total_bars': 0}
    result = store.upsert_bars(df)
    logger.info("[data-worker] prime_store %s: backfilled %d bars "
                "(total=%d span covers ~%dmin)", store.symbol,
                result.get('appended', 0), result.get('total_bars', 0),
                window_minutes)
    return result


# ──────────────────────────────────────────────────────────────────────
# Phase 2.5 — Tier-2 (1-Minute coarse) ingest
# ──────────────────────────────────────────────────────────────────────

def _polygon_1m(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch Polygon 1-Minute bars for [start_dt, end_dt] (UTC-aware).

    Mirrors `_polygon_1s` but with minute granularity. Direct REST so we
    bypass `load_market_data`'s optional bar_cache layer — Tier-2 owns
    its own cache, in memory.
    """
    from data_loader import (
        _to_polygon_ticker, _polygon_fetch_bars, _polygon_bars_to_df,
    )
    ticker = _to_polygon_ticker(symbol)
    from_ms = int(start_dt.timestamp() * 1000)
    to_ms = int(end_dt.timestamp() * 1000)
    results = _polygon_fetch_bars(ticker, 1, 'minute', str(from_ms), str(to_ms))
    df = _polygon_bars_to_df(results)
    if len(df) > 0:
        lo = pd.Timestamp(start_dt)
        hi = pd.Timestamp(end_dt)
        df = df[(df.index >= lo) & (df.index <= hi)]
    return df


def prime_coarse_store(coarse_store, days: int) -> dict:
    """One-time cold-start backfill — pull the full rolling 1Min window.

    Defensive: `_polygon_fetch_bars` silently returns partial results on
    non-200 / mid-stream exceptions (`data_loader.py:415-433`). We log a
    WARNING if the prime came back short (< 0.7 × RTH-only expected
    floor: 390 RTH bars/day × `days` × 0.7). The streaming-loop's metrics
    line carries `coarse: bars=… span=…` so a short prime is visible.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    end = now - timedelta(minutes=1)  # only fully-closed 1Min bars
    try:
        df = _polygon_1m(coarse_store.symbol, start, end)
    except Exception as e:
        logger.warning("[data-worker] prime_coarse_store fetch failed %s: %s",
                        coarse_store.symbol, e)
        return {'appended': 0, 'total_bars': 0}
    result = coarse_store.upsert_bars(df)
    floor = int(390 * days * 0.7)  # RTH-only conservative floor
    bar_count = result.get('total_bars', 0)
    if bar_count < floor:
        logger.warning(
            "[data-worker] prime_coarse_store %s SHORT: bars=%d < floor=%d "
            "(Polygon may have returned partial — refresh cycles will fill "
            "gaps; cross-TF strategies may be temporarily under-warmed)",
            coarse_store.symbol, bar_count, floor)
    else:
        logger.info(
            "[data-worker] prime_coarse_store %s: backfilled %d 1Min bars "
            "(total=%d, span covers ~%dd)", coarse_store.symbol,
            result.get('appended', 0), bar_count, days)
    return result


def run_coarse_ingest_cycle(coarse_store, metrics: IngestMetrics) -> None:
    """Incremental Tier-2 refresh — append new 1Min bars + revise the tail.

    Pulls `[last_bar_ts - 5min overlap, now - 1min]` and upserts. The
    5-min overlap lets last-write-wins overwrite the most recent
    provisional 1Min bar with the late-print-corrected one (mirrors
    Tier-1's recon philosophy). Cheap — typically 1-3 new bars per cycle.
    """
    cov = coarse_store.coverage()
    now = datetime.now(timezone.utc)
    end = now - timedelta(minutes=1)
    if cov['count'] == 0 or cov['last_ts'] is None:
        # No prior data — fall back to a one-day backfill to seed the
        # series; full prime is meant to be called separately at startup.
        start = now - timedelta(days=1)
    else:
        start = cov['last_ts'].to_pydatetime() - timedelta(minutes=5)
    try:
        df = _polygon_1m(coarse_store.symbol, start, end)
    except Exception as e:
        metrics.record_coarse_fetch_error()
        logger.warning("[data-worker] coarse ingest fetch failed %s: %s",
                        coarse_store.symbol, e)
        return
    result = coarse_store.upsert_bars(df)
    metrics.record_coarse(result)
    if result.get('revised'):
        logger.info("[data-worker] coarse %s: revised=%d appended=%d "
                    "total=%d", coarse_store.symbol, result['revised'],
                    result['appended'], result['total_bars'])


def run_recon_cycle(store, metrics: IngestMetrics, window_minutes: int) -> None:
    """One reconciliation cycle — re-pull the trailing window, upsert.

    Last-write-wins in the store overwrites provisional bars whose close
    moved due to late prints.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=window_minutes)
    end = now - timedelta(seconds=INGEST_COMMIT_LAG_SECONDS)
    try:
        df = _polygon_1s(store.symbol, start, end)
    except Exception as e:
        metrics.record_fetch_error()
        logger.warning("[data-worker] recon fetch failed %s: %s",
                        store.symbol, e)
        return
    result = store.upsert_bars(df)
    metrics.record_recon(result)
    if result.get('revised'):
        logger.info("[data-worker] recon %s: revised=%d appended=%d "
                    "total=%d", store.symbol, result['revised'],
                    result['appended'], result['total_bars'])
