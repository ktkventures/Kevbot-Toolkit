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
