#!/usr/bin/env python3
"""
Ralph Wiggum Alert Engine — O(1) incremental alert detection.

Standalone process that subscribes to real-time market data, builds OHLCV
bars, incrementally updates indicators, evaluates triggers, and fires
alerts/webhooks with sub-second latency.

Usage:
    python ralph_engine.py              # Start the engine
    python ralph_engine.py --status     # Print engine status
    python ralph_engine.py --stop       # Send SIGTERM to running engine

Architecture:
    Ticks → BarBuilder → bar close → IncrementalIndicators (O(1))
          → TriggerEvaluator (single-row) → PositionStateMachine
          → AlertDispatcher (save + webhook)
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from unified_engine import (
    MAX_HISTORY, VWAP_SESSION_GAP_SECONDS,
    TEMPLATE_REQUIREMENTS, TRIGGER_PREFIX_TO_TEMPLATE,
    INTRABAR_LEVEL_MAP, _IB_L_TYPE_TRIGGERS,
    get_trigger_exec_type, _strip_exec_suffix,
    IndicatorState, IncrementalIndicatorEngine,
    TriggerEvaluator, PositionState, PositionStateMachine,
    resolve_strategy_requirements, _resolve_trigger_id, _resolve_trigger_ids,
    _GP_SESSION_WINDOWS, _eval_gp_scalar,
    _load_enabled_general_packs, _evaluate_general_packs,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENGINE_STATUS_FILE = _SCRIPT_DIR / "engine_status.json"
_ENGINE_STATE_FILE = _SCRIPT_DIR / "engine_state.json"
_ENGINE_AUDIT_FILE = _SCRIPT_DIR / "engine_audit.jsonl"
_ENGINE_RELOAD_FLAG = _SCRIPT_DIR / "engine_reload.flag"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s  %(message)s"
_LOG_DATE_FMT = "%H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATE_FMT)
logger = logging.getLogger("ralph")


# ─── Hot-path profiler (2026-05-19, A-lag optimization) ──────────────────
# Lightweight cumulative timing of the per-second processing path. Behind
# HOT_PATH_PROFILE env flag (default off). Dumped every 30s aligned with
# the A-channel lag histogram in _record_a_lag.
_HOT_PATH_PROFILE = os.environ.get(
    'HOT_PATH_PROFILE', '').strip().lower() in ('1', 'true', 'yes', 'on')
_prof_acc: dict = {}  # label -> [total_seconds, call_count]


class _prof:
    """`with _prof('label'):` — accumulates wall time under that label.
    Near-zero overhead when HOT_PATH_PROFILE is off."""
    __slots__ = ('label', '_t0')

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        if _HOT_PATH_PROFILE:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _HOT_PATH_PROFILE:
            d = _prof_acc.get(self.label)
            dt = time.perf_counter() - self._t0
            if d is None:
                _prof_acc[self.label] = [dt, 1]
            else:
                d[0] += dt
                d[1] += 1
        return False


def _prof_fn(label: str):
    """Decorator — accumulate wall time of a function under `label`.
    Returns the function unchanged when HOT_PATH_PROFILE is off."""
    def deco(fn):
        if not _HOT_PATH_PROFILE:
            return fn
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                d = _prof_acc.get(label)
                dt = time.perf_counter() - _t0
                if d is None:
                    _prof_acc[label] = [dt, 1]
                else:
                    d[0] += dt
                    d[1] += 1
        return wrapper
    return deco


def _prof_dump_and_reset() -> None:
    """Log accumulated hot-path timings and clear. Called every 30s."""
    if not _HOT_PATH_PROFILE or not _prof_acc:
        return
    parts = []
    for label, (tot, n) in sorted(
            _prof_acc.items(), key=lambda kv: -kv[1][0]):
        parts.append('%s=%.0fms/%d(%.2fms)' % (
            label, tot * 1000.0, n, tot / n * 1000.0 if n else 0.0))
    logger.info("HOT-PATH PROFILE (30s window): %s", ' | '.join(parts))
    _prof_acc.clear()


# Add file handler with rotation (5 MB max, keep 3 backups).
# Guard: this module runs both as __main__ and via `from ralph_engine import X`
# (separate sys.modules entries), so getLogger("ralph") would accumulate
# duplicate handlers without this check.
if not any(isinstance(h, logging.handlers.RotatingFileHandler)
           for h in logger.handlers):
    try:
        _log_file = Path(__file__).resolve().parent / "ralph_engine.log"
        _fh = logging.handlers.RotatingFileHandler(
            str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=3)
        _fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT))
        _fh.setLevel(logging.DEBUG)
        logger.addHandler(_fh)
    except Exception:
        pass  # Non-critical — console logging still works

# ---------------------------------------------------------------------------
# Constants (shared constants imported from unified_engine above)
# ---------------------------------------------------------------------------
TIMEFRAME_SECONDS = {
    "5Sec": 5, "10Sec": 10, "15Sec": 15, "30Sec": 30,
    "1Min": 60, "2Min": 120, "3Min": 180, "5Min": 300,
    "10Min": 600, "15Min": 900, "30Min": 1800,
    "1Hour": 3600, "2Hour": 7200, "4Hour": 14400,
    "1Day": 86400, "1Week": 604800, "1Month": 2592000,
}
SECONDS_TO_TIMEFRAME = {v: k for k, v in TIMEFRAME_SECONDS.items()}

# Grace window (seconds) for sub-minute bar force-close. See
# BarBuilder.force_close_stale_bar — picked from the 2026-05-19 RTH
# sweep (TSLA 95-100% / SPY 87% within $0.10 vs REST; diminishing
# returns past 5). Future: per-strategy parameter via the Live
# Execution Fidelity spec (`ws_agg_reconciled` live_model).
SUBMINUTE_FORCE_CLOSE_GRACE_SEC = 5

# Bar gap-fill (2026-06-10): when False, empty/no-trade periods produce NO bar
# — matching the backtest's data_loader.resample_to_timeframe(...).dropna(
# subset=['open']) (TradingView convention). Previously the live engine
# carried forward flat zero-volume bars through empty periods (after-hours),
# polluting incremental indicators and diverging from the backtest. Keep this
# False; the constant exists only as a one-flip rollback during the first
# market-hours observation window. Locked by test_ralph_bar_no_gapfill.py.
BAR_GAP_FILL_ENABLED = False

# Short TF labels → seconds (both cases for normalization)
_LABEL_TO_TF_SECONDS = {}
for _tf_name, _tf_sec in TIMEFRAME_SECONDS.items():
    _short = _tf_name.replace('Min', 'M').replace('Hour', 'H').replace(
        'Day', 'D').replace('Week', 'W').replace('Sec', 's')
    _LABEL_TO_TF_SECONDS[_short] = _tf_sec
    _LABEL_TO_TF_SECONDS[_short.lower()] = _tf_sec


def _normalize_confluence_label(record: str) -> str:
    """Normalize TF prefix to uppercase (e.g., '15m-...' → '15M-...').

    GEN- records pass through unchanged.
    """
    if record.startswith('GEN-'):
        return record
    dash = record.find('-')
    if dash < 0:
        return record
    prefix = record[:dash]
    return prefix.upper() + record[dash:]


# Chart pickle defaults per timeframe
CHART_BAR_COUNTS = {
    60: 300, 300: 200, 900: 150, 1800: 100, 3600: 100,
}
DEFAULT_CHART_BARS = 300

# Reconciliation interval (REST bar correction)
RECONCILE_INTERVAL = 60  # seconds
RECONCILE_GRACE_BARS = 2  # don't touch the last N bars

# Hot-reload interval
STRATEGY_REFRESH_INTERVAL = 300  # seconds

# Partial bar pickle write interval
PICKLE_WRITE_INTERVAL = 2.0  # seconds

# Trade condition codes excluded from OHLCV bar building.
# These match the standard CTA/UTP exclusions that Alpaca applies
# to their historical bar aggregation.  Form T ('T') is intentionally
# included because strategies trade extended hours sessions.
EXCLUDED_TRADE_CONDITIONS = {
    'B',  # Average Price Trade (CTA)
    'W',  # Average Price Trade (UTP)
    'C',  # Cash Trade
    'G',  # Bunched Sold Trade
    'H',  # Price Variation Trade
    'I',  # Odd Lot Trade
    'L',  # Sold Last
    'M',  # Market Center Official Close
    'N',  # Next Day
    'P',  # Prior Reference Price
    'Q',  # Market Center Official Open
    'R',  # Seller
    'U',  # Extended Hours (Out of Sequence)
    'V',  # Contingent Trade
    'Z',  # Sold (Out of Sequence)
    '4',  # Derivatively Priced
    '7',  # Qualified Contingent Trade (QCT)
    '9',  # Corrected Consolidated Close
}


# ═══════════════════════════════════════════════════════════════════════════
# BAR BUILDER — standalone OHLCV bar assembly from tick data (no dependencies)
# ═══════════════════════════════════════════════════════════════════════════

class PartialBar:
    """Represents a partial OHLCV bar being built from tick data."""

    __slots__ = ('open', 'high', 'low', 'close', 'volume', 'bar_start',
                 'bar_duration_seconds', 'tick_count')

    def __init__(self, price: float, timestamp: datetime,
                 bar_duration_seconds: int = 60):
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = 0
        self.bar_start = timestamp
        self.bar_duration_seconds = bar_duration_seconds
        self.tick_count = 0

    def update(
        self,
        price: float,
        volume: int = 1,
        update_ohlc: bool = True,
        update_volume: bool = True,
    ):
        """Update partial bar from a tick.

        Phase G (2026-05-15): trades flagged by Polygon's canonical
        eligibility rules as OHLC-ineligible-but-vol-eligible (Form T,
        Odd Lot, etc.) now contribute to volume only — not to H/L/C.
        This mirrors flat_file_ingestion._Bar.update() so live cache
        and observable bars apply identical rules.
        """
        if update_ohlc:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
            self.close = price
        if update_volume:
            self.volume += volume
            self.tick_count += 1

    def to_dict(self) -> dict:
        return {
            'open': self.open, 'high': self.high,
            'low': self.low, 'close': self.close,
            'volume': self.volume,
            'timestamp': self.bar_start.isoformat(),
        }


class BarBuilder:
    """Aggregates ticks into clock-aligned OHLCV bars for one timeframe."""

    def __init__(self, tf_seconds: int):
        self.tf_seconds = tf_seconds
        self.history: pd.DataFrame = pd.DataFrame()
        self._partial: Optional[PartialBar] = None
        self._bar_count = 0
        # M8.7 rebroadcast handling (2026-05-02): set by accept_bar /
        # accept_second_bar after each call. True if the incoming bar was
        # a Polygon WS rebroadcast for the most recent history row (we
        # replaced the row in place); False for a new bar (appended).
        # Callers that care about duplicate-detection read this AFTER
        # calling accept_*. Default False so existing callers (tests,
        # audit scripts) work unchanged.
        self.last_was_duplicate: bool = False
        # last_was_correction: True only for the FIRST rebroadcast of a
        # given (newer) bucket — callers gate recompute_from_history on it.
        # A closed bucket gets many late / out-of-order per-second
        # re-deliveries when the worker drains a backlog; recomputing for
        # every one is a self-feeding lag spiral (2026-05-19). The
        # monotonic _last_rebroadcast_recompute_ts marker bounds it to one
        # recompute per bucket. Genuine Polygon corrections march forward
        # in time, so they still get their recompute.
        self.last_was_correction: bool = False
        self._last_rebroadcast_recompute_ts = None
        # Per-builder grace override — set by RalphEngine._refresh_builder_
        # _grace as max(grace_seconds) across active monitors on this
        # (symbol, TF) builder (LEF Phase 2a, 2026-05-20). None → fall
        # back to SUBMINUTE_FORCE_CLOSE_GRACE_SEC. 1Min+ ignores grace
        # regardless of this value (see force_close_stale_bar).
        self.grace_sec_override: Optional[int] = None
        # Phase C (2026-05-28): bounded per-second history retention so
        # rest_verifier can splice REST per-second values at sub-bar
        # granularity. Bucketed by bar_start; each entry is a list of
        # per-second OHLCV dicts. Capped to K bar_starts via OrderedDict
        # eviction in accept_second_bar to keep memory bounded
        # (~10 bars × ~10 sec/bar × ~100 bytes/sec ≈ 10KB per builder).
        from collections import OrderedDict as _OD
        import os as _os
        _per_sec_k = int(_os.environ.get(
            'BARBUILDER_PER_SEC_HISTORY_K', '10'))
        self._per_second_history: '_OD[pd.Timestamp, list]' = _OD()
        self._per_second_history_max_buckets = max(_per_sec_k, 1)
        # 2026-06-12 (clobber merge fix): constituent bookkeeping for the
        # most recently CLOSED bar, retained at _close_bar time from
        # _partial_seen_subbars. Lets a late sub-bar (a 1Min fan-out
        # delivery arriving after flush_stale_bars already closed the
        # >=120s period — flush has zero grace for 1Min+ and almost
        # always wins the close race) MERGE into the closed row with
        # volume deduped and open/close attributed by constituent order,
        # instead of wholesale-replacing the row (the 2m clobber: volume
        # 0.50x REST, range collapsed to one minute). None when the last
        # close had no constituent info (e.g., REST-seeded history row).
        self._closed_seen_subbars: Optional[set] = None
        self._closed_min_subbar_ts: Optional[pd.Timestamp] = None
        self._closed_max_subbar_ts: Optional[pd.Timestamp] = None
        self._closed_period_start: Optional[pd.Timestamp] = None

    def seed_history(self, df: pd.DataFrame):
        """Seed builder history from a DataFrame.

        2026-05-13 fix: if the trailing row's index matches the currently
        forming period, split it into `self._partial` instead of leaving
        it as `history.iloc[-1]`. Otherwise every incoming bar that aligns
        to the same period_start triggers the rebroadcast branch in
        `accept_second_bar`, which returns early without ever creating a
        partial — meaning the next boundary cross produces no completed
        bar (no live_bars write, no shadow.on_bar_close).

        Symptom captured 2026-05-13 via DIAG-FANOUT logs:
          history_len=147 last_idx=17:30:00 partial=None
          completed=ts=17:30:00 close=742.51 last_was_duplicate=True
        After this fix, last_idx becomes the previous closed bar (17:15)
        and partial gets the forming bar's OHLCV — incoming 1Min bars
        update partial, and 17:45 boundary closes the partial cleanly.
        """
        if df is None or len(df) == 0:
            return

        now_utc = pd.Timestamp.now(tz='UTC')
        period_size_ns = int(self.tf_seconds) * 1_000_000_000
        current_period_start = pd.Timestamp(
            (now_utc.value // period_size_ns) * period_size_ns, tz='UTC')

        last_idx = df.index[-1]
        if last_idx.tzinfo is None:
            last_idx = last_idx.tz_localize('UTC')

        if last_idx >= current_period_start:
            last_row = df.iloc[-1]
            try:
                self._partial = PartialBar(
                    float(last_row['open']),
                    last_idx.to_pydatetime(),
                    self.tf_seconds)
                self._partial.high = float(last_row['high'])
                self._partial.low = float(last_row['low'])
                self._partial.close = float(last_row['close'])
                vol = last_row.get('volume', 0)
                self._partial.volume = float(vol) if vol is not None else 0.0
                self._partial.tick_count = 1
                self.history = df.iloc[:-1].tail(MAX_HISTORY).copy()
                logger.info(
                    "BarBuilder.seed_history: split forming bar at %s "
                    "into _partial for tf=%ss; history=%d closed bars",
                    last_idx, self.tf_seconds, len(self.history))
            except Exception as e:
                # Defensive: if the row is malformed, fall back to seeding
                # the full df. Worse case = the rebroadcast cascade bug
                # we just diagnosed; better than crashing the boot path.
                logger.warning(
                    "BarBuilder.seed_history: forming-bar split failed "
                    "for tf=%ss: %s — falling back to full seed",
                    self.tf_seconds, e)
                self.history = df.tail(MAX_HISTORY).copy()
        else:
            self.history = df.tail(MAX_HISTORY).copy()

        self._bar_count = len(self.history)

    def process_tick(
        self,
        price: float,
        volume: int,
        timestamp: datetime,
        update_ohlc: bool = True,
        update_volume: bool = True,
    ) -> Optional[dict]:
        """Process a single tick into the builder's partial bar.

        Phase G (2026-05-15): added update_ohlc + update_volume flags
        so vol-only trades (Polygon OHLC-ineligible but vol-eligible)
        contribute to volume without disturbing H/L/C. Default values
        preserve backward-compatibility — callers without the flag
        still get full OHLC + volume update.
        """
        # If this tick is volume-only AND no partial exists yet,
        # we can't seed OHLC from it. Drop. Rare; an OHLC-eligible
        # trade is almost always the first tick of any new period.
        if self._partial is None and not update_ohlc:
            return None

        period_start = self._align_to_period(timestamp)
        if self._partial is None:
            self._partial = PartialBar(price, period_start, self.tf_seconds)
            self._partial.update(price, volume,
                                 update_ohlc=update_ohlc,
                                 update_volume=update_volume)
            return None
        if period_start > self._partial.bar_start:
            # Capture before close (close may clear the partial).
            fill_close = self._partial.close
            old_bar_end = self._partial.bar_start + timedelta(
                seconds=self.tf_seconds)
            completed = self._close_bar()
            # Gap-fill (2026-06-10): empty periods produce NO bar when
            # BAR_GAP_FILL_ENABLED is False — matches backtest dropna. The
            # flat carry-forward bars polluted incremental indicators.
            if BAR_GAP_FILL_ENABLED:
                gap_ts = old_bar_end
                while gap_ts < period_start:
                    self._append_to_history({
                        'timestamp': gap_ts.isoformat(),
                        'open': fill_close, 'high': fill_close,
                        'low': fill_close, 'close': fill_close,
                        'volume': 0,
                    })
                    self._bar_count += 1
                    gap_ts += timedelta(seconds=self.tf_seconds)
            # Edge case: if first tick of new period is vol-only,
            # we have no real OHLC to seed. Skip starting a new
            # partial; next OHLC-eligible tick will start it.
            if not update_ohlc:
                return completed
            self._partial = PartialBar(price, period_start, self.tf_seconds)
            self._partial.update(price, volume,
                                 update_ohlc=update_ohlc,
                                 update_volume=update_volume)
            return completed
        self._partial.update(price, volume,
                             update_ohlc=update_ohlc,
                             update_volume=update_volume)
        return None

    @_prof_fn('b_accept_second')
    def accept_second_bar(self, bar_dict: dict,
                          close_on_boundary: bool = True,
                          skip_volume: bool = False,
                          full_period_input: bool = True) -> Optional[dict]:
        """Aggregate a per-second OHLCV bar into the current period's partial.

        M8.5 Phase B+: enables sub-minute primary-TF aggregation AND forming-bar
        snapshots for any TF, all from Polygon's A.{ticker} per-second channel.

        Aggregation rules MIRROR data_loader.resample_to_timeframe() exactly so
        live and backtest produce bar-for-bar identical primary-TF bars:
            open   = first per-second bar's open
            high   = max across the period
            low    = min across the period
            close  = most recent per-second bar's close
            volume = sum across the period

        No gap-fill: empty periods produce no bar (matches backtest's
        dropna(subset=['open']) behavior).

        Args:
            bar_dict: per-second OHLCV bar dict
            close_on_boundary:
                True  (default — sub-minute primary-TF use): on period boundary,
                       close the prior partial via `_close_bar()` (appends to
                       history, increments bar_count) and return the completed
                       bar dict. This is the canonical sub-minute aggregation.
                False (1Min+ chart-visual use): on period boundary, just discard
                       the prior partial without appending. The AM channel
                       canonical bar is what actually closes the bar via
                       `accept_bar()`. Always returns None.

        Returns the completed bar dict on period close (when close_on_boundary
        is True), None otherwise. Locked by `test_ralph_subminute_parity.py`.
        """
        ts_raw = bar_dict.get('timestamp')
        if isinstance(ts_raw, str):
            ts = pd.Timestamp(ts_raw).to_pydatetime()
        elif isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw.to_pydatetime()
        else:
            ts = ts_raw
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        period_start = self._align_to_period(ts)

        sec_open = float(bar_dict['open'])
        sec_high = float(bar_dict['high'])
        sec_low = float(bar_dict['low'])
        sec_close = float(bar_dict['close'])
        sec_volume = float(bar_dict.get('volume', 0))

        # Phase C (2026-05-28): retain per-second bar dict in bucketed
        # history so rest_verifier can later splice REST per-second
        # corrections at sub-bar granularity. Only retain for
        # close_on_boundary=True mode (the canonical sub-minute path);
        # chart-visual mode (False, used for 1Min+ aesthetics) has no
        # downstream consumer that needs per-second history.
        if close_on_boundary:
            period_ts = pd.Timestamp(period_start)
            if period_ts.tzinfo is None:
                period_ts = period_ts.tz_localize('UTC')
            bucket = self._per_second_history.get(period_ts)
            if bucket is None:
                bucket = []
                self._per_second_history[period_ts] = bucket
                # Evict the oldest bucket if we're over capacity.
                while (len(self._per_second_history)
                        > self._per_second_history_max_buckets):
                    self._per_second_history.popitem(last=False)
            sec_ts = pd.Timestamp(ts)
            if sec_ts.tzinfo is None:
                sec_ts = sec_ts.tz_localize('UTC')
            bucket.append({
                'timestamp': sec_ts,
                'open': sec_open, 'high': sec_high,
                'low': sec_low, 'close': sec_close,
                'volume': sec_volume,
            })

        # M8.7 rebroadcast handling (2026-05-02): if the incoming bar's
        # period_start matches the most recent history row, this is a
        # Polygon WS rebroadcast (correction within the 15-min FINRA
        # late-print window) for the most recently closed bar. Replace
        # the row in place. Don't pollute the current partial. Only
        # applies to close_on_boundary=True mode (where we maintain
        # canonical history); for chart-visual mode (False) we don't
        # have history to replace.
        if close_on_boundary and len(self.history) > 0:
            last_ts = self.history.index[-1]
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize('UTC')
            if last_ts == pd.Timestamp(period_start):
                # 2026-06-12 (clobber merge fix): a SUB-PERIOD input (a
                # single 1Min bar from the AM/ws_agg fan-out into a
                # >=120s builder) must MERGE into the closed row, never
                # replace it. flush_stale_bars (zero grace for 1Min+)
                # almost always closes the period before its final minute
                # can be delivered; that late minute lands here. The old
                # wholesale replace was the 2m clobber (volume 0.50x
                # REST, range collapsed to one minute); the 247afec/
                # 0d767d9 drop-guard starved live_bars writes instead
                # (the "freeze" — this branch's return was the only
                # >=120s write path). Merge keeps both correct.
                if not full_period_input:
                    row_idx = self.history.index[-1]
                    row = self.history.iloc[-1]
                    _sub_ts = pd.Timestamp(ts)
                    _same_period = (self._closed_period_start is not None
                                    and self._closed_period_start
                                    == pd.Timestamp(period_start))
                    _seen = (self._closed_seen_subbars
                             if _same_period else None)
                    _hi = max(float(row['high']), sec_high)
                    _lo = min(float(row['low']), sec_low)
                    _open = float(row['open'])
                    _close = float(row['close'])
                    _vol = float(row['volume'])
                    if _seen is not None:
                        # constituent-ordered open/close attribution
                        if _sub_ts <= self._closed_min_subbar_ts:
                            _open = sec_open
                            self._closed_min_subbar_ts = _sub_ts
                        if _sub_ts >= self._closed_max_subbar_ts:
                            _close = sec_close
                            self._closed_max_subbar_ts = _sub_ts
                        if not skip_volume and _sub_ts not in _seen:
                            _vol += sec_volume
                        _seen.add(_sub_ts)
                    # else: no constituent info for this row (REST-seeded
                    # or post-restart) — REST values are full-period
                    # truth; merge prices conservatively, never volume.
                    _vals = [_open, _hi, _lo, _close, _vol]
                    _old = [float(row[c]) for c in
                            ('open', 'high', 'low', 'close', 'volume')]
                    _changed = any(a != b for a, b in zip(_vals, _old))
                    if _changed:
                        self.history.loc[
                            row_idx,
                            ['open', 'high', 'low', 'close', 'volume']
                        ] = _vals
                    self.last_was_duplicate = True
                    # Recompute at most once per bucket, and only when
                    # the merge actually changed values (lag-spiral
                    # protection, 2026-05-19).
                    _bts = pd.Timestamp(period_start)
                    if (_changed
                            and (self._last_rebroadcast_recompute_ts is None
                                 or _bts >= self._last_rebroadcast_recompute_ts)):
                        self._last_rebroadcast_recompute_ts = _bts
                        self.last_was_correction = True
                    else:
                        self.last_was_correction = False
                    return {
                        'timestamp': period_start.isoformat(),
                        'open': _vals[0], 'high': _vals[1],
                        'low': _vals[2], 'close': _vals[3],
                        'volume': _vals[4],
                    }
                # Full-period input: replace history row with corrected
                # aggregation values. The CALLER (on_second_bar sub-minute
                # primary) is feeding the post-correction full-period
                # aggregation; we trust those values.
                #
                # 2026-05-13 fix: assign per-column instead of via
                # `iloc[-1] = pd.Series(5-keys)` because seeded history
                # from REST often carries extra columns (vwap,
                # trade_count, etc.). The 5-key Series broke with
                # "Must have equal len keys and value when setting with
                # an iterable" once secondary builders started getting
                # seeded by the hot-reload path.
                _cols = ['open', 'high', 'low', 'close', 'volume']
                _new = [sec_open, sec_high, sec_low, sec_close, sec_volume]
                self.history.loc[self.history.index[-1], _cols] = _new
                self.last_was_duplicate = True
                # Recompute at most once per bucket — see __init__ note.
                _bts = pd.Timestamp(period_start)
                if (self._last_rebroadcast_recompute_ts is None
                        or _bts > self._last_rebroadcast_recompute_ts):
                    self._last_rebroadcast_recompute_ts = _bts
                    self.last_was_correction = True
                else:
                    self.last_was_correction = False
                # Return the corrected bar dict so caller knows this was a
                # close-on-boundary (treats it like a completed-bar return).
                return {
                    'timestamp': period_start.isoformat(),
                    'open': sec_open, 'high': sec_high, 'low': sec_low,
                    'close': sec_close, 'volume': sec_volume,
                }

        if self._partial is None or period_start > self._partial.bar_start:
            completed = None
            if self._partial is not None and close_on_boundary:
                completed = self._close_bar()
            # else: discard prior partial (chart-visual mode for 1Min+)
            self._partial = PartialBar(
                sec_open, period_start, self.tf_seconds)
            self._partial.high = sec_high
            self._partial.low = sec_low
            self._partial.close = sec_close
            # Volume integrity (2026-06-12): skip_volume callers (the
            # 1Min+ per-second chart-visual path) must never contribute
            # volume — the AM/ws_agg fan-out is the canonical volume
            # source, and double-feeding inflated every >60s bar ~2x
            # (the documented VWAP_V2/RVOL_V2 caveat; live RVOL gate
            # read EXTREME near-constantly → sid 291 at 1.0%).
            self._partial.volume = 0.0 if skip_volume else sec_volume
            self._partial.tick_count = 1
            # Track which sub-bar timestamps have contributed VOLUME to
            # THIS partial, so the same sub-bar delivered twice (AM and
            # ws_agg fan-outs both feed the same minute into >=120s
            # builders) counts its volume exactly once. skip_volume
            # callers (per-second chart-visual on 1Min+ primaries) must
            # NOT register — their :00 second's ts collides with the
            # fan-out minute's ts and would silently drop the minute's
            # volume (2026-06-12).
            self._partial_seen_subbars = (
                set() if skip_volume else {pd.Timestamp(ts)})
            self.last_was_duplicate = False
            return completed

        # Late bar: period_start is OLDER than the current partial — its
        # bucket already closed. Drop it. Aggregating it into the current
        # partial would misattribute its volume/price to the wrong period
        # — the cause of flat empty sub-minute bars (the real bucket
        # closed empty) plus volume dumped into a later bucket, under
        # WebSocket delivery jitter. A late bar for the immediately-
        # previous closed bar is already handled above as a rebroadcast
        # correction; this catches anything older.
        if period_start < self._partial.bar_start:
            self.last_was_duplicate = False
            return None

        # Same period — aggregate into existing partial
        self._partial.high = max(self._partial.high, sec_high)
        self._partial.low = min(self._partial.low, sec_low)
        self._partial.close = sec_close
        # Volume integrity (2026-06-12): count each sub-bar's volume at
        # most once per partial (AM + ws_agg fan-outs deliver the SAME
        # minute to >=120s builders), and never from skip_volume callers
        # (per-second chart-visual on 1Min+ primaries).
        _sub_ts = pd.Timestamp(ts)
        if not hasattr(self, '_partial_seen_subbars'):
            self._partial_seen_subbars = set()
        if not skip_volume:
            if _sub_ts not in self._partial_seen_subbars:
                self._partial.volume += sec_volume
            # Only volume-contributing callers register: skip_volume
            # (chart-visual seconds) polluting the set dropped the
            # colliding :00 fan-out minute's volume (2026-06-12).
            self._partial_seen_subbars.add(_sub_ts)
        self._partial.tick_count += 1
        self.last_was_duplicate = False
        return None

    @property
    def partial_bar(self) -> Optional[PartialBar]:
        return self._partial

    def get_df_with_partial(self) -> pd.DataFrame:
        if self._partial is None or len(self.history) == 0:
            return self.history.copy()
        bar = self._partial.to_dict()
        ts = pd.Timestamp(bar['timestamp'])
        partial_row = pd.DataFrame(
            [{k: v for k, v in bar.items() if k != 'timestamp'}],
            index=pd.DatetimeIndex([ts], name='timestamp'),
        )
        return pd.concat([self.history, partial_row])

    def _align_to_period(self, ts: datetime) -> datetime:
        epoch = int(ts.timestamp())
        aligned = epoch - (epoch % self.tf_seconds)
        return datetime.fromtimestamp(aligned, tz=timezone.utc)

    def _close_bar(self) -> dict:
        bar = self._partial.to_dict()
        self._append_to_history(bar)
        self._bar_count += 1
        # Retain constituent bookkeeping so late sub-bar deliveries for
        # this just-closed period can merge with correct volume dedup and
        # open/close attribution (see __init__ note, 2026-06-12).
        seen = getattr(self, '_partial_seen_subbars', None)
        if seen:
            self._closed_seen_subbars = set(seen)
            self._closed_min_subbar_ts = min(seen)
            self._closed_max_subbar_ts = max(seen)
        else:
            self._closed_seen_subbars = None
            self._closed_min_subbar_ts = None
            self._closed_max_subbar_ts = None
        self._closed_period_start = pd.Timestamp(bar['timestamp'])
        return bar

    def _append_to_history(self, bar_dict: dict):
        ts = pd.Timestamp(bar_dict['timestamp'])
        new_row = pd.DataFrame(
            [{k: v for k, v in bar_dict.items() if k != 'timestamp'}],
            index=pd.DatetimeIndex([ts], name='timestamp'),
        )
        if len(self.history) == 0:
            self.history = new_row
        else:
            self.history = pd.concat([self.history, new_row])
        if len(self.history) > MAX_HISTORY:
            self.history = self.history.iloc[-MAX_HISTORY:]

    @_prof_fn('b_accept_bar')
    def accept_bar(self, bar_dict: dict) -> dict:
        """Accept a pre-built bar (from Polygon WebSocket).

        Appends to history, increments bar count. Empty periods produce NO
        bar (no gap-fill) when BAR_GAP_FILL_ENABLED is False — matches the
        backtest's resample().dropna(). Returns the bar dict for downstream.

        M8.7 rebroadcast handling (2026-05-02): if the incoming bar's
        bar_start matches the most recent history row's timestamp, this
        is a Polygon WS rebroadcast (correction within the 15-min FINRA
        late-print window). We REPLACE the history row in place and set
        `self.last_was_duplicate=True` for the caller. Otherwise we
        append as before and set `last_was_duplicate=False`.
        """
        bar_ts = pd.Timestamp(bar_dict['timestamp'])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize('UTC')
        bar_start = self._align_to_period(bar_ts.to_pydatetime())

        # Rebroadcast detection: same bar_start as the most recent history
        # row. Replace, don't append. Don't increment bar_count.
        if len(self.history) > 0:
            last_ts = self.history.index[-1]
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize('UTC')
            if last_ts == pd.Timestamp(bar_start):
                # Replace the row's OHLCV with the corrected values.
                # 2026-05-13: column-targeted assignment (see
                # accept_second_bar above for the same fix).
                _cols = ['open', 'high', 'low', 'close', 'volume']
                _new = [
                    float(bar_dict['open']),
                    float(bar_dict['high']),
                    float(bar_dict['low']),
                    float(bar_dict['close']),
                    float(bar_dict.get('volume', 0)),
                ]
                self.history.loc[self.history.index[-1], _cols] = _new
                self.last_was_duplicate = True
                # Recompute at most once per bucket — see __init__ note.
                _bts = pd.Timestamp(bar_start)
                if (self._last_rebroadcast_recompute_ts is None
                        or _bts > self._last_rebroadcast_recompute_ts):
                    self._last_rebroadcast_recompute_ts = _bts
                    self.last_was_correction = True
                else:
                    self.last_was_correction = False
                return bar_dict

        # Gap-fill (2026-06-10): only when explicitly enabled. Empty periods
        # otherwise produce NO bar — matching backtest dropna. The flat
        # carry-forward bars polluted the incremental indicators after-hours.
        if BAR_GAP_FILL_ENABLED and len(self.history) > 0:
            last_ts = self.history.index[-1]
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize('UTC')
            last_period = self._align_to_period(last_ts.to_pydatetime())
            expected_next = last_period + timedelta(seconds=self.tf_seconds)
            fill_close = self.history['close'].iloc[-1]
            gap_ts = expected_next
            while gap_ts < bar_start:
                self._append_to_history({
                    'timestamp': gap_ts.isoformat(),
                    'open': fill_close, 'high': fill_close,
                    'low': fill_close, 'close': fill_close,
                    'volume': 0,
                })
                self._bar_count += 1
                gap_ts += timedelta(seconds=self.tf_seconds)

        # Append the actual bar
        self._append_to_history(bar_dict)
        self._bar_count += 1
        # Clear partial since this bar is complete
        self._partial = None
        self.last_was_duplicate = False
        return bar_dict

    def force_close_stale_bar(self, now: datetime) -> Optional[dict]:
        """Close the partial bar if wall-clock time has passed bar_end.

        Sub-minute bars get a grace window past bar_end before force-close
        to absorb Polygon's ~3-7s A-channel ingestion lag. Without it,
        force_close fired BEFORE the last per-second data could arrive
        and emitted a flat carry-forward bar (Bug B — the SPY/TSLA 10Sec
        empty-bar symptom from 2026-05-18). 1Min+ doesn't need a grace
        — cache vs REST is already 100% there (see Backtest_Speed_
        Analysis_2026-05-19.md §3).

        Grace = 5s is the value picked from the 2026-05-19 RTH grace
        sweep: TSLA 95-100% exact vs REST, SPY 87% within $0.10 / 100%
        within $0.25 (unbiased microstructure noise). Diminishing
        returns past grace 5. Future: per-strategy parameter via the
        Live Execution Fidelity spec (`ws_agg_reconciled` live_model).

        Returns the completed bar dict, or None if no bar was stale.
        No gap-fill (2026-06-10): empty periods produce no bar (matches
        backtest dropna) unless BAR_GAP_FILL_ENABLED.
        """
        if self._partial is None:
            return None
        bar_end = self._partial.bar_start + timedelta(seconds=self.tf_seconds)
        # Sub-minute grace: per-builder override (set by
        # RalphEngine._refresh_builder_grace from max(active monitor
        # grace_seconds)) if present, else the global default. 1Min+
        # ignores grace regardless.
        if self.tf_seconds < 60:
            grace_sec = (self.grace_sec_override
                         if self.grace_sec_override is not None
                         else SUBMINUTE_FORCE_CLOSE_GRACE_SEC)
        else:
            grace_sec = 0
        if now < bar_end + timedelta(seconds=grace_sec):
            return None  # Bar still forming (or within sub-minute grace)
        fill_close = self._partial.close
        completed = self._close_bar()
        if BAR_GAP_FILL_ENABLED:
            # Fill gap bars between bar_end and current period
            current_period = self._align_to_period(now)
            gap_ts = bar_end
            while gap_ts < current_period:
                self._append_to_history({
                    'timestamp': gap_ts.isoformat(),
                    'open': fill_close, 'high': fill_close,
                    'low': fill_close, 'close': fill_close,
                    'volume': 0,
                })
                self._bar_count += 1
                gap_ts += timedelta(seconds=self.tf_seconds)
            # Start a new partial bar at the current period using last known
            # close, so that arriving ticks will update it normally.
            if current_period >= bar_end:
                self._partial = PartialBar(fill_close, current_period,
                                           self.tf_seconds)
        else:
            # No gap-fill: the real bar is closed; a fresh partial will be
            # seeded by the next real per-second bar / tick at its own period.
            self._partial = None
        return completed


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY MONITOR — one per strategy, owns indicator + trigger + position
# ═══════════════════════════════════════════════════════════════════════════

# Fields whose values define a monitor's behavior. The worker's hot-reload
# (see worker.py) compares the hash before/after to detect config changes
# on RUNNING strategies and re-instantiate the monitor — pre-2026-05-20
# the hot-reload only diffed by ID, so a UI edit to a running strategy's
# config silently did nothing until the next deploy.
# Lookup is permissive: each field is checked at top level first, then
# inside `config`, because different load paths flatten the JSONB
# differently. Volatile fields (stored_trades, kpis, equity_curve_data,
# live_executions, discrepancies, data_refreshed_at) are intentionally
# excluded so a normal data refresh doesn't trigger a re-instantiation.
_MONITOR_CONFIG_FIELDS = (
    'live_model', 'timeframe', 'symbol', 'direction',
    'entry_trigger_confluence_id', 'entry_trigger',
    'exit_trigger', 'exit_triggers', 'confluence',
    'stop_config', 'target_config', 'stop_atr_mult',
    'risk_per_trade', 'starting_balance', 'position_size',
    'trading_session', 'data_seed', 'strategy_origin',
    'grace_seconds',
    'alert_tracking_enabled',
)


def _monitor_config_hash(strategy: dict) -> str:
    """Short stable hash of the fields that affect monitor behavior.

    Worker hot-reload uses this to detect config changes on existing
    monitors and re-instantiate them. 12-char hex is plenty.
    """
    import hashlib
    import json
    cfg = strategy.get('config') or {}
    parts: dict = {}
    for f in _MONITOR_CONFIG_FIELDS:
        if f in strategy:
            parts[f] = strategy[f]
        elif f in cfg:
            parts[f] = cfg[f]
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


class StrategyMonitor:
    """Monitors one strategy: indicators → triggers → position state."""

    def __init__(self, strategy: dict,
                 position_state: Optional[PositionState] = None,
                 general_packs: Optional[list] = None):
        self.strategy = strategy
        self.strat_id = strategy['id']
        self.strat_name = strategy.get('name', f'Strategy {self.strat_id}')
        self.symbol = strategy.get('symbol', 'SPY')
        # user_id propagates from strategy row (DB) or is stamped by worker.
        # Used by SymbolHub to route Supabase Realtime live-bar broadcasts.
        self.user_id = strategy.get('user_id', '')
        self.tf_seconds = TIMEFRAME_SECONDS.get(
            strategy.get('timeframe', '1Min'), 60)
        # Crypto symbols trade 24/7 — override session
        if '/' in self.symbol:
            self.session = '24/7'
        else:
            self.session = strategy.get('trading_session', 'RTH')

        # Phase C (2026-05-05): live_model selection drives bar-source
        # routing in _run_monitor_pipeline_for_completed_bar's gate.
        # `ws_with_corrections` (default) consumes Polygon AM events only;
        # `ws_agg_locked` / `ws_agg_with_rest_backfill` consume client-side
        # 1Min bars built from per-second A.* events (lock at minute close).
        # Strategies with stale or unknown live_model values fall back to
        # the platform default — never crash on a deleted experimental ID.
        from strategy_models import (
            get_default_live_model, is_valid_live_model,
            resolve_grace_seconds,
        )
        declared = strategy.get('live_model') or strategy.get('config', {}).get('live_model')
        self.live_model = (
            declared if (declared and is_valid_live_model(declared))
            else get_default_live_model()
        )
        # LEF Phase 2b (2026-05-20): cached strategy-level grace + the
        # "already-fired this bucket" tracker.
        #
        # `grace_seconds` is consumed by the new fire-at-strategy-grace
        # path; `_fired_bucket` is the epoch-second bar_start of the
        # most recent bucket this monitor fired on via fire_on_partial_
        # bucket, used to suppress double-fire within a single bucket.
        # Both are NO-OPs for live_model != 'ws_agg_reconciled' — the
        # existing on_bar_close path stays the source of truth.
        self.grace_seconds: int = resolve_grace_seconds(strategy)
        self._fired_bucket: Optional[int] = None

        # Resolve requirements (uses confluence mapping for trigger IDs)
        req_ind, req_interp, req_trig, params = (
            resolve_strategy_requirements(strategy))

        # Resolve entry/exit trigger IDs through confluence mapping
        resolved_entry = _resolve_trigger_id(strategy, 'entry_trigger')
        resolved_exits = _resolve_trigger_ids(strategy, 'exit_triggers')
        if not resolved_exits:
            exit_single = _resolve_trigger_id(strategy, 'exit_trigger')
            if exit_single:
                resolved_exits = [exit_single]

        self.indicators = IncrementalIndicatorEngine(req_ind, params)
        self.trigger_eval = TriggerEvaluator(
            req_interp, req_trig, params['ema_periods'])

        # Tier 3 §8.3 (always-start-flat, 2026-05-20): a restart-context
        # `position_state` carrying IN_POSITION is NOT inherited. The
        # engine starts FLAT regardless; the inherited state is captured
        # to `_tier3_inherited_carryover` for the worker to persist as a
        # `position_carryover` entry (§4.2). Operators see the broker may
        # still hold the underlying position — surfaced via the Data
        # Fidelity / Configuration tab Open Trade Carryover card.
        #
        # When position_state is None or FLAT, no carryover is recorded —
        # this is the cold-start case (worker first boot, brand-new
        # strategy, or already-flat at last shutdown).
        self._tier3_inherited_carryover: Optional[dict] = None
        if position_state is not None and getattr(
                position_state, 'status', 'FLAT') == 'IN_POSITION':
            from dataclasses import asdict as _asdict
            try:
                carryover_data = _asdict(position_state)
            except TypeError:
                # Defensive: position_state might not be a dataclass
                # under some test paths. Convert via __dict__.
                carryover_data = dict(getattr(position_state, '__dict__', {}))
            self._tier3_inherited_carryover = {
                'type': 'position_carryover',
                'strategy_id': self.strat_id,
                'recorded_at': datetime.now(timezone.utc).isoformat(),
                # Audit fix 2026-05-20: capture broker-relevant fields
                # too so the user has the full picture for a manual
                # close (stop + target + age) — not just the entry.
                'entry_time': carryover_data.get('entry_time'),
                'entry_price': carryover_data.get('entry_price'),
                'stop_price': carryover_data.get('stop_price'),
                'target_price': carryover_data.get('target_price'),
                'entry_bar_count': carryover_data.get('entry_bar_count'),
                'direction': carryover_data.get('direction'),
                'entry_trigger': carryover_data.get('entry_trigger'),
                'reason': 'tier3_restart_flat',
            }
            logger.warning(
                "Tier 3 MIGRATION sid=%s user=%s (%s/%ds): pre-restart "
                "position was IN_POSITION (entry_price=%s, stop_price=%s, "
                "target_price=%s, entry_time=%s) — engine now starts FLAT "
                "per always-start-flat contract. Broker may still hold "
                "the position; verify and close manually if needed. "
                "Carryover queued for persistence.",
                self.strat_id, (self.user_id[:8] if self.user_id else '?'),
                self.symbol, self.tf_seconds,
                carryover_data.get('entry_price'),
                carryover_data.get('stop_price'),
                carryover_data.get('target_price'),
                carryover_data.get('entry_time'))
        # Pass None to PositionStateMachine — engine ALWAYS starts FLAT
        # at construction. The Tier 3 contract is enforced here, not
        # downstream.
        self.position = PositionStateMachine(
            strategy, None,
            resolved_entry=resolved_entry,
            resolved_exits=resolved_exits)

        # Confluence records for current bar
        self._current_confluence: Set[str] = set()
        self._interp_keys = list(req_interp)

        # Bar-diagnostics (#57C, 2026-06-05). Resolve declared columns
        # once at init — by-strategy lookup that walks the pack manifests.
        # Empty list ⇒ logging disabled cleanly for this strategy. Buffer
        # writes flush every _DIAG_FLUSH_EVERY bars in on_bar_close to
        # keep the hot tick path lean.
        self._diag_buffer: List[dict] = []
        self._diag_cols: List[str] = []
        # Lower threshold than backtest (60) — live engine has no end-of-run
        # flush, so we want diagnostic rows visible within ~5 minutes of
        # the bar closing. At 10Sec primary TF: 30 bars × 10s = 5 min.
        self._DIAG_FLUSH_EVERY = 30
        try:
            from pack_registry import get_diagnostic_columns_for_strategy
            self._diag_cols = get_diagnostic_columns_for_strategy(strategy) or []
        except Exception as _diag_e:
            logger.warning("ralph bar_diagnostics setup sid=%s failed: %s",
                           self.strat_id, _diag_e)
            self._diag_cols = []

        # General Pack references (for GEN- confluence records).
        # Only keep packs referenced by this strategy's general_confluences.
        # Match by the full `GEN-{PACK_ID}-` prefix, NOT split('-')[1] — pack
        # ids can contain dashes (e.g. 'time_of_day-11:30-14:00'), so token[1]
        # would yield just 'TIME_OF_DAY' and never match the real id → the pack
        # is silently dropped and the gate never fires live (while the backtest,
        # which evaluates all packs, still has it → backtest↔live divergence).
        _needed_recs = [r for r in strategy.get('general_confluences', [])
                        if isinstance(r, str) and r.startswith('GEN-')]
        self._general_packs = [
            p for p in (general_packs or [])
            if any(r.upper().startswith(f'GEN-{p.id.upper()}-')
                   for r in _needed_recs)
        ] if _needed_recs else []

        # Normalize confluence labels to uppercase (backtest may store lowercase)
        if self.position.confluence_set:
            self.position.confluence_set = {
                _normalize_confluence_label(r)
                for r in self.position.confluence_set
            }

        # Parse required secondary TFs from confluence records
        self._required_secondary_tf: Set[int] = set()
        all_conf = (list(strategy.get('confluence', []))
                    + list(strategy.get('general_confluences', [])))
        for rec in all_conf:
            if rec.startswith('GEN-'):
                continue
            tf_part = rec.split('-', 1)[0]
            sec = _LABEL_TO_TF_SECONDS.get(tf_part, 0)
            if sec and sec != self.tf_seconds:
                self._required_secondary_tf.add(sec)

        # M8.5 B+: populate _intrabar_triggers from resolved entry/exit
        # triggers, filtered to those known L-type (intra-bar level cross)
        # triggers from `_IB_L_TYPE_TRIGGERS`. Used by the WS subscription
        # logic and by intra-bar evaluation. Was never set before, which
        # silently broke L-type live triggers entirely.
        self._intrabar_triggers: Set[str] = set()
        candidate_triggers = set()
        if resolved_entry:
            candidate_triggers.add(resolved_entry)
        if resolved_exits:
            candidate_triggers.update(resolved_exits)
        for t in candidate_triggers:
            if t in _IB_L_TYPE_TRIGGERS:
                self._intrabar_triggers.add(t)

        logger.info("StrategyMonitor: %s (%s/%ds) — indicators=%s, "
                     "triggers=%s, interpreters=%s, entry=%s, exits=%s%s, "
                     "ema_periods=%s",
                     self.strat_name, self.symbol, self.tf_seconds,
                     req_ind, req_trig, req_interp,
                     resolved_entry, resolved_exits,
                     f", secondary_tfs={self._required_secondary_tf}"
                     if self._required_secondary_tf else "",
                     params.get('ema_periods'))

        # Hash of the behavior-driving config fields — the worker's
        # hot-reload (worker.py db_hot_reload) compares this before/after
        # on each cycle and re-instantiates the monitor when it changes,
        # so live edits (live_model / grace_seconds / triggers / stops)
        # activate without a deploy.
        self._config_hash: str = _monitor_config_hash(strategy)

    def warmup(self, df: pd.DataFrame):
        """Initialize indicator state from historical bars."""
        self.indicators.warmup(df)
        # Live engine: enable pre-bar snapshotting so a Polygon rebroadcast
        # correction resolves in O(1) (apply_last_bar_correction) rather
        # than an O(N) recompute_from_history replay.
        self.indicators._snapshot_enabled = True

    @_prof_fn('m_on_bar_close')
    def on_bar_close(self, bar: dict,
                     bar_count: int,
                     mtf_confluence: Dict[Tuple[int, str], Set[str]] = None,
                     ) -> Tuple[List[dict], dict]:
        """Process a completed bar.

        Args:
            mtf_confluence: Cross-TF confluence buffer from SymbolHub.
                Maps (tf_seconds, session) → latest confluence records
                from other TFs (session is 'RTH' unless
                RORT_MTF_SESSION_SHADOWS — Bug Hunt Wave 1 #1).

        Returns:
            (signals, audit_data) — signals list and dict for fidelity logging.
        """
        signals = []

        # 1. Update indicators (O(1))
        current = self.indicators.update_bar(bar)
        prev = self.indicators.get_prev_values()

        # 1a. Bar-diagnostics capture (#57C). Project declared columns
        # from `current`; buffer; flush every N bars. Skipped if pack
        # didn't declare diagnostic_columns. Best-effort — never blocks
        # the tick path. Source='live' distinguishes from backtest/algo
        # writes coming through unified_engine.
        if self._diag_cols:
            _vals: dict = {}
            for _c in self._diag_cols:
                _v = current.get(_c)
                if _v is None:
                    continue
                try:
                    _fv = float(_v)
                    if _fv == _fv and _fv not in (float('inf'), float('-inf')):
                        _vals[_c] = _fv
                except (TypeError, ValueError):
                    continue
            if _vals:
                _bts = bar.get('timestamp')
                if _bts is not None:
                    _bts_iso = (_bts.isoformat()
                                if hasattr(_bts, 'isoformat') else str(_bts))
                    self._diag_buffer.append({
                        'strategy_id': self.strat_id,
                        'bar_ts': _bts_iso,
                        'source': 'live',
                        'values': _vals,
                    })
                    if len(self._diag_buffer) >= self._DIAG_FLUSH_EVERY:
                        try:
                            from db import save_bar_diagnostics_batch
                            save_bar_diagnostics_batch(self._diag_buffer)
                        except Exception as _diag_e:
                            logger.warning(
                                "ralph bar_diagnostics flush sid=%s failed: %s",
                                self.strat_id, _diag_e)
                        self._diag_buffer.clear()

        # 1b. Feed high/low into position buffer (for swing stops/targets)
        self.position.update_high_low(bar['high'], bar['low'])

        # 2. Evaluate triggers
        interps, trigger_bools = self.trigger_eval.evaluate_bar_close(
            current, prev, self.indicators.state.prev2_macd_hist)

        # 3. Build confluence records
        tf_label = SECONDS_TO_TIMEFRAME.get(self.tf_seconds, '1Min')
        # Shorten label for confluence format (1Min→1M, 5Min→5M, etc.)
        short_label = tf_label.replace('Min', 'M').replace(
            'Hour', 'H').replace('Day', 'D')
        self._current_confluence = set()
        for ikey, state_val in interps.items():
            self._current_confluence.add(
                f'{short_label}-{ikey}-{state_val}')

        # 3b. Evaluate General Pack conditions (time/calendar filters)
        if self._general_packs:
            bar_ts = bar.get('timestamp')
            # BarBuilder.to_dict() emits 'timestamp' as an ISO STRING
            # (self.bar_start.isoformat()), so the live bar-close path passes
            # a str here — not a datetime. The old isinstance(datetime) guard
            # was therefore always False live, so _evaluate_general_packs was
            # never called, no GEN- record entered _current_confluence, and
            # EVERY general-pack-gated strategy was silently blocked live
            # (the backtest path uses a datetime index, so it worked → the
            # tz fix only ever helped backtest). Coerce string → datetime so
            # gen-gating actually fires live. Matches the pd.Timestamp parse
            # pattern used elsewhere for bar['timestamp'].
            if isinstance(bar_ts, str) and bar_ts:
                try:
                    bar_ts = pd.Timestamp(bar_ts).to_pydatetime()
                except Exception:
                    bar_ts = None
            if isinstance(bar_ts, datetime):
                self._current_confluence |= _evaluate_general_packs(
                    self._general_packs, bar_ts)

        # 3c. Merge cross-TF confluence records from hub's shared buffer.
        # Bug Hunt Wave 1 #1: merge ONLY records keyed to this monitor's
        # own session ('RTH' with the flag off — legacy behavior, since
        # every key is then ('tf', 'RTH')). A union across sessions would
        # let e.g. an Extended-Hours 1d state satisfy an RTH monitor's
        # gate (and vice versa) — the exact cross-contamination the
        # session-keyed shadows exist to prevent.
        if mtf_confluence:
            _own_sess = (self.session if MTF_SESSION_SHADOWS else 'RTH')
            for (other_tf, _sess), records in mtf_confluence.items():
                if other_tf != self.tf_seconds and _sess == _own_sess:
                    self._current_confluence |= records

        # Use bar_start timestamp for bar-close signals — matches backtest
        # convention (df.index = bar-start) and chart candle positioning.
        raw_ts = bar.get('timestamp', '')
        if isinstance(raw_ts, str) and raw_ts:
            bar_time = raw_ts
        elif isinstance(raw_ts, datetime):
            bar_time = raw_ts.isoformat()
        else:
            bar_time = ''

        # 4. Handle pending HM exit at bar open price
        if self.position.state.pending_hm_exit:
            exit_sig = self.position._signal_exit(
                'unconfirmed_hm', bar.get('open', current.get('open', 0)),
                bar_time, bar_count)
            exit_sig['atr'] = current.get('atr', 0)
            signals.append(exit_sig)

        # 5. Check exit (before entry — same bar exit takes priority)
        exit_sig = self.position.check_exit_bar_close(
            trigger_bools, current, bar_count, bar_time)
        if exit_sig:
            exit_sig['atr'] = current.get('atr', 0)
            signals.append(exit_sig)

        # 6. Check entry (only if FLAT after exit check)
        entry_sig = self.position.check_entry(
            trigger_bools, current, bar_count, bar_time,
            confluence_records=self._current_confluence)
        if entry_sig:
            signals.append(entry_sig)
            # GATE_DIAG (temporary, iter 0609a): when a GATED strategy fires
            # an entry, record whether the gate was actually satisfied.
            # subset_ok=0 => fired ungated (fail-open path); subset_ok=1 =>
            # gate passed legitimately (=> live/BT shadow divergence, NOT
            # fail-open). Remove after diagnosis.
            if self.position.confluence_set:
                _recs = self._current_confluence
                _ok = self.position.confluence_set.issubset(_recs)
                logger.info(
                    "GATE_DIAG strat=%s fired entry; gate=%s subset_ok=%d "
                    "records=%s mtf_keys=%s",
                    self.strat_id, sorted(self.position.confluence_set),
                    1 if _ok else 0, sorted(_recs),
                    sorted(mtf_confluence.keys()) if mtf_confluence else None)
        elif (self.position.state.status == 'FLAT'
              and self.position.confluence_set
              and any(trigger_bools.values())):
            # XTF_BLOCK_DIAG (2026-06-23): a gated entry trigger fired while
            # FLAT but produced no entry signal — the confluence gate suppressed
            # it. Logs needed confluence_set vs what's actually in
            # _current_confluence, plus the cross-TF (mtf) buffer keys — to pin
            # whether the secondary-TF records (e.g. 2m/5m SWING_123 for
            # 327/328/329) are ABSENT (broken cross-TF feed) or PRESENT-but-not
            # -matching. Mirrors the gen-gate diagnosis. Remove after diagnosis.
            _recs = self._current_confluence
            _need = self.position.confluence_set
            logger.info(
                "XTF_BLOCK_DIAG strat=%s active=%s need=%s subset_ok=%d "
                "present=%s mtf_keys=%s",
                self.strat_id, [k for k, v in trigger_bools.items() if v],
                sorted(_need), 1 if _need.issubset(_recs) else 0,
                sorted(_recs), sorted(mtf_confluence.keys()) if mtf_confluence else None)

        # 7. Confirmation check for HM/HL entries made THIS bar
        if (self.position.state.status == 'IN_POSITION' and
                self.position.state.entry_bar_count == bar_count and
                self.position.state.exec_type in ('HM', 'HL')):
            base_trigger = _strip_exec_suffix(
                self.position.state.entry_trigger)
            if not self.trigger_eval.check_confirmation(
                    base_trigger, current):
                if self.position.state.exec_type == 'HM':
                    self.position.state.pending_hm_exit = True
                else:
                    self.position.state.pending_hl_limit = True

        # 8. Audit data for fidelity logger
        audit_data = {
            'indicator_values': current,
            'trigger_booleans': trigger_bools,
            'interpreter_states': interps,
            'position_state': self.position.state.status,
        }

        return signals, audit_data

    def compute_tentative_state(
        self, forming_bar: dict,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Run indicator + interpreter pipeline on a forming bar without
        mutating committed engine state. Snapshots IndicatorState, applies
        the bar as if closed, reads interpreter outputs, then restores.

        Returns (indicator_values, interpreter_states). If the engine isn't
        warmed up yet, returns empty dicts — caller should treat that as
        "no live data available yet for this strategy."

        Used by SymbolHub to pack live heatmap / oscillator / overlay values
        into the forming-bar broadcast so the frontend can animate all panes
        in sync with the price candle.
        """
        if not self.indicators._initialized:
            return {}, {}

        import copy as _copy
        state_snapshot = _copy.deepcopy(self.indicators.state)
        init_snapshot = self.indicators._initialized
        # The forming-bar update_bar() below overwrites _pre_bar_snapshot
        # (it snapshots at entry). Hold the real pre-bar token and restore
        # it, or a later Polygon rebroadcast correction would rewind to
        # the wrong (post-bar) state. Reference hold is enough — update_bar
        # replaces the attribute, never mutates the token in place.
        prebar_snapshot = getattr(
            self.indicators, '_pre_bar_snapshot', None)
        # User-pack incremental classes hold their own internal state (e.g.
        # UtBotV4Incremental's _trail_stop / _prev_close). Forming-bar
        # tentative computation MUST snapshot/restore these alongside the
        # built-in state — otherwise per-second forming-bar updates pollute
        # `_prev_close` and the next real bar's flip detection breaks
        # silently. This was the root cause of strategy 132 (ut_bot_v4)
        # never firing live alerts despite the indicator computing
        # correctly per bar.
        user_pack_snapshot = {
            slug: _copy.deepcopy(eng)
            for slug, eng in getattr(self.indicators,
                                     '_user_pack_engines', {}).items()
        }
        try:
            current = self.indicators.update_bar(forming_bar)
            prev = self.indicators.get_prev_values()
            interps, _triggers = self.trigger_eval.evaluate_bar_close(
                current, prev, self.indicators.state.prev2_macd_hist)
        except Exception:
            interps = {}
            current = {}
        finally:
            self.indicators.state = state_snapshot
            self.indicators._initialized = init_snapshot
            if user_pack_snapshot:
                self.indicators._user_pack_engines = user_pack_snapshot
            self.indicators._pre_bar_snapshot = prebar_snapshot
        return current, interps

    @_prof_fn('m_fire_partial')
    def fire_on_partial_bucket(self, partial_bar: dict,
                                bar_count: int,
                                mtf_confluence=None) -> tuple:
        """Fire on a still-forming sub-minute bucket at strategy-grace.

        LEF Phase 2b (2026-05-20). Snapshots indicator state and the
        user-pack engines, runs the FULL `on_bar_close` path on
        `partial_bar` (which evaluates triggers + may mutate position),
        then RESTORES the indicator state — leaving the position state
        machine's transition intact (a fired alert means we're in
        position; that must persist) but rolling back the per-bar
        indicator commits so the official commit happens later at the
        builder's bar-close with the FINAL aggregate.

        Returns (signals, audit_data) like on_bar_close — the caller
        dispatches alerts.

        Failure mode: on any exception in the snapshot/run/restore
        block, returns ([], {}) — the caller treats it as "no fire,"
        and the next builder bar-close runs on_bar_close normally as
        the fallback. No silent alert drop because the regular
        bar-close path is the safety net.
        """
        if not self.indicators._initialized:
            return [], {}
        import copy as _copy
        state_snapshot = _copy.deepcopy(self.indicators.state)
        init_snapshot = self.indicators._initialized
        prebar_snapshot = getattr(
            self.indicators, '_pre_bar_snapshot', None)
        user_pack_snapshot = {
            slug: _copy.deepcopy(eng)
            for slug, eng in getattr(
                self.indicators, '_user_pack_engines', {}).items()
        }
        try:
            signals, audit_data = self.on_bar_close(
                partial_bar, bar_count, mtf_confluence=mtf_confluence)
        except Exception as e:
            logger.warning(
                "fire_on_partial_bucket strat=%s bucket=%s failed: %s — "
                "falling back to builder bar-close",
                self.strat_id, partial_bar.get('timestamp'), e)
            return [], {}
        finally:
            # Restore INDICATOR state only — position state must keep
            # whatever transition just happened, so any fired alert
            # corresponds to a real entry/exit on the strategy.
            self.indicators.state = state_snapshot
            self.indicators._initialized = init_snapshot
            if user_pack_snapshot:
                self.indicators._user_pack_engines = user_pack_snapshot
            self.indicators._pre_bar_snapshot = prebar_snapshot
        return signals, audit_data

    @_prof_fn('m_on_tick')
    def on_tick(self, price: float, timestamp: str,
                bar_count: int) -> List[dict]:
        """Process a tick for intra-bar detection. Returns list of signals."""
        signals = []

        # Check stop/target on every tick
        exit_sig = self.position.check_exit_tick(price, timestamp)
        if exit_sig:
            exit_sig['atr'] = self.indicators.state.current.get('atr', 0)
            self.position.state.last_exit_bar_count = bar_count
            signals.append(exit_sig)
            return signals

        # Check intra-bar level crossings
        ib_result = self.trigger_eval.check_intrabar(price)
        if ib_result:
            ib_trigger, fill_price = ib_result
            # Check if it's an exit trigger
            exit_sig = self.position.check_exit_intrabar(
                ib_trigger, fill_price, timestamp)
            if exit_sig:
                exit_sig['atr'] = self.indicators.state.current.get('atr', 0)
                self.position.state.last_exit_bar_count = bar_count
                signals.append(exit_sig)
            else:
                # Check if it's an entry trigger
                entry_sig = self.position.check_entry_intrabar(
                    ib_trigger, fill_price,
                    self.indicators.state.current,
                    bar_count, timestamp,
                    self._current_confluence)
                if entry_sig:
                    signals.append(entry_sig)

        return signals


# ═══════════════════════════════════════════════════════════════════════════
# WS_AGG SHADOW MINUTE BUILDER — client-side 1Min from Polygon A.* per-second
# ═══════════════════════════════════════════════════════════════════════════
#
# Live_Model_Decision.md Mode 2 candidate.  Aggregates Polygon's per-second
# `A.<symbol>` events into 1Min bars on our side and writes them to
# `live_bars` with source='ws_agg', alongside the existing source='ws' rows
# from `AM.<symbol>` events.  Purely additive: NEVER routes to monitors,
# NEVER fires alerts, NEVER affects engine decisions.  Exists only to
# generate the comparison dataset that picks between AM-only / A-agg /
# AM-with-fallback as the production live source.
#
# Scope:
#   - Only fires on symbols where the worker is already subscribed to
#     `A.<symbol>` (i.e. has at least one L-type or sub-minute strategy).
#     For SPY today this means we get rich comparison data; pure-1Min
#     symbols like AAPL/AMD wait for a Phase 2 subscription expansion.
#   - Gated by env flag `WS_AGG_SHADOW_ENABLED=true` so the change is
#     reversible at deploy time without code edits.
#
# Aggregation rules (textbook):
#   open  = first per-second bar's open
#   high  = max of all per-second highs in the minute
#   low   = min of all per-second lows in the minute
#   close = last per-second bar's close
#   volume = sum of per-second volumes
#
# Boundaries are floor-to-minute on the per-second bar's UTC timestamp.

class WsAggMinuteBuilder:
    """Aggregate per-second bars into 1Min bars at minute boundaries."""

    __slots__ = (
        'symbol', '_current_minute_unix', '_open', '_high', '_low',
        '_close', '_volume', '_tick_count',
    )

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._current_minute_unix: int = -1
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._volume: float = 0.0
        self._tick_count: int = 0

    @_prof_fn('wsagg_accept')
    def accept_second_bar(self, bar_dict: dict) -> Optional[dict]:
        """Feed one per-second bar.  Returns a completed 1Min bar dict
        when the minute boundary closes (i.e. a per-second bar arrives
        for the next minute), otherwise None.

        Returned dict shape mirrors what `live_bars_writer.write_bar`
        already accepts: {timestamp, open, high, low, close, volume}.
        """
        ts_raw = bar_dict.get('timestamp', '')
        if not ts_raw:
            return None
        try:
            if isinstance(ts_raw, str):
                ts = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
            elif isinstance(ts_raw, datetime):
                ts = ts_raw
            else:
                ts = pd.Timestamp(ts_raw).to_pydatetime()
        except (ValueError, TypeError):
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        minute_unix = int(ts.timestamp()) // 60 * 60

        completed: Optional[dict] = None
        if self._current_minute_unix < 0:
            # First bar ever — start the current minute.
            self._reset_to(minute_unix, bar_dict)
        elif minute_unix == self._current_minute_unix:
            # Same minute — accumulate.
            try:
                h = float(bar_dict.get('high', self._high))
                lo = float(bar_dict.get('low', self._low))
                c = float(bar_dict.get('close', self._close))
                v = float(bar_dict.get('volume', 0))
            except (TypeError, ValueError):
                return None
            if h > self._high:
                self._high = h
            if lo < self._low:
                self._low = lo
            self._close = c
            self._volume += v
            self._tick_count += 1
        elif minute_unix > self._current_minute_unix:
            # Boundary crossed — flush previous minute, start new.
            if self._tick_count > 0:
                completed = {
                    'timestamp': datetime.fromtimestamp(
                        self._current_minute_unix, tz=timezone.utc
                    ).isoformat(),
                    'open': self._open,
                    'high': self._high,
                    'low': self._low,
                    'close': self._close,
                    'volume': self._volume,
                }
            self._reset_to(minute_unix, bar_dict)
        # else: minute_unix < current — out-of-order per-second event
        # (rebroadcast or replay).  Don't disturb the running aggregate.
        return completed

    def _reset_to(self, minute_unix: int, bar_dict: dict) -> None:
        try:
            self._open = float(bar_dict.get('open', 0))
            self._high = float(bar_dict.get('high', 0))
            self._low = float(bar_dict.get('low', 0))
            self._close = float(bar_dict.get('close', 0))
            self._volume = float(bar_dict.get('volume', 0))
        except (TypeError, ValueError):
            self._open = self._high = self._low = self._close = 0.0
            self._volume = 0.0
        self._current_minute_unix = minute_unix
        self._tick_count = 1


def _ws_agg_shadow_enabled() -> bool:
    """Env flag — disabled by default.  Flip to 'true' on Railway worker
    to start writing source='ws_agg' rows to live_bars."""
    val = os.environ.get('WS_AGG_SHADOW_ENABLED', '').strip().lower()
    return val in ('1', 'true', 'yes', 'on')


def _ws_agg_secondary_fanout_enabled() -> bool:
    """Task #11 (2026-05-14): default ON. When True, ws_agg 1Min minute
    closes also feed ≥60s secondary builders via the same fan-out helper
    used by the AM channel. This keeps `_mtf_confluence[sec_tf]` fresh
    even when Polygon's AM channel is intermittent.

    Operator can flip OFF by setting WS_AGG_SECONDARY_FANOUT_ENABLED=false
    on the Railway worker without redeploy.
    """
    val = os.environ.get(
        'WS_AGG_SECONDARY_FANOUT_ENABLED', 'true').strip().lower()
    return val in ('1', 'true', 'yes', 'on')


def _polygon_canonical_filter_enabled() -> bool:
    """Phase G (2026-05-15): default ON. When True, the trade filter
    in `_make_on_trade` uses Polygon's canonical /v3/reference/conditions
    rules (via polygon_conditions._classify_eligibility) split into
    OHLC vs volume eligibility — matching exactly what
    flat_file_ingestion uses for observable bars.

    Diagnostic motivation: Phase E proved Cache ≠ Observable on 5-13;
    suspect #1 was that the legacy CTA/UTP letter-code filter
    (EXCLUDED_TRADE_CONDITIONS) doesn't match Polygon WS's numeric
    codes, so the filter is effectively a no-op and cache includes
    every trade — including Form T / Odd Lot / Cash Sale prints that
    Polygon REST excludes from OHLC.

    When OFF: legacy letter-code filter (original behavior, rollback path).
    When ON (default): canonical numeric filter + volume-only path for
    OHLC-ineligible-but-vol-eligible trades.

    Rollback: set RALPH_USE_POLYGON_CANONICAL_FILTER=false on the
    Worker service and the next worker restart picks it up.
    """
    val = os.environ.get(
        'RALPH_USE_POLYGON_CANONICAL_FILTER', 'true').strip().lower()
    return val in ('1', 'true', 'yes', 'on')


def _ws_agg_primary_fanout_enabled() -> bool:
    """Task #12 (2026-05-14): default ON. When True, ws_agg 1Min minute
    closes also feed >60s PRIMARY builders (5Min, 1Hour, 1Day). Without
    this, ≥5Min primaries have NO close path when Polygon's AM channel
    is silent — monitors never fire on_bar_close, alerts never dispatch,
    and own_records cascading to dependent strategies (e.g. sid 126
    needing sid 137's 5m secondary) is blocked.

    Caveat: this path coexists with the per-second forming-bar update in
    on_second_bar's primary loop (close_on_boundary=False). For the
    forming partial of a >60s primary, BOTH paths contribute volume.
    Closed history bars therefore reflect ~2x volume for the forming
    period. Affects volume-based interpreters (VWAP_V2, RVOL_V2) only;
    price-based interpreters (EMA, MACD, SWING_123, UT_BOT_V4) unaffected.

    Operator can flip OFF by setting WS_AGG_PRIMARY_FANOUT_ENABLED=false
    on the Railway worker without redeploy.
    """
    val = os.environ.get(
        'WS_AGG_PRIMARY_FANOUT_ENABLED', 'true').strip().lower()
    return val in ('1', 'true', 'yes', 'on')


# Bug 5 (2026-06-18): the live warmup used a flat days=7 for EVERY builder,
# which starves coarse-TF gates — a 1Day gate got ~5 bars and its indicator
# (UT_BOT, etc.) never warmed, so the cross-TF gate state was never valid and
# gated strategies never entered live (310/313 fired 0 live vs 33/103 backtest).
# Size the warmup window to the TF so the gate converges the way it does in
# backtest. Target a generous bar count (>= what the backtest window yields),
# floored at 7 days (fine TFs already warm in <7d).
_SHADOW_WARMUP_TARGET_BARS = 250


def _secondary_warmup_days(tf_str: str, is_crypto_sym: bool = False) -> int:
    """Calendar days of 1-min history needed to warm ~_SHADOW_WARMUP_TARGET_BARS
    bars of `tf_str`. The shadow is then built by resampling that 1-min history
    to `tf_str` — mirroring the backtest's resample-from-1min construction
    (CLAUDE.md), so live and backtest gate states align. Sub-minute TFs keep
    the flat 7-day native load (can't resample 1-min finer)."""
    import math as _m
    from data_loader import BARS_PER_DAY, CRYPTO_BARS_PER_DAY
    bpd_map = CRYPTO_BARS_PER_DAY if is_crypto_sym else BARS_PER_DAY
    bpd = bpd_map.get(tf_str, 390)
    if bpd <= 0:
        return 7
    trading_days = _SHADOW_WARMUP_TARGET_BARS / bpd
    # trading-days -> calendar-days. Equities skip weekends (×7/5); crypto is
    # 24/7. Add a holiday/weekend buffer either way.
    cal = trading_days if is_crypto_sym else trading_days * 7.0 / 5.0
    return max(7, int(_m.ceil(cal)) + 5)


def _load_warmup_df(sym: str, tf_seconds: int, session: str) -> pd.DataFrame:
    """Load warmup bars for one builder/shadow TF (Bug 5, 2026-06-18).

    Single source of truth for every warmup path (startup `_warmup_all` +
    hot-reload). Sub-minute TFs load natively (can't resample 1-min finer).
    >=60s TFs are built by resampling a TF-scaled 1-min window — matching the
    backtest's resample-from-1min construction (CLAUDE.md) AND the live
    ws_agg/AM fan-out — so coarse gates (4h/1d) actually converge live instead
    of being starved by the old flat days=7 (a 1Day gate got ~5 bars → never
    warmed → gated strategies never entered live). Validated: recent daily
    UT_BOT_V4 states match backtest 0/12 mismatches."""
    from data_loader import (
        load_market_data, is_crypto, resample_to_timeframe)
    tf_str = SECONDS_TO_TIMEFRAME.get(tf_seconds, '1Min')
    # KILL-SWITCH (2026-06-19): RORT_TF_SCALED_WARMUP=0 reverts to the
    # pre-Bug-5 flat days=7 native load for ALL TFs — instant rollback if the
    # TF-scaled resample-from-1min warmup shows any live divergence Monday
    # (flip the env var, no code revert/redeploy of logic). 1Min stays native
    # either way (the resample-to-1Min no-op is invalid).
    _tf_scaled = os.getenv('RORT_TF_SCALED_WARMUP', '1') == '1'
    if tf_seconds <= 60 or not _tf_scaled:
        return load_market_data(
            sym, days=7, timeframe=tf_str, feed='sip', session=session)
    wd = _secondary_warmup_days(tf_str, is_crypto(sym))
    df1 = load_market_data(
        sym, days=wd, timeframe='1Min', feed='sip', session=session)
    if df1 is None or len(df1) == 0:
        return pd.DataFrame()
    out = resample_to_timeframe(
        df1[['open', 'high', 'low', 'close', 'volume']].copy(), tf_str)
    logger.info("[Bug5] warmup %s tf=%s: 1min_days=%d -> %d resampled bars "
                "(was flat days=7)", sym, tf_str, wd, len(out))
    return out


# Bug Hunt Wave 1 #1 — coarse-gate SESSION MISMATCH (2026-07-06,
# docs/_active/Bug_Hunt_Wave1_2026-07-06.md).
#
# The live coarse-secondary gate shadows (_ShadowIndicatorEngine) were keyed
# by tf_seconds ONLY, and the hub warmup resolved the warmup session from
# "first monitor whose PRIMARY tf == this tf" — which can NEVER match a
# secondary TF, so every coarse gate shadow hard-defaulted to 'RTH'. The
# backtest preps gate columns with the STRATEGY's session. For non-RTH
# strategies (sid 338: TSLA 10Sec, session='Extended Hours', gate
# 1d-SWING_123-NEUTRAL) the identical daily engine derives NEUTRAL on RTH
# dailies but BEAR_C2 on extended dailies — live over-fired 65 entries on a
# day the backtest correctly fired 0. RTH strategies matched by luck.
#
# Fix (flag ON): shadows and the cross-TF gate buffer are keyed per
# (tf_seconds, session); each shadow warms up + reloads from ITS session's
# bars, and monitors merge ONLY records from their own session's key. A
# UNION of sessions' records would wrongly satisfy both gates (TSLA/1d is
# required by 338 Extended AND 310/312/313 RTH) — hence per-key isolation,
# not merging.
#
# Flag OFF (default): both dicts still use tuple keys internally, but the
# session component is hard-pinned to 'RTH' everywhere — behavior is exactly
# today's (single shadow per tf, 'RTH' warmup).
#
# Read once at module import (like GAP_SKIP in shadow_manager.py) — flip
# requires a worker restart.
MTF_SESSION_SHADOWS = os.getenv(
    "RORT_MTF_SESSION_SHADOWS", "0").strip().lower() in (
    "1", "true", "yes", "on")


def _shadow_session_for(monitor: 'StrategyMonitor') -> str:
    """Session key a monitor's cross-TF gate lookups/stores use.

    Flag ON: the monitor's own trading session (verbatim — same string
    that's passed to load_market_data at the monitor warmup sites).
    Flag OFF: hard 'RTH', reproducing the legacy single-shadow behavior.
    getattr-defensive: StrategyMonitor always sets .session, but test
    doubles / duck-typed monitors may not.
    """
    if not MTF_SESSION_SHADOWS:
        return 'RTH'
    return getattr(monitor, 'session', 'RTH')


def _closed_bars_only(df: pd.DataFrame, tf_seconds: int) -> pd.DataFrame:
    """Drop the currently-forming bar from a warmup df (index >= current
    period start). Mirrors what BarBuilder.seed_history does for the
    builder-backed warmup path; used for session-shadow warmups that
    bypass the builder (the builder holds the RTH-flavored feed)."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    now_utc = pd.Timestamp.now(tz='UTC')
    period_ns = int(tf_seconds) * 1_000_000_000
    cur_start = pd.Timestamp(
        (now_utc.value // period_ns) * period_ns, tz='UTC')
    if getattr(df.index, 'tz', None) is None:
        df = df.copy()
        df.index = df.index.tz_localize('UTC')
    return df[df.index < cur_start]


# ═══════════════════════════════════════════════════════════════════════════
# SHADOW INDICATOR ENGINE — secondary TF interpreter state for MTF confluence
# ═══════════════════════════════════════════════════════════════════════════

class _ShadowIndicatorEngine:
    """Runs indicator/interpreter pipeline for a secondary TF.

    Produces confluence records but does NOT manage positions, triggers, or
    alerts.  Used by SymbolHub to provide cross-TF interpreter states to
    primary monitors for MTF confluence gating.
    """

    def __init__(self, tf_seconds: int, req_ind: Set[str],
                 req_interp: Set[str], params: Dict[str, Any],
                 session: str = 'RTH'):
        self.tf_seconds = tf_seconds
        # Bug Hunt Wave 1 #1: the trading session whose bars this shadow's
        # state is derived from ('RTH' unless RORT_MTF_SESSION_SHADOWS).
        self.session = session
        self.indicators = IncrementalIndicatorEngine(req_ind, params)
        # TriggerEvaluator needs interpreters but no triggers (empty set)
        self.trigger_eval = TriggerEvaluator(
            req_interp, set(), params['ema_periods'])

        tf_label = SECONDS_TO_TIMEFRAME.get(tf_seconds, '1Min')
        self._tf_short_label = tf_label.replace(
            'Min', 'M').replace('Hour', 'H').replace(
            'Day', 'D').replace('Week', 'W')
        self._current_confluence: Set[str] = set()

    def warmup(self, df: pd.DataFrame):
        self.indicators.warmup(df)
        # Live shadow engine — enable O(1) rebroadcast correction.
        self.indicators._snapshot_enabled = True

    @_prof_fn('sh_on_bar_close')
    def on_bar_close(self, bar: dict) -> Set[str]:
        """Process a completed bar and return confluence records."""
        current = self.indicators.update_bar(bar)
        prev = self.indicators.get_prev_values()

        interps, _ = self.trigger_eval.evaluate_bar_close(
            current, prev, self.indicators.state.prev2_macd_hist)

        self._current_confluence = set()
        for ikey, state_val in interps.items():
            self._current_confluence.add(
                f'{self._tf_short_label}-{ikey}-{state_val}')

        return self._current_confluence

    def recompute_confluence(self, history_df: pd.DataFrame) -> Set[str]:
        """Re-sync indicators from corrected history AND re-emit confluence
        records — the rebroadcast-cascade counterpart to `on_bar_close`.

        2026-06-09 (iter 0609a): the rebroadcast/duplicate path used to call
        `indicators.recompute_from_history()` only, leaving the shadow's
        `_current_confluence` (and therefore `SymbolHub._mtf_confluence[tf]`)
        FROZEN at whatever state it held on the first real close after worker
        start. For 2m+ secondary-TF gates whose closes are flagged duplicate
        every bucket, that meant the cross-TF gate state never moved — it
        passed (or blocked) forever based on the boot-time secondary state.
        This recomputes indicators from the corrected history and then
        re-derives the records so the gate stays live. Shadow records drive
        only gating (no triggers/positions/alerts), so re-emitting is safe —
        the original "don't re-emit" rule was inherited from the primary
        own_records path, where re-emit could double-fire alerts.
        """
        self.indicators.recompute_from_history(history_df)
        return self._derive_confluence_records()

    def _derive_confluence_records(self) -> Set[str]:
        """Re-derive `_current_confluence` from the indicators' CURRENT
        state without touching indicator state itself. Factored out of
        `recompute_confluence` (2026-06-11, gap healer) so callers that
        already ran their own recompute (e.g. the insert-aware replay in
        apply_rest_insert) can refresh the records without a second
        indicator pass."""
        current = self.indicators.get_values()
        prev = self.indicators.get_prev_values()

        interps, _ = self.trigger_eval.evaluate_bar_close(
            current, prev, self.indicators.state.prev2_macd_hist)

        self._current_confluence = set()
        for ikey, state_val in interps.items():
            self._current_confluence.add(
                f'{self._tf_short_label}-{ikey}-{state_val}')

        return self._current_confluence


# ═══════════════════════════════════════════════════════════════════════════
# ALERT DISPATCHER — save + enrich + webhook
# ═══════════════════════════════════════════════════════════════════════════

class AlertDispatcher:
    """Saves alerts and fires webhooks. Reuses existing alert infrastructure."""

    def __init__(self):
        self._deliver_alert_fn = None
        try:
            from alert_monitor import deliver_alert
            self._deliver_alert_fn = deliver_alert
        except Exception as e:
            logger.warning("Could not import deliver_alert: %s", e)

    def dispatch(self, signal: dict, strategy: dict,
                 config: dict) -> Optional[dict]:
        """Build, enrich, save, and deliver an alert.

        Returns the saved alert dict (with id and timestamp).
        Webhook delivery is offloaded to a thread pool to avoid blocking
        the async event loop.
        """
        from alerts import (save_alert, enrich_signal_with_portfolio_context,
                            load_alert_config)

        # Derive order_action: buy/sell for entries, close for exits
        direction = strategy.get('direction', 'LONG')
        sig_type = signal['type']
        if sig_type == 'exit_signal':
            order_action = 'close'
        elif direction == 'LONG':
            order_action = 'buy'
        else:
            order_action = 'sell'

        # algo_model split (2026-05-07): mirror of DBAlertDispatcher logic
        # in worker.py — stamp strategy's live_model on alert payload at
        # fire time. Per feedback_dispatcher_override_pitfall.md, both
        # dispatcher classes must carry symmetric field-passing logic.
        # 2026-06-04: added get_default_live_model() fallback so strategies
        # that haven't explicitly opted into a model still get the resolved
        # default stamped on the alert. Without this, rest_verifier
        # short-circuits (alert.live_model != 'ws_rest_spliced') and Layer 1
        # (verification_status) stays NULL for ~90% of the fleet.
        try:
            from strategy_models import get_default_live_model
            _default_live_model = get_default_live_model()
        except Exception:
            _default_live_model = None
        live_model_at_fire = (
            (strategy.get('config') or {}).get('live_model')
            or strategy.get('live_model')
            or _default_live_model
        )

        alert = {
            'type': sig_type,
            'trigger': signal.get('trigger', ''),
            'price': signal.get('price', 0),
            'bar_time': signal.get('bar_time', ''),
            'stop_price': signal.get('stop_price'),
            'target_price': signal.get('target_price'),
            'atr': signal.get('atr', 0),
            'level': 'strategy',
            'strategy_id': strategy['id'],
            'strategy_name': strategy.get('name', ''),
            'symbol': strategy.get('symbol', '?'),
            'direction': direction,
            'order_action': order_action,
            'risk_per_trade': strategy.get('risk_per_trade', 100.0),
            'timeframe': strategy.get('timeframe', '1Min'),
            'strategy_alerts_visible': True,
            'source': 'ralph',
            'live_model': live_model_at_fire,
        }

        if signal['type'] == 'exit_signal':
            alert['entry_price'] = signal.get('entry_price')
            alert['entry_stop_price'] = signal.get('entry_stop_price')

        # M8.7 M4 (2026-05-02): carry the indicator snapshot from
        # _fire_alert through to the alert dict. _alert_to_row() routes
        # any field not in ALERT_COLUMN_FIELDS into the alerts.data JSONB,
        # so this lands as alert.data.indicator_snapshot in the DB.
        snapshot = signal.get('indicator_snapshot')
        if snapshot:
            alert['indicator_snapshot'] = snapshot

        alert = enrich_signal_with_portfolio_context(alert, strategy['id'])
        alert = save_alert(alert)

        logger.info("ALERT: %s for %s (%s) trigger=%s price=%.2f",
                     signal['type'], strategy.get('name'), strategy.get('symbol'),
                     signal.get('trigger'), signal.get('price', 0))

        # ws_rest_spliced verifier hook (2026-05-28): mirror of the
        # DBAlertDispatcher hook in worker.py. Two dispatcher paths exist
        # (dev/local AlertDispatcher here; production DBAlertDispatcher
        # in worker.py) — both call save_alert and both fire webhooks,
        # so both need the verifier hook for symmetric coverage. No-op
        # when REST_VERIFY_ENABLED unset or alert not on ws_rest_spliced.
        try:
            from rest_verifier import queue_verify_for_alert
            queue_verify_for_alert(alert, signal, strategy)
        except Exception as e:
            logger.warning(
                "rest_verifier hook failed (sid=%s): %s",
                strategy.get('id'), e)

        # Webhook delivery — offloaded to thread pool (non-blocking)
        if self._deliver_alert_fn:
            self._schedule_webhook(alert, config)

        return alert

    def _schedule_webhook(self, alert: dict, config: dict):
        """Fire webhook in a background thread to avoid blocking the event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None, self._deliver_webhook_sync, alert, config)
        except RuntimeError:
            # No running event loop — fall back to synchronous
            self._deliver_webhook_sync(alert, config)

    def _deliver_webhook_sync(self, alert: dict, config: dict):
        """Synchronous webhook delivery (runs in thread pool)."""
        try:
            self._deliver_alert_fn(alert, config)
        except Exception as e:
            logger.error("Webhook delivery failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# FIDELITY AUDIT LOGGER
# ═══════════════════════════════════════════════════════════════════════════

class FidelityAuditor:
    """Logs indicator values and trigger states for offline verification."""

    def __init__(self, path: Path = _ENGINE_AUDIT_FILE):
        self.path = path

    def log_bar_close(self, symbol: str, tf_seconds: int,
                      bar: dict, indicator_values: Dict[str, float],
                      trigger_booleans: Dict[str, bool],
                      interpreter_states: Dict[str, str],
                      positions: Dict[int, str]):
        """Append a bar-close audit record."""
        record = {
            'ts': bar.get('timestamp', ''),
            'symbol': symbol,
            'tf': tf_seconds,
            'bar_close': bar.get('close', 0),
            'indicators': {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in indicator_values.items()
                          if k not in ('open', 'high', 'low', 'close',
                                       'volume')},
            'triggers': {k: v for k, v in trigger_booleans.items() if v},
            'interpreters': interpreter_states,
            'positions': positions,
        }
        try:
            with open(self.path, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception:
            pass  # Non-critical — don't crash the engine


# ═══════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def save_engine_state(positions: Dict[int, PositionState],
                      path: Path = _ENGINE_STATE_FILE):
    """Persist position states to JSON."""
    state = {
        'positions': {str(k): v.to_dict() for k, v in positions.items()},
        'saved_at': datetime.now(timezone.utc).isoformat(),
    }
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, str(path))


def load_engine_state(path: Path = _ENGINE_STATE_FILE,
                      ) -> Dict[int, PositionState]:
    """Load position states from JSON. Returns empty dict if missing."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {int(k): PositionState.from_dict(v)
                for k, v in data.get('positions', {}).items()}
    except Exception as e:
        logger.warning("Failed to load engine state: %s", e)
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOL HUB — one per symbol, owns BarBuilders and routes to monitors
# ═══════════════════════════════════════════════════════════════════════════

class SymbolHub:
    """Per-symbol tick dispatcher with BarBuilders and StrategyMonitors."""

    def __init__(self, symbol: str, publisher: Optional['LiveBarPublisher'] = None):
        self.symbol = symbol
        self.builders: Dict[int, BarBuilder] = {}  # tf_seconds → BarBuilder
        self.monitors: Dict[int, StrategyMonitor] = {}  # strat_id → monitor
        self.tick_count = 0
        self.last_tick_time: Optional[datetime] = None
        # Trade_Timestamps_Spec (2026-04-20): latest per-second close price
        # for this symbol, updated on every per-second bar arrival (A
        # channel). Used by alert dispatch to stamp `actual_price` on each
        # alert — the near-live market price at the moment the alert fires,
        # as distinct from the theoretical fill price from the engine. Gap
        # between them = price slippage (complements the time slippage in
        # the delta column).
        self.latest_close: Optional[float] = None
        # Cross-TF confluence: (tf_seconds, session) → latest interpreter
        # confluence records. Bug Hunt Wave 1 #1: keys are ALWAYS tuples;
        # session is 'RTH' unless RORT_MTF_SESSION_SHADOWS is on.
        self._mtf_confluence: Dict[Tuple[int, str], Set[str]] = {}
        # Shadow engines for secondary TFs not covered by real monitors,
        # keyed (tf_seconds, session) — same convention as _mtf_confluence.
        self._shadow_engines: Dict[
            Tuple[int, str], '_ShadowIndicatorEngine'] = {}
        # Live-bar publisher (Supabase Realtime broadcast, Phase M8.5).
        # None is a valid value: all publish calls become silent no-ops.
        self._publisher = publisher
        # Held refs to prevent GC of fire-and-forget asyncio tasks.
        self._pending_publish_tasks: Set['asyncio.Task'] = set()
        # Optional bar-close hook: an async callable installed by the
        # worker/engine host to trigger per-strategy forward-test
        # recompute on every primary-TF bar close. Signature:
        #   async def hook(strategy_id: int, user_id: str, tf_seconds: int)
        # Never blocks the tick loop — task is fire-and-forget and any
        # exceptions are swallowed by the caller's own error handling.
        self.on_bar_close_hook: Optional[Callable] = None
        self._pending_bar_close_tasks: Set['asyncio.Task'] = set()
        # WS_AGG shadow builder (Live_Model_Decision.md Mode 2 candidate).
        # Aggregates per-second A.* bars into 1Min on our side; writes to
        # live_bars with source='ws_agg'.  Lazy-init on first use so hubs
        # for symbols that never see A events don't allocate the builder.
        self._ws_agg_minute: Optional[WsAggMinuteBuilder] = None

    def _fire_alert(self, sig: dict, monitor: 'StrategyMonitor',
                    config: dict, alert_callback: Optional[Callable]) -> None:
        """Stamp actual_price onto the signal and invoke alert_callback.

        Trade_Timestamps_Spec (2026-04-20): actual_price = the most recent
        per-second close we've seen for this symbol. Gap between
        actual_price and sig['price'] (engine's theoretical fill) =
        price slippage. Zero latency — in-memory dict lookup.

        M8.7 M4 (2026-05-02): also snapshot the engine's indicator state
        at this moment onto the signal. Lets the alert record carry a
        precise "what did the engine see when it decided to fire" view
        for later analysis. Flows through to the alert dict (and its
        `data` JSONB) via the dispatch path.
        """
        if not alert_callback or not sig:
            return
        if self.latest_close is not None and 'actual_price' not in sig:
            sig['actual_price'] = self.latest_close
        # Snapshot engine indicator state. monitor.indicators.state.current is
        # a Dict[str, float|bool] populated by IncrementalIndicatorEngine
        # (see unified_engine.py). Defensive copy + scalar coercion so the
        # JSON serializer downstream never trips on numpy types.
        try:
            if hasattr(monitor, 'indicators') and \
                    hasattr(monitor.indicators, 'state') and \
                    'indicator_snapshot' not in sig:
                src = monitor.indicators.state.current or {}
                snapshot: dict = {}
                for k, v in src.items():
                    if isinstance(v, bool):
                        snapshot[k] = bool(v)
                    elif isinstance(v, (int, float)):
                        snapshot[k] = float(v)
                    else:
                        snapshot[k] = str(v)
                sig['indicator_snapshot'] = snapshot
        except Exception as e:
            logger.warning(
                "Failed to snapshot indicator state for alert (sid=%s): %s",
                getattr(monitor, 'strat_id', '?'), e)
        alert_callback(sig, monitor.strategy, config)

    def _gather_tentative_state(
        self, tf_seconds: int, forming_bar: dict,
    ) -> Dict[str, Any]:
        """Collect tentative indicator values + interpreter states for every
        monitor on this TF, plus last-committed cross-TF interpreter states
        from the shared MTF confluence buffer. Used to enrich live-bar
        broadcasts so the frontend can animate indicator overlays, oscillator
        panes, and CB-fidelity heatmap cells in sync with the price candle.

        Returns a dict with three keys:
            indicators: {col: value}  — merged across all monitors on this TF
            states: {interp_key: state}  — primary-TF interpreter states
            state_cross_tf: {col: state}  — cross-TF interpreter states (last
                committed bar; sub-minute indicators on higher TFs don't
                recompute intra-bar).

        Returns {} on any failure — broadcast still goes out with OHLCV only.
        """
        try:
            indicators: Dict[str, float] = {}
            states: Dict[str, str] = {}
            _tf_sessions: Set[str] = set()
            for m in self.monitors.values():
                if m.tf_seconds != tf_seconds:
                    continue
                _tf_sessions.add(_shadow_session_for(m))
                ind, interp = m.compute_tentative_state(forming_bar)
                indicators.update(ind)
                states.update(interp)
            if not _tf_sessions:
                _tf_sessions = {'RTH'}

            # Bug Hunt Wave 1 #1: only surface cross-TF records whose key
            # session matches a monitor on this TF (flag OFF: everything
            # is keyed 'RTH', so all records pass — legacy behavior).
            state_cross_tf: Dict[str, str] = {}
            for (other_tf, _sess), records in self._mtf_confluence.items():
                if other_tf == tf_seconds:
                    continue
                if _sess not in _tf_sessions:
                    continue
                for rec in records:
                    if rec.startswith('GEN-'):
                        continue
                    parts = rec.split('-', 2)
                    if len(parts) < 3:
                        continue
                    tf_label, interp_key, state_val = parts
                    col = f"{interp_key}__{tf_label.lower()}"
                    state_cross_tf[col] = state_val

            return {
                'indicators': indicators,
                'states': states,
                'state_cross_tf': state_cross_tf,
            }
        except Exception as e:
            logger.debug("_gather_tentative_state failed: %s", e)
            return {}

    def _publish_completed_bar(self, tf_seconds: int, bar: dict,
                                is_forming: bool = False,
                                extras: Optional[Dict[str, Any]] = None,
                                ) -> None:
        """Fire-and-forget broadcast of a bar snapshot to Supabase Realtime.

        Side effect only — MUST NOT block or raise into the tick handler.
        Publishes once per (symbol, tf) regardless of how many monitors share
        the timeframe (user_id is taken from the first matching monitor).

        is_forming=True signals the publisher to apply the per-(symbol, tf)
        throttle window (250ms by default). is_forming=False (default) always
        sends immediately and resets the throttle — used for canonical bar
        closes that must never be dropped.

        `extras` is merged into the broadcast payload alongside `bar` /
        `is_forming`. Used for live indicator values, interpreter states,
        and cross-TF state snapshots.
        """
        if self._publisher is None:
            return
        user_id = None
        for m in self.monitors.values():
            if m.tf_seconds == tf_seconds and getattr(m, 'user_id', ''):
                user_id = m.user_id
                break
        if not user_id:
            return
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(
                self._publisher.publish_async(
                    user_id, self.symbol, tf_seconds, bar,
                    is_forming=is_forming, extras=extras,
                )
            )
            self._pending_publish_tasks.add(task)
            task.add_done_callback(self._pending_publish_tasks.discard)
        except Exception as e:
            logger.debug("LiveBar publish schedule failed: %s", e)

    def add_monitor(self, monitor: StrategyMonitor):
        self.monitors[monitor.strat_id] = monitor
        # Ensure a BarBuilder exists for this timeframe
        if monitor.tf_seconds not in self.builders:
            self.builders[monitor.tf_seconds] = BarBuilder(monitor.tf_seconds)
        # Also create BarBuilders for required secondary TFs
        for sec_tf in getattr(monitor, '_required_secondary_tf', set()):
            if sec_tf not in self.builders:
                self.builders[sec_tf] = BarBuilder(sec_tf)
                logger.info("Created BarBuilder for secondary TF %ss (%s)",
                            sec_tf, self.symbol)
        # LEF Phase 2a: refresh the primary builder's grace override
        # (max grace_seconds across active monitors on this TF). 1Min+
        # builders skip this — force_close_stale_bar ignores grace there.
        if monitor.tf_seconds < 60:
            self._refresh_builder_grace(monitor.tf_seconds)

    def _check_strategy_grace_fires(self, tf_seconds: int, builder,
                                     alert_callback, config) -> None:
        """LEF Phase 2b — fire ws_agg_reconciled monitors at their own
        strategy-grace on the current partial bucket.

        For each monitor on this (symbol, TF) with live_model
        'ws_agg_reconciled' whose grace window has just elapsed on the
        current partial bucket (and which hasn't already fired for this
        bucket), build a partial_bar dict from the builder's partial
        and call `fire_on_partial_bucket`. Dispatch any returned
        signals via `_fire_alert` and mark `_fired_bucket` so subsequent
        per-second events don't double-fire.

        Cheap no-op short-circuit when no monitor on this hub uses
        ws_agg_reconciled — the common case while the model is still
        `available: False` in `LIVE_MODELS`.

        Misses (e.g. monitor added mid-bucket so the grace deadline
        already passed) fall through to the existing on_bar_close
        path at builder bar-close — never a silent drop.
        """
        # Short-circuit: nothing reconciled on this hub.
        # 2026-05-28: ws_rest_spliced uses the same fire-at-grace
        # behavior — added to the gate so its monitors fire at grace too.
        if not any(m.live_model in ('ws_agg_reconciled', 'ws_rest_spliced')
                   for m in self.monitors.values()):
            return
        partial = getattr(builder, '_partial', None)
        if partial is None:
            return
        bucket_start_dt = partial.bar_start
        if bucket_start_dt.tzinfo is None:
            bucket_start_dt = bucket_start_dt.replace(tzinfo=timezone.utc)
        bucket_start_epoch = int(bucket_start_dt.timestamp())
        # Wall clock = the timestamp of the latest per-second event we
        # processed (set in on_second_bar). Falls back to UTC now() if
        # the engine hasn't received any tick yet (cold start).
        wall = self.last_tick_time
        if wall is None:
            wall = datetime.now(timezone.utc)
        elif wall.tzinfo is None:
            wall = wall.replace(tzinfo=timezone.utc)
        # Build the partial bar dict once — every reconciled monitor on
        # this builder sees the same forming-bar snapshot.
        partial_bar = {
            'open': float(partial.open),
            'high': float(partial.high),
            'low': float(partial.low),
            'close': float(partial.close),
            'volume': float(partial.volume),
            'timestamp': bucket_start_dt.isoformat(),
        }
        bar_count_for_partial = getattr(builder, '_bar_count', 0) + 1
        for monitor in self.monitors.values():
            if monitor.tf_seconds != tf_seconds:
                continue
            if monitor.live_model not in (
                'ws_agg_reconciled', 'ws_rest_spliced',
            ):
                continue
            if monitor._fired_bucket == bucket_start_epoch:
                continue  # already fired this bucket
            due_at = bucket_start_dt + timedelta(
                seconds=tf_seconds + monitor.grace_seconds)
            if wall < due_at:
                continue  # grace window still open
            try:
                signals, _audit = monitor.fire_on_partial_bucket(
                    partial_bar, bar_count_for_partial,
                    mtf_confluence=self._mtf_confluence)
            except Exception as e:
                logger.warning(
                    "strategy-grace fire failed strat=%s tf=%ss: %s — "
                    "monitor will catch up at builder bar-close",
                    monitor.strat_id, tf_seconds, e)
                continue
            monitor._fired_bucket = bucket_start_epoch
            if signals and alert_callback:
                logger.info(
                    "STRATEGY_GRACE_FIRE strat=%s tf=%ss grace=%ss bar=%s "
                    "signals=%d",
                    monitor.strat_id, tf_seconds, monitor.grace_seconds,
                    partial_bar['timestamp'], len(signals))
                for sig in signals:
                    self._fire_alert(sig, monitor, config, alert_callback)

    def _refresh_builder_grace(self, tf_seconds: int) -> None:
        """Recompute and apply a sub-minute builder's `grace_sec_override`
        as max(resolve_grace_seconds) across active monitors on that
        timeframe.

        LEF Phase 2a — foundational wiring. With no strategy overriding
        `config.grace_seconds`, the resolver returns the global default
        (5), so the override == the existing SUBMINUTE_FORCE_CLOSE_
        GRACE_SEC constant and behavior is unchanged. The override only
        diverges from default once a strategy carries an explicit
        `grace_seconds` in its config.
        """
        builder = self.builders.get(tf_seconds)
        if builder is None or tf_seconds >= 60:
            return
        from strategy_models import resolve_grace_seconds
        graces = [
            resolve_grace_seconds(m.strategy)
            for m in self.monitors.values()
            if m.tf_seconds == tf_seconds
        ]
        builder.grace_sec_override = max(graces) if graces else None

    def _shadow_items_for_tf(
        self, tf_seconds: int,
    ) -> List[Tuple[str, '_ShadowIndicatorEngine']]:
        """All (session, shadow) pairs registered for one TF.

        Bug Hunt Wave 1 #1: with RORT_MTF_SESSION_SHADOWS off this is at
        most one item — ('RTH', shadow) — matching the legacy
        single-shadow-per-tf layout."""
        return [(k[1], sh) for k, sh in self._shadow_engines.items()
                if k[0] == tf_seconds]

    def _shadow_tf_set(self) -> Set[int]:
        """Distinct TFs that have at least one shadow engine."""
        return {k[0] for k in self._shadow_engines}

    def _close_shadow_with_bar(self, tf_seconds: int, session: str,
                               shadow: '_ShadowIndicatorEngine',
                               completed: dict) -> None:
        """Refresh one session-shadow's gate records on a TF close.

        RTH-session shadows consume the completed bar directly — exactly
        today's behavior (the fan-out/builder bars are RTH-flavored).
        Non-RTH shadows (RORT_MTF_SESSION_SHADOWS only) must NOT eat that
        bar — it's the wrong candle for their session — so they reload
        session-correct history via _load_warmup_df and recompute. That
        reload runs once per coarse-TF close (a few times/day for the
        1H/4H/1D gates this exists for). On any failure the previous
        records are kept — never crash the close path."""
        key = (tf_seconds, session)
        if session == 'RTH':
            self._mtf_confluence[key] = shadow.on_bar_close(completed)
            return
        try:
            df = _load_warmup_df(self.symbol, tf_seconds, session)
            df = _closed_bars_only(df, tf_seconds)
            if df is not None and len(df) > 0:
                self._mtf_confluence[key] = shadow.recompute_confluence(df)
            else:
                logger.warning(
                    "[MTF-SESSION] %s tf=%ss session=%s: empty session "
                    "reload on close — keeping previous gate records",
                    self.symbol, tf_seconds, session)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[MTF-SESSION] %s tf=%ss session=%s: close reload failed "
                "(%s) — keeping previous gate records",
                self.symbol, tf_seconds, session, e)

    def finalize_shadow_engines(self):
        """Create shadow indicator engines for secondary TFs not covered by
        real monitors.  Call after all monitors have been added.  Idempotent.
        """
        # Collect all secondary TFs needed across monitors, keyed by
        # (tf, session) — Bug Hunt Wave 1 #1. Flag OFF pins session='RTH'
        # so the grouping collapses to the legacy per-tf layout.
        needed: Dict[Tuple[int, str], List[StrategyMonitor]] = {}
        for monitor in self.monitors.values():
            for sec_tf in getattr(monitor, '_required_secondary_tf', set()):
                key = (sec_tf, _shadow_session_for(monitor))
                needed.setdefault(key, []).append(monitor)

        # For each needed secondary key, check if a real monitor covers it
        for (sec_tf, sh_session), requesting_monitors in needed.items():
            if (sec_tf, sh_session) in self._shadow_engines:
                continue  # already created
            # A real monitor on that TF produces the records — but (flag
            # ON) only for ITS OWN session key; a different-session gate
            # still needs a shadow.
            has_real = any(
                m.tf_seconds == sec_tf
                and (not MTF_SESSION_SHADOWS or m.session == sh_session)
                for m in self.monitors.values())
            if has_real:
                continue  # real monitor produces confluence records

            # Collect indicator/interpreter requirements from requesting monitors
            req_ind: Set[str] = {'atr'}
            req_interp: Set[str] = set()
            shadow_params: Dict[str, Any] = {}

            for monitor in requesting_monitors:
                all_conf = (list(monitor.strategy.get('confluence', []))
                            + list(monitor.strategy.get('general_confluences', [])))
                for rec in all_conf:
                    if rec.startswith('GEN-'):
                        continue
                    parts = rec.split('-', 2)
                    if len(parts) < 3:
                        continue
                    tf_sec = _LABEL_TO_TF_SECONDS.get(parts[0], 0)
                    if tf_sec != sec_tf:
                        continue
                    interp_key = parts[1]
                    req_interp.add(interp_key)
                    matched_builtin = False
                    for _tpl_key, (ind_set, ikey) in TEMPLATE_REQUIREMENTS.items():
                        if ikey == interp_key:
                            req_ind.update(ind_set)
                            matched_builtin = True
                            break
                    if not matched_builtin:
                        # User-pack interpreter on a secondary TF — add the
                        # `_user_pack_<slug>` marker so the shadow engine's
                        # IncrementalIndicatorEngine instantiates the pack's
                        # incremental_class. Without this the shadow engine
                        # carries the interpreter requirement but no
                        # underlying indicator values, and evaluate_bar_close
                        # has nothing to dispatch on. This was bug #4 in the
                        # user-pack live integration set (sids 132/134).
                        try:
                            import pack_registry
                            for slug, pack in (
                                pack_registry.get_registered_packs().items()
                            ):
                                if interp_key in pack.manifest.get(
                                    'interpreters', []):
                                    req_ind.add(f'_user_pack_{slug}')
                                    break
                        except Exception as _e:
                            logger.warning(
                                "shadow user-pack lookup failed for %s: %s",
                                interp_key, _e)

                # Merge params from every requesting monitor so the shadow's
                # union of required indicators is fully covered. Shared keys
                # (macd_fast_period, vwap_sd1_mult, etc.) should agree across
                # monitors since they come from the same user-wide group;
                # last-wins on conflict (rare, only if user has multiple
                # variant groups for the same indicator family).
                _, _, _, monitor_params = resolve_strategy_requirements(
                    monitor.strategy)
                shadow_params.update(monitor_params)

            if req_interp:
                shadow = _ShadowIndicatorEngine(
                    sec_tf, req_ind, req_interp, shadow_params,
                    session=sh_session)
                self._shadow_engines[(sec_tf, sh_session)] = shadow
                logger.info("Created shadow engine for %s/%ss session=%s: "
                            "ind=%s interp=%s",
                            self.symbol, sec_tf, sh_session,
                            req_ind, req_interp)

    def seed_history(self, tf_seconds: int, df: pd.DataFrame):
        builder = self.builders.get(tf_seconds)
        if builder:
            builder.seed_history(df)
            # Cache reconcile (2026-06-11): worker restarts leave a
            # ~1-3 min hole in live_bars (engine history is whole via
            # this REST seed, but nothing wrote the cache during the
            # switchover). Backfill the recent window with the seeded
            # bars as source='warmup_seed' — engine-consumed, so the
            # alert lens / parity ribbon stay faithful across deploys.
            try:
                from live_bars_writer import reconcile_seeded_history
                reconcile_seeded_history(self.symbol, tf_seconds, df)
            except Exception as e:
                logger.warning(
                    "seed_history cache reconcile failed sym=%s tf=%ss: %s",
                    self.symbol, tf_seconds, e)

    def seed_mtf_from_shadow(self, tf_seconds: int,
                             session: str = 'RTH') -> bool:
        """Publish a freshly-warmed shadow's confluence into the cross-TF gate
        dict (`_mtf_confluence[tf]`) right after warmup.

        BUG FIX (2026-06-26): `_mtf_confluence[tf]` was ONLY ever populated on a
        live `shadow.on_bar_close` — i.e. when a bar of that TF actually CLOSES
        live. For COARSE secondary TFs (1Hour/4Hour/1Day) a bar almost never
        closes during a session (a 1Day bar closes once/day), so their gate
        state stayed ABSENT from `_mtf_confluence` → any entry trigger needing a
        `1D-…`/`4H-…`/`1H-…` confluence had `subset_ok=0` FOREVER → the strategy
        fired 0 alerts live while its backtest fired normally (310/313: 77/85
        backtest trades in the live window, 0 alerts). The Bug-5 warmup already
        seeds the shadow INDICATOR (e.g. 246 daily bars) — this derives the
        confluence records from that warmed state (no extra indicator pass, via
        `_derive_confluence_records`) and seeds the gate dict so coarse gates are
        evaluable immediately, matching what the backtest sees (last CLOSED
        coarse bar). Updated normally thereafter on each live coarse close.
        Kill-switch RORT_SEED_MTF_FROM_WARMUP=0 reverts (instant rollback).

        Bug Hunt Wave 1 #1: seeds ONE (tf, session) key per call — the
        warmup loop calls it once per session-shadow of the TF."""
        if os.getenv('RORT_SEED_MTF_FROM_WARMUP', '1') != '1':
            return False
        shadow = self._shadow_engines.get((tf_seconds, session))
        if not (shadow and getattr(shadow.indicators, '_initialized', False)):
            return False
        try:
            recs = shadow._derive_confluence_records()
            self._mtf_confluence[(tf_seconds, session)] = recs
            logger.info("[MTF-SEED] %s tf=%ss session=%s: seeded cross-TF "
                        "gate from warmup (%d records) — coarse gates now "
                        "evaluable pre-close",
                        self.symbol, tf_seconds, session, len(recs))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[MTF-SEED] seed_mtf_from_shadow failed %s tf=%ss "
                           "session=%s: %s",
                           self.symbol, tf_seconds, session, e)
            return False

    def on_tick(self, price: float, volume: int, timestamp: datetime,
                alert_callback: Callable = None, config: dict = None,
                auditor: 'FidelityAuditor' = None,
                update_ohlc: bool = True, update_volume: bool = True):
        """Route tick to bar builders and strategy monitors.

        Phase G (2026-05-15): update_ohlc + update_volume flags plumb
        through to each builder. Vol-only trades (Polygon OHLC-
        ineligible but vol-eligible) contribute to volume without
        moving H/L/C. Default values preserve backward-compat.
        """
        self.tick_count += 1
        self.last_tick_time = timestamp
        ts_str = timestamp.isoformat()

        for tf_seconds, builder in self.builders.items():
            completed = builder.process_tick(
                price, volume, timestamp,
                update_ohlc=update_ohlc,
                update_volume=update_volume,
            )

            if completed is not None:
                # Bar close — update shared confluence buffer FIRST
                # so monitors on other TFs can see this TF's state.
                # Bug Hunt Wave 1 #1: one shadow per (tf, session); RTH
                # shadows eat the bar, non-RTH shadows reload their own
                # session's history inside _close_shadow_with_bar.
                for _sh_sess, shadow in self._shadow_items_for_tf(
                        tf_seconds):
                    if shadow.indicators._initialized:
                        self._close_shadow_with_bar(
                            tf_seconds, _sh_sess, shadow, completed)

                # Run all monitors for this timeframe
                bar_audit_positions = {}

                for monitor in self.monitors.values():
                    if monitor.tf_seconds != tf_seconds:
                        continue
                    if not _is_in_session(timestamp, monitor.session):
                        if self.tick_count % 5000 == 0:
                            logger.debug("Session skip: strat=%s session=%s ts=%s",
                                         monitor.strat_id, monitor.session, timestamp)
                        continue

                    signals, audit_data = monitor.on_bar_close(
                        completed, builder._bar_count,
                        mtf_confluence=self._mtf_confluence)

                    # Log every bar-close evaluation for diagnostics
                    pos_state = audit_data.get('position_state', '?')
                    trigger_bools = audit_data.get('trigger_booleans', {})
                    active_triggers = [k for k, v in trigger_bools.items() if v]
                    ind_vals = audit_data.get('indicator_values', {})
                    interp_states = audit_data.get('interpreter_states', {})
                    # Key indicator values for quick diagnosis
                    diag_parts = []
                    for k in ('utbot_direction', 'utbot_stop', 'utbot_atr'):
                        if k in ind_vals:
                            v = ind_vals[k]
                            diag_parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
                    interp_str = ",".join(f"{k}={v}" for k, v in interp_states.items()) if interp_states else "none"
                    # MTF confluence diagnostic
                    mtf_count = sum(
                        len(r) for k, r in self._mtf_confluence.items()
                        if k[0] != tf_seconds) if self._mtf_confluence else 0
                    conf_str = (f" mtf_records={mtf_count} "
                                f"confluence={len(monitor._current_confluence)}"
                                if mtf_count else "")
                    logger.info("BAR_CLOSE strat=%s bar=%d close=%.2f pos=%s "
                                "signals=%d triggers=%s ind=[%s] interp=[%s]%s",
                                monitor.strat_id, builder._bar_count,
                                completed['close'], pos_state,
                                len(signals) if signals else 0,
                                active_triggers if active_triggers else "none",
                                " ".join(diag_parts) if diag_parts else "none",
                                interp_str, conf_str)

                    if signals and alert_callback:
                        for sig in signals:
                            self._fire_alert(sig, monitor, config, alert_callback)

                    # Collect per-strategy position state for audit
                    bar_audit_positions[monitor.strat_id] = (
                        audit_data.get('position_state', 'FLAT'))

                    # Log fidelity audit for this monitor's bar close
                    if auditor:
                        auditor.log_bar_close(
                            symbol=self.symbol,
                            tf_seconds=tf_seconds,
                            bar=completed,
                            indicator_values=audit_data['indicator_values'],
                            trigger_booleans=audit_data['trigger_booleans'],
                            interpreter_states=audit_data[
                                'interpreter_states'],
                            positions={monitor.strat_id:
                                       audit_data['position_state']},
                        )

                # Publish this TF's confluence to shared buffer (from real
                # monitor if no shadow engine covers its (tf, session) key)
                # Bug Hunt Wave 1 #1: real monitors are primary-TF, so the
                # session belongs to the monitor — store under
                # (tf, monitor's session), first monitor per session wins
                # (flag OFF: single 'RTH' session == legacy first-monitor).
                _stored_sessions: Set[str] = set()
                for m in self.monitors.values():
                    if m.tf_seconds != tf_seconds:
                        continue
                    _m_sess = _shadow_session_for(m)
                    if _m_sess in _stored_sessions:
                        continue
                    _key = (tf_seconds, _m_sess)
                    if _key in self._mtf_confluence and \
                            self._shadow_engines.get(_key):
                        continue  # session-shadow owns this key
                    _stored_sessions.add(_m_sess)
                    # 2026-06-12 (B5 TRUE root cause): publish
                    # ONLY the monitor's OWN-TF records. The
                    # monitor's _current_confluence also carries
                    # records MERGED from every other TF (its own
                    # gating input, step 3c) — publishing the
                    # whole set re-broadcast stale copies of
                    # other TFs' states; round-trips accumulated
                    # until _mtf_confluence held ALL states of
                    # ALL interpreters (observed live: GATE_DIAG
                    # records carried all four 2M-UT_BOT states
                    # at once) → every gate's subset check passed
                    # → gates permanently open → the fleet-wide
                    # phantom-flood ladder of 2026-06-12.
                    _lbl = SECONDS_TO_TIMEFRAME.get(
                        tf_seconds, '1Min').replace(
                        'Min', 'M').replace('Hour', 'H').replace(
                        'Day', 'D').replace('Week', 'W')
                    own_records = {r for r in m._current_confluence
                                   if r.startswith(_lbl + '-')}
                    self._mtf_confluence[_key] = own_records

                # M8.5: broadcast completed bar to Supabase Realtime (live chart)
                self._publish_completed_bar(tf_seconds, completed)

            # Intra-bar tick checks for all monitors on this timeframe
            for monitor in self.monitors.values():
                if monitor.tf_seconds != tf_seconds:
                    continue
                if not _is_in_session(timestamp, monitor.session):
                    continue
                if not monitor.indicators._initialized:
                    continue

                signals = monitor.on_tick(price, ts_str,
                                          builder._bar_count)
                if signals and alert_callback:
                    for sig in signals:
                        self._fire_alert(sig, monitor, config, alert_callback)

    def emit_gate_telemetry(self, now: datetime) -> None:
        """Gate-state telemetry (2026-06-12, Kevin's 'show me what the
        engine actually sees'). Periodically snapshot `_mtf_confluence`
        for every TF that any gated monitor on this hub depends on, and
        persist CHANGES to bar_diagnostics (source='live_gate', one row
        per gated strategy per change) — recorded ground truth of the
        live gate, the thing every lens until now only RECONSTRUCTED.

        Also the freeze alarm: if a depended-on TF's records haven't
        changed in > 3 periods while the engine is processing, WARN —
        the silent >=120s close freeze (247afec, reverted 2026-06-12)
        would have surfaced in minutes instead of via a manual cache
        audit. Called from the periodic flush loop; cheap (set compares).
        """
        try:
            if not hasattr(self, '_gate_telemetry_state'):
                # (tf, session) -> (records, ts) — Bug Hunt Wave 1 #1
                self._gate_telemetry_state = {}
            needed: dict = {}
            for m in self.monitors.values():
                for sec in getattr(m, '_required_secondary_tf', ()):
                    needed.setdefault(
                        (sec, _shadow_session_for(m)), []).append(m.strat_id)
            if not needed:
                return
            rows = []
            for (tf, sess), sids in needed.items():
                cur = frozenset(self._mtf_confluence.get((tf, sess), set()))
                prev, prev_ts = self._gate_telemetry_state.get(
                    (tf, sess), (None, None))
                if cur != prev:
                    self._gate_telemetry_state[(tf, sess)] = (cur, now)
                    bar_iso = datetime.fromtimestamp(
                        int(now.timestamp()) - int(now.timestamp()) % tf,
                        tz=timezone.utc).isoformat()
                    _vals = {'tf': tf, 'records': sorted(cur)}
                    if MTF_SESSION_SHADOWS:
                        _vals['session'] = sess
                    for sid in sids:
                        rows.append({
                            'strategy_id': sid,
                            'bar_ts': bar_iso,
                            'source': 'live_gate',
                            'values': _vals,
                        })
                elif prev_ts is not None:
                    age = (now - prev_ts).total_seconds()
                    if age > 3 * tf and self.tick_count > 0:
                        logger.warning(
                            "GATE-FREEZE? sym=%s tf=%ss session=%s records "
                            "unchanged for %.0fs (>3 periods) — secondary "
                            "closes may have stopped (cf. 2026-06-12 "
                            "247afec freeze)",
                            self.symbol, tf, sess, age)
                        # re-arm so the warning fires once per stale period
                        self._gate_telemetry_state[(tf, sess)] = (prev, now)
            if rows:
                from live_bars_writer import _get_executor
                from db import save_bar_diagnostics_batch
                _get_executor().submit(save_bar_diagnostics_batch, rows)
        except Exception as e:
            logger.debug("gate telemetry emit failed: %s", e)

    def flush_stale_bars(self, now: datetime,
                         alert_callback: Callable = None,
                         config: dict = None,
                         auditor: 'FidelityAuditor' = None):
        """Force-close any bars whose period has expired (wall-clock driven).

        This ensures bar-close signals fire promptly even when no ticks
        arrive (e.g., after-hours low liquidity).  Called from the
        periodic tasks loop every PICKLE_WRITE_INTERVAL seconds.
        """
        for tf_seconds, builder in self.builders.items():
            completed = builder.force_close_stale_bar(now)
            if completed is None:
                continue

            # Gap healer (2026-06-11): force-close is the DOMINANT close
            # path for thin symbols (sparse per-second arrivals rarely
            # cross close_on_boundary at the right moment) — without this
            # hook their gaps only heal via the 120s sweeper. Found via
            # the KO canary: 26 ws closes / 12 min, zero at-close
            # detections.
            self._queue_gap_heal_if_gap(tf_seconds, builder)

            # M8.7 hotfix #2 (2026-05-01): only write SUB-MINUTE bars
            # from the flush path. For tf >= 60, the canonical write
            # comes from on_polygon_bar (primary AM) or its secondary
            # fan-out — the 60s+ builder's `_partial` here is fed by
            # per-second bars for chart-visual purposes ONLY (close_
            # on_boundary=False) and contains incomplete data. Writing
            # it would overwrite the canonical row with partial garbage.
            # Symptom: 1Min cache rows with `volume` 99% lower than
            # `first_volume` (canonical). Sub-minute primaries (10Sec,
            # 30Sec) legitimately rely on flush-driven writes because
            # Polygon's per-second A events arrive too sparsely to
            # always trigger close_on_boundary=True at the right moment.
            if tf_seconds < 60:
                try:
                    from live_bars_writer import write_bar as _live_bars_write
                    _live_bars_write(self.symbol, tf_seconds, completed,
                                     source='ws')
                except Exception:
                    pass
            elif (tf_seconds in self._shadow_tf_set()
                  and tf_seconds not in
                  {m.tf_seconds for m in self.monitors.values()}):
                # 2026-06-12 (clobber merge fix): PURE-SECONDARY >=60s
                # builders (the 2m gate) are fed ONLY by fan-out minutes
                # — their partial is canonical, so the flush close is
                # writable. Without this, the only >=120s live_bars
                # write path was the rebroadcast-replace branch (i.e.
                # the clobber itself): flush wins the close race nearly
                # always (zero grace for 1Min+), and once the clobber
                # was guarded, 2m rows stopped entirely (the 0612
                # "freeze" — write starvation, not close starvation).
                # Primary >=60s builders stay excluded (hotfix #2): their
                # partial carries per-second chart-visual price updates.
                try:
                    from live_bars_writer import write_bar as _live_bars_write
                    _live_bars_write(self.symbol, tf_seconds, completed,
                                     source='ws')
                except Exception:
                    pass

            # Update shared confluence buffer first (shadow or real monitor)
            # Bug Hunt Wave 1 #1: per-(tf, session) — see _close_shadow_with_bar
            for _sh_sess, shadow in self._shadow_items_for_tf(tf_seconds):
                if shadow.indicators._initialized:
                    self._close_shadow_with_bar(
                        tf_seconds, _sh_sess, shadow, completed)

            # Run all monitors for this timeframe — same as on_tick bar-close
            for monitor in self.monitors.values():
                if monitor.tf_seconds != tf_seconds:
                    continue
                if not _is_in_session(now, monitor.session):
                    continue

                signals, audit_data = monitor.on_bar_close(
                    completed, builder._bar_count,
                    mtf_confluence=self._mtf_confluence)

                # Diagnostic logging (same as tick-driven bar close)
                pos_state = audit_data.get('position_state', '?')
                trigger_bools = audit_data.get('trigger_booleans', {})
                active_triggers = [k for k, v in trigger_bools.items() if v]
                ind_vals = audit_data.get('indicator_values', {})
                interp_states = audit_data.get('interpreter_states', {})
                diag_parts = []
                # Surface any user-pack indicator value of interest. The
                # original list only handled utbot_v2 keys; here we include
                # utv4_* (UT Bot V4 user pack) and any prefixed booleans so
                # silent user-pack failures are visible in worker logs.
                _diag_keys = (
                    'utbot_direction', 'utbot_stop', 'utbot_atr',
                    'utv4_trailing_stop', 'utv4_trailing_stop_prev',
                    'utv4_bull_flip', 'utv4_bear_flip',
                )
                for k in _diag_keys:
                    if k in ind_vals:
                        v = ind_vals[k]
                        if isinstance(v, float):
                            diag_parts.append(f"{k}={v:.4f}")
                        elif isinstance(v, bool):
                            diag_parts.append(f"{k}={int(v)}")
                        else:
                            diag_parts.append(f"{k}={v}")
                interp_str = ",".join(f"{k}={v}" for k, v in interp_states.items()) if interp_states else "none"
                logger.info("BAR_CLOSE(flush) strat=%s bar=%d close=%.2f pos=%s "
                            "signals=%d triggers=%s ind=[%s] interp=[%s]",
                            monitor.strat_id, builder._bar_count,
                            completed['close'], pos_state,
                            len(signals) if signals else 0,
                            active_triggers if active_triggers else "none",
                            " ".join(diag_parts) if diag_parts else "none",
                            interp_str)

                if signals and alert_callback:
                    for sig in signals:
                        self._fire_alert(sig, monitor, config, alert_callback)

                if auditor:
                    auditor.log_bar_close(
                        symbol=self.symbol,
                        tf_seconds=tf_seconds,
                        bar=completed,
                        indicator_values=audit_data['indicator_values'],
                        trigger_booleans=audit_data['trigger_booleans'],
                        interpreter_states=audit_data[
                            'interpreter_states'],
                        positions={monitor.strat_id:
                                   audit_data['position_state']},
                    )

            # Publish real monitor's confluence to shared buffer
            # Bug Hunt Wave 1 #1: per (tf, monitor session) key; first
            # monitor per session (flag OFF == legacy first-monitor 'RTH').
            _stored_sessions: Set[str] = set()
            for m in self.monitors.values():
                if m.tf_seconds != tf_seconds:
                    continue
                _m_sess = _shadow_session_for(m)
                if _m_sess in _stored_sessions:
                    continue
                if self._shadow_engines.get((tf_seconds, _m_sess)):
                    continue  # session-shadow owns this key
                _stored_sessions.add(_m_sess)
                # B5 own-TF filter (2026-06-12) — see the
                # on_polygon_bar site for the full story.
                _lbl = SECONDS_TO_TIMEFRAME.get(
                    tf_seconds, '1Min').replace(
                    'Min', 'M').replace('Hour', 'H').replace(
                    'Day', 'D').replace('Week', 'W')
                own_records = {r for r in m._current_confluence
                               if r.startswith(_lbl + '-')}
                self._mtf_confluence[(tf_seconds, _m_sess)] = own_records

            # M8.5: broadcast completed bar to Supabase Realtime (live chart)
            self._publish_completed_bar(tf_seconds, completed)

    @_prof_fn('s_mon_pipeline')
    def _run_monitor_pipeline_for_completed_bar(
        self,
        tf_seconds: int,
        bar_dict: dict,
        bar_count: int,
        alert_callback: Optional[Callable] = None,
        config: dict = None,
        auditor: 'FidelityAuditor' = None,
        source_label: str = 'polygon',
    ) -> None:
        """Shared monitor pipeline for any completed bar (canonical AM bar OR
        sub-minute bar built via accept_second_bar). Updates shadow engines,
        runs monitors, dispatches alerts, logs audit, refreshes MTF confluence,
        and publishes the bar to the Realtime broadcast channel.

        Caller is responsible for adding the bar to the builder's history
        (via accept_bar or _close_bar inside accept_second_bar).
        """
        # Per-bar REST verification trigger (#57E+F, 2026-06-05). Queue a
        # bar-scoped verify for every closed bar — closes the drift gap on
        # bars that don't fire alerts. One queue per (symbol, tf_seconds,
        # bar_start) regardless of how many monitors are listening; the
        # splice writes propagate through the shared BarBuilder. Best-
        # effort: queue_verify_for_bar is no-op when REST_VERIFY_ENABLED
        # is unset, so the worker stays safe pre-rollout.
        try:
            from rest_verifier import queue_verify_for_bar, is_enabled as _rv_enabled
            if _rv_enabled():
                _bar_ts_raw = bar_dict.get('timestamp')
                _bar_close = bar_dict.get('close')
                if _bar_ts_raw is not None and _bar_close is not None:
                    if isinstance(_bar_ts_raw, str):
                        from datetime import datetime as _dt
                        _bts = _dt.fromisoformat(_bar_ts_raw.replace('Z', '+00:00'))
                    else:
                        _bts = _bar_ts_raw
                    # TF-based grace: faster check for sub-minute, slower
                    # for 1Min+ where settle time is forgivingly longer.
                    _grace = 5.0 if tf_seconds < 60 else 30.0
                    queue_verify_for_bar(
                        self.symbol, tf_seconds, _bts, float(_bar_close),
                        grace_seconds=_grace, pass_num=1,
                    )
        except Exception as _e:
            logger.warning(
                "queue_verify_for_bar dispatch failed sym=%s tf=%ss: %s",
                self.symbol, tf_seconds, _e)

        # Update shared confluence buffer (shadow engines first)
        # Bug Hunt Wave 1 #1: per-(tf, session) — see _close_shadow_with_bar
        for _sh_sess, _shadow in self._shadow_items_for_tf(tf_seconds):
            if _shadow.indicators._initialized:
                self._close_shadow_with_bar(
                    tf_seconds, _sh_sess, _shadow, bar_dict)

        # Collect committed indicator values + interpreter states from each
        # monitor so the live-chart broadcast carries fresh state on bar
        # close (matches what the backend chart-data endpoint would return
        # for this bar on next fetch).
        committed_indicators: Dict[str, float] = {}
        committed_states: Dict[str, str] = {}

        # Run all monitors for this timeframe
        for monitor in self.monitors.values():
            if monitor.tf_seconds != tf_seconds:
                continue
            if not _is_in_session(self.last_tick_time, monitor.session):
                continue
            # Phase C (2026-05-05): live_model gate — symmetric skip sets
            # so each monitor only fires on its declared bar source.
            #   ws_with_corrections → consume only AM-source bars (polygon)
            #   ws_agg_locked / ws_agg_with_rest_backfill → consume only
            #     ws_agg-source bars (client-side per-second aggregation)
            # Existing source labels: 'polygon' (on_polygon_bar AM events),
            # 'subm' (on_second_bar sub-minute primary), 'ws_agg' (Phase C
            # ws_agg dispatch from on_second_bar).  'subm' is sub-minute
            # only and never reaches 1Min monitors, so it's gate-irrelevant.
            if source_label == 'polygon' and monitor.live_model in (
                'ws_agg_locked', 'ws_agg_with_rest_backfill',
                # ws_rest_spliced (2026-05-28) consumes ws_agg-source
                # bars exactly like ws_agg_reconciled does. Add to the
                # polygon-skip list so AM events bypass this monitor.
                'ws_rest_spliced',
            ):
                continue
            if source_label == 'ws_agg' and monitor.live_model == 'ws_with_corrections':
                continue

            signals, audit_data = monitor.on_bar_close(
                bar_dict, bar_count,
                mtf_confluence=self._mtf_confluence)

            # LEF Phase 2b (2026-05-20): if a ws_agg_reconciled monitor
            # already fired at strategy-grace for this bucket, suppress
            # any signals on_bar_close re-emits — we already dispatched
            # at the partial-bucket fire-time. Indicator state still
            # commits (engine.update_bar ran inside on_bar_close), which
            # is exactly what we want: official commit at builder close.
            #
            # ws_rest_spliced (2026-05-28): identical suppression behavior
            # — fires at grace, suppresses re-fire at close. The ONLY
            # difference from ws_agg_reconciled is the rest_verifier hook
            # on the alert-fire path (handled in dispatch, not here).
            if monitor.live_model in (
                'ws_agg_reconciled', 'ws_rest_spliced',
            ) and signals:
                try:
                    _bs_dt = pd.Timestamp(bar_dict['timestamp'])
                    if _bs_dt.tzinfo is None:
                        _bs_dt = _bs_dt.tz_localize('UTC')
                    if monitor._fired_bucket == int(_bs_dt.timestamp()):
                        signals = []
                except Exception:
                    pass  # if timestamp parsing fails, let signals through

            # M8.7 M6 (2026-05-02): capture engine indicator state at bar
            # close for engine-truth replay/analysis. Fire-and-forget,
            # gated by BAR_ENGINE_STATE_WRITE_ENABLED env flag. Skipped
            # silently when disabled (default).
            try:
                from bar_engine_state_writer import write_state as _bes_write
                _bes_write(
                    strategy_id=monitor.strat_id,
                    bar_start=bar_dict.get('timestamp'),
                    indicator_state=dict(monitor.indicators.state.current or {}),
                    source='ws',
                )
            except Exception:
                pass  # never block bar processing on state capture

            pos_state = audit_data.get('position_state', '?')
            trigger_bools = audit_data.get('trigger_booleans', {})
            active_triggers = [k for k, v in trigger_bools.items() if v]
            logger.info("BAR_CLOSE(%s) strat=%s bar=%d tf=%ss close=%.2f "
                        "pos=%s signals=%d triggers=%s",
                        source_label, monitor.strat_id, bar_count, tf_seconds,
                        bar_dict['close'], pos_state,
                        len(signals) if signals else 0,
                        active_triggers if active_triggers else "none")

            committed_indicators.update(audit_data.get('indicator_values', {}))
            committed_states.update(audit_data.get('interpreter_states', {}))

            if signals and alert_callback:
                for sig in signals:
                    self._fire_alert(sig, monitor, config, alert_callback)

            # Bar-close hook: fire-and-forget async task to trigger the
            # forward-test recompute of stored_trades. Must not block
            # the tick loop. Handler is responsible for its own
            # throttling/concurrency control.
            if self.on_bar_close_hook is not None:
                try:
                    loop = asyncio.get_event_loop()
                    task = loop.create_task(self.on_bar_close_hook(
                        strategy_id=monitor.strat_id,
                        user_id=monitor.strategy.get('user_id'),
                        tf_seconds=tf_seconds,
                    ))
                    self._pending_bar_close_tasks.add(task)
                    task.add_done_callback(
                        self._pending_bar_close_tasks.discard)
                except RuntimeError:
                    # No running event loop — best-effort, non-critical.
                    pass
                except Exception as e:
                    logger.debug("bar_close_hook scheduling failed: %s", e)

            if auditor:
                auditor.log_bar_close(
                    symbol=self.symbol, tf_seconds=tf_seconds,
                    bar=bar_dict,
                    indicator_values=audit_data['indicator_values'],
                    trigger_booleans=audit_data['trigger_booleans'],
                    interpreter_states=audit_data['interpreter_states'],
                    positions={monitor.strat_id: audit_data['position_state']},
                )

        # Publish real monitor's confluence
        # Bug Hunt Wave 1 #1: per (tf, monitor session) key; first monitor
        # per session (flag OFF == legacy first-monitor 'RTH').
        _stored_sessions: Set[str] = set()
        _tf_sessions: Set[str] = set()
        for m in self.monitors.values():
            if m.tf_seconds != tf_seconds:
                continue
            _m_sess = _shadow_session_for(m)
            _tf_sessions.add(_m_sess)
            if _m_sess in _stored_sessions:
                continue
            if self._shadow_engines.get((tf_seconds, _m_sess)):
                continue  # session-shadow owns this key
            _stored_sessions.add(_m_sess)
            # B5 own-TF filter (2026-06-12) — see the
            # on_polygon_bar site for the full story.
            _lbl = SECONDS_TO_TIMEFRAME.get(
                tf_seconds, '1Min').replace(
                'Min', 'M').replace('Hour', 'H').replace(
                'Day', 'D').replace('Week', 'W')
            own_records = {r for r in m._current_confluence
                           if r.startswith(_lbl + '-')}
            self._mtf_confluence[(tf_seconds, _m_sess)] = own_records
        if not _tf_sessions:
            _tf_sessions = {'RTH'}

        # Build cross-TF state snapshot the same way as the forming-bar path
        # Bug Hunt Wave 1 #1: session-filtered like _gather_tentative_state.
        state_cross_tf: Dict[str, str] = {}
        for (other_tf, _sess), records in self._mtf_confluence.items():
            if other_tf == tf_seconds:
                continue
            if _sess not in _tf_sessions:
                continue
            for rec in records:
                if rec.startswith('GEN-'):
                    continue
                parts = rec.split('-', 2)
                if len(parts) < 3:
                    continue
                tf_label, interp_key, state_val = parts
                col = f"{interp_key}__{tf_label.lower()}"
                state_cross_tf[col] = state_val

        # M8.5: broadcast completed bar to Supabase Realtime (live chart)
        extras = {
            'indicators': committed_indicators,
            'states': committed_states,
            'state_cross_tf': state_cross_tf,
        } if (committed_indicators or committed_states or state_cross_tf) else None
        self._publish_completed_bar(
            tf_seconds, bar_dict, is_forming=False, extras=extras)

    @_prof_fn('s_fanout_sec')
    def _fanout_to_secondary_builders(
        self,
        bar_dict: dict,
        source_label: str = 'ws',
    ) -> None:
        """Fan out a 1Min bar to all ≥60s secondary builders.

        Task #11 (2026-05-14): called from BOTH the AM channel
        (on_polygon_bar) AND the ws_agg minute close path (on_second_bar).
        Each call MAY advance a secondary builder's partial and close it
        on boundary. live_bars writes use the supplied `source_label`
        ('ws' for AM, 'ws_agg' for ws_agg path) so DB rows disambiguate
        which path the close came from.

        Sub-minute shadows (<60s) are NOT handled here — they're fed
        from the dedicated sub-minute fan-out in on_second_bar at the
        per-second cadence.

        ⚠️ Volume / tick_count caveat: when BOTH paths deliver the same
        1Min bar (typical case during active markets), the secondary
        builder's `_partial.volume` and `_partial.tick_count` accumulate
        twice. Once AM arrives and the bar transitions from forming to
        history, AM's rebroadcast branch (BarBuilder.accept_second_bar
        lines ~303-323) replaces history.iloc[-1] with AM's canonical
        values — healing the double-count for closed bars. Periods
        where AM never arrives keep ws_agg's accurate per-second volume.
        Affects: VWAP_V2, RVOL_V2 (volume-based interpreters).
        Not affected: SWING_123, EMA_*, MACD_*, UT_BOT_V4 (price-based).
        """
        # Bug Hunt Wave 1 #1: shadow keys are (tf, session) — iterate each
        # distinct TF once (insertion order preserved).
        for sec_tf in list(dict.fromkeys(
                k[0] for k in self._shadow_engines)):
            if sec_tf < 60:
                continue  # sub-minute handled in on_second_bar
            sec_builder = self.builders.get(sec_tf)
            if sec_builder is None:
                continue
            try:
                completed = sec_builder.accept_second_bar(
                    bar_dict, close_on_boundary=True,
                    full_period_input=False)
            except Exception as e:
                logger.warning(
                    "secondary-TF aggregation failed (%ss, src=%s): %s",
                    sec_tf, source_label, e)
                continue
            if completed is None:
                continue
            sec_was_duplicate = sec_builder.last_was_duplicate
            if not sec_was_duplicate:
                # Gap healer (2026-06-11): secondary builders (the 2m
                # gate) gap-heal independently of the primary.
                self._queue_gap_heal_if_gap(sec_tf, sec_builder)
            try:
                from live_bars_writer import write_bar as _live_bars_write
                _live_bars_write(
                    self.symbol, sec_tf, completed, source=source_label)
            except Exception:
                pass
            shadow_items = self._shadow_items_for_tf(sec_tf)
            if not shadow_items:
                continue

            if sec_was_duplicate:
                # M8.7 (2026-05-02): rebroadcast cascade. Recompute shadow
                # indicators from corrected history.
                # 2026-06-09 (iter 0609a): ALSO re-emit confluence records.
                # Previously this branch left `_mtf_confluence[sec_tf]`
                # frozen at the boot-time state, so 2m+ cross-TF gates went
                # stale (passed/blocked forever based on the secondary TF's
                # state at the first close after worker start). See
                # `_ShadowIndicatorEngine.recompute_confluence`.
                # Bug Hunt Wave 1 #1: RTH shadow only — non-RTH shadows
                # never consumed this RTH-built fan-out history, so a
                # correction to it can't affect their state; their records
                # refresh via the session reload on real closes.
                for _sh_sess, shadow in shadow_items:
                    if _sh_sess != 'RTH':
                        continue
                    try:
                        self._mtf_confluence[(sec_tf, 'RTH')] = \
                            shadow.recompute_confluence(sec_builder.history)
                        logger.info(
                            "Rebroadcast cascade: shadow %s/%ss recomputed, "
                            "records refreshed=%s (src=%s)",
                            self.symbol, sec_tf,
                            self._mtf_confluence[(sec_tf, 'RTH')],
                            source_label)
                    except Exception as e:
                        logger.warning(
                            "shadow recompute_confluence failed (%ss): %s",
                            sec_tf, e)
                continue

            for _sh_sess, shadow in shadow_items:
                try:
                    self._close_shadow_with_bar(
                        sec_tf, _sh_sess, shadow, completed)
                    logger.info(
                        "shadow_close %s/%ss session=%s close=%.2f "
                        "records=%s (src=%s)",
                        self.symbol, sec_tf, _sh_sess,
                        float(completed['close']),
                        self._mtf_confluence.get((sec_tf, _sh_sess)),
                        source_label)
                except Exception as e:
                    logger.warning(
                        "shadow on_bar_close failed (%ss session=%s): %s",
                        sec_tf, _sh_sess, e)

    @_prof_fn('s_fanout_pri')
    def _fanout_to_primary_builders_gt_60(
        self,
        completed_min: dict,
        alert_callback: Callable = None,
        config: dict = None,
        auditor: 'FidelityAuditor' = None,
    ) -> None:
        """Fan out a 1Min ws_agg bar to >60s PRIMARY builders.

        Task #12 (2026-05-14): When AM channel is silent, ≥5Min primary
        builders (5Min, 1H, 1D) have NO close path — monitors never fire
        on_bar_close, alerts never dispatch, and the strategy's own
        `_mtf_confluence[tf]` (own_records buffer) stays stale, cascade-
        blocking dependent strategies (sid 126 needs sid 137's 5m).

        Eligibility: only fires for monitors whose live_model is
        'ws_agg_locked' or 'ws_agg_with_rest_backfill'. The pipeline's
        internal source-label gate (in _run_monitor_pipeline_for_completed_bar)
        also re-checks, but filtering at the call site avoids unnecessary
        accept_second_bar calls when nobody's listening.

        Rebroadcast handling: if AM later delivers the same period, the
        builder's accept_bar duplicate-detection branch will correct
        history.iloc[-1] and recompute indicators (existing on_polygon_bar
        path). This helper's `last_was_duplicate` check handles the
        SYMMETRIC case (AM first, then ws_agg) by recomputing silently.

        ⚠️ Volume caveat: the per-second loop in on_second_bar continues
        to update the >60s primary's _partial with close_on_boundary=False
        for chart-visual forming-bar publish. So during the forming period,
        BOTH the per-second loop AND this fan-out (via the per-minute
        accept_second_bar(close_on_boundary=True)) contribute to volume.
        Closed history bars therefore reflect ~2x volume for the forming
        period when ws_agg drives the close. Affects volume-based
        interpreters (VWAP_V2, RVOL_V2) only. Price-based interpreters
        (EMA, MACD, SWING_123, UT_BOT_V4) are unaffected. Future cleanup
        would add a `skip_volume` param to BarBuilder.accept_second_bar.
        """
        primary_tfs = {m.tf_seconds for m in self.monitors.values()}
        for tf in primary_tfs:
            if tf <= 60:
                continue  # 1Min has its own Phase C path; sub-minute is per-second
            eligible_at_tf = any(
                m.tf_seconds == tf
                and m.live_model in ('ws_agg_locked', 'ws_agg_with_rest_backfill')
                for m in self.monitors.values()
            )
            if not eligible_at_tf:
                continue
            builder = self.builders.get(tf)
            if builder is None:
                continue
            try:
                completed = builder.accept_second_bar(
                    completed_min, close_on_boundary=True,
                    full_period_input=False)
            except Exception as e:
                logger.warning(
                    "primary >60s aggregation failed (%ss, src=ws_agg): %s",
                    tf, e)
                continue
            if completed is None:
                continue
            primary_was_duplicate = builder.last_was_duplicate
            primary_was_correction = builder.last_was_correction
            if not primary_was_duplicate:
                # Gap healer (2026-06-11).
                self._queue_gap_heal_if_gap(tf, builder)
            try:
                from live_bars_writer import write_bar as _live_bars_write
                _live_bars_write(
                    self.symbol, tf, completed, source='ws_agg')
            except Exception:
                pass

            if primary_was_duplicate:
                # AM beat us to this period and ws_agg is the rebroadcast.
                # Only recompute if values actually changed — a pure
                # re-delivery (identical OHLCV) needs no work. Never
                # re-fire alerts. See last_was_correction (spiral fix).
                if primary_was_correction:
                    for monitor in self.monitors.values():
                        if monitor.tf_seconds != tf:
                            continue
                        try:
                            monitor.indicators.recompute_from_history(builder.history)
                        except Exception as e:
                            logger.warning(
                                "Rebroadcast recompute failed for strat %s "
                                "(tf=%ss, src=ws_agg): %s",
                                monitor.strat_id, tf, e)
                    logger.info(
                        "Rebroadcast applied: %s tf=%ss bar=%s — "
                        "indicators recomputed (>60s primary, src=ws_agg), "
                        "no alerts", self.symbol, tf,
                        completed.get('timestamp'))
                continue

            try:
                self._run_monitor_pipeline_for_completed_bar(
                    tf_seconds=tf,
                    bar_dict=completed,
                    bar_count=builder._bar_count,
                    alert_callback=alert_callback,
                    config=config,
                    auditor=auditor,
                    source_label='ws_agg',
                )
            except Exception as e:
                logger.warning(
                    "ws_agg primary pipeline dispatch failed "
                    "(%ss, %s): %s", tf, self.symbol, e)

    @_prof_fn('s_on_polygon_bar')
    def on_polygon_bar(self, bar_dict: dict, tf_seconds: int,
                        alert_callback: Callable = None,
                        config: dict = None,
                        auditor: 'FidelityAuditor' = None):
        """Handle a pre-aggregated bar from Polygon WebSocket.

        Bypasses tick aggregation — the bar arrives already complete.
        Routes through the same monitor pipeline as tick-built bars.
        """
        builder = self.builders.get(tf_seconds)
        if builder is None:
            return

        timestamp = pd.Timestamp(bar_dict['timestamp'])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        self.last_tick_time = timestamp.to_pydatetime()
        self.tick_count += 1

        # Trade_Timestamps_Spec: record the bar close so alert dispatch
        # can stamp actual_price on any alert that fires before the next
        # per-second bar arrives.
        close = bar_dict.get('close')
        if close:
            self.latest_close = float(close)

        # Accept the pre-built bar into the builder (handles gap-fill).
        # M8.7 (2026-05-02): if Polygon WS rebroadcasts a correction for
        # the most recent bar within the 15-min FINRA window, accept_bar
        # replaces the history row in place and sets last_was_duplicate.
        builder.accept_bar(bar_dict)
        was_duplicate = builder.last_was_duplicate
        was_correction = builder.last_was_correction
        if not was_duplicate:
            # Gap healer (2026-06-11): a fresh close is the cheapest
            # moment to spot a hole between the last two history bars.
            self._queue_gap_heal_if_gap(tf_seconds, builder)

        # M8.7: record the WS-aggregated bar to live_bars (fire-and-forget).
        # Always write — even on duplicate, since this is how cache.close
        # captures Polygon's correction (the trigger preserves first_close).
        try:
            from live_bars_writer import write_bar as _live_bars_write
            _live_bars_write(self.symbol, tf_seconds, bar_dict, source='ws')
        except Exception:
            pass  # never block bar processing on cache write

        if was_duplicate:
            # Polygon rebroadcast: silently recompute indicator state from
            # the corrected history so future bars use accurate values.
            # Do NOT re-fire alerts — once an alert is saved it's fact;
            # the corrected bar's own trigger evaluation was already done
            # with the original values at the moment it first arrived.
            # Skip the recompute entirely on a pure re-delivery (identical
            # OHLCV) — that work is wasted and drives the lag spiral.
            if was_correction:
                for monitor in self.monitors.values():
                    if monitor.tf_seconds != tf_seconds:
                        continue
                    try:
                        monitor.indicators.recompute_from_history(builder.history)
                    except Exception as e:
                        logger.warning(
                            "Rebroadcast recompute failed for strat %s "
                            "(tf=%ss): %s",
                            monitor.strat_id, tf_seconds, e)
                    # User-pack engines: if the monitor has any, they live on
                    # IncrementalIndicatorEngine via _user_pack_engines; the
                    # __init__ called by recompute_from_history rebuilds them
                    # cleanly with fresh state.
                # Also recompute any shadow engine for this TF.
                # Bug Hunt Wave 1 #1: RTH key only — builder.history is the
                # RTH-flavored live feed; non-RTH shadows never consume it.
                shadow_self = self._shadow_engines.get((tf_seconds, 'RTH'))
                if shadow_self is not None:
                    try:
                        shadow_self.indicators.recompute_from_history(builder.history)
                    except Exception as e:
                        logger.warning(
                            "Rebroadcast recompute failed for shadow %ss: %s",
                            tf_seconds, e)
                logger.info(
                    "Rebroadcast applied: %s tf=%ss bar=%s — indicators "
                    "recomputed, no alerts fired",
                    self.symbol, tf_seconds, bar_dict.get('timestamp'))
        else:
            self._run_monitor_pipeline_for_completed_bar(
                tf_seconds=tf_seconds,
                bar_dict=bar_dict,
                bar_count=builder._bar_count,
                alert_callback=alert_callback,
                config=config,
                auditor=auditor,
                source_label='polygon',
            )

        # Phase 31 polygon path stops here for the primary TF only.
        # Fan out the just-closed 1Min AM bar to all ≥60s secondary
        # builders so cross-TF confluence shadows stay current.
        # Extracted to helper 2026-05-14 (task #11) for reuse by the
        # ws_agg minute-close path in on_second_bar.
        self._fanout_to_secondary_builders(bar_dict, source_label='ws')

    def apply_rest_correction(self, tf_seconds: int,
                              rest_bar_dict: dict,
                              rest_per_second_bars: list = None) -> bool:
        # ws_rest_spliced (2026-05-28, refactored same day for Phase A+B+C):
        # accepts a corrected bar from REST and updates the BarBuilder
        # history + indicator state to converge on REST-aligned values.
        # Three paths depending on where the bar is in history:
        #   (1) Bar is the latest in builder.history → existing K=1 fast
        #       path via accept_bar + apply_last_bar_correction. Preserves
        #       Polygon rebroadcast semantics.
        #   (2) Bar is in [latest-K, latest-1] window → multi-bar-back
        #       replay via the new apply_bar_correction. Drops the prior
        #       "stale" rejection — sub-minute TFs need this because REST
        #       settle is usually >1 bar period.
        #   (3) Bar is older than the K-window or not in history at all →
        #       reject (logged), can't be corrected.
        # If rest_per_second_bars is provided AND BarBuilder has per-second
        # history for the bar, splice at per-second granularity and
        # re-aggregate (Phase C). Otherwise replace the whole-bar OHLCV
        # with rest_bar_dict's values (Phase B fallback).
        # Invoked off-thread by rest_verifier; per-symbol mutex held by
        # the verifier.
        builder = self.builders.get(tf_seconds)
        if builder is None or len(builder.history) == 0:
            return False
        rest_ts = pd.Timestamp(rest_bar_dict['timestamp'])
        if rest_ts.tzinfo is None:
            rest_ts = rest_ts.tz_localize('UTC')
        # 2026-05-28: defense-in-depth alignment. The verifier should
        # already pass a bar-boundary timestamp, but L-type intra-bar
        # alerts have historically stamped bar_time = fill_ts (intra-bar
        # second). Align down to the bar boundary so the history lookup
        # succeeds for those.
        try:
            _aligned = builder._align_to_period(rest_ts.to_pydatetime())
            rest_ts = pd.Timestamp(_aligned)
            if rest_ts.tzinfo is None:
                rest_ts = rest_ts.tz_localize('UTC')
        except Exception:
            pass
        # Locate the target bar in history.
        idx_loc = None
        for i in range(len(builder.history) - 1, -1, -1):
            ts = builder.history.index[i]
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            if ts == rest_ts:
                idx_loc = i
                break
        if idx_loc is None:
            logger.info(
                "apply_rest_correction: bar not in history sym=%s tf=%ss "
                "rest_bar=%s — skipping",
                self.symbol, tf_seconds, rest_ts)
            return False
        offset_from_latest = (len(builder.history) - 1) - idx_loc

        # Phase C: if per-second list provided AND per-second history
        # exists for this bar_start, splice at per-second granularity and
        # re-aggregate. Otherwise use the whole-bar values directly.
        per_sec_used = False
        effective_bar = dict(rest_bar_dict)
        if (rest_per_second_bars
                and hasattr(builder, '_per_second_history')
                and rest_ts in builder._per_second_history):
            try:
                ws_per_sec = list(builder._per_second_history[rest_ts])
                # Build a {ts: rest_per_sec_dict} index for splice.
                rest_index = {}
                for rb in rest_per_second_bars:
                    rb_ts = pd.Timestamp(rb.get('timestamp'))
                    if rb_ts.tzinfo is None:
                        rb_ts = rb_ts.tz_localize('UTC')
                    rest_index[rb_ts] = rb
                # Splice: for each WS per-sec entry, if a REST entry
                # exists at the same timestamp, use REST's OHLCV;
                # otherwise keep WS's. Append any REST per-sec entries
                # that fall in window but have no WS counterpart.
                spliced: list = []
                ws_ts_set = set()
                for wb in ws_per_sec:
                    wb_ts = pd.Timestamp(wb.get('timestamp'))
                    if wb_ts.tzinfo is None:
                        wb_ts = wb_ts.tz_localize('UTC')
                    ws_ts_set.add(wb_ts)
                    if wb_ts in rest_index:
                        spliced.append(rest_index[wb_ts])
                    else:
                        spliced.append(wb)
                bar_end = rest_ts + pd.Timedelta(seconds=tf_seconds)
                for rb_ts, rb in rest_index.items():
                    if rb_ts in ws_ts_set:
                        continue
                    if rest_ts <= rb_ts < bar_end:
                        spliced.append(rb)
                # Persist the spliced per-second list (so future
                # corrections see corrected state).
                builder._per_second_history[rest_ts] = spliced
                # Re-aggregate using accept_second_bar's aggregation
                # rules: open=first, high=max, low=min, close=last,
                # volume=sum. Sort by timestamp first.
                spliced.sort(key=lambda b: pd.Timestamp(b.get('timestamp')))
                if spliced:
                    agg_open = float(spliced[0]['open'])
                    agg_close = float(spliced[-1]['close'])
                    agg_high = max(float(b['high']) for b in spliced)
                    agg_low = min(float(b['low']) for b in spliced)
                    agg_vol = sum(float(b.get('volume', 0))
                                  for b in spliced)
                    effective_bar = {
                        'timestamp': rest_ts,
                        'open': agg_open, 'high': agg_high,
                        'low': agg_low, 'close': agg_close,
                        'volume': agg_vol,
                    }
                    per_sec_used = True
            except Exception as e:
                logger.warning(
                    "apply_rest_correction: per-second splice failed "
                    "sym=%s tf=%ss bar=%s: %s — falling back to whole-bar",
                    self.symbol, tf_seconds, rest_ts, e)
                effective_bar = dict(rest_bar_dict)

        # Replace history row with effective_bar's OHLCV (handles both
        # the K=1 latest case and any K-back case uniformly).
        _cols = ['open', 'high', 'low', 'close', 'volume']
        _new = [
            float(effective_bar['open']),
            float(effective_bar['high']),
            float(effective_bar['low']),
            float(effective_bar['close']),
            float(effective_bar.get('volume', 0)),
        ]
        builder.history.loc[builder.history.index[idx_loc], _cols] = _new

        # Indicator state recompute — pick the right path.
        path_label = 'bar_level'
        any_applied = False
        for monitor in self.monitors.values():
            if monitor.tf_seconds != tf_seconds:
                continue
            # Build per-monitor post-correction replay callback that emits
            # source='live_corrected' rows to bar_diagnostics. Only attach
            # when the monitor has diagnostic_columns declared and we're
            # going through the K>1 replay path (K=1 fast path doesn't
            # replay multiple bars). The callback projects each replayed
            # bar's update_bar return into the diagnostic schema and
            # writes via the existing save_bar_diagnostics_batch (best-
            # effort, swallows errors). 2026-06-05 #57G.
            _diag_cols = getattr(monitor, '_diag_cols', None) or []
            _live_corr_buf: list = []
            if _diag_cols:
                def _replay_cb(_ts, _vals, _sid=monitor.strat_id, _cols=_diag_cols, _buf=_live_corr_buf):
                    _row: dict = {}
                    for _c in _cols:
                        _v = (_vals or {}).get(_c)
                        if _v is None:
                            continue
                        try:
                            _fv = float(_v)
                            if _fv == _fv and _fv not in (float('inf'), float('-inf')):
                                _row[_c] = _fv
                        except (TypeError, ValueError):
                            continue
                    if not _row:
                        return
                    _bts_iso = (_ts.isoformat()
                                if hasattr(_ts, 'isoformat') else str(_ts))
                    _buf.append({
                        'strategy_id': _sid,
                        'bar_ts': _bts_iso,
                        'source': 'live_corrected',
                        'values': _row,
                    })
            else:
                _replay_cb = None
            try:
                if offset_from_latest == 0:
                    # K=1 fast path — preserves existing apply_last_bar_
                    # correction behavior. After it returns, the engine's
                    # state.current holds the post-correction values dict
                    # (including user-pack outputs — see unified_engine
                    # line 1368). Project diagnostic columns from that.
                    ok = monitor.indicators.apply_last_bar_correction(
                        builder.history)
                    if ok and _diag_cols:
                        try:
                            _post = monitor.indicators.state.current or {}
                            _row: dict = {}
                            for _c in _diag_cols:
                                _v = _post.get(_c)
                                if _v is None:
                                    continue
                                try:
                                    _fv = float(_v)
                                    if _fv == _fv and _fv not in (float('inf'), float('-inf')):
                                        _row[_c] = _fv
                                except (TypeError, ValueError):
                                    continue
                            if _row:
                                _live_corr_buf.append({
                                    'strategy_id': monitor.strat_id,
                                    'bar_ts': rest_ts.isoformat(),
                                    'source': 'live_corrected',
                                    'values': _row,
                                })
                        except Exception as _post_e:
                            logger.warning(
                                "live_corrected K=1 snapshot sid=%s failed: %s",
                                monitor.strat_id, _post_e)
                else:
                    # K>1 path — restore from snapshot buffer, replay
                    # forward. Callback fires per replayed bar.
                    ok = monitor.indicators.apply_bar_correction(
                        rest_ts, builder.history,
                        replay_callback=_replay_cb)
                if not ok:
                    # Snapshot wasn't in buffer (engine cold-restarted,
                    # or bar aged out). Full replay fallback.
                    monitor.indicators.recompute_from_history(builder.history)
                any_applied = True
                # Flush this monitor's live_corrected buffer (covers both
                # K=1 explicit snapshot and K>1 callback-emitted rows).
                if _live_corr_buf:
                    try:
                        from db import save_bar_diagnostics_batch
                        save_bar_diagnostics_batch(_live_corr_buf)
                    except Exception as _e:
                        logger.warning(
                            "live_corrected flush failed sid=%s: %s",
                            monitor.strat_id, _e)

                # Post-hoc reclassification (#57H, 2026-06-05). For any
                # alert that fired on a corrected bar, check whether the
                # REST-aligned indicator state would have ALSO fired the
                # same trigger. The post-correction state.current dict
                # contains the trigger booleans under their trigger_id
                # keys (e.g. 'utv4_bull_flip'). If False after splice,
                # the alert was a phantom_pre_correction (structural WS
                # noise, not a real bug). Stored on the alerts row;
                # strategy_health classifier reads it to surface a
                # phantom_pre_correction bucket separate from
                # needs_investigation. Best-effort, never raises.
                if ok:
                    try:
                        _post_current = (
                            getattr(monitor.indicators, 'state', None)
                            and monitor.indicators.state.current
                        ) or {}
                        if _post_current:
                            from db import get_admin_client as _gac
                            _rcl = _gac()
                            # Query alerts in [bar_start, bar_start+tf_seconds).
                            # L-type alerts stamp bar_time = intra-bar fill_ts
                            # (not the boundary), so exact equality misses
                            # them. The range covers C-type (boundary) AND
                            # L-type (intra-bar) alerts for this bar.
                            _bar_end = (rest_ts + pd.Timedelta(seconds=tf_seconds))
                            _ar = (_rcl.table('alerts')
                                   .select('id,trigger_id,bar_time')
                                   .eq('strategy_id', monitor.strat_id)
                                   .gte('bar_time', rest_ts.isoformat())
                                   .lt('bar_time', _bar_end.isoformat())
                                   .execute())
                            for _alert in _ar.data or []:
                                _trig = _alert.get('trigger_id')
                                if not _trig:
                                    continue
                                # Strip exec-type suffix to find the base
                                # trigger column. Suffixes: _ib, _lc, _cc,
                                # _hm, _hl. The indicator class returns
                                # values under the base trigger name only
                                # (e.g. 'utv4_bull_flip').
                                _base = _trig
                                for _suf in ('_ib', '_lc', '_cc', '_hm', '_hl'):
                                    if _base.endswith(_suf):
                                        _base = _base[:-len(_suf)]
                                        break
                                _would_fire = bool(_post_current.get(_base, False))
                                (_rcl.table('alerts')
                                 .update({'would_fire_post_correction': _would_fire})
                                 .eq('id', _alert['id']).execute())
                                logger.info(
                                    "post_correction reclass sid=%s alert=%s "
                                    "trigger=%s would_fire=%s",
                                    monitor.strat_id, _alert['id'],
                                    _trig, _would_fire)
                    except Exception as _rc_e:
                        logger.warning(
                            "post_correction reclass sid=%s failed: %s",
                            monitor.strat_id, _rc_e)
            except Exception as e:
                logger.warning(
                    "REST correction recompute failed sym=%s strat=%s "
                    "tf=%ss K=%d: %s",
                    self.symbol, monitor.strat_id, tf_seconds,
                    offset_from_latest, e)
        # Shadow engine (cross-TF confluence) — same logic.
        # Bug Hunt Wave 1 #1: RTH key only — the corrected builder.history
        # is the RTH-flavored feed; non-RTH shadows never consume it (their
        # records refresh via session reloads on closes).
        shadow_self = self._shadow_engines.get((tf_seconds, 'RTH'))
        if shadow_self is not None:
            try:
                if offset_from_latest == 0:
                    ok = shadow_self.indicators.apply_last_bar_correction(
                        builder.history)
                else:
                    ok = shadow_self.indicators.apply_bar_correction(
                        rest_ts, builder.history)
                if not ok:
                    shadow_self.indicators.recompute_from_history(
                        builder.history)
                # 2026-06-12 (B5 gate-flood fix): re-derive the gate
                # records from the CORRECTED state. This branch recomputed
                # the shadow's indicators but left `_mtf_confluence[tf]`
                # frozen on pre-correction state for up to a full
                # secondary period after EVERY REST correction (~23/window
                # on SPY) — the live gate kept passing/blocking entries on
                # stale state while the backtest used corrected bars.
                # Fleet evidence 2026-06-12: gated phantom ratio tracks
                # gate flip-frequency (INSIDE 11% < UT_BOT 42% < SWING 63%
                # < SUPERTREND 73% < ungated ~91-97%) with matched pairs
                # ~100%. Mirrors the 0609a fan-out rebroadcast fix, which
                # never reached this path.
                self._mtf_confluence[(tf_seconds, 'RTH')] = \
                    shadow_self._derive_confluence_records()
            except Exception as e:
                logger.warning(
                    "REST correction shadow recompute failed sym=%s "
                    "tf=%ss K=%d: %s",
                    self.symbol, tf_seconds, offset_from_latest, e)
        try:
            from live_bars_writer import write_bar as _live_bars_write
            _live_bars_write(self.symbol, tf_seconds, effective_bar,
                             source='rest_correction')
        except Exception:
            pass
        if any_applied or shadow_self is not None:
            if per_sec_used:
                path_label = 'per_second'
            logger.info(
                "REST correction applied: sym=%s tf=%ss bar=%s K=%d "
                "path=%s — indicators recomputed, no alerts re-fired",
                self.symbol, tf_seconds, rest_ts.isoformat(),
                offset_from_latest, path_label)
        return True

    def apply_rest_insert(self, tf_seconds: int, rest_bar_dict: dict,
                          rest_per_second_bars: list = None) -> str:
        """Insert a WS-MISSED bar (real Polygon REST data) into
        builder.history and recompute indicator state forward.

        Gap healer (2026-06-11) — the sibling of apply_rest_correction
        for bars the WebSocket never delivered at all (correction can
        only fix values of bars already in history; ~14% of RTH 10s
        windows were missing entirely, drifting incremental indicators
        vs the backtest's REST series).

        Returns: 'inserted' (snapshot-path replay) | 'inserted_full'
        (full-replay fallback) | 'exists' | 'rejected_forming' |
        'no_builder' | 'failed'.

        Invoked off-thread by gap_healer; the caller holds the SAME
        per-symbol mutex as rest_verifier corrections. INDICATOR STATE
        ONLY: never calls on_bar_close/on_tick, so no alerts are fired
        for the healed window and PositionStateMachine is untouched.
        NEVER synthesizes bars — callers must pass real REST data.
        """
        builder = self.builders.get(tf_seconds)
        if builder is None:
            return 'no_builder'
        rest_ts = pd.Timestamp(rest_bar_dict['timestamp'])
        if rest_ts.tzinfo is None:
            rest_ts = rest_ts.tz_localize('UTC')
        try:
            _aligned = builder._align_to_period(rest_ts.to_pydatetime())
            rest_ts = pd.Timestamp(_aligned)
            if rest_ts.tzinfo is None:
                rest_ts = rest_ts.tz_localize('UTC')
        except Exception:
            pass

        # Forming-bar guard: only heal bars strictly older than the
        # previous fully-closed period (the live WS path owns the
        # forming bar and the just-closed bar still inside grace).
        now_utc = datetime.now(timezone.utc)
        current_period = pd.Timestamp(
            builder._align_to_period(now_utc))
        if current_period.tzinfo is None:
            current_period = current_period.tz_localize('UTC')
        if rest_ts > current_period - pd.Timedelta(
                seconds=2 * tf_seconds):
            return 'rejected_forming'
        if builder._partial is not None:
            partial_start = pd.Timestamp(builder._partial.bar_start)
            if partial_start.tzinfo is None:
                partial_start = partial_start.tz_localize('UTC')
            if rest_ts >= partial_start:
                return 'rejected_forming'

        def _norm(ts):
            ts = pd.Timestamp(ts)
            return ts.tz_localize('UTC') if ts.tzinfo is None else ts

        def _in_history(target) -> bool:
            for i in range(len(builder.history) - 1, -1, -1):
                if _norm(builder.history.index[i]) == target:
                    return True
            return False

        if len(builder.history) > 0 and _in_history(rest_ts):
            # A late WS rebroadcast or a verifier pass got there first.
            # Value corrections belong to the 4-pass verifier, not here.
            return 'exists'

        # Build the new row with the same column shape _append_to_history
        # produces (extra seeded columns stay NaN, same as a WS append).
        new_row = pd.DataFrame(
            [{
                'open': float(rest_bar_dict['open']),
                'high': float(rest_bar_dict['high']),
                'low': float(rest_bar_dict['low']),
                'close': float(rest_bar_dict['close']),
                'volume': float(rest_bar_dict.get('volume', 0)),
            }],
            index=pd.DatetimeIndex([rest_ts], name='timestamp'))

        # Insert with one re-validation retry: the WS thread can append
        # a new bar between our read and the ref swap (builder.history
        # has no lock of its own — same exposure the correction path
        # accepts). A lost update self-heals either way: a lost healed
        # bar reverts to a detectable gap; a lost WS bar becomes a gap
        # at the next close.
        try:
            for _attempt in (1, 2):
                snapshot_last = (builder.history.index[-1]
                                 if len(builder.history) else None)
                merged = pd.concat([builder.history, new_row]).sort_index()
                if len(merged) > MAX_HISTORY:
                    merged = merged.iloc[-MAX_HISTORY:]
                cur_last = (builder.history.index[-1]
                            if len(builder.history) else None)
                if (cur_last is None or snapshot_last is None
                        or cur_last == snapshot_last):
                    builder.history = merged
                    break
                if _attempt == 2:
                    logger.warning(
                        "gap_heal: history moved twice during insert "
                        "sym=%s tf=%ss bar=%s — leaving for next sweep",
                        self.symbol, tf_seconds, rest_ts.isoformat())
                    return 'failed'
            builder._bar_count += 1
        except Exception as e:
            logger.warning(
                "gap_heal: history insert failed sym=%s tf=%ss bar=%s: %s",
                self.symbol, tf_seconds, rest_ts.isoformat(), e)
            return 'failed'

        # Per-second history (sub-minute TFs): keep the REST per-second
        # list so later verifier passes can splice this bar at sub-bar
        # granularity. Rebuild sorted + re-cap — buckets evict in
        # insertion order, and an out-of-order insert would corrupt the
        # FIFO eviction.
        if (tf_seconds < 60 and rest_per_second_bars
                and hasattr(builder, '_per_second_history')):
            try:
                from collections import OrderedDict as _OD
                psh = builder._per_second_history
                psh[rest_ts] = list(rest_per_second_bars)
                rebuilt = _OD(sorted(psh.items(), key=lambda kv: kv[0]))
                cap = getattr(
                    builder, '_per_second_history_max_buckets', 10)
                while len(rebuilt) > cap:
                    rebuilt.popitem(last=False)
                builder._per_second_history = rebuilt
            except Exception as e:
                logger.warning(
                    "gap_heal: per-second bucket insert failed sym=%s "
                    "tf=%ss bar=%s: %s — bar-level only",
                    self.symbol, tf_seconds, rest_ts.isoformat(), e)

        # Indicator recompute — insert-aware replay, NEVER the plain
        # recompute_from_history (its fast path re-applies only the
        # last bar and would silently omit the inserted bar).
        used_full = False
        for monitor in self.monitors.values():
            if monitor.tf_seconds != tf_seconds:
                continue
            try:
                ok = monitor.indicators.apply_inserted_bar_replay(
                    rest_ts, builder.history)
                if not ok:
                    monitor.indicators.recompute_from_history(
                        builder.history, force_full=True)
                    used_full = True
            except Exception as e:
                logger.warning(
                    "gap_heal: monitor recompute failed sym=%s strat=%s "
                    "tf=%ss: %s",
                    self.symbol, monitor.strat_id, tf_seconds, e)
        # Shadow engine (cross-TF confluence) — recompute AND re-derive
        # records so _mtf_confluence reflects healed data (the gate
        # must not stay frozen on pre-heal state; mirrors the 0609a
        # rebroadcast-cascade fix).
        # Bug Hunt Wave 1 #1: RTH key only — the healed builder.history is
        # the RTH-flavored feed; non-RTH shadows never consume it.
        shadow_self = self._shadow_engines.get((tf_seconds, 'RTH'))
        if shadow_self is not None:
            try:
                ok = shadow_self.indicators.apply_inserted_bar_replay(
                    rest_ts, builder.history)
                if not ok:
                    shadow_self.indicators.recompute_from_history(
                        builder.history, force_full=True)
                    used_full = True
                self._mtf_confluence[(tf_seconds, 'RTH')] = \
                    shadow_self._derive_confluence_records()
            except Exception as e:
                logger.warning(
                    "gap_heal: shadow recompute failed sym=%s tf=%ss: %s",
                    self.symbol, tf_seconds, e)

        # Persist to the live_bars cache with the engine-consumed
        # 'rest_insert' source (NOT 'rest_backfill' — that one is
        # cosmetic and excluded from faithful lens views).
        try:
            from live_bars_writer import write_bar as _live_bars_write
            _live_bars_write(self.symbol, tf_seconds,
                             dict(rest_bar_dict, timestamp=rest_ts),
                             source='rest_insert')
        except Exception:
            pass

        # Hand the healed bar to the normal verifier settle machinery
        # (it's in history now, so passes 2-4 can correct late Polygon
        # adjustments like any WS bar).
        try:
            from rest_verifier import (
                queue_verify_for_bar, is_enabled as _rv_enabled)
            if _rv_enabled():
                queue_verify_for_bar(
                    self.symbol, tf_seconds,
                    rest_ts.to_pydatetime(),
                    ws_close=float(rest_bar_dict['close']),
                    grace_seconds=0.0, pass_num=2)
        except Exception as e:
            logger.warning(
                "gap_heal: verifier handoff failed sym=%s tf=%ss: %s",
                self.symbol, tf_seconds, e)

        logger.info(
            "gap_heal: INSERTED sym=%s tf=%ss bar=%s close=%.4f vol=%.0f "
            "replay=%s monitors=%d shadow=%s — indicators recomputed, "
            "no alerts re-fired",
            self.symbol, tf_seconds, rest_ts.isoformat(),
            float(rest_bar_dict['close']),
            float(rest_bar_dict.get('volume', 0)),
            'full' if used_full else 'snapshot',
            sum(1 for m in self.monitors.values()
                if m.tf_seconds == tf_seconds),
            shadow_self is not None)
        return 'inserted_full' if used_full else 'inserted'

    def _queue_gap_heal_if_gap(self, tf_seconds: int,
                               builder: 'BarBuilder') -> None:
        """After a non-duplicate bar close lands in history: if there is
        a hole between the last two bars, queue the missing boundaries
        for healing. Cheap no-op when the healer is disabled. MUST never
        break bar processing — fully exception-guarded."""
        try:
            import gap_healer
            if not gap_healer.is_enabled():
                return
            if len(builder.history) < 2:
                return
            last_ts = builder.history.index[-1]
            prev_ts = builder.history.index[-2]
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize('UTC')
            if prev_ts.tzinfo is None:
                prev_ts = prev_ts.tz_localize('UTC')
            delta = (last_ts - prev_ts).total_seconds()
            if delta <= tf_seconds:
                return
            missing = []
            cursor = prev_ts + pd.Timedelta(seconds=tf_seconds)
            cap = gap_healer.GAP_HEAL_MAX_WINDOWS_PER_DETECT
            while cursor < last_ts and len(missing) < cap:
                missing.append(cursor.to_pydatetime())
                cursor += pd.Timedelta(seconds=tf_seconds)
            if missing:
                gap_healer.queue_heal(
                    self.symbol, tf_seconds, missing,
                    detected_via='bar_close')
        except Exception as e:
            logger.warning(
                "gap_heal: at-close detection failed sym=%s tf=%ss: %s",
                self.symbol, tf_seconds, e)

    @_prof_fn('s_on_second_bar')
    def on_second_bar(self, bar_dict: dict,
                       alert_callback: Callable = None,
                       config: dict = None,
                       auditor: 'FidelityAuditor' = None):
        """Handle a per-second bar from Polygon A. channel.

        Two responsibilities (in order, latency-critical first):
          1. L-type intra-bar trigger detection: walk this second's high then
             low through every monitor's intra-bar trigger evaluator. Fires
             alerts within ~1s of the trigger event.
          2. M8.5 Phase B+ — sub-minute primary-TF aggregation AND forming-bar
             snapshots for any TF, all from per-second bars:
               - Sub-minute builders (tf_seconds < 60): aggregate via
                 BarBuilder.accept_second_bar(close_on_boundary=True). On a
                 completed bar, run the full monitor pipeline (mirror of
                 on_polygon_bar) and publish the completed bar.
               - 1Min+ builders (tf_seconds >= 60): aggregate via
                 BarBuilder.accept_second_bar(close_on_boundary=False) so the
                 partial is updated for chart visual but never appended to
                 history (the canonical AM bar from on_polygon_bar is what
                 actually closes the bar — preserves backtest parity).
             Forming-bar broadcasts are throttled inside the publisher.
        """
        ts = bar_dict.get('timestamp', '')
        high = bar_dict.get('high', 0)
        low = bar_dict.get('low', 0)

        if not high or not low:
            return

        # Trade_Timestamps_Spec: record the per-second close so alert
        # dispatch can stamp actual_price = most recent tick we've seen.
        close = bar_dict.get('close')
        if close:
            self.latest_close = float(close)

        # ---- (0) WS_AGG BUILDER + DISPATCH (Phase B + C) ----
        # Aggregate Polygon's per-second A.* events into 1Min on our side.
        # Two side effects on each completed-minute boundary:
        #   1. Write to live_bars with source='ws_agg' (cache-side data
        #      collection — feeds dashboards, REST diff validator, future
        #      backtest models cache_locked / cache_corrected).
        #   2. Phase C: dispatch the completed bar through the standard
        #      monitor pipeline if any monitor on this symbol declared
        #      live_model in (ws_agg_locked, ws_agg_with_rest_backfill).
        #      _run_monitor_pipeline_for_completed_bar's gate filters
        #      monitors by source_label='ws_agg' so only opted-in monitors
        #      receive it; ws_with_corrections monitors keep firing on
        #      AM events as before.
        # Both behaviors gated by WS_AGG_SHADOW_ENABLED env flag — set on
        # the Worker container.  Independent of monitor opt-in: the cache
        # write happens regardless of whether any monitor consumes it
        # (lets us collect comparison data even when no strategy uses
        # ws_agg yet).
        if _ws_agg_shadow_enabled():
            try:
                if self._ws_agg_minute is None:
                    self._ws_agg_minute = WsAggMinuteBuilder(self.symbol)
                completed_min = self._ws_agg_minute.accept_second_bar(bar_dict)
                if completed_min is not None:
                    from live_bars_writer import write_bar as _live_bars_write
                    _live_bars_write(self.symbol, 60, completed_min,
                                     source='ws_agg')
                    # Task #11 (2026-05-14): fan-out the just-completed
                    # 1Min ws_agg bar to ≥60s secondary builders. Keeps
                    # `_mtf_confluence[sec_tf]` fresh even when Polygon's
                    # AM channel is intermittent — the longstanding
                    # cause of stale gate state on cross-TF strategies.
                    # Independent feature flag so this can be disabled
                    # without affecting ws_agg cache writes.
                    if _ws_agg_secondary_fanout_enabled():
                        try:
                            self._fanout_to_secondary_builders(
                                completed_min, source_label='ws_agg')
                        except Exception as _e:
                            logger.warning(
                                "ws_agg secondary fan-out failed "
                                "for %s: %s", self.symbol, _e)
                    # Task #12 (2026-05-14): fan-out the just-completed
                    # 1Min ws_agg bar to >60s PRIMARY builders (5Min, 1H,
                    # 1D). Without this, ≥5Min primaries depend solely on
                    # Polygon's AM channel to close — intermittent for
                    # SPY/TSLA. Sid 137 (TSLA 5Min) silent for hours;
                    # cascade-blocks sid 126 (TSLA 10Sec needs 5m secondary).
                    # Independent feature flag from task #11; disable via
                    # WS_AGG_PRIMARY_FANOUT_ENABLED=false on Railway.
                    if _ws_agg_primary_fanout_enabled():
                        try:
                            self._fanout_to_primary_builders_gt_60(
                                completed_min,
                                alert_callback=alert_callback,
                                config=config,
                                auditor=auditor,
                            )
                        except Exception as _e:
                            logger.warning(
                                "ws_agg primary fan-out failed "
                                "for %s: %s", self.symbol, _e)
                    # Phase C dispatch: only invoke monitor pipeline when
                    # at least one 1Min monitor on this symbol opted in.
                    # Avoids the cost of running the pipeline (with a
                    # source_label='ws_agg' that all ws_with_corrections
                    # monitors would skip anyway) when nobody's listening.
                    eligible = any(
                        m.live_model in ('ws_agg_locked', 'ws_agg_with_rest_backfill')
                        and m.tf_seconds == 60
                        for m in self.monitors.values()
                    )
                    if eligible:
                        builder = self.builders.get(60)
                        bar_count = builder._bar_count if builder is not None else 0
                        # Append to the canonical builder history so a
                        # later AM rebroadcast (if AM ever recovers) sees
                        # the bar already in place and updates indicator
                        # state cleanly via accept_bar's duplicate path.
                        if builder is not None:
                            try:
                                builder.accept_bar(completed_min)
                            except Exception as _e:
                                logger.debug(
                                    "ws_agg accept_bar into builder failed "
                                    "for %s: %s", self.symbol, _e)
                        try:
                            self._run_monitor_pipeline_for_completed_bar(
                                tf_seconds=60,
                                bar_dict=completed_min,
                                bar_count=bar_count,
                                alert_callback=alert_callback,
                                config=config,
                                source_label='ws_agg',
                                auditor=auditor,
                            )
                        except Exception as _pe:
                            logger.warning(
                                "ws_agg pipeline dispatch failed for %s: %s",
                                self.symbol, _pe)
            except Exception as e:
                # Builder must NEVER take down live alerting.  Log + continue.
                logger.warning("ws_agg pipeline failed for %s: %s",
                               self.symbol, e)

        # Find which TF's bar_count to use (use the primary 60s builder)
        builder = self.builders.get(60)
        bar_count = builder._bar_count if builder is not None else 0

        # ---- (1) L-TYPE DETECTION (latency-critical, runs first) ----
        for monitor in self.monitors.values():
            if not monitor.indicators._initialized:
                continue
            if not _is_in_session(
                datetime.fromisoformat(ts) if isinstance(ts, str) and ts else datetime.now(timezone.utc),
                monitor.session
            ):
                continue

            # Check high price (catches upward level crosses)
            signals_high = monitor.on_tick(high, ts, bar_count)
            if signals_high and alert_callback:
                for sig in signals_high:
                    self._fire_alert(sig, monitor, config, alert_callback)
                continue  # Don't double-fire on same second

            # Check low price (catches downward level crosses)
            signals_low = monitor.on_tick(low, ts, bar_count)
            if signals_low and alert_callback:
                for sig in signals_low:
                    self._fire_alert(sig, monitor, config, alert_callback)

        # ---- (2) PER-BUILDER AGGREGATION + FORMING-BAR PUBLISH ----
        # Update last_tick_time for session checks downstream
        try:
            self.last_tick_time = datetime.fromisoformat(ts) if isinstance(ts, str) and ts else datetime.now(timezone.utc)
            if self.last_tick_time.tzinfo is None:
                self.last_tick_time = self.last_tick_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            self.last_tick_time = datetime.now(timezone.utc)

        # Primary TF builders are driven by per-second bars below.
        # Secondary TF builders split by granularity:
        #   - >= 60s: fed from the canonical 1Min AM bars in on_polygon_bar
        #     (commit ca57cac).
        #   - < 60s: fed here from per-second bars (commit follows ca57cac
        #     and heals sub-minute cross-TF gates like sid 125's
        #     `5s-SWING_123_TEST-NEUTRAL` and sid 102's `15s-...`).
        primary_tfs = {m.tf_seconds for m in self.monitors.values()}
        secondary_subminute_tfs = {
            k[0] for k in self._shadow_engines if k[0] < 60
        }

        # Sub-minute secondary aggregation: feed each per-second bar into
        # any sub-minute shadow builder; on close, refresh _mtf_confluence.
        for sec_tf in secondary_subminute_tfs:
            if sec_tf in primary_tfs:
                continue  # primary path handles it below
            sec_builder = self.builders.get(sec_tf)
            if sec_builder is None:
                continue
            try:
                completed = sec_builder.accept_second_bar(
                    bar_dict, close_on_boundary=True)
            except Exception as e:
                logger.warning(
                    "sub-minute secondary aggregation failed (%ss): %s",
                    sec_tf, e)
                continue
            if completed is None:
                continue
            sec_was_duplicate = sec_builder.last_was_duplicate
            sec_was_correction = sec_builder.last_was_correction
            if not sec_was_duplicate:
                # Gap healer (2026-06-11): sub-minute secondary builders
                # gap-heal independently too.
                self._queue_gap_heal_if_gap(sec_tf, sec_builder)
            # M8.7: record sub-minute secondary-TF bar to live_bars.
            try:
                from live_bars_writer import write_bar as _live_bars_write
                _live_bars_write(self.symbol, sec_tf, completed, source='ws')
            except Exception:
                pass
            shadow_items = self._shadow_items_for_tf(sec_tf)
            if not shadow_items:
                continue

            if sec_was_duplicate:
                # M8.7 (2026-05-02): Polygon per-second rebroadcast for
                # the most recent sub-minute period. Recompute shadow
                # indicator state from corrected history; skip on_bar_close
                # to avoid emitting confluence records that already fired.
                # Skip the recompute on a pure re-delivery (no value change)
                # — that is the lag-spiral hot path (2026-05-19).
                # Bug Hunt Wave 1 #1: sub-minute builders are fed RAW
                # per-second bars (not RTH-built fan-out), so every
                # session-shadow of this TF consumed the same live stream
                # — recompute all of them from the shared history.
                if sec_was_correction:
                    for _sh_sess, shadow in shadow_items:
                        try:
                            shadow.indicators.recompute_from_history(sec_builder.history)
                            logger.info(
                                "Rebroadcast cascade: shadow %s/%ss recomputed",
                                self.symbol, sec_tf)
                        except Exception as e:
                            logger.warning(
                                "shadow recompute_from_history failed (%ss): %s",
                                sec_tf, e)
                continue

            # Note: do NOT gate on shadow.indicators._initialized here.
            # shadow.on_bar_close → update_bar handles cold-start via the
            # incremental class's `_first` flag, and skipping when not
            # initialized would prevent the shadow from ever initializing
            # via live data alone (chicken-and-egg). worker.py.warmup()
            # is an optimization, not a hard requirement. Found while
            # investigating Q4 data fidelity (2026-04-27).
            # Bug Hunt Wave 1 #1: sub-minute closes come from RAW
            # per-second data (valid for any session), so — unlike the
            # >=60s fan-out path — EVERY session-shadow eats the bar
            # directly; no per-close session reloads at this cadence.
            for _sh_sess, shadow in shadow_items:
                try:
                    self._mtf_confluence[(sec_tf, _sh_sess)] = \
                        shadow.on_bar_close(completed)
                    logger.info(
                        "shadow_close %s/%ss close=%.2f records=%s",
                        self.symbol, sec_tf, float(completed['close']),
                        self._mtf_confluence[(sec_tf, _sh_sess)])
                except Exception as e:
                    logger.warning(
                        "shadow on_bar_close failed (%ss): %s", sec_tf, e)

        for tf_seconds, b in self.builders.items():
            if tf_seconds not in primary_tfs:
                continue

            if tf_seconds < 60:
                # Sub-minute: per-second is the canonical source. Aggregate.
                completed = b.accept_second_bar(bar_dict, close_on_boundary=True)
                # LEF Phase 2b (2026-05-20): for any ws_agg_reconciled or
                # ws_rest_spliced (2026-05-28) monitor on this (symbol, TF)
                # whose strategy-grace deadline just elapsed on the *current
                # partial bucket*, fire on the forming bar via
                # fire_on_partial_bucket. Cheap no-op when no monitor uses
                # either model.
                self._check_strategy_grace_fires(
                    tf_seconds, b, alert_callback, config)
                if completed is not None:
                    primary_was_duplicate = b.last_was_duplicate
                    primary_was_correction = b.last_was_correction
                    if not primary_was_duplicate:
                        # Gap healer (2026-06-11): the sub-minute primary
                        # (10s cohort) is where WS gaps hurt most.
                        self._queue_gap_heal_if_gap(tf_seconds, b)
                    # M8.7: record sub-minute primary-TF bar to live_bars.
                    try:
                        from live_bars_writer import write_bar as _live_bars_write
                        _live_bars_write(self.symbol, tf_seconds, completed,
                                         source='ws')
                    except Exception:
                        pass

                    if primary_was_duplicate:
                        # M8.7 (2026-05-02): Polygon per-second rebroadcast
                        # for the most recent sub-minute primary bar. Recompute
                        # indicator state silently; skip alert pipeline.
                        # Pure re-deliveries (identical OHLCV — rampant when
                        # the worker drains a per-second backlog) skip the
                        # recompute: this is THE lag-spiral fix (2026-05-19).
                        if primary_was_correction:
                            for monitor in self.monitors.values():
                                if monitor.tf_seconds != tf_seconds:
                                    continue
                                try:
                                    monitor.indicators.recompute_from_history(b.history)
                                except Exception as e:
                                    logger.warning(
                                        "Rebroadcast recompute failed for strat %s "
                                        "(tf=%ss): %s",
                                        monitor.strat_id, tf_seconds, e)
                            # Bug Hunt Wave 1 #1: sub-minute primary — raw
                            # per-second history is session-agnostic, so
                            # recompute every session-shadow of this TF.
                            for _sh_sess, shadow_self in \
                                    self._shadow_items_for_tf(tf_seconds):
                                try:
                                    shadow_self.indicators.recompute_from_history(b.history)
                                except Exception as e:
                                    logger.warning(
                                        "Rebroadcast recompute failed for shadow "
                                        "%ss: %s", tf_seconds, e)
                            logger.info(
                                "Rebroadcast applied: %s tf=%ss bar=%s — "
                                "indicators recomputed (sub-minute), no alerts",
                                self.symbol, tf_seconds,
                                completed.get('timestamp'))
                    else:
                        # Run the full monitor pipeline on the closed sub-minute bar
                        self._run_monitor_pipeline_for_completed_bar(
                            tf_seconds=tf_seconds,
                            bar_dict=completed,
                            bar_count=b._bar_count,
                            alert_callback=alert_callback,
                            config=config,
                            auditor=auditor,
                            source_label='subm',
                        )
                # Publish forming snapshot of the (new) partial bar, with
                # tentative indicators + interpreter states so the frontend
                # heatmap / oscillator / overlay panes stay in sync with
                # the price candle.
                if b._partial is not None:
                    partial_dict = b._partial.to_dict()
                    extras = self._gather_tentative_state(
                        tf_seconds, partial_dict)
                    self._publish_completed_bar(
                        tf_seconds, partial_dict,
                        is_forming=True, extras=extras)
            else:
                # 1Min+: AM channel is canonical. Use per-second only to
                # update the partial for chart visual. NEVER appends to
                # history. NEVER runs monitors here. skip_volume: the
                # fan-out minutes are the canonical volume source —
                # per-second contributions double-counted it (2026-06-12
                # volume-integrity fix).
                b.accept_second_bar(bar_dict, close_on_boundary=False,
                                    skip_volume=True)
                if b._partial is not None:
                    partial_dict = b._partial.to_dict()
                    extras = self._gather_tentative_state(
                        tf_seconds, partial_dict)
                    self._publish_completed_bar(
                        tf_seconds, partial_dict,
                        is_forming=True, extras=extras)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION CHECK
# ═══════════════════════════════════════════════════════════════════════════

_SESSION_HOURS = {
    "RTH": (9, 30, 16, 0),
    "Pre-Market": (4, 0, 9, 30),
    "After Hours": (16, 0, 20, 0),
    "Extended Hours": (4, 0, 20, 0),
}
try:
    import pytz as _pytz
    _ET_TZ = _pytz.timezone("America/New_York")
except ImportError:
    _ET_TZ = timezone(timedelta(hours=-5))  # fallback EST


def _is_in_session(timestamp: datetime, session: str) -> bool:
    """Check if a UTC timestamp falls within the given trading session (ET).
    Returns True unconditionally for '24/7' session (crypto).
    """
    if session == '24/7':
        return True
    from datetime import time as _time
    et_time = timestamp.astimezone(_ET_TZ).time()
    sh, sm, eh, em = _SESSION_HOURS.get(session, (9, 30, 16, 0))
    return _time(sh, sm) <= et_time < _time(eh, em)


# ═══════════════════════════════════════════════════════════════════════════
# RALPH ENGINE — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class RalphEngine:
    """Main alert engine. Connects to data, routes ticks, dispatches alerts."""

    def __init__(self):
        self.hubs: Dict[str, SymbolHub] = {}
        self.strategies: List[dict] = []
        self.monitors: Dict[int, StrategyMonitor] = {}
        self.dispatcher = AlertDispatcher()
        self.auditor = FidelityAuditor()
        self._running = False
        self._config: dict = {}
        self._stream_ref = None
        self._crypto_stream_ref = None
        self._event_loop = None  # Set when async loop starts
        self._ws_confirmed = False  # True only after first trade received
        self._start_time: Optional[str] = None
        self._last_reconcile = 0.0
        self._last_pickle_write = 0.0
        self._last_strategy_refresh = 0.0
        self._subscribed_symbols: List[str] = []
        # Tier 3 §8.3 (always-start-flat, 2026-05-20): collected from
        # monitor._tier3_inherited_carryover during start(). The worker
        # registers `_carryover_persister` to a callable that writes
        # each entry to the strategy's position_carryovers field. The
        # engine calls the persister synchronously between monitor
        # construction and the event-loop start (so the persistence
        # is durable BEFORE any live alerts can fire). When no
        # persister is registered (CLI dry-run, tests), the list is
        # left available for inspection but not auto-written.
        self._pending_carryovers: List[dict] = []
        self._carryover_persister = None  # type: Optional[callable]
        # M8.5: Live-bar publisher — set by worker before start().
        # None = live chart disabled (no broadcasts). Harmless default.
        self._publisher: Optional['LiveBarPublisher'] = None
        # Phase 3 (Alert_Recovery_Plan_2026-04-17): ThreadPoolExecutor for
        # alert-dispatch DB + webhook I/O. Set by the worker before start().
        # When None, alert dispatch runs synchronously on the event loop
        # (legacy behavior, fine for CLI / tests — blocks ~200-500ms per alert).
        self._alert_executor = None
        # Phase H.1 (2026-05-15): shadow trade-channel bar builder. Subscribes
        # to Polygon T.* channels for SPY+TSLA and writes the resulting OHLCV
        # to `live_bars_trades` at three wait-time variants in parallel.
        # Doesn't touch the production alert path. Disabled by setting
        # POLYGON_TRADE_SHADOW_ENABLED=false on the worker.
        from trade_bar_builder import TradeBarShadowManager
        self._trade_shadow = TradeBarShadowManager()
        # Grace-window shadow (2026-05-18): builds 10Sec bars at 2/3/4s
        # grace from the A per-second stream → live_bars_trades, so Bars
        # Comparison can A/B which grace window best matches REST.
        from grace_shadow import GraceShadowManager
        self._grace_shadow = GraceShadowManager()
        # Phase H+ diag (2026-05-18): A-channel ingestion-lag histogram.
        # Sizes the sub-minute builder grace window — how late per-second
        # bars reach us drives how long a 10Sec bucket must stay open.
        self._a_lag_n = 0
        self._a_lag_sum = 0.0
        self._a_lag_min = float('inf')
        self._a_lag_max = 0.0
        self._a_lag_hist = [0, 0, 0, 0, 0, 0, 0]
        self._a_lag_last_log = 0.0

    def _record_a_lag(self, bar_start_ms) -> None:
        """Accumulate A-channel ingestion lag (wall-clock minus the
        per-second bar's start ms) and log a cumulative histogram every
        30s. Diagnostic only — no behavior change."""
        try:
            start_s = float(bar_start_ms) / 1000.0
        except (TypeError, ValueError):
            return
        if start_s <= 0:
            return
        now = time.time()
        lag = now - start_s
        self._a_lag_n += 1
        self._a_lag_sum += lag
        if lag < self._a_lag_min:
            self._a_lag_min = lag
        if lag > self._a_lag_max:
            self._a_lag_max = lag
        edges = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
        placed = False
        for i, e in enumerate(edges):
            if lag <= e:
                self._a_lag_hist[i] += 1
                placed = True
                break
        if not placed:
            self._a_lag_hist[6] += 1
        if now - self._a_lag_last_log >= 30.0:
            self._a_lag_last_log = now
            n = self._a_lag_n
            avg = self._a_lag_sum / n if n else 0.0
            lo = self._a_lag_min if self._a_lag_min != float('inf') else 0.0
            run = 0
            cum = []
            for i, e in enumerate(edges):
                run += self._a_lag_hist[i]
                cum.append('<=%ss:%.1f%%' % (
                    e, 100.0 * run / n if n else 0.0))
            logger.info(
                "A-channel lag (30s window): n=%d avg=%.2fs min=%.2fs "
                "max=%.2fs | %s >10s:%d",
                n, avg, lo, self._a_lag_max, ' '.join(cum),
                self._a_lag_hist[6])
            self._a_lag_n = 0
            self._a_lag_sum = 0.0
            self._a_lag_min = float('inf')
            self._a_lag_max = 0.0
            self._a_lag_hist = [0, 0, 0, 0, 0, 0, 0]
            _prof_dump_and_reset()

    def rest_correction_callback(self, symbol: str, tf_seconds: int,
                                  rest_bar_dict: dict,
                                  rest_per_second_bars: list = None
                                  ) -> bool:
        # ws_rest_spliced (2026-05-28): module-level callback registered
        # with rest_verifier at worker startup. Routes the correction to
        # the SymbolHub that owns BarBuilders for this symbol. Phase C
        # passes rest_per_second_bars when available so SymbolHub.
        # apply_rest_correction can splice at sub-bar granularity.
        # Returns False (silent no-op) when the symbol has no live hub.
        hub = self.hubs.get(symbol)
        if hub is None:
            return False
        return hub.apply_rest_correction(
            tf_seconds, rest_bar_dict,
            rest_per_second_bars=rest_per_second_bars)

    def rest_insert_callback(self, symbol: str, tf_seconds: int,
                             rest_bar_dict: dict,
                             rest_per_second_bars: list = None) -> str:
        # Gap healer (2026-06-11): module-level callback registered with
        # gap_healer at worker startup. Routes the insert of a WS-missed
        # bar to the SymbolHub that owns BarBuilders for this symbol.
        hub = self.hubs.get(symbol)
        if hub is None:
            return 'no_hub'
        return hub.apply_rest_insert(
            tf_seconds, rest_bar_dict,
            rest_per_second_bars=rest_per_second_bars)

    def scan_gap_candidates(self, lookback_seconds: int = None,
                            max_windows_per_builder: int = None) -> list:
        """Gap healer sweeper provider (2026-06-11). Read-only walk of
        every hub's builders; returns [(symbol, tf_seconds,
        [missing_bar_start_dt])]. Two gap classes per builder:
          (a) interior gaps — consecutive history index deltas > tf
              within the lookback window;
          (b) tail gaps — no forming partial and the last close is more
              than 2*tf+grace stale (WS silent / force_close path).
        Bounded by lookback + per-builder cap. Same no-lock read posture
        as the verifier; a torn read at worst yields a candidate that
        apply_rest_insert later rejects ('exists'/'rejected_forming')."""
        import gap_healer as _gh
        lookback = lookback_seconds or _gh.GAP_HEAL_LOOKBACK_SEC
        cap = max_windows_per_builder or _gh.GAP_HEAL_MAX_WINDOWS_PER_DETECT
        now = datetime.now(timezone.utc)
        floor_ts = pd.Timestamp(now - timedelta(seconds=lookback))
        out: list = []
        for symbol, hub in self.hubs.items():
            for tf_seconds, builder in hub.builders.items():
                try:
                    hist = builder.history
                    if hist is None or len(hist) < 1:
                        continue
                    missing: list = []
                    # (a) interior gaps within the lookback window.
                    if len(hist) >= 2:
                        idx = hist.index
                        for i in range(len(idx) - 1, 0, -1):
                            cur = idx[i]
                            prev = idx[i - 1]
                            if cur.tzinfo is None:
                                cur = cur.tz_localize('UTC')
                            if prev.tzinfo is None:
                                prev = prev.tz_localize('UTC')
                            if cur < floor_ts:
                                break
                            delta = (cur - prev).total_seconds()
                            if delta <= tf_seconds:
                                continue
                            cursor = prev + pd.Timedelta(seconds=tf_seconds)
                            while (cursor < cur
                                   and len(missing) < cap):
                                if cursor >= floor_ts:
                                    missing.append(cursor.to_pydatetime())
                                cursor += pd.Timedelta(seconds=tf_seconds)
                            if len(missing) >= cap:
                                break
                    # (b) tail gap: builder idle past 2*tf + grace.
                    if builder._partial is None and len(missing) < cap:
                        last_ts = hist.index[-1]
                        if last_ts.tzinfo is None:
                            last_ts = last_ts.tz_localize('UTC')
                        grace = getattr(builder, 'grace_sec_override',
                                        None) or 5
                        stale_after = 2 * tf_seconds + grace
                        if (pd.Timestamp(now) - last_ts).total_seconds() \
                                > stale_after:
                            cursor = last_ts + pd.Timedelta(
                                seconds=tf_seconds)
                            edge = pd.Timestamp(now) - pd.Timedelta(
                                seconds=2 * tf_seconds)
                            while (cursor < edge
                                   and len(missing) < cap):
                                if cursor >= floor_ts:
                                    missing.append(cursor.to_pydatetime())
                                cursor += pd.Timedelta(seconds=tf_seconds)
                    if missing:
                        out.append((symbol, tf_seconds, sorted(missing)))
                except Exception as e:
                    logger.warning(
                        "gap_heal: scan failed sym=%s tf=%ss: %s",
                        symbol, tf_seconds, e)
        return out

    def start(self, strategies: list, config: dict):
        """Start the engine (blocking — runs the async event loop)."""
        self._config = config
        self.strategies = strategies
        self._running = True

        if not strategies:
            logger.info("No strategies to monitor")
            self._running = False
            return

        # Load saved position state
        saved_positions = load_engine_state()

        # Load enabled general packs for GEN- confluence evaluation.
        # CRITICAL (2026-06-22): the worker thread has NO thread-local user
        # context at startup, so the context-dependent _load_enabled_general_packs()
        # silently loaded 0 packs → every monitor was built with an empty
        # _general_packs → ALL general-pack gating was blocked live (confirmed
        # via GATE_BLOCK_DIAG: gen_packs=0). The hot-reload path already loads
        # via the explicit admin path; mirror it here using the user_id stamped
        # on the strategies. Falls back to the context path for CLI/dry-run.
        _uid = next((s.get('user_id') for s in strategies if s.get('user_id')), None)
        self._general_packs = []
        if _uid:
            try:
                from db import load_general_packs_admin
                from general_packs import _parse_pack_list, get_enabled_general_packs
                self._general_packs = get_enabled_general_packs(
                    _parse_pack_list(load_general_packs_admin(_uid)))
            except Exception as _gp_e:
                logger.warning("startup general-pack admin load failed (%s); "
                               "falling back to context load", _gp_e)
                self._general_packs = _load_enabled_general_packs()
        else:
            self._general_packs = _load_enabled_general_packs()
        logger.info("startup: loaded %d enabled general packs (uid=%s)",
                    len(self._general_packs), str(_uid)[:8] if _uid else None)

        # Group strategies by symbol
        by_symbol: Dict[str, List[dict]] = {}
        for strat in strategies:
            sym = strat.get('symbol', 'SPY')
            by_symbol.setdefault(sym, []).append(strat)

        # Create hubs and monitors
        for sym, strats in by_symbol.items():
            hub = SymbolHub(sym, publisher=self._publisher)

            for strat in strats:
                strat_id = strat['id']
                pos_state = saved_positions.get(strat_id)

                monitor = StrategyMonitor(
                    strat, pos_state, general_packs=self._general_packs)
                hub.add_monitor(monitor)
                self.monitors[strat_id] = monitor

                # Tier 3 §8.3: drain inherited-position carryover for the
                # worker to persist after engine.start() completes.
                if monitor._tier3_inherited_carryover is not None:
                    self._pending_carryovers.append(
                        monitor._tier3_inherited_carryover)

            self.hubs[sym] = hub

        # Tier 3 §8.3: flush pending carryovers to the worker's persister
        # before any monitor goes live. If the persister raises, log it
        # but proceed — the engine itself is FLAT either way; loss of
        # the carryover write is a UI-surface degradation, not a
        # correctness failure.
        if self._pending_carryovers and self._carryover_persister is not None:
            try:
                self._carryover_persister(list(self._pending_carryovers))
                logger.info(
                    "Tier 3: persisted %d position_carryover entries",
                    len(self._pending_carryovers))
            except Exception as e:
                logger.error(
                    "Tier 3: carryover persister raised %s — UI banner "
                    "will not surface but engine is FLAT correctly",
                    e, exc_info=True)
            self._pending_carryovers.clear()

        # Create shadow indicator engines for cross-TF confluence
        for sym, hub in self.hubs.items():
            hub.finalize_shadow_engines()

        # Load warmup data and initialize indicators
        self._warmup_all()

        # Rebase bar-count fields for restored positions.  After restart
        # the bar_count counter starts fresh from warmup length, so the
        # old session's values are meaningless.
        for sym, hub in self.hubs.items():
            for tf_seconds, builder in hub.builders.items():
                cur_bc = builder._bar_count
                for monitor in hub.monitors.values():
                    if monitor.tf_seconds != tf_seconds:
                        continue

                    # Reset last_exit_bar_count so the 1-bar cooldown
                    # doesn't permanently block entries after restart.
                    old_lebc = monitor.position.state.last_exit_bar_count
                    if old_lebc > cur_bc:
                        monitor.position.state.last_exit_bar_count = 0
                        logger.info(
                            "Reset last_exit_bar_count for %s: %d → 0",
                            monitor.strat_name, old_lebc)

                    if monitor.position.state.status == 'IN_POSITION':
                        old_ebc = monitor.position.state.entry_bar_count
                        if old_ebc > cur_bc:
                            monitor.position.state.entry_bar_count = cur_bc
                            logger.info(
                                "Rebased entry_bar_count for %s: %d → %d",
                                monitor.strat_name, old_ebc, cur_bc)

        # For strategies with no saved position state, attempt to
        # determine current position by running generate_trades() on
        # the warmup data. This handles first-ever startup where
        # engine_state.json doesn't exist yet.
        if saved_positions:
            strats_needing_sync = [
                s for s in strategies if s['id'] not in saved_positions]
        else:
            strats_needing_sync = strategies
        if strats_needing_sync:
            self._sync_initial_positions(strats_needing_sync)

        # Write PID and status
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._write_status(running=True)

        logger.info("Ralph engine starting: %d strategies, %d symbols",
                     len(strategies), len(self.hubs))

        # Run the async event loop — choose data provider
        from dotenv import load_dotenv as _ld
        _ld(_SCRIPT_DIR / '.env', override=True)
        _data_provider = os.getenv("DATA_PROVIDER", "auto").lower()
        _polygon_key = os.getenv("POLYGON_API_KEY", "")
        _use_polygon = _data_provider in ("polygon", "auto") and bool(
            _polygon_key and not _polygon_key.startswith("YOUR_"))

        try:
            if _use_polygon:
                logger.info("Using Polygon.io data provider")
                asyncio.run(self._stream_data_polygon(_polygon_key))
            else:
                logger.info("Using Alpaca data provider")
                asyncio.run(self._stream_data())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        except Exception as e:
            logger.error("Engine error: %s", e)
        finally:
            self._running = False
            self._save_all_positions()
            self._write_status(running=False)
            logger.info("Ralph engine stopped")

    def stop(self):
        """Signal the engine to stop."""
        self._running = False
        # Close all active WebSocket streams
        for stream in (self._stream_ref, self._crypto_stream_ref):
            if stream:
                try:
                    stream._should_run = False
                except Exception:
                    pass
                try:
                    loop = self._event_loop
                    if loop and loop.is_running():
                        loop.call_soon_threadsafe(
                            loop.create_task, stream.close())
                except Exception:
                    pass
        # M8.7: drain in-flight live_bars writes so we don't lose the
        # last few seconds of data on engine restart.
        try:
            from live_bars_writer import shutdown as _live_bars_shutdown
            _live_bars_shutdown(wait=True)
        except Exception:
            pass
        # M8.7 M6 (2026-05-02): drain in-flight bar_engine_states writes too.
        try:
            from bar_engine_state_writer import shutdown as _bes_shutdown
            _bes_shutdown(wait=True)
        except Exception:
            pass
        # Phase H.1 (2026-05-15): drain shadow trade-bar writes so the last
        # few bars of the session land in live_bars_trades cleanly.
        try:
            from trade_bar_builder import shutdown as _trade_bar_shutdown
            _trade_bar_shutdown(wait=True)
        except Exception:
            pass

    def _warmup_all(self):
        """Load historical data and initialize all monitors."""
        # Bug Hunt Wave 1 #1: dedup cache keyed by (sym, tf, session).
        seen_tf: Dict[Tuple[str, int, str], pd.DataFrame] = {}

        for sym, hub in self.hubs.items():
            for tf_seconds, builder in hub.builders.items():
                tf_str = SECONDS_TO_TIMEFRAME.get(tf_seconds, '1Min')
                # Determine session from first monitor using this TF as
                # its PRIMARY. (Bug Hunt Wave 1 #1: for pure-secondary
                # TFs this never matches → 'RTH'; that was the broken
                # session resolution for gate shadows. The builder + real
                # monitors legitimately keep this resolution — the
                # session-aware shadow warmup is the pass below.)
                session = 'RTH'
                for m in hub.monitors.values():
                    if m.tf_seconds == tf_seconds:
                        session = m.session
                        break
                cache_key = (sym, tf_seconds, session)
                if cache_key in seen_tf:
                    df = seen_tf[cache_key]
                else:
                    try:
                        # Bug 5 (2026-06-18): shared warmup loader — sub-minute
                        # native, >=60s resampled-from-1min over a TF-scaled
                        # window so coarse gates converge (was flat days=7).
                        df = _load_warmup_df(sym, tf_seconds, session)
                    except Exception as e:
                        logger.error("Warmup failed for %s/%s: %s",
                                     sym, tf_str, e)
                        df = pd.DataFrame()
                    seen_tf[cache_key] = df

                # Seed bar builder. seed_history now splits any forming
                # bar into builder._partial, leaving builder.history with
                # only CLOSED bars. Use builder.history (not the raw df)
                # for monitor.warmup and shadow.warmup so indicators don't
                # get warmed up with partial-period data — otherwise
                # shadow.on_bar_close at the next boundary would
                # double-count the forming bar's contribution.
                hub.seed_history(tf_seconds, df)
                warmup_df = builder.history

                # Warmup each monitor's indicators
                for monitor in hub.monitors.values():
                    if monitor.tf_seconds == tf_seconds and len(warmup_df) > 0:
                        monitor.warmup(warmup_df)
                        logger.info("Warmup %s: strat=%s tf=%ss bars=%d initialized=%s",
                                    sym, monitor.strat_id, tf_seconds,
                                    len(warmup_df),
                                    monitor.indicators._initialized)
                    elif monitor.tf_seconds == tf_seconds:
                        logger.warning("Warmup SKIPPED %s: strat=%s tf=%ss — no historical data",
                                       sym, monitor.strat_id, tf_seconds)

            # Warmup shadow engines — session-aware pass (Bug Hunt Wave 1
            # #1). Iterate the shadow ENGINE KEYS instead of resolving the
            # session from primary monitors (which can never match a
            # secondary TF → hard 'RTH' — the root cause of the 338
            # session mismatch). RTH shadows keep warming from the seeded
            # builder history (byte-identical to the legacy path); non-RTH
            # shadows (flag ON only) load their own session's bars.
            for (tf_seconds, sh_session), shadow in \
                    hub._shadow_engines.items():
                builder = hub.builders.get(tf_seconds)
                if sh_session == 'RTH':
                    warmup_df = (builder.history if builder is not None
                                 else pd.DataFrame())
                else:
                    cache_key = (sym, tf_seconds, sh_session)
                    if cache_key in seen_tf:
                        df = seen_tf[cache_key]
                    else:
                        try:
                            df = _load_warmup_df(sym, tf_seconds, sh_session)
                        except Exception as e:
                            logger.error(
                                "Shadow warmup load failed for %s/%ss "
                                "session=%s: %s",
                                sym, tf_seconds, sh_session, e)
                            df = pd.DataFrame()
                        seen_tf[cache_key] = df
                    # The builder holds the RTH-flavored feed, so the
                    # forming-bar split it provides isn't available here —
                    # trim the forming bar explicitly.
                    warmup_df = _closed_bars_only(df, tf_seconds)
                if len(warmup_df) > 0:
                    shadow.warmup(warmup_df)
                    # Publish the warmed coarse-TF state into the cross-TF gate
                    # dict so 1H/4H/1D gates are evaluable immediately (2026-06-26
                    # fix — they were absent until a live close that never comes
                    # intraday → coarse-gated strategies silent live).
                    hub.seed_mtf_from_shadow(tf_seconds, sh_session)
                    logger.info("Shadow warmup %s: tf=%ss session=%s bars=%d "
                                "initialized=%s",
                                sym, tf_seconds, sh_session, len(warmup_df),
                                shadow.indicators._initialized)

        logger.info("Warmup complete for %d symbols", len(self.hubs))

    def _sync_initial_positions(self, strategies: List[dict]):
        """Determine initial position state for strategies with no saved state.

        Runs generate_trades() on the warmup bars for each strategy to check
        if the last trade is still open. If so, sets the PositionStateMachine
        to IN_POSITION with the appropriate entry details.
        """
        try:
            from triggers import generate_trades
            from indicators import run_all_indicators
            from interpreters import detect_all_triggers
        except ImportError as e:
            logger.warning("Cannot sync initial positions: %s", e)
            return

        for strat in strategies:
            sid = strat['id']
            monitor = self.monitors.get(sid)
            if not monitor:
                continue

            sym = strat.get('symbol', 'SPY')
            hub = self.hubs.get(sym)
            if not hub:
                continue

            builder = hub.builders.get(monitor.tf_seconds)
            if not builder or len(builder.history) < 10:
                continue

            try:
                # Run the batch pipeline on warmup bars
                df = builder.history.copy()
                df = run_all_indicators(df)
                df = detect_all_triggers(df)

                # Use resolved trigger IDs from the monitor
                entry_trigger = monitor.position.entry_trigger
                exit_triggers = list(monitor.position.exit_triggers)
                trades = generate_trades(
                    df,
                    direction=strat.get('direction', 'LONG'),
                    entry_trigger=entry_trigger,
                    exit_triggers=exit_triggers or None,
                    risk_per_trade=strat.get('risk_per_trade', 100.0),
                    stop_atr_mult=strat.get('stop_atr_mult', 1.5),
                    stop_config=strat.get('stop_config'),
                    target_config=strat.get('target_config'),
                    bar_count_exit=strat.get('bar_count_exit'),
                )

                if isinstance(trades, pd.DataFrame) and len(trades) > 0:
                    last_trade = trades.iloc[-1]
                    # An open trade has NaN exit_price
                    has_exit = (
                        'exit_price' in last_trade.index
                        and pd.notna(last_trade.get('exit_price')))
                    if not has_exit:
                        entry_price = float(
                            last_trade.get('entry_price', 0) or 0)
                        stop_price = last_trade.get('stop_price')
                        if pd.notna(stop_price):
                            stop_price = float(stop_price)
                        else:
                            stop_price = 0.0
                        entry_bar = int(
                            last_trade.get('entry_bar_index', 0) or 0)

                        monitor.position.state.status = 'IN_POSITION'
                        monitor.position.state.entry_price = entry_price
                        monitor.position.state.stop_price = stop_price
                        monitor.position.state.entry_bar_count = entry_bar
                        logger.info(
                            "  Synced S%d (%s): IN_POSITION @ %.2f "
                            "stop=%.4f",
                            sid, strat.get('name'), entry_price,
                            stop_price or 0)
                    else:
                        logger.debug("  S%d (%s): FLAT (last trade closed)",
                                    sid, strat.get('name'))
                else:
                    logger.debug("  S%d (%s): FLAT (no trades)",
                                sid, strat.get('name'))
            except Exception as e:
                logger.warning("Position sync failed for S%d: %s", sid, e)

    def _on_alert(self, signal: dict, strategy: dict, config: dict):
        """Callback when a strategy monitor fires a signal.

        Called synchronously from inside the async WS consumer coroutine
        (via hub.on_tick / hub.on_second_bar / hub.on_polygon_bar). When
        an _alert_executor is wired (by the Railway worker), the actual
        DB + webhook work is scheduled on a thread so the event loop is
        not blocked on Supabase HTTP round-trips. This was the main cause
        of the post-M8.5 alert-latency regression — see Phase 3 of
        docs/Alert_Recovery_Plan_2026-04-17.md.
        """
        if self._alert_executor is None:
            self._dispatch_alert_sync(signal, strategy, config)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not in an async context — fall back to sync.
            self._dispatch_alert_sync(signal, strategy, config)
            return
        future = loop.run_in_executor(
            self._alert_executor,
            self._dispatch_alert_sync, signal, strategy, config)
        future.add_done_callback(self._log_alert_task_exception)

    def _dispatch_alert_sync(self, signal: dict, strategy: dict,
                             config: dict) -> None:
        """Actual alert-dispatch body — runs on executor thread when wired."""
        try:
            alert = self.dispatcher.dispatch(signal, strategy, config or self._config)
            if alert:
                self._save_all_positions()
        except Exception as e:
            logger.error("Alert dispatch failed for %s (%s): %s",
                         strategy.get('name', '?'), signal.get('type', '?'), e)

    @staticmethod
    def _log_alert_task_exception(future) -> None:
        """Log any exception from a fire-and-forget alert executor task."""
        exc = future.exception()
        if exc is not None:
            logger.error("Alert executor task raised: %s", exc)

    async def _stream_data_polygon(self, api_key: str):
        """Subscribe to Polygon.io pre-aggregated bar WebSocket + periodic tasks.

        Polygon's AM.{ticker} channel delivers completed minute bars.
        No tick aggregation needed — bars arrive pre-built with OHLCV.
        """
        self._event_loop = asyncio.get_running_loop()

        import websockets

        # Launch independent periodic task loop
        periodic_task = asyncio.ensure_future(self._periodic_tasks_loop())

        backoff = 5
        max_backoff = 120

        async def _interruptible_sleep(seconds):
            for _ in range(int(seconds)):
                if not self._running:
                    return
                await asyncio.sleep(1)

        try:
            while self._running:
                all_symbols = list(self.hubs.keys())
                stock_symbols = [s for s in all_symbols if '/' not in s]
                crypto_symbols = [s for s in all_symbols if '/' in s]
                self._subscribed_symbols = all_symbols

                # Build subscription channels
                # AM.{ticker} = per-minute aggregates (stocks) — NO LONGER
                #   subscribed (see below).
                # A.{ticker}  = per-second aggregates — subscribed when ANY
                #   monitor on this symbol needs per-second data:
                #     * L-type intrabar trigger (level-cross detection), or
                #     * sub-minute primary TF (5s/10s/15s/30s aggregation), or
                #     * Phase C live_model in (ws_agg_locked,
                #       ws_agg_with_rest_backfill) — the engine consumes
                #       client-side 1Min bars built from per-second A bars.
                #   Avoids unconditional A subscription on every symbol —
                #   the unconditional version flooded the event loop with
                #   per-second forming-bar work and was the main contributor
                #   to the post-M8.5 alert-latency regression. See
                #   docs/Alert_Recovery_Plan_2026-04-17.md Phase 2.
                # XA.X:{BASE}{QUOTE} = per-minute aggregates (crypto)
                #
                # AM.* is NOT subscribed for stocks (2026-05-18): ws_agg
                # (built from A.* per-second) is the sole live_bars writer.
                # Subscribing AM created a second writer racing the same
                # PK, which contaminated the first_* decision-time columns
                # (whichever path inserted first won the snapshot). Every
                # monitor runs a ws_agg live_model, so AM contributed
                # nothing to alert decisions — only cache noise and WS load.
                stock_channels: List[str] = []
                for sym in stock_symbols:
                    hub = self.hubs.get(sym)
                    if hub is None:
                        continue
                    has_ltype = any(
                        any(t in _IB_L_TYPE_TRIGGERS
                            for t in getattr(m, '_intrabar_triggers', set()))
                        for m in hub.monitors.values()
                    )
                    has_subminute = any(
                        m.tf_seconds < 60 for m in hub.monitors.values())
                    has_ws_agg = any(
                        getattr(m, 'live_model', None) in (
                            'ws_agg_locked', 'ws_agg_with_rest_backfill'
                        )
                        for m in hub.monitors.values()
                    )
                    if has_ltype or has_subminute or has_ws_agg:
                        stock_channels.append(f"A.{sym}")
                # Phase H.1 (2026-05-15): shadow trade channels for SPY/TSLA.
                # Only added for symbols the manager is configured to track AND
                # that we already have hubs for, so we never subscribe to a
                # symbol we're not otherwise streaming.
                shadow_channels = [
                    ch for ch in self._trade_shadow.trade_channels()
                    if ch.split(".", 1)[-1] in stock_symbols
                ]
                stock_channels.extend(shadow_channels)
                crypto_channels = [f"XA.X:{s.replace('/', '')}" for s in crypto_symbols]

                all_channels = stock_channels + crypto_channels

                if not all_channels:
                    logger.warning("No symbols to subscribe — sleeping")
                    await _interruptible_sleep(10)
                    continue

                # Determine WebSocket URLs needed
                ws_tasks = []

                async def _run_polygon_ws(ws_url, channels, is_crypto=False):
                    """Connect to Polygon WS, auth, subscribe, and consume bars."""
                    async with websockets.connect(
                        ws_url,
                        ping_interval=10,
                        ping_timeout=180,
                        max_queue=1024,
                    ) as ws:
                        # Wait for connection message
                        msg = await ws.recv()
                        logger.debug("Polygon WS connect: %s", str(msg)[:200])

                        # Auth
                        await ws.send(json.dumps({
                            "action": "auth", "params": api_key
                        }))
                        auth_resp = await ws.recv()
                        auth_data = json.loads(auth_resp)
                        if isinstance(auth_data, list):
                            auth_data = auth_data[0]
                        if auth_data.get('status') != 'auth_success':
                            if is_crypto:
                                logger.warning("Polygon crypto WS auth failed: %s "
                                               "— crypto requires separate subscription. "
                                               "Crypto symbols will not stream.", auth_data)
                                return  # Exit gracefully, don't crash stock stream
                            raise ValueError(f"Polygon auth failed: {auth_data}")
                        logger.info("Polygon WS authenticated (%s)",
                                    "crypto" if is_crypto else "stocks")

                        # Subscribe
                        await ws.send(json.dumps({
                            "action": "subscribe",
                            "params": ",".join(channels)
                        }))
                        sub_resp = await ws.recv()
                        logger.info("Polygon subscribed to %d channels: %s",
                                    len(channels), str(sub_resp)[:200])

                        if not self._ws_confirmed:
                            self._ws_confirmed = True
                            backoff = 5
                            self._write_status(running=True, connected=True)

                        # Consume bars
                        async for raw_msg in ws:
                            if not self._running:
                                break
                            try:
                                events = json.loads(raw_msg)
                                if not isinstance(events, list):
                                    events = [events]
                                for ev in events:
                                    ev_type = ev.get('ev', '')
                                    # Phase H.1: route trade events to the
                                    # shadow bar builder. Returns fast when
                                    # the symbol isn't in shadow scope so we
                                    # don't pay any cost for non-shadowed
                                    # streams.
                                    if ev_type == 'T':
                                        try:
                                            self._trade_shadow.on_trade_event(ev)
                                        except Exception as _e:
                                            logger.debug(
                                                "trade shadow ingest error: %s",
                                                _e)
                                        continue
                                    if ev_type not in ('AM', 'XA', 'A', 'XAS'):
                                        continue  # skip status/other messages

                                    # Parse the symbol
                                    sym_raw = ev.get('sym', ev.get('pair', ''))
                                    if is_crypto and sym_raw.startswith('X:'):
                                        base = sym_raw[2:]
                                        if base.endswith('USD'):
                                            sym_raw = base[:-3] + '/USD'
                                        else:
                                            sym_raw = base

                                    hub = self.hubs.get(sym_raw)
                                    if hub is None:
                                        continue

                                    # Build bar dict in standard format
                                    bar_ts = datetime.fromtimestamp(
                                        ev.get('s', ev.get('e', 0)) / 1000,
                                        tz=timezone.utc
                                    )
                                    bar_dict = {
                                        'timestamp': bar_ts.isoformat(),
                                        'open': ev.get('o', 0),
                                        'high': ev.get('h', 0),
                                        'low': ev.get('l', 0),
                                        'close': ev.get('c', 0),
                                        'volume': ev.get('v', 0),
                                        # Optional Polygon fields — kept if
                                        # present so live_bars writes can
                                        # capture them. Aggregated downstream
                                        # bars don't carry these.
                                        'vwap': ev.get('vw'),
                                        'trade_count': ev.get('n'),
                                    }

                                    if ev_type in ('A', 'XAS'):
                                        # Phase H+ diag: A-channel ingestion lag.
                                        self._record_a_lag(
                                            ev.get('s', ev.get('e', 0)))
                                        # Grace-window shadow: 10Sec bars
                                        # at 2/3/4s grace (shadow-only).
                                        try:
                                            self._grace_shadow.on_second_bar(
                                                sym_raw, bar_dict)
                                        except Exception as _ge:
                                            logger.debug(
                                                "grace shadow ingest error: %s",
                                                _ge)
                                        # Per-second bar → L-type intra-bar detection
                                        hub.on_second_bar(
                                            bar_dict,
                                            alert_callback=self._on_alert,
                                            config=self._config,
                                        )
                                    else:
                                        # Per-minute bar → full bar-close processing
                                        tf_seconds = 60
                                        hub.on_polygon_bar(
                                            bar_dict, tf_seconds,
                                            alert_callback=self._on_alert,
                                            config=self._config,
                                            auditor=self.auditor,
                                        )
                            except json.JSONDecodeError:
                                logger.warning("Polygon WS bad JSON: %s",
                                               str(raw_msg)[:100])
                            except Exception as e:
                                logger.warning("Polygon bar processing error: %s", e)

                try:
                    tasks = []
                    if stock_channels:
                        tasks.append(_run_polygon_ws(
                            "wss://socket.polygon.io/stocks",
                            stock_channels, is_crypto=False))
                    if crypto_channels:
                        tasks.append(_run_polygon_ws(
                            "wss://socket.polygon.io/crypto",
                            crypto_channels, is_crypto=True))

                    done, pending = await asyncio.wait(
                        [asyncio.ensure_future(t) for t in tasks],
                        return_when=asyncio.FIRST_EXCEPTION)
                    for p in pending:
                        p.cancel()
                    for d in done:
                        if d.exception():
                            raise d.exception()

                except (websockets.WebSocketException, ConnectionError) as e:
                    self._ws_confirmed = False
                    self._write_status(running=True, connected=False)
                    if not self._running:
                        break
                    logger.warning("Polygon WS disconnected: %s — reconnect "
                                   "in %ds", e, backoff)
                    await _interruptible_sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

                except ValueError as e:
                    self._ws_confirmed = False
                    self._write_status(running=True, connected=False)
                    if not self._running:
                        break
                    logger.error("Polygon auth/subscribe error: %s — "
                                 "retrying in %ds", e, backoff)
                    await _interruptible_sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

                except Exception as e:
                    self._ws_confirmed = False
                    self._write_status(running=True, connected=False)
                    if not self._running:
                        break
                    logger.warning("Polygon stream error: %s — reconnect "
                                   "in %ds", e, backoff)
                    await _interruptible_sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

        finally:
            periodic_task.cancel()
            try:
                await periodic_task
            except asyncio.CancelledError:
                pass

    async def _stream_data(self):
        """Subscribe to Alpaca real-time trade data + run periodic tasks."""
        self._event_loop = asyncio.get_running_loop()

        from dotenv import load_dotenv
        load_dotenv(_SCRIPT_DIR / '.env', override=True)

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            logger.error("Alpaca API keys not configured")
            return

        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.enums import DataFeed
        except ImportError:
            logger.error("alpaca-py not installed")
            return

        # Feed selection: ALPACA_DATA_FEED env var (default: sip)
        feed_name = os.getenv("ALPACA_DATA_FEED", "sip").lower()
        data_feed = DataFeed.IEX if feed_name == "iex" else DataFeed.SIP
        logger.info("Using %s data feed", data_feed.value)

        # Patch the Alpaca SDK's _run_forever to propagate "connection
        # limit exceeded" errors instead of spinning in a tight retry
        # loop. The SDK catches ValueError internally but only breaks
        # for "insufficient subscription" — all others just log and
        # retry with sleep(0), which floods the server with requests.
        async def _patched_run_forever(stream_instance):
            """Wrapper that re-raises connection limit errors."""
            import websockets
            stream_instance._loop = asyncio.get_running_loop()
            while not any(
                v for k, v in stream_instance._handlers.items()
                if k not in ("cancelErrors", "corrections")
            ):
                if not stream_instance._stop_stream_queue.empty():
                    stream_instance._stop_stream_queue.get(timeout=1)
                    return
                await asyncio.sleep(0)
            stream_instance._should_run = True
            stream_instance._running = False
            while True:
                try:
                    if not stream_instance._should_run:
                        return
                    if not stream_instance._running:
                        await stream_instance._start_ws()
                        await stream_instance._send_subscribe_msg()
                        stream_instance._running = True
                    await stream_instance._consume()
                except websockets.WebSocketException as wse:
                    await stream_instance.close()
                    stream_instance._running = False
                    raise  # Let outer handler apply backoff
                except ValueError as ve:
                    await stream_instance.close()
                    stream_instance._running = False
                    msg = str(ve)
                    if "insufficient subscription" in msg:
                        return  # Unrecoverable — don't retry
                    raise  # All other ValueErrors get backoff
                except Exception as e:
                    await stream_instance.close()
                    stream_instance._running = False
                    raise  # Let outer handler apply backoff

        # Launch independent periodic task loop (not tick-driven)
        periodic_task = asyncio.ensure_future(self._periodic_tasks_loop())

        backoff = 5
        max_backoff = 120

        async def _interruptible_sleep(seconds):
            """Sleep in 1-second increments so stop() takes effect quickly."""
            for _ in range(int(seconds)):
                if not self._running:
                    return
                await asyncio.sleep(1)
            remainder = seconds - int(seconds)
            if remainder > 0 and self._running:
                await asyncio.sleep(remainder)

        # Shared trade handler for both stock and crypto streams.
        # Must be async — CryptoDataStream requires coroutine handlers.
        def _make_on_trade(symbol_list, is_crypto_stream=False):
            confirmed = [False]
            async def on_trade(trade):
                nonlocal backoff
                if not self._running:
                    return
                if not confirmed[0]:
                    confirmed[0] = True
                    if not self._ws_confirmed:
                        self._ws_confirmed = True
                        backoff = 5
                        self._write_status(running=True, connected=True)
                    stream_type = "Crypto" if is_crypto_stream else "Stock"
                    logger.info("%s WebSocket confirmed — receiving "
                                "trades for %d symbols",
                                stream_type, len(symbol_list))

                # Phase G (2026-05-15): condition filtering. Crypto
                # trades don't have CTA/UTP condition codes — skip
                # filter for them. Stocks: branch on feature flag.
                update_ohlc = True
                update_volume = True
                if not is_crypto_stream:
                    conditions = getattr(trade, 'conditions', None) or []
                    if _polygon_canonical_filter_enabled():
                        # Canonical numeric-code filter (matches what
                        # flat_file_ingestion uses for observable bars).
                        from polygon_conditions import _classify_eligibility
                        ohlc_ok, vol_ok = _classify_eligibility(conditions)
                        if not ohlc_ok and not vol_ok:
                            return  # rare; skip entirely
                        # Vol-only trades (Form T, Odd Lot, etc.) contribute
                        # to volume without moving H/L/C.
                        update_ohlc = ohlc_ok
                        update_volume = vol_ok
                    else:
                        # Legacy letter-code filter (rollback path).
                        if conditions and EXCLUDED_TRADE_CONDITIONS.intersection(
                                conditions):
                            return

                hub = self.hubs.get(trade.symbol)
                if hub:
                    hub.on_tick(
                        price=float(trade.price),
                        volume=int(trade.size)
                        if hasattr(trade, 'size') else 1,
                        timestamp=trade.timestamp,
                        alert_callback=self._on_alert,
                        config=self._config,
                        auditor=self.auditor,
                        update_ohlc=update_ohlc,
                        update_volume=update_volume,
                    )
            return on_trade

        try:
            while self._running:
                # Re-read symbols on each reconnect (may have changed
                # via hot-reload). Split into stock vs crypto.
                all_symbols = list(self.hubs.keys())
                stock_symbols = [s for s in all_symbols if '/' not in s]
                crypto_symbols = [s for s in all_symbols if '/' in s]
                self._subscribed_symbols = all_symbols

                try:
                    tasks = []

                    # Stock stream (if any stock symbols)
                    if stock_symbols:
                        stream = StockDataStream(
                            api_key, secret_key, feed=data_feed,
                            websocket_params={
                                'ping_interval': 10,
                                'ping_timeout': 180,
                                'max_queue': 1024,
                            })
                        self._stream_ref = stream
                        self._ws_confirmed = False

                        on_stock_trade = _make_on_trade(stock_symbols,
                                                        is_crypto_stream=False)
                        stream.subscribe_trades(on_stock_trade, *stock_symbols)
                        logger.info("Connecting to Alpaca STOCK stream for "
                                    "%d symbols (backoff=%ds)…",
                                    len(stock_symbols), backoff)
                        tasks.append(_patched_run_forever(stream))

                    # Crypto stream (if any crypto symbols)
                    if crypto_symbols:
                        try:
                            from alpaca.data.live import CryptoDataStream
                        except ImportError:
                            logger.error("CryptoDataStream not available in "
                                         "alpaca-py — crypto symbols skipped")
                            crypto_symbols = []

                    if crypto_symbols:
                        crypto_stream = CryptoDataStream(
                            api_key, secret_key,
                            websocket_params={
                                'ping_interval': 10,
                                'ping_timeout': 180,
                                'max_queue': 1024,
                            })
                        self._crypto_stream_ref = crypto_stream
                        if not stock_symbols:
                            self._stream_ref = crypto_stream
                            self._ws_confirmed = False

                        on_crypto_trade = _make_on_trade(crypto_symbols,
                                                         is_crypto_stream=True)
                        crypto_stream.subscribe_trades(on_crypto_trade,
                                                       *crypto_symbols)
                        logger.info("Connecting to Alpaca CRYPTO stream for "
                                    "%d symbols (backoff=%ds)…",
                                    len(crypto_symbols), backoff)
                        tasks.append(_patched_run_forever(crypto_stream))

                    if not tasks:
                        logger.warning("No symbols to subscribe — sleeping")
                        await _interruptible_sleep(10)
                        continue

                    # Run all streams concurrently; if any fails, all
                    # are cancelled and we reconnect
                    done, pending = await asyncio.wait(
                        [asyncio.ensure_future(t) for t in tasks],
                        return_when=asyncio.FIRST_EXCEPTION)
                    for p in pending:
                        p.cancel()
                    # Re-raise the first exception
                    for d in done:
                        if d.exception():
                            raise d.exception()

                except ValueError as e:
                    self._ws_confirmed = False
                    self._write_status(running=True, connected=False)
                    if not self._running:
                        break
                    msg = str(e).lower()
                    if "auth failed" in msg:
                        logger.error("Alpaca auth failed — check API keys "
                                     "in .env. Retrying in %ds", backoff)
                    elif "connection limit" in msg:
                        # Cap connection limit backoff at 30s — the old
                        # connection clears on its own, no need for 120s waits
                        conn_backoff = min(backoff, 30)
                        logger.warning("Connection limit exceeded — "
                                       "another stream may be active. "
                                       "Retrying in %ds", conn_backoff)
                        await _interruptible_sleep(conn_backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    else:
                        logger.warning("Stream error: %s — retrying in "
                                       "%ds", e, backoff)
                    await _interruptible_sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

                except Exception as e:
                    self._ws_confirmed = False
                    self._write_status(running=True, connected=False)
                    if not self._running:
                        break
                    logger.warning("Stream disconnected: %s — reconnect "
                                   "in %ds", e, backoff)
                    await _interruptible_sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
        finally:
            periodic_task.cancel()
            try:
                await periodic_task
            except asyncio.CancelledError:
                pass

    async def _periodic_tasks_loop(self):
        """Independent timer-driven loop for pickle writes, reconciliation,
        and strategy hot-reload. Runs regardless of tick arrival.

        Heavy I/O work (pickle writes, reconciliation) is offloaded to
        the default thread pool to avoid blocking the event loop.
        """
        logger.info("Periodic tasks loop started")
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                await asyncio.sleep(PICKLE_WRITE_INTERVAL)
                if not self._running:
                    break

                now = time.monotonic()

                # Flush stale bars — force-close any bars whose period has
                # expired based on wall-clock time.  This ensures bar-close
                # signals (especially bar_count_exit) fire promptly even
                # during low-liquidity periods (after-hours) where ticks
                # may arrive many seconds after bar period ends.
                # Runs in the event loop (not thread pool) to serialize
                # with the tick handler — no lock needed.
                try:
                    utc_now = datetime.now(timezone.utc)
                    for hub in self.hubs.values():
                        hub.flush_stale_bars(
                            utc_now,
                            alert_callback=self._on_alert,
                            config=self._config,
                            auditor=self.auditor,
                        )
                        # Gate-state telemetry + freeze alarm
                        # (2026-06-12) — change-detected, cheap.
                        hub.emit_gate_telemetry(utc_now)
                except Exception as e:
                    logger.debug("Stale bar flush error: %s", e)

                # Phase H.1: timer-driven close for shadow trade bars,
                # so bars from quiet-period buckets eventually emit
                # even when no follow-on trade arrives.
                try:
                    self._trade_shadow.flush_due()
                except Exception as e:
                    logger.debug("Trade shadow flush error: %s", e)
                try:
                    self._grace_shadow.flush_due()
                except Exception as e:
                    logger.debug("Grace shadow flush error: %s", e)

                # Pickle writes — offloaded to thread pool
                try:
                    await loop.run_in_executor(
                        None, self._write_chart_pickles)
                except Exception as e:
                    logger.debug("Pickle write error: %s", e)

                # Reconciliation — offloaded to thread pool
                if now - self._last_reconcile >= RECONCILE_INTERVAL:
                    self._last_reconcile = now
                    try:
                        await loop.run_in_executor(
                            None, self._reconcile_bars)
                    except Exception as e:
                        logger.debug("Reconcile error: %s", e)

                # Strategy hot-reload (every STRATEGY_REFRESH_INTERVAL
                # or immediately on engine_reload.flag IPC)
                reload_flag = _ENGINE_RELOAD_FLAG.exists()
                if reload_flag or now - self._last_strategy_refresh >= (
                        STRATEGY_REFRESH_INTERVAL):
                    self._last_strategy_refresh = now
                    if reload_flag:
                        try:
                            _ENGINE_RELOAD_FLAG.unlink()
                        except OSError:
                            pass
                        logger.info("Reload flag detected — refreshing "
                                    "strategies")
                    try:
                        self._hot_reload_strategies()
                    except Exception as e:
                        logger.debug("Hot-reload error: %s", e)

                # Status file update
                self._write_status(
                    running=True,
                    connected=self._ws_confirmed)
        except asyncio.CancelledError:
            pass
        logger.info("Periodic tasks loop stopped")

    def _write_chart_pickles(self):
        """Write enriched DataFrames to pickle for live chart consumption.

        Every write includes batch indicator enrichment so the live chart
        can render indicator overlays and generate_trades() can produce
        trade markers.  This runs on a 2-second throttle which is fast
        enough for the chart's 5-second refresh cycle.

        The batch pipeline runs on the chart slice (typically 300 bars) —
        not the full history — so it completes in <50ms per symbol.
        """
        import pickle
        from indicators import run_all_indicators, run_indicators_for_group
        from interpreters import run_all_interpreters, detect_all_triggers
        from confluence_groups import get_enabled_groups
        import general_packs as gp_module

        # Cache group + pack configs once per write cycle (avoid repeated
        # disk reads across the symbol/timeframe loop iterations).
        try:
            enabled_groups = get_enabled_groups()
        except Exception:
            enabled_groups = []
        try:
            gen_packs = gp_module.load_general_packs()
            enabled_gen = gp_module.get_enabled_general_packs(gen_packs)
        except Exception:
            enabled_gen = []

        for sym, hub in self.hubs.items():
            for tf_seconds, builder in hub.builders.items():
                df = builder.get_df_with_partial()
                if len(df) == 0:
                    continue

                max_bars = CHART_BAR_COUNTS.get(tf_seconds, DEFAULT_CHART_BARS)
                df_out = df.iloc[-max_bars:] if len(df) > max_bars else df

                # Full enrichment pipeline matching prepare_data_with_indicators()
                # so the live chart renders identical indicators and trade markers.
                try:
                    df_out = run_all_indicators(df_out)
                    for group in enabled_groups:
                        df_out = run_indicators_for_group(df_out, group)
                    df_out = run_all_interpreters(df_out)
                    df_out = detect_all_triggers(df_out)
                    for gpack in enabled_gen:
                        col_name = gpack.get_condition_column()
                        df_out[col_name] = gp_module.evaluate_condition(
                            df_out, gpack)
                except Exception as e:
                    logger.debug("Pickle enrichment error for %s/%s: %s",
                                 sym, tf_seconds, e)

                # Sanitize symbol for filesystem (BTC/USD → BTC_USD)
                _safe_sym = sym.replace('/', '_')
                pkl_path = _SCRIPT_DIR / f"live_data_{_safe_sym}_{tf_seconds}.pkl"
                tmp_path = str(pkl_path) + ".tmp"
                try:
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(df_out, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, str(pkl_path))
                except Exception:
                    pass

    def _reconcile_bars(self):
        """Fetch recent bars from REST API to correct tick-built bar drift.

        No-op when using Polygon data provider (WS bars = REST bars by construction).
        """
        _dp = os.getenv("DATA_PROVIDER", "auto").lower()
        _pk = os.getenv("POLYGON_API_KEY", "")
        if _dp in ("polygon", "auto") and _pk and not _pk.startswith("YOUR_"):
            return  # Polygon WS bars match REST — no reconciliation needed

        try:
            from data_loader import load_market_data
        except ImportError:
            return

        for sym, hub in self.hubs.items():
            for tf_seconds, builder in hub.builders.items():
                if len(builder.history) == 0:
                    continue
                # Sub-minute TFs not available via REST API — skip gracefully
                if tf_seconds < 60:
                    continue
                tf_str = SECONDS_TO_TIMEFRAME.get(tf_seconds, '1Min')
                session = 'RTH'
                for m in hub.monitors.values():
                    if m.tf_seconds == tf_seconds:
                        session = m.session
                        break
                try:
                    df_rest = load_market_data(
                        sym, days=1, timeframe=tf_str,
                        feed='sip', session=session)
                except Exception:
                    continue

                if df_rest is None or len(df_rest) == 0:
                    continue

                # Only correct bars BEFORE the grace window (don't touch
                # the last N bars which may still be forming).
                # Take a snapshot reference — if _append_to_history runs
                # concurrently and reassigns builder.history via pd.concat,
                # we work on the copy and reassign atomically at the end.
                hist = builder.history.copy()
                if len(hist) < RECONCILE_GRACE_BARS + 1:
                    continue

                correctable = hist.iloc[:-RECONCILE_GRACE_BARS]
                corrections = 0
                for ts in correctable.index:
                    if ts in df_rest.index:
                        rest_row = df_rest.loc[ts]
                        for col in ('open', 'high', 'low', 'close', 'volume'):
                            if col in rest_row and col in hist.columns:
                                old_val = hist.at[ts, col]
                                new_val = rest_row[col]
                                if abs(float(old_val) - float(new_val)) > 1e-6:
                                    hist.at[ts, col] = new_val
                                    corrections += 1

                if corrections > 0:
                    builder.history = hist
                    logger.debug("Reconciled %d values for %s/%s",
                                corrections, sym, tf_str)

    def _hot_reload_strategies(self):
        """Check for new/removed strategies and update monitors."""
        try:
            from alerts import load_alert_config
            from alert_monitor import get_monitored_strategies

            config = load_alert_config()
            strategies = get_monitored_strategies(config)
        except Exception as e:
            logger.warning("Strategy hot-reload failed: %s", e)
            return

        # Refresh general packs (user may have enabled/disabled packs)
        self._general_packs = _load_enabled_general_packs()

        new_ids = {s['id'] for s in strategies}
        current_ids = set(self.monitors.keys())

        # Remove strategies no longer monitored
        removed = current_ids - new_ids
        for sid in removed:
            monitor = self.monitors.pop(sid, None)
            if monitor:
                # Remove from hub
                hub = self.hubs.get(monitor.symbol)
                if hub:
                    hub.monitors.pop(sid, None)
                logger.info("Hot-reload: removed strategy %d (%s)",
                           sid, monitor.strat_name)

        # Add new strategies
        added = new_ids - current_ids
        # Track hubs that gained NEW secondary TF builders so we can
        # finalize shadow engines + warm them up after monitors are in.
        # Without this, cross-TF confluence on hot-reloaded strategies
        # silently breaks: the secondary builder accumulates bars but no
        # shadow indicator engine exists, so the MTF confluence buffer
        # never updates and `live_bars` writes for the secondary TF
        # don't happen (the fan-out at on_polygon_bar:1912 iterates
        # `_shadow_engines.keys()`). Bug found 2026-05-13 — sid 169 was
        # firing 0 alerts because its 15Min SWING_123 confluence shadow
        # never got built when the strategy was created via the UI.
        hubs_with_new_secondary_tfs: Dict[str, Set[int]] = {}
        for strat in strategies:
            if strat['id'] not in added:
                continue

            sid = strat['id']
            sym = strat.get('symbol', 'SPY')

            # Ensure hub exists
            if sym not in self.hubs:
                self.hubs[sym] = SymbolHub(sym)

            hub = self.hubs[sym]
            # Snapshot builder TFs BEFORE add_monitor — anything new
            # afterward is a fresh secondary TF that needs warmup +
            # shadow-engine setup.
            tfs_before = set(hub.builders.keys())

            monitor = StrategyMonitor(
                strat, general_packs=self._general_packs)
            hub.add_monitor(monitor)
            self.monitors[sid] = monitor

            tfs_after = set(hub.builders.keys())
            new_secondary_tfs = tfs_after - tfs_before - {monitor.tf_seconds}
            if new_secondary_tfs:
                hubs_with_new_secondary_tfs.setdefault(sym, set()).update(
                    new_secondary_tfs)

            # Warmup the new monitor (primary TF)
            tf_seconds = monitor.tf_seconds
            builder = hub.builders.get(tf_seconds)
            if builder and len(builder.history) > 0:
                monitor.warmup(builder.history)
            else:
                # Load historical data for warmup
                try:
                    # Bug 5: shared warmup loader (TF-scaled, resample-from-1min
                    # for >=60s). Sub-minute primary stays native days=7.
                    df = _load_warmup_df(sym, tf_seconds, monitor.session)
                    if df is not None and len(df) > 0:
                        # seed_history splits any forming bar into
                        # _partial; warm up monitor from the resulting
                        # closed-only history (not raw df).
                        hub.seed_history(tf_seconds, df)
                        if builder and len(builder.history) > 0:
                            monitor.warmup(builder.history)
                except Exception as e:
                    logger.error("Warmup failed for new strategy %d: %s",
                                sid, e)

            # Seed history for any NEW secondary TF builders this monitor
            # introduced. They were created empty by add_monitor; without
            # historical bars the shadow engine can't initialize.
            for sec_tf in new_secondary_tfs:
                sec_builder = hub.builders.get(sec_tf)
                if sec_builder is None or len(sec_builder.history) > 0:
                    continue  # already seeded by another monitor this pass
                try:
                    sec_tf_str = SECONDS_TO_TIMEFRAME.get(sec_tf, '1Min')
                    # Bug 5: TF-scaled, resample-from-1min warmup for the new
                    # secondary/gate TF (was flat days=7 → coarse gates starved
                    # → never warmed → gated strategy never entered live).
                    sec_df = _load_warmup_df(sym, sec_tf, monitor.session)
                    if sec_df is not None and len(sec_df) > 0:
                        # Drop the in-progress (currently-forming) bar
                        # if present. load_market_data sometimes returns
                        # a row at the current period's start before that
                        # period has closed. If we seed it, every
                        # subsequent 1Min Polygon bar that aligns to the
                        # same period_start would match history.index[-1]
                        # and false-trigger the rebroadcast branch in
                        # accept_second_bar — blocking the partial-bar
                        # accumulator and spamming pandas errors.
                        now_utc = pd.Timestamp.now(tz='UTC')
                        period_size_ns = int(sec_tf) * 1_000_000_000
                        current_period_start = pd.Timestamp(
                            (now_utc.value // period_size_ns) * period_size_ns,
                            tz='UTC',
                        )
                        last_idx = sec_df.index[-1]
                        if last_idx.tzinfo is None:
                            last_idx = last_idx.tz_localize('UTC')
                        if last_idx >= current_period_start:
                            sec_df = sec_df.iloc[:-1]
                            logger.info(
                                "Hot-reload: dropped forming bar from "
                                "seed %s/%ss (last=%s, current_period=%s)",
                                sym, sec_tf, last_idx, current_period_start)
                    if sec_df is not None and len(sec_df) > 0:
                        hub.seed_history(sec_tf, sec_df)
                        logger.info(
                            "Hot-reload: seeded secondary TF %s/%ss "
                            "(%d bars) for new strat %d",
                            sym, sec_tf, len(sec_df), sid)
                except Exception as e:
                    logger.error(
                        "Hot-reload: secondary TF seed failed for "
                        "strat %d %s/%ss: %s",
                        sid, sym, sec_tf, e)

            logger.info("Hot-reload: added strategy %d (%s / %s)",
                       sid, strat.get('name'), sym)

        # Finalize shadow engines on any hub whose monitor set changed
        # (added OR removed — removal can drop a strategy whose required
        # secondary TF is no longer needed, but finalize is idempotent
        # and safe). Then warm up any newly-created shadow engines from
        # the seeded secondary-TF history.
        if added or removed:
            for sym, hub in self.hubs.items():
                # Only do this work on hubs that touched new strategies.
                # Removed-only hubs don't need new shadow engines.
                if sym not in hubs_with_new_secondary_tfs and not added:
                    continue
                # Bug Hunt Wave 1 #1: shadow keys are (tf, session).
                pre_shadow_keys = set(hub._shadow_engines.keys())
                hub.finalize_shadow_engines()
                post_shadow_keys = set(hub._shadow_engines.keys())
                new_shadow_keys = post_shadow_keys - pre_shadow_keys
                for (sec_tf, sh_session) in new_shadow_keys:
                    sec_builder = hub.builders.get(sec_tf)
                    shadow = hub._shadow_engines.get((sec_tf, sh_session))
                    if sec_builder is None or shadow is None:
                        continue
                    if sh_session != 'RTH':
                        # Flag-ON non-RTH shadow: the builder history is
                        # the RTH-flavored feed — load session-correct
                        # bars instead (mirrors _warmup_all).
                        try:
                            sess_df = _closed_bars_only(
                                _load_warmup_df(sym, sec_tf, sh_session),
                                sec_tf)
                            if len(sess_df) > 0:
                                shadow.warmup(sess_df)
                                logger.info(
                                    "Hot-reload: shadow %s/%ss session=%s "
                                    "warmed up (%d bars) initialized=%s",
                                    sym, sec_tf, sh_session, len(sess_df),
                                    shadow.indicators._initialized)
                            else:
                                logger.warning(
                                    "Hot-reload: shadow %s/%ss session=%s "
                                    "— no session history; stays cold",
                                    sym, sec_tf, sh_session)
                        except Exception as e:
                            logger.error(
                                "Hot-reload: session-shadow warmup failed "
                                "%s/%ss session=%s: %s",
                                sym, sec_tf, sh_session, e)
                        continue
                    if len(sec_builder.history) == 0:
                        logger.warning(
                            "Hot-reload: shadow %s/%ss created but builder "
                            "has no history — indicators stay cold until "
                            "live bars accumulate",
                            sym, sec_tf)
                        continue
                    try:
                        shadow.warmup(sec_builder.history)
                        logger.info(
                            "Hot-reload: shadow %s/%ss warmed up "
                            "(%d bars) initialized=%s",
                            sym, sec_tf, len(sec_builder.history),
                            shadow.indicators._initialized)
                    except Exception as e:
                        logger.error(
                            "Hot-reload: shadow warmup failed %s/%ss: %s",
                            sym, sec_tf, e)

        if added or removed:
            self._config = config
            self.strategies = strategies
            logger.info("Hot-reload complete: %d strategies active",
                       len(self.monitors))

            # Check if new symbols need stream subscription — if so,
            # close the current stream to trigger a reconnect with the
            # updated symbol list.
            new_symbols = {
                s.get('symbol', 'SPY') for s in strategies
                if s['id'] in added
            } - set(self._subscribed_symbols or [])
            if new_symbols and self._stream_ref:
                logger.info("New symbols detected (%s) — forcing "
                            "stream reconnect", ', '.join(new_symbols))
                try:
                    self._stream_ref.close()
                except Exception:
                    pass

    def _save_all_positions(self):
        """Persist all position states."""
        positions = {sid: m.position.state
                     for sid, m in self.monitors.items()}
        save_engine_state(positions)

    def _write_status(self, running: bool, connected: bool = False):
        """Write engine status file for UI consumption."""
        status = {
            'running': running,
            'pid': os.getpid(),
            'started_at': self._start_time,
            'connected': connected,
            'symbols': list(self.hubs.keys()),
            'strategies': len(self.strategies),
            'tick_count': sum(h.tick_count for h in self.hubs.values()),
            'streaming_connected': connected,
        }
        tmp = str(_ENGINE_STATUS_FILE) + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(status, f, indent=2)
            os.replace(tmp, str(_ENGINE_STATUS_FILE))
        except Exception:
            pass

    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            'running': self._running,
            'symbols': list(self.hubs.keys()),
            'strategies': len(self.strategies),
            'tick_count': sum(h.tick_count for h in self.hubs.values()),
            'started_at': self._start_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _load_config_and_strategies():
    """Load alert config and resolve monitored strategies."""
    sys.path.insert(0, str(_SCRIPT_DIR))
    from alerts import load_alert_config
    from alert_monitor import get_monitored_strategies

    from dotenv import load_dotenv
    load_dotenv(_SCRIPT_DIR / '.env')

    config = load_alert_config()
    strategies = get_monitored_strategies(config)
    return config, strategies


def cmd_start():
    """Start the Ralph engine."""
    # Check if already running
    if _ENGINE_STATUS_FILE.exists():
        try:
            with open(_ENGINE_STATUS_FILE) as f:
                status = json.load(f)
            pid = status.get('pid', 0)
            if status.get('running') and pid:
                # Check if process is actually alive
                try:
                    os.kill(pid, 0)
                    logger.error("Engine already running (PID %d)", pid)
                    return
                except OSError:
                    pass  # PID not alive, stale status file
        except Exception:
            pass

    config, strategies = _load_config_and_strategies()

    if not strategies:
        logger.error("No strategies to monitor. Configure portfolios with "
                     "active webhooks first.")
        return

    logger.info("Resolved %d strategies to monitor:", len(strategies))
    symbols = set()
    for s in strategies:
        symbols.add(s.get('symbol'))
        logger.info("  %d: %s (%s / %s)",
                     s['id'], s.get('name'), s.get('symbol'),
                     s.get('timeframe'))
    logger.info("Symbols: %s", symbols)

    # Set up signal handlers
    engine = RalphEngine()

    def handle_signal(signum, frame):
        logger.info("Received signal %d — stopping", signum)
        engine.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    engine.start(strategies, config)


def cmd_status():
    """Print engine status."""
    if not _ENGINE_STATUS_FILE.exists():
        print("Engine status: NOT RUNNING (no status file)")
        return

    try:
        with open(_ENGINE_STATUS_FILE) as f:
            status = json.load(f)
    except Exception as e:
        print(f"Error reading status: {e}")
        return

    pid = status.get('pid', 0)
    is_alive = False
    if pid:
        try:
            os.kill(pid, 0)
            is_alive = True
        except OSError:
            pass

    if status.get('running') and is_alive:
        print(f"Engine status: RUNNING (PID {pid})")
        print(f"  Connected: {status.get('connected', False)}")
        print(f"  Symbols: {status.get('symbols', [])}")
        print(f"  Strategies: {status.get('strategies', 0)}")
        print(f"  Ticks: {status.get('tick_count', 0):,}")
        print(f"  Started: {status.get('started_at', 'unknown')}")

        # Show position states if engine_state.json exists
        if _ENGINE_STATE_FILE.exists():
            try:
                positions = load_engine_state()
                if positions:
                    print("  Positions:")
                    for sid, ps in positions.items():
                        state_str = ps.state if hasattr(ps, 'state') else str(ps)
                        extra = ""
                        if hasattr(ps, 'entry_price') and ps.entry_price:
                            extra = f" @ {ps.entry_price:.2f}"
                        if hasattr(ps, 'stop_price') and ps.stop_price:
                            extra += f" stop={ps.stop_price:.4f}"
                        print(f"    S{sid}: {state_str}{extra}")
            except Exception:
                pass
    else:
        print(f"Engine status: STOPPED (PID {pid} not alive)")


def cmd_stop():
    """Send SIGTERM to the running engine."""
    if not _ENGINE_STATUS_FILE.exists():
        print("No engine status file found")
        return

    try:
        with open(_ENGINE_STATUS_FILE) as f:
            status = json.load(f)
    except Exception as e:
        print(f"Error reading status: {e}")
        return

    pid = status.get('pid', 0)
    if not pid:
        print("No PID in status file")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
    except OSError as e:
        print(f"Failed to signal PID {pid}: {e}")


def cmd_dry_run():
    """Load config, resolve strategies, warmup, and print diagnostics.

    Does NOT connect to the live stream. Useful for verifying that
    strategy resolution, indicator warmup, and position sync all work
    before starting the real engine.
    """
    config, strategies = _load_config_and_strategies()
    if not strategies:
        print("No strategies to monitor.")
        return

    print(f"Resolved {len(strategies)} strategies:")
    for s in strategies:
        print(f"  S{s['id']}: {s.get('name')} ({s.get('symbol')} / "
              f"{s.get('timeframe')})")

    symbols = {s.get('symbol') for s in strategies}
    print(f"\nSymbols: {sorted(symbols)}")

    engine = RalphEngine()
    engine.strategies = strategies
    engine._config = config
    engine._running = True
    engine._general_packs = _load_enabled_general_packs()

    # Group by symbol, create hubs/monitors
    by_symbol: Dict[str, List[dict]] = {}
    for strat in strategies:
        sym = strat.get('symbol', 'SPY')
        by_symbol.setdefault(sym, []).append(strat)

    saved_positions = load_engine_state()

    for sym, strats in by_symbol.items():
        hub = SymbolHub(sym)
        for strat in strats:
            pos_state = saved_positions.get(strat['id'])
            monitor = StrategyMonitor(
                strat, pos_state, general_packs=engine._general_packs)
            hub.add_monitor(monitor)
            engine.monitors[strat['id']] = monitor
            # Tier 3 §8.3: dry-run collects carryovers for diagnostic
            # display; same semantic as the production path.
            if monitor._tier3_inherited_carryover is not None:
                engine._pending_carryovers.append(
                    monitor._tier3_inherited_carryover)
        engine.hubs[sym] = hub

    print("\nRunning warmup...")
    engine._warmup_all()
    print("Warmup complete.")

    # Position sync
    if saved_positions:
        strats_needing_sync = [
            s for s in strategies if s['id'] not in saved_positions]
    else:
        strats_needing_sync = strategies
    if strats_needing_sync:
        print(f"\nSyncing initial positions for {len(strats_needing_sync)} "
              f"strategies...")
        engine._sync_initial_positions(strats_needing_sync)

    # Print final state
    print("\nFinal monitor states:")
    for sid, m in sorted(engine.monitors.items()):
        pos = m.position.state.status
        indicators = list(m.indicators.required)
        entry = m.position.state.entry_price
        stop = m.position.state.stop_price
        extra = ""
        if pos == 'IN_POSITION':
            extra = f" entry={entry:.2f} stop={stop:.4f}"
        print(f"  S{sid} ({m.strat_name}): {pos}{extra} | "
              f"indicators={sorted(indicators)}")

    engine._running = False
    print("\nDry run complete — no live connection was made.")


def main():
    parser = argparse.ArgumentParser(
        description="Ralph Wiggum Alert Engine")
    parser.add_argument('--status', action='store_true',
                        help='Print engine status')
    parser.add_argument('--stop', action='store_true',
                        help='Stop the running engine')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load, warmup, and print diagnostics without '
                             'connecting to the live stream')
    args = parser.parse_args()

    # Ensure we're in the right directory
    os.chdir(str(_SCRIPT_DIR))

    if args.status:
        cmd_status()
    elif args.stop:
        cmd_stop()
    elif args.dry_run:
        cmd_dry_run()
    else:
        cmd_start()


if __name__ == '__main__':
    main()
