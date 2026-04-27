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

    def update(self, price: float, volume: int = 1):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
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

    def seed_history(self, df: pd.DataFrame):
        if df is not None and len(df) > 0:
            self.history = df.tail(MAX_HISTORY).copy()
            self._bar_count = len(self.history)

    def process_tick(self, price: float, volume: int,
                     timestamp: datetime) -> Optional[dict]:
        period_start = self._align_to_period(timestamp)
        if self._partial is None:
            self._partial = PartialBar(price, period_start, self.tf_seconds)
            self._partial.update(price, volume)
            return None
        if period_start > self._partial.bar_start:
            fill_close = self._partial.close
            old_bar_end = self._partial.bar_start + timedelta(
                seconds=self.tf_seconds)
            completed = self._close_bar()
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
            self._partial = PartialBar(price, period_start, self.tf_seconds)
            self._partial.update(price, volume)
            return completed
        self._partial.update(price, volume)
        return None

    def accept_second_bar(self, bar_dict: dict,
                          close_on_boundary: bool = True) -> Optional[dict]:
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
            self._partial.volume = sec_volume
            self._partial.tick_count = 1
            return completed

        # Same period — aggregate into existing partial
        self._partial.high = max(self._partial.high, sec_high)
        self._partial.low = min(self._partial.low, sec_low)
        self._partial.close = sec_close
        self._partial.volume += sec_volume
        self._partial.tick_count += 1
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

    def accept_bar(self, bar_dict: dict) -> dict:
        """Accept a pre-built bar (from Polygon WebSocket).

        Appends to history, increments bar count, gap-fills missing bars.
        Returns the bar dict for downstream consumption.
        """
        bar_ts = pd.Timestamp(bar_dict['timestamp'])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize('UTC')
        bar_start = self._align_to_period(bar_ts.to_pydatetime())

        # Gap-fill if there are missing bars between last history and this bar
        if len(self.history) > 0:
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
        return bar_dict

    def force_close_stale_bar(self, now: datetime) -> Optional[dict]:
        """Close the partial bar if wall-clock time has passed bar_end.

        Returns the completed bar dict, or None if no bar was stale.
        Gap-fills any missing intermediate bars (same logic as process_tick).
        """
        if self._partial is None:
            return None
        bar_end = self._partial.bar_start + timedelta(seconds=self.tf_seconds)
        if now < bar_end:
            return None  # Bar still forming
        fill_close = self._partial.close
        completed = self._close_bar()
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
        return completed


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY MONITOR — one per strategy, owns indicator + trigger + position
# ═══════════════════════════════════════════════════════════════════════════

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
        self.position = PositionStateMachine(
            strategy, position_state,
            resolved_entry=resolved_entry,
            resolved_exits=resolved_exits)

        # Confluence records for current bar
        self._current_confluence: Set[str] = set()
        self._interp_keys = list(req_interp)

        # General Pack references (for GEN- confluence records)
        # Only keep packs referenced by this strategy's general_confluences
        gp_ids_needed = set()
        for rec in strategy.get('general_confluences', []):
            parts = rec.split('-', 2)
            if len(parts) >= 2:
                gp_ids_needed.add(parts[1].lower())
        self._general_packs = [
            p for p in (general_packs or [])
            if p.id in gp_ids_needed
        ] if gp_ids_needed else []

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

    def warmup(self, df: pd.DataFrame):
        """Initialize indicator state from historical bars."""
        self.indicators.warmup(df)

    def on_bar_close(self, bar: dict,
                     bar_count: int,
                     mtf_confluence: Dict[int, Set[str]] = None,
                     ) -> Tuple[List[dict], dict]:
        """Process a completed bar.

        Args:
            mtf_confluence: Cross-TF confluence buffer from SymbolHub.
                Maps tf_seconds → latest confluence records from other TFs.

        Returns:
            (signals, audit_data) — signals list and dict for fidelity logging.
        """
        signals = []

        # 1. Update indicators (O(1))
        current = self.indicators.update_bar(bar)
        prev = self.indicators.get_prev_values()

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
            if isinstance(bar_ts, datetime):
                self._current_confluence |= _evaluate_general_packs(
                    self._general_packs, bar_ts)

        # 3c. Merge cross-TF confluence records from hub's shared buffer
        if mtf_confluence:
            for other_tf, records in mtf_confluence.items():
                if other_tf != self.tf_seconds:
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
        return current, interps

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
# SHADOW INDICATOR ENGINE — secondary TF interpreter state for MTF confluence
# ═══════════════════════════════════════════════════════════════════════════

class _ShadowIndicatorEngine:
    """Runs indicator/interpreter pipeline for a secondary TF.

    Produces confluence records but does NOT manage positions, triggers, or
    alerts.  Used by SymbolHub to provide cross-TF interpreter states to
    primary monitors for MTF confluence gating.
    """

    def __init__(self, tf_seconds: int, req_ind: Set[str],
                 req_interp: Set[str], params: Dict[str, Any]):
        self.tf_seconds = tf_seconds
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
        }

        if signal['type'] == 'exit_signal':
            alert['entry_price'] = signal.get('entry_price')
            alert['entry_stop_price'] = signal.get('entry_stop_price')

        alert = enrich_signal_with_portfolio_context(alert, strategy['id'])
        alert = save_alert(alert)

        logger.info("ALERT: %s for %s (%s) trigger=%s price=%.2f",
                     signal['type'], strategy.get('name'), strategy.get('symbol'),
                     signal.get('trigger'), signal.get('price', 0))

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
        # Cross-TF confluence: tf_seconds → latest interpreter confluence records
        self._mtf_confluence: Dict[int, Set[str]] = {}
        # Shadow engines for secondary TFs not covered by real monitors
        self._shadow_engines: Dict[int, '_ShadowIndicatorEngine'] = {}
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

    def _fire_alert(self, sig: dict, monitor: 'StrategyMonitor',
                    config: dict, alert_callback: Optional[Callable]) -> None:
        """Stamp actual_price onto the signal and invoke alert_callback.

        Trade_Timestamps_Spec (2026-04-20): actual_price = the most recent
        per-second close we've seen for this symbol. Gap between
        actual_price and sig['price'] (engine's theoretical fill) =
        price slippage. Zero latency — in-memory dict lookup.
        """
        if not alert_callback or not sig:
            return
        if self.latest_close is not None and 'actual_price' not in sig:
            sig['actual_price'] = self.latest_close
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
            for m in self.monitors.values():
                if m.tf_seconds != tf_seconds:
                    continue
                ind, interp = m.compute_tentative_state(forming_bar)
                indicators.update(ind)
                states.update(interp)

            state_cross_tf: Dict[str, str] = {}
            for other_tf, records in self._mtf_confluence.items():
                if other_tf == tf_seconds:
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

    def finalize_shadow_engines(self):
        """Create shadow indicator engines for secondary TFs not covered by
        real monitors.  Call after all monitors have been added.  Idempotent.
        """
        # Collect all secondary TFs needed across monitors
        needed: Dict[int, List[StrategyMonitor]] = {}
        for monitor in self.monitors.values():
            for sec_tf in getattr(monitor, '_required_secondary_tf', set()):
                needed.setdefault(sec_tf, []).append(monitor)

        # For each needed secondary TF, check if a real monitor covers it
        for sec_tf, requesting_monitors in needed.items():
            if sec_tf in self._shadow_engines:
                continue  # already created
            has_real = any(m.tf_seconds == sec_tf for m in self.monitors.values())
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
                    for _tpl_key, (ind_set, ikey) in TEMPLATE_REQUIREMENTS.items():
                        if ikey == interp_key:
                            req_ind.update(ind_set)
                            break

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
                    sec_tf, req_ind, req_interp, shadow_params)
                self._shadow_engines[sec_tf] = shadow
                logger.info("Created shadow engine for %s/%ss: ind=%s interp=%s",
                            self.symbol, sec_tf, req_ind, req_interp)

    def seed_history(self, tf_seconds: int, df: pd.DataFrame):
        builder = self.builders.get(tf_seconds)
        if builder:
            builder.seed_history(df)

    def on_tick(self, price: float, volume: int, timestamp: datetime,
                alert_callback: Callable = None, config: dict = None,
                auditor: 'FidelityAuditor' = None):
        """Route tick to bar builders and strategy monitors."""
        self.tick_count += 1
        self.last_tick_time = timestamp
        ts_str = timestamp.isoformat()

        for tf_seconds, builder in self.builders.items():
            completed = builder.process_tick(price, volume, timestamp)

            if completed is not None:
                # Bar close — update shared confluence buffer FIRST
                # so monitors on other TFs can see this TF's state
                shadow = self._shadow_engines.get(tf_seconds)
                if shadow and shadow.indicators._initialized:
                    self._mtf_confluence[tf_seconds] = shadow.on_bar_close(completed)
                else:
                    # Use first real monitor's confluence as the TF's records
                    for m in self.monitors.values():
                        if m.tf_seconds == tf_seconds:
                            # Will be updated after on_bar_close, but for
                            # same-TF monitors the records don't matter
                            break

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
                        len(r) for tf, r in self._mtf_confluence.items()
                        if tf != tf_seconds) if self._mtf_confluence else 0
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
                # monitor if no shadow engine covers it)
                if tf_seconds not in self._mtf_confluence or \
                        not self._shadow_engines.get(tf_seconds):
                    for m in self.monitors.values():
                        if m.tf_seconds == tf_seconds:
                            # Use interpreter-only records (exclude GEN-)
                            own_records = {r for r in m._current_confluence
                                           if not r.startswith('GEN-')}
                            self._mtf_confluence[tf_seconds] = own_records
                            break

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

            # Update shared confluence buffer first (shadow or real monitor)
            shadow = self._shadow_engines.get(tf_seconds)
            if shadow and shadow.indicators._initialized:
                self._mtf_confluence[tf_seconds] = shadow.on_bar_close(completed)

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
            if not self._shadow_engines.get(tf_seconds):
                for m in self.monitors.values():
                    if m.tf_seconds == tf_seconds:
                        own_records = {r for r in m._current_confluence
                                       if not r.startswith('GEN-')}
                        self._mtf_confluence[tf_seconds] = own_records
                        break

            # M8.5: broadcast completed bar to Supabase Realtime (live chart)
            self._publish_completed_bar(tf_seconds, completed)

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
        # Update shared confluence buffer (shadow engines first)
        shadow = self._shadow_engines.get(tf_seconds)
        if shadow and shadow.indicators._initialized:
            self._mtf_confluence[tf_seconds] = shadow.on_bar_close(bar_dict)

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

            signals, audit_data = monitor.on_bar_close(
                bar_dict, bar_count,
                mtf_confluence=self._mtf_confluence)

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
        if not self._shadow_engines.get(tf_seconds):
            for m in self.monitors.values():
                if m.tf_seconds == tf_seconds:
                    own_records = {r for r in m._current_confluence
                                   if not r.startswith('GEN-')}
                    self._mtf_confluence[tf_seconds] = own_records
                    break

        # Build cross-TF state snapshot the same way as the forming-bar path
        state_cross_tf: Dict[str, str] = {}
        for other_tf, records in self._mtf_confluence.items():
            if other_tf == tf_seconds:
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

        # Accept the pre-built bar into the builder (handles gap-fill)
        builder.accept_bar(bar_dict)

        self._run_monitor_pipeline_for_completed_bar(
            tf_seconds=tf_seconds,
            bar_dict=bar_dict,
            bar_count=builder._bar_count,
            alert_callback=alert_callback,
            config=config,
            auditor=auditor,
            source_label='polygon',
        )

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

        # Set of primary timeframes (each monitor's tf_seconds). Skip secondary
        # TF builders — those are seeded at warmup and used for cross-TF
        # confluence, not driven by live per-second data.
        primary_tfs = {m.tf_seconds for m in self.monitors.values()}

        for tf_seconds, b in self.builders.items():
            if tf_seconds not in primary_tfs:
                continue

            if tf_seconds < 60:
                # Sub-minute: per-second is the canonical source. Aggregate.
                completed = b.accept_second_bar(bar_dict, close_on_boundary=True)
                if completed is not None:
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
                # history. NEVER runs monitors here.
                b.accept_second_bar(bar_dict, close_on_boundary=False)
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
        # M8.5: Live-bar publisher — set by worker before start().
        # None = live chart disabled (no broadcasts). Harmless default.
        self._publisher: Optional['LiveBarPublisher'] = None
        # Phase 3 (Alert_Recovery_Plan_2026-04-17): ThreadPoolExecutor for
        # alert-dispatch DB + webhook I/O. Set by the worker before start().
        # When None, alert dispatch runs synchronously on the event loop
        # (legacy behavior, fine for CLI / tests — blocks ~200-500ms per alert).
        self._alert_executor = None

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

        # Load enabled general packs for GEN- confluence evaluation
        self._general_packs = _load_enabled_general_packs()

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

            self.hubs[sym] = hub

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

    def _warmup_all(self):
        """Load historical data and initialize all monitors."""
        from data_loader import load_market_data, is_crypto

        seen_tf: Dict[Tuple[str, int], pd.DataFrame] = {}

        for sym, hub in self.hubs.items():
            for tf_seconds, builder in hub.builders.items():
                cache_key = (sym, tf_seconds)
                if cache_key in seen_tf:
                    df = seen_tf[cache_key]
                else:
                    tf_str = SECONDS_TO_TIMEFRAME.get(tf_seconds, '1Min')
                    # Determine session from first monitor using this TF
                    session = 'RTH'
                    for m in hub.monitors.values():
                        if m.tf_seconds == tf_seconds:
                            session = m.session
                            break

                    try:
                        # Crypto: session is irrelevant (24/7 market),
                        # load_market_data auto-routes via is_crypto()
                        df = load_market_data(
                            sym, days=7, timeframe=tf_str,
                            feed='sip', session=session)
                    except Exception as e:
                        logger.error("Warmup failed for %s/%s: %s",
                                     sym, tf_str, e)
                        df = pd.DataFrame()
                    seen_tf[cache_key] = df

                # Seed bar builder
                hub.seed_history(tf_seconds, df)

                # Warmup each monitor's indicators
                for monitor in hub.monitors.values():
                    if monitor.tf_seconds == tf_seconds and len(df) > 0:
                        monitor.warmup(df)
                        logger.info("Warmup %s: strat=%s tf=%ss bars=%d initialized=%s",
                                    sym, monitor.strat_id, tf_seconds, len(df),
                                    monitor.indicators._initialized)
                    elif monitor.tf_seconds == tf_seconds:
                        logger.warning("Warmup SKIPPED %s: strat=%s tf=%ss — no historical data",
                                       sym, monitor.strat_id, tf_seconds)

                # Warmup shadow engine for this TF (if exists)
                shadow = hub._shadow_engines.get(tf_seconds)
                if shadow and len(df) > 0:
                    shadow.warmup(df)
                    logger.info("Shadow warmup %s: tf=%ss bars=%d initialized=%s",
                                sym, tf_seconds, len(df),
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
                # AM.{ticker} = per-minute aggregates (stocks)
                # A.{ticker}  = per-second aggregates — only subscribed when
                #   a strategy on this symbol actually needs sub-minute data:
                #     * L-type intrabar trigger (level-cross detection), or
                #     * sub-minute primary TF (5s/10s/15s/30s aggregation).
                #   Unconditional A.{ticker} flooded the event loop with
                #   per-second forming-bar work and was the main contributor
                #   to the post-M8.5 alert-latency regression. See
                #   docs/Alert_Recovery_Plan_2026-04-17.md Phase 2.
                # XA.X:{BASE}{QUOTE} = per-minute aggregates (crypto)
                stock_channels = [f"AM.{s}" for s in stock_symbols]
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
                    if has_ltype or has_subminute:
                        stock_channels.append(f"A.{sym}")
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
                                    }

                                    if ev_type in ('A', 'XAS'):
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

                # Skip trade condition filtering for crypto
                # (crypto trades don't have CTA/UTP condition codes)
                if not is_crypto_stream:
                    conditions = getattr(trade, 'conditions', None) or []
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
                except Exception as e:
                    logger.debug("Stale bar flush error: %s", e)

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
        for strat in strategies:
            if strat['id'] not in added:
                continue

            sid = strat['id']
            sym = strat.get('symbol', 'SPY')

            # Ensure hub exists
            if sym not in self.hubs:
                self.hubs[sym] = SymbolHub(sym)

            hub = self.hubs[sym]
            monitor = StrategyMonitor(
                strat, general_packs=self._general_packs)
            hub.add_monitor(monitor)
            self.monitors[sid] = monitor

            # Warmup the new monitor
            tf_seconds = monitor.tf_seconds
            builder = hub.builders.get(tf_seconds)
            if builder and len(builder.history) > 0:
                monitor.warmup(builder.history)
            else:
                # Load historical data for warmup
                try:
                    from data_loader import load_market_data
                    tf_str = SECONDS_TO_TIMEFRAME.get(tf_seconds, '1Min')
                    df = load_market_data(
                        sym, days=7, timeframe=tf_str,
                        feed='sip', session=monitor.session)
                    if df is not None and len(df) > 0:
                        hub.seed_history(tf_seconds, df)
                        monitor.warmup(df)
                except Exception as e:
                    logger.error("Warmup failed for new strategy %d: %s",
                                sid, e)

            logger.info("Hot-reload: added strategy %d (%s / %s)",
                       sid, strat.get('name'), sym)

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
