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
        self.tf_seconds = TIMEFRAME_SECONDS.get(
            strategy.get('timeframe', '1Min'), 60)
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
            req_interp, req_trig, params.get('ema_periods', [8, 21, 50]))
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

        logger.info("StrategyMonitor: %s (%s/%ds) — indicators=%s, "
                     "triggers=%d, interpreters=%d",
                     self.strat_name, self.symbol, self.tf_seconds,
                     req_ind, len(req_trig), len(req_interp))

    def warmup(self, df: pd.DataFrame):
        """Initialize indicator state from historical bars."""
        self.indicators.warmup(df)

    def on_bar_close(self, bar: dict,
                     bar_count: int) -> Tuple[List[dict], dict]:
        """Process a completed bar.

        Returns:
            (signals, audit_data) — signals list and dict for fidelity logging.
        """
        signals = []

        # 1. Update indicators (O(1))
        current = self.indicators.update_bar(bar)
        prev = self.indicators.get_prev_values()

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

        # Use bar_end (bar_start + timeframe - 1s) for bar-close signals
        # so the timestamp reflects when the bar actually closed.
        raw_ts = bar.get('timestamp', '')
        if isinstance(raw_ts, str) and raw_ts:
            bar_end = (datetime.fromisoformat(raw_ts)
                       + timedelta(seconds=self.tf_seconds - 1))
            bar_time = bar_end.isoformat()
        elif isinstance(raw_ts, datetime):
            bar_time = (raw_ts + timedelta(seconds=self.tf_seconds - 1)).isoformat()
        else:
            bar_time = ''

        # 4. Handle pending HM exit at bar open price
        if self.position.state.pending_hm_exit:
            exit_sig = self.position._signal_exit(
                'unconfirmed_hm', bar.get('open', current.get('open', 0)),
                bar_time)
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

    def on_tick(self, price: float, timestamp: str,
                bar_count: int) -> List[dict]:
        """Process a tick for intra-bar detection. Returns list of signals."""
        signals = []

        # Check stop/target on every tick
        exit_sig = self.position.check_exit_tick(price, timestamp)
        if exit_sig:
            exit_sig['atr'] = self.indicators.state.current.get('atr', 0)
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

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.builders: Dict[int, BarBuilder] = {}  # tf_seconds → BarBuilder
        self.monitors: Dict[int, StrategyMonitor] = {}  # strat_id → monitor
        self.tick_count = 0
        self.last_tick_time: Optional[datetime] = None

    def add_monitor(self, monitor: StrategyMonitor):
        self.monitors[monitor.strat_id] = monitor
        # Ensure a BarBuilder exists for this timeframe
        if monitor.tf_seconds not in self.builders:
            self.builders[monitor.tf_seconds] = BarBuilder(monitor.tf_seconds)

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
                # Bar close — run all monitors for this timeframe
                # Collect audit data keyed by strategy id
                bar_audit_positions = {}

                for monitor in self.monitors.values():
                    if monitor.tf_seconds != tf_seconds:
                        continue
                    if not _is_in_session(timestamp, monitor.session):
                        continue

                    signals, audit_data = monitor.on_bar_close(
                        completed, builder._bar_count)
                    if signals and alert_callback:
                        for sig in signals:
                            alert_callback(sig, monitor.strategy, config)

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
                        alert_callback(sig, monitor.strategy, config)


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
    """Check if a UTC timestamp falls within the given trading session (ET)."""
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
        self._ws_confirmed = False  # True only after first trade received
        self._start_time: Optional[str] = None
        self._last_reconcile = 0.0
        self._last_pickle_write = 0.0
        self._last_strategy_refresh = 0.0
        self._subscribed_symbols: List[str] = []

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
            hub = SymbolHub(sym)

            for strat in strats:
                strat_id = strat['id']
                pos_state = saved_positions.get(strat_id)

                monitor = StrategyMonitor(
                    strat, pos_state, general_packs=self._general_packs)
                hub.add_monitor(monitor)
                self.monitors[strat_id] = monitor

            self.hubs[sym] = hub

        # Load warmup data and initialize indicators
        self._warmup_all()

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

        # Run the async event loop
        try:
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
        stream = self._stream_ref
        if stream:
            try:
                stream._should_run = False
            except Exception:
                pass
            # Force-close the WebSocket to unblock the blocking _consume()
            # await. Without this, the engine hangs until ping_timeout (180s).
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        loop.create_task, stream.close())
            except Exception:
                pass

    def _warmup_all(self):
        """Load historical data and initialize all monitors."""
        from data_loader import load_market_data

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
        """Callback when a strategy monitor fires a signal."""
        alert = self.dispatcher.dispatch(signal, strategy, config or self._config)
        if alert:
            self._save_all_positions()

    async def _stream_data(self):
        """Subscribe to Alpaca real-time trade data + run periodic tasks."""
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

        try:
            while self._running:
                # Re-read symbols on each reconnect (may have changed
                # via hot-reload)
                symbols = list(self.hubs.keys())
                self._subscribed_symbols = symbols

                try:
                    stream = StockDataStream(
                        api_key, secret_key, feed=data_feed,
                        websocket_params={
                            'ping_interval': 10,
                            'ping_timeout': 180,
                            'max_queue': 1024,
                        })
                    self._stream_ref = stream
                    self._ws_confirmed = False

                    async def on_trade(trade):
                        nonlocal backoff
                        if not self._running:
                            return
                        if not self._ws_confirmed:
                            self._ws_confirmed = True
                            backoff = 5
                            self._write_status(running=True, connected=True)
                            logger.info("WebSocket confirmed — receiving "
                                        "trades for %d symbols", len(symbols))

                        # Filter out trades with excluded condition codes
                        # (odd lots, average price, out-of-sequence, etc.)
                        # to match Alpaca's historical bar aggregation.
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

                    stream.subscribe_trades(on_trade, *symbols)
                    logger.info("Connecting to Alpaca stream for %d symbols "
                                "(backoff=%ds)…", len(symbols), backoff)

                    # Use patched _run_forever that properly
                    # propagates connection limit errors
                    await _patched_run_forever(stream)

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
                        logger.warning("Connection limit exceeded — "
                                       "another stream may be active. "
                                       "Retrying in %ds", backoff)
                    else:
                        logger.warning("Stream error: %s — retrying in "
                                       "%ds", e, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

                except Exception as e:
                    self._ws_confirmed = False
                    self._write_status(running=True, connected=False)
                    if not self._running:
                        break
                    logger.warning("Stream disconnected: %s — reconnect "
                                   "in %ds", e, backoff)
                    await asyncio.sleep(backoff)
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

                pkl_path = _SCRIPT_DIR / f"live_data_{sym}_{tf_seconds}.pkl"
                tmp_path = str(pkl_path) + ".tmp"
                try:
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(df_out, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, str(pkl_path))
                except Exception:
                    pass

    def _reconcile_bars(self):
        """Fetch recent bars from REST API to correct tick-built bar drift."""
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
