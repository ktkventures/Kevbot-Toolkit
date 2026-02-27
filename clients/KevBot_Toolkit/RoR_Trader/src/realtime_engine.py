"""
Unified Streaming Alert Engine for RoR Trader (Phase 14B).

WebSocket-first architecture that subscribes to Alpaca SIP tick data,
builds OHLCV bars locally from ticks, and evaluates strategy triggers
at bar close.  Reduces alert latency from ~4s (polling) to sub-second.

Architecture:
    Ticks → BarBuilder (per symbol/timeframe) → bar close → full pipeline
          → detect_signals → enrich → save_alert → deliver_alert (thread pool)

The existing alert_monitor.py poller acts as a degraded-mode fallback
when the streaming connection is unavailable — coordinated via the
``streaming_connected`` flag in monitor_status.json.
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Configure file handler so engine logs are visible even from daemon threads
# (Streamlit doesn't capture stdout/stderr from non-main threads).
# Guard against duplicate handlers on module reimport.
if not logger.handlers:
    _log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streaming_engine.log")
    _fh = logging.FileHandler(_log_file, mode='a')
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
    logger.addHandler(_fh)
    logger.setLevel(logging.DEBUG)

# Route alpaca-py WebSocket logger to the same file so connection errors
# (e.g. "connection limit exceeded") appear in streaming_engine.log.
_alpaca_ws_logger = logging.getLogger("alpaca.data.live.websocket")
if not _alpaca_ws_logger.handlers:
    _alpaca_ws_logger.addHandler(_fh)
    _alpaca_ws_logger.setLevel(logging.DEBUG)

# Maximum rolling bars kept per (symbol, timeframe).
# Must be large enough to match the price chart's data_days (default 30) so that
# indicators (EMA-200, VWAP, MACD) warm up identically to the backtest pipeline.
MAX_HISTORY = 25_000

# Window of recent bars passed to generate_trades() for performance.
# The full DataFrame is used for indicator calculation (proper warmup), but
# trade generation only needs recent bars since strategies are intraday.
GENERATE_TRADES_WINDOW = 2000

# Phase 27C: Alpaca REST bar backfill
BACKFILL_INTERVAL = 60   # seconds between backfill cycles
BACKFILL_GRACE_BARS = 2  # don't touch the last N bars (REST API finalization delay)


def _is_in_session(timestamp: datetime, session: str) -> bool:
    """Check if a UTC timestamp falls within the given trading session (ET)."""
    from data_loader import SESSION_HOURS
    import pytz
    import datetime as dt_mod

    et = pytz.timezone("America/New_York")
    et_time = timestamp.astimezone(et).time()
    sh, sm, eh, em = SESSION_HOURS.get(session, (9, 30, 16, 0))
    return dt_mod.time(sh, sm) <= et_time < dt_mod.time(eh, em)

# Engine singleton
_engine_instance: Optional['UnifiedStreamingEngine'] = None
_engine_lock = threading.Lock()

# Local timeframe → seconds mapping.  Duplicated from alert_monitor.py to
# avoid importing that module at module level (it registers signal handlers
# at import time which fails from non-main threads).
TIMEFRAME_SECONDS = {
    "5Sec": 5, "10Sec": 10, "15Sec": 15, "30Sec": 30,
    "1Min": 60, "2Min": 120, "3Min": 180, "5Min": 300,
    "10Min": 600, "15Min": 900, "30Min": 1800,
    "1Hour": 3600, "2Hour": 7200, "4Hour": 14400,
    "1Day": 86400, "1Week": 604800, "1Month": 2592000,
}

# Reverse mapping: tf_seconds → timeframe string for REST API calls (Phase 27C)
SECONDS_TO_TIMEFRAME = {v: k for k, v in TIMEFRAME_SECONDS.items()}


class PartialBar:
    """Represents a partial OHLCV bar being built from tick data."""

    __slots__ = ('open', 'high', 'low', 'close', 'volume', 'bar_start',
                 'bar_duration_seconds', 'tick_count')

    def __init__(self, price: float, timestamp: datetime, bar_duration_seconds: int = 60):
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = 0
        self.bar_start = timestamp
        self.bar_duration_seconds = bar_duration_seconds
        self.tick_count = 0

    def update(self, price: float, volume: int = 1):
        """Update the partial bar with a new tick."""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.tick_count += 1

    def is_complete(self, current_time: datetime) -> bool:
        """Check if the bar period has elapsed."""
        elapsed = (current_time - self.bar_start).total_seconds()
        return elapsed >= self.bar_duration_seconds

    def to_dict(self) -> dict:
        """Convert to OHLCV dict format."""
        return {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.bar_start.isoformat(),
        }


# =============================================================================
# BarBuilder — clock-aligned OHLCV bar aggregation from ticks
# =============================================================================

class BarBuilder:
    """Aggregates raw ticks into clock-aligned OHLCV bars for one timeframe.

    Maintains a rolling ``pd.DataFrame`` of completed bars (capped at
    *MAX_HISTORY*) and a ``PartialBar`` for the in-progress period.
    """

    def __init__(self, tf_seconds: int):
        self.tf_seconds = tf_seconds
        self.history: pd.DataFrame = pd.DataFrame()
        self._partial: Optional[PartialBar] = None
        self._bar_count = 0

    def seed_history(self, df: pd.DataFrame):
        """Pre-load historical bars for indicator warmup."""
        if df is not None and len(df) > 0:
            self.history = df.tail(MAX_HISTORY).copy()
            self._bar_count = len(self.history)
            logger.info("BarBuilder(%ds): seeded %d bars", self.tf_seconds, len(self.history))

    def process_tick(self, price: float, volume: int, timestamp: datetime) -> Optional[dict]:
        """Ingest a tick.  Returns completed bar dict on period close, else None."""
        period_start = self._align_to_period(timestamp)

        if self._partial is None:
            # First tick — start a new bar
            self._partial = PartialBar(price, period_start, self.tf_seconds)
            self._partial.update(price, volume)
            return None

        if period_start > self._partial.bar_start:
            # New period → close the old bar, start a new one
            fill_close = self._partial.close
            old_bar_end = self._partial.bar_start + timedelta(seconds=self.tf_seconds)
            completed = self._close_bar()

            # Fill any gap bars for skipped periods (e.g. WebSocket lag)
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

        # Same period — update in-progress bar
        self._partial.update(price, volume)
        return None

    @property
    def partial_bar(self) -> Optional[PartialBar]:
        return self._partial

    def get_df_with_partial(self) -> pd.DataFrame:
        """Return history + current partial bar as a single DataFrame.

        If no partial bar exists, returns ``history.copy()``.  Used by the
        throttled pipeline evaluator to evaluate indicators against the
        most recent tick data without waiting for bar close.
        """
        if self._partial is None or len(self.history) == 0:
            return self.history.copy()
        bar = self._partial.to_dict()
        ts = pd.Timestamp(bar['timestamp'])
        partial_row = pd.DataFrame(
            [{k: v for k, v in bar.items() if k != 'timestamp'}],
            index=pd.DatetimeIndex([ts], name='timestamp'),
        )
        return pd.concat([self.history, partial_row])

    # -- private --------------------------------------------------------

    def _align_to_period(self, ts: datetime) -> datetime:
        """Snap a timestamp to its clock-aligned bar-period start.

        E.g. for 60-second bars: 09:31:23 → 09:31:00.
        """
        epoch = int(ts.timestamp())
        aligned = epoch - (epoch % self.tf_seconds)
        return datetime.fromtimestamp(aligned, tz=timezone.utc)

    def _close_bar(self) -> dict:
        """Finalize the current partial bar, append to history, return dict."""
        bar = self._partial.to_dict()
        self._append_to_history(bar)
        self._bar_count += 1
        return bar

    def _append_to_history(self, bar_dict: dict):
        """Add a completed bar to the rolling DataFrame (trim to MAX_HISTORY)."""
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


# =============================================================================
# AlertCooldown — prevents duplicate alerts within a time window
# =============================================================================

class AlertCooldown:
    """Prevents duplicate alert firing within a cooldown window.

    Keyed by ``"strategy_id:signal_type"`` — e.g. ``"18:entry_signal"``.
    """

    def __init__(self):
        self._last_fired: Dict[str, float] = {}  # key → epoch

    def can_fire(self, key: str, timestamp: datetime, cooldown_seconds: float = 60.0) -> bool:
        """Return True if this key hasn't fired within the cooldown window."""
        now_epoch = timestamp.timestamp()
        last = self._last_fired.get(key)
        if last is not None and (now_epoch - last) < cooldown_seconds:
            return False
        self._last_fired[key] = now_epoch
        return True

    def reset(self, key: str):
        self._last_fired.pop(key, None)

    def clear_all(self):
        self._last_fired.clear()


# =============================================================================
# TriggerStateTracker — trigger-agnostic False→True transition detection
# =============================================================================

class TriggerStateTracker:
    """Tracks trig_* boolean column states across pipeline evaluations.

    The core insight: comparing boolean column values between evaluations
    is trigger-agnostic.  Works for UT Bot, EMA crosses, VWAP, MACD, or
    any future indicator pack — no per-indicator logic needed.

    A signal fires when a trig_* column transitions False→True on the
    last row.  This is identical to the backtest's transition-based
    detection: ``(direction == 1) & (direction.shift(1) != 1)``.
    """

    def __init__(self):
        # strategy_id → {col_name: bool} — last-known state of each trig column
        self._prev_state: Dict[int, Dict[str, bool]] = {}

    def seed(self, strategy_id: int, df: pd.DataFrame, trigger_cols: List[str]):
        """Initialize state from warmup data (prevents false fires on first eval).

        Args:
            strategy_id: Strategy ID for namespacing.
            df: Enriched DataFrame with trig_* columns.
            trigger_cols: List of column names to track (e.g. ``["trig_utbot_buy"]``).
        """
        if len(df) == 0:
            return
        last_row = df.iloc[-1]
        state = {}
        for col in trigger_cols:
            if col in df.columns:
                state[col] = bool(last_row.get(col, False))
        self._prev_state[strategy_id] = state

    def evaluate(self, strategy_id: int, df: pd.DataFrame,
                 trigger_cols: List[str]) -> List[str]:
        """Check for False→True transitions on the last row.

        Args:
            strategy_id: Strategy ID for namespacing.
            df: Enriched DataFrame with trig_* columns.
            trigger_cols: List of column names to check.

        Returns:
            List of column names that transitioned False→True.
        """
        if len(df) == 0:
            return []
        last_row = df.iloc[-1]
        prev = self._prev_state.get(strategy_id, {})
        fired = []
        new_state = {}
        for col in trigger_cols:
            if col not in df.columns:
                continue
            current = bool(last_row.get(col, False))
            new_state[col] = current
            was_true = prev.get(col, False)
            if current and not was_true:
                fired.append(col)
        self._prev_state[strategy_id] = new_state
        return fired

    def clear(self, strategy_id: Optional[int] = None):
        """Clear state for one or all strategies."""
        if strategy_id is not None:
            self._prev_state.pop(strategy_id, None)
        else:
            self._prev_state.clear()


# =============================================================================
# TradeListTracker — generate_trades() diff approach (Phase 27B)
# =============================================================================

class TradeListTracker:
    """Detect new entries and exits by diffing generate_trades() output.

    On each evaluation, runs generate_trades() on the enriched DataFrame
    and compares against the previous result. Fires alerts when:
    - A new trade appears (entry alert)
    - A previously-open trade now has an exit (exit alert)

    This is the SAME function that produces chart entries/exits,
    guaranteeing chart = alerts = one source of truth.
    """

    def __init__(self):
        self._prev_trades: Dict[int, pd.DataFrame] = {}  # strategy_id → DataFrame of trades
        self._fired_exits: set = set()  # (strategy_id, entry_time_str) tuples — dedup for managed exits

    def evaluate(self, strategy_id: int, df_enriched: pd.DataFrame,
                 strat: dict) -> List[dict]:
        """Run generate_trades() and diff against previous result.

        Args:
            strategy_id: Strategy ID
            df_enriched: Enriched DataFrame from pipeline (with indicators/interpreters/triggers)
            strat: Strategy config dict

        Returns:
            List of signal dicts (entry_signal or exit_signal) for new events
        """
        from alerts import _get_base_trigger_id
        from triggers import generate_trades

        # Resolve trigger names
        entry_trigger = strat.get('entry_trigger') or ''
        if strat.get('entry_trigger_confluence_id'):
            entry_trigger = _get_base_trigger_id(strat['entry_trigger_confluence_id'])

        exit_trigger = strat.get('exit_trigger') or ''
        if strat.get('exit_trigger_confluence_id'):
            exit_trigger = _get_base_trigger_id(strat['exit_trigger_confluence_id'])

        exit_triggers_list = None
        if strat.get('exit_trigger_confluence_ids'):
            exit_triggers_list = [_get_base_trigger_id(t) for t in strat['exit_trigger_confluence_ids'] if t]
        elif strat.get('exit_triggers'):
            exit_triggers_list = [et for et in strat['exit_triggers'] if et]

        # Build confluence set — include BOTH confluence AND general_confluences
        confluence_set = set(strat.get('confluence', [])) | set(strat.get('general_confluences', []))
        confluence_set = confluence_set if confluence_set else None

        # General columns for confluence records (Time of Day, Day of Week, etc.)
        general_cols = [c for c in df_enriched.columns if c.startswith("GP_")]

        # Secondary TF map for MTF confluence
        sec_tf_map = self._get_secondary_tf_map(df_enriched)

        # Slice to recent bars for generate_trades() performance.
        # Indicators are already computed on the full DataFrame (proper warmup),
        # but row-by-row trade generation only needs recent bars.
        df_trades = df_enriched.iloc[-GENERATE_TRADES_WINDOW:] if len(df_enriched) > GENERATE_TRADES_WINDOW else df_enriched

        try:
            new_trades = generate_trades(
                df_trades,
                direction=strat.get('direction', 'LONG'),
                entry_trigger=entry_trigger,
                exit_trigger=exit_trigger,
                exit_triggers=exit_triggers_list,
                confluence_required=confluence_set,
                risk_per_trade=strat.get('risk_per_trade', 100.0),
                stop_atr_mult=strat.get('stop_atr_mult', 1.5),
                stop_config=strat.get('stop_config'),
                target_config=strat.get('target_config'),
                bar_count_exit=strat.get('bar_count_exit'),
                general_columns=general_cols if general_cols else None,
                secondary_tf_map=sec_tf_map if sec_tf_map else None,
            )
        except Exception as e:
            logger.error("TradeListTracker: generate_trades() failed for strategy %d: %s",
                         strategy_id, e)
            return []

        prev_trades = self._prev_trades.get(strategy_id, pd.DataFrame())
        signals = self._diff_trades(strategy_id, prev_trades, new_trades, strat, df_enriched)
        self._prev_trades[strategy_id] = new_trades
        return signals

    def seed(self, strategy_id: int, df_enriched: pd.DataFrame,
             strat: dict) -> pd.DataFrame:
        """Initialize with warmup data to prevent false fires on first eval.

        Returns the trade DataFrame for position state initialization.
        """
        from alerts import _get_base_trigger_id
        from triggers import generate_trades

        entry_trigger = strat.get('entry_trigger') or ''
        if strat.get('entry_trigger_confluence_id'):
            entry_trigger = _get_base_trigger_id(strat['entry_trigger_confluence_id'])

        exit_trigger = strat.get('exit_trigger') or ''
        if strat.get('exit_trigger_confluence_id'):
            exit_trigger = _get_base_trigger_id(strat['exit_trigger_confluence_id'])

        exit_triggers_list = None
        if strat.get('exit_trigger_confluence_ids'):
            exit_triggers_list = [_get_base_trigger_id(t) for t in strat['exit_trigger_confluence_ids'] if t]
        elif strat.get('exit_triggers'):
            exit_triggers_list = [et for et in strat['exit_triggers'] if et]

        confluence_set = set(strat.get('confluence', [])) | set(strat.get('general_confluences', []))
        confluence_set = confluence_set if confluence_set else None

        general_cols = [c for c in df_enriched.columns if c.startswith("GP_")]
        sec_tf_map = self._get_secondary_tf_map(df_enriched)

        # Slice to recent bars for performance (same as evaluate)
        df_trades = df_enriched.iloc[-GENERATE_TRADES_WINDOW:] if len(df_enriched) > GENERATE_TRADES_WINDOW else df_enriched

        try:
            trades = generate_trades(
                df_trades,
                direction=strat.get('direction', 'LONG'),
                entry_trigger=entry_trigger,
                exit_trigger=exit_trigger,
                exit_triggers=exit_triggers_list,
                confluence_required=confluence_set,
                risk_per_trade=strat.get('risk_per_trade', 100.0),
                stop_atr_mult=strat.get('stop_atr_mult', 1.5),
                stop_config=strat.get('stop_config'),
                target_config=strat.get('target_config'),
                bar_count_exit=strat.get('bar_count_exit'),
                general_columns=general_cols if general_cols else None,
                secondary_tf_map=sec_tf_map if sec_tf_map else None,
            )
        except Exception as e:
            logger.error("TradeListTracker: seed failed for strategy %d: %s",
                         strategy_id, e)
            trades = pd.DataFrame()

        self._prev_trades[strategy_id] = trades
        return trades

    def mark_exit_fired(self, strategy_id: int, entry_time_str: str):
        """Mark an exit as already fired (by managed exits) to prevent duplicates."""
        self._fired_exits.add((strategy_id, entry_time_str))

    def clear(self, strategy_id: Optional[int] = None):
        """Clear state for one or all strategies."""
        if strategy_id is not None:
            self._prev_trades.pop(strategy_id, None)
            self._fired_exits = {k for k in self._fired_exits if k[0] != strategy_id}
        else:
            self._prev_trades.clear()
            self._fired_exits.clear()

    def _diff_trades(self, strategy_id: int, prev: pd.DataFrame,
                     current: pd.DataFrame, strat: dict,
                     df_enriched: pd.DataFrame) -> List[dict]:
        """Compare trade lists using stable count+tail approach.

        Instead of diffing ALL trades by entry_time (which is unstable when
        partial bar data shifts indicator values), we track:
        1. Trade count — when it grows, a new entry happened
        2. Last trade open/closed status — when it changes, an exit happened

        This prevents phantom alerts from historical trades shifting entry
        times across evaluations due to partial bar recalculation.
        """
        signals = []

        if len(current) == 0:
            return signals

        prev_count = len(prev)
        curr_count = len(current)

        last_row = df_enriched.iloc[-1]
        atr_val = last_row.get('atr', float(last_row['close']) * 0.01)
        if pd.isna(atr_val) or atr_val <= 0:
            atr_val = float(last_row['close']) * 0.01

        # Case 1: Trade count increased — new entry (fire for the LAST new trade only)
        if curr_count > prev_count:
            # The newest trade is at the end of the list
            new_trade = current.iloc[-1]
            et_str = str(new_trade['entry_time'])

            signals.append({
                'type': 'entry_signal',
                'trigger': new_trade.get('entry_trigger', strat.get('entry_trigger', '')),
                'bar_time': str(new_trade['entry_time']),
                'price': float(new_trade['entry_price']),
                'stop_price': float(new_trade['stop_price']) if pd.notna(new_trade.get('stop_price')) else None,
                'atr': float(atr_val),
            })

            # If the new trade is already closed (entered and exited within
            # the same eval cycle), also fire the exit
            if pd.notna(new_trade.get('exit_time')):
                signals.append({
                    'type': 'exit_signal',
                    'trigger': new_trade.get('exit_reason', 'signal_exit'),
                    'bar_time': str(new_trade['exit_time']),
                    'price': float(new_trade['exit_price']),
                    'stop_price': None,
                    'atr': float(atr_val),
                    'entry_price': float(new_trade['entry_price']),
                    'entry_stop_price': float(new_trade['stop_price']) if pd.notna(new_trade.get('stop_price')) else None,
                })

        # Case 2: Same count, but last trade was open and now has an exit
        elif curr_count == prev_count and prev_count > 0:
            prev_last = prev.iloc[-1]
            curr_last = current.iloc[-1]

            if pd.isna(prev_last.get('exit_time')) and pd.notna(curr_last.get('exit_time')):
                et_str = str(curr_last['entry_time'])

                # Check dedup — managed exits may have already fired this
                if (strategy_id, et_str) in self._fired_exits:
                    self._fired_exits.discard((strategy_id, et_str))
                else:
                    signals.append({
                        'type': 'exit_signal',
                        'trigger': curr_last.get('exit_reason', 'signal_exit'),
                        'bar_time': str(curr_last['exit_time']),
                        'price': float(curr_last['exit_price']),
                        'stop_price': None,
                        'atr': float(atr_val),
                        'entry_price': float(curr_last['entry_price']),
                        'entry_stop_price': float(curr_last['stop_price']) if pd.notna(curr_last.get('stop_price')) else None,
                    })

        return signals

    @staticmethod
    def _get_secondary_tf_map(df: pd.DataFrame) -> Optional[Dict[str, List[str]]]:
        """Extract secondary TF map from DataFrame columns (same as app.py's get_secondary_tf_map)."""
        tf_map: dict = {}
        for col in df.columns:
            if "__" in col:
                parts = col.rsplit("__", 1)
                if len(parts) == 2:
                    tf_label = parts[1]
                    tf_map.setdefault(tf_label, []).append(col)
        return tf_map if tf_map else None


# =============================================================================
# INTRABAR LEVEL MAP — maps trigger bases to indicator columns + cross direction
# =============================================================================

# Keyed by base trigger name (without _ib suffix).  The _ib companion triggers
# strip their suffix to look up the level spec here.
INTRABAR_LEVEL_MAP: Dict[str, Dict[str, str]] = {
    # VWAP
    "vwap_cross_above":          {"column": "vwap",            "cross": "above"},
    "vwap_cross_below":          {"column": "vwap",            "cross": "below"},
    "vwap_enter_upper_extreme":  {"column": "vwap_sd2_upper",  "cross": "above"},
    "vwap_enter_lower_extreme":  {"column": "vwap_sd2_lower",  "cross": "below"},
    # UT Bot
    "utbot_buy":                 {"column": "utbot_stop",      "cross": "above"},
    "utbot_sell":                {"column": "utbot_stop",      "cross": "below"},
    # SuperTrend (user pack)
    "st_bull_flip":              {"column": "st_line",         "cross": "above"},
    "st_bear_flip":              {"column": "st_line",         "cross": "below"},
    # Bollinger Bands (user pack)
    "bb_cross_upper":            {"column": "bb_upper",        "cross": "above"},
    "bb_cross_lower":            {"column": "bb_lower",        "cross": "below"},
    "bb_cross_basis_up":         {"column": "bb_basis",        "cross": "above"},
    "bb_cross_basis_down":       {"column": "bb_basis",        "cross": "below"},
    # SR Channels (user pack)
    "src_resistance_broken":     {"column": "src_nearest_top", "cross": "above"},
    "src_support_broken":        {"column": "src_nearest_bot", "cross": "below"},
    # EMA Price Position — column is dynamic based on group parameters
    "ema_pp_cross_short_up":     {"column": "ema_9",  "cross": "above", "param_key": "short_period"},
    "ema_pp_cross_short_down":   {"column": "ema_9",  "cross": "below", "param_key": "short_period"},
    "ema_pp_cross_mid_up":       {"column": "ema_21", "cross": "above", "param_key": "mid_period"},
    "ema_pp_cross_mid_down":     {"column": "ema_21", "cross": "below", "param_key": "mid_period"},
    # UT Bot (Confirmed) — fill at PREVIOUS bar's trailing stop
    "utbot_v2_buy":              {"column": "utbot_stop_prev", "cross": "above"},
    "utbot_v2_sell":             {"column": "utbot_stop_prev", "cross": "below"},
    # EMA Price Position (Confirmed) — fill at PREVIOUS bar's EMA level
    "ema_pp_v2_cross_short_up":  {"column": "ema_9_prev",  "cross": "above", "param_key": "short_period"},
    "ema_pp_v2_cross_short_down": {"column": "ema_9_prev",  "cross": "below", "param_key": "short_period"},
    "ema_pp_v2_cross_mid_up":    {"column": "ema_21_prev", "cross": "above", "param_key": "mid_period"},
    "ema_pp_v2_cross_mid_down":  {"column": "ema_21_prev", "cross": "below", "param_key": "mid_period"},
}


# =============================================================================
# TriggerLevelCache — O(1) intra-bar crossing detection
# =============================================================================

class TriggerLevelCache:
    """Cache of trigger levels for O(1) intra-bar price comparisons.

    On each bar close, ``update_from_indicators()`` seeds ``_prev_side``
    from the bar-close price.  Between bar closes, ``check()`` detects the
    first crossing relative to that bar-close reference.

    "Once Per Bar" semantics: ``_prev_side`` is updated only at bar close
    and on the first detected crossing — NOT on every tick.  This mirrors
    TradingView's "Once Per Bar" alert mode and the backtest's transition-
    based trigger logic, preventing tick-level noise from causing rapid
    entry/exit cycling that diverges from the chart.
    """

    def __init__(self):
        self._levels: Dict[str, float] = {}             # key → level price
        self._cross_dir: Dict[str, str] = {}             # key → "above"/"below"
        self._prev_side: Dict[str, Optional[str]] = {}   # key → last side of level

    def update_from_indicators(self, strategy_id: int, trigger_base: str,
                               df: pd.DataFrame, group_params: Optional[dict] = None):
        """Extract the trigger level from the last bar's indicators.

        Args:
            strategy_id: The strategy ID for namespacing.
            trigger_base: The base trigger name (with or without ``_ib`` suffix).
            df: OHLCV+indicators DataFrame; level is read from ``iloc[-1]``.
            group_params: Group parameters dict, used to resolve dynamic EMA
                columns (e.g., ``ema_{short_period}``).
        """
        base = trigger_base.removesuffix("_ib")
        spec = INTRABAR_LEVEL_MAP.get(base)
        if spec is None or len(df) == 0:
            return

        # Resolve column name (dynamic for EMA)
        col = spec["column"]
        if "param_key" in spec and group_params:
            period = group_params.get(spec["param_key"])
            if period is not None:
                col = f"ema_{period}"

        last_bar = df.iloc[-1]
        if col not in last_bar.index or pd.isna(last_bar[col]):
            return

        key = f"{strategy_id}:{trigger_base}"
        level = float(last_bar[col])
        close = float(last_bar["close"])

        self._levels[key] = level
        self._cross_dir[key] = spec["cross"]
        # Initialize prev_side from bar close relative to new level
        self._prev_side[key] = "above" if close > level else "below" if close < level else None

    def check(self, key: str, price: float) -> bool:
        """O(1) crossing detection — "Once Per Bar" semantics.

        Returns True the first time the tick price crosses the cached level
        in the required direction since the last bar close.  After a crossing
        is detected, ``_prev_side`` is locked to the new side so no further
        crossings register until the next ``update_from_indicators()`` call
        resets it.  This is trigger-agnostic — works for UT Bot, EMA crosses,
        VWAP, or any future indicator in INTRABAR_LEVEL_MAP.

        Args:
            key: ``"strategy_id:trigger_base"`` string.
            price: Current tick price.
        """
        level = self._levels.get(key)
        if level is None:
            return False

        cross_dir = self._cross_dir.get(key)
        current_side = "above" if price > level else "below" if price < level else None
        prev = self._prev_side.get(key)
        # Do NOT update _prev_side on every tick — only on crossing detection.
        # _prev_side is seeded at bar close by update_from_indicators().

        if prev is None or current_side is None:
            return False

        if cross_dir == "above" and prev == "below" and current_side == "above":
            self._prev_side[key] = current_side  # lock to new side
            return True
        if cross_dir == "below" and prev == "above" and current_side == "below":
            self._prev_side[key] = current_side  # lock to new side
            return True

        return False

    def get_level(self, key: str) -> Optional[float]:
        """Return the cached level price for a key, or None."""
        return self._levels.get(key)

    def clear(self):
        """Clear all cached levels and state."""
        self._levels.clear()
        self._cross_dir.clear()
        self._prev_side.clear()


# =============================================================================
# SymbolHub — per-symbol tick dispatcher + strategy evaluator
# =============================================================================

class SymbolHub:
    """Per-symbol tick dispatcher.

    Manages ``BarBuilder`` instances (one per timeframe) and routes ticks to
    each.  On bar completion, runs the full indicator → interpreter → trigger
    pipeline via ``detect_signals()`` and fires alert callbacks.
    """

    def __init__(self, symbol: str, alert_callback: Optional[Callable] = None):
        self.symbol = symbol
        self.builders: Dict[int, BarBuilder] = {}  # tf_seconds → BarBuilder
        self.strategies: List[dict] = []
        self._alert_callback = alert_callback
        self._cooldown = AlertCooldown()
        self._trigger_tracker = TriggerStateTracker()  # kept for backward compat, unused by Phase 27B
        self._trade_tracker = TradeListTracker()  # Phase 27B: generate_trades() diff approach
        self._position_state: Dict[int, bool] = {}       # strategy_id → True if in position
        self._position_entry: Dict[int, dict] = {}      # strategy_id → entry details when in position
        self._first_bar_closed = False                   # True after first real bar close (not warmup)
        self._last_eval_time: float = 0.0                # monotonic time of last pipeline eval
        self._last_enriched_df: Dict[int, pd.DataFrame] = {}  # tf_seconds → last enriched df
        self.tick_count = 0
        self.last_tick_time: Optional[datetime] = None
        # Phase 27C: Alpaca REST bar backfill
        self._last_backfill_time: float = 0.0
        self._streaming_start_ts: Optional[datetime] = None
        self._backfill_executor: Optional[object] = None  # set by UnifiedStreamingEngine

    def add_timeframe(self, tf_seconds: int, warmup_df: pd.DataFrame):
        """Register a timeframe with historical warmup data."""
        if tf_seconds not in self.builders:
            builder = BarBuilder(tf_seconds)
            builder.seed_history(warmup_df)
            self.builders[tf_seconds] = builder

    def add_strategy(self, strategy: dict):
        """Register a strategy on this symbol."""
        self.strategies.append(strategy)

    def _init_position_state(self):
        """Initialize position state and seed TradeListTracker from warmup history.

        Runs the pipeline once per timeframe, then seeds TradeListTracker
        per strategy (which runs generate_trades() internally). This establishes
        the baseline trade list so the first throttled evaluation doesn't
        fire alerts for pre-existing trades.
        """
        from alerts import _run_pipeline
        from indicators import run_indicators_for_group
        from interpreters import detect_all_triggers as _detect_triggers, run_all_interpreters as _run_interpreters
        from confluence_groups import get_enabled_groups

        enabled_groups = get_enabled_groups()
        _warmup_enriched: Dict[int, pd.DataFrame] = {}  # tf_seconds → enriched df

        for strat in self.strategies:
            strat_id = strat['id']
            strat_tf = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)
            builder = self.builders.get(strat_tf)
            if not builder or len(builder.history) < 10:
                self._position_state[strat_id] = False
                continue

            try:
                # Reuse enriched df across strategies on the same timeframe
                if strat_tf in _warmup_enriched:
                    df_pipeline = _warmup_enriched[strat_tf]
                else:
                    df_pipeline = _run_pipeline(builder.history.copy())
                    for group in enabled_groups:
                        df_pipeline = run_indicators_for_group(df_pipeline, group)
                    # Re-run interpreters + triggers after group indicators
                    # so group-dependent interpreters (UTBOT) get classified
                    df_pipeline = _run_interpreters(df_pipeline)
                    df_pipeline = _detect_triggers(df_pipeline)
                    _warmup_enriched[strat_tf] = df_pipeline

                # Seed TradeListTracker — runs generate_trades() internally
                trades = self._trade_tracker.seed(strat_id, df_pipeline, strat)

                in_position = False
                if len(trades) > 0:
                    last_trade = trades.iloc[-1]
                    if pd.isna(last_trade.get('exit_time')):
                        in_position = True
                self._position_state[strat_id] = in_position
                if in_position and len(trades) > 0:
                    last_trade = trades.iloc[-1]
                    self._position_entry[strat_id] = {
                        'entry_price': float(last_trade.get('entry_price', 0)),
                        'stop_price': float(last_trade['stop_price']) if pd.notna(last_trade.get('stop_price')) else None,
                        'entry_bar_count': builder._bar_count,
                        'direction': strat.get('direction', 'LONG'),
                    }
                logger.info("Position state for %s: %s (%d warmup trades)",
                            strat.get('name'), 'IN_POSITION' if in_position else 'FLAT',
                            len(trades))

            except Exception as e:
                logger.error("Failed to init position state for %s: %s", strat.get('name'), e)
                self._position_state[strat_id] = False

        # Write warmup enriched data to pickle so Live Chart tab is immediately visible
        if _warmup_enriched:
            import pickle
            src_dir = os.path.dirname(os.path.abspath(__file__))
            for tf_sec, df_e in _warmup_enriched.items():
                self._last_enriched_df[tf_sec] = df_e
                pkl_path = os.path.join(src_dir, f"live_data_{self.symbol}_{tf_sec}.pkl")
                tmp_path = pkl_path + ".tmp"
                try:
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(df_e, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, pkl_path)
                    logger.info("Warmup data written to %s (%d bars)", pkl_path, len(df_e))
                except Exception as e:
                    logger.error("Failed to write warmup pickle: %s", e)

    def _init_position_state_for(self, strat: dict):
        """Initialize position state and seed TradeListTracker for a single strategy.

        Used by hot-reload to set up a newly-added strategy without
        re-running the full _init_position_state() loop.
        """
        from alerts import _run_pipeline
        from indicators import run_indicators_for_group
        from interpreters import detect_all_triggers as _detect_triggers, run_all_interpreters as _run_interpreters
        from confluence_groups import get_enabled_groups

        strat_id = strat['id']
        strat_tf = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)
        builder = self.builders.get(strat_tf)
        if not builder or len(builder.history) < 10:
            self._position_state[strat_id] = False
            return

        try:
            enabled_groups = get_enabled_groups()
            df_pipeline = _run_pipeline(builder.history.copy())
            for group in enabled_groups:
                df_pipeline = run_indicators_for_group(df_pipeline, group)
            df_pipeline = _run_interpreters(df_pipeline)
            df_pipeline = _detect_triggers(df_pipeline)

            # Seed TradeListTracker — runs generate_trades() internally
            trades = self._trade_tracker.seed(strat_id, df_pipeline, strat)

            in_position = False
            if len(trades) > 0:
                last_trade = trades.iloc[-1]
                if pd.isna(last_trade.get('exit_time')):
                    in_position = True
            self._position_state[strat_id] = in_position
            if in_position and len(trades) > 0:
                last_trade = trades.iloc[-1]
                self._position_entry[strat_id] = {
                    'entry_price': float(last_trade.get('entry_price', 0)),
                    'stop_price': float(last_trade['stop_price']) if pd.notna(last_trade.get('stop_price')) else None,
                    'entry_bar_count': builder._bar_count,
                    'direction': strat.get('direction', 'LONG'),
                }
            logger.info("Position state for %s: %s (%d warmup trades)",
                        strat.get('name'), 'IN_POSITION' if in_position else 'FLAT',
                        len(trades))

        except Exception as e:
            logger.error("Failed to init position state for %s: %s", strat.get('name'), e)
            self._position_state[strat_id] = False

    def on_tick(self, price: float, volume: int, timestamp: datetime):
        """Route a tick to all bar builders; run pipeline on bar close."""
        self.tick_count += 1
        self.last_tick_time = timestamp
        for tf_seconds, builder in self.builders.items():
            completed = builder.process_tick(price, volume, timestamp)
            if completed is not None:
                self._on_bar_close(tf_seconds, builder, timestamp)
        # Throttled pipeline evaluation (runs every 500ms, not every tick)
        self._evaluate_pipeline_throttled(price, timestamp)
        # Phase 27C: periodically replace streaming bars with canonical REST bars
        self._maybe_backfill()

    # -- private --------------------------------------------------------

    def _add_mtf_columns(self, strat: dict, df_enriched: pd.DataFrame,
                         enriched_dfs: Dict[int, pd.DataFrame]):
        """Add secondary-timeframe interpreter columns to df_enriched for MTF confluence.

        If the strategy's confluence references secondary timeframes (e.g., 5M, 15M),
        forward-fills those interpreter columns into df_enriched so generate_trades()
        can evaluate MTF confluence gating.
        """
        from data_loader import get_required_tfs_from_confluence, get_tf_from_label
        from confluence_groups import get_enabled_interpreter_keys

        all_conf = list(strat.get('confluence', [])) + list(strat.get('general_confluences', []))
        req_labels = get_required_tfs_from_confluence(all_conf)
        if not req_labels:
            return

        interp_keys = get_enabled_interpreter_keys()
        for lbl in req_labels:
            sec_tf_str = get_tf_from_label(lbl)
            sec_tf_sec = TIMEFRAME_SECONDS.get(sec_tf_str)
            if sec_tf_sec and sec_tf_sec in self.builders:
                sec_enriched = enriched_dfs.get(sec_tf_sec)
                if sec_enriched is not None and len(sec_enriched) > 0:
                    for icol in interp_keys:
                        if icol in sec_enriched.columns:
                            scol = f"{icol}__{lbl}"
                            if scol not in df_enriched.columns:
                                df_enriched[scol] = sec_enriched[icol].reindex(
                                    df_enriched.index, method='ffill')

    def _evaluate_pipeline_throttled(self, price: float, timestamp: datetime):
        """Unified pipeline evaluator (Phase 27B) — generate_trades() as single source of truth.

        Runs the full indicator → interpreter → trigger pipeline every 500ms on
        ``history + partial bar``.  Then runs ``generate_trades()`` per strategy via
        ``TradeListTracker`` and diffs against the previous result.  New entries/exits
        in the trade list fire alerts — this is identical to what the chart shows.

        The pipeline runs once per timeframe and is shared across all strategies
        on that timeframe.
        """
        # Throttle: skip if <500ms since last eval
        now = time.monotonic()
        if now - self._last_eval_time < 0.5:
            return
        self._last_eval_time = now

        # Gate: skip until first real bar has closed (need warmup data)
        if not self._first_bar_closed:
            return

        from alerts import (
            _run_pipeline,
            save_alert, enrich_signal_with_portfolio_context, load_alert_config,
        )
        from indicators import run_indicators_for_group
        from interpreters import detect_all_triggers as _detect_triggers, run_all_interpreters as _run_interpreters
        from confluence_groups import get_enabled_groups

        enabled_groups = get_enabled_groups()
        config = load_alert_config()

        # ── Phase 1: Run pipeline once per timeframe ──
        enriched_dfs: Dict[int, pd.DataFrame] = {}
        for tf_seconds, builder in self.builders.items():
            df = builder.get_df_with_partial()
            if len(df) < 10:
                continue
            try:
                df_enriched = _run_pipeline(df)
                for group in enabled_groups:
                    df_enriched = run_indicators_for_group(df_enriched, group)
                # Re-run interpreters + triggers AFTER group indicators so that
                # group-specific interpreters (UTBOT state) and triggers
                # (utbot_buy, etc.) get created.  _run_pipeline() calls these
                # before group indicators add columns like utbot_stop/utbot_direction.
                df_enriched = _run_interpreters(df_enriched)
                df_enriched = _detect_triggers(df_enriched)
                enriched_dfs[tf_seconds] = df_enriched
                self._last_enriched_df[tf_seconds] = df_enriched
            except Exception as e:
                logger.error("Pipeline error for %s/%ds: %s", self.symbol, tf_seconds, e)

        if not enriched_dfs:
            return

        # ── Phase 2: Evaluate each strategy via TradeListTracker (Phase 27B) ──
        # generate_trades() handles ALL entry/exit logic internally:
        # trigger resolution, confluence gating, position tracking, stop/target.
        # We just diff the trade list and fire alerts on new events.
        for strat in self.strategies:
            strat_id = strat['id']
            strat_tf = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)
            df_enriched = enriched_dfs.get(strat_tf)
            if df_enriched is None:
                continue

            # Session gate
            if not _is_in_session(timestamp, strat.get('trading_session', 'RTH')):
                continue

            try:
                # Add MTF secondary TF columns if needed by confluence
                self._add_mtf_columns(strat, df_enriched, enriched_dfs)

                # Run generate_trades() and diff against previous result
                signals = self._trade_tracker.evaluate(strat_id, df_enriched, strat)
                if not signals:
                    continue

                builder = self.builders.get(strat_tf)

                for raw_sig in signals:
                    sig_type = raw_sig['type']

                    # Update position state for managed exits tracking
                    if sig_type == 'entry_signal':
                        self._position_state[strat_id] = True
                        self._position_entry[strat_id] = {
                            'entry_price': raw_sig.get('price', 0),
                            'stop_price': raw_sig.get('stop_price'),
                            'entry_bar_count': builder._bar_count if builder else 0,
                            'direction': strat.get('direction', 'LONG'),
                        }
                    elif sig_type == 'exit_signal':
                        self._position_state[strat_id] = False
                        self._position_entry.pop(strat_id, None)

                    # Build full signal dict
                    sig = {
                        'type': sig_type,
                        'trigger': raw_sig.get('trigger', ''),
                        'bar_time': raw_sig.get('bar_time', ''),
                        'price': raw_sig.get('price', 0),
                        'stop_price': raw_sig.get('stop_price'),
                        'atr': raw_sig.get('atr', 0),
                        'level': 'strategy',
                        'strategy_id': strat_id,
                        'strategy_name': strat.get('name', f"Strategy {strat_id}"),
                        'symbol': strat.get('symbol', '?'),
                        'direction': strat.get('direction', '?'),
                        'risk_per_trade': strat.get('risk_per_trade', 100.0),
                        'timeframe': strat.get('timeframe', '1Min'),
                        'strategy_alerts_visible': True,
                        'source': 'streaming',
                    }

                    # For exit signals, attach entry parameters
                    if sig_type == 'exit_signal':
                        sig['entry_price'] = raw_sig.get('entry_price')
                        sig['entry_stop_price'] = raw_sig.get('entry_stop_price')

                    sig = enrich_signal_with_portfolio_context(sig, strat_id)
                    alert = save_alert(sig)

                    logger.info("TradeList alert: %s for %s (%s) trigger=%s @ %.2f",
                                sig_type, strat.get('name'), self.symbol,
                                raw_sig.get('trigger', ''), raw_sig.get('price', 0))

                    if self._alert_callback:
                        self._alert_callback(alert, config)

            except Exception as e:
                logger.error("Error in throttled eval for %s on %s: %s",
                             strat.get('name', strat['id']), self.symbol, e)

        # ── Phase 3: Write enriched data for live chart ──
        try:
            import pickle
            src_dir = os.path.dirname(os.path.abspath(__file__))
            for tf_seconds, df_e in enriched_dfs.items():
                pkl_path = os.path.join(src_dir, f"live_data_{self.symbol}_{tf_seconds}.pkl")
                tmp_path = pkl_path + ".tmp"
                with open(tmp_path, 'wb') as f:
                    pickle.dump(df_e, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, pkl_path)
        except Exception as e:
            logger.debug("Failed to write live chart data: %s", e)

    def _on_bar_close(self, tf_seconds: int, builder: BarBuilder, timestamp: datetime):
        """Housekeeping on bar close — enables pipeline and evaluates managed exits.

        The full pipeline evaluation is handled by ``_evaluate_pipeline_throttled()``
        which runs every 500ms from ``on_tick()``.  This method only handles:
        1. Setting ``_first_bar_closed`` flag on first close
        2. Running managed exits (stop loss, bar count) which need completed bar data
        """
        if len(builder.history) < 10:
            return

        # Mark warmup complete — throttled evaluator is now safe to run
        if not self._first_bar_closed:
            self._first_bar_closed = True
            self._streaming_start_ts = timestamp
            logger.info("First real bar closed for %s — pipeline evaluator enabled", self.symbol)

        # Evaluate managed exits (stop/bar-count) for strategies in position
        from alerts import save_alert, enrich_signal_with_portfolio_context, load_alert_config

        for strat in self.strategies:
            strat_tf = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)
            if strat_tf != tf_seconds:
                continue
            if not _is_in_session(timestamp, strat.get('trading_session', 'RTH')):
                continue

            strat_id = strat['id']
            in_pos = self._position_state.get(strat_id, False)
            if not in_pos:
                continue

            try:
                exit_sig = self._check_managed_exits(strat, builder, timestamp)
                if exit_sig:
                    # Attach entry details for webhook
                    prior_entry = self._position_entry.get(strat_id)
                    if prior_entry:
                        exit_sig['entry_price'] = prior_entry.get('entry_price')
                        exit_sig['entry_stop_price'] = prior_entry.get('stop_price')

                    exit_sig['level'] = 'strategy'
                    exit_sig['strategy_id'] = strat_id
                    exit_sig['strategy_name'] = strat.get('name', f"Strategy {strat_id}")
                    exit_sig['symbol'] = strat.get('symbol', '?')
                    exit_sig['direction'] = strat.get('direction', '?')
                    exit_sig['risk_per_trade'] = strat.get('risk_per_trade', 100.0)
                    exit_sig['timeframe'] = strat.get('timeframe', '1Min')
                    exit_sig['strategy_alerts_visible'] = True
                    exit_sig['source'] = 'streaming'

                    # Mark exit as fired in TradeListTracker to prevent duplicate
                    # alerts when generate_trades() catches up at next eval
                    prev_trades = self._trade_tracker._prev_trades.get(strat_id, pd.DataFrame())
                    if len(prev_trades) > 0:
                        last_prev = prev_trades.iloc[-1]
                        if pd.isna(last_prev.get('exit_time')):
                            entry_time_str = str(last_prev['entry_time'])
                            self._trade_tracker.mark_exit_fired(strat_id, entry_time_str)

                    self._position_state[strat_id] = False
                    self._position_entry.pop(strat_id, None)

                    exit_sig = enrich_signal_with_portfolio_context(exit_sig, strat_id)
                    config = load_alert_config()
                    alert = save_alert(exit_sig)

                    logger.info("Managed exit alert: %s for %s (%s)",
                                exit_sig.get('trigger'), strat.get('name'), self.symbol)

                    if self._alert_callback:
                        self._alert_callback(alert, config)
            except Exception as e:
                logger.error("Error checking managed exits for %s: %s",
                             strat.get('name', strat['id']), e)

    # ── Phase 27C: Alpaca REST bar backfill ──────────────────────────

    def _maybe_backfill(self):
        """Submit a REST backfill cycle if enough time has elapsed."""
        now = time.monotonic()
        if now - self._last_backfill_time < BACKFILL_INTERVAL:
            return
        if self._streaming_start_ts is None:
            return
        self._last_backfill_time = now
        if self._backfill_executor:
            self._backfill_executor.submit(self._backfill_from_rest)

    def _backfill_from_rest(self):
        """Replace streaming bars with canonical Alpaca REST bars.

        Runs on the ThreadPoolExecutor (non-blocking).  Fetches bars from
        ``_streaming_start_ts`` to ``now - grace`` and overwrites OHLCV in
        ``builder.history`` for matching timestamps.  The next pipeline eval
        cycle (within 500ms) will recalculate indicators with corrected data.
        """
        try:
            from data_loader import load_from_alpaca
        except ImportError:
            return

        start_ts = self._streaming_start_ts

        # Use the most permissive session across all strategies on this symbol
        sessions = [s.get('trading_session', 'RTH') for s in self.strategies]
        session = 'Extended Hours' if 'Extended Hours' in sessions else (
            sessions[0] if sessions else 'RTH')

        for tf_seconds, builder in self.builders.items():
            if len(builder.history) == 0:
                continue

            tf_str = SECONDS_TO_TIMEFRAME.get(tf_seconds)
            if not tf_str:
                continue  # Sub-minute TFs not available via REST — skip

            # Grace period scales with timeframe (2 bars worth)
            grace_seconds = BACKFILL_GRACE_BARS * tf_seconds
            end_ts = datetime.now(timezone.utc) - timedelta(seconds=grace_seconds)

            if start_ts >= end_ts:
                continue  # Not enough time has passed for this TF

            try:
                rest_df = load_from_alpaca(
                    self.symbol,
                    start_date=start_ts,
                    end_date=end_ts,
                    timeframe=tf_str,
                    feed='sip',
                    session=session,
                )
            except Exception as e:
                logger.warning("Backfill fetch failed for %s/%s: %s",
                               self.symbol, tf_str, e)
                continue

            if rest_df is None or len(rest_df) == 0:
                continue

            # Replace OHLCV for matching timestamps
            common_idx = builder.history.index.intersection(rest_df.index)
            if len(common_idx) > 0:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in rest_df.columns:
                        builder.history.loc[common_idx, col] = rest_df.loc[common_idx, col]

            logger.info("Backfill %s/%ds: replaced %d bars (%s to %s)",
                        self.symbol, tf_seconds, len(common_idx),
                        start_ts.strftime('%H:%M'), end_ts.strftime('%H:%M'))

    def _check_managed_exits(self, strat: dict, builder: BarBuilder,
                             timestamp: datetime) -> Optional[dict]:
        """Evaluate stop-loss and bar-count exits for positions tracked by the engine.

        Called when _position_state says we're in position but detect_signals()
        returned no signals (typically because the strategy has no explicit exit
        triggers — exits are managed by stop/target/bar-count instead).

        Returns an exit signal dict if exit conditions are met, else None.
        """
        strat_id = strat['id']
        entry = self._position_entry.get(strat_id)
        if not entry:
            return None

        last_bar = builder.history.iloc[-1]
        direction = entry.get('direction', 'LONG')
        stop_price = entry.get('stop_price')
        close_price = float(last_bar['close'])

        atr_val = last_bar.get('atr', close_price * 0.01)
        if pd.isna(atr_val) or atr_val <= 0:
            atr_val = close_price * 0.01

        bar_time = str(last_bar.name) if hasattr(last_bar, 'name') else timestamp.isoformat()

        # Check stop-loss
        if stop_price is not None:
            stopped = False
            open_price = float(last_bar['open'])
            if direction == 'LONG' and float(last_bar['low']) <= float(stop_price):
                stopped = True
                fill_price = min(float(stop_price), open_price)
            elif direction == 'SHORT' and float(last_bar['high']) >= float(stop_price):
                stopped = True
                fill_price = max(float(stop_price), open_price)
            if stopped:
                logger.info("Managed exit: stop hit for %s (%s) fill=%.2f",
                            strat.get('name'), self.symbol, fill_price)
                return {
                    "type": "exit_signal",
                    "trigger": "stop_loss",
                    "confluence_met": [],
                    "bar_time": bar_time,
                    "price": fill_price,
                    "stop_price": None,
                    "atr": float(atr_val),
                    "entry_price": entry.get('entry_price'),
                    "entry_stop_price": entry.get('stop_price'),
                }

        # Check bar-count exit
        bar_count_exit = strat.get('bar_count_exit')
        if bar_count_exit:
            bars_held = builder._bar_count - entry.get('entry_bar_count', builder._bar_count)
            if bars_held >= int(bar_count_exit):
                logger.info("Managed exit: bar count (%d >= %d) for %s (%s)",
                            bars_held, int(bar_count_exit), strat.get('name'), self.symbol)
                return {
                    "type": "exit_signal",
                    "trigger": "bar_count_exit",
                    "confluence_met": [],
                    "bar_time": bar_time,
                    "price": close_price,
                    "stop_price": None,
                    "atr": float(atr_val),
                    "entry_price": entry.get('entry_price'),
                    "entry_stop_price": entry.get('stop_price'),
                }

        return None


# =============================================================================
# UnifiedStreamingEngine — lifecycle manager
# =============================================================================

class UnifiedStreamingEngine:
    """WebSocket-first streaming alert engine.

    Subscribes to Alpaca SIP tick data, builds bars via SymbolHubs,
    evaluates triggers at bar close, and dispatches webhook delivery
    via a thread pool.

    Coordinates with ``alert_monitor.py`` via the ``streaming_connected``
    flag in ``monitor_status.json`` — when *True* the poller sleeps.
    """

    def __init__(self):
        self.hubs: Dict[str, SymbolHub] = {}
        self.strategies: List[dict] = []
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stream_ref = None  # holds StockDataStream for clean shutdown
        self._start_time: Optional[str] = None
        self._deliver_alert_fn: Optional[Callable] = None

    # -- public ---------------------------------------------------------

    def start(self, strategies: list, alert_config: dict):
        """Start the streaming engine.

        1. Group strategies by symbol
        2. Create SymbolHub per symbol with warmup data
        3. Start thread pool for webhook delivery
        4. Spawn daemon thread with asyncio WebSocket loop
        """
        if self._running:
            logger.warning("Streaming engine already running")
            return

        # Set _running immediately to prevent concurrent start() calls.
        # Streamlit reruns during warmup can trigger additional start() calls;
        # the flag must be True BEFORE the slow warmup phase begins.
        self._running = True

        from alerts import compute_signal_detection_bars
        from data_loader import load_latest_bars

        # Pre-import deliver_alert from alert_monitor.  This import must
        # happen in the calling thread because alert_monitor registers signal
        # handlers at module level (requires main-ish thread context).
        try:
            from alert_monitor import deliver_alert
            self._deliver_alert_fn = deliver_alert
        except Exception as exc:
            logger.warning("Could not import deliver_alert: %s — "
                           "webhook delivery disabled", exc)
            self._deliver_alert_fn = None

        self.strategies = strategies
        if not strategies:
            logger.info("No strategies to monitor — engine not started")
            self._running = False
            return

        # Group strategies by symbol
        by_symbol: Dict[str, List[dict]] = {}
        for strat in strategies:
            sym = strat.get('symbol', 'SPY')
            by_symbol.setdefault(sym, []).append(strat)

        # Create SymbolHubs with warmup data
        self.hubs = {}
        for sym, strats in by_symbol.items():
            hub = SymbolHub(sym, alert_callback=self._queue_alert_delivery)
            for strat in strats:
                hub.add_strategy(strat)

            # Register unique (timeframe, session) combos for this symbol
            # Includes both primary TFs and any secondary TFs required by MTF confluence
            from data_loader import get_required_tfs_from_confluence, get_tf_from_label
            seen_tf: set = set()
            for strat in strats:
                tf_str = strat.get('timeframe', '1Min')
                tf_sec = TIMEFRAME_SECONDS.get(tf_str, 60)
                session = strat.get('trading_session', 'RTH')

                # Collect primary + required secondary timeframes
                tfs_to_register = [(tf_str, tf_sec)]
                req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
                for lbl in req_labels:
                    sec_tf_str = get_tf_from_label(lbl)
                    sec_tf_sec = TIMEFRAME_SECONDS.get(sec_tf_str)
                    if sec_tf_sec is not None:
                        tfs_to_register.append((sec_tf_str, sec_tf_sec))

                for reg_tf_str, reg_tf_sec in tfs_to_register:
                    tf_key = (reg_tf_sec, session)
                    if tf_key in seen_tf:
                        continue
                    seen_tf.add(tf_key)

                    # Match price chart warmup depth: use data_days from strategy
                    # config so indicators initialize identically to the backtest.
                    from data_loader import load_market_data as _load_market_data
                    data_days = strat.get('data_days', 30)
                    try:
                        warmup_df = _load_market_data(
                            sym, days=data_days, timeframe=reg_tf_str,
                            seed=strat.get('data_seed', 42), feed='sip',
                            session=session,
                        )
                    except Exception as e:
                        logger.error("Warmup load failed for %s/%s: %s", sym, reg_tf_str, e)
                        warmup_df = pd.DataFrame()
                    hub.add_timeframe(reg_tf_sec, warmup_df)

            # Initialize position state and trigger cache from warmup data
            hub._init_position_state()

            self.hubs[sym] = hub

        # Thread pool for non-blocking webhook delivery
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="alert-delivery")

        # Wire executor to each hub for REST backfill (Phase 27C)
        for hub in self.hubs.values():
            hub._backfill_executor = self._executor

        self._start_time = datetime.now(timezone.utc).isoformat()

        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="streaming-engine")
        self._thread.start()

        logger.info("Streaming engine started: %d strategies, %d symbols",
                     len(strategies), len(self.hubs))

    def stop(self):
        """Stop the streaming engine gracefully."""
        self._running = False
        self._set_streaming_status(False)

        # Close WebSocket stream to unblock _run_forever()
        if self._stream_ref is not None:
            try:
                self._stream_ref.close()
            except Exception:
                pass
            self._stream_ref = None

        # Wait for the streaming thread to exit (avoids stale threads on restart)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

        logger.info("Streaming engine stopped")

    def get_status(self) -> dict:
        """Return engine status dict for UI display."""
        total_ticks = sum(h.tick_count for h in self.hubs.values())
        last_tick = None
        for h in self.hubs.values():
            if h.last_tick_time and (last_tick is None or h.last_tick_time > last_tick):
                last_tick = h.last_tick_time
        return {
            'running': self._running,
            'connected': self._connected,
            'symbols': sorted(self.hubs.keys()),
            'strategy_count': len(self.strategies),
            'tick_count': total_ticks,
            'last_tick_time': last_tick.isoformat() if last_tick else None,
            'started_at': self._start_time,
        }

    STRATEGY_REFRESH_INTERVAL = 300  # 5 minutes

    def _refresh_strategies(self):
        """Hot-reload strategies from disk without disrupting the WebSocket.

        Handles new strategies on already-subscribed symbols instantly.
        New symbols require a manual restart (logged as warning).
        """
        try:
            from alert_monitor import get_monitored_strategies
            from alerts import load_alert_config, compute_signal_detection_bars
            from data_loader import load_latest_bars

            config = load_alert_config()
            fresh = get_monitored_strategies(config)
        except Exception as e:
            logger.error("Strategy refresh failed to load: %s", e)
            return

        current_ids = {s['id'] for s in self.strategies}
        fresh_ids = {s['id'] for s in fresh}
        added_ids = fresh_ids - current_ids
        removed_ids = current_ids - fresh_ids

        if not added_ids and not removed_ids:
            return

        # Remove strategies that are no longer monitored
        for rid in removed_ids:
            for hub in self.hubs.values():
                hub.strategies = [s for s in hub.strategies if s['id'] != rid]
            self.strategies = [s for s in self.strategies if s['id'] != rid]
            logger.info("Strategy refresh: removed strategy %d", rid)

        # Add new strategies
        new_symbols_needed = set()
        for strat in fresh:
            if strat['id'] not in added_ids:
                continue
            sym = strat.get('symbol', 'SPY')
            hub = self.hubs.get(sym)
            if hub is None:
                new_symbols_needed.add(sym)
                continue

            # Add to existing hub
            hub.add_strategy(strat)
            self.strategies.append(strat)

            # Register timeframes if needed
            from data_loader import get_required_tfs_from_confluence, get_tf_from_label
            tf_str = strat.get('timeframe', '1Min')
            tf_sec = TIMEFRAME_SECONDS.get(tf_str, 60)
            session = strat.get('trading_session', 'RTH')

            tfs_to_check = [(tf_str, tf_sec)]
            req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
            for lbl in req_labels:
                sec_tf_str = get_tf_from_label(lbl)
                sec_tf_sec = TIMEFRAME_SECONDS.get(sec_tf_str)
                if sec_tf_sec is not None:
                    tfs_to_check.append((sec_tf_str, sec_tf_sec))

            for reg_tf_str, reg_tf_sec in tfs_to_check:
                if reg_tf_sec not in hub.builders:
                    from data_loader import load_market_data as _load_market_data
                    data_days = strat.get('data_days', 30)
                    try:
                        warmup_df = _load_market_data(
                            sym, days=data_days, timeframe=reg_tf_str,
                            seed=strat.get('data_seed', 42), feed='sip',
                            session=session,
                        )
                    except Exception as e:
                        logger.error("Warmup load for new strategy %s/%s: %s", sym, reg_tf_str, e)
                        warmup_df = pd.DataFrame()
                    hub.add_timeframe(reg_tf_sec, warmup_df)

            # Initialize position state for the new strategy
            hub._init_position_state_for(strat)

            logger.info("Strategy refresh: added '%s' (%s) to live monitoring",
                        strat.get('name'), sym)

        if new_symbols_needed:
            logger.warning("Strategy refresh: new symbols %s require monitor restart",
                           new_symbols_needed)

    # -- private --------------------------------------------------------

    def _run_loop(self):
        """Thread entry point — run the async WebSocket loop."""
        try:
            asyncio.run(self._stream_data())
        except Exception as e:
            logger.error("Streaming engine loop error: %s", e)
        finally:
            self._running = False
            self._set_streaming_status(False)
            logger.info("Streaming engine loop exited")

    async def _stream_data(self):
        """Subscribe to Alpaca SIP real-time data with exponential backoff."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            logger.error("Alpaca API keys not configured — cannot start stream")
            return

        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.enums import DataFeed
        except ImportError:
            logger.error("alpaca-py not installed.  pip install alpaca-py")
            return

        symbols = list(self.hubs.keys())
        backoff = 5  # seconds — doubles on each failure up to max_backoff
        max_backoff = 60

        while self._running:
            try:
                stream = StockDataStream(api_key, secret_key, feed=DataFeed.SIP)
                self._stream_ref = stream
                _ws_confirmed = False
                _last_refresh = time.monotonic()

                async def on_trade(trade):
                    nonlocal _ws_confirmed, backoff, _last_refresh
                    if not self._running:
                        return
                    if not _ws_confirmed:
                        _ws_confirmed = True
                        self._connected = True
                        self._set_streaming_status(True)
                        backoff = 5
                        logger.info("WebSocket confirmed — receiving trades for %d symbols", len(symbols))

                    # Periodic strategy hot-reload
                    now = time.monotonic()
                    if now - _last_refresh >= self.STRATEGY_REFRESH_INTERVAL:
                        _last_refresh = now
                        self._refresh_strategies()

                    hub = self.hubs.get(trade.symbol)
                    if hub:
                        hub.on_tick(
                            price=float(trade.price),
                            volume=int(trade.size) if hasattr(trade, 'size') else 1,
                            timestamp=trade.timestamp,
                        )

                stream.subscribe_trades(on_trade, *symbols)

                logger.info("Connecting to Alpaca stream for %d symbols …", len(symbols))

                await stream._run_forever()

            except Exception as e:
                self._connected = False
                self._set_streaming_status(False)
                if not self._running:
                    break
                logger.warning("Stream disconnected: %s — reconnecting in %ds",
                               e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        self._connected = False
        self._set_streaming_status(False)

    def _queue_alert_delivery(self, alert: dict, config: dict):
        """Submit alert delivery to the thread pool (non-blocking)."""
        if self._executor is None or self._deliver_alert_fn is None:
            return
        try:
            self._executor.submit(self._deliver_alert_fn, alert, config)
        except Exception as e:
            logger.error("Failed to queue alert delivery: %s", e)

    def _set_streaming_status(self, connected: bool):
        """Update ``streaming_connected`` in monitor_status.json."""
        try:
            from alerts import load_monitor_status, save_monitor_status
            status = load_monitor_status()
            status['streaming_connected'] = connected
            save_monitor_status(status)
        except Exception as e:
            logger.debug("Could not update streaming status: %s", e)

    @staticmethod
    def _has_intrabar_triggers(strategy: dict) -> bool:
        """Check if strategy uses any [I] (intra_bar) execution type triggers."""
        try:
            from confluence_groups import get_all_triggers
            all_triggers = get_all_triggers()
            trigger_ids = [strategy.get('entry_trigger_confluence_id', ''),
                           strategy.get('exit_trigger_confluence_id', '')]
            trigger_ids.extend(strategy.get('exit_trigger_confluence_ids', []))
            for tid in trigger_ids:
                if tid and tid in all_triggers:
                    if getattr(all_triggers[tid], 'execution', 'bar_close') == 'intra_bar':
                        return True
        except Exception:
            pass
        return False


# =============================================================================
# MODULE-LEVEL CONVENIENCE API
# =============================================================================

def get_engine() -> UnifiedStreamingEngine:
    """Get or create the singleton engine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = UnifiedStreamingEngine()
    return _engine_instance


def start_engine(strategies: list, alert_config: dict):
    """Start the streaming engine (convenience wrapper)."""
    engine = get_engine()
    if engine._running:
        logger.info("Engine already running")
        return engine
    engine.start(strategies, alert_config)
    return engine


def stop_engine():
    """Stop the streaming engine (convenience wrapper)."""
    get_engine().stop()


def engine_status() -> dict:
    """Get engine status (convenience wrapper)."""
    return get_engine().get_status()
