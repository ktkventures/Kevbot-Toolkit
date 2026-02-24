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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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

# Maximum rolling bars kept per (symbol, timeframe) — covers EMA-200 warmup
MAX_HISTORY = 500


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
            completed = self._close_bar()
            self._partial = PartialBar(price, period_start, self.tf_seconds)
            self._partial.update(price, volume)
            return completed

        # Same period — update in-progress bar
        self._partial.update(price, volume)
        return None

    @property
    def partial_bar(self) -> Optional[PartialBar]:
        return self._partial

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

    On each bar close, ``update_from_indicators()`` extracts the latest
    indicator value for each active intra-bar trigger.  Between bar closes,
    ``check()`` compares tick prices against cached levels with crossing-
    direction tracking.
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
        """O(1) crossing detection.  Returns True on first crossing in the required direction.

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
        self._prev_side[key] = current_side

        if prev is None or current_side is None:
            return False

        if cross_dir == "above" and prev == "below" and current_side == "above":
            return True
        if cross_dir == "below" and prev == "above" and current_side == "below":
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

    def __init__(self, symbol: str, alert_callback: Optional[Callable] = None,
                 trigger_cache: Optional[TriggerLevelCache] = None):
        self.symbol = symbol
        self.builders: Dict[int, BarBuilder] = {}  # tf_seconds → BarBuilder
        self.strategies: List[dict] = []
        self._alert_callback = alert_callback
        self._cooldown = AlertCooldown()
        self._trigger_cache = trigger_cache or TriggerLevelCache()
        self._confluence_met: Dict[int, bool] = {}       # strategy_id → confluence ok?
        self._intrabar_fired: Dict[str, set] = {}        # "strat_id:tf" → {trigger_bases}
        self._ib_cooldown = AlertCooldown()               # separate cooldown for intra-bar
        self._position_state: Dict[int, bool] = {}       # strategy_id → True if in position
        self._position_entry: Dict[int, dict] = {}      # strategy_id → entry details when in position
        self._first_bar_closed = False                   # True after first real bar close (not warmup)
        self.tick_count = 0
        self.last_tick_time: Optional[datetime] = None

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
        """Initialize position state and trigger cache from warmup history.

        Runs the pipeline once per strategy to determine position state
        (via generate_trades) and seed the trigger cache for intra-bar triggers.
        """
        from alerts import _run_pipeline, _get_base_trigger_id
        from indicators import run_indicators_for_group
        from confluence_groups import get_enabled_groups
        from triggers import generate_trades

        enabled_groups = get_enabled_groups()

        for strat in self.strategies:
            strat_id = strat['id']
            strat_tf = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)
            builder = self.builders.get(strat_tf)
            if not builder or len(builder.history) < 10:
                self._position_state[strat_id] = False
                continue

            try:
                # Run pipeline once and reuse for both position state + trigger cache
                df_pipeline = _run_pipeline(builder.history.copy())
                for group in enabled_groups:
                    df_pipeline = run_indicators_for_group(df_pipeline, group)

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

                confluence_set = set(strat.get('confluence', [])) if strat.get('confluence') else None
                trades = generate_trades(
                    df_pipeline,
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
                )
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
                logger.info("Position state for %s: %s",
                            strat.get('name'), 'IN_POSITION' if in_position else 'FLAT')

                # Seed trigger cache (reuse same df_pipeline)
                ib_bases = self._get_intrabar_trigger_bases(strat)
                for trigger_base in ib_bases:
                    group_params = self._get_group_params_for_trigger(trigger_base)
                    self._trigger_cache.update_from_indicators(
                        strat_id, trigger_base, df_pipeline, group_params)

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
        # Intra-bar trigger evaluation on every tick
        self._check_intrabar_triggers(price, timestamp)

    # -- private --------------------------------------------------------

    def _get_intrabar_trigger_bases(self, strat: dict) -> List[str]:
        """Return trigger names for intra-bar triggers in this strategy.

        Uses the strategy's own trigger fields (e.g. ``utbot_v2_buy_ib``)
        which map directly to INTRABAR_LEVEL_MAP keys after removing ``_ib``.
        """
        bases = []
        entry = strat.get('entry_trigger') or ''
        if entry.endswith('_ib'):
            bases.append(entry)
        for et in (strat.get('exit_triggers') or []):
            if et and et.endswith('_ib'):
                bases.append(et)
        exit_t = strat.get('exit_trigger') or ''
        if exit_t and exit_t.endswith('_ib'):
            bases.append(exit_t)
        return bases

    def _get_group_params_for_trigger(self, trigger_base: str) -> Optional[dict]:
        """Look up the group parameters for a trigger base (needed for EMA dynamic columns)."""
        try:
            from confluence_groups import get_enabled_groups, get_group_triggers
            for group in get_enabled_groups():
                for trig in get_group_triggers(group):
                    # Match either short base ("buy_ib") or full name ("utbot_v2_buy_ib")
                    full_name = f"{group.base_template}_{trig.base_trigger}"
                    if trig.base_trigger == trigger_base or full_name == trigger_base:
                        return group.parameters
        except Exception:
            pass
        return None

    def _check_intrabar_triggers(self, price: float, timestamp: datetime):
        """Evaluate intra-bar triggers against cached levels on every tick."""
        # Skip until the first real bar close — warmup-seeded levels have a stale
        # prev_side that causes false crossings on the first tick.
        if not self._first_bar_closed:
            return

        from alerts import save_alert, enrich_signal_with_portfolio_context, load_alert_config
        from triggers import calculate_stop_price

        for strat in self.strategies:
            strat_id = strat['id']
            strat_tf = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)

            # Session gate
            if not _is_in_session(timestamp, strat.get('trading_session', 'RTH')):
                continue

            ib_bases = self._get_intrabar_trigger_bases(strat)
            if not ib_bases:
                continue

            # Confluence gate: only fire if confluence was met at last bar close
            if not self._confluence_met.get(strat_id, True):
                continue

            dedup_key = f"{strat_id}:{strat_tf}"

            for trigger_base in ib_bases:
                cache_key = f"{strat_id}:{trigger_base}"

                if self._trigger_cache.check(cache_key, price):
                    # Check dedup — don't re-fire same trigger this bar
                    fired_set = self._intrabar_fired.get(dedup_key, set())
                    base_no_ib = trigger_base.removesuffix("_ib")
                    if base_no_ib in fired_set:
                        continue

                    # Cooldown — min(tf_seconds / 2, 30) seconds
                    cooldown_secs = min(strat_tf / 2.0, 30.0)
                    ib_cooldown_key = f"ib:{strat_id}:{trigger_base}"
                    if not self._ib_cooldown.can_fire(ib_cooldown_key, timestamp,
                                                      cooldown_seconds=cooldown_secs):
                        continue

                    # Determine signal type based on trigger direction
                    entry_trigger = strat.get('entry_trigger') or ''
                    sig_type = 'entry_signal' if trigger_base == entry_trigger else 'exit_signal'

                    # Position gate: only fire entry when flat, exit when in position
                    in_pos = self._position_state.get(strat_id, False)
                    if sig_type == 'entry_signal' and in_pos:
                        continue
                    if sig_type == 'exit_signal' and not in_pos:
                        continue

                    # Mark as fired for dedup
                    if dedup_key not in self._intrabar_fired:
                        self._intrabar_fired[dedup_key] = set()
                    self._intrabar_fired[dedup_key].add(base_no_ib)

                    # Capture entry details BEFORE clearing (needed for exit quantity)
                    prior_entry = self._position_entry.get(strat_id)

                    # Update position state and entry tracking
                    if sig_type == 'entry_signal':
                        self._position_state[strat_id] = True
                    elif sig_type == 'exit_signal':
                        self._position_state[strat_id] = False
                        self._position_entry.pop(strat_id, None)

                    level = self._trigger_cache.get_level(cache_key)

                    # Calculate stop price and ATR for quantity derivation
                    stop_price = None
                    atr_val = None
                    if sig_type == 'entry_signal':
                        builder = self.builders.get(strat_tf)
                        if builder and len(builder.history) > 0:
                            last_bar = builder.history.iloc[-1]
                            atr_val = last_bar.get('atr', price * 0.01)
                            if pd.isna(atr_val) or atr_val <= 0:
                                atr_val = price * 0.01
                            direction = strat.get('direction', 'LONG')
                            stop_cfg = strat.get('stop_config') or {
                                "method": "atr",
                                "atr_mult": strat.get("stop_atr_mult", 1.5),
                            }
                            try:
                                stop_price = calculate_stop_price(
                                    price, direction, last_bar,
                                    builder.history, len(builder.history) - 1,
                                    stop_cfg,
                                )
                            except Exception:
                                pass

                    # Track entry details for managed exit monitoring
                    if sig_type == 'entry_signal':
                        entry_builder = self.builders.get(strat_tf)
                        self._position_entry[strat_id] = {
                            'entry_price': price,
                            'stop_price': float(stop_price) if stop_price is not None else None,
                            'entry_bar_count': entry_builder._bar_count if entry_builder else 0,
                            'direction': strat.get('direction', 'LONG'),
                        }

                    sig = {
                        'type': sig_type,
                        'trigger': trigger_base,
                        'bar_time': timestamp.isoformat(),
                        'price': price,
                        'stop_price': float(stop_price) if stop_price is not None else None,
                        'atr': float(atr_val) if atr_val is not None else None,
                        'level': 'strategy',
                        'strategy_id': strat_id,
                        'strategy_name': strat.get('name', f"Strategy {strat_id}"),
                        'symbol': strat.get('symbol', '?'),
                        'direction': strat.get('direction', '?'),
                        'risk_per_trade': strat.get('risk_per_trade', 100.0),
                        'timeframe': strat.get('timeframe', '1Min'),
                        'strategy_alerts_visible': True,
                        'source': 'intra_bar',
                        'intrabar_level': float(level) if level is not None else None,
                    }

                    # For exit signals, attach entry parameters so webhook
                    # quantity matches the entry size
                    if sig_type == 'exit_signal' and prior_entry:
                        sig['entry_price'] = prior_entry.get('entry_price')
                        sig['entry_stop_price'] = prior_entry.get('stop_price')

                    sig = enrich_signal_with_portfolio_context(sig, strat_id)
                    config = load_alert_config()
                    alert = save_alert(sig)

                    logger.info("Intra-bar alert: %s for %s (%s) @ %.2f (level=%.2f)",
                                sig_type, strat.get('name'), self.symbol, price,
                                level if level else 0)

                    if self._alert_callback:
                        self._alert_callback(alert, config)

    def _on_bar_close(self, tf_seconds: int, builder: BarBuilder, timestamp: datetime):
        """Run full pipeline and evaluate signals for all matching strategies.

        Uses _position_state as the single source of truth for position tracking,
        rather than re-running generate_trades() each bar (which can't see
        intra-bar entries or streaming-only position changes).
        """
        from alerts import (
            detect_signals,
            save_alert,
            enrich_signal_with_portfolio_context,
            load_alert_config,
            _run_pipeline,
        )
        from data_loader import get_required_tfs_from_confluence, get_tf_from_label, get_tf_label
        from indicators import run_indicators_for_group
        from confluence_groups import get_enabled_groups

        df = builder.history
        if len(df) < 10:
            return  # Not enough data for reliable signals

        # Mark warmup complete — intra-bar triggers are now safe to fire
        if not self._first_bar_closed:
            self._first_bar_closed = True
            logger.info("First real bar closed for %s — intra-bar triggers enabled", self.symbol)

        config = load_alert_config()

        # Build enriched DataFrame with group-specific indicators (e.g. utbot_stop_prev).
        # These must be present BEFORE detect_signals → _run_pipeline → detect_all_triggers
        # so that group trigger columns (trig_utbot_v2_buy, etc.) get created.
        enabled_groups = get_enabled_groups()
        df_enriched = _run_pipeline(df.copy())
        for group in enabled_groups:
            df_enriched = run_indicators_for_group(df_enriched, group)

        for strat in self.strategies:
            strat_tf = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)
            if strat_tf != tf_seconds:
                continue

            # Session gate: skip evaluation if bar is outside strategy's session
            if not _is_in_session(timestamp, strat.get('trading_session', 'RTH')):
                continue

            strat_id = strat['id']
            dedup_key = f"{strat_id}:{tf_seconds}"

            try:
                # Gather secondary TF histories for MTF confluence
                secondary_tf_dfs = None
                req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
                if req_labels:
                    secondary_tf_dfs = {}
                    for lbl in req_labels:
                        sec_tf_str = get_tf_from_label(lbl)
                        sec_tf_sec = TIMEFRAME_SECONDS.get(sec_tf_str)
                        if sec_tf_sec and sec_tf_sec in self.builders:
                            sec_history = self.builders[sec_tf_sec].history
                            if len(sec_history) > 0:
                                secondary_tf_dfs[lbl] = sec_history.copy()

                # Use _position_state as authoritative position tracker
                in_pos = self._position_state.get(strat_id, False)
                signals = detect_signals(strat, df=df_enriched.copy(),
                                         secondary_tf_dfs=secondary_tf_dfs,
                                         override_in_position=in_pos)

                # Track confluence state for intra-bar gating
                confluence_set = set(strat.get('confluence', []))
                if confluence_set:
                    from interpreters import get_confluence_records
                    from confluence_groups import get_enabled_interpreter_keys
                    interp_keys = get_enabled_interpreter_keys()
                    last_bar = df.iloc[-1]
                    current_conf = get_confluence_records(last_bar, "1M", interp_keys)
                    self._confluence_met[strat_id] = confluence_set.issubset(current_conf)
                else:
                    self._confluence_met[strat_id] = True

                # Update trigger cache for intra-bar triggers (reuse enriched df)
                ib_bases = self._get_intrabar_trigger_bases(strat)
                if ib_bases:
                    for trigger_base in ib_bases:
                        group_params = self._get_group_params_for_trigger(trigger_base)
                        self._trigger_cache.update_from_indicators(
                            strat_id, trigger_base, df_enriched, group_params)

                # Evaluate stop/bar-count exits for strategies without
                # explicit exit triggers (managed by the engine)
                if in_pos and not signals:
                    exit_sig = self._check_managed_exits(strat, builder, timestamp)
                    if exit_sig:
                        signals.append(exit_sig)

                # Get fired intra-bar trigger bases for dedup at bar-close
                # (must read BEFORE clearing for next bar period)
                fired_ib = self._intrabar_fired.get(dedup_key, set()).copy()

                # Clear intra-bar dedup set for this strategy/TF (new bar period)
                self._intrabar_fired.pop(dedup_key, None)

                for sig in signals:
                    # Enrich — mirrors alert_monitor.py poll_strategies() pattern
                    sig['level'] = 'strategy'
                    sig['strategy_id'] = strat_id
                    sig['strategy_name'] = strat.get('name', f"Strategy {strat_id}")
                    sig['symbol'] = strat.get('symbol', '?')
                    sig['direction'] = strat.get('direction', '?')
                    sig['risk_per_trade'] = strat.get('risk_per_trade', 100.0)
                    sig['timeframe'] = strat.get('timeframe', '1Min')
                    sig['strategy_alerts_visible'] = True
                    sig['source'] = 'streaming'

                    sig_type = sig.get('type', '')

                    # For exit signals, attach entry parameters so webhook
                    # quantity matches the entry size (risk / stop-distance)
                    if sig_type == 'exit_signal':
                        entry = self._position_entry.get(strat_id)
                        if entry:
                            sig['entry_price'] = entry.get('entry_price')
                            sig['entry_stop_price'] = entry.get('stop_price')

                    # Position gate (safety net — detect_signals uses override,
                    # but guard against edge cases)
                    if sig_type == 'entry_signal' and self._position_state.get(strat_id, False):
                        logger.debug("Bar-close entry suppressed (already in position): %s",
                                     strat.get('name'))
                        continue
                    if sig_type == 'exit_signal' and not self._position_state.get(strat_id, False):
                        logger.debug("Bar-close exit suppressed (not in position): %s",
                                     strat.get('name'))
                        continue

                    # Dedup: suppress bar-close if trigger already fired intra-bar
                    trigger_name = sig.get('trigger', '')
                    if trigger_name.removesuffix('_ib') in fired_ib:
                        logger.debug("Bar-close suppressed (intra-bar already fired): %s", trigger_name)
                        continue

                    sig = enrich_signal_with_portfolio_context(sig, strat_id)

                    # Cooldown — prevents re-firing the same signal type within one bar
                    cooldown_key = f"{strat_id}:{sig_type}"
                    if not self._cooldown.can_fire(cooldown_key, timestamp,
                                                   cooldown_seconds=float(tf_seconds)):
                        logger.debug("Alert suppressed by cooldown: %s", cooldown_key)
                        continue

                    # Update position state and entry tracking AFTER all gates pass
                    if sig_type == 'entry_signal':
                        self._position_state[strat_id] = True
                        self._position_entry[strat_id] = {
                            'entry_price': sig.get('price'),
                            'stop_price': sig.get('stop_price'),
                            'entry_bar_count': builder._bar_count,
                            'direction': strat.get('direction', 'LONG'),
                        }
                    elif sig_type == 'exit_signal':
                        self._position_state[strat_id] = False
                        self._position_entry.pop(strat_id, None)

                    alert = save_alert(sig)
                    logger.info("Streaming alert: %s for %s (%s)",
                                sig_type, strat.get('name'), self.symbol)

                    if self._alert_callback:
                        self._alert_callback(alert, config)

            except Exception as e:
                logger.error("Error evaluating %s on %s: %s",
                             strat.get('name', strat['id']), self.symbol, e)

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
            if direction == 'LONG' and float(last_bar['low']) <= float(stop_price):
                stopped = True
            elif direction == 'SHORT' and float(last_bar['high']) >= float(stop_price):
                stopped = True
            if stopped:
                logger.info("Managed exit: stop hit for %s (%s)", strat.get('name'), self.symbol)
                return {
                    "type": "exit_signal",
                    "trigger": "stop_loss",
                    "confluence_met": [],
                    "bar_time": bar_time,
                    "price": close_price,
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
        self._trigger_cache = TriggerLevelCache()
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
            return

        # Group strategies by symbol
        by_symbol: Dict[str, List[dict]] = {}
        for strat in strategies:
            sym = strat.get('symbol', 'SPY')
            by_symbol.setdefault(sym, []).append(strat)

        # Create SymbolHubs with warmup data
        self.hubs = {}
        for sym, strats in by_symbol.items():
            hub = SymbolHub(sym, alert_callback=self._queue_alert_delivery,
                          trigger_cache=self._trigger_cache)
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

                    bars_needed = max(compute_signal_detection_bars(reg_tf_str, session), MAX_HISTORY)
                    try:
                        warmup_df = load_latest_bars(
                            sym, bars=bars_needed, timeframe=reg_tf_str,
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

        self._running = True
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
        self._trigger_cache.clear()

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

                async def on_trade(trade):
                    if not self._running:
                        return
                    hub = self.hubs.get(trade.symbol)
                    if hub:
                        hub.on_tick(
                            price=float(trade.price),
                            volume=int(trade.size) if hasattr(trade, 'size') else 1,
                            timestamp=trade.timestamp,
                        )

                stream.subscribe_trades(on_trade, *symbols)

                self._connected = True
                self._set_streaming_status(True)
                backoff = 5  # reset on successful connect
                logger.info("WebSocket connected — streaming %d symbols", len(symbols))

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
