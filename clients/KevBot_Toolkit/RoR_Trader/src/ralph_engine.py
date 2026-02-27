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
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENGINE_STATUS_FILE = _SCRIPT_DIR / "engine_status.json"
_ENGINE_STATE_FILE = _SCRIPT_DIR / "engine_state.json"
_ENGINE_AUDIT_FILE = _SCRIPT_DIR / "engine_audit.jsonl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s  %(message)s"
_LOG_DATE_FMT = "%H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATE_FMT)
logger = logging.getLogger("ralph")

# Add file handler with rotation (5 MB max, keep 3 backups)
try:
    from logging.handlers import RotatingFileHandler
    _log_file = Path(__file__).resolve().parent / "ralph_engine.log"
    _fh = RotatingFileHandler(
        str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=3)
    _fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT))
    _fh.setLevel(logging.DEBUG)
    logger.addHandler(_fh)
except Exception:
    pass  # Non-critical — console logging still works

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_HISTORY = 25_000

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

# Warmup bar count
WARMUP_BARS = 250  # enough for EMA-200 convergence + margin

# Session gap threshold for VWAP reset
VWAP_SESSION_GAP_SECONDS = 30 * 60  # 30 minutes


# ═══════════════════════════════════════════════════════════════════════════
# BAR BUILDER (copied from realtime_engine.py — standalone, no dependencies)
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
# INCREMENTAL INDICATOR ENGINE — O(1) updates per bar close
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorState:
    """Cached state for O(1) incremental indicator computation."""

    # EMA states: {period: prev_ema}
    ema: Dict[int, float] = field(default_factory=dict)

    # MACD (fast=12, slow=26, signal=9 by default)
    macd_ema_fast: float = 0.0
    macd_ema_slow: float = 0.0
    macd_signal_ema: float = 0.0
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    # ATR (ewm-based, alpha = 2/(period+1))
    atr_value: float = 0.0
    atr_period: int = 14
    prev_close: float = 0.0

    # VWAP (resets on session boundary)
    vwap_cum_pv: float = 0.0
    vwap_cum_vol: float = 0.0
    vwap_cum_sq_dev_vol: float = 0.0
    vwap_value: float = 0.0
    vwap_std: float = 0.0
    vwap_sd1_mult: float = 1.0
    vwap_sd2_mult: float = 2.0
    vwap_prev_ts: Optional[datetime] = None

    # UT Bot (Wilder ATR, alpha = 1/period)
    utbot_atr: float = 0.0
    utbot_atr_period: int = 10
    utbot_atr_mult: float = 1.0
    utbot_trail_stop: float = 0.0
    utbot_direction: int = 0  # 1=bull, -1=bear, 0=unset
    utbot_prev_close: float = 0.0

    # Volume SMA (circular buffer)
    vol_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    vol_sum: float = 0.0
    vol_count: int = 0
    vol_sma_period: int = 20

    # Previous bar values (for trigger crossover detection and v2 variants)
    prev_values: Dict[str, float] = field(default_factory=dict)

    # Current bar indicator values (set after each update)
    current: Dict[str, float] = field(default_factory=dict)

    # MACD history for histogram triggers (need macd_hist from 2 bars ago)
    prev_macd_hist: float = 0.0
    prev2_macd_hist: float = 0.0


class IncrementalIndicatorEngine:
    """Strategy-scoped incremental indicator computation.

    Only computes indicators that the strategy actually needs, determined
    by its triggers and confluences.
    """

    def __init__(self, required_indicators: Set[str], params: Dict[str, Any]):
        """
        Args:
            required_indicators: Set of indicator keys needed, e.g.
                {'ema', 'atr', 'utbot', 'macd', 'vwap', 'rvol'}
            params: Parameters from confluence group config, e.g.
                {'ema_periods': [8, 21, 50], 'atr_period': 14,
                 'utbot_atr_period': 10, 'utbot_atr_mult': 1.0, ...}
        """
        self.required = required_indicators
        self.params = params
        self.state = IndicatorState()
        self._initialized = False

        # Configure state from params
        if 'ema' in self.required:
            for p in params.get('ema_periods', [8, 21, 50]):
                self.state.ema[p] = 0.0

        if 'macd' in self.required:
            self.state.macd_fast_period = params.get('macd_fast_period', 12)
            self.state.macd_slow_period = params.get('macd_slow_period', 26)
            self.state.macd_signal_period = params.get('macd_signal_period', 9)

        if 'atr' in self.required:
            self.state.atr_period = params.get('atr_period', 14)

        if 'vwap' in self.required:
            self.state.vwap_sd1_mult = params.get('vwap_sd1_mult', 1.0)
            self.state.vwap_sd2_mult = params.get('vwap_sd2_mult', 2.0)

        if 'utbot' in self.required:
            self.state.utbot_atr_period = params.get('utbot_atr_period', 10)
            self.state.utbot_atr_mult = params.get('utbot_atr_mult', 1.0)

        if 'rvol' in self.required:
            period = params.get('vol_sma_period', 20)
            self.state.vol_sma_period = period
            self.state.vol_buffer = deque(maxlen=period)

    def warmup(self, df: pd.DataFrame):
        """Initialize state from historical bars (runs batch pipeline once).

        After warmup, the state contains the final indicator values from
        the last bar, ready for incremental O(1) updates.
        """
        if len(df) < 2:
            logger.warning("Warmup: insufficient data (%d bars)", len(df))
            return

        # Process each bar sequentially to build state
        for i in range(len(df)):
            row = df.iloc[i]
            bar = {
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
                'timestamp': df.index[i],
            }
            is_first = (i == 0)
            self._update_indicators(bar, is_first=is_first)

        self._initialized = True
        logger.info("Warmup complete: %d bars, indicators: %s",
                     len(df), self.required)

    def update_bar(self, bar: dict) -> Dict[str, float]:
        """Incremental O(1) update for a new completed bar.

        Args:
            bar: {'open', 'high', 'low', 'close', 'volume', 'timestamp'}

        Returns:
            Dict of current indicator values.
        """
        # Save previous values before updating
        self.state.prev2_macd_hist = self.state.prev_macd_hist
        self.state.prev_macd_hist = self.state.current.get('macd_hist', 0.0)
        self.state.prev_values = dict(self.state.current)

        self._update_indicators(bar, is_first=False)
        return dict(self.state.current)

    def get_values(self) -> Dict[str, float]:
        """Get current indicator values."""
        return dict(self.state.current)

    def get_prev_values(self) -> Dict[str, float]:
        """Get previous bar's indicator values."""
        return dict(self.state.prev_values)

    def _update_indicators(self, bar: dict, is_first: bool = False):
        """Core update logic — called for both warmup and live bars."""
        close = bar['close']
        high = bar['high']
        low = bar['low']
        volume = bar['volume']
        timestamp = bar['timestamp']

        vals: Dict[str, float] = {
            'close': close, 'high': high, 'low': low,
            'open': bar['open'], 'volume': volume,
        }

        # ── EMA ──
        if 'ema' in self.required:
            for period, prev_ema in self.state.ema.items():
                alpha = 2.0 / (period + 1)
                if is_first:
                    new_ema = close
                else:
                    new_ema = alpha * close + (1.0 - alpha) * prev_ema
                self.state.ema[period] = new_ema
                vals[f'ema_{period}'] = new_ema

        # ── ATR (ewm-based) ──
        if 'atr' in self.required:
            if is_first:
                tr = high - low
                self.state.atr_value = tr
                self.state.prev_close = close
            else:
                tr1 = high - low
                tr2 = abs(high - self.state.prev_close)
                tr3 = abs(low - self.state.prev_close)
                tr = max(tr1, tr2, tr3)
                alpha = 2.0 / (self.state.atr_period + 1)
                self.state.atr_value = (
                    alpha * tr + (1.0 - alpha) * self.state.atr_value)
                self.state.prev_close = close
            vals['atr'] = self.state.atr_value

        # ── MACD ──
        if 'macd' in self.required:
            fp = self.state.macd_fast_period
            sp = self.state.macd_slow_period
            sig_p = self.state.macd_signal_period
            af = 2.0 / (fp + 1)
            a_s = 2.0 / (sp + 1)
            a_sig = 2.0 / (sig_p + 1)

            if is_first:
                self.state.macd_ema_fast = close
                self.state.macd_ema_slow = close
                self.state.macd_signal_ema = 0.0
            else:
                self.state.macd_ema_fast = (
                    af * close + (1.0 - af) * self.state.macd_ema_fast)
                self.state.macd_ema_slow = (
                    a_s * close + (1.0 - a_s) * self.state.macd_ema_slow)

            macd_line = self.state.macd_ema_fast - self.state.macd_ema_slow

            if is_first:
                self.state.macd_signal_ema = macd_line
            else:
                self.state.macd_signal_ema = (
                    a_sig * macd_line
                    + (1.0 - a_sig) * self.state.macd_signal_ema)

            macd_hist = macd_line - self.state.macd_signal_ema
            vals['macd_line'] = macd_line
            vals['macd_signal'] = self.state.macd_signal_ema
            vals['macd_hist'] = macd_hist

        # ── VWAP ──
        if 'vwap' in self.required:
            tp = (high + low + close) / 3.0

            # Session reset: gap > 30 min from previous bar
            if self.state.vwap_prev_ts is not None:
                if hasattr(timestamp, 'timestamp'):
                    ts_epoch = timestamp.timestamp()
                else:
                    ts_epoch = pd.Timestamp(timestamp).timestamp()
                prev_epoch = (self.state.vwap_prev_ts.timestamp()
                              if hasattr(self.state.vwap_prev_ts, 'timestamp')
                              else pd.Timestamp(
                                  self.state.vwap_prev_ts).timestamp())
                if ts_epoch - prev_epoch > VWAP_SESSION_GAP_SECONDS:
                    self.state.vwap_cum_pv = 0.0
                    self.state.vwap_cum_vol = 0.0
                    self.state.vwap_cum_sq_dev_vol = 0.0

            if is_first:
                self.state.vwap_cum_pv = 0.0
                self.state.vwap_cum_vol = 0.0
                self.state.vwap_cum_sq_dev_vol = 0.0

            self.state.vwap_cum_pv += tp * volume
            self.state.vwap_cum_vol += volume

            if self.state.vwap_cum_vol > 0:
                vwap = self.state.vwap_cum_pv / self.state.vwap_cum_vol
                sq_dev = volume * (tp - vwap) ** 2
                self.state.vwap_cum_sq_dev_vol += sq_dev
                std = math.sqrt(
                    self.state.vwap_cum_sq_dev_vol / self.state.vwap_cum_vol)
            else:
                vwap = close
                std = 0.0

            self.state.vwap_value = vwap
            self.state.vwap_std = std
            self.state.vwap_prev_ts = timestamp

            vals['vwap'] = vwap
            m1 = self.state.vwap_sd1_mult
            m2 = self.state.vwap_sd2_mult
            vals['vwap_sd1_upper'] = vwap + m1 * std
            vals['vwap_sd1_lower'] = vwap - m1 * std
            vals['vwap_sd2_upper'] = vwap + m2 * std
            vals['vwap_sd2_lower'] = vwap - m2 * std

        # ── UT Bot (Wilder ATR) ──
        if 'utbot' in self.required:
            period = self.state.utbot_atr_period
            mult = self.state.utbot_atr_mult
            alpha_w = 1.0 / period  # Wilder smoothing

            if is_first:
                tr = high - low
                self.state.utbot_atr = tr
                self.state.utbot_trail_stop = close - mult * tr
                self.state.utbot_direction = 0
                self.state.utbot_prev_close = close
            else:
                prev_c = self.state.utbot_prev_close
                tr1 = high - low
                tr2 = abs(high - prev_c)
                tr3 = abs(low - prev_c)
                tr = max(tr1, tr2, tr3)

                self.state.utbot_atr = (
                    self.state.utbot_atr + alpha_w
                    * (tr - self.state.utbot_atr))

                n_loss = mult * self.state.utbot_atr
                prev_stop = self.state.utbot_trail_stop
                prev_dir = self.state.utbot_direction

                # Trailing stop ratchet logic
                if close > prev_stop and prev_c > prev_stop:
                    new_stop = max(prev_stop, close - n_loss)
                elif close < prev_stop and prev_c < prev_stop:
                    new_stop = min(prev_stop, close + n_loss)
                elif close > prev_stop:
                    new_stop = close - n_loss  # flip to bull
                else:
                    new_stop = close + n_loss  # flip to bear

                # Direction detection
                if prev_c < prev_stop and close > prev_stop:
                    direction = 1  # bullish flip
                elif prev_c > prev_stop and close < prev_stop:
                    direction = -1  # bearish flip
                else:
                    direction = prev_dir  # carry forward

                self.state.utbot_trail_stop = new_stop
                self.state.utbot_direction = direction
                self.state.utbot_prev_close = close

            vals['utbot_stop'] = self.state.utbot_trail_stop
            vals['utbot_direction'] = self.state.utbot_direction
            vals['utbot_atr'] = self.state.utbot_atr
            # v2 (confirmed) — previous bar's stop level
            prev_stop_val = self.state.prev_values.get(
                'utbot_stop', self.state.utbot_trail_stop)
            vals['utbot_stop_prev'] = prev_stop_val

        # ── Volume SMA / RVOL ──
        if 'rvol' in self.required:
            buf = self.state.vol_buffer
            if len(buf) == buf.maxlen:
                self.state.vol_sum -= buf[0]
            buf.append(volume)
            self.state.vol_sum += volume
            self.state.vol_count = min(
                self.state.vol_count + 1, self.state.vol_sma_period)

            if self.state.vol_count >= 5:
                vol_sma = self.state.vol_sum / len(buf)
                rvol = volume / vol_sma if vol_sma > 0 else 0.0
            else:
                vol_sma = 0.0
                rvol = 0.0
            vals['vol_sma'] = vol_sma
            vals['rvol'] = rvol

        # ── Store previous EMA values for v2 triggers ──
        if 'ema' in self.required:
            for period in self.state.ema:
                prev_key = f'ema_{period}_prev'
                vals[prev_key] = self.state.prev_values.get(
                    f'ema_{period}', vals.get(f'ema_{period}', 0.0))

        self.state.current = vals


# ═══════════════════════════════════════════════════════════════════════════
# TRIGGER EVALUATOR — single-row interpreter + trigger evaluation
# ═══════════════════════════════════════════════════════════════════════════

class TriggerEvaluator:
    """Evaluates interpreters and triggers on a single bar's data.

    Bar-close triggers compare current vs previous bar values.
    Intra-bar triggers check tick price against cached levels.
    """

    def __init__(self, required_interpreters: Set[str],
                 required_triggers: Set[str],
                 ema_periods: List[int] = None):
        self.required_interpreters = required_interpreters
        self.required_triggers = required_triggers
        self.ema_periods = ema_periods or [8, 21, 50]

        # Intra-bar state: once-per-bar firing
        self._ib_fired: Dict[str, bool] = {}
        self._cached_levels: Dict[str, float] = {}

    def evaluate_bar_close(self, current: Dict[str, float],
                           prev: Dict[str, float],
                           prev2_macd_hist: float = 0.0,
                           ) -> Tuple[Dict[str, str], Dict[str, bool]]:
        """Evaluate interpreters and triggers on bar close.

        Returns:
            (interpreter_states, trigger_booleans)
        """
        interps: Dict[str, str] = {}
        triggers: Dict[str, bool] = {}

        # ── Interpreters ──

        if 'EMA_STACK' in self.required_interpreters:
            s, m, l_ = self._get_ema_triple(current)
            if s is not None:
                interps['EMA_STACK'] = self._classify_ema_stack(s, m, l_)

        if 'EMA_PRICE_POSITION' in self.required_interpreters:
            interps['EMA_PRICE_POSITION'] = self._classify_ema_price_pos(
                current)

        if 'EMA_PRICE_POSITION_V2' in self.required_interpreters:
            interps['EMA_PRICE_POSITION_V2'] = self._classify_ema_price_pos(
                current)

        if 'MACD_LINE' in self.required_interpreters:
            ml = current.get('macd_line', 0)
            ms = current.get('macd_signal', 0)
            if ml > ms:
                interps['MACD_LINE'] = 'M>S+' if ml > 0 else 'M>S-'
            else:
                interps['MACD_LINE'] = 'M<S-' if ml <= 0 else 'M<S+'

        if 'MACD_HISTOGRAM' in self.required_interpreters:
            mh = current.get('macd_hist', 0)
            pmh = prev.get('macd_hist', 0)
            if mh > 0:
                interps['MACD_HISTOGRAM'] = (
                    'H+up' if mh > pmh else 'H+dn')
            else:
                interps['MACD_HISTOGRAM'] = (
                    'H-dn' if mh <= pmh else 'H-up')

        if 'VWAP' in self.required_interpreters:
            interps['VWAP'] = self._classify_vwap(current)

        if 'RVOL' in self.required_interpreters:
            rv = current.get('rvol', 0)
            if rv > 3.0:
                interps['RVOL'] = 'EXTREME'
            elif rv > 1.5:
                interps['RVOL'] = 'HIGH'
            elif rv > 0.75:
                interps['RVOL'] = 'NORMAL'
            elif rv > 0.5:
                interps['RVOL'] = 'LOW'
            else:
                interps['RVOL'] = 'MINIMAL'

        if 'UTBOT' in self.required_interpreters:
            d = current.get('utbot_direction', 0)
            if d == 1:
                interps['UTBOT'] = 'BULL'
            elif d == -1:
                interps['UTBOT'] = 'BEAR'

        if 'UTBOT_V2' in self.required_interpreters:
            d = current.get('utbot_direction', 0)
            if d == 1:
                interps['UTBOT_V2'] = 'BULL'
            elif d == -1:
                interps['UTBOT_V2'] = 'BEAR'

        # ── Triggers (crossover detection: current vs prev) ──

        # EMA Stack triggers
        if 'ema_cross_bull' in self.required_triggers:
            triggers['ema_cross_bull'] = (
                current.get('ema_8', 0) > current.get('ema_21', 0)
                and prev.get('ema_8', 0) <= prev.get('ema_21', 0))
        if 'ema_cross_bear' in self.required_triggers:
            triggers['ema_cross_bear'] = (
                current.get('ema_8', 0) < current.get('ema_21', 0)
                and prev.get('ema_8', 0) >= prev.get('ema_21', 0))
        if 'ema_mid_cross_bull' in self.required_triggers:
            triggers['ema_mid_cross_bull'] = (
                current.get('ema_21', 0) > current.get('ema_50', 0)
                and prev.get('ema_21', 0) <= prev.get('ema_50', 0))
        if 'ema_mid_cross_bear' in self.required_triggers:
            triggers['ema_mid_cross_bear'] = (
                current.get('ema_21', 0) < current.get('ema_50', 0)
                and prev.get('ema_21', 0) >= prev.get('ema_50', 0))

        # EMA Price Position triggers
        for prefix in ('ema_pp', 'ema_pp_v2'):
            sp = self.ema_periods[0] if len(self.ema_periods) > 0 else 8
            mp = self.ema_periods[1] if len(self.ema_periods) > 1 else 21

            s_key = f'ema_{sp}'
            m_key = f'ema_{mp}'
            # For v2, use _prev versions of EMA for the level
            if prefix == 'ema_pp_v2':
                s_prev_key = f'ema_{sp}_prev'
                m_prev_key = f'ema_{mp}_prev'
            else:
                s_prev_key = s_key
                m_prev_key = m_key

            t = f'{prefix}_cross_short_up'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) > current.get(s_key, 0)
                    and prev.get('close', 0) <= prev.get(s_key, 0))
            t = f'{prefix}_cross_short_down'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) < current.get(s_key, 0)
                    and prev.get('close', 0) >= prev.get(s_key, 0))
            t = f'{prefix}_cross_mid_up'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) > current.get(m_key, 0)
                    and prev.get('close', 0) <= prev.get(m_key, 0))
            t = f'{prefix}_cross_mid_down'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) < current.get(m_key, 0)
                    and prev.get('close', 0) >= prev.get(m_key, 0))

        # MACD triggers
        if 'macd_cross_bull' in self.required_triggers:
            triggers['macd_cross_bull'] = (
                current.get('macd_line', 0) > current.get('macd_signal', 0)
                and prev.get('macd_line', 0) <= prev.get('macd_signal', 0))
        if 'macd_cross_bear' in self.required_triggers:
            triggers['macd_cross_bear'] = (
                current.get('macd_line', 0) < current.get('macd_signal', 0)
                and prev.get('macd_line', 0) >= prev.get('macd_signal', 0))
        if 'macd_zero_cross_up' in self.required_triggers:
            triggers['macd_zero_cross_up'] = (
                current.get('macd_line', 0) > 0
                and prev.get('macd_line', 0) <= 0)
        if 'macd_zero_cross_down' in self.required_triggers:
            triggers['macd_zero_cross_down'] = (
                current.get('macd_line', 0) < 0
                and prev.get('macd_line', 0) >= 0)

        # MACD Histogram triggers
        if 'macd_hist_flip_pos' in self.required_triggers:
            triggers['macd_hist_flip_pos'] = (
                current.get('macd_hist', 0) > 0
                and prev.get('macd_hist', 0) <= 0)
        if 'macd_hist_flip_neg' in self.required_triggers:
            triggers['macd_hist_flip_neg'] = (
                current.get('macd_hist', 0) < 0
                and prev.get('macd_hist', 0) >= 0)
        if 'macd_hist_momentum_shift_up' in self.required_triggers:
            mh = current.get('macd_hist', 0)
            pmh = prev.get('macd_hist', 0)
            triggers['macd_hist_momentum_shift_up'] = (
                mh > pmh and pmh < prev2_macd_hist)
        if 'macd_hist_momentum_shift_down' in self.required_triggers:
            mh = current.get('macd_hist', 0)
            pmh = prev.get('macd_hist', 0)
            triggers['macd_hist_momentum_shift_down'] = (
                mh < pmh and pmh > prev2_macd_hist)

        # VWAP triggers
        if 'vwap_cross_above' in self.required_triggers:
            triggers['vwap_cross_above'] = (
                current.get('close', 0) > current.get('vwap', 0)
                and prev.get('close', 0) <= prev.get('vwap', 0))
        if 'vwap_cross_below' in self.required_triggers:
            triggers['vwap_cross_below'] = (
                current.get('close', 0) < current.get('vwap', 0)
                and prev.get('close', 0) >= prev.get('vwap', 0))
        if 'vwap_enter_upper_extreme' in self.required_triggers:
            triggers['vwap_enter_upper_extreme'] = (
                current.get('close', 0) > current.get('vwap_sd2_upper', 0)
                and prev.get('close', 0)
                <= prev.get('vwap_sd2_upper', 0))
        if 'vwap_enter_lower_extreme' in self.required_triggers:
            triggers['vwap_enter_lower_extreme'] = (
                current.get('close', 0) < current.get('vwap_sd2_lower', 0)
                and prev.get('close', 0)
                >= prev.get('vwap_sd2_lower', 0))
        if 'vwap_return_to_vwap' in self.required_triggers:
            pc = prev.get('close', 0)
            pv1u = prev.get('vwap_sd1_upper', 0)
            pv1l = prev.get('vwap_sd1_lower', 0)
            was_extreme = pc > pv1u or pc < pv1l
            v = current.get('vwap', 0)
            v1u = current.get('vwap_sd1_upper', 0)
            half_sd = (v1u - v) * 0.5 if v1u > v else v * 0.001
            c = current.get('close', 0)
            now_at_vwap = (v - half_sd) <= c <= (v + half_sd)
            triggers['vwap_return_to_vwap'] = was_extreme and now_at_vwap

        # RVOL triggers
        if 'rvol_spike' in self.required_triggers:
            triggers['rvol_spike'] = (
                current.get('rvol', 0) > 1.5
                and prev.get('rvol', 0) <= 1.5)
        if 'rvol_extreme' in self.required_triggers:
            triggers['rvol_extreme'] = (
                current.get('rvol', 0) > 3.0
                and prev.get('rvol', 0) <= 3.0)
        if 'rvol_fade' in self.required_triggers:
            triggers['rvol_fade'] = (
                current.get('rvol', 0) < 1.0
                and prev.get('rvol', 0) >= 1.0)

        # UT Bot triggers
        if 'utbot_buy' in self.required_triggers:
            triggers['utbot_buy'] = (
                current.get('utbot_direction', 0) == 1
                and prev.get('utbot_direction', 0) != 1)
        if 'utbot_sell' in self.required_triggers:
            triggers['utbot_sell'] = (
                current.get('utbot_direction', 0) == -1
                and prev.get('utbot_direction', 0) != -1)
        # UT Bot v2 (same boolean logic, different fill price)
        if 'utbot_v2_buy' in self.required_triggers:
            triggers['utbot_v2_buy'] = (
                current.get('utbot_direction', 0) == 1
                and prev.get('utbot_direction', 0) != 1)
        if 'utbot_v2_sell' in self.required_triggers:
            triggers['utbot_v2_sell'] = (
                current.get('utbot_direction', 0) == -1
                and prev.get('utbot_direction', 0) != -1)

        # Update cached levels for intra-bar detection
        self._update_cached_levels(current)
        # Reset intra-bar fired flags on bar close
        self._ib_fired.clear()

        return interps, triggers

    def check_intrabar(self, price: float) -> Optional[Tuple[str, float]]:
        """Check if tick price crosses any cached level (O(1) per level).

        Returns (trigger_id, fill_price) if a crossing is detected,
        or None.  Each trigger fires at most once per bar.
        """
        for trigger_id, (level, direction) in self._get_ib_checks():
            if self._ib_fired.get(trigger_id, False):
                continue
            if direction == 'above' and price > level:
                self._ib_fired[trigger_id] = True
                return (trigger_id, level)
            elif direction == 'below' and price < level:
                self._ib_fired[trigger_id] = True
                return (trigger_id, level)
        return None

    def _get_ib_checks(self) -> List[Tuple[str, Tuple[float, str]]]:
        """Return list of (trigger_id, (level, direction)) for intra-bar."""
        checks = []
        # Map of base trigger → (level_key, direction)
        IB_MAP = {
            'vwap_cross_above': ('vwap', 'above'),
            'vwap_cross_below': ('vwap', 'below'),
            'vwap_enter_upper_extreme': ('vwap_sd2_upper', 'above'),
            'vwap_enter_lower_extreme': ('vwap_sd2_lower', 'below'),
            'utbot_buy': ('utbot_stop', 'above'),
            'utbot_sell': ('utbot_stop', 'below'),
            'utbot_v2_buy': ('utbot_stop_prev', 'above'),
            'utbot_v2_sell': ('utbot_stop_prev', 'below'),
        }
        # EMA price position — dynamic by period
        if self.ema_periods:
            sp = self.ema_periods[0]
            mp = self.ema_periods[1] if len(self.ema_periods) > 1 else 21
            IB_MAP.update({
                'ema_pp_cross_short_up': (f'ema_{sp}', 'above'),
                'ema_pp_cross_short_down': (f'ema_{sp}', 'below'),
                'ema_pp_cross_mid_up': (f'ema_{mp}', 'above'),
                'ema_pp_cross_mid_down': (f'ema_{mp}', 'below'),
                'ema_pp_v2_cross_short_up': (f'ema_{sp}_prev', 'above'),
                'ema_pp_v2_cross_short_down': (f'ema_{sp}_prev', 'below'),
                'ema_pp_v2_cross_mid_up': (f'ema_{mp}_prev', 'above'),
                'ema_pp_v2_cross_mid_down': (f'ema_{mp}_prev', 'below'),
            })

        for base_trigger, (level_key, direction) in IB_MAP.items():
            ib_trigger = f'{base_trigger}_ib'
            if ib_trigger in self.required_triggers:
                level = self._cached_levels.get(level_key)
                if level is not None and level > 0:
                    checks.append((ib_trigger, (level, direction)))
        return checks

    def _update_cached_levels(self, current: Dict[str, float]):
        """Cache indicator levels for intra-bar crossing detection."""
        for key in ('vwap', 'vwap_sd2_upper', 'vwap_sd2_lower',
                    'utbot_stop', 'utbot_stop_prev'):
            if key in current:
                self._cached_levels[key] = current[key]
        # EMA levels
        for period in self.ema_periods:
            for suffix in ('', '_prev'):
                key = f'ema_{period}{suffix}'
                if key in current:
                    self._cached_levels[key] = current[key]

    def _get_ema_triple(self, vals):
        if len(self.ema_periods) >= 3:
            s = vals.get(f'ema_{self.ema_periods[0]}')
            m = vals.get(f'ema_{self.ema_periods[1]}')
            l_ = vals.get(f'ema_{self.ema_periods[2]}')
            return s, m, l_
        return None, None, None

    @staticmethod
    def _classify_ema_stack(s, m, l_) -> str:
        if s > m > l_:
            return 'SML'
        elif s > l_ > m:
            return 'SLM'
        elif m > s > l_:
            return 'MSL'
        elif m > l_ > s:
            return 'MLS'
        elif l_ > s > m:
            return 'LSM'
        elif l_ > m > s:
            return 'LMS'
        return 'SML'  # exact equality fallback

    def _classify_ema_price_pos(self, vals) -> str:
        s, m, l_ = self._get_ema_triple(vals)
        c = vals.get('close', 0)
        if s is None:
            return 'PSML'
        items = [('P', c), ('S', s), ('M', m), ('L', l_)]
        items.sort(key=lambda x: -x[1])
        return ''.join(x[0] for x in items)

    @staticmethod
    def _classify_vwap(vals) -> str:
        c = vals.get('close', 0)
        v = vals.get('vwap', 0)
        v2u = vals.get('vwap_sd2_upper', v)
        v1u = vals.get('vwap_sd1_upper', v)
        v1l = vals.get('vwap_sd1_lower', v)
        v2l = vals.get('vwap_sd2_lower', v)
        half_sd = (v1u - v) * 0.5 if v1u > v else v * 0.001

        if c > v2u:
            return '>+2sigma'
        elif c > v1u:
            return '>+1sigma'
        elif c > v + half_sd:
            return '>V'
        elif c >= v - half_sd:
            return '@V'
        elif c >= v1l:
            return '<V'
        elif c >= v2l:
            return '<-1sigma'
        else:
            return '<-2sigma'


# ═══════════════════════════════════════════════════════════════════════════
# POSITION STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PositionState:
    """Persistent position state for one strategy."""
    status: str = 'FLAT'  # 'FLAT' or 'IN_POSITION'
    entry_price: float = 0.0
    entry_time: Optional[str] = None
    stop_price: float = 0.0
    target_price: Optional[float] = None
    entry_bar_count: int = 0
    direction: str = 'LONG'
    entry_trigger: str = ''

    def to_dict(self) -> dict:
        return {
            'status': self.status,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'entry_bar_count': self.entry_bar_count,
            'direction': self.direction,
            'entry_trigger': self.entry_trigger,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PositionState':
        return cls(
            status=d.get('status', 'FLAT'),
            entry_price=d.get('entry_price', 0.0),
            entry_time=d.get('entry_time'),
            stop_price=d.get('stop_price', 0.0),
            target_price=d.get('target_price'),
            entry_bar_count=d.get('entry_bar_count', 0),
            direction=d.get('direction', 'LONG'),
            entry_trigger=d.get('entry_trigger', ''),
        )


class PositionStateMachine:
    """Manages FLAT ↔ IN_POSITION transitions for one strategy."""

    def __init__(self, strategy: dict, state: Optional[PositionState] = None,
                 resolved_entry: str = '', resolved_exits: List[str] = None):
        self.strategy = strategy
        self.state = state or PositionState(
            direction=strategy.get('direction', 'LONG'))
        self.strat_id = strategy['id']

        # Use resolved trigger IDs (from confluence mapping)
        self.entry_trigger = resolved_entry or strategy.get('entry_trigger', '')
        self.exit_triggers = set()
        if resolved_exits:
            self.exit_triggers = set(resolved_exits)
        elif strategy.get('exit_triggers'):
            self.exit_triggers = set(strategy['exit_triggers'])
        elif strategy.get('exit_trigger'):
            et = strategy['exit_trigger']
            if et and et not in ('opposite_signal',):
                self.exit_triggers = {et}

        self.bar_count_exit = strategy.get('bar_count_exit')
        self.stop_config = strategy.get('stop_config') or {
            'method': 'atr', 'atr_mult': strategy.get('stop_atr_mult', 1.5)}
        self.target_config = strategy.get('target_config')
        self.confluence_set = set(strategy.get('confluence', [])) | set(
            strategy.get('general_confluences', []))
        self.confluence_set = self.confluence_set or None

    def check_entry(self, trigger_booleans: Dict[str, bool],
                    interpreter_states: Dict[str, str],
                    current_values: Dict[str, float],
                    bar_count: int,
                    bar_time: str,
                    confluence_records: Set[str] = None,
                    ) -> Optional[dict]:
        """Check for entry signal. Returns signal dict or None."""
        if self.state.status != 'FLAT':
            return None

        # Check if entry trigger fired
        trigger_id = self.entry_trigger
        # For _ib triggers, the bar-close base is the same boolean
        base_trigger = (trigger_id[:-3]
                        if trigger_id.endswith('_ib') else trigger_id)

        if not trigger_booleans.get(trigger_id, False) and \
           not trigger_booleans.get(base_trigger, False):
            return None

        # Check confluence
        if self.confluence_set and confluence_records:
            if not self.confluence_set.issubset(confluence_records):
                return None

        # Compute stop price
        close = current_values.get('close', 0)
        atr = current_values.get('atr', close * 0.01)
        if not atr or atr <= 0:
            atr = close * 0.01

        stop_price = self._compute_stop(close, atr, current_values)
        target_price = self._compute_target(close, stop_price, atr,
                                            current_values)

        # Transition to IN_POSITION
        self.state.status = 'IN_POSITION'
        self.state.entry_price = close
        self.state.entry_time = bar_time
        self.state.stop_price = stop_price
        self.state.target_price = target_price
        self.state.entry_bar_count = bar_count
        self.state.entry_trigger = trigger_id

        return {
            'type': 'entry_signal',
            'trigger': trigger_id,
            'price': close,
            'stop_price': stop_price,
            'target_price': target_price,
            'bar_time': bar_time,
            'atr': atr,
        }

    def check_entry_intrabar(self, trigger_id: str, fill_price: float,
                             current_values: Dict[str, float],
                             bar_count: int, timestamp: str,
                             confluence_records: Set[str] = None,
                             ) -> Optional[dict]:
        """Check intra-bar entry. Returns signal dict or None."""
        if self.state.status != 'FLAT':
            return None

        # Verify this is our entry trigger
        if trigger_id != self.entry_trigger:
            return None

        # Confluence check
        if self.confluence_set and confluence_records:
            if not self.confluence_set.issubset(confluence_records):
                return None

        atr = current_values.get('atr', fill_price * 0.01)
        if not atr or atr <= 0:
            atr = fill_price * 0.01

        stop_price = self._compute_stop(fill_price, atr, current_values)
        target_price = self._compute_target(fill_price, stop_price, atr,
                                            current_values)

        self.state.status = 'IN_POSITION'
        self.state.entry_price = fill_price
        self.state.entry_time = timestamp
        self.state.stop_price = stop_price
        self.state.target_price = target_price
        self.state.entry_bar_count = bar_count
        self.state.entry_trigger = trigger_id

        return {
            'type': 'entry_signal',
            'trigger': trigger_id,
            'price': fill_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'bar_time': timestamp,
            'atr': atr,
        }

    def check_exit_bar_close(self, trigger_booleans: Dict[str, bool],
                             current_values: Dict[str, float],
                             bar_count: int,
                             bar_time: str) -> Optional[dict]:
        """Check for exit on bar close. Returns signal dict or None."""
        if self.state.status != 'IN_POSITION':
            return None

        close = current_values.get('close', 0)
        high = current_values.get('high', close)
        low = current_values.get('low', close)
        direction = self.state.direction

        # Priority 1: Stop loss (check bar's extreme)
        if self.state.stop_price:
            if direction == 'LONG' and low <= self.state.stop_price:
                fill = min(self.state.stop_price,
                           current_values.get('open', close))
                return self._exit('stop_loss', fill, bar_time)
            elif direction == 'SHORT' and high >= self.state.stop_price:
                fill = max(self.state.stop_price,
                           current_values.get('open', close))
                return self._exit('stop_loss', fill, bar_time)

        # Priority 2: Target
        if self.state.target_price:
            if direction == 'LONG' and high >= self.state.target_price:
                fill = max(self.state.target_price,
                           current_values.get('open', close))
                return self._exit('target', fill, bar_time)
            elif direction == 'SHORT' and low <= self.state.target_price:
                fill = min(self.state.target_price,
                           current_values.get('open', close))
                return self._exit('target', fill, bar_time)

        # Priority 3: Signal exit
        for et in self.exit_triggers:
            base_et = et[:-3] if et.endswith('_ib') else et
            if trigger_booleans.get(et, False) or \
               trigger_booleans.get(base_et, False):
                return self._exit(et, close, bar_time)

        # Priority 4: Bar count exit
        if self.bar_count_exit is not None:
            bars_held = bar_count - self.state.entry_bar_count
            if bars_held >= self.bar_count_exit:
                return self._exit('bar_count_exit', close, bar_time)

        return None

    def check_exit_tick(self, price: float,
                        timestamp: str) -> Optional[dict]:
        """Check stop/target on each tick (O(1))."""
        if self.state.status != 'IN_POSITION':
            return None

        direction = self.state.direction

        # Stop loss
        if self.state.stop_price:
            if direction == 'LONG' and price <= self.state.stop_price:
                return self._exit('stop_loss', self.state.stop_price,
                                  timestamp)
            elif direction == 'SHORT' and price >= self.state.stop_price:
                return self._exit('stop_loss', self.state.stop_price,
                                  timestamp)

        # Target
        if self.state.target_price:
            if direction == 'LONG' and price >= self.state.target_price:
                return self._exit('target', self.state.target_price,
                                  timestamp)
            elif direction == 'SHORT' and price <= self.state.target_price:
                return self._exit('target', self.state.target_price,
                                  timestamp)

        return None

    def check_exit_intrabar(self, trigger_id: str, fill_price: float,
                            timestamp: str) -> Optional[dict]:
        """Check signal exit from intra-bar level cross."""
        if self.state.status != 'IN_POSITION':
            return None

        base_id = trigger_id[:-3] if trigger_id.endswith('_ib') else trigger_id
        if base_id in self.exit_triggers or trigger_id in self.exit_triggers:
            return self._exit(trigger_id, fill_price, timestamp)
        return None

    def _exit(self, reason: str, price: float, bar_time: str) -> dict:
        """Execute exit transition."""
        sig = {
            'type': 'exit_signal',
            'trigger': reason,
            'price': price,
            'bar_time': bar_time,
            'entry_price': self.state.entry_price,
            'entry_stop_price': self.state.stop_price,
            'atr': 0,  # filled by caller
        }
        self.state.status = 'FLAT'
        self.state.entry_price = 0.0
        self.state.entry_time = None
        self.state.stop_price = 0.0
        self.state.target_price = None
        return sig

    def _compute_stop(self, entry_price: float, atr: float,
                      vals: Dict[str, float]) -> float:
        method = self.stop_config.get('method', 'atr')
        direction = self.state.direction

        if method == 'atr':
            mult = self.stop_config.get('atr_mult', 1.5)
            distance = atr * mult
        elif method == 'fixed_dollar':
            distance = self.stop_config.get('dollar_amount', 1.0)
        elif method == 'percentage':
            pct = self.stop_config.get('percentage', 0.5)
            distance = entry_price * (pct / 100.0)
        elif method == 'swing':
            # Swing stop needs recent bar history — use ATR fallback
            # for live engine (historical swing computed at warmup)
            distance = atr * 1.5
        else:
            distance = atr * 1.5

        if direction == 'LONG':
            return entry_price - distance
        else:
            return entry_price + distance

    def _compute_target(self, entry_price: float, stop_price: float,
                        atr: float, vals: Dict[str, float]) -> Optional[float]:
        if self.target_config is None:
            return None

        method = self.target_config.get('method')
        if method is None:
            return None

        risk = abs(entry_price - stop_price)
        direction = self.state.direction

        if method == 'risk_reward':
            rr = self.target_config.get('rr_ratio', 2.0)
            distance = risk * rr
        elif method == 'atr':
            mult = self.target_config.get('atr_mult', 2.0)
            distance = atr * mult
        elif method == 'fixed_dollar':
            distance = self.target_config.get('dollar_amount', 2.0)
        elif method == 'percentage':
            pct = self.target_config.get('percentage', 1.0)
            distance = entry_price * (pct / 100.0)
        else:
            return None

        if direction == 'LONG':
            return entry_price + distance
        else:
            return entry_price - distance


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY RESOLVER — maps strategy config to required indicators/triggers
# ═══════════════════════════════════════════════════════════════════════════

# Intra-bar level map — used by triggers.py for backtest fill-price resolution
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


# Template key → required indicator set and interpreter key
TEMPLATE_REQUIREMENTS = {
    'ema_stack': ({'ema'}, 'EMA_STACK'),
    'ema_price_position': ({'ema'}, 'EMA_PRICE_POSITION'),
    'ema_price_position_v2': ({'ema'}, 'EMA_PRICE_POSITION_V2'),
    'macd_line': ({'macd'}, 'MACD_LINE'),
    'macd_histogram': ({'macd'}, 'MACD_HISTOGRAM'),
    'vwap': ({'vwap'}, 'VWAP'),
    'rvol': ({'rvol'}, 'RVOL'),
    'utbot': ({'utbot'}, 'UTBOT'),
    'utbot_v2': ({'utbot'}, 'UTBOT_V2'),
    'bar_count': (set(), None),
}

# Trigger prefix → template key
TRIGGER_PREFIX_TO_TEMPLATE = {
    'ema': 'ema_stack',
    'ema_pp': 'ema_price_position',
    'ema_pp_v2': 'ema_price_position_v2',
    'macd': 'macd_line',
    'macd_hist': 'macd_histogram',
    'vwap': 'vwap',
    'rvol': 'rvol',
    'utbot': 'utbot',
    'utbot_v2': 'utbot_v2',
    'bar_count': 'bar_count',
}


def _resolve_trigger_id(strategy: dict, key: str) -> str:
    """Resolve a strategy's trigger ID using confluence mapping.

    Converts confluence_trigger_ids (e.g. 'ema_stack_default_cross_bull')
    to prefixed trigger IDs (e.g. 'ema_cross_bull') using the same logic
    as the existing alert/realtime engine.
    """
    confluence_key = key.replace('trigger', 'trigger_confluence_id')
    cid = strategy.get(confluence_key)
    if cid:
        try:
            from alerts import _get_base_trigger_id
            return _get_base_trigger_id(cid)
        except Exception:
            pass
    return strategy.get(key, '')


def _resolve_trigger_ids(strategy: dict, key: str) -> List[str]:
    """Resolve multiple trigger IDs (exit_triggers)."""
    confluence_key = key.replace('triggers', 'trigger_confluence_ids')
    cids = strategy.get(confluence_key, [])
    if cids:
        try:
            from alerts import _get_base_trigger_id
            return [_get_base_trigger_id(c) for c in cids if c]
        except Exception:
            pass
    return [t for t in strategy.get(key, []) if t]


def resolve_strategy_requirements(strategy: dict) -> Tuple[
        Set[str], Set[str], Set[str], Dict[str, Any]]:
    """Resolve what indicators, interpreters, and triggers a strategy needs.

    Returns:
        (required_indicators, required_interpreters,
         required_triggers, indicator_params)
    """
    indicators: Set[str] = set()
    interpreters: Set[str] = set()
    triggers: Set[str] = set()
    params: Dict[str, Any] = {}

    # Always need ATR for stop calculations
    indicators.add('atr')

    def _process_trigger_id(trigger_id: str):
        """Map a trigger ID to its template and register requirements."""
        if not trigger_id:
            return
        # Strip _ib suffix for mapping
        base = trigger_id[:-3] if trigger_id.endswith('_ib') else trigger_id
        triggers.add(trigger_id)

        # Find template by matching longest prefix
        matched_template = None
        for prefix in sorted(TRIGGER_PREFIX_TO_TEMPLATE.keys(),
                             key=len, reverse=True):
            if base.startswith(prefix + '_') or base == prefix:
                matched_template = TRIGGER_PREFIX_TO_TEMPLATE[prefix]
                break

        if matched_template and matched_template in TEMPLATE_REQUIREMENTS:
            ind_set, interp_key = TEMPLATE_REQUIREMENTS[matched_template]
            indicators.update(ind_set)
            if interp_key:
                interpreters.add(interp_key)

    def _process_confluence_record(record: str):
        """Map a confluence record to interpreter requirements."""
        # Format: "1M-EMA_STACK-SML" or "GEN-TOD_NY_OPEN-IN_WINDOW"
        parts = record.split('-', 2)
        if len(parts) < 3:
            return
        tf_label, interp_key, _state = parts
        if tf_label == 'GEN':
            return  # General packs don't need indicators

        interpreters.add(interp_key)
        # Find template for this interpreter
        for tpl_key, (ind_set, ikey) in TEMPLATE_REQUIREMENTS.items():
            if ikey == interp_key:
                indicators.update(ind_set)
                break

    # Resolve entry trigger via confluence mapping
    entry = _resolve_trigger_id(strategy, 'entry_trigger')
    _process_trigger_id(entry)

    # Resolve exit triggers via confluence mapping
    exit_triggers = _resolve_trigger_ids(strategy, 'exit_triggers')
    for et in exit_triggers:
        _process_trigger_id(et)
    if not exit_triggers:
        exit_single = _resolve_trigger_id(strategy, 'exit_trigger')
        if exit_single:
            _process_trigger_id(exit_single)

    # Process confluences
    for conf in strategy.get('confluence', []):
        _process_confluence_record(conf)
    for conf in strategy.get('general_confluences', []):
        _process_confluence_record(conf)

    # Build params from strategy config
    # Default EMA periods — could be overridden by group params
    params['ema_periods'] = [8, 21, 50]

    return indicators, interpreters, triggers, params


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY MONITOR — one per strategy, owns indicator + trigger + position
# ═══════════════════════════════════════════════════════════════════════════

class StrategyMonitor:
    """Monitors one strategy: indicators → triggers → position state."""

    def __init__(self, strategy: dict,
                 position_state: Optional[PositionState] = None):
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

        logger.info("StrategyMonitor: %s (%s/%ds) — indicators=%s, "
                     "triggers=%d, interpreters=%d",
                     self.strat_name, self.symbol, self.tf_seconds,
                     req_ind, len(req_trig), len(req_interp))

    def warmup(self, df: pd.DataFrame):
        """Initialize indicator state from historical bars."""
        self.indicators.warmup(df)

    def on_bar_close(self, bar: dict,
                     bar_count: int) -> List[dict]:
        """Process a completed bar. Returns list of signals."""
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

        bar_time = bar.get('timestamp', '')
        if isinstance(bar_time, datetime):
            bar_time = bar_time.isoformat()

        # 4. Check exit (before entry — same bar exit takes priority)
        exit_sig = self.position.check_exit_bar_close(
            trigger_bools, current, bar_count, bar_time)
        if exit_sig:
            exit_sig['atr'] = current.get('atr', 0)
            signals.append(exit_sig)

        # 5. Check entry (only if FLAT after exit check)
        entry_sig = self.position.check_entry(
            trigger_bools, interps, current, bar_count, bar_time,
            self._current_confluence)
        if entry_sig:
            signals.append(entry_sig)

        return signals

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
        """
        from alerts import (save_alert, enrich_signal_with_portfolio_context,
                            load_alert_config)

        alert = {
            'type': signal['type'],
            'trigger': signal.get('trigger', ''),
            'price': signal.get('price', 0),
            'bar_time': signal.get('bar_time', ''),
            'stop_price': signal.get('stop_price'),
            'atr': signal.get('atr', 0),
            'level': 'strategy',
            'strategy_id': strategy['id'],
            'strategy_name': strategy.get('name', ''),
            'symbol': strategy.get('symbol', '?'),
            'direction': strategy.get('direction', 'LONG'),
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

        # Webhook delivery
        if self._deliver_alert_fn:
            try:
                self._deliver_alert_fn(alert, config)
            except Exception as e:
                logger.error("Webhook delivery failed: %s", e)

        return alert


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
                alert_callback: Callable = None, config: dict = None):
        """Route tick to bar builders and strategy monitors."""
        self.tick_count += 1
        self.last_tick_time = timestamp
        ts_str = timestamp.isoformat()

        for tf_seconds, builder in self.builders.items():
            completed = builder.process_tick(price, volume, timestamp)

            if completed is not None:
                # Bar close — run all monitors for this timeframe
                for monitor in self.monitors.values():
                    if monitor.tf_seconds != tf_seconds:
                        continue
                    if not _is_in_session(timestamp, monitor.session):
                        continue

                    signals = monitor.on_bar_close(
                        completed, builder._bar_count)
                    if signals and alert_callback:
                        for sig in signals:
                            alert_callback(sig, monitor.strategy, config)

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

def _is_in_session(timestamp: datetime, session: str) -> bool:
    """Check if a UTC timestamp falls within the given trading session (ET)."""
    import pytz
    import datetime as dt_mod

    SESSION_HOURS = {
        "RTH": (9, 30, 16, 0),
        "Pre-Market": (4, 0, 9, 30),
        "After Hours": (16, 0, 20, 0),
        "Extended Hours": (4, 0, 20, 0),
    }
    et = pytz.timezone("America/New_York")
    et_time = timestamp.astimezone(et).time()
    sh, sm, eh, em = SESSION_HOURS.get(session, (9, 30, 16, 0))
    return dt_mod.time(sh, sm) <= et_time < dt_mod.time(eh, em)


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
        self._start_time: Optional[str] = None
        self._last_reconcile = 0.0
        self._last_pickle_write = 0.0
        self._last_strategy_refresh = 0.0

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

                monitor = StrategyMonitor(strat, pos_state)
                hub.add_monitor(monitor)
                self.monitors[strat_id] = monitor

            self.hubs[sym] = hub

        # Load warmup data and initialize indicators
        self._warmup_all()

        # Write PID and status
        self._write_status(running=True)
        self._start_time = datetime.now(timezone.utc).isoformat()

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
        if self._stream_ref:
            try:
                self._stream_ref.close()
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

    def _on_alert(self, signal: dict, strategy: dict, config: dict):
        """Callback when a strategy monitor fires a signal."""
        alert = self.dispatcher.dispatch(signal, strategy, config or self._config)
        if alert:
            self._save_all_positions()

    async def _stream_data(self):
        """Subscribe to Alpaca real-time trade data."""
        from dotenv import load_dotenv
        load_dotenv(_SCRIPT_DIR / '.env')

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

        symbols = list(self.hubs.keys())
        backoff = 5
        max_backoff = 60

        while self._running:
            try:
                stream = StockDataStream(api_key, secret_key,
                                         feed=DataFeed.SIP)
                self._stream_ref = stream
                ws_confirmed = False

                async def on_trade(trade):
                    nonlocal ws_confirmed, backoff
                    if not self._running:
                        return
                    if not ws_confirmed:
                        ws_confirmed = True
                        backoff = 5
                        self._write_status(running=True, connected=True)
                        logger.info("WebSocket confirmed — receiving "
                                    "trades for %d symbols", len(symbols))

                    hub = self.hubs.get(trade.symbol)
                    if hub:
                        hub.on_tick(
                            price=float(trade.price),
                            volume=int(trade.size)
                            if hasattr(trade, 'size') else 1,
                            timestamp=trade.timestamp,
                            alert_callback=self._on_alert,
                            config=self._config,
                        )

                    # Periodic tasks
                    now = time.monotonic()
                    if now - self._last_pickle_write >= PICKLE_WRITE_INTERVAL:
                        self._last_pickle_write = now
                        self._write_chart_pickles()
                    if now - self._last_reconcile >= RECONCILE_INTERVAL:
                        self._last_reconcile = now
                        self._reconcile_bars()
                    if now - self._last_strategy_refresh >= STRATEGY_REFRESH_INTERVAL:
                        self._last_strategy_refresh = now
                        self._hot_reload_strategies()

                stream.subscribe_trades(on_trade, *symbols)
                logger.info("Connecting to Alpaca stream for %d symbols…",
                            len(symbols))
                await stream._run_forever()

            except Exception as e:
                self._write_status(running=True, connected=False)
                if not self._running:
                    break
                logger.warning("Stream disconnected: %s — reconnect in %ds",
                               e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

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
        from indicators import run_all_indicators
        from interpreters import detect_all_triggers

        for sym, hub in self.hubs.items():
            for tf_seconds, builder in hub.builders.items():
                df = builder.get_df_with_partial()
                if len(df) == 0:
                    continue

                max_bars = CHART_BAR_COUNTS.get(tf_seconds, DEFAULT_CHART_BARS)
                df_out = df.iloc[-max_bars:] if len(df) > max_bars else df

                # Enrich with indicators + triggers for chart display
                try:
                    df_out = run_all_indicators(df_out)
                    df_out = detect_all_triggers(df_out)
                except Exception:
                    pass  # Write un-enriched data if enrichment fails

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
                # the last N bars which may still be forming)
                hist = builder.history
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
            monitor = StrategyMonitor(strat)
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

            # If this is a new symbol, we need to re-subscribe
            # (handled on next stream reconnect)

        if added or removed:
            self._config = config
            self.strategies = strategies
            logger.info("Hot-reload complete: %d strategies active",
                       len(self.monitors))

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


def main():
    parser = argparse.ArgumentParser(
        description="Ralph Wiggum Alert Engine")
    parser.add_argument('--status', action='store_true',
                        help='Print engine status')
    parser.add_argument('--stop', action='store_true',
                        help='Stop the running engine')
    args = parser.parse_args()

    # Ensure we're in the right directory
    os.chdir(str(_SCRIPT_DIR))

    if args.status:
        cmd_status()
    elif args.stop:
        cmd_stop()
    else:
        cmd_start()


if __name__ == '__main__':
    main()
