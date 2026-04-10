"""Stop loss and take profit method registry.

Pluggable replacement for the hardcoded if/elif branches in
unified_engine.py:_compute_stop() and _compute_target().

Each method is a class with a `compute(ctx)` method. Adding a new method
type means adding a new class and registering it — no engine changes needed.

Design:
- StopMethod / TargetMethod: abstract base classes
- StopContext / TargetContext: dataclasses bundling everything a method needs
- STOP_METHODS / TARGET_METHODS: name → instance lookup tables

Backward compatibility:
- Method names match the existing hardcoded if/elif strings ('atr', 'fixed_dollar', etc.)
- Default fallback parameters match the hardcoded defaults
- Return values are computed identically to the original implementation
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional


# =============================================================================
# Context dataclasses
# =============================================================================

@dataclass
class StopContext:
    """Everything a StopMethod needs to compute the initial stop price."""
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    atr: float
    high_low_buffer: deque  # rolling (high, low) tuples for swing methods
    config: dict = field(default_factory=dict)  # method-specific params


@dataclass
class TargetContext:
    """Everything a TargetMethod needs to compute the target price."""
    entry_price: float
    stop_price: float  # required for risk_reward method
    direction: str  # 'LONG' or 'SHORT'
    atr: float
    high_low_buffer: deque
    config: dict = field(default_factory=dict)


# =============================================================================
# Stop method base + implementations
# =============================================================================

class StopMethod(ABC):
    """Base class for stop loss computation methods."""

    @abstractmethod
    def compute(self, ctx: StopContext) -> float:
        """Return the initial stop price for a new position."""
        ...


class ATRStop(StopMethod):
    """ATR-based stop: stop = entry ± (atr × multiplier)."""

    def compute(self, ctx: StopContext) -> float:
        mult = ctx.config.get('atr_mult', 1.5)
        distance = ctx.atr * mult
        return ctx.entry_price - distance if ctx.direction == 'LONG' else ctx.entry_price + distance


class FixedDollarStop(StopMethod):
    """Fixed dollar offset stop: stop = entry ± dollar_amount."""

    def compute(self, ctx: StopContext) -> float:
        distance = ctx.config.get('dollar_amount', 1.0)
        return ctx.entry_price - distance if ctx.direction == 'LONG' else ctx.entry_price + distance


class PercentageStop(StopMethod):
    """Percentage offset stop: stop = entry ± (entry × pct / 100)."""

    def compute(self, ctx: StopContext) -> float:
        pct = ctx.config.get('percentage', 0.5)
        distance = ctx.entry_price * (pct / 100.0)
        return ctx.entry_price - distance if ctx.direction == 'LONG' else ctx.entry_price + distance


class SwingStop(StopMethod):
    """Swing stop: lowest low (LONG) or highest high (SHORT) over lookback bars,
    with optional padding. Excludes the current bar from the lookback window."""

    def compute(self, ctx: StopContext) -> float:
        lookback = ctx.config.get('lookback', 5)
        padding = ctx.config.get('padding', 0.0)
        # Exclude current bar (last entry in buffer) — matches original behavior
        buf = list(ctx.high_low_buffer)[:-1]
        window = buf[-lookback:] if len(buf) >= lookback else buf
        if not window:
            # Fallback: ATR × 1.5 (same as original)
            distance = ctx.atr * 1.5
            return ctx.entry_price - distance if ctx.direction == 'LONG' else ctx.entry_price + distance
        if ctx.direction == 'LONG':
            return min(low for _, low in window) - padding
        else:
            return max(high for high, _ in window) + padding


# Registry: method name → instance
STOP_METHODS: Dict[str, StopMethod] = {
    'atr': ATRStop(),
    'fixed_dollar': FixedDollarStop(),
    'percentage': PercentageStop(),
    'swing': SwingStop(),
}


def compute_stop(ctx: StopContext) -> float:
    """Dispatch to the registered stop method. Falls back to ATR × 1.5 if unknown."""
    method_name = ctx.config.get('method', 'atr')
    method = STOP_METHODS.get(method_name)
    if method is None:
        # Fallback: ATR × 1.5 (same as original `else` branch in _compute_stop)
        distance = ctx.atr * 1.5
        return ctx.entry_price - distance if ctx.direction == 'LONG' else ctx.entry_price + distance
    return method.compute(ctx)


# =============================================================================
# Target method base + implementations
# =============================================================================

class TargetMethod(ABC):
    """Base class for take profit computation methods."""

    @abstractmethod
    def compute(self, ctx: TargetContext) -> Optional[float]:
        """Return the target price, or None if no target should be set."""
        ...


class RiskRewardTarget(TargetMethod):
    """Risk:reward target: target = entry ± (risk × rr_ratio).
    Risk = |entry_price - stop_price|."""

    def compute(self, ctx: TargetContext) -> Optional[float]:
        rr = ctx.config.get('rr_ratio', 2.0)
        risk = abs(ctx.entry_price - ctx.stop_price)
        distance = risk * rr
        return ctx.entry_price + distance if ctx.direction == 'LONG' else ctx.entry_price - distance


class ATRTarget(TargetMethod):
    """ATR-based target: target = entry ± (atr × multiplier)."""

    def compute(self, ctx: TargetContext) -> Optional[float]:
        mult = ctx.config.get('atr_mult', 2.0)
        distance = ctx.atr * mult
        return ctx.entry_price + distance if ctx.direction == 'LONG' else ctx.entry_price - distance


class FixedDollarTarget(TargetMethod):
    """Fixed dollar offset target."""

    def compute(self, ctx: TargetContext) -> Optional[float]:
        distance = ctx.config.get('dollar_amount', 2.0)
        return ctx.entry_price + distance if ctx.direction == 'LONG' else ctx.entry_price - distance


class PercentageTarget(TargetMethod):
    """Percentage offset target."""

    def compute(self, ctx: TargetContext) -> Optional[float]:
        pct = ctx.config.get('percentage', 1.0)
        distance = ctx.entry_price * (pct / 100.0)
        return ctx.entry_price + distance if ctx.direction == 'LONG' else ctx.entry_price - distance


class SwingTarget(TargetMethod):
    """Swing target: highest high (LONG) or lowest low (SHORT) over lookback bars."""

    def compute(self, ctx: TargetContext) -> Optional[float]:
        lookback = ctx.config.get('lookback', 5)
        padding = ctx.config.get('padding', 0.0)
        buf = list(ctx.high_low_buffer)[:-1]
        window = buf[-lookback:] if len(buf) >= lookback else buf
        if not window:
            return None
        if ctx.direction == 'LONG':
            return max(high for high, _ in window) + padding
        else:
            return min(low for _, low in window) - padding


# Registry: method name → instance
TARGET_METHODS: Dict[str, TargetMethod] = {
    'risk_reward': RiskRewardTarget(),
    'atr': ATRTarget(),
    'fixed_dollar': FixedDollarTarget(),
    'percentage': PercentageTarget(),
    'swing': SwingTarget(),
}


def compute_target(ctx: TargetContext) -> Optional[float]:
    """Dispatch to the registered target method. Returns None if no target configured."""
    if ctx.config is None:
        return None
    method_name = ctx.config.get('method')
    if method_name is None:
        return None
    method = TARGET_METHODS.get(method_name)
    if method is None:
        return None
    return method.compute(ctx)


# =============================================================================
# Execution type support (for stop/target hit detection)
# =============================================================================

VALID_STOP_TARGET_EXEC_TYPES = ('L', 'C', 'LC', 'CC')


def get_exec_type(config: Optional[dict]) -> str:
    """Extract exec_type from a stop/target config dict, defaulting to 'L'."""
    if not config:
        return 'L'
    et = config.get('exec_type', 'L')
    return et if et in VALID_STOP_TARGET_EXEC_TYPES else 'L'
