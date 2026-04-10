"""Parity test for stop_target_methods.py refactor.

Verifies that the new pluggable method registry produces IDENTICAL stop and
target prices to the old hardcoded if/elif branches in
unified_engine.py:_compute_stop() and _compute_target().

Run: cd src && ../.venv/bin/python test_stop_target_parity.py
"""

from collections import deque
from stop_target_methods import StopContext, TargetContext, compute_stop, compute_target


# =============================================================================
# Reference implementations (copied verbatim from unified_engine.py BEFORE refactor)
# =============================================================================

def _compute_stop_OLD(stop_config, entry_price, atr, direction, high_low_buffer):
    """Original hardcoded if/elif from unified_engine.py:_compute_stop()."""
    method = stop_config.get('method', 'atr')

    if method == 'atr':
        mult = stop_config.get('atr_mult', 1.5)
        distance = atr * mult
    elif method == 'fixed_dollar':
        distance = stop_config.get('dollar_amount', 1.0)
    elif method == 'percentage':
        pct = stop_config.get('percentage', 0.5)
        distance = entry_price * (pct / 100.0)
    elif method == 'swing':
        lookback = stop_config.get('lookback', 5)
        padding = stop_config.get('padding', 0.0)
        buf = list(high_low_buffer)[:-1]
        window = buf[-lookback:] if len(buf) >= lookback else buf
        if not window:
            distance = atr * 1.5
        elif direction == 'LONG':
            return min(low for _, low in window) - padding
        else:
            return max(high for high, _ in window) + padding
    else:
        distance = atr * 1.5

    if direction == 'LONG':
        return entry_price - distance
    else:
        return entry_price + distance


def _compute_target_OLD(target_config, entry_price, stop_price, atr, direction, high_low_buffer):
    """Original hardcoded if/elif from unified_engine.py:_compute_target()."""
    if target_config is None:
        return None

    method = target_config.get('method')
    if method is None:
        return None

    risk = abs(entry_price - stop_price)

    if method == 'risk_reward':
        rr = target_config.get('rr_ratio', 2.0)
        distance = risk * rr
    elif method == 'atr':
        mult = target_config.get('atr_mult', 2.0)
        distance = atr * mult
    elif method == 'fixed_dollar':
        distance = target_config.get('dollar_amount', 2.0)
    elif method == 'percentage':
        pct = target_config.get('percentage', 1.0)
        distance = entry_price * (pct / 100.0)
    elif method == 'swing':
        lookback = target_config.get('lookback', 5)
        padding = target_config.get('padding', 0.0)
        buf = list(high_low_buffer)[:-1]
        window = buf[-lookback:] if len(buf) >= lookback else buf
        if not window:
            return None
        elif direction == 'LONG':
            return max(high for high, _ in window) + padding
        else:
            return min(low for _, low in window) - padding
    else:
        return None

    if direction == 'LONG':
        return entry_price + distance
    else:
        return entry_price - distance


# =============================================================================
# New implementations (via the registry)
# =============================================================================

def _compute_stop_NEW(stop_config, entry_price, atr, direction, high_low_buffer):
    ctx = StopContext(
        entry_price=entry_price,
        direction=direction,
        atr=atr,
        high_low_buffer=high_low_buffer,
        config=stop_config,
    )
    return compute_stop(ctx)


def _compute_target_NEW(target_config, entry_price, stop_price, atr, direction, high_low_buffer):
    if target_config is None:
        return None
    ctx = TargetContext(
        entry_price=entry_price,
        stop_price=stop_price,
        direction=direction,
        atr=atr,
        high_low_buffer=high_low_buffer,
        config=target_config,
    )
    return compute_target(ctx)


# =============================================================================
# Test cases
# =============================================================================

def make_swing_buffer():
    """Build a sample high/low buffer with 10 bars + 1 current."""
    buf = deque(maxlen=50)
    # 10 historical bars
    bars = [
        (100.5, 99.5),  # 0
        (101.0, 100.0),  # 1
        (101.5, 100.5),  # 2
        (102.0, 101.0),  # 3
        (101.8, 100.8),  # 4
        (102.5, 101.2),  # 5
        (103.0, 101.5),  # 6
        (102.7, 101.7),  # 7
        (103.5, 102.0),  # 8
        (104.0, 102.5),  # 9
    ]
    for h, l in bars:
        buf.append((h, l))
    # Current bar (excluded by [:-1] slice)
    buf.append((104.5, 103.0))
    return buf


def assert_close(a, b, label, tol=1e-10):
    if a is None and b is None:
        return
    if a is None or b is None:
        raise AssertionError(f"{label}: one is None — old={a} new={b}")
    if abs(a - b) > tol:
        raise AssertionError(f"{label}: mismatch — old={a} new={b} delta={abs(a-b)}")


def test_stop_methods():
    """Verify each stop method produces identical results."""
    buf = make_swing_buffer()
    test_cases = [
        # ATR
        ({'method': 'atr', 'atr_mult': 1.5}, 100.0, 1.0, 'LONG'),
        ({'method': 'atr', 'atr_mult': 1.5}, 100.0, 1.0, 'SHORT'),
        ({'method': 'atr', 'atr_mult': 2.0}, 250.75, 0.823, 'LONG'),
        ({'method': 'atr'}, 100.0, 1.5, 'LONG'),  # default mult
        # Fixed dollar
        ({'method': 'fixed_dollar', 'dollar_amount': 0.50}, 100.0, 1.0, 'LONG'),
        ({'method': 'fixed_dollar', 'dollar_amount': 2.50}, 100.0, 1.0, 'SHORT'),
        ({'method': 'fixed_dollar'}, 100.0, 1.0, 'LONG'),  # default $1.00
        # Percentage
        ({'method': 'percentage', 'percentage': 1.0}, 100.0, 1.0, 'LONG'),
        ({'method': 'percentage', 'percentage': 0.25}, 250.0, 1.0, 'SHORT'),
        ({'method': 'percentage'}, 100.0, 1.0, 'LONG'),  # default 0.5%
        # Swing
        ({'method': 'swing', 'lookback': 5, 'padding': 0.05}, 103.5, 1.0, 'LONG', buf),
        ({'method': 'swing', 'lookback': 5, 'padding': 0.05}, 103.5, 1.0, 'SHORT', buf),
        ({'method': 'swing', 'lookback': 10}, 103.5, 1.0, 'LONG', buf),
        ({'method': 'swing'}, 103.5, 1.0, 'LONG', buf),  # all defaults
        # Swing with empty buffer (fallback to ATR×1.5)
        ({'method': 'swing', 'lookback': 5}, 100.0, 2.0, 'LONG', deque(maxlen=50)),
        # Unknown method (fallback)
        ({'method': 'unknown_method'}, 100.0, 1.5, 'LONG'),
    ]
    failures = 0
    for tc in test_cases:
        cfg, entry, atr, direction = tc[:4]
        b = tc[4] if len(tc) > 4 else deque(maxlen=50)
        old = _compute_stop_OLD(cfg, entry, atr, direction, b)
        new = _compute_stop_NEW(cfg, entry, atr, direction, b)
        try:
            assert_close(old, new, f"stop {cfg} entry={entry}")
            print(f"  PASS  {cfg} entry={entry} dir={direction} → {old}")
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failures += 1
    return failures


def test_target_methods():
    """Verify each target method produces identical results."""
    buf = make_swing_buffer()
    test_cases = [
        # No target config
        (None, 100.0, 99.0, 1.0, 'LONG'),
        # Risk:reward
        ({'method': 'risk_reward', 'rr_ratio': 2.0}, 100.0, 99.0, 1.0, 'LONG'),
        ({'method': 'risk_reward', 'rr_ratio': 3.0}, 100.0, 101.0, 1.0, 'SHORT'),
        ({'method': 'risk_reward'}, 100.0, 99.5, 1.0, 'LONG'),  # default 2.0
        # ATR
        ({'method': 'atr', 'atr_mult': 2.0}, 100.0, 99.0, 1.5, 'LONG'),
        ({'method': 'atr', 'atr_mult': 3.0}, 100.0, 101.0, 1.5, 'SHORT'),
        ({'method': 'atr'}, 100.0, 99.0, 1.5, 'LONG'),  # default 2.0
        # Fixed dollar
        ({'method': 'fixed_dollar', 'dollar_amount': 2.5}, 100.0, 99.0, 1.0, 'LONG'),
        ({'method': 'fixed_dollar'}, 100.0, 99.0, 1.0, 'LONG'),
        # Percentage
        ({'method': 'percentage', 'percentage': 2.0}, 100.0, 99.0, 1.0, 'LONG'),
        ({'method': 'percentage'}, 100.0, 99.0, 1.0, 'LONG'),
        # Swing
        ({'method': 'swing', 'lookback': 5, 'padding': 0.05}, 103.5, 102.0, 1.0, 'LONG', buf),
        ({'method': 'swing', 'lookback': 5}, 103.5, 102.0, 1.0, 'SHORT', buf),
        ({'method': 'swing'}, 103.5, 102.0, 1.0, 'LONG', buf),
        # Empty swing → None
        ({'method': 'swing', 'lookback': 5}, 100.0, 99.0, 1.0, 'LONG', deque(maxlen=50)),
        # Unknown method → None
        ({'method': 'unknown'}, 100.0, 99.0, 1.0, 'LONG'),
        # No method
        ({'lookback': 5}, 100.0, 99.0, 1.0, 'LONG'),
    ]
    failures = 0
    for tc in test_cases:
        cfg, entry, stop, atr, direction = tc[:5]
        b = tc[5] if len(tc) > 5 else deque(maxlen=50)
        old = _compute_target_OLD(cfg, entry, stop, atr, direction, b)
        new = _compute_target_NEW(cfg, entry, stop, atr, direction, b)
        try:
            assert_close(old, new, f"target {cfg} entry={entry} stop={stop}")
            print(f"  PASS  {cfg} entry={entry} stop={stop} dir={direction} → {old}")
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failures += 1
    return failures


if __name__ == '__main__':
    print("=== Stop Method Parity ===")
    stop_fails = test_stop_methods()
    print()
    print("=== Target Method Parity ===")
    target_fails = test_target_methods()
    print()
    total = stop_fails + target_fails
    if total == 0:
        print("ALL PARITY TESTS PASSED")
    else:
        print(f"FAILURES: {total}")
        exit(1)
