#!/usr/bin/env python3
"""Test M8.7 (2026-05-02) rebroadcast handling — duplicate-bar detection
in BarBuilder + IncrementalIndicatorEngine.recompute_from_history.

When Polygon WS rebroadcasts a corrected version of the most recent bar
within their 15-min FINRA late-print window, our worker must:
  1. REPLACE (not append) the history row in BarBuilder
  2. RECOMPUTE indicator state from corrected history (not double-apply)

Validates that the post-rebroadcast indicator state matches what a fresh
engine would compute against the corrected sequence — i.e., Option B
of the bug fix is mathematically correct.

Usage:
    cd src && ../.venv/bin/python test_rebroadcast_recompute.py
"""

import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, '.')

from ralph_engine import BarBuilder
from unified_engine import IncrementalIndicatorEngine


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def generate_test_bars(n: int = 50, seed: int = 42) -> list:
    """Generate a list of bar dicts with realistic OHLCV at 1Min spacing."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = close * 0.003 * rng.uniform(0.5, 1.5, n)
    high = close + spread * rng.uniform(0, 1, n)
    low = close - spread * rng.uniform(0, 1, n)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.randint(1000, 50000, n).astype(float)

    base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
    bars = []
    for i in range(n):
        ts = base + timedelta(minutes=i)
        bars.append({
            'timestamp': ts.isoformat(),
            'open': float(open_[i]),
            'high': float(high[i]),
            'low': float(low[i]),
            'close': float(close[i]),
            'volume': float(volume[i]),
        })
    return bars


def assert_close(name: str, a: float, b: float,
                 rtol: float = 1e-8, atol: float = 1e-10):
    if math.isnan(a) and math.isnan(b):
        return
    if math.isnan(a) or math.isnan(b):
        raise AssertionError(f"{name}: {a} vs {b} (one is NaN)")
    if abs(b) < atol:
        if abs(a - b) > atol:
            raise AssertionError(f"{name}: {a:.10f} vs {b:.10f}")
        return
    rel = abs(a - b) / max(abs(b), 1e-15)
    if rel > rtol:
        raise AssertionError(
            f"{name}: {a:.10f} vs {b:.10f}, rel_diff={rel:.2e}")


def state_to_dict(eng: IncrementalIndicatorEngine) -> dict:
    """Snapshot all stateful fields for comparison."""
    s = eng.state
    return {
        'ema': dict(s.ema),
        'macd_ema_fast': s.macd_ema_fast,
        'macd_ema_slow': s.macd_ema_slow,
        'macd_signal_ema': s.macd_signal_ema,
        'atr_value': s.atr_value,
        'prev_close': s.prev_close,
        'vwap_cum_pv': s.vwap_cum_pv,
        'vwap_cum_vol': s.vwap_cum_vol,
        'vwap_value': s.vwap_value,
        'utbot_atr': s.utbot_atr,
        'utbot_trail_stop': s.utbot_trail_stop,
        'utbot_direction': s.utbot_direction,
        'utbot_prev_close': s.utbot_prev_close,
        'current': dict(s.current),
        'prev_values': dict(s.prev_values),
        'prev_macd_hist': s.prev_macd_hist,
    }


def assert_state_equivalent(name: str, a: dict, b: dict, rtol: float = 1e-8):
    """Compare two state dicts; raises on first mismatch."""
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            for sub in va:
                if sub not in vb:
                    raise AssertionError(f"{name}: {key}.{sub} missing in B")
                if isinstance(va[sub], (int, float)):
                    assert_close(f"{name}: {key}.{sub}", va[sub], vb[sub], rtol)
        elif isinstance(va, (int, float)):
            assert_close(f"{name}: {key}", va, vb, rtol)
        else:
            if va != vb:
                raise AssertionError(f"{name}: {key} differ — {va!r} vs {vb!r}")


def make_engine() -> IncrementalIndicatorEngine:
    """Standard 1Min engine with EMA + MACD + ATR + VWAP for testing."""
    required = {'ema', 'macd', 'atr'}
    params = {
        'ema_periods': [9, 21],
        'macd_fast_period': 12,
        'macd_slow_period': 26,
        'macd_signal_period': 9,
        'atr_period': 14,
    }
    return IncrementalIndicatorEngine(required, params)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: BarBuilder.accept_bar duplicate detection
# ═══════════════════════════════════════════════════════════════════════════

def test_accept_bar_detects_duplicate_timestamp():
    """A second accept_bar with the same bar_start must REPLACE, not append."""
    print("Test 1: accept_bar duplicate detection...", end=" ")
    builder = BarBuilder(tf_seconds=60)
    bars = generate_test_bars(5)

    for bar in bars:
        builder.accept_bar(bar)
        assert builder.last_was_duplicate is False, \
            "fresh bar should not be flagged as duplicate"

    assert len(builder.history) == 5, f"expected 5 rows, got {len(builder.history)}"
    initial_count = builder._bar_count
    assert initial_count == 5

    # Now rebroadcast bar[4] with corrected close
    corrected = dict(bars[4])
    corrected['close'] = bars[4]['close'] + 0.50  # arbitrary correction
    corrected['volume'] = bars[4]['volume'] + 1000  # late prints add volume

    builder.accept_bar(corrected)

    assert builder.last_was_duplicate is True, \
        "rebroadcast for last history row should set last_was_duplicate=True"
    assert len(builder.history) == 5, \
        f"history should still have 5 rows (replaced, not appended), " \
        f"got {len(builder.history)}"
    assert builder._bar_count == initial_count, \
        f"bar_count should not change on duplicate, got {builder._bar_count}"

    last_row = builder.history.iloc[-1]
    assert abs(float(last_row['close']) - corrected['close']) < 1e-9, \
        f"corrected close should be in history, got {last_row['close']}"
    assert abs(float(last_row['volume']) - corrected['volume']) < 1e-9, \
        f"corrected volume should be in history, got {last_row['volume']}"

    print("PASS")


def test_accept_bar_new_bar_appends_normally():
    """A bar with a new bar_start must append + increment + flag False."""
    print("Test 2: accept_bar fresh bar appends...", end=" ")
    builder = BarBuilder(tf_seconds=60)
    bars = generate_test_bars(3)

    for bar in bars:
        builder.accept_bar(bar)

    assert builder.last_was_duplicate is False
    assert len(builder.history) == 3
    assert builder._bar_count == 3
    print("PASS")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: accept_second_bar duplicate detection
# ═══════════════════════════════════════════════════════════════════════════

def test_accept_second_bar_detects_duplicate_period():
    """A close_on_boundary=True call for the most recent history period
    must REPLACE the history row, not pollute the current partial."""
    print("Test 3: accept_second_bar duplicate detection...", end=" ")
    # 10s builder
    builder = BarBuilder(tf_seconds=10)
    base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)

    # Feed 3 full 10-sec periods of per-second bars to close 2 bars
    # Period A: 09:30:00-09:30:10 (10 per-second bars)
    # Period B: 09:30:10-09:30:20 (10 per-second bars)
    # Period C: 09:30:20+ (1 per-second bar to start the partial)
    for sec in range(21):
        ts = base + timedelta(seconds=sec)
        builder.accept_second_bar({
            'timestamp': ts.isoformat(),
            'open': 100.0 + sec * 0.01,
            'high': 100.0 + sec * 0.01,
            'low': 100.0 + sec * 0.01,
            'close': 100.0 + sec * 0.01,
            'volume': 100.0,
        }, close_on_boundary=True)
        assert builder.last_was_duplicate is False, \
            f"sec {sec}: should not flag duplicate during fresh ingestion"

    # By now, history should have closed periods A and B (period C still partial)
    assert len(builder.history) == 2, \
        f"expected 2 closed bars, got {len(builder.history)}"
    initial_count = builder._bar_count

    # Polygon rebroadcasts a corrected version of period B (the most recent
    # closed period). The bar_dict's timestamp is a per-second one within
    # period B, with corrected aggregation values for the WHOLE period.
    rebroadcast_ts = base + timedelta(seconds=15)  # mid-period B
    builder.accept_second_bar({
        'timestamp': rebroadcast_ts.isoformat(),
        'open': 100.10,
        'high': 100.50,  # corrected high (with late-print)
        'low': 100.05,
        'close': 100.45,  # corrected close
        'volume': 5000.0,  # corrected volume
    }, close_on_boundary=True)

    assert builder.last_was_duplicate is True, \
        "rebroadcast for last closed period should set last_was_duplicate=True"
    assert len(builder.history) == 2, \
        f"history should still have 2 rows after rebroadcast, got {len(builder.history)}"
    assert builder._bar_count == initial_count, \
        f"bar_count should not change on rebroadcast, got {builder._bar_count}"

    last_row = builder.history.iloc[-1]
    assert abs(float(last_row['close']) - 100.45) < 1e-9, \
        f"corrected close should be in history, got {last_row['close']}"
    assert abs(float(last_row['high']) - 100.50) < 1e-9, \
        f"corrected high should be in history, got {last_row['high']}"

    print("PASS")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: recompute_from_history yields equivalent state
# ═══════════════════════════════════════════════════════════════════════════

def test_recompute_from_history_resets_state():
    """recompute_from_history(df) on engine X must produce identical state to
    a fresh engine warmup'd with the same df."""
    print("Test 4: recompute_from_history equivalence...", end=" ")
    bars = generate_test_bars(50)
    df = pd.DataFrame([
        {'open': b['open'], 'high': b['high'], 'low': b['low'],
         'close': b['close'], 'volume': b['volume']} for b in bars
    ], index=pd.DatetimeIndex(
        [pd.Timestamp(b['timestamp']) for b in bars], name='timestamp'))

    # Engine A: fresh warmup
    engine_a = make_engine()
    engine_a.warmup(df)
    state_a = state_to_dict(engine_a)

    # Engine B: warmup with first half then recompute_from_history
    engine_b = make_engine()
    engine_b.warmup(df.iloc[:25])
    # Apply some incremental updates to dirty state
    for i in range(25, 35):
        bar = bars[i]
        engine_b.update_bar({
            'open': bar['open'], 'high': bar['high'], 'low': bar['low'],
            'close': bar['close'], 'volume': bar['volume'],
            'timestamp': pd.Timestamp(bar['timestamp']),
        })
    # Now recompute from full corrected history
    engine_b.recompute_from_history(df)
    state_b = state_to_dict(engine_b)

    assert_state_equivalent("recompute equivalence", state_a, state_b, rtol=1e-8)
    print("PASS")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: End-to-end rebroadcast scenario
# ═══════════════════════════════════════════════════════════════════════════

def test_rebroadcast_correction_yields_clean_state():
    """End-to-end:
       Engine A: warmup with CORRECTED sequence (bar X has corrected close)
       Engine B: warmup with ORIGINAL sequence, then receive a rebroadcast
                 for bar X (BarBuilder replaces history), call recompute
       Final states must match.
    """
    print("Test 5: end-to-end rebroadcast scenario...", end=" ")
    n = 30
    bars_orig = generate_test_bars(n)
    # Last bar gets a correction
    bars_corrected = list(bars_orig)
    last = dict(bars_orig[-1])
    last['close'] = last['close'] + 0.50
    last['volume'] = last['volume'] + 5000
    bars_corrected[-1] = last

    # Engine A path: fresh feed with corrected sequence via BarBuilder
    builder_a = BarBuilder(tf_seconds=60)
    for bar in bars_corrected:
        builder_a.accept_bar(bar)
    engine_a = make_engine()
    engine_a.warmup(builder_a.history)
    state_a = state_to_dict(engine_a)

    # Engine B path: feed original sequence, then rebroadcast last bar
    builder_b = BarBuilder(tf_seconds=60)
    for bar in bars_orig:
        builder_b.accept_bar(bar)
    engine_b = make_engine()
    engine_b.warmup(builder_b.history)

    # Simulate Polygon rebroadcast for the last bar
    builder_b.accept_bar(last)
    assert builder_b.last_was_duplicate is True, \
        "rebroadcast must be detected as duplicate"

    # Caller's responsibility post-duplicate: recompute indicators
    engine_b.recompute_from_history(builder_b.history)
    state_b = state_to_dict(engine_b)

    # Final states should be equivalent
    assert_state_equivalent("end-to-end", state_a, state_b, rtol=1e-8)

    # Sanity: confirm builder_b's history reflects the corrected last bar
    last_row = builder_b.history.iloc[-1]
    assert abs(float(last_row['close']) - last['close']) < 1e-9
    assert abs(float(last_row['volume']) - last['volume']) < 1e-9
    assert len(builder_b.history) == n  # NOT n+1 — was a replace, not an append

    print("PASS")


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def run_all():
    tests = [
        test_accept_bar_detects_duplicate_timestamp,
        test_accept_bar_new_bar_appends_normally,
        test_accept_second_bar_detects_duplicate_period,
        test_recompute_from_history_resets_state,
        test_rebroadcast_correction_yields_clean_state,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"ERROR: {t.__name__} — {type(e).__name__}: {e}")
            failed.append(t.__name__)
    print()
    if failed:
        print(f"{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"{len(tests)} passed, 0 failed")


if __name__ == '__main__':
    run_all()
