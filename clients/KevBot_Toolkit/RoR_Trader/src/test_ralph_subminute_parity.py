"""Parity test for Ralph's BarBuilder.accept_second_bar vs backtest's resample.

Locks bar-for-bar equality between:
  - Backtest path: data_loader.resample_to_timeframe(df_1s, target_tf)
  - Live path:     BarBuilder.accept_second_bar() called per-second

Any future change to either path that breaks parity will fail this test.
This is THE safety net for backtest↔live divergence on sub-minute strategies.

Run:  cd src && ../.venv/bin/python test_ralph_subminute_parity.py
Or:   cd src && ../.venv/bin/pytest test_ralph_subminute_parity.py -v
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data_loader import resample_to_timeframe
from ralph_engine import BarBuilder


# =============================================================================
# Fixtures
# =============================================================================

def _make_synthetic_1s(seed: int = 42, n_seconds: int = 600,
                       start: str = "2026-04-14T13:30:00+00:00") -> pd.DataFrame:
    """Generate realistic per-second OHLCV bars.

    n_seconds=600 gives us 10 minutes of data — enough to cover at least 60
    aggregated 10s bars, 40 aggregated 15s bars, 20 aggregated 30s bars, and
    10 aggregated 1Min bars. Includes random walk, varying volume, and
    intra-second high/low ranges (NOT just close-derived) so a bug like
    'using close as high' would fail.
    """
    rng = np.random.default_rng(seed)
    base = 500.0
    rows = []
    t0 = pd.Timestamp(start)
    for i in range(n_seconds):
        # Random walk on close
        delta = rng.normal(0, 0.05)
        base = max(1.0, base + delta)
        close = base
        # Realistic intra-second range: open near previous close,
        # high and low both extend beyond [open, close]
        open_ = close - rng.normal(0, 0.04)
        spread = abs(rng.normal(0.08, 0.04))
        high = max(open_, close) + spread
        low = min(open_, close) - spread
        volume = float(rng.integers(100, 5000))
        ts = t0 + timedelta(seconds=i)
        rows.append({
            'open': open_, 'high': high, 'low': low,
            'close': close, 'volume': volume,
        })
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(
        [t0 + timedelta(seconds=i) for i in range(n_seconds)],
        name='timestamp', tz='UTC',
    )
    return df


def _make_synthetic_with_gaps(seed: int = 7, n_seconds: int = 300) -> pd.DataFrame:
    """Synthetic 1s bars with random missing seconds — exercises the
    no-gap-fill behavior that backtest's resample().dropna() relies on."""
    full = _make_synthetic_1s(seed=seed, n_seconds=n_seconds)
    rng = np.random.default_rng(seed + 100)
    keep_mask = rng.random(len(full)) > 0.15  # drop ~15% of seconds
    return full[keep_mask].copy()


# =============================================================================
# Helpers
# =============================================================================

def _ralph_aggregate(df_1s: pd.DataFrame, tf_seconds: int) -> pd.DataFrame:
    """Run BarBuilder.accept_second_bar() over a 1s DataFrame and return the
    resulting completed-bar DataFrame in the same shape as resample output.

    Note: backtest's resample includes the FINAL bar even if not fully complete
    within the window. To mirror, we capture builder._partial at end and
    treat it as the final completed bar (matches resample semantics).
    """
    builder = BarBuilder(tf_seconds=tf_seconds)
    completed_bars = []
    for ts, row in df_1s.iterrows():
        bar = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
            'timestamp': ts,
        }
        result = builder.accept_second_bar(bar)
        if result is not None:
            completed_bars.append(result)
    # Capture the final partial as an in-progress bar (resample includes it)
    if builder._partial is not None:
        completed_bars.append(builder._partial.to_dict())

    if not completed_bars:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    out = pd.DataFrame([
        {
            'open': b['open'], 'high': b['high'], 'low': b['low'],
            'close': b['close'], 'volume': b['volume'],
            'timestamp': pd.Timestamp(b['timestamp']),
        }
        for b in completed_bars
    ])
    out = out.set_index('timestamp')
    if out.index.tz is None:
        out.index = out.index.tz_localize('UTC')
    return out


def _backtest_resample(df_1s: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Run backtest's resample, drop trade_count if present (Ralph doesn't
    propagate it through the partial), and return only OHLCV columns."""
    out = resample_to_timeframe(df_1s, target_tf)
    keep = ['open', 'high', 'low', 'close', 'volume']
    out = out[[c for c in keep if c in out.columns]]
    return out


def _assert_parity(df_ralph: pd.DataFrame, df_backtest: pd.DataFrame,
                   label: str = ""):
    """Bar-for-bar OHLCV equality + matching index."""
    assert len(df_ralph) == len(df_backtest), (
        f"{label}: bar count mismatch — ralph={len(df_ralph)}, "
        f"backtest={len(df_backtest)}\n"
        f"ralph idx head: {list(df_ralph.index[:5])}\n"
        f"backtest idx head: {list(df_backtest.index[:5])}"
    )
    # Compare index timestamps
    assert (df_ralph.index == df_backtest.index).all(), (
        f"{label}: timestamp index mismatch.\n"
        f"diffs at: {df_ralph.index[df_ralph.index != df_backtest.index]}"
    )
    # Compare OHLCV column-by-column
    for col in ['open', 'high', 'low', 'close', 'volume']:
        a = df_ralph[col].values
        b = df_backtest[col].values
        if not np.allclose(a, b, rtol=0, atol=1e-9):
            mismatches = np.where(~np.isclose(a, b, rtol=0, atol=1e-9))[0]
            head = mismatches[:5]
            raise AssertionError(
                f"{label}: {col} mismatch at {len(mismatches)} bars. "
                f"First mismatches (idx, ralph, backtest):\n"
                + "\n".join(
                    f"  [{i}] ts={df_ralph.index[i]} "
                    f"ralph={a[i]:.6f} backtest={b[i]:.6f}"
                    for i in head
                )
            )


# =============================================================================
# Tests
# =============================================================================

def test_5s_aggregation_continuous():
    df_1s = _make_synthetic_1s(n_seconds=600)
    ralph = _ralph_aggregate(df_1s, tf_seconds=5)
    backtest = _backtest_resample(df_1s, '5Sec')
    _assert_parity(ralph, backtest, label="5Sec continuous")


def test_10s_aggregation_continuous():
    df_1s = _make_synthetic_1s(n_seconds=600)
    ralph = _ralph_aggregate(df_1s, tf_seconds=10)
    backtest = _backtest_resample(df_1s, '10Sec')
    _assert_parity(ralph, backtest, label="10Sec continuous")


def test_15s_aggregation_continuous():
    df_1s = _make_synthetic_1s(n_seconds=600)
    ralph = _ralph_aggregate(df_1s, tf_seconds=15)
    backtest = _backtest_resample(df_1s, '15Sec')
    _assert_parity(ralph, backtest, label="15Sec continuous")


def test_30s_aggregation_continuous():
    df_1s = _make_synthetic_1s(n_seconds=600)
    ralph = _ralph_aggregate(df_1s, tf_seconds=30)
    backtest = _backtest_resample(df_1s, '30Sec')
    _assert_parity(ralph, backtest, label="30Sec continuous")


def test_10s_aggregation_with_gaps():
    """Per-second feed with ~15% missing seconds — exercises the
    no-gap-fill behavior that mirrors backtest's resample.dropna()."""
    df_1s = _make_synthetic_with_gaps(n_seconds=600)
    ralph = _ralph_aggregate(df_1s, tf_seconds=10)
    backtest = _backtest_resample(df_1s, '10Sec')
    _assert_parity(ralph, backtest, label="10Sec with gaps")


def test_aggregation_uses_intra_bar_high_low_not_close():
    """Critical regression test: a naive implementation that uses only
    `close` to track high/low (process_tick semantics) would fail this.
    The per-second bars below have intra-second highs/lows that exceed
    closes, and accept_second_bar must capture them."""
    t0 = pd.Timestamp("2026-04-14T13:30:00+00:00")
    rows = [
        # open  high  low   close volume — close is FLAT but high/low spike
        (100.0, 105.0, 99.0, 100.0, 1000),
        (100.0, 110.0, 95.0, 100.0, 1000),
        (100.0, 102.0, 98.0, 100.0, 1000),
        (100.0, 100.5, 99.5, 100.0, 1000),
        (100.0, 103.0, 97.0, 100.0, 1000),
    ]
    df_1s = pd.DataFrame(
        [{'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
         for o, h, l, c, v in rows],
        index=pd.DatetimeIndex(
            [t0 + timedelta(seconds=i) for i in range(5)],
            name='timestamp', tz='UTC',
        ),
    )
    ralph = _ralph_aggregate(df_1s, tf_seconds=5)
    backtest = _backtest_resample(df_1s, '5Sec')
    # Both should produce ONE bar with high=110, low=95 (the spikes)
    assert ralph.iloc[0]['high'] == 110.0, f"Ralph high lost the spike: {ralph.iloc[0]['high']}"
    assert ralph.iloc[0]['low'] == 95.0, f"Ralph low lost the spike: {ralph.iloc[0]['low']}"
    _assert_parity(ralph, backtest, label="intra-bar high/low")


# =============================================================================
# Self-execution path (no pytest required)
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_5s_aggregation_continuous,
        test_10s_aggregation_continuous,
        test_15s_aggregation_continuous,
        test_30s_aggregation_continuous,
        test_10s_aggregation_with_gaps,
        test_aggregation_uses_intra_bar_high_low_not_close,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}\n  {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}\n  {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
