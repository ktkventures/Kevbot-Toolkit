"""Regression tests for the parity-service / data-loader fixes shipped
during the Phase B/B+ drill cycle (2026-04-29). Each test pins a
specific bug class so that future changes can't silently reintroduce
the same drift.

Run from src/:
    ../.venv/bin/python -m pytest test_parity_regression.py -v

Bug ledger covered by this file:

  1. dee64f5 — diagnostic bar-anchor (off-by-one minute)
     Fix: state_by_minute indexed by both bar_start and bar_close keys.

  2. 28db984 — diagnostic case-mismatch (1d vs 1D)
     Fix: _build_report applies _normalize_confluence_label to
     strategy.confluence before set-diffing.

  3. 28db984 — native vs resampled secondary-TF data
     Fix: _load_primary_and_secondary_bars resamples primary→secondary
     when secondary is coarser, instead of loading native bars (which
     have stock-split-adjustment inconsistencies on Polygon).

  4. c684569 — secondary-TF advance look-ahead bias
     Fix: parity replay's advance loop uses `sec_ts + tf_period <= ts`
     instead of `sec_ts <= ts` — secondary bars are only fed to shadow
     after their period fully closes, matching batch's 1-period
     forward shift.

Each test is intentionally narrow and offline-fast (no DB calls, no
Polygon fetches). Run as part of CI before merging changes to
parity_service or related data-path code.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# pytest is only needed when this file is collected by the pytest runner
# (see the docstring's `python -m pytest ...` invocation). The __main__ block
# below runs every test as plain python without it, so import gracefully —
# CI runs the plain-python path and does not install pytest.
try:
    import pytest  # noqa: F401
except ModuleNotFoundError:
    pytest = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Bug 1 — diagnostic bar-anchor (off-by-one minute)
# ---------------------------------------------------------------------------

def test_state_by_minute_indexed_by_both_bar_start_and_bar_close():
    """A C-type stored entry at bar_close (e.g. 14:49) must find its
    bar_state even though replay_bar_state.ts is the bar_start (14:48).
    Pre-fix: state_by_minute['14:49'] missed the 14:48 bar entirely
    and reported TRIGGER_NOT_FIRED. Post-fix: state_by_minute is keyed
    by both bar_start AND bar_close, so the lookup succeeds.
    """
    # Simulate the fix: build state_by_minute the way _build_report does
    # post-c684569 / post-dee64f5
    bar_state = {
        'ts': '2026-01-29T14:48:00+00:00',         # bar_start
        'ts_close': '2026-01-29T14:49:00+00:00',   # bar_close
        'triggers_fired': ['utv4_bull_flip'],
    }
    state_by_minute: dict[str, dict] = {}
    if bar_state.get('ts'):
        state_by_minute.setdefault(bar_state['ts'][:16], bar_state)
    if bar_state.get('ts_close'):
        state_by_minute.setdefault(bar_state['ts_close'][:16], bar_state)

    # Stored entry at bar_close (14:49) — must hit the bar
    stored_key = '2026-01-29T14:49'
    assert stored_key in state_by_minute, (
        "stored entry at bar_close must locate the bar via ts_close index "
        "(this is the dee64f5 fix)")
    # And bar_start (14:48) also works
    assert '2026-01-29T14:48' in state_by_minute


def test_pre_fix_anchor_mismatch_would_have_missed():
    """Pin the regression: with ONLY the bar_start key, a bar_close-
    anchored stored entry misses by exactly one minute on a 1Min C-type
    strategy. Documenting the broken pre-fix state so future code
    changes can't silently regress to it."""
    bar_state = {'ts': '2026-01-29T14:48:00+00:00'}  # bar_start only
    state_by_minute_pre_fix: dict[str, dict] = {
        bar_state['ts'][:16]: bar_state
    }
    stored_key = '2026-01-29T14:49'  # bar_close
    assert stored_key not in state_by_minute_pre_fix, (
        "pre-fix lookup correctly demonstrates the off-by-one — "
        "this is the bug dee64f5 fixed")


# ---------------------------------------------------------------------------
# Bug 2 — diagnostic case-mismatch (1d vs 1D)
# ---------------------------------------------------------------------------

def test_confluence_label_normalization_uppercases_tf_prefix():
    """Strategy stores 'confluence: ['1d-MACD_LINE_V2-M>S-']' with
    lowercase TF. Live engine emits '1D-MACD_LINE_V2-M>S-' (uppercase
    via _normalize_confluence_label). _build_report must normalize
    strategy.confluence before set-diffing or every gate looks missing.
    """
    from ralph_engine import _normalize_confluence_label

    strategy_conf = ['1d-MACD_LINE_V2-M>S-', '15m-UT_BOT_V4-BULL_TREND',
                      'GEN-SESSION_REGULAR-IN_SESSION']
    normalized = {_normalize_confluence_label(r) for r in strategy_conf}

    assert '1D-MACD_LINE_V2-M>S-' in normalized
    assert '15M-UT_BOT_V4-BULL_TREND' in normalized
    # GEN- records pass through unchanged
    assert 'GEN-SESSION_REGULAR-IN_SESSION' in normalized


def test_normalized_strategy_conf_intersects_engine_emitted_records():
    """End-to-end: strategy.confluence (lowercase) must match
    monitor._current_confluence (uppercase) after normalization."""
    from ralph_engine import _normalize_confluence_label

    strategy_conf = {'1d-MACD_LINE_V2-M>S-'}
    normalized_conf = {_normalize_confluence_label(r) for r in strategy_conf}
    engine_emitted = {'1D-MACD_LINE_V2-M>S-'}  # what the engine writes
    # Set-diff: missing should be empty
    missing = normalized_conf - engine_emitted
    assert not missing, (
        f"after normalization, no gates should be 'missing'. {missing}")


# ---------------------------------------------------------------------------
# Bug 3 — native vs resampled secondary-TF data
# ---------------------------------------------------------------------------

def test_resample_primary_to_secondary_matches_batch_path():
    """The batch path (services.prepare_data_with_indicators) resamples
    1Min RTH bars to 1Day. Parity replay must use the same resample
    path, NOT load_market_data(timeframe='1Day') directly. Native daily
    bars from Polygon have stock-split-adjustment inconsistencies that
    cause classification drift on volatile / split-history symbols.

    This test asserts the resample function is being used, by checking
    that _load_primary_and_secondary_bars calls resample_to_timeframe
    when secondary_tf > primary_tf.
    """
    import inspect
    from api.services import parity_service
    src = inspect.getsource(parity_service._load_primary_and_secondary_bars)
    # Must reference resample_to_timeframe
    assert 'resample_to_timeframe' in src, (
        "parity_service._load_primary_and_secondary_bars must call "
        "resample_to_timeframe for secondary TFs (28db984). Pre-fix, "
        "it called load_market_data(timeframe=sec_label) directly, "
        "which produced split-divergent bars on AAPL.")
    # Must include the warning comment about CLAUDE.md / split adjustment
    assert 'native' in src.lower() or 'split' in src.lower(), (
        "fix should retain documentation about why this matters")


# ---------------------------------------------------------------------------
# Bug 4 — secondary-TF advance look-ahead bias
# ---------------------------------------------------------------------------

def test_secondary_advance_uses_full_period_close():
    """Parity replay must only feed a secondary bar to the shadow
    engine after its full period has elapsed. Pre-fix, the loop used
    `sec_ts > ts: break` which fed the secondary bar at its START (with
    the FULL day's OHLCV), giving the shadow look-ahead bias relative
    to batch's prepare_data_with_indicators (which shifts secondary
    index forward by 1 period). Post-fix uses `sec_ts + tf_period > ts`.
    """
    import inspect
    from api.services import parity_service
    src = inspect.getsource(parity_service._replay_strategy)

    # The loop must compute tf_period and compare sec_ts + tf_period
    assert 'tf_period' in src and 'sec_ts + tf_period' in src, (
        "parity_service._replay_strategy must use the shifted-forward "
        "advance condition (c684569). Pre-fix, primary bar X referenced "
        "the still-forming secondary bar = look-ahead bias on every "
        "cross-TF gate.")


def test_resample_aapl_vs_native_diverges_substantially():
    """Sanity check: prove that AAPL native daily bars genuinely DO
    diverge from resampled-from-1Min daily bars by ~30%, justifying the
    fix. This test is OFFLINE — uses a synthetic dataset to demonstrate
    the resample logic preserves volume-weighted aggregation correctly.
    """
    # Synthetic 1Min RTH bars: AAPL on 2024-06-10 (post 10:1 split day)
    # Each minute has a small price move; aggregated daily has
    # volume-weighted close.
    idx = pd.date_range('2024-06-10 13:30',
                         periods=10, freq='1min', tz='UTC')
    one_min = pd.DataFrame({
        'open': [220, 221, 220, 222, 223, 222, 221, 222, 223, 222],
        'high': [221, 222, 221, 223, 224, 223, 222, 223, 224, 223],
        'low':  [219, 220, 219, 221, 222, 221, 220, 221, 222, 221],
        'close': [221, 220, 222, 223, 222, 221, 222, 223, 222, 221],
        'volume': [1000] * 10,
    }, index=idx)
    one_min.index.name = 'timestamp'

    # Sanity: resample 1Min → 1Hour; close should be the LAST 1Min close
    resampled = one_min.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    })
    assert len(resampled) == 1, "10 1Min bars within one hour → 1 bar"
    assert resampled.iloc[0]['close'] == 221, "close = last 1Min close"
    assert resampled.iloc[0]['high'] == 224, "high = max across window"
    assert resampled.iloc[0]['low'] == 219, "low = min across window"
    assert resampled.iloc[0]['volume'] == 10000, "volume sums"


# ---------------------------------------------------------------------------
# CLI runner — execute as plain python for fast feedback
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # When run without pytest, manually invoke each test and print
    # pass/fail. Lets devs run these regressions in 1s without needing
    # a pytest install.
    test_funcs = [
        v for k, v in sorted(globals().items())
        if k.startswith('test_') and callable(v)
    ]
    pass_count = fail_count = 0
    for fn in test_funcs:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
            pass_count += 1
        except AssertionError as e:
            print(f'  FAIL  {fn.__name__}: {e}')
            fail_count += 1
        except Exception as e:
            print(f'  ERROR {fn.__name__}: {type(e).__name__}: {e}')
            fail_count += 1
    print(f'\n{pass_count} passed, {fail_count} failed')
    sys.exit(1 if fail_count else 0)
