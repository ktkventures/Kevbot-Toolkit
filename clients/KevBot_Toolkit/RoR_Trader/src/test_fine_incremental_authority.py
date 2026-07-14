"""Lock RORT_MTF_FINE_INCREMENTAL_AUTHORITY (2026-07-14).

The ≥60s fan-out DUPLICATE branch used to recompute fine-TF RTH shadow
records from `sec_builder.history` (shallow, fan-out-flavored series —
a different anchor than the backtest-family warmup) and REPUBLISH on
every AM-after-ws_agg duplicate (~every 2-3 min per TF). Path-dependent
interpreters (SWING_123 counts) land off-by-one from that anchor, so the
wrong-family state OWNED the gate key's steady state between boundary
closes — the live 5m key sat at BULL_C3 for 50 min on 2026-07-13 while
settled truth said NEUTRAL/BULL_C2 (327/328/329's whole gate surface).

Flag ON: the duplicate branch SKIPS recompute AND republish for fine RTH
shadows — the boot-seeded deep-anchor incremental state stays
authoritative; the thread-side refresher (RORT_MTF_STATE_REFRESH_S) does
canonical re-trues. Coarse shadows keep their CLASS 1a reload. Flag OFF:
byte-identical legacy.

Run:  cd src && ../.venv/bin/pytest test_fine_incremental_authority.py -v
"""

import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import ralph_engine  # noqa: E402
from ralph_engine import (  # noqa: E402
    BarBuilder, SymbolHub, _ShadowIndicatorEngine)

PARAMS = {
    'ema_periods': [9, 21],
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'atr_period': 14,
}
REQUIRED = {'ema', 'macd', 'atr'}
SEC_TF = 120
T0 = pd.Timestamp("2026-04-14T13:30:00+00:00")


@contextmanager
def flag(value: str):
    old = os.environ.get('RORT_MTF_FINE_INCREMENTAL_AUTHORITY')
    os.environ['RORT_MTF_FINE_INCREMENTAL_AUTHORITY'] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('RORT_MTF_FINE_INCREMENTAL_AUTHORITY', None)
        else:
            os.environ['RORT_MTF_FINE_INCREMENTAL_AUTHORITY'] = old


def make_min_bars(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    idx = pd.date_range(T0, periods=n, freq='60s', tz='UTC')
    return pd.DataFrame({
        'open': close - 0.02, 'high': close + 0.05,
        'low': close - 0.05, 'close': close,
        'volume': rng.randint(1000, 5000, n).astype(float),
    }, index=idx)


def bar_dict(df: pd.DataFrame, i: int) -> dict:
    row = df.iloc[i]
    return {
        'timestamp': df.index[i].isoformat(),
        'open': float(row['open']), 'high': float(row['high']),
        'low': float(row['low']), 'close': float(row['close']),
        'volume': float(row['volume']),
    }


def build_hub_with_fine_shadow():
    hub = SymbolHub('TEST')
    hub.builders[SEC_TF] = BarBuilder(tf_seconds=SEC_TF)
    shadow = _ShadowIndicatorEngine(
        SEC_TF, set(REQUIRED), {'MACD_LINE'}, dict(PARAMS))
    hub._shadow_engines[(SEC_TF, 'RTH')] = shadow
    return hub, shadow


def feed_until_duplicate(hub, df):
    """Feed 1Min bars through the fan-out until a 2m bucket has CLOSED,
    then return the index of a 1Min bar INSIDE that closed bucket —
    re-feeding it is a rebroadcast for an already-closed period, which is
    what sets BarBuilder.last_was_duplicate (see accept_second_bar's
    rebroadcast branch)."""
    for i in range(len(df)):
        hub._fanout_to_secondary_builders(bar_dict(df, i), source_label='t')
        hist = hub.builders[SEC_TF].history
        if len(hist) > 0 and (SEC_TF, 'RTH') in hub._mtf_confluence \
                and i >= 6:
            last_closed_start = hist.index[-1]
            for j in range(i, -1, -1):
                if last_closed_start <= df.index[j] \
                        < last_closed_start + pd.Timedelta(seconds=SEC_TF):
                    return j
    raise AssertionError("no closed 2m bucket + publish — test feed bad")


def test_flag_off_duplicate_recomputes_and_republishes():
    """Legacy lock: duplicate → recompute_confluence output replaces the
    key (byte-identity of the kill-switch state)."""
    df = make_min_bars(12)
    with flag('0'):
        hub, shadow = build_hub_with_fine_shadow()
        i = feed_until_duplicate(hub, df)
        sentinel = {'2M-MACD_LINE-SENTINEL'}
        shadow.recompute_confluence = lambda hist: set(sentinel)
        hub._fanout_to_secondary_builders(bar_dict(df, i), source_label='t')
        assert hub.builders[SEC_TF].last_was_duplicate, \
            "re-fed bar was not flagged duplicate — test premise broken"
        assert hub._mtf_confluence[(SEC_TF, 'RTH')] == sentinel, (
            "flag OFF: duplicate branch no longer republishes the "
            "recompute output — legacy behavior changed")


def test_flag_on_duplicate_skips_recompute_and_keeps_records():
    """Fix lock: duplicate → NO recompute call, key records untouched,
    shadow indicator state untouched."""
    df = make_min_bars(12)
    with flag('1'):
        hub, shadow = build_hub_with_fine_shadow()
        i = feed_until_duplicate(hub, df)
        before_records = set(hub._mtf_confluence[(SEC_TF, 'RTH')])
        before_state = dict(shadow.indicators.get_values())

        def _boom(hist):
            raise AssertionError(
                "recompute_confluence called on the duplicate branch "
                "with the flag ON — the skip is broken")
        shadow.recompute_confluence = _boom

        hub._fanout_to_secondary_builders(bar_dict(df, i), source_label='t')
        assert hub.builders[SEC_TF].last_was_duplicate
        assert hub._mtf_confluence[(SEC_TF, 'RTH')] == before_records, \
            "flag ON: key records changed on a duplicate"
        after_state = dict(shadow.indicators.get_values())
        assert after_state == before_state, \
            "flag ON: shadow indicator state changed on a duplicate"


def test_flag_on_boundary_closes_still_publish():
    """Fix lock: with the flag ON, genuine boundary closes still update
    the key (only the duplicate recompute is suppressed)."""
    df = make_min_bars(12)
    with flag('1'):
        hub, _ = build_hub_with_fine_shadow()
        feed_until_duplicate(hub, df)
        assert hub._mtf_confluence.get((SEC_TF, 'RTH')), (
            "flag ON: boundary closes stopped publishing — the skip is "
            "too broad")


def test_flag_on_coarse_reload_still_runs(monkeypatch):
    """Fix lock: the CLASS 1a coarse reload branch precedes the fine skip
    and still runs with the flag ON."""
    df = make_min_bars(12)
    with flag('1'):
        hub, shadow = build_hub_with_fine_shadow()
        i = feed_until_duplicate(hub, df)
        called = []
        monkeypatch.setattr(
            hub, '_reload_coarse_rth_shadow',
            lambda tf, sh: called.append(tf))
        monkeypatch.setattr(
            hub, '_coarse_rth_reload_active', lambda tf: True)
        hub._fanout_to_secondary_builders(bar_dict(df, i), source_label='t')
        assert hub.builders[SEC_TF].last_was_duplicate
        assert called == [SEC_TF], \
            "coarse reload was skipped by the fine-authority flag"
