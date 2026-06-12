"""Caller-level repro for the 2m clobber + the 247afec/0d767d9 "freeze".

ROOT CAUSE (2026-06-12 evening, second bisection): the freeze was never a
close freeze — it was WRITE STARVATION.

  1. flush_stale_bars closes >=120s partials with ZERO grace
     (force_close_stale_bar: grace_sec = 0 for tf >= 60), every ~2s.
  2. The ws_agg minute that completes a 2m period can only be DELIVERED
     after that period ends (the 1Min builder needs a next-minute second
     to close), so flush nearly always wins the close race.
  3. flush only WRITES live_bars rows for tf < 60 (M8.7 hotfix #2) — so
     the only live_bars write path for >=120s bars was the late fan-out
     minute hitting accept_second_bar's rebroadcast-replace branch, which
     returns a "completed" bar dict that _fanout_to_secondary_builders
     writes (and which REPLACED the whole 2m bar with one minute's OHLCV:
     the clobber, volume 0.50x REST).
  4. The 247afec guard dropped those late minutes -> no clobber, but also
     no >=120s live_bars writes ever again -> watcher saw 0 rows ->
     "freeze" -> revert. Closes (and gates) were fine the whole time.

THE FIX (this repro is its gate):
  - Late sub-bars (full_period_input=False) MERGE into the closed row:
    high=max, low=min, close from the latest constituent, volume deduped
    via the retained seen-subbars set. Never wholesale replace, never drop.
  - flush_stale_bars writes pure-secondary >=60s closes to live_bars.

Run:  cd src && ../.venv/bin/python -m pytest test_clobber_repro.py -v
"""
import sys
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, '.')
import live_bars_writer
from ralph_engine import BarBuilder, SymbolHub, _ShadowIndicatorEngine

PARAMS = {'ema_periods': [9, 21], 'macd_fast_period': 12,
          'macd_slow_period': 26, 'macd_signal_period': 9, 'atr_period': 14}

T0 = pd.Timestamp('2026-06-12T14:00:00+00:00')


def _minute(i, vol=1000.0):
    ts = T0 + timedelta(minutes=i)
    px = 100.0 + i * 0.05
    return {'timestamp': ts.isoformat(), 'open': px, 'high': px + 0.3,
            'low': px - 0.3, 'close': px + 0.1, 'volume': vol}


def _seed_df(n_minutes=30):
    idx = pd.date_range(T0 - timedelta(minutes=n_minutes), periods=n_minutes,
                        freq='1min', tz='UTC')
    px = pd.Series(99.0 + np.arange(n_minutes) * 0.02, index=idx)
    return pd.DataFrame({'open': px, 'high': px + 0.2, 'low': px - 0.2,
                         'close': px + 0.05,
                         'volume': [1000.0] * n_minutes}, index=idx)


def _resample(df, tf_seconds):
    return df.resample(f'{tf_seconds}s').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close': 'last', 'volume': 'sum'}).dropna(subset=['open'])


def build_prod_like_hub(tf_seconds=120):
    hub = SymbolHub('TEST')
    b = BarBuilder(tf_seconds=tf_seconds)
    hub.builders[tf_seconds] = b
    # production wiring: the fan-out only feeds TFs with a shadow engine
    shadow = _ShadowIndicatorEngine(tf_seconds, {'ema', 'macd', 'atr'},
                                    {'MACD_LINE'}, dict(PARAMS))
    seed = _resample(_seed_df(), tf_seconds)
    shadow.warmup(seed)
    hub._shadow_engines[tf_seconds] = shadow
    # production: warmup seeds the secondary builder's history
    hub.seed_history(tf_seconds, seed)
    return hub, b


@pytest.fixture
def write_capture(monkeypatch):
    """Capture live_bars writes. Both flush_stale_bars and the fan-outs
    import write_bar function-locally (`from live_bars_writer import
    write_bar`), so patching the module attribute intercepts every call."""
    writes = []

    def _fake_write(symbol, tf_seconds, bar, source='ws', **kw):
        writes.append({'tf': tf_seconds, 'source': source, **dict(bar)})

    monkeypatch.setattr(live_bars_writer, 'write_bar', _fake_write)
    return writes


def run_prod_feed(hub, tf_seconds=120, minutes=12, slow_tape=True):
    """PRODUCTION sequence per minute boundary i (wall clock = T0+i min):

      1. flush_stale_bars fires within ~2s of every period end and closes
         the >=120s partial BEFORE the minute that completes it can be
         delivered (1Min ws_agg only closes when a NEXT-minute second
         arrives). slow_tape=True models this dominant ordering.
      2. ws_agg delivers minute i-1 (now complete).
      3. AM redelivers minute i-2 (~5s lag) — hits the closed period.
    """
    for i in range(minutes):
        now = (T0 + timedelta(minutes=i, seconds=1)).to_pydatetime()
        if slow_tape:
            hub.flush_stale_bars(now)
        if i >= 1:
            hub._fanout_to_secondary_builders(_minute(i - 1),
                                              source_label='ws_agg')
        if i >= 2:
            hub._fanout_to_secondary_builders(_minute(i - 2),
                                              source_label='ws')
        if not slow_tape:
            hub.flush_stale_bars(now)


@pytest.mark.parametrize('tf_seconds', [120, 300])
def test_closes_keep_happening(tf_seconds, write_capture):
    """THE FREEZE GATE (revised): closes were never the problem, but they
    must obviously keep happening under the full production sequence."""
    hub, b = build_prod_like_hub(tf_seconds)
    n0 = len(b.history)
    run_prod_feed(hub, tf_seconds, minutes=12)
    per_period = 60 // (tf_seconds // 60) if tf_seconds <= 720 else 1
    expected = (12 // (tf_seconds // 60)) - 1  # last period still forming
    closed = len(b.history) - n0
    assert closed >= expected - 1, (
        f"tf={tf_seconds}: expected ~{expected} closes, got {closed}")


@pytest.mark.parametrize('tf_seconds', [120, 300])
def test_no_write_starvation(tf_seconds, write_capture):
    """THE REAL 247afec REGRESSION GATE: every closed >=120s period must
    produce at least one live_bars write. The drop-guard starved writes
    to zero (flush doesn't write >=60s; the replace branch was the only
    writer and the guard silenced it)."""
    hub, b = build_prod_like_hub(tf_seconds)
    n0 = len(b.history)
    run_prod_feed(hub, tf_seconds, minutes=12)
    closed_ts = set(b.history.index[n0:].tz_convert('UTC'))
    written_ts = {pd.Timestamp(w['timestamp']).tz_convert('UTC')
                  for w in write_capture if w['tf'] == tf_seconds}
    unwritten = closed_ts - written_ts
    assert not unwritten, (
        f"tf={tf_seconds}: closed periods never written to live_bars: "
        f"{sorted(unwritten)} — write starvation (the 'freeze')")


@pytest.mark.parametrize('tf_seconds', [120, 300])
def test_no_clobber_values_converge(tf_seconds, write_capture):
    """THE CLOBBER GATE: after late AM/ws_agg minutes land on closed
    periods, in-memory history must equal the REST-style resample of the
    constituent minutes — full volume (no 0.50x single-minute replace),
    true high/low, last constituent's close."""
    hub, b = build_prod_like_hub(tf_seconds)
    n0 = len(b.history)
    minutes = 12
    run_prod_feed(hub, tf_seconds, minutes=minutes)

    fed = pd.DataFrame([_minute(i) for i in range(minutes - 1)])
    fed.index = pd.DatetimeIndex(pd.to_datetime(fed.pop('timestamp')))
    expected = _resample(fed, tf_seconds)

    got = b.history.iloc[n0:]
    # exclude the newest closed period: its late minutes may not have
    # been redelivered yet by the end of the feed (converges next round)
    for ts, row in got.iloc[:-1].iterrows():
        exp = expected.loc[ts]
        for col in ('open', 'high', 'low', 'close', 'volume'):
            assert float(row[col]) == pytest.approx(float(exp[col])), (
                f"tf={tf_seconds} bar={ts} {col}: live={row[col]} "
                f"expected={exp[col]} — clobber/half-bar not healed")


def test_final_writes_not_clobbered(write_capture):
    """The LAST write for each period must carry merged (full) values —
    pre-fix, the last write was the single-minute clobber (0.50x vol)."""
    hub, b = build_prod_like_hub(120)
    run_prod_feed(hub, 120, minutes=12)
    last_write = {}
    for w in write_capture:
        if w['tf'] == 120:
            last_write[pd.Timestamp(w['timestamp']).tz_convert('UTC')] = w
    # all but the newest period must show the deduped two-minute volume
    for ts in sorted(last_write)[:-1]:
        vol = float(last_write[ts]['volume'])
        assert vol == pytest.approx(2000.0), (
            f"bar={ts}: final live_bars write volume={vol} "
            f"(single-minute clobber would be 1000.0)")
