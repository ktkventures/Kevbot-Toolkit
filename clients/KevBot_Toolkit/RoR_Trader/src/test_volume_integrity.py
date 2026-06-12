"""Lock the volume-integrity fix (2026-06-12, iter 0612b).

Live >=60s bars carried ~2x volume: (a) AM and ws_agg fan-outs both
deliver the SAME minute into >=120s builders — each add summed volume
again; (b) per-second chart-visual updates on 1Min+ primaries added a
third stream. Volume-dependent packs broke (RVOL gate read EXTREME
near-constantly → sid 291 at 1.0% combined; VWAP integrated the bias
cumulatively → 304/305). Price packs unaffected (~90-97%).

Fix: per-partial sub-bar timestamp dedup (volume counts once per
sub-bar) + skip_volume on the chart-visual caller.

Run:  cd src && ../.venv/bin/python -m pytest test_volume_integrity.py -v
"""
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, '.')
from ralph_engine import BarBuilder

T0 = pd.Timestamp('2026-06-12T14:00:00+00:00')


def _minute(i, vol=1000.0):
    ts = T0 + timedelta(minutes=i)
    px = 100.0 + i * 0.1
    return {'timestamp': ts.isoformat(), 'open': px, 'high': px + 0.2,
            'low': px - 0.2, 'close': px + 0.1, 'volume': vol}


def test_same_minute_double_feed_counts_volume_once():
    """AM + ws_agg fan-outs deliver the same minute → volume once."""
    b = BarBuilder(tf_seconds=120)
    b.accept_second_bar(_minute(0), close_on_boundary=True)   # ws_agg
    b.accept_second_bar(_minute(0), close_on_boundary=True)   # AM (same minute)
    b.accept_second_bar(_minute(1), close_on_boundary=True)
    b.accept_second_bar(_minute(1), close_on_boundary=True)
    completed = b.accept_second_bar(_minute(2), close_on_boundary=True)
    assert completed is not None
    assert completed['volume'] == 2000.0, completed['volume']  # NOT 4000


def test_skip_volume_contributes_zero():
    """Chart-visual per-second path must not add volume."""
    b = BarBuilder(tf_seconds=120)
    b.accept_second_bar(_minute(0), close_on_boundary=True)        # canonical
    sec = {'timestamp': (T0 + timedelta(seconds=70)).isoformat(),
           'open': 100, 'high': 100.3, 'low': 99.9, 'close': 100.2,
           'volume': 500.0}
    b.accept_second_bar(sec, close_on_boundary=False, skip_volume=True)
    assert b._partial.volume == 1000.0                              # unchanged
    # OHLC still updates from the visual feed
    assert b._partial.high >= 100.3


def test_normal_per_second_path_unchanged():
    """Distinct per-second bars on a 10s builder sum normally."""
    b = BarBuilder(tf_seconds=10)
    for s in range(10):
        bar = {'timestamp': (T0 + timedelta(seconds=s)).isoformat(),
               'open': 100, 'high': 100.1, 'low': 99.9, 'close': 100,
               'volume': 10.0}
        b.accept_second_bar(bar, close_on_boundary=True)
    done = b.accept_second_bar(
        {'timestamp': (T0 + timedelta(seconds=10)).isoformat(),
         'open': 100, 'high': 100.1, 'low': 99.9, 'close': 100,
         'volume': 10.0}, close_on_boundary=True)
    assert done is not None and done['volume'] == 100.0
