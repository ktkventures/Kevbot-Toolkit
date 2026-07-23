"""Lock RORT_BAR_DUP_GUARD behavior (2026-07-23) — BarBuilder duplicate-period
protection.

The bug (sid 136, found by the 07-23 nightly bug-hunt): after a period's
authoritative close (accept_bar AM bar) clears the partial, a late
out-of-order tick stamped inside that period re-opens a partial for it, and
the next rollover closes it as a SECOND history row and increments _bar_count
again. bars_held = bar_count - entry_bar_count then over-counts, firing
max_hold_bars exits one bar early (live 16:59 vs backtest+algo 17:00,
2026-07-22), and the duplicate row (stale partial OHLC) feeds the incremental
indicator stream.

Flag OFF (shipped default): byte-identical legacy behavior — the duplicate
row and count inflation still occur (this test DOCUMENTS the bug so the
flag-off case is locked, mirroring test_ralph_bar_no_gapfill's approach).
Flag ON: the late tick / stale-partial close is dropped — one row per period,
honest _bar_count, no spurious completed-bar dispatch.

Both modes: the BAR_DUP_GUARD tripwire warning fires on detection (validated
here via the rate-limit marker), so live frequency is measurable pre-arm.

Run:  cd src && ../.venv/bin/python test_ralph_bar_dup_guard.py
Or:   cd src && ../.venv/bin/pytest test_ralph_bar_dup_guard.py -v
"""

import os
from datetime import datetime, timezone

from ralph_engine import BarBuilder


def _dt(h, m, s):
    return datetime(2026, 7, 22, h, m, s, tzinfo=timezone.utc)


def _drive_tick_am_late_tick(builder):
    """The sid-136 sequence: tick-open, rollover, AM authority close, late
    out-of-order tick for the closed period, next-period tick."""
    builder.process_tick(100.0, 10, _dt(16, 55, 1))    # partial(16:55)
    builder.process_tick(101.0, 10, _dt(16, 56, 1))    # closes 16:55
    builder.accept_bar({'timestamp': _dt(16, 56, 0).isoformat(),
                        'open': 101.0, 'high': 101.6, 'low': 100.9,
                        'close': 101.5, 'volume': 500})  # AM closes 16:56
    builder.process_tick(101.4, 5, _dt(16, 56, 59))    # LATE tick in 16:56
    return builder.process_tick(102.0, 10, _dt(16, 57, 1))  # rollover


def test_flag_off_legacy_duplicate():
    """OFF: the duplicate row + inflated count occur (locked legacy)."""
    os.environ['RORT_BAR_DUP_GUARD'] = '0'
    b = BarBuilder(tf_seconds=60)
    completed = _drive_tick_am_late_tick(b)
    assert b._bar_count == 3, b._bar_count            # inflated (2 real periods)
    assert len(b.history) == 3
    assert b.history.index.duplicated().any()          # duplicate 16:56 row
    assert completed is not None                       # spurious dispatch
    # tripwire fired even at OFF (rate-limit marker set to the stale period)
    assert b._dup_guard_warned_period is not None


def test_flag_on_drops_late_tick():
    """ON: late tick for a closed period is dropped — honest count, no dup."""
    os.environ['RORT_BAR_DUP_GUARD'] = '1'
    try:
        b = BarBuilder(tf_seconds=60)
        completed = _drive_tick_am_late_tick(b)
        assert b._bar_count == 2, b._bar_count         # honest closed count
        assert len(b.history) == 2
        assert not b.history.index.duplicated().any()
        assert completed is None                       # no spurious dispatch
        assert b._dup_guard_warned_period is not None  # tripwire fired
        # the in-order stream continues normally after the drop
        got = b.process_tick(103.0, 10, _dt(16, 58, 1))  # closes 16:57
        assert got is not None and b._bar_count == 3
        assert not b.history.index.duplicated().any()
    finally:
        os.environ['RORT_BAR_DUP_GUARD'] = '0'


def test_flag_on_stale_partial_close_guard():
    """ON: a stale partial (opened while OFF, then flag flips ON) is discarded
    at close time by the _close_bar guard — defense in depth."""
    os.environ['RORT_BAR_DUP_GUARD'] = '0'
    b = BarBuilder(tf_seconds=60)
    b.process_tick(100.0, 10, _dt(16, 55, 1))
    b.process_tick(101.0, 10, _dt(16, 56, 1))
    b.accept_bar({'timestamp': _dt(16, 56, 0).isoformat(),
                  'open': 101.0, 'high': 101.6, 'low': 100.9,
                  'close': 101.5, 'volume': 500})
    b.process_tick(101.4, 5, _dt(16, 56, 59))          # stale partial opens (OFF)
    os.environ['RORT_BAR_DUP_GUARD'] = '1'
    try:
        completed = b.process_tick(102.0, 10, _dt(16, 57, 1))
        assert completed is None                       # stale close dropped
        assert b._bar_count == 2 and len(b.history) == 2
        assert not b.history.index.duplicated().any()
    finally:
        os.environ['RORT_BAR_DUP_GUARD'] = '0'


def test_flag_on_second_bar_path():
    """ON: accept_second_bar (canonical sub-minute path) refuses to re-open a
    period older than the last closed row."""
    os.environ['RORT_BAR_DUP_GUARD'] = '1'
    try:
        b = BarBuilder(tf_seconds=10)
        for s in (0, 5):                               # 16:55:00 bucket
            b.accept_second_bar({'timestamp': _dt(16, 55, s).isoformat(),
                                 'open': 100, 'high': 100.5, 'low': 99.9,
                                 'close': 100.2, 'volume': 10})
        b.accept_second_bar({'timestamp': _dt(16, 55, 11).isoformat(),
                             'open': 100.3, 'high': 100.6, 'low': 100.2,
                             'close': 100.4, 'volume': 10})   # closes :00 bucket
        b.accept_second_bar({'timestamp': _dt(16, 55, 21).isoformat(),
                             'open': 100.5, 'high': 100.7, 'low': 100.4,
                             'close': 100.6, 'volume': 10})   # closes :10 bucket
        assert b._bar_count == 2
        # force a None partial (as force_close_stale_bar would leave it), then
        # deliver a sub-bar for the :00 bucket — OLDER than the last closed row
        b._partial = None
        got = b.accept_second_bar({'timestamp': _dt(16, 55, 3).isoformat(),
                                   'open': 100.1, 'high': 100.2, 'low': 100.0,
                                   'close': 100.1, 'volume': 5})
        assert got is None
        assert b._partial is None                      # not re-opened
        assert b._bar_count == 2
        assert not b.history.index.duplicated().any()
        # in-order stream resumes
        b.accept_second_bar({'timestamp': _dt(16, 55, 31).isoformat(),
                             'open': 100.7, 'high': 100.8, 'low': 100.6,
                             'close': 100.7, 'volume': 10})
        assert b._partial is not None
    finally:
        os.environ['RORT_BAR_DUP_GUARD'] = '0'


if __name__ == '__main__':
    fails = 0
    for fn in (test_flag_off_legacy_duplicate,
               test_flag_on_drops_late_tick,
               test_flag_on_stale_partial_close_guard,
               test_flag_on_second_bar_path):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if fails else 0)
