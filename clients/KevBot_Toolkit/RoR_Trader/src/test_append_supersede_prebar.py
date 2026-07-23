"""Lock RORT_APPEND_SUPERSEDE_OPEN + RORT_HIFI_NO_PREBAR_FILL (2026-07-23, board #101).

Class (07-23 triage): append writers are insert-only, so a full recompute's
open row (exit IS NULL) either gets DUPLICATED by the windowed append (same
entry, one 'open' + one closed row — unique index treats NULL exit as
distinct) or its close is SKIPPED forever by the cron append's entry-position
gate. Separately, the Hi-Fi 1s refiners walk PADDED windows with no session
filter, so fills can stamp pre-bar (premarket) seconds live could never act
on (stop_loss@13:29:55Z for a 13:30-bar exit).

Both flags default OFF = byte-identical legacy.

Run:  cd src && ../.venv/bin/python test_append_supersede_prebar.py
"""

import os
from datetime import datetime, timezone, timedelta

import pandas as pd


def _bars(start, n, step_s=5):
    idx = pd.DatetimeIndex([start + timedelta(seconds=step_s * i)
                            for i in range(n)])
    return pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5,
                         'close': 100.2, 'volume': 10}, index=idx)


def test_clamp_prebar_off_is_noop():
    os.environ['RORT_HIFI_NO_PREBAR_FILL'] = '0'
    from api.services.backtest_service import _clamp_prebar
    ws = datetime(2026, 7, 23, 13, 30, 0, tzinfo=timezone.utc)
    bars = _bars(ws - timedelta(seconds=5), 4)   # 13:29:55 .. 13:30:10
    out = _clamp_prebar(bars, ws)
    assert len(out) == 4 and out.index[0] < pd.Timestamp(ws)


def test_clamp_prebar_on_drops_left_pad():
    os.environ['RORT_HIFI_NO_PREBAR_FILL'] = '1'
    try:
        from api.services.backtest_service import _clamp_prebar
        ws = datetime(2026, 7, 23, 13, 30, 0, tzinfo=timezone.utc)
        bars = _bars(ws - timedelta(seconds=5), 4)
        out = _clamp_prebar(bars, ws)
        assert len(out) == 3
        assert out.index[0] == pd.Timestamp(ws)      # 13:30:00 first
    finally:
        os.environ['RORT_HIFI_NO_PREBAR_FILL'] = '0'


def test_supersede_calls_lane_scoped_delete(monkeypatch=None):
    os.environ['RORT_APPEND_SUPERSEDE_OPEN'] = '1'
    try:
        import db
        import api.services.forward_test_service as fts
        calls = []
        orig = db.delete_trade_placeholder_admin

        def _spy(sid, entry, data_source_like=None):
            calls.append((sid, entry, data_source_like))
            return 1
        db.delete_trade_placeholder_admin = _spy
        try:
            fts._supersede_open_rows(
                999, {'entry_fill_ts': '2026-07-22T20:00:00+00:00',
                      'exit_fill_ts': '2026-07-23T13:30:00+00:00'},
                'backtest_%', 'TEST')
            assert calls == [(999, '2026-07-22T20:00:00+00:00', 'backtest_%')]
            # open record (no exit) → no delete attempted
            fts._supersede_open_rows(
                999, {'entry_fill_ts': '2026-07-22T20:00:00+00:00',
                      'exit_fill_ts': None}, 'backtest_%', 'TEST')
            assert len(calls) == 1
        finally:
            db.delete_trade_placeholder_admin = orig
    finally:
        os.environ['RORT_APPEND_SUPERSEDE_OPEN'] = '0'


def test_flag_off_supersede_inert():
    os.environ['RORT_APPEND_SUPERSEDE_OPEN'] = '0'
    import api.services.forward_test_service as fts
    assert fts._supersede_open_flag() is False


def test_db_helpers_lane_scope_live():
    """Prod-safe smoke: nonexistent sid → 0 deletions / empty set, and the
    data_source_like param round-trips through PostgREST."""
    from dotenv import load_dotenv
    load_dotenv('.env')
    from db import delete_trade_placeholder_admin, get_open_trade_entries_admin
    assert delete_trade_placeholder_admin(
        -1, '2026-01-01T00:00:00+00:00', data_source_like='backtest_%') == 0
    assert get_open_trade_entries_admin(-1, 'backtest_%') == set()


if __name__ == '__main__':
    fails = 0
    for fn in (test_clamp_prebar_off_is_noop,
               test_clamp_prebar_on_drops_left_pad,
               test_supersede_calls_lane_scoped_delete,
               test_flag_off_supersede_inert,
               test_db_helpers_lane_scope_live):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if fails else 0)
