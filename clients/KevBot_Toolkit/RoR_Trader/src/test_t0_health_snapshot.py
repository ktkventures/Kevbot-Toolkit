"""Tests for the daily T+0 health snapshot (board #160).

Covers the three risky surfaces:
  1. _should_capture_now — the scheduling predicate (weekend / before-mark /
     after-mark / already-captured / late-cutoff).
  2. build_snapshot_rows — the pure merge of the 10s + 60s get_strategy_health
     passes into one upsert row per strategy (missing tolerance, algo-lane
     presence tri-state, defensive coercion).
  3. capture_snapshot — end-to-end with get_strategy_health + the admin client
     mocked: verifies the idempotent upsert (on_conflict) and the payload shape.

Plus flag gating (is_enabled / start_capture_thread) and tunable parsing.

No DB, no clock, no threads spawned. Run:
  cd src && ../.venv/bin/python test_t0_health_snapshot.py
"""
from __future__ import annotations

import os
import unittest.mock as mock
from datetime import date, datetime, time as dtime, timezone


def _setup_env():
    os.environ.setdefault('SUPABASE_URL', 'http://stub.local')
    os.environ.setdefault('SUPABASE_SERVICE_ROLE_KEY', 'stub-service-role-key')
    os.environ.setdefault('SUPABASE_ANON_KEY', 'stub-anon-key')


_setup_env()
import t0_health_snapshot as t0  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Supabase client — enough of the PostgREST builder chain for the
# select/like/gte/eq/limit/upsert calls this module makes.
# ---------------------------------------------------------------------------
class _FakeQuery:
    def __init__(self, table, ctx):
        self._table = table
        self._ctx = ctx
        self._mode = 'select'
        self._payload = None
        self._on_conflict = None

    def select(self, *_a, **_k):
        self._mode = 'select'
        return self

    def like(self, *_a, **_k):
        return self

    def gte(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def range(self, *_a, **_k):
        # Single-page fake: staged rows come back on the first page; the
        # paginator stops because len(chunk) < page size for our small fixtures.
        return self

    def upsert(self, payload, on_conflict=None):
        self._mode = 'upsert'
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def execute(self):
        if self._mode == 'upsert':
            self._ctx['upserts'].append(
                (self._table, self._payload, self._on_conflict))
            return mock.MagicMock(data=self._payload)
        # select — return whatever the ctx has staged for this table.
        return mock.MagicMock(data=self._ctx['selects'].get(self._table, []))


class _FakeClient:
    def __init__(self):
        self.ctx = {'upserts': [], 'selects': {}}

    def stage_select(self, table, rows):
        self.ctx['selects'][table] = rows

    def table(self, name):
        return _FakeQuery(name, self.ctx)


def _health_row(sid, *, combined=None, paired=None, phantom=None, missed=None,
                tbd=None, symbol='TSLA', tf='5Min'):
    return {
        'strategy_id': sid, 'user_id': f'u{sid}', 'name': f's{sid}',
        'symbol': symbol, 'timeframe': tf, 'direction': 'LONG',
        'data_source': 'backtest_rest_hifi', 'backtest_model': 'rest_hifi',
        'forward_testing': True, 'trade_count_backtest': 42,
        'combined_pct': combined, 'paired_count_cov': paired,
        'phantom_count_cov': phantom, 'missed_count_cov': missed,
        'tbd_count': tbd,
        'last_recompute_until_ts': '2026-07-28T14:45:00+00:00',
        'bt_current_through': '2026-07-28T14:44:00+00:00',
        'bt_heartbeat_at': '2026-07-28T14:59:00+00:00',
        'snapshot_age_sec': 30, 'last_alert_at': '2026-07-28T14:58:00+00:00',
        'last_alert_age_sec': 120,
    }


# ---------------------------------------------------------------------------
# 1) _should_capture_now
# ---------------------------------------------------------------------------
_TARGET = dtime(15, 0, tzinfo=timezone.utc)


def _dt(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_should_capture_after_mark():
    # 2026-07-28 is a Tuesday.
    assert t0._should_capture_now(_dt(2026, 7, 28, 15, 0), None, _TARGET, 120)
    assert t0._should_capture_now(_dt(2026, 7, 28, 15, 30), None, _TARGET, 120)


def test_should_not_capture_before_mark():
    assert not t0._should_capture_now(_dt(2026, 7, 28, 14, 59), None, _TARGET, 120)


def test_should_not_capture_weekend():
    # 2026-08-01 is a Saturday, 2026-08-02 a Sunday.
    assert not t0._should_capture_now(_dt(2026, 8, 1, 16, 0), None, _TARGET, 120)
    assert not t0._should_capture_now(_dt(2026, 8, 2, 16, 0), None, _TARGET, 120)


def test_should_not_recapture_same_day():
    today = date(2026, 7, 28)
    assert not t0._should_capture_now(_dt(2026, 7, 28, 16, 0), today, _TARGET, 120)


def test_late_cutoff_blocks_stale_capture():
    # 120-min cutoff: 17:00 is exactly at the edge (allowed), 17:01 is past it.
    assert t0._should_capture_now(_dt(2026, 7, 28, 17, 0), None, _TARGET, 120)
    assert not t0._should_capture_now(_dt(2026, 7, 28, 17, 1), None, _TARGET, 120)
    # cutoff 0 disables the late guard entirely.
    assert t0._should_capture_now(_dt(2026, 7, 28, 19, 0), None, _TARGET, 0)


# ---------------------------------------------------------------------------
# 2) build_snapshot_rows
# ---------------------------------------------------------------------------
_DAY = date(2026, 7, 28)
_CAP = _dt(2026, 7, 28, 15, 0)


def _build(health_by_tol, algo=None):
    return t0.build_snapshot_rows(
        health_by_tol, snapshot_day=_DAY, captured_at=_CAP,
        target_utc='15:00', health_now='2026-07-28T15:00:00+00:00',
        window_hours=24, algo_lane_sids=algo)


def test_build_merges_two_tolerances():
    rows = _build({
        10.0: [_health_row(267, combined=91.0, paired=10, phantom=1, missed=0, tbd=2)],
        60.0: [_health_row(267, combined=95.5, paired=11, phantom=0, missed=0, tbd=2)],
    }, algo={267})
    assert len(rows) == 1
    r = rows[0]
    assert r['strategy_id'] == 267
    assert r['snapshot_date'] == '2026-07-28'
    # 10s counts
    assert r['combined_pct_10'] == 91.0 and r['paired_10'] == 10
    assert r['phantom_10'] == 1 and r['tbd_10'] == 2
    # 60s counts
    assert r['combined_pct_60'] == 95.5 and r['paired_60'] == 11
    assert r['phantom_60'] == 0
    # descriptors from the 60s (default) pass
    assert r['symbol'] == 'TSLA' and r['forward_testing'] is True
    assert r['trade_count_backtest'] == 42
    # coverage context carried through
    assert r['last_recompute_until_ts'] == '2026-07-28T14:45:00+00:00'
    assert r['last_alert_age_sec'] == 120
    # algo lane present + full row retained
    assert r['algo_lane_present'] is True
    assert r['health_row']['combined_pct'] == 95.5


def test_build_algo_lane_tri_state():
    # sid present in algo set → True; absent → False; None set → NULL.
    rows_present = _build({60.0: [_health_row(267)]}, algo={999})
    assert rows_present[0]['algo_lane_present'] is False
    rows_none = _build({60.0: [_health_row(267)]}, algo=None)
    assert rows_none[0]['algo_lane_present'] is None


def test_build_single_tolerance_robust():
    # Only a 60s pass: 10s fields NULL, descriptors still from 60s.
    rows = _build({60.0: [_health_row(341, combined=88.0, paired=8)]})
    r = rows[0]
    assert r['combined_pct_60'] == 88.0
    assert r['combined_pct_10'] is None and r['paired_10'] is None
    assert r['symbol'] == 'TSLA'


def test_build_defensive_coercion():
    # None counts stay None; combined_pct None stays None (no crash).
    rows = _build({10.0: [_health_row(1)], 60.0: [_health_row(1)]})
    r = rows[0]
    assert r['paired_10'] is None and r['combined_pct_10'] is None
    assert r['missed_60'] is None


def test_build_union_of_sids_across_tols():
    rows = _build({
        10.0: [_health_row(1, combined=90.0)],
        60.0: [_health_row(2, combined=80.0)],
    })
    sids = sorted(r['strategy_id'] for r in rows)
    assert sids == [1, 2]


# ---------------------------------------------------------------------------
# 3) capture_snapshot end-to-end
# ---------------------------------------------------------------------------
def test_capture_snapshot_upserts_idempotently():
    import api.routers.strategy_health as sh_mod

    fake = _FakeClient()
    fake.stage_select('trades', [{'strategy_id': 267}])  # algo-lane presence

    def _fake_health(user=None, window_hours=24, tolerance_seconds=60.0,
                     **_k):
        combined = 91.0 if tolerance_seconds == 10.0 else 96.0
        return {'now': '2026-07-28T15:00:00+00:00', 'window_hours': window_hours,
                'rows': [_health_row(267, combined=combined, paired=10)]}

    with mock.patch.object(sh_mod, 'get_strategy_health', _fake_health), \
         mock.patch('db.get_admin_client', return_value=fake):
        summary = t0.capture_snapshot(now=_CAP, tolerances=(10.0, 60.0))

    assert summary['strategies'] == 1
    assert summary['written'] == 1
    assert summary['algo_lane_sids'] == 1
    assert len(fake.ctx['upserts']) == 1
    table, payload, on_conflict = fake.ctx['upserts'][0]
    assert table == 't0_health_snapshots'
    assert on_conflict == 'snapshot_date,strategy_id'  # idempotent key
    assert payload[0]['combined_pct_10'] == 91.0
    assert payload[0]['combined_pct_60'] == 96.0
    assert payload[0]['algo_lane_present'] is True


def test_capture_snapshot_dry_run_writes_nothing():
    import api.routers.strategy_health as sh_mod

    fake = _FakeClient()
    fake.stage_select('trades', [])

    def _fake_health(user=None, window_hours=24, tolerance_seconds=60.0, **_k):
        return {'now': '2026-07-28T15:00:00+00:00',
                'rows': [_health_row(267, combined=90.0)]}

    with mock.patch.object(sh_mod, 'get_strategy_health', _fake_health), \
         mock.patch('db.get_admin_client', return_value=fake):
        summary = t0.capture_snapshot(now=_CAP, dry_run=True)

    assert summary['dry_run'] is True
    assert summary['written'] == 0
    assert fake.ctx['upserts'] == []       # nothing written
    assert 'rows' in summary and len(summary['rows']) == 1


def test_capture_snapshot_scopes_only_sids_and_restores():
    import api.routers.strategy_health as sh_mod

    fake = _FakeClient()
    fake.stage_select('trades', [])
    seen = {}

    def _fake_health(user=None, window_hours=24, tolerance_seconds=60.0, **_k):
        seen['only_sids_during_call'] = sh_mod._ONLY_SIDS
        return {'now': '2026-07-28T15:00:00+00:00',
                'rows': [_health_row(267)]}

    prev = sh_mod._ONLY_SIDS
    with mock.patch.object(sh_mod, 'get_strategy_health', _fake_health), \
         mock.patch('db.get_admin_client', return_value=fake):
        t0.capture_snapshot(now=_CAP, only_sids=[267], tolerances=(60.0,))

    assert seen['only_sids_during_call'] == [267]   # scoped during the call
    assert sh_mod._ONLY_SIDS == prev                # restored after


def test_already_captured_today_guard():
    fake = _FakeClient()
    fake.stage_select('t0_health_snapshots', [{'id': 1}])
    with mock.patch('db.get_admin_client', return_value=fake):
        assert t0.already_captured_today(_DAY) is True
    fake2 = _FakeClient()  # no staged rows → empty
    with mock.patch('db.get_admin_client', return_value=fake2):
        assert t0.already_captured_today(_DAY) is False


# ---------------------------------------------------------------------------
# 4) flag gating + tunables
# ---------------------------------------------------------------------------
def test_is_enabled_default_off():
    prev = os.environ.pop('RORT_T0_HEALTH_SNAPSHOT', None)
    try:
        assert t0.is_enabled() is False
        assert t0.start_capture_thread(__import__('threading').Event()) is None
        os.environ['RORT_T0_HEALTH_SNAPSHOT'] = '1'
        assert t0.is_enabled() is True
    finally:
        if prev is None:
            os.environ.pop('RORT_T0_HEALTH_SNAPSHOT', None)
        else:
            os.environ['RORT_T0_HEALTH_SNAPSHOT'] = prev


def test_tunable_parsing():
    with mock.patch.dict(os.environ, {'RORT_T0_HEALTH_SNAPSHOT_AT_UTC': '13:30'}):
        assert t0._target_time_utc() == dtime(13, 30, tzinfo=timezone.utc)
    with mock.patch.dict(os.environ, {'RORT_T0_HEALTH_SNAPSHOT_AT_UTC': 'garbage'}):
        assert t0._target_time_utc() == dtime(15, 0, tzinfo=timezone.utc)
    with mock.patch.dict(os.environ, {'RORT_T0_HEALTH_SNAPSHOT_TOLS': '5, 15 ,30'}):
        assert t0._tolerances() == [5.0, 15.0, 30.0]
    with mock.patch.dict(os.environ, {'RORT_T0_HEALTH_SNAPSHOT_TOLS': ''}):
        assert t0._tolerances() == [10.0, 60.0]
    with mock.patch.dict(os.environ, {'RORT_T0_HEALTH_SNAPSHOT_LATE_MIN': 'x'}):
        assert t0._late_cutoff_min() == 120


# ---------------------------------------------------------------------------
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} passed")


if __name__ == '__main__':
    _run_all()
