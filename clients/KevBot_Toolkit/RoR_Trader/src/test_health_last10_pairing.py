"""Unit tests for health_last10 (board #122 _greedy_pair + board #119 algo basis).

The E-sign-off nit: the walk used to return a set of paired edge
TIMESTAMPS, so two edges sharing one timestamp double-collapsed — one
matched alert marked both edges paired, inflating points. The fix keys
the returned set by index into the edges argument.

The board #119 tests exercise `_score_sid` end-to-end through a fake
Supabase client, proving the third 'algo' basis pairs the last-10 backtest
edges against the ALGO-LANE (cache_%) trades and surfaces as `points_algo`.

Run:  cd src && ../.venv/bin/pytest test_health_last10_pairing.py -v
"""
from types import SimpleNamespace

from api.routers.health_last10 import _greedy_pair, _score_sid


def _reference_paired_count(edges, alerts, tolerance):
    """Verbatim count-only port of strategy_health._pair_phantom_missed's
    greedy two-pointer walk (the closure at strategy_health.py:349) — the
    semantics _greedy_pair must reproduce."""
    edges_sorted = sorted(edges)
    alerts_sorted = sorted(alerts)
    count = i = j = 0
    while i < len(edges_sorted) and j < len(alerts_sorted):
        diff = alerts_sorted[j] - edges_sorted[i]
        if abs(diff) <= tolerance:
            count += 1
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1
    return count


def test_dup_timestamp_one_alert_pairs_exactly_one():
    """Two edges at the same second, ONE alert in tolerance → 1 point.
    (Timestamp-keyed set counted 2 — the board #122 double-collapse.)"""
    edges = [100.0, 100.0, 500.0]
    alerts = [101.0]
    paired = _greedy_pair(edges, alerts, 10.0)
    assert len(paired) == 1
    assert paired <= {0, 1}  # one of the two duplicate slots, not the 500 edge


def test_dup_timestamp_two_alerts_pair_both():
    """Each duplicate edge is its own pairing slot when alerts exist for
    both."""
    edges = [100.0, 100.0]
    alerts = [99.0, 101.0]
    paired = _greedy_pair(edges, alerts, 10.0)
    assert paired == {0, 1}


def test_one_alert_cannot_serve_two_edges():
    """1:1 pairing — distinct-timestamp edges still consume alerts
    individually."""
    edges = [100.0, 105.0]
    alerts = [102.0]
    paired = _greedy_pair(edges, alerts, 10.0)
    assert len(paired) == 1


def test_indices_refer_to_original_positions_when_unsorted():
    """The walk sorts internally; returned indices must point at the
    caller's original list positions."""
    edges = [500.0, 100.0, 300.0]  # only 100.0 (index 1) has an alert
    alerts = [101.0]
    assert _greedy_pair(edges, alerts, 10.0) == {1}


def test_out_of_tolerance_pairs_nothing():
    assert _greedy_pair([100.0], [200.0], 10.0) == set()
    assert _greedy_pair([], [100.0], 10.0) == set()
    assert _greedy_pair([100.0], [], 10.0) == set()


def test_count_matches_reference_walk():
    """len(_greedy_pair(...)) must equal the canonical count walk in every
    scenario, INCLUDING duplicate timestamps (where the old set-of-ts
    version diverged)."""
    scenarios = [
        ([100.0, 100.0, 500.0], [101.0]),            # the #122 dup case
        ([100.0, 100.0], [99.0, 101.0]),             # dup, both served
        ([100.0, 100.0, 100.0], [95.0, 105.0]),      # triple dup, 2 alerts
        ([500.0, 100.0, 300.0], [101.0, 299.0]),     # unsorted input
        ([100.0, 200.0, 300.0], [100.0, 200.0, 300.0]),
        ([100.0, 111.0], [105.0]),                   # boundary: only 1 in tol
        ([], []),
    ]
    for edges, alerts in scenarios:
        got = len(_greedy_pair(edges, alerts, 10.0))
        want = _reference_paired_count(edges, alerts, 10.0)
        assert got == want, (edges, alerts, got, want)


# ── board #119 — ALGO basis (BT edges vs algo-lane cache_% trades) ──────

class _FakeQuery:
    """Chainable stand-in for a supabase-py query. Records the table, the
    `like` patterns, then hands the recorded state back to the client to
    resolve at execute() — enough to drive _score_sid's three fetches
    (backtest trades / alerts / algo trades) + the tf lookup."""
    def __init__(self, client, table):
        self._c = client
        self._table = table
        self._like = {}

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def like(self, col, pat):
        self._like[col] = pat
        return self

    @property
    def not_(self):
        return self

    def is_(self, *a, **k):
        return self

    def gte(self, *a, **k):
        return self

    def lte(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return SimpleNamespace(data=self._c._resolve(self._table, self._like))


class _FakeClient:
    def __init__(self, backtest=None, alerts=None, algo=None, tf="1Min"):
        self._backtest = backtest or []
        self._alerts = alerts or []
        self._algo = algo or []
        self._tf = tf

    def table(self, name):
        return _FakeQuery(self, name)

    def _resolve(self, table, like):
        if table == "trades":
            pat = like.get("data_source", "")
            if pat.startswith("backtest"):
                return list(self._backtest)
            if pat.startswith("cache"):
                return list(self._algo)
            return []
        if table == "alerts":
            return list(self._alerts)
        if table == "strategies":
            return [{"timeframe": self._tf}]
        return []


# Two completed backtest trades (returned exit-desc, as the DB query orders):
# trade #2 @15:00, trade #1 @14:00 → 4 edges, denom 4.
_BT = [
    {"id": 2, "entry_fill_ts": "2026-07-25T15:00:00+00:00",
     "exit_fill_ts": "2026-07-25T15:05:00+00:00"},
    {"id": 1, "entry_fill_ts": "2026-07-25T14:00:00+00:00",
     "exit_fill_ts": "2026-07-25T14:05:00+00:00"},
]
# Alerts near all four BT edges → fired + theo both pair 4/4.
_ALERTS = [
    {"timestamp": "2026-07-25T14:00:03+00:00", "fill_ts": "2026-07-25T14:00:00+00:00",
     "trigger_ts": None, "event_type": "entry"},
    {"timestamp": "2026-07-25T14:05:04+00:00", "fill_ts": "2026-07-25T14:05:00+00:00",
     "trigger_ts": None, "event_type": "exit"},
    {"timestamp": "2026-07-25T15:00:03+00:00", "fill_ts": "2026-07-25T15:00:00+00:00",
     "trigger_ts": None, "event_type": "entry"},
    {"timestamp": "2026-07-25T15:05:03+00:00", "fill_ts": "2026-07-25T15:05:00+00:00",
     "trigger_ts": None, "event_type": "exit"},
]
# Algo lane (cache_%) covers ONLY the 14:00 trade's two edges → algo 2/4.
_ALGO = [
    {"entry_fill_ts": "2026-07-25T14:00:02+00:00",
     "exit_fill_ts": "2026-07-25T14:05:01+00:00"},
]


def test_algo_basis_scored_from_cache_lane():
    """points_algo pairs BT edges against algo-lane edges, independent of the
    alert-derived fired/theo scores. Here delivery is perfect (4/4) but the
    engine only computed one of the two trades → algo 2/4 (the algo-low =
    logic/state divergence diagnostic)."""
    out = _score_sid(_FakeClient(_BT, _ALERTS, _ALGO), 42, 10.0, False)
    assert out["denom"] == 4
    assert out["points"] == 4        # fired: every edge has an alert
    assert out["points_theo"] == 4
    assert out["points_algo"] == 2   # only the 14:00 trade has cache_% edges


def test_algo_basis_zero_when_no_cache_trades():
    """No algo-lane rows → points_algo 0, but fired/theo unaffected (a
    delivery-fine / no-engine-record split)."""
    out = _score_sid(_FakeClient(_BT, _ALERTS, algo=[]), 42, 10.0, False)
    assert out["points"] == 4
    assert out["points_algo"] == 0


def test_algo_window_guard_drops_far_edges():
    """An algo edge outside the ±1h bracket is guarded out (never pairs, never
    inflates) — the DB filter is on entry_fill_ts, the per-edge unix guard
    covers exits that stray past the window."""
    far = [{"entry_fill_ts": "2026-07-25T14:00:02+00:00",   # in window (pairs)
            "exit_fill_ts": "2026-07-26T14:05:01+00:00"}]    # +1 day (guarded)
    out = _score_sid(_FakeClient(_BT, _ALERTS, far), 42, 10.0, False)
    assert out["points_algo"] == 1   # only the entry paired; far exit dropped


def test_detail_carries_algo_basis_per_trade():
    """Detail mode adds the 'algo' basis object to every trade row so the
    modal can render the algo pairing column."""
    out = _score_sid(_FakeClient(_BT, _ALERTS, _ALGO), 42, 10.0, True)
    assert out["points_algo"] == 2
    assert len(out["trades"]) == 2
    t1, t2 = out["trades"]           # reversed to oldest→newest: 14:00, 15:00
    for row in out["trades"]:
        assert "algo" in row
        assert {"entry_paired", "exit_paired", "entry_nearest_alert_ts",
                "exit_nearest_alert_ts", "entry_delta_sec",
                "exit_delta_sec"} <= set(row["algo"].keys())
    assert t1["algo"]["entry_paired"] is True
    assert t1["algo"]["exit_paired"] is True
    assert t2["algo"]["entry_paired"] is False   # no cache_% edge near 15:00
    assert t2["algo"]["exit_paired"] is False


def test_response_shape_backward_compatible():
    """The pre-#119 keys stay present + typed (frontend deploy-skew safety)."""
    out = _score_sid(_FakeClient(_BT, _ALERTS, _ALGO), 42, 10.0, False)
    for k in ("strategy_id", "points", "points_theo", "points_algo",
              "denom", "trade_count", "tolerance_seconds"):
        assert k in out
    assert isinstance(out["points_algo"], int)
