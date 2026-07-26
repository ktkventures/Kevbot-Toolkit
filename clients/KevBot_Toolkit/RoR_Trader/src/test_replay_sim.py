"""Offline unit tests for the replay-SIM basis producer (board #120, V1.16).

Maps 1:1 to the design's test plan (§9) and divergence ledger (§8). Everything
here runs WITHOUT a DB or the heavy replay_harness — the job's pure helpers
(window sizing, ledger, coverage, staleness, fingerprint), the read-path scorer
(`score_sim_basis`, driven by a fake Supabase client), and the poller's
guardrails. The actual engine replay is exercised by replay_harness itself.

Run:  cd src && ../.venv/bin/pytest test_replay_sim.py -v
"""
import os
import sys
from collections import deque
from datetime import datetime, timedelta, timezone

# Make the poller importable (lives at ../tools/team_sim/).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools", "team_sim"))

import replay_sim_job as J  # noqa: E402


def _iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _trade(entry, exit_):
    return {"entry_fill_ts": entry, "exit_fill_ts": exit_}


# ─────────────────────────── session floor / spans ──────────────────────────

def test_session_floor_is_0930_et_as_utc():
    # 2026-07-24 15:00Z (Fri, EDT = 11:00 ET) → 09:30 ET = 13:30Z.
    got = J.session_floor_utc(_iso("2026-07-24T15:00:00Z"))
    assert got == _iso("2026-07-24T13:30:00Z")


def test_trading_day_span_counts_weekdays_inclusive():
    # Fri 07-24 → Mon 07-27 inclusive = Fri, Mon = 2 weekdays (Sat/Sun skipped).
    n = J._trading_day_span(_iso("2026-07-24T13:30:00Z"),
                            _iso("2026-07-27T20:00:00Z"))
    assert n == 2


# ─────────────────────────── window sizing (§6) ─────────────────────────────

def test_size_window_basic_since_floor_until_pad():
    trades = [_trade("2026-07-24T14:00:00Z", "2026-07-24T14:30:00Z"),
              _trade("2026-07-24T15:00:00Z", "2026-07-24T15:30:00Z")]
    w = J.size_window(trades, max_span_trading_days=5)
    assert w["since"] == _iso("2026-07-24T13:30:00Z")            # session floor
    assert w["until"] == _iso("2026-07-24T15:30:00Z") + timedelta(seconds=120)
    assert w["capped"] is False
    assert w["bt_trade_count"] == 2


def test_size_window_span_cap_drops_oldest_and_marks_capped():
    trades = [_trade("2026-07-14T14:00:00Z", "2026-07-14T14:30:00Z"),
              _trade("2026-07-18T14:00:00Z", "2026-07-18T14:30:00Z"),
              _trade("2026-07-24T14:00:00Z", "2026-07-24T14:30:00Z")]
    w = J.size_window(trades, max_span_trading_days=2)
    assert w["capped"] is True
    assert w["bt_trade_count"] == 1                              # only newest fits
    assert w["covered"][0]["entry_fill_ts"] == "2026-07-24T14:00:00Z"
    assert J._trading_day_span(w["since"], w["until"]) <= 2


def test_size_window_empty_input():
    w = J.size_window([], max_span_trading_days=5)
    assert w["covered"] == [] and w["since"] is None and w["capped"] is False


def test_size_window_never_caps_below_one_trade():
    # A single trade whose span exceeds the cap is still replayed (K>=1), capped.
    trades = [_trade("2026-07-06T14:00:00Z", "2026-07-13T14:30:00Z")]  # ~6 wkdays
    w = J.size_window(trades, max_span_trading_days=2)
    assert w["bt_trade_count"] == 1


# ─────────────────────────── coverage (§8 D4) ───────────────────────────────

def test_coverage_zero_when_no_decision_time_bars():
    cov = J.measure_coverage(_iso("2026-07-24T13:30:00Z"),
                             _iso("2026-07-24T20:00:00Z"), 60, n_closes=0)
    assert cov == 0.0                                            # → caller: unres


def test_coverage_full_and_partial():
    since, until = _iso("2026-07-24T13:30:00Z"), _iso("2026-07-24T20:00:00Z")
    full = J.measure_coverage(since, until, 60, n_closes=10_000)  # >> expected
    assert full == 1.0
    part = J.measure_coverage(since, until, 60, n_closes=195)     # ~half of 390
    assert 0.0 < part < 1.0


# ─────────────────────────── divergence ledger (§8) ─────────────────────────

def _ids(ledger):
    return [e["id"] for e in ledger]


def test_ledger_structural_always_present():
    led = J.build_divergence_ledger({}, 1.0, False, 120, "fp123",
                                    "2026-07-26T00:00:00Z", corrected=False)
    for d in ("D1", "D2", "D3", "D5", "D6", "D9", "D10", "D11", "D4"):
        assert d in _ids(led)


def test_ledger_d10_absent_when_refresher_retired():
    led = J.build_divergence_ledger({}, 1.0, False, 0, "fp", "t", corrected=False)
    assert "D10" not in _ids(led)


def test_ledger_d7_unres_and_d8_drift_footnotes():
    diag = {"unresolved": ["GEN-X: GEN-PACK-MISSING"],
            "label_drift": ["10S-STOCH: prefix 10Sec never matches"]}
    led = J.build_divergence_ledger(diag, 1.0, False, 120, "fp", "t", False)
    assert "D7" in _ids(led) and "D8" in _ids(led)


def test_ledger_window_capped_surface_and_corrected():
    led = J.build_divergence_ledger({}, 0.8, True, 120, "fp", "t", corrected=True)
    d11 = [e for e in led if e["id"] == "D11"][0]
    assert "window-capped" in d11["surface"]
    assert "corrected" in _ids(led)


# ─────────────────────────── status classification ──────────────────────────

def test_classify_unres_beats_everything():
    assert J.classify_status({"unresolved": ["x"]}, 5, True, 1.0) == "unres"
    assert J.classify_status({}, 0, False, 1.0) == "unres"
    assert J.classify_status({}, 5, False, 0.0) == "unres"


def test_classify_partial_and_ok():
    assert J.classify_status({}, 5, True, 1.0) == "partial"       # capped
    assert J.classify_status({}, 5, False, 0.5) == "partial"      # coverage<1
    assert J.classify_status({}, 5, False, 1.0) == "ok"


# ─────────────────────────── flag fingerprint (§8 D9) ───────────────────────

def test_flags_fingerprint_stable_and_sensitive():
    a = {"RORT_MTF_PB_DEFER": "1", "RORT_GATE_FAIL_CLOSED": "1"}
    b = {"RORT_GATE_FAIL_CLOSED": "1", "RORT_MTF_PB_DEFER": "1"}   # reordered
    c = {"RORT_MTF_PB_DEFER": "0", "RORT_GATE_FAIL_CLOSED": "1"}
    assert J.flags_fingerprint(a) == J.flags_fingerprint(b)       # order-independent
    assert J.flags_fingerprint(a) != J.flags_fingerprint(c)       # value-sensitive
    assert len(J.flags_fingerprint(a)) == 12


# ─────────────────────────── staleness (§4.3) ───────────────────────────────

def test_staleness_covered_subset_and_flag():
    since, until = _iso("2026-07-24T13:30:00Z"), _iso("2026-07-24T16:00:00Z")
    inside = _trade("2026-07-24T14:00:00Z", "2026-07-24T14:30:00Z")
    beyond = _trade("2026-07-24T15:30:00Z", "2026-07-24T17:00:00Z")  # exit past until
    s = J.apply_staleness([inside, beyond], since, until)
    assert s["n_covered"] == 1 and s["stale"] is True


def test_staleness_none_window():
    s = J.apply_staleness([_trade("2026-07-24T14:00:00Z", "2026-07-24T14:30:00Z")],
                          None, None)
    assert s["covered"] == [] and s["stale"] is True


# ─────────────────────────── fake Supabase client ───────────────────────────

class _Res:
    def __init__(self, data):
        self.data = data


class _Q:
    """Ignores every filter; returns the table's next canned response. A table
    with multiple canned responses advances one per .execute() (single-response
    tables keep returning the same), so a sequence of queries on one table can
    return different rows (the score_sim_basis none→fallback path)."""
    def __init__(self, dq):
        self._dq = dq
    def __getattr__(self, _name):
        return lambda *a, **k: self          # select/eq/in_/like/order/limit/...
    @property
    def not_(self):
        return self
    def execute(self):
        if not self._dq:
            return _Res([])
        return _Res(self._dq.popleft() if len(self._dq) > 1 else self._dq[0])


class FakeClient:
    def __init__(self, tables):
        # tables: {name: [resp, resp, ...]}
        self._tables = {k: deque(v) for k, v in tables.items()}
    def table(self, name):
        return _Q(self._tables.get(name, deque([[]])))


# ─────────────────────────── score_sim_basis (read path) ────────────────────

def _score(sid, tables, tol=10.0):
    from api.routers.replay_sim import score_sim_basis
    return score_sim_basis(FakeClient(tables), sid, tol)


def test_score_none_when_no_cache():
    out = _score(1, {"replay_sim_basis": [[], []], "trades": [[]]})
    assert out["status"] == "none"


def test_score_unres_fallback_surfaces_ledger():
    unres_row = {"status": "unres", "divergences": [{"id": "D7"}],
                 "computed_at": "2026-07-26T00:00:00Z"}
    out = _score(1, {"replay_sim_basis": [[], [unres_row]], "trades": [[]]})
    assert out["status"] == "unres"
    assert out["divergences"] == [{"id": "D7"}]


def test_score_pairs_cached_edges_vs_current_bt():
    row = {
        "id": 42, "status": "ok",
        "window_since": "2026-07-24T13:30:00Z",
        "window_until": "2026-07-24T16:00:00Z",
        "replay_entries": ["2026-07-24T14:00:00+00:00"],
        "replay_exits": ["2026-07-24T14:30:00+00:00"],
        "coverage": 1.0, "corrected": False, "divergences": [],
        "flags_fp": "abc", "computed_at": "2026-07-26T00:00:00Z",
        "compute_secs": 30.0, "engine_sha": "deadbee",
    }
    # bt trade within 10s of both replay edges → both pair.
    bt = [_trade("2026-07-24T14:00:03Z", "2026-07-24T14:30:02Z")]
    out = _score(1, {"replay_sim_basis": [[row]], "trades": [bt],
                     "system_settings": [[]]})
    assert out["status"] == "ok"
    assert out["points"] == 2 and out["denom"] == 2
    assert out["trade_count"] == 1
    assert out["stale"] is False
    assert out["flags_stale"] is False


def test_score_flags_stale_when_fingerprint_moved():
    row = {"id": 1, "status": "ok",
           "window_since": "2026-07-24T13:30:00Z",
           "window_until": "2026-07-24T16:00:00Z",
           "replay_entries": [], "replay_exits": [], "coverage": 1.0,
           "corrected": False, "divergences": [], "flags_fp": "abc",
           "computed_at": "2026-07-26T00:00:00Z"}
    out = _score(1, {"replay_sim_basis": [[row]], "trades": [[]],
                     "system_settings": [[{"value": {"fp": "xyz"}}]]})
    assert out["flags_stale"] is True


def test_score_stale_when_bt_extends_past_window():
    row = {"id": 1, "status": "ok",
           "window_since": "2026-07-24T13:30:00Z",
           "window_until": "2026-07-24T16:00:00Z",
           "replay_entries": ["2026-07-24T14:00:00+00:00"], "replay_exits": [],
           "coverage": 1.0, "corrected": False, "divergences": [],
           "flags_fp": "abc", "computed_at": "2026-07-26T00:00:00Z"}
    bt = [_trade("2026-07-24T14:00:00Z", "2026-07-24T14:30:00Z"),
          _trade("2026-07-24T18:00:00Z", "2026-07-24T18:30:00Z")]  # past window
    out = _score(1, {"replay_sim_basis": [[row]], "trades": [bt],
                     "system_settings": [[]]})
    assert out["stale"] is True
    assert out["covered_of_total"] == [1, 2]


# ─────────────────────────── poller guardrails ──────────────────────────────

def test_offhours_window():
    import replay_sim_poller as P
    # July 2026 = EDT (UTC-4).
    assert P.in_offhours_window(_iso("2026-07-25T18:00:00Z")) is True   # Sat
    assert P.in_offhours_window(_iso("2026-07-24T14:00:00Z")) is False  # Fri 10:00 ET
    assert P.in_offhours_window(_iso("2026-07-24T21:00:00Z")) is True   # Fri 17:00 ET
    assert P.in_offhours_window(_iso("2026-07-24T12:00:00Z")) is True   # Fri 08:00 ET


def test_no_active_recompute_guard():
    import replay_sim_poller as P
    assert P.no_active_recompute(FakeClient({"compute_jobs": [[]]})) is True
    assert P.no_active_recompute(
        FakeClient({"compute_jobs": [[{"id": "x", "status": "running"}]]})) is False


def test_is_cache_fresh_guard():
    import replay_sim_poller as P
    now = _iso("2026-07-26T00:00:00Z")
    fresh = [{"computed_at": "2026-07-25T00:00:00Z"}]      # 1 day old
    old = [{"computed_at": "2026-07-10T00:00:00Z"}]        # 16 days old
    assert P.is_cache_fresh(FakeClient({"replay_sim_basis": [fresh]}), 1, 3, now) is True
    assert P.is_cache_fresh(FakeClient({"replay_sim_basis": [old]}), 1, 3, now) is False
    assert P.is_cache_fresh(FakeClient({"replay_sim_basis": [[]]}), 1, 3, now) is False
