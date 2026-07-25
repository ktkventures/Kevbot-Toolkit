"""Unit tests for health_last10._greedy_pair (board #122).

The E-sign-off nit: the walk used to return a set of paired edge
TIMESTAMPS, so two edges sharing one timestamp double-collapsed — one
matched alert marked both edges paired, inflating points. The fix keys
the returned set by index into the edges argument.

Run:  cd src && ../.venv/bin/pytest test_health_last10_pairing.py -v
"""
from api.routers.health_last10 import _greedy_pair


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
