"""Tier 3 §8.4 regression — KPI boundary filter.

Verifies that `split_trades_at_boundary` (services.py) implements the
Tier 3 §6.3 contract: trades opened AT or AFTER the boundary count
toward the post-boundary (forward) window.

Run: cd src && .venv-relative-python _test_tier3_kpi_boundary.py
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from services import split_trades_at_boundary


def _make_trades(rows):
    """Build a minimal trades DF from a list of (entry_time, label) tuples."""
    return pd.DataFrame({
        'entry_time': pd.to_datetime([r[0] for r in rows], utc=True),
        'label': [r[1] for r in rows],
    })


def test_split_uses_ge_for_forward():
    """The forward bucket should include trades AT the boundary
    (Tier 3 §6.3: 'trades opened at or after the boundary count')."""
    boundary = pd.Timestamp('2026-04-01 00:00:00', tz='UTC')
    trades = _make_trades([
        ('2026-03-15 10:00:00+00:00', 'pre'),
        ('2026-03-31 23:59:59+00:00', 'pre_edge'),
        ('2026-04-01 00:00:00+00:00', 'AT_BOUNDARY'),  # the critical one
        ('2026-04-01 00:00:01+00:00', 'just_after'),
        ('2026-04-15 14:00:00+00:00', 'forward'),
    ])
    backtest, forward = split_trades_at_boundary(trades, boundary)

    bt_labels = set(backtest['label'])
    fwd_labels = set(forward['label'])

    expected_bt = {'pre', 'pre_edge'}
    expected_fwd = {'AT_BOUNDARY', 'just_after', 'forward'}

    assert bt_labels == expected_bt, (
        f"backtest mismatch: got {bt_labels}, want {expected_bt}"
    )
    assert fwd_labels == expected_fwd, (
        f"forward mismatch: got {fwd_labels}, want {expected_fwd}"
    )
    print("  ✓ AT_BOUNDARY trade goes to forward bucket (§6.3 contract)")
    print(f"    backtest={sorted(bt_labels)}, forward={sorted(fwd_labels)}")


def test_empty_input():
    """Defensive: empty trades input returns two empty frames."""
    boundary = pd.Timestamp('2026-04-01 00:00:00', tz='UTC')
    bt, fwd = split_trades_at_boundary(pd.DataFrame(), boundary)
    assert len(bt) == 0 and len(fwd) == 0
    print("  ✓ empty trades input handled cleanly")


def test_all_before_boundary():
    """All trades pre-boundary → backtest gets all, forward empty."""
    boundary = pd.Timestamp('2026-04-01 00:00:00', tz='UTC')
    trades = _make_trades([
        ('2026-03-01 10:00:00+00:00', 'a'),
        ('2026-03-15 14:00:00+00:00', 'b'),
    ])
    bt, fwd = split_trades_at_boundary(trades, boundary)
    assert len(bt) == 2 and len(fwd) == 0
    print("  ✓ all-pre-boundary handled correctly")


def test_all_at_or_after_boundary():
    """All trades >= boundary → forward gets all, backtest empty."""
    boundary = pd.Timestamp('2026-04-01 00:00:00', tz='UTC')
    trades = _make_trades([
        ('2026-04-01 00:00:00+00:00', 'a'),  # AT
        ('2026-04-15 14:00:00+00:00', 'b'),
    ])
    bt, fwd = split_trades_at_boundary(trades, boundary)
    assert len(bt) == 0 and len(fwd) == 2
    print("  ✓ all-at-or-after-boundary handled correctly")


def test_naive_boundary_with_utc_trades():
    """A naive boundary timestamp should be localized to match the
    trades column's tz so the comparison succeeds. (Defensive: this
    used to bite us in app.py via the timezone mismatch gotcha.)"""
    import datetime as _dt
    boundary_naive = _dt.datetime(2026, 4, 1)  # naive
    trades = _make_trades([
        ('2026-03-15 10:00:00+00:00', 'pre'),
        ('2026-04-15 10:00:00+00:00', 'forward'),
    ])
    bt, fwd = split_trades_at_boundary(trades, boundary_naive)
    assert set(bt['label']) == {'pre'}
    assert set(fwd['label']) == {'forward'}
    print("  ✓ naive boundary auto-localizes to trades column tz")


if __name__ == '__main__':
    print("\n=== Tier 3 §8.4 KPI boundary regression ===\n")
    test_split_uses_ge_for_forward()
    test_empty_input()
    test_all_before_boundary()
    test_all_at_or_after_boundary()
    test_naive_boundary_with_utc_trades()
    print("\n=== ALL PASS ===\n")
