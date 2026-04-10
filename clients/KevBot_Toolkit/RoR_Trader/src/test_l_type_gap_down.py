"""Verify L-type stops produce > -1R losses on gap-down scenarios.

Tests two cases:
1. Bar opens AT or above stop, wicks below: L-type fills at stop, R = -1.0
2. Bar opens BELOW stop (gap-down): L-type fills at open, R < -1.0
"""

from unified_engine import PositionStateMachine


def _make_psm(stop_exec='L', entry=100.0, stop=99.0):
    strategy = {
        'id': 'test', 'direction': 'LONG',
        'entry_trigger': 'dummy', 'exit_triggers': [], 'confluence': [],
        'stop_config': {'method': 'atr', 'atr_mult': 1.0,
                        'exec_type': stop_exec},
        'target_config': None,
    }
    psm = PositionStateMachine(strategy)
    psm.state.status = 'IN_POSITION'
    psm.state.direction = 'LONG'
    psm.state.entry_price = entry
    psm.state.stop_price = stop
    psm.state.initial_stop_price = stop
    psm.state.entry_bar_count = 0
    psm.state.exec_type = 'C'  # not L — so same_bar_ltype is False
    return psm


def test_l_stop_no_gap_caps_at_minus_1():
    """Bar opens at 99.5 (above stop 99), wicks to 97. L fills at min(99, 99.5)=99, R=-1.0."""
    psm = _make_psm()
    current = {'open': 99.5, 'high': 99.8, 'low': 97.0, 'close': 98.5}
    rec = psm.check_exit({}, current, 5, 't')
    assert rec is not None
    assert rec['exit_price'] == 99.0
    assert rec['r_multiple'] == -1.0
    print(f'  [OK] no-gap: open=99.5 → fill=99.0, r=-1.0 (capped at risk distance)')


def test_l_stop_gap_down_fills_below_stop():
    """Bar opens at 95 (gap-down below stop 99). L fills at min(99, 95)=95.
    pnl = 95-100 = -5, risk = 1, R = -5.0 — much worse than -1R."""
    psm = _make_psm()
    current = {'open': 95.0, 'high': 96.0, 'low': 94.0, 'close': 95.5}
    rec = psm.check_exit({}, current, 5, 't')
    assert rec is not None
    assert rec['exit_price'] == 95.0, f'L fill should be open=95.0 (gap), got {rec["exit_price"]}'
    assert rec['r_multiple'] == -5.0, f'R should be -5.0 on gap-down, got {rec["r_multiple"]}'
    print(f'  [OK] gap-down: open=95.0 → fill=95.0, r=-5.0 (uncapped)')


def test_l_stop_huge_gap_short():
    """SHORT entry at 100, stop at 101. Bar opens at 200 (huge gap-up).
    L fills at max(101, 200)=200, pnl = 100-200 = -100, risk = 1, R = -100."""
    psm = _make_psm(entry=100.0, stop=101.0)
    psm.state.direction = 'SHORT'
    current = {'open': 200.0, 'high': 205.0, 'low': 195.0, 'close': 198.0}
    rec = psm.check_exit({}, current, 5, 't')
    assert rec is not None
    assert rec['exit_price'] == 200.0
    assert rec['r_multiple'] == -100.0, f'SHORT gap-up R should be -100, got {rec["r_multiple"]}'
    print(f'  [OK] short gap-up: open=200.0 → fill=200.0, r=-100.0 (uncapped)')


def test_c_stop_gap_down_fills_at_close():
    """C-type same scenario: fills at close (95.5), R = -4.5."""
    psm = _make_psm(stop_exec='C')
    current = {'open': 95.0, 'high': 96.0, 'low': 94.0, 'close': 95.5}
    rec = psm.check_exit({}, current, 5, 't')
    assert rec is not None
    assert rec['exit_price'] == 95.5
    assert rec['r_multiple'] == -4.5
    print(f'  [OK] C-type gap-down: fill=95.5 (close), r=-4.5')


TESTS = [
    test_l_stop_no_gap_caps_at_minus_1,
    test_l_stop_gap_down_fills_below_stop,
    test_l_stop_huge_gap_short,
    test_c_stop_gap_down_fills_at_close,
]


if __name__ == '__main__':
    import sys
    print('=' * 70)
    print('L-type gap-aware fill tests')
    print('=' * 70)
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'  [FAIL] {t.__name__}: {e}')
            failed += 1
    print('=' * 70)
    print(f'Results: {len(TESTS) - failed} passed, {failed} failed')
    print()
    print('CONCLUSION: L-type fills are gap-aware. Losses exceed -1R when the')
    print('bar OPENS past the stop (gap). Without a gap, every L-type fill is')
    print('at the stop level by definition → R = -1.0 exactly. Not a cap.')
    sys.exit(0 if failed == 0 else 1)
