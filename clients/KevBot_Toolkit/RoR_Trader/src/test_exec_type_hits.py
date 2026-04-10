"""Unit tests for exec_type-aware stop/target hit detection.

Validates `PositionStateMachine._check_stop_hit` and `_check_target_hit`
across all four exec types: L, C, LC, CC.

Run: cd src && ../.venv/bin/python test_exec_type_hits.py
"""

from unified_engine import PositionStateMachine, PositionState


def _make_psm(direction='LONG', stop=99.0, target=103.0,
              stop_exec='L', target_exec='L'):
    """Build a PositionStateMachine in a position with the given config."""
    strategy = {
        'id': 'test',
        'direction': direction,
        'entry_trigger': 'dummy',
        'exit_triggers': [],
        'confluence': [],
        'stop_config': {'method': 'atr', 'atr_mult': 1.5,
                        'exec_type': stop_exec},
        'target_config': {'method': 'fixed_dollar', 'dollar_amount': 3.0,
                          'exec_type': target_exec},
    }
    psm = PositionStateMachine(strategy)
    psm.state.status = 'IN_POSITION'
    psm.state.direction = direction
    psm.state.entry_price = 100.0
    psm.state.stop_price = stop
    psm.state.initial_stop_price = stop
    psm.state.target_price = target
    psm.state.entry_bar_count = 0
    psm.state.exec_type = 'C'
    return psm


# ==============================================================================
# L-type (default — intra-bar touch)
# ==============================================================================

def test_l_stop_long_touched_no_gap():
    """LONG L-stop fires on touch, fills at stop (no gap)."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='L')
    fill = psm._check_stop_hit(bar_open=100.0, high=100.5, low=98.5,
                               close=99.5, bar_count=5)
    assert fill == 99.0, f'Expected fill=99.0, got {fill}'


def test_l_stop_long_gap_down():
    """LONG L-stop with gap-down fills at open (worse than stop)."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='L')
    fill = psm._check_stop_hit(bar_open=98.0, high=98.5, low=97.5,
                               close=98.0, bar_count=5)
    assert fill == 98.0, f'Expected fill=98.0 (gap), got {fill}'


def test_l_stop_short_touched():
    """SHORT L-stop fires on touch (high crosses stop)."""
    psm = _make_psm(direction='SHORT', stop=101.0, stop_exec='L')
    fill = psm._check_stop_hit(bar_open=100.0, high=101.5, low=99.5,
                               close=100.5, bar_count=5)
    assert fill == 101.0, f'Expected fill=101.0, got {fill}'


def test_l_stop_short_gap_up():
    """SHORT L-stop with gap-up fills at open."""
    psm = _make_psm(direction='SHORT', stop=101.0, stop_exec='L')
    fill = psm._check_stop_hit(bar_open=102.0, high=102.5, low=101.5,
                               close=102.0, bar_count=5)
    assert fill == 102.0, f'Expected fill=102.0 (gap), got {fill}'


def test_l_stop_no_touch():
    """L-stop ignored when low never crosses stop."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='L')
    fill = psm._check_stop_hit(bar_open=100.5, high=101.0, low=99.5,
                               close=100.0, bar_count=5)
    assert fill is None


# ==============================================================================
# C-type (bar close past — filters wicks)
# ==============================================================================

def test_c_stop_wick_does_not_fire():
    """C-stop ignores wicks: low touches but close above stop → no exit."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='C')
    fill = psm._check_stop_hit(bar_open=100.0, high=100.5, low=98.5,
                               close=99.5, bar_count=5)
    assert fill is None, f'C-stop should ignore wick, got {fill}'


def test_c_stop_close_past_fires():
    """C-stop fires when bar closes past stop (fill = close)."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='C')
    fill = psm._check_stop_hit(bar_open=99.5, high=99.8, low=98.0,
                               close=98.5, bar_count=5)
    assert fill == 98.5, f'Expected fill=98.5 (close), got {fill}'


def test_c_stop_short_wick_ignored():
    """SHORT C-stop ignores upper wick if close still below stop."""
    psm = _make_psm(direction='SHORT', stop=101.0, stop_exec='C')
    fill = psm._check_stop_hit(bar_open=100.0, high=101.5, low=99.5,
                               close=100.5, bar_count=5)
    assert fill is None


def test_c_target_close_past_fires():
    """C-target fires on bar close past target."""
    psm = _make_psm(direction='LONG', target=103.0, target_exec='C')
    fill = psm._check_target_hit(bar_open=102.5, high=104.0, low=102.0,
                                 close=103.5, bar_count=5)
    assert fill == 103.5


def test_c_target_wick_ignored():
    """C-target ignores intra-bar touch when close is below target."""
    psm = _make_psm(direction='LONG', target=103.0, target_exec='C')
    fill = psm._check_target_hit(bar_open=102.0, high=103.5, low=101.5,
                                 close=102.5, bar_count=5)
    assert fill is None


# ==============================================================================
# LC-type (touch + close confirms)
# ==============================================================================

def test_lc_stop_touch_only_no_fire():
    """LC-stop: touch but close above stop → no fire."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='LC')
    fill = psm._check_stop_hit(bar_open=100.0, high=100.5, low=98.5,
                               close=99.5, bar_count=5)
    assert fill is None


def test_lc_stop_no_touch_no_close_no_fire():
    """LC-stop: neither touch nor close past → no fire."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='LC')
    fill = psm._check_stop_hit(bar_open=100.5, high=101.0, low=99.5,
                               close=100.0, bar_count=5)
    assert fill is None


def test_lc_stop_touch_and_close_fires():
    """LC-stop: both touch and close past → fires at close."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='LC')
    fill = psm._check_stop_hit(bar_open=99.5, high=99.8, low=98.0,
                               close=98.5, bar_count=5)
    assert fill == 98.5, f'Expected fill=98.5 (close), got {fill}'


def test_lc_target_short_touch_and_close():
    """SHORT LC-target: low touches target AND close below target → fires."""
    psm = _make_psm(direction='SHORT', target=97.0, target_exec='LC')
    fill = psm._check_target_hit(bar_open=98.0, high=98.5, low=96.5,
                                 close=96.8, bar_count=5)
    assert fill == 96.8, f'Expected fill=96.8 (close), got {fill}'


def test_lc_target_touch_only_no_fire():
    """LC-target: low touches target but close above → no fire."""
    psm = _make_psm(direction='SHORT', target=97.0, target_exec='LC')
    fill = psm._check_target_hit(bar_open=98.0, high=98.5, low=96.5,
                                 close=97.5, bar_count=5)
    assert fill is None


# ==============================================================================
# CC-type (close past + next bar close also past)
# ==============================================================================

def test_cc_stop_first_bar_arms_pending():
    """CC-stop: first close-past bar arms pending state (no fire yet)."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='CC')
    fill = psm._check_stop_hit(bar_open=99.5, high=99.8, low=98.0,
                               close=98.5, bar_count=5)
    assert fill is None
    assert psm.state.pending_stop_confirm_bar == 5


def test_cc_stop_two_bar_confirm_fires():
    """CC-stop: two consecutive closes past confirm exit."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='CC')
    # Bar 5: first close past — arms
    fill1 = psm._check_stop_hit(bar_open=99.5, high=99.8, low=98.0,
                                close=98.5, bar_count=5)
    assert fill1 is None
    # Bar 6: second close past — confirms
    fill2 = psm._check_stop_hit(bar_open=98.5, high=98.8, low=97.5,
                                close=98.0, bar_count=6)
    assert fill2 == 98.0, f'Expected fill=98.0 (confirm close), got {fill2}'
    assert psm.state.pending_stop_confirm_bar is None


def test_cc_stop_confirmation_failure_resets():
    """CC-stop: if next bar closes back above stop, pending resets."""
    psm = _make_psm(direction='LONG', stop=99.0, stop_exec='CC')
    # Bar 5: arms
    psm._check_stop_hit(bar_open=99.5, high=99.8, low=98.0,
                        close=98.5, bar_count=5)
    assert psm.state.pending_stop_confirm_bar == 5
    # Bar 6: closes back above → reset, no fire
    fill = psm._check_stop_hit(bar_open=98.5, high=99.5, low=98.5,
                               close=99.5, bar_count=6)
    assert fill is None
    assert psm.state.pending_stop_confirm_bar is None


def test_cc_target_two_bar_confirm():
    """CC-target: two closes past target → fires on second close."""
    psm = _make_psm(direction='LONG', target=103.0, target_exec='CC')
    # Bar 5: first close past — arms
    fill1 = psm._check_target_hit(bar_open=102.5, high=103.5, low=102.5,
                                  close=103.2, bar_count=5)
    assert fill1 is None
    assert psm.state.pending_target_confirm_bar == 5
    # Bar 6: second close past — confirms at close
    fill2 = psm._check_target_hit(bar_open=103.2, high=103.8, low=103.0,
                                  close=103.5, bar_count=6)
    assert fill2 == 103.5
    assert psm.state.pending_target_confirm_bar is None


def test_cc_short_two_bar_confirm():
    """SHORT CC-stop: two consecutive closes above stop → confirms."""
    psm = _make_psm(direction='SHORT', stop=101.0, stop_exec='CC')
    # Bar 5: first close past
    fill1 = psm._check_stop_hit(bar_open=100.5, high=101.8, low=100.0,
                                close=101.5, bar_count=5)
    assert fill1 is None
    # Bar 6: second close past
    fill2 = psm._check_stop_hit(bar_open=101.5, high=102.0, low=101.2,
                                close=101.8, bar_count=6)
    assert fill2 == 101.8


# ==============================================================================
# Test runner
# ==============================================================================

TESTS = [
    # L-type
    test_l_stop_long_touched_no_gap,
    test_l_stop_long_gap_down,
    test_l_stop_short_touched,
    test_l_stop_short_gap_up,
    test_l_stop_no_touch,
    # C-type
    test_c_stop_wick_does_not_fire,
    test_c_stop_close_past_fires,
    test_c_stop_short_wick_ignored,
    test_c_target_close_past_fires,
    test_c_target_wick_ignored,
    # LC-type
    test_lc_stop_touch_only_no_fire,
    test_lc_stop_no_touch_no_close_no_fire,
    test_lc_stop_touch_and_close_fires,
    test_lc_target_short_touch_and_close,
    test_lc_target_touch_only_no_fire,
    # CC-type
    test_cc_stop_first_bar_arms_pending,
    test_cc_stop_two_bar_confirm_fires,
    test_cc_stop_confirmation_failure_resets,
    test_cc_target_two_bar_confirm,
    test_cc_short_two_bar_confirm,
]


def main():
    print('=' * 60)
    print('Exec-type stop/target hit detection tests')
    print('=' * 60)
    passed = 0
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'  [OK]   {t.__name__}')
            passed += 1
        except AssertionError as e:
            print(f'  [FAIL] {t.__name__}: {e}')
            failed += 1
        except Exception as e:
            print(f'  [ERR]  {t.__name__}: {type(e).__name__}: {e}')
            failed += 1
    print('=' * 60)
    print(f'Results: {passed} passed, {failed} failed')
    print('=' * 60)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
