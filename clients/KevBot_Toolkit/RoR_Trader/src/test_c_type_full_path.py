"""End-to-end check_exit test for C-type stops.

Verifies that when check_exit() processes a bar that closes past the stop,
the resulting trade record has:
  - exit_price = bar's close (NOT stop_price)
  - r_multiple = pnl / risk where pnl reflects the actual close-vs-entry move

This catches any bug where C-type stops accidentally fill at stop_price
(which would always cap losses at exactly -1R).
"""

from collections import deque
from unified_engine import PositionStateMachine


def _make_psm(stop_exec='C', direction='LONG', entry=100.0, stop=99.0):
    strategy = {
        'id': 'test',
        'direction': direction,
        'entry_trigger': 'dummy',
        'exit_triggers': [],
        'confluence': [],
        'stop_config': {'method': 'atr', 'atr_mult': 1.0, 'exec_type': stop_exec},
        'target_config': None,
    }
    psm = PositionStateMachine(strategy)
    psm.state.status = 'IN_POSITION'
    psm.state.direction = direction
    psm.state.entry_price = entry
    psm.state.stop_price = stop
    psm.state.initial_stop_price = stop
    psm.state.target_price = None
    psm.state.entry_bar_count = 0
    psm.state.exec_type = 'C'  # entry exec_type — C means stop checks aren't skipped on entry bar
    return psm


def test_c_stop_long_close_past_full_record():
    """Drive check_exit through a bar that closes past the stop and inspect
    the trade record. exit_price MUST be the close, r_multiple MUST be < -1."""
    psm = _make_psm(stop_exec='C', direction='LONG', entry=100.0, stop=99.0)
    # Bar: opens at 99.5, wicks down to 97.0, closes at 98.0 (1R past stop)
    current = {'open': 99.5, 'high': 99.8, 'low': 97.0,
               'close': 98.0, 'volume': 1000}
    record = psm.check_exit(
        trigger_booleans={}, current_values=current,
        bar_count=10, bar_time='2026-04-09T10:00:00')
    assert record is not None, 'C-stop must fire on close past'
    assert record['exit_price'] == 98.0, \
        f'exit_price should be CLOSE (98.0), got {record["exit_price"]}. ' \
        f'If this is 99.0 (stop), C-type is filling at stop level — BUG.'
    # risk = 100-99 = 1, pnl = 98-100 = -2, r = -2.0
    assert record['r_multiple'] == -2.0, \
        f'r_multiple should be -2.0 (close 2R past stop), got {record["r_multiple"]}'
    print(f'  [OK] LONG C-stop fills at close, r_mult={record["r_multiple"]}')


def test_l_stop_long_for_comparison():
    """Same bar with L-type stop fills at stop level → r=-1."""
    psm = _make_psm(stop_exec='L', direction='LONG', entry=100.0, stop=99.0)
    # entry_bar_count=0 and bar_count=10 → not entry bar, ltype check ok
    current = {'open': 99.5, 'high': 99.8, 'low': 97.0,
               'close': 98.0, 'volume': 1000}
    record = psm.check_exit(
        trigger_booleans={}, current_values=current,
        bar_count=10, bar_time='2026-04-09T10:00:00')
    assert record is not None
    # L-type fill = min(stop, open) = min(99, 99.5) = 99
    assert record['exit_price'] == 99.0, \
        f'L-type fill should be 99 (stop), got {record["exit_price"]}'
    assert record['r_multiple'] == -1.0, \
        f'L-type r should be -1.0, got {record["r_multiple"]}'
    print(f'  [OK] LONG L-stop fills at stop, r_mult={record["r_multiple"]}')


def test_c_stop_short_close_past_full_record():
    """SHORT C-stop. Stop above entry. Bar closes ABOVE stop → exit at close."""
    psm = _make_psm(stop_exec='C', direction='SHORT', entry=100.0, stop=101.0)
    # Bar: opens at 100.5, wicks up to 103.0, closes at 102.0 (1R past stop)
    current = {'open': 100.5, 'high': 103.0, 'low': 100.2,
               'close': 102.0, 'volume': 1000}
    record = psm.check_exit(
        trigger_booleans={}, current_values=current,
        bar_count=10, bar_time='2026-04-09T10:00:00')
    assert record is not None
    assert record['exit_price'] == 102.0, \
        f'SHORT C-stop fill should be 102 (close), got {record["exit_price"]}'
    # risk = 101-100 = 1, pnl = 100 - 102 = -2 (SHORT lost 2)
    assert record['r_multiple'] == -2.0, \
        f'SHORT r should be -2.0, got {record["r_multiple"]}'
    print(f'  [OK] SHORT C-stop fills at close, r_mult={record["r_multiple"]}')


def test_l_type_entry_skips_stop_on_entry_bar():
    """If entry was L-type, the entry bar's stop check is skipped (preserved
    legacy behavior). Verify this doesn't accidentally apply to non-entry bars."""
    psm = _make_psm(stop_exec='C', direction='LONG', entry=100.0, stop=99.0)
    psm.state.exec_type = 'L'  # L-type entry
    psm.state.entry_bar_count = 10
    # Same bar as entry — should be skipped
    current = {'open': 99.5, 'high': 99.8, 'low': 97.0,
               'close': 98.0, 'volume': 1000}
    record = psm.check_exit(
        trigger_booleans={}, current_values=current,
        bar_count=10, bar_time='2026-04-09T10:00:00')
    assert record is None, \
        f'L-entry on same bar must skip stop check, got record={record}'
    # Now next bar — should fire
    record2 = psm.check_exit(
        trigger_booleans={}, current_values=current,
        bar_count=11, bar_time='2026-04-09T10:01:00')
    assert record2 is not None
    assert record2['exit_price'] == 98.0
    assert record2['r_multiple'] == -2.0
    print(f'  [OK] L-entry skips entry bar, fires at close on bar+1')


TESTS = [
    test_c_stop_long_close_past_full_record,
    test_l_stop_long_for_comparison,
    test_c_stop_short_close_past_full_record,
    test_l_type_entry_skips_stop_on_entry_bar,
]


if __name__ == '__main__':
    import sys
    print('=' * 60)
    print('check_exit() end-to-end C-type tests')
    print('=' * 60)
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'  [FAIL] {t.__name__}:\n         {e}')
            failed += 1
        except Exception as e:
            print(f'  [ERR]  {t.__name__}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            failed += 1
    print('=' * 60)
    print(f'Results: {len(TESTS) - failed} passed, {failed} failed')
    sys.exit(0 if failed == 0 else 1)
