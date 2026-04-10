"""Sanity check: C-type swing stop must NOT fire on a wick-only bar.

Builds a synthetic LONG position with a swing stop and steps the engine
through bars that wick past the stop but close above it. Verifies:
  1. C-type stop does NOT fire (no exit)
  2. L-type stop DOES fire (for comparison)
  3. When C-type stop DOES fire (close past), exit_price == bar close
     (NOT stop_price — guards against accidental L-type fill)
"""

from unified_engine import PositionStateMachine


def _make_psm(stop_exec='C'):
    strategy = {
        'id': 'test',
        'direction': 'LONG',
        'entry_trigger': 'dummy',
        'exit_triggers': [],
        'confluence': [],
        'stop_config': {'method': 'swing', 'lookback': 3, 'padding': 0.0,
                        'exec_type': stop_exec},
        'target_config': None,
    }
    psm = PositionStateMachine(strategy)
    psm.state.status = 'IN_POSITION'
    psm.state.direction = 'LONG'
    psm.state.entry_price = 100.0
    psm.state.stop_price = 99.0  # Stop at 99 — wicks below shouldn't trigger C
    psm.state.initial_stop_price = 99.0
    psm.state.target_price = None
    psm.state.entry_bar_count = 0
    psm.state.exec_type = 'C'
    return psm


def test_c_type_ignores_wick():
    """LONG with C-type swing stop @ 99: bar wicks to 98 but closes at 99.5
    must NOT trigger the stop."""
    psm = _make_psm(stop_exec='C')
    fill = psm._check_stop_hit(
        bar_open=100.0, high=100.5, low=98.0, close=99.5, bar_count=5)
    assert fill is None, f'C-type should ignore wick! got fill={fill}'
    print('  [OK] test_c_type_ignores_wick — wick at 98, close 99.5, no fire')


def test_l_type_fires_on_wick():
    """Same bar with L-type stop SHOULD fire."""
    psm = _make_psm(stop_exec='L')
    fill = psm._check_stop_hit(
        bar_open=100.0, high=100.5, low=98.0, close=99.5, bar_count=5)
    assert fill is not None, 'L-type should fire on wick'
    # L-type fill is gap-aware: min(stop, open) for LONG → min(99, 100) = 99
    assert fill == 99.0, f'L-type fill should be stop=99.0, got {fill}'
    print('  [OK] test_l_type_fires_on_wick — fill=99.0 (stop level)')


def test_c_type_fires_on_close_past_with_close_fill():
    """LONG with C-type stop @ 99: bar that closes at 98.5 (past stop)
    fires the stop, with fill_price == close (NOT stop)."""
    psm = _make_psm(stop_exec='C')
    fill = psm._check_stop_hit(
        bar_open=99.5, high=99.8, low=98.0, close=98.5, bar_count=5)
    assert fill is not None, 'C-type should fire when close past stop'
    assert fill == 98.5, \
        f'C-type fill must be CLOSE (98.5), got {fill}. ' \
        'If this returned 99.0 (stop), the C-type is silently L-type — ' \
        'likely a look-ahead-equivalent bug.'
    print('  [OK] test_c_type_fires_on_close_past_with_close_fill — fill=98.5')


def test_get_trade_record_carries_exec_types():
    """The trade record built by get_trade_record() must include
    stop_exec_type and target_exec_type so the chart can shift markers."""
    psm = _make_psm(stop_exec='C')
    psm.state.entry_time = '2026-04-09T09:30:00'
    rec = psm.get_trade_record(
        exit_price=98.5, exit_time='2026-04-09T09:35:00',
        exit_reason='stop_loss', exit_trigger='stop_loss', bar_count=5)
    assert rec['stop_exec_type'] == 'C', \
        f'Trade record missing stop_exec_type=C: {rec.get("stop_exec_type")}'
    assert rec['target_exec_type'] == 'L', \
        f'Trade record target_exec_type default mismatch: ' \
        f'{rec.get("target_exec_type")} (expected L for None target_config)'
    assert rec['exit_price'] == 98.5
    print('  [OK] test_get_trade_record_carries_exec_types')


TESTS = [
    test_c_type_ignores_wick,
    test_l_type_fires_on_wick,
    test_c_type_fires_on_close_past_with_close_fill,
    test_get_trade_record_carries_exec_types,
]


if __name__ == '__main__':
    import sys
    print('=' * 60)
    print('C-type swing stop sanity tests')
    print('=' * 60)
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'  [FAIL] {t.__name__}: {e}')
            failed += 1
        except Exception as e:
            print(f'  [ERR]  {t.__name__}: {type(e).__name__}: {e}')
            failed += 1
    print('=' * 60)
    print(f'Results: {len(TESTS) - failed} passed, {failed} failed')
    sys.exit(0 if failed == 0 else 1)
