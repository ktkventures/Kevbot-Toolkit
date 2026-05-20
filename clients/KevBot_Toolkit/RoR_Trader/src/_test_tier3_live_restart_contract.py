"""Tier 3 §8.3 regression — live restart contract.

Verifies that `StrategyMonitor.__init__` enforces the always-start-flat
contract regardless of whether saved-state load_engine_state() returns
an IN_POSITION PositionState.

Run: cd src && .venv-relative-python _test_tier3_live_restart_contract.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Force JSON mode so we don't hit Supabase from a test
os.environ.setdefault('USE_DB', 'false')

from ralph_engine import StrategyMonitor, PositionState


def _make_strategy() -> dict:
    return {
        'id': 999_997,
        'name': 'TEST-TIER3-LIVE-RESTART',
        'symbol': 'SPY',
        'direction': 'LONG',
        'timeframe': '1Min',
        'entry_trigger': '1Min-EMA_STACK-S',
        'entry_trigger_confluence_id': '1Min-EMA_STACK-S',
        'exit_trigger': 'opposite_signal',
        'confluence': [],
        'risk_per_trade': 100.0,
        'starting_balance': 10_000.0,
        'stop_atr_mult': 1.5,
        'stop_config': {'method': 'atr', 'atr_mult': 1.5},
        'data_seed': 42,
        'strategy_origin': 'standard',
        'trading_session': 'RTH',
        'user_id': 'test-user-id',
    }


def test_construction_with_no_saved_state_starts_flat():
    strat = _make_strategy()
    monitor = StrategyMonitor(strat, None, general_packs=[])
    assert monitor.position.state.status == 'FLAT', (
        f"Expected FLAT, got {monitor.position.state.status}"
    )
    assert monitor._tier3_inherited_carryover is None, (
        "No carryover should be recorded when position_state is None"
    )
    print("  ✓ no saved state → FLAT, no carryover")


def test_construction_with_saved_flat_state_starts_flat():
    strat = _make_strategy()
    saved = PositionState(status='FLAT', direction='LONG')
    monitor = StrategyMonitor(strat, saved, general_packs=[])
    assert monitor.position.state.status == 'FLAT'
    assert monitor._tier3_inherited_carryover is None, (
        "No carryover should be recorded when saved status is FLAT"
    )
    print("  ✓ saved FLAT state → FLAT, no carryover")


def test_construction_with_saved_in_position_starts_flat():
    """The critical Tier 3 §8.3 property."""
    strat = _make_strategy()
    saved = PositionState(
        status='IN_POSITION',
        direction='LONG',
        entry_price=425.50,
        entry_time='2026-05-19T15:30:00+00:00',
        stop_price=420.00,
        target_price=435.00,
        entry_bar_count=5,
        entry_trigger='1Min-EMA_STACK-S',
    )
    monitor = StrategyMonitor(strat, saved, general_packs=[])

    # Engine FLAT regardless of saved state
    assert monitor.position.state.status == 'FLAT', (
        f"Tier 3 contract violated: engine should be FLAT despite "
        f"saved IN_POSITION. Got: {monitor.position.state.status}"
    )

    # Carryover captured for persistence
    carry = monitor._tier3_inherited_carryover
    assert carry is not None, (
        "Carryover should be recorded when saved status was IN_POSITION"
    )

    # Carryover has the right shape
    assert carry['type'] == 'position_carryover'
    assert carry['strategy_id'] == strat['id']
    assert carry['entry_price'] == 425.50
    assert carry['direction'] == 'LONG'
    assert carry['entry_time'] == '2026-05-19T15:30:00+00:00'
    assert carry['reason'] == 'tier3_restart_flat'
    assert 'recorded_at' in carry

    print(f"  ✓ saved IN_POSITION → FLAT, carryover recorded:")
    print(f"    type={carry['type']}, sid={carry['strategy_id']}, "
          f"price=${carry['entry_price']}, dir={carry['direction']}")


def test_position_state_machine_does_not_inherit():
    """Double-check: even if we forge a sneaky way to pass IN_POSITION
    through, the engine starts FLAT because StrategyMonitor passes
    None to PositionStateMachine, not the saved state."""
    strat = _make_strategy()
    saved = PositionState(
        status='IN_POSITION', direction='LONG',
        entry_price=100.0, entry_time='2026-05-19T10:00:00+00:00',
    )
    monitor = StrategyMonitor(strat, saved, general_packs=[])
    # Direct introspection: the PositionStateMachine inside the monitor
    # should NOT have entry_price=100.0 from the saved state.
    psm_state = monitor.position.state
    assert psm_state.entry_price == 0.0, (
        f"PositionStateMachine.state.entry_price leaked from saved "
        f"state — got {psm_state.entry_price}, expected 0.0 (default)"
    )
    assert psm_state.entry_time is None, (
        f"PositionStateMachine.state.entry_time leaked — got "
        f"{psm_state.entry_time}, expected None (default)"
    )
    print("  ✓ PositionStateMachine fields all default — no leak from saved")


if __name__ == '__main__':
    print("\n=== Tier 3 §8.3 live restart contract regression ===\n")
    test_construction_with_no_saved_state_starts_flat()
    test_construction_with_saved_flat_state_starts_flat()
    test_construction_with_saved_in_position_starts_flat()
    test_position_state_machine_does_not_inherit()
    print("\n=== ALL PASS ===\n")
