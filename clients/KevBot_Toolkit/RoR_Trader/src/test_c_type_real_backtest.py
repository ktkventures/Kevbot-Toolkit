"""Run a synthetic backtest with C-type swing stop and verify R-multiples
distribution. This catches any path-level cap that the unit tests might miss.

Generates 200 synthetic 1Min bars with controlled noise so that some bars
will close significantly past a swing low. Then runs a synthetic LONG strategy
through the unified engine and checks that:
  1. Some r_multiples are < -1 (proves no cap)
  2. exit_price for stop_loss exits == bar's close (not stop_price)
"""

import pandas as pd
import numpy as np
from collections import deque
from unified_engine import PositionStateMachine


def synthesize_bars(n_bars=200, seed=42):
    """Build synthetic OHLC bars with realistic wicks and gaps."""
    rng = np.random.default_rng(seed)
    base = 100.0
    bars = []
    price = base
    for i in range(n_bars):
        # Step price by random walk
        delta = rng.normal(0, 0.3)
        # Inject occasional gap-down/recovery (creates 1.5R+ losses for C-type)
        if i % 23 == 0 and i > 50:
            delta -= 1.5
        new_price = max(50, price + delta)
        bar_open = price
        bar_close = new_price
        # Wick range around the body
        wick_size = abs(rng.normal(0, 0.2))
        bar_high = max(bar_open, bar_close) + wick_size
        bar_low = min(bar_open, bar_close) - wick_size
        bars.append({
            'open': bar_open,
            'high': bar_high,
            'low': bar_low,
            'close': bar_close,
            'volume': 1000,
            'timestamp': f'2026-04-09T{9 + i // 60:02d}:{30 + i % 60:02d}:00',
        })
        price = bar_close
    return bars


def run_synthetic_strategy(stop_exec='C'):
    """Drive a LONG position through synthetic bars with C-type swing stop.
    Records every stop_loss exit and returns the trade list."""
    bars = synthesize_bars(n_bars=200)

    strategy = {
        'id': 'synth',
        'direction': 'LONG',
        'entry_trigger': 'dummy',
        'exit_triggers': [],
        'confluence': [],
        'stop_config': {'method': 'swing', 'lookback': 5, 'padding': 0.0,
                        'exec_type': stop_exec},
        'target_config': None,
    }

    trades = []
    psm = PositionStateMachine(strategy)
    in_position = False

    for i, bar in enumerate(bars):
        # Update high/low buffer for swing lookback
        psm.update_high_low(bar['high'], bar['low'])

        if not in_position:
            # Enter every 10 bars at the bar's close
            if i > 5 and i % 10 == 0:
                # Compute swing stop from buffer
                buf = list(psm._high_low_buffer)[:-1]
                window = buf[-5:] if len(buf) >= 5 else buf
                if not window:
                    continue
                stop_price = min(low for _, low in window)
                if stop_price >= bar['close']:
                    continue  # Invalid stop
                psm.state.status = 'IN_POSITION'
                psm.state.direction = 'LONG'
                psm.state.entry_price = bar['close']
                psm.state.stop_price = stop_price
                psm.state.initial_stop_price = stop_price
                psm.state.target_price = None
                psm.state.entry_bar_count = i
                psm.state.exec_type = 'C'
                psm.state.entry_time = bar['timestamp']
                in_position = True
                continue

        if in_position:
            current = {
                'open': bar['open'], 'high': bar['high'],
                'low': bar['low'], 'close': bar['close'],
                'volume': bar['volume'],
            }
            record = psm.check_exit(
                trigger_booleans={}, current_values=current,
                bar_count=i, bar_time=bar['timestamp'])
            if record is not None:
                # Attach the bar's close so we can verify the fill
                record['_bar_close'] = bar['close']
                record['_bar_low'] = bar['low']
                record['_bar_high'] = bar['high']
                trades.append(record)
                in_position = False

    return trades


def main():
    print('=' * 70)
    print('Synthetic C-type swing stop backtest')
    print('=' * 70)

    c_trades = run_synthetic_strategy(stop_exec='C')
    l_trades = run_synthetic_strategy(stop_exec='L')

    print(f'\nC-type stop fired {len(c_trades)} stop-outs')
    print(f'L-type stop fired {len(l_trades)} stop-outs')

    if not c_trades:
        print('NO C-type trades — synthetic data did not produce any close-past events')
        return 1

    c_rs = [t['r_multiple'] for t in c_trades]
    l_rs = [t['r_multiple'] for t in l_trades]

    print(f'\nC-type R-multiples: min={min(c_rs):.3f}, max={max(c_rs):.3f}, '
          f'mean={sum(c_rs)/len(c_rs):.3f}')
    print(f'L-type R-multiples: min={min(l_rs):.3f}, max={max(l_rs):.3f}, '
          f'mean={sum(l_rs)/len(l_rs):.3f}')

    # Critical check: C-type should have at least SOME losses below -1R
    below_neg_1 = [t for t in c_trades if t['r_multiple'] < -1.001]
    print(f'\nC-type trades with r_multiple < -1.0: {len(below_neg_1)} of {len(c_trades)}')
    if not below_neg_1:
        print('  *** BUG ***: C-type stops are CAPPED at -1R. Engine has a problem.')
        return 1
    print('  Sample <-1R trades:')
    for t in below_neg_1[:5]:
        print(f"    entry={t['entry_price']:.2f} stop={t['initial_stop_price']:.2f} "
              f"close={t['_bar_close']:.2f} fill={t['exit_price']:.2f} "
              f"r={t['r_multiple']:.3f}")

    # Verify exit_price matches bar close for every stop_loss
    print('\nVerifying exit_price == bar close for C-type stop fills:')
    mismatches = 0
    for t in c_trades:
        if t['exit_reason'] == 'stop_loss':
            if abs(t['exit_price'] - t['_bar_close']) > 0.001:
                mismatches += 1
                print(f"  MISMATCH: exit_price={t['exit_price']:.2f}, "
                      f"bar_close={t['_bar_close']:.2f}, "
                      f"stop={t['initial_stop_price']:.2f}")
    if mismatches == 0:
        print(f'  All {len(c_trades)} C-type fills equal bar close — correct.')
    else:
        print(f'  *** BUG ***: {mismatches} fills do not match bar close')
        return 1

    print('\n' + '=' * 70)
    print('PASSED — C-type stops produce uncapped losses with close-based fills')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
