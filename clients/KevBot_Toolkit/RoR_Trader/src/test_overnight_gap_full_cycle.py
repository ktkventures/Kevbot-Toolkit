"""End-to-end test: drive process_bar through entry → overnight gap → exit.

Reproduces what the user observed in image 53/54: an L-type stop on an
overnight hold should fill at the gap-down open, not at the stop level.
"""

import pandas as pd
from datetime import datetime
from unified_engine import UnifiedStrategy


def make_df_with_gap():
    """Build a synthetic OHLC sequence with a clear overnight gap-down.

    Layout:
      - 5 bars trending up to set up a swing stop and entry signal
      - 1 entry bar (around bar 6) — close near 100
      - GAP: next bar opens at 80 (way below entry, way below stop)
      - Continue with low prices to trigger stop-out
    """
    bars = []
    base = 95.0
    # Bars 0-4: uptrend, building swing low
    for i in range(5):
        o = base + i * 0.5
        c = o + 0.4
        h = c + 0.1
        l = o - 0.1
        bars.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1000})
    # Bar 5: entry candidate — close at 99.5 (near top)
    bars.append({'open': 97.5, 'high': 99.7, 'low': 97.4, 'close': 99.5, 'volume': 1000})
    # Bar 6: GAP DOWN — opens at 80, closes at 79 — way below the swing low (~94.9)
    bars.append({'open': 80.0, 'high': 80.5, 'low': 78.0, 'close': 79.0, 'volume': 1000})
    # Bars 7-9: continue low so position stays out
    for i in range(3):
        bars.append({'open': 79.0, 'high': 79.2, 'low': 78.5, 'close': 78.8, 'volume': 1000})

    df = pd.DataFrame(bars)
    df.index = pd.date_range(start='2026-04-09 09:30', periods=len(df), freq='1min')
    return df


def main():
    print('=' * 70)
    print('Overnight gap-down end-to-end test')
    print('=' * 70)

    df = make_df_with_gap()
    print(f'\nSynthetic bars:')
    for i, (ts, row) in enumerate(df.iterrows()):
        print(f'  bar[{i}] @ {ts}: O={row["open"]:.2f} H={row["high"]:.2f} '
              f'L={row["low"]:.2f} C={row["close"]:.2f}')

    # Build a UnifiedStrategy that we'll drive directly through process_bar.
    # Force-enter on bar 5 by directly poking the position state after process_bar.
    strategy = {
        'id': 'test',
        'symbol': 'TEST',
        'direction': 'LONG',
        'timeframe': '1Min',
        'entry_trigger': 'dummy',  # never fires
        'exit_triggers': [],
        'confluence': [],
        'stop_config': {'method': 'swing', 'lookback': 5, 'padding': 0.0,
                        'exec_type': 'L'},
        'target_config': None,
    }

    strat = UnifiedStrategy(strategy, general_packs=None)

    # Drive bars 0-4 to populate the swing buffer
    for i in range(5):
        bar = {**df.iloc[i].to_dict(), 'timestamp': df.index[i]}
        strat.process_bar(bar)

    # Manually inject the entry on bar 5 (close = 99.5, swing stop = ~94.9)
    bar5 = {**df.iloc[5].to_dict(), 'timestamp': df.index[5]}
    strat.process_bar(bar5)
    psm = strat.position
    psm.state.status = 'IN_POSITION'
    psm.state.direction = 'LONG'
    psm.state.entry_price = 99.5
    # Compute stop from buffer (mimic what _compute_stop would do for swing)
    # Swing low over the lookback window (excluding most recent bar)
    buf = list(psm._high_low_buffer)[:-1]
    swing_low = min(low for _, low in buf[-5:])
    psm.state.stop_price = swing_low
    psm.state.initial_stop_price = swing_low
    psm.state.target_price = None
    psm.state.entry_bar_count = 5
    psm.state.exec_type = 'C'  # C-type entry — same_bar_ltype check uses this
    psm.state.entry_time = df.index[5]
    print(f'\nManually entered LONG at bar 5: '
          f'entry={psm.state.entry_price:.2f}, stop={psm.state.stop_price:.2f}, '
          f'risk={psm.state.entry_price - psm.state.stop_price:.2f}')

    # Now drive bar 6 (the gap-down) and see what happens
    bar6 = {**df.iloc[6].to_dict(), 'timestamp': df.index[6]}
    print(f'\nProcessing bar 6 (GAP DOWN): O={bar6["open"]:.2f} '
          f'H={bar6["high"]:.2f} L={bar6["low"]:.2f} C={bar6["close"]:.2f}')
    print(f'  Stop price: {psm.state.stop_price:.2f}')
    print(f'  bar_open ({bar6["open"]}) < stop ({psm.state.stop_price})? '
          f'{bar6["open"] < psm.state.stop_price}')

    trades, _, _, _ = strat.process_bar(bar6)
    print(f'\n  Trades returned by process_bar: {len(trades)}')
    if trades:
        t = trades[0]
        print(f'  Exit reason:  {t["exit_reason"]}')
        print(f'  Exit price:   {t["exit_price"]:.4f}')
        print(f'  Stop price:   {t["stop_price"]:.4f}')
        print(f'  Initial stop: {t["initial_stop_price"]:.4f}')
        print(f'  Entry price:  {t["entry_price"]:.4f}')
        print(f'  R-multiple:   {t["r_multiple"]:.4f}')
        print(f'  stop_exec_type field: {t.get("stop_exec_type", "MISSING")}')

        # Verify gap-aware fill
        expected_fill = min(psm.state.stop_price if psm.state.stop_price > 0 else 99,
                            bar6['open'])
        # After the trade, state is reset, so use initial_stop_price from record
        risk = abs(t['entry_price'] - t['initial_stop_price'])
        expected_r = (t['exit_price'] - t['entry_price']) / risk

        print(f'\n  Expected fill: min(stop=??, open={bar6["open"]:.2f}) = ??')
        print(f'  Risk: {risk:.4f}')
        print(f'  Computed R from fields: {expected_r:.4f}')

        # The bug check: if exit_price is the stop level (not the open), that's wrong
        if abs(t['exit_price'] - t['initial_stop_price']) < 0.001:
            print(f'\n  *** BUG: exit_price ({t["exit_price"]:.4f}) == stop ({t["initial_stop_price"]:.4f})')
            print(f'  *** Should have been bar_open ({bar6["open"]:.4f}) for a gap-down')
            return 1
        if abs(t['exit_price'] - bar6['open']) < 0.001:
            print(f'\n  CORRECT: exit_price == bar_open (gap-aware fill working)')
            return 0
        print(f'\n  UNEXPECTED: exit_price ({t["exit_price"]:.4f}) is neither stop nor open')
        return 1
    else:
        print(f'  *** BUG: No trade generated on the gap-down bar!')
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
