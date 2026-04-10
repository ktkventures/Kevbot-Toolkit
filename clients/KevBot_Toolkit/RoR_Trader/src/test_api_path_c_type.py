"""Reproduce the API backtest path with synthetic data and C-type swing stop.

This drives services.unified_trades() with a strategy dict matching exactly
what the API builds, then inspects the resulting trades_df. Catches any bug
between the API request and the engine that the direct PSM tests miss.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def synthesize_df(n_bars=300, seed=42):
    """Build a synthetic OHLCV DataFrame indexed by timestamp."""
    rng = np.random.default_rng(seed)
    base = 100.0
    rows = []
    price = base
    start = datetime(2026, 4, 1, 9, 30)
    for i in range(n_bars):
        delta = rng.normal(0, 0.4)
        if i % 17 == 0 and i > 30:
            delta -= 1.5  # Inject occasional sharp drop
        new_price = max(50, price + delta)
        bar_open = price
        bar_close = new_price
        wick = abs(rng.normal(0, 0.25))
        bar_high = max(bar_open, bar_close) + wick
        bar_low = min(bar_open, bar_close) - wick
        rows.append({
            'open': bar_open, 'high': bar_high, 'low': bar_low,
            'close': bar_close, 'volume': 1000,
        })
        price = bar_close
    df = pd.DataFrame(rows)
    df.index = pd.date_range(start=start, periods=n_bars, freq='1min')
    return df


def main():
    print('=' * 70)
    print('API path C-type test — reproduces backtest_service behavior')
    print('=' * 70)

    df = synthesize_df()
    print(f'Synthetic df: {len(df)} bars, {df.index[0]} → {df.index[-1]}')

    # Build a strategy dict matching backtest_service.run_backtest construction.
    # We use a simple entry_trigger that the engine recognizes (not a real one
    # — we'll mark every bar as a potential entry via direct state).
    # For a fair test, use a real built-in trigger that fires often.
    strategy = {
        'symbol': 'TEST',
        'direction': 'LONG',
        'timeframe': '1Min',
        'trading_session': 'RTH',
        'entry_trigger_confluence_id': '1M-EMA_STACK_DEFAULT-LR_BULL_STACK',
        'exit_trigger_confluence_ids': [],
        'confluence': [],
        'stop_config': {
            'method': 'swing', 'lookback': 5, 'padding': 0.0,
            'exec_type': 'C',  # ← CRITICAL — this is what the backtest_service injects
        },
        'target_config': None,
        'stop_atr_mult': 1.5,
        'risk_per_trade': 100.0,
        'bar_count_exit': None,
    }

    # Call the unified engine directly (same path as svc.unified_trades).
    # We bypass prepare_data_with_indicators because we're testing the engine
    # not the data pipeline.
    from unified_engine import run_unified_backtest

    trades_df, _ = run_unified_backtest(
        df, strategy, general_packs=None,
        secondary_tf_map=None, include_open_position=False)

    print(f'\nTrades produced: {len(trades_df)}')
    if len(trades_df) == 0:
        print('No trades — entry trigger may not have fired')
        return 0

    # Filter to stop_loss exits only
    stops = trades_df[trades_df['exit_reason'] == 'stop_loss']
    print(f'Stop-loss exits: {len(stops)}')

    if len(stops) == 0:
        print('No stop-loss exits to inspect')
        return 0

    # Print exec_type info
    print(f'\nstop_exec_type values: {stops["stop_exec_type"].unique()}')
    print(f'\nR-multiple distribution:')
    print(f'  min:  {stops["r_multiple"].min():.4f}')
    print(f'  max:  {stops["r_multiple"].max():.4f}')
    print(f'  mean: {stops["r_multiple"].mean():.4f}')

    capped_at_1 = (stops['r_multiple'].round(3) == -1.000).sum()
    below_1 = (stops['r_multiple'] < -1.001).sum()
    print(f'\n  trades at exactly -1.000R: {capped_at_1}')
    print(f'  trades below -1.001R:      {below_1}')

    if below_1 == 0 and len(stops) > 5:
        print('\n*** REPRODUCED BUG: every stop-loss is exactly -1R ***')
        # Inspect first few to find why
        print('\nFirst 5 stop-out trades:')
        cols = ['entry_price', 'stop_price', 'initial_stop_price',
                'exit_price', 'r_multiple', 'stop_exec_type']
        print(stops[cols].head().to_string())
        return 1

    print(f'\nSample stop-outs (sorted by r_multiple):')
    cols = ['entry_price', 'initial_stop_price', 'exit_price',
            'r_multiple', 'stop_exec_type']
    print(stops[cols].nsmallest(8, 'r_multiple').to_string())
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
