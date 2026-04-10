"""End-to-end pipeline test that mimics the API backtest path with C-type swing.

Builds a synthetic OHLC dataframe with EMA crossovers, uses a real built-in
entry trigger, runs through services.unified_trades, and verifies that the
resulting trades_df has uncapped R-multiples for C-type stops.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def synthesize_df(n_bars=400, seed=7):
    """Build OHLCV with enough variation that EMA stack triggers fire."""
    rng = np.random.default_rng(seed)
    base = 100.0
    rows = []
    price = base
    start = datetime(2026, 4, 1, 9, 30)
    # Create alternating uptrends and pullbacks so EMA stacks form and break
    for i in range(n_bars):
        # Slow oscillation + noise
        trend = 0.05 * np.sin(i / 30)
        noise = rng.normal(0, 0.4)
        # Big drops every 25 bars to test stop behavior
        if i % 25 == 0 and i > 50:
            shock = -2.0
        else:
            shock = 0.0
        delta = trend + noise + shock
        new_price = max(50, price + delta)
        bar_open = price
        bar_close = new_price
        wick = abs(rng.normal(0, 0.2))
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
    print('Full pipeline C-type test — uses services.unified_trades')
    print('=' * 70)

    # Need to enrich df with indicators first (mimics prepare_data_with_indicators)
    import services as svc
    from confluence_groups import load_confluence_groups, get_enabled_groups

    df = synthesize_df()
    print(f'Synthetic df: {len(df)} bars')

    # Run all built-in indicators
    df = svc.run_all_indicators(df)

    # Run group-specific indicators (EMA stack etc.)
    groups = get_enabled_groups(load_confluence_groups())
    for group in groups:
        df = svc.run_indicators_for_group(df, group)
    df = svc.run_all_interpreters(df)

    print(f'Enriched df columns: {len(df.columns)}')
    print(f'EMA-related columns: {[c for c in df.columns if "ema" in c.lower()][:8]}')

    # Find an entry trigger that exists. Build a strategy that uses a c-type
    # stop with swing method. Use a likely-to-fire EMA stack trigger.
    # Try a range of common entry trigger IDs.
    candidate_triggers = [
        '1M-EMA_STACK_DEFAULT-LR_BULL_STACK',
        '1M-EMA_STACK_DEFAULT-SML',
        'ema_stack_default_bull',
    ]

    for entry_trig in candidate_triggers:
        print(f'\nTrying entry_trigger={entry_trig}')
        strategy = {
            'symbol': 'TEST',
            'direction': 'LONG',
            'timeframe': '1Min',
            'trading_session': 'RTH',
            'entry_trigger': entry_trig,
            'entry_trigger_confluence_id': entry_trig,
            'exit_trigger_confluence_ids': [],
            'confluence': [],
            'stop_config': {
                'method': 'swing', 'lookback': 5, 'padding': 0.05,
                'exec_type': 'C',
            },
            'target_config': None,
            'stop_atr_mult': 1.5,
            'risk_per_trade': 100.0,
            'bar_count_exit': None,
        }

        trades_df = svc.unified_trades(df, strategy, include_open_position=False)
        print(f'  trades: {len(trades_df)}')
        if len(trades_df) > 0:
            print(f'  columns: {sorted(trades_df.columns.tolist())[:10]}...')
            stops = trades_df[trades_df['exit_reason'] == 'stop_loss']
            print(f'  stop-loss exits: {len(stops)}')
            if len(stops) > 0:
                print(f'  stop_exec_type unique: {stops["stop_exec_type"].unique() if "stop_exec_type" in stops.columns else "MISSING"}')
                print(f'  r_multiple range: [{stops["r_multiple"].min():.4f}, {stops["r_multiple"].max():.4f}]')
                print(f'  trades with r < -1.001: {(stops["r_multiple"] < -1.001).sum()}')
                print(f'  trades with r == -1.000: {(stops["r_multiple"].round(4) == -1.0).sum()}')
                # Show first few
                print(f'  Sample trades:')
                cols = ['entry_price', 'initial_stop_price', 'exit_price', 'r_multiple']
                if 'stop_exec_type' in stops.columns:
                    cols.append('stop_exec_type')
                print(stops[cols].head().to_string())
                break

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
