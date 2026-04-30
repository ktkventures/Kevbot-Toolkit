"""Replay live engine path locally to get live's incremental MACD values
at sid 136's 15:25 UTC entry. Compare to batch.

The live engine uses:
  IncrementalIndicatorEngine (unified_engine.py)
  → user_pack engines (e.g., MacdLineV2Incremental)
  → state.current updated bar-by-bar

If we instantiate the same class and feed today's 1Min bars sequentially,
we get the same values the live worker would produce — modulo BarBuilder
boundary effects (which the worker has but our 1Min REST data doesn't
capture).
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
set_admin_user_context("19d47e46-f718-49a6-af32-5f5407f5b170")

from data_loader import load_market_data

# Load fresh SPY 1Min bars for today (and a bit of yesterday for warmup)
df = load_market_data(symbol='SPY', days=2, timeframe='1Min', session='RTH')
print(f"Loaded {len(df)} 1Min bars: {df.index[0]} → {df.index[-1]}")

# Instantiate the macd_line_v2 incremental class with default params
pack = pack_registry.get_pack('macd_line_v2')
defaults = {k: v.get('default') for k, v in (pack.manifest.get('parameters_schema') or {}).items()}
print(f"\nmacd_line_v2 default params: {defaults}")

engine = pack.incremental_class(**defaults)

# Walk every bar through the engine, capturing values at the bars of interest
target_window_start = pd.Timestamp('2026-04-30 15:23:00', tz='UTC')
target_window_end   = pd.Timestamp('2026-04-30 15:27:00', tz='UTC')

print(f"\nLIVE incremental walk — values at 15:23–15:27 UTC:")
print(f"  bar              mlv2_line    mlv2_signal  mlv2_hist  cross_bull")
print(f"  " + "-" * 80)

for i in range(len(df)):
    ts = df.index[i]
    row = df.iloc[i]
    bar = {
        'open': float(row['open']), 'high': float(row['high']),
        'low': float(row['low']), 'close': float(row['close']),
        'volume': float(row.get('volume', 0)),
        'timestamp': ts,
    }
    out = engine.update_bar(bar)
    if target_window_start <= ts <= target_window_end:
        line = out.get('mlv2_line', 0)
        sig = out.get('mlv2_signal', 0)
        hist = out.get('mlv2_hist', 0)
        cross = out.get('mlv2_cross_bull', False)
        print(f"  {ts}  {line:>10.6f}  {sig:>10.6f}  {hist:>10.6f}  {cross}")

# Compare with batch at the same timestamps
print(f"\n\nBATCH (prepare_data_with_indicators) at the same bars:")
from services import prepare_data_with_indicators
df_batch = prepare_data_with_indicators(symbol='SPY', days=2, timeframe='1Min', session='RTH', secondary_tfs=())
mask = (df_batch.index >= target_window_start) & (df_batch.index <= target_window_end)
print(f"  bar              mlv2_line    mlv2_signal  mlv2_hist  trig_cross_bull")
print(f"  " + "-" * 80)
for ts, brow in df_batch.loc[mask].iterrows():
    print(f"  {ts}  {brow.get('mlv2_line', 0):>10.6f}  {brow.get('mlv2_signal', 0):>10.6f}  "
          f"{brow.get('mlv2_hist', 0):>10.6f}  {brow.get('trig_mlv2_cross_bull', False)}")

print(f"\n\nIf incremental and batch values match exactly → divergence is")
print(f"elsewhere (BarBuilder boundary, live worker state, etc.)")
print(f"If they differ → state-warmup or computation drift.")
