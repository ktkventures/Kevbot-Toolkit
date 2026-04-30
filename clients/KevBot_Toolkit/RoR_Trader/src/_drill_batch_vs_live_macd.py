"""Drill batch vs live MACD values for sid 136's 15:25 UTC entry.

Strategy gates on 1M-MACD_LINE_V2-M>S- (1Min MACD line above signal,
below zero). Live fired the entry → live's view said gate satisfied.
Heatmap (batch view) shows red → batch's view says gate not satisfied.

Approach:
  1. Pull batch's mlv2_line / mlv2_signal / mlv2_hist values at 15:25
     UTC and surrounding bars from prepare_data_with_indicators.
  2. Try to find live's view of same bars — first check if there's a
     pickle file (live_data_*.pkl), then fall back to instrumenting
     the live engine path.
  3. Compare values. If they differ, that's the root cause.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import pandas as pd
from pathlib import Path

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
set_admin_user_context("19d47e46-f718-49a6-af32-5f5407f5b170")

# ---------------------------------------------------------------------------
# Step 1: batch view
# ---------------------------------------------------------------------------
print("=" * 80)
print("BATCH VIEW (prepare_data_with_indicators)")
print("=" * 80)

from services import prepare_data_with_indicators

df_batch = prepare_data_with_indicators(
    symbol='SPY', days=1, timeframe='1Min', session='RTH',
    secondary_tfs=('15Min',),
)

target_window_start = pd.Timestamp('2026-04-30 15:23:00', tz='UTC')
target_window_end = pd.Timestamp('2026-04-30 15:27:00', tz='UTC')
mask = (df_batch.index >= target_window_start) & (df_batch.index <= target_window_end)
slice_batch = df_batch.loc[mask]

print(f"\nBatch 1Min bars around 15:25 UTC entry:")
print(f"  cols of interest: mlv2_line, mlv2_signal, mlv2_hist, MACD_LINE_V2 (state)")
print()
for ts, row in slice_batch.iterrows():
    print(f"  {ts}")
    print(f"    mlv2_line:    {row.get('mlv2_line', 'MISSING'):>10.6f}" if isinstance(row.get('mlv2_line'), float) else f"    mlv2_line:    {row.get('mlv2_line', 'MISSING')}")
    print(f"    mlv2_signal:  {row.get('mlv2_signal', 'MISSING'):>10.6f}" if isinstance(row.get('mlv2_signal'), float) else f"    mlv2_signal:  {row.get('mlv2_signal', 'MISSING')}")
    print(f"    mlv2_hist:    {row.get('mlv2_hist', 'MISSING'):>10.6f}" if isinstance(row.get('mlv2_hist'), float) else f"    mlv2_hist:    {row.get('mlv2_hist', 'MISSING')}")
    print(f"    MACD_LINE_V2 state: {row.get('MACD_LINE_V2', 'MISSING')}")
    # Trigger booleans
    print(f"    trig_mlv2_cross_bull: {row.get('trig_mlv2_cross_bull', 'MISSING')}")

# ---------------------------------------------------------------------------
# Step 2: live view via pickle
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("LIVE VIEW (pickle files written by Ralph engine)")
print("=" * 80)

# Find live pickles. Worker writes them per memory's notes.
candidates = [
    Path('/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/live_data_SPY_60.pkl'),
    Path('/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/src/live_data_SPY_60.pkl'),
    Path.home() / 'projects' / 'Kevbot-Toolkit' / 'live_data_SPY_60.pkl',
]

live_pickle = None
for p in candidates:
    if p.exists():
        live_pickle = p
        print(f"  Found: {p}  (size {p.stat().st_size:,} bytes)")
        break
if live_pickle is None:
    print("  No live_data_SPY_60.pkl found in expected paths.")
    print("  (This is expected on local dev — pickles are on Railway worker.)")
    print()
    print("Without the live pickle, we can't directly read live's mlv2_line")
    print("at 15:25 UTC. To verify live vs batch divergence, options:")
    print("  (a) Pull pickle from Railway")
    print("  (b) Reproduce the live engine path locally on the same data")
    print("  (c) Check the alerts table — alert.data may contain indicator")
    print("      snapshot at trigger time (verify by querying)")
else:
    import pickle
    with open(live_pickle, 'rb') as f:
        live_df = pickle.load(f)
    print(f"  loaded: type={type(live_df).__name__}")
    if isinstance(live_df, pd.DataFrame):
        print(f"  shape: {live_df.shape}, cols: {list(live_df.columns)[:10]}")
        if pd.Timestamp('2026-04-30 15:25:00', tz='UTC') in live_df.index:
            row = live_df.loc[pd.Timestamp('2026-04-30 15:25:00', tz='UTC')]
            print(f"  mlv2_line at 15:25:   {row.get('mlv2_line')}")
            print(f"  mlv2_signal at 15:25: {row.get('mlv2_signal')}")
            print(f"  mlv2_hist at 15:25:   {row.get('mlv2_hist')}")
