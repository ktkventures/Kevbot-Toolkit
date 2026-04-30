"""Direct comparison: batch path vs incremental path for ut_bot_v4 on
AAPL/1Min/120days. If batch fires utv4_bull_flip on bars X and
incremental fires it on bars Y, with X != Y, that's a parity bug
between the pack's two paths.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
set_admin_user_context("19d47e46-f718-49a6-af32-5f5407f5b170")

from services import prepare_data_with_indicators
from data_loader import load_market_data

print("=== BATCH path: prepare_data_with_indicators ===")
df_batch = prepare_data_with_indicators(
    symbol="AAPL", days=120, timeframe="1Min", session="RTH"
)
print(f"  bars: {len(df_batch)}")
print(f"  has utv4_bull_flip column: {'utv4_bull_flip' in df_batch.columns}")
print(f"  has trig_utv4_bull_flip column: {'trig_utv4_bull_flip' in df_batch.columns}")
print(f"  utv4-related cols: {[c for c in df_batch.columns if 'utv4' in c.lower() or 'ut_bot_v4' in c.lower()][:15]}")
# Use the trig_ prefixed column if present
fire_col = 'trig_utv4_bull_flip' if 'trig_utv4_bull_flip' in df_batch.columns else 'utv4_bull_flip'
if fire_col in df_batch.columns:
    batch_fires = df_batch[fire_col].astype(bool).sum()
    print(f"  {fire_col} True count (batch): {batch_fires}")
    batch_fire_minutes = set()
    for ts in df_batch.index[df_batch[fire_col].astype(bool)]:
        batch_fire_minutes.add(str(ts)[:16])
    print(f"  unique fire minutes: {len(batch_fire_minutes)}")
else:
    batch_fire_minutes = set()
    batch_fires = 0

print("\n=== INCREMENTAL path: ut_bot_v4 update_bar() walk ===")
df_inc = load_market_data(symbol="AAPL", days=120, timeframe="1Min", session="RTH")
pack = pack_registry.get_pack("ut_bot_v4")
defaults = {k: v.get("default") for k, v in (pack.manifest.get("parameters_schema") or {}).items()}
engine = pack.incremental_class(**defaults)

inc_fire_minutes = set()
inc_fire_count = 0
for i in range(len(df_inc)):
    row = df_inc.iloc[i]
    bar = {
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
        "timestamp": df_inc.index[i],
    }
    out = engine.update_bar(bar)
    if isinstance(out, dict) and out.get("utv4_bull_flip"):
        inc_fire_count += 1
        inc_fire_minutes.add(str(df_inc.index[i])[:16])

print(f"  bars: {len(df_inc)}")
print(f"  utv4_bull_flip True count (incremental): {inc_fire_count}")
print(f"  unique fire minutes: {len(inc_fire_minutes)}")

print("\n=== OVERLAP ANALYSIS ===")
if batch_fire_minutes:
    batch_only = batch_fire_minutes - inc_fire_minutes
    inc_only = inc_fire_minutes - batch_fire_minutes
    overlap = batch_fire_minutes & inc_fire_minutes
    print(f"  batch fires:       {len(batch_fire_minutes)}")
    print(f"  incremental fires: {len(inc_fire_minutes)}")
    print(f"  overlap:           {len(overlap)}")
    print(f"  batch_only:        {len(batch_only)} (batch fires here, incremental doesn't)")
    print(f"  inc_only:          {len(inc_only)}")
    if batch_only:
        print(f"\n  batch_only sample (first 5): {sorted(batch_only)[:5]}")
    if inc_only:
        print(f"  inc_only sample (first 5):   {sorted(inc_only)[:5]}")
