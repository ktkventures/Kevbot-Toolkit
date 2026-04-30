"""Probe Bug A — AAPL user-pack trigger black hole.

Loads AAPL/1Min bars via the same `load_market_data` path parity_service
uses, then directly instantiates `ut_bot_v4`'s incremental class and
walks bars one at a time. Counts how many `utv4_bull_flip=True` events
the incremental class produces on this exact data.

Outcomes:
- 0 flips → bar data itself is wrong (load_market_data returns bad
  bars for AAPL, RTH filter dropping everything, etc.)
- Many flips → integration bug between StrategyMonitor and the user
  pack engine; data + pack are fine, but they're not being wired
  together correctly during replay.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timezone

import pack_registry
pack_registry.scan_and_load_all()

from data_loader import load_market_data

# Match what parity_service does for sid 145
SYMBOL = "AAPL"
TIMEFRAME = "1Min"
SESSION = "RTH"
DAYS = 120

print(f"\n=== Loading {SYMBOL}/{TIMEFRAME} via load_market_data ({DAYS} days, {SESSION}) ===")
df = load_market_data(symbol=SYMBOL, days=DAYS, timeframe=TIMEFRAME, session=SESSION)
print(f"  bars loaded: {len(df) if df is not None else None}")
if df is not None and len(df) > 0:
    print(f"  index dtype: {df.index.dtype}")
    print(f"  range:       {df.index[0]}  →  {df.index[-1]}")
    print(f"  columns:     {list(df.columns)}")
    print(f"  sample tail (5 bars):")
    print(df.tail(5))

if df is None or len(df) == 0:
    print("\nNO DATA LOADED — Bug A is a data-loader issue.")
    raise SystemExit(1)

# Instantiate the user-pack incremental class directly. Use the exact
# default params from manifest — sid 145's strategy uses the default
# group, so this matches.
print(f"\n=== Instantiating ut_bot_v4 incremental class with default params ===")
pack = pack_registry.get_pack("ut_bot_v4")
if pack is None:
    print("ERROR: ut_bot_v4 not registered in pack_registry")
    raise SystemExit(2)

defaults = {}
for k, spec in (pack.manifest.get("parameters_schema") or {}).items():
    defaults[k] = spec.get("default")
print(f"  defaults: {defaults}")

engine = pack.incremental_class(**defaults)

# Walk bars one at a time, simulating the live engine's per-bar update.
print(f"\n=== Walking {len(df)} bars via engine.update_bar() ===")
bull_flip_count = 0
bear_flip_count = 0
bull_flip_examples: list[str] = []
bear_flip_examples: list[str] = []
none_state_count = 0
non_dict_count = 0

for i in range(len(df)):
    row = df.iloc[i]
    bar = {
        "open":   float(row["open"]),
        "high":   float(row["high"]),
        "low":    float(row["low"]),
        "close":  float(row["close"]),
        "volume": float(row.get("volume", 0)),
        "timestamp": df.index[i],
    }
    out = engine.update_bar(bar)
    if not isinstance(out, dict):
        non_dict_count += 1
        continue
    if out.get("utv4_bull_flip"):
        bull_flip_count += 1
        if len(bull_flip_examples) < 5:
            bull_flip_examples.append(str(df.index[i]))
    if out.get("utv4_bear_flip"):
        bear_flip_count += 1
        if len(bear_flip_examples) < 5:
            bear_flip_examples.append(str(df.index[i]))

print(f"\n  utv4_bull_flip: {bull_flip_count} events")
print(f"  utv4_bear_flip: {bear_flip_count} events")
if bull_flip_examples:
    print(f"  bull_flip example timestamps: {bull_flip_examples}")
if bear_flip_examples:
    print(f"  bear_flip example timestamps: {bear_flip_examples}")
print(f"  bars where update_bar returned non-dict: {non_dict_count}")

# 4. Check whether stored_trades' entry timestamps land on bars in this df.
# If not, the data window doesn't cover the trades — explains zero replay
# fires (the parity replay would have walked OTHER bars).
print(f"\n=== Cross-check vs sid 145 stored_trades ===")
from db import get_admin_client, set_admin_user_context
import json
set_admin_user_context("19d47e46-f718-49a6-af32-5f5407f5b170")
c = get_admin_client()
r = c.table("strategies").select("stored_trades").eq("id", 145).single().execute()
st = r.data.get("stored_trades") or []
if isinstance(st, str):
    st = json.loads(st)

stored_entry_ts = sorted([t.get("entry_fill_ts") for t in st if t.get("entry_fill_ts")])
print(f"  stored_trades entries: {len(stored_entry_ts)}")
if stored_entry_ts:
    print(f"  first entry: {stored_entry_ts[0]}")
    print(f"  last entry:  {stored_entry_ts[-1]}")

# Check overlap with df range
df_first = df.index[0]
df_last  = df.index[-1]

def _to_dt(s):
    return datetime.fromisoformat(str(s).replace('Z', '+00:00'))

inside = sum(1 for s in stored_entry_ts if df_first <= _to_dt(s) <= df_last)
print(f"  stored entries inside loaded df window: {inside}/{len(stored_entry_ts)}")

# Sample: do the bars at stored entry timestamps actually exist in df?
hits = 0
misses = 0
for s in stored_entry_ts[:10]:
    target = _to_dt(s)
    if target in df.index:
        hits += 1
    else:
        misses += 1
print(f"  exact-index hits on first 10 stored entries: {hits} hits / {misses} misses")
