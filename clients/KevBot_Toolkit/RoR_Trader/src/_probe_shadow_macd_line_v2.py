"""Probe — instantiate the AAPL 1Day shadow engine for sid 145, walk
real daily bars through it, and dump what `current` and `interps`
contain after each step. Tells us whether:

  (a) the macd_line_v2 incremental class is producing the indicator
      columns (mlv2_line, mlv2_signal, mlv2_hist),
  (b) the interpreter dispatch in TriggerEvaluator.evaluate_bar_close
      is returning a state for MACD_LINE_V2,
  (c) the resulting confluence record is being emitted with the right
      label.

If (a) is missing → user-pack engine isn't running in shadow.
If (a) present but (b) is None → interpreter dispatch fails (likely
the missing-indicator-column case, or NaN propagation).
If (b) present → confluence record assembly is broken.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import get_strategy_by_id_db, set_admin_user_context
from data_loader import load_market_data
from ralph_engine import StrategyMonitor, SymbolHub, _ShadowIndicatorEngine

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SID = 145

set_admin_user_context(USER)
strat = get_strategy_by_id_db(SID)

# Build hub/monitor the way parity_service does
hub = SymbolHub(strat["symbol"], publisher=None)
monitor = StrategyMonitor(strat, None, general_packs=[])
hub.add_monitor(monitor)
hub.finalize_shadow_engines()

print(f"\nShadow engines on hub: {list(hub._shadow_engines.keys())}")

# Pull the 1Day shadow
shadow_1d = hub._shadow_engines.get(86400)
if shadow_1d is None:
    print("ERROR: no 1Day shadow engine")
    raise SystemExit(1)

print(f"\n1Day shadow:")
print(f"  required indicators: {sorted(shadow_1d.indicators.required)}")
print(f"  required interpreters: {sorted(shadow_1d.trigger_eval.required_interpreters)}")
print(f"  user_pack_engines: {list(shadow_1d.indicators._user_pack_engines.keys())}")

# Load 1Day AAPL bars
df_1d = load_market_data(symbol="AAPL", days=120, timeframe="1Day", session="RTH")
print(f"\n1Day AAPL bars: {len(df_1d)}")
if len(df_1d) > 0:
    print(f"  range: {df_1d.index[0]} → {df_1d.index[-1]}")

# Warmup with first 25 bars (mirrors parity warmup logic for shadows)
warmup = min(50, max(5, len(df_1d) // 4))
shadow_1d.warmup(df_1d.iloc[:warmup])
print(f"\nAfter warmup ({warmup} bars):")
print(f"  current keys: {sorted(shadow_1d.indicators.state.current.keys())}")
print(f"  current['mlv2_line']: {shadow_1d.indicators.state.current.get('mlv2_line')}")
print(f"  current['mlv2_signal']: {shadow_1d.indicators.state.current.get('mlv2_signal')}")
print(f"  current['mlv2_hist']: {shadow_1d.indicators.state.current.get('mlv2_hist')}")

# Walk a few bars and dump records
print(f"\nWalking bars {warmup}..{warmup+10}:")
for i in range(warmup, min(warmup + 10, len(df_1d))):
    row = df_1d.iloc[i]
    bar = {
        "open": float(row["open"]), "high": float(row["high"]),
        "low":  float(row["low"]), "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
        "timestamp": df_1d.index[i],
    }
    records = shadow_1d.on_bar_close(bar)
    cur = shadow_1d.indicators.state.current
    mlv2_keys = {k: cur.get(k) for k in ['mlv2_line', 'mlv2_signal', 'mlv2_hist']}
    print(f"  i={i} ts={df_1d.index[i]} mlv2={mlv2_keys} records={records}")

# Now run an additional 30 bars and tally records produced
record_states = []
for i in range(warmup + 10, len(df_1d)):
    row = df_1d.iloc[i]
    bar = {
        "open": float(row["open"]), "high": float(row["high"]),
        "low":  float(row["low"]), "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
        "timestamp": df_1d.index[i],
    }
    records = shadow_1d.on_bar_close(bar)
    record_states.append(set(records))

from collections import Counter
all_record_set = set()
for r in record_states:
    all_record_set |= r
print(f"\nDistinct records produced across {len(record_states)} bars: {sorted(all_record_set)}")

# How often is each record present?
record_counter = Counter()
for r in record_states:
    for rec in r:
        record_counter[rec] += 1
print(f"Record frequencies:")
for rec, n in record_counter.most_common():
    print(f"  {rec}: {n}/{len(record_states)}")
