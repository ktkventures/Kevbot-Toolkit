"""Probe Bug A continuation — same data, but now wrap it in
StrategyMonitor (the way parity_service does) and verify whether
`_user_pack_engines` is populated and whether `utv4_bull_flip` shows
up in `state.current` after `_update_indicators(bar)`.

If `_user_pack_engines` is empty or doesn't contain ut_bot_v4 → the
pack registry isn't loading the pack into the engine for AAPL.
If it IS loaded but state.current['utv4_bull_flip'] is missing/False
on the bars where direct-pack-walk produced True → the engine isn't
calling pack.update_bar() in the inner loop.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import get_strategy_by_id_db, set_admin_user_context
from data_loader import load_market_data
from ralph_engine import StrategyMonitor, SymbolHub

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SID = 145

set_admin_user_context(USER)
strat = get_strategy_by_id_db(SID)  # IMPORTANT: returns FLATTENED dict (unpacks config JSONB)
print(f"\nStrategy: sid {SID}: {strat['name']} ({strat['symbol']}/{strat['timeframe']})")
print(f"  entry_trigger_confluence_id: {strat.get('entry_trigger_confluence_id')}")
print(f"  exit_trigger_confluence_ids: {strat.get('exit_trigger_confluence_ids')}")
print(f"  confluence: {strat.get('confluence')}")

# Build hub + monitor the way parity_service does
hub = SymbolHub(strat["symbol"], publisher=None)
monitor = StrategyMonitor(strat, None, general_packs=[])
hub.add_monitor(monitor)
hub.finalize_shadow_engines()

print(f"\n--- Inspecting monitor.indicators._user_pack_engines ---")
ind_engine = monitor.indicators
upe = getattr(ind_engine, "_user_pack_engines", None)
print(f"  _user_pack_engines: {list(upe.keys()) if upe is not None else 'ATTR NOT SET'}")
print(f"  required (interpreters): {sorted(ind_engine.required)[:15]}")
print(f"  required_triggers (trigger_eval): {sorted(monitor.trigger_eval.required_triggers)[:15]}")

# Load AAPL bars (same path)
df = load_market_data(symbol=strat["symbol"], days=120,
                      timeframe=strat["timeframe"], session="RTH")
print(f"\n  bars loaded: {len(df)}")

# Warmup with first 200 bars (same as parity_service)
print(f"\n--- Running warmup with first 200 bars ---")
monitor.warmup(df.iloc[:200])
print(f"  _user_pack_engines after warmup: {list(getattr(ind_engine, '_user_pack_engines', {}).keys())}")
print(f"  state.current keys (sample): {sorted(list(ind_engine.state.current.keys()))[:20]}")
print(f"  utv4_bull_flip in state.current: {ind_engine.state.current.get('utv4_bull_flip', 'NOT PRESENT')}")

# Now walk a chunk of bars after warmup and count fires
print(f"\n--- Walking 1000 bars after warmup, counting fires ---")
fires_bull = 0
fires_bear = 0
state_current_bull_flip_seen = 0
for i in range(200, min(1200, len(df))):
    row = df.iloc[i]
    bar = {
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
        "timestamp": df.index[i],
    }
    signals, audit = monitor.on_bar_close(bar, i, mtf_confluence={})

    # Check what the trigger_booleans audit contains
    t_b = audit.get("trigger_booleans") or {}
    if t_b.get("utv4_bull_flip"):
        fires_bull += 1
    if t_b.get("utv4_bear_flip"):
        fires_bear += 1

    # Also: did state.current contain utv4_bull_flip=True at all?
    if ind_engine.state.current.get("utv4_bull_flip"):
        state_current_bull_flip_seen += 1

print(f"  audit.trigger_booleans utv4_bull_flip True count: {fires_bull}")
print(f"  audit.trigger_booleans utv4_bear_flip True count: {fires_bear}")
print(f"  state.current[utv4_bull_flip] True count: {state_current_bull_flip_seen}")

# Compare to direct-pack-walk on same chunk
print(f"\n--- Comparison: direct ut_bot_v4 walk on bars 0-1200 (incl warmup) ---")
pack = pack_registry.get_pack("ut_bot_v4")
defaults = {k: v.get("default") for k, v in (pack.manifest.get("parameters_schema") or {}).items()}
direct_engine = pack.incremental_class(**defaults)
direct_bulls_in_range = 0
for i in range(0, min(1200, len(df))):
    row = df.iloc[i]
    bar = {
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
        "timestamp": df.index[i],
    }
    out = direct_engine.update_bar(bar)
    if isinstance(out, dict) and out.get("utv4_bull_flip"):
        direct_bulls_in_range += 1
print(f"  direct walk bull_flip count (bars 0-1200): {direct_bulls_in_range}")
