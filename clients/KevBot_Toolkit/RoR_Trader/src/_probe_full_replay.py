"""Probe — replicate parity_service._replay_strategy LOCALLY for sid 145
and count audit.trigger_booleans utv4_bull_flip True events across ALL
replayed bars. If this produces ~2088 fires (matching direct walk), the
bug is in parity_service's diff phase — NOT in the engine. If it
produces 0, there's something specific to the parity loop that breaks
the engine.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import time
from collections import Counter

import pack_registry
pack_registry.scan_and_load_all()

from db import get_strategy_by_id_db, set_admin_user_context
from data_loader import load_market_data
from ralph_engine import StrategyMonitor, SymbolHub, SECONDS_TO_TIMEFRAME

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SID = 145
DAYS = 120

set_admin_user_context(USER)
strat = get_strategy_by_id_db(SID)

# Build hub + monitor — exactly like parity_service does
hub = SymbolHub(strat["symbol"], publisher=None)
monitor = StrategyMonitor(strat, None, general_packs=[])
hub.add_monitor(monitor)
hub.finalize_shadow_engines()

# Load primary + secondary bars
primary_df = load_market_data(symbol=strat["symbol"], days=DAYS,
                              timeframe=strat["timeframe"], session="RTH")
print(f"primary bars: {len(primary_df)}")

sec_dfs = {}
for sec_tf in monitor._required_secondary_tf:
    label = SECONDS_TO_TIMEFRAME.get(sec_tf)
    sec_dfs[sec_tf] = load_market_data(
        symbol=strat["symbol"], days=DAYS, timeframe=label, session="RTH")
    print(f"  secondary {sec_tf}s ({label}): {len(sec_dfs[sec_tf])} bars")

# Warmup primary
warmup = 200
monitor.warmup(primary_df.iloc[:warmup])

# Warmup shadows
for sec_tf, sec_df in sec_dfs.items():
    shadow = hub._shadow_engines.get(sec_tf)
    if shadow is None or sec_df is None or len(sec_df) < 5:
        continue
    sec_warmup = min(50, max(5, len(sec_df) // 4))
    try:
        shadow.warmup(sec_df.iloc[:sec_warmup])
    except Exception as e:
        print(f"shadow warmup failed: {e}")

# Walk ALL primary bars after warmup, advancing secondary TFs in lockstep
sec_state = {}
for sec_tf, sec_df in sec_dfs.items():
    if sec_df is None or len(sec_df) < 5:
        continue
    sec_warmup = min(50, max(5, len(sec_df) // 4))
    sec_state[sec_tf] = {
        'idx': sec_warmup,
        'len': len(sec_df),
        'sec_df': sec_df,
    }

print(f"\nWalking bars {warmup}..{len(primary_df)} (= {len(primary_df) - warmup} bars)")
fire_counter = Counter()
bull_flip_bar_minutes = set()  # set of YYYY-MM-DDTHH:MM strings where bull_flip fired
bar_count = warmup
t0 = time.time()
for i in range(warmup, len(primary_df)):
    ts = primary_df.index[i]

    for sec_tf, st in sec_state.items():
        shadow = hub._shadow_engines.get(sec_tf)
        if shadow is None:
            continue
        s_idx = st['idx']
        s_len = st['len']
        sec_df = st['sec_df']
        while s_idx < s_len:
            sec_ts = sec_df.index[s_idx]
            if sec_ts > ts:
                break
            sec_bar = {
                'open': float(sec_df['open'].iloc[s_idx]),
                'high': float(sec_df['high'].iloc[s_idx]),
                'low':  float(sec_df['low'].iloc[s_idx]),
                'close': float(sec_df['close'].iloc[s_idx]),
                'volume': float(sec_df.get('volume', 0).iloc[s_idx]) if 'volume' in sec_df.columns else 0,
                'timestamp': sec_ts,
            }
            try:
                records = shadow.on_bar_close(sec_bar)
                hub._mtf_confluence[sec_tf] = records
            except Exception as e:
                print(f"shadow bar_close fail {sec_tf}@{sec_ts}: {e}")
            s_idx += 1
        st['idx'] = s_idx

    bar = {
        'open': float(primary_df['open'].iloc[i]),
        'high': float(primary_df['high'].iloc[i]),
        'low':  float(primary_df['low'].iloc[i]),
        'close': float(primary_df['close'].iloc[i]),
        'volume': float(primary_df.get('volume', 0).iloc[i]) if 'volume' in primary_df.columns else 0,
        'timestamp': ts,
    }
    try:
        signals, audit = monitor.on_bar_close(
            bar, bar_count, mtf_confluence=dict(hub._mtf_confluence))
    except Exception as e:
        print(f"monitor.on_bar_close fail @ {ts}: {e}")
        bar_count += 1
        continue

    for k, v in (audit.get('trigger_booleans') or {}).items():
        if v:
            fire_counter[k] += 1
            if k == 'utv4_bull_flip':
                bull_flip_bar_minutes.add(str(ts)[:16])

    bar_count += 1

elapsed = time.time() - t0
print(f"\nElapsed: {elapsed:.1f}s")
print(f"Fire counts (audit.trigger_booleans True across {len(primary_df)-warmup} bars):")
for k, v in fire_counter.most_common():
    print(f"  {k}: {v}")

# Cross-check stored_trades entry timestamps against bull_flip fire bars
from db import get_admin_client
import json
c = get_admin_client()
r = c.table("strategies").select("stored_trades").eq("id", SID).single().execute()
st = r.data.get("stored_trades") or []
if isinstance(st, str):
    st = json.loads(st)

stored_minutes = []
for t in st:
    ts_field = t.get("entry_fill_ts") or t.get("entry_time")
    if ts_field:
        # Normalize: convert to UTC ISO and truncate to minute
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(str(ts_field).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            normalized = dt.astimezone(timezone.utc).isoformat()[:16]
            stored_minutes.append(normalized)
        except Exception:
            pass

print(f"\nStored-trade entry minutes: {len(stored_minutes)}")
print(f"Bull-flip fire bar minutes (from probe replay): {len(bull_flip_bar_minutes)}")
overlap = set(stored_minutes) & bull_flip_bar_minutes
print(f"Overlap (stored entry IS in fire-bar set): {len(overlap)}")
if overlap:
    print(f"  example overlaps: {sorted(overlap)[:5]}")
not_in_fire = set(stored_minutes) - bull_flip_bar_minutes
print(f"Stored entries that are NOT in fire bars: {len(not_in_fire)}")
if not_in_fire:
    print(f"  example missing: {sorted(not_in_fire)[:5]}")

# Check 1-bar offsets — does live fire on stored_minute + 1 minute?
from datetime import datetime, timedelta
def _shift_minute(s: str, delta: int) -> str:
    dt = datetime.fromisoformat(s + ":00+00:00")
    dt = dt + timedelta(minutes=delta)
    return dt.isoformat()[:16]

shifted_back_match = sum(1 for s in stored_minutes if _shift_minute(s, -1) in bull_flip_bar_minutes)
shifted_fwd_match  = sum(1 for s in stored_minutes if _shift_minute(s, 1) in bull_flip_bar_minutes)
print(f"\nOffset analysis (do live fires land 1 bar before/after stored entries?):")
print(f"  stored - 1min in live fires: {shifted_back_match}")
print(f"  stored + 1min in live fires: {shifted_fwd_match}")

# Sample first 5 stored minutes — show what's in the +/-3 minute window of fires
print(f"\nNeighborhood scan for first 5 stored entries:")
for s in sorted(stored_minutes)[:5]:
    nearby = []
    for delta in range(-3, 4):
        candidate = _shift_minute(s, delta)
        if candidate in bull_flip_bar_minutes:
            nearby.append(f"{delta:+d}m")
    print(f"  stored: {s}  →  fires found at offsets: {nearby or '(none in -3..+3)'}")
