"""Drill the sid 141 RED item — 1D VWAP_V2 batch produces `>+2σ` on 78
bars where shadow produces `<-2σ`. Both paths now use resampled daily
bars (post 28db984 fix), yet they classify opposite states. Possible
causes: sample-vs-population std-dev divisor, accumulator precision
drift, daily-reset semantics differing between batch and incremental.

Procedure:
  1. Load TSLL 1Min 180d, resample to 1Day.
  2. BATCH path: run `prepare_data_with_indicators` and inspect the
     per-day vwap_v2_* columns + VWAP_V2 interpreter state.
  3. INCREMENTAL path: instantiate vwap_v2 incremental class directly,
     walk the same resampled daily bars, dump per-bar
     vwap_v2_upper_2sigma / vwap_v2_lower_2sigma / state classification.
  4. Diff: where do they disagree, and on what column values?
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
set_admin_user_context("19d47e46-f718-49a6-af32-5f5407f5b170")

from data_loader import load_market_data, resample_to_timeframe

# 1. Load TSLL primary 1Min and resample to 1Day
print("=== Loading TSLL/1Min 180 days ===")
df_1m = load_market_data(symbol="TSLL", days=180, timeframe="1Min", session="RTH")
print(f"  bars: {len(df_1m)}")

print("\n=== Resampling 1Min → 1Day ===")
df_1d = resample_to_timeframe(df_1m[['open','high','low','close','volume']].copy(), '1Day')
print(f"  daily bars: {len(df_1d)}")
print(f"  range: {df_1d.index[0]} → {df_1d.index[-1]}")

# 2. BATCH path: instantiate via run_indicators_for_group(...) the way
#    services.prepare_data_with_indicators does it
print("\n=== BATCH path: run_indicators_for_group ===")
from confluence_groups import load_confluence_groups
from indicators import run_indicators_for_group, run_all_indicators
df_batch = run_all_indicators(df_1d.copy())
groups = load_confluence_groups()
for g in groups:
    if g.base_template == 'vwap_v2':
        df_batch = run_indicators_for_group(df_batch, g)
        print(f"  ran group: {g.id} (params={g.parameters})")
        break
else:
    print("  WARNING: no vwap_v2 group found — using default")

# Run interpreter
from interpreters import INTERPRETER_FUNCS
if 'VWAP_V2' in INTERPRETER_FUNCS:
    df_batch['VWAP_V2'] = INTERPRETER_FUNCS['VWAP_V2'](df_batch)
else:
    print("  WARNING: VWAP_V2 not in INTERPRETER_FUNCS")

vwap_v2_cols = [c for c in df_batch.columns if 'vwap' in c.lower() or 'VWAP' in c]
print(f"  vwap-related cols: {sorted(vwap_v2_cols)[:15]}")

# 3. INCREMENTAL path: walk same resampled daily bars through pack's
#    incremental class
print("\n=== INCREMENTAL path: vwap_v2 incremental walk ===")
pack = pack_registry.get_pack('vwap_v2')
defaults = {k: v.get('default') for k, v in (pack.manifest.get('parameters_schema') or {}).items()}
print(f"  defaults: {defaults}")
engine = pack.incremental_class(**defaults)

# Capture per-bar incremental output
inc_records = []
for i in range(len(df_1d)):
    row = df_1d.iloc[i]
    bar = {
        'open': float(row['open']), 'high': float(row['high']),
        'low': float(row['low']), 'close': float(row['close']),
        'volume': float(row.get('volume', 0)),
        'timestamp': df_1d.index[i],
    }
    out = engine.update_bar(bar)
    inc_records.append({'ts': df_1d.index[i], **(out if isinstance(out, dict) else {})})

# 4. Diff
print(f"\n=== Comparison (first 10 bars after warmup) ===")
print(f"  {'ts':<28} {'src':<8} {'vwap_v2_line':>15} {'+2σ':>15} {'-2σ':>15} {'state':<12}")
print(f"  {'-'*100}")
for i in range(20, min(40, len(df_1d))):
    ts = df_1d.index[i]
    # Batch row
    if ts in df_batch.index:
        b = df_batch.loc[ts]
        b_vwap = b.get('vwap_v2_line') or b.get('vwap')
        b_upper = b.get('vwap_v2_upper_2sigma') or b.get('vwap_sd2_upper')
        b_lower = b.get('vwap_v2_lower_2sigma') or b.get('vwap_sd2_lower')
        b_state = b.get('VWAP_V2', '?')
        print(f"  {str(ts):<28} {'batch':<8} "
              f"{(b_vwap or 0):>15.4f} {(b_upper or 0):>15.4f} "
              f"{(b_lower or 0):>15.4f} {str(b_state):<12}")
    # Inc row
    inc = inc_records[i]
    i_vwap = inc.get('vwap_v2_line') or inc.get('vwap')
    i_upper = inc.get('vwap_v2_upper_2sigma') or inc.get('vwap_sd2_upper')
    i_lower = inc.get('vwap_v2_lower_2sigma') or inc.get('vwap_sd2_lower')
    print(f"  {'':<28} {'inc':<8} "
          f"{(i_vwap or 0):>15.4f} {(i_upper or 0):>15.4f} "
          f"{(i_lower or 0):>15.4f} {'(no state, raw inc)':<20}")

# 5. Quick column comparison summary
print(f"\n=== Column-by-column divergence on common keys ===")
inc_keys = set(inc_records[20].keys())
batch_keys = set(df_batch.columns)
common = inc_keys & batch_keys
print(f"  common cols: {sorted(common)}")
for k in sorted(common):
    if k == 'ts': continue
    diffs = []
    for i in range(20, min(60, len(df_1d))):
        ts = df_1d.index[i]
        if ts not in df_batch.index: continue
        b = df_batch.loc[ts, k] if k in df_batch.columns else None
        i_v = inc_records[i].get(k)
        if b is None and i_v is None: continue
        try:
            if abs(float(b) - float(i_v)) > 1e-6:
                diffs.append((ts, b, i_v))
        except Exception:
            pass
    if diffs:
        print(f"  {k}: {len(diffs)} divergent bars; example {diffs[0]}")
    else:
        print(f"  {k}: identical across sample")
