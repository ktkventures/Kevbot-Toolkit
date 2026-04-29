"""Probe v2: compare unified_engine.run_unified_backtest vs StrategyMonitor
on the same bars and find where they diverge.

The 4Q pack parity passes; the strategy-level parity all-FAILs. Engine
core code (TriggerEvaluator.evaluate_bar_close) ALREADY pulls user-pack
trigger booleans from `current` into `triggers` (unified_engine.py:1331).
So the bug isn't lack of dispatch — it's something else.

Probe steps:
  1. Pick sid 137 (TSLA/5Min, swing_123 entry).
  2. Load 60 days of bars (matches what parity replay would use).
  3. Run unified_engine.run_unified_backtest(df, strategy) → trades_a
  4. Run a manual StrategyMonitor.on_bar_close loop → fires_b
  5. Compare:
     - count of trades vs fires
     - timestamps of trades vs fires
     - first divergent bar's trigger_bools / interpreter_states / confluence_records
  6. Report exactly where they differ.

Run from src/:
    ../.venv/bin/python _probe_parity_divergence.py
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context, get_strategy_by_id_db
from data_loader import load_market_data
from ralph_engine import StrategyMonitor, SymbolHub, SECONDS_TO_TIMEFRAME

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SID = 137  # TSLA/5Min, swing_123 — backtest=141, replay=0
DAYS = 60  # roughly the parity replay's window for this strategy

set_admin_user_context(USER)
print(f"\n=== Probe v2: divergence on sid {SID} ===\n")

strategy = get_strategy_by_id_db(SID)
print(f"Strategy: {strategy['name']}  symbol={strategy['symbol']}  tf={strategy['timeframe']}")
print(f"  entry={strategy.get('entry_trigger_confluence_id')}")
print(f"  confluence={strategy.get('confluence', [])}")
print()

# Load bars
df = load_market_data(symbol=strategy['symbol'], days=DAYS,
                     timeframe=strategy['timeframe'], session='RTH')
print(f"Loaded {len(df)} primary bars")

# ─── Path A: unified_engine.run_unified_backtest ───
import services as svc

# Determine secondary TFs from confluence (matches services.get_strategy_trades)
from data_loader import get_required_tfs_from_confluence, get_tf_from_label
req_labels = get_required_tfs_from_confluence(strategy.get('confluence', []))
sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels if get_tf_from_label(lbl)))
print(f"Secondary TFs: {sec_tfs}")
print()

# Run via prepare_data_with_indicators + unified_trades (the standard backtest path)
print("─── Path A: unified backtest ───")
df_enriched = svc.prepare_data_with_indicators(
    strategy['symbol'], DAYS, 42,
    timeframe=strategy['timeframe'], session='RTH',
    secondary_tfs=sec_tfs)
print(f"  enriched df: {len(df_enriched)} rows × {len(df_enriched.columns)} cols")

trades_a = svc.unified_trades(df_enriched, strategy)
print(f"  trades from unified_engine: {len(trades_a)}")
if len(trades_a) > 0:
    print(f"  first 3 trade entry timestamps: "
          f"{[str(t)[:19] for t in trades_a['entry_time'].head(3)]}")
print()

# ─── Path B: StrategyMonitor + SymbolHub shadow engines ───
# Mirrors parity_service._replay_strategy exactly, including cross-TF
# fan-out so we can tell whether the bug is in shadow-engine output OR
# in how the primary engine consumes the cross-TF buffer.
print("─── Path B: live engine replay (with shadow engines) ───")
hub = SymbolHub(strategy['symbol'], publisher=None)
monitor = StrategyMonitor(strategy, None, general_packs=[])
hub.add_monitor(monitor)
hub.finalize_shadow_engines()

# Load secondary TF bars
secondary_tfs_seconds = monitor._required_secondary_tf
print(f"  required_secondary_tf = {secondary_tfs_seconds}")
sec_dfs = {}
for sec_tf in secondary_tfs_seconds:
    sec_label = SECONDS_TO_TIMEFRAME.get(sec_tf)
    if not sec_label:
        continue
    sec_df = load_market_data(symbol=strategy['symbol'], days=DAYS,
                              timeframe=sec_label, session='RTH')
    if sec_df is not None and len(sec_df) >= 5:
        sec_dfs[sec_tf] = sec_df
        print(f"  loaded {len(sec_df)} bars for secondary TF {sec_label}")

# Warmup primary
WARMUP = 200
monitor.warmup(df.iloc[:WARMUP])

# Warmup shadows
for sec_tf, sec_df in sec_dfs.items():
    sw = min(200, max(5, len(sec_df) // 4))
    shadow = hub._shadow_engines.get(sec_tf)
    if shadow is not None:
        try:
            shadow.warmup(sec_df.iloc[:sw])
            print(f"  warmed up shadow {sec_tf}s on {sw} bars")
        except Exception as e:
            print(f"  shadow {sec_tf}s warmup failed: {e}")

# Track secondary iter positions
sec_state = {}
for sec_tf, sec_df in sec_dfs.items():
    sw = min(200, max(5, len(sec_df) // 4))
    sec_state[sec_tf] = {'df': sec_df, 'idx': sw}

print(f"  hub._shadow_engines keys: {list(hub._shadow_engines.keys())}")
for sec_tf, sh in hub._shadow_engines.items():
    print(f"    shadow {sec_tf}s: req_ind={sh.indicators.required} "
          f"req_interp={sh.trigger_eval.required_interpreters}")

fires_b = []
true_bar_indices = []
true_bars_with_full_confluence = []
sample_audits_at_trigger = []  # save first 3 trigger-fired bars for inspection
sec_feed_count = {tf: 0 for tf in sec_state}  # how many secondary bars fed during replay
sec_records_seen = {tf: set() for tf in sec_state}  # all distinct records emitted
for i in range(WARMUP, len(df)):
    ts = df.index[i]

    # Advance secondary TFs (mirrors parity_service replay loop)
    for sec_tf, st in sec_state.items():
        shadow = hub._shadow_engines.get(sec_tf)
        if shadow is None:
            continue
        sec_df = st['df']
        while st['idx'] < len(sec_df):
            sec_ts = sec_df.index[st['idx']]
            if sec_ts > ts:
                break
            sec_row = sec_df.iloc[st['idx']]
            sec_bar = {
                'open':   float(sec_row['open']),
                'high':   float(sec_row['high']),
                'low':    float(sec_row['low']),
                'close':  float(sec_row['close']),
                'volume': float(sec_row.get('volume', 0)),
                'timestamp': sec_ts,
            }
            try:
                records = shadow.on_bar_close(sec_bar)
                hub._mtf_confluence[sec_tf] = records
                sec_feed_count[sec_tf] += 1
                if records:
                    sec_records_seen[sec_tf].update(records)
            except Exception as e:
                print(f"shadow bar_close failed: {e}")
            st['idx'] += 1

    row = df.iloc[i]
    bar = {
        'open':   float(row['open']),
        'high':   float(row['high']),
        'low':    float(row['low']),
        'close':  float(row['close']),
        'volume': float(row.get('volume', 0)),
        'timestamp': ts,
    }
    signals, audit = monitor.on_bar_close(
        bar, i, mtf_confluence=dict(hub._mtf_confluence))

    if audit['trigger_booleans'].get('sw123_bull_c2'):
        true_bar_indices.append((i, ts))
        # Sample bars: first 1, middle 1, last 1 — to avoid all sampling
        # from the early-replay window where shadows haven't yet emitted.
        sample_audits_at_trigger.append({
            'idx': i, 'ts': ts,
            'confluence': sorted(monitor._current_confluence),
            'mtf_buf': {k: sorted(v) for k, v in hub._mtf_confluence.items()},
            'position': audit['position_state'],
        })
    for sig in signals:
        if 'entry' in (sig.get('signal_type') or sig.get('type', '')).lower():
            fires_b.append({'bar_idx': i, 'ts': str(ts)[:19]})

print(f"  fires from StrategyMonitor: {len(fires_b)}")
print(f"  bars where sw123_bull_c2=True (regardless of confluence/position): {len(true_bar_indices)}")
print(f"\n  Secondary feed counts during replay:")
for tf, n in sec_feed_count.items():
    print(f"    tf={tf}s: {n} bars fed")
print(f"\n  Distinct records emitted by each shadow:")
for tf, recs in sec_records_seen.items():
    print(f"    tf={tf}s: {len(recs)} distinct, sample: {sorted(recs)[:5]}")
if fires_b:
    print(f"  first 3 fire timestamps: {[f['ts'] for f in fires_b[:3]]}")
print()

# ─── Compare ───
print("─── Divergence analysis ───")
if len(trades_a) == 0 and len(fires_b) == 0:
    print("  Both paths produce 0 trades. Insufficient data window or strict gates.")
elif len(trades_a) > 0 and len(fires_b) == 0:
    print(f"  ⚠ unified_backtest produced {len(trades_a)} trades but live engine produced 0!")
    print(f"  Trigger fired on {len(true_bar_indices)} bars in live path, but no signals emitted.")
    print(f"  Likely: confluence gate or position state machine blocking entry.\n")

    required = set(strategy.get('confluence', []))
    print(f"  Strategy required confluence: {sorted(required)}\n")

    # Pick first / middle / last from sample_audits_at_trigger
    if len(sample_audits_at_trigger) >= 3:
        picked = [
            sample_audits_at_trigger[0],
            sample_audits_at_trigger[len(sample_audits_at_trigger) // 2],
            sample_audits_at_trigger[-1],
        ]
    else:
        picked = sample_audits_at_trigger
    print(f"  Sampling {len(picked)} bars (first/middle/last of {len(sample_audits_at_trigger)} trigger-fires):")
    for i, sample in enumerate(picked, 1):
        print(f"\n  --- Sample {i} | bar {sample['idx']} @ {sample['ts']} ---")
        print(f"    monitor _current_confluence: {sample['confluence']}")
        print(f"    hub._mtf_confluence buffer: {sample['mtf_buf']}")
        print(f"    position_state: {sample['position']}")
        seen = set(sample['confluence'])
        missing = required - seen
        if missing:
            print(f"    ⚠ Missing required confluence: {sorted(missing)}")
        else:
            print(f"    All required confluence present (so why did entry block?)")
elif len(fires_b) > 0 and len(trades_a) == 0:
    print(f"  ⚠ Inverse: live fires {len(fires_b)} but backtest produced 0 trades")
else:
    print(f"  Both produced trades: A={len(trades_a)} B={len(fires_b)}")


# ─── Diagnostic: compare BATCH vs LIVE interpreter on the same 1D df ───
print("\n─── Compare batch vs live MACD_HISTOGRAM_V2 on 1D bars ───")
sec_1d_df = sec_dfs.get(86400)
if sec_1d_df is None:
    print("  No 1D bars loaded; skipping diagnostic.")
else:
    # Need to compute the 1D indicators (mhv2_hist) before running interpreter
    from indicators import run_all_indicators
    from confluence_groups import load_confluence_groups, get_enabled_groups, get_template
    from interpreters import run_all_interpreters

    # Load the strategy's enabled groups + run group-specific indicators
    # like services.prepare_data_with_indicators does
    enriched_1d = run_all_indicators(sec_1d_df.copy())
    # Run group-specific indicators (incl. mhv2_*)
    from services import run_indicators_for_group
    for group in get_enabled_groups(load_confluence_groups()):
        try:
            enriched_1d = run_indicators_for_group(enriched_1d, group)
        except Exception:
            pass

    # Run user-pack interpreter on the full 1D df
    pack = pack_registry.get_registered_packs().get('macd_histogram_v2')
    if pack and pack.interpreter_func:
        batch_states = pack.interpreter_func(enriched_1d)
        from collections import Counter
        batch_counts = Counter(batch_states.dropna().tolist())
        print(f"  BATCH state distribution across {len(sec_1d_df)} 1D bars: {dict(batch_counts)}")

        # Sample a few mhv2_hist values from enriched
        if 'mhv2_hist' in enriched_1d.columns:
            print(f"  Sample mhv2_hist values:")
            for j, idx in enumerate(enriched_1d.index[-5:]):
                h = enriched_1d.loc[idx, 'mhv2_hist']
                s = batch_states.iloc[-5 + j] if len(batch_states) >= 5 else None
                print(f"    {idx}  mhv2_hist={h:.4f}  state={s}")
        else:
            print(f"  ⚠ mhv2_hist NOT in enriched_1d columns — indicator pipeline didn't run for this group")
            print(f"  enriched_1d columns sample: {list(enriched_1d.columns)[:20]}")
    else:
        print("  pack macd_histogram_v2 not registered or has no interpreter_func")

print("\nDone.")
