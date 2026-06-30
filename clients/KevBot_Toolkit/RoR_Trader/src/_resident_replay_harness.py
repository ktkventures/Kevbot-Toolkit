"""M-RS4 Phase 3 — Step A/C POSITIVE byte-identity proof (offline, no live market).

Sibling of `_shadow_replay_harness.py`. That harness proved the NEGATIVE result:
snapshot-CHAINED resume (PATH B) is NOT byte-identical to a from-cold run (~4% of
trades drop/change at chunk boundaries) — the "append breaks things" root, and exactly
what `data_worker_engine.run_store_fed_window` does today.

This harness proves the POSITIVE premise Phase 3 is built on: a SINGLE warmed engine fed
each bar ONCE, incrementally, never re-windowed / never resumed (the continuous-resident
design — `shadow_engine.ResidentStrategyEngine`) IS byte-identical to a from-cold full
recompute. It exercises the ACTUAL production class, so if run_unified_backtest's per-bar
loop ever drifts from ResidentStrategyEngine's, this test goes red.

  PATH A  from-cold one-shot      : run_unified_backtest(full_df)         — today's recompute
  PATH C  resident, single feed   : ONE ResidentStrategyEngine.feed(full_df) bar-by-bar
  PATH D  resident, BOOTSTRAP+poll: warm on df[:split], then feed the tail in K batches —
                                    exactly the service's "bootstrap once, then poll for
                                    new settled bars" loop. Trades (warm + all batches)
                                    must still equal PATH A.

All paths run over the SAME prepared df, isolating "one-shot loop" vs "resident feed" vs
"resident bootstrap+batched feed." A==C==D byte-identical (the trade_snapshot KEY/VAL
gate) across the canaries ⇒ the resident design is faithful BY CONSTRUCTION, including
the bootstrap-then-steady-state path. The cold-RESTART heal across a serialize boundary
is Step D (separate; this proves the in-process warm needs no heal).

Usage:  PYTHONPATH=. ../.venv/bin/python _resident_replay_harness.py [SIDS] [DAYS]
  SIDS   comma-separated strategy ids (default a single canary). The GATE is >=3 canaries
         green INCLUDING a sub-minute primary and a secondary-TF-gated one (e.g. 267 =
         10Sec + 2Min covers both; pair with a 1Min + a multi/coarse-secondary one).
  DAYS   window length anchored on each strategy's latest real backtest trade (default 1).
"""
import os, sys, logging
from datetime import datetime, timedelta, timezone

# Determinism: full warmup load, no secondary-TF snapshot fast path. All paths share
# ONE df, so this only affects df-prep cost, never the A/C/D comparison.
os.environ["RORT_SECONDARY_TF_SNAPSHOT"] = "0"
os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
logging.basicConfig(level=logging.WARNING)
for n in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(n).setLevel(logging.ERROR)

import math
SIDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [263]
DAYS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
WARMUP = int(os.environ.get("HARNESS_WARMUP", "300"))
BATCHES = int(os.environ.get("HARNESS_BATCHES", "5"))   # PATH D poll-batch count
SPLIT_FRAC = float(os.environ.get("HARNESS_SPLIT_FRAC", "0.6"))  # PATH D warm fraction
ADMIN_USER = "19d47e46-f718-49a6-af32-5f5407f5b170"

import db
db.set_admin_user_context(ADMIN_USER)
import pack_registry
_np = pack_registry.scan_and_load_all()
print(f"registry: {len(_np) if isinstance(_np, dict) else _np} packs", flush=True)

import pandas as pd
from db import get_admin_client, get_strategy_by_id_admin
from services import prepare_data_with_indicators, get_secondary_tf_map
from data_loader import (
    BARS_PER_DAY, get_required_tfs_from_confluence, get_tf_from_label,
)
import general_packs as gp_module
from unified_engine import run_unified_backtest
from shadow_engine import ResidentStrategyEngine
from trade_snapshot import KEY_FIELDS, VAL_FIELDS


def keyed(df):
    """trades_df -> {entry|exit : {val fields}} using the trade_snapshot gate fields."""
    out = {}
    if df is None or len(df) == 0:
        return out
    for t in df.to_dict("records"):
        k = "|".join(str(t.get(f)) for f in KEY_FIELDS)
        out[k] = {f: (None if t.get(f) is None else t.get(f)) for f in VAL_FIELDS}
    return out


def prepare_window_df(strat, model, since_dt, until_dt):
    """Mirror get_strategy_trades_for_window's cold (no-resume) df-prep so all paths
    share an identical, fully-warmed df."""
    timeframe = strat.get('timeframe', '1Min')
    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))
    all_tfs = [timeframe] + list(sec_tfs)
    bpds = [BARS_PER_DAY.get(tf, 390) for tf in all_tfs]
    binding_bpd = min(bpd for bpd in bpds if bpd > 0) if bpds else 390
    warmup_days = max(1, math.ceil(WARMUP / max(binding_bpd, 0.001) * 365 / 252))

    since_naive = since_dt.replace(tzinfo=None) if since_dt.tzinfo else since_dt
    end_date = until_dt.replace(tzinfo=None) if until_dt.tzinfo else until_dt
    start_date = since_naive - timedelta(days=warmup_days)

    df = prepare_data_with_indicators(
        strat['symbol'], seed=strat.get('data_seed', 42),
        start_date=start_date, end_date=end_date, timeframe=timeframe,
        data_feed="sip", session=strat.get('trading_session', 'RTH'),
        secondary_tfs=sec_tfs, secondary_tf_dfs=None, strat=strat,
        model_override=model,
    )
    if len(df) == 0:
        return df, sec_tfs
    end_clip = pd.Timestamp(end_date)
    if df.index.tz is not None and end_clip.tz is None:
        end_clip = end_clip.tz_localize('UTC')
    elif df.index.tz is None and end_clip.tz is not None:
        end_clip = end_clip.tz_localize(None)
    df = df[df.index <= end_clip]
    return df, sec_tfs


def _cmp(label, A, X, sid):
    ka, kx = set(A), set(X)
    added, removed = kx - ka, ka - kx
    changed = [k for k in (ka & kx) if A[k] != X[k]]
    ok = not added and not removed and not changed
    print(f"    {label}: {len(X):4d} trades  added(X-only)={len(added)} "
          f"removed(A-only)={len(removed)} changed={len(changed)}  "
          f"{'✅' if ok else '❌'}", flush=True)
    if not ok:
        for k in list(removed)[:2]:
            print(f"        A-only: {k}\n            {A[k]}", flush=True)
        for k in list(added)[:2]:
            print(f"        {label}-only: {k}\n            {X[k]}", flush=True)
        for k in changed[:3]:
            diffs = {f: (A[k][f], X[k][f]) for f in VAL_FIELDS if A[k][f] != X[k][f]}
            print(f"        changed {k}: {diffs}", flush=True)
    return ok


def run_sid(sid):
    strat = get_strategy_by_id_admin(sid, ADMIN_USER)
    if not strat:
        print(f"sid {sid}: NOT FOUND for admin user — skipping", flush=True)
        return None
    tf = strat.get("timeframe")
    model = strat.get("backtest_model")

    c = get_admin_client()
    mx = (c.table("trades").select("entry_fill_ts").eq("strategy_id", sid)
          .like("data_source", "backtest_%").order("entry_fill_ts", desc=True)
          .limit(1).execute().data or [])
    if not mx:
        print(f"sid {sid}: no backtest trades to anchor on — skipping", flush=True)
        return None
    max_entry = datetime.fromisoformat(str(mx[0]["entry_fill_ts"]).replace("Z", "+00:00"))
    if max_entry.tzinfo is None:
        max_entry = max_entry.replace(tzinfo=timezone.utc)
    T_end = max_entry + timedelta(hours=1)
    T0 = T_end - timedelta(days=DAYS)
    print(f"\n--- sid={sid} tf={tf} model={model}  window {T0.date()}->{T_end.date()} "
          f"({DAYS}d, warmup={WARMUP}) ---", flush=True)

    df, sec_tfs = prepare_window_df(strat, model, T0, T_end)
    if len(df) < 2:
        print(f"sid {sid}: only {len(df)} bars in window — skipping", flush=True)
        return None
    sec_tf_map = get_secondary_tf_map(df)
    sec_tf_map = sec_tf_map if sec_tf_map else None
    enabled_gen = gp_module.get_enabled_general_packs(gp_module.load_general_packs())
    print(f"    df: {len(df)} bars  secondary_tfs={sec_tfs or '()'}", flush=True)

    import time
    # PATH A — from-cold one-shot (the reference)
    t = time.time()
    df_A, _ = run_unified_backtest(
        df, strat, general_packs=enabled_gen, secondary_tf_map=sec_tf_map,
        include_open_position=False, last_bar_partial=False)
    A = keyed(df_A)
    print(f"    PATH A (one-shot)   : {len(A):4d} trades  {time.time()-t:5.1f}s", flush=True)

    # PATH C — resident, single feed over the whole df
    eng_c = ResidentStrategyEngine(strat, enabled_gen)
    C = keyed(pd.DataFrame(eng_c.feed(df, sec_tf_map) or []))
    ok_c = _cmp("PATH C (resident 1-feed)", A, C, sid)

    # PATH D — resident BOOTSTRAP + poll: warm on df[:split], feed the tail in K batches
    split = max(1, int(len(df) * SPLIT_FRAC))
    eng_d = ResidentStrategyEngine(strat, enabled_gen)
    trades_d = list(eng_d.feed(df.iloc[:split], sec_tf_map) or [])   # bootstrap warm
    tail = df.iloc[split:]
    if len(tail) > 0:
        edges = [int(len(tail) * i / BATCHES) for i in range(BATCHES + 1)]
        for b in range(BATCHES):
            seg = tail.iloc[edges[b]:edges[b + 1]]
            if len(seg) > 0:
                trades_d.extend(eng_d.feed(seg, sec_tf_map) or [])
    D = keyed(pd.DataFrame(trades_d or []))
    ok_d = _cmp(f"PATH D (warm@{split}+{BATCHES}×)", A, D, sid)

    ok = ok_c and ok_d
    print(f"    {'✅' if ok else '❌'} sid {sid}: resident {'==' if ok else '!='} "
          f"from-cold (C {'ok' if ok_c else 'FAIL'}, D {'ok' if ok_d else 'FAIL'})",
          flush=True)
    return ok


results = {}
for sid in SIDS:
    try:
        results[sid] = run_sid(sid)
    except Exception as e:
        import traceback
        print(f"sid {sid}: ERROR {e}", flush=True)
        traceback.print_exc()
        results[sid] = None

print("\n" + "=" * 64)
print("STEP A/C GATE — continuous-resident byte-identity (single feed + bootstrap+poll)")
green = [s for s, r in results.items() if r is True]
red = [s for s, r in results.items() if r is False]
skipped = [s for s, r in results.items() if r is None]
print(f"  green={green}  red={red}  skipped={skipped}")
if red:
    print("  ❌ GATE RED — do NOT flip RORT_BACKTEST_LANE_MODE=shadow.")
elif len(green) >= 3:
    print("  ✅ GATE GREEN (>=3 canaries) — resident engine core validated.")
elif green:
    print(f"  🟡 {len(green)} green, none red — add canaries to reach the >=3 gate "
          "(incl. a sub-minute + a secondary-TF-gated strategy).")
else:
    print("  ⚠️  no canaries evaluated.")
print("=" * 64)
