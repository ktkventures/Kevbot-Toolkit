"""M-RS4 Phase 3 — ResidentEngineManager offline validation (Step C gate).

Proves the MANAGER (not just the engine core) emits settled trades byte-identical to a
from-cold full recompute — driving the real `ResidentEngineManager.advance()` exactly as
the live shadow-worker will: bootstrap a resident engine, then run K polls that EACH
re-prepare a fresh warmup-bounded window and feed only the new bars in.

This is stricter than the engine-core harness (PATH C/D fed slices of ONE prepared df):
here every poll calls `prepare_data_with_indicators` independently, so it also proves the
windowed re-prepare reproduces the one-shot's user-pack + secondary-TF columns (warmup
stability). Compared over the STEADY region (entry > T0) — the bootstrap warms with the
same left edge as the reference, so state at T0 is identical and the comparison isolates
the per-poll re-prepare + resident feed.

  REFERENCE  run_unified_backtest over [T0-warmup, T_end]   (from-cold), entry > T0
  MANAGER    advance(slot, until=T0, since=T0)  then  K polls advance(slot, until_k)

Usage:  PYTHONPATH=. ../.venv/bin/python _shadow_manager_validate.py [SIDS] [DAYS]
"""
import os, sys, logging
from datetime import datetime, timedelta, timezone

os.environ["RORT_SECONDARY_TF_SNAPSHOT"] = "0"
os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
logging.basicConfig(level=logging.WARNING)
for n in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(n).setLevel(logging.ERROR)

SIDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [263]
DAYS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
POLLS = int(os.environ.get("VALIDATE_POLLS", "6"))
ADMIN_USER = "19d47e46-f718-49a6-af32-5f5407f5b170"

import db
db.set_admin_user_context(ADMIN_USER)
import pack_registry
_np = pack_registry.scan_and_load_all()
print(f"registry: {len(_np) if isinstance(_np, dict) else _np} packs", flush=True)

import pandas as pd
from db import get_admin_client, get_strategy_by_id_admin
from services import get_secondary_tf_map
import general_packs as gp_module
from unified_engine import run_unified_backtest
from shadow_manager import ResidentEngineManager, EngineSlot, prepare_window
from trade_snapshot import KEY_FIELDS, VAL_FIELDS


def keyed(df, after=None):
    out = {}
    if df is None or len(df) == 0:
        return out
    for t in df.to_dict("records"):
        if after is not None:
            ent = t.get('entry_fill_ts')
            if ent is None or str(ent) <= after:
                continue
        k = "|".join(str(t.get(f)) for f in KEY_FIELDS)
        out[k] = {f: (None if t.get(f) is None else t.get(f)) for f in VAL_FIELDS}
    return out


def run_sid(sid):
    strat = get_strategy_by_id_admin(sid, ADMIN_USER)
    if not strat:
        print(f"sid {sid}: NOT FOUND — skipping", flush=True)
        return None
    model = strat.get("backtest_model")
    tf = strat.get("timeframe")

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
    T0_iso = T0.isoformat()
    print(f"\n--- sid={sid} tf={tf} model={model}  window {T0.date()}->{T_end.date()} "
          f"({DAYS}d, {POLLS} polls) ---", flush=True)

    mgr = ResidentEngineManager(shard_symbols={strat['symbol']}, dry_run=True)
    slot = EngineSlot(sid, ADMIN_USER, strat)
    if not slot.eligible:
        print(f"sid {sid}: ineligible ({slot.ineligible_reason}) — skipping", flush=True)
        return None

    # REFERENCE — from-cold over [T0-warmup, T_end].
    import time
    t = time.time()
    df_ref = prepare_window(strat, model, T0, T_end, slot.timeframe, slot.sec_tfs)
    if df_ref is None or len(df_ref) < 2:
        print(f"sid {sid}: only {0 if df_ref is None else len(df_ref)} bars — skipping",
              flush=True)
        return None
    sec_tf_map = get_secondary_tf_map(df_ref) or None
    enabled_gen = gp_module.get_enabled_general_packs(gp_module.load_general_packs())
    df_A, _ = run_unified_backtest(
        df_ref, strat, general_packs=enabled_gen, secondary_tf_map=sec_tf_map,
        include_open_position=False, last_bar_partial=False)
    A = keyed(df_A, after=T0_iso)
    print(f"    REFERENCE (from-cold) : {len(A):4d} steady trades  {time.time()-t:5.1f}s",
          flush=True)

    # MANAGER — bootstrap to T0 (same left edge), then K polls re-preparing each time.
    # RESTART (Step D): if VALIDATE_RESTART_POLL=p, after poll p we DISCARD the resident
    # engine and cold re-bootstrap at that boundary — simulating a crash + restart with
    # NO snapshot. If the combined trades still equal from-cold, crash recovery via cold
    # re-bootstrap is byte-identical (no snapshot/heal needed for correctness).
    restart_poll = int(os.environ.get("VALIDATE_RESTART_POLL", "0"))
    t = time.time()
    mgr.advance(slot, until_dt=T0, since_override=T0)   # bootstrap warm; discard its trades
    edges = [T0 + (T_end - T0) * (i / POLLS) for i in range(POLLS + 1)]
    acc = []
    for i in range(POLLS):
        acc.extend(mgr.advance(slot, until_dt=edges[i + 1]))
        if restart_poll and (i + 1) == restart_poll:
            slot.engine = None
            slot.last_processed_ts = None
            mgr.advance(slot, until_dt=edges[i + 1], since_override=edges[i + 1])  # re-warm
            print(f"    ↻ simulated crash + cold re-bootstrap after poll {restart_poll}",
                  flush=True)
    M = keyed(pd.DataFrame(acc) if acc else pd.DataFrame(), after=T0_iso)
    tag = f"bootstrap+{POLLS}×" + (f", restart@{restart_poll}" if restart_poll else "")
    print(f"    MANAGER ({tag}): {len(M):4d} steady trades  {time.time()-t:5.1f}s",
          flush=True)

    ka, km = set(A), set(M)
    added, removed = km - ka, ka - km
    changed = [k for k in (ka & km) if A[k] != M[k]]
    ok = not added and not removed and not changed
    print(f"    A={len(A)} M={len(M)} added(M-only)={len(added)} "
          f"removed(A-only)={len(removed)} changed={len(changed)}  "
          f"{'✅ byte-identical' if ok else '❌ DIVERGENCE'}", flush=True)
    if not ok:
        for k in list(removed)[:3]:
            print(f"        A-only: {k}\n            {A[k]}", flush=True)
        for k in list(added)[:3]:
            print(f"        M-only: {k}\n            {M[k]}", flush=True)
        for k in changed[:5]:
            diffs = {f: (A[k][f], M[k][f]) for f in VAL_FIELDS if A[k][f] != M[k][f]}
            print(f"        changed {k}: {diffs}", flush=True)
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
print("STEP C MANAGER GATE — settled trades == from-cold (live-poll simulation)")
green = [s for s, r in results.items() if r is True]
red = [s for s, r in results.items() if r is False]
skipped = [s for s, r in results.items() if r is None]
print(f"  green={green}  red={red}  skipped={skipped}")
if red:
    print("  ❌ GATE RED — manager diverges; do NOT wire shadow-mode to write.")
elif len(green) >= 3:
    print("  ✅ GATE GREEN (>=3 canaries) — manager settled lane validated.")
elif green:
    print(f"  🟡 {len(green)} green, none red — add canaries to reach >=3.")
else:
    print("  ⚠️  no canaries evaluated.")
print("=" * 64)
