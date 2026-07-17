"""M-RS5a Phase-2 probe: is the resident engine SELF-SUFFICIENT for a user-pack
strategy with OHLCV-only feeding? Every fleet pack ships an incremental_class, and
IncrementalIndicatorEngine._update_indicators drives them (unified_engine.py:1609) —
so the engine may compute user-pack indicators+triggers itself, making the df columns
redundant. Test: run the manager with a FORCED OHLCV-only resident frame vs the normal
full-prep path (both bootstrap identically at T0), compare trades. IDENTICAL => the
resident frame needs NO user-pack derived columns for trigger-only strategies; eligibility
can broaden to incremental-pack slots (no secondary). Read-only, dry_run, no DB writes.
"""
import os, sys
os.environ.setdefault("RORT_SECONDARY_TF_SNAPSHOT", "0")
os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import logging
logging.basicConfig(level=logging.ERROR)
for n in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(n).setLevel(logging.ERROR)
from datetime import datetime, timedelta, timezone
import db
ADMIN = "19d47e46-f718-49a6-af32-5f5407f5b170"
db.set_admin_user_context(ADMIN)
import pack_registry
pack_registry.scan_and_load_all()
import pandas as pd
from db import get_admin_client, get_strategy_by_id_admin
import shadow_manager as _sm
from shadow_manager import ResidentEngineManager, EngineSlot
from data_worker_engine import _UserContext
from trade_snapshot import KEY_FIELDS, VAL_FIELDS

POLLS = 6
SIDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [263]
DAYS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


def keyed(df, after):
    out = {}
    if df is None or len(df) == 0:
        return out
    for t in df.to_dict("records"):
        ent = t.get('entry_fill_ts')
        if ent is None or str(ent) <= after:
            continue
        k = "|".join(str(t.get(f)) for f in KEY_FIELDS)
        out[k] = {f: (None if t.get(f) is None else t.get(f)) for f in VAL_FIELDS}
    return out


def run_manager(sid, strat, use_frame, T0, edges, T0_iso):
    mgr = ResidentEngineManager(shard_symbols={strat['symbol']}, dry_run=True)
    slot = EngineSlot(sid, ADMIN, strat)
    _sm.RESIDENT_FRAME = False
    mgr.advance(slot, until_dt=T0, since_override=T0)          # bootstrap (full-prep)
    if use_frame:
        _sm.RESIDENT_FRAME = True
        with _UserContext(ADMIN):
            dfb = _sm.prepare_window(strat, slot.bt_model, T0, T0,
                                     slot.timeframe, slot.sec_tfs)
        eng_ts = slot.engine.last_bar_ts
        frame = _sm.ResidentFrame(dfb[dfb.index <= eng_ts])
        frame.trim(_sm.WARMUP_BARS)
        slot.frame = frame
        slot.frame_eligible = True
        used = (frame.tail is not None and pd.Timestamp(frame.tail) == pd.Timestamp(eng_ts))
        if not used:
            print(f"    [warn] forced frame tail {frame.tail} != cursor {eng_ts}")
    else:
        _sm.RESIDENT_FRAME = False
    acc = []
    for i in range(POLLS):
        acc.extend(mgr.advance(slot, until_dt=edges[i + 1]))
    _sm.RESIDENT_FRAME = False
    return keyed(pd.DataFrame(acc) if acc else pd.DataFrame(), after=T0_iso), \
        (slot.frame is not None)


def run_sid(sid):
    strat = get_strategy_by_id_admin(sid, ADMIN)
    tf = strat.get("timeframe")
    c = get_admin_client()
    mx = (c.table("trades").select("entry_fill_ts").eq("strategy_id", sid)
          .like("data_source", "backtest_%").order("entry_fill_ts", desc=True)
          .limit(1).execute().data or [])
    if not mx:
        print(f"sid {sid}: no anchor trades — skip"); return None
    # Pin a SETTLED historical window (env) to avoid live-edge bar_cache drift between the
    # two sequential manager runs; else anchor on the latest trade (live edge — drifts).
    if os.environ.get("PIN_T0"):
        T0 = datetime.fromisoformat(os.environ["PIN_T0"])
        if T0.tzinfo is None:
            T0 = T0.replace(tzinfo=timezone.utc)
        T_end = (datetime.fromisoformat(os.environ["PIN_TEND"]).replace(tzinfo=timezone.utc)
                 if os.environ.get("PIN_TEND") else T0 + timedelta(days=DAYS))
    else:
        me = datetime.fromisoformat(str(mx[0]["entry_fill_ts"]).replace("Z", "+00:00"))
        if me.tzinfo is None:
            me = me.replace(tzinfo=timezone.utc)
        T_end = me + timedelta(hours=1)
        T0 = T_end - timedelta(days=DAYS)
    T0_iso = str(pd.Timestamp(T0))
    edges = [T0 + (T_end - T0) * (i / POLLS) for i in range(POLLS + 1)]
    # EDGE_OFFSET_S: shift the POLL bounds (not the bootstrap T0) off bar boundaries to
    # force a PARTIAL forming bucket at each poll's `until` — the LIVE scenario (settled_until
    # = now-15min is not bar-aligned), where internal-incremental vs batch user-pack values
    # for the partial bar could diverge. 0 = bar-aligned hourly edges.
    _off = int(os.environ.get("EDGE_OFFSET_S", "0"))
    if _off:
        edges = [edges[0]] + [e + timedelta(seconds=_off) for e in edges[1:]]
    print(f"\n--- sid={sid} tf={tf} window {T0.date()}->{T_end.date()} ({DAYS}d, {POLLS}p) ---")

    full, _ = run_manager(sid, strat, False, T0, edges, T0_iso)
    frame, frame_survived = run_manager(sid, strat, True, T0, edges, T0_iso)

    def diff(a, b):
        ka, kb = set(a), set(b)
        return (kb - ka, ka - kb, [k for k in (ka & kb) if a[k] != b[k]])

    def report(label, a, b):
        added, removed, changed = diff(a, b)
        ok = not added and not removed and not changed
        print(f"    [{label}] FULL={len(a)} FRAME={len(b)} added={len(added)} "
              f"removed={len(removed)} changed={len(changed)}  "
              f"{'✅ identical' if ok else '❌ diverges'}")
        for k in list(removed)[:2]:
            print(f"        full-only : {k}  exit={a[k].get('exit_fill_ts')}")
        for k in list(added)[:2]:
            print(f"        frame-only: {k}")
        for k in changed[:2]:
            d = {f: (a[k][f], b[k][f]) for f in VAL_FIELDS if a[k][f] != b[k][f]}
            print(f"        changed {k}: {d}")
        return ok

    # Live only COMMITS settled trades (exit <= now-LAG). Partition the comparison:
    # full window vs settled-only (exit <= T_end - 15min) — a divergence confined to the
    # unsettled tail is harmless (never committed); a settled divergence is real Phase-2 work.
    SETTLED = str(pd.Timestamp(T_end - timedelta(minutes=15)))
    def settled_only(d):
        # exit_fill_ts is the 2nd KEY field (entry|exit|...), not in VAL_FIELDS.
        out = {}
        for k, v in d.items():
            parts = k.split("|")
            ex = parts[1] if len(parts) > 1 else None
            if ex and ex not in ("None", "NaT") and ex <= SETTLED:
                out[k] = v
        return out
    print(f"    frame_survived={frame_survived}  settled_cutoff(exit<=)={SETTLED}")
    ok_full = report("full-window", full, frame)
    ok_settled = report("settled-only", settled_only(full), settled_only(frame))
    return ok_settled


res = {}
for sid in SIDS:
    try:
        res[sid] = run_sid(sid)
    except Exception as e:
        import traceback; print(f"sid {sid} ERROR {e}"); traceback.print_exc(); res[sid] = None
print("\n" + "=" * 60)
print("USER-PACK OHLCV-SELF-SUFFICIENCY PROBE:",
      {s: ("SELF-SUFF" if r else "DIVERGES" if r is False else "skip") for s, r in res.items()})
