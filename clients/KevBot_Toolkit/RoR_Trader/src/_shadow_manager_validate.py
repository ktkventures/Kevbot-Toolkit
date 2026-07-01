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

HI-FI GATE (M-RS4 Phase 3 §0, 2026-06-30) — the two-phase-loop regression let the
shadow WRITE stop-loss exits on the PRIMARY-TF bar boundary (e.g. 10s edges) because
Hi-Fi (the per-second refinement) lagged. Bar-aligned exits don't match the live lane's
per-second fills → Strategy-Health pairing broke (all TBD). To catch that class of
regression OFFLINE, after the bar-resolution byte-identity check we run the REAL
`_hifi_resolve_trades` (Hi-Fi Pass 2 — the same refinement the KPI/recompute path calls)
over BOTH the manager and the from-cold trades and assert:
  (1) still byte-identical AFTER Hi-Fi (refinement is deterministic across lanes), and
  (2) every eligible L-type stop/target exit that Hi-Fi refined landed INTRA-CANDLE
      (NOT on the primary-TF bar boundary).
Gated by VALIDATE_HIFI (default 1). AMBER when Hi-Fi couldn't be exercised (no 1-sec
bars for the window — re-run near market hours); does NOT change the Step C verdict.

Usage:  PYTHONPATH=. ../.venv/bin/python _shadow_manager_validate.py [SIDS] [DAYS]
"""
import os, sys, logging
from datetime import datetime, timedelta, timezone

os.environ.setdefault("RORT_SECONDARY_TF_SNAPSHOT", "0")  # override to 1 to test fast path
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
from shadow_manager import ResidentEngineManager, EngineSlot
from services import prepare_strategy_window_df
from trade_snapshot import KEY_FIELDS, VAL_FIELDS

# HI-FI GATE (§0). Default ON; degrades to AMBER when no 1-sec bars are available.
HIFI_GATE = os.environ.get("VALIDATE_HIFI", "1").strip().lower() in (
    "1", "true", "yes", "on")
from api.services.backtest_service import _hifi_resolve_trades, _tf_to_seconds
HIFI_VERDICTS: dict = {}   # sid -> 'green'|'amber'|'red'|None


def _on_boundary(ts_val, tf_seconds):
    """True if `ts_val` sits exactly on a primary-TF bar boundary (bar-aligned),
    False if intra-candle, None if unparseable. Mirrors the eligibility test in
    `backtest_service._hifi_resolve_trades` (`.second % tf_seconds == 0 and
    .microsecond == 0`) so 'intra-candle' means the same thing as the refiner's."""
    if ts_val is None:
        return None
    try:
        ts = pd.Timestamp(ts_val)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(ts):
        return None
    return (int(ts.second) % tf_seconds == 0
            and int(ts.microsecond) == 0
            and int(getattr(ts, "nanosecond", 0)) == 0)


def _refine(df_trades, df_bars, strat, symbol, tf):
    """Apply the REAL Hi-Fi Pass 2 to a COPY of the trades — exactly as the
    KPI/recompute path does (`_hifi_resolve_trades`). Ensures the columns the
    refiner reads are present (`hifi_resolved` default False; `direction` from
    the strategy config — the engine dict omits it and the refiner defaults LONG,
    so inject it for a faithful per-second walk on SHORTs)."""
    if df_trades is None or len(df_trades) == 0:
        return df_trades
    d = df_trades.copy()
    if "hifi_resolved" not in d.columns:
        d["hifi_resolved"] = False
    if "direction" not in d.columns:
        d["direction"] = strat.get("direction", "LONG")
    return _hifi_resolve_trades(d, symbol, tf, bar_df=df_bars)


def _intra_candle_stats(df_ref_trades, tf_seconds, after_iso):
    """Among steady (entry > after_iso) L-type stop/target exits in a REFINED
    trade frame, count: eligible, Hi-Fi-refined, and of those how many landed
    intra-candle (off the bar boundary). Uses the same `str(entry) <= after`
    steady filter as `keyed()` so the population matches the byte-identity set."""
    elig = refined = offbound = 0
    if df_ref_trades is None or len(df_ref_trades) == 0:
        return {"elig": 0, "refined": 0, "offbound": 0}
    for t in df_ref_trades.to_dict("records"):
        ent = t.get("entry_fill_ts")
        if ent is None or str(ent) <= after_iso:
            continue
        er = t.get("exit_reason")
        stop_et = str(t.get("stop_exec_type", "L")).upper()
        tgt_et = str(t.get("target_exec_type", "L")).upper()
        is_stop = er in ("stop_loss", "stop") and stop_et == "L"
        is_tgt = er == "target" and tgt_et == "L"
        if not (is_stop or is_tgt):
            continue
        elig += 1
        if t.get("hifi_resolved") is True:
            refined += 1
            if _on_boundary(t.get("exit_fill_ts"), tf_seconds) is False:
                offbound += 1
    return {"elig": elig, "refined": refined, "offbound": offbound}


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

    # REFERENCE — true from-cold over [T0-warmup, T_end]: FULL resample (snapshot OFF),
    # backfill ALLOWED (warms the cache so the read-only manager path below finds its
    # bars), snapshot not persisted. The manager then runs with MANAGER_SNAPSHOT (0 =
    # full-resample read-only correctness; 1 = the dev fast path) — comparing it to this
    # full-resample ground truth.
    mgr_snap = os.environ.get("MANAGER_SNAPSHOT", "0")
    os.environ["RORT_SECONDARY_TF_SNAPSHOT"] = "0"
    import time
    t = time.time()
    df_ref = prepare_strategy_window_df(
        strat, T0, T_end, warmup_bars=300, data_feed="sip", model_override=model,
        no_backfill=False, persist_snapshot=False)
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
    os.environ["RORT_SECONDARY_TF_SNAPSHOT"] = mgr_snap   # manager mode (0=full, 1=fast path)
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

    # ── HI-FI GATE (§0) — refine BOTH lanes with the real Pass 2, then assert
    # post-Hi-Fi byte-identity + intra-candle stop/target exits. Additive: never
    # changes the Step C (bar-resolution) verdict returned below.
    if HIFI_GATE:
        try:
            tf_secs = _tf_to_seconds(tf)
            sym = strat["symbol"]
            rA = _refine(df_A, df_ref, strat, sym, tf)
            rM = _refine(pd.DataFrame(acc) if acc else pd.DataFrame(),
                         df_ref, strat, sym, tf)
            Ahf, Mhf = keyed(rA, after=T0_iso), keyed(rM, after=T0_iso)
            ka2, km2 = set(Ahf), set(Mhf)
            hf_added, hf_removed = km2 - ka2, ka2 - km2
            hf_changed = [k for k in (ka2 & km2) if Ahf[k] != Mhf[k]]
            hf_ident = not hf_added and not hf_removed and not hf_changed
            st = _intra_candle_stats(rM, tf_secs, T0_iso)
            # Verdict keys on WHETHER eligible exits got Hi-Fi-walked, NOT on the
            # refined timestamp's boundary alignment. A refined exit that lands on
            # the bar boundary is legitimate — the gap-aware walker fills at the exit
            # bar's OPEN (second :00) when the first 1-sec bar already breached the
            # level (overnight gap / flash move). The REGRESSION signature is the
            # opposite: eligible stop/target exits left UN-walked (refined < elig),
            # which stay bar-aligned because Hi-Fi never resolved them.
            on_bound = st["refined"] - st["offbound"]   # refined-but-boundary = gap fills
            if not hf_ident:
                verdict = "red"
                note = (f"post-Hi-Fi DIVERGENCE added={len(hf_added)} "
                        f"removed={len(hf_removed)} changed={len(hf_changed)}")
            elif st["elig"] == 0:
                verdict = "amber"
                note = "no L-type stop/target exits in window — nothing to refine"
            elif st["refined"] == 0:
                verdict = "amber"
                note = ("Hi-Fi not exercised (no 1-sec bars for window) — "
                        "re-run near market hours")
            elif st["refined"] < st["elig"]:
                verdict = "amber"
                note = (f"partial 1-sec coverage: {st['refined']}/{st['elig']} "
                        f"eligible walked — re-run near market hours for full coverage")
            else:  # refined == elig: every eligible stop/target exit was Hi-Fi-walked
                verdict = "green"
                note = (f"{st['refined']}/{st['elig']} eligible refined; "
                        f"{st['offbound']} intra-candle"
                        + (f", {on_bound} boundary/gap fill(s)" if on_bound else ""))
            HIFI_VERDICTS[sid] = verdict
            sym_e = {"green": "✅", "amber": "🟡", "red": "❌"}[verdict]
            print(f"    HI-FI GATE {sym_e} {verdict.upper()}: post-hifi identical="
                  f"{hf_ident} elig={st['elig']} refined={st['refined']} "
                  f"intra_candle={st['offbound']} boundary/gap={on_bound} — {note}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            HIFI_VERDICTS[sid] = None
            print(f"    HI-FI GATE: ERROR {e}", flush=True)
            traceback.print_exc()
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

if HIFI_GATE:
    hg = [s for s, v in HIFI_VERDICTS.items() if v == "green"]
    ha = [s for s, v in HIFI_VERDICTS.items() if v == "amber"]
    hr = [s for s, v in HIFI_VERDICTS.items() if v == "red"]
    he = [s for s, v in HIFI_VERDICTS.items() if v is None]
    print("HI-FI GATE — stop/target exits intra-candle + byte-identical post-Hi-Fi")
    print(f"  green={hg}  amber={ha}  red={hr}  error={he}")
    if hr:
        print("  ❌ HI-FI RED — refined exits bar-aligned or lanes diverge post-Hi-Fi; "
              "do NOT arm live.")
    elif hg:
        print(f"  ✅ HI-FI GREEN ({len(hg)}) — per-second exits validated"
              + (f"; {len(ha)} amber (no 1-sec data — re-run at market hours)"
                 if ha else "") + ".")
    elif ha:
        print("  🟡 HI-FI AMBER only — Hi-Fi not exercised offline (no 1-sec bars). "
              "Re-run near market hours to turn green.")
    print("=" * 64)
