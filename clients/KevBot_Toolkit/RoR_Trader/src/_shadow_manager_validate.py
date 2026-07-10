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

DEEP-REFERENCE variant (default ON — Mechanism #1 gate hole, Bug_Hunt_Wave1
2026-07-06): the original gate gave the reference the SAME left edge as the
manager (both `since=T0, warmup_bars=300`), so a warmup-window truncation that
hits both identically (e.g. the calendar-days formula delivering ZERO pre-T0
bars after a weekend/holiday, re-seeding recursive user-pack columns) could
never fail the gate. With VALIDATE_DEEP_REF=1 the reference's `since` is pushed
back so its left edge sits >=2 TRADING days deeper than every window the
manager prepares — the reference is then converged truth, and any manager-side
warmup truncation shows up as divergence. The measured edge depth is asserted
per sid (DEPTH-INVALID = the run cannot certify the bug class; deepen
VALIDATE_DEEP_REF_DAYS).

Env:
  VALIDATE_DEEP_REF=1        deep reference ON (default; 0 = legacy same-edge)
  VALIDATE_DEEP_REF_DAYS=10  min calendar days the reference `since` precedes T0
                             (actual = max(this, manager-formula-days + 7))
  VALIDATE_T0 / VALIDATE_TEND  pin the window explicitly (ISO, e.g.
                             2026-07-06T13:30:00+00:00) — the Monday-after-
                             holiday case; skips the latest-trade anchor.

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

# DEEP-REFERENCE gate (Mechanism #1 — see module docstring)
DEEP_REF = os.environ.get("VALIDATE_DEEP_REF", "1") == "1"
DEEP_REF_DAYS = int(os.environ.get("VALIDATE_DEEP_REF_DAYS", "10"))
PIN_T0 = os.environ.get("VALIDATE_T0")      # e.g. 2026-07-06T13:30:00+00:00
PIN_TEND = os.environ.get("VALIDATE_TEND")  # e.g. 2026-07-06T20:00:00+00:00

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


def _mgr_formula_days(strat, warmup_bars=300):
    """Replicate prepare_strategy_window_df's FULL-path calendar-days warmup
    formula — the deepest start the manager's windowed prep would request via
    the formula — so the deep reference can be sized strictly deeper than it."""
    import math
    from data_loader import (BARS_PER_DAY, get_required_tfs_from_confluence,
                             get_tf_from_label)
    tf = strat.get('timeframe', '1Min')
    req = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec = [get_tf_from_label(lbl) for lbl in req]
    bpds = [BARS_PER_DAY.get(t, 390) for t in [tf] + sec]
    binding = min(b for b in bpds if b > 0) if bpds else 390
    return max(1, math.ceil(warmup_bars / max(binding, 0.001) * 365 / 252))


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

    if PIN_T0:
        # Pinned window (Monday-after-holiday case): validate an EXPLICIT
        # boundary instead of anchoring on the latest stored trade.
        T0 = datetime.fromisoformat(PIN_T0)
        if T0.tzinfo is None:
            T0 = T0.replace(tzinfo=timezone.utc)
        if PIN_TEND:
            T_end = datetime.fromisoformat(PIN_TEND)
            if T_end.tzinfo is None:
                T_end = T_end.replace(tzinfo=timezone.utc)
        else:
            T_end = T0 + timedelta(days=DAYS)
    else:
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
    # Steady-region cut ('entry > T0') is a STRING compare against str(entry_fill_ts),
    # which stringifies pandas-style ('2026-07-06 13:30:00+00:00', SPACE separator).
    # datetime.isoformat() uses 'T', and ' ' < 'T' — so with a T-separated cut every
    # same-date entry compared lexicographically below it and was silently dropped
    # (vacuous pass on same-day pinned windows; the legacy anchored mode dodged it
    # only because T0 lands the prior evening). Use the pandas string form so the
    # lexicographic compare is chronological.
    T0_iso = str(pd.Timestamp(T0))
    print(f"\n--- sid={sid} tf={tf} model={model}  window {T0.date()}->{T_end.date()} "
          f"({DAYS}d, {POLLS} polls{', PINNED T0=' + T0_iso if PIN_T0 else ''}) ---",
          flush=True)

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
    # DEEP-REFERENCE (Mechanism #1 gate): push the reference `since` back so its
    # left edge is >=2 TRADING days deeper than anything the manager prepares.
    # Same-edge (legacy, VALIDATE_DEEP_REF=0) is blind to warmup-window
    # truncation because both sides truncate identically. The steady-region
    # comparison (entry > T0) is unchanged — the deeper since only adds
    # converged warmup history left of T0.
    deep_days = max(DEEP_REF_DAYS, _mgr_formula_days(strat) + 7) if DEEP_REF else 0
    T_ref = T0 - timedelta(days=deep_days)
    df_ref = prepare_strategy_window_df(
        strat, T_ref, T_end, warmup_bars=300, data_feed="sip", model_override=model,
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
    ref_tag = (f"from-cold, DEEP since={T_ref.date().isoformat()}" if DEEP_REF
               else "from-cold")
    print(f"    REFERENCE ({ref_tag}) : {len(A):4d} steady trades  "
          f"{time.time()-t:5.1f}s", flush=True)

    # MANAGER — bootstrap to T0 (same left edge), then K polls re-preparing each time.
    # RESTART (Step D): if VALIDATE_RESTART_POLL=p, after poll p we DISCARD the resident
    # engine and cold re-bootstrap at that boundary — simulating a crash + restart with
    # NO snapshot. If the combined trades still equal from-cold, crash recovery via cold
    # re-bootstrap is byte-identical (no snapshot/heal needed for correctness).
    restart_poll = int(os.environ.get("VALIDATE_RESTART_POLL", "0"))
    os.environ["RORT_SECONDARY_TF_SNAPSHOT"] = mgr_snap   # manager mode (0=full, 1=fast path)
    t = time.time()
    # Record every window the manager actually prepares (left edge = first bar
    # DELIVERED) so the deep-ref depth claim is MEASURED, not assumed.
    import shadow_manager as _sm
    _mgr_left_edges = []
    _orig_prepare = _sm.prepare_window
    def _rec_prepare(*a, **kw):
        d = _orig_prepare(*a, **kw)
        try:
            if d is not None and len(d):
                _mgr_left_edges.append(d.index[0])
        except Exception:
            pass
        return d
    _sm.prepare_window = _rec_prepare
    try:
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
    finally:
        _sm.prepare_window = _orig_prepare
    M = keyed(pd.DataFrame(acc) if acc else pd.DataFrame(), after=T0_iso)
    tag = f"bootstrap+{POLLS}×" + (f", restart@{restart_poll}" if restart_poll else "")
    print(f"    MANAGER ({tag}): {len(M):4d} steady trades  {time.time()-t:5.1f}s",
          flush=True)

    # DEEP-REF PATH gate: certify the reference really sat >=2 TRADING days
    # deeper than the SHALLOWEST window the manager delivered — otherwise this
    # run cannot catch the Mechanism #1 bug class and must not count as green.
    if DEEP_REF:
        if not _mgr_left_edges:
            print("    DEEP-REF depth: no manager windows recorded — "
                  "⚠️ DEPTH-INVALID (treating as skipped)", flush=True)
            return None
        mgr_min_edge = min(_mgr_left_edges)
        ref_edge = df_ref.index[0]
        _between = df_ref.index[(df_ref.index >= ref_edge)
                                & (df_ref.index < mgr_min_edge)]
        tdays_deeper = len(_between.normalize().unique())
        depth_ok = tdays_deeper >= 2
        print(f"    DEEP-REF depth: ref_edge={ref_edge}  mgr_min_edge={mgr_min_edge}  "
              f"trading_days_deeper={tdays_deeper}  "
              f"{'✅ >=2' if depth_ok else '⚠️ DEPTH-INVALID (raise VALIDATE_DEEP_REF_DAYS)'}",
              flush=True)
        if not depth_ok:
            return None

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
