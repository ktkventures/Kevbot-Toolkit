#!/usr/bin/env python3
"""
local_update.py — Run Update-All-Data / Update-New-Data LOCALLY, the correct way.

Mirrors the EXACT production dispatch the Railway UI uses
(`recompute_jobs.py:_run_job_worker`, the per-strategy block ~lines 300-327) so
local results are byte-identical to Railway. Verified 2026-06-23:
  - 263 local recompute = 314 trades = Railway's 314.
  - 321 append-built lane vs full registry-correct recompute = 529/529 entries
    identical to the second.

WHY THIS EXISTS — the trap it makes impossible:
  An ad-hoc local script that forgets `pack_registry.scan_and_load_all()` makes
  user-pack entry triggers (e.g. ut_bot_v4) silently fire ZERO -> 0 trades. It
  looks exactly like a divergence / empty-lane bug or a "local != Railway"
  problem, but it is purely a missing init. (That caused a multi-hour false
  alarm on 2026-06-23.) The API always loads the registry at startup
  (api/main.py:103). THIS TOOL loads it the same way AND refuses to write a
  single trade until a known-answer SMOKE TEST proves the environment is
  correct.

SAFETY GATES (all on by default):
  1. Production-DB check — confirms SUPABASE_URL is the expected prod project.
  2. SMOKE TEST — read-only recompute of a fixed-window user-pack canary must be
     bit-perfect vs its persisted lane (>0 trades, no overlap mismatch), else
     ABORT before any write. This is what makes the 0-trades trap impossible.
  3. Per-strategy BACKUP of all trade rows before any write (restore-able).
  4. The recompute no-wipe guard (forward_test_service.py:223-237) means a
     0-trade result NEVER deletes a lane — but a 0 on a user-pack/normal
     strategy is reported LOUDLY as suspicious.

CAVEATS the tool enforces / prints:
  - Strategies with NO fixed `backtest_start_date` (Bucket A canaries, e.g. 263)
    window to "now - N days" -> their recompute floats with wall-clock and is
    NOT reproducible run-to-run. The tool flags them; never use them as a
    bit-perfect reference.
  - Update-New (append) lands the backtest lane ~15 min behind live alerts by
    design (Polygon REST finalization). Anything inside that window is TBD, not
    phantom.

MAINTENANCE (future-proofing — READ BEFORE EDITING):
  The per-strategy dispatch in `_dispatch()` MUST stay in lock-step with
  `recompute_jobs.py:_run_job_worker`. If that file changes (new lanes, new
  args, a new mode), update `_dispatch()` here and re-run the smoke test. The 4
  imported function names must match that file's import block exactly.

USAGE (run from the src/ directory):
  python local_update.py --mode new  --strategies 321,308
  python local_update.py --mode all  --strategies 263
  python local_update.py --mode new  --strategies 321 --force
  Flags: --no-backup, --skip-smoke (only for trusted repeat runs in one session),
         --canary <sid> (default 321), --user <uuid>, --force-db (override prod check)
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# --- constants ---------------------------------------------------------------
PROD_SUPABASE_HOST = "lxgrvnzfarampvbgghft"          # the production project ref
DEFAULT_USER = "19d47e46-f718-49a6-af32-5f5407f5b170"  # Kevin (ktkventures)
DEFAULT_CANARY = 321  # fixed-window (Jun-15), ut_bot_v4 user-pack, known-good lane
HIFI_NOTE = "Update-New runs Hi-Fi internally (append_new_backtest_trades_for_strategy)."


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(msg):
    print(f"[{_ts()}] {msg}", flush=True)


def _preflight(user, force_db):
    """Set up env, admin context, and load packs EXACTLY like the API does."""
    os.environ.setdefault("RORT_SECONDARY_TF_SNAPSHOT", "1")  # match Railway api
    os.environ["USE_DB"] = "true"
    from dotenv import load_dotenv
    load_dotenv(".env", override=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    url = os.getenv("SUPABASE_URL", "")
    is_prod = PROD_SUPABASE_HOST in url
    log(f"SUPABASE_URL = {url}  ({'PRODUCTION' if is_prod else 'NON-PROD'})")
    if not is_prod and not force_db:
        sys.exit("ABORT: not pointed at the production DB (use --force-db to override).")

    import db
    db.set_admin_user_context(user)
    log(f"admin user context set: {user}")

    # Load the user-pack registry exactly like api/main.py:103.
    import pack_registry
    n = pack_registry.scan_and_load_all()
    log(f"pack_registry.scan_and_load_all() -> {n} packs loaded")
    return is_prod


def _entries_to_sec(rows_or_df):
    """Normalize a list[dict] or DataFrame of trades to a sorted set of
    entry_fill_ts truncated to the second (string compare, tz-naive-safe)."""
    import pandas as pd
    if hasattr(rows_or_df, "columns"):  # DataFrame
        vals = rows_or_df["entry_fill_ts"].tolist() if "entry_fill_ts" in rows_or_df.columns else []
    else:
        vals = [r.get("entry_fill_ts") for r in rows_or_df]
    return sorted(str(v)[:19].replace("T", " ") for v in vals if v is not None)


def _smoke_test(user, canary):
    """Read-only: recompute the canary and require it to be bit-perfect vs its
    persisted lane over the overlapping window. ABORT if the environment is not
    correctly initialized (this is the 0-trades trap detector)."""
    from db import get_admin_client, get_strategy_by_id_admin
    import services as svc
    c = get_admin_client()
    lane = (c.table("trades").select("entry_fill_ts").eq("strategy_id", canary)
            .like("data_source", "backtest_%").order("entry_fill_ts").execute().data or [])
    if not lane:
        sys.exit(f"ABORT smoke test: canary {canary} has no backtest lane to verify against. "
                 f"Pick a --canary with a known-good fixed-window lane.")
    lane_secs = _entries_to_sec(lane)
    # 2026-07-16: verify on a HISTORICAL window only. The canary itself is
    # live-trading, so its NEWEST trades (appended intra/near-session) never
    # bit-match a clean recompute until they settle — that raced the gate
    # during/after RTH (07-15 forward-check aborted on lane_only=4 despite a
    # 99.8% historical match). Trades newer than SMOKE_HISTORY_CUTOFF_H
    # (default 24h) are excluded from BOTH sides; the gate still proves
    # registry/packs/flags correctness — the 0-trades trap and any HISTORICAL
    # mismatch still abort.
    from datetime import datetime, timedelta, timezone
    _cut_h = float(os.environ.get("SMOKE_HISTORY_CUTOFF_H", "24"))
    _cutoff = (datetime.now(timezone.utc) - timedelta(hours=_cut_h)
               ).strftime("%Y-%m-%d %H:%M:%S")
    hist_lane = [x for x in lane_secs if x <= _cutoff]
    if not hist_lane:
        sys.exit(f"ABORT smoke test: canary {canary} has no lane trades older "
                 f"than the {_cut_h:.0f}h historical cutoff — pick a canary "
                 f"with settled history.")
    last = hist_lane[-1]
    strat = get_strategy_by_id_admin(canary, user)
    t = time.time()
    tr = svc.get_strategy_trades(strat)
    dt = time.time() - t
    n = len(tr) if tr is not None else 0
    if n == 0:
        sys.exit(f"ABORT smoke test: canary {canary} recompute produced 0 trades. "
                 f"Environment is NOT correctly initialized (registry/packs not loaded). "
                 f"DO NOT trust any write from this process.")
    rec = _entries_to_sec(tr)
    overlap = [x for x in rec if x <= last]
    A, B = set(hist_lane), set(overlap)
    lane_only, rec_only = A - B, B - A
    ok = not lane_only and not rec_only
    log(f"SMOKE TEST canary {canary}: hist_lane={len(hist_lane)}/{len(lane_secs)} "
        f"(cutoff {_cut_h:.0f}h) recompute={n} "
        f"overlap_match={len(A & B)}/{len(hist_lane)} lane_only={len(lane_only)} "
        f"rec_only={len(rec_only)}  [{dt:.0f}s]")
    if not ok:
        sys.exit(f"ABORT smoke test: canary {canary} lane does NOT match a registry-correct "
                 f"recompute (lane_only={len(lane_only)}, rec_only={len(rec_only)}). "
                 f"Resolve before writing.")
    log("SMOKE TEST PASSED ✅ — environment matches Railway; safe to proceed.")


def _backup(user, sid, backup_dir):
    from db import get_admin_client
    c = get_admin_client()
    rows = c.table("trades").select("*").eq("strategy_id", sid).execute().data or []
    path = os.path.join(backup_dir, f"backup_sid{sid}_{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(rows, f, default=str)
    bt = sum(1 for r in rows if (r.get("data_source") or "").startswith("backtest_"))
    return path, len(rows), bt


def _count(sid, prefix):
    from db import get_admin_client
    c = get_admin_client()
    return (c.table("trades").select("id", count="exact").eq("strategy_id", sid)
            .like("data_source", prefix + "%").limit(1).execute().count or 0)


def _dispatch(sid, user, mode, force):
    """MIRROR of recompute_jobs.py:_run_job_worker per-strategy block.
    Keep in lock-step with that file. Returns (bt_result, algo_result)."""
    from api.services.forward_test_service import (
        append_new_trades_for_strategy,
        recompute_and_persist_stored_trades,
        append_new_backtest_trades_for_strategy,
        recompute_and_persist_algo_trades,
    )
    from recompute_jobs import _get_strategy_lock
    with _get_strategy_lock(sid):
        if mode == "new":   # == job_type 'append_recent'
            try:
                bt = append_new_backtest_trades_for_strategy(sid, user, force=force)
            except Exception as e:
                bt = {"status": "error", "reason": str(e)}
            try:
                algo = append_new_trades_for_strategy(sid, user, force=force)
            except Exception as e:
                algo = {"status": "error", "reason": str(e)}
        else:               # mode == 'all' == job_type 'full_recompute'
            try:
                bt = recompute_and_persist_stored_trades(sid, user, compute_parity=False)
            except Exception as e:
                bt = {"status": "error", "reason": str(e)}
            try:
                algo = recompute_and_persist_algo_trades(sid, user)
            except Exception as e:
                algo = {"status": "error", "reason": str(e)}
    return bt, algo


def main():
    ap = argparse.ArgumentParser(description="Run Update-All / Update-New locally, correctly.")
    ap.add_argument("--mode", choices=["all", "new"], required=True)
    ap.add_argument("--strategies", required=True, help="comma-separated strategy ids")
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--force", action="store_true", help="force append even if throttled (mode=new)")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true", help="skip the self-verify smoke test (trusted repeat runs only)")
    ap.add_argument("--canary", type=int, default=DEFAULT_CANARY)
    ap.add_argument("--force-db", action="store_true", help="override the production-DB check")
    ap.add_argument("--backup-dir", default=None)
    args = ap.parse_args()

    sids = [int(x) for x in args.strategies.split(",") if x.strip()]
    backup_dir = args.backup_dir or os.path.join(os.getcwd(), "_local_update_backups")
    os.makedirs(backup_dir, exist_ok=True)

    log(f"=== local_update  mode={args.mode}  strategies={sids}  force={args.force} ===")
    if args.mode == "new":
        log(HIFI_NOTE)
    _preflight(args.user, args.force_db)

    if args.skip_smoke:
        log("⚠️  SMOKE TEST SKIPPED (--skip-smoke) — only safe if a prior run this session passed.")
    else:
        _smoke_test(args.user, args.canary)

    from db import get_strategy_by_id_admin
    results = []
    for sid in sids:
        strat = get_strategy_by_id_admin(sid, args.user)
        if strat is None:
            log(f"sid {sid}: NOT FOUND for user — skip"); continue
        bt_start = (strat.get("config") or {}).get("backtest_start_date") or strat.get("backtest_start_date")
        floating = bt_start is None
        if floating:
            log(f"sid {sid}: ⚠️ no fixed backtest_start_date — window FLOATS with now (non-reproducible).")
        bt0, algo0 = _count(sid, "backtest_"), _count(sid, "cache_")
        backup_path = None
        if not args.no_backup:
            backup_path, total, bt = _backup(args.user, sid, backup_dir)
            log(f"sid {sid}: backup -> {os.path.basename(backup_path)} ({total} rows, {bt} backtest)")

        t = time.time()
        bt_r, algo_r = _dispatch(sid, args.user, args.mode, args.force)
        elapsed = time.time() - t
        bt1, algo1 = _count(sid, "backtest_"), _count(sid, "cache_")

        bt_status = bt_r.get("status"); algo_status = algo_r.get("status")
        log(f"sid {sid}: backtest[{bt_status}] {bt0}->{bt1} ({bt1-bt0:+d}) | "
            f"algo[{algo_status}] {algo0}->{algo1} ({algo1-algo0:+d})  [{elapsed:.0f}s]")
        # loud flags
        if bt1 < bt0 or algo1 < algo0:
            log(f"   ⚠️⚠️ COUNT DROPPED for sid {sid} — inspect backup {backup_path}")
        if args.mode == "all" and bt_status == "no_trades" and not floating:
            log(f"   ⚠️ recompute produced 0 trades on a fixed-window strategy — "
                f"suspect coarse-gate warmup (Bucket B) or config; lane preserved by no-wipe guard.")
        results.append({"sid": sid, "bt": bt_status, "algo": algo_status,
                        "bt_delta": bt1 - bt0, "algo_delta": algo1 - algo0, "elapsed_s": round(elapsed, 1)})

    log("=== SUMMARY ===")
    for r in results:
        log(f"  sid {r['sid']}: bt={r['bt']}({r['bt_delta']:+d}) algo={r['algo']}({r['algo_delta']:+d}) {r['elapsed_s']}s")
    log("Done. (Update-New lands the lane ~15 min behind live by REST design — inside that window is TBD, not phantom.)")


if __name__ == "__main__":
    main()
