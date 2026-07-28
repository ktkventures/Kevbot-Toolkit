#!/usr/bin/env python3
"""
local_update.py — Run Update-All-Data / Update-New-Data LOCALLY, the correct way.

Mirrors the EXACT production dispatch the Railway UI uses
(`recompute_jobs.py:_run_job_worker`, the per-strategy block ~lines 300-327) so
local results are byte-identical to Railway — BUT ONLY when the local environment
matches the prod compute service. This tool no longer *assumes* that match; it
ASSERTS it (flag parity + dependency parity, board #161) and fails closed before
any write. Verified 2026-06-23: 263 local recompute = 314 trades = Railway's 314;
321 append-built lane = 529/529 identical to a full registry-correct recompute.

⚠️ BYTE-IDENTITY IS A CONDITIONAL CLAIM. Recomputes run on the **batch-worker**
Railway service, under its RORT_ flag set and its pinned pandas/numpy/pyarrow. A
local run that silently differs on either axis writes materially different code's
output to the PRODUCTION DB. On 2026-07-27 (board #161) local held only 6 of
batch-worker's 26 flags and a major-version-stale pandas, and BOTH sailed through
the smoke test — because a known-answer canary only detects the faults its
particular strategy is SENSITIVE to (sid 338 is blind to SUPPRESS_EOD_REENTRY).
So the smoke test is NOT the environment guard. The two parity assertions below
are; the smoke test is a registry / 0-trades-trap detector and behavioral
cross-check, nothing more.

WHY THIS EXISTS — the trap it makes impossible:
  An ad-hoc local script that forgets `pack_registry.scan_and_load_all()` makes
  user-pack entry triggers (e.g. ut_bot_v4) silently fire ZERO -> 0 trades. It
  looks exactly like a divergence / empty-lane bug or a "local != Railway"
  problem, but it is purely a missing init. (That caused a multi-hour false
  alarm on 2026-06-23.) The API always loads the registry at startup
  (api/main.py:103). THIS TOOL loads it the same way; the SMOKE TEST then proves
  the registry actually fired (>0 canary trades reproducing the persisted lane).

SAFETY GATES (all on by default; fail closed):
  1. FLAG PARITY — src/.env must mirror ALL of batch-worker's RORT_ flags at
     their prod values (missing / extra / wrong-value / forbidden-scheduler-flag
     all ABORT). This is the board #161 guard against the 6-of-26 drift.
  2. DEPENDENCY PARITY — the venv's pandas/numpy/pyarrow must match prod
     requirements.txt; MAJOR-version skew ABORTS (minor/patch warns loudly).
  3. Production-DB check — confirms SUPABASE_URL is the expected prod project.
  4. SMOKE TEST — read-only recompute of a fixed-window user-pack canary must be
     bit-perfect vs its persisted lane (>0 trades, no overlap mismatch), else
     ABORT. Detects the 0-trades / registry-not-loaded trap and any drift the
     canary is sensitive to. NOT a general environment guard (see above) —
     gates 1 & 2 are.
  5. Per-strategy BACKUP of all trade rows before any write (restore-able).
  6. The recompute no-wipe guard (forward_test_service.py:223-237) means a
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
  - The per-strategy dispatch in `_dispatch()` MUST stay in lock-step with
    `recompute_jobs.py:_run_job_worker`. If that file changes (new lanes, new
    args, a new mode), update `_dispatch()` here and re-run the smoke test. The 4
    imported function names must match that file's import block exactly.
  - The BATCH_WORKER_* flag manifest below is a captured SNAPSHOT of prod.
    Whenever batch-worker vars change, re-capture and bump PROD_FLAGS_CAPTURED:
      railway variables --service batch-worker | grep RORT_
    Keep src/.env in lock-step — flag parity will ABORT until you do.

USAGE (run from the src/ directory):
  python local_update.py --mode new  --strategies 321,308
  python local_update.py --mode all  --strategies 263
  python local_update.py --mode new  --strategies 321 --force
  Flags: --no-backup, --skip-smoke (only for trusted repeat runs in one session),
         --skip-env-check (DANGER: bypass flag+dependency parity — deliberate,
         understood deviations only), --canary <sid> (default 321),
         --user <uuid>, --force-db (override prod check)
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

# --- prod batch-worker RORT_ flag manifest (board #161) ----------------------
# Authoritative SNAPSHOT of the RORT_ flags on the **batch-worker** Railway
# service — the service that actually runs recomputes (api enqueues to it via
# RORT_COMPUTE_REMOTE=1). A local recompute must reproduce THIS environment, not
# the Worker (live engine) one. `src/.env` is the local mirror of this manifest;
# _assert_flag_parity() fails CLOSED on any drift between the two.
#
# Captured 2026-07-27 (E3, board #159) via:
#     railway variables --service batch-worker | grep RORT_
# ⚠️ RE-CAPTURE whenever batch-worker vars change and bump PROD_FLAGS_CAPTURED;
#    flag parity ABORTS until src/.env is brought back in lock-step.
PROD_FLAGS_CAPTURED = "2026-07-27"

# MIRROR: present on batch-worker AND required in .env at EXACTLY this value.
BATCH_WORKER_MIRROR_FLAGS = {
    "RORT_APPEND_SUPERSEDE_OPEN": "1",
    "RORT_CANONICAL_SUBMIN_STATE": "1",
    "RORT_COARSE_SECONDARY_FROM_1MIN": "1",
    "RORT_ENFORCE_1MIN_GATE": "1",
    "RORT_HIFI_NO_PREBAR_FILL": "1",
    "RORT_PRIME1S_CACHE_MAX_ROWS": "2000000",
    "RORT_RESAMPLED_STORE_READ": "1",
    "RORT_RESAMPLED_STORE_SERVE": "1",
    "RORT_RESUME_INHERIT_POSITION": "1",
    "RORT_RIGHTSIZE_WARMUP": "1",
    "RORT_SCOPE_CONFLUENCE_GROUPS": "1",
    "RORT_SECONDARY_TF_CACHE": "1",
    "RORT_SECONDARY_TF_SNAPSHOT": "1",
    "RORT_SUPPRESS_EOD_REENTRY": "1",
    "RORT_UAD_GUARD_EMPTY_LANE": "1",
    "RORT_UPDATE_ALL_SKIP_ALGO": "1",
    "RORT_USE_DOUGH_CACHE": "1",
    "RORT_NIGHTLY_SETTLE_RETRUE": "1",
    "RORT_NIGHTLY_SETTLE_RETRUE_WINDOW": "5",
    "RORT_PARITY_SELF_HEAL": "1",
    "RORT_PARITY_SETTLED_MIN_TDAYS": "4",
}
# OVERRIDE: on batch-worker at prod_value, but the CORRECT local value differs
# (documented deviation). name -> (expected_local_value, prod_value, reason)
BATCH_WORKER_OVERRIDE_FLAGS = {
    "RORT_COMPUTE_REMOTE": ("0", "1",
        "batch-worker enqueues to the remote compute service; local_update calls "
        "forward_test_service directly, so the job store is bypassed."),
}
# OMIT: on batch-worker but MUST be ABSENT locally — scheduler/orchestration
# only; setting them on a workstation is actively wrong. name -> reason
BATCH_WORKER_OMIT_FLAGS = {
    "RORT_NIGHTLY_RECOMPUTE": "nightly scheduler trigger — would fire recomputes on a workstation",
    "RORT_NIGHTLY_RECOMPUTE_AT": "nightly scheduler time",
    "RORT_RECOMPUTE_PARALLELISM": "remote-worker pool sizing",
    "RORT_RECOMPUTE_SUPERVISOR": "remote-worker supervisor toggle",
}
# Total batch-worker RORT_ flags this manifest accounts for (drift tripwire — 26).
BATCH_WORKER_FLAG_COUNT = (len(BATCH_WORKER_MIRROR_FLAGS)
                           + len(BATCH_WORKER_OVERRIDE_FLAGS)
                           + len(BATCH_WORKER_OMIT_FLAGS))

# Packages whose version changes can shift numeric trade results (byte-identity
# risk). Checked against the prod-pinned requirements.txt.
NUMERIC_PACKAGES = ("pandas", "numpy", "pyarrow")


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(msg):
    print(f"[{_ts()}] {msg}", flush=True)


def _here(*parts):
    """Resolve a path relative to this file (src/), CWD-independent."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _find_env_path():
    for p in (_here(".env"), ".env"):
        if os.path.exists(p):
            return p
    return _here(".env")


def _find_requirements_path():
    for p in (_here("..", "requirements.txt"), "../requirements.txt", "requirements.txt"):
        if os.path.exists(p):
            return p
    return None


def _parse_env_rort_flags(env_path):
    """RORT_ flags as written in the .env FILE (not os.environ), so the mirror
    block is validated as authored, independent of any shell exports."""
    from dotenv import dotenv_values
    return {k: v for k, v in dotenv_values(env_path).items()
            if k.startswith("RORT_") and v is not None}


def _assert_flag_parity():
    """GATE 1 — fail CLOSED unless src/.env mirrors the batch-worker RORT_ flag
    set the prod recompute runs under. Catches the four drift classes behind
    board #161: MISSING (under-mirror, the 6-of-26 bug), WRONG value, STRAY
    (an .env flag prod doesn't have), and FORBIDDEN (a scheduler-only flag that
    must never be set on a workstation). Also checks the EFFECTIVE environment so
    a shell-exported flag can't smuggle a deviation past the file check."""
    env_path = _find_env_path()
    env_flags = _parse_env_rort_flags(env_path)
    problems = []

    # MIRROR + OVERRIDE must be present in .env at the expected value.
    expected = dict(BATCH_WORKER_MIRROR_FLAGS)
    for name, (local_val, _prod, _why) in BATCH_WORKER_OVERRIDE_FLAGS.items():
        expected[name] = local_val
    for name, want in expected.items():
        got = env_flags.get(name)
        if got is None:
            problems.append(f"MISSING   {name} (wants {want!r}) — .env under-mirrors prod")
        elif str(got) != str(want):
            problems.append(f"WRONG     {name}={got!r} but manifest wants {want!r}")

    # OMIT must be absent from .env AND from the effective environment.
    for name, why in BATCH_WORKER_OMIT_FLAGS.items():
        if name in env_flags:
            problems.append(f"FORBIDDEN {name} present in .env — {why}")
        if name in os.environ:
            problems.append(f"FORBIDDEN {name} present in environment — {why}")

    # STRAY: any RORT_ in .env the manifest doesn't know about.
    known = set(expected) | set(BATCH_WORKER_OMIT_FLAGS)
    for name in sorted(set(env_flags) - known):
        problems.append(f"STRAY     {name}={env_flags[name]!r} not in the batch-worker "
                        f"manifest (new prod flag? update the manifest — or remove from .env)")

    n = BATCH_WORKER_FLAG_COUNT
    if problems:
        detail = "\n  ".join(problems)
        sys.exit(f"ABORT flag parity: local .env does NOT match the {n}-flag batch-worker "
                 f"manifest (captured {PROD_FLAGS_CAPTURED}). A recompute here would run "
                 f"different code than prod and persist it to the PROD DB. Resolve:\n  "
                 f"{detail}\n"
                 f"Re-diff: railway variables --service batch-worker | grep RORT_")
    log(f"FLAG PARITY OK ✅ — .env mirrors all {n} batch-worker RORT_ flags "
        f"({len(expected)} required, {len(BATCH_WORKER_OMIT_FLAGS)} correctly omitted; "
        f"manifest captured {PROD_FLAGS_CAPTURED}).")


def _major(v):
    try:
        return int(str(v).split(".", 1)[0])
    except (ValueError, AttributeError):
        return -1


def _parse_requirements_pins(req_path, packages):
    """{package: (op, version)} for `packages` from requirements.txt."""
    import re
    want = {p.lower() for p in packages}
    pins = {}
    with open(req_path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|~=|>=|>)\s*([0-9][^\s,;]*)", line)
            if not m:
                continue
            name = m.group(1).lower()
            if name in want:
                pins[name] = (m.group(2), m.group(3))
    return pins


def _assert_dependency_parity():
    """GATE 2 — fail CLOSED on MAJOR-version skew (loud WARN on minor/patch) for
    the packages whose version changes can shift numeric trade results. prod's
    requirements.txt is the pinned truth; the local venv must not silently
    diverge (board #161: venv pandas 2.3.3 vs prod 3.0.3 wrote off-version
    numerics while the smoke test passed green)."""
    import importlib.metadata as im
    reqfile = _find_requirements_path()
    if reqfile is None:
        sys.exit("ABORT dependency parity: could not locate requirements.txt to diff against.")
    pins = _parse_requirements_pins(reqfile, NUMERIC_PACKAGES)
    fails, warns, rows = [], [], []
    for pkg in NUMERIC_PACKAGES:
        try:
            installed = im.version(pkg)
        except im.PackageNotFoundError:
            fails.append(f"{pkg}: NOT INSTALLED in this venv (prod pins {pins.get(pkg)})")
            continue
        pin = pins.get(pkg)
        if pin is None:
            warns.append(f"{pkg}=={installed}: no pin in requirements.txt — cannot verify")
            continue
        op, want = pin
        rows.append(f"{pkg} {installed} vs {op}{want}")
        if op == "==" and installed != want:
            if _major(installed) != _major(want):
                fails.append(f"{pkg}: MAJOR skew installed={installed} vs prod pin =={want}")
            else:
                warns.append(f"{pkg}: minor/patch skew installed={installed} vs prod pin =={want}")
        elif op != "==" and _major(installed) < _major(want):
            fails.append(f"{pkg}: installed major {installed} below prod floor {op}{want}")
    log("DEPENDENCY PARITY: " + (" | ".join(rows) if rows else "(no rows parsed)"))
    for w in warns:
        log(f"   ⚠️ dep warn: {w}")
    if fails:
        detail = "\n  ".join(fails)
        sys.exit(f"ABORT dependency parity: local venv diverges from prod requirements.txt on "
                 f"numeric-affecting packages — trade results may NOT be byte-identical:\n  "
                 f"{detail}\n"
                 f"Fix: align the venv to the pinned versions (pip install -r {reqfile}).")
    log(f"DEPENDENCY PARITY OK ✅ — {', '.join(NUMERIC_PACKAGES)} match prod major versions.")


def _preflight(user, force_db, skip_env_check=False):
    """Set up env, admin context, and load packs EXACTLY like the API does."""
    os.environ.setdefault("RORT_SECONDARY_TF_SNAPSHOT", "1")  # match Railway api
    os.environ["USE_DB"] = "true"
    from dotenv import load_dotenv
    load_dotenv(".env", override=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # GATES 1 & 2 — environment parity (board #161). These, NOT the smoke test,
    # are the guard that a local recompute runs prod's code against prod's deps.
    if skip_env_check:
        log("⚠️  ENV PARITY CHECKS SKIPPED (--skip-env-check) — flag & dependency "
            "parity NOT verified; only safe for deliberate, understood deviations.")
    else:
        _assert_flag_parity()
        _assert_dependency_parity()

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
    """GATE 4 — read-only: recompute the canary and require it to be bit-perfect
    vs its persisted lane over the overlapping window. ABORT if the registry/
    packs did not load (the 0-trades trap) or the canary drifts.

    SCOPE (board #161): this is a REGISTRY / 0-trades-trap detector and a
    behavioral cross-check — NOT a general environment guard. A known-answer
    canary only detects the faults its own strategy is SENSITIVE to (sid 338 is
    blind to SUPPRESS_EOD_REENTRY, so a 20-flag mismatch once passed green here).
    Environment correctness is asserted directly by _assert_flag_parity (GATE 1)
    and _assert_dependency_parity (GATE 2); do not lean on this test for it."""
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
    log("SMOKE TEST PASSED ✅ — canary reproduced its persisted lane (registry/packs "
        "loaded, 0-trades trap clear). Environment parity is proven by GATES 1 & 2, "
        "not this test.")


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
    ap.add_argument("--skip-smoke", action="store_true", help="skip the behavioral smoke test (trusted repeat runs only)")
    ap.add_argument("--skip-env-check", action="store_true",
                    help="DANGER: bypass flag + dependency parity gates (deliberate deviations only)")
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
    _preflight(args.user, args.force_db, args.skip_env_check)

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
