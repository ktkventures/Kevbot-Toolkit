"""M-RS1 byte-identical gate — right-sized warmup vs legacy ×2.

Read-only. Runs `services.get_strategy_trades(strat)` twice per strategy —
once with RORT_RIGHTSIZE_WARMUP=0 (legacy `visible_days*2`) and once with
=1 (per-TF right-sized) — and asserts the VISIBLE-WINDOW trades are identical
(entry ts, exit ts, exec_type to the second). Also reports wall-time so we can
quantify the speedup.

Registry-safe (calls scan_and_load_all per feedback_local_script_pack_registry)
and writes NOTHING to the DB. See Plan_M-RS1_Warmup_Rightsizing.md.

Interpretation note: a 1Day/1Week-gated strategy may show a diff because the
LEGACY path *under*-warmed daily gates (visible_days*2 < 250 daily bars). That
is a CORRECTNESS FIX, not a regression — flagged separately below.

Usage:
  .venv/bin/python src/_compare_rightsize_warmup.py --user <uuid> --sids 325 309 313
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _norm_trades(df):
    """Normalize a trades DataFrame to a sorted set of
    (entry_sec, exit_sec, exec_type) tuples — second-truncated, tz-safe."""
    if df is None or len(df) == 0:
        return set()

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    e = col("entry_fill_ts", "entry_time", "entry_trigger_ts")
    x = col("exit_fill_ts", "exit_time", "exit_trigger_ts")
    xt = col("exit_exec_type", "entry_exec_type", "exec_type")

    def sec(v):
        return str(v)[:19].replace("T", " ") if v is not None else ""

    out = set()
    for _, r in df.iterrows():
        out.add((
            sec(r.get(e)) if e else "",
            sec(r.get(x)) if x else "",
            str(r.get(xt)) if xt else "",
        ))
    return out


def _run(strat, flag, label):
    import services as svc
    os.environ["RORT_RIGHTSIZE_WARMUP"] = flag
    sid = strat.get("id")
    print(f"  [sid {sid}] {label} run START (flag={flag})...", flush=True)
    t = time.time()
    tr = svc.get_strategy_trades(strat)
    dt = time.time() - t
    n = len(tr) if tr is not None else 0
    print(f"  [sid {sid}] {label} run DONE: {n} trades in {dt:.0f}s", flush=True)
    return tr, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user",
                    default=os.getenv("ROR_ADMIN_USER",
                                      "19d47e46-f718-49a6-af32-5f5407f5b170"),
                    help="admin user_id (uuid); default = Kevin (local_update.DEFAULT_USER)")
    ap.add_argument("--sids", type=int, nargs="+", required=True)
    ap.add_argument("--pin-days", type=int, default=0,
                    help="Pin BOTH runs to a recent N-calendar-day visible "
                         "window (lookback_mode='Date Range') for a fast, "
                         "window-agnostic byte-identical signal. 0 = real config.")
    ap.add_argument("--settle-min", type=int, default=30,
                    help="Exclude trades entered within the last N minutes from "
                         "the compare (right-edge unsettled-tip is fetch-timing, "
                         "not warmup). 0 = include the live tip.")
    args = ap.parse_args()
    if not args.user:
        sys.exit("ERROR: --user <uuid> required (or set ROR_ADMIN_USER).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    # Enable DB + load creds BEFORE importing db (USE_DB is read at import
    # time). Keep the secondary-TF snapshot OFF: seeding it writes the
    # strategy config (services._secondary_snapshot_persist), and this gate
    # must stay strictly read-only. get_strategy_trades (full path) doesn't
    # read the snapshot anyway, so OFF is correct for both runs.
    os.environ["USE_DB"] = "true"
    os.environ["RORT_SECONDARY_TF_SNAPSHOT"] = "0"
    from dotenv import load_dotenv
    load_dotenv(os.path.join(script_dir, ".env"), override=True)
    import db
    import pack_registry
    db.set_admin_user_context(args.user)
    n = pack_registry.scan_and_load_all()
    n_packs = len(n) if hasattr(n, "__len__") else n
    print(f"[init] registry: {n_packs} packs loaded; user={args.user}")

    from db import get_strategy_by_id_admin

    any_fail = False
    for sid in args.sids:
        strat = get_strategy_by_id_admin(sid, args.user)
        if not strat:
            print(f"[sid {sid}] NOT FOUND — skipping")
            continue
        tf = strat.get("timeframe")
        conf = strat.get("confluence", []) or []
        is_daily_gate = any(str(c).startswith(("1d-", "1D-", "1w-", "1W-"))
                            for c in conf)

        if args.pin_days > 0:
            import datetime as _dt
            end = _dt.datetime.now(_dt.timezone.utc)
            start = end - _dt.timedelta(days=args.pin_days)
            strat["lookback_mode"] = "Date Range"
            strat["lookback_start_date"] = start.isoformat()
            strat["lookback_end_date"] = end.isoformat()
            # Report the warmup each path will use on this pinned window so a
            # rightsized>legacy "flip" (coarse-gate on a short window) is visible.
            import strategy_data as _sd
            os.environ["RORT_RIGHTSIZE_WARMUP"] = "0"
            _leg = _sd.compute_warmup_days(strat, float(args.pin_days))
            os.environ["RORT_RIGHTSIZE_WARMUP"] = "1"
            _rs = _sd.compute_warmup_days(strat, float(args.pin_days))
            flip = " (FLIP: rightsized loads MORE — coarse gate on short window)" \
                if _rs > _leg else ""
            print(f"[sid {sid}] PINNED {args.pin_days}d window | "
                  f"legacy_warmup={_leg:.0f}d rightsized_warmup={_rs:.0f}d{flip}",
                  flush=True)

        print(f"[sid {sid}] tf={tf} conf={len(conf)} — comparing...", flush=True)
        rs_tr, rs_dt = _run(strat, "1", "rightsized")   # fast run first
        legacy_tr, legacy_dt = _run(strat, "0", "legacy")  # slow baseline

        A = _norm_trades(legacy_tr)
        B = _norm_trades(rs_tr)
        # Right-edge settle margin: the two runs fetch "now" minutes apart, so
        # the unsettled tip differs by fetch-timing — NOT by warmup. Exclude
        # entries newer than (now - settle_min) for an apples-to-apples compare
        # (the project's *_fair convention). Off with --settle-min 0.
        if args.settle_min > 0:
            import datetime as _dt2
            cutoff = (_dt2.datetime.now(_dt2.timezone.utc)
                      - _dt2.timedelta(minutes=args.settle_min)
                      ).strftime("%Y-%m-%d %H:%M:%S")
            A = {t for t in A if t[0] and t[0] <= cutoff}
            B = {t for t in B if t[0] and t[0] <= cutoff}
        legacy_only = A - B
        rs_only = B - A
        ident = not legacy_only and not rs_only
        speedup = (legacy_dt / rs_dt) if rs_dt > 0 else 0.0

        tag = "✅ IDENTICAL" if ident else (
            "⚠️ DIFF (daily-gate under-warm fix — expected)" if is_daily_gate
            else "❌ DIFF — INVESTIGATE")
        if not ident and not is_daily_gate:
            any_fail = True

        print(f"[sid {sid}] tf={tf} conf={len(conf)} | "
              f"legacy={len(A)}tr/{legacy_dt:.0f}s  rightsized={len(B)}tr/{rs_dt:.0f}s  "
              f"({speedup:.1f}x faster) | legacy_only={len(legacy_only)} "
              f"rs_only={len(rs_only)} | {tag}")
        if legacy_only or rs_only:
            vstart = min((t[0] for t in (A | B) if t[0]), default="?")
            vend = max((t[0] for t in (A | B) if t[0]), default="?")
            print(f"    [diff] visible entries span {vstart} .. {vend}", flush=True)
            for t in sorted(legacy_only)[:10]:
                print(f"    [legacy_only] entry={t[0]} exit={t[1]} exec={t[2]}",
                      flush=True)
            for t in sorted(rs_only)[:10]:
                print(f"    [rs_only]     entry={t[0]} exit={t[1]} exec={t[2]}",
                      flush=True)

    print("\nRESULT:", "❌ at least one non-daily diff — DO NOT flip default"
          if any_fail else "✅ all non-daily-gate strategies byte-identical")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
