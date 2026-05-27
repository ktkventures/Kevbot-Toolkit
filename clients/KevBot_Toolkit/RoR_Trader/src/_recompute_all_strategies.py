#!/usr/bin/env python3
"""Bulk recompute every strategy in DB — applies today's fidelity fixes
(unified warmup helper c1869a5, Hi-Fi exit-timestamp fix 362c845,
rest_hifi default 1f7d0ea) to every strategy's existing trades.

The data-worker's incremental catchup path doesn't rewrite existing
trades — it only appends new ones. So bar-aligned pre-fix trades stay
in the DB until manually re-recomputed. This script does that for
every strategy in one pass.

Calls `recompute_and_persist_stored_trades(sid, user_id)` directly via
the admin DB client (no HTTP roundtrip). Sequential — one strategy at
a time with a 5s pause between to avoid Supabase pressure.
`compute_parity=False` so parity checks don't add latency; user can
trigger those individually if needed.

Usage:
    cd src && USE_DB=true ../.venv/bin/python _recompute_all_strategies.py

Options:
    --dry-run        — list strategies, don't recompute
    --sleep N        — pause N seconds between strategies (default 5)
    --skip-existing  — skip strategies updated in the past hour
    --only SID,SID   — only recompute these strategy IDs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("USE_DB", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import get_admin_client, set_admin_user_context
from api.services.forward_test_service import (
    recompute_and_persist_stored_trades,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List strategies, don't recompute.")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="Seconds to pause between strategies. Default 5.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip strategies whose kpis_computed_at is within "
                          "the past hour (already fresh).")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated strategy IDs to recompute. "
                          "Default: all.")
    args = ap.parse_args()

    only_ids = None
    if args.only:
        try:
            only_ids = {int(s.strip()) for s in args.only.split(",")
                        if s.strip()}
        except ValueError:
            print(f"Invalid --only value: {args.only!r}", flush=True)
            return 1

    c = get_admin_client()
    print(f"[{_now_iso()}] Loading strategies from DB...", flush=True)
    rows = (c.table("strategies")
            .select("id,user_id,name,symbol,timeframe,"
                    "kpis_computed_at,strategy_origin")
            .order("id")
            .execute().data or [])

    # Filter
    targets = []
    skipped = []
    now = datetime.now(timezone.utc)
    for r in rows:
        sid = r.get("id")
        if sid is None:
            continue
        if only_ids is not None and sid not in only_ids:
            continue
        # Webhook strategies have no engine to recompute against.
        if r.get("strategy_origin") == "webhook_inbound":
            skipped.append((sid, "webhook_inbound"))
            continue
        if args.skip_existing and r.get("kpis_computed_at"):
            try:
                ts = datetime.fromisoformat(
                    r["kpis_computed_at"].replace("Z", "+00:00"))
                if (now - ts).total_seconds() < 3600:
                    skipped.append((sid, "fresh < 1h"))
                    continue
            except (ValueError, TypeError):
                pass
        targets.append(r)

    print(f"[{_now_iso()}] {len(targets)} strategies to recompute, "
          f"{len(skipped)} skipped (webhook_inbound or fresh<1h).",
          flush=True)
    if skipped and args.dry_run:
        print("  Skipped detail:", flush=True)
        for sid, reason in skipped[:20]:
            print(f"    sid={sid}: {reason}", flush=True)

    if args.dry_run:
        print("[dry-run] Would recompute:", flush=True)
        for r in targets:
            print(f"  sid={r['id']:5}  {r.get('symbol')}/{r.get('timeframe')}"
                  f"  '{r.get('name', '?')}'", flush=True)
        return 0

    if not targets:
        print(f"[{_now_iso()}] Nothing to recompute.", flush=True)
        return 0

    ok_count = 0
    fail_count = 0
    no_trades = 0
    started = time.monotonic()
    last_user_id = None

    for i, r in enumerate(targets, start=1):
        sid = r["id"]
        user_id = r["user_id"]
        name = r.get("name", "?")
        sym = r.get("symbol", "?")
        tf = r.get("timeframe", "?")

        # Set the admin user context once per user (not per-strategy —
        # most are the same user). The engine needs this so
        # confluence_groups + user packs load correctly.
        if user_id != last_user_id:
            set_admin_user_context(user_id)
            last_user_id = user_id

        t0 = time.monotonic()
        try:
            res = recompute_and_persist_stored_trades(
                sid, user_id, compute_parity=False)
            elapsed = round(time.monotonic() - t0, 1)
            status = res.get("status", "?")
            n_trades = res.get("trades", 0)
            if status == "no_trades":
                no_trades += 1
            ok_count += 1
            print(f"[{_now_iso()}] [{i}/{len(targets)}] sid={sid} "
                  f"{sym}/{tf} '{name[:30]}' → {status} "
                  f"trades={n_trades} wall={elapsed}s",
                  flush=True)
        except Exception as e:
            fail_count += 1
            elapsed = round(time.monotonic() - t0, 1)
            print(f"[{_now_iso()}] [{i}/{len(targets)}] sid={sid} "
                  f"{sym}/{tf} '{name[:30]}' → FAILED ({elapsed}s): "
                  f"{type(e).__name__}: {e}",
                  flush=True)

        # Pause between strategies — gives Supabase breathing room
        # and lets the data-worker's catchup interleave without
        # tripping the circuit breaker.
        if i < len(targets) and args.sleep > 0:
            time.sleep(args.sleep)

    total_wall = round(time.monotonic() - started, 1)
    print(f"\n[{_now_iso()}] Done. Wall={total_wall}s. "
          f"ok={ok_count} (no_trades={no_trades}) failed={fail_count} "
          f"skipped={len(skipped)}",
          flush=True)
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
