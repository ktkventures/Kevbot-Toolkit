"""Streaming-mode toggle verification.

Run from src/ via the project venv:
    ../.venv/bin/python _verify_streaming_toggle.py

What this does (read-only by default — does NOT flip the toggle):
1. Prints current system_settings.data_worker_streaming_enabled value
2. Prints the most-recent engine_snapshot_at across the fleet (the
   freshness signal — if streaming is ON, this advances every ~1 min
   during market hours)
3. Polls every 15s for 2 minutes, watching for snapshot freshness
   changes
4. Lists any recent update_jobs (the on-demand alternative to streaming)

Use this BEFORE flipping the toggle to capture a baseline, then again
AFTER flipping to confirm the worker actually paused/resumed.

Optional: --kick-job <sid>  Triggers a single update_job for the given
strategy_id so you can verify the on-demand path works while streaming
is paused.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone


def _load_env():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def _print_setting(client):
    """Show current value of the streaming-enabled flag."""
    r = (client.table("system_settings")
         .select("key,value,source:value,updated_at,updated_by")
         .eq("key", "data_worker_streaming_enabled")
         .maybe_single().execute())
    if not r or not r.data:
        env_default = os.environ.get(
            "DATA_WORKER_STREAMING_ENABLED", "true"
        ).strip().lower() in ("1", "true", "yes", "on")
        print(f"  system_settings row: <missing> — falling back to env var "
              f"DATA_WORKER_STREAMING_ENABLED → {'ON' if env_default else 'OFF'}")
        return env_default, "env"
    val = r.data.get("value")
    if isinstance(val, str):
        val = val.strip().lower() in ("1", "true", "yes", "on")
    print(f"  system_settings.data_worker_streaming_enabled = {val}")
    print(f"  updated_at = {r.data.get('updated_at')}")
    print(f"  updated_by = {r.data.get('updated_by')}")
    return bool(val), "db"


def _fleet_snapshot_freshness(client):
    """Freshness of streaming pipeline output. `kpis_computed_at` advances
    every ~5 minutes when the KPI recompute loop is running, so it's a
    good proxy for whether streaming work is actually happening."""
    r = (client.table("strategies")
         .select("id,name,kpis_computed_at,data_refreshed_at")
         .eq("alert_tracking_enabled", True)
         .not_.is_("kpis_computed_at", "null")
         .limit(500).execute())
    rows = [x for x in (r.data or []) if x.get("kpis_computed_at")]
    if not rows:
        print("  no strategies with kpis_computed_at — fleet idle or "
              "streaming has never run")
        return None
    ages = []
    now = datetime.now(timezone.utc)
    for row in rows:
        try:
            t = datetime.fromisoformat(
                row["kpis_computed_at"].replace("Z", "+00:00"))
            ages.append((now - t).total_seconds())
        except Exception:
            continue
    ages.sort()
    fresh = sum(1 for a in ages if a < 120)
    stale = sum(1 for a in ages if a > 600)
    n = len(ages)
    print(f"  strategies sampled: {n}")
    print(f"  snapshot age   min={ages[0]:>6.0f}s  "
          f"median={ages[n//2]:>6.0f}s  max={ages[-1]:>6.0f}s")
    print(f"  fresh (<2min): {fresh:>3d}   stale (>10min): {stale:>3d}")
    return ages[n // 2]


def _recent_update_jobs(client, limit=5):
    r = (client.table("update_jobs")
         .select("id,name,status,mode,scope,strategy_ids,created_at,updated_at")
         .order("created_at", desc=True).limit(limit).execute())
    rows = r.data or []
    if not rows:
        print("  no update_jobs yet")
        return
    for j in rows:
        ts = (j.get("created_at") or "")[:19].replace("T", " ")
        sids = j.get("strategy_ids") or []
        print(f"  #{j['id']:>10}  {j.get('status'):>10}  "
              f"{j.get('mode'):>4}/{j.get('scope'):>6}  "
              f"sids={len(sids)}  {ts}   {j.get('name', '')[:50]}")


def _kick_job(client, sid: int, user_id: str) -> int | None:
    """Insert a single-strategy update_job row + return its id. The job
    won't actually run unless the api service picks it up — but if you
    triggered it via the UI it should already have started."""
    print(f"\nTo kick an on-demand job for sid={sid}, POST to the api:")
    print(f"  curl -X POST -H 'Authorization: Bearer <token>' "
          f"-H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"strategy_ids\":[{sid}],\"mode\":\"new\"}}' \\")
    print(f"    https://<api-host>/api/update-jobs/run")
    print("(Or use the /admin/update-jobs UI's kickoff buttons.)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=120,
                    help="how long to keep polling (default 120s)")
    ap.add_argument("--interval", type=int, default=15,
                    help="poll interval (default 15s)")
    ap.add_argument("--kick-job", type=int, default=None,
                    help="suggest a curl command to kick an on-demand job "
                         "for this strategy_id while streaming is paused")
    args = ap.parse_args()

    _load_env()
    from db import get_admin_client
    client = get_admin_client()

    print("=" * 70)
    print("STREAMING-MODE TOGGLE VERIFICATION")
    print(f"now={datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print("\n[1] Current setting:")
    enabled, source = _print_setting(client)

    print("\n[2] Fleet snapshot freshness (baseline):")
    baseline_median = _fleet_snapshot_freshness(client)

    print("\n[3] Recent update_jobs:")
    _recent_update_jobs(client)

    if args.kick_job:
        _kick_job(client, args.kick_job, None)

    print("\n" + "=" * 70)
    print(f"Polling every {args.interval}s for {args.poll_seconds}s — "
          f"watching for streaming-state change…")
    print("Flip the toggle in /admin/update-jobs NOW to see the effect.")
    print("=" * 70)

    t0 = time.monotonic()
    last_enabled = enabled
    while time.monotonic() - t0 < args.poll_seconds:
        time.sleep(args.interval)
        print(f"\n--- tick at +{int(time.monotonic()-t0):>3d}s ---")
        cur_enabled, _ = _print_setting(client)
        if cur_enabled != last_enabled:
            print(f"  *** STREAMING FLAG CHANGED: "
                  f"{last_enabled} → {cur_enabled} ***")
            last_enabled = cur_enabled
        cur_median = _fleet_snapshot_freshness(client)
        if baseline_median and cur_median:
            delta = cur_median - baseline_median
            sign = "+" if delta >= 0 else ""
            print(f"  median snapshot age delta from baseline: {sign}{delta:.0f}s "
                  f"(advancing = streaming working; stuck/growing = paused)")

    print("\nDone. Compare:")
    print("  - If you flipped to MANUAL: median snapshot age should be "
          "GROWING (data worker no longer touching them).")
    print("  - If you flipped to AUTO:   median age should be SHRINKING "
          "back toward <120s.")
    print("  - Data Worker log: search for "
          "`effective streaming flag = ENABLED` / `DISABLED`")
    print("    in Railway logs to confirm the worker saw the change.")


if __name__ == "__main__":
    main()
