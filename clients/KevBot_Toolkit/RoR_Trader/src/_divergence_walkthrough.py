"""Divergence backlog walkthrough tool.

Wraps `/api/admin/strategy-health/backlog` so we get apples-to-apples
results with the UI, plus pulls ±2 min context for each event so we can
classify quickly. Designed for the workflow described in
`docs/SOP_Divergence_Investigation_DEPRECATED.md`.

Usage:
    python _divergence_walkthrough.py --window-hours 1 --max-clusters 20
    python _divergence_walkthrough.py --strategy-id 151 --window-hours 3
    python _divergence_walkthrough.py --denom    # also print the per-strategy paired-count denominator
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-hours", type=int, default=1)
    p.add_argument("--strategy-id", type=int, default=None)
    p.add_argument("--max-clusters", type=int, default=20)
    p.add_argument("--include-known", action="store_true",
                   help="Include all events, not just needs_investigation")
    p.add_argument("--denom", action="store_true",
                   help="Also fetch paired/total counts from the overview endpoint as the 'good' denominator")
    args = p.parse_args()

    # Admin context — match how the UI calls the endpoint
    from dotenv import load_dotenv
    load_dotenv(override=True)
    os.environ["DEV_BYPASS_AUTH"] = "true"
    os.environ.setdefault(
        "DEV_USER_ID", "19d47e46-f718-49a6-af32-5f5407f5b170")
    from db import set_admin_user_context, get_admin_client
    set_admin_user_context(os.environ["DEV_USER_ID"])
    from api.routers.strategy_health import (
        get_strategy_health_backlog, get_strategy_health)

    user = {"id": os.environ["DEV_USER_ID"], "email": "admin@local"}

    # Backlog query — same defaults as UI
    bl = get_strategy_health_backlog(
        user=user,
        window_hours=args.window_hours,
        start=None, end=None,
        only_needs_investigation=not args.include_known,
        apples_to_apples=True,
        max_rows=500,
        strategy_id=args.strategy_id,
    )
    rows = bl.get("rows") or []
    rows.sort(key=lambda r: r.get("timestamp") or "")
    print(f"=== Divergence backlog ===")
    print(f"window: {bl.get('window_start')[:19]} -> {bl.get('window_end')[:19]}")
    print(f"events: {bl.get('row_count')} (of {bl.get('total_count')}; "
          f"truncated={bl.get('truncated')})")
    print(f"apples_to_apples=True, only_needs_investigation="
          f"{not args.include_known}")
    print()

    if args.denom:
        oh = get_strategy_health(
            user=user, window_hours=args.window_hours,
            start=None, end=None)
        oh_rows = oh.get("rows") or []
        # Pull paired/phantom/missed by sid for strategies in the backlog
        sids_in_bl = {r.get("strategy_id") for r in rows}
        # 2026-05-29: prefer the GLOBAL fair counts for cross-strategy
        # comparison — they apply a single fleet-wide cutoff so identical
        # strategies actually report identical numbers. Raw counts include
        # batch-lag at the right edge and produced misleading per-strategy
        # asymmetry (sid 170 vs 172 case 2026-05-29). Raw counts shown in
        # parens for context.
        global_cutoff = (oh_rows[0].get("global_fair_cutoff_ts")
                          if oh_rows else None)
        print(f"=== Per-strategy denominator (overview, GLOBAL fair) ===")
        if global_cutoff:
            print(f"  Global fair cutoff: {global_cutoff[:19]}")
            print(f"  (events newer than cutoff excluded as batch-lag)")
        for orow in oh_rows:
            sid = orow.get("strategy_id")
            if sid not in sids_in_bl:
                continue
            p_global = orow.get("phantom_count_global_fair")
            m_global = orow.get("missed_count_global_fair")
            paired_global = orow.get("paired_count_global_fair")
            p_raw = orow.get("phantom_count")
            paired_raw = orow.get("paired_count")
            print(f"  sid={sid:4d} {orow.get('symbol'):5s} "
                  f"{orow.get('timeframe'):6s} "
                  f"paired={paired_global} phantom={p_global} "
                  f"missed={m_global}  "
                  f"(raw: paired={paired_raw} phantom={p_raw})")
        print()

    # Dedupe to clusters — events with same (timestamp, edge, event_type)
    # affecting multiple strategies are one root cause repeated.
    clusters = defaultdict(list)
    for r in rows:
        key = (r.get("timestamp"), r.get("edge"),
               r.get("event_type"), r.get("exit_reason"))
        clusters[key].append(r)

    print(f"=== Clustered: {len(clusters)} distinct (ts, edge, type, "
          f"exit_reason) combinations ===")
    print()

    c = get_admin_client()
    cluster_list = list(clusters.items())[: args.max_clusters]
    for idx, (key, members) in enumerate(cluster_list, 1):
        ts, edge, evt, er = key
        sid_list = sorted(m.get("strategy_id") for m in members)
        sym = members[0].get("symbol")
        tf = members[0].get("timeframe")
        ts_short = (ts or "")[:19]
        print(f"--- Cluster {idx}: {evt.upper()} at {ts_short} "
              f"{sym} {tf} edge={edge} exit_reason={er} ---")
        print(f"   strategies ({len(sid_list)}): {sid_list}")
        # First sid's alert_id / trade_id as the cluster's representative
        rep = members[0]
        if rep.get("alert_id"):
            print(f"   alert_id (rep): {rep.get('alert_id')}")
        if rep.get("trade_id"):
            print(f"   trade_id (rep): {rep.get('trade_id')}")

        # Pull ±2 min context for representative sid
        sid_rep = sid_list[0]
        try:
            t0 = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            lo = (t0 - timedelta(seconds=120)).isoformat()
            hi = (t0 + timedelta(seconds=120)).isoformat()
            ar = (c.table("alerts")
                  .select("id,fill_ts,bar_time,trigger_id,side,price,"
                          "exec_type,verification_status,"
                          "verification_close_delta")
                  .eq("strategy_id", sid_rep)
                  .gte("fill_ts", lo).lte("fill_ts", hi)
                  .order("fill_ts").execute())
            print(f"   nearby alerts on sid={sid_rep} (±2 min):")
            for a in ar.data:
                vs = a.get("verification_status") or "NULL"
                vd = a.get("verification_close_delta")
                fts = (a.get("fill_ts") or "")[:19]
                bt = (a.get("bar_time") or "")[:19]
                print(f"     id={a['id']} fill_ts={fts} bar_time={bt} "
                      f"{a.get('side')} {a.get('trigger_id'):20s} "
                      f"exec={a.get('exec_type')} px={a.get('price')} "
                      f"verify={vs} Δ={vd}")
            # trades context (entry_trigger column doesn't exist — use exit_trigger)
            tr = (c.table("trades")
                  .select("id,entry_fill_ts,exit_fill_ts,exit_trigger,"
                          "exit_reason")
                  .eq("strategy_id", sid_rep)
                  .gte("entry_fill_ts", lo).lte("entry_fill_ts", hi)
                  .execute())
            if tr.data:
                print(f"   nearby trades on sid={sid_rep}:")
                for t in tr.data:
                    print(f"     id={t['id']} entry={(t.get('entry_fill_ts') or '')[:19]} "
                          f"exit={(t.get('exit_fill_ts') or '')[:19]} "
                          f"exit_trigger={t.get('exit_trigger')} "
                          f"reason={t.get('exit_reason')}")
        except Exception as e:
            print(f"   [ctx fetch failed: {e}]")
        print()


if __name__ == "__main__":
    main()
