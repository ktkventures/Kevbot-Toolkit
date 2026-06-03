"""Re-measure backtest↔alert pair rates at ±5s tolerance.

The /api/admin/strategy-health endpoint hardcodes ±60s (legacy default).
This script mirrors its pairing logic but at the stricter ±5s standard
we want as the healthy-system target. Used to remeasure Tier B mid-range
packs after worker has been stable, and to get a fleet-wide picture of
backtest-to-live divergence over the last N hours.

Usage:
    python _remeasure_pair_rates_5s.py --window-hours 24
    python _remeasure_pair_rates_5s.py --window-hours 4 --tolerance 5
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple


def _parse_iso(s) -> Optional[datetime]:
    if not s:
        return None
    try:
        if isinstance(s, datetime):
            return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _pair_two_pointer(edge_isos: List[str],
                      alert_isos: List[str],
                      tolerance_sec: float,
                      upper_bound: Optional[datetime] = None
                      ) -> Tuple[int, int, int]:
    """Greedy two-pointer pairing within ±tolerance. Returns
    (phantom, missed, paired). edges = trade entry/exit timestamps;
    alerts = alert fill timestamps. upper_bound caps both lists."""
    edges = sorted(t.timestamp() for t in (_parse_iso(s) for s in edge_isos) if t)
    alerts = sorted(t.timestamp() for t in (_parse_iso(s) for s in alert_isos) if t)
    if upper_bound is not None:
        cap = upper_bound.timestamp()
        edges = [e for e in edges if e <= cap]
        alerts = [a for a in alerts if a <= cap]
    i = j = 0
    paired = 0
    while i < len(edges) and j < len(alerts):
        diff = alerts[j] - edges[i]
        if abs(diff) <= tolerance_sec:
            paired += 1
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1
    phantom = len(alerts) - paired
    missed = len(edges) - paired
    return phantom, missed, paired


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-hours", type=int, default=24)
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="Pair tolerance in seconds (default 5)")
    p.add_argument("--name-filter", default="PACKTEST",
                   help="Strategy name LIKE filter (default PACKTEST)")
    p.add_argument("--include-fair", action="store_true", default=True,
                   help="Apply fair_cutoff (min of last_bt_processed, last_alert) to avoid lag-tail inflation")
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(override=True)
    from db import get_admin_client
    ac = get_admin_client()

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=args.window_hours)
    window_start_iso = window_start.isoformat()
    window_end_iso = window_end.isoformat()

    # Pull all strategies matching the filter
    strategies = ac.table("strategies").select(
        "id,name,symbol,timeframe,user_id"
    ).like("name", f"{args.name_filter}%").execute().data
    if not strategies:
        print(f"No strategies matching '{args.name_filter}%'")
        return
    sids = [s["id"] for s in strategies]

    # Pull all alerts (event_type='fill' or legacy 'entry_signal'/'exit_signal')
    # for these strategies in the window. Paginate to avoid 1k cap.
    alerts_by_sid: dict = defaultdict(list)
    last_alert_per_sid: dict = {}
    offset = 0
    page_size = 1000
    while True:
        page = ac.table("alerts").select(
            "strategy_id,fill_ts"
        ).in_("strategy_id", sids) \
         .gte("fill_ts", window_start_iso) \
         .lte("fill_ts", window_end_iso) \
         .order("fill_ts") \
         .range(offset, offset + page_size - 1) \
         .execute().data
        if not page:
            break
        for r in page:
            sid = r["strategy_id"]
            ts = r["fill_ts"]
            if ts:
                alerts_by_sid[sid].append(ts)
                prev = last_alert_per_sid.get(sid)
                if prev is None or ts > prev:
                    last_alert_per_sid[sid] = ts
        if len(page) < page_size:
            break
        offset += page_size

    # Pull all trade edges (entry_fill_ts + exit_fill_ts) for these strategies
    # in the window, restricted to backtest data sources.
    edges_by_sid: dict = defaultdict(list)
    last_bt_edge_per_sid: dict = {}
    offset = 0
    while True:
        page = ac.table("trades").select(
            "strategy_id,entry_fill_ts,exit_fill_ts,data_source"
        ).in_("strategy_id", sids) \
         .like("data_source", "backtest%") \
         .or_(f"entry_fill_ts.gte.{window_start_iso},exit_fill_ts.gte.{window_start_iso}") \
         .order("entry_fill_ts") \
         .range(offset, offset + page_size - 1) \
         .execute().data
        if not page:
            break
        for r in page:
            sid = r["strategy_id"]
            for col in ("entry_fill_ts", "exit_fill_ts"):
                ts = r.get(col)
                if not ts:
                    continue
                if ts < window_start_iso or ts > window_end_iso:
                    continue
                edges_by_sid[sid].append(ts)
                prev = last_bt_edge_per_sid.get(sid)
                if prev is None or ts > prev:
                    last_bt_edge_per_sid[sid] = ts
        if len(page) < page_size:
            break
        offset += page_size

    # Score each strategy
    rows = []
    for s in strategies:
        sid = s["id"]
        name = s["name"]
        edges = edges_by_sid.get(sid, [])
        alerts = alerts_by_sid.get(sid, [])

        # Raw counts (no fair cutoff)
        phantom_raw, missed_raw, paired_raw = _pair_two_pointer(
            edges, alerts, args.tolerance, upper_bound=None)

        # Fair-cutoff counts: cap both lists at min(last_bt_edge, last_alert)
        last_bt = _parse_iso(last_bt_edge_per_sid.get(sid))
        last_al = _parse_iso(last_alert_per_sid.get(sid))
        if last_bt is not None and last_al is not None:
            fair_cutoff = min(last_bt, last_al)
        elif last_bt is not None:
            fair_cutoff = last_bt
        elif last_al is not None:
            fair_cutoff = last_al
        else:
            fair_cutoff = None

        if fair_cutoff is not None and args.include_fair:
            phantom, missed, paired = _pair_two_pointer(
                edges, alerts, args.tolerance, upper_bound=fair_cutoff)
        else:
            phantom, missed, paired = phantom_raw, missed_raw, paired_raw

        total = paired + phantom + missed
        rate = (paired / total * 100.0) if total else 0.0
        rows.append({
            "sid": sid,
            "name": name,
            "paired": paired,
            "phantom": phantom,
            "missed": missed,
            "total": total,
            "rate": rate,
            "raw_total": paired_raw + phantom_raw + missed_raw,
        })

    # Sort by name (then sid) for readable output
    rows.sort(key=lambda r: (r["name"], r["sid"]))

    print(f"\n=== Pair-rate remeasure ±{args.tolerance}s tolerance ===")
    print(f"window: {window_start_iso[:19]} -> {window_end_iso[:19]} ({args.window_hours}h)")
    print(f"fair_cutoff per-strategy: {'ON' if args.include_fair else 'OFF'}")
    print(f"{'sid':>4}  {'rate':>6}  {'paired':>6}  {'phantom':>7}  {'missed':>6}  name")
    print("-" * 100)
    for r in rows:
        rate_s = f"{r['rate']:5.1f}%" if r["total"] else "  --  "
        print(f"  {r['sid']:>3}  {rate_s:>6}  {r['paired']:>6}  {r['phantom']:>7}  {r['missed']:>6}  {r['name']}")

    # Aggregate stats
    active_rows = [r for r in rows if r["total"] > 0]
    if active_rows:
        total_paired = sum(r["paired"] for r in active_rows)
        total_phantom = sum(r["phantom"] for r in active_rows)
        total_missed = sum(r["missed"] for r in active_rows)
        total_all = total_paired + total_phantom + total_missed
        overall_rate = total_paired / total_all * 100.0 if total_all else 0.0
        print("-" * 100)
        print(f"FLEET TOTAL (active only, n={len(active_rows)}): "
              f"rate={overall_rate:.1f}%, "
              f"paired={total_paired}, phantom={total_phantom}, missed={total_missed}")
        silent = [r for r in rows if r["total"] == 0]
        if silent:
            print(f"SILENT strategies (no events in window): {len(silent)} "
                  f"({', '.join(str(r['sid']) for r in silent[:10])})")


if __name__ == "__main__":
    main()
