"""Comprehensive 48h pre-UAD baseline snapshot.

Captures fleet-wide alerts + BT trade counts per strategy per hour so
we have a fallback reference point if a subsequent Update All Data
materially changes the divergence picture.

Usage: cd src && ../.venv/bin/python _pre_uad_baseline.py
"""
from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from db import get_admin_client  # noqa: E402


ACTIVE_SIDS = [
    136, 174, 194, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275,
    276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290,
    291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
]


def main():
    c = get_admin_client()
    now = datetime.now(timezone.utc)
    t48h = (now - timedelta(hours=48)).isoformat()

    print(f"NOW: {now.isoformat()[:19]}  cutoff: {t48h[:19]}")
    print("Pulling alerts...")
    alerts_by_sid_hour: dict = defaultdict(lambda: Counter())
    total_alerts = 0
    start = 0
    while True:
        r = (c.table("alerts").select("strategy_id,fill_ts")
             .gte("fill_ts", t48h).order("fill_ts")
             .range(start, start + 999).execute())
        chunk = r.data or []
        for a in chunk:
            sid = a["strategy_id"]
            if sid not in ACTIVE_SIDS:
                continue
            fts = a.get("fill_ts")
            if not fts:
                continue
            alerts_by_sid_hour[sid][fts[:13]] += 1
            total_alerts += 1
        if len(chunk) < 1000:
            break
        start += 1000
    print(f"  total alerts: {total_alerts}")

    print("Pulling BT trades...")
    bt_entry_by_sid_hour: dict = defaultdict(lambda: Counter())
    bt_exit_by_sid_hour: dict = defaultdict(lambda: Counter())
    total_bt = 0
    start = 0
    while True:
        r = (c.table("trades")
             .select("strategy_id,entry_fill_ts,exit_fill_ts,data_source")
             .gte("entry_fill_ts", t48h).like("data_source", "backtest_%")
             .order("entry_fill_ts").range(start, start + 999).execute())
        chunk = r.data or []
        for t in chunk:
            sid = t["strategy_id"]
            if sid not in ACTIVE_SIDS:
                continue
            ents = t.get("entry_fill_ts")
            exs = t.get("exit_fill_ts")
            if ents:
                bt_entry_by_sid_hour[sid][ents[:13]] += 1
            if exs and exs >= t48h:
                bt_exit_by_sid_hour[sid][exs[:13]] += 1
            total_bt += 1
        if len(chunk) < 1000:
            break
        start += 1000
    print(f"  total BT pairs: {total_bt}")

    out_path = "/tmp/pre_uad_baseline_2026-06-06.json"
    snapshot = {
        "created_at": now.isoformat(),
        "cohort_size": len(ACTIVE_SIDS),
        "cohort_ids": ACTIVE_SIDS,
        "window_hours": 48,
        "alerts_by_sid_hour": {
            str(sid): dict(h) for sid, h in alerts_by_sid_hour.items()},
        "bt_entries_by_sid_hour": {
            str(sid): dict(h) for sid, h in bt_entry_by_sid_hour.items()},
        "bt_exits_by_sid_hour": {
            str(sid): dict(h) for sid, h in bt_exit_by_sid_hour.items()},
    }
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nRaw snapshot saved to: {out_path}")

    print()
    print(f'{"sid":>4} {"alerts":>7} {"BT_pairs":>9} '
          f'{"BT_evts":>8} {"ratio":>7}  verdict')
    print("-" * 75)
    results = []
    for sid in ACTIVE_SIDS:
        n_alerts = sum(alerts_by_sid_hour[sid].values())
        n_bt_pairs = sum(bt_entry_by_sid_hour[sid].values())
        n_bt_exits_in_window = sum(bt_exit_by_sid_hour[sid].values())
        n_bt_events = n_bt_pairs + n_bt_exits_in_window
        if n_alerts == 0:
            ratio = float("nan")
            v = "no_activity"
        else:
            ratio = 100 * n_bt_events / n_alerts
            if abs(ratio - 100) < 10:
                v = "OK"
            elif ratio > 110:
                v = "BT>alerts"
            elif ratio > 70:
                v = "soft gap"
            elif ratio > 30:
                v = "big gap"
            else:
                v = "critical"
        results.append((sid, n_alerts, n_bt_pairs, n_bt_events, ratio, v))
        ratio_s = f"{ratio:>6.1f}%" if n_alerts else "      —"
        print(f"{sid:>4} {n_alerts:>7} {n_bt_pairs:>9} {n_bt_events:>8} "
              f"{ratio_s}  {v}")

    tsv_path = "/tmp/pre_uad_summary_2026-06-06.tsv"
    with open(tsv_path, "w") as f:
        f.write("sid\talerts\tbt_pairs\tbt_events\tratio_pct\tverdict\n")
        for row in results:
            ratio_field = f"{row[4]:.1f}" if row[1] else "nan"
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t"
                    f"{ratio_field}\t{row[5]}\n")
    print(f"\nSummary TSV saved to: {tsv_path}")


if __name__ == "__main__":
    main()
