"""Comprehensive hourly fleet divergence analysis covering 2026-05-28
through 2026-06-05 (inclusive — 9 days). Per-strategy pairing at ±5s,
combined paired-rate = paired / (paired + phantom + missed).

This is the comprehensive evidence base for the question: "do we need
to roll back any recent code?"

Output:
- /tmp/fleet_hourly_may28_jun5/<sid>_hourly.csv — per strategy CSV
- /tmp/fleet_hourly_may28_jun5/all_strategies.json — master JSON
- /tmp/fleet_hourly_may28_jun5/cohort_by_hour.csv — cohort aggregate per hour
- /tmp/fleet_hourly_may28_jun5/cohort_by_day.csv — per-day per-strategy summary
- /tmp/fleet_hourly_may28_jun5/best_worst_hours.csv — ranked hours by clean count

Cohort: 43 active strategies. The 3 with PRE-5/28 data (136, 174, 194)
are flagged so we can also show the pre-range tail.

Usage: cd src && ../.venv/bin/python _fleet_hourly_may28_jun5.py
"""
from __future__ import annotations

import os
import sys
import json
import bisect
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from db import get_admin_client  # noqa: E402


# All active strategies from the fleet baseline
COHORT = [
    136, 174, 194, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275,
    276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
    290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
    303, 304, 305,
]

# Strategies with PRE-5/28 data — for the comparison call-out
PRE_5_28_AVAILABLE = [136, 174, 194]

# Range: 9 full days
RANGE_START_ISO = "2026-05-28T00:00:00+00:00"
RANGE_END_ISO = "2026-06-06T00:00:00+00:00"  # exclusive

PAIR_WINDOW_S = 5
OUT_DIR = "/tmp/fleet_hourly_may28_jun5"


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def pair_events(a_ts, b_ts, window_s):
    """Symmetric ±window_s pairing of two sorted timestamp arrays."""
    a_paired = [False] * len(a_ts)
    b_paired = [False] * len(b_ts)
    if not a_ts or not b_ts:
        return a_paired, b_paired
    for i, t in enumerate(a_ts):
        idx = bisect.bisect_left(b_ts, t)
        best = None
        best_delta = window_s + 1
        for j in (idx - 1, idx):
            if 0 <= j < len(b_ts):
                d = abs(b_ts[j] - t)
                if d < best_delta:
                    best_delta = d
                    best = j
        if best is not None and best_delta <= window_s:
            a_paired[i] = True
            b_paired[best] = True
    for j, t in enumerate(b_ts):
        if b_paired[j]:
            continue
        idx = bisect.bisect_left(a_ts, t)
        for i in (idx - 1, idx):
            if 0 <= i < len(a_ts):
                d = abs(a_ts[i] - t)
                if d <= window_s:
                    b_paired[j] = True
                    a_paired[i] = True
                    break
    return a_paired, b_paired


def analyze_strategy(c, sid: int):
    """Pull data + compute hourly stats for one strategy in range."""
    # Alerts
    alerts = []
    start = 0
    while True:
        r = (c.table("alerts")
             .select("fill_ts,verification_status,trigger_id,side")
             .eq("strategy_id", sid)
             .gte("fill_ts", RANGE_START_ISO).lt("fill_ts", RANGE_END_ISO)
             .order("fill_ts").range(start, start + 999).execute())
        chunk = r.data or []
        alerts.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    # BT trades
    trades = []
    start = 0
    while True:
        r = (c.table("trades")
             .select("entry_fill_ts,exit_fill_ts,exit_reason,data_source")
             .eq("strategy_id", sid).like("data_source", "backtest_%")
             .gte("entry_fill_ts", RANGE_START_ISO).lt("entry_fill_ts", RANGE_END_ISO)
             .order("entry_fill_ts").range(start, start + 999).execute())
        chunk = r.data or []
        trades.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    # Build timestamp arrays
    a_dts = []
    a_originals = []
    for i, a in enumerate(alerts):
        ts = parse_ts(a.get("fill_ts"))
        if ts is None:
            continue
        a_dts.append(ts.timestamp())
        a_originals.append(a)

    b_events = []  # list of (ts_sec, kind)
    for t in trades:
        ets = parse_ts(t.get("entry_fill_ts"))
        if ets is not None:
            b_events.append((ets.timestamp(), "entry"))
        xts = parse_ts(t.get("exit_fill_ts"))
        if xts is not None:
            b_events.append((xts.timestamp(), "exit"))

    # Sort
    a_sorted_idx = sorted(range(len(a_dts)), key=lambda k: a_dts[k])
    a_sorted_ts = [a_dts[k] for k in a_sorted_idx]
    b_sorted = sorted(b_events, key=lambda x: x[0])
    b_sorted_ts = [x[0] for x in b_sorted]

    a_paired_sorted, b_paired_sorted = pair_events(
        a_sorted_ts, b_sorted_ts, PAIR_WINDOW_S)

    # Map back
    a_paired = [False] * len(a_dts)
    for k, sorted_i in enumerate(a_sorted_idx):
        a_paired[sorted_i] = a_paired_sorted[k]

    # Hour-bin
    per_hour = defaultdict(lambda: {
        "alerts": 0, "bt_events": 0, "paired_a": 0, "phantom_a": 0,
        "missed_b": 0, "v_total": 0, "v_drift_unc": 0,
        "v_verified": 0, "v_corrected": 0, "v_null": 0,
    })

    for i, a in enumerate(a_originals):
        ts = parse_ts(a.get("fill_ts"))
        if ts is None:
            continue
        h = ts.strftime("%Y-%m-%dT%H")
        bucket = per_hour[h]
        bucket["alerts"] += 1
        if a_paired[i]:
            bucket["paired_a"] += 1
        else:
            bucket["phantom_a"] += 1
        vs = a.get("verification_status") or "NULL"
        bucket["v_total"] += 1
        if vs == "drift_uncorrected":
            bucket["v_drift_unc"] += 1
        elif vs == "verified":
            bucket["v_verified"] += 1
        elif vs == "corrected":
            bucket["v_corrected"] += 1
        elif vs == "NULL":
            bucket["v_null"] += 1

    for k, (ts_sec, _kind) in enumerate(b_sorted):
        ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        h = ts.strftime("%Y-%m-%dT%H")
        bucket = per_hour[h]
        bucket["bt_events"] += 1
        if not b_paired_sorted[k]:
            bucket["missed_b"] += 1

    rows = []
    for h in sorted(per_hour.keys()):
        b = per_hour[h]
        alerts_n = b["alerts"]
        bt_n = b["bt_events"]
        paired = b["paired_a"]
        phantom = b["phantom_a"]
        missed = b["missed_b"]
        denom = paired + phantom + missed
        combined_pct = (100 * paired / denom) if denom > 0 else 0.0
        a_pair_denom = paired + phantom
        alert_pair_pct = (100 * paired / a_pair_denom) if a_pair_denom else 0.0
        drift_pct = (100 * b["v_drift_unc"] / max(1, b["v_total"])
                     if b["v_total"] else 0.0)
        rows.append({
            "hour_utc": h,
            "alerts": alerts_n,
            "bt_events": bt_n,
            "paired": paired,
            "phantom": phantom,
            "missed": missed,
            "combined_pct": round(combined_pct, 1),
            "alert_pair_pct": round(alert_pair_pct, 1),
            "drift_unc_pct": round(drift_pct, 2),
            "verified": b["v_verified"],
            "corrected": b["v_corrected"],
            "null_verify": b["v_null"],
        })

    return {
        "sid": sid,
        "alert_total": len(alerts),
        "bt_trade_total": len(trades),
        "hours": rows,
    }


def main():
    c = get_admin_client()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Pulling fleet hourly data for {len(COHORT)} strategies, "
          f"window {RANGE_START_ISO[:10]} → {RANGE_END_ISO[:10]} "
          f"(±{PAIR_WINDOW_S}s pairing)")
    print()

    all_results = {}
    for sid in COHORT:
        print(f"  sid {sid}...", end=" ", flush=True)
        try:
            res = analyze_strategy(c, sid)
            all_results[sid] = res
            csv_path = f"{OUT_DIR}/sid_{sid}_hourly.csv"
            with open(csv_path, "w") as f:
                f.write("hour_utc,alerts,bt_events,paired,phantom,missed,"
                        "combined_pct,alert_pair_pct,drift_unc_pct,"
                        "verified,corrected,null_verify\n")
                for r in res["hours"]:
                    f.write(",".join(str(r[k]) for k in [
                        "hour_utc", "alerts", "bt_events", "paired",
                        "phantom", "missed", "combined_pct", "alert_pair_pct",
                        "drift_unc_pct", "verified", "corrected",
                        "null_verify"]) + "\n")
            print(f"alerts={res['alert_total']} trades={res['bt_trade_total']} "
                  f"hours={len(res['hours'])}")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Master JSON
    json_path = f"{OUT_DIR}/all_strategies.json"
    with open(json_path, "w") as f:
        json.dump({str(sid): res for sid, res in all_results.items()},
                  f, indent=2)
    print(f"\nMaster JSON: {json_path}")

    # Cohort aggregate per hour
    hour_agg = defaultdict(lambda: {
        "n_strategies_active": 0,  # had alerts
        "n_clean_strategies": 0,    # combined_pct >= 90 with alerts >= 5
        "n_excellent": 0,           # combined_pct >= 95 with alerts >= 30
        "total_alerts": 0,
        "total_paired": 0,
        "total_phantom": 0,
        "total_missed": 0,
    })
    for sid, res in all_results.items():
        for r in res["hours"]:
            h = r["hour_utc"]
            if r["alerts"] >= 1:
                hour_agg[h]["n_strategies_active"] += 1
            hour_agg[h]["total_alerts"] += r["alerts"]
            hour_agg[h]["total_paired"] += r["paired"]
            hour_agg[h]["total_phantom"] += r["phantom"]
            hour_agg[h]["total_missed"] += r["missed"]
            if r["alerts"] >= 5 and r["combined_pct"] >= 90:
                hour_agg[h]["n_clean_strategies"] += 1
            if r["alerts"] >= 30 and r["combined_pct"] >= 95:
                hour_agg[h]["n_excellent"] += 1

    cohort_csv = f"{OUT_DIR}/cohort_by_hour.csv"
    with open(cohort_csv, "w") as f:
        f.write("hour_utc,n_strategies_active,n_clean,n_excellent,"
                "agg_combined_pct,total_alerts,total_paired,total_phantom,"
                "total_missed\n")
        for h in sorted(hour_agg.keys()):
            q = hour_agg[h]
            denom = q["total_paired"] + q["total_phantom"] + q["total_missed"]
            agg = (100 * q["total_paired"] / denom) if denom > 0 else 0
            f.write(f"{h},{q['n_strategies_active']},{q['n_clean_strategies']},"
                    f"{q['n_excellent']},{agg:.1f},{q['total_alerts']},"
                    f"{q['total_paired']},{q['total_phantom']},"
                    f"{q['total_missed']}\n")
    print(f"Cohort hourly: {cohort_csv}")

    # Per-day per-strategy summary
    day_csv = f"{OUT_DIR}/per_day_per_strategy.csv"
    with open(day_csv, "w") as f:
        f.write("sid,date,alerts,bt_events,paired,phantom,missed,"
                "combined_pct,alert_pair_pct,drift_unc_pct\n")
        for sid, res in all_results.items():
            day_agg = defaultdict(lambda: {
                "alerts": 0, "bt_events": 0, "paired": 0,
                "phantom": 0, "missed": 0,
                "v_drift_unc": 0, "v_total": 0,
            })
            for r in res["hours"]:
                d = r["hour_utc"][:10]
                day_agg[d]["alerts"] += r["alerts"]
                day_agg[d]["bt_events"] += r["bt_events"]
                day_agg[d]["paired"] += r["paired"]
                day_agg[d]["phantom"] += r["phantom"]
                day_agg[d]["missed"] += r["missed"]
                # Drift weighted by alert volume
                day_agg[d]["v_drift_unc"] += r["alerts"] * (
                    r["drift_unc_pct"] / 100.0)
                day_agg[d]["v_total"] += r["alerts"]
            for d in sorted(day_agg.keys()):
                v = day_agg[d]
                denom = v["paired"] + v["phantom"] + v["missed"]
                cpct = (100 * v["paired"] / denom) if denom > 0 else 0
                apr_denom = v["paired"] + v["phantom"]
                apr = (100 * v["paired"] / apr_denom) if apr_denom else 0
                drift = (100 * v["v_drift_unc"] / v["v_total"]
                         if v["v_total"] else 0)
                f.write(f"{sid},{d},{v['alerts']},{v['bt_events']},"
                        f"{v['paired']},{v['phantom']},{v['missed']},"
                        f"{cpct:.1f},{apr:.1f},{drift:.2f}\n")
    print(f"Per-day per-strategy: {day_csv}")

    # Best hours cohort-wide (top 50)
    sorted_hours = sorted(
        hour_agg.items(),
        key=lambda x: (-x[1]["n_excellent"], -x[1]["n_clean_strategies"],
                       -x[1]["total_alerts"]))
    best_csv = f"{OUT_DIR}/best_worst_hours.csv"
    with open(best_csv, "w") as f:
        f.write("hour_utc,n_strategies_active,n_clean,n_excellent,"
                "agg_combined_pct,total_alerts\n")
        for h, q in sorted_hours[:100]:
            denom = q["total_paired"] + q["total_phantom"] + q["total_missed"]
            agg = (100 * q["total_paired"] / denom) if denom > 0 else 0
            f.write(f"{h},{q['n_strategies_active']},{q['n_clean_strategies']},"
                    f"{q['n_excellent']},{agg:.1f},{q['total_alerts']}\n")
    print(f"Best hours ranking: {best_csv}")

    # Console summary
    print()
    print("=== TOP 25 hours cohort-wide (most strategies excellent) ===")
    print(f"{'hour_utc':<16} {'active':>7} {'clean':>6} {'excellent':>10} "
          f"{'agg_pct':>9} {'alerts':>7}")
    for h, q in sorted_hours[:25]:
        denom = q["total_paired"] + q["total_phantom"] + q["total_missed"]
        agg = (100 * q["total_paired"] / denom) if denom > 0 else 0
        print(f"{h:<16} {q['n_strategies_active']:>7} "
              f"{q['n_clean_strategies']:>6} {q['n_excellent']:>10} "
              f"{agg:>8.1f}% {q['total_alerts']:>7}")


if __name__ == "__main__":
    main()
