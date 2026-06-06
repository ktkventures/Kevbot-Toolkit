"""Hour-by-hour BT-vs-live divergence analysis for the same UT Bot
cohort, but covering LAST WEEK (Mon 2026-05-25 through Sun
2026-05-31). Kevin recalls parity was excellent going into the
weekend last week and wants the hourly data preserved for direct
comparison against this week's data before clicking UAD.

Same methodology as _hourly_utbot_history.py:
- For each UT Bot strategy, pull alerts + BT trades in the window
- Local pairing at ±5s (Kevin's preferred tight window)
- Hour-bin everything, compute per-hour:
    alerts, bt_events, paired, phantom, missed, paired_pct,
    drift_unc_pct
- Save per-strategy CSV + master JSON to a separate directory so
  this week's data isn't overwritten

Output goes to /tmp/utbot_hourly_lastweek/.

Usage: cd src && ../.venv/bin/python _hourly_utbot_history_lastweek.py
"""
from __future__ import annotations

import os
import sys
import json
import bisect
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from db import get_admin_client  # noqa: E402


UT_BOT_SIDS = [
    263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275,
    277, 279, 281, 283, 285, 287, 291, 293, 295, 297, 299,
    301, 302, 303, 305,
]

TRUSTED_REFERENCE = [263, 268, 270, 273, 302]

# Last week window: Mon 2026-05-25 00:00 UTC through Sun 2026-05-31
# 23:59 UTC. Memorial Day Monday means markets are closed but
# strategies were still subscribed so any pre-market activity will
# still surface.
SINCE_ISO = "2026-05-25T00:00:00+00:00"
UNTIL_ISO = "2026-06-01T00:00:00+00:00"

PAIR_WINDOW_S = 5


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def pair_events(a_ts: list[float], b_ts: list[float],
                window_s: float) -> tuple[list[bool], list[bool]]:
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


def analyze_strategy(c, sid: int) -> dict:
    alerts = []
    start = 0
    while True:
        r = (c.table("alerts")
             .select("fill_ts,verification_status,trigger_id")
             .eq("strategy_id", sid)
             .gte("fill_ts", SINCE_ISO).lt("fill_ts", UNTIL_ISO)
             .order("fill_ts").range(start, start + 999).execute())
        chunk = r.data or []
        alerts.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    trades = []
    start = 0
    while True:
        r = (c.table("trades")
             .select("entry_fill_ts,exit_fill_ts,exit_reason,data_source")
             .eq("strategy_id", sid).like("data_source", "backtest_%")
             .gte("entry_fill_ts", SINCE_ISO).lt("entry_fill_ts", UNTIL_ISO)
             .order("entry_fill_ts").range(start, start + 999).execute())
        chunk = r.data or []
        trades.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    a_dts = []
    a_indices = []
    for i, a in enumerate(alerts):
        ts = parse_ts(a.get("fill_ts"))
        if ts is None:
            continue
        a_dts.append(ts.timestamp())
        a_indices.append(i)
    b_events = []
    for i, t in enumerate(trades):
        ets = parse_ts(t.get("entry_fill_ts"))
        if ets is not None:
            b_events.append((ets.timestamp(), "entry", i))
        xts = parse_ts(t.get("exit_fill_ts"))
        if xts is not None:
            b_events.append((xts.timestamp(), "exit", i))
    a_sorted_idx = sorted(range(len(a_dts)), key=lambda k: a_dts[k])
    a_sorted_ts = [a_dts[k] for k in a_sorted_idx]
    b_sorted = sorted(b_events, key=lambda x: x[0])
    b_sorted_ts = [x[0] for x in b_sorted]

    a_paired_sorted, b_paired_sorted = pair_events(
        a_sorted_ts, b_sorted_ts, PAIR_WINDOW_S)

    a_paired = [False] * len(a_dts)
    for k, sorted_i in enumerate(a_sorted_idx):
        a_paired[sorted_i] = a_paired_sorted[k]

    per_hour = defaultdict(lambda: {
        "alerts": 0, "bt_events": 0, "paired_a": 0, "paired_b": 0,
        "phantom_a": 0, "missed_b": 0, "v_total": 0, "v_drift_unc": 0,
        "v_verified": 0, "v_corrected": 0, "v_null": 0,
    })

    for i, a in enumerate(alerts):
        ts = parse_ts(a.get("fill_ts"))
        if ts is None:
            continue
        h = ts.strftime("%Y-%m-%dT%H")
        bucket = per_hour[h]
        bucket["alerts"] += 1
        idx_in_a = a_indices.index(i) if i in a_indices else -1
        if idx_in_a != -1:
            if a_paired[idx_in_a]:
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

    for k, (ts_sec, _kind, _trade_idx) in enumerate(b_sorted):
        ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        h = ts.strftime("%Y-%m-%dT%H")
        bucket = per_hour[h]
        bucket["bt_events"] += 1
        if b_paired_sorted[k]:
            bucket["paired_b"] += 1
        else:
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
        paired_pct = 100 * paired / max(1, denom) if denom > 0 else 0.0
        bt_per_alert = (100 * bt_n / max(1, alerts_n)) if alerts_n else 0
        drift_pct = (100 * b["v_drift_unc"] / max(1, b["v_total"])
                     if b["v_total"] else 0.0)
        rows.append({
            "hour_utc": h,
            "alerts": alerts_n,
            "bt_events": bt_n,
            "paired": paired,
            "phantom": phantom,
            "missed": missed,
            "paired_pct": round(paired_pct, 1),
            "bt_per_alert_pct": round(bt_per_alert, 1),
            "drift_unc_pct": round(drift_pct, 2),
            "verified": b["v_verified"],
            "corrected": b["v_corrected"],
            "null_verify": b["v_null"],
        })
    return {"sid": sid, "alert_total": len(alerts),
            "bt_trade_total": len(trades), "hours": rows}


def main():
    c = get_admin_client()
    print(f"Pulling LAST WEEK ({SINCE_ISO[:10]} → {UNTIL_ISO[:10]}) "
          f"for {len(UT_BOT_SIDS)} UT Bot strategies at ±{PAIR_WINDOW_S}s pairing")

    out_dir = "/tmp/utbot_hourly_lastweek"
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for sid in UT_BOT_SIDS:
        print(f"  sid {sid}...", end=" ", flush=True)
        try:
            res = analyze_strategy(c, sid)
            all_results[sid] = res
            csv_path = f"{out_dir}/sid_{sid}_hourly.csv"
            with open(csv_path, "w") as f:
                f.write("hour_utc,alerts,bt_events,paired,phantom,"
                        "missed,paired_pct,bt_per_alert_pct,"
                        "drift_unc_pct,verified,corrected,null_verify\n")
                for row in res["hours"]:
                    f.write(",".join(str(row[k]) for k in [
                        "hour_utc", "alerts", "bt_events", "paired",
                        "phantom", "missed", "paired_pct",
                        "bt_per_alert_pct", "drift_unc_pct",
                        "verified", "corrected", "null_verify"]) + "\n")
            print(f"alerts={res['alert_total']} trades={res['bt_trade_total']} "
                  f"hours={len(res['hours'])} -> {csv_path}")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    json_path = f"{out_dir}/all_strategies.json"
    with open(json_path, "w") as f:
        json.dump(
            {str(sid): res for sid, res in all_results.items()}, f, indent=2)
    print(f"\nMaster JSON: {json_path}")

    hour_quality = defaultdict(lambda: {
        "n_clean_strategies": 0, "total_alerts": 0,
        "total_paired": 0, "total_phantom": 0, "total_missed": 0,
        "strategies_with_activity": 0,
    })
    for sid, res in all_results.items():
        for row in res["hours"]:
            h = row["hour_utc"]
            if row["alerts"] >= 5:
                hour_quality[h]["strategies_with_activity"] += 1
                hour_quality[h]["total_alerts"] += row["alerts"]
                hour_quality[h]["total_paired"] += row["paired"]
                hour_quality[h]["total_phantom"] += row["phantom"]
                hour_quality[h]["total_missed"] += row["missed"]
                if row["paired_pct"] >= 90:
                    hour_quality[h]["n_clean_strategies"] += 1

    print()
    print("=== Top 30 hours by # clean strategies (paired_pct >= 90, "
          "alerts >= 5) — LAST WEEK ===")
    print(f"{'hour_utc':<16} {'n_clean':>8} {'active':>7} {'paired%':>9} "
          f"{'alerts':>7} {'phantoms':>9} {'missed':>7}")
    sorted_hours = sorted(hour_quality.items(),
                          key=lambda x: (-x[1]["n_clean_strategies"],
                                         -x[1]["total_alerts"]))
    for h, q in sorted_hours[:30]:
        denom = q["total_paired"] + q["total_phantom"] + q["total_missed"]
        agg = 100 * q["total_paired"] / max(1, denom)
        print(f"{h:<16} {q['n_clean_strategies']:>8} "
              f"{q['strategies_with_activity']:>7} {agg:>8.1f}% "
              f"{q['total_alerts']:>7} {q['total_phantom']:>9} "
              f"{q['total_missed']:>7}")

    hq_path = f"{out_dir}/hour_quality_summary.csv"
    with open(hq_path, "w") as f:
        f.write("hour_utc,n_clean_strategies,strategies_with_activity,"
                "agg_paired_pct,total_alerts,total_paired,total_phantom,"
                "total_missed\n")
        for h, q in sorted_hours:
            denom = q["total_paired"] + q["total_phantom"] + q["total_missed"]
            agg = 100 * q["total_paired"] / max(1, denom)
            f.write(f"{h},{q['n_clean_strategies']},"
                    f"{q['strategies_with_activity']},{agg:.1f},"
                    f"{q['total_alerts']},{q['total_paired']},"
                    f"{q['total_phantom']},{q['total_missed']}\n")
    print(f"\nHour-quality summary: {hq_path}")

    # Per-strategy daily summary for trusted set
    print()
    print("=== TRUSTED reference UT Bot strategies — daily summary LAST WEEK ===")
    for sid in TRUSTED_REFERENCE:
        if sid not in all_results:
            print(f"\nsid {sid}: not found")
            continue
        res = all_results[sid]
        daily: dict = {}
        for r in res["hours"]:
            d = r["hour_utc"][:10]
            if d not in daily:
                daily[d] = {"alerts": 0, "paired": 0, "phantom": 0,
                            "missed": 0, "bt": 0}
            daily[d]["alerts"] += r["alerts"]
            daily[d]["paired"] += r["paired"]
            daily[d]["phantom"] += r["phantom"]
            daily[d]["missed"] += r["missed"]
            daily[d]["bt"] += r["bt_events"]
        print(f"\n  sid {sid}  total alerts={res['alert_total']} "
              f"BT pairs={res['bt_trade_total']}:")
        print(f"    {'date':<11} {'alerts':>7} {'bt_evts':>8} "
              f"{'paired':>7} {'phantom':>8} {'missed':>7} {'paired%':>8}")
        for d in sorted(daily.keys()):
            v = daily[d]
            denom = v["paired"] + v["phantom"] + v["missed"]
            p_pct = 100 * v["paired"] / max(1, denom)
            print(f"    {d:<11} {v['alerts']:>7} {v['bt']:>8} "
                  f"{v['paired']:>7} {v['phantom']:>8} {v['missed']:>7} "
                  f"{p_pct:>7.1f}%")


if __name__ == "__main__":
    main()
