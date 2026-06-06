"""Hour-by-hour 5-day BT-vs-live divergence analysis for the 26 UT Bot
strategies in the active cohort.

For each strategy:
- Pull last 5 days of alerts (fill_ts, verification_status)
- Pull last 5 days of BT trades (entry_fill_ts + exit_fill_ts, both as
  trade events)
- Local pairing: each alert pairs with the nearest BT trade event
  within ±60s; vice versa for unpaired BT events
- Hour-bin everything by UTC hour. Compute per-hour:
    alerts, bt_events, paired, phantom (alert unpaired),
    missed (BT event unpaired), paired_pct, drift_unc_pct
- Save per-strategy CSV
- Build a summary highlighting the cleanest hours cohort-wide

Usage: cd src && ../.venv/bin/python _hourly_utbot_history.py
"""
from __future__ import annotations

import os
import sys
import json
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from db import get_admin_client  # noqa: E402


UT_BOT_SIDS = [
    263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275,
    277, 279, 281, 283, 285, 287, 291, 293, 295, 297, 299,
    301, 302, 303, 305,
]

# Most-trusted subset (Kevin's reference set, per session notes):
# pure utv4 signal canaries — no exotic packs gating
TRUSTED_REFERENCE = [263, 268, 270, 273, 302]

# Kevin's preferred tight pairing window (2026-06-06): catches
# intra-second timing quality, not just count balance. ±5s means
# an alert at 14:35:10 only pairs with a BT trade event at
# 14:35:05-14:35:15.
PAIR_WINDOW_S = 5
DAYS = 5


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def pair_events(a_ts: list[float], b_ts: list[float],
                window_s: float) -> tuple[list[bool], list[bool]]:
    """For each entry in a_ts, find nearest b_ts within ±window_s.
    Returns (a_paired_mask, b_paired_mask)."""
    a_paired = [False] * len(a_ts)
    b_paired = [False] * len(b_ts)
    if not a_ts or not b_ts:
        return a_paired, b_paired
    for i, t in enumerate(a_ts):
        # Find insertion point in b_ts
        idx = bisect.bisect_left(b_ts, t)
        # Candidates: idx-1 and idx
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
    # Second pass: for b's that didn't pair via a's lookup, do the
    # reverse so high-density regions don't miss matches. Cheap.
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


def analyze_strategy(c, sid: int, since_iso: str) -> dict:
    """Pull data + compute hourly stats for one strategy."""
    # Pull alerts
    alerts = []
    start = 0
    while True:
        r = (c.table("alerts")
             .select("fill_ts,verification_status,trigger_id")
             .eq("strategy_id", sid).gte("fill_ts", since_iso)
             .order("fill_ts").range(start, start + 999).execute())
        chunk = r.data or []
        alerts.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    # Pull BT trades
    trades = []
    start = 0
    while True:
        r = (c.table("trades")
             .select("entry_fill_ts,exit_fill_ts,exit_reason,data_source")
             .eq("strategy_id", sid).like("data_source", "backtest_%")
             .gte("entry_fill_ts", since_iso)
             .order("entry_fill_ts").range(start, start + 999).execute())
        chunk = r.data or []
        trades.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    # Build sorted timestamp arrays for pairing
    a_dts = []
    a_indices = []
    for i, a in enumerate(alerts):
        ts = parse_ts(a.get("fill_ts"))
        if ts is None:
            continue
        a_dts.append(ts.timestamp())
        a_indices.append(i)
    # BT events = each entry + each exit (separate events). We also
    # tag each event so we can label them later.
    b_events = []  # list of (ts, kind, trade_idx)
    for i, t in enumerate(trades):
        ets = parse_ts(t.get("entry_fill_ts"))
        if ets is not None:
            b_events.append((ets.timestamp(), "entry", i))
        xts = parse_ts(t.get("exit_fill_ts"))
        if xts is not None:
            b_events.append((xts.timestamp(), "exit", i))
    # Sort both
    a_sorted_idx = sorted(range(len(a_dts)), key=lambda k: a_dts[k])
    a_sorted_ts = [a_dts[k] for k in a_sorted_idx]
    b_sorted = sorted(b_events, key=lambda x: x[0])
    b_sorted_ts = [x[0] for x in b_sorted]

    a_paired_sorted, b_paired_sorted = pair_events(
        a_sorted_ts, b_sorted_ts, PAIR_WINDOW_S)

    # Map back to original alert/event order
    a_paired = [False] * len(a_dts)
    for k, sorted_i in enumerate(a_sorted_idx):
        a_paired[sorted_i] = a_paired_sorted[k]

    # Now hour-bin: collect counts per UTC hour string "YYYY-MM-DDTHH"
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
        if idx_in_a == -1:
            # alert had no parseable ts above — already excluded
            pass
        else:
            if a_paired[idx_in_a]:
                bucket["paired_a"] += 1
            else:
                bucket["phantom_a"] += 1
        # Verification status
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

    for k, (ts_sec, kind, trade_idx) in enumerate(b_sorted):
        ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        h = ts.strftime("%Y-%m-%dT%H")
        bucket = per_hour[h]
        bucket["bt_events"] += 1
        if b_paired_sorted[k]:
            bucket["paired_b"] += 1
        else:
            bucket["missed_b"] += 1

    # Build final per-hour table
    rows = []
    for h in sorted(per_hour.keys()):
        b = per_hour[h]
        alerts_n = b["alerts"]
        bt_n = b["bt_events"]
        paired = b["paired_a"]  # paired_a should equal paired_b
        phantom = b["phantom_a"]
        missed = b["missed_b"]
        total_evts = alerts_n + bt_n - paired  # avoid double-count of paired
        paired_pct = (100 * paired / max(1, paired + phantom + missed)
                      if (paired + phantom + missed) > 0 else 0.0)
        # Live coverage: BT events as fraction of expected
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
    since = datetime.now(timezone.utc) - timedelta(days=DAYS)
    since_iso = since.isoformat()
    print(f"Pulling last {DAYS}d for {len(UT_BOT_SIDS)} UT Bot strategies "
          f"(since {since_iso[:19]})")

    out_dir = "/tmp/utbot_hourly_5d"
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for sid in UT_BOT_SIDS:
        print(f"  sid {sid}...", end=" ", flush=True)
        try:
            res = analyze_strategy(c, sid, since_iso)
            all_results[sid] = res
            # Per-strategy CSV
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

    # Master JSON
    json_path = f"{out_dir}/all_strategies.json"
    with open(json_path, "w") as f:
        json.dump(
            {str(sid): res for sid, res in all_results.items()}, f, indent=2)
    print(f"\nMaster JSON: {json_path}")

    # Find the cleanest hours cohort-wide.
    # Definition: an hour is "clean" for a strategy if alerts ≥ 5 AND
    # paired_pct ≥ 90. Aggregate across cohort: how many strategies
    # met that bar in each hour?
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

    # Sort hours by n_clean_strategies descending — find the cleanest periods
    print()
    print("=== Top 20 hours by # of clean strategies (paired_pct >= 90 "
          "with alerts >= 5) ===")
    print(f"{'hour_utc':<16} {'n_clean':>8} {'active':>7} {'paired%':>9} "
          f"{'alerts':>7} {'phantoms':>9} {'missed':>7}")
    sorted_hours = sorted(hour_quality.items(),
                          key=lambda x: (-x[1]["n_clean_strategies"],
                                         -x[1]["total_alerts"]))
    for h, q in sorted_hours[:20]:
        denom = q["total_paired"] + q["total_phantom"] + q["total_missed"]
        agg = 100 * q["total_paired"] / max(1, denom)
        print(f"{h:<16} {q['n_clean_strategies']:>8} "
              f"{q['strategies_with_activity']:>7} {agg:>8.1f}% "
              f"{q['total_alerts']:>7} {q['total_phantom']:>9} "
              f"{q['total_missed']:>7}")

    # Save hour-quality summary
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


if __name__ == "__main__":
    main()
