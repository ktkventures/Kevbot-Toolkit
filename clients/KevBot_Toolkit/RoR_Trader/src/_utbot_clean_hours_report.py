"""Compile clean-hours report + per-strategy drill-down for the
trusted UT Bot set. Reads /tmp/utbot_hourly_5d/all_strategies.json
produced by _hourly_utbot_history.py."""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TOP_HOURS = [
    "2026-06-04T14", "2026-06-04T13", "2026-06-04T17",
    "2026-06-03T14", "2026-06-03T15", "2026-06-02T14",
    "2026-06-04T18", "2026-06-04T16", "2026-06-04T15",
    "2026-06-05T18", "2026-06-05T19",
]

TRUSTED = [263, 268, 270, 273, 302]


def main():
    with open("/tmp/utbot_hourly_5d/all_strategies.json") as f:
        data = json.load(f)

    print("=== Strategies that hit paired_pct >= 90 with alerts >= 5 "
          "in each top hour ===")
    for h in TOP_HOURS:
        clean = []
        activity = []
        for sid_str, res in data.items():
            for row in res["hours"]:
                if row["hour_utc"] == h:
                    if row["alerts"] >= 5:
                        activity.append((
                            int(sid_str), row["paired_pct"],
                            row["alerts"], row["phantom"], row["missed"]))
                        if row["paired_pct"] >= 90:
                            clean.append((
                                int(sid_str), row["paired_pct"], row["alerts"]))
                    break
        print(f"\n  {h}  ({len(clean)} clean of {len(activity)} active):")
        for sid, pct, a in sorted(clean, key=lambda x: -x[1]):
            print(f"    sid {sid:>4}  paired={pct}%  alerts={a}")
        if not clean:
            top3 = sorted(activity, key=lambda x: -x[1])[:3]
            print(f"    (none clean — best 3: {top3})")

    print()
    print("=== TRUSTED reference UT Bot strategies — top 10 cleanest "
          "hours each ===")
    for sid in TRUSTED:
        sid_str = str(sid)
        if sid_str not in data:
            print(f"\nsid {sid}: not in data")
            continue
        res = data[sid_str]
        hours_active = [r for r in res["hours"] if r["alerts"] >= 5]
        if not hours_active:
            print(f"\nsid {sid}: no hours with alerts >= 5")
            continue
        top = sorted(
            hours_active,
            key=lambda r: (-r["paired_pct"], -r["alerts"]))[:10]
        print(f"\n  sid {sid}  total {res['alert_total']} alerts / "
              f"{res['bt_trade_total']} BT pairs:")
        print(f"    {'hour':<16} {'alerts':>7} {'bt_evts':>8} "
              f"{'paired':>7} {'phantom':>8} {'missed':>7} {'paired%':>8}")
        for r in top:
            print(f"    {r['hour_utc']:<16} {r['alerts']:>7} "
                  f"{r['bt_events']:>8} {r['paired']:>7} {r['phantom']:>8} "
                  f"{r['missed']:>7} {r['paired_pct']:>7.1f}%")

    print()
    print("=== DAILY summary per trusted strategy ===")
    for sid in TRUSTED:
        sid_str = str(sid)
        if sid_str not in data:
            continue
        res = data[sid_str]
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
        print(f"\n  sid {sid}:")
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
