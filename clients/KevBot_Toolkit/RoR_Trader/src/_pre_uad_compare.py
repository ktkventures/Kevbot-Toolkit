"""Today RTH vs Yesterday RTH comparison + cohort trend signal."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def hr_range(date_str: str, h0: int, h1: int) -> list[str]:
    return [f"{date_str}T{h:02d}" for h in range(h0, h1)]


def window_sum(by_sid_hour: dict, sid: int, hours: list[str]) -> int:
    h = by_sid_hour.get(str(sid), {})
    return sum(h.get(hh, 0) for hh in hours)


def main():
    with open("/tmp/pre_uad_baseline_2026-06-06.json") as f:
        snap = json.load(f)
    cohort = snap["cohort_ids"]

    # RTH (13:30-20:00 UTC ≈ regular session)
    hours_rth_today = hr_range("2026-06-05", 13, 20)
    hours_rth_yest = hr_range("2026-06-04", 13, 20)
    hours_eth = hr_range("2026-06-05", 20, 24) + hr_range("2026-06-06", 0, 4)

    a_y = {sid: window_sum(snap["alerts_by_sid_hour"], sid, hours_rth_yest)
           for sid in cohort}
    b_y = {sid: window_sum(snap["bt_entries_by_sid_hour"], sid, hours_rth_yest)
              + window_sum(snap["bt_exits_by_sid_hour"], sid, hours_rth_yest)
           for sid in cohort}
    a_t = {sid: window_sum(snap["alerts_by_sid_hour"], sid, hours_rth_today)
           for sid in cohort}
    b_t = {sid: window_sum(snap["bt_entries_by_sid_hour"], sid, hours_rth_today)
              + window_sum(snap["bt_exits_by_sid_hour"], sid, hours_rth_today)
           for sid in cohort}
    a_e = {sid: window_sum(snap["alerts_by_sid_hour"], sid, hours_eth)
           for sid in cohort}
    b_e = {sid: window_sum(snap["bt_entries_by_sid_hour"], sid, hours_eth)
              + window_sum(snap["bt_exits_by_sid_hour"], sid, hours_eth)
           for sid in cohort}

    T_a_y, T_b_y = sum(a_y.values()), sum(b_y.values())
    T_a_t, T_b_t = sum(a_t.values()), sum(b_t.values())
    T_a_e, T_b_e = sum(a_e.values()), sum(b_e.values())

    print("=== COHORT TOTAL ===")
    print(f"                          Alerts   BT_evts   Ratio")
    print(f"  RTH 2026-06-04 (yest):  {T_a_y:>6}    {T_b_y:>6}   {100*T_b_y/max(1,T_a_y):>5.1f}%")
    print(f"  RTH 2026-06-05 (today): {T_a_t:>6}    {T_b_t:>6}   {100*T_b_t/max(1,T_a_t):>5.1f}%")
    print(f"  After-hours 06-05+06:   {T_a_e:>6}    {T_b_e:>6}   {100*T_b_e/max(1,T_a_e):>5.1f}%")
    print()

    print("=== Per-strategy: BT/alert ratio comparison ===")
    print(f"{'sid':>4} {'yest_RTH':>11} {'today_RTH':>11} {'trend':>9}")
    print("-" * 60)
    better, worse, flat = [], [], []
    for sid in cohort:
        ya, yb = a_y[sid], b_y[sid]
        ta, tb = a_t[sid], b_t[sid]
        y_r = (100 * yb / ya) if ya > 0 else None
        t_r = (100 * tb / ta) if ta > 0 else None
        if y_r is None and t_r is None:
            continue
        if y_r is not None and t_r is not None:
            delta = t_r - y_r
            if abs(delta) < 5:
                trend = "flat"
                flat.append((sid, delta))
            elif delta > 0:
                trend = f"+{delta:.1f}"
                better.append((sid, delta))
            else:
                trend = f"{delta:.1f}"
                worse.append((sid, delta))
        else:
            trend = "—"
        y_s = f"{y_r:.1f}%" if y_r is not None else "—"
        t_s = f"{t_r:.1f}%" if t_r is not None else "—"
        print(f"{sid:>4} {y_s:>11} {t_s:>11} {trend:>9}")

    print()
    print(f"Trend summary: {len(better)} BETTER, {len(worse)} WORSE, {len(flat)} flat")
    if better:
        print(f"  top improvements: "
              f"{sorted(better, key=lambda x: -x[1])[:5]}")
    if worse:
        print(f"  top regressions:  "
              f"{sorted(worse, key=lambda x: x[1])[:5]}")

    tsv = "/tmp/pre_uad_trend_2026-06-06.tsv"
    with open(tsv, "w") as f:
        f.write("sid\talerts_yest\tbt_evts_yest\tratio_yest\t"
                "alerts_today\tbt_evts_today\tratio_today\ttrend_delta\n")
        for sid in cohort:
            ya, yb = a_y[sid], b_y[sid]
            ta, tb = a_t[sid], b_t[sid]
            y_r = (100 * yb / ya) if ya > 0 else ""
            t_r = (100 * tb / ta) if ta > 0 else ""
            d = (t_r - y_r) if (y_r != "" and t_r != "") else ""
            f.write(f"{sid}\t{ya}\t{yb}\t{y_r}\t{ta}\t{tb}\t{t_r}\t{d}\n")
    print(f"\nTrend TSV: {tsv}")


if __name__ == "__main__":
    main()
