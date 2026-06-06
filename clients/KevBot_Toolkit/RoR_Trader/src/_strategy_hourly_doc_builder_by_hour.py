"""Pivot of the per-strategy hourly doc — one table per hour, rows are
strategies. Lets you see which strategies were cleanest in any given
hour cohort-wide.

Output: docs/Strategy_Hourly_By_Hour_2026-06-06.md
"""
from __future__ import annotations

import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COHORT = [
    136, 174, 194, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275,
    276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
    290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
    303, 304, 305,
]


def load_data():
    with open("/tmp/fleet_hourly_may28_jun5/all_strategies.json") as f:
        fleet = json.load(f)
    try:
        with open("/tmp/utbot_hourly_lastweek/all_strategies.json") as f:
            lastweek = json.load(f)
    except FileNotFoundError:
        lastweek = {}
    merged = {}
    for sid in COHORT:
        sid_str = str(sid)
        hours = {}
        if sid_str in lastweek:
            for r in lastweek[sid_str]["hours"]:
                if r["alerts"] >= 1 or r["bt_events"] >= 1:
                    hours[r["hour_utc"]] = dict(r)
        if sid_str in fleet:
            for r in fleet[sid_str]["hours"]:
                h = r["hour_utc"]
                if r["alerts"] == 0 and r["bt_events"] == 0:
                    continue
                if h not in hours:
                    hours[h] = dict(r)
                else:
                    if r["bt_events"] > hours[h]["bt_events"]:
                        hours[h] = dict(r)
        merged[sid_str] = hours
    return merged


def build_hour_table(hour_utc: str, hour_data: dict, names: dict) -> str:
    """Build the markdown for one hour. hour_data: {sid: row}."""
    if not hour_data:
        return ""
    # Compute combined % and alert-pair % per strategy
    rows = []
    for sid, r in hour_data.items():
        a = r["alerts"]
        bt = r["bt_events"]
        p = r["paired"]
        ph = r["phantom"]
        m = r["missed"]
        denom = p + ph + m
        cpct = (100 * p / denom) if denom > 0 else 0.0
        a_denom = p + ph
        apr = (100 * p / a_denom) if a_denom > 0 else 0.0
        rows.append({
            "sid": sid,
            "alerts": a,
            "bt_events": bt,
            "paired": p,
            "phantom": ph,
            "missed": m,
            "cpct": cpct,
            "apr": apr,
        })
    # Rank within hour by combined % descending (alerts >=1 AND bt >=1)
    rankable = [r for r in rows
                if r["alerts"] >= 1 and r["bt_events"] >= 1]
    rankable.sort(key=lambda r: (-r["cpct"], -r["alerts"]))
    rank_map = {r["sid"]: i + 1 for i, r in enumerate(rankable)}
    n_ranked = len(rankable)

    # KPI line: avg / min / max combined % across RANKED strategies
    # (those with alerts >=1 AND bt_events >=1 — meaningful pair attempt)
    if rankable:
        cpcts = [r["cpct"] for r in rankable]
        kpi_avg = sum(cpcts) / len(cpcts)
        kpi_min = min(cpcts)
        kpi_max = max(cpcts)
        kpi_line = (f"**KPIs (across {len(rankable)} ranked strategies):** "
                    f"avg combined = **{kpi_avg:.1f}%** · "
                    f"max = **{kpi_max:.1f}%** · "
                    f"min = **{kpi_min:.1f}%**")
    else:
        kpi_line = "**KPIs:** no ranked strategies (no meaningful pair attempts)"

    # Output sorted by sid number
    out = []
    out.append(f"## Hour {hour_utc} UTC\n")
    out.append(f"_Strategies with activity this hour: {len(rows)}; "
               f"ranked (alerts≥1, BT≥1): {n_ranked}_\n")
    out.append(kpi_line + "\n")
    out.append(f"| sid | Strategy name | Alerts | BT events | Paired | "
               f"Phantom | Missed | Combined % | Alert-pair % | Rank |")
    out.append(f"|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x["sid"]):
        sid = r["sid"]
        name = (names.get(int(sid), "(unknown)") or "(unknown)")[:45]
        rank = rank_map.get(sid)
        rank_str = f"{rank}/{n_ranked}" if rank else "—"
        out.append(
            f"| {sid} | {name} | {r['alerts']} | {r['bt_events']} | "
            f"{r['paired']} | {r['phantom']} | {r['missed']} | "
            f"{r['cpct']:.1f}% | {r['apr']:.1f}% | {rank_str} |")
    out.append("")
    return "\n".join(out)


def main():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".env"))
    from db import get_admin_client
    c = get_admin_client()
    names = {}
    for sid in COHORT:
        r = (c.table("strategies")
             .select("name").eq("id", sid).single().execute())
        names[sid] = (r.data.get("name") if r.data else "(unknown)") or "(unknown)"

    merged = load_data()

    # Pivot: hour -> {sid_str: row}
    by_hour = defaultdict(dict)
    for sid_str, hours in merged.items():
        for h, r in hours.items():
            by_hour[h][sid_str] = r

    sorted_hours = sorted(by_hour.keys())

    # Pre-compute KPIs per hour so we can rank hours
    hour_kpis = {}
    for h in sorted_hours:
        rows = by_hour[h]
        rankable_cpcts = []
        for sid_str, r in rows.items():
            if r["alerts"] >= 1 and r["bt_events"] >= 1:
                denom = r["paired"] + r["phantom"] + r["missed"]
                cpct = (100 * r["paired"] / denom) if denom > 0 else 0.0
                rankable_cpcts.append(cpct)
        if rankable_cpcts:
            hour_kpis[h] = {
                "n_ranked": len(rankable_cpcts),
                "avg": sum(rankable_cpcts) / len(rankable_cpcts),
                "max": max(rankable_cpcts),
                "min": min(rankable_cpcts),
            }
        else:
            hour_kpis[h] = {"n_ranked": 0, "avg": None,
                            "max": None, "min": None}

    # Rank hours by each metric (rank 1 = highest avg/max/min)
    # Skip hours with None (no ranked strategies)
    ranked_hours_with_data = [h for h in sorted_hours
                              if hour_kpis[h]["avg"] is not None]

    def rank_by(metric: str) -> dict:
        sorted_by = sorted(ranked_hours_with_data,
                            key=lambda h: -hour_kpis[h][metric])
        return {h: i + 1 for i, h in enumerate(sorted_by)}

    avg_rank = rank_by("avg")
    max_rank = rank_by("max")
    min_rank = rank_by("min")
    n_with_data = len(ranked_hours_with_data)

    # Identify top-5 by each metric for highlighting
    top5_avg = set(h for h in ranked_hours_with_data if avg_rank[h] <= 5)
    top5_max = set(h for h in ranked_hours_with_data if max_rank[h] <= 5)
    top5_min = set(h for h in ranked_hours_with_data if min_rank[h] <= 5)

    # Build doc
    doc = []
    doc.append("# Hourly Cohort Tables — 2026-06-06\n")
    doc.append(
        "> Same data as `Strategy_Hourly_Tables_2026-06-06.md` but pivoted: "
        "one table per HOUR timestamp, rows are strategies. Lets you see "
        "which strategies were cleanest in any given hour cohort-wide.\n"
        "> Pair window: ±5s. Combined % = paired / (paired + phantom + "
        "missed). Alert-pair % = paired / (paired + phantom). Rank is "
        "per-hour cohort-wide: rank 1 = the cleanest strategy in this "
        "hour by Combined %. Only strategies with alerts ≥ 1 AND "
        "BT events ≥ 1 in that hour are ranked.\n"
    )

    # SUMMARY TABLE — all hours chronologically with KPIs + cross-hour ranks
    doc.append("## Cross-hour summary — KPIs + ranks\n")
    doc.append(f"_All {n_with_data} hours with ranked-strategy activity, "
               f"sorted chronologically. The Avg/Max/Min rank columns show "
               f"how each hour compares to the other hours by that metric "
               f"(rank 1 = best). Top 5 by each metric marked with ⭐._\n")
    doc.append(
        "| Hour (UTC) | Ranked N | Avg Combined % | Max Combined % | "
        "Min Combined % | Avg Rank | Max Rank | Min Rank |")
    doc.append(
        "|---|---|---|---|---|---|---|---|")
    for h in sorted_hours:
        kpi = hour_kpis[h]
        if kpi["avg"] is None:
            doc.append(f"| {h} | {kpi['n_ranked']} | — | — | — | — | — | — |")
            continue
        avg_marker = "⭐ " if h in top5_avg else ""
        max_marker = "⭐ " if h in top5_max else ""
        min_marker = "⭐ " if h in top5_min else ""
        doc.append(
            f"| {h} | {kpi['n_ranked']} | "
            f"{kpi['avg']:.1f}% | {kpi['max']:.1f}% | {kpi['min']:.1f}% | "
            f"{avg_marker}{avg_rank[h]}/{n_with_data} | "
            f"{max_marker}{max_rank[h]}/{n_with_data} | "
            f"{min_marker}{min_rank[h]}/{n_with_data} |"
        )
    doc.append("")
    doc.append("---\n")

    doc.append("## Index of hours\n")
    for h in sorted_hours:
        n_active = len(by_hour[h])
        # Tag the index with strategy count so reader can skim
        doc.append(f"- [{h} UTC](#hour-{h.lower().replace(':','-')}-utc) — "
                   f"{n_active} strategies with activity")
    doc.append("")
    doc.append("---\n")

    for h in sorted_hours:
        section = build_hour_table(h, by_hour[h], names)
        doc.append(section)
        doc.append("---\n")

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "Strategy_Hourly_By_Hour_2026-06-06.md")
    with open(out_path, "w") as f:
        f.write("\n".join(doc))
    print(f"Wrote {out_path}")
    print(f"Total hours covered: {len(sorted_hours)}")
    total_rows = sum(len(by_hour[h]) for h in sorted_hours)
    print(f"Total strategy-hour rows: {total_rows}")


if __name__ == "__main__":
    main()
