"""Build a doc with hourly tables for every active strategy.

For each strategy in the cohort, produce a markdown table with every
hour that had any activity (alerts or BT events) since 2026-05-28,
plus a per-strategy rank column showing which hour was closest to
100% combined paired-rate.

Output: docs/Strategy_Hourly_Tables_2026-06-06.md
"""
from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cohort: same 43 active strategies from the fleet analysis
COHORT = [
    136, 174, 194, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275,
    276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
    290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
    303, 304, 305,
]


def load_data():
    """Return {sid_str: hours_list} merging fleet + lastweek where available."""
    with open("/tmp/fleet_hourly_may28_jun5/all_strategies.json") as f:
        fleet = json.load(f)
    # The lastweek data only has UT Bot strategies and extends to 5/25-5/31.
    # For sid 268 etc., it has 5/29 hours that fleet doesn't (because fleet
    # window starts 5/28 but BT data for those sids was rebuilt by UAD and
    # the lastweek data captured pre-UAD state).
    try:
        with open("/tmp/utbot_hourly_lastweek/all_strategies.json") as f:
            lastweek = json.load(f)
    except FileNotFoundError:
        lastweek = {}

    merged = {}
    for sid in COHORT:
        sid_str = str(sid)
        hours = {}
        # Start with lastweek if present (older, may have data not in fleet)
        if sid_str in lastweek:
            for r in lastweek[sid_str]["hours"]:
                if r["alerts"] >= 1 or r["bt_events"] >= 1:
                    hours[r["hour_utc"]] = dict(r)
        # Overlay/extend with fleet (newer, more complete for 5/28+)
        if sid_str in fleet:
            for r in fleet[sid_str]["hours"]:
                h = r["hour_utc"]
                if r["alerts"] == 0 and r["bt_events"] == 0:
                    continue
                if h not in hours:
                    hours[h] = dict(r)
                else:
                    # Prefer source with more bt_events
                    existing = hours[h]
                    if r["bt_events"] > existing["bt_events"]:
                        hours[h] = dict(r)
        merged[sid_str] = hours
    return merged


def build_ranks(hours_dict):
    """Add rank field. Rank only hours where alerts>=1 AND bt_events>=1
    (meaningful pair attempt). Other hours get rank None."""
    rankable = []
    for h, r in hours_dict.items():
        if r["alerts"] >= 1 and r["bt_events"] >= 1:
            denom = r["paired"] + r["phantom"] + r["missed"]
            cpct = (100 * r["paired"] / denom) if denom > 0 else 0.0
            rankable.append((h, cpct))
    # Sort descending by combined %, then by alerts (more alerts = more
    # confident measurement) so a 100% hour with 5 alerts ranks below a
    # 95% hour with 50 alerts… actually no, Kevin wants raw ranking by
    # combined %. Tie-break by alerts descending so high-volume hours
    # at the same % rank higher.
    rankable.sort(key=lambda x: (-x[1], -hours_dict[x[0]]["alerts"]))
    rank_map = {}
    for i, (h, _) in enumerate(rankable, 1):
        rank_map[h] = i
    return rank_map


def build_strategy_table(sid: int, hours_dict: dict, strategy_meta: dict) -> str:
    """Build the markdown for one strategy."""
    if not hours_dict:
        return (f"## sid {sid}\n\n"
                f"No data in range.\n")

    rank_map = build_ranks(hours_dict)
    n_ranked = len(rank_map)

    name = strategy_meta.get("name", "(unknown)")
    out = []
    out.append(f"## sid {sid} — {name}\n")
    out.append(f"_Total hours with activity: {len(hours_dict)}; "
               f"ranked hours (alerts≥1, BT≥1): {n_ranked}_\n")
    out.append("")
    out.append(f"| Hour (UTC) | Alerts | BT events | Paired | Phantom | "
               f"Missed | Combined % | Alert-pair % | Rank |")
    out.append(f"|---|---|---|---|---|---|---|---|---|")
    for h in sorted(hours_dict.keys()):
        r = hours_dict[h]
        a = r["alerts"]
        bt = r["bt_events"]
        p = r["paired"]
        ph = r["phantom"]
        m = r["missed"]
        denom = p + ph + m
        cpct = (100 * p / denom) if denom > 0 else 0.0
        a_denom = p + ph
        apr = (100 * p / a_denom) if a_denom > 0 else 0.0
        rank = rank_map.get(h)
        rank_str = f"{rank}/{n_ranked}" if rank else "—"
        out.append(f"| {h} | {a} | {bt} | {p} | {ph} | {m} | "
                   f"{cpct:.1f}% | {apr:.1f}% | {rank_str} |")
    out.append("")
    return "\n".join(out)


def main():
    # Load strategy names
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    from db import get_admin_client
    c = get_admin_client()
    names = {}
    for sid in COHORT:
        r = (c.table("strategies")
             .select("name").eq("id", sid).single().execute())
        names[sid] = (r.data.get("name") if r.data else "(unknown)") or "(unknown)"

    merged = load_data()

    # Build the doc
    doc = []
    doc.append("# Per-Strategy Hourly Tables — 2026-06-06\n")
    doc.append(
        "> Hourly pair-rate tables for every active strategy since "
        "2026-05-28. Pair window: ±5s. Combined % = paired / (paired + "
        "phantom + missed). Alert-pair % = paired / (paired + phantom). "
        "Rank: each strategy's hours sorted by Combined %, where rank 1 "
        "is that strategy's cleanest hour. Hours with no alerts AND no "
        "BT events are omitted. Hours with alerts=0 OR BT=0 are shown "
        "but unranked (no meaningful pair attempt).\n"
    )
    doc.append("## Index\n")
    for sid in COHORT:
        nm = names.get(sid, "(unknown)")[:80]
        doc.append(f"- [sid {sid}](#sid-{sid}-{nm.lower().replace(' ', '-').replace(chr(46), '').replace(chr(183), '').replace(chr(58), '')[:50]}) — {nm}")
    doc.append("")
    doc.append("---\n")

    for sid in COHORT:
        sid_str = str(sid)
        meta = {"name": names.get(sid, "(unknown)")}
        section = build_strategy_table(sid, merged.get(sid_str, {}), meta)
        doc.append(section)
        doc.append("---\n")

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "Strategy_Hourly_Tables_2026-06-06.md")
    with open(out_path, "w") as f:
        f.write("\n".join(doc))
    print(f"Wrote {out_path}")
    print(f"Total strategies covered: {len(COHORT)}")
    total_hours = sum(len(merged.get(str(s), {})) for s in COHORT)
    print(f"Total hour rows: {total_hours}")


if __name__ == "__main__":
    main()
