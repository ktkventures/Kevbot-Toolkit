#!/usr/bin/env python3
"""ONE-OFF: correct the eight cars that shipped but still read `Staged` -- #249.

`car_ship_tick.py` ticked each car's ship step and never moved its status, so
eight tasks that merged into `dev` still carry `status = 'Staged'` -- *"reviewed,
waiting for a release train"* -- and the dispatch page counts them as awaiting a
train that already ran. The sweep is fixed forward; these eight are the cars it
mis-stated before the fix existed, and nothing else will ever correct them.

WHY A NAMED LIST AND NOT A RULE
-------------------------------
The obvious shortcut -- "move every `Staged` task whose ship step is ticked" --
is a rule, and a rule runs against rows nobody checked. `Staged` is a legitimate
status: it is where a car SHOULD sit while it waits for its train, and the board
also carries `Staged` tasks that are not cars at all (two sit in the E lane
right now as parked work). A rule that walked the board would rewrite them the
first time it ran and again every time anyone re-ran it.

So this tool is a LIST. Every id below was read on the board on 07-31, confirmed
merged and confirmed mis-stated. It cannot touch anything else: the ids are the
query, not a filter applied after one.

  #238  Release brief: PROBE the service after deploy-watch
  #240  M-Session Dashboard: bound the activity log
  #242  A release train never ticks its CARS' ship steps   (the sweep itself)
  #243  Shipping lane shows the BUILDER, not the assignee
  #252  New top-level `Team` container
  #257  Kevin (owner) dashboard
  #261  Task-type model: explicit `task_type`
  #262  Goal task type: params, rules and the context block

WHAT IT REFUSES
---------------
Per car, and out loud:
  * not in the list                     -- unreachable by construction
  * ship step not ticked                -- then it did NOT ship, and the record
                                           that says so is the tick itself. This
                                           tool has no git proof of its own and
                                           does not pretend to: it corrects the
                                           STATUS of cars whose tick already
                                           passed the merge gate, and refuses
                                           anything else.
  * status is not `Staged`              -- the world moved; the mis-statement
                                           this was written for is gone
  * two-touch (board #136)              -- Kevin's stamp still governs the close
Target status comes from `car_ship_tick.plan_status` -- the SAME rule the fixed
sweep uses, so the backfill cannot drift from it.

DRY RUN BY DEFAULT. `--apply` writes; anything else shows and changes nothing.

    python3 tools/release_brief/backfill_car_status_249.py            # show
    python3 tools/release_brief/backfill_car_status_249.py --apply    # write

Exit codes: 0 = clean (or nothing left to do); 1 = a write failed; 2 = wiring.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import brief_gen as B          # noqa: E402
import car_ship_tick as C      # noqa: E402

# The eight, by id. Read from the board 07-31; every one Staged, ship step
# ticked, next step Kevin's. NOT a rule -- see the module docstring.
BACKFILL_IDS = (238, 240, 242, 243, 252, 257, 261, 262)

# The one status this tool will correct FROM. A car that has moved on since the
# list was written is somebody else's business.
EXPECTED_STATUS = "Staged"


def shipped_step_index(task):
    """Index of the ship step IF it is already ticked, else None.

    Deliberately NOT `car_ship_tick.ship_step_index` -- that one asks "is the
    ship step the next OPEN step" (a car waiting for a train). Here the question
    is the opposite: has the ship step already been completed. Same owners, same
    chain, mirror-image question.
    """
    for i, s in enumerate(task.get("checklist") or []):
        if not isinstance(s, dict):
            continue
        if (B._step_owner(s) or "").upper() not in C.SHIP_OWNERS:
            continue
        if s.get("done"):
            return i
    return None


def plan_backfill(tasks, actor, allowed=BACKFILL_IDS):
    """[decision] -- one per NAMED id, in id order. Nothing else is considered."""
    out = []
    for t in sorted((t for t in tasks if t["id"] in allowed),
                    key=lambda t: t["id"]):
        d = {"task_id": t["id"], "title": t.get("title") or "",
             "status": t.get("status"), "target": None}
        idx = shipped_step_index(t)
        if idx is None:
            d["action"] = "skip"
            d["reason"] = ("no TICKED ship step -- there is no record that it "
                           "shipped, so its status is not this tool's to move")
        elif t.get("status") != EXPECTED_STATUS:
            d["action"] = "skip"
            d["reason"] = (f"status is {t.get('status')!r}, not "
                           f"{EXPECTED_STATUS!r} -- already corrected, or moved "
                           "on since the list was written")
        else:
            target, why = C.plan_status(t, t.get("status"), actor)
            if target is None:
                d["action"] = "skip"
                d["reason"] = why
            else:
                d["action"] = "set"
                d["target"] = target
                d["reason"] = why
        out.append(d)
    return out


def apply_backfill(decisions, set_status, actor):
    """Write each `set`, one at a time. Returns (results, failures)."""
    results, failures = [], []
    for d in decisions:
        if d["action"] != "set":
            continue
        r = dict(d)
        try:
            set_status(d["task_id"], d["status"], d["target"], actor)
        except Exception as e:                       # noqa: BLE001 -- loud
            r["result"] = "error"
            r["detail"] = f"{type(e).__name__}: {e}"
            failures.append(r)
        else:
            r["result"] = "moved"
            r["detail"] = f"{d['status']} → {d['target']}"
        results.append(r)
    return results, failures


def render(decisions, results, apply):
    sets = [d for d in decisions if d["action"] == "set"]
    by_id = {r["task_id"]: r for r in results}
    out = [f"#249 backfill -- {len(BACKFILL_IDS)} named car(s): "
           f"{len(sets)} to correct, {len(decisions) - len(sets)} already fine"]
    for d in decisions:
        r = by_id.get(d["task_id"])
        if d["action"] != "set":
            mark, tail = "--  ", d["reason"]
        elif r is None:
            mark, tail = "DRY ", f"{d['reason']} (dry run -- nothing written)"
        else:
            mark = "OK  " if r["result"] == "moved" else "!!  "
            tail = r["detail"]
        out.append(f"  {mark}#{d['task_id']} [{d['status']}] "
                   f"{d['title'][:52]} -- {tail}")
    if sets and not apply:
        out.append("  (dry run -- re-run with --apply to write)")
    return "\n".join(out)


def live_tasks(ids=BACKFILL_IDS):
    q = (f"dev_tasks?id=in.({','.join(str(i) for i in ids)})"
         "&select=id,title,status,assignee,kevin_final,checklist")
    return B.api("GET", q) or []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--actor", default="M-A·auto",
                    help="who to record on the status change (default M-A·auto)")
    ap.add_argument("--apply", action="store_true",
                    help="write the corrections (default: show only)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    try:
        decisions = plan_backfill(live_tasks(), args.actor)
        results, failures = (apply_backfill(decisions, C.live_set_status,
                                            args.actor)
                             if args.apply else ([], []))
    except Exception as e:                           # noqa: BLE001
        print(f"backfill_car_status_249: could not run -- {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"decisions": decisions, "results": results,
                          "failures": failures}, indent=1, ensure_ascii=False))
    else:
        print(render(decisions, results, args.apply))

    if failures:
        print(f"\n--- {len(failures)} status write(s) failed -- hand to M ---",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
