#!/usr/bin/env python3
"""Tick each shipped CAR's own ship step -- board #242.

A release train is its own board task with its own chain, and it ticks **its
own** steps. Nothing ticks the CARS'. So every car a train merges keeps an open
`Ship via a release train` step, holds `Staged`/`Review`, and renders in the
shipping lane exactly like work that never went anywhere.

Observed three times: after Wave 23 (#217, #218, #224), Wave 24 (#220, #228,
#232, #234, #235) and Wave 25 (#225, #229, #239). M hand-ticked each time, and
on the third **Kevin saw the stale lane before M did** -- which is the real
cost: the board stopped being trustworthy at a glance.

`brief_gen.py` already refuses an already-merged car loudly ("it shipped on an
earlier train and its R step was never ticked"). That is a genuine safety net
but a LATE one -- it fires at the NEXT train, and between trains the lane is
wrong. This tool closes the gap where it opens.

WHY A POST-MERGE SWEEP, and not a tick interleaved into the merge sequence
--------------------------------------------------------------------------
Three placements were on the table (task #242 step 2). This is a sweep the
train runs once the merges have landed, and the reasoning is:

  1. **The tick must FOLLOW the merge, never the intent.** This tool never
     takes anyone's word for what shipped: every candidate is gated on
     `git merge-base --is-ancestor <branch> origin/dev` after a fresh fetch. A
     car that failed to merge, or that R-A *meant* to merge and aborted on,
     cannot be ticked -- there is no code path that ticks an unmerged branch.
     An interleaved tick is a write driven by intent at exactly the moment the
     merge sequence is most likely to break.
  2. **The merge sequence is where a train ABORTS**, and board writes wedged
     between merges are the fragile thing to be doing there. A sweep is
     idempotent and re-runnable, so it is also safe on the abort path: run it
     after a partial train and it ticks precisely the cars that really landed.
  3. **`brief_gen` reporting alone is too late** -- that is the existing net,
     and it is the one that already failed three times.

The sweep needs NO argument list, deliberately. It re-derives the candidate set
from the board the same way `brief_gen` derives cars (first incomplete step is
R/R-A-owned) and resolves branches with the same resolver, so it cannot be fed
a wrong list of "what shipped". Truth comes from git, twice: the branch, and
whether dev contains it.

WHAT THIS TOOL WILL NEVER DO: merge, push, rebase, run a gate, flip a flag or
touch a worktree. It fetches, reads the board, and POSTs `/steps/complete` --
the same route agents already use, so the hand-off, `completed_at` and
`completed_by` are recorded properly rather than hand-patched.

Usage (from the app root, `clients/KevBot_Toolkit/RoR_Trader`):

    python3 tools/release_brief/car_ship_tick.py --dry-run   # show, change nothing
    python3 tools/release_brief/car_ship_tick.py             # tick what merged
    python3 tools/release_brief/car_ship_tick.py --actor 'R-A·auto' --only 225,229

Exit codes: 0 = nothing to do or everything ticked and VERIFIED; 1 = a tick was
attempted and could not be verified (loud -- hand to M); 2 = usage/wiring error.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import brief_gen as B  # noqa: E402

# A ship step is owned by the release lane. BOTH letters count: `R` is the
# session and `R-A` the headless executor (charter §1, board #229), and real
# chains carry both -- #225/#229 name `R-A`, older ones name `R`. Matching only
# "R" would make the newer convention invisible, which is the same silent-drop
# shape this whole task is about.
SHIP_OWNERS = ("R", "R-A")

# Statuses a shipped-but-unticked car can be sitting in. The merge gate is what
# actually protects this tool, so the scan is deliberately WIDER than
# `brief_gen`'s `Staged`-only entry point: the stale cars of Waves 23-25 sat in
# `Staged` AND `Review`, and a car cannot be ticked from any status unless dev
# already contains its branch.
SCAN_STATUSES = ("Staged", "Review", "In Progress", "Todo", "Blocked")

API_TIMEOUT_S = 20


# ---------------------------------------------------------------------------
# Pure core -- every external fact is injected, so the rails run with no board,
# no git and no network. See test_car_ship_tick.py.
# ---------------------------------------------------------------------------

def ship_step_index(task):
    """Index of the FIRST INCOMPLETE step when that step is ship-owned, else None.

    Reading the first incomplete step (rather than "any R step") is what makes
    the answer mean *reviewed, waiting on a train* -- every earlier step is
    ticked by construction. It is also a hard safety requirement here and not
    just a definition: `/steps/complete` ticks whatever the first incomplete
    step IS. If a chain still has an open M step in front of its R step, asking
    the API to complete "the ship step" would silently complete the M step
    instead. So the two must be the same index or we do not call at all.
    """
    cl = task.get("checklist") or []
    if not isinstance(cl, list):
        return None
    for i, s in enumerate(cl):
        if not isinstance(s, dict):
            continue
        if s.get("done"):
            continue
        owner = (B._step_owner(s) or "").upper()
        return i if owner in SHIP_OWNERS else None
    return None


def plan_ticks(tasks, resolve_branch, is_merged):
    """[decision] for every task whose CURRENT step is an open ship step.

    resolve_branch(task) -> (branch|None, source)
    is_merged(branch)    -> True when origin/dev already contains the branch tip

    Tasks with no open ship step are not decisions at all and are omitted --
    this tool has no opinion about them. Among the rest:

      tick -- branch resolved AND merged into dev. It shipped; the step is owed.
      skip -- branch not merged (the normal case: a car waiting for its train),
              or unresolvable (say so; never guess).
    """
    out = []
    for t in sorted(tasks, key=lambda t: t["id"]):
        idx = ship_step_index(t)
        if idx is None:
            continue
        step = (t.get("checklist") or [])[idx]
        d = {
            "task_id": t["id"],
            "title": t.get("title") or "",
            "status": t.get("status"),
            "step_index": idx,
            "step_number": idx + 1,
            "step_title": B._step_title(step),
            "step_owner": B._step_owner(step),
            "branch": None,
            "branch_source": None,
        }
        branch, src = resolve_branch(t)
        d["branch"], d["branch_source"] = branch, src
        if not branch:
            d["action"] = "skip"
            d["reason"] = ("no branch found in run_history.pushed_branch, the task "
                           "thread, or local branches -- cannot prove it shipped")
        elif not is_merged(branch):
            d["action"] = "skip"
            d["reason"] = (f"`{branch}` is NOT an ancestor of {B.BASE} -- it has not "
                           "shipped, so its ship step stays open")
        else:
            d["action"] = "tick"
            d["reason"] = f"`{branch}` is already merged into {B.BASE}"
        out.append(d)
    return out


def verify_tick(row, decision, actor):
    """(ok, detail) -- did the POST tick the step we MEANT to tick?

    `/steps/complete` completes the first incomplete step as the SERVER sees it,
    which is not necessarily the one this process read a moment earlier: another
    agent can tick in between. Re-reading the returned row closes that race
    loudly instead of letting the tool report a tick it did not make.
    """
    if not isinstance(row, dict):
        return False, "the API returned no task row"
    cl = row.get("checklist") or []
    i = decision["step_index"]
    if i >= len(cl) or not isinstance(cl[i], dict):
        return False, f"step {i + 1} is missing from the returned chain"
    step = cl[i]
    if not step.get("done"):
        return False, (f'step {i + 1} "{B._step_title(step)}" is still OPEN after '
                       "the call -- a different step was completed")
    by = step.get("completed_by")
    if by != actor:
        return False, (f'step {i + 1} is done but completed_by={by!r}, not {actor!r} '
                       "-- it was already ticked by someone else")
    return True, f'step {i + 1} "{B._step_title(step)}" ticked by {actor}'


def apply_ticks(decisions, complete_step, actor):
    """POST /steps/complete for every `tick`, then VERIFY each one.

    complete_step(task_id, actor) -> the updated task row (or raises)

    Returns (results, failures). A failure is never swallowed: an unverified
    tick means the board and this report disagree, and that is exactly the
    condition the whole task exists to stop being quiet about.
    """
    results, failures = [], []
    for d in decisions:
        if d["action"] != "tick":
            continue
        r = dict(d)
        try:
            row = complete_step(d["task_id"], actor)
        except Exception as e:                       # noqa: BLE001 -- loud, not fatal
            r["result"] = "error"
            r["detail"] = f"{type(e).__name__}: {e}"
            results.append(r)
            failures.append(r)
            continue
        ok, detail = verify_tick(row, d, actor)
        r["result"] = "ticked" if ok else "unverified"
        r["detail"] = detail
        r["assignee_after"] = (row or {}).get("assignee")
        results.append(r)
        if not ok:
            failures.append(r)
    return results, failures


def render(decisions, results, dry_run):
    ticks = [d for d in decisions if d["action"] == "tick"]
    skips = [d for d in decisions if d["action"] == "skip"]
    out = [f"car ship-step sweep -- {len(decisions)} open ship step(s): "
           f"{len(ticks)} shipped, {len(skips)} still waiting"]
    by_id = {r["task_id"]: r for r in results}
    for d in ticks:
        r = by_id.get(d["task_id"])
        mark = {"ticked": "OK  ", "unverified": "!!  ", "error": "!!  "}.get(
            (r or {}).get("result"), "DRY ")
        tail = f" -- {r['detail']}" if r else f" -- {d['reason']} (dry run)"
        out.append(f"  {mark}#{d['task_id']} step {d['step_number']} "
                   f"\"{d['step_title']}\" [{d['branch']}]{tail}")
        if r and r.get("assignee_after"):
            out.append(f"        handed off to {r['assignee_after']}")
    for d in skips:
        out.append(f"  --  #{d['task_id']} step {d['step_number']} "
                   f"\"{d['step_title']}\" -- {d['reason']}")
    if dry_run and ticks:
        out.append("  (dry run -- nothing was written)")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Live wiring
# ---------------------------------------------------------------------------

def board_api_host():
    return os.getenv("RORT_BOARD_API_URL", B.API_HOST).rstrip("/")


def service_key():
    """`RORT_BOARD_SERVICE_KEY` (exported into every dispatched run) or `src/.env`."""
    key = os.getenv("RORT_BOARD_SERVICE_KEY")
    if key:
        return key
    return B._creds()[1]


def live_complete_step(task_id, actor):
    import urllib.request
    url = f"{board_api_host()}/api/dev-tasks/{task_id}/steps/complete"
    req = urllib.request.Request(
        url, data=json.dumps({"actor": actor}).encode(),
        headers={"Authorization": f"Bearer {service_key()}",
                 "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as r:
        txt = r.read().decode()
    return json.loads(txt) if txt.strip() else None


def live_tasks(only=None):
    # `In Progress` has a space, which urllib refuses to put in a URL. Encode the
    # value rather than dropping the status -- a shipped car sitting In Progress
    # is exactly the kind of row this sweep must still see.
    from urllib.parse import quote
    statuses = quote(",".join(f'"{s}"' for s in SCAN_STATUSES), safe=',"')
    q = (f"dev_tasks?status=in.({statuses})"
         "&select=id,title,status,assignee,checklist")
    if only:
        q += f"&id=in.({','.join(str(i) for i in only)})"
    return B.api("GET", q) or []


def sweep(fetch, load_tasks, resolve_branch, is_merged, complete_step,
          actor, dry_run=False, only=None):
    """Fetch FIRST, then decide, then act.

    The fetch is load-bearing and is not a tidiness step: the cars this sweep
    exists for were merged SECONDS ago, and against a stale `origin/dev` ref
    every one of them reads as unmerged. The failure would be silent -- a clean
    "0 shipped" report on a train that just shipped five cars -- which is the
    exact shape of the bug being fixed. So it happens before any merge check,
    unconditionally, including on a dry run.
    """
    fetch()
    decisions = plan_ticks(load_tasks(only), resolve_branch, is_merged)
    if dry_run:
        return decisions, [], []
    results, failures = apply_ticks(decisions, complete_step, actor)
    return decisions, results, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--actor", default="R-A·auto",
                    help="who to record as completing the step (default R-A·auto)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be ticked; write nothing")
    ap.add_argument("--only", default="",
                    help="restrict to these board ids (comma-separated)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    only = [int(x) for x in args.only.replace(" ", "").split(",") if x] or None

    try:
        decisions, results, failures = sweep(
            fetch=lambda: B.git("fetch", "origin", "--quiet"),
            load_tasks=live_tasks,
            resolve_branch=B.live_resolve_branch,
            is_merged=B.live_already_merged,
            complete_step=live_complete_step,
            actor=args.actor, dry_run=args.dry_run, only=only)
    except Exception as e:                           # noqa: BLE001
        print(f"car_ship_tick: could not run -- {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"decisions": decisions, "results": results,
                          "failures": failures}, indent=1, ensure_ascii=False))
    else:
        print(render(decisions, results, args.dry_run))

    if failures:
        print(f"\n--- {len(failures)} tick(s) could NOT be verified -- the board and "
              "this report disagree. Hand to M; do not re-run blindly. ---",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
