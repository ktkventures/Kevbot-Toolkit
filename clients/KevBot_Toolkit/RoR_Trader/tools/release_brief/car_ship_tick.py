#!/usr/bin/env python3
"""Tick each shipped CAR's own ship step AND move its status -- boards #242/#249.

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

THE STEP IS NOT THE STATUS -- board #249
----------------------------------------
Ticking the step is only half the record, and shipping the half alone produced a
second wrong board: on 07-30 the dispatch page's STAGED tile read **7** under
the caption *"reviewed, waiting for a release train"* while **five of those
seven had already shipped**. Kevin read the tile against an empty shipping lane
and asked why they disagreed.

`Staged` has exactly one meaning -- *reviewed, has code, waiting for a train*.
Once the train has run that sentence is false, and every count built on `status`
inherits the lie (`waitBucket('Staged')` is a pure status count: correct code,
wrong input). The step tick records **what happened**; the status records
**where the task now is**, and only the second drives the tiles.

So a verified tick is followed by a status advance, on the SAME evidence:

  * chain finished        -> `Done`
  * anything still open   -> `Review` (the work ran; it awaits sign-off)
  * `Closed`              -> NEVER. That is Kevin's own status, after `Done`.

and under these rails:

  * The status rides on the merge proof the tick already required. There is no
    second, weaker path: if the tick is not owed, the status move is not either.
  * Only cars THIS RUN ticked, and only ticks that VERIFIED. A blanket "fix all
    Staged" sweep would rewrite tasks no train ever touched -- the cars already
    mis-stated are corrected by an explicit id list in
    `backfill_car_status_249.py`, deliberately a separate, reviewable tool.
  * Neither target is dispatchable (`Review`/`Done` sit in the API's
    `_UNDISPATCHABLE_STAGES` / `FINISHED_STATUSES`), so no status this tool
    writes can hand a shipped car back to the dispatcher queue.
  * The two-touch guard (board #136) is HONOURED, not bypassed. The status write
    goes over PostgREST because `PATCH /api/dev-tasks/<id>` is user-gated and
    401s for a service-key holder (#217), so the guard the endpoint would have
    applied is re-applied here rather than lost by the change of route.
  * Every status write is ATTRIBUTED -- a `status: X -> Y (by <actor>)` system
    comment in the same format the API writes, which is what board #248 needs.

WHAT THIS TOOL WILL NEVER DO: merge, push, rebase, run a gate, flip a flag or
touch a worktree. It fetches, reads the board, POSTs `/steps/complete` -- the
same route agents already use, so the hand-off, `completed_at` and
`completed_by` are recorded properly rather than hand-patched -- and PATCHes
`status` for the cars that route just ticked.

Usage (from the app root, `clients/KevBot_Toolkit/RoR_Trader`):

    python3 tools/release_brief/car_ship_tick.py --dry-run   # show, change nothing
    python3 tools/release_brief/car_ship_tick.py             # tick what merged
    python3 tools/release_brief/car_ship_tick.py --actor 'R-A·auto' --only 225,229

Exit codes: 0 = nothing to do or everything ticked and VERIFIED; 1 = a tick was
attempted and could not be verified, or its status write failed (loud -- hand to
M); 2 = usage/wiring error.
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

# Where a car can LAND once its ship step is ticked (board #249). Two values, and
# the set is closed on purpose:
#   Review -- the code shipped and the chain still has a step to run. That is
#             what `Review` means on this board ("the work ran, awaiting
#             sign-off", boards #136/#197) and it is what the tile says.
#   Done   -- the chain is finished; there is nothing left to wait for.
# `Closed` is ABSENT and must stay absent: it is Kevin's own status, reached from
# `Done` by his hand, and a tool that can write it can manufacture his sign-off.
STATUS_SHIPPED_OPEN = "Review"
STATUS_SHIPPED_DONE = "Done"
ALLOWED_TARGETS = (STATUS_SHIPPED_OPEN, STATUS_SHIPPED_DONE)

# Mirrors dev_tasks.FINISHED_STATUSES (board #232): a task that is already over
# is never re-stated by this sweep.
FINISHED_STATUSES = ("Done", "Closed")

# Mirrors the two-touch close guard in `PATCH /api/dev-tasks/<id>` (board #136).
# The status write cannot use that endpoint (it is user-gated; a service key gets
# 401 -- board #217), so the guard travels with the rule instead of with the
# route. Losing a refusal to a change of transport is exactly the silent
# weakening this task is about.
TWO_TOUCH_GUARDED = ("Staged", "Done")

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


def next_open_step(chain):
    """The first incomplete step of a chain, or None when it is finished."""
    for s in chain or []:
        if isinstance(s, dict) and not s.get("done"):
            return s
    return None


def target_status(chain):
    """Where a car IS once its ship step is ticked -- `Done` or `Review`.

    Read from the chain AFTER the tick, because that is the only description of
    the task's position that the tick itself just changed. Two cases, and no
    third:

      * nothing left open -> `Done`. The chain is the task's definition of
        finished, so there is nothing left to be waiting for.
      * anything left open -> `Review`. The code shipped; what remains is
        somebody looking at it. `Review` is this board's word for exactly that
        ("output done, awaiting sign-off" -- boards #136/#197), and it is the
        caption the tile carries.

    The kevin-owned close step is the COMMON case of the second, not a separate
    rule. Making it the only case would have left #242 itself sitting in
    `Staged` after Wave 26 -- its post-ship step was M's, not Kevin's -- which is
    the same lie in a less obvious costume. What matters is that the answer is
    never `Closed` (Kevin's alone) and never `Done` while a step is open.
    """
    return STATUS_SHIPPED_DONE if next_open_step(chain) is None \
        else STATUS_SHIPPED_OPEN


def plan_status(row, current, actor):
    """(target|None, reason) -- the status write owed after a VERIFIED tick.

    `row` is the task as the server returned it FROM the tick, so the chain read
    here is the post-tick one and no second board read can race it.

    None is returned for every case where no write is owed, each with the reason
    it is owed nothing -- the report prints them, because "nothing to do" and
    "refused" must not look alike.
    """
    if not isinstance(row, dict) or not isinstance(row.get("checklist"), list):
        return None, "the API returned no chain -- cannot tell where the task is"
    if current in FINISHED_STATUSES:
        return None, f"already {current} -- a finished task is not re-stated"
    target = target_status(row["checklist"])
    if target == current:
        return None, f"already {current}"
    # Two-touch (board #136): Kevin stamped "I review before Done", so only his
    # hand moves the task into Staged/Done -- except Staged -> Done, which IS the
    # release train shipping work he already signed off. Same condition the API
    # applies; re-applied because the API's route is closed to a service key.
    if (row.get("kevin_final") and target in TWO_TOUCH_GUARDED
            and current not in TWO_TOUCH_GUARDED and actor != "kevin"):
        return None, (f"two-touch task -- only Kevin moves it {current} -> "
                      f"{target}; the step is ticked, the status is his call")
    return target, f"shipped -- {current or '—'} -> {target}"


def apply_ticks(decisions, complete_step, set_status, actor):
    """POST /steps/complete for every `tick`, VERIFY it, then move the STATUS.

    complete_step(task_id, actor)            -> the updated task row (or raises)
    set_status(task_id, old, new, actor)     -> writes the status (or raises)

    ORDER IS THE RAIL. The status write happens only after the tick VERIFIED --
    not after the POST returned, and never on its own. An unverified tick means
    someone else moved the chain underneath this run, and a status written on
    that reading would be a guess. `set_status` is a required argument for the
    same reason: a caller cannot half-use this function and silently ship the
    #249 bug back.

    Returns (results, failures). A failure is never swallowed: an unverified
    tick, or a status write that did not land, means the board and this report
    disagree, and that is exactly the condition the whole task exists to stop
    being quiet about. A REFUSAL (two-touch, nothing owed) is not a failure --
    it is reported, in full, as the deliberate answer it is.
    """
    results, failures = [], []
    for d in decisions:
        if d["action"] != "tick":
            continue
        r = dict(d)
        r["status_before"] = d.get("status")
        r["status_after"] = None
        try:
            row = complete_step(d["task_id"], actor)
        except Exception as e:                       # noqa: BLE001 -- loud, not fatal
            r["result"] = "error"
            r["detail"] = f"{type(e).__name__}: {e}"
            r["status_result"] = "not attempted"
            r["status_detail"] = "the step was not ticked, so no status is owed"
            results.append(r)
            failures.append(r)
            continue
        ok, detail = verify_tick(row, d, actor)
        r["result"] = "ticked" if ok else "unverified"
        r["detail"] = detail
        r["assignee_after"] = (row or {}).get("assignee")
        if not ok:
            r["status_result"] = "not attempted"
            r["status_detail"] = ("the tick did not verify, so the status is not "
                                  "moved on it")
            results.append(r)
            failures.append(r)
            continue
        # The status the SERVER just returned, not the one this run read a
        # moment earlier -- same reason `verify_tick` re-reads the chain.
        current = row.get("status") or d.get("status")
        r["status_before"] = current
        target, why = plan_status(row, current, actor)
        r["status_detail"] = why
        if target is None:
            r["status_result"] = "unchanged"
        else:
            try:
                set_status(d["task_id"], current, target, actor)
            except Exception as e:                   # noqa: BLE001
                r["status_result"] = "error"
                r["status_detail"] = f"{type(e).__name__}: {e}"
                results.append(r)
                failures.append(r)
                continue
            r["status_result"] = "moved"
            r["status_after"] = target
        results.append(r)
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
        # The status line is printed for EVERY outcome, including the ones where
        # nothing moved (board #249): a silent status is how a shipped car went
        # on reading `Staged` for a week.
        if r and r.get("status_result"):
            smark = {"moved": "status", "unchanged": "status --",
                     "error": "status !!", "not attempted": "status --"}.get(
                         r["status_result"], "status")
            arrow = (f" {r['status_before'] or '—'} -> {r['status_after']}"
                     if r["status_result"] == "moved" else "")
            out.append(f"        {smark}{arrow} ({r.get('status_detail')})")
        elif not r:
            out.append(f"        status: would move (dry run) from "
                       f"{d.get('status') or '—'}")
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


def live_set_status(task_id, old, new, actor):
    """PATCH the status, VERIFY it stuck, and ATTRIBUTE it on the thread.

    Over PostgREST rather than `PATCH /api/dev-tasks/<id>`, which is user-gated
    and 401s for a service-key holder (board #217) -- the same constraint that
    put `/steps/complete` behind `get_service_or_user`. Everything the endpoint
    would have done is therefore done here EXPLICITLY, never dropped:

      * the two-touch refusal, in `plan_status` (board #136);
      * `ai_eligible=false` on a finished status, so the board never carries a
        `Done` + armed row (board #198's disarm-on-close);
      * the `status: X -> Y (by <actor>)` system comment, in the API's own
        wording, which is the audit trail board #248 is about. Written HERE
        rather than left to the reader to infer.

    Then it reads the row back. A PATCH that matched no row returns 204 exactly
    like one that matched -- reporting a status move that never happened is the
    failure mode this whole task exists to kill, so it is checked, not assumed.
    """
    body = {"status": new}
    if new in FINISHED_STATUSES:
        body["ai_eligible"] = False
    B.api("PATCH", f"dev_tasks?id=eq.{task_id}", body=body)
    rows = B.api("GET", f"dev_tasks?id=eq.{task_id}&select=id,status") or []
    got = rows[0].get("status") if rows else None
    if got != new:
        raise RuntimeError(
            f"status write did not land -- #{task_id} reads {got!r}, not {new!r}")
    B.api("POST", "dev_task_comments",
          body={"task_id": task_id, "author": "system",
                "body": f"status: {old or '—'} → {new or '—'} (by {actor})"})


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
          set_status, actor, dry_run=False, only=None):
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
    results, failures = apply_ticks(decisions, complete_step, set_status, actor)
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
            set_status=live_set_status,
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
        print(f"\n--- {len(failures)} car(s) could NOT be completed: the tick did "
              "not verify, or its status write did not land -- the board and this "
              "report disagree. Hand to M; do not re-run blindly. ---",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
