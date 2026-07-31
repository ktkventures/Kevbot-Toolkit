#!/usr/bin/env python3
"""Rails for the car ship-step sweep (boards #242 and #249).

Hermetic: no database, no git, no network. `sweep()` takes its six external
facts as callables, so every rail runs against fixtures.

Every rail has a NEGATIVE test -- one that removes the rail's precondition and
asserts the answer CHANGES. The load-bearing pair is:

  * a merged car ends TICKED, with the assignee advanced AND its status moved
    off `Staged` (rails 1 and 11);
  * a car whose merge ABORTED gets NEITHER -- no tick, no status write -- and no
    amount of intent changes that (rails 2 and 12).

Run:  python3 tools/release_brief/test_car_ship_tick.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_ship_tick as C  # noqa: E402

PASS = FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        # `detail` is routinely a decision list, so stringify rather than
        # concatenate -- a rail that fails must PRINT why, not raise TypeError
        # inside the reporter and hide it behind "<test> raised".
        FAILURES.append(f"{name}{(' -- ' + str(detail)) if detail else ''}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def step(owner, title, done=True):
    return {"owner": owner, "title": title, "done": done,
            "id": f"s{abs(hash((owner, title))) % 10 ** 6}"}


def task(tid, title="a car", steps=None, status="Staged", kevin_final=False):
    return {"id": tid, "title": title, "status": status, "assignee": "R-A",
            "kevin_final": kevin_final,
            "checklist": steps if steps is not None else shipped_chain()}


def shipped_chain(ship_owner="R-A"):
    """Reviewed and waiting on a train -- the shape every car has."""
    return [
        step("kevin", "Approve"),
        step("M-A", "Build it"),
        step("M", "M review, then ship"),
        step(ship_owner, "Ship via a release train", done=False),
        step("M", "Verify it fires", done=False),
    ]


class Board:
    """A tiny stand-in for the board + the /steps/complete route.

    Reproduces the ONE behaviour the sweep's safety depends on: the route ticks
    the FIRST INCOMPLETE step, whichever that is -- not the one the caller had
    in mind.
    """

    def __init__(self, tasks):
        self.tasks = {t["id"]: t for t in tasks}
        self.calls = []
        self.status_calls = []           # (task_id, old, new, actor)

    def load(self, only=None):
        rows = list(self.tasks.values())
        return [t for t in rows if only is None or t["id"] in only]

    def complete(self, task_id, actor):
        self.calls.append((task_id, actor))
        t = self.tasks[task_id]
        for s in t["checklist"]:
            if not s.get("done"):
                s["done"] = True
                s["completed_at"] = "2026-07-30T00:00:00Z"
                s["completed_by"] = actor
                break
        nxt = [s for s in t["checklist"] if not s.get("done")]
        t["assignee"] = (nxt[0]["owner"] if nxt else "M")
        # `/steps/complete` moves the chain and the assignee and NOTHING else --
        # in particular it does not touch `status`. That is board #249's defect
        # reproduced in the fixture: without the sweep's own status write, a
        # shipped car stays `Staged` here exactly as it did on the real board.
        return t

    def set_status(self, task_id, old, new, actor):
        self.status_calls.append((task_id, old, new, actor))
        self.tasks[task_id]["status"] = new


def run_sweep(board, merged=(), branches=None, actor="R-A·auto",
              dry_run=False, only=None, fetch=None, set_status=None):
    branches = branches if branches is not None else {
        t["id"]: f"feat/thing-{t['id']}" for t in board.load()}
    fetched = {"n": 0}

    def _fetch():
        fetched["n"] += 1
        if fetch:
            fetch()

    def _merged(b):
        # Mirrors `git merge-base --is-ancestor` reading a ref that only knows
        # what the last fetch told it.
        if fetched["n"] == 0:
            raise AssertionError("merge check ran BEFORE the fetch")
        return b in merged

    d, r, f = C.sweep(_fetch, board.load,
                      lambda t: (branches.get(t["id"]), "fixture"),
                      _merged, board.complete, set_status or board.set_status,
                      actor, dry_run=dry_run, only=only)
    return d, r, f, fetched["n"]


def by_id(rows):
    return {r["task_id"]: r for r in rows}


# ---------------------------------------------------------------------------
# RAIL 1 -- a merged car ends TICKED, with the hand-off recorded.
# ---------------------------------------------------------------------------

def t_merged_car_is_ticked():
    b = Board([task(225)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-225"})
    check("merged car is planned as a tick", by_id(d)[225]["action"] == "tick", d)
    check("merged car reports ticked", by_id(r)[225]["result"] == "ticked", r)
    check("merged car has no failures", not f, f)

    chain = b.tasks[225]["checklist"]
    check("the SHIP step is the one that got ticked", chain[3]["done"] is True)
    check("completed_by is recorded", chain[3]["completed_by"] == "R-A·auto",
          chain[3])
    check("completed_at is recorded", bool(chain[3].get("completed_at")))
    check("the task ADVANCED to the next step's owner",
          b.tasks[225]["assignee"] == "M", b.tasks[225]["assignee"])
    check("the tick is reported as a hand-off",
          by_id(r)[225]["assignee_after"] == "M", r)
    # Board #249 -- the other half of the same record. Ticking alone left the
    # card reading `Staged` ("waiting for a release train") after the train ran.
    check("the STATUS moved off Staged as well",
          b.tasks[225]["status"] == "Review", b.tasks[225]["status"])
    check("the status move is reported", by_id(r)[225]["status_result"] == "moved"
          and by_id(r)[225]["status_after"] == "Review", r)

    body = C.render(d, r, dry_run=False)
    check("the report names the ticked car", "#225" in body and "OK" in body, body)


def t_negative_no_sweep_leaves_it_open():
    """Without the sweep, the shipped car keeps its open ship step -- the bug."""
    b = Board([task(225)])
    check("a shipped car starts with an OPEN ship step",
          b.tasks[225]["checklist"][3]["done"] is False)
    check("...and stays open with no sweep",
          b.tasks[225]["checklist"][3]["done"] is False)


# ---------------------------------------------------------------------------
# RAIL 2 -- THE tick follows the MERGE, never the intent. A car whose merge
# aborted is not ticked, and there is no code path that can tick it.
# ---------------------------------------------------------------------------

def t_unmerged_car_is_never_ticked():
    b = Board([task(240)])
    d, r, f, _ = run_sweep(b, merged=set())          # the merge aborted
    check("unmerged car is planned as a SKIP", by_id(d)[240]["action"] == "skip", d)
    check("unmerged car names why", "NOT an ancestor" in by_id(d)[240]["reason"],
          d)
    check("no /steps/complete call was made for it", b.calls == [], b.calls)
    check("its ship step is STILL OPEN",
          b.tasks[240]["checklist"][3]["done"] is False)
    check("assignee unchanged", b.tasks[240]["assignee"] == "R-A")
    # Board #249 -- THE rail the SOP names: an unmerged branch gets NEITHER.
    check("no status write was made for it", b.status_calls == [], b.status_calls)
    check("its status is untouched", b.tasks[240]["status"] == "Staged",
          b.tasks[240]["status"])
    check("nothing failed -- an unmerged car is normal, not an error", not f, f)


def t_partial_train_ticks_only_what_landed():
    """Cars 1-2 merged, car 3 conflicted and the train aborted."""
    b = Board([task(301), task(302), task(303)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-301", "feat/thing-302"})
    check("exactly the two merged cars are ticked",
          sorted(x["task_id"] for x in r if x["result"] == "ticked") == [301, 302],
          r)
    check("the aborted car keeps its open ship step",
          b.tasks[303]["checklist"][3]["done"] is False)
    check("only the merged cars were POSTed",
          sorted(c[0] for c in b.calls) == [301, 302], b.calls)


def t_negative_intent_cannot_tick():
    """The only input that can produce a tick is the MERGE check.

    Same board, same branches, same everything -- flip only `is_merged` and the
    action flips with it. If the tool ever ticked on intent (a car list, a
    brief, an agent's say-so), this test could not distinguish the two runs.
    """
    b1, b2 = Board([task(310)]), Board([task(310)])
    d1, _, _, _ = run_sweep(b1, merged=set())
    d2, _, _, _ = run_sweep(b2, merged={"feat/thing-310"})
    check("merge state alone decides the action",
          (by_id(d1)[310]["action"], by_id(d2)[310]["action"]) == ("skip", "tick"),
          f"{d1} / {d2}")


# ---------------------------------------------------------------------------
# RAIL 3 -- never complete a step that is not the current one.
# `/steps/complete` ticks the FIRST incomplete step, so a chain with an open M
# step in front of its ship step must not be touched at all.
# ---------------------------------------------------------------------------

def t_open_step_in_front_is_not_touched():
    chain = shipped_chain()
    chain[2]["done"] = False                          # M review not ticked
    b = Board([task(320, steps=chain)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-320"})
    check("a chain mid-review yields NO decision", d == [], d)
    check("no call was made", b.calls == [], b.calls)
    check("the M review step is still open", b.tasks[320]["checklist"][2]["done"]
          is False)


def t_negative_wrong_step_would_be_ticked():
    """Proof of what rail 3 prevents: complete() ticks the M step, not the ship step."""
    chain = shipped_chain()
    chain[2]["done"] = False
    b = Board([task(321, steps=chain)])
    b.complete(321, "R-A·auto")                       # what a naive tool would do
    check("the API would have ticked the M REVIEW step",
          b.tasks[321]["checklist"][2]["done"] is True
          and b.tasks[321]["checklist"][3]["done"] is False,
          b.tasks[321]["checklist"])


def t_no_ship_step_is_no_decision():
    chain = [step("kevin", "Approve"), step("M", "Build", done=False)]
    b = Board([task(330, steps=chain)])
    d, _, _, _ = run_sweep(b, merged={"feat/thing-330"})
    check("a chain whose current step is not ship-owned is ignored", d == [], d)


def t_finished_chain_is_no_decision():
    chain = shipped_chain()
    for s in chain:
        s["done"] = True
    b = Board([task(331, steps=chain)])
    d, _, _, _ = run_sweep(b, merged={"feat/thing-331"})
    check("an already-ticked chain produces no decision (idempotent)", d == [], d)


def t_sweep_is_idempotent():
    b = Board([task(340)])
    run_sweep(b, merged={"feat/thing-340"})
    calls_after_first = len(b.calls)
    d2, r2, f2, _ = run_sweep(b, merged={"feat/thing-340"})
    check("a second sweep makes no further calls",
          len(b.calls) == calls_after_first, b.calls)
    check("a second sweep has nothing to do", d2 == [] and r2 == [] and not f2,
          f"{d2} / {r2}")


# ---------------------------------------------------------------------------
# RAIL 4 -- both release-lane owners count (`R` the session, `R-A` the executor).
# ---------------------------------------------------------------------------

def t_both_ship_owners_are_seen():
    for owner in ("R", "R-A", "r-a"):
        b = Board([task(350, steps=shipped_chain(owner))])
        d, _, _, _ = run_sweep(b, merged={"feat/thing-350"})
        check(f"owner {owner!r} is recognised as a ship step",
              len(d) == 1 and d[0]["action"] == "tick", d)


def t_negative_r_only_would_miss_ra():
    """The convention moved to `R-A` (board #229) -- matching only "R" loses it."""
    b = Board([task(351, steps=shipped_chain("R-A"))])
    idx = C.ship_step_index(b.tasks[351])
    owner = b.tasks[351]["checklist"][idx]["owner"]
    check("the modern chains this must catch are owned by R-A", owner == "R-A")
    check("a naive owner == 'R' test would NOT match it", owner.upper() != "R")


def t_other_owners_are_not_ship_steps():
    b = Board([task(352, steps=shipped_chain("F"))])
    d, _, _, _ = run_sweep(b, merged={"feat/thing-352"})
    check("an F-owned current step is not a ship step", d == [], d)


# ---------------------------------------------------------------------------
# RAIL 5 -- the FETCH happens before any merge check.
# Without it the just-merged cars read as unmerged and the sweep silently ticks
# NOTHING on the very train it exists for.
# ---------------------------------------------------------------------------

def t_fetch_precedes_every_merge_check():
    b = Board([task(360)])
    d, r, _, n = run_sweep(b, merged={"feat/thing-360"})
    check("fetch ran exactly once", n == 1, n)
    check("and the merge check saw fresh state", by_id(r)[360]["result"] == "ticked")


def t_negative_stale_ref_ticks_nothing():
    """Simulates the no-fetch world: dev's ref does not know about the merge."""
    b = Board([task(361)])
    d, r, f, _ = run_sweep(b, merged=set())          # stale ref == "not merged"
    check("a stale origin/dev produces zero ticks",
          all(x["action"] == "skip" for x in d) and r == [], f"{d} / {r}")
    check("and that silence is exactly why the fetch is unconditional",
          b.tasks[361]["checklist"][3]["done"] is False)


def t_fetch_runs_on_a_dry_run_too():
    b = Board([task(362)])
    d, r, f, n = run_sweep(b, merged={"feat/thing-362"}, dry_run=True)
    check("dry run still fetches", n == 1, n)
    check("dry run plans the tick", by_id(d)[362]["action"] == "tick", d)
    check("dry run writes NOTHING", b.calls == [] and r == [] and not f, b.calls)
    check("dry run says so in the report",
          "dry run" in C.render(d, r, dry_run=True), C.render(d, r, True))


# ---------------------------------------------------------------------------
# RAIL 6 -- an unresolvable branch is skipped OUT LOUD, never guessed.
# ---------------------------------------------------------------------------

def t_unresolvable_branch_is_loud():
    b = Board([task(370)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-370"}, branches={370: None})
    check("no branch -> skip", by_id(d)[370]["action"] == "skip", d)
    check("no branch -> reason names it", "no branch found" in by_id(d)[370]["reason"])
    check("no branch -> no call", b.calls == [], b.calls)
    check("no branch -> it appears in the report",
          "#370" in C.render(d, r, False), C.render(d, r, False))


# ---------------------------------------------------------------------------
# RAIL 7 -- a tick is VERIFIED against the returned row; a race is loud.
# ---------------------------------------------------------------------------

def t_race_is_reported_not_swallowed():
    b = Board([task(380)])

    def racing_complete(task_id, actor):
        # Someone else ticked the ship step a moment ago, so the route completes
        # the NEXT step instead and hands the chain on.
        t = b.tasks[task_id]
        t["checklist"][3]["done"] = True
        t["checklist"][3]["completed_by"] = "M"
        return b.complete(task_id, actor)

    d = C.plan_ticks(b.load(), lambda t: ("feat/thing-380", "fixture"),
                     lambda x: True)
    r, f = C.apply_ticks(d, racing_complete, b.set_status, "R-A·auto")
    check("a raced tick is reported UNVERIFIED", r[0]["result"] == "unverified", r)
    check("...and named as a failure", len(f) == 1, f)
    check("...with the real completer named", "completed_by='M'" in r[0]["detail"]
          or "'M'" in r[0]["detail"], r[0]["detail"])


def t_step_left_open_is_unverified():
    d = [{"task_id": 1, "step_index": 3, "action": "tick"}]
    row = {"checklist": [{"done": True}] * 3 + [{"done": False, "title": "Ship"}]}
    ok, detail = C.verify_tick(row, d[0], "R-A·auto")
    check("a step still open after the call is NOT a tick", ok is False, detail)
    check("...and says a different step was completed",
          "different step" in detail, detail)


def t_api_error_is_a_failure_not_a_crash():
    b = Board([task(390)])

    def boom(task_id, actor):
        raise RuntimeError("409 every step is already complete")

    d = C.plan_ticks(b.load(), lambda t: ("feat/thing-390", "fixture"),
                     lambda x: True)
    r, f = C.apply_ticks(d, boom, b.set_status, "R-A·auto")
    check("an API error is recorded as a failure", len(f) == 1
          and r[0]["result"] == "error", r)
    check("...and the message is kept", "409" in r[0]["detail"], r[0]["detail"])


def t_negative_unverified_would_pass_silently():
    """Without verify_tick, an unverified call would look identical to a tick."""
    row = {"checklist": [{"done": True}] * 3 + [{"done": False, "title": "Ship"}]}
    ok, _ = C.verify_tick(row, {"step_index": 3}, "R-A·auto")
    check("the naive 'the POST returned 200' reading would be wrong", ok is False)


# ---------------------------------------------------------------------------
# RAIL 8 -- `--only` narrows the scan; the merge gate still applies.
# ---------------------------------------------------------------------------

def t_only_narrows_without_weakening():
    b = Board([task(401), task(402)])
    d, r, _, _ = run_sweep(b, merged={"feat/thing-401", "feat/thing-402"},
                           only=[401])
    check("--only restricts the candidate set", [x["task_id"] for x in d] == [401],
          d)
    check("the unlisted car is untouched",
          b.tasks[402]["checklist"][3]["done"] is False)

    b2 = Board([task(403)])
    d2, r2, _, _ = run_sweep(b2, merged=set(), only=[403])
    check("--only cannot force a tick on an unmerged car",
          by_id(d2)[403]["action"] == "skip" and r2 == [], f"{d2} / {r2}")


# ---------------------------------------------------------------------------
# RAIL 9 -- the sweep is status-agnostic; the merge gate is the only gate.
# The stale cars of Waves 23-25 sat in `Staged` AND `Review`.
# ---------------------------------------------------------------------------

def t_review_status_cars_are_swept_too():
    b = Board([task(410, status="Review"), task(411, status="Staged")])
    d, r, _, _ = run_sweep(b, merged={"feat/thing-410", "feat/thing-411"})
    check("both a Review car and a Staged car are ticked",
          sorted(x["task_id"] for x in r if x["result"] == "ticked") == [410, 411],
          r)


def t_scan_statuses_cover_the_observed_lane():
    check("the scan covers Staged and Review at minimum",
          {"Staged", "Review"} <= set(C.SCAN_STATUSES), C.SCAN_STATUSES)


# ---------------------------------------------------------------------------
# RAIL 10 -- the report is readable and complete (this is what M/Kevin reads).
# ---------------------------------------------------------------------------

def t_report_covers_both_kinds():
    b = Board([task(420), task(421)])
    d, r, _, _ = run_sweep(b, merged={"feat/thing-420"})
    body = C.render(d, r, dry_run=False)
    check("report counts the open ship steps", "2 open ship step(s)" in body, body)
    check("report names the shipped car", "#420" in body, body)
    check("report names the waiting car", "#421" in body, body)
    check("report says one shipped, one waiting",
          "1 shipped, 1 still waiting" in body, body)


# ---------------------------------------------------------------------------
# RAIL 11 (board #249) -- the STATUS follows the chain: `Review` while a step is
# open, `Done` when the chain is finished, and never anything else.
# ---------------------------------------------------------------------------

def t_status_target_follows_the_chain():
    open_chain = shipped_chain()
    open_chain[3]["done"] = True                      # ship step just ticked
    check("a chain with a step still open targets Review",
          C.target_status(open_chain) == "Review", C.target_status(open_chain))

    done_chain = shipped_chain()
    for s in done_chain:
        s["done"] = True
    check("a finished chain targets Done",
          C.target_status(done_chain) == "Done", C.target_status(done_chain))


def t_kevin_close_step_yields_review():
    """The common car shape: ship, then Kevin looks. That is `Review`."""
    b = Board([task(500)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-500"})
    check("the car lands in Review", b.tasks[500]["status"] == "Review",
          b.tasks[500]["status"])
    check("the write is recorded once, with both ends and the actor",
          b.status_calls == [(500, "Staged", "Review", "R-A·auto")], b.status_calls)
    check("no failures", not f, f)


def t_last_step_shipped_yields_done():
    chain = shipped_chain()
    chain[4]["done"] = True                           # only the ship step is open
    b = Board([task(501, steps=chain)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-501"})
    check("a car whose chain finishes on the ship step lands Done",
          b.tasks[501]["status"] == "Done", b.tasks[501]["status"])
    check("and it is reported as such", by_id(r)[501]["status_after"] == "Done", r)


def t_post_ship_step_owned_by_M_still_moves():
    """#242's OWN shape: the step after the train was M's, not Kevin's.

    A rule that only fired on a kevin-owned next step would have left #242
    sitting in `Staged` after Wave 26 -- the same lie, one step further along.
    """
    chain = [step("kevin", "Approve"), step("M-A", "Build"),
             step("M", "M review, then ship"),
             step("R-A", "Ship via a release train", done=False),
             step("M", "Verify on the NEXT train", done=False),
             step("kevin", "Validate and CLOSE", done=False)]
    b = Board([task(502, steps=chain)])
    run_sweep(b, merged={"feat/thing-502"})
    check("an M-owned next step still moves the car off Staged",
          b.tasks[502]["status"] == "Review", b.tasks[502]["status"])


def t_never_closed_never_a_dispatchable_status():
    """`Closed` is Kevin's alone; `Todo` would re-enter the dispatcher queue."""
    check("the allowed targets are exactly Review and Done",
          set(C.ALLOWED_TARGETS) == {"Review", "Done"}, C.ALLOWED_TARGETS)
    check("`Closed` is not among them", "Closed" not in C.ALLOWED_TARGETS)
    seen = set()
    for n_open in range(0, 6):                        # every chain shape
        chain = shipped_chain()
        for i, s in enumerate(chain):
            s["done"] = i < len(chain) - n_open
        seen.add(C.target_status(chain))
    check("no chain shape can produce anything else", seen <= {"Review", "Done"},
          seen)
    check("...and never a status the dispatcher would pick up",
          "Todo" not in seen and "In Progress" not in seen, seen)


def t_negative_status_left_alone_is_the_bug():
    """The pre-#249 world, stated as a test: tick only, and the tile still lies."""
    b = Board([task(503)])
    b.complete(503, "R-A·auto")                       # what the tool used to do
    check("the ship step is ticked", b.tasks[503]["checklist"][3]["done"] is True)
    check("...and the card STILL reads Staged, counted as awaiting a train",
          b.tasks[503]["status"] == "Staged", b.tasks[503]["status"])


# ---------------------------------------------------------------------------
# RAIL 12 (board #249) -- the status rides on the SAME proof as the tick. No
# second path: not an unmerged branch, not an unverified tick, not a car this
# run never touched.
# ---------------------------------------------------------------------------

def t_status_moves_only_for_cars_this_run_ticked():
    b = Board([task(510), task(511), task(512, status="Staged")])
    # 510 merged; 511 did not; 512 is mid-chain (an open M step in front), so the
    # sweep has no opinion about it at all.
    chain = shipped_chain()
    chain[2]["done"] = False
    b.tasks[512]["checklist"] = chain
    run_sweep(b, merged={"feat/thing-510", "feat/thing-512"})
    check("exactly one status write", [c[0] for c in b.status_calls] == [510],
          b.status_calls)
    check("the unmerged car keeps its status", b.tasks[511]["status"] == "Staged")
    check("the mid-chain car keeps its status", b.tasks[512]["status"] == "Staged")


def t_negative_blanket_status_sweep_would_touch_them_all():
    """Why the id list and the merge gate, and not `UPDATE ... WHERE status='Staged'`."""
    b = Board([task(513), task(514)])
    staged = [t["id"] for t in b.load() if t["status"] == "Staged"]
    run_sweep(b, merged={"feat/thing-513"})
    check("both cars read Staged before the sweep", staged == [513, 514], staged)
    check("a blanket 'fix all Staged' would have rewritten 2; this rewrote 1",
          len(b.status_calls) == 1 and b.status_calls[0][0] == 513, b.status_calls)


def t_unverified_tick_writes_no_status():
    b = Board([task(520)])

    def racing_complete(task_id, actor):
        t = b.tasks[task_id]
        t["checklist"][3]["done"] = True
        t["checklist"][3]["completed_by"] = "M"       # someone else got there
        return b.complete(task_id, actor)

    d = C.plan_ticks(b.load(), lambda t: ("feat/thing-520", "fixture"),
                     lambda x: True)
    r, f = C.apply_ticks(d, racing_complete, b.set_status, "R-A·auto")
    check("the tick is unverified", r[0]["result"] == "unverified", r)
    check("so NO status was written", b.status_calls == [], b.status_calls)
    check("and the report says why", "did not verify" in r[0]["status_detail"],
          r[0]["status_detail"])
    check("it is a failure", len(f) == 1, f)


def t_failed_tick_writes_no_status():
    b = Board([task(521)])

    def boom(task_id, actor):
        raise RuntimeError("409 every step is already complete")

    d = C.plan_ticks(b.load(), lambda t: ("feat/thing-521", "fixture"),
                     lambda x: True)
    r, f = C.apply_ticks(d, boom, b.set_status, "R-A·auto")
    check("a failed tick writes no status", b.status_calls == [], b.status_calls)
    check("...and says so", "no status is owed" in r[0]["status_detail"],
          r[0]["status_detail"])


def t_dry_run_writes_no_status():
    b = Board([task(522)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-522"}, dry_run=True)
    check("dry run plans the tick", by_id(d)[522]["action"] == "tick", d)
    check("dry run writes no status", b.status_calls == [], b.status_calls)
    check("dry run leaves the card alone", b.tasks[522]["status"] == "Staged")


def t_status_write_failure_is_loud():
    b = Board([task(523)])

    def refuse(task_id, old, new, actor):
        raise RuntimeError("status write did not land -- #523 reads 'Staged'")

    d, r, f, _ = run_sweep(b, merged={"feat/thing-523"}, set_status=refuse)
    check("the tick still counts as ticked", by_id(r)[523]["result"] == "ticked", r)
    check("but the status write is an error",
          by_id(r)[523]["status_result"] == "error", r)
    check("...and it is a FAILURE, not a quiet note", len(f) == 1, f)
    check("...naming what did not land", "did not land" in by_id(r)[523]["status_detail"],
          r)


# ---------------------------------------------------------------------------
# RAIL 13 (board #249) -- nothing is written when nothing is owed, and the
# two-touch close guard (board #136) survives the change of route.
# ---------------------------------------------------------------------------

def t_no_write_when_already_correct():
    chain = shipped_chain()
    b = Board([task(530, steps=chain, status="Review")])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-530"})
    check("a car already in Review is not re-written", b.status_calls == [],
          b.status_calls)
    check("...and says so", by_id(r)[530]["status_result"] == "unchanged"
          and "already Review" in by_id(r)[530]["status_detail"], r)
    check("...and it is not a failure", not f, f)


def t_two_touch_task_is_not_closed_by_the_sweep():
    """kevin_final + Review -> Done is Kevin's hand only; the API refuses it, so
    this refuses it too rather than winning by using a different route."""
    chain = shipped_chain()
    chain[4]["done"] = True                           # only the ship step is open
    b = Board([task(531, steps=chain, status="Review", kevin_final=True)])
    d, r, f, _ = run_sweep(b, merged={"feat/thing-531"})
    check("the ship step is still ticked", b.tasks[531]["checklist"][3]["done"]
          is True)
    check("but the status is NOT moved to Done", b.status_calls == [],
          b.status_calls)
    check("...and the refusal names two-touch",
          "two-touch" in by_id(r)[531]["status_detail"], r)
    check("...and a refusal is not an error", not f, f)


def t_negative_same_shape_without_kevin_final_moves():
    """Flip only `kevin_final` and the answer flips -- the guard is what stopped it."""
    chain = shipped_chain()
    chain[4]["done"] = True
    b = Board([task(532, steps=chain, status="Review", kevin_final=False)])
    run_sweep(b, merged={"feat/thing-532"})
    check("without the stamp the same car closes",
          b.tasks[532]["status"] == "Done", b.tasks[532]["status"])


def t_two_touch_staged_to_done_is_allowed():
    """The API's own exemption: Staged -> Done IS the train shipping signed-off work."""
    chain = shipped_chain()
    chain[4]["done"] = True
    b = Board([task(533, steps=chain, status="Staged", kevin_final=True)])
    run_sweep(b, merged={"feat/thing-533"})
    check("Staged -> Done is not blocked by the two-touch guard",
          b.tasks[533]["status"] == "Done", b.tasks[533]["status"])


def t_finished_task_is_not_restated():
    target, why = C.plan_status({"checklist": [{"done": True}], "kevin_final": False},
                                "Closed", "R-A·auto")
    check("a Closed task is never re-stated", target is None, why)
    check("...and says why", "finished" in why, why)


# ---------------------------------------------------------------------------
# RAIL 14 (board #249) -- the report tells M what happened to the STATUS,
# including when nothing happened.
# ---------------------------------------------------------------------------

def t_report_names_the_status_move():
    b = Board([task(540)])
    d, r, _, _ = run_sweep(b, merged={"feat/thing-540"})
    body = C.render(d, r, dry_run=False)
    check("the report shows the status move", "Staged -> Review" in body, body)
    check("the report marks it as a status line", "status" in body, body)


def t_report_names_a_refusal():
    chain = shipped_chain()
    chain[4]["done"] = True
    b = Board([task(541, steps=chain, status="Review", kevin_final=True)])
    d, r, _, _ = run_sweep(b, merged={"feat/thing-541"})
    body = C.render(d, r, dry_run=False)
    check("a refused status move is printed, not hidden",
          "two-touch" in body, body)


TESTS = [t_merged_car_is_ticked, t_negative_no_sweep_leaves_it_open,
         t_unmerged_car_is_never_ticked, t_partial_train_ticks_only_what_landed,
         t_negative_intent_cannot_tick,
         t_open_step_in_front_is_not_touched, t_negative_wrong_step_would_be_ticked,
         t_no_ship_step_is_no_decision, t_finished_chain_is_no_decision,
         t_sweep_is_idempotent,
         t_both_ship_owners_are_seen, t_negative_r_only_would_miss_ra,
         t_other_owners_are_not_ship_steps,
         t_fetch_precedes_every_merge_check, t_negative_stale_ref_ticks_nothing,
         t_fetch_runs_on_a_dry_run_too,
         t_unresolvable_branch_is_loud,
         t_race_is_reported_not_swallowed, t_step_left_open_is_unverified,
         t_api_error_is_a_failure_not_a_crash,
         t_negative_unverified_would_pass_silently,
         t_only_narrows_without_weakening,
         t_review_status_cars_are_swept_too, t_scan_statuses_cover_the_observed_lane,
         t_report_covers_both_kinds,
         # board #249 -- the status half
         t_status_target_follows_the_chain, t_kevin_close_step_yields_review,
         t_last_step_shipped_yields_done, t_post_ship_step_owned_by_M_still_moves,
         t_never_closed_never_a_dispatchable_status,
         t_negative_status_left_alone_is_the_bug,
         t_status_moves_only_for_cars_this_run_ticked,
         t_negative_blanket_status_sweep_would_touch_them_all,
         t_unverified_tick_writes_no_status, t_failed_tick_writes_no_status,
         t_dry_run_writes_no_status, t_status_write_failure_is_loud,
         t_no_write_when_already_correct,
         t_two_touch_task_is_not_closed_by_the_sweep,
         t_negative_same_shape_without_kevin_final_moves,
         t_two_touch_staged_to_done_is_allowed, t_finished_task_is_not_restated,
         t_report_names_the_status_move, t_report_names_a_refusal]


def run():
    for t in TESTS:
        try:
            t()
        except Exception as e:                        # a crashing rail is a failing rail
            check(f"{t.__name__} raised", False, f"{type(e).__name__}: {e}")
    print(f"\n{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print("  FAIL:", f)
    if FAIL == 0:
        print("ALL PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
