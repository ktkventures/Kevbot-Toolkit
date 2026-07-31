#!/usr/bin/env python3
"""Rails for the ONE-OFF #249 status backfill.

Hermetic: no board, no git, no network -- `plan_backfill` takes rows and
`apply_backfill` takes the writer.

The rail that matters most is the one about what it CANNOT do. A backfill is a
blunt instrument pointed at live rows, so every test here is really the same
question asked from a different angle: *can this thing touch a task nobody
named?* Each has a NEGATIVE twin showing what the rule prevents.

Run:  python3 tools/release_brief/test_backfill_car_status_249.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backfill_car_status_249 as BF  # noqa: E402

PASS = FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}{(' -- ' + str(detail)) if detail else ''}")


# ---------------------------------------------------------------------------
# Fixtures -- the real shape of the eight: ship step ticked, Kevin's close step
# still open, status stuck on Staged.
# ---------------------------------------------------------------------------

def step(owner, title, done=True):
    return {"owner": owner, "title": title, "done": done,
            "id": f"s{abs(hash((owner, title))) % 10 ** 6}"}


def shipped_car(tid, status="Staged", kevin_final=False, ship_done=True,
                ship_owner="R-A"):
    return {"id": tid, "title": f"car {tid}", "status": status,
            "assignee": "kevin", "kevin_final": kevin_final,
            "checklist": [
                step("kevin", "Approve"),
                step("F", "Build it"),
                step("M", "M review, then ship"),
                step(ship_owner, "Ship via a release train", done=ship_done),
                step("kevin", "Confirm, then close", done=False)]}


class Writer:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, task_id, old, new, actor):
        if task_id in self.fail_on:
            raise RuntimeError(f"status write did not land -- #{task_id}")
        self.calls.append((task_id, old, new, actor))


# ---------------------------------------------------------------------------
# RAIL 1 -- the named cars, and ONLY the named cars.
# ---------------------------------------------------------------------------

def t_the_list_is_the_eight_read_off_the_board():
    check("the list is exactly the eight ids from the task",
          sorted(BF.BACKFILL_IDS) == [238, 240, 242, 243, 252, 257, 261, 262],
          BF.BACKFILL_IDS)
    check("...and it is a fixed literal, not a query result",
          isinstance(BF.BACKFILL_IDS, tuple), type(BF.BACKFILL_IDS))


def t_only_named_cars_are_planned():
    rows = [shipped_car(238), shipped_car(999)]      # 999 = identical, unnamed
    d = BF.plan_backfill(rows, "M-A·auto")
    check("the unnamed car is not even a decision",
          [x["task_id"] for x in d] == [238], d)


def t_negative_a_rule_would_have_taken_the_stranger():
    """The shortcut this tool refuses: 'every Staged car with a ticked ship step'."""
    rows = [shipped_car(238), shipped_car(999)]
    would_match = [r["id"] for r in rows
                   if r["status"] == "Staged"
                   and BF.shipped_step_index(r) is not None]
    d = BF.plan_backfill(rows, "M-A·auto")
    check("a rule would have matched BOTH", would_match == [238, 999], would_match)
    check("the named list matched one", [x["task_id"] for x in d] == [238], d)


def t_the_live_query_asks_for_the_named_ids_only():
    """Not a filter applied after a board-wide read -- the ids ARE the query."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "backfill_car_status_249.py"), encoding="utf-8").read()
    check("the live loader selects by id=in.(...)", "id=in.(" in src, src[:0])
    check("...built from BACKFILL_IDS", "def live_tasks(ids=BACKFILL_IDS)" in src)


# ---------------------------------------------------------------------------
# RAIL 2 -- a car with no TICKED ship step is refused. The tick is the record
# that the merge gate passed; without it this tool has no proof at all.
# ---------------------------------------------------------------------------

def t_unticked_ship_step_is_refused():
    d = BF.plan_backfill([shipped_car(238, ship_done=False)], "M-A·auto")
    check("no ticked ship step -> skip", d[0]["action"] == "skip", d)
    check("...and it names the missing proof",
          "no TICKED ship step" in d[0]["reason"], d)
    w = Writer()
    BF.apply_backfill(d, w, "M-A·auto")
    check("...and nothing is written", w.calls == [], w.calls)


def t_negative_the_same_car_ticked_is_corrected():
    """Flip only the tick and the answer flips -- the tick is the whole proof."""
    a = BF.plan_backfill([shipped_car(238, ship_done=False)], "M-A·auto")[0]
    b = BF.plan_backfill([shipped_car(238, ship_done=True)], "M-A·auto")[0]
    check("the ticked ship step alone decides it",
          (a["action"], b["action"]) == ("skip", "set"), f"{a} / {b}")


def t_a_chain_with_no_ship_step_at_all_is_refused():
    row = shipped_car(240)
    row["checklist"] = [step("kevin", "Approve"), step("F", "Build")]
    d = BF.plan_backfill([row], "M-A·auto")
    check("a chain with no R/R-A step is skipped", d[0]["action"] == "skip", d)


def t_both_ship_owners_count():
    for owner in ("R", "R-A"):
        d = BF.plan_backfill([shipped_car(242, ship_owner=owner)], "M-A·auto")
        check(f"a {owner}-owned ship step is recognised",
              d[0]["action"] == "set", d)


# ---------------------------------------------------------------------------
# RAIL 3 -- it corrects `Staged`, and nothing else. Anything already moved is
# left exactly where it is, which is also what makes a re-run harmless.
# ---------------------------------------------------------------------------

def t_only_staged_is_corrected():
    for status in ("Review", "Done", "Closed", "In Progress"):
        d = BF.plan_backfill([shipped_car(243, status=status)], "M-A·auto")
        check(f"a {status} car is left alone", d[0]["action"] == "skip",
              f"{status}: {d}")


def t_rerun_is_a_no_op():
    rows = [shipped_car(252)]
    d1 = BF.plan_backfill(rows, "M-A·auto")
    w = Writer()
    BF.apply_backfill(d1, w, "M-A·auto")
    check("the first run corrects it", [c[0] for c in w.calls] == [252], w.calls)
    rows[0]["status"] = w.calls[0][2]                # the board now reads Review
    d2 = BF.plan_backfill(rows, "M-A·auto")
    BF.apply_backfill(d2, w, "M-A·auto")
    check("the second run writes nothing more", len(w.calls) == 1, w.calls)
    check("...and says it is already correct",
          d2[0]["action"] == "skip" and "already corrected" in d2[0]["reason"], d2)


# ---------------------------------------------------------------------------
# RAIL 4 -- the target comes from the SWEEP's rule, not a copy of it.
# ---------------------------------------------------------------------------

def t_target_is_the_sweeps_own_rule():
    d = BF.plan_backfill([shipped_car(257)], "M-A·auto")
    check("a car whose chain still has Kevin's close step targets Review",
          d[0]["target"] == "Review", d)

    finished = shipped_car(261)
    for s in finished["checklist"]:
        s["done"] = True
    d2 = BF.plan_backfill([finished], "M-A·auto")
    check("a finished chain targets Done", d2[0]["target"] == "Done", d2)


def t_never_closed():
    targets = set()
    for kf in (False, True):
        for n_open in range(0, 3):
            row = shipped_car(262, kevin_final=kf)
            for i, s in enumerate(row["checklist"]):
                s["done"] = i < len(row["checklist"]) - n_open
            row["checklist"][3]["done"] = True       # ship step always ticked
            d = BF.plan_backfill([row], "M-A·auto")
            targets.add(d[0]["target"])
    check("no shape produces `Closed` -- that stays Kevin's",
          "Closed" not in targets, targets)
    check("...and no shape produces a dispatchable status",
          targets <= {None, "Review", "Done"}, targets)


def t_two_touch_is_honoured_here_too():
    """Same guard as the sweep: the backfill cannot close a two-touch task."""
    row = shipped_car(261, status="Staged", kevin_final=True)
    for s in row["checklist"]:
        s["done"] = True
    row["status"] = "Staged"
    d = BF.plan_backfill([row], "M-A·auto")
    check("Staged -> Done is the API's own exemption, so it proceeds",
          d[0]["action"] == "set" and d[0]["target"] == "Done", d)


# ---------------------------------------------------------------------------
# RAIL 5 -- dry run by default; a failed write is loud.
# ---------------------------------------------------------------------------

def t_dry_run_is_the_default():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "backfill_car_status_249.py"), encoding="utf-8").read()
    check("writing takes an explicit --apply",
          '"--apply", action="store_true"' in src)
    check("...and the writer is only called under it",
          "if args.apply else ([], [])" in src)
    d = BF.plan_backfill([shipped_car(238)], "M-A·auto")
    body = BF.render(d, [], apply=False)
    check("the dry-run report says nothing was written",
          "nothing written" in body and "--apply" in body, body)


def t_failed_write_is_a_failure():
    d = BF.plan_backfill([shipped_car(240), shipped_car(242)], "M-A·auto")
    w = Writer(fail_on=(240,))
    results, failures = BF.apply_backfill(d, w, "M-A·auto")
    check("the failing car is reported as an error", len(failures) == 1
          and failures[0]["task_id"] == 240, failures)
    check("...and the other car still landed", [c[0] for c in w.calls] == [242],
          w.calls)
    body = BF.render(d, results, apply=True)
    check("the report marks the failure", "!!" in body, body)


def t_write_is_attributed():
    d = BF.plan_backfill([shipped_car(238)], "M-A·auto")
    w = Writer()
    BF.apply_backfill(d, w, "M-A·auto")
    check("the actor travels with the write",
          w.calls == [(238, "Staged", "Review", "M-A·auto")], w.calls)


def t_report_lists_every_named_car():
    rows = [shipped_car(i) for i in BF.BACKFILL_IDS]
    rows[0]["status"] = "Review"                     # one already fine
    d = BF.plan_backfill(rows, "M-A·auto")
    body = BF.render(d, [], apply=False)
    for i in BF.BACKFILL_IDS:
        check(f"#{i} appears in the report", f"#{i}" in body, body)
    check("the header counts both kinds", "7 to correct, 1 already fine" in body,
          body)


TESTS = [t_the_list_is_the_eight_read_off_the_board, t_only_named_cars_are_planned,
         t_negative_a_rule_would_have_taken_the_stranger,
         t_the_live_query_asks_for_the_named_ids_only,
         t_unticked_ship_step_is_refused, t_negative_the_same_car_ticked_is_corrected,
         t_a_chain_with_no_ship_step_at_all_is_refused, t_both_ship_owners_count,
         t_only_staged_is_corrected, t_rerun_is_a_no_op,
         t_target_is_the_sweeps_own_rule, t_never_closed,
         t_two_touch_is_honoured_here_too,
         t_dry_run_is_the_default, t_failed_write_is_a_failure,
         t_write_is_attributed, t_report_lists_every_named_car]


def run():
    for t in TESTS:
        try:
            t()
        except Exception as e:                       # a crashing rail is a failing rail
            check(f"{t.__name__} raised", False, f"{type(e).__name__}: {e}")
    print(f"\n{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print("  FAIL:", f)
    if FAIL == 0:
        print("ALL PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
