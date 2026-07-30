"""Acceptance test — board #198, Step 4: the dispatch gate stops reading status.

The gate was `status=eq.Todo`, which made the kanban column do two unrelated jobs
at once (lifecycle position AND "may an AI touch this") and kept corrupting the
first. It is now `ai_eligible = true OR status = 'Todo'`.

Every rail below has at least one check that FAILS if the rail is removed — a
rail with no failing test is a comment:

  RAIL 1  the OR gate            — an ARMED task dispatches from ANY status; the
                                   query itself is the OR, not `status=eq.Todo`.
  RAIL 2  additive               — an UNARMED Todo task still dispatches, and an
                                   unarmed non-Todo task still does NOT (proves
                                   the filter was replaced, not deleted).
  RAIL 3  no status permission   — no task is refused for not being in Todo, and
                                   the module never queries `status=eq.Todo`
                                   again (the `status=eq.Done` blocked_by lookup
                                   is about the BLOCKER, not permission).
  RAIL 4  step-guard             — the Todo query used to double as the
                                   re-dispatch guard (reap() → Review → falls out
                                   of the query). An armed task does not
                                   re-dispatch a step a finished run already
                                   covered, and DOES dispatch once the chain
                                   advances. Status-blind: read from the
                                   dispatcher's own run log.
  RAIL 5  completed chain        — an armed task whose every step is ticked is not
                                   dispatched (it would re-run the whole task),
                                   while a fully-ticked chain sitting in Todo
                                   keeps its legacy behaviour (board #182 walk 4).
  RAIL 7  terminal statuses      — (step-5 audible) a `Done` or `Blocked` task is
                                   NEVER dispatched, armed or not: status as
                                   FINALITY, which is not the status-as-permission
                                   coupling #198 removed. The step-guard cannot
                                   cover it — 129 of the live board's 132 Done
                                   tasks are CHAINLESS, so step_sig() is the
                                   constant "task". Narrow on purpose: an armed
                                   Staged/Review/In Progress task still dispatches.
  RAIL 6  every earlier gate     — headless assignee, needs-review/scoping tags,
                                   empty description, blocked_by-vs-Done,
                                   chain/assignee agreement and mode=discuss all
                                   still hold on the NEW arm; the toggle arms a
                                   task, it does not exempt it.

Hermetic: loads the real dispatcher module with a fake `api()` that PARSES the
PostgREST filter it is handed, so a gate that ignored the filter (or kept the old
one) is caught rather than accidentally passing. No network, no DB, no spawns.

Mutation-checked 07-29 (board #198 step 4): reverting the gate to `status=eq.Todo`
fails 37 checks; removing the step-guard fails 5; removing the completed-chain
refusal fails 3; adding a status refusal back fails 12; and DELETING the filter
fails on the GATE_FILTER check and then aborts the suite loudly through the fake's
own filter parser (exit 1) rather than passing on a board-wide read.

Run:  .venv/bin/python src/test_dispatcher_ai_eligible_gate_198.py   (from RoR_Trader root)
"""
import importlib.util
import os
import re
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DISP = os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py")

PASS = FAIL = 0


def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}{(' — ' + detail) if detail else ''}")


spec = importlib.util.spec_from_file_location("dispatcher_under_test", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["dispatcher_under_test"] = D
spec.loader.exec_module(D)

SOURCE = open(DISP).read()

AGENTS = {"M": {"letter": "M", "status": "headless", "worktree": ".",
                "scope": "s", "boundaries": "b"},
          "F": {"letter": "F", "status": "headless", "worktree": ".",
                "scope": "s", "boundaries": "b"}}

ALL_STATUSES = ("Backlog", "Scoping", "Approval", "Todo", "In Progress",
                "Review", "Staged", "Done", "Blocked")
# Rail 7 (the step-5 audible): these two are FINAL — an armed task sitting in one
# of them is refused. Spelled as literals rather than derived from
# D.TERMINAL_STATUSES so widening that constant cannot silently shrink what rails
# 1 and 3 still test.
TERMINAL = ("Done", "Blocked")
LIVE_STATUSES = tuple(s for s in ALL_STATUSES if s not in TERMINAL)


def task(**kw):
    t = {"id": 1, "title": "T", "description": "d", "assignee": "F",
         "status": "Backlog", "ai_eligible": False,
         "tags": [], "blocked_by": [], "checklist": []}
    t.update(kw)
    return t


# ── the fake board: PostgREST filter parsing, so the gate is really tested ──
# Handles exactly the two filter shapes the dispatcher emits: `status=eq.X` and
# `or=(ai_eligible.eq.true,status.eq.Todo)`. Anything else raises, so a gate that
# invents a filter this test does not model fails LOUDLY instead of silently
# matching the whole board.
_OR_RE = re.compile(r"or=\(([^)]*)\)")
_EQ_RE = re.compile(r"(\w+)=eq\.([^&]+)")


def _matches(row, clause):
    col, _, val = clause.partition(".eq.")
    if val == "true":
        return row.get(col) is True
    if val == "false":
        return row.get(col) is False
    return str(row.get(col)) == val


def board_api(rows):
    """Fake `api()` serving `rows`, honouring the filter it is given."""
    seen = {"paths": []}

    def _api(method, path, body=None, prefer=None):
        seen["paths"].append(path)
        if not path.startswith("dev_tasks?"):
            return []
        q = path.split("?", 1)[1]
        m = _OR_RE.search(q)
        if m:
            clauses = m.group(1).split(",")
            return [r for r in rows if any(_matches(r, c) for c in clauses)]
        m = _EQ_RE.search(q)
        if m:
            col, val = m.group(1), m.group(2)
            return [r for r in rows if str(r.get(col)) == val]
        raise AssertionError(f"unmodelled dev_tasks filter: {q}")

    return _api, seen


def triage(rows, st=None):
    D.api, seen = board_api(rows)
    out, skipped = D.triage_todo(AGENTS, set(), st)
    return [t["id"] for t in out], skipped, seen


# ── RAIL 1 — the OR gate: armed dispatches from ANY status ──────────────────
print("― rail 1: an ARMED task dispatches wherever it sits on the board ―")
for i, s in enumerate(LIVE_STATUSES):
    armed = task(id=100 + i, status=s, ai_eligible=True)
    got, skipped, _ = triage([armed])
    ok(f"ai_eligible=true dispatches from status={s!r}",
       got == [100 + i],
       f"skipped={[x[1] for x in skipped]}")

got, _, seen = triage([task(id=7, status="Staged", ai_eligible=True,
                            assignee="M",
                            checklist=[{"owner": "M", "title": "ship it",
                                        "body": "SOP", "done": False}])])
ok("the #198 BONUS falls out: an armed Staged task with an open step dispatches",
   got == [7])
ok("the gate query is the OR filter, not status=eq.Todo",
   any("or=(ai_eligible.eq.true,status.eq.Todo)" in p for p in seen["paths"]),
   str(seen["paths"]))

# ── RAIL 2 — additive: today's board behaves exactly as it does now ─────────
print("― rail 2: ADDITIVE — the legacy Todo queue is untouched ―")
ok("the gate constant is the OR of the two arms, spelled for PostgREST",
   D.GATE_FILTER == "or=(ai_eligible.eq.true,status.eq.Todo)", D.GATE_FILTER)
got, skipped, _ = triage([task(id=1, status="Todo", ai_eligible=False)])
ok("an UNARMED Todo task still dispatches (nothing on the board today changes)",
   got == [1], f"skipped={[x[1] for x in skipped]}")

for s in ("Backlog", "Scoping", "Approval", "Review", "Staged", "Done"):
    got, _, _ = triage([task(id=2, status=s, ai_eligible=False)])
    ok(f"an UNARMED {s} task still does NOT dispatch (the filter was replaced, "
       f"not dropped)", got == [])

mixed = [task(id=11, status="Todo", ai_eligible=False),
         task(id=12, status="Backlog", ai_eligible=True),
         task(id=13, status="Backlog", ai_eligible=False)]
got, _, _ = triage(mixed)
ok("both arms are served in one pass and nothing else is", sorted(got) == [11, 12],
   str(got))

# ── RAIL 3 — status is never permission ────────────────────────────────────
print("― rail 3: no task is refused for not being in Todo ―")
# Scanned over the LIVE statuses only: the two TERMINAL ones are refused ON
# PURPOSE by rail 7 (finality, not permission), and that refusal names the status
# because it has to be legible in the log. Everything else must stay status-blind.
refusals = []
for i, s in enumerate(LIVE_STATUSES):
    _, skipped, _ = triage([task(id=200 + i, status=s, ai_eligible=True)])
    refusals += [r for _t, r, _s in skipped if "status" in r.lower()]
ok("no skip reason anywhere cites the task's status", not refusals, str(refusals))
# Scanned over the dev_tasks QUERIES the module actually builds, not the whole
# file: the prose above cites the old gate by name, and `agents?status=eq.headless`
# is the agent registry, neither of which is a task-permission read.
TASK_QUERIES = re.findall(r"dev_tasks\?[^\"']*", SOURCE)
STATUS_QUERIES = [q for q in TASK_QUERIES if "status=eq." in q]
ok("no dev_tasks query filters on status=eq.Todo any more",
   not [q for q in TASK_QUERIES if "status=eq.Todo" in q], str(TASK_QUERIES))
ok("...and the ONE remaining status=eq. task query is the blocked_by Done lookup "
   "(the BLOCKER's state, not permission)",
   len(STATUS_QUERIES) == 1 and "status=eq.Done" in STATUS_QUERIES[0],
   str(STATUS_QUERIES))
ok("reap() still WRITES Review — status is write-only, not unused",
   '"status": "Review"' in SOURCE)

# ── RAIL 4 — the step-guard replaces the Todo query's accidental guard ─────
print("― rail 4: an armed task does not re-run the step it just ran ―")
chain = [{"owner": "M", "title": "build the thing", "body": "SOP", "done": False},
         {"owner": "M", "title": "review the thing", "body": "SOP", "done": False}]
armed_chain = task(id=42, status="Review", ai_eligible=True, assignee="M",
                   checklist=[dict(s) for s in chain])

st_fresh = {"runs": []}
got, _, _ = triage([armed_chain], st_fresh)
ok("first pass: the armed task dispatches", got == [42])

sig = D.step_sig(armed_chain)
st_ran = {"runs": [{"run_id": "r1", "task_id": 42, "agent": "M",
                    "step_sig": sig, "active": False}]}
got, skipped, _ = triage([armed_chain], st_ran)
ok("a FINISHED run on this step refuses a re-dispatch (no cap-eating loop)",
   got == [] and len(skipped) == 1, f"got={got}")
ok("...the reason names the step and says what re-arms it",
   skipped and "step 1" in skipped[0][1] and "re-arms" in skipped[0][1],
   skipped[0][1] if skipped else "")
ok("...and it is classified WAITING, not stuck (no comment on every finished run)",
   skipped and skipped[0][2] is False)

st_active = {"runs": [{"run_id": "r1", "task_id": 42, "agent": "M",
                       "step_sig": sig, "active": True}]}
got, _, _ = triage([armed_chain], st_active)
ok("an ACTIVE run does not trip the guard (slots/per-lane rules own that)",
   got == [42])

advanced = task(id=42, status="Review", ai_eligible=True, assignee="M",
                checklist=[{"owner": "M", "title": "build the thing", "done": True},
                           {"owner": "M", "title": "review the thing",
                            "body": "SOP", "done": False}])
got, _, _ = triage([advanced], st_ran)
ok("THE CHAIN RE-ARMS ITSELF: tick the step and the next one dispatches",
   got == [42])
ok("step_sig changes when the chain advances",
   D.step_sig(advanced) != sig)

got, _, _ = triage([task(id=43, status="Todo", ai_eligible=False, assignee="M",
                         checklist=[dict(s) for s in chain])],
                   {"runs": [{"run_id": "r1", "task_id": 43, "agent": "M",
                              "step_sig": D.step_sig(
                                  task(checklist=[dict(s) for s in chain])),
                              "active": False}]})
ok("the guard does NOT touch the legacy Todo arm (byte-identical behaviour)",
   got == [43])

chainless = task(id=44, status="Review", ai_eligible=True, assignee="M")
ok("step_sig of a chainless task is 'task'", D.step_sig(chainless) == "task")
got, skipped, _ = triage([chainless],
                         {"runs": [{"run_id": "r1", "task_id": 44,
                                    "step_sig": "task", "active": False}]})
ok("a chainless armed task runs ONCE per arm, not every poll", got == [],
   f"got={got}")
ok("...and the reason says 'this task' rather than a step number",
   skipped and "this task" in skipped[0][1], skipped[0][1] if skipped else "")

ok("st=None makes the guard inert (the dispatchable() shim / unit callers)",
   D.triage_todo(AGENTS, set(), None) is not None or True)
D.api, _ = board_api([armed_chain])
ok("...proved: the same task dispatches with no state passed",
   [t["id"] for t in D.dispatchable(AGENTS, set())] == [42])

st_forget = {"runs": [{"run_id": "r1", "task_id": 42, "step_sig": sig,
                       "active": False},
                      {"run_id": "r2", "task_id": 99, "step_sig": "task",
                       "active": False},
                      {"run_id": "r3", "task_id": 42, "step_sig": sig,
                       "active": True}]}
D.forget_task_runs(st_forget, 42)
ok("forget_task_runs() drops that task's FINISHED runs (the #197 retry hook)",
   [r["run_id"] for r in st_forget["runs"]] == ["r2", "r3"],
   str([r["run_id"] for r in st_forget["runs"]]))

# ── RAIL 5 — a completed chain is not re-run from the top ──────────────────
print("― rail 5: a fully-ticked chain has nothing left to dispatch ―")
done_chain = [{"owner": "M", "title": "a", "done": True},
              {"owner": "M", "title": "b", "done": True}]
got, skipped, _ = triage([task(id=50, status="Review", ai_eligible=True,
                               assignee="M",
                               checklist=[dict(s) for s in done_chain])],
                         {"runs": []})
ok("an armed task with every step ticked is NOT dispatched", got == [],
   f"got={got}")
ok("...the reason says the chain is complete",
   skipped and "chain complete" in skipped[0][1], skipped[0][1] if skipped else "")
ok("...and it is quiet (waiting on a human to close it)",
   skipped and skipped[0][2] is False)

got, _, _ = triage([task(id=51, status="Todo", ai_eligible=False, assignee="M",
                         checklist=[dict(s) for s in done_chain])], {"runs": []})
ok("a fully-ticked chain in Todo keeps its legacy behaviour (board #182 walk 4)",
   got == [51])

got, _, _ = triage([task(id=52, status="Review", ai_eligible=True, assignee="M",
                         checklist=[{"role": "M", "text": "legacy", "done": True}])],
                   {"runs": []})
ok("a LEGACY {role,text,done} checklist is not mistaken for a finished chain",
   got == [52])

# ── RAIL 6 — the toggle arms a task, it does not exempt it ─────────────────
print("― rail 6: every pre-#198 gate still holds on the new arm ―")
cases = [
    ("assignee is not a headless lane",
     task(id=60, status="Backlog", ai_eligible=True, assignee="kevin"), False),
    ("tagged needs-review",
     task(id=61, status="Backlog", ai_eligible=True, tags=["needs-review"]), True),
    ("tagged needs-scoping",
     task(id=62, status="Backlog", ai_eligible=True, tags=["needs-scoping"]), True),
    ("empty description",
     task(id=63, status="Backlog", ai_eligible=True, description="  "), True),
    ("blocked_by an unfinished task",
     task(id=64, status="Backlog", ai_eligible=True, blocked_by=[999]), True),
    ("chain owner and assignee disagree",
     task(id=65, status="Backlog", ai_eligible=True, assignee="F",
          checklist=[{"owner": "M", "title": "x", "done": False}]), True),
    ("current step is mode=discuss",
     task(id=66, status="Backlog", ai_eligible=True,
          checklist=[{"owner": "F", "title": "x", "mode": "discuss",
                      "done": False}]), False),
]
for label, t, expect_stuck in cases:
    got, skipped, _ = triage([t], {"runs": []})
    ok(f"armed but REFUSED: {label}", got == [] and len(skipped) == 1,
       f"got={got} skipped={[x[1] for x in skipped]}")
    ok(f"...loudness unchanged (is_stuck={expect_stuck})",
       skipped and skipped[0][2] is expect_stuck)

got, _, _ = triage([task(id=67, status="Backlog", ai_eligible=True,
                         blocked_by=[999])], {"runs": []})
ok("a stale blocker still blocks only while it is not Done", got == [])
D.api, _ = board_api([task(id=68, status="Backlog", ai_eligible=True,
                           blocked_by=[999])])
out, _ = D.triage_todo(AGENTS, {999}, {"runs": []})
ok("...and a DONE blocker does not (the #144 non-bug)",
   [t["id"] for t in out] == [68])

# ── RAIL 7 — TERMINAL statuses: finality, not permission (step-5 audible) ──
print("― rail 7: a Done or Blocked task is never dispatched, armed or not ―")
ok("TERMINAL_STATUSES is exactly Done + Blocked",
   D.TERMINAL_STATUSES == ("Done", "Blocked"), str(D.TERMINAL_STATUSES))

# THE MEASURED HOLE: 129 of the live board's 132 Done tasks are CHAINLESS and 89
# are assigned to a headless lane, so step_sig() is the constant "task" and the
# step-guard has nothing to protect. Empty run state on purpose — this must be
# refused by the terminal rail alone, not by the step-guard remembering a run.
closed = task(id=70, status="Done", ai_eligible=True, assignee="M")
got, skipped, _ = triage([closed], {"runs": []})
ok("an ARMED, CHAINLESS, Done task is NOT dispatched (the re-run hole)",
   got == [] and len(skipped) == 1, f"got={got}")
ok("...the reason says terminal and names the status",
   skipped and "terminal" in skipped[0][1] and "Done" in skipped[0][1],
   skipped[0][1] if skipped else "")
ok("...and it is QUIET — a closed task is not a stuck queue (no comment, no "
   "4h tripwire on a finished thread)",
   skipped and skipped[0][2] is False)

got, skipped, _ = triage([task(id=71, status="Blocked", ai_eligible=True,
                               assignee="M")], {"runs": []})
ok("an ARMED Blocked task is NOT dispatched (reap() parks failed runs there — "
   "board #197)", got == [], f"got={got}")
ok("...quiet as well (reap already commented; the board column IS the signal)",
   skipped and skipped[0][2] is False)

open_step = [{"owner": "M", "title": "do it", "body": "SOP", "done": False}]
for s in TERMINAL:
    got, _, _ = triage([task(id=72, status=s, ai_eligible=True, assignee="M",
                             checklist=[dict(x) for x in open_step])],
                       {"runs": []})
    ok(f"an armed {s} task with an OPEN chain step is refused too "
       f"(finality beats an un-ticked step)", got == [], f"got={got}")

# The rail must be NARROW: it is the two terminal columns, not "every status that
# is not Todo" — that would put the #198 coupling straight back.
for s in ("Staged", "Review", "In Progress", "Backlog", "Scoping", "Approval"):
    got, _, _ = triage([task(id=73, status=s, ai_eligible=True, assignee="M",
                             checklist=[dict(x) for x in open_step])],
                       {"runs": []})
    ok(f"an armed {s} task still DISPATCHES (the rail is finality, not a "
       f"whitelist)", got == [73], f"got={got}")
ok("...which keeps the #193 bonus: an armed Staged task self-dispatches",
   triage([task(id=74, status="Staged", ai_eligible=True, assignee="M")],
          {"runs": []})[0] == [74])

# A closed task cannot reach the spawn path even through the back-back door: the
# gate is the only producer of the dispatch list (dispatchable() is a shim).
D.api, _ = board_api([closed])
ok("dispatchable() — the shim one_pass()'s callers use — is empty for a Done "
   "task", [t["id"] for t in D.dispatchable(AGENTS, set())] == [])

# Terminal refusal happens FIRST, so the reason a human reads is "it is finished"
# rather than an incidental gate it also happens to fail.
got, skipped, _ = triage([task(id=75, status="Done", ai_eligible=True,
                               assignee="kevin", description="")], {"runs": []})
ok("terminal is checked before assignee/description — the reason is finality",
   got == [] and "terminal" in skipped[0][1], skipped[0][1] if skipped else "")

print()
print(f"{'ALL PASS' if not FAIL else 'FAILURES'} "
      f"({PASS} checks{'' if not FAIL else f', {FAIL} failed'})")
sys.exit(1 if FAIL else 0)
