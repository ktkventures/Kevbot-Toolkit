"""Acceptance test — board #171: assignee-authoritative dispatch + LOUD skips.

Two changes, one file (tools/team_dispatcher/dispatcher.py):

1. ASSIGNEE IS AUTHORITATIVE. The old next_actor() gate let an un-ticked
   checklist step OVERRIDE the assignee and silently made the whole Todo queue
   undispatchable for ~41h on 07-26/27. triage_todo() now routes on `assignee`
   alone; the checklist gates nothing. (The dispatch side of that is also
   covered end-to-end in test_stamp_ticks_kevin_step_151.py walk 2.)

2. SKIPS ARE LOUD. The pre-#171 organic queue skipped tasks with a bare
   `continue` — a task could fail a gate every poll for days emitting NOTHING.
   process_skips() now: logs a skip on reason-CHANGE (deduped), comments ONCE
   per STUCK task (headless-assigned but hard-gated), and trips a one-time
   staleness flag after STALE_SKIP_S. A task merely WAITING on a human (assignee
   not a headless agent) is the designed state and stays quiet.

Offline + hermetic, same shape as the other board tests: the real dispatcher.py
runs against a recorded fake api() and a controllable clock.

Run:  .venv/bin/python src/test_dispatcher_loud_skips_171.py   (from RoR_Trader root)
"""
import importlib.util
import os
import sys
import types

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


# ── load the real dispatcher module ────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "team_dispatcher",
    os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)

# Controllable clock (process_skips is the only thing under test that reads it)
# and a temp state file so save_state() doesn't touch the real dispatcher state.
CLOCK = [1_000_000.0]
disp.time = types.SimpleNamespace(time=lambda: CLOCK[0])
scratch = os.environ.get("TMPDIR", "/tmp")
disp.STATE_FILE = os.path.join(scratch, "test171_state.json")

# Recorded fake api(): capture every write; serve the Todo queue for GETs.
CALLS = []
TODO_ROWS = []


def fake_api(method, path, body=None, prefer=None):
    CALLS.append((method, path, body))
    # Board #198 replaced the gate filter (`status=eq.Todo`) with the
    # ai_eligible-OR-Todo form; keyed off the module's own constant so this fake
    # follows the gate instead of pinning one spelling of it.
    if method == "GET" and path.startswith(f"dev_tasks?{disp.GATE_FILTER}"):
        return list(TODO_ROWS)
    return []


disp.api = fake_api

AGENTS = {"F": {"letter": "F", "status": "headless", "worktree": ".",
               "scope": "s", "boundaries": "b"}}


def comments_for(tid):
    return [b["body"] for m, p, b in CALLS
            if m == "POST" and p == "dev_task_comments" and b and b["task_id"] == tid]


def task(tid, **kw):
    base = {"id": tid, "status": "Todo", "assignee": "F", "tags": [],
            "description": "d", "blocked_by": [], "checklist": []}
    base.update(kw)
    return base


# ── walk 1: triage_todo classifies waiting vs stuck vs dispatchable ─────────
print("― walk 1: triage_todo — assignee authoritative, checklist ignored ―")
WAIT = task(201, assignee="kevin")                    # human-waiting (quiet)
STUCK = task(202, blocked_by=[999])                   # headless but hard-gated
# Un-ticked Kevin-first checklist that WOULD have blocked pre-#171:
DISP = task(203, checklist=[{"text": "Kevin stamps", "done": False, "role": "kevin"},
                            {"text": "build", "done": False, "role": "F"}])
TODO_ROWS = [WAIT, STUCK, DISP]
out, skipped = disp.triage_todo(AGENTS, set())   # 999 not Done → STUCK stays stuck
ok("dispatchable = the assignee-F task, checklist notwithstanding",
   [t["id"] for t in out] == [203])
ok("kevin-assigned task is skipped as WAITING (is_stuck=False)",
   (201, False) in [(t["id"], s) for t, _, s in skipped])
ok("blocked headless task is skipped as STUCK (is_stuck=True)",
   (202, True) in [(t["id"], s) for t, _, s in skipped])

# ── walk 2: process_skips comments ONCE on a stuck task, never on waiting ───
print("― walk 2: loud comment once on STUCK, silence on WAITING ―")
st = {"runs": []}
disp.process_skips(st, skipped, live=True, exclude_ids=set())
ok("STUCK task got exactly one '⛔ dispatch skipped' comment",
   sum("dispatch skipped" in b for b in comments_for(202)) == 1)
ok("WAITING (kevin) task got NO comment — that is the designed state",
   comments_for(201) == [])
ok("dedupe state recorded the stuck task as commented",
   st["skips"]["202"]["commented"] is True)

# ── walk 3: same reason next pass → deduped (no second comment) ─────────────
print("― walk 3: dedupe — unchanged reason does not re-comment ―")
n_before = len(comments_for(202))
disp.process_skips(st, skipped, live=True, exclude_ids=set())
ok("no second comment while the skip reason is unchanged",
   len(comments_for(202)) == n_before)

# ── walk 4: reason CHANGES → re-comment (log-on-change semantics) ───────────
print("― walk 4: a changed reason re-comments ―")
TODO_ROWS = [task(202, blocked_by=[998])]        # different blocker → different reason
_, skipped2 = disp.triage_todo(AGENTS, set())
ok("changed-reason skip really differs from the recorded one",
   skipped2[0][1] != st["skips"]["202"]["reason"])
disp.process_skips(st, skipped2, live=True, exclude_ids=set())
ok("changed reason produces a second dispatch-skipped comment",
   sum("dispatch skipped" in b for b in comments_for(202)) == 2)
ok("dedupe 'since' reset when the reason changed",
   st["skips"]["202"]["flagged"] is False)

# ── walk 5: staleness tripwire fires ONCE after STALE_SKIP_S ────────────────
print("― walk 5: staleness tripwire — one-time flag past STALE_SKIP_S ―")
CLOCK[0] += disp.STALE_SKIP_S + 1
disp.process_skips(st, skipped2, live=True, exclude_ids=set())
ok("a stuck task past STALE_SKIP_S gets a STALENESS TRIPWIRE comment",
   sum("STALENESS TRIPWIRE" in b for b in comments_for(202)) == 1)
CLOCK[0] += disp.STALE_SKIP_S + 1
disp.process_skips(st, skipped2, live=True, exclude_ids=set())
ok("the tripwire fires only ONCE (flagged latch holds)",
   sum("STALENESS TRIPWIRE" in b for b in comments_for(202)) == 1)

# ── walk 6: exclude_ids suppresses process_skips (run-request path owns it) ─
print("― walk 6: exclude_ids — no double-comment with the run-request path ―")
n_before = len(comments_for(202))
skipped3 = [(task(202, blocked_by=[997]), "blocked by [997] (not Done)", True)]
disp.process_skips(st, skipped3, live=True, exclude_ids={202})
ok("an excluded task is not commented by process_skips",
   len(comments_for(202)) == n_before)

# ── walk 7: dry-run makes ZERO writes; unstuck cleanup forgets the entry ────
print("― walk 7: dry-run is silent to the board; unstuck cleanup ―")
CALLS.clear()
disp.process_skips(st, skipped2, live=False, exclude_ids=set())
ok("live=False posts NO comments (prints only)",
   not any(m == "POST" for m, p, b in CALLS))
disp.process_skips(st, [], live=True, exclude_ids=set())  # 202 no longer skipped
ok("a task that leaves the skip set is dropped from the dedupe map",
   "202" not in st["skips"])

print(f"\nALL PASS ({PASS} checks)")
