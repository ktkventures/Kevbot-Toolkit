"""Acceptance test — board #151: the /stamp flow ticks the pending Kevin
checklist step so the dispatcher's next-actor gate stops blocking dispatch.

Bug: Kevin stamped the first board-dispatched release (#150). The stamp moved
Approval → Todo and set kevin_final, but the checklist's first step ("Kevin
stamps", role=kevin) stayed UNCHECKED. dispatcher.next_actor() therefore kept
returning 'kevin', the eligibility gate (next_actor == assignee) skipped the
task, and it sat in Todo with dispatchable=0 — looking approved but going
nowhere.

Fix: stamp_approval marks the FIRST un-done kevin checklist step done (that
step IS the stamp) and names it in the system comment. Guard: only the first
such step — a later two-touch review step must stay open.

Board #171 UPDATE: the dispatcher's next-actor gate was later REMOVED entirely
(assignee is authoritative; the checklist is advisory). A missed tick can no
longer strand a task — so this bug class is deleted, not just fixed. The stamp
still ticks the step for the board's progress bar (walks 1/3/4, unchanged);
walk 2 now asserts the gate is GONE and dispatch holds with an un-ticked
checklist either way.

Board #225 UPDATE — the reason this file grew walks 5-8: every walk above builds
a LEGACY checklist ({role, text}), and the handler matched raw step["role"] /
step["text"]. #182 process chains carry {owner, title}, so on every CHAINED task
the loop matched nothing: 2 of 3 stamps on 07-30 (#222, #224) left their Kevin
step un-ticked and the task sitting `Todo`/`kevin`, which will not dispatch and
trips preflight invariant (7). The test was GREEN while the behaviour was
BROKEN. Walks 5-8 close that by exercising the chain schema directly, and cover
the second half of the same gap: the stamp never wrote `assignee`, so ticking
alone still left the task on kevin. Both halves fail without the fix.

Offline + hermetic, same shape as test_board_lifecycle_136.py: the real
`api.routers.dev_tasks` router functions run against an in-memory fake Supabase
client, and the real `tools/team_dispatcher/dispatcher.py` gate functions run
against a recorded fake `api()` — proving the end-to-end regression is closed.

Run:  .venv/bin/python src/test_stamp_ticks_kevin_step_151.py   (from RoR_Trader root)
"""
import importlib.util
import os
import sys
import types

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


# ── In-memory fake Supabase (query-builder subset the router uses) ─────────

class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, tname):
        self.store, self.tname = store, tname
        self.op, self.payload, self.filters = "select", None, []

    def select(self, *_):
        self.op = "select"
        return self

    def insert(self, row):
        self.op, self.payload = "insert", dict(row)
        return self

    def update(self, row):
        self.op, self.payload = "update", dict(row)
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def order(self, *_, **__):
        return self

    def limit(self, *_):
        return self

    def _match(self, row):
        return all(row.get(k) == v for k, v in self.filters)

    def execute(self):
        rows = self.store.setdefault(self.tname, [])
        if self.op == "insert":
            row = dict(self.payload)
            row["id"] = self.store["_seq"] = self.store.get("_seq", 0) + 1
            if self.tname == "dev_tasks":
                for k, v in (("status", "Backlog"), ("impact", "contained"),
                             ("kevin_final", False), ("standing_approval", False),
                             ("tags", []), ("checklist", []), ("blocked_by", []),
                             ("origin", "planned"), ("assignee", None),
                             ("description", ""), ("parent_id", None)):
                    row.setdefault(k, v)
            rows.append(row)
            return _Res([_deep(row)])
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(_deep(self.payload))
            return _Res([_deep(r) for r in hit])
        # Deep-copy on read so callers mutating the returned checklist can't
        # touch the store — matches the real client (fresh JSON per read).
        return _Res([_deep(r) for r in rows if self._match(r)])


def _deep(row):
    import copy
    return copy.deepcopy(row)


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return _Query(self.store, name)


FAKE = FakeSupabase()

db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

from api.routers import dev_tasks as dt                # noqa: E402


def task_row(tid):
    return next(r for r in FAKE.store["dev_tasks"] if r["id"] == tid)


def comments_for(tid):
    return [c["body"] for c in FAKE.store.get("dev_task_comments", [])
            if c["task_id"] == tid]


# ── walk 1: stamp ticks the first pending kevin step, spares a later one ───

print("― walk 1: stamp ticks the first pending Kevin step ―")
CHECKLIST = [
    {"text": "Kevin stamps", "done": False, "role": "kevin"},   # the stamp gate
    {"text": "Build / investigate", "done": False, "role": "F"},
    {"text": "Kevin reviews before Done", "done": False, "role": "kevin"},  # two-touch
]
t = dt.create_task({"title": "board-dispatched release", "description": "d",
                    "assignee": "F", "checklist": CHECKLIST}, user=None)
tid = t["id"]
dt.update_task(tid, {"status": "Approval", "actor": "M"}, user=None)
r = dt.stamp_approval(tid, {"mode": "start", "author": "kevin"}, user=None)

cl = r["checklist"]
ok("first Kevin step is now done (the stamp IS that step)",
   cl[0]["text"] == "Kevin stamps" and cl[0]["done"] is True)
ok("agent build step untouched", cl[1]["done"] is False)
ok("LATER Kevin (review) step stays OPEN — only the first is ticked",
   cl[2]["text"] == "Kevin reviews before Done" and cl[2]["done"] is False)
# Board #232: ONE stamp, and it no longer decides who closes. The kevin_final
# COLUMN survives (legacy rows + the API's close guard depend on it) — the stamp
# simply stops writing it, so a freshly stamped task is never two-touch.
ok("status → Todo, and the stamp does NOT set kevin_final (board #232)",
   r["status"] == "Todo" and r["kevin_final"] is False)
ok("stamp comment names the ticked step",
   any('checklist: ticked "Kevin stamps"' in b for b in comments_for(tid)))
ok("still exactly ONE system comment for the stamp",
   sum("stamp: approved" in b for b in comments_for(tid)) == 1)

# ── walk 2: board #171 — the checklist no longer gates dispatch ────────────
# Pre-#171 this walk proved the stamp HAD to tick the Kevin step or the
# dispatcher's next_actor() gate skipped the task. That gate is deleted: the
# dispatcher routes on `assignee` alone. So the trap that stranded the queue for
# ~41h — an un-ticked Kevin-first checklist — is now harmless, and we assert
# dispatchability holds EVEN with the checklist left un-ticked.
print("― walk 2: dispatcher routes on assignee; the checklist does not gate ―")
spec = importlib.util.spec_from_file_location(
    "team_dispatcher",
    os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)

ok("next_actor gate is GONE — the checklist no longer drives control flow",
   not hasattr(disp, "next_actor"))

agents = {"F": {"letter": "F", "status": "headless", "worktree": ".",
                "scope": "s", "boundaries": "b"}}
stamped = task_row(tid)
# Force the DELIBERATELY-UNTICKED Kevin-first checklist — the exact pre-#171 trap.
untick = {**stamped, "checklist": CHECKLIST}
disp.api = lambda m, p, b=None, prefer=None: [dict(untick)]
got, skipped = disp.triage_todo(agents, set())
ok("assignee-F Todo is DISPATCHABLE despite an un-ticked Kevin-first checklist "
   "(the ~41h dead-queue bug cannot recur)",
   [t2["id"] for t2 in got] == [tid] and skipped == [])
ok("back-compat dispatchable() shim returns the same subset",
   [t2["id"] for t2 in disp.dispatchable(agents, set())] == [tid])

# ── walk 3: back-compat — no Kevin step means no checklist write / tick note

print("― walk 3: back-compat (no Kevin step in the chain) ―")
t3 = dt.create_task({"title": "no kevin step", "description": "d",
                     "assignee": "F",
                     "checklist": [{"text": "Ship", "done": False, "role": "M"}]},
                    user=None)
tid3 = t3["id"]
dt.update_task(tid3, {"status": "Approval", "actor": "M"}, user=None)
r3 = dt.stamp_approval(tid3, {"mode": "delegate", "author": "kevin"}, user=None)
ok("checklist untouched when there is no pending Kevin step",
   r3["checklist"][0]["done"] is False)
ok("comment has NO tick note when nothing was ticked",
   all("checklist: ticked" not in b for b in comments_for(tid3))
   and any("stamp: approved to start (by kevin)" in b
           for b in comments_for(tid3)))

# ── walk 4: an already-done first Kevin step is left alone, next one ticks ─

print("― walk 4: skip an already-done Kevin step, tick the first UN-done one ―")
t4 = dt.create_task({"title": "kevin step already done", "description": "d",
                     "assignee": "F", "checklist": [
                         {"text": "Kevin pre-ok", "done": True, "role": "kevin"},
                         {"text": "Kevin final stamp", "done": False, "role": "Kevin"},
                     ]}, user=None)
tid4 = t4["id"]
dt.update_task(tid4, {"status": "Approval", "actor": "M"}, user=None)
r4 = dt.stamp_approval(tid4, {"mode": "delegate"}, user=None)
ok("already-done Kevin step stays done", r4["checklist"][0]["done"] is True)
ok("role match is case-insensitive ('Kevin' ticks too)",
   r4["checklist"][1]["done"] is True)
ok("comment names the second (first un-done) Kevin step",
   any('checklist: ticked "Kevin final stamp"' in b for b in comments_for(tid4)))

# ── walks 5-8: board #225 — the SAME contract against the #182 CHAIN schema ─
# Everything above builds a LEGACY checklist ({role, text}). That is precisely
# why #151 stayed green while the behaviour was broken: process chains carry
# {owner, title}, the handler matched raw step["role"], and so it matched
# NOTHING on any chained task — 2 of 3 stamps on 07-30 left their Kevin step
# un-ticked and the task sitting `Todo`/`kevin`, which will not dispatch and
# trips preflight invariant (7). These walks fail without the fix; walk 1 keeps
# the legacy path honest alongside them.

print("― walk 5: chain schema (owner/title) — stamp ticks + hands off ―")
CHAIN = [
    {"owner": "kevin", "title": "Approve the fix (fast stamp)", "done": False,
     "mode": "discuss", "origin": "planned"},
    {"owner": "F", "title": "Build it", "done": False, "mode": "execute"},
    {"owner": "M", "title": "Review", "done": False, "mode": "discuss"},
    {"owner": "kevin", "title": "Kevin looks before Done", "done": False,
     "mode": "discuss"},                                  # later two-touch step
]
t5 = dt.create_task({"title": "chained task", "description": "d",
                     "assignee": "kevin", "checklist": CHAIN}, user=None)
tid5 = t5["id"]
dt.update_task(tid5, {"status": "Approval", "actor": "M"}, user=None)
r5 = dt.stamp_approval(tid5, {"mode": "start", "author": "kevin"}, user=None)

cl5 = r5["checklist"]
ok("chain step matched on `owner` — the Kevin step IS ticked",
   cl5[0]["done"] is True)
ok("agent build step untouched", cl5[1]["done"] is False)
ok("LATER kevin two-touch step stays OPEN (guard survives the chain schema)",
   cl5[3]["done"] is False)
ok("assignee HANDED OFF kevin → F (next incomplete step's owner) — the task "
   "can now dispatch and invariant (7) holds",
   r5["assignee"] == "F")
ok("tick note reads the chain's `title`, so the stamp comment is auditable",
   any('checklist: ticked "Approve the fix (fast stamp)"' in b
       for b in comments_for(tid5)))
ok("stamp comment records the hand-off",
   any("handed off: kevin → F" in b for b in comments_for(tid5)))
ok("still exactly ONE system comment for the stamp",
   sum("stamp: approved" in b for b in comments_for(tid5)) == 1)
# The hand-off must AGREE with /steps/complete's, not merely resemble it —
# same helper, same answer, or the two paths drift (board #202).
ok("hand-off matches _next_assignee, the helper /steps/complete uses",
   dt._next_assignee(cl5, "kevin") == "F")

print("― walk 6: chain — a kevin step that is NOT first still ticks + hands off ―")
CHAIN6 = [
    {"owner": "M", "title": "Spec it", "done": True, "mode": "execute"},
    {"owner": "kevin", "title": "Approve the spec", "done": False,
     "mode": "discuss"},
    {"owner": "E", "title": "Implement", "done": False, "mode": "execute"},
]
t6 = dt.create_task({"title": "chain, kevin mid-list", "description": "d",
                     "assignee": "kevin", "checklist": CHAIN6}, user=None)
tid6 = t6["id"]
dt.update_task(tid6, {"status": "Approval", "actor": "M"}, user=None)
r6 = dt.stamp_approval(tid6, {"mode": "start"}, user=None)
ok("already-done chain step stays done", r6["checklist"][0]["done"] is True)
ok("first UN-done kevin step ticks", r6["checklist"][1]["done"] is True)
ok("assignee advances kevin → E", r6["assignee"] == "E")

print("― walk 7: chain — Kevin owns the LAST step → hand off to M, not kevin ―")
# T2: the chain never ends on kevin; closing is M's follow-through.
CHAIN7 = [
    {"owner": "F", "title": "Ship it", "done": True, "mode": "execute"},
    {"owner": "Kevin", "title": "Kevin signs off", "done": False,
     "mode": "discuss"},                       # capitalised — case-insensitive
]
t7 = dt.create_task({"title": "chain ends on kevin", "description": "d",
                     "assignee": "kevin", "checklist": CHAIN7}, user=None)
tid7 = t7["id"]
dt.update_task(tid7, {"status": "Approval", "actor": "M"}, user=None)
r7 = dt.stamp_approval(tid7, {"mode": "start"}, user=None)
ok("owner match is case-insensitive on the chain schema too ('Kevin')",
   r7["checklist"][1]["done"] is True)
ok("chain complete → assignee M (never left on kevin — T2)",
   r7["assignee"] == "M")

print("― walk 8: no kevin step in the chain → no tick, and NO hand-off ―")
# The hand-off rides on the tick. A stamp that ticked nothing must not quietly
# reassign the task out from under whoever holds it.
CHAIN8 = [{"owner": "F", "title": "Just build it", "done": False,
           "mode": "execute"}]
t8 = dt.create_task({"title": "chain, no kevin step", "description": "d",
                     "assignee": "P", "checklist": CHAIN8}, user=None)
tid8 = t8["id"]
dt.update_task(tid8, {"status": "Approval", "actor": "M"}, user=None)
r8 = dt.stamp_approval(tid8, {"mode": "start"}, user=None)
ok("nothing ticked when the chain has no kevin step",
   r8["checklist"][0]["done"] is False)
ok("assignee untouched when nothing was ticked", r8["assignee"] == "P")
ok("comment carries neither a tick note nor a hand-off note",
   all("checklist: ticked" not in b and "handed off" not in b
       for b in comments_for(tid8)))

print(f"\nALL PASS ({PASS} checks)")
