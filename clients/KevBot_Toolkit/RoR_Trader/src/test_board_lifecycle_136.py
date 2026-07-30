"""Acceptance tests — board #136 kanban lifecycle (Approval/Review/Staged,
two-touch stamps, impact column, dispatcher gates).

Offline + hermetic: the real `api.routers.dev_tasks` router functions run
against an in-memory fake Supabase client (no network, no DB), and the real
`tools/team_dispatcher/dispatcher.py` gate functions run against a recorded
fake `api()`. Covers the agreed acceptance walk:

  1. Full lifecycle on a test task: Backlog → Scoping → Approval →
     [stamp] → Todo → In Progress → Review → (Staged →) Done,
     BOTH stamp paths ("Approve — M closes" and "Approve + I review before
     Done"), system comments logged, two-touch close guard enforced
     (403 unless actor=kevin; Staged → Done release-train exempt).
  2. Field validation: impact whitelist, kevin_final/standing_approval bools.
  3. Run-request refusals for the new pipeline stages.
  4. Dispatcher: organic queue is status=Todo ONLY (by construction);
     run_requested hard-refuses Approval/Review/Staged; reap moves finished
     runs to Review (needs-review tag retired).
  5. Migration file is additive + idempotent by construction.

Run:  .venv/bin/python src/test_board_lifecycle_136.py   (from RoR_Trader root)
"""
import importlib.util
import json
import os
import sys
import time
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

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, k, v):
        self.filters.append(("eq", k, v))
        return self

    def neq(self, k, v):
        self.filters.append(("neq", k, v))
        return self

    def order(self, *_ , **__):
        return self

    def limit(self, *_ ):
        return self

    def _match(self, row):
        for op, k, v in self.filters:
            if op == "eq" and row.get(k) != v:
                return False
            if op == "neq" and row.get(k) == v:
                return False
        return True

    def execute(self):
        rows = self.store.setdefault(self.tname, [])
        if self.op == "insert":
            row = dict(self.payload)
            row["id"] = self.store["_seq"] = self.store.get("_seq", 0) + 1
            if self.tname == "dev_tasks":
                # Mirror dev_tasks_table.sql + dev_tasks_lifecycle.sql defaults.
                for k, v in (("status", "Backlog"), ("impact", "contained"),
                             ("kevin_final", False), ("standing_approval", False),
                             ("tags", []), ("checklist", []), ("blocked_by", []),
                             ("origin", "planned"), ("assignee", None),
                             ("description", ""), ("parent_id", None)):
                    row.setdefault(k, v)
            rows.append(row)
            return _Res([dict(row)])
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(self.payload)
            return _Res([dict(r) for r in hit])
        if self.op == "delete":
            self.store[self.tname] = [r for r in rows if not self._match(r)]
            return _Res([])
        return _Res([dict(r) for r in rows if self._match(r)])


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return _Query(self.store, name)


FAKE = FakeSupabase()

# Stub `db` BEFORE importing the router (api.deps does `from db import …`).
db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

from fastapi import HTTPException                      # noqa: E402
from api.routers import dev_tasks as dt                # noqa: E402


def expect_http(label, code, fn, *a, **k):
    try:
        fn(*a, **k)
    except HTTPException as e:
        ok(label, e.status_code == code, f"(got {e.status_code}: {e.detail})")
        return e
    print(f"FAIL: {label} — no HTTPException raised")
    sys.exit(1)


def comments_for(tid):
    return [c["body"] for c in FAKE.store.get("dev_task_comments", [])
            if c["task_id"] == tid]


print("― lifecycle walk A: stamp 'delegate' (Approve — M closes) ―")
t = dt.create_task({"title": "walk A", "description": "test", "assignee": "F"},
                   user=None)
tid = t["id"]
ok("created with defaults", t["status"] == "Backlog"
   and t["impact"] == "contained" and t["kevin_final"] is False
   and t["standing_approval"] is False)
dt.update_task(tid, {"status": "Scoping", "actor": "M"}, user=None)
r = dt.update_task(tid, {"status": "Approval", "actor": "M"}, user=None)
ok("walked Backlog → Scoping → Approval", r["status"] == "Approval")
expect_http("run-request refused in Approval", 400,
            dt.request_run, tid, {"author": "kevin"}, user=None)
expect_http("stamp rejects a bad mode", 400,
            dt.stamp_approval, tid, {"mode": "yolo"}, user=None)
r = dt.stamp_approval(tid, {"mode": "delegate", "author": "kevin"}, user=None)
ok("delegate stamp → Todo, kevin_final stays false",
   r["status"] == "Todo" and r["kevin_final"] is False)
ok("stamp logged ONE system comment",
   sum("stamp: approved — M closes (by kevin)" in b
       for b in comments_for(tid)) == 1)
expect_http("second stamp refused (409 — not in Approval)", 409,
            dt.stamp_approval, tid, {"mode": "delegate"}, user=None)
dt.update_task(tid, {"status": "In Progress", "actor": "F"}, user=None)
dt.update_task(tid, {"status": "Review", "actor": "F"}, user=None)
r = dt.update_task(tid, {"status": "Done", "actor": "M"}, user=None)
ok("M closes a non-two-touch task from Review", r["status"] == "Done")
ok("status flips logged as system comments",
   any("status: Backlog → Scoping (by M)" in b for b in comments_for(tid))
   and any("status: In Progress → Review (by F)" in b for b in comments_for(tid)))

print("― lifecycle walk B: stamp 'final' (two-touch) ―")
t2 = dt.create_task({"title": "walk B", "description": "test", "assignee": "F"},
                    user=None)
tid2 = t2["id"]
dt.update_task(tid2, {"status": "Approval", "actor": "M"}, user=None)
r = dt.stamp_approval(tid2, {"mode": "final", "author": "kevin"}, user=None)
ok("final stamp → Todo + kevin_final TRUE",
   r["status"] == "Todo" and r["kevin_final"] is True)
ok("stamp comment names two-touch",
   any("two-touch (Kevin reviews before Done)" in b for b in comments_for(tid2)))
dt.update_task(tid2, {"status": "In Progress", "actor": "F"}, user=None)
dt.update_task(tid2, {"status": "Review", "actor": "F"}, user=None)
expect_http("two-touch guard: M cannot move Review → Staged", 403,
            dt.update_task, tid2, {"status": "Staged", "actor": "M"}, user=None)
expect_http("two-touch guard: F cannot move Review → Done", 403,
            dt.update_task, tid2, {"status": "Done", "actor": "F"}, user=None)
expect_http("run-request refused in Review", 400,
            dt.request_run, tid2, {}, user=None)
r = dt.update_task(tid2, {"status": "Staged", "actor": "kevin"}, user=None)
ok("Kevin signs off Review → Staged", r["status"] == "Staged")
expect_http("run-request refused in Staged", 400,
            dt.request_run, tid2, {}, user=None)
r = dt.update_task(tid2, {"status": "Done", "actor": "R"}, user=None)
ok("release train (R) ships Staged → Done despite two-touch",
   r["status"] == "Done")

print("― field validation ―")
expect_http("impact whitelist enforced", 400,
            dt.update_task, tid, {"impact": "galaxy"}, user=None)
r = dt.update_task(tid, {"impact": "engine"}, user=None)
ok("valid impact lands", r["impact"] == "engine")
expect_http("kevin_final must be boolean", 400,
            dt.update_task, tid, {"kevin_final": "yes"}, user=None)
expect_http("standing_approval must be boolean", 400,
            dt.update_task, tid, {"standing_approval": 1}, user=None)
r = dt.update_task(tid, {"standing_approval": True}, user=None)
ok("standing_approval flag settable", r["standing_approval"] is True)

# ── Dispatcher gates (real dispatcher.py, recorded fake api) ───────────────

print("― dispatcher: Todo-only organic queue + stage refusals + reap→Review ―")
spec = importlib.util.spec_from_file_location(
    "team_dispatcher",
    os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)

calls = []
fake_rows = []


def fake_api(method, path, body=None, prefer=None):
    calls.append((method, path, body))
    if method == "GET" and path.startswith("dev_tasks?"):
        return list(fake_rows)
    return []


disp.api = fake_api

agents = {"F": {"letter": "F", "status": "headless", "worktree": ".",
                "scope": "s", "boundaries": "b"}}
todo_row = {"id": 1, "status": "Todo", "assignee": "F", "tags": [],
            "description": "d", "blocked_by": [], "checklist": []}
fake_rows = [todo_row]
got = disp.dispatchable(agents, set())
q = calls[-1][1]
# Board #198: the gate is `ai_eligible = true OR status = 'Todo'`. Still "by
# construction" — the SELECT is the gate — but status is now one arm of it rather
# than the whole thing, and it no longer refuses anything on its own.
ok("organic queue queries the #198 OR gate (ai_eligible OR Todo), by construction",
   "or=(ai_eligible.eq.true,status.eq.Todo)" in q, q)
ok("...and the legacy Todo arm is still in it (nothing on the board today changed)",
   "status.eq.Todo" in q, q)
ok("eligible Todo task passes the organic gates", got == [todo_row])

fake_rows = [
    dict(todo_row, id=10, status="Approval", tags=["run-requested"]),
    dict(todo_row, id=11, status="Review", tags=["run-requested"]),
    dict(todo_row, id=12, status="Staged", tags=["run-requested"]),
    dict(todo_row, id=13, status="Todo", tags=["run-requested"]),
]
good, bad = disp.run_requested(agents, set())
ok("run_requested refuses Approval/Review/Staged",
   sorted(t["id"] for t, _ in bad) == [10, 11, 12]
   and all(r == f"status is {t['status']}" for t, r in bad))
ok("run_requested still serves an approved Todo task",
   [t["id"] for t in good] == [13])

# reap: a finished run must land the task in Review (needs-review tag retired).
scratch = os.environ.get("TMPDIR", "/tmp")
log_path = os.path.join(scratch, "test136_run.log")
with open(log_path, "w") as f:
    f.write(json.dumps({"result": "report text", "is_error": False}) + "\n")
disp.STATE_FILE = os.path.join(scratch, "test136_state.json")
dead_pid = 4194304  # > default pid_max — /proc entry cannot exist
calls.clear()
st = {"runs": [{"run_id": "rTEST-1", "task_id": 42, "agent": "F",
                "pid": dead_pid, "t0": time.time(), "log": log_path,
                "ts": "2026-07-25T00:00:00+00:00", "active": True}]}
disp.reap(st)
patched = [b for m, p, b in calls if m == "PATCH" and p.startswith("dev_tasks")]
ok("reap moves the finished task to Review",
   {"status": "Review"} in patched)
ok("reap no longer writes the retired needs-review tag",
   not any(b and "tags" in b for b in patched))
ok("reap logs the status flip as a system comment",
   any(m == "POST" and p == "dev_task_comments"
       and "→ Review" in (b or {}).get("body", "")
       for m, p, b in calls))
ok("reap closes run_history with the agent's outcome",
   any(m == "PATCH" and p.startswith("run_history?run_id=eq.rTEST-1")
       and (b or {}).get("outcome") == "ok" for m, p, b in calls))

print("― migration: additive + idempotent by construction ―")
mig = open(os.path.join(SRC, "migrations", "dev_tasks_lifecycle.sql")).read()
sql = "\n".join(ln for ln in mig.splitlines()
                if not ln.strip().startswith("--"))
stmts = [s.strip() for s in sql.split(";") if s.strip()]
ok("exactly the 3 agreed columns",
   len(stmts) == 3 and all("ADD COLUMN IF NOT EXISTS" in s for s in stmts)
   and all(k in mig for k in ("impact", "kevin_final", "standing_approval")))
ok("no destructive statements",
   not any(k in mig.upper() for k in ("DROP ", "DELETE ", "TRUNCATE")))

print(f"\nALL PASS ({PASS} checks)")
