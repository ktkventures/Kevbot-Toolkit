"""Acceptance test — board #148: the open-modal liveness poll is one
consolidated round-trip, and its board-change probe is a cheap max(updated_at).

Waste: TaskDetailModal polled every 7s firing 3-4 calls/tick — loadThread +
loadRuns(scoped) + loadRuns(task) + a FULL 120-row board refetch — so an open
modal drove 30+ requests/min indefinitely (13.6k seq scans on a 120-row table,
and it inflated the Supabase request/error dashboards).

Fix (server half): a single GET /api/dev-tasks/{id}/poll returns the scoped
thread + run rows, the header task's run rows only when a subtask is scoped,
and board_max_updated_at — the single most-recent dev_tasks.updated_at (the
trg_dev_tasks_touch trigger keeps it fresh). The client compares that probe to
its last-seen value and refetches the whole board ONLY when it moved.

This also pins the "no manufactured errors" property: every read uses list
semantics (`.data or []`), never `.single()`, so a task with zero comments or
zero run rows returns `[]` — never a no-rows PGRST116/406 that Supabase logs as
an error (87 of 120 tasks have zero run_history rows).

Offline + hermetic, same shape as test_stamp_ticks_kevin_step_151.py: the real
`api.routers.dev_tasks.poll_task` runs against an in-memory fake Supabase
client. The fake honors order()/limit() so the max(updated_at) probe is really
exercised, and bumps dev_tasks.updated_at on every write like the real trigger.

Run:  .venv/bin/python src/test_board_poll_efficiency_148.py   (from RoR_Trader root)
"""
import os
import sys
import types

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


# ── In-memory fake Supabase (query-builder subset poll_task uses) ──────────
# Extends the #151 fake with order(col, desc) + limit(n) honored, and a
# monotonic dev_tasks.updated_at clock that stands in for trg_dev_tasks_touch.

class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, tname):
        self.store, self.tname = store, tname
        self.op, self.payload, self.filters = "select", None, []
        self.orderings, self._limit = [], None

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

    def order(self, col, desc=False, **__):
        self.orderings.append((col, desc))
        return self

    def limit(self, n=None):
        self._limit = n
        return self

    def _match(self, row):
        return all(row.get(k) == v for k, v in self.filters)

    def _touch(self, row):
        # Mirror trg_dev_tasks_touch: bump updated_at on every dev_tasks write.
        if self.tname == "dev_tasks":
            self.store["_clock"] = self.store.get("_clock", 0) + 1
            row["updated_at"] = f"{self.store['_clock']:010d}"

    def execute(self):
        rows = self.store.setdefault(self.tname, [])
        if self.op == "insert":
            row = dict(self.payload)
            row["id"] = self.store["_seq"] = self.store.get("_seq", 0) + 1
            self._touch(row)
            rows.append(row)
            return _Res([_deep(row)])
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(_deep(self.payload))
                self._touch(r)
            return _Res([_deep(r) for r in hit])
        out = [_deep(r) for r in rows if self._match(r)]
        # Apply orderings last-first so the first-added key is primary (stable).
        for col, desc in reversed(self.orderings):
            out.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=desc)
        if self._limit is not None:
            out = out[: self._limit]
        return _Res(out)


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


def seed_comment(tid, body):
    FAKE.table("dev_task_comments").insert(
        {"task_id": tid, "author": "system", "body": body}).execute()


def seed_run(tid, outcome):
    FAKE.table("run_history").insert(
        {"task_id": tid, "agent_letter": "F", "outcome": outcome}).execute()


def task_updated_at(tid):
    return next(r for r in FAKE.store["dev_tasks"] if r["id"] == tid)["updated_at"]


# ── setup: a vision task + one subtask, each with its own thread + runs ─────

print("― setup: two tasks with distinct threads / runs ―")
parent = dt.create_task({"title": "polling efficiency", "description": "d",
                         "assignee": "F"}, user=None)
pid = parent["id"]
sub = dt.create_task({"title": "sub", "description": "d", "assignee": "F",
                      "parent_id": pid}, user=None)
sid = sub["id"]
# A THIRD task that is the most-recently-updated one, to prove the probe reads
# the global max, not the polled task's own updated_at.
other = dt.create_task({"title": "other", "description": "d"}, user=None)
oid = other["id"]

seed_comment(pid, "parent comment 1")
seed_comment(pid, "parent comment 2")
seed_comment(sid, "sub comment 1")
seed_run(pid, "ok")
seed_run(pid, "requested")
seed_run(sid, "running")


# ── walk 1: poll the parent (no scoped_id) ─────────────────────────────────

print("― walk 1: consolidated poll of the header task ―")
r = dt.poll_task(pid, user=None)

ok("returns exactly the four poll keys",
   set(r) == {"comments", "runs", "header_runs", "board_max_updated_at"})
ok("comments are the parent's thread (list, filtered by task_id)",
   [c["body"] for c in r["comments"]] == ["parent comment 1", "parent comment 2"])
ok("runs are the parent's run_history rows only",
   {x["outcome"] for x in r["runs"]} == {"ok", "requested"} and len(r["runs"]) == 2)
ok("header_runs is None when scoped == task (no redundant second fetch)",
   r["header_runs"] is None)
ok("board_max_updated_at is the GLOBAL max (the 'other' task, updated last)",
   r["board_max_updated_at"] == task_updated_at(oid)
   and r["board_max_updated_at"] == max(t["updated_at"] for t in FAKE.store["dev_tasks"]))


# ── walk 2: poll a selected subtask (scoped_id != task_id) ─────────────────

print("― walk 2: subtask scoped — header runs come back too ―")
r2 = dt.poll_task(pid, scoped_id=sid, user=None)
ok("comments re-scope to the subtask",
   [c["body"] for c in r2["comments"]] == ["sub comment 1"])
ok("runs re-scope to the subtask",
   {x["outcome"] for x in r2["runs"]} == {"running"})
ok("header_runs now carries the PARENT's runs (header RunButton stays fresh)",
   r2["header_runs"] is not None
   and {x["outcome"] for x in r2["header_runs"]} == {"ok", "requested"})


# ── walk 3: the no-manufactured-errors property (empty lanes → []) ─────────

print("― walk 3: zero comments / zero runs → [] , never a no-rows error ―")
bare = dt.create_task({"title": "no activity", "description": "d"}, user=None)
rb = dt.poll_task(bare["id"], user=None)
ok("empty comment lane returns [] (list semantics, not .single()/PGRST116)",
   rb["comments"] == [])
ok("empty run lane returns [] (list semantics, not .single()/PGRST116)",
   rb["runs"] == [])
ok("board probe still populated even for a quiet task",
   isinstance(rb["board_max_updated_at"], str) and rb["board_max_updated_at"])
# Belt-and-suspenders: the router source never CALLS single()/maybe_single()
# (AST, so a docstring mentioning them by name doesn't false-positive).
import ast  # noqa: E402
_tree = ast.parse(open(os.path.join(SRC, "api", "routers", "dev_tasks.py")).read())
_single_calls = [n for n in ast.walk(_tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("single", "maybe_single")]
ok("dev_tasks router makes NO .single()/.maybe_single() call anywhere",
   _single_calls == [])


# ── walk 4: the probe MOVES on a real change (change-detection basis) ───────

print("― walk 4: probe advances when any task changes ―")
before = dt.poll_task(pid, user=None)["board_max_updated_at"]
dt.update_task(sid, {"status": "In Progress", "actor": "F"}, user=None)
after = dt.poll_task(pid, user=None)["board_max_updated_at"]
ok("board_max_updated_at strictly advances after an update (client refetches)",
   after > before)
ok("an unchanged board yields a STABLE probe (client skips the refetch)",
   dt.poll_task(pid, user=None)["board_max_updated_at"] == after)

print(f"\nALL PASS ({PASS} checks)")
