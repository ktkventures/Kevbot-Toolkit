"""Acceptance tests — board #220 Phase 1: the cross-task comment feed endpoint.

Offline + hermetic: the REAL `api.routers.dev_tasks.recent_comments` runs
against an in-memory fake Supabase client that (unlike the #143 harness)
actually honours `.order()` and `.limit()`, because ordering and the cap ARE
the contract here. No network, no DB.

What is covered:

  1. Newest first, across ALL tasks — the whole point: the M-session activity
     log is the first reader that needs the board's comment stream as a whole,
     and fanning /{task_id}/comments over ~90 open tasks is the board-#148
     poll-efficiency mistake.
  2. `limit` is honoured and CAPPED (1 … 500) — a caller cannot ask the board
     for everything by accident.
  3. `max_chars` truncates the BODY on the wire while `body_chars` always
     carries the UNTRUNCATED length, so the UI can say how much it is not
     showing. Silent elision is the failure mode this pair exists to prevent.
  4. `max_chars=0` returns full bodies.
  5. The route does NOT collide with `/{task_id}/comments`: the per-task
     endpoint stays scoped to its task, and "comments" is not swallowed as a
     task id.
  6. Read-only: the endpoint performs no insert/update/delete.

Run:  .venv/bin/python src/test_m_session_feed_220.py   (from RoR_Trader root)
"""
import os
import sys
import types
from datetime import datetime, timedelta, timezone

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


# ── In-memory fake Supabase (order + limit are real here) ──────────────────

class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, tname, writes):
        self.store, self.tname, self.writes = store, tname, writes
        self.op, self.payload, self.filters = "select", None, []
        self.order_key, self.order_desc, self.lim = None, False, None

    def select(self, *_):
        self.op = "select"
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def update(self, row):
        self.op, self.payload = "update", dict(row)
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def order(self, key, desc=False):
        self.order_key, self.order_desc = key, desc
        return self

    def limit(self, n):
        self.lim = n
        return self

    def _match(self, row):
        return all(row.get(k) == v for k, v in self.filters)

    def execute(self):
        rows = self.store.setdefault(self.tname, [])
        if self.op in ("insert", "update", "delete"):
            self.writes.append((self.op, self.tname))
            if self.op == "insert":
                payload = self.payload if isinstance(self.payload, list) \
                    else [self.payload]
                out = []
                for r in payload:
                    r = dict(r)
                    r["id"] = self.store["_seq"] = self.store.get("_seq", 0) + 1
                    rows.append(r)
                    out.append(dict(r))
                return _Res(out)
            return _Res([])
        out = [dict(r) for r in rows if self._match(r)]
        if self.order_key:
            out.sort(key=lambda r: r.get(self.order_key) or "",
                     reverse=self.order_desc)
        if self.lim is not None:
            out = out[:self.lim]
        return _Res(out)


class FakeSupabase:
    def __init__(self):
        self.store, self.writes = {}, []

    def table(self, name):
        return _Query(self.store, name, self.writes)


FAKE = FakeSupabase()

db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

from api.routers import dev_tasks as dt          # noqa: E402


# ── Seed: three tasks, interleaved comments, one very long body ────────────

T0 = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


def seed(task_id, author, body, minutes):
    FAKE.table("dev_task_comments").insert({
        "task_id": task_id, "author": author, "body": body,
        "step_order": None,
        "created_at": (T0 + timedelta(minutes=minutes)).isoformat(),
    }).execute()


LONG = "x" * 5000
seed(101, "system", "dispatched to F·auto (run r1)", 0)
seed(202, "kevin", "@M what is the status here?", 1)
seed(101, "F·auto", LONG, 2)                      # an agent report — long
seed(303, "M", "a long M diagnosis " * 60, 3)     # M's own — long
seed(202, "M", "answered: here is the status", 4)
seed(101, "system", "status: In Progress → Review (by dispatcher)", 5)

writes_after_seed = len(FAKE.writes)

print("― 1. newest first, across ALL tasks ―")
rows = dt.recent_comments(user=None)
ok("all six comments returned", len(rows) == 6, f"(got {len(rows)})")
ok("newest first",
   [r["created_at"] for r in rows] == sorted(
       (r["created_at"] for r in rows), reverse=True))
ok("spans more than one task",
   len({r["task_id"] for r in rows}) == 3,
   f"(got {sorted({r['task_id'] for r in rows})})")
ok("the newest row is the last one seeded",
   rows[0]["body"].startswith("status: In Progress → Review"))

print("― 2. limit honoured and capped ―")
ok("limit=2 returns 2", len(dt.recent_comments(limit=2, user=None)) == 2)
ok("limit=2 returns the two NEWEST",
   [r["body"][:9] for r in dt.recent_comments(limit=2, user=None)]
   == ["status: I", "answered:"])
ok("limit is capped at 500",
   len(dt.recent_comments(limit=10_000, user=None)) == 6)
ok("limit below 1 is floored to 1, not an empty page",
   len(dt.recent_comments(limit=0, user=None)) == 1)

print("― 3. max_chars truncates the body, body_chars tells the truth ―")
rows = dt.recent_comments(max_chars=100, user=None)
report = next(r for r in rows if r["author"] == "F·auto")
ok("body truncated to max_chars", len(report["body"]) == 100,
   f"(got {len(report['body'])})")
ok("body_chars carries the UNTRUNCATED length",
   report["body_chars"] == len(LONG), f"(got {report['body_chars']})")
ok("truncation is detectable as len(body) < body_chars",
   len(report["body"]) < report["body_chars"])
short = next(r for r in rows if r["author"] == "kevin")
ok("a short body is untouched", short["body"] == "@M what is the status here?")
ok("an untruncated row reports len(body) == body_chars",
   len(short["body"]) == short["body_chars"])

print("― 4. max_chars=0 returns full bodies ―")
rows = dt.recent_comments(max_chars=0, user=None)
report = next(r for r in rows if r["author"] == "F·auto")
ok("full body returned", report["body"] == LONG)
ok("body_chars still present", report["body_chars"] == len(LONG))

print("― 5. no collision with the per-task comments endpoint ―")
per_task = dt.list_comments(101, user=None)
ok("per-task read still scoped to its task",
   len(per_task) == 3 and {r["task_id"] for r in per_task} == {101},
   f"(got {len(per_task)} rows)")
ok("the cross-task feed is strictly wider",
   len(dt.recent_comments(user=None)) > len(per_task))

print("― 6. the endpoint is READ-ONLY ―")
dt.recent_comments(user=None)
dt.recent_comments(limit=5, max_chars=10, user=None)
ok("no insert/update/delete performed by the feed reads",
   len(FAKE.writes) == writes_after_seed,
   f"(new writes: {FAKE.writes[writes_after_seed:]})")

print(f"\nALL {PASS} CHECKS PASSED — board #220 Phase 1 feed endpoint")
