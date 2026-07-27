"""Acceptance tests — board #143 mentions + unread Messages inbox (V2.12).

Offline + hermetic: the REAL `api.routers.dev_tasks.add_comment` (which writes
mentions at POST time) and the REAL `api.routers.mentions` read/seen endpoints
run against an in-memory fake Supabase client — no network, no DB. Covers the
spec's acceptance walk:

  1. @tokens in a posted comment fan out to task_mentions — one row per
     DISTINCT registry role (letters + kevin, case-insensitive); unknown
     tokens and email local parts produce NO row; system/plain comments too.
  2. GET /api/mentions?for=R&unseen=true is the inbox (seen_at IS NULL),
     enriched with task title/status + comment excerpt.
  3. POST /api/mentions/{id}/seen clears one; the inbox shrinks.
  4. POST /api/mentions/seen-all?for=R clears the rest; returns the count.
  5. unseen=false reveals recently-seen history (the "show already-seen" view).
  6. A comment write NEVER fails even if the mentions table is missing
     (best-effort fan-out) — additive feature, comments are the primary action.
  7. Migration is additive + idempotent by construction.

Run:  .venv/bin/python src/test_mentions_143.py   (from RoR_Trader root)
"""
import os
import re
import sys
import types
from datetime import datetime, timedelta, timezone

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


# ── In-memory fake Supabase (the query-builder subset both routers use) ────

class _Res:
    def __init__(self, data):
        self.data = data


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class _Query:
    def __init__(self, store, tname):
        self.store, self.tname = store, tname
        self.op, self.payload, self.filters, self.ors = "select", None, [], None

    def select(self, *_):
        self.op = "select"
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row       # dict OR list of dicts
        return self

    def update(self, row):
        self.op, self.payload = "update", dict(row)
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, k, v):
        self.filters.append(("eq", k, v)); return self

    def neq(self, k, v):
        self.filters.append(("neq", k, v)); return self

    def is_(self, k, v):
        self.filters.append(("is", k, v)); return self

    def in_(self, k, vs):
        self.filters.append(("in", k, list(vs))); return self

    def gte(self, k, v):
        self.filters.append(("gte", k, v)); return self

    def or_(self, expr):
        self.ors = expr; return self

    def order(self, *_, **__):
        return self

    def limit(self, *_):
        return self

    @staticmethod
    def _cond(op, k, v, row):
        cur = row.get(k)
        if op == "eq":
            return cur == v
        if op == "neq":
            return cur != v
        if op == "is":
            return cur is None if v == "null" else cur == v
        if op == "in":
            return cur in v
        if op == "gte":
            return cur is not None and cur >= v
        return True

    def _match_ors(self, row):
        # minimal PostgREST or() parser: "col.op.val,col.op.val" (val may
        # contain dots — split into exactly 3 parts).
        for term in self.ors.split(","):
            col, op, val = term.split(".", 2)
            if self._cond(op, col, None if val == "null" else val, row):
                return True
        return False

    def _match(self, row):
        for op, k, v in self.filters:
            if not self._cond(op, k, v, row):
                return False
        if self.ors is not None and not self._match_ors(row):
            return False
        return True

    def _insert_one(self, rows, row):
        row = dict(row)
        row["id"] = self.store["_seq"] = self.store.get("_seq", 0) + 1
        if self.tname == "task_mentions":
            row.setdefault("created_at", _iso(datetime.now(timezone.utc)))
            row.setdefault("seen_at", None)
        if self.tname == "dev_task_comments":
            row.setdefault("created_at", _iso(datetime.now(timezone.utc)))
        if self.tname == "dev_tasks":
            row.setdefault("status", "Backlog")   # DB default (dev_tasks_table.sql)
        rows.append(row)
        return dict(row)

    def execute(self):
        rows = self.store.setdefault(self.tname, [])
        if self.op == "insert":
            payload = self.payload
            if isinstance(payload, list):
                return _Res([self._insert_one(rows, r) for r in payload])
            return _Res([self._insert_one(rows, payload)])
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

# Stub `db` BEFORE importing the routers (api.deps does `from db import …`).
db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

from fastapi import HTTPException          # noqa: E402
from api.routers import dev_tasks as dt    # noqa: E402
from api.routers import mentions as mn     # noqa: E402


def expect_http(label, code, fn, *a, **k):
    try:
        fn(*a, **k)
    except HTTPException as e:
        ok(label, e.status_code == code, f"(got {e.status_code}: {e.detail})")
        return
    print(f"FAIL: {label} — no HTTPException raised")
    sys.exit(1)


def mentions_rows():
    return FAKE.store.get("task_mentions", [])


# Seed the agents registry (the source of valid @tokens) + a couple of tasks.
for letter in ("M", "E", "E2", "F", "P", "R", "kevin"):
    FAKE.table("agents").insert({"letter": letter}).execute()
t1 = dt.create_task({"title": "Build the thing", "description": "d",
                     "assignee": "F"}, user=None)
t2 = dt.create_task({"title": "Second task", "description": "d",
                     "assignee": "E"}, user=None)
tid1, tid2 = t1["id"], t2["id"]

print("― 1. parse + write at comment-POST time ―")
c = dt.add_comment(tid1, {"body": "hey @F please build this, and @kevin "
                          "sign off. ping ktk@gmail.com, not @nobody",
                          "author": "kevin"}, user=None)
rows = mentions_rows()
ok("two mention rows written (F, kevin)",
   sorted(r["mentioned"] for r in rows) == ["F", "kevin"],
   f"(got {[r['mentioned'] for r in rows]})")
ok("email local part + unknown token made NO row",
   all(r["mentioned"] in ("F", "kevin") for r in rows))
ok("rows carry comment_id/task_id/author, seen_at NULL",
   all(r["comment_id"] == c["id"] and r["task_id"] == tid1
       and r["author"] == "kevin" and r["seen_at"] is None for r in rows))

# case-insensitive dedup: @f @F @kevin → one F, one kevin (distinct only)
dt.add_comment(tid2, {"body": "@f @F @kevin @KEVIN duplicate tags",
                      "author": "M"}, user=None)
t2_rows = [r for r in mentions_rows() if r["task_id"] == tid2]
ok("distinct-only per comment (case-insensitive dedup)",
   sorted(r["mentioned"] for r in t2_rows) == ["F", "kevin"])

# plain comment with no tokens writes no rows
before = len(mentions_rows())
dt.add_comment(tid1, {"body": "just a normal note", "author": "F"}, user=None)
ok("no-token comment writes nothing", len(mentions_rows()) == before)

print("― 2. inbox read (GET, enriched) ―")
inbox_f = mn.list_mentions(for_="F", unseen=True, user=None)
ok("F inbox has 2 unseen (from t1 + t2)", len(inbox_f) == 2)
ok("inbox enriched with task title/status + excerpt",
   all(m.get("task_title") and m.get("task_status")
       and isinstance(m.get("excerpt"), str) for m in inbox_f))
ok("kevin sees the direct tag from the agents",
   len(mn.list_mentions(for_="kevin", unseen=True, user=None)) == 2)
ok("a role nobody tagged has an empty inbox",
   mn.list_mentions(for_="P", unseen=True, user=None) == [])
expect_http("GET requires 'for'", 400, mn.list_mentions, for_="", user=None)

print("― 3. mark one seen ―")
one = inbox_f[0]
seen = mn.mark_seen(one["id"], user=None)
ok("mark_seen stamps seen_at", seen["seen_at"] is not None)
ok("F inbox shrinks to 1",
   len(mn.list_mentions(for_="F", unseen=True, user=None)) == 1)
expect_http("mark_seen 404 on unknown id", 404, mn.mark_seen, 999999, user=None)

print("― 4. mark all seen ―")
res = mn.mark_all_seen(for_="F", user=None)
ok("seen-all reports the count it cleared", res["marked"] == 1)
ok("F inbox now empty",
   mn.list_mentions(for_="F", unseen=True, user=None) == [])
ok("seen-all on an empty inbox marks 0",
   mn.mark_all_seen(for_="F", user=None)["marked"] == 0)

print("― 5. show already-seen (last 7d) ―")
seen_view = mn.list_mentions(for_="F", unseen=False, days=7, user=None)
ok("unseen=false reveals the recently-seen history", len(seen_view) == 2)

print("― 6. mentions never break a comment (best-effort) ―")
# Drop the table mid-flight: add_comment must still succeed.
saved = FAKE.store.pop("task_mentions", [])


class _Broken(FakeSupabase):
    def table(self, name):
        if name == "task_mentions":
            raise RuntimeError("relation \"task_mentions\" does not exist")
        return super().table(name)


broken = _Broken()
broken.store = FAKE.store
db_stub.get_admin_client = lambda: broken
out = dt.add_comment(tid1, {"body": "@M this must not 500", "author": "kevin"},
                     user=None)
ok("comment still posts when the mentions table is unavailable",
   out.get("body") == "@M this must not 500")
db_stub.get_admin_client = lambda: FAKE
FAKE.store["task_mentions"] = saved

print("― 7. migration additive + idempotent by construction ―")
mig = os.path.join(SRC, "migrations", "task_mentions_table.sql")
sql = open(mig).read()
ok("CREATE TABLE IF NOT EXISTS", "CREATE TABLE IF NOT EXISTS task_mentions" in sql)
ok("CREATE INDEX IF NOT EXISTS (mentioned, seen_at)",
   re.search(r"CREATE INDEX IF NOT EXISTS.*\(mentioned, seen_at\)", sql, re.S)
   is not None)
ok("FKs cascade from comments + tasks",
   "REFERENCES dev_task_comments(id) ON DELETE CASCADE" in sql
   and "REFERENCES dev_tasks(id) ON DELETE CASCADE" in sql)
ok("UNIQUE (comment_id, mentioned) makes writes idempotent",
   "UNIQUE (comment_id, mentioned)" in sql)
ok("RLS enabled, backfill guarded ON CONFLICT DO NOTHING",
   "ENABLE ROW LEVEL SECURITY" in sql
   and "ON CONFLICT (comment_id, mentioned) DO NOTHING" in sql)

print(f"\nALL {PASS} CHECKS PASSED ✅")
