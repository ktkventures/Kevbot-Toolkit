"""Acceptance tests — board #220 step 4: sections 3 + 4 and M's own store.

Offline + hermetic: the REAL `api.routers.m_session` runs against an in-memory
fake Supabase client and against REAL files on disk for the rules render. No
network, no DB, no DDL.

The task's rail is *"Every rail must have a test that FAILS without it — a rail
with no failing test is a comment, not a rail."* So every check below is
paired to a rail, and each one goes red if its rail is removed:

  RAIL 1 · THE RULES ARE RENDERED FROM SOURCE, VERBATIM.
    Not a retyped summary — a hand-maintained copy is a second source of truth
    that diverges silently (the untracked-copy failure, three times on 07-30).
    Tests: every rendered block is a byte-for-byte SUBSTRING of its file, and
    the charter §4 block really does carry the assignee rule's own words.

  RAIL 2 · A MISSING ANCHOR FAILS LOUD.
    A renamed heading must produce an ERROR, never an empty block: "there are
    no rules" and "the rules moved" render identically and debug differently.
    Tests: missing anchor → error, missing file → error naming the absolute
    path tried, empty section → error. `found` is False in every case and the
    endpoint still returns 200 so the PAGE can show the failure.

  RAIL 3 · THE STORE NEVER SHADOWS A BOARD-OWNED FIELD.
    Design principle B: "if it starts storing task state, it will drift from
    the board and Kevin will have two answers to the same question." Tests:
    every write endpoint 400s on status/assignee/title/priority — at the
    payload top level AND inside an ordering item — and an ordering item may
    carry nothing but task_id + why.

  RAIL 4 · RANK IS THE ARRAY POSITION, AND THE PLAN IS REPLACED WHOLE.
    A caller states "this is my plan, in order"; it cannot hand in colliding
    or stale ranks, and a task dropped from the plan LOSES its rank rather
    than keeping a stale intention.

  RAIL 5 · THE ASK-AI REGEX MUST NOT MATCH `@M-A` (step-3 review BLOCKER).
    Board #222 makes `M-A` a real owner. The shipped lookahead is read out of
    `mSessionFeed.ts` and exercised here, so this test is bound to the actual
    source rather than to a copy of it.

  RAIL 6 · AN UNPROVISIONED STORE DEGRADES LOUDLY, NOT SILENTLY.
    The migration is authored and UNAPPLIED. A read must say so in a form the
    page can render in red — never an empty canvas that looks like M wrote
    nothing.

Run:  .venv/bin/python src/test_m_session_state_220.py   (from RoR_Trader root)
"""
import os
import re
import sys
import types
from pathlib import Path

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


def raises(label, fn, status=None, contains=None):
    from fastapi import HTTPException
    try:
        fn()
    except HTTPException as e:
        detail = str(e.detail)
        if status is not None and e.status_code != status:
            ok(label, False, f"(status {e.status_code}, wanted {status})")
        if contains and contains.lower() not in detail.lower():
            ok(label, False, f"(detail {detail!r} lacks {contains!r})")
        ok(label, True)
        return
    ok(label, False, "(no HTTPException raised)")


# ── In-memory fake Supabase (upsert + delete are real here) ────────────────

class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, tname):
        self.store, self.tname = store, tname
        self.op, self.payload, self.filters, self.negs = "select", None, [], []
        self.order_key, self.order_desc, self.lim = None, False, None

    def select(self, *_):
        self.op = "select"
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def upsert(self, row):
        self.op, self.payload = "upsert", row
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def neq(self, k, v):
        self.negs.append((k, v))
        return self

    def order(self, key, desc=False):
        self.order_key, self.order_desc = key, desc
        return self

    def limit(self, n):
        self.lim = n
        return self

    def _match(self, row):
        return (all(row.get(k) == v for k, v in self.filters)
                and all(row.get(k) != v for k, v in self.negs))

    def _pk(self):
        return {"m_session_canvas": "id",
                "m_session_queue_order": "task_id",
                "m_session_event_marks": "event_key"}.get(self.tname)

    def execute(self):
        if self.tname in MISSING_TABLES:
            raise RuntimeError(
                f'Could not find the table \'public.{self.tname}\' in the '
                f'schema cache (PGRST205)')
        rows = self.store.setdefault(self.tname, [])
        if self.op == "insert":
            payload = self.payload if isinstance(self.payload, list) \
                else [self.payload]
            rows.extend(dict(r) for r in payload)
            return _Res([dict(r) for r in payload])
        if self.op == "upsert":
            pk = self._pk()
            payload = self.payload if isinstance(self.payload, list) \
                else [self.payload]
            for r in payload:
                for i, existing in enumerate(rows):
                    if existing.get(pk) == r.get(pk):
                        rows[i] = dict(r)
                        break
                else:
                    rows.append(dict(r))
            return _Res([dict(r) for r in payload])
        if self.op == "delete":
            keep = [r for r in rows if not self._match(r)]
            removed = len(rows) - len(keep)
            self.store[self.tname] = keep
            return _Res([{"removed": removed}])
        out = [dict(r) for r in rows if self._match(r)]
        if self.order_key:
            out.sort(key=lambda r: (r.get(self.order_key) is None,
                                    r.get(self.order_key) or ""),
                     reverse=self.order_desc)
        if self.lim is not None:
            out = out[:self.lim]
        return _Res(out)


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return _Query(self.store, name)


FAKE = FakeSupabase()
MISSING_TABLES = set()

db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

from api.routers import m_session as ms          # noqa: E402

ROR_ROOT = Path(SRC).parent


# ── RAIL 1 · the rules are rendered FROM SOURCE, verbatim ──────────────────

print("― RAIL 1. rendered from source, byte-for-byte ―")
payload = ms.rules(user=None)
blocks = {b["id"]: b for b in payload["sources"]}
ok("every declared source is rendered",
   len(payload["sources"]) == len(ms.RULE_SOURCES))
ok("no source is missing on a full checkout", payload["missing"] == 0,
   f'(missing: {[b["id"] for b in payload["sources"] if not b["found"]]})')

for src in ms.RULE_SOURCES:
    b = blocks[src.id]
    raw = src.resolve().read_text(encoding="utf-8")
    ok(f"{src.id}: text is a VERBATIM substring of {src.path}",
       b["text"] and b["text"] in raw)
    ok(f"{src.id}: block starts at its anchor {src.anchor!r}",
       b["text"].startswith(src.anchor))

# The point of rendering rather than restating: the actual binding words are
# present. If someone summarised §4 this check would go red.
ok("charter §4 carries the assignee rule's own words",
   "THE ASSIGNEE IS WHOEVER THE TASK IS WAITING ON" in blocks["charter-4"]["text"])
ok("the rails block carries the verbatim migration rail",
   "Author the migration, do NOT apply it" in blocks["chain-rails"]["text"])
ok("the priority order puts Ask AI first",
   "Ask AI" in blocks["priority-order"]["text"]
   and blocks["priority-order"]["text"].index("Ask AI")
   < blocks["priority-order"]["text"].index("dispatch/ship"))

# A section stops at the next heading of the SAME OR SHALLOWER depth — so §4
# does not swallow §5, and a block containing sub-headings keeps them.
ok("§4 stops before §5", "## 5." not in blocks["charter-4"]["text"])
ok("§7 stops before §8", "## 8." not in blocks["charter-7"]["text"])
ok("the rails block keeps its own sub-headings (### A/B/C)",
   blocks["chain-rails"]["text"].count("\n### ") >= 3)
ok("the rails block stops at the next ## heading",
   "## Procedure" not in blocks["chain-rails"]["text"])


# ── RAIL 2 · a missing anchor / file FAILS LOUD ────────────────────────────

print("― RAIL 2. fail loud on a missing anchor or file ―")
body, err = ms.extract_section("## 4. Real\ntext\n", "## 9.")
ok("a missing anchor returns an ERROR, not an empty string",
   body is None and err and "not found" in err)
body, err = ms.extract_section("## 4. Real\n\n\n## 5. Next\n", "## 4.")
ok("an anchor matching an EMPTY section is an error too",
   body is None and err and "EMPTY" in err)
body, err = ms.extract_section("## 4. Real\nkeep\n## 5. Next\n", "## 4.")
ok("a real anchor still renders", body == "## 4. Real\nkeep" and err is None)
ok("a '#' inside a line is not mistaken for a heading",
   ms.extract_section("## A\nsee #220 here\n## B\n", "## A")[0]
   == "## A\nsee #220 here")

gone = ms.RuleSource("gone", "Gone", "ror", "docs/_active/__nope__.md",
                     "## 1.", "n/a")
blk = ms.render_rule_source(gone)
ok("a missing FILE is reported, not skipped",
   blk["found"] is False and "not found" in (blk["error"] or ""))
ok("the failure names the ABSOLUTE path it tried",
   blk["resolved_path"].endswith("docs/_active/__nope__.md")
   and blk["resolved_path"].startswith("/"))

bad_anchor = ms.RuleSource("bad", "Bad", "ror",
                           "docs/_active/Session_Charters.md",
                           "## 99. Nope", "n/a")
blk = ms.render_rule_source(bad_anchor)
ok("a renamed/renumbered heading is reported as an error",
   blk["found"] is False and "not found" in (blk["error"] or "")
   and blk["text"] == "")

# The endpoint must still return 200 with a `missing` count — the PAGE is what
# renders the red. A raise here would blank the section instead.
saved = list(ms.RULE_SOURCES)
try:
    ms.RULE_SOURCES.append(gone)
    p2 = ms.rules(user=None)
    ok("the endpoint still returns a payload when a source is missing",
       isinstance(p2, dict) and p2["missing"] == 1)
    ok("`missing` is the count the page renders in red",
       sum(1 for b in p2["sources"] if not b["found"]) == p2["missing"])
finally:
    ms.RULE_SOURCES[:] = saved


# ── RAIL 3 · the store never shadows a board-owned field ───────────────────

print("― RAIL 3. board-owned fields are REFUSED, not stripped ―")
for field in ("status", "assignee", "title", "priority_seq", "checklist"):
    raises(f"canvas rejects a payload carrying `{field}`",
           lambda f=field: ms.put_canvas({"body": "x", f: "Done"}, user=None),
           status=400, contains="owned by the board")
    raises(f"order rejects a payload carrying `{field}`",
           lambda f=field: ms.put_order({"items": [], f: "Done"}, user=None),
           status=400, contains="owned by the board")
    raises(f"marks rejects a payload carrying `{field}`",
           lambda f=field: ms.post_mark(
               {"event_key": "comment:1", "state": "seen", f: "Done"}, user=None),
           status=400, contains="owned by the board")

raises("an ORDER ITEM carrying `status` is refused",
       lambda: ms.put_order(
           {"items": [{"task_id": 220, "status": "Done"}]}, user=None),
       status=400, contains="owned by the board")
raises("an order item may carry ONLY task_id + why",
       lambda: ms.put_order(
           {"items": [{"task_id": 220, "why": "next", "rank": 99}]}, user=None),
       status=400, contains="unexpected field")
ok("BOARD_OWNED_FIELDS covers the fields the board actually owns",
   {"status", "assignee", "title", "priority_phase", "priority_seq"}
   <= ms.BOARD_OWNED_FIELDS)


# ── RAIL 4 · rank is the array position; the plan is replaced whole ────────

print("― RAIL 4. rank comes from position, plan replaced whole ―")
res = ms.put_order({"items": [{"task_id": 220, "why": "Ask AI first"},
                              {"task_id": 143},
                              {"task_id": 182, "why": "then review"}],
                    "actor": "M"}, user=None)
ok("all three items stored", res["count"] == 3)
ok("rank is 1..n in the order given",
   [r["rank"] for r in res["items"]] == [1, 2, 3])
ok("rank follows POSITION, not any caller-supplied number",
   [r["task_id"] for r in res["items"]] == [220, 143, 182])
ok("`why` round-trips, and is None when absent",
   res["items"][0]["why"] == "Ask AI first" and res["items"][1]["why"] is None)

state = ms.get_state(user=None)
ok("the store reads back in rank order",
   [r["task_id"] for r in state["order"]] == [220, 143, 182])
ok("the stored row holds NO board-owned field",
   not (ms.BOARD_OWNED_FIELDS & set(state["order"][0])),
   f"(row keys: {sorted(state['order'][0])})")

ms.put_order({"items": [{"task_id": 143, "why": "moved to the top"}]}, user=None)
state = ms.get_state(user=None)
ok("a task dropped from the plan LOSES its rank (no stale intention)",
   [r["task_id"] for r in state["order"]] == [143])
raises("a duplicate task_id is refused",
       lambda: ms.put_order(
           {"items": [{"task_id": 7}, {"task_id": 7}]}, user=None),
       status=400, contains="duplicate")
raises("a non-integer task_id is refused",
       lambda: ms.put_order({"items": [{"task_id": "abc"}]}, user=None),
       status=400)

print("― canvas + marks round-trip ―")
ms.put_canvas({"body": "## Doing\nshipping #220", "actor": "M"}, user=None)
state = ms.get_state(user=None)
ok("canvas body round-trips", state["canvas"]["body"] == "## Doing\nshipping #220")
ok("canvas records who wrote it", state["canvas"]["updated_by"] == "M")
ms.put_canvas({"body": "rewritten", "actor": "M"}, user=None)
ok("the canvas is a singleton — a second write REPLACES, never appends",
   len(FAKE.store["m_session_canvas"]) == 1
   and ms.get_state(user=None)["canvas"]["body"] == "rewritten")
raises("a canvas write with no body is refused",
       lambda: ms.put_canvas({"actor": "M"}, user=None), status=400)

ms.post_mark({"event_key": "comment:41", "state": "actioned",
              "note": "replied on the thread", "task_id": 220}, user=None)
ms.post_mark({"event_key": "env:stranded-branch", "state": "no-action",
              "note": "known, tracked on #218"}, user=None)
marks = {m["event_key"]: m for m in ms.get_state(user=None)["marks"]}
ok("a comment event is marked", marks["comment:41"]["state"] == "actioned")
ok("an ENV event with NO task can be marked — Phase 2 needs no migration",
   marks["env:stranded-branch"]["state"] == "no-action"
   and marks["env:stranded-branch"]["task_id"] is None)
ms.post_mark({"event_key": "comment:41", "state": "seen"}, user=None)
ok("re-marking the same key UPDATES rather than duplicating",
   len([m for m in ms.get_state(user=None)["marks"]
        if m["event_key"] == "comment:41"]) == 1)
raises("an unknown mark state is refused",
       lambda: ms.post_mark(
           {"event_key": "comment:9", "state": "probably-fine"}, user=None),
       status=400, contains="state")
ms.delete_mark("comment:41", user=None)
ok("a mark can be UNDONE — a wrong 'I looked' is false coverage",
   "comment:41" not in {m["event_key"]
                        for m in ms.get_state(user=None)["marks"]})


# ── RAIL 5 · the Ask-AI regex must not match `@M-A` (step-3 BLOCKER) ───────

print("― RAIL 5. ASK_AI_RE excludes the hyphen (#222 `M-A`) ―")
FEED_TS = (ROR_ROOT / "frontend/src/views/mSessionFeed.ts").read_text(encoding="utf-8")


def js_regex(name):
    """Pull a regex LITERAL out of the shipped TS and compile it in Python.

    Read from the source rather than restated here on purpose: a copy of the
    pattern in this file would pass while the shipped one was wrong, which is
    exactly the failure this test exists to catch.
    """
    m = re.search(rf"export const {name} = /(.+?)/([a-z]*);", FEED_TS)
    assert m, f"{name} literal not found in mSessionFeed.ts"
    flags = re.IGNORECASE if "i" in m.group(2) else 0
    return re.compile(m.group(1), flags)


ASK_AI = js_regex("ASK_AI_RE")
ASK_KEVIN = js_regex("ASK_KEVIN_RE")

ok("`@M please look` MATCHES — Kevin's fast lane still fires",
   bool(ASK_AI.search("@M please look")))
ok("`@M\\n` matches (the button posts '@M ' then the body)",
   bool(ASK_AI.search("@M\nwhat is the status?")))
ok("leading whitespace is tolerated", bool(ASK_AI.search("  @M hi")))
ok("`@M-A build this` does NOT match — #222 makes M-A a real owner, and "
   "agent traffic in this inbox degrades the one signal it exists for",
   not ASK_AI.search("@M-A build this"))
ok("`@M-A2 ...` does not match either", not ASK_AI.search("@M-A2 go"))
ok("`@MA nope` does not match", not ASK_AI.search("@MA nope"))
ok("`@Meta no` does not match", not ASK_AI.search("@Meta no"))
ok("a mid-body `@M` is a mention, not the fast lane",
   not ASK_AI.search("please ask @M about this"))
ok("ASK_KEVIN_RE matches `@kevin` and is case-insensitive",
   bool(ASK_KEVIN.search("@Kevin can you confirm?")))
ok("ASK_KEVIN_RE excludes the hyphen too",
   not ASK_KEVIN.search("@kevin-bot ping"))


# ── RAIL 6 · an unprovisioned store degrades LOUDLY ────────────────────────

print("― RAIL 6. unapplied migration says so; it never looks empty ―")
MISSING_TABLES.update({"m_session_canvas", "m_session_queue_order",
                       "m_session_event_marks"})
state = ms.get_state(user=None)
ok("a read reports available=False rather than 500ing",
   state["available"] is False)
ok("the reason names the unapplied migration",
   "m_session_state.sql" in (state["reason"] or ""))
ok("an unprovisioned read returns EMPTY collections, not stale data",
   state["order"] == [] and state["marks"] == [] and state["canvas"] is None)
raises("a canvas WRITE against an unprovisioned store 503s with the reason",
       lambda: ms.put_canvas({"body": "x"}, user=None),
       status=503, contains="not provisioned")
raises("an order WRITE against an unprovisioned store 503s",
       lambda: ms.put_order({"items": []}, user=None), status=503)
raises("a mark WRITE against an unprovisioned store 503s",
       lambda: ms.post_mark({"event_key": "k", "state": "seen"}, user=None),
       status=503)
MISSING_TABLES.clear()

# A real error must NOT be swallowed as "unprovisioned" — that would turn a
# genuine outage into a quiet "the migration isn't applied yet".
ok("_unprovisioned() is specific to a missing table",
   ms._unprovisioned(RuntimeError("PGRST205 could not find the table"))
   and not ms._unprovisioned(RuntimeError("connection reset by peer")))


# ── The migration is AUTHORED and matches what the code reads ──────────────

print("― the migration file exists, and is not applied by anything here ―")
mig = ROR_ROOT / "src/migrations/m_session_state.sql"
sql = mig.read_text(encoding="utf-8")
ok("m_session_state.sql is authored", mig.exists())
for t in ("m_session_canvas", "m_session_queue_order", "m_session_event_marks"):
    ok(f"{t} is created by the migration", f"CREATE TABLE IF NOT EXISTS {t}" in sql)
ok("the migration is additive — no DROP/ALTER of an existing table",
   "DROP TABLE" not in sql.upper()
   and "ALTER TABLE DEV_TASKS" not in sql.upper())
ok("RLS is enabled on all three tables, matching the dev_tasks posture",
   sql.upper().count("ENABLE ROW LEVEL SECURITY") == 3)
# The store must not carry a board-owned column — the schema-level half of
# RAIL 3. `task_id` is a reference, not a copy of task state. Comments are
# stripped first: the file NAMES those fields to say it is not storing them,
# and a check that could not tell prose from DDL would be untrustworthy in
# both directions.
ddl = " ".join(line.split("--")[0] for line in sql.splitlines())
for col in ("status", "assignee", "priority_seq", "checklist", "title"):
    ok(f"the migration declares no `{col}` column",
       not re.search(rf"^\s*{col}\s", ddl, re.I)
       and f" {col} " not in ddl)
ok("no code in this router applies DDL",
   "CREATE TABLE" not in (ROR_ROOT / "src/api/routers/m_session.py")
   .read_text(encoding="utf-8").upper())

print(f"\nALL {PASS} CHECKS PASSED — board #220 step 4 "
      f"(sections 3 + 4, M's store, the #222 hyphen fix)")
