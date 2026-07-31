"""Acceptance test — board #248: a status change names its actor, and the
PostgREST bypass that logged nothing at all is refused.

#248 was FILED as "the board has no `closed_by` column and no audit table" and
that filing was wrong: `update_task` has logged `status: X → Y (by <actor>)` to
the comment thread since #136, the browser sends `actor` on every PATCH, and so
does the dispatcher. The real gap is two-sided, and the bigger side was M's own
practice — measured against the live board on 2026-07-31:

  * 175 of 178 finished tasks have NO arrival line at ALL. Those closes were
    PATCHed through service-role PostgREST (`d.api('PATCH', 'dev_tasks?...')`),
    which bypasses `/api/dev-tasks` entirely, so nothing was ever written. Not
    anonymous — ABSENT. It is why the board watcher, with no data to read, fell
    back to convention and announced one of M's closes as "CLOSED by kevin".
  * 12 activity lines DID go through the API but carried no `actor`, so they
    read `status: Backlog → Todo` with nobody's name on them.

(The filing's "65 of the last 200 system comments" counted every system comment
lacking a `(by …)` suffix — 183 all-time, of which 145 are `dispatched to X·auto
(run …)` lines that record a dispatch, not a status change, and correctly have
no actor. The status/assignee-line figure is 12 of 182.)

So the fix is a refusal on each path, plus the callers that would have tripped
it. Every rail here fails without its line of the fix:

  R1  API: PATCH {"status": …} with no actor   → 400 naming the field
  R2  API: … with a blank/whitespace actor     → 400 (present ≠ named)
  R3  API: PATCH {"status": …, "actor": "M"}   → 200 + the attributed line
  R4  API: a NON-status PATCH is unaffected    → title/tags/ai_eligible still 200
  R5  API: assignee-ONLY is deliberately open  → the #171 hand-off path survives
  R6  dispatcher.api(): a raw status PATCH     → AnonymousStatusWrite, no HTTP
  R7  … other dev_tasks PATCHes still pass     → the guard is status-scoped
  R8  set_status() writes the attributed line  → same wording as the API's
  R9  set_status() keeps the patch ATOMIC      → `extra` rides with the status
  R10 set_status() with no actor is refused    → no silent "system" default
  R11 set_status() on a no-op does NOT log     → history is not manufactured
  R12 set_status(old=…) does NO read           → an unreadable board can't kill a reap
  R13 the dispatcher's own status writes all   → 0 raw sites left in the file
      go through it

R1–R5 run over real HTTP through a FastAPI TestClient so `Depends(...)` is
actually exercised. R6–R11 drive the real `dispatcher.api()` with `_creds`
stubbed, so the refusal is proven at the chokepoint rather than at a mock of it.
Offline + hermetic: fake Supabase, no DB, no network.

Run:  .venv/bin/python src/test_status_change_actor_248.py   (from RoR_Trader root)
"""
import ast
import base64
import copy
import importlib
import importlib.util
import json
import os
import re
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


# ── In-memory fake Supabase (same subset as test_steps_actor_auth_217.py) ────

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
        self.op = "insert"
        self.payload = [dict(r) for r in row] if isinstance(row, list) \
            else [dict(row)]
        return self

    def update(self, row):
        self.op, self.payload = "update", dict(row)
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def neq(self, *_):
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
            out = []
            for payload in self.payload:
                row = dict(payload)
                row["id"] = self.store["_seq"] = self.store.get("_seq", 0) + 1
                rows.append(row)
                out.append(copy.deepcopy(row))
            return _Res(out)
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(copy.deepcopy(self.payload))
            return _Res([copy.deepcopy(r) for r in hit])
        return _Res([copy.deepcopy(r) for r in rows if self._match(r)])


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return _Query(self.store, name)


FAKE = FakeSupabase()
FAKE.store["agents"] = [{"letter": ltr} for ltr in ("M", "E", "F", "P", "R")]
db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

SERVICE_KEY = "eyJhbGciOiJIUzI1NiJ9." + base64.urlsafe_b64encode(
    json.dumps({"role": "service_role",
                "exp": int(time.time()) + 10 * 365 * 86400}).encode()
).decode().rstrip("=") + ".sig"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = SERVICE_KEY
# The rails under test are about ATTRIBUTION, not authentication — #217 already
# proves the auth gate. Bypass keeps the client focused on the 400 vs 200 line.
os.environ["DEV_BYPASS_AUTH"] = "true"

from api import deps                                    # noqa: E402
importlib.reload(deps)
from api.routers import dev_tasks as dt                 # noqa: E402
importlib.reload(dt)

from fastapi import FastAPI                             # noqa: E402
from fastapi.testclient import TestClient               # noqa: E402

app = FastAPI()
app.include_router(dt.router)
client = TestClient(app)


def make_task(**over):
    row = {"title": "task under test", "assignee": "M", "status": "Todo"}
    row.update(over)
    return dt.create_task(row, user=None)["id"]


def comments_for(tid):
    return [c["body"] for c in FAKE.store.get("dev_task_comments", [])
            if c["task_id"] == tid]


def status_lines(tid):
    return [b for b in comments_for(tid) if b.startswith("status: ")]


# ── R1: a status PATCH with no actor is REFUSED ─────────────────────────────
print("― R1: PATCH {status} with no actor → 400 ―")
tid = make_task()
r = client.patch(f"/api/dev-tasks/{tid}", json={"status": "In Progress"})
ok("no-actor status PATCH → 400 (was 200 + an anonymous line)",
   r.status_code == 400, f"got {r.status_code}: {r.text[:200]}")
ok("the refusal NAMES the field the caller must send",
   "actor" in r.text, f"got {r.text[:200]}")
ok("the row did NOT move — the refusal is before the write",
   FAKE.store["dev_tasks"][-1]["status"] == "Todo",
   f"got {FAKE.store['dev_tasks'][-1]['status']}")
ok("and nothing was logged for it",
   status_lines(tid) == [], f"got {status_lines(tid)}")

# ── R2: PRESENT is not the same as NAMED ────────────────────────────────────
print("― R2: a blank actor is still no actor ―")
for blank in ("", "   ", None):
    r = client.patch(f"/api/dev-tasks/{tid}",
                     json={"status": "In Progress", "actor": blank})
    ok(f"actor={blank!r} → 400", r.status_code == 400,
       f"got {r.status_code}: {r.text[:120]}")

# ── R3: with an actor it works, and the line names them ─────────────────────
print("― R3: PATCH {status, actor} → 200 + attributed line ―")
r = client.patch(f"/api/dev-tasks/{tid}",
                 json={"status": "In Progress", "actor": "M"})
ok("named status PATCH → 200", r.status_code == 200,
   f"got {r.status_code}: {r.text[:200]}")
ok("the row moved", r.json()["status"] == "In Progress")
ok("the activity line names the actor",
   any(b == "status: Todo → In Progress (by M)" for b in status_lines(tid)),
   f"got {status_lines(tid)}")

# ── R4: a NON-status PATCH is untouched ─────────────────────────────────────
print("― R4: non-status PATCHes are unaffected ―")
t2 = make_task()
for field, value in (("title", "renamed"), ("tags", ["x"]),
                     ("ai_eligible", True), ("notes", "n"),
                     ("priority_seq", 3)):
    r = client.patch(f"/api/dev-tasks/{t2}", json={field: value})
    ok(f"PATCH {{{field}}} with no actor → 200 (still open, by design)",
       r.status_code == 200, f"got {r.status_code}: {r.text[:160]}")
ok("the edits landed", FAKE.store["dev_tasks"][-1]["title"] == "renamed")

# ── R5: assignee-only stays open — the #171 hand-off path ───────────────────
print("― R5: assignee-only is deliberately NOT gated (board #171) ―")
r = client.patch(f"/api/dev-tasks/{t2}", json={"assignee": "kevin"})
ok("assignee-only PATCH with no actor → 200 — every headless run does this "
   "on its way out; gating it would break the hand-off",
   r.status_code == 200, f"got {r.status_code}: {r.text[:160]}")
ok("it still logs the transition (anonymously — the known, accepted residue)",
   any("assignee: M → kevin" in b for b in comments_for(t2)),
   f"got {comments_for(t2)}")
r = client.patch(f"/api/dev-tasks/{t2}",
                 json={"status": "Review", "assignee": "M"})
ok("but a patch carrying BOTH is gated on the status half",
   r.status_code == 400, f"got {r.status_code}: {r.text[:160]}")

# ── dispatcher side ─────────────────────────────────────────────────────────
DISPATCHER_PY = os.environ.get(
    "DISPATCHER_PY", os.path.join(ROOT, "tools", "team_dispatcher",
                                  "dispatcher.py"))
spec = importlib.util.spec_from_file_location("team_dispatcher", DISPATCHER_PY)
D = importlib.util.module_from_spec(spec)
sys.modules["team_dispatcher"] = D
spec.loader.exec_module(D)
print(f"\ndispatcher under test: {DISPATCHER_PY}")

# A board that RECORDS what the real api() would have sent. _creds is stubbed
# rather than api(), so the guard in the real api() is what runs.
BOARD = {"dev_tasks": [{"id": 900, "status": "Todo", "tags": ["run-requested"]}],
         "dev_task_comments": []}
SENT = []


def fake_urlopen(req, timeout=None):
    path = req.full_url.split("/rest/v1/", 1)[1]
    body = json.loads(req.data.decode()) if req.data else None
    SENT.append((req.get_method(), path, body))
    out = None
    if req.get_method() == "GET" and path.startswith("dev_tasks?id=eq."):
        tid = int(re.search(r"id=eq\.(\d+)", path).group(1))
        out = [r for r in BOARD["dev_tasks"] if r["id"] == tid]
    elif req.get_method() == "PATCH" and path.startswith("dev_tasks?id=eq."):
        tid = int(re.search(r"id=eq\.(\d+)", path).group(1))
        for r in BOARD["dev_tasks"]:
            if r["id"] == tid:
                r.update(body)
        out = []
    elif req.get_method() == "POST" and path == "dev_task_comments":
        BOARD["dev_task_comments"].append(body)
        out = []

    class _R:
        def read(self_inner):
            return json.dumps(out if out is not None else []).encode()

        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False
    return _R()


D._creds = lambda: ("https://fake.supabase.co", "svc-key")
D.urllib.request.urlopen = fake_urlopen


def dcomments():
    return [c["body"] for c in BOARD["dev_task_comments"]]


# ── R6: the raw bypass is REFUSED at the chokepoint ─────────────────────────
print("― R6: dispatcher.api() refuses a raw status PATCH ―")
before = len(SENT)
try:
    D.api("PATCH", "dev_tasks?id=eq.900", body={"status": "Done"})
    ok("raw status PATCH raises", False, "it was allowed through")
except D.AnonymousStatusWrite as e:
    ok("raw status PATCH raises AnonymousStatusWrite", True)
    ok("the message points at set_status", "set_status" in str(e), str(e)[:160])
    ok("…and says WHY (it bypasses the API / logs nothing)",
       "bypass" in str(e).lower() and "248" in str(e), str(e)[:200])
ok("NO HTTP request was made — refused before the wire",
   len(SENT) == before, f"{len(SENT) - before} request(s) escaped")
ok("the board did not move",
   BOARD["dev_tasks"][0]["status"] == "Todo")

# ── R7: the guard is STATUS-scoped, not a blanket dev_tasks ban ─────────────
print("― R7: other dev_tasks PATCHes still pass ―")
D.api("PATCH", "dev_tasks?id=eq.900", body={"tags": []})
ok("a tags-only PATCH goes through (clear_run_request depends on it)",
   BOARD["dev_tasks"][0]["tags"] == [])
D.api("PATCH", "run_history?task_id=eq.900", body={"outcome": "ok"})
ok("a run_history PATCH carrying 'status'-free keys goes through", True)
D.api("PATCH", "run_history?task_id=eq.900", body={"status": "whatever"})
ok("a NON-dev_tasks PATCH is not policed even with a status key — the guard "
   "is about the board's audit trail, not the word 'status'", True)

# ── R8: set_status writes the API's own line ────────────────────────────────
print("― R8: set_status() PATCHes AND attributes ―")
old = D.set_status(900, "In Progress", "dispatcher · run r1-248",
                   note="claimed for M-A·auto")
ok("it returns the PREVIOUS status (the #197 pre-dispatch read)",
   old == "Todo", f"got {old}")
ok("the row moved", BOARD["dev_tasks"][0]["status"] == "In Progress")
line = [b for b in dcomments() if b.startswith("status: ")]
ok("exactly one status line was written", len(line) == 1, f"got {line}")
ok("it matches the API's wording, with the actor named",
   line[0] == "status: Todo → In Progress (by dispatcher · run r1-248) "
              "— claimed for M-A·auto", f"got {line[0]!r}")
ok("and it is authored 'system', like the API's",
   BOARD["dev_task_comments"][-1]["author"] == "system")

# ── R9: the claim stays ATOMIC ──────────────────────────────────────────────
print("― R9: `extra` rides in the SAME patch as the status ―")
BOARD["dev_tasks"][0]["tags"] = ["run-requested"]
SENT.clear()
D.set_status(900, "Review", "dispatcher · run r2-248",
             extra={"tags": [], "notes": "done"})
patches = [b for m, p, b in SENT if m == "PATCH" and p.startswith("dev_tasks")]
ok("one PATCH, not two", len(patches) == 1, f"got {patches}")
ok("carrying status AND the extras together",
   patches[0] == {"tags": [], "notes": "done", "status": "Review"},
   f"got {patches[0]}")
ok("board reflects both", BOARD["dev_tasks"][0]["tags"] == []
   and BOARD["dev_tasks"][0]["status"] == "Review")

# ── R10: no silent default actor ────────────────────────────────────────────
print("― R10: set_status() with no actor is refused ―")
for blank in ("", "   ", None):
    try:
        D.set_status(900, "Done", blank)
        ok(f"actor={blank!r} raises", False, "it was allowed")
    except D.AnonymousStatusWrite:
        ok(f"actor={blank!r} raises AnonymousStatusWrite — no 'system' "
           f"fallback, which is how the field became worthless", True)
ok("the board did not move on any of them",
   BOARD["dev_tasks"][0]["status"] == "Review")

# ── R11: a no-op does not manufacture history ───────────────────────────────
print("― R11: re-setting the same status writes no line ―")
n = len(dcomments())
D.set_status(900, "Review", "M")
ok("row unchanged", BOARD["dev_tasks"][0]["status"] == "Review")
ok("no new activity line", len(dcomments()) == n, f"{len(dcomments()) - n} added")

# ── R12: an unreadable board must not break a reap (board #197's rail) ──────
# Caught by #197's own suite while building this: the first cut of set_status
# did a blind `GET dev_tasks?id=eq.N` for the previous status, which reintroduced
# exactly the failure #197 fixed — reap MUST finish on a board it cannot read.
# Every caller in dispatcher.py knows the old status already, so they pass it.
print("― R12: set_status(old=…) does no read; a read failure never propagates ―")
SENT.clear()
D.set_status(900, "Done", "M", old="Review")
gets = [p for m, p, _ in SENT if m == "GET"]
ok("passing `old` issues NO read at all", not gets, f"got {gets}")
ok("and still writes the line from the supplied old",
   dcomments()[-1] == "status: Review → Done (by M)", f"got {dcomments()[-1]!r}")

real_urlopen = D.urllib.request.urlopen


def _blind(req, timeout=None):
    if req.get_method() == "GET":
        raise RuntimeError("simulated board read failure")
    return real_urlopen(req, timeout)


D.urllib.request.urlopen = _blind
D.set_status(900, "Blocked", "dispatcher · run r3-248")   # `old` omitted
ok("an omitted `old` + unreadable board DEGRADES instead of raising",
   BOARD["dev_tasks"][0]["status"] == "Blocked")
ok("…and says so on the line rather than inventing a previous status",
   "(unreadable) → Blocked (by dispatcher · run r3-248)" in dcomments()[-1],
   f"got {dcomments()[-1]!r}")
D.urllib.request.urlopen = real_urlopen

# Parsed, not grepped: `set_status(` also appears in the docstring and in the
# refusal message, and counting strings is how a rail passes for a bad reason.
src_now = open(DISPATCHER_PY, encoding="utf-8").read()
calls = [n for n in ast.walk(ast.parse(src_now))
         if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "set_status"]
blind = [n.lineno for n in calls
         if "old" not in [k.arg for k in n.keywords]]
ok(f"all {len(calls)} set_status calls in the dispatcher pass `old` — none "
   f"needs a read, so none can die on one", not blind,
   f"blind calls at line(s) {blind}")

# ── R13: no raw status PATCH survives in the dispatcher ─────────────────────
print("― R13: the dispatcher has no bypass left ―")
src = open(DISPATCHER_PY, encoding="utf-8").read()
# Every `api("PATCH", f"dev_tasks...` call site, with its body argument.
sites = re.findall(r'api\(\s*"PATCH",\s*f?"dev_tasks[^)]*?\)', src, re.S)
offenders = [s for s in sites if '"status"' in s]
ok(f"0 of {len(sites)} dev_tasks PATCH sites write a raw status "
   f"(pre-fix: 5)", not offenders, f"offenders: {offenders}")
ok("set_status is used by the reap/claim legs",
   src.count("set_status(") >= 6,
   f"only {src.count('set_status(')} references")
ok("the hand-rolled copy of the audit line is gone — one writer, no drift",
   "(by dispatcher — " not in src)

print(f"\nALL {PASS} CHECKS PASSED — board #248: a status change names its "
      f"actor, on both paths.")
