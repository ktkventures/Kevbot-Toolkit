"""Acceptance test — board #198, Step 2: the `ai_eligible` column + API.

#198 splits the two jobs the kanban `status` had been doing at once — WHERE a
task sits in its lifecycle, and MAY an AI agent run it. Step 2 is the data +
API half only:

  • round-trip   — ai_eligible is settable at create, readable back, and
                   flips both ways through PATCH;
  • default OFF  — a task created without it is FALSE (the migration's
                   `NOT NULL DEFAULT false`), so no existing task is newly armed;
  • fail loud    — a non-boolean is a 400, not a silent coerce
                   (feedback_no_silent_defaults);
  • the stamp arms it — Kevin's Approval stamp sets ai_eligible=true in BOTH
                   modes and names it on the system comment. His review stays
                   the arming gate; the guarantee just stops riding on `Todo`;
  • INERT until Step 4 — the dispatcher still gates on `status=eq.Todo`, so an
                   ai_eligible task parked anywhere else does NOT dispatch yet
                   and a Todo task with ai_eligible=false STILL does. This is
                   the "no existing task's behaviour changed" proof: Step 2
                   changes nothing about what runs.

Offline + hermetic, same shape as test_stamp_ticks_kevin_step_151.py: the real
`api.routers.dev_tasks` router functions run against an in-memory fake Supabase
client, and the real `tools/team_dispatcher/dispatcher.py` gate runs against a
recorded fake `api()` that honours the `status=eq.` filter. No DB, no network.

Run:  .venv/bin/python src/test_ai_eligible_198.py   (from RoR_Trader root)
"""
import copy
import importlib.util
import os
import re
import sys
import types

from fastapi import HTTPException

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


def raises(label, code, fn):
    try:
        fn()
    except HTTPException as e:
        ok(f"{label} (HTTP {code})", e.status_code == code,
           f"got {e.status_code}: {e.detail}")
        return
    ok(f"{label} (HTTP {code})", False, "no HTTPException raised")


# ── In-memory fake Supabase (query-builder subset the router uses) ─────────

def _deep(row):
    return copy.deepcopy(row)


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
                # Column defaults, mirroring the migrations — ai_eligible's
                # `NOT NULL DEFAULT false` is dev_tasks_ai_eligible.sql.
                for k, v in (("status", "Backlog"), ("impact", "contained"),
                             ("kevin_final", False),
                             ("standing_approval", False),
                             ("ai_eligible", False),
                             ("tags", []), ("checklist", []),
                             ("blocked_by", []), ("origin", "planned"),
                             ("assignee", None), ("description", ""),
                             ("is_urgent", False), ("priority_phase", 1),
                             ("priority_seq", 99), ("parent_id", None)):
                    row.setdefault(k, v)
            rows.append(row)
            return _Res([_deep(row)])
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(_deep(self.payload))
            return _Res([_deep(r) for r in hit])
        return _Res([_deep(r) for r in rows if self._match(r)])


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


def comments_for(tid):
    return [c["body"] for c in FAKE.store.get("dev_task_comments", [])
            if c["task_id"] == tid]


def mk(title, **kw):
    """Create a task with the fields the dispatcher gate needs to pass."""
    payload = {"title": title, "description": "scoped", "assignee": "F"}
    payload.update(kw)
    return dt.create_task(payload, user=None)


# ── walk 1: the column round-trips through the API ─────────────────────────

print("― walk 1: ai_eligible round-trips (create → read → PATCH both ways) ―")
t = mk("armed at birth", ai_eligible=True)
ok("create accepts ai_eligible=true and returns it",
   t["ai_eligible"] is True)

off = dt.update_task(t["id"], {"ai_eligible": False}, user=None)
ok("PATCH disarms it (true → false)", off["ai_eligible"] is False)

on = dt.update_task(t["id"], {"ai_eligible": True}, user=None)
ok("PATCH re-arms it (false → true)", on["ai_eligible"] is True)

ok("a plain read sees the persisted value",
   dt.update_task(t["id"], {}, user=None)["ai_eligible"] is True)

ok("arming does NOT move the card — status is untouched",
   on["status"] == "Backlog")


# ── walk 2: default OFF — nothing is armed by the migration landing ────────

print("― walk 2: default OFF (no existing task is newly armed) ―")
plain = mk("untouched legacy task")
ok("a task created without the field is FALSE, not None",
   plain["ai_eligible"] is False)

legacy = dt.update_task(plain["id"], {"status": "Todo", "actor": "M"},
                        user=None)
ok("editing other fields never sets it implicitly",
   legacy["ai_eligible"] is False and legacy["status"] == "Todo")


# ── walk 3: fail loud on a non-boolean (no silent coercion) ────────────────

print("― walk 3: non-boolean is a 400, never coerced ―")
for bad in ("true", 1, None, [], {"on": True}):
    raises(f"create with ai_eligible={bad!r} refused", 400,
           lambda b=bad: dt.create_task(
               {"title": "bad", "ai_eligible": b}, user=None))
raises("PATCH with ai_eligible='yes' refused", 400,
       lambda: dt.update_task(plain["id"], {"ai_eligible": "yes"}, user=None))
ok("the refused PATCH left the row alone",
   dt.update_task(plain["id"], {}, user=None)["ai_eligible"] is False)


# ── walk 4: Kevin's Approval stamp arms it ─────────────────────────────────
# Board #232 cut the two stamp modes down to ONE ("approved to start"); this walk
# still runs the legacy 'delegate' alias beside it, because the arming behaviour
# #198 added must be identical on both spellings.

print("― walk 4: the Approval stamp sets ai_eligible=true ―")
for mode, label in (("start", "approved to start"), ("delegate", "legacy alias")):
    st = mk(f"awaiting stamp ({mode})",
            checklist=[{"text": "Kevin stamps", "done": False,
                        "role": "kevin"},
                       {"text": "Build", "done": False, "role": "F"}])
    sid = st["id"]
    ok(f"[{mode}] unstamped task starts disarmed",
       st["ai_eligible"] is False)
    dt.update_task(sid, {"status": "Approval", "actor": "M"}, user=None)
    ok(f"[{mode}] still disarmed while it sits in Approval",
       dt.update_task(sid, {}, user=None)["ai_eligible"] is False)

    r = dt.stamp_approval(sid, {"mode": mode, "author": "kevin"}, user=None)
    ok(f"[{mode}] the stamp ARMS the task", r["ai_eligible"] is True)
    ok(f"[{mode}] the stamp still moves Approval → Todo (unchanged)",
       r["status"] == "Todo")
    ok(f"[{mode}] the stamp no longer sets kevin_final (board #232 — the "
       f"stamp is permission to START, not a promise to review)",
       r["kevin_final"] is False)
    ok(f"[{mode}] the system comment names the arming",
       any("ai_eligible: true" in b and "stamp: approved" in b
           for b in comments_for(sid)))
    ok(f"[{mode}] still exactly ONE stamp comment",
       sum("stamp: approved" in b for b in comments_for(sid)) == 1)
    ok(f"[{mode}] #151 checklist tick survives (comment keeps both notes)",
       any('checklist: ticked "Kevin stamps"' in b
           for b in comments_for(sid)))

raises("stamping a task that is not in Approval is still a 409", 409,
       lambda: dt.stamp_approval(plain["id"], {"mode": "start"}, user=None))


# ── walk 5: END TO END — the API's column reaches the real dispatch gate ───
# Written for Step 2, when the dispatcher still gated on `status=eq.Todo` and this
# walk proved the column was INERT. Step 4 landed the gate, so the same walk now
# proves the other half of the additive claim: what the API arms, the dispatcher
# dispatches — and what it does not arm behaves exactly as it did before #198.
# (The gate's own rails — step-guard, completed-chain, every pre-#198 refusal —
# live in test_dispatcher_ai_eligible_gate_198.py.)

print("― walk 5: an ARMED task reaches the real dispatch gate ―")
spec = importlib.util.spec_from_file_location(
    "team_dispatcher",
    os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)

# `Review`, not `Staged`: the step-8 audible made `Staged` TERMINAL in the gate
# (a release ships via R on M's brief, never by self-dispatch), so it can no
# longer stand in for "some non-Todo column". `Review` is mid-pipeline and still
# dispatches, which is what this walk actually asserts.
armed_elsewhere = mk("armed but in Review", ai_eligible=True)
dt.update_task(armed_elsewhere["id"], {"status": "Review", "actor": "M"},
               user=None)
todo_unarmed = mk("plain Todo, never armed")
dt.update_task(todo_unarmed["id"], {"status": "Todo", "actor": "M"},
               user=None)

STATUS_RE = re.compile(r"status=eq\.([^&]+)")
OR_RE = re.compile(r"or=\(([^)]*)\)")

backlog_unarmed = mk("plain Backlog, never armed")
dt.update_task(backlog_unarmed["id"], {"status": "Backlog", "actor": "M"},
               user=None)


def fake_api(method, path, body=None, prefer=None):
    """Serve the dispatcher's PostgREST query from the fake board, HONOURING the
    filter it was given — the Step-4 OR gate or a bare `status=eq.` — so a gate
    that ignored its own filter would be caught."""
    board = [_deep(r) for r in FAKE.store["dev_tasks"]]
    m = OR_RE.search(path)
    if m:
        keep = []
        for r in board:
            for clause in m.group(1).split(","):
                col, _, val = clause.partition(".eq.")
                if (r.get(col) is True if val == "true"
                        else str(r.get(col)) == val):
                    keep.append(r)
                    break
        return keep
    m = STATUS_RE.search(path)
    if m:
        board = [r for r in board if r.get("status") == m.group(1)]
    return board


disp.api = fake_api
agents = {"F": {"letter": "F", "status": "headless", "worktree": ".",
                "scope": "s", "boundaries": "b"}}
got, _skipped = disp.triage_todo(agents, set())
got_ids = [x["id"] for x in got]

ok("ai_eligible=true on a NON-Todo (Review) task DISPATCHES — the column the "
   "API sets is the column the gate reads (Step 4)",
   armed_elsewhere["id"] in got_ids, f"got={got_ids}")
ok("a Todo task with ai_eligible=false STILL dispatches "
   "(today's queue is untouched)",
   todo_unarmed["id"] in got_ids)
ok("an UNARMED non-Todo task still does NOT dispatch (additive, not a free-for-all)",
   backlog_unarmed["id"] not in got_ids, f"got={got_ids}")
ok("the stamped tasks from walk 4 are in the queue as before",
   len(got_ids) >= 3)
ok("the dispatcher gate reads ai_eligible (Step 4 shipped in the same branch)",
   "ai_eligible" in
   open(os.path.join(ROOT, "tools", "team_dispatcher",
                     "dispatcher.py")).read())


# ── walk 6: whitelist discipline holds ─────────────────────────────────────

print("― walk 6: unknown fields are still dropped ―")
ok("ai_eligible is on the PATCH whitelist", "ai_eligible" in dt._EDITABLE)
w = dt.update_task(plain["id"],
                   {"ai_eligible": True, "bogus_column": "x"}, user=None)
ok("the real field landed", w["ai_eligible"] is True)
ok("the unknown field never reached the row", "bogus_column" not in w)


# ── walk 7: closing a task DISARMS it (board #198 audible, step 5) ─────────
# The other half of the audible. The dispatcher's TERMINAL_STATUSES rail is the
# guarantee; this is hygiene, so the refusal never has to fire and the board's
# toggle column tells the truth about a closed card.

print("― walk 7: status → Done clears ai_eligible ―")
closed = mk("armed, then closed", ai_eligible=True)
ok("armed before the close", closed["ai_eligible"] is True)
r = dt.update_task(closed["id"], {"status": "Done", "actor": "M"}, user=None)
ok("closing the task DISARMS it", r["ai_eligible"] is False)
ok("...and the close itself still happened", r["status"] == "Done")
ok("...the status transition comment is unchanged",
   any("status:" in b and "Done" in b for b in comments_for(closed["id"])))
ok("...it stays disarmed on a later no-op read",
   dt.update_task(closed["id"], {}, user=None)["ai_eligible"] is False)

contra = mk("closed and armed in one patch")
r = dt.update_task(contra["id"],
                   {"status": "Done", "ai_eligible": True, "actor": "M"},
                   user=None)
ok("a CONTRADICTORY patch (Done + armed) resolves to disarmed — closing wins",
   r["ai_eligible"] is False and r["status"] == "Done")

# Narrow: only a CLOSE disarms. Review / In Progress are mid-flight, so disarming
# there would stop the chain re-arming itself — the whole point of #198. `Staged`
# keeps its flag too, even though the gate now refuses to dispatch it (step-8
# audible): the flag records that Kevin ARMED the task, and it must survive the
# release so the task is still armed for whatever follows the train. The gate, not
# the flag, is what withholds a Staged dispatch.
for s in ("Review", "Staged", "In Progress", "Blocked"):
    t = mk(f"armed and moved to {s}", ai_eligible=True)
    r = dt.update_task(t["id"], {"status": s, "actor": "M"}, user=None)
    ok(f"moving to {s} does NOT disarm (only a close does)",
       r["ai_eligible"] is True, str(r["ai_eligible"]))

reopened = dt.update_task(closed["id"], {"status": "Todo", "actor": "M"},
                          user=None)
ok("re-opening a closed task does not silently re-arm it (Kevin's stamp is "
   "still the arming gate)", reopened["ai_eligible"] is False)

# BELT AND BRACES: a `Done` + armed row that got there WITHOUT this endpoint (a
# direct Supabase write, or a future close path that forgets) is still refused by
# the dispatcher — which is exactly why both rails exist.
bypass = mk("armed Done via a path that skipped the API", ai_eligible=True)
for row in FAKE.store["dev_tasks"]:
    if row["id"] == bypass["id"]:
        row["status"] = "Done"        # no PATCH → no disarm
ok("the bypassed row really is Done AND armed",
   [r for r in FAKE.store["dev_tasks"] if r["id"] == bypass["id"]][0]
   ["ai_eligible"] is True)
disp.api = fake_api
got_ids = [x["id"] for x in disp.triage_todo(agents, set(), {"runs": []})[0]]
ok("the dispatcher refuses it anyway (TERMINAL_STATUSES — the rail that cannot "
   "be forgotten)", bypass["id"] not in got_ids, f"got={got_ids}")

print(f"\nALL PASS ({PASS} checks)")
