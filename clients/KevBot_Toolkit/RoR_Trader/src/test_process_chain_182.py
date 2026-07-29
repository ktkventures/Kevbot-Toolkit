"""Acceptance test — board #182, Step 3: process-chain data model + server-side
transition rules on the dev_tasks API.

Proves the migration/API layer for the accordion process chain WITHOUT any UI:
  • back-compat  — legacy {role,text,done} checklists keep their exact pre-#182
                   free-toggle behavior (the 42 existing tasks are untouched);
  • widened step — richer step objects validate + get stable ids at birth;
  • T3 strict order — a generic PATCH cannot tick/untick a step; only
                   /steps/complete does, and only the first incomplete step;
  • T1/T2 hand-off — completing a step reassigns to the next owner; finishing
                   the chain reassigns to M (never kevin);
  • T4' immutability — a PATCH cannot untick or delete a completed step;
  • T8 stamp gate — a stamp-gated step can't complete until /steps/stamp
                   approves it;
  • T9 escalation — Raise-an-issue (comment required) and a rejected stamp both
                   route to M without ticking;
  • kevin_final derived — TRUE if any kevin-owned step carries a required stamp,
                   and never cleared;
  • step_order — comments are auto-tagged with the task's current step.

Offline + hermetic, same shape as test_stamp_ticks_kevin_step_151.py: the real
`api.routers.dev_tasks` router functions run against an in-memory fake Supabase
client. No DB, no network.

Run:  .venv/bin/python src/test_process_chain_182.py   (from RoR_Trader root)
"""
import copy
import os
import sys
import types

from fastapi import HTTPException

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
        # supabase accepts a single dict or a list of dicts.
        self.op = "insert"
        self.payload = [dict(r) for r in row] if isinstance(row, list) \
            else [dict(row)]
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
                if self.tname == "dev_tasks":
                    for k, v in (("status", "Backlog"), ("impact", "contained"),
                                 ("kevin_final", False),
                                 ("standing_approval", False), ("tags", []),
                                 ("checklist", []), ("blocked_by", []),
                                 ("origin", "planned"), ("assignee", None),
                                 ("description", ""), ("parent_id", None)):
                        row.setdefault(k, v)
                rows.append(row)
                out.append(_deep(row))
            return _Res(out)
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(_deep(self.payload))
            return _Res([_deep(r) for r in hit])
        if self.op == "delete":
            keep = [r for r in rows if not self._match(r)]
            self.store[self.tname] = keep
            return _Res([])
        return _Res([_deep(r) for r in rows if self._match(r)])


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return _Query(self.store, name)


FAKE = FakeSupabase()
# Seed the agents registry so @-mention fan-out resolves (as it does in prod).
FAKE.store["agents"] = [{"letter": ltr} for ltr in ("M", "E", "F", "P", "R")]
db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

from api.routers import dev_tasks as dt                # noqa: E402


def comments_for(tid):
    return [c for c in FAKE.store.get("dev_task_comments", [])
            if c["task_id"] == tid]


def bodies_for(tid):
    return [c["body"] for c in comments_for(tid)]


# A three-step chain: F builds, kevin stamps (gated), M closes.
def make_chain():
    return [
        {"owner": "F", "title": "Build the thing", "body": "SOP...",
         "mode": "execute", "done": False},
        {"owner": "kevin", "title": "Review — STAMP", "mode": "discuss",
         "stamp": {"required": True, "state": "pending"}, "done": False},
        {"owner": "M", "title": "Close", "mode": "execute", "done": False},
    ]


# ── walk 1: legacy checklists are untouched (the 42 existing tasks) ─────────
print("― walk 1: legacy {role,text,done} keeps pre-#182 free-toggle behavior ―")
legacy = [{"text": "a", "done": False, "role": "F"},
          {"text": "b", "done": False, "role": "M"}]
t = dt.create_task({"title": "legacy", "assignee": "F", "checklist": legacy},
                   user=None)
tid = t["id"]
ok("legacy checklist gets NO step ids (not a chain)",
   all("id" not in s for s in t["checklist"]))
# Free-toggle the SECOND step (out of order) — legacy path must allow it.
r = dt.update_task(tid, {"checklist": [
    {"text": "a", "done": False, "role": "F"},
    {"text": "b", "done": True, "role": "M"}]}, user=None)
ok("legacy out-of-order tick is allowed (no strict order pre-#182)",
   r["checklist"][1]["done"] is True and r["checklist"][0]["done"] is False)
ok("legacy PATCH does NOT auto-reassign", r["assignee"] == "F")
dt.add_comment(tid, {"body": "hi", "author": "F"}, user=None)
ok("legacy comment gets step_order = NULL (not a chain)",
   comments_for(tid)[-1]["step_order"] is None)


# ── walk 2: a process chain — birth, validation, PATCH guards ──────────────
print("― walk 2: process chain — ids at birth, PATCH is structure-only ―")
t2 = dt.create_task({"title": "chain", "assignee": "F",
                     "checklist": make_chain()}, user=None)
tid2 = t2["id"]
ok("every chain step gets a stable id at birth",
   all(s.get("id") for s in t2["checklist"]))
ok("kevin_final derived TRUE (a kevin step has a required stamp)",
   t2["kevin_final"] is True)
ids0 = [s["id"] for s in t2["checklist"]]

# Structural edit (rewrite an SOP body) is allowed and preserves ids.
edited = _deep(t2["checklist"])
edited[0]["body"] = "SOP rewritten"
r2 = dt.update_task(tid2, {"checklist": edited}, user=None)
ok("structural edit (body) is allowed",
   r2["checklist"][0]["body"] == "SOP rewritten")
ok("ids are preserved across a structural edit",
   [s["id"] for s in r2["checklist"]] == ids0)

# Ticking a step via the generic PATCH is refused (T3/T4' — API refusal).
tick = _deep(r2["checklist"])
tick[0]["done"] = True
raises("PATCH cannot tick a step", 409,
       lambda: dt.update_task(tid2, {"checklist": tick}, user=None))

# Inserting a step that arrives already-done is refused.
bad_new = _deep(r2["checklist"]) + [
    {"owner": "F", "title": "sneaky", "done": True}]
raises("a newly inserted step cannot start completed", 400,
       lambda: dt.update_task(tid2, {"checklist": bad_new}, user=None))


# ── walk 3: /steps/complete — strict order + hand-off (T1/T3) ──────────────
print("― walk 3: /steps/complete ticks the current step and hands off ―")
r3 = dt.complete_step(tid2, {"actor": "F"}, user=None)
ok("step 1 is now done with an audit stamp",
   r3["checklist"][0]["done"] is True
   and r3["checklist"][0]["completed_by"] == "F"
   and r3["checklist"][0]["completed_at"])
ok("assignee handed off to the next step's owner (kevin)",
   r3["assignee"] == "kevin")
ok("one hand-off system comment tagged to the completed step",
   any('step 1 "Build the thing" complete' in c["body"]
       and c["step_order"] == 0 for c in comments_for(tid2)))


# ── walk 4: T8 stamp gate — can't complete a gated step until approved ─────
print("― walk 4: T8 — stamp gate blocks completion until approved ―")
raises("gated step 2 cannot be completed while stamp is pending", 409,
       lambda: dt.complete_step(tid2, {"actor": "kevin"}, user=None))
# Rejecting routes to M and does not tick (T9).
rj = dt.stamp_step(tid2, {"decision": "reject", "actor": "kevin"}, user=None)
ok("reject sets stamp.state=rejected and does NOT tick",
   rj["checklist"][1]["stamp"]["state"] == "rejected"
   and rj["checklist"][1]["done"] is False)
ok("reject routes to M (T9 escalation)", rj["assignee"] == "M")
# Now approve, then complete succeeds.
ap = dt.stamp_step(tid2, {"decision": "approve", "actor": "kevin"}, user=None)
ok("approve sets stamp.state=approved", ap["checklist"][1]["stamp"]["state"] == "approved")
r4 = dt.complete_step(tid2, {"actor": "kevin"}, user=None)
ok("gated step 2 completes once approved", r4["checklist"][1]["done"] is True)
ok("assignee handed off to M (step 3 owner)", r4["assignee"] == "M")


# ── walk 5: T2 — finishing the chain lands on M, never kevin ───────────────
print("― walk 5: T2 — end of chain reassigns to M ―")
r5 = dt.complete_step(tid2, {"actor": "M"}, user=None)
ok("all steps done", all(s["done"] for s in r5["checklist"]))
ok("chain complete → assignee M (never kevin)", r5["assignee"] == "M")
raises("completing an already-finished chain is refused", 409,
       lambda: dt.complete_step(tid2, {"actor": "M"}, user=None))

# End-of-chain lands on M even when the LAST step is kevin-owned.
tk = dt.create_task({"title": "kevin-last", "assignee": "kevin",
                     "checklist": [
                         {"owner": "F", "title": "build", "done": False},
                         {"owner": "kevin", "title": "final look", "done": False},
                     ]}, user=None)
dt.complete_step(tk["id"], {"actor": "F"}, user=None)
rk = dt.complete_step(tk["id"], {"actor": "kevin"}, user=None)
ok("chain ending on a kevin step still hands to M (T2)", rk["assignee"] == "M")


# ── walk 6: T9 — Raise-an-issue requires a comment and routes to M ─────────
print("― walk 6: T9 — raise-an-issue ―")
t6 = dt.create_task({"title": "issue", "assignee": "F",
                     "checklist": make_chain()}, user=None)
tid6 = t6["id"]
raises("raise-issue with no comment is refused", 400,
       lambda: dt.raise_issue(tid6, {"actor": "F"}, user=None))
r6 = dt.raise_issue(tid6, {"body": "this SOP is wrong @M", "actor": "F"},
                    user=None)
ok("raise-issue routes to M", r6["assignee"] == "M")
ok("raise-issue does NOT tick any step",
   all(not s["done"] for s in r6["checklist"]))
ok("the explanation comment is posted, tagged to the current step (0)",
   any(c["body"] == "this SOP is wrong @M" and c["step_order"] == 0
       for c in comments_for(tid6)))
ok("@mention in the issue comment was fanned out",
   any(m["mentioned"] == "M" for m in FAKE.store.get("task_mentions", [])))


# ── walk 7: step_order auto-tag follows the current step ───────────────────
print("― walk 7: comment step_order tracks the current step ―")
t7 = dt.create_task({"title": "tagging", "assignee": "F",
                     "checklist": make_chain()}, user=None)
tid7 = t7["id"]
dt.add_comment(tid7, {"body": "on step 1", "author": "F"}, user=None)
ok("comment while step 1 is current → step_order 0",
   comments_for(tid7)[-1]["step_order"] == 0)
dt.complete_step(tid7, {"actor": "F"}, user=None)
dt.add_comment(tid7, {"body": "on step 2", "author": "kevin"}, user=None)
ok("comment after advancing → step_order 1",
   [c for c in comments_for(tid7) if c["body"] == "on step 2"][0]["step_order"] == 1)


# ── walk 8: kevin_final is derived and never cleared ───────────────────────
print("― walk 8: kevin_final derivation ―")
t8 = dt.create_task({"title": "no kevin stamp", "assignee": "F",
                     "checklist": [
                         {"owner": "F", "title": "x", "done": False}]},
                    user=None)
ok("no kevin-stamp step → kevin_final stays FALSE",
   t8["kevin_final"] is False)
# Add a kevin-owned required-stamp step via PATCH → derived TRUE.
r8 = dt.update_task(t8["id"], {"checklist": [
    {"owner": "F", "title": "x", "done": False},
    {"owner": "kevin", "title": "stamp", "stamp": {"required": True},
     "done": False}]}, user=None)
ok("adding a kevin required-stamp step derives kevin_final TRUE",
   r8["kevin_final"] is True)
# Removing it again must NOT clear an existing TRUE.
r8b = dt.update_task(t8["id"], {"checklist": [
    {"owner": "F", "title": "x", "done": False, "id": r8["checklist"][0]["id"]}]},
    user=None)
ok("kevin_final is never cleared once TRUE (protects the #136 guard)",
   r8b["kevin_final"] is True)


# ── walk 9: T4' — a completed step cannot be deleted by a PATCH ────────────
print("― walk 9: T4' — completed steps are immutable ―")
t9 = dt.create_task({"title": "immutable", "assignee": "F",
                     "checklist": [
                         {"owner": "F", "title": "one", "done": False},
                         {"owner": "M", "title": "two", "done": False}]},
                    user=None)
tid9 = t9["id"]
r9 = dt.complete_step(tid9, {"actor": "F"}, user=None)
survivor_id = r9["checklist"][1]["id"]
# Try to delete the completed step 1 (send only step 2).
raises("deleting a completed step is refused", 409,
       lambda: dt.update_task(tid9, {"checklist": [
           {"owner": "M", "title": "two", "done": False, "id": survivor_id}]},
           user=None))
# Un-ticking the completed step via PATCH is refused.
raises("un-ticking a completed step is refused", 409,
       lambda: dt.update_task(tid9, {"checklist": [
           {**r9["checklist"][0], "done": False}, r9["checklist"][1]]},
           user=None))


# ── walk 10: migration is additive + idempotent + non-destructive ──────────
print("― walk 10: migration hygiene ―")
mig = open(os.path.join(SRC, "migrations", "dev_tasks_process_chain.sql")).read()
ok("adds step_order with IF NOT EXISTS (idempotent)",
   "ADD COLUMN IF NOT EXISTS step_order" in mig)
low = mig.lower()
ok("no destructive statements",
   not any(x in low for x in ("drop table", "drop column", "delete from",
                              "truncate", "alter column")))


print(f"\nALL PASS ({PASS} checks)")
