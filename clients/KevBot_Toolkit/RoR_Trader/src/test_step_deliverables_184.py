"""Acceptance test — board #184, Step 3: step deliverables (capture surface).

Proves the API/storage layer WITHOUT any UI, mirroring
src/test_process_chain_182.py (same in-memory fake Supabase, same walk style):

  • back-compat   — a step with NO `deliverables` key behaves byte-identically
                    to pre-#184 (the rail that makes this shippable at all);
  • §2 ownership  — a generic PATCH may edit deliverable SPECS but can NEVER
                    fill or clear one (409); a NEW deliverable arriving filled
                    is a 400; a COMPLETED step's deliverables are frozen (409);
                    dropping a FILLED one on an open step is ALLOWED and posts a
                    system comment naming what was lost;
  • §1 limits     — >8 per step, over-long label, over-long text, non-URL link,
                    over-size file (413), disallowed extension (415);
  • §3 endpoints  — fill / upload / clear / signed-url, incl. wrong-kind 409,
                    unknown-id 404, and fill on an OPEN NON-CURRENT step
                    (allowed — evidence arrives early);
  • T10           — complete with a required-unfilled deliverable → 409 naming
                    the label; with only optional-unfilled → SUCCEEDS, and the
                    system comment names them;
  • §3.6 stamps   — a task-level /stamp on a CHAIN task writes
                    step.stamp{state,by,at} AND completed_at/completed_by AND
                    hands off AND is subject to T10; on a LEGACY checklist it is
                    unchanged; a kevin step that is not the current step falls
                    back to the pre-#184 direct tick (never ticks the wrong one).

Offline + hermetic: the real `api.routers.dev_tasks` functions run against an
in-memory fake Supabase client, with a fake Storage that records uploads. No DB,
no network, no bucket.

Run:  .venv/bin/python src/test_step_deliverables_184.py   (from RoR_Trader root)
      (bare `python3` raises ModuleNotFoundError: fastapi)
"""
import base64
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
        return e
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
            self.store[self.tname] = [r for r in rows if not self._match(r)]
            return _Res([])
        return _Res([_deep(r) for r in rows if self._match(r)])


class _FakeBucketOps:
    """Records uploads + hands back a fake signed URL. The point is that the
    router uses the SAME `get_admin_client().storage.from_(B)` shape
    bar_cache_store.py:125 does — one client path, not two."""

    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload(self, path, blob, opts=None):
        if self.store.get("_fail"):
            raise RuntimeError("Bucket not found")
        self.store.setdefault("_objects", {})[path] = (blob, dict(opts or {}))
        return {"path": path}

    def create_signed_url(self, path, ttl):
        if path not in self.store.get("_objects", {}):
            raise RuntimeError("object not found")
        return {"signedURL": f"https://sig.example/{self.name}/{path}?ttl={ttl}"}


class _FakeStorage:
    def __init__(self, store):
        self.store = store

    def from_(self, name):
        return _FakeBucketOps(self.store, name)


class FakeSupabase:
    def __init__(self):
        self.store = {}
        self.storage = _FakeStorage(self.store)

    def table(self, name):
        return _Query(self.store, name)


FAKE = FakeSupabase()
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


def task_row(tid):
    return [r for r in FAKE.store["dev_tasks"] if r["id"] == tid][0]


def deliv(tid, step_i, d_i=0):
    return task_row(tid)["checklist"][step_i]["deliverables"][d_i]


def make_deliv(kind="link", label="link to the merged PR", required=False,
               hint=""):
    return {"kind": kind, "label": label, "required": required, "hint": hint}


# ══ walk 1: BACK-COMPAT — a step with no `deliverables` key is untouched ═════
print("― walk 1: back-compat — no `deliverables` key ⇒ pre-#184 behaviour ―")
plain = [
    {"owner": "F", "title": "Build", "mode": "execute", "done": False},
    {"owner": "M", "title": "Close", "mode": "execute", "done": False},
]
t = dt.create_task({"title": "plain chain", "assignee": "F",
                    "checklist": copy.deepcopy(plain)}, user=None)
tid = t["id"]
ok("a chain step with no deliverables gets NO deliverables key written",
   all("deliverables" not in s for s in t["checklist"]))
r = dt.complete_step(tid, {"actor": "F"}, user=None)
ok("complete_step still ticks + hands off", r["checklist"][0]["done"] is True
   and r["assignee"] == "M", r["assignee"])
ok("its system comment has NO optional-deliverable tail",
   "optional deliverable" not in bodies_for(tid)[-1], bodies_for(tid)[-1])
ok("completion provenance is recorded as before",
   r["checklist"][0]["completed_by"] == "F"
   and bool(r["checklist"][0]["completed_at"]))

# Legacy {role,text,done} checklists must stay legacy — `deliverables` is a
# new-shape key, so it must not accidentally promote a legacy list.
legacy = [{"text": "a", "done": False, "role": "F"}]
t_leg = dt.create_task({"title": "legacy", "assignee": "F",
                        "checklist": copy.deepcopy(legacy)}, user=None)
ok("legacy checklist is still not a chain (no ids assigned)",
   all("id" not in s for s in t_leg["checklist"]))


# ══ walk 2: authoring specs through the generic PATCH ════════════════════════
print("― walk 2: §2 — a PATCH authors SPECS, never fills ―")
chain = [
    {"owner": "F", "title": "Build the thing", "mode": "execute", "done": False,
     "deliverables": [make_deliv(required=True),
                      make_deliv(kind="file", label="a screenshot")]},
    {"owner": "kevin", "title": "Review — STAMP", "mode": "discuss",
     "stamp": {"required": True, "state": "pending"}, "done": False},
    {"owner": "M", "title": "Close", "mode": "execute", "done": False},
]
t2 = dt.create_task({"title": "deliv chain", "assignee": "F",
                     "checklist": copy.deepcopy(chain)}, user=None)
t2id = t2["id"]
ds = t2["checklist"][0]["deliverables"]
ok("deliverables get server-assigned stable ids at birth",
   all(len(d.get("id", "")) == 12 for d in ds), ds)
ok("a step carrying only `deliverables` counts as a chain (step ids assigned)",
   all(s.get("id") for s in t2["checklist"]))
d0_id, d1_id = ds[0]["id"], ds[1]["id"]

# Editing a SPEC is allowed.
cl = copy.deepcopy(t2["checklist"])
cl[0]["deliverables"][0]["label"] = "link to the PR (merged)"
cl[0]["deliverables"][0]["hint"] = "the github URL"
r = dt.update_task(t2id, {"checklist": cl}, user=None)
ok("a PATCH may edit a deliverable spec (label/hint)",
   r["checklist"][0]["deliverables"][0]["label"] == "link to the PR (merged)")

# Filling through a generic PATCH is refused.
cl = copy.deepcopy(r["checklist"])
cl[0]["deliverables"][0]["value"] = "https://github.com/x/y/pull/1"
e = raises("a PATCH that FILLS a deliverable", 409,
           lambda: dt.update_task(t2id, {"checklist": cl}, user=None))
ok("...and the refusal names the endpoint that owns fills",
   "/steps/deliverables" in e.detail, e.detail)

# A NEW deliverable may not arrive pre-filled.
cl = copy.deepcopy(r["checklist"])
cl[0]["deliverables"].append(
    dict(make_deliv(kind="text", label="a note"), value="already answered"))
raises("a newly inserted deliverable arriving FILLED", 400,
       lambda: dt.update_task(t2id, {"checklist": cl}, user=None))

# A brand-NEW step may not arrive with filled deliverables either.
cl = copy.deepcopy(r["checklist"])
cl.append({"owner": "P", "title": "smuggled", "done": False,
           "deliverables": [dict(make_deliv(kind="text", label="x"),
                                 value="smuggled in")]})
raises("a newly inserted STEP carrying a filled deliverable", 400,
       lambda: dt.update_task(t2id, {"checklist": cl}, user=None))


# ══ walk 3: §1 limits — every one fails loud and names itself ════════════════
print("― walk 3: §1 limits — loud, named refusals ―")


def patch_first_step_deliverables(items):
    cl = copy.deepcopy(task_row(t2id)["checklist"])
    cl[0]["deliverables"] = items
    return lambda: dt.update_task(t2id, {"checklist": cl}, user=None)


raises("> 8 deliverables on one step", 400, patch_first_step_deliverables(
    [make_deliv(kind="text", label=f"q{i}") for i in range(9)]))
raises("a label over 120 chars", 400,
       patch_first_step_deliverables([make_deliv(label="x" * 121)]))
raises("an unknown kind", 400,
       patch_first_step_deliverables([make_deliv(kind="dropdown")]))
raises("an empty label", 400, patch_first_step_deliverables([make_deliv(label="  ")]))
raises("required that isn't a boolean", 400,
       patch_first_step_deliverables([make_deliv(required="yes")]))
ok("the stored chain survived every refusal unchanged",
   [d["id"] for d in task_row(t2id)["checklist"][0]["deliverables"]] == [d0_id, d1_id])


# ══ walk 4: §3.1/§3.3 — fill and clear a text/link deliverable ═══════════════
print("― walk 4: §3.1/§3.3 — fill · clear · the refusals around them ―")
raises("filling an unknown deliverable id", 404,
       lambda: dt.fill_deliverable(t2id, "deadbeefcafe",
                                   {"value": "x", "actor": "F"}, user=None))
raises("posting a text value to a FILE deliverable", 409,
       lambda: dt.fill_deliverable(t2id, d1_id,
                                   {"value": "x", "actor": "F"}, user=None))
raises("a link value that is not an http(s) URL", 400,
       lambda: dt.fill_deliverable(t2id, d0_id,
                                   {"value": "not a url", "actor": "F"}, user=None))
raises("an empty value", 400,
       lambda: dt.fill_deliverable(t2id, d0_id,
                                   {"value": "   ", "actor": "F"}, user=None))
r = dt.fill_deliverable(t2id, d0_id,
                        {"value": "https://github.com/x/y/pull/9 ", "actor": "F"},
                        user=None)
filled = r["checklist"][0]["deliverables"][0]
ok("fill records value + filled_by + filled_at",
   filled["value"] == "https://github.com/x/y/pull/9"
   and filled["filled_by"] == "F" and bool(filled["filled_at"]), filled)
ok("fill posts a system comment naming the label and the step",
   'deliverable "link to the PR (merged)" filled (by F) on step 1'
   in bodies_for(t2id)[-1], bodies_for(t2id)[-1])
ok("...tagged to that step's index",
   comments_for(t2id)[-1]["step_order"] == 0)

# Fill on an OPEN, NON-CURRENT step is allowed (evidence arrives early).
cl = copy.deepcopy(task_row(t2id)["checklist"])
cl[2]["deliverables"] = [make_deliv(kind="text", label="closing note")]
dt.update_task(t2id, {"checklist": cl}, user=None)
late_id = task_row(t2id)["checklist"][2]["deliverables"][0]["id"]
dt.fill_deliverable(t2id, late_id, {"value": "shipped in wave 33", "actor": "M"},
                    user=None)
ok("a deliverable on a LATER open step can be filled early",
   deliv(t2id, 2)["value"] == "shipped in wave 33")

# Clear.
r = dt.clear_deliverable(t2id, late_id, actor="M", user=None)
ok("clear nulls every fill field",
   all(r["checklist"][2]["deliverables"][0][f] is None
       for f in dt._DELIVERABLE_FILL_FIELDS))
ok("clear names what was dropped on the thread",
   "cleared (by M)" in bodies_for(t2id)[-1]
   and "shipped in wave 33" in bodies_for(t2id)[-1], bodies_for(t2id)[-1])
n_before = len(comments_for(t2id))
dt.clear_deliverable(t2id, late_id, actor="M", user=None)
ok("clearing an already-empty deliverable is a silent no-op (idempotent)",
   len(comments_for(t2id)) == n_before)


# ══ walk 5: T10 — required blocks, optional is recorded ═════════════════════
print("― walk 5: T10 — required BLOCKS, optional is RECORDED (never blocked) ―")
# Step 1 has: d0 (required link, now FILLED) + d1 (optional file, unfilled).
# Add a second REQUIRED one and prove completion refuses.
cl = copy.deepcopy(task_row(t2id)["checklist"])
cl[0]["deliverables"].append(
    make_deliv(kind="text", label="the measured before/after", required=True))
dt.update_task(t2id, {"checklist": cl}, user=None)
req_id = task_row(t2id)["checklist"][0]["deliverables"][2]["id"]
e = raises("complete with a required-unfilled deliverable", 409,
           lambda: dt.complete_step(t2id, {"actor": "F"}, user=None))
ok("...the 409 names the LABEL",
   "the measured before/after" in e.detail, e.detail)
ok("...and names the escalation path in line (D3)",
   "Raise issue" in e.detail, e.detail)
ok("...and the step is genuinely NOT ticked",
   task_row(t2id)["checklist"][0]["done"] is False)

raises("a text value over 4000 chars", 400,
       lambda: dt.fill_deliverable(t2id, req_id, {"value": "x" * 4001,
                                                  "actor": "F"}, user=None))
dt.fill_deliverable(t2id, req_id, {"value": "22 → 31 paired", "actor": "F"},
                    user=None)
r = dt.complete_step(t2id, {"actor": "F"}, user=None)
ok("with every REQUIRED one filled, completion succeeds",
   r["checklist"][0]["done"] is True)
ok("...even though an OPTIONAL one is unfilled (D2: never blocked)",
   r["checklist"][0]["deliverables"][1].get("value") is None)
ok("...and the system comment NAMES the skipped optional one",
   '1 optional deliverable left unfilled: "a screenshot"' in bodies_for(t2id)[-1],
   bodies_for(t2id)[-1])
ok("...and the hand-off still happened", r["assignee"] == "kevin", r["assignee"])


# ══ walk 6: T4' — a completed step's deliverables are frozen ═════════════════
print("― walk 6: T4' — a completed step's deliverables are frozen ―")
raises("filling a deliverable on a COMPLETED step", 409,
       lambda: dt.fill_deliverable(t2id, task_row(t2id)["checklist"][0]
                                   ["deliverables"][1]["id"],
                                   {"value": "x", "actor": "F"}, user=None))
raises("clearing a deliverable on a COMPLETED step", 409,
       lambda: dt.clear_deliverable(t2id, d0_id, actor="F", user=None))
cl = copy.deepcopy(task_row(t2id)["checklist"])
cl[0]["deliverables"][0]["label"] = "retconned"
raises("editing a COMPLETED step's deliverable spec via PATCH", 409,
       lambda: dt.update_task(t2id, {"checklist": cl}, user=None))
cl = copy.deepcopy(task_row(t2id)["checklist"])
cl[0]["deliverables"] = cl[0]["deliverables"][:1]
raises("deleting a COMPLETED step's deliverable via PATCH", 409,
       lambda: dt.update_task(t2id, {"checklist": cl}, user=None))


# ══ walk 7: dropping a FILLED deliverable on an OPEN step ═══════════════════
print("― walk 7: dropping a FILLED deliverable on an OPEN step is allowed ―")
t3 = dt.create_task({"title": "drop test", "assignee": "F", "checklist": [
    {"owner": "F", "title": "Build", "done": False,
     "deliverables": [make_deliv(kind="text", label="the note")]}]}, user=None)
t3id = t3["id"]
n_id = t3["checklist"][0]["deliverables"][0]["id"]
dt.fill_deliverable(t3id, n_id, {"value": "answered", "actor": "F"}, user=None)
cl = copy.deepcopy(task_row(t3id)["checklist"])
cl[0]["deliverables"] = []
r = dt.update_task(t3id, {"checklist": cl}, user=None)
ok("the drop is allowed (the author may have asked for the wrong thing)",
   r["checklist"][0]["deliverables"] == [])
ok("...but the thread records WHAT was dropped, so the trail survives",
   "was removed from step" in bodies_for(t3id)[-1]
   and "answered" in bodies_for(t3id)[-1], bodies_for(t3id)[-1])


# ══ walk 8: §3.2/§3.4 — upload + signed URL ═════════════════════════════════
print("― walk 8: §3.2/§3.4 — upload · limits · the click-time signed URL ―")
t4 = dt.create_task({"title": "upload test", "assignee": "F", "checklist": [
    {"owner": "F", "title": "Build", "done": False,
     "deliverables": [make_deliv(kind="file", label="the chart")]}]}, user=None)
t4id = t4["id"]
f_id = t4["checklist"][0]["deliverables"][0]["id"]
b64 = base64.b64encode(b"\x89PNG fake bytes").decode()
raises("uploading a disallowed extension", 415,
       lambda: dt.upload_deliverable(t4id, f_id, {
           "filename": "payload.exe", "content_type": "application/x-msdownload",
           "b64": b64, "actor": "F"}, user=None))
raises("uploading over the size limit", 413,
       lambda: dt.upload_deliverable(t4id, f_id, {
           "filename": "huge.png", "content_type": "image/png",
           "b64": base64.b64encode(b"x" * (10 * 1024 * 1024 + 1)).decode(),
           "actor": "F"}, user=None))
raises("uploading bytes that are not base64", 400,
       lambda: dt.upload_deliverable(t4id, f_id, {
           "filename": "a.png", "content_type": "image/png",
           "b64": "!!!not base64!!!", "actor": "F"}, user=None))
raises("uploading a file to a LINK deliverable", 409,
       lambda: dt.upload_deliverable(t2id, late_id, {
           "filename": "a.png", "content_type": "image/png", "b64": b64,
           "actor": "M"}, user=None))
ok("no object was stored by any refused upload",
   not FAKE.store.get("_objects"), FAKE.store.get("_objects"))

r = dt.upload_deliverable(t4id, f_id, {
    "filename": "health chart.png", "content_type": "image/png",
    "b64": b64, "actor": "F"}, user=None)
stored = r["checklist"][0]["deliverables"][0]
ok("upload records the OBJECT PATH, not a URL",
   stored["value"].startswith(f"task-{t4id}/step-")
   and "://" not in stored["value"], stored["value"])
ok("...the filename is preserved verbatim for display",
   stored["filename"] == "health chart.png")
ok("...the path itself is sanitized (no spaces)", " " not in stored["value"])
ok("...size + content_type + provenance are recorded",
   stored["size_bytes"] == len(b"\x89PNG fake bytes")
   and stored["content_type"] == "image/png" and stored["filled_by"] == "F")
ok("...and the bytes actually reached the bucket",
   FAKE.store["_objects"][stored["value"]][0] == b"\x89PNG fake bytes")

u = dt.deliverable_url(t4id, f_id, user=None)
ok("the signed URL is minted on demand and points at the object",
   u["url"].startswith("https://sig.example/task-deliverables/")
   and "ttl=3600" in u["url"], u)
raises("asking for a URL on an UNFILLED file deliverable", 404,
       lambda: dt.deliverable_url(t2id, d1_id, user=None))

# A storage failure must be LOUD, and must not half-write the checklist.
FAKE.store["_fail"] = True
t5 = dt.create_task({"title": "no bucket", "assignee": "F", "checklist": [
    {"owner": "F", "title": "Build", "done": False,
     "deliverables": [make_deliv(kind="file", label="the chart")]}]}, user=None)
t5id, f5 = t5["id"], t5["checklist"][0]["deliverables"][0]["id"]
e = raises("an upload when the bucket does not exist", 502,
           lambda: dt.upload_deliverable(t5id, f5, {
               "filename": "a.png", "content_type": "image/png", "b64": b64,
               "actor": "F"}, user=None))
ok("...and the 502 names the missing bucket, not a generic failure",
   "task-deliverables" in e.detail, e.detail)
ok("...and NOTHING was written to the checklist",
   task_row(t5id)["checklist"][0]["deliverables"][0].get("value") is None)
FAKE.store["_fail"] = False


# ══ walk 9: §3.6 — the stamp lands IN the step it was made on ═══════════════
print("― walk 9: §3.6 — a chain stamp writes step.stamp + provenance + hand-off ―")
t6 = dt.create_task({"title": "stamp chain", "assignee": "kevin",
                     "status": "Approval", "checklist": [
                         {"owner": "kevin", "title": "Review the spec — STAMP",
                          "mode": "discuss", "done": False},
                         {"owner": "F", "title": "Build", "done": False}]},
                    user=None)
t6id = t6["id"]
r = dt.stamp_approval(t6id, {"mode": "start", "author": "kevin"}, user=None)
st = r["checklist"][0]
ok("the stamp is recorded ON THE STEP (state/by/at) — #184's core ask",
   (st.get("stamp") or {}).get("state") == "approved"
   and st["stamp"]["by"] == "kevin" and bool(st["stamp"]["at"]), st.get("stamp"))
ok("...on a step that declared no stamp gate, required stays False",
   st["stamp"]["required"] is False)
ok("...the tick carries completion provenance (the #244 half)",
   st["done"] is True and st["completed_by"] == "kevin"
   and bool(st["completed_at"]), st)
ok("...the hand-off fired to the next step's owner",
   r["assignee"] == "F", r["assignee"])
ok("...status/arming still flipped", r["status"] == "Todo"
   and r["ai_eligible"] is True)
ok("...and ONE stamp comment names the stamp, the tick and the hand-off",
   'stamp: approved to start (by kevin)' in bodies_for(t6id)[-1]
   and 'ticked "Review the spec — STAMP"' in bodies_for(t6id)[-1]
   and "handed off: kevin → F" in bodies_for(t6id)[-1], bodies_for(t6id)[-1])

# A gated step keeps required:True and settles it (the #244 behaviour, kept).
t7 = dt.create_task({"title": "gated stamp", "assignee": "kevin",
                     "status": "Approval", "checklist": [
                         {"owner": "kevin", "title": "Approve — STAMP",
                          "stamp": {"required": True, "state": "pending"},
                          "done": False},
                         {"owner": "M", "title": "Close", "done": False}]},
                    user=None)
r = dt.stamp_approval(t7["id"], {"mode": "start", "author": "kevin"}, user=None)
ok("a GATED kevin step keeps required:True and settles to approved",
   r["checklist"][0]["stamp"] == {"required": True, "state": "approved",
                                  "by": "kevin", "at": r["checklist"][0]["stamp"]["at"]},
   r["checklist"][0]["stamp"])

# T10 applies to Kevin's step too, through the shared path.
t8 = dt.create_task({"title": "stamp + deliverable", "assignee": "kevin",
                     "status": "Approval", "checklist": [
                         {"owner": "kevin", "title": "Review — STAMP",
                          "done": False,
                          "deliverables": [make_deliv(kind="text",
                                                      label="the ruling",
                                                      required=True)]},
                         {"owner": "F", "title": "Build", "done": False}]},
                    user=None)
t8id = t8["id"]
e = raises("a stamp on a step with a required-unfilled deliverable", 409,
           lambda: dt.stamp_approval(t8id, {"mode": "start", "author": "kevin"},
                                     user=None))
ok("...T10 fires through the SHARED path, naming the label",
   "the ruling" in e.detail, e.detail)
ok("...and the stamp changed NOTHING (status still Approval)",
   task_row(t8id)["status"] == "Approval"
   and task_row(t8id)["checklist"][0]["done"] is False)

# LEGACY checklists: byte-identical to pre-#184 (no step.stamp invented).
t9 = dt.create_task({"title": "legacy stamp", "assignee": "kevin",
                     "status": "Approval", "checklist": [
                         {"role": "kevin", "text": "Kevin approves", "done": False},
                         {"role": "F", "text": "build", "done": False}]},
                    user=None)
r = dt.stamp_approval(t9["id"], {"mode": "start", "author": "kevin"}, user=None)
ok("a LEGACY checklist stamp ticks exactly as before (no stamp block invented)",
   r["checklist"][0]["done"] is True and "stamp" not in r["checklist"][0],
   r["checklist"][0])
ok("...with the same provenance #244 added, and the same hand-off",
   r["checklist"][0]["completed_by"] == "kevin" and r["assignee"] == "F")

# A kevin step BEHIND an open step: fall back to the direct tick, never tick
# the wrong (current) step.
t10 = dt.create_task({"title": "kevin not current", "assignee": "kevin",
                      "status": "Approval", "checklist": [
                          {"owner": "F", "title": "Build first", "done": False},
                          {"owner": "kevin", "title": "Then approve",
                           "done": False}]}, user=None)
r = dt.stamp_approval(t10["id"], {"mode": "start", "author": "kevin"}, user=None)
ok("a kevin step that is NOT current is ticked directly — F's step is untouched",
   r["checklist"][0]["done"] is False and r["checklist"][1]["done"] is True,
   [s["done"] for s in r["checklist"]])
ok("...and no stamp block is invented on it (pre-#184 fall-back, byte-identical)",
   "stamp" not in r["checklist"][1], r["checklist"][1])
ok("...and the task hands back to the still-open F step",
   r["assignee"] == "F", r["assignee"])


print(f"\n✅ ALL {PASS} CHECKS PASSED — board #184 step deliverables")
