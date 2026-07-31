"""Acceptance test — board #257: Kevin's (owner) dashboard.

WHAT THIS PINS
--------------
The page exists to answer ONE question — *"what is waiting on ME, and can I
stop worrying about the rest?"* — and the way it fails is by answering a
different one. Two shapes in particular:

  • A COUNT THAT LIES. Shipped work still sits in `Staged` (#249), so a
    "needs your look" module built on `status` would have shown Kevin six
    already-merged tasks as things awaiting his eyes. That is the exact bug he
    found by looking at a count that lied, and rail 2 is a replay of it.
  • A SECOND MANAGER DASHBOARD. Everything he cannot act on is noise, and noise
    hands the triage back to the person the page was built to spare.

RAILS, each with a check that FAILS without it
----------------------------------------------
A · YOUR PLATE
  1. the four buckets are evaluated IN ORDER and a task lands in EXACTLY ONE —
     a task matching both `stamp` and `close` is a STAMP (work that cannot
     start outranks work that is finished).
  2. #249 — "needs your LOOK" reads the chain's SHIP STEP, never `status`.
     Staged + ship step ticked + parked on Kevin = LOOK; Staged + ship step
     OPEN = mid-chain, not look. And six merged-but-Staged tasks with no Kevin
     step are not on his plate AT ALL.
  3. a chain with NO release-lane step is UNREADABLE, not unshipped — it falls
     through to mid-chain rather than being guessed at in either direction.
  4. only the FIRST OPEN step counts: a pending stamp further down the chain is
     not his move yet, and an already-approved stamp is not his move any more.
  5. `Closed` never appears — it is the state his own action produces.
  6. AGE is measured from the step that HANDED IT TO HIM, not from task age,
     and the row NAMES which stamp answered.
  7. every row carries what it unblocks: the next step's owner, and a count of
     open tasks naming it in `blocked_by` (#246 rendered as a number).
  8. rows are BOUNDED (#240) while `count` stays the FULL number.

C · STUCK
  9. `Stand By` is NOT stuck — board #232 created it so Kevin could park
     something deliberately, and a page that calls it stalled has un-invented
     the status.
 10. anything already on his plate is EXCLUDED — his queue is not a stall.
 11. "a car waiting for a train" is IMPORTED from rSessionSignals (#251), not
     restated; feeding its output produces the stall.
 12. a run past its 45m lease is flagged; one inside its lease is not. The
     lease rule is `isLeaseLive` (#193), imported.
 13. loop `down` is a stall; loop `unknown` is NOT — an absence of evidence
     must not cry wolf on every quiet evening (#157).
 14. a `never` heartbeat is reported and WORDED DIFFERENTLY from a `down` one:
     "never started" and "stopped" want different reactions.
 15. the panel is bounded, and it reports WHAT IT CHECKED so that empty reads
     as reassurance rather than as absence.

B · CAN ANYTHING EVEN RUN
 16. `loopState` has THREE answers. A recent claim is `up`; a run request left
     unclaimed past the grace is `down`; a quiet board is `unknown` and never
     `up`.

E · ACTIVITY
 17. the feed EXCLUDES what M named — status ticks, assignee changes, dispatch
     claims, step ticks, heartbeat/poll chatter — and INCLUDES stamps, Done /
     Closed, failed runs, trains, deploys and armed flags.
 18. events matching no rule are COUNTED, never silently discarded.

THE PAGE ITSELF
 19. every module has calm, affirmative empty-state copy; the page never says
     "No results" and never renders a bare zero as its whole answer.
 20. the modal is the EXISTING `TaskDetailModal`, called with the same props
     /admin/tasks passes, wrapped in `AgentRegistryContext.Provider`.
 21. /admin/tasks still opens the modal EXACTLY as before, and TaskDetailModal
     is unchanged on disk relative to `origin/dev`.
 22. NO FILTER BAR — the page is pre-filtered by construction.
 23. wiring: the route, the nav entry under `Team`, and SignOfLife reused.
 24. no migration is introduced by this task.

Run:  .venv/bin/python src/test_kevin_dashboard_257.py
      (from the RoR_Trader root)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
FE = os.path.join(ROOT, "frontend", "src")

FAILED = []
PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}" + (f" — {detail}" if detail else ""))
        FAILED.append(label)


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


KD = read(FE, "views", "kevinDashboard.ts")
KPAGE = read(FE, "views", "AdminKevinPage.tsx")
DPAGE = read(FE, "views", "AdminDispatchPage.tsx")
SHARED = read(FE, "views", "taskBoardShared.tsx")
TASKS = read(FE, "views", "AdminTasksPage.tsx")
SIDEBAR = read(FE, "components", "Sidebar.tsx")


# ── Lifting the pure TS out so it can actually be RUN ───────────────────────
# Same technique as test_r_session_dashboard_251.py / _245.py: only the
# SIGNATURE is de-typed and a handful of TS-only annotations are stripped from
# the body; everything else is copied VERBATIM, so what runs here is what ships.
# A static substring assertion would pass against a module that returns the
# wrong rows, and the bucket rules are exactly the part that has to be right.

def _split_params(params):
    out, depth, buf = [], 0, ""
    for ch in params:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf)
    return [p.split(":")[0].strip() for p in out if p.strip()]


def _detype(body):
    body = re.sub(r"\b(const|let) (\w+): [^=\n]+ = ", r"\1 \2 = ", body)
    body = re.sub(r"new (Map|Set)<[^>]*>\(", r"new \1(", body)
    body = re.sub(r"\((\w+): [^)=]+\) =>", r"(\1) =>", body)
    body = re.sub(r"\.forEach\(\((\w+): [^)]+\)", r".forEach((\1)", body)
    return body


def js_body(src, name):
    m = re.search(rf"^export function {name}\(", src, re.M)
    assert m, f"{name} not found — the derivation is missing"
    line_end = src.index("\n", m.start())
    sig = src[m.start():line_end]
    params = sig[sig.index("(") + 1:sig.rindex(")")]
    i, depth = line_end, 1
    while depth:
        i += 1
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
    return (f"function {name}({', '.join(_split_params(params))}) {{\n"
            + _detype(src[line_end + 1:i + 1]))


def js_const(src, name):
    """Lift `export const NAME = <expr>;` — scanning to the terminating `;` with
    bracket/quote awareness rather than a lazy `[^;]+`, because several of these
    values are objects whose PROSE contains semicolons."""
    m = re.search(rf"^export const {name} = ", src, re.M)
    assert m, f"{name} not found"
    i = m.end()
    depth, quote, esc = 0, None, False
    while i < len(src):
        ch = src[i]
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth == 0:
            break
        i += 1
    return f"const {name} = {_detype(src[m.end():i])};\n"


NODE = shutil.which("node")
ok("node is available to execute the lifted derivations", bool(NODE))

STALE_GRACE = re.search(r"^const STALE_GRACE_S = (\d+);", DPAGE, re.M)
assert STALE_GRACE, "STALE_GRACE_S not found on the dispatch page"

NOW_ISO = "2026-07-31T04:00:00Z"
PRELUDE = (
    "const NOW = Date.parse('%s');\nDate.now = () => NOW;\n" % NOW_ISO
    + js_const(SHARED, "FINISHED_STATUSES")
    + "const isFinished = (s) => !!s && FINISHED_STATUSES.includes(s);\n"
    + "const stepOwner = (s) => (s.owner !== undefined && s.owner !== null) ? s.owner : "
      "(s.role !== undefined && s.role !== null ? s.role : null);\n"
    + "const stepTitle = (s) => s.title || s.text || '';\n"
    + js_const(SHARED, "DISPATCHER")
    + f"const STALE_GRACE_S = {STALE_GRACE.group(1)};\n"
    + js_const(DPAGE, "SHIP_STEP_OWNERS")
    + js_body(SHARED, "hoursSince") + "\n"
    + js_body(DPAGE, "isLeaseLive") + "\n"
    + js_const(KD, "KEVIN")
    + js_const(KD, "PLATE_BUCKETS")
    + js_const(KD, "CALM_STATUSES")
    + js_const(KD, "STARVED_H")
    + js_const(KD, "FAILED_WINDOW_H")
    + js_const(KD, "FAILED_OUTCOMES")
    + js_const(KD, "LOOP_CLAIM_GRACE_S")
    + js_const(KD, "LOOP_UP_WINDOW_S")
    + js_const(KD, "KEVIN_FEED_DROP")
    + js_const(KD, "KEVIN_FEED_KEEP")
    + js_body(KD, "firstOpenStep") + "\n"
    + js_body(KD, "parkedOnKevin") + "\n"
    + js_body(KD, "pendingStamp") + "\n"
    + js_body(KD, "shipStepState") + "\n"
    + js_body(KD, "plateBucketOf") + "\n"
    + js_body(KD, "landedOnKevin") + "\n"
    + js_body(KD, "unblocksWho") + "\n"
    + js_body(KD, "buildPlate") + "\n"
    + js_body(KD, "loopState") + "\n"
    + js_body(KD, "buildStalls") + "\n"
    + js_body(KD, "decisionsAndOutcomes") + "\n"
)


def run_js(tail, payload):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(PRELUDE + "const a = JSON.parse(process.argv[2]);\n" + tail)
        path = fh.name
    try:
        out = subprocess.run([NODE, path, json.dumps(payload)],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:1200]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


def plate(tasks, limit=12):
    return run_js(
        "const p = buildPlate(a.tasks, a.limit);\n"
        "console.log(JSON.stringify({\n"
        "  buckets: Object.fromEntries(p.buckets.map(b => [b.key, b.rows.map(r => r.task.id)])),\n"
        "  counts: Object.fromEntries(p.buckets.map(b => [b.key, b.count])),\n"
        "  total: p.total, oldestH: p.oldestH,\n"
        "  rows: Object.fromEntries(p.buckets.flatMap(b => b.rows.map(r => [r.task.id, {\n"
        "    ageH: r.ageH, ageFrom: r.ageFrom, unblocks: r.unblocks,\n"
        "    blocking: r.blocking, bucket: r.bucket, stepTitle: r.stepTitle }]))),\n"
        "  of: Object.fromEntries(a.tasks.map(t => [t.id, plateBucketOf(t)])),\n"
        "  ship: Object.fromEntries(a.tasks.map(t => [t.id, shipStepState(t)])),\n"
        "}));\n", {"tasks": tasks, "limit": limit})


def step(title, owner, done, stamp=None, completed_at=None):
    s = {"title": title, "owner": owner, "done": done}
    if stamp is not None:
        s["stamp"] = stamp
    if completed_at is not None:
        s["completed_at"] = completed_at
    return s


def task(tid, title, status, checklist, assignee=None, blocked_by=None,
         updated_at="2026-07-31T00:00:00Z"):
    return {"id": tid, "title": title, "status": status, "checklist": checklist,
            "assignee": assignee, "blocked_by": blocked_by or [],
            "description": "", "updated_at": updated_at}


PENDING = {"required": True, "state": "pending"}
APPROVED = {"required": True, "state": "approved"}

print("\n── RAIL 1 · four buckets, evaluation order, exactly one ────────────")
# A task that satisfies BOTH "pending Kevin stamp" and "Done + assigned kevin".
# Without an explicit order it lands in two buckets and the page double-counts.
BOTH = task(1, "both shapes at once", "Done", [
    step("Approve the plan", "kevin", False, PENDING),
], assignee="kevin")
r = plate([BOTH])
ok("a task matching two buckets resolves to STAMP (work that cannot start outranks finished work)",
   r["of"]["1"] == "stamp", r["of"])
ok("...and appears in exactly one bucket's rows",
   sum(len(v) for v in r["buckets"].values()) == 1, r["buckets"])
ok("the plate total is 1, not 2", r["total"] == 1, r["total"])

CLOSE = task(2, "M marked it Done", "Done", [
    step("Build", "F", True), step("Ship", "R-A", True),
], assignee="kevin")
r = plate([CLOSE])
ok("Done + assigned kevin is the CLOSE bucket", r["of"]["2"] == "close", r["of"])
ok("a Done task NOT assigned to kevin is not on his plate",
   plate([dict(CLOSE, assignee="M")])["total"] == 0)


print("\n── RAIL 2 · #249 — LOOK reads the SHIP STEP, never `status` ────────")
# The chain that would break a status-based rule: `Staged` (which is where
# SHIPPED work still sits on this board) with the release-lane step TICKED.
SHIPPED_PARKED = task(10, "shipped, now parked on Kevin", "Staged", [
    step("Build", "F", True, completed_at="2026-07-30T22:00:00Z"),
    step("Ship via a release train", "R-A", True, completed_at="2026-07-31T01:00:00Z"),
    step("Use it for a day, then close", "kevin", False),
])
UNSHIPPED_PARKED = task(11, "not shipped, parked on Kevin", "Staged", [
    step("Build", "F", True, completed_at="2026-07-30T22:00:00Z"),
    step("Approve the direction", "kevin", False),
    step("Ship via a release train", "R-A", False),
])
r = plate([SHIPPED_PARKED, UNSHIPPED_PARKED])
ok("ship step TICKED + parked on Kevin = needs your LOOK",
   r["of"]["10"] == "look", r["of"])
ok("ship step OPEN = mid-chain, NOT look (the #249 lie)",
   r["of"]["11"] == "midchain", r["of"])
ok("both are `Staged`, so a status-based rule could not have told them apart",
   SHIPPED_PARKED["status"] == UNSHIPPED_PARKED["status"] == "Staged")
ok("shipStepState reports done/open, not a status echo",
   r["ship"]["10"] == "done" and r["ship"]["11"] == "open", r["ship"])

# The measured #249 shape: merged work still reading `Staged`, with NO Kevin
# step at all. A status-based module would have painted all six at Kevin.
SIX = [task(100 + i, f"merged but still Staged #{i}", "Staged", [
    step("Build", "F", True),
    step("Ship via a release train", "R-A", True),
]) for i in range(6)]
r = plate(SIX)
ok("six merged-but-Staged tasks with no Kevin step are NOT on his plate at all",
   r["total"] == 0, r["counts"])


print("\n── RAIL 3 · no release-lane step = UNREADABLE, not unshipped ───────")
NO_SHIP = task(20, "legacy chain, no R step", "In Progress", [
    step("Build", "F", True), step("Decide", "kevin", False),
])
r = plate([NO_SHIP])
ok("a chain with no R/R-A step reports ship state `none`", r["ship"]["20"] == "none")
ok("...and falls through to mid-chain rather than being guessed as shipped",
   r["of"]["20"] == "midchain", r["of"])


print("\n── RAIL 4 · only the FIRST OPEN step counts ────────────────────────")
LATER_STAMP = task(30, "his stamp is two steps away", "In Progress", [
    step("Build", "F", False),
    step("Approve", "kevin", False, PENDING),
])
ok("a pending Kevin stamp BELOW an open F step is not his move yet",
   plate([LATER_STAMP])["of"]["30"] is None)
APPROVED_STEP = task(31, "already stamped", "In Progress", [
    step("Approve", "kevin", False, APPROVED),
    step("Build", "F", False),
])
r = plate([APPROVED_STEP])
ok("an ALREADY-APPROVED stamp is no longer a STAMP item",
   r["of"]["31"] != "stamp", r["of"])
ok("...it is still parked on him, so it stays visible as mid-chain",
   r["of"]["31"] == "midchain", r["of"])
NOT_REQUIRED = task(32, "kevin step, no stamp gate", "In Progress", [
    step("Look at it", "kevin", False, {"required": False, "state": "pending"}),
])
ok("a Kevin step whose stamp is not REQUIRED is not a stamp item",
   plate([NOT_REQUIRED])["of"]["32"] == "midchain")


print("\n── RAIL 5 · `Closed` never comes back ──────────────────────────────")
ok("a Closed task assigned kevin is off the plate — the page must not ask twice",
   plate([task(40, "already closed", "Closed", [
       step("Build", "F", True)], assignee="kevin")])["total"] == 0)
ok("a Closed task still parked on a Kevin step is off the plate too",
   plate([task(41, "closed mid-chain", "Closed", [
       step("Decide", "kevin", False)], assignee="kevin")])["total"] == 0)


print("\n── RAIL 5b · deliberately parked is not waiting ────────────────────")
# MEASURED against the live board on 07-31: five of twenty plate items were
# `Stand By`, FOUR of them under "Needs your STAMP" — i.e. the page telling
# Kevin that work cannot start on things he himself chose to park. That is a
# count he cannot act on, in the module that has to be perfect.
PARKED_STAMP = task(45, "he parked this himself", "Stand By", [
    step("Approve", "kevin", False, PENDING)], assignee="kevin")
ok("a `Stand By` task with a pending Kevin stamp is NOT on his plate",
   plate([PARKED_STAMP])["total"] == 0, plate([PARKED_STAMP])["of"])
ok("...nor is a `Stand By` task parked mid-chain on him",
   plate([task(46, "parked mid-chain", "Stand By", [
       step("Decide", "kevin", False)])])["total"] == 0)
ok("`Backlog` is quiet on the plate too",
   plate([task(47, "captured, not next", "Backlog", [
       step("Approve", "kevin", False, PENDING)])])["total"] == 0)
ok("the SAME task in an active status IS on his plate (the rail bites)",
   plate([dict(PARKED_STAMP, status="Approval")])["of"]["45"] == "stamp")
ok("the plate and the stall panel share ONE definition of parked",
   KD.count("export const CALM_STATUSES") == 1)


print("\n── RAIL 6 · AGE is since it landed on HIM ──────────────────────────")
# Task edited long ago, handed to him an hour ago. Task age would say 30h.
HANDED = task(50, "handed over an hour ago", "In Progress", [
    step("Build", "F", True, completed_at="2026-07-29T22:00:00Z"),
    step("M review", "M", True, completed_at="2026-07-31T03:00:00Z"),
    step("Your call", "kevin", False),
], updated_at="2026-07-30T00:00:00Z")
r = plate([HANDED])
row = r["rows"]["50"]
ok("age is ~1h (from the step that handed it over), NOT 28h+ (task age)",
   0.9 < row["ageH"] < 1.1, row)
ok("the row NAMES which stamp answered — an unlabelled proxy is how a dashboard lies",
   "step" in row["ageFrom"], row["ageFrom"])
NO_STAMPS = task(51, "chainless-ish", "In Progress", [
    step("Your call", "kevin", False),
], updated_at="2026-07-31T02:00:00Z")
r = plate([NO_STAMPS])
ok("with no completed step it falls back to updated_at and SAYS so",
   1.9 < r["rows"]["51"]["ageH"] < 2.1
   and "board edit" in r["rows"]["51"]["ageFrom"], r["rows"]["51"])


print("\n── RAIL 7 · what it unblocks — #246 rendered as a number ───────────")
BLOCKER = task(60, "he is the holdup", "In Progress", [
    step("Build", "F", True, completed_at="2026-07-31T03:00:00Z"),
    step("Approve", "kevin", False, PENDING),
    step("Ship via a release train", "R-A", False),
])
DEPENDENT_A = task(61, "cannot start", "Todo", [step("Build", "F", False)], blocked_by=[60])
DEPENDENT_B = task(62, "cannot start either", "Todo", [step("Build", "E", False)], blocked_by=[60])
DONE_DEP = task(63, "already finished", "Done", [step("Build", "F", True)], blocked_by=[60])
r = plate([BLOCKER, DEPENDENT_A, DEPENDENT_B, DONE_DEP])
ok("the row names who moves next once he acts", r["rows"]["60"]["unblocks"] == "R-A",
   r["rows"]["60"])
ok("...and counts the OPEN tasks that cannot start until he does",
   r["rows"]["60"]["blocking"] == 2, r["rows"]["60"])
ok("a FINISHED dependent is not counted — it is not waiting on anything",
   r["rows"]["60"]["blocking"] == 2)


print("\n── RAIL 8 · bounded rows, full count (#240) ────────────────────────")
MANY = [task(200 + i, f"stamp me {i}", "In Progress", [
    step("Approve", "kevin", False, PENDING)],
    updated_at=f"2026-07-30T{i % 24:02d}:00:00Z") for i in range(30)]
r = plate(MANY, limit=12)
ok("only `limit` rows are painted", len(r["buckets"]["stamp"]) == 12,
   len(r["buckets"]["stamp"]))
ok("...while the COUNT is the full 30 — nothing is dropped, only unpainted",
   r["counts"]["stamp"] == 30, r["counts"])
ok("the plate total reflects the full count too", r["total"] == 30, r["total"])
ok("rows are OLDEST FIRST — the top of the list is the longest-standing holdup",
   r["rows"][str(r["buckets"]["stamp"][0])]["ageH"]
   >= r["rows"][str(r["buckets"]["stamp"][1])]["ageH"])


# ── C · STUCK ───────────────────────────────────────────────────────────────

def stalls(tasks=None, runs=None, cars=None, plate_ids=None, dispatchable=None,
           lanes_free=4, loop=None, sessions=None, limit=20):
    payload = {
        "input": {
            "cars": cars, "runs": runs or [], "tasks": tasks or [],
            "plateIds": plate_ids or [], "dispatchableIds": dispatchable or [],
            "lanesFree": lanes_free,
            "loop": loop or {"state": "up", "why": "test", "lastClaimAt": None},
            "sessions": sessions or [],
            "nowMs": 0,
        },
        "limit": limit,
    }
    return run_js(
        "const s = buildStalls(a.input, a.limit);\n"
        "console.log(JSON.stringify({kinds: s.rows.map(r => r.kind),\n"
        "  ids: s.rows.map(r => r.taskId), count: s.count,\n"
        "  details: s.rows.map(r => r.detail), checked: s.checked,\n"
        "  titles: s.rows.map(r => r.title)}));\n", payload)


def car_panel(rows):
    return {"rows": rows, "count": len(rows), "covered": [], "openTrainIds": [],
            "oldestH": max([r["ageH"] for r in rows] or [0]), "warning": None,
            "trainOwed": len(rows) > 0}


print("\n── RAIL 9 · `Stand By` is NOT stuck ────────────────────────────────")
PARKED = task(70, "Kevin parked this on purpose", "Stand By", [
    step("Build", "F", False)])
CAR_PARKED = car_panel([{"task": PARKED, "branch": "feat/x", "ageFrom": "branch",
                         "ageH": 40, "mergeWhy": "", "mergeUnknown": False}])
r = stalls(tasks=[PARKED], cars=CAR_PARKED)
ok("a `Stand By` task is never reported as stalled (board #232 invented it for this)",
   r["count"] == 0, r)
r = stalls(tasks=[PARKED], dispatchable=[70], lanes_free=2)
ok("...not even when it is otherwise dispatchable and idle", r["count"] == 0, r)
BACKLOG = task(71, "captured, not next", "Backlog", [step("Build", "F", False)])
ok("`Backlog` is quiet too",
   stalls(tasks=[BACKLOG], dispatchable=[71])["count"] == 0)


print("\n── RAIL 10 · his own plate is not a stall ──────────────────────────")
ONPLATE = task(72, "waiting on Kevin, and dispatch-eligible", "Todo", [
    step("Approve", "kevin", False, PENDING)],
    updated_at="2026-07-30T00:00:00Z")
r = stalls(tasks=[ONPLATE], dispatchable=[72], plate_ids=[72])
ok("a task already on his plate is EXCLUDED — his queue is not a stall",
   r["count"] == 0, r)
r = stalls(tasks=[ONPLATE], dispatchable=[72], plate_ids=[])
ok("...and the same task IS a stall when it is not on his plate (the rail bites)",
   r["kinds"] == ["starved"], r)


print("\n── RAIL 11 · cars waiting — R's definition, IMPORTED ───────────────")
CAR = task(246, "Tasks page — multiselect filters", "Staged",
           [step("Ship via a release train", "R-A", False)])
r = stalls(tasks=[CAR], cars=car_panel([
    {"task": CAR, "branch": "feat/tasks-multiselect-filters-246", "ageFrom": "branch",
     "ageH": 4, "mergeWhy": "", "mergeUnknown": False}]))
ok("#246's 07-30 shape arrives as a `car-waiting` stall",
   r["kinds"] == ["car-waiting"] and r["ids"] == [246], r)
ok("...and it says a train is OWED", "train is OWED" in r["details"][0], r["details"])
ok("the module IMPORTS carsWaitingForATrain rather than restating clause 1/2/3",
   "carsWaitingForATrain" in KD and "from './rSessionSignals'" in KD)
ok("...and does not re-implement 'is this Staged with an unmerged branch'",
   "mergeStateOf" not in KD.split("import {")[-1].split("}")[0]
   and KD.count("status !== 'Staged'") == 0)


print("\n── RAIL 12 · a run past its lease, and one inside it ───────────────")
LIVE_RUN = {"id": 1, "task_id": 80, "agent_letter": "F", "outcome": "running",
            "requested_at": "2026-07-31T03:50:00Z", "started_at": "2026-07-31T03:50:00Z",
            "finished_at": None}
DEAD_RUN = dict(LIVE_RUN, id=2, started_at="2026-07-31T02:00:00Z",
                requested_at="2026-07-31T02:00:00Z")
T80 = task(80, "long-running", "In Progress", [step("Build", "F", False)])
ok("a run INSIDE its 45m lease is not a stall",
   stalls(tasks=[T80], runs=[LIVE_RUN])["count"] == 0)
r = stalls(tasks=[T80], runs=[DEAD_RUN])
ok("a run 2h past its start IS a stall (the lease rule is #193's isLeaseLive)",
   r["kinds"] == ["run-overrun"], r)
ok("the page imports isLeaseLive rather than restating the lease arithmetic",
   "isLeaseLive" in KD and "RUN_TIMEOUT_S" not in KD)


print("\n── RAIL 13 · loop `unknown` must not cry wolf ──────────────────────")
DOWN = {"state": "down", "why": "3 run requests unclaimed", "lastClaimAt": None}
UNKNOWN = {"state": "unknown", "why": "nothing claimable", "lastClaimAt": None}
r = stalls(loop=DOWN)
ok("loop DOWN is a stall", r["kinds"] == ["loop-down"], r)
ok("...and carries the loop's own reason, not a restatement",
   r["details"] == ["3 run requests unclaimed"], r["details"])
ok("loop UNKNOWN is NOT a stall — an absence of evidence is not an alarm",
   stalls(loop=UNKNOWN)["count"] == 0)


print("\n── RAIL 14 · `never` and `down` are different words (#157) ─────────")
NEVER = {"letter": "R", "key": "session_heartbeat_R", "tracked": True,
         "state": "never", "at": None, "at_source": None, "age_s": None,
         "actor": None, "note": None, "updated_at": None, "why": ""}
STOPPED = dict(NEVER, state="down", age_s=7200)
FRESH = dict(NEVER, state="fresh", age_s=60)
UNTRACKED = dict(NEVER, tracked=False)
r = stalls(sessions=[NEVER])
ok("a session that never wrote a heartbeat is reported", r["kinds"] == ["session-down"], r)
ok("...worded as 'never'", "never written" in r["details"][0], r["details"])
r2 = stalls(sessions=[STOPPED])
ok("a session that STOPPED is reported with different words",
   "stopped" in r2["details"][0] and r2["details"][0] != r["details"][0], r2["details"])
ok("a fresh session is not a stall", stalls(sessions=[FRESH])["count"] == 0)
ok("an UNTRACKED beat is not reported — only declared sessions are owed a pulse",
   stalls(sessions=[UNTRACKED])["count"] == 0)


print("\n── RAIL 15 · bounded, and it says what it checked ──────────────────")
MANY_RUNS = [dict(DEAD_RUN, id=100 + i, task_id=300 + i) for i in range(30)]
MANY_T = [task(300 + i, f"t{i}", "In Progress", [step("Build", "F", False)]) for i in range(30)]
r = stalls(tasks=MANY_T, runs=MANY_RUNS, limit=20)
ok("rows are bounded to the limit", len(r["kinds"]) == 20, len(r["kinds"]))
ok("...while count is the full 30", r["count"] == 30, r["count"])
r = stalls()
ok("an EMPTY panel still reports what it checked, so zero reads as reassurance",
   len(r["checked"]) >= 3 and any("lane" in c for c in r["checked"]), r["checked"])


# ── B · LOOP ────────────────────────────────────────────────────────────────

def loop_of(runs, starved=0, now="2026-07-31T04:00:00Z"):
    return run_js(
        "console.log(JSON.stringify(loopState(a.runs, a.starved, Date.parse(a.now))));\n",
        {"runs": runs, "starved": starved, "now": now})


print("\n── RAIL 16 · loopState has THREE answers ───────────────────────────")
RECENT = {"id": 1, "task_id": 1, "agent_letter": "F", "outcome": "ok",
          "requested_at": "2026-07-31T03:50:00Z", "started_at": "2026-07-31T03:50:00Z",
          "finished_at": "2026-07-31T03:58:00Z"}
ok("a run claimed 10m ago = UP", loop_of([RECENT])["state"] == "up", loop_of([RECENT]))
STALE_REQ = {"id": 2, "task_id": 2, "agent_letter": "F", "outcome": "requested",
             "requested_at": "2026-07-31T03:00:00Z", "started_at": None,
             "finished_at": None}
r = loop_of([STALE_REQ])
ok("a run request unclaimed for an hour = DOWN", r["state"] == "down", r)
ok("...and the reason names the poll cadence", "20s" in r["why"], r["why"])
r = loop_of([])
ok("a quiet board is UNKNOWN, never UP", r["state"] == "unknown", r)
ok("...and says so in words — 'Not a green light'", "green light" in r["why"], r["why"])
r = loop_of([], starved=3)
ok("starvation with a free lane flips it to DOWN", r["state"] == "down", r)
OLD = dict(RECENT, started_at="2026-07-30T04:00:00Z", requested_at="2026-07-30T04:00:00Z",
           finished_at="2026-07-30T04:10:00Z")
ok("a claim from yesterday does NOT count as UP", loop_of([OLD])["state"] == "unknown")


# ── E · ACTIVITY ────────────────────────────────────────────────────────────

def feed(events, limit=40):
    return run_js(
        "const f = decisionsAndOutcomes(a.events, a.limit);\n"
        "console.log(JSON.stringify({keys: f.rows.map(r => r.key), count: f.count,\n"
        "  dropped: f.dropped, unrecognised: f.unrecognised}));\n",
        {"events": events, "limit": limit})


def ev(key, headline, actor="M-A", source="comment", kind="system", task_id=1):
    return {"key": key, "at": "2026-07-31T03:00:00Z", "actor": actor,
            "source": source, "kind": kind, "taskId": task_id, "headline": headline}


print("\n── RAIL 17 · what the feed EXCLUDES is the spec ────────────────────")
EXCLUDED = [
    ev("e1", "status: Todo → In Progress (by F)"),
    ev("e2", "assignee: F → M"),
    ev("e3", "dispatched to F·auto (run r123-257)"),
    ev("e4", "step 3 \"Build the dashboard\" complete (by F)"),
    ev("e5", "heartbeat written for R"),
]
r = feed(EXCLUDED)
ok("status ticks, assignee changes, dispatch claims, step ticks and heartbeats are ALL excluded",
   r["count"] == 0 and r["dropped"] == 5, r)
INCLUDED = [
    ev("k1", "stamp approved on step 1 \"Approve the module list\" (by kevin)"),
    ev("k2", "status: Review → Done (by M)"),
    ev("k3", "status: Done → Closed (by kevin)"),
    ev("k4", "run error — r999", source="run", kind="run-error"),
    ev("k5", "R · Release train Wave 30 merged"),
    ev("k6", "deploy watch: 7/7 success"),
    ev("k7", "RORT_MTF_PB_PREV_EPOCH armed on 4 services"),
    ev("k8", "⚠️ issue raised on step 2"),
    ev("k9", "I'm good with the modules you recommended", actor="kevin", kind="comment"),
]
r = feed(INCLUDED)
ok("stamps / Done / Closed / failed runs / trains / deploys / armed flags / issues / Kevin are KEPT",
   r["count"] == 9, r)
ok("a status tick to Done is NOT dropped by the status-tick rule (order matters)",
   "k2" in r["keys"] and "k3" in r["keys"], r["keys"])
ok("a successful run is not an event on this page — only its report is",
   feed([ev("s1", "run ok — r1", source="run", kind="run-ok")])["count"] == 0)


print("\n── RAIL 18 · unrecognised events are COUNTED, never silently gone ──")
r = feed([ev("u1", "some shape no rule has met yet")])
ok("an unrecognised event is not painted", r["count"] == 0, r)
ok("...but it IS counted, so the page can say how much it left out",
   r["unrecognised"] == 1, r)
r = feed(INCLUDED, limit=3)
ok("the feed is bounded", len(r["keys"]) == 3, r["keys"])
ok("...while count stays the full number", r["count"] == 9, r["count"])


# ── THE PAGE ────────────────────────────────────────────────────────────────

print("\n── RAIL 19 · zero looks like a good answer ─────────────────────────")
for copy in ["Nothing is waiting on you.", "Nothing is stalled.", "Quiet since"]:
    ok(f"empty-state copy present: “{copy}”", copy in KD, "missing from kevinDashboard.ts")
ok("the page renders the plate's empty copy", "EMPTY_COPY.plate" in KPAGE)
ok("the page renders the stuck empty copy", "EMPTY_COPY.stuck" in KPAGE)
ok("the page renders the activity empty copy", "EMPTY_COPY.activity" in KPAGE)
ok("each PLATE bucket carries its OWN empty line, so a zero group is never blank",
   KD.count("empty: '") == 4, KD.count("empty: '"))
ok("the empty stuck panel prints what it CHECKED, not just a zero",
   "stalls.checked" in KPAGE)
ok("the page never says “No results”", "No results" not in KPAGE)


print("\n── RAIL 20 · ONE modal, two callers ────────────────────────────────")
ok("the page imports the EXISTING TaskDetailModal",
   "import TaskDetailModal from './TaskDetailModal'" in KPAGE)
ok("...and does not define its own", "function TaskDetailModal" not in KPAGE)
for prop in ["task=", "allTasks=", "visionOptions=", "roles=", "patch=", "del=",
             "onClose=", "onOpenTask=", "commentAuthor=", "onPickAuthor=",
             "headless=", "onRunRequested=", "onPollTick="]:
    ok(f"prop supplied to the modal: {prop.rstrip('=')}",
       prop in KPAGE.split("<TaskDetailModal")[-1])
ok("the modal is wrapped in AgentRegistryContext.Provider (or role chips fall back "
   "to their outage rendering — a quiet visual regression)",
   "AgentRegistryContext.Provider" in KPAGE)
ok("the registry value comes from agentRegistryMap, as the board's does",
   "agentRegistryMap(agents)" in KPAGE)
ok("there is exactly ONE TaskDetailModal component file",
   len([p for p in os.listdir(os.path.join(FE, "views"))
        if p.startswith("TaskDetailModal")]) == 1)


print("\n── RAIL 21 · /admin/tasks still opens the modal exactly as before ──")
ok("AdminTasksPage still renders TaskDetailModal", "<TaskDetailModal" in TASKS)
ok("...still inside its AgentRegistryContext.Provider",
   "AgentRegistryContext.Provider" in TASKS)
for prop in ["task={modalTask}", "allTasks={tasks}", "visionOptions={visionItems}",
             "patch={patch}", "del={del}", "onPollTick={pollRefresh}"]:
    ok(f"AdminTasksPage prop unchanged: {prop}", prop in TASKS)
try:
    diff = subprocess.run(
        ["git", "-C", ROOT, "diff", "origin/dev", "--",
         "clients/KevBot_Toolkit/RoR_Trader/frontend/src/views/TaskDetailModal.tsx"],
        capture_output=True, text=True, timeout=30)
    if diff.returncode == 0:
        ok("TaskDetailModal.tsx is BYTE-UNCHANGED vs origin/dev — called from a second "
           "page, not modified for it", diff.stdout.strip() == "",
           diff.stdout[:400])
    else:
        print("  --: origin/dev not resolvable here; the byte-identity check was skipped")
except Exception as e:                                        # pragma: no cover
    print(f"  --: git unavailable ({e}); byte-identity check skipped")


print("\n── RAIL 22 · NO FILTER BAR ─────────────────────────────────────────")
ok("the page renders no MultiSelectFilter", "MultiSelectFilter" not in KPAGE)
ok("...no status/area/assignee filter state, and no filter predicates",
   not any(n in KPAGE for n in ["statusFilter", "areaFilter", "assigneeFilter",
                                "taskMatchesSelects", "activeSelectCount"]))
ok("...and it SAYS why, pointing at the shared surface instead",
   "no filter bar" in KPAGE and "/admin/tasks" in KPAGE)


print("\n── RAIL 23 · wiring ────────────────────────────────────────────────")
route = os.path.join(FE, "app", "admin", "kevin", "page.tsx")
ok("the route exists", os.path.exists(route))
ok("...and mounts the view", "AdminKevinPage" in read(route))
ok("the nav links it under the `Team` section (board #252)",
   "'/admin/kevin'" in SIDEBAR)
team = SIDEBAR.split("label: 'Team'")[-1].split("},\n  {")[0]
ok("...inside Team's children, not somewhere else",
   "/admin/kevin" in team, team[:200])
ok("SignOfLife is REUSED, not reimplemented (#251)",
   "import SignOfLife from './SignOfLife'" in KPAGE
   and "function SignOfLife" not in KPAGE)
ok("the shipping-lane resolver is reused for the car list, not forked",
   "shippingRows" in KPAGE and "from './AdminDispatchPage'" in KPAGE)
ok("dispatch eligibility is the EXISTING runIneligibleReason",
   "runIneligibleReason" in KPAGE)
ok("the daily cap is resolved with the loop's own policy (#228), not restated",
   "resolveDailyCap" in KPAGE)
ok("the page polls with list GETs joined in memory, never per-row (#148)",
   "Promise.all" in KPAGE and "/api/dev-tasks?include_done=true" in KPAGE)


print("\n── RAIL 24 · no schema change ──────────────────────────────────────")
mig = os.path.join(ROOT, "migrations")
new_mig = [f for f in os.listdir(mig) if "257" in f or "kevin_dash" in f] \
    if os.path.isdir(mig) else []
ok("board #257 introduces NO migration", new_mig == [], new_mig)


print("\n" + "=" * 72)
print(f"{PASS} passed, {len(FAILED)} failed")
if FAILED:
    for f in FAILED:
        print(f"  ✗ {f}")
    sys.exit(1)
print("ALL RAILS GREEN")
