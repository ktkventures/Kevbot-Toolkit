"""Acceptance test — board #251: the R session dashboard.

WHAT THIS PINS
--------------
The page exists to catch ONE shape: on 07-30 #246 was approved, built, tested
and PR'd inside an hour, then sat in `Staged` for FOUR HOURS because nobody
created the train that ships it. Nothing was broken. Nobody was watching.

So the headline rail is a replay of that exact board state — and a version of
this module that would not have flagged #246 is not worth shipping.

RAILS, each with a check that FAILS without it
----------------------------------------------
CARS WAITING FOR A TRAIN (M's three-clause definition, board #251 step 2)
  1. #246's HISTORICAL SHAPE IS FLAGGED — `Staged`, branch
     `feat/tasks-multiselect-filters-246` unmerged, no train task in existence.
  2. clause 1 — a task with unmerged code that is NOT `Staged` is absent
     (the shipping lane is wider on purpose; a train is owed only for parked
     work).
  3. clause 3 — the SAME task drops off the moment an open train names it, and
     is reported under `covered` rather than silently vanishing.
  4. a train that is `Done` does NOT cover a car — a shipped train cannot be
     the reason a new car is ignored.
  5. a train does not cover ITSELF (its own `#id` appears in its own body).
  6. `trainOwed` is the flag, and it is false only when the list is empty.
  7. clause 2 is NOT re-implemented — the module IMPORTS `shippingRows` from
     AdminDispatchPage. A second definition of "has unmerged code" would drift
     from the #245 one that already has tests.
  8. the row list is BOUNDED (#240 — the M-session activity log painted ~435
     rows and stalled Kevin's browser) while `count` stays the FULL number.

TRAIN CLASSIFICATION
  9. an R-A ship step at a LATER index is a CAR, never its own train (that
     would report every car as already having one — a false green of exactly
     the shape being fixed). AND, measured on the live board: step-1 ownership
     ALONE is not sufficient either — #234/#235/#237/#238 all match it and are
     not trains, and #238 is open, so the loose rule would silently HIDE any
     car named in its body. A train must also say it is one.
 10. car ids are read from the title AND the description.

SIGN OF LIFE (router)
 11. a fresh beat is `fresh`; a 20m-old beat is `stale`; a 2h-old beat is `down`.
 12. a key that was NEVER WRITTEN is `never`, NOT `down` — the #157 lesson: an
     absent signal must not render as a confident measurement.
 13. an unparseable/absent timestamp is `never`, never `fresh`.
 14. freshness prefers `value.at` and NAMES which stamp it used.
 15. a session letter is VALIDATED before it becomes a settings key.
 16. TRACKED sessions with no row are still RETURNED (a page that lists only
     the sessions it can find would show an empty, reassuring strip).

DEPLOY LOG (router)
 17. wave numbers are parsed out of real Deploy_Log.md lines.
 18. the parse is BOUNDED by `limit`.
 19. a missing file is a STATED error carrying the absolute path — never an
     empty list, which would read as "no deploys have happened".

WIRING
 20. the router is registered in api/main.py (imported AND included).
 21. the page route + view exist, and BOTH session dashboards render SignOfLife.
 22. no migration is introduced by this task.

Run:  .venv/bin/python src/test_r_session_dashboard_251.py
      (from the RoR_Trader root)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
FE = os.path.join(ROOT, "frontend", "src")
sys.path.insert(0, SRC)

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


PAGE = read(FE, "views", "AdminDispatchPage.tsx")
SHARED = read(FE, "views", "taskBoardShared.tsx")
SIGNALS = read(FE, "views", "rSessionSignals.ts")
RPAGE = read(FE, "views", "AdminRSessionPage.tsx")
MPAGE = read(FE, "views", "AdminMSessionPage.tsx")


# ── Lifting the pure TS out so it can actually be RUN ───────────────────────
# Same technique as test_shipping_lane_unmerged_branch_245.py: only the
# SIGNATURE is de-typed, the body is copied verbatim, so what runs here is what
# ships. A static substring assertion would pass against a module that returns
# the wrong rows.

def _split_params(params):
    """Split a parameter list on TOP-LEVEL commas only (generics are nested)."""
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
    body = src[line_end + 1:i + 1]
    # Strip TS-only type arguments on locals that node cannot parse.
    body = re.sub(r"const (\w+): [^=]+ = ", r"const \1 = ", body)
    body = re.sub(r"new Map<[^>]*>\(\)", "new Map()", body)
    body = re.sub(r"\.forEach\(\((\w+): [^)]+\)", r".forEach((\1)", body)
    return f"function {name}({', '.join(_split_params(params))}) {{\n" + body


def js_const(src, name):
    m = re.search(rf"^export const {name} = ([^;]+);", src, re.M)
    assert m, f"{name} not found"
    return f"const {name} = {m.group(1)};\n"


NODE = shutil.which("node")
ok("node is available to execute the lifted derivations", bool(NODE))

FINISHED_SET = re.search(r"export const FINISHED_STATUSES = [^;]+;", SHARED)
FINISHED = re.search(r"export const isFinished = [^;]+;", SHARED)
PRELUDE = (
    FINISHED_SET.group(0).replace("export ", "") + "\n"
    + re.sub(r"\(status\?[^)]*\):\s*boolean\s*=>", "(status) =>",
             FINISHED.group(0).replace("export ", "")) + "\n"
    + "const stepOwner = (s) => (s.owner !== undefined && s.owner !== null) ? s.owner : (s.role ?? null);\n"
    + "const stepTitle = (s) => s.title || s.text || '';\n"
    + "const NOW = Date.parse('2026-07-31T01:00:00Z');\n"
    + "const hoursSince = (t) => t ? (NOW - Date.parse(t)) / 3600000 : 0;\n"
)


def run_js(tasks, runs_by_task, pushed_known, limit=25):
    """Run the SHIPPED shippingRows → carsWaitingForATrain chain on a fixture."""
    harness = (
        PRELUDE
        + js_const(PAGE, "SHIP_STEP_OWNERS")
        + js_const(SIGNALS, "TRAIN_TITLE_RE")
        + js_const(SIGNALS, "CAR_REF_RE")
        + js_const(SIGNALS, "TRAIN_WORD_RE")
        + js_body(PAGE, "mergeStateOf") + "\n"
        + js_body(PAGE, "shippingRows") + "\n"
        + js_body(SIGNALS, "trainWave") + "\n"
        + js_body(SIGNALS, "isTrainTask") + "\n"
        + js_body(SIGNALS, "trainCarIds") + "\n"
        + js_body(SIGNALS, "openTrains") + "\n"
        + js_body(SIGNALS, "carsWaitingForATrain") + "\n"
        + "const a = JSON.parse(process.argv[2]);\n"
        + "const m = new Map(Object.entries(a.runs).map(([k, v]) => [Number(k), v]));\n"
        + "const lane = shippingRows(a.tasks, m, a.pushedKnown);\n"
        + "const cars = carsWaitingForATrain(lane, a.tasks, a.limit);\n"
        + "console.log(JSON.stringify({\n"
        + "  laneIds: lane.rows.map(r => r.task.id),\n"
        + "  ids: cars.rows.map(r => r.task.id), count: cars.count,\n"
        + "  branches: cars.rows.map(r => r.branch), oldestH: cars.oldestH,\n"
        + "  covered: cars.covered, openTrainIds: cars.openTrainIds,\n"
        + "  trainOwed: cars.trainOwed, warning: cars.warning,\n"
        + "  trains: a.tasks.filter(isTrainTask).map(t => t.id),\n"
        + "  carIds: Object.fromEntries(a.tasks.map(t => [t.id, trainCarIds(t)])),\n"
        + "}));\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        out = subprocess.run(
            [NODE, path, json.dumps({"tasks": tasks, "runs": runs_by_task,
                                     "pushedKnown": pushed_known, "limit": limit})],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:900]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


# ── Board fixtures ──────────────────────────────────────────────────────────

def task(tid, title, status, checklist, description=""):
    return {"id": tid, "title": title, "status": status, "description": description,
            "checklist": checklist, "updated_at": "2026-07-30T20:00:00Z"}


def step(title, owner, done):
    return {"title": title, "owner": owner, "done": done}


def run(tid, branch, pushed_at="2026-07-30T21:00:00Z"):
    return [{"id": tid * 10, "task_id": tid, "agent_letter": "F",
             "requested_at": "2026-07-30T20:30:00Z",
             "finished_at": "2026-07-30T21:00:00Z", "outcome": "ok",
             "pushed_branch": branch, "pushed_at": pushed_at}]


# #246 verbatim: a real chain, built, Staged, ship step open, branch unmerged.
CAR_246 = task(246, "Tasks page — multiselect filters", "Staged", [
    step("Approve scope", "kevin", True),
    step("Build the filters", "F", True),
    step("M review and stage", "M", True),
    step("Ship via a release train", "R-A", False),
])
RUNS_246 = {"246": run(246, "feat/tasks-multiselect-filters-246")}


print("\n── RAIL 1 · #246's 07-30 shape is FLAGGED ──────────────────────────")
r = run_js([CAR_246], RUNS_246, True)
ok("#246 is in the shipping lane at all (clause 2 — unmerged branch)",
   r["laneIds"] == [246], r["laneIds"])
ok("#246 IS FLAGGED as a car waiting for a train (the 07-30 four-hour stall)",
   r["ids"] == [246], r)
ok("its branch is named on the row", r["branches"] == ["feat/tasks-multiselect-filters-246"],
   r["branches"])
ok("trainOwed is TRUE — this is the actionable signal", r["trainOwed"] is True)
ok("no open train is reported, because none existed", r["openTrainIds"] == [],
   r["openTrainIds"])
ok("age is non-zero, so the panel can lead with how long it has waited",
   r["oldestH"] > 0, r["oldestH"])


print("\n── RAIL 2 · clause 1 — only Staged ─────────────────────────────────")
in_review = dict(CAR_246, status="Review")
r = run_js([in_review], RUNS_246, True)
ok("a Review task with the SAME unmerged branch is in the shipping lane",
   r["laneIds"] == [246], r["laneIds"])
ok("...but is NOT a car waiting for a train (a train is owed for PARKED work)",
   r["ids"] == [], r["ids"])
ok("trainOwed is false when nothing is Staged", r["trainOwed"] is False)


print("\n── RAIL 3 · clause 3 — an open train covers its cars ───────────────")
TRAIN = task(252, "R · Release train Wave 28 (07-31) — #246", "Todo", [
    step("Merge the cars in order, gate combined", "R-A", False),
], description="Car 1: #246 `feat/tasks-multiselect-filters-246`")
r = run_js([CAR_246, TRAIN], RUNS_246, True)
ok("with an OPEN train naming it, #246 drops off the waiting list",
   r["ids"] == [], r["ids"])
ok("...and is reported under `covered`, never silently vanished",
   [c["id"] for c in r["covered"]] == [246], r["covered"])
ok("the covering train is NAMED on the covered row",
   r["covered"] and r["covered"][0]["trainId"] == 252, r["covered"])
ok("the open train is listed", r["openTrainIds"] == [252], r["openTrainIds"])
ok("trainOwed is FALSE — somebody is coming for it", r["trainOwed"] is False)


print("\n── RAIL 4 · a FINISHED train does not cover a car ──────────────────")
done_train = dict(TRAIN, status="Done")
r = run_js([CAR_246, done_train], RUNS_246, True)
ok("a Done train does not count as coverage — #246 is flagged again",
   r["ids"] == [246], r)
ok("...and it is not listed as an open train", r["openTrainIds"] == [],
   r["openTrainIds"])
closed_train = dict(TRAIN, status="Closed")
r = run_js([CAR_246, closed_train], RUNS_246, True)
ok("a Closed train does not count as coverage either", r["ids"] == [246], r["ids"])


print("\n── RAIL 5 · a train does not cover ITSELF ──────────────────────────")
# A train's own `#id` appears in its body; self-coverage would hide a stalled
# train from the very panel meant to surface it.
self_ref = task(252, "R · Release train Wave 28 — #252", "Staged", [
    step("Merge the cars", "R-A", False),
], description="This train is #252.")
r = run_js([self_ref], {"252": run(252, "release/wave28-252")}, True)
ok("a Staged train with an unmerged branch flags ITSELF as waiting",
   r["ids"] == [252], r)
ok("...and does not appear in its own `covered` list",
   [c["id"] for c in r["covered"]] == [], r["covered"])


print("\n── RAIL 7 · clause 2 is IMPORTED, not re-implemented ───────────────")
ok("rSessionSignals imports shippingRows from the #245 resolver",
   re.search(r"import \{[^}]*shippingRows[^}]*\} from '\./AdminDispatchPage'", SIGNALS)
   or re.search(r"import \{[^}]*ShipLane[^}]*\} from '\./AdminDispatchPage'", SIGNALS),
   "the module must consume the shipping-lane resolver, never fork it")
ok("rSessionSignals does NOT define its own merge-state function",
   "function mergeStateOf" not in SIGNALS and "merge-base" not in SIGNALS,
   "a second definition of merged-ness would drift from #245's")
ok("the R page calls shippingRows rather than re-deriving membership",
   "shippingRows(tasks, runsByTask, pushedKnown)" in RPAGE)


print("\n── RAIL 8 · every list is BOUNDED, counts are not ──────────────────")
many = [dict(CAR_246, id=300 + i, title=f"car {i}") for i in range(40)]
many_runs = {str(300 + i): run(300 + i, f"feat/car-{i}") for i in range(40)}
r = run_js(many, many_runs, True, limit=25)
ok("40 waiting cars render only 25 rows (#240 — bound the DOM)",
   len(r["ids"]) == 25, len(r["ids"]))
ok("...but `count` still reports all 40 (render less, never KEEP less)",
   r["count"] == 40, r["count"])
ok("R_PAGE_ROWS is a declared bound, not a magic number in the JSX",
   "export const R_PAGE_ROWS" in SIGNALS)
ok("the page passes that bound to every list",
   RPAGE.count("R_PAGE_ROWS") >= 3, RPAGE.count("R_PAGE_ROWS"))


print("\n── RAIL 9/10 · train classification ────────────────────────────────")
r = run_js([CAR_246, TRAIN], RUNS_246, True)
ok("step-1-owned-by-R-A IS a train", 252 in r["trains"], r["trains"])
ok("a CAR with an R-A ship step at a LATER index is NOT a train",
   246 not in r["trains"],
   "classifying cars as their own trains would report every car as covered")
ok("car ids are read from the train's TITLE", 246 in r["carIds"]["252"])
body_only = task(253, "R · train", "Todo",
                 [step("Merge", "R-A", False)], description="cars: #246 and #240")
r2 = run_js([CAR_246, body_only], RUNS_246, True)
ok("car ids are also read from the DESCRIPTION",
   set(r2["carIds"]["253"]) >= {246, 240}, r2["carIds"]["253"])
ok("a body-named car is therefore covered too", r2["ids"] == [], r2["ids"])

# MEASURED ON THE LIVE BOARD, 07-30: four tasks have step 1 owned by R/R-A and
# are NOT trains — #234 (M hand-off doc), #235 (dev is RED), #237 (the api-502
# outage) and #238 (a release-brief change). #238 is Staged and OPEN, so a
# step-1-only rule would treat it as an open train and SILENTLY HIDE any car id
# in its body. A hidden car is an invisible stall — the 07-30 failure again.
NOT_A_TRAIN = task(238, "Release brief: PROBE the service after deploy-watch",
                   "Staged", [step("Add the probe step", "R-A", False)],
                   description="Follows from #246 and #241.")
r3 = run_js([CAR_246, NOT_A_TRAIN], RUNS_246, True)
ok("a release-lane task whose step 1 is R-A but whose title says no train is "
   "NOT classified as one (#238's real shape)", 238 not in r3["trains"], r3["trains"])
ok("...so it CANNOT hide #246 — the car is still flagged", 246 in r3["ids"], r3["ids"])
ok("...and it is not counted as an open train", r3["openTrainIds"] == [],
   r3["openTrainIds"])
HAND_TRAIN = task(260, "R-A · release train, hand-assembled", "Todo",
                  [step("Merge the cars", "R-A", False)], description="cars: #246")
r4 = run_js([CAR_246, HAND_TRAIN], RUNS_246, True)
ok("a hand-authored train that SAYS train is still recognised",
   260 in r4["trains"] and r4["ids"] == [], r4)


print("\n── RAIL 6 · the lane's own warning is carried, not swallowed ───────")
r = run_js([CAR_246], {"246": [dict(run(246, None)[0], pushed_branch=None)]}, False)
ok("with `pushed_branch` unreadable the panel still renders the legacy fallback",
   r["ids"] == [246], r)
ok("...and RE-STATES the resolver's warning rather than hiding the degrade",
   bool(r["warning"]) and "pushed_branch" in r["warning"], r["warning"])


# ── The router: sign of life, deploy log ───────────────────────────────────
print("\n── RAILS 11-16 · SIGN OF LIFE ─────────────────────────────────────")
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_KEY", "test")
from api.routers import r_session as R  # noqa: E402

NOW = datetime(2026, 7, 31, 1, 0, 0, tzinfo=timezone.utc)


def beat(minutes_ago, actor="R", key_at=True):
    at = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    return {"key": "session_heartbeat_R",
            "value": ({"at": at, "actor": actor} if key_at else {"actor": actor}),
            "updated_at": at}


s = R.heartbeat_state(beat(2), NOW)
ok("a 2-minute-old beat is `fresh`", s["state"] == "fresh", s)
s = R.heartbeat_state(beat(20), NOW)
ok("a 20-minute-old beat is `stale` (amber, probably mid-task)",
   s["state"] == "stale", s)
s = R.heartbeat_state(beat(120), NOW)
ok("a 2-hour-old beat is `down` → the UI reads NOT RUNNING",
   s["state"] == "down", s)
s = R.heartbeat_state(None, NOW)
ok("a key that was NEVER WRITTEN is `never`, NOT `down` (#157: an absent "
   "signal must not render as a measurement)", s["state"] == "never", s)
ok("...and it says so in `why`", "never written" in s["why"], s["why"])
s = R.heartbeat_state({"key": "session_heartbeat_R", "value": {"actor": "R"},
                       "updated_at": "not-a-timestamp"}, NOW)
ok("an UNPARSEABLE timestamp is `never`, never `fresh`", s["state"] == "never", s)
s = R.heartbeat_state(beat(1), NOW)
ok("freshness prefers `value.at`", s["at_source"] == "value.at", s)
s = R.heartbeat_state(beat(1, key_at=False), NOW)
ok("...and falls back to `updated_at`, NAMING which stamp answered",
   s["at_source"] == "updated_at", s)
s = R.heartbeat_state(beat(-30), NOW)
ok("a FUTURE beat is not clamped to healthy — clock skew stays visible",
   s["age_s"] is not None and s["age_s"] < 0, s)

for bad in ("", None, "r session", "../../etc", "RR", "session_heartbeat_R"):
    try:
        R.parse_letter(bad)
        ok(f"letter {bad!r} is REFUSED (it becomes a settings key)", False)
    except Exception as e:  # HTTPException
        ok(f"letter {bad!r} is REFUSED (it becomes a settings key)",
           getattr(e, "status_code", None) == 400, str(e)[:120])
ok("'r' is accepted and normalised to 'R'", R.parse_letter("r") == "R")
ok("'R-A' is a valid session letter", R.parse_letter("R-A") == "R-A")
ok("the key is built from the validated letter",
   R.heartbeat_key("R") == "session_heartbeat_R")
ok("M and R are BOTH tracked, so a page lists a session that never wrote a beat",
   set(R.TRACKED_SESSIONS) >= {"M", "R"}, R.TRACKED_SESSIONS)
ok("thresholds are module constants the API returns (never restated in the UI)",
   R.FRESH_MAX_S > 0 and R.STALE_MAX_S > R.FRESH_MAX_S)
ok("the UI reads the thresholds from the payload rather than hard-coding them",
   "thresholds.fresh_max_s" in read(FE, "views", "SignOfLife.tsx"))
ok("`never` and `down` get DIFFERENT labels but both read NOT RUNNING",
   "never started" in SIGNALS and SIGNALS.count("NOT RUNNING") >= 2)
ok("the heartbeat write puts the ACTOR in the JSONB value, not `updated_by` "
   "(which is a UUID FK to auth.users)",
   "updated_by" not in read(SRC, "api", "routers", "r_session.py")
   .split("def write_heartbeat")[1].split("def ")[0].replace(
       "# `updated_by` is a UUID FK to auth.users — the ACTOR goes in the JSONB", ""))


print("\n── RAILS 17-19 · DEPLOY LOG ───────────────────────────────────────")
LOG = """# Deploy Log

## 2026-07-30

- **22:22:23 UTC (16:22 MT 07-30)** — `2164afb6` **Wave-26 train — five cars.**
  - Notes: bounded
- **20:32:07 UTC (14:32 MT 07-30)** — `a41c39d1` **Wave-25 train — three cars.**

## 2026-07-29

- **18:00:00 UTC (12:00 MT 07-29)** — `deadbeef` **Wave 24 train.**
"""
e = R.parse_deploy_log(LOG)
ok("every entry is parsed", len(e) == 3, e)
ok("wave numbers are read (`Wave-26` and `Wave 24` both)",
   [x["wave"] for x in e] == [26, 25, 24], [x["wave"] for x in e])
ok("shas are read", [x["sha"] for x in e] == ["2164afb6", "a41c39d1", "deadbeef"])
ok("the day heading is carried onto the entry", e[0]["day"] == "2026-07-30", e[0])
ok("the parse is BOUNDED by limit", len(R.parse_deploy_log(LOG, 1)) == 1)
ok("excerpts are truncated, not shipped whole",
   all(len(x["excerpt"]) <= R.DEPLOY_LOG_EXCERPT for x in e))
ok("a log with no matching lines returns [] rather than raising",
   R.parse_deploy_log("nothing here") == [])
ok("the real Deploy_Log.md parses to at least one wave",
   len([x for x in R.parse_deploy_log(read(ROOT, "docs", "Deploy_Log.md")) if x["wave"]]) > 0)
RS = read(SRC, "api", "routers", "r_session.py")
ok("a missing deploy log is a STATED error carrying the absolute path",
   'resolved_path' in RS and 'could not read' in RS)
ok("...and never an empty 200 that reads as 'no deploys have happened'",
   '"found": False' in RS)


print("\n── RAILS 20-22 · WIRING ───────────────────────────────────────────")
MAIN = read(SRC, "api", "main.py")
ok("the router is IMPORTED in api/main.py",
   "from api.routers.r_session import router as r_session_router" in MAIN)
ok("...and INCLUDED (an import alone serves nothing)",
   "app.include_router(r_session_router)" in MAIN)
ok("the route page exists at /admin/r-session",
   os.path.exists(os.path.join(FE, "app", "admin", "r-session", "page.tsx")))
ok("the R page renders CARS WAITING FOR A TRAIN as module 1",
   RPAGE.index("Cars waiting for a train") < RPAGE.index("Recent trains"),
   "the lead module must be the thing that failed on 07-30")
ok("the R page renders all five modules in M's order",
   all(s in RPAGE for s in ("Cars waiting for a train", "Recent trains",
                            "Deploy log", "/health", "Blocked")))
ok("the R page renders SignOfLife", "<SignOfLife" in RPAGE)
ok("the M page ALSO renders SignOfLife (both dashboards, one component)",
   "<SignOfLife" in MPAGE and "from './SignOfLife'" in MPAGE)
ok("SignOfLife lives in ONE file, so the two pages cannot disagree",
   os.path.exists(os.path.join(FE, "views", "SignOfLife.tsx")))
mig = os.path.join(SRC, "migrations")
new_mig = [f for f in os.listdir(mig) if "r_session" in f or "heartbeat" in f]
ok("NO migration is introduced — `system_settings` already exists",
   new_mig == [], new_mig)
ok("the router says so explicitly, so a reviewer does not go looking",
   "THERE ISN'T ONE" in RS or "no migration" in RS.lower())
ok("the page prints WHAT IT CANNOT SEE (the #157 dead-alarm lesson)",
   "CANNOT SEE" in RPAGE and "known_gaps" in RPAGE)


print(f"\n{'=' * 70}")
print(f"{PASS} checks passed, {len(FAILED)} failed")
if FAILED:
    for f in FAILED:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("ALL PASS")
