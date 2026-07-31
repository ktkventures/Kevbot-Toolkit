"""Acceptance test — board #232: `Stand By` + `Closed`, and ONE shared
"finished" set.

Kevin added two statuses on 07-30 and renamed the Approval stamp:

  • `Stand By` — he has SEEN it, wants it, and chose *not yet*. NOT `Blocked`:
    Blocked means something PREVENTS progress and someone should come look.
    Collapsing the two buries real blockers among deliberately parked work.
  • `Closed`  — his version of done, AFTER M's `Done`. Two payoffs: visibility
    (~10 tasks reached Done on 07-30 without him seeing any of them) and a
    follow-up hook while the work is fresh. It BLOCKS NOTHING — the work has
    already shipped by then, which is the right place for a non-blocking check.
  • the stamp is now `approved to start`, and it is the ONLY mode. The retired
    two-touch stamp was a PRE-commitment to review, days before the work landed;
    `Closed` is the POST check that replaces it. Keeping both = reviewing twice.

THE EXPENSIVE HALF IS NOT THE STATUSES. `Done` meant "finished" in ~14
load-bearing places, and the sharpest — `done_ids` in dispatcher.one_pass() — is
how `blocked_by` is RESOLVED. Spelled `status=eq.Done`, a blocker moving
Done → Closed drops out of that set and every dependent task silently re-blocks:
a task that dispatched fine yesterday just stops, with no error, no log line and
no comment. That is rail 1 below, and it is the reason #232 is a sweep rather
than a one-line addition.

Every rail here FAILS without its fix. Rail 1 in particular is mutation-tested
against the real filter: the fake board honours PostgREST `status=in.(…)` and
`status=eq.…` literally, so reverting FINISHED_FILTER to the old spelling makes
the Closed blocker vanish from `done_ids` and the dispatch stops.

  1. blocked_by a CLOSED task still dispatches — end to end through one_pass()
     (and the control: blocked_by a BLOCKED task still blocks, because Blocked
     is undispatchable but emphatically NOT finished)
  2. an ARMED `Stand By` task is NOT dispatched, and is reported QUIET-not-stuck
  3. an ARMED `Closed` task, same
  4. the Run button refuses both, for the two different reasons
  5. ONE shared finished set on each side, and no hand-written `== 'Done'`
     finished-check left in the board UI
  6. both statuses render: STATUSES / STATUS_DEF / STATUS_COLOR, colour distinct
     from Blocked's
  7. the stamp: one mode, labelled `approved to start`, never sets kevin_final —
     and the retired mode is refused LOUDLY, not silently downgraded
  8. the SessionStart hook's open-queue excludes Closed too

Offline + hermetic: the real `tools/team_dispatcher/dispatcher.py` runs against a
fake board that parses the filters it emits, the real `api.routers.dev_tasks`
constants are imported, and the frontend is source-scanned.

Run:  .venv/bin/python src/test_board_statuses_232.py   (from RoR_Trader root)
"""
import importlib.util
import io
import contextlib
import os
import re
import sys
import tempfile
import types

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
VIEWS = os.path.join(ROOT, "frontend", "src", "views")
sys.path.insert(0, SRC)

PASS = FAIL = 0


def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label} — {detail}")


# ── the real dispatcher ─────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "team_dispatcher",
    os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

AGENTS = {"F": {"letter": "F", "status": "headless", "worktree": ".",
                "scope": "s", "boundaries": "b"},
          "M-A": {"letter": "M-A", "status": "headless", "worktree": ".",
                  "scope": "s", "boundaries": "b"}}


def task(**kw):
    t = {"id": 1, "title": "T", "description": "d", "assignee": "F",
         "status": "Todo", "ai_eligible": False, "tags": [], "blocked_by": [],
         "checklist": []}
    t.update(kw)
    return t


# The fake board parses the filters the dispatcher actually emits, so a gate
# that changes its query is really re-tested rather than silently matching
# everything. Three shapes: `or=(…)`, `status=in.(A,B)` and `col=eq.v`.
_OR_RE = re.compile(r"or=\(([^)]*)\)")
_IN_RE = re.compile(r"(\w+)=in\.\(([^)]*)\)")
_EQ_RE = re.compile(r"(\w+)=eq\.([^&]+)")


def _or_match(row, clause):
    col, _, val = clause.partition(".eq.")
    if val in ("true", "false"):
        return row.get(col) is (val == "true")
    return str(row.get(col)) == val


def board_api(rows):
    """Fake `api()` over `rows`, honouring the filter it is handed."""
    def _api(method, path, body=None, prefer=None):
        if method != "GET" or not path.startswith("dev_tasks?"):
            return []
        q = path.split("?", 1)[1]
        if q.startswith("tags=cs."):        # the Run-button lane
            tag = q.split("{", 1)[1].split("}", 1)[0]
            return [dict(r) for r in rows if tag in (r.get("tags") or [])]
        m = _OR_RE.search(q)
        if m:
            return [dict(r) for r in rows
                    if any(_or_match(r, c) for c in m.group(1).split(","))]
        m = _IN_RE.search(q)
        if m:
            vals = [v.strip() for v in m.group(2).split(",")]
            return [dict(r) for r in rows if str(r.get(m.group(1))) in vals]
        m = _EQ_RE.search(q)
        if m:
            return [dict(r) for r in rows
                    if str(r.get(m.group(1))) == m.group(2)]
        raise AssertionError(f"unmodelled dev_tasks filter: {q}")
    return _api


def triage(rows, st=None):
    D.api = board_api(rows)
    done_ids = {t["id"] for t in
                D.api("GET", f"dev_tasks?{D.FINISHED_FILTER}&select=id")}
    out, skipped = D.triage_todo(AGENTS, done_ids, st)
    return [t["id"] for t in out], skipped


def dry_pass(rows, only_task):
    """Run the REAL one_pass() in DRY-RUN over `rows` and return its stdout.

    This is what makes rail 1 an end-to-end rail rather than a restatement of
    the constant: `done_ids` is assembled by the shipped line in one_pass(),
    from the fake board, through the filter the module actually builds.
    """
    tmp = tempfile.mkdtemp(prefix="t232-")
    saved = {k: getattr(D, k) for k in
             ("api", "LOG_DIR", "headless_agents", "load_state", "save_state",
              "reap", "effective_daily_cap", "today_run_count", "mentions_for",
              "build_prompt", "PAUSE_FILE")}
    try:
        D.api = board_api(rows)
        D.LOG_DIR = tmp
        D.PAUSE_FILE = os.path.join(tmp, "NO_SUCH_PAUSE")
        D.headless_agents = lambda: (AGENTS, "stub")
        D.load_state = lambda: {"runs": []}
        D.save_state = lambda st: None
        D.reap = lambda st: None
        D.effective_daily_cap = lambda: (50, "stub")
        D.today_run_count = lambda st: 0
        D.mentions_for = lambda letter: ("", [])
        D.build_prompt = lambda *a, **k: "PROMPT"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            D.one_pass(live=False, only_task=only_task)
        return buf.getvalue()
    finally:
        for k, v in saved.items():
            setattr(D, k, v)


# ── RAIL 1 — THE SILENT ONE: blocked_by a CLOSED task still dispatches ──────
print("― rail 1: a task blocked_by a CLOSED task still dispatches ―")

blocker_done = task(id=900, status="Done", assignee="M-A")
blocker_closed = task(id=901, status="Closed", assignee="M-A")
blocker_blocked = task(id=902, status="Blocked", assignee="M-A")

out = dry_pass([blocker_closed, task(id=910, blocked_by=[901])], only_task=910)
ok("one_pass() DISPATCHES #910 whose blocker is Closed (the silent re-block "
   "this whole task exists to prevent)",
   "WOULD DISPATCH #910" in out, out.strip())

out = dry_pass([blocker_done, task(id=911, blocked_by=[900])], only_task=911)
ok("...and a Done blocker still works exactly as it always did",
   "WOULD DISPATCH #911" in out, out.strip())

out = dry_pass([blocker_done, blocker_closed,
                task(id=912, blocked_by=[900, 901])], only_task=912)
ok("...and a MIX of Done + Closed blockers dispatches",
   "WOULD DISPATCH #912" in out, out.strip())

# THE CONTROL, and the reason FINISHED_STATUSES and TERMINAL_STATUSES are two
# different constants: `Blocked` is undispatchable but NOT finished, so it must
# keep blocking. A "fix" that widened done_ids to every terminal status would
# pass rail 1 and break this.
out = dry_pass([blocker_blocked, task(id=913, blocked_by=[902])], only_task=913)
ok("a BLOCKED blocker still BLOCKS — finished ≠ terminal",
   "WOULD DISPATCH #913" not in out and "blocked by [902]" in out, out.strip())

got, skipped = triage([blocker_closed, task(id=914, blocked_by=[901])])
ok("triage_todo agrees when handed the same done_ids", got == [914],
   f"skipped={[s[1] for s in skipped]}")

_, skipped = triage([task(id=915, blocked_by=[999])])
ok("the skip reason no longer says 'not Done' — it names the SET",
   skipped and "not finished" in skipped[0][1] and "Closed" in skipped[0][1],
   skipped[0][1] if skipped else "")


# ── RAIL 2/3 — an ARMED Stand By / Closed task is refused, and QUIETLY ──────
print("― rails 2+3: armed Stand By / Closed are refused, quiet-not-stuck ―")

for status in ("Stand By", "Closed"):
    got, skipped = triage([task(id=920, status=status, ai_eligible=True)],
                          {"runs": []})
    ok(f"an ARMED {status!r} task is NOT dispatched", got == [], f"got={got}")
    ok(f"...the reason says terminal and names {status!r}",
       skipped and "terminal" in skipped[0][1] and status in skipped[0][1],
       skipped[0][1] if skipped else "")
    # QUIET is the requirement, not a detail: is_stuck=True posts a "dispatch
    # skipped" comment and arms the 4h staleness tripwire. On `Stand By` that
    # would nag Kevin daily about a decision he has already made — which is
    # exactly the difference from `Blocked`, where someone SHOULD come look.
    ok(f"...and it is QUIET, not stuck ({status!r} needs nobody to come look)",
       skipped and skipped[0][2] is False)

ok("TERMINAL_STATUSES carries both new statuses beside the old three",
   D.TERMINAL_STATUSES == ("Done", "Closed", "Blocked", "Staged", "Stand By"),
   str(D.TERMINAL_STATUSES))
# Blocked's meaning is explicitly NOT changed by #232.
_, skipped = triage([task(id=921, status="Blocked", ai_eligible=True)],
                    {"runs": []})
ok("Blocked still behaves exactly as before (#232 did not touch its meaning)",
   skipped and "terminal" in skipped[0][1] and skipped[0][2] is False)


# ── RAIL 4 — the Run button refuses both ───────────────────────────────────
print("― rail 4: the Run button refuses Closed (finished) and Stand By (parked) ―")
rows = [task(id=930, status="Closed", tags=["run-requested"]),
        task(id=931, status="Stand By", tags=["run-requested"]),
        task(id=932, status="Todo", tags=["run-requested"])]
D.api = board_api(rows)
good, bad = D.run_requested(AGENTS, set())
ok("run_requested refuses Closed and Stand By, and still serves Todo",
   sorted(t["id"] for t, _ in bad) == [930, 931]
   and [t["id"] for t in good] == [932],
   f"good={[t['id'] for t in good]} bad={[t['id'] for t, _ in bad]}")


# ── RAIL 5 — ONE shared set on each side ───────────────────────────────────
print("― rail 5: one shared finished set, used everywhere ―")

ok("dispatcher.FINISHED_STATUSES is exactly Done + Closed",
   D.FINISHED_STATUSES == ("Done", "Closed"), str(D.FINISHED_STATUSES))
ok("dispatcher.is_finished() is the predicate behind it",
   D.is_finished("Done") and D.is_finished("Closed")
   and not D.is_finished("Blocked") and not D.is_finished("Stand By")
   and not D.is_finished(None))
ok("FINISHED_FILTER is the PostgREST spelling of the same set",
   D.FINISHED_FILTER == "status=in.(Done,Closed)", D.FINISHED_FILTER)

DISP_SRC = open(os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"),
                encoding="utf-8").read()
# The task queries the module BUILDS (not prose): none may pin one status.
built = re.findall(r'"dev_tasks\?[^"]*"|f"dev_tasks\?[^"]*"', DISP_SRC)
ok("no dev_tasks query the dispatcher builds pins status to a single value",
   not [q for q in built if "status=eq." in q], str(built))
ok("...the blocked_by lookup interpolates the shared filter",
   any("{FINISHED_FILTER}" in q for q in built), str(built))

# The API side mirrors it. Stub `db` before the router import (api.deps does
# `from db import …`), exactly as the #136 suite does.
db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: None
sys.modules.setdefault("db", db_stub)
from api.routers import dev_tasks as dt                # noqa: E402

ok("api.routers.dev_tasks mirrors the set",
   dt.FINISHED_STATUSES == ("Done", "Closed"), str(dt.FINISHED_STATUSES))
ok("..._is_finished() agrees with the dispatcher's predicate",
   dt._is_finished("Closed") and dt._is_finished("Done")
   and not dt._is_finished("Stand By"))
ok("Stand By joins the API's undispatchable stages",
   "Stand By" in dt._UNDISPATCHABLE_STAGES, str(dt._UNDISPATCHABLE_STAGES))

API_SRC = open(os.path.join(SRC, "api", "routers", "dev_tasks.py"),
               encoding="utf-8").read()
# Deliberate exception, called out in the #232 report: the two-touch close guard
# is a TRANSITION rule about specific columns (Review → Staged/Done), not a
# question of "is this task over", so it keeps the literal spelling on purpose.
guard = API_SRC[API_SRC.index("Two-touch close guard"):]
guard = guard[:guard.index("c.table(\"dev_tasks\").update(row)")]
ok("the two-touch close guard keeps its LITERAL Done — a transition rule, not "
   "a finished-check (and the exception is documented in place)",
   '("Staged", "Done")' in guard and "LITERAL `Done` ON PURPOSE" in API_SRC)

# ── the frontend half of the shared set ────────────────────────────────────
SHARED = open(os.path.join(VIEWS, "taskBoardShared.tsx"), encoding="utf-8").read()
BOARD_FILES = {
    "taskBoardShared.tsx": SHARED,
    "AdminTasksPage.tsx": open(os.path.join(VIEWS, "AdminTasksPage.tsx"),
                               encoding="utf-8").read(),
    "AdminRoadmapPage.tsx": open(os.path.join(VIEWS, "AdminRoadmapPage.tsx"),
                                 encoding="utf-8").read(),
    "AdminDispatchPage.tsx": open(os.path.join(VIEWS, "AdminDispatchPage.tsx"),
                                  encoding="utf-8").read(),
    "TaskDetailModal.tsx": open(os.path.join(VIEWS, "TaskDetailModal.tsx"),
                                encoding="utf-8").read(),
}

ok("taskBoardShared exports the shared set + predicate",
   "export const FINISHED_STATUSES = ['Done', 'Closed'];" in SHARED
   and "export const isFinished =" in SHARED)

# THE SWEEP RAIL — what makes status number three cheap. Comments are stripped
# first so the explanatory prose above each fix does not mask a real one.
CODE_ONLY = {name: re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)
             for name, src in BOARD_FILES.items()}
DONE_CMP = re.compile(r"status\s*(?:!==|===)\s*'Done'")
leftovers = {name: DONE_CMP.findall(code)
             for name, code in CODE_ONLY.items() if DONE_CMP.search(code)}
ok("NO hand-written `status === 'Done'` finished-check survives in the board "
   "UI — every one goes through isFinished()", not leftovers, str(leftovers))

for name, needle in (
        ("AdminTasksPage.tsx", "tasks.filter((t) => !isFinished(t.status)).length"),
        ("AdminDispatchPage.tsx", "!isFinished(t.status)"),
        ("AdminRoadmapPage.tsx", "!isFinished(v.status)"),
        ("TaskDetailModal.tsx", "isFinished(s.status)")):
    ok(f"{name} reads the shared predicate", needle in BOARD_FILES[name])

ok("openCount specifically excludes BOTH finished statuses (Kevin's own closed "
   "work must not sit in the open count forever)",
   "const openCount = tasks.filter((t) => !isFinished(t.status)).length"
   in BOARD_FILES["AdminTasksPage.tsx"])
ok("the blocked_by mirror behind the Run button's tooltip uses it too",
   "!isFinished(allTasks.find((x) => x.id === b)?.status)" in SHARED)


# ── RAIL 6 — both statuses render ──────────────────────────────────────────
print("― rail 6: both statuses render, with their own colour and description ―")

statuses = re.search(r"export const STATUSES = \[(.*?)\];", SHARED, re.S).group(1)
order = [s.strip().strip("'") for s in statuses.split(",")]
ok("STATUSES carries Stand By and Closed", "Stand By" in order and "Closed" in order,
   str(order))
ok("Closed comes AFTER Done — it is Kevin's step past M's close, not a rival to it",
   order.index("Closed") > order.index("Done"), str(order))
ok("Stand By sits beside Blocked, the other anywhere-exception",
   abs(order.index("Stand By") - order.index("Blocked")) == 1, str(order))
# Board #261 folded VISION_STATUSES into the TASK_TYPES map. The list moved; the
# #232 guarantee did not — EVERY container type still offers both statuses, and
# asserting it per type (rather than on one lifted list) means a fourth type
# cannot quietly drop them.
type_map = re.search(r"export const TASK_TYPES[^{]*\{(.*)\n\};", SHARED, re.S).group(1)
per_type = dict(re.findall(r"\n  (\w+): \{(.*?)\n  \},", type_map, re.S))
ok("the type map lists every type #261 declares",
   set(per_type) == {"action", "vision", "goal"}, str(sorted(per_type)))
ok("vision rows get them too",
   "'Stand By'" in per_type["vision"] and "'Closed'" in per_type["vision"])
ok("...and so does every other type (action inherits STATUSES itself)",
   all("'Stand By'" in v and "'Closed'" in v
       for k, v in per_type.items() if k != "action")
   and "statuses: STATUSES" in per_type["action"], str(sorted(per_type)))

defs = re.search(r"export const STATUS_DEF[^{]*\{(.*?)\n\};", SHARED, re.S).group(1)
ok("Stand By has a description that says it is NOT Blocked",
   "'Stand By':" in defs and "Not Blocked" in defs)
ok("Closed has a description that says it blocks nothing",
   "'Closed':" in defs and "Blocks NOTHING" in defs)

colors = re.search(r"export const STATUS_COLOR[^{]*\{(.*?)\n\};",
                   SHARED, re.S).group(1)
picked = dict(re.findall(r"'([^']+)': '([^']+)'", colors))
ok("both have a colour", "Stand By" in picked and "Closed" in picked, str(picked))
ok("Stand By's colour is DISTINCT from Blocked's — mistaking parked work for a "
   "real blocker is the exact failure the status exists to prevent",
   picked.get("Stand By") not in (picked.get("Blocked"), None),
   f"StandBy={picked.get('Stand By')} Blocked={picked.get('Blocked')}")
ok("Closed's colour is distinct from Done's as well",
   picked.get("Closed") not in (picked.get("Done"), None))


# ── RAIL 7 — the stamp: one mode, `approved to start`, no kevin_final ───────
print("― rail 7: ONE stamp, labelled 'approved to start' ―")

ok("the API keeps exactly one canonical mode (+ the legacy alias)",
   dt._STAMP_MODE == "start" and dt._STAMP_MODE_ALIASES == ("start", "delegate"),
   f"{dt._STAMP_MODE} {dt._STAMP_MODE_ALIASES}")
ok("...labelled for what it actually is: permission to execute",
   dt._STAMP_LABEL == "approved to start", dt._STAMP_LABEL)
stamp_src = API_SRC[API_SRC.index("def stamp_approval"):]
stamp_src = stamp_src[:stamp_src.index("\n@router")]
# Comments and the docstring stripped: they EXPLAIN the absence, and matching
# them would make the rail pass on prose.
stamp_code = re.sub(r"#[^\n]*", "", stamp_src)
stamp_code = re.sub(r'"""[\s\S]*?"""', "", stamp_code)
ok("the stamp NEVER writes kevin_final (board #232 — it is vestigial now)",
   "kevin_final" not in stamp_code, stamp_code)
ok("...and the retired 'final' mode is refused LOUDLY, not downgraded in silence",
   'mode == "final"' in stamp_src and "retired by board #232" in stamp_src)
# The column and its guard STAY. Ripping them out would mean rewriting two
# lifecycle suites to make room for a status change — out of proportion, and a
# separate follow-up.
ok("the kevin_final COLUMN is still editable and still validated (legacy rows "
   "must keep behaving exactly as stamped)",
   '"kevin_final"' in API_SRC and "kevin_final" in dt._EDITABLE)

ok("the UI ships ONE stamp button, and it says 'Approve to start'",
   SHARED.count("onStamp(t.id,") == 1 and "✅ Approve to start" in SHARED)
ok("...typed as a one-member union, so a second mode cannot be added by accident",
   "export type StampMode = 'start';" in SHARED)
ok("the legacy two-touch CHIP still renders for rows stamped before #232",
   "TwoTouchChip" in SHARED and "kevin_final" in SHARED)


# ── RAIL 8 — the SessionStart hook's open queue ────────────────────────────
print("― rail 8: the session-start board digest excludes Closed too ―")
HOOK = open(os.path.join(ROOT, "tools", "team_dispatcher", "hooks",
                         "team_board_context.py"), encoding="utf-8").read()
ok("the hook's open-task query excludes BOTH finished statuses",
   "status=not.in.(Done,Closed)" in HOOK and "status=neq.Done" not in HOOK)


print()
print(f"{'ALL PASS' if not FAIL else 'FAILURES'} "
      f"({PASS} checks{'' if not FAIL else f', {FAIL} failed'})")
sys.exit(1 if FAIL else 0)
