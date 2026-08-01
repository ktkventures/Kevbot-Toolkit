"""Acceptance test — board #289: per-goal assignee lanes (`G281`) in the UI.

A goal task carries its own session, and Kevin's ruling 08-01 is that the lane
is PER-GOAL — `G281`, not one shared `G`: *"the point of a goal is to have a
dedicated session fully focused on that goal."* #281 hit the gap within minutes
of the first goal being spawned: the session set its own `assignee` to `G` (the
API has no assignee vocabulary, so it worked) and NO dropdown could express it.

Two things this task is NOT, both asserted so they cannot creep in:

  * it is not an API change. The premise M filed it on — *"a goal task cannot be
    assigned to the session doing the work"* — was wrong, and rail 5 pins the
    correction: `dev_tasks.py` validates `origin` and `task_type` and never
    `assignee`. The data half already worked; this is the UI half only.

  * it is emphatically not a registry row. That omission is the SAFETY RAIL
    (rail 4): `headless_agents()` selects `status=eq.headless` and `triage_todo()`
    refuses any assignee absent from that dict as *waiting, not stuck* — which
    is the only thing stopping the loop from spawning goal sessions on its own.
    Kevin spawns goals. Rail 4 fails the moment someone "helpfully" adds one.

Rails, each of which FAILS without its fix:

  RAIL 1  THE DERIVATION      — `openGoalLanes` EXECUTED: one `G<id>` per
                                UNFINISHED goal, id-sorted. A fixed array cannot
                                express an unbounded vocabulary, so the lanes are
                                derived from the board's own goal rows. Finished
                                goals drop out; vision/action rows never produce
                                a lane at all.
  RAIL 2  THE PREFIX COLOUR   — `roleColor` EXECUTED returns ONE colour for every
                                `G<n>`, via a prefix rule rather than N map
                                entries. A BARE `G` is NOT a lane — that is
                                Kevin's per-goal ruling made executable — and
                                neither is `g281`, `G-A` or `Goal`. Every
                                existing role colour is byte-unchanged.
  RAIL 3  SELECTABLE          — `rolesWithGoalLanes` EXECUTED appends without
                                duplicating and preserves registry order; and in
                                SOURCE, every assignee/owner picker on both board
                                pages reads that list, not the raw registry one.
                                A helper nothing calls is the #219 failure shape.
  RAIL 4  NO REGISTRY ROW,    — the real `triage_todo()` is EXECUTED against a
          AND THE LOOP STILL    task assigned `G281`: it must be SKIPPED, with
          REFUSES               is_stuck=False, through the same code path that
                                refuses `kevin`. No `G` letter check was added to
                                the dispatcher, no `G` in STUB_AGENTS, no
                                migration inserts one, and no `G` was hard-coded
                                into the frontend's fixed role lists.
  RAIL 5  THE MIRRORS         — `dev_tasks.py` has no assignee vocabulary to
                                mirror (proved POSITIVELY: the file validates
                                other fields, so its silence on `assignee` is a
                                finding and not an absence of machinery), and the
                                dispatcher gained no role knowledge. Mirror drift
                                cost hours on #219; this is the check that would
                                have caught it.
  RAIL 6  NOT MENTIONABLE     — the API's REAL mention parser is imported and
                                called: `@G281` produces no mention row, because
                                only registry letters do. So the UI must not
                                OFFER it — a token that posts and silently
                                reaches nobody is worse than no token.
  RAIL 7  RENDERS HONESTLY    — `isSessionLane('G281', …)` EXECUTED is true even
                                against a fully-populated registry, so the chip
                                can never claim *"the dispatcher can auto-run
                                this lane"* about a goal. And the avatar grows to
                                fit four characters instead of truncating: two
                                lanes that both render "G…" defeat the ruling.

The frontend derivations are EXECUTED in node, never grepped — board #245's
lesson is that a string-compare "mirror" is not a mirror. Offline + hermetic: no
DB, no network.

Run:  .venv/bin/python src/test_goal_assignee_lanes_289.py   (from RoR_Trader root)
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
VIEWS = os.path.join(ROOT, "frontend", "src", "views")
DISP = os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py")
MIGRATIONS = os.path.join(SRC, "migrations")
sys.path.insert(0, SRC)

PASS = 0
FAILED = []


def ok(label, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL: {label}{(' — ' + str(detail)) if detail else ''}")


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return fh.read()


SHARED = read(VIEWS, "taskBoardShared.tsx")
TASKS_PAGE = read(VIEWS, "AdminTasksPage.tsx")
KEVIN_PAGE = read(VIEWS, "AdminKevinPage.tsx")
MODAL = read(VIEWS, "TaskDetailModal.tsx")
DISPATCHER_SRC = read(DISP)
API_SRC = read(SRC, "api", "routers", "dev_tasks.py")

NODE = shutil.which("node")


# ── lifting the real TS (same technique as test_goal_task_type_262.py) ──────
def lift_literal(src, name):
    """`export const NAME[: type] = <literal>;` → runnable JS, brace-matched."""
    m = re.search(rf"^export const {name}(?::[^=]+)? = ", src, re.M)
    assert m, f"{name} not found in the source"
    i = m.end()
    depth, start = 0, i
    while True:
        ch = src[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                i += 1
                break
        elif ch == ";" and depth == 0:
            break
        i += 1
    return f"const {name} = {src[start:i]};\n"


def lift_expr(src, name, exported=True):
    """`[export ]const NAME = <expression>;` → runnable JS. For a value with no
    leading brace, which `lift_literal` cannot brace-match. `exported=False`
    lifts a module-private const (SESSION_FALLBACK)."""
    prefix = "export " if exported else ""
    m = re.search(rf"^{prefix}const {name}(?::[^=]+)? = ", src, re.M)
    assert m, f"{name} not found in the source"
    i, depth, start = m.end(), 0, m.end()
    while True:
        ch = src[i]
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == ";" and depth == 0:
            break
        i += 1
    return f"const {name} = {src[start:i]};\n"


def lift_arrow(src, name):
    """`export const NAME = (args): T => …;` → runnable JS with types stripped.

    The 262/261 lifter, with ONE addition: an OPTIONAL parameter (`status?:
    string | null`) is de-typed rather than crashing the lifter. `isFinished`
    and `isGoalLane` both have one, so without it this suite could not execute
    the very derivations it exists to execute."""
    m = re.search(rf"^export const {name} = \(", src, re.M)
    assert m, f"{name} not found in the source"
    i, depth = m.end() - 1, 0
    body = []
    while True:
        ch = src[i]
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == ";" and depth == 0:
            break
        body.append(ch)
        i += 1
    text = "".join(body)
    sig_end = text.index("=>")
    sig, rest = text[:sig_end], text[sig_end:]
    params = sig[sig.index("(") + 1:sig.rindex(")")]
    out = []
    for part in re.split(r",(?![^<(\[]*[>)\]])", params):
        part = part.strip()
        if not part:
            continue
        mdef = re.match(r"(\w+)\s*\??\s*(?::[^=]+)?(=\s*.+)?$", part, re.S)
        assert mdef, f"could not de-type parameter {part!r} of {name}"
        out.append((mdef.group(1) + " " + (mdef.group(2) or "")).strip())
    body_js = re.sub(r"\s+as\s+[\w\[\]]+(?:\s*\|\s*[\w\[\]]+)*", "", rest)
    return f"const {name} = ({', '.join(out)}) {body_js};\n"


def run_js(program):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(program)
        path = fh.name
    try:
        out = subprocess.run([NODE, path], capture_output=True, text=True,
                             timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:1500]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


# Declaration order matters: every lifted piece must exist before one that
# references it. This IS the dependency graph of the derivation under test.
PRELUDE = (
    lift_literal(SHARED, "STATUSES")
    + lift_literal(SHARED, "FINISHED_STATUSES")
    + lift_arrow(SHARED, "isFinished")
    + lift_literal(SHARED, "TASK_TYPES")
    + lift_arrow(SHARED, "taskTypeOf")
    + lift_expr(SHARED, "GOAL_LANE_PREFIX")
    + lift_arrow(SHARED, "goalLaneFor")
    + lift_arrow(SHARED, "isGoalLane")
    + lift_arrow(SHARED, "openGoalLanes")
    + lift_arrow(SHARED, "rolesWithGoalLanes")
    + lift_literal(SHARED, "ROLE_COLOR")
    + lift_expr(SHARED, "GOAL_LANE_COLOR")
    + lift_arrow(SHARED, "roleColor")
    + lift_arrow(SHARED, "roleAbbrev")
    + lift_expr(SHARED, "SESSION_FALLBACK", exported=False)
    + lift_arrow(SHARED, "isSessionLane")
    + lift_literal(SHARED, "LANE_KIND_TIP")
    + lift_literal(SHARED, "ASSIGNEES")
    + lift_literal(SHARED, "MENTION_ROLES")
)


def js(expr_block):
    return run_js(PRELUDE + expr_block)


print("\n=== RAIL 1 — THE DERIVATION (executed) ===")

BOARD = """
const board = [
  {id: 281, task_type: 'goal',   status: 'In Progress', tags: []},
  {id: 12,  task_type: 'goal',   status: 'Scoping',     tags: []},
  {id: 99,  task_type: 'goal',   status: 'Done',        tags: []},
  {id: 98,  task_type: 'goal',   status: 'Closed',      tags: []},
  {id: 40,  task_type: 'vision', status: 'In Progress', tags: []},
  {id: 41,  task_type: 'action', status: 'Todo',        tags: []},
  {id: 42,  task_type: '',       status: 'Todo',        tags: ['vision']},
];
"""

r1 = js(BOARD + """
console.log(JSON.stringify({
  lanes: openGoalLanes(board),
  empty: openGoalLanes([]),
  nullish: openGoalLanes(null),
  laneFor: goalLaneFor({id: 281}),
  onlyGoals: openGoalLanes(board.filter((t) => t.task_type !== 'goal')),
}));
""")

ok("one lane per UNFINISHED goal, `G<id>`, lowest id first",
   r1["lanes"] == ["G12", "G281"], r1["lanes"])
ok("a FINISHED goal drops out — Done and Closed both (#232's shared set)",
   "G99" not in r1["lanes"] and "G98" not in r1["lanes"], r1["lanes"])
ok("vision / action / tag-derived-container rows never produce a lane",
   r1["onlyGoals"] == [], r1["onlyGoals"])
ok("an empty or missing board derives no lanes (never a crash)",
   r1["empty"] == [] and r1["nullish"] == [], r1)
ok("goalLaneFor is the single spelling of the lane name",
   r1["laneFor"] == "G281", r1["laneFor"])

print("\n=== RAIL 2 — THE PREFIX COLOUR RULE (executed) ===")

r2 = js("""
const roles = ['M','M-A','E','E2','F','P','R','R-A','kevin','claude','system'];
console.log(JSON.stringify({
  goals:   ['G1','G12','G281','G99999'].map(roleColor),
  goalCol: GOAL_LANE_COLOR,
  fixed:   Object.fromEntries(roles.map((r) => [r, roleColor(r)])),
  unknown: roleColor('Z'),
  bareG:   roleColor('G'),
  isLane:  Object.fromEntries(
    ['G281','G1','G','g281','G-A','Goal','GA','','281']
      .map((r) => [r || '(empty)', isGoalLane(r)])),
  abbrev:  roleAbbrev('G281'),
}));
""")

ok("EVERY per-goal lane takes the one goal colour, by prefix",
   set(r2["goals"]) == {r2["goalCol"]} and len(r2["goals"]) == 4, r2["goals"])
ok("the goal colour is distinct from every fixed role colour",
   r2["goalCol"] not in r2["fixed"].values(), r2["goalCol"])
ok("existing role colours are byte-unchanged by the new rule",
   r2["fixed"] == {"M": "#7c5cff", "M-A": "#a48cff", "E": "#d9534f",
                   "E2": "#e08a3c", "F": "#3b82f6", "P": "#2e9e5b",
                   "R": "#2aa8a0", "R-A": "#5fc9c2", "kevin": "#c9a227",
                   "claude": "#64748b", "system": "#556070"}, r2["fixed"])
ok("an unknown role still falls back to slate",
   r2["unknown"] == "#64748b", r2["unknown"])
ok("a BARE `G` is NOT a lane — Kevin's per-goal ruling, executable",
   r2["isLane"]["G"] is False and r2["bareG"] == "#64748b", r2["bareG"])
ok("`g281` / `G-A` / `Goal` / `GA` / `281` are not lanes either",
   [r2["isLane"][k] for k in ("g281", "G-A", "Goal", "GA", "281")]
   == [False] * 5, r2["isLane"])
ok("`G281` and `G1` ARE lanes",
   r2["isLane"]["G281"] is True and r2["isLane"]["G1"] is True, r2["isLane"])
ok("the chip label keeps the goal id — the digits ARE the identity",
   r2["abbrev"] == "G281", r2["abbrev"])

print("\n=== RAIL 3 — SELECTABLE (executed + wired) ===")

r3 = js(BOARD + """
const registry = ['', 'M', 'M-A', 'E', 'F', 'kevin'];
const once = rolesWithGoalLanes(registry, board);
console.log(JSON.stringify({
  once,
  idempotent: rolesWithGoalLanes(once, board),
  noGoals:    rolesWithGoalLanes(registry, []),
  sameRef:    rolesWithGoalLanes(registry, []) === registry,
}));
""")

ok("registry lanes keep their order, goal lanes append after",
   r3["once"] == ["", "M", "M-A", "E", "F", "kevin", "G12", "G281"], r3["once"])
ok("applying it twice adds nothing — no duplicate options",
   r3["idempotent"] == r3["once"], r3["idempotent"])
ok("a board with no open goal changes the list not at all",
   r3["noGoals"] == ["", "M", "M-A", "E", "F", "kevin"] and r3["sameRef"],
   r3)

# WIRED — a derivation nothing calls is exactly the #219 shape (a constant
# shipped without its mirror). Each of these is a distinct assignee/owner
# surface, and every one of them has to read the widened list.
ok("AdminTasksPage builds roleOptions from rolesWithGoalLanes",
   "rolesWithGoalLanes(roles, tasks)" in TASKS_PAGE
   and "rolesWithGoalLanes" in TASKS_PAGE.split("} from './taskBoardShared'")[0])
ok("AdminKevinPage builds roleOptions from rolesWithGoalLanes",
   "rolesWithGoalLanes(roles, tasks)" in KEVIN_PAGE
   and "rolesWithGoalLanes" in KEVIN_PAGE.split("} from './taskBoardShared'")[0])
ok("the board table's assignee select offers goal lanes",
   "withLegacy(roleOptions, t.assignee || '')" in TASKS_PAGE)
ok("the new-task assignee select offers goal lanes",
   "{roleOptions.map((a) => <option key={a} value={a}>{a || '—'}</option>)}"
   in TASKS_PAGE)
ok("the assignee FILTER offers goal lanes (a lane you cannot filter to is "
   "half a lane)",
   "const known = roleOptions.filter(Boolean);" in TASKS_PAGE)
ok("both pages hand the widened list to TaskDetailModal (assignee picker, "
   "step owners, comment author)",
   TASKS_PAGE.count("roles={roleOptions}") == 1
   and KEVIN_PAGE.count("roles={roleOptions}") == 1)
ok("Kevin's dashboard hand-off picker offers goal lanes",
   "options={roleOptions.filter(Boolean)}" in KEVIN_PAGE)
_stale = [ln.strip() for ln in (TASKS_PAGE + KEVIN_PAGE).splitlines()
          if re.search(r"\broles(\.filter\(Boolean\)|\.map)|roles=\{roles\}"
                       r"|withLegacy\(roles,", ln)]
ok("no assignee surface still reads the un-widened `roles`",
   not _stale, _stale)

print("\n=== RAIL 4 — NO REGISTRY ROW, AND THE LOOP STILL REFUSES ===")

spec = importlib.util.spec_from_file_location("dispatcher_under_test_289", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["dispatcher_under_test_289"] = D
spec.loader.exec_module(D)

# The registry as it must remain: headless lanes only, no `G*` anywhere.
AGENTS = {
    "M-A": {"letter": "M-A", "status": "headless", "worktree": ".",
            "scope": "s", "boundaries": "b"},
    "F": {"letter": "F", "status": "headless", "worktree": ".",
          "scope": "s", "boundaries": "b"},
}

_OR_RE = re.compile(r"or=\(([^)]*)\)")
_EQ_RE = re.compile(r"(\w+)=eq\.([^&]+)")


def _matches(row, clause):
    col, _, val = clause.partition(".eq.")
    if val == "true":
        return row.get(col) is True
    if val == "false":
        return row.get(col) is False
    return str(row.get(col)) == val


def board_api(rows):
    def _api(method, path, body=None, prefer=None):
        if not path.startswith("dev_tasks?"):
            return []
        q = path.split("?", 1)[1]
        m = _OR_RE.search(q)
        if m:
            return [r for r in rows
                    if any(_matches(r, c) for c in m.group(1).split(","))]
        m = _EQ_RE.search(q)
        if m:
            return [r for r in rows if str(r.get(m.group(1))) == m.group(2)]
        raise AssertionError(f"unmodelled dev_tasks filter: {q}")
    return _api


def task(**kw):
    t = {"id": 1, "title": "T", "description": "a real description",
         "assignee": "F", "status": "Todo", "ai_eligible": True,
         "tags": [], "blocked_by": [], "checklist": []}
    t.update(kw)
    return t


def triage(rows):
    D.api = board_api(rows)
    out, skipped = D.triage_todo(AGENTS, set(), {"runs": []})
    return [t["id"] for t in out], skipped


def verdict(assignee, status="Todo"):
    """(dispatched?, reason, is_stuck) for one maximally-eligible task."""
    rows = [task(assignee=assignee, status=status)]
    ids, skipped = triage(rows)
    if ids:
        return True, "DISPATCHED", None
    for t, reason, stuck in skipped:
        if t["id"] == 1:
            return False, reason, stuck
    return False, "vanished", None

# A goal lane must behave EXACTLY as `kevin` does — same code path, same
# classification. If these two ever differ, someone taught the gate about G.
g_disp, g_reason, g_stuck = verdict("G281")
k_disp, k_reason, k_stuck = verdict("kevin")
f_disp, _, _ = verdict("F")

ok("a Todo + ai_eligible task assigned `G281` is NOT dispatched",
   g_disp is False, g_reason)
ok("...and it is classified WAITING, not stuck (no 4h staleness nag)",
   g_stuck is False, g_stuck)
ok("...for the same reason `kevin` is refused — one code path, no letter check",
   g_reason == "waiting on G281 (not a headless agent)"
   and k_reason == "waiting on kevin (not a headless agent)"
   and g_stuck == k_stuck, (g_reason, k_reason))
ok("...from EVERY live status, not just Todo",
   all(verdict("G281", s)[0] is False
       for s in ("Backlog", "Scoping", "Approval", "Todo", "In Progress",
                 "Review")), "a G lane dispatched from some status")
ok("a real headless lane still dispatches — the gate was not broken to get here",
   f_disp is True)

ok("⛔ no `G*` row in the dispatcher's fallback roster",
   not [k for k in D.STUB_AGENTS if re.fullmatch(r"G\d*", k)],
   list(D.STUB_AGENTS))
ok("⛔ no migration inserts a `G*` agent",
   not [f for f in os.listdir(MIGRATIONS)
        if f.endswith(".sql") and re.search(r"'G\d*'", read(MIGRATIONS, f))
        and "agents" in read(MIGRATIONS, f)],
   "a migration adds a G agent")
ok("the dispatcher gained no goal knowledge at all",
   not re.search(r"\bG\d+\b|goal_lane|goalLane|GOAL_LANE", DISPATCHER_SRC),
   "dispatcher.py mentions a goal lane")
_fixed_roles = js("console.log(JSON.stringify("
                  "ASSIGNEES.concat(MENTION_ROLES, Object.keys(ROLE_COLOR))));")
ok("⛔ no `G*` hard-coded into the frontend's FIXED role lists",
   not [r for r in _fixed_roles if re.fullmatch(r"G\d*", r)],
   "a G leaked into ASSIGNEES / MENTION_ROLES / ROLE_COLOR")

print("\n=== RAIL 5 — THE MIRRORS (#219: a constant without its mirror) ===")

# POSITIVE control first: the file DOES validate other fields, so its silence on
# `assignee` is a finding about assignee — not "this file validates nothing".
ok("dev_tasks.py validates `origin` (the positive control)",
   "_ORIGINS = {" in API_SRC and "_ORIGINS" in API_SRC.replace("_ORIGINS = {", "", 1))
ok("dev_tasks.py accepts `assignee` as an editable field",
   '"assignee"' in API_SRC.split("_EDITABLE = {")[1].split("}")[0])
ok("dev_tasks.py has NO assignee vocabulary — nothing to mirror, which is why "
   "#281's goal session could already set itself to `G`",
   not re.search(r"_ASSIGNEES\s*=|ASSIGNEE_VALUES\s*=|assignee\s+not\s+in\s+",
                 API_SRC), "an assignee enum appeared in the API")
ok("the shared module still declares the ONE role list the pages fall back to",
   "export const ASSIGNEES = ['', 'M', 'M-A', 'E', 'E2', 'F', 'P', 'R', "
   "'R-A', 'kevin'];" in SHARED)

print("\n=== RAIL 6 — ASSIGNABLE, BUT NOT MENTIONABLE ===")

from api.routers.dev_tasks import _parse_mentions  # noqa: E402

# The registry as `_write_mentions` builds it: real letters + kevin. No G.
VALID = {"m": "M", "m-a": "M-A", "e": "E", "f": "F", "kevin": "kevin"}
ok("the REAL parser drops `@G281` — a goal lane has no registry row, so the "
   "mention would post and reach nobody",
   _parse_mentions("@G281 please look, @F", VALID) == ["F"],
   _parse_mentions("@G281 please look, @F", VALID))
ok("...so the UI does not OFFER it: the mention picker filters goal lanes out",
   "roles.filter((r) => r && !isGoalLane(r))" in MODAL)
ok("...while the comment-AUTHOR picker keeps them (a goal session signs its "
   "own comments)",
   "withLegacy(roles.filter(Boolean), commentAuthor)" in MODAL)

print("\n=== RAIL 7 — THE CHIP TELLS THE TRUTH ===")

r7 = js("""
const live = new Map([
  ['M',   {letter:'M',   status:'live-session'}],
  ['M-A', {letter:'M-A', status:'headless'}],
  ['F',   {letter:'F',   status:'headless'}],
  ['kevin', {letter:'kevin', status:'live-session', kind:'human'}],
]);
console.log(JSON.stringify({
  goalIsSession: isSessionLane('G281', live),
  goalOffline:   isSessionLane('G281', new Map()),
  agentUnchanged: isSessionLane('F', live),
  sessionUnchanged: isSessionLane('M', live),
  humanUnchanged: isSessionLane('kevin', live),
  tips: Object.keys(LANE_KIND_TIP),
  goalTip: LANE_KIND_TIP.goal,
}));
""")

ok("a goal lane is NEVER rendered as auto-dispatchable — even against a fully "
   "populated registry that has no row for it",
   r7["goalIsSession"] is True and r7["goalOffline"] is True, r7)
ok("registry-backed lanes are unaffected by the carve-out",
   r7["agentUnchanged"] is False and r7["sessionUnchanged"] is True
   and r7["humanUnchanged"] is True, r7)
ok("the goal lane gets its OWN tooltip — the other two claim a registry status "
   "it deliberately does not have",
   "goal" in r7["tips"] and "no registry row" in r7["goalTip"], r7["goalTip"])
ok("...and it says who spawns a goal, since that is the whole rail",
   "Kevin spawns a goal" in r7["goalTip"], r7["goalTip"])
ok("the avatar GROWS for a four-character lane instead of truncating",
   "width: goal ? 'auto' : size, minWidth: size" in SHARED
   and "padding: goal ?" in SHARED)
ok("the lane-kind chip labels it a goal session",
   "'⚑ goal session'" in SHARED)

print(f"\n{'=' * 62}")
if FAILED:
    print(f"FAILED ({len(FAILED)}):")
    for f in FAILED:
        print(f"  - {f}")
    print(f"{PASS} passed, {len(FAILED)} FAILED")
    sys.exit(1)
print(f"ALL {PASS} CHECKS PASSED — board #289 per-goal assignee lanes")
