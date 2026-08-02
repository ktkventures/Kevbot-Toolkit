"""Acceptance test — board #297: the help manual is GENERATED, not written.

Kevin's ask was a help button on /admin/tasks opening a modal: nav on the left,
detail on the right. The build is trivial; the thing that decides whether it is
worth having is whether it can go stale.

The rules governing this board already live in five places — the code model, a
381-line charter, ten skills, M's memory files, and task descriptions. A
hand-typed modal is a SIXTH, and a sixth copy drifts within a week, at which
point it is worse than no manual at all: it looks authoritative, it is wrong,
and Kevin has said outright he wants the agents to trust it.

So the whole of this suite is one claim, made unfakeable:

    every row the manual shows is DERIVED from taskBoardShared's model,
    and a future hand-edit that hardcodes one turns this file red.

Rails, each of which FAILS without its fix:

  RAIL 1  IT OPENS            — the button exists on the tasks page, at the TOP
                                (Kevin's words), it is wired to state, and the
                                modal is rendered from that state. A component
                                nothing mounts is the #219 failure shape.
  RAIL 2  TYPES ARE DERIVED   — `helpTypeRows` EXECUTED against the REAL
                                `TASK_TYPES`: one row per type in declaration
                                order, each carrying that type's own `label` /
                                `icon` / `def` / `statuses` / `children` /
                                `session` BY VALUE. A fourth type added to the
                                map appears with no edit here — proved by
                                executing the builder against a mutated map.
  RAIL 3  STATUSES ARE DERIVED— `helpStatusRows` EXECUTED: one row per status,
                                pipeline order, each with the model's OWN
                                `STATUS_DEF` string, and `types` COMPUTED by
                                asking each type whether it lists that status.
                                Membership is a computation, never a table.
  RAIL 4  ASSIGNEES ARE DERIVED— `helpAssigneeRows` EXECUTED: the fixed lanes
                                come from `ASSIGNEES`, and the per-goal `G*`
                                lanes from `openGoalLanes` over the board's real
                                rows — the unbounded half a fixed array
                                structurally cannot hold (board #289). Lane kind
                                is read through `laneKindLabel`, i.e. from the
                                registry, never from the letter.
  RAIL 5  NO SECOND COPY      — THE ANTI-DRIFT RAIL, and the reason the task
                                exists. No model value may appear in the modal
                                as a standalone string literal, and no `def` /
                                `STATUS_DEF` sentence may appear in it at all.
                                Retyping one line of the model into the JSX
                                fails here.
  RAIL 6  THE WARTS ARE STATED— the four behaviours Kevin must see before he
                                redesigns against them are present, each with a
                                board ref AND a file+symbol citation. Every
                                cited file is OPENED and the named symbol found,
                                so a rename breaks the test instead of quietly
                                leaving a lie in a modal.
  RAIL 7  SCOPE HELD          — this pass documents only what renders itself.
                                Process chains, stamps and the release/train
                                lifecycle are NOT written up here (they are
                                prose, and prose is the sixth copy); nothing in
                                the modal writes, and the written sources it
                                points at exist on disk.

The derivations are EXECUTED in node, never grepped — board #245's lesson is
that a string-compare "mirror" is not a mirror. The lifters are the ones
test_goal_assignee_lanes_289.py established. Offline + hermetic: no DB, no
network.

Run:  .venv/bin/python src/test_task_help_modal_297.py   (from RoR_Trader root)
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
VIEWS = os.path.join(ROOT, "frontend", "src", "views")

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
HELP = read(VIEWS, "TaskHelpModal.tsx")
TASKS_PAGE = read(VIEWS, "AdminTasksPage.tsx")

NODE = shutil.which("node")


# ── lifting the real TS (the 289/262/261 technique, verbatim) ──────────────
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
    """`[export ]const NAME = <expression>;` → runnable JS."""
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
    # `Object.keys(TASK_TYPES) as TaskType[]` — the 289 lifter never met a cast
    # outside an arrow body, and TASK_TYPE_VALUES (the type vocabulary IN ORDER)
    # is exactly one, so the de-typing `lift_arrow` already does moves here too.
    body = re.sub(r"\s+as\s+[\w\[\]]+(?:\s*\|\s*[\w\[\]]+)*", "", src[start:i])
    return f"const {name} = {body};\n"


def lift_arrow(src, name):
    """`export const NAME = (args): T => …;` → runnable JS, types stripped."""
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
# references it. This IS the dependency graph of the manual's derivation — and
# the fact that it is entirely taskBoardShared symbols, with nothing lifted out
# of TaskHelpModal but the three builders themselves, is the point.
PRELUDE = (
    lift_literal(SHARED, "STATUSES")
    + lift_literal(SHARED, "FINISHED_STATUSES")
    + lift_arrow(SHARED, "isFinished")
    + lift_literal(SHARED, "TASK_TYPES")
    + lift_expr(SHARED, "TASK_TYPE_VALUES")
    + lift_expr(SHARED, "DEFAULT_TASK_TYPE")
    + lift_arrow(SHARED, "taskTypeOf")
    + lift_expr(SHARED, "GOAL_LANE_PREFIX")
    + lift_arrow(SHARED, "goalLaneFor")
    + lift_arrow(SHARED, "isGoalLane")
    + lift_arrow(SHARED, "openGoalLanes")
    + lift_literal(SHARED, "ROLE_COLOR")
    + lift_expr(SHARED, "GOAL_LANE_COLOR")
    + lift_arrow(SHARED, "roleColor")
    + lift_arrow(SHARED, "roleAbbrev")
    + lift_expr(SHARED, "SESSION_FALLBACK", exported=False)
    + lift_arrow(SHARED, "isSessionLane")
    + lift_arrow(SHARED, "laneKindLabel")
    + lift_literal(SHARED, "LANE_KIND_TIP")
    + lift_literal(SHARED, "STATUS_DEF")
    + lift_literal(SHARED, "STATUS_COLOR")
    + lift_literal(SHARED, "ASSIGNEES")
    # …and the three builders under test, lifted out of the modal itself.
    + lift_arrow(HELP, "helpTypeRows")
    + lift_arrow(HELP, "helpStatusRows")
    + lift_arrow(HELP, "helpAssigneeRow")
    + lift_arrow(HELP, "helpAssigneeRows")
)


def js(expr_block, prelude=PRELUDE):
    return run_js(prelude + expr_block)


# The model, parsed once on the PYTHON side, so every expectation below is
# compared against the real map rather than against a copy in this file.
MODEL = js("console.log(JSON.stringify({"
           "types: TASK_TYPES, order: TASK_TYPE_VALUES, statuses: STATUSES,"
           "defs: STATUS_DEF, assignees: ASSIGNEES, finished: FINISHED_STATUSES"
           "}));")

print("\n=== RAIL 1 — IT OPENS, FROM THE TOP OF THE PAGE ===")

ok("the tasks page imports the help modal",
   "import TaskHelpModal from './TaskHelpModal'" in TASKS_PAGE)
ok("a Help button exists and is wired to open it",
   re.search(r"Help</button>", TASKS_PAGE) is not None
   and "setHelpOpen(true)" in TASKS_PAGE)
ok("the modal is actually MOUNTED from that state (not merely imported)",
   re.search(r"helpOpen && <TaskHelpModal", TASKS_PAGE) is not None)
ok("it is fed the live task list, so goal lanes are the board's real ones",
   re.search(r"<TaskHelpModal tasks=\{tasks\}", TASKS_PAGE) is not None)

# "at the top" = above the board, and specifically in the page's h1 row. Anchor
# on the `<h1` TAG, not on the page title text — that string appears first in
# the file's own docstring, which is not a row anything renders.
H1 = TASKS_PAGE.find("<h1")
ok("the button sits in the page's title row, at the TOP (Kevin's ask)",
   -1 < H1 < TASKS_PAGE.find("Help</button>") < H1 + 700,
   (H1, TASKS_PAGE.find("Help</button>")))
ok("it opens BEFORE the board table renders, not below it",
   TASKS_PAGE.find("Help</button>") < TASKS_PAGE.find("<tbody>"))

# Left nav / right detail — Kevin's shape, and four topics reachable.
ok("left nav + right detail, both present in the shell",
   "<nav" in HELP and "setTopic(" in HELP)
for t in ("types", "statuses", "assignees", "sources"):
    ok(f"topic `{t}` has a panel bound to it",
       re.search(rf"topic === '{t}' && <", HELP) is not None)
ok("Escape closes it (a reference surface must never trap you)",
   "'Escape'" in HELP and "onClose()" in HELP)

print("\n=== RAIL 2 — TASK TYPES ARE DERIVED (executed) ===")

r2 = js("console.log(JSON.stringify({rows: helpTypeRows()}));")["rows"]

ok("one row per type, in the model's declaration order",
   [r["type"] for r in r2] == MODEL["order"], [r["type"] for r in r2])
ok("every row carries its type's OWN label/icon/def, by value",
   all(r["label"] == MODEL["types"][r["type"]]["label"]
       and r["icon"] == MODEL["types"][r["type"]]["icon"]
       and r["def"] == MODEL["types"][r["type"]]["def"] for r in r2))
ok("every row carries its type's OWN status set, by value",
   all(r["statuses"] == MODEL["types"][r["type"]]["statuses"] for r in r2))
ok("children / session are read from the map, not asserted per type",
   all(r["children"] == MODEL["types"][r["type"]]["children"]
       and r["session"] == MODEL["types"][r["type"]]["session"] for r in r2))
ok("`missing` is a SUBTRACTION from the pipeline, never a second list",
   all(sorted(r["missing"])
       == sorted(s for s in MODEL["statuses"] if s not in r["statuses"])
       for r in r2))
ok("exactly one type is flagged as the default, and it is the model's",
   [r["type"] for r in r2 if r["isDefault"]]
   == [js("console.log(JSON.stringify(DEFAULT_TASK_TYPE));")])

# THE claim: a type added to the map appears with no edit to the modal.
GHOST = PRELUDE.replace(
    "const TASK_TYPE_VALUES =",
    "TASK_TYPES.spike = {label: 'Spike', icon: '※', children: false,"
    " session: false, statuses: ['Backlog'], def: 'spike — a timeboxed probe'};"
    "\nconst TASK_TYPE_VALUES =", 1)
r2b = js("console.log(JSON.stringify(helpTypeRows()));", prelude=GHOST)
ok("a FOURTH type added to the map renders here with NO edit to the modal",
   [r["type"] for r in r2b] == MODEL["order"] + ["spike"]
   and r2b[-1]["def"] == "spike — a timeboxed probe", [r["type"] for r in r2b])

print("\n=== RAIL 3 — STATUSES ARE DERIVED (executed) ===")

r3 = js("console.log(JSON.stringify(helpStatusRows()));")

ok("one row per status, in the model's pipeline order",
   [r["status"] for r in r3] == MODEL["statuses"], [r["status"] for r in r3])
ok("each definition is the model's OWN STATUS_DEF string, byte-for-byte",
   all(r["def"] == MODEL["defs"].get(r["status"], "") for r in r3))
ok("every status carries a definition (none silently blank)",
   all(r["def"] for r in r3), [r["status"] for r in r3 if not r["def"]])
ok("`types` is COMPUTED by asking each type, not tabulated",
   all(r["types"] == [t for t in MODEL["order"]
                      if r["status"] in MODEL["types"][t]["statuses"]]
       for r in r3))
ok("the container types really are narrower — a status they exclude proves it",
   any(r["types"] == ["action"] for r in r3),
   [r["status"] for r in r3 if r["types"] == ["action"]])
ok("`finished` comes from FINISHED_STATUSES, not from a hand-marked flag",
   [r["status"] for r in r3 if r["finished"]] == MODEL["finished"])

# A status only a container names must widen the union — proved, not promised.
WIDE = PRELUDE.replace(
    "const TASK_TYPE_VALUES =",
    "TASK_TYPES.vision.statuses = TASK_TYPES.vision.statuses.concat(['Parked']);"
    "\nconst TASK_TYPE_VALUES =", 1)
r3b = js("console.log(JSON.stringify(helpStatusRows()));", prelude=WIDE)
ok("a status only a CONTAINER type names still gets a row (union, not copy)",
   any(r["status"] == "Parked" and r["types"] == ["vision"] for r in r3b))

print("\n=== RAIL 4 — ASSIGNEES ARE DERIVED (executed) ===")

BOARD = """
const board = [
  {id: 297, task_type: 'action', status: 'In Progress', tags: []},
  {id: 281, task_type: 'goal',   status: 'In Progress', tags: []},
  {id: 12,  task_type: 'goal',   status: 'Scoping',     tags: []},
  {id: 99,  task_type: 'goal',   status: 'Done',        tags: []},
  {id: 40,  task_type: 'vision', status: 'In Progress', tags: []},
];
const reg = new Map([
  ['M', {letter: 'M', status: 'live-session'}],
  ['F', {letter: 'F', status: 'headless'}],
  ['kevin', {letter: 'kevin', status: 'live-session', kind: 'human'}],
]);
"""
r4 = js(BOARD + "console.log(JSON.stringify({"
        "rows: helpAssigneeRows(reg, board),"
        "none: helpAssigneeRows(reg, []),"
        "nullish: helpAssigneeRows(reg, null)}));")
rows, fixed = r4["rows"], [r for r in r4["rows"] if not r["derived"]]
derived = [r for r in r4["rows"] if r["derived"]]

ok("the fixed lanes are exactly ASSIGNEES minus the empty option",
   [r["role"] for r in fixed] == [a for a in MODEL["assignees"] if a],
   [r["role"] for r in fixed])
ok("the per-goal lanes are DERIVED from the board's unfinished goal rows",
   [r["role"] for r in derived] == ["G12", "G281"],
   [r["role"] for r in derived])
ok("a FINISHED goal contributes no lane; vision/action rows never do",
   all(r["role"] not in ("G99", "G40", "G297") for r in derived))
ok("an empty or missing board yields the fixed lanes and nothing else",
   [r["role"] for r in r4["none"]] == [r["role"] for r in fixed]
   and [r["role"] for r in r4["nullish"]] == [r["role"] for r in fixed])
ok("goal lanes carry the goal id, so the panel can say WHICH goal",
   [r["goalId"] for r in derived] == [12, 281])
ok("lane kind is read from the REGISTRY, never from the letter",
   {r["role"]: r["kind"] for r in fixed if r["role"] in ("M", "F", "kevin")}
   == {"M": "session", "F": "agent", "kevin": "session"},
   {r["role"]: r["kind"] for r in fixed})
ok("a goal lane gets its OWN kind — neither registry answer is true of it",
   all(r["kind"] == "goal" for r in derived))
ok("every row's tip is a model string, not one written in the modal",
   all(r["kindTip"] in js("console.log(JSON.stringify("
                          "Object.values(LANE_KIND_TIP)));")
       for r in rows))
ok("colours come through roleColor, so every G-lane shares the prefix colour",
   len({r["color"] for r in derived}) == 1
   and {r["color"] for r in derived}.isdisjoint(
       {r["color"] for r in fixed}), [r["color"] for r in derived])

print("\n=== RAIL 5 — NO SECOND COPY OF THE MODEL (the anti-drift rail) ===")

# The modal may TALK about a status; it may not carry one as a literal. A
# standalone quoted literal equal to a model value is what a hand-typed table
# looks like, and it is the exact shape this rail refuses.
LITERALS = set(re.findall(r"'([^'\\\n]{1,60})'", HELP)) \
    | set(re.findall(r'"([^"\\\n]{1,60})"', HELP))

leaked_status = sorted(LITERALS & set(MODEL["statuses"]))
ok("no status name appears as a standalone string literal in the modal",
   not leaked_status, leaked_status)
leaked_role = sorted(LITERALS & {a for a in MODEL["assignees"] if a})
ok("no role letter appears as a standalone string literal in the modal",
   not leaked_role, leaked_role)
leaked_label = sorted(LITERALS & {t["label"] for t in MODEL["types"].values()})
ok("no type label appears as a standalone string literal in the modal",
   not leaked_label, leaked_label)

# The prose is the bigger risk: a `def` sentence pasted into JSX reads fine and
# is a copy. None of them may appear ANYWHERE in the file, quoted or not.
leaked_def = [t for t, d in MODEL["types"].items() if d["def"] in HELP]
ok("no type `def` sentence is reproduced anywhere in the modal", not leaked_def,
   leaked_def)
leaked_sdef = [s for s, d in MODEL["defs"].items() if d and d in HELP]
ok("no STATUS_DEF sentence is reproduced anywhere in the modal",
   not leaked_sdef, leaked_sdef)

# …and positively: it reads the model rather than re-declaring one.
IMPORT = re.search(r"import \{([^}]+)\} from './taskBoardShared';", HELP)
ok("the modal imports its vocabulary from taskBoardShared", IMPORT is not None)
imported = {s.strip() for s in (IMPORT.group(1) if IMPORT else "").split(",")}
for sym in ("TASK_TYPES", "TASK_TYPE_VALUES", "STATUSES", "STATUS_DEF",
            "ASSIGNEES", "FINISHED_STATUSES", "openGoalLanes", "laneKindLabel"):
    ok(f"`{sym}` is read from the model, not redefined here",
       sym in imported and f"const {sym} =" not in HELP)
# The ONLY module-level arrays the modal may own: its nav structure (TOPICS —
# which panels exist, a UI fact and not a rule), the warts (the one thing that
# cannot be derived, since the model is what they contradict), and the pointers
# to written sources. A fourth array is a vocabulary being retyped.
ALLOWED_ARRAYS = {"TOPICS", "HELP_WARTS", "HELP_SOURCES"}
own_arrays = set(re.findall(
    r"^(?:export )?const ([A-Z_]+)(?::[^=]+)? = \[", HELP, re.M))
ok("the modal declares no vocabulary array of its own beyond nav/warts/sources",
   own_arrays == ALLOWED_ARRAYS, sorted(own_arrays))

print("\n=== RAIL 6 — THE WARTS ARE STATED, AND CITED ===")

WARTS = js(lift_literal(HELP, "HELP_WARTS").replace(
    "const HELP_WARTS", "const W") + "console.log(JSON.stringify(W));",
    prelude="")
by_id = {w["id"]: w for w in WARTS}

for wid, ref in (("backlog-does-not-hold", "#296"),
                 ("statuses-not-enforced", "#261"),
                 ("assignee-unvalidated", "#289"),
                 ("staged-is-terminal", "#198")):
    ok(f"wart `{wid}` is documented, with board ref {ref}",
       wid in by_id and ref in by_id[wid]["refs"],
       by_id.get(wid, {}).get("refs"))

ok("every wart names a board ref and a file+symbol that proves it",
   all(w["refs"] and w["source"] and w["symbol"] for w in WARTS))
ok("every wart is attached to a panel that actually exists",
   {w["section"] for w in WARTS} <= {"types", "statuses", "assignees"},
   sorted({w["section"] for w in WARTS}))
ok("every wart is RENDERED (the panels filter to their own section)",
   "wartsFor(" in HELP and HELP.count("wartsFor('") >= 3)

# The citations are checked against the real files — this is what stops a wart
# outliving the behaviour it describes.
for w in WARTS:
    path = os.path.join(ROOT, w["source"])
    exists = os.path.isfile(path)
    ok(f"wart `{w['id']}` cites a file that exists: {w['source']}", exists)
    if exists:
        ok(f"…and `{w['symbol']}` is still in it",
           w["symbol"] in read(path), w["symbol"])

# The two warts whose whole point is what the code DOES — pinned at the source,
# so this suite fails if the behaviour is fixed and the manual is not updated.
DISP = read(ROOT, "tools", "team_dispatcher", "dispatcher.py")
ok("the OR gate the `Backlog` wart describes is really there",
   "or=(ai_eligible.eq.true,status.eq.Todo)" in DISP)
ok("`Staged` really is in the dispatcher's terminal set",
   re.search(r"TERMINAL_STATUSES = \([^)]*\"Staged\"", DISP) is not None)
API = read(SRC, "api", "routers", "dev_tasks.py")
ok("the API really validates task_type but NOT status — the wart, proved",
   '"task_type" in row' in API
   and not re.search(r'"status" in row and row\["status"\] not in', API))
ok("the API really has no assignee vocabulary — proved positively",
   '"origin" in row' in API and '"assignee" in row and' not in API)

print("\n=== RAIL 7 — SCOPE HELD ===")

ok("no prose write-up of process chains / stamps / release trains in this pass",
   not re.search(r"HELP_(CHAINS|STAMPS|TRAINS|LIFECYCLE)", HELP))
ok("the modal is read-only — no PATCH/POST/fetch anywhere in it",
   not re.search(r"apiFetch|fetch\(|PATCH|POST", HELP))
SOURCES = js(lift_literal(HELP, "HELP_SOURCES").replace(
    "const HELP_SOURCES", "const S") + "console.log(JSON.stringify(S));",
    prelude="")
ok("the manual cites at least one written source rather than restating it",
   len(SOURCES) >= 1)
for s in SOURCES:
    ok(f"cited source exists on disk: {s['path']}",
       os.path.isfile(os.path.join(ROOT, s["path"])))
ok("the charter is among them — roles are prose and stay prose",
   any("Session_Charters.md" in s["path"] for s in SOURCES))

print(f"\n{'=' * 60}")
if FAILED:
    print(f"RED — {len(FAILED)} failed / {PASS} passed")
    for f in FAILED:
        print(f"   ✗ {f}")
    sys.exit(1)
print(f"GREEN — {PASS} checks passed")
