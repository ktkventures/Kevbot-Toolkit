"""Acceptance test — board #262: the `goal` task type's behaviour.

#261 gave `goal` a row in the type map and shipped the `goal_params` column.
#262 gives it its RULES and its OUTPUT: the parameters that define a goal, the
context block that starts its session, and the two guards that stop a goal
being a runaway.

WHAT IS NOT BEING BUILT, and is asserted so it never creeps in: a `/goal`
skill, a runner or a scheduler. Kevin uses Claude's BUILT-IN `/goal` (his
ruling, 07-31) — everything here is INPUT to that command.

Rails, each of which FAILS without its fix:

  1. THE SHAPE — `GOAL_PARAM_FIELDS` declares M's seven fields with the right
     REQUIRED set, and `autonomy` is declared as COLUMN-backed
  2. THE REFUSAL, which is the whole gate — `renderGoalBlock` EXECUTED returns
     NO BLOCK for a goal with a blank `done_when`, and a real block once it is
     filled in. Refusal is TOTAL: `block` is null, not a block with a hole
  3. THE WIRING — autonomy is read from and written to the EXISTING
     `standing_approval` column. A stray `goal_params.autonomy` does NOT win,
     and is stripped on write. No parallel field
  4. INHERITANCE — a goal's child inherits its autonomy unless it overrides it
  5. THE NESTING GUARD — a goal may not carry a `parent_id`, in the UI AND at
     the API, in BOTH patch directions (parent onto a goal / goal onto a
     parented row)
  6. THE BLOCK'S CONTENT — the things a fresh session cannot guess and M named
     as easy to forget: `parent_id`, M reviews, E drives every Railway action /
     flag flip / migration, and the standing rails emitted whether or not the
     author filled `boundaries` in
  7. GOAL CHILDREN ARE VISIBLE — `groupBoard` groups on CONTAINER, not on
     `vision`, or a goal's subtasks vanish off the board entirely
  8. THE FENCES HOLD — no API status enforcement was added (#261's fence,
     #168's follow-up), no schema change, and no `/goal` skill

The frontend derivations are EXECUTED in node, not grepped: board #245's
lesson is that a string-compare "mirror" is not a mirror. Offline + hermetic —
the real `api.routers.dev_tasks` validator is imported and called, no DB and no
network.

Run:  .venv/bin/python src/test_goal_task_type_262.py   (from RoR_Trader root)
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
        print(f"  FAIL: {label} — {detail}")


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return fh.read()


SHARED = read(VIEWS, "taskBoardShared.tsx")
MODAL = read(VIEWS, "TaskDetailModal.tsx")
API_SRC = read(SRC, "api", "routers", "dev_tasks.py")

NODE = shutil.which("node")


# ── lifting the real TS (same technique as test_task_type_model_261.py) ─────
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


def lift_expr(src, name):
    """`export const NAME = <expression>;` → runnable JS. For a value built by
    a call chain (no leading brace), which `lift_literal` cannot brace-match."""
    m = re.search(rf"^export const {name}(?::[^=]+)? = ", src, re.M)
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
    """`export const NAME = (args): T => …;` → runnable JS with types stripped."""
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
        mdef = re.match(r"(\w+)\s*(?::[^=]+)?(=\s*.+)?$", part, re.S)
        assert mdef, f"could not de-type parameter {part!r} of {name}"
        out.append((mdef.group(1) + " " + (mdef.group(2) or "")).strip())
    body_js = re.sub(r"\s+as\s+[\w\[\]]+(?:\s*\|\s*[\w\[\]]+)*", "", rest)
    return f"const {name} = ({', '.join(out)}) {body_js};\n"


def lift_group_board(src):
    """`export function groupBoard(...)` → runnable JS.

    A named function with an inline return-type annotation, which neither
    lifter above handles. The three substitutions below are EXPLICIT rather
    than a general TS stripper, so a change to the signature breaks this
    loudly instead of lifting some other function's body by accident.
    """
    start = src.index("export function groupBoard")
    end = src.index("\n}\n", start) + 3
    text = src[start:end]
    text = text.replace("export function groupBoard(tasks: Task[]): {",
                        "function groupBoard(tasks) { /*")
    text = text.replace(
        "byParent: Map<number, Task[]>; visionItems: Task[]; looseTasks: Task[];\n} {",
        "*/")
    text = text.replace("new Map<number, Task[]>()", "new Map()")
    assert ": Task[]" not in text and "Map<" not in text, \
        f"groupBoard signature changed — the lifter needs updating:\n{text[:400]}"
    return text + "\n"


def run_js(program):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(program)
        path = fh.name
    try:
        out = subprocess.run([NODE, path], capture_output=True, text=True,
                             timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:1200]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


# Declaration order matters: every lifted piece must exist before one that
# CALLS it runs.
PRELUDE = (
    lift_literal(SHARED, "STATUSES")
    + lift_literal(SHARED, "TASK_TYPES")
    + lift_arrow(SHARED, "taskTypeOf")
    + lift_arrow(SHARED, "isContainerTask")
    + lift_group_board(SHARED)
    + lift_literal(SHARED, "AUTONOMY_VALUES")
    + lift_literal(SHARED, "AUTONOMY_DEF")
    + lift_literal(SHARED, "DEFAULT_GOAL_LANES")
    + lift_literal(SHARED, "GOAL_PARAM_FIELDS")
    + lift_expr(SHARED, "GOAL_PARAM_KEYS")
    + lift_literal(SHARED, "GOAL_STANDING_RAILS")
    + lift_arrow(SHARED, "autonomyOf")
    + lift_arrow(SHARED, "autonomyPatch")
    + lift_arrow(SHARED, "goalParamsOf")
    + lift_arrow(SHARED, "inheritedAutonomy")
    + lift_arrow(SHARED, "goalLanes")
    + lift_arrow(SHARED, "goalParamsPatch")
    + lift_arrow(SHARED, "goalVal")
    + lift_arrow(SHARED, "missingGoalParams")
    + lift_arrow(SHARED, "goalNestError")
    + lift_arrow(SHARED, "renderGoalBlock")
)

# A goal that is fully authored, and the same goal with `done_when` blanked.
GOAL_JS = """
const goal = (params, extra) => Object.assign({
  id: 262, title: 'Drive paired-% to 90 on the gated fleet',
  tags: [], parent_id: null, task_type: 'goal',
  standing_approval: false, goal_params: params,
}, extra || {});
const FULL = {
  objective: 'Every monitored GATED strategy trades at 90% paired or better.',
  done_when: 'get_strategy_health reports >=90% paired for all monitored GATED strategies over the recent-activation window, two sessions running.',
  scope: 'engine divergence work; no new strategies',
  lanes: ['E', 'F'],
  oversight: 'Kevin checks the morning brief.',
  boundaries: 'Do not touch the live-trade pause.',
};
"""

print("=" * 72)
print("board #262 — the goal type: params, refusal, autonomy wiring, nesting")
print("=" * 72)

ok("node is available to EXECUTE the shipped derivations "
   "(a string compare is not a test of behaviour)", bool(NODE))

print("\n── RAIL 1 · THE SHAPE: M's seven fields, the right ones required ───")
fields = run_js(PRELUDE + "console.log(JSON.stringify(GOAL_PARAM_FIELDS));\n")
by_key = {f["key"]: f for f in fields}
ok("all seven of M's spec'd params are declared",
   sorted(by_key) == sorted(["objective", "done_when", "boundaries", "autonomy",
                             "scope", "lanes", "oversight"]),
   str(sorted(by_key)))
ok("REQUIRED = objective, done_when, autonomy — exactly M's table",
   sorted(f["key"] for f in fields if f["required"])
   == sorted(["objective", "done_when", "autonomy"]),
   str([f["key"] for f in fields if f["required"]]))
ok("`done_when` carries WHY it is required, so the next author does not "
   "delete it as boilerplate",
   "never ends" in by_key["done_when"]["hint"], by_key["done_when"]["hint"])
ok("`autonomy` is declared COLUMN-backed — `standing_approval`, by name",
   by_key["autonomy"].get("column") == "standing_approval",
   json.dumps(by_key["autonomy"]))
ok("...and it is the ONLY column-backed field (everything else is JSONB)",
   [f["key"] for f in fields if f.get("column")] == ["autonomy"])
keys = run_js(PRELUDE + "console.log(JSON.stringify(GOAL_PARAM_KEYS));\n")
ok("the JSONB key list EXCLUDES autonomy — it is not a params key at all",
   "autonomy" not in keys and "done_when" in keys, str(keys))

print("\n── RAIL 2 · THE REFUSAL IS THE GATE ────────────────────────────────")
res = run_js(PRELUDE + GOAL_JS + """
const noDone = Object.assign({}, FULL); delete noDone.done_when;
const blankDone = Object.assign({}, FULL, {done_when: '   '});
const noObj = Object.assign({}, FULL); delete noObj.objective;
console.log(JSON.stringify({
  full:      renderGoalBlock(goal(FULL), 'goal', 'https://api.example'),
  missing:   renderGoalBlock(goal(noDone), 'goal', 'https://api.example'),
  blank:     renderGoalBlock(goal(blankDone), 'goal', 'https://api.example'),
  noObj:     renderGoalBlock(goal(noObj), 'goal', 'https://api.example'),
  notAGoal:  renderGoalBlock(goal(FULL), 'action', 'https://api.example'),
  emptyGoal: renderGoalBlock(goal(null), 'goal', 'https://api.example'),
}));
""")
ok("a goal with NO `done_when` is REFUSED", res["missing"]["ok"] is False)
ok("...and the refusal is TOTAL — `block` is null, not a block with a hole "
   "in it (a half-rendered block is the one that gets pasted anyway)",
   res["missing"]["block"] is None, json.dumps(res["missing"])[:300])
ok("...and it says WHY, in the words that make the rule stick",
   "never ends" in res["missing"]["refusal"]
   and "done_when" in res["missing"]["refusal"], res["missing"]["refusal"])
ok("...and it names what is missing, so it is actionable",
   res["missing"]["missing"] == ["done_when"], str(res["missing"]["missing"]))
ok("WHITESPACE is not an end condition — '   ' refuses too",
   res["blank"]["ok"] is False and res["blank"]["block"] is None)
ok("a goal missing `objective` is refused as well (M marked it required)",
   res["noObj"]["ok"] is False and res["noObj"]["missing"] == ["objective"],
   str(res["noObj"]["missing"]))
ok("a goal with NO params at all refuses and names BOTH missing requireds",
   res["emptyGoal"]["ok"] is False
   and sorted(res["emptyGoal"]["missing"]) == ["done_when", "objective"],
   str(res["emptyGoal"]["missing"]))
ok("a FULLY authored goal RENDERS — the refusal is a gate, not a wall",
   res["full"]["ok"] is True and bool(res["full"]["block"]))
ok("a non-goal task has no goal block at all",
   res["notAGoal"]["ok"] is False and res["notAGoal"]["block"] is None)
BLOCK = res["full"]["block"]
ok("the rendered block carries the objective and the end condition VERBATIM "
   "— not a paraphrase the session has to trust",
   FULL_OBJ := True and "Every monitored GATED strategy trades at 90%" in BLOCK
   and "two sessions running" in BLOCK)

print("\n── RAIL 3 · THE WIRING: autonomy IS `standing_approval` ────────────")
wired = run_js(PRELUDE + GOAL_JS + """
const off = goal(FULL);
const on  = goal(FULL, {standing_approval: true});
// A goal whose JSONB claims the OPPOSITE of its column. The column must win.
const lying = goal(Object.assign({}, FULL, {autonomy: 'standing'}),
                   {standing_approval: false});
console.log(JSON.stringify({
  off: autonomyOf(off),
  on:  autonomyOf(on),
  patchStanding: autonomyPatch('standing'),
  patchApprove:  autonomyPatch('approve_each'),
  lyingReads: autonomyOf(lying),
  strippedOnWrite: goalParamsPatch(lying, 'scope', 'x'),
  blockOff: renderGoalBlock(off, 'goal', 'h').block,
  blockOn:  renderGoalBlock(on,  'goal', 'h').block,
}));
""")
ok("`standing_approval` false → approve_each; true → standing",
   wired["off"] == "approve_each" and wired["on"] == "standing", str(wired))
ok("setting autonomy PATCHES THE COLUMN and writes no params key",
   wired["patchStanding"] == {"standing_approval": True}
   and wired["patchApprove"] == {"standing_approval": False},
   str(wired["patchStanding"]))
ok("a stray `goal_params.autonomy` does NOT win — the column does "
   "(two switches for one decision is what this task exists to avoid)",
   wired["lyingReads"] == "approve_each", wired["lyingReads"])
ok("...and it is STRIPPED on the next write, so it cannot linger and confuse "
   "a later reader",
   "autonomy" not in wired["strippedOnWrite"]["goal_params"],
   json.dumps(wired["strippedOnWrite"]))
ok("the BLOCK's autonomy section tracks the column, both ways",
   "## Autonomy — approve_each" in wired["blockOff"]
   and "assign\nit to `kevin`, and WAIT" in wired["blockOff"]
   and "## Autonomy — standing" in wired["blockOn"]
   and "You do NOT need Kevin" in wired["blockOn"])
ok("the modal's autonomy control writes `autonomyPatch`, never a params key",
   "autonomyPatch(e.target.value" in MODAL
   and "goal_params: { autonomy" not in MODAL)
ok("the reserved checkbox at the Config tab is the SAME column, and now says "
   "what reads it",
   "standing_approval: e.target.checked" in MODAL
   and "no pipeline logic reads it yet" not in MODAL
   and "board #262" in MODAL)

print("\n── RAIL 4 · CHILDREN INHERIT THE GOAL'S AUTONOMY ───────────────────")
inh = run_js(PRELUDE + GOAL_JS + """
const kid = (params) => ({id: 900, title: 'child', tags: [], parent_id: 262,
                          task_type: 'action', standing_approval: false,
                          goal_params: params || null});
const standingGoal = goal(FULL, {standing_approval: true});
const cautiousGoal = goal(FULL, {standing_approval: false});
console.log(JSON.stringify({
  inheritsStanding: inheritedAutonomy(kid(), standingGoal),
  inheritsApprove:  inheritedAutonomy(kid(), cautiousGoal),
  overrideDown: inheritedAutonomy(kid({autonomy_override: 'approve_each'}), standingGoal),
  overrideUp:   inheritedAutonomy(kid({autonomy_override: 'standing'}), cautiousGoal),
  bogusOverride: inheritedAutonomy(kid({autonomy_override: 'yolo'}), standingGoal),
  orphan: inheritedAutonomy(kid(), null),
}));
""")
ok("a child with no override INHERITS the goal's autonomy — both values",
   inh["inheritsStanding"] == "standing"
   and inh["inheritsApprove"] == "approve_each", str(inh))
ok("...even though the child's OWN standing_approval column is false — "
   "inheritance is not the child's column read twice",
   inh["inheritsStanding"] == "standing")
ok("an explicit override wins, in BOTH directions (a boolean column cannot "
   "carry 'unset', so the override is its own explicit key)",
   inh["overrideDown"] == "approve_each" and inh["overrideUp"] == "standing")
ok("a BOGUS override does not silently become 'standing' — it falls back to "
   "the inherited value (no-silent-defaults)",
   inh["bogusOverride"] == "standing" and
   run_js(PRELUDE + GOAL_JS + """
     console.log(JSON.stringify(inheritedAutonomy(
       {id: 9, tags: [], parent_id: 262, standing_approval: false,
        goal_params: {autonomy_override: 'yolo'}},
       goal(FULL, {standing_approval: false}))));
   """) == "approve_each")
ok("the BLOCK tells the session its children inherit — a rule nobody applies "
   "is not a rule",
   "Children of this goal inherit this autonomy" in BLOCK)
ok("the modal SHOWS a child its inherited autonomy rather than applying it "
   "silently", "inheritedAutonomy(scoped, scopedParent)" in MODAL)

print("\n── RAIL 5 · THE NESTING GUARD: a goal is top-level ─────────────────")
nest = run_js(PRELUDE + GOAL_JS + """
console.log(JSON.stringify({
  goalUnderParent: goalNestError('goal', 41),
  goalTopLevel:    goalNestError('goal', null),
  actionUnder:     goalNestError('action', 41),
  visionUnder:     goalNestError('vision', 41),
}));
""")
ok("a goal with a parent is REFUSED in the UI, naming the parent",
   nest["goalUnderParent"] and "#41" in nest["goalUnderParent"],
   str(nest["goalUnderParent"]))
ok("...and the refusal names BOTH forbidden containers, because they are one "
   "rule and a reader will otherwise assume only one was meant",
   "vision" in nest["goalUnderParent"] and "goal" in nest["goalUnderParent"])
ok("a top-level goal is fine, and the guard does not fire on other types",
   nest["goalTopLevel"] is None and nest["actionUnder"] is None
   and nest["visionUnder"] is None, str(nest))
ok("the modal renders the guard INSTEAD of a parent select on a goal — not a "
   "select whose every choice the API refuses",
   "scopedType === 'goal' ?" in MODAL and "goalNestError('goal'" in MODAL)

from api.routers import dev_tasks as dt  # noqa: E402


class _Stored:
    """Minimal stand-in for the admin client, with no DB.

    It DISTINGUISHES the validator's queries by their filter column, because
    the validator asks three different questions of the same table — "what is
    stored for this id", "does this parent exist and is it itself a child",
    and "does this row have children of its own". A stub that answered all
    three with the same row would report every task as having subtasks and
    turn a passing rail red for a reason that has nothing to do with goals.
    """

    def __init__(self, row, children=None):
        self.row = row
        self.children = children or []
        self.col = None

    def table(self, _name):
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, col, _val):
        self.col = col
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        data = self.children if self.col == "parent_id" \
            else ([self.row] if self.row else [])
        return type("R", (), {"data": data})()


def api_validate(row, task_id=None, stored=None, children=None):
    """Run the REAL server validator; return the 400 detail, or None."""
    try:
        dt._validate_team_fields(_Stored(stored, children), dict(row),
                                 task_id=task_id)
    except Exception as e:
        return getattr(e, "detail", str(e))
    return None


created = api_validate({"task_type": "goal", "parent_id": 41})
ok("CREATE — a goal born with a parent is refused by the API",
   created is not None and "top-level" in str(created), str(created))
patched = api_validate({"parent_id": 41}, task_id=262,
                       stored={"task_type": "goal", "parent_id": None})
ok("PATCH direction A — filing an EXISTING goal under a parent is refused",
   patched is not None and "top-level" in str(patched), str(patched))
retyped = api_validate({"task_type": "goal"}, task_id=900,
                       stored={"task_type": "action", "parent_id": 41})
ok("PATCH direction B — RETYPING a parented task to `goal` is refused too "
   "(a patch arrives as either half; one check, both directions)",
   retyped is not None and "top-level" in str(retyped), str(retyped))
ok("a top-level goal is accepted on create AND on patch",
   api_validate({"task_type": "goal", "parent_id": None}) is None
   and api_validate({"task_type": "goal"}, task_id=262,
                    stored={"task_type": "goal", "parent_id": None}) is None)
ok("a goal's CHILDREN are still perfectly legal — a goal contains action "
   "tasks; that is what it is for",
   api_validate({"parent_id": 262}, task_id=900,
                stored={"task_type": "action", "parent_id": None},
                ) is None)
ok("filing a VISION under a parent is unchanged — this guard is goal-only",
   api_validate({"task_type": "vision", "parent_id": None}) is None)

print("\n── RAIL 6 · WHAT THE BLOCK MUST SAY (M's easy-to-forget list) ──────")
ok("it tells the session to set `parent_id` to the goal's own id — the one "
   "mechanical thing a fresh session cannot guess",
   "set `parent_id` to 262" in BLOCK and "Do not create top-level tasks" in BLOCK,
   BLOCK[:200])
ok("it says M reviews its output", "M reviews your output" in BLOCK)
ok("KEVIN'S RULING — every Railway action, flag flip and migration goes "
   "through E, stated as the FIRST boundary",
   "Railway action, flag flip and migration goes through E" in BLOCK
   and BLOCK.index("goes through E")
   < BLOCK.index("Never force-push"), BLOCK[BLOCK.index("## Boundaries"):][:300])
ok("...and it is the existing rail UNCHANGED, not a widened one",
   "This rail is not widened for goals" in BLOCK)
for rail in ("Never force-push", "Never reset `dev` or `main`",
             "Never apply a migration", "Never edit the main checkout",
             "Never mark a task `Closed`"):
    ok(f"standing rail present: {rail}", rail in BLOCK)
noBounds = run_js(PRELUDE + GOAL_JS + """
const p = Object.assign({}, FULL); delete p.boundaries;
console.log(JSON.stringify(renderGoalBlock(goal(p), 'goal', 'h').block));
""")
ok("the rails are emitted even when the author wrote NO `boundaries` — they "
   "are not the author's to omit",
   all(r in noBounds for r in ("goes through E", "Never force-push",
                               "Never mark a task `Closed`")))
ok("...and the author's own boundaries are appended when present",
   "Do not touch the live-trade pause" in BLOCK
   and "Do not touch the live-trade pause" not in noBounds)
ok("it names the honest exit — what to do when the goal turns out to be "
   "unreachable (every goal session eventually meets one)",
   "unreachable or wrong, say THAT and stop" in BLOCK)
ok("it forbids quietly substituting a reachable goal",
   "do not quietly\nsubstitute a goal you can reach" in BLOCK)
ok("the lanes are written IN, defaulted declaratively so the session never "
   "has to guess",
   "## Lanes you may dispatch to\nE, F —" in BLOCK
   and "E, F, M-A" in run_js(PRELUDE + GOAL_JS + """
     const p = Object.assign({}, FULL); delete p.lanes;
     console.log(JSON.stringify(renderGoalBlock(goal(p), 'goal', 'h').block));
   """))
ok("the API host is substituted in, not left for the session to find",
   "https://api.example/api/dev-tasks" in BLOCK)
ok("SIGN OF LIFE uses #251's existing endpoint — no new infrastructure",
   "/api/r-session/heartbeat" in BLOCK and "Do not fake it" in BLOCK)
# `parse_letter` (api/routers/r_session.py) pins ^[A-Z](?:[0-9]|-[A-Z])?$, so a
# `G262` letter would 400 on every beat and the dashboards would show the
# session dead forever. The KEY stays a legal letter; the goal id rides in
# `actor`, which is a free string. Asserted against the SHIPPED regex, so this
# fails the day either side moves.
from api.routers.r_session import parse_letter  # noqa: E402
beat = re.search(r'\{"letter": "([^"]+)", "actor": "([^"]+)"\}', BLOCK)
ok("the heartbeat letter in the block is one the SHIPPED validator accepts "
   "(a rejected letter is a session that looks dead forever)",
   bool(beat) and parse_letter(beat.group(1)) == beat.group(1),
   str(beat.group(0) if beat else BLOCK[-400:]))
ok("...and the goal id is still carried, in `actor` where it is a free string",
   bool(beat) and beat.group(2) == "goal-262", str(beat and beat.group(2)))

print("\n── RAIL 7 · A GOAL'S CHILDREN ARE VISIBLE ON THE BOARD ─────────────")
grouped = run_js(PRELUDE + """
const t = (o) => Object.assign({id: 0, title: '', tags: [], parent_id: null,
                                task_type: 'action'}, o);
const tasks = [
  t({id: 1, task_type: 'vision'}),
  t({id: 2, parent_id: 1}),
  t({id: 3, task_type: 'goal'}),
  t({id: 4, parent_id: 3}),
  t({id: 5}),
];
const g = groupBoard(tasks);
console.log(JSON.stringify({
  containers: g.visionItems.map((x) => x.id),
  loose: g.looseTasks.map((x) => x.id),
  goalKids: (g.byParent.get(3) || []).map((x) => x.id),
  goalIsContainer: isContainerTask(t({id: 3, task_type: 'goal'}), true),
  childlessGoalIsContainer: isContainerTask(t({id: 3, task_type: 'goal'}), false),
}));
""")
ok("a GOAL is grouped as a container alongside visions — without this its "
   "children belong to no container and vanish off the board entirely",
   grouped["containers"] == [1, 3], str(grouped["containers"]))
ok("...so its children are reachable under it",
   grouped["goalKids"] == [4], str(grouped["goalKids"]))
ok("...and the goal itself is NOT dumped in with the loose tasks",
   grouped["loose"] == [5], str(grouped["loose"]))
ok("a goal is a container even before it has any children — it is one by "
   "TYPE, not by accident of having subtasks yet",
   grouped["childlessGoalIsContainer"] is True)
ok("the predicate reads the type map's `children` flag, so container type "
   "number three is one map entry and no code change",
   "TASK_TYPES[taskTypeOf(t, hasChildren)]?.children" in SHARED)
ok("`groupBoard` asks isContainerTask, not isVisionTask",
   SHARED.count("isContainerTask(t, byParent.has(t.id))") == 2
   and "isVisionTask(t, byParent))" not in SHARED)
ok("the modal treats a goal as a container too (selector strip, roll-up "
   "counts, pipeline exemption) — the board and the modal must not disagree",
   "const vision = !!TASK_TYPES[taskType]?.children;" in MODAL)

print("\n── RAIL 8 · THE FENCES HOLD ────────────────────────────────────────")
ok("`goal_params` is reachable through the API (it is in the whitelist)",
   "goal_params" in dt._EDITABLE, str(sorted(dt._EDITABLE)))
ok("...and its SHAPE is validated — an object or null, never a string",
   api_validate({"goal_params": {"objective": "x"}}) is None
   and api_validate({"goal_params": None}) is None
   and api_validate({"goal_params": "done_when=whenever"}) is not None
   and api_validate({"goal_params": ["a"]}) is not None)
ok("NO required-key enforcement at the API — a half-authored goal must still "
   "SAVE; the render is the gate (M's step-1 decision)",
   api_validate({"task_type": "goal", "goal_params": {}}) is None
   and api_validate({"task_type": "goal",
                     "goal_params": {"objective": "x"}}) is None)
val_src = API_SRC[API_SRC.index("def _validate_team_fields"):]
val_src = val_src[:val_src.index("\ndef _admin")]
val_code = re.sub(r"#[^\n]*", "", val_src)
val_code = re.sub(r'"""[\s\S]*?"""', "", val_code)
ok("...and no `done_when` check leaked into the validator at all",
   "done_when" not in val_code, val_code[-400:])
ok("#261's fence held: still NO status enforcement, for any type",
   all(api_validate({"status": s, "task_type": "goal"}) is None
       for s in ("In Review", "wip", "Scoping", "Backlog"))
   and '"status"' not in val_code and "'status'" not in val_code)
ok("NO schema change — `goal_params` already shipped in #261's migration",
   os.path.exists(os.path.join(SRC, "migrations", "dev_tasks_task_type.sql"))
   and "goal_params jsonb" in read(SRC, "migrations", "dev_tasks_task_type.sql")
   and not [f for f in os.listdir(os.path.join(SRC, "migrations"))
            if "goal" in f.lower()])
skills = os.path.join(ROOT, ".claude", "skills")
ok("NO `/goal` skill was created — Kevin uses Claude's BUILT-IN one and was "
   "explicit about it",
   not os.path.isdir(os.path.join(skills, "goal")),
   os.path.join(skills, "goal"))
_goal_hdr = SHARED[SHARED.index("THE GOAL TYPE (board #262)"):][:1400].lower()
ok("...and the code says so where the next author will read it",
   "built-in" in _goal_hdr
   and "runner or a scheduler" in _goal_hdr
   and "no skill, no runner, no scheduler" in MODAL, _goal_hdr[:300])
ok("the block is rendered CLIENT-SIDE for pasting — no runner, no scheduler, "
   "no server-side goal execution anywhere",
   "renderGoalBlock" not in API_SRC and "goal_params" in SHARED)

print("\n" + "=" * 72)
print(f"{PASS} passed, {len(FAILED)} failed")
if FAILED:
    for f in FAILED:
        print(f"  ✗ {f}")
    sys.exit(1)
print("ALL RAILS GREEN")
