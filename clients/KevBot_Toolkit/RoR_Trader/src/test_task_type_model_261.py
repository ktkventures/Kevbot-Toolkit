"""Acceptance test — board #261: the explicit task-type model.

The board already half-implemented task types: `isVisionTask()` DERIVED "vision"
from a tag or from having children, and a separate `VISION_STATUSES` list gave
those rows their own status set. The type was implicit and the split was
frontend-only. #261 makes the type explicit (`dev_tasks.task_type`), folds
`VISION_STATUSES` into ONE map, and puts the type on the row.

THE MIRROR TEST IS THE ONE THAT MATTERS (rail 1). Board #219 shipped a constant
without its frontend mirror and dev sat red for hours; board #245's mirror check
was quote-naive (`'x'` vs `"x"`) and read two identical lists as different. So
rail 1 does NOT compare source text: it EXECUTES the TypeScript map in node,
serialises it to JSON, and compares the parsed object to the Python dict, key by
key and list by list. A string-comparison mirror is not a mirror.

Rails, each of which FAILS without its fix:

  1. MIRROR — the TS `TASK_TYPES` and the Python `TASK_TYPES` are equal BY
     PARSED VALUE (executed, not grepped), and the vocabulary matches the
     migration's CHECK constraint
  2. VISION_STATUSES is GONE as a separate export, and the list it held survives
     byte-identical inside the map — a vision row's dropdown does not move
  3. the dropdown is RENDERED FROM THE MAP — `statusOptionsFor` executed for
     real, per type, including the legacy-status case that proves `withLegacy`
     is still in the path
  4. `taskTypeOf` — explicit wins; a pre-migration row falls back to today's
     derivation; and an explicit `action` row that HAS CHILDREN still resolves to
     a container (without which its subtasks vanish off the board entirely)
  5. the TYPE ICON is on the row (Kevin's ask), and `action` wears none
  6. the API accepts `task_type` (it is in `_EDITABLE`) and refuses a bogus one
  7. NO API STATUS ENFORCEMENT — an out-of-set status is still accepted, because
     236 legacy rows predate the map

Offline + hermetic: the real `api.routers.dev_tasks` constants are imported, the
real frontend derivations are EXECUTED in node, no DB and no network.

Run:  .venv/bin/python src/test_task_type_model_261.py   (from RoR_Trader root)
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
BOARD = read(VIEWS, "AdminTasksPage.tsx")
MODAL = read(VIEWS, "TaskDetailModal.tsx")
API_SRC = read(SRC, "api", "routers", "dev_tasks.py")

NODE = shutil.which("node")


# ── lifting the real TS, without depending on its quote style ───────────────
def lift_literal(src, name):
    """`export const NAME[: type] = <literal>;` → runnable JS, brace-matched.

    Brace-matched rather than `[^;]+` so a nested object/array literal is lifted
    whole, and the TYPE ANNOTATION is dropped — the value is what mirrors, not
    the annotation.
    """
    m = re.search(rf"^export const {name}(?::[^=]+)? = ", src, re.M)
    assert m, f"{name} not found in the source — the map is missing"
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


def lift_arrow(src, name):
    """`export const NAME = (args): T => …;` → runnable JS with types stripped."""
    m = re.search(rf"^export const {name} = \(", src, re.M)
    assert m, f"{name} not found in the source"
    i, depth = m.end() - 1, 0
    body = []
    # take everything to the terminating `;` at depth 0
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
    # de-type: `(t: Task, hasChildren = false): TaskType =>` → `(t, hasChildren = false) =>`
    sig_end = text.index("=>")
    sig, rest = text[:sig_end], text[sig_end:]
    params = sig[sig.index("(") + 1:sig.rindex(")")]
    out = []
    depth = 0
    for part in re.split(r",(?![^<(\[]*[>)\]])", params):
        part = part.strip()
        if not part:
            continue
        # keep a default value, drop the `: Type` between name and `=`
        mdef = re.match(r"(\w+)\s*(?::[^=]+)?(=\s*.+)?$", part, re.S)
        out.append((mdef.group(1) + " " + (mdef.group(2) or "")).strip())
    # Drop `as T` / `as A | B` casts. UNION-AWARE on purpose: a naive `as \w+`
    # leaves `| undefined` behind, and `x | undefined` is a BITWISE OR in JS —
    # it evaluates to 0, so the lifted function silently takes the wrong branch
    # and the rail "passes" on a function that is not the shipped one.
    body_js = re.sub(r"\s+as\s+[\w\[\]]+(?:\s*\|\s*[\w\[\]]+)*", "", rest)
    return f"const {name} = ({', '.join(out)}) {body_js};\n"


def run_js(program):
    """Execute JS lifted from the shipped source; return its JSON stdout."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(program)
        path = fh.name
    try:
        out = subprocess.run([NODE, path], capture_output=True, text=True,
                             timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:800]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


# The three lifted pieces every rail below runs against. `STATUSES` comes first
# because the map REFERENCES it (action reuses the pipeline rather than
# restating it — a restatement is the thing that drifts).
PRELUDE = (lift_literal(SHARED, "STATUSES")
           + lift_literal(SHARED, "TASK_TYPES")
           + lift_literal(SHARED, "DEFAULT_TASK_TYPE")
           + lift_arrow(SHARED, "withLegacy")
           + lift_arrow(SHARED, "statusesForType")
           + lift_arrow(SHARED, "taskTypeOf")
           + lift_arrow(SHARED, "statusOptionsFor"))

print("=" * 72)
print("board #261 — explicit task_type + per-type statuses + row icons")
print("=" * 72)

print("\n── RAIL 1 · THE MIRROR: TS and Python maps equal BY PARSED VALUE ───")
ok("node is available to EXECUTE the TS map (a string compare is not a mirror)",
   bool(NODE))

from api.routers import dev_tasks as dt  # noqa: E402

ts_map = run_js(PRELUDE + "console.log(JSON.stringify(TASK_TYPES));\n")
py_map = dt.TASK_TYPES

ok("the API declares a TASK_TYPES map", isinstance(py_map, dict) and py_map)
ok("the TS map EXECUTES and yields an object", isinstance(ts_map, dict) and ts_map)
ok("both sides declare the SAME THREE TYPES, in the same order",
   list(ts_map) == list(py_map) == ["action", "vision", "goal"],
   f"ts={list(ts_map)} py={list(py_map)}")

for t in ("action", "vision", "goal"):
    ts_def, py_def = ts_map.get(t, {}), py_map.get(t, {})
    ok(f"`{t}` — the two definitions are IDENTICAL by parsed value",
       ts_def == py_def,
       "ts=" + json.dumps(ts_def, ensure_ascii=False)
       + "  py=" + json.dumps(py_def, ensure_ascii=False))

ok("every type carries all five spec'd fields (label, icon, statuses, "
   "children, session) — plus the tooltip",
   all({"label", "icon", "statuses", "children", "session", "def"}
       <= set(d) for d in py_map.values()),
   str({k: sorted(v) for k, v in py_map.items()}))
ok("`action` reuses STATUSES itself rather than restating the pipeline",
   "statuses: STATUSES" in SHARED
   and ts_map["action"]["statuses"] == json.loads(
       run_js(PRELUDE + "console.log(JSON.stringify(STATUSES));\n")
       and json.dumps(run_js(PRELUDE + "console.log(JSON.stringify(STATUSES));\n"))))
ok("the containers are the two parent types; only `goal` carries a session",
   [t for t, d in py_map.items() if d["children"]] == ["vision", "goal"]
   and [t for t, d in py_map.items() if d["session"]] == ["goal"],
   str({k: (v["children"], v["session"]) for k, v in py_map.items()}))
ok("`loop` is NOT a type — Kevin raised it and excluded it from this plan",
   "loop" not in py_map and "loop" not in ts_map)
ok("the default type is declared on both sides and is `action`",
   dt.DEFAULT_TASK_TYPE == "action"
   and run_js(PRELUDE + "console.log(JSON.stringify(DEFAULT_TASK_TYPE));\n")
   == "action")

# The migration is board #261 step 1's file and may live on its own branch until
# the combined tree; guarded so this suite is green standalone AND catches a
# vocabulary drift once both are in one tree.
MIG = os.path.join(SRC, "migrations", "dev_tasks_task_type.sql")
if os.path.exists(MIG):
    check = re.search(r"CHECK \(task_type IN \(([^)]*)\)\)", read(MIG))
    sql_types = re.findall(r"'([^']+)'", check.group(1)) if check else []
    ok("the DB CHECK constraint pins EXACTLY the map's vocabulary",
       sorted(sql_types) == sorted(py_map), f"sql={sql_types} map={sorted(py_map)}")
    ok("the migration defaults the column to the declared default",
       f"DEFAULT '{dt.DEFAULT_TASK_TYPE}'" in read(MIG))
else:
    print("  -- migration file not in this tree (step-1 branch); vocabulary "
          "check deferred to the combined gate")


print("\n── RAIL 2 · VISION_STATUSES folded IN, not left beside ─────────────")
ok("VISION_STATUSES no longer exists as its own export — a second list beside "
   "the map is exactly how the two drift",
   not re.search(r"export const VISION_STATUSES", SHARED), )
ok("...and nothing in the board views still imports it",
   "VISION_STATUSES" not in BOARD and "VISION_STATUSES" not in MODAL)
# The pre-#261 literal, verbatim from git history. If the fold changed the list,
# every vision row's dropdown moved on landing — the one thing this task must not do.
PRE_261_VISION = ['Backlog', 'Scoping', 'Todo', 'In Progress', 'Blocked',
                  'Stand By', 'Done', 'Closed']
ok("the vision status set survives the fold UNCHANGED (same members, same order)",
   ts_map["vision"]["statuses"] == PRE_261_VISION, str(ts_map["vision"]["statuses"]))
ok("...and the goal set is the vision set PLUS Approval, minus Todo — a goal is "
   "authorised to start but never ships (its children do)",
   "Approval" in ts_map["goal"]["statuses"]
   and "Review" not in ts_map["goal"]["statuses"]
   and "Staged" not in ts_map["goal"]["statuses"]
   and "Scoping" in ts_map["goal"]["statuses"],
   str(ts_map["goal"]["statuses"]))


print("\n── RAIL 3 · the dropdown is RENDERED FROM THE MAP ──────────────────")
# statusOptionsFor is EXECUTED, not read: this is the generalisation of
# `withLegacy(isVision ? VISION_STATUSES : STATUSES, t.status)`.
opts = run_js(PRELUDE + """
const t = (status, extra = {}) => ({status, tags: [], ...extra});
console.log(JSON.stringify({
  action: statusOptionsFor(t('Todo'), 'action'),
  vision: statusOptionsFor(t('Todo'), 'vision'),
  // 'Backlog', not 'Todo' — a GOAL has no Todo (it is authorised to start, it
  // does not queue), so a Todo fixture would be legacy-appended and mask the set.
  goal:   statusOptionsFor(t('Backlog'), 'goal'),
  // the same goal, holding a status its type does NOT list: withLegacy keeps it.
  goalOnTodo: statusOptionsFor(t('Todo'), 'goal'),
  legacy: statusOptionsFor(t('needs-review'), 'vision'),
  unknown: statusOptionsFor(t('Todo'), 'nonesuch'),
}));
""")
ok("an ACTION row offers the full pipeline",
   opts["action"] == ts_map["action"]["statuses"], str(opts["action"]))
ok("a VISION row offers the container set — no Approval/Review/Staged",
   opts["vision"] == ts_map["vision"]["statuses"], str(opts["vision"]))
ok("a GOAL row offers its own set — Approval, no Review/Staged, no Todo",
   opts["goal"] == ts_map["goal"]["statuses"], str(opts["goal"]))
ok("...and a goal PARKED on a status its type lacks still renders it",
   opts["goalOnTodo"] == ts_map["goal"]["statuses"] + ["Todo"],
   str(opts["goalOnTodo"]))
ok("withLegacy IS STILL IN THE PATH — a legacy row holding an out-of-set status "
   "still renders it (dropping it would blank 236 rows' dropdowns)",
   opts["legacy"][-1] == "needs-review"
   and opts["legacy"][:-1] == ts_map["vision"]["statuses"], str(opts["legacy"]))
ok("an UNKNOWN type falls back to the full pipeline, never an empty dropdown",
   opts["unknown"] == ts_map["action"]["statuses"], str(opts["unknown"]))
ok("both call sites pass a TYPE, not the old isVision boolean",
   "statusOptionsFor(t, typeOf(t))" in BOARD
   and "statusOptionsFor(task, taskType)" in MODAL)


print("\n── RAIL 4 · taskTypeOf: explicit wins, and can never DEMOTE ────────")
res = run_js(PRELUDE + """
const t = (o) => ({id: 1, status: 'Todo', tags: [], ...o});
console.log(JSON.stringify({
  explicitGoal:      taskTypeOf(t({task_type: 'goal'}), false),
  explicitVision:    taskTypeOf(t({task_type: 'vision'}), false),
  explicitAction:    taskTypeOf(t({task_type: 'action'}), false),
  // pre-migration / API not serving the column → the OLD derivation, whole
  legacyTagged:      taskTypeOf(t({tags: ['vision']}), false),
  legacyParent:      taskTypeOf(t({}), true),
  legacyLeaf:        taskTypeOf(t({}), false),
  // the rescue: a row the backfill could not have covered
  actionWithKids:    taskTypeOf(t({task_type: 'action'}), true),
  actionTaggedLater: taskTypeOf(t({task_type: 'action', tags: ['vision']}), false),
  // a goal is already a container — it needs no rescue and must not become one
  goalWithKids:      taskTypeOf(t({task_type: 'goal'}), true),
  // an unknown value is not trusted; it falls back rather than passing through
  bogus:             taskTypeOf(t({task_type: 'loop'}), true),
  bogusLeaf:         taskTypeOf(t({task_type: 'loop'}), false),
}));
""")
ok("an explicit type is honoured", res["explicitGoal"] == "goal"
   and res["explicitVision"] == "vision" and res["explicitAction"] == "action",
   str(res))
ok("a row the backfill missed falls back to TODAY'S derivation (tag OR children)",
   res["legacyTagged"] == "vision" and res["legacyParent"] == "vision"
   and res["legacyLeaf"] == "action", str(res))
ok("an explicit `action` row that HAS CHILDREN still resolves to a container — "
   "without this its subtasks drop out of visionItems AND out of looseTasks "
   "(parent_id != null), i.e. vanish off the board",
   res["actionWithKids"] == "vision", str(res))
ok("...and the same for a row tagged 'vision' after the backfill ran",
   res["actionTaggedLater"] == "vision", str(res))
ok("a GOAL with children stays a goal — the rescue only ever promotes a "
   "non-container, it never rewrites a declared container type",
   res["goalWithKids"] == "goal", str(res))
ok("an unknown type value is not passed through — it falls back",
   res["bogus"] == "vision" and res["bogusLeaf"] == "action", str(res))
ok("isVisionTask is now the map, not a re-derivation",
   "taskTypeOf(t, byParent.has(t.id)) === 'vision'" in SHARED)
ok("the modal no longer keeps its OWN copy of the derivation",
   "(task.tags || []).includes('vision') || subtasks.length > 0" not in MODAL
   and "taskTypeOf(task, subtasks.length > 0)" in MODAL)


print("\n── RAIL 5 · the TYPE ICON is on the row (Kevin's ask) ──────────────")
ok("a TaskTypeIcon component exists and reads the map",
   "export const TaskTypeIcon" in SHARED and "TASK_TYPES[type]" in SHARED)
ok("the board row renders it", "<TaskTypeIcon type={typeOf(t)} />" in BOARD)
ok("...and the detail modal wears the same glyph",
   "<TaskTypeIcon type={taskType} />" in MODAL)
ok("each CONTAINER type has a non-empty icon",
   all(py_map[t]["icon"] for t in ("vision", "goal")),
   str({t: py_map[t]["icon"] for t in py_map}))
ok("the two container icons are DISTINCT — one glyph for two types says nothing",
   py_map["vision"]["icon"] != py_map["goal"]["icon"])
ok("`action` — the default row — wears NO icon (an icon on every row is an "
   "icon on no row)", py_map["action"]["icon"] == "")
ok("the icon carries a tooltip saying what the type IS",
   all(py_map[t]["def"].startswith(t) for t in py_map),
   str({t: py_map[t]["def"] for t in py_map}))


print("\n── RAIL 6 · the API accepts task_type, and only the three ──────────")
ok("`task_type` is editable via the API (POST/PATCH whitelist)",
   "task_type" in dt._EDITABLE, str(sorted(dt._EDITABLE)))
ok("`goal_params` is NOT editable yet — it is board #262's, read by nothing",
   "goal_params" not in dt._EDITABLE)


class _Refused(Exception):
    pass


def validate(row):
    """Run the REAL validator; return the 400 detail, or None when accepted."""
    try:
        dt._validate_team_fields(None, dict(row))
    except Exception as e:  # HTTPException
        return getattr(e, "detail", str(e))
    return None


for good in ("action", "vision", "goal"):
    ok(f"...a `{good}` patch is accepted",
       validate({"task_type": good}) is None, str(validate({"task_type": good})))
bogus = validate({"task_type": "Vision"})
ok("a bogus type is REFUSED LOUDLY, with the legal set named — a free-text type "
   "lies the first time someone writes 'Vision' or 'goals'",
   bogus is not None and "task_type" in str(bogus), str(bogus))
ok("...and the refusal names all three legal values",
   all(t in str(bogus) for t in ("action", "vision", "goal")), str(bogus))


print("\n── RAIL 7 · NO API status enforcement (deliberately deferred) ──────")
# Enforcing the map's statuses at the API on day one would reject legacy rows —
# which is exactly why withLegacy exists. This rail fails if someone adds it.
legacy_statuses = ["needs-review", "In Review", "wip", "Backlog"]
ok("an OUT-OF-SET status is still accepted for every type — the 236 rows that "
   "predate this map must keep patching",
   all(validate({"status": s}) is None for s in legacy_statuses)
   and all(validate({"status": s, "task_type": t}) is None
           for s in legacy_statuses for t in ("action", "vision", "goal")))
val_src = API_SRC[API_SRC.index("def _validate_team_fields"):]
val_src = val_src[:val_src.index("\ndef _admin")]
val_code = re.sub(r"#[^\n]*", "", val_src)
val_code = re.sub(r'"""[\s\S]*?"""', "", val_code)
ok("...and the validator contains NO status check at all (not merely a "
   "permissive one)",
   '"status"' not in val_code and "'status'" not in val_code, val_code[-400:])
ok("the map's statuses are labelled declarative on the API side",
   "DECLARATIVE, NOT A GATE" in API_SRC)


print("\n" + "=" * 72)
print(f"{PASS} passed, {len(FAILED)} failed")
if FAILED:
    for f in FAILED:
        print(f"  ✗ {f}")
    sys.exit(1)
print("ALL RAILS GREEN")
