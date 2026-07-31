"""Acceptance suite — board #246: multi-select area / assignee / status filters
on the team board, with checkboxes.

Kevin, 07-30: *"There's an area dropdown and a status dropdown, but there's no
status filter… can we make those multi-select so that way I can select which
ones I want to filter to see? My thought is you just have a checkbox next to
each option."*

WHAT WAS ACTUALLY THERE, measured: `areaFilter` and `assigneeFilter` were
single-select strings (`useState('')`), and there was NO status filter — status
was reachable only through two booleans that were status filters in disguise
(`reviewOnly` = `status === 'Review'`, `awaitingOk` = `status === 'Approval'`).
Kevin ruled on 07-30 that the real filter REPLACES them, so this suite pins
that the second path is gone: two paths to the same predicate let the page
contradict itself (tick "in Review", also tick "Staged", get an empty board,
read it as a bug).

THE RAIL THAT MATTERS MOST is rail 1. An EMPTY selection means NO FILTER — show
everything — never "show nothing". An empty board reads as data loss, and Kevin
twice on 07-30 concluded from a display artefact that work was being dropped
(the shipping-lane avatar, the stale ship steps). A filter that defaults to
hiding every row would be the third.

HOW THE BEHAVIOURAL RAILS ARE REAL RATHER THAN PROSE: the frontend has no JS
test runner (no jest/vitest, no `test` script — measured, not assumed), and the
repo's other frontend suites are source-scans. A source-scan cannot honestly
assert "empty selection shows ALL rows". So the predicates live in
`frontend/src/views/taskFilters.js` — dependency-free ES-module JavaScript —
and rails 1-7 EXECUTE THAT EXACT FILE under `node`. Reverting the
`selected.length === 0 ||` short-circuit fails rail 1 for real; a Python
re-implementation would have passed while the shipped code hid everything.

Rails:
  1. empty selection → every row passes (each filter alone, and all three)
  2. UNION within one filter — tick Review + Staged, see both
  3. INTERSECTION across filters — Review ∧ frontend, not Review ∨ frontend
  4. '' is a real assignee value "(unassigned)", no longer the all-pass sentinel
  5. toggleSelected ticks/unticks and returns a NEW array
  6. filterSummary never says nothing — 'all', the values, or a count
  7. activeSelectCount counts only CONSTRAINING filters
  8. the status option list is derived from STATUSES, never a literal
  9. STATUSES still carries all 11 (incl. Stand By / Closed) — a 12th is free
 10. the two redundant booleans are GONE — one path, per Kevin's ruling
 11. all three filters are `string[]` state initialised to `[]`
 12. the view always says what it is doing — summary, clear button, empty state
 13. taskFilters.js stays dependency-free (the property rails 1-7 rest on)

Run:  .venv/bin/python src/test_tasks_multiselect_filters_246.py   (from RoR_Trader root)
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

PASS = FAIL = 0
FAILURES = []


def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        FAILURES.append(f"{label}{(' — ' + detail) if detail else ''}")
        print(f"  FAIL: {label} — {detail}")


def read(*parts):
    p = os.path.join(*parts)
    if not os.path.exists(p):
        print(f"FAIL: missing file {os.path.relpath(p, ROOT)}")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return f.read()


FILTERS_PATH = os.path.join(FE, "views", "taskFilters.js")
FILTERS = read(FE, "views", "taskFilters.js")
PAGE = read(FE, "views", "AdminTasksPage.tsx")
SHARED = read(FE, "views", "taskBoardShared.tsx")


# ── the harness: run the REAL module under node ─────────────────────────────
HARNESS = r"""
import {
  matchesMulti, toggleSelected, filterSummary, taskMatchesSelects,
  activeSelectCount, UNASSIGNED_LABEL,
} from './taskFilters.mjs';

// A miniature board covering the shapes that matter: two areas, three
// statuses, an assigned and an UNASSIGNED row.
const BOARD = [
  { id: 1, status: 'Review',      area: 'frontend', assignee: 'F' },
  { id: 2, status: 'Staged',      area: 'engine',   assignee: 'E' },
  { id: 3, status: 'Review',      area: 'engine',   assignee: 'M' },
  { id: 4, status: 'In Progress', area: 'frontend', assignee: 'F' },
  { id: 5, status: 'Todo',        area: 'docs',     assignee: null },
];
const ids = (sel) => BOARD.filter((t) => taskMatchesSelects(t, sel)).map((t) => t.id);

const out = {
  // 1 — empty selection means NO FILTER
  none:            ids({}),
  allEmptyArrays:  ids({ areas: [], assignees: [], statuses: [] }),
  emptyStatusOnly: ids({ areas: [], assignees: [], statuses: [] }).length,
  emptyAreaWithStatus: ids({ areas: [], statuses: ['Review'] }),
  matchesMultiEmpty:   matchesMulti([], 'anything'),
  matchesMultiEmptyNull: matchesMulti([], null),
  // 2 — union WITHIN a filter
  unionStatus: ids({ statuses: ['Review', 'Staged'] }),
  unionArea:   ids({ areas: ['frontend', 'docs'] }),
  // 3 — intersection ACROSS filters
  crossStatusArea: ids({ statuses: ['Review'], areas: ['frontend'] }),
  crossThree:      ids({ statuses: ['Review', 'In Progress'], areas: ['frontend'], assignees: ['F'] }),
  crossEmptyResult: ids({ statuses: ['Staged'], areas: ['frontend'] }),
  // 4 — '' is a real value (unassigned), not the all-pass sentinel
  unassigned:      ids({ assignees: [''] }),
  unassignedPlusF: ids({ assignees: ['', 'F'] }),
  unassignedLabel: UNASSIGNED_LABEL,
  // 5 — toggle
  toggleAdd:    toggleSelected(['Review'], 'Staged'),
  toggleRemove: toggleSelected(['Review', 'Staged'], 'Review'),
  toggleFresh:  (() => { const a = ['Review']; const b = toggleSelected(a, 'Staged'); return a.length === 1 && b.length === 2; })(),
  toggleFromEmpty: toggleSelected([], 'Review'),
  // 6 — the closed control always says something
  summaryEmpty: filterSummary([]),
  summaryEmptyCustom: filterSummary([], 'everyone'),
  summaryOne:   filterSummary(['Review']),
  summaryTwo:   filterSummary(['Review', 'Staged']),
  summaryMany:  filterSummary(['Review', 'Staged', 'Todo']),
  summaryUnassigned: filterSummary(['']),
  // 7 — active count
  activeNone: activeSelectCount({ areas: [], assignees: [], statuses: [] }),
  activeOne:  activeSelectCount({ statuses: ['Review'] }),
  activeTwo:  activeSelectCount({ statuses: ['Review'], areas: ['frontend'] }),
  activeAll:  activeSelectCount({ statuses: ['Review'], areas: ['frontend'], assignees: ['F'] }),
};
process.stdout.write(JSON.stringify(out));
"""


def run_node():
    node = shutil.which("node")
    if not node:
        print("FAIL: `node` not found — rails 1-7 execute the real taskFilters.js "
              "and cannot be honestly reported without it. Not skipping (board #239: "
              "a suite that cannot run must not report green).")
        sys.exit(1)
    with tempfile.TemporaryDirectory() as td:
        # .mjs so bare `node` treats the (unmodified) source as an ES module.
        shutil.copyfile(FILTERS_PATH, os.path.join(td, "taskFilters.mjs"))
        hp = os.path.join(td, "harness.mjs")
        with open(hp, "w", encoding="utf-8") as f:
            f.write(HARNESS)
        r = subprocess.run([node, hp], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"FAIL: node harness exited {r.returncode}\n{r.stderr.strip()}")
            sys.exit(1)
        return json.loads(r.stdout)


print("― rails 1-7: the REAL taskFilters.js, executed under node ―")
R = run_node()

print("― rail 1: an EMPTY selection means NO FILTER, never 'show nothing' ―")
ok("no filters at all → every row", R["none"] == [1, 2, 3, 4, 5], f"got {R['none']}")
ok("three empty arrays → every row", R["allEmptyArrays"] == [1, 2, 3, 4, 5],
   f"got {R['allEmptyArrays']}")
ok("matchesMulti([], value) passes", R["matchesMultiEmpty"] is True)
ok("matchesMulti([], null) passes — a null value is not a filtered-out value",
   R["matchesMultiEmptyNull"] is True)
ok("an empty filter beside a set one constrains only by the set one",
   R["emptyAreaWithStatus"] == [1, 3], f"got {R['emptyAreaWithStatus']}")

print("― rail 2: UNION within one filter — 'show me Review AND Staged' ―")
ok("statuses [Review, Staged] → both", R["unionStatus"] == [1, 2, 3], f"got {R['unionStatus']}")
ok("areas [frontend, docs] → both", R["unionArea"] == [1, 4, 5], f"got {R['unionArea']}")

print("― rail 3: INTERSECTION across filters ―")
ok("Review ∧ frontend → only the row that is both", R["crossStatusArea"] == [1],
   f"got {R['crossStatusArea']} (a union would be [1,3,4])")
ok("status ∧ area ∧ assignee all applied", R["crossThree"] == [1, 4], f"got {R['crossThree']}")
ok("a genuinely empty intersection is empty (the filter really filters)",
   R["crossEmptyResult"] == [], f"got {R['crossEmptyResult']}")

print("― rail 4: '' is 'unassigned', not the all-pass sentinel it was ―")
ok("assignees [''] → only the unassigned row", R["unassigned"] == [5], f"got {R['unassigned']}")
ok("'' unions with a real role", R["unassignedPlusF"] == [1, 4, 5], f"got {R['unassignedPlusF']}")
ok("it renders as a word, not a blank row", R["unassignedLabel"] == "(unassigned)")

print("― rail 5: toggleSelected ―")
ok("tick adds", R["toggleAdd"] == ["Review", "Staged"], f"got {R['toggleAdd']}")
ok("untick removes", R["toggleRemove"] == ["Staged"], f"got {R['toggleRemove']}")
ok("from empty, ticking selects exactly one", R["toggleFromEmpty"] == ["Review"])
ok("returns a NEW array (React state must not be mutated in place)", R["toggleFresh"] is True)

print("― rail 6: the closed control never looks like an empty board ―")
ok("nothing ticked reads 'all'", R["summaryEmpty"] == "all", f"got {R['summaryEmpty']!r}")
ok("the all-label is overridable", R["summaryEmptyCustom"] == "everyone")
ok("one value names it", R["summaryOne"] == "Review")
ok("two values name both", R["summaryTwo"] == "Review, Staged")
ok("past that it counts, so the control never overflows",
   R["summaryMany"] == "3 selected", f"got {R['summaryMany']!r}")
ok("'' summarises as (unassigned), not as blank", R["summaryUnassigned"] == "(unassigned)")

print("― rail 7: activeSelectCount counts only CONSTRAINING filters ―")
ok("no filters → 0", R["activeNone"] == 0)
ok("one → 1", R["activeOne"] == 1)
ok("two → 2", R["activeTwo"] == 2)
ok("three → 3", R["activeAll"] == 3)

print("― rail 8: the status list is DERIVED from STATUSES, never a literal ―")
ok("the page feeds STATUSES to the status filter",
   re.search(r'<MultiSelectFilter\s+label="status"\s+options=\{STATUSES\}', PAGE) is not None,
   "expected <MultiSelectFilter label=\"status\" options={STATUSES} …>")
ok("STATUSES is imported from the shared module", re.search(r"\bSTATUSES\b", PAGE) is not None)
# A hand-written status list in the page is the failure this rail exists for: a
# twelfth status would then be invisible in the filter and nobody would notice.
page_literals = re.findall(r"\[[^\[\]]*'(?:Backlog|Scoping|Staged|Stand By)'[^\[\]]*\]", PAGE)
ok("no hard-coded status array in AdminTasksPage", page_literals == [],
   f"found {page_literals}")
ok("area options come from AREAS", 'options={AREAS}' in PAGE)
ok("assignee options come from the registry-backed list",
   'options={assigneeOptions}' in PAGE)

print("― rail 9: STATUSES still carries all eleven (a twelfth is then free) ―")
m = re.search(r"export const STATUSES = \[([^\]]*)\]", SHARED)
ok("taskBoardShared exports STATUSES", m is not None)
statuses = re.findall(r"'([^']+)'", m.group(1)) if m else []
ok("eleven statuses", len(statuses) == 11, f"got {len(statuses)}: {statuses}")
for s in ("Backlog", "Scoping", "Approval", "Todo", "In Progress", "Review",
          "Staged", "Blocked", "Stand By", "Done", "Closed"):
    ok(f"STATUSES includes {s!r}", s in statuses)

print("― rail 10: the two redundant booleans are GONE (Kevin's 07-30 ruling) ―")
ok("no reviewOnly state", re.search(r"useState.*\breviewOnly\b|setReviewOnly", PAGE) is None,
   "reviewOnly still has its own state — two paths to `status === 'Review'`")
ok("no awaitingOk state", re.search(r"const \[awaitingOk, setAwaitingOk\]", PAGE) is None,
   "awaitingOk still has its own state — two paths to `status === 'Approval'`")
ok("no `in Review` checkbox left behind", "checked={reviewOnly}" not in PAGE,
   "the checkbox was this filter with one box ticked")
ok("`Awaiting my OK` survives as a PRESET over the status filter",
   re.search(r"setStatusFilter\(awaitingOk \? \[\] : \['Approval'\]\)", PAGE) is not None,
   "the preset must SET the status filter, not carry its own boolean")
ok("the preset is derived from the status filter, not stored separately",
   re.search(r"const awaitingOk = statusFilter\.", PAGE) is not None)
ok("the four NON-status toggles are untouched (they are not statuses)",
   all(f"const [{n}, set" in PAGE for n in ("liveOnly", "stuckOnly", "armedOnly", "runningOnly")))
ok("hideDone is untouched too", "const [hideDone, setHideDone]" in PAGE)

print("― rail 11: all three filters are multi-select state, initialised EMPTY ―")
for name in ("areaFilter", "assigneeFilter", "statusFilter"):
    setter = "set" + name[0].upper() + name[1:]
    ok(f"{name} is string[] initialised to []",
       f"const [{name}, {setter}] = useState<string[]>([])" in PAGE,
       "single-select string state would make 'none selected' ambiguous again")
ok("the predicate goes through taskMatchesSelects", "taskMatchesSelects(t, {" in PAGE)
ok("all three feed the memo's dep list",
   re.search(r"areaFilter, assigneeFilter, statusFilter\]", PAGE) is not None,
   "a missing dep means the table does not re-filter on a tick")

print("― rail 12: a filtered view must never be mistakable for an empty board ―")
ok("the closed control renders the summary", "filterSummary(selected, allLabel)" in SHARED)
ok("the control goes bold while constraining", "fontWeight: active ? 700 : 400" in SHARED)
ok("the panel spells out that none-ticked = all",
   "none ticked = all" in SHARED)
ok("a per-filter clear is offered", "onChange([])" in SHARED)
ok("a global clear appears while any filter is active",
   re.search(r"✕ clear \{activeFilters\}", PAGE) is not None)
ok("the empty state names how many rows were LOADED",
   "{tasks.length} loaded" in PAGE,
   "'No tasks match the filters.' alone is what reads as data loss")
ok("the empty state says nothing is lost and offers a way back",
   "Nothing hidden by a filter is lost" in PAGE and "show all</button>" in PAGE)
ok("checkboxes, as Kevin asked", SHARED.count('type="checkbox"') >= 1
   and 'checked={selected.includes(o)}' in SHARED)

print("― rail 13: taskFilters.js stays dependency-free (rails 1-7 rest on it) ―")
ok("no import statements", re.search(r"^\s*import\s", FILTERS, re.M) is None,
   "an import makes the module un-runnable under bare node and rails 1-7 die")
ok("no require()", "require(" not in FILTERS)
CODE = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", FILTERS, flags=re.S))
ok("no JSX and no React in the CODE (the JSDoc may name them)",
   "React" not in CODE and "</" not in CODE)
ok("no TypeScript-only syntax — the file must be valid JS as written",
   re.search(r"\binterface\s+\w+|\bas\s+\w+\s*[;,)]|:\s*(string|number|boolean)\b", CODE) is None,
   "a type annotation would stop bare node from running it")
ok("types are carried in JSDoc, so tsc still checks it under allowJs",
   FILTERS.count("@param") >= 6 and FILTERS.count("@returns") >= 4)
ok("exports the five primitives the page and the control use",
   all(f"export const {n}" in FILTERS for n in
       ("matchesMulti", "toggleSelected", "filterSummary", "taskMatchesSelects",
        "activeSelectCount")))
ok("the reason it is .js rather than .ts is written down, not folklore",
   "no JS test runner" in FILTERS or "WHY A PLAIN .js MODULE" in FILTERS)

print()
if FAIL:
    print(f"FAILED — {FAIL} failure(s), {PASS} passed")
    for f in FAILURES:
        print(f"  · {f}")
    sys.exit(1)
print(f"ALL PASS — {PASS} assertions (board #246)")
