"""Acceptance test — board #193, Step 3: the dispatch dashboard (Half 1).

`Spec_Dispatch_Dashboard.md` is the contract. The page itself is a pure READ —
no schema change, no new endpoint, no new capture — so there is no server
behaviour to exercise. What CAN rot silently, and what this test pins, is the
set of invariants the spec spends its length on:

  • CONSTANT DRIFT — `DISPATCHER` in taskBoardShared.tsx mirrors dispatcher.py's
    CONCURRENCY / DAILY_CAP / RUN_TIMEOUT_S because the app cannot read Python.
    A silent drift here makes the lane panel lie about saturation, which is the
    one number Kevin reads at a glance. dispatcher.py is the SSOT: this test
    parses BOTH and fails on any divergence (spec §4).
  • REUSE, DO NOT FORK (§6) — the page must IMPORT #198's AiEligibleToggle /
    RunStatePill / RunButton and #182's chain helpers, never redefine them. Two
    implementations of the input/output split would recreate the exact defect
    the split exists to fix.
  • POLL SHAPE (§10) — three list GETs joined in memory, `include_done=true`,
    and no per-row fetch (the board-#148 lesson).
  • HONEST BLIND SPOTS (§8) — the known-gaps footer is present, the page never
    claims the dispatcher is alive, and PR/merge state renders as untracked
    rather than as a false negative (§7). This is the #157 dead-alarm lesson:
    a dashboard whose blind spots are invisible manufactures false confidence.
  • WIRING — the route, the sidebar entry and the reciprocal Agents link exist.

Static-source assertions, so it is offline, hermetic and runs anywhere.

Run:  .venv/bin/python src/test_dispatch_dashboard_193.py   (from RoR_Trader root)
"""
import os
import re
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
FE = os.path.join(ROOT, "frontend", "src")

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


def read(*parts):
    p = os.path.join(*parts)
    ok(f"exists: {os.path.relpath(p, ROOT)}", os.path.exists(p))
    with open(p, encoding="utf-8") as f:
        return f.read()


PAGE = read(FE, "views", "AdminDispatchPage.tsx")
SHARED = read(FE, "views", "taskBoardShared.tsx")
ROUTE = read(FE, "app", "admin", "dispatch", "page.tsx")
SIDEBAR = read(FE, "components", "Sidebar.tsx")
AGENTS = read(FE, "views", "AdminAgentsPage.tsx")
DISP = read(ROOT, "tools", "team_dispatcher", "dispatcher.py")

print("― §4 dispatcher constants: dispatcher.py is the SSOT, the UI mirrors it ―")

def py_const(name):
    m = re.search(rf"^{name}\s*=\s*([^#\n]+)", DISP, re.M)
    ok(f"dispatcher.py defines {name}", bool(m))
    return eval(m.group(1).strip(), {}, {})   # noqa: S307 — literal arithmetic

m = re.search(r"export const DISPATCHER = \{([^}]*)\}", SHARED)
ok("taskBoardShared exports the DISPATCHER mirror", bool(m))
ui = dict(re.findall(r"(\w+):\s*(\d+)", m.group(1)))
for name in ("CONCURRENCY", "DAILY_CAP", "RUN_TIMEOUT_S"):
    want = py_const(name)
    ok(f"{name} mirrors dispatcher.py ({want})", int(ui.get(name, -1)) == want,
       f"UI has {ui.get(name)!r}")
ok("the mirror names dispatcher.py as SSOT in a comment",
   "tools/team_dispatcher/dispatcher.py" in SHARED)
ok("POLL_S is the OPERATIONAL value (--loop --poll 20), not the module default",
   int(ui.get("POLL_S", -1)) == 20)
# The starvation the spec insists on SHOWING rather than hiding.
ok("§4: more headless lanes than slots is rendered, not hidden",
   "headless lanes vs" in PAGE and "starved" in PAGE)

print("― §4 lane occupancy: a stale `running` row must NOT hold a slot ―")
ok("isLeaseLive gates on RUN_TIMEOUT_S + a grace", "RUN_TIMEOUT_S + STALE_GRACE_S" in PAGE)
ok("slots are filled from liveRuns, never from raw running rows",
   re.search(r"slots\s*=\s*Array\.from\(\{\s*length:\s*DISPATCHER\.CONCURRENCY\s*\}[^\n]*liveRuns", PAGE) is not None)
ok("stale rows get their own loud banner", "staleRuns.length > 0" in PAGE
   and "past" in PAGE and "lease" in PAGE)
ok("the banner says stale rows are excluded from the cap",
   "not</b> counted against the lane cap" in PAGE)

print("― §5 the STEP: shared helpers, never a re-implementation ―")
for helper in ("isProcessChain", "stepOwner", "stepTitle"):
    ok(f"imports {helper} from taskBoardShared", re.search(rf"\b{helper}\b", PAGE) is not None)
    ok(f"does NOT redefine {helper}",
       re.search(rf"(const|function)\s+{helper}\s*[=(]", PAGE) is None)
ok("renders STEP n of N", "STEP {info.n} of {info.total}" in PAGE)
ok("a legacy checklist is labelled, not silently rendered as a chain",
   "no chain — whole-task dispatch" in PAGE)

print("― §6 two controls, deliberately separate — imported, not forked ―")
for comp in ("AiEligibleToggle", "RunStatePill", "RunButton", "runIneligibleReason"):
    ok(f"imports {comp}", re.search(rf"\b{comp}\b", PAGE) is not None)
    ok(f"does NOT redefine {comp} page-locally",
       re.search(rf"(export\s+)?(const|function)\s+{comp}\s*[=(]", PAGE) is None)
    ok(f"{comp} still lives in taskBoardShared (the one implementation)",
       re.search(rf"export (const|function) {comp}", SHARED) is not None)
ok("the Run button keeps its one-off-queue-jump label",
   "▶ Run once" in SHARED and "Run once" not in PAGE.split("import {")[0])
ok("§6 degrade-never-lie: an absent ai_eligible falls back to a LABELLED legacy gate",
   "aiEligibleKnown" in PAGE and "legacy Todo gate" in PAGE)

print("― §7 shipping lane: age is the headline, PR'd is untracked, unknown ≠ no ―")
ok("age thresholds live in one shared helper", "export function ageColor" in SHARED)
ok("thresholds are board #166's 12h amber / 24h red",
   re.search(r"hours >= 24.*var\(--red\)", SHARED, re.S) is not None
   and re.search(r"hours >= 12.*amber", SHARED, re.S) is not None)
ok("rows are sorted oldest-first", "sort((a, b) => (b.ageH) - (a.ageH))" in PAGE)
ok("PR'd renders as untracked, not as a negative", 'state="untracked"' in PAGE)
ok("the PR'd tooltip names what would unlock it (pr_number + a GitHub token)",
   "pr_number" in PAGE and "GitHub token" in PAGE)
ok("an absent pushed_branch renders UNKNOWN, never 'not pushed'",
   "'unknown'" in PAGE and "not \"not pushed\"" in PAGE)
ok("board state is labelled as a proxy for the merge, not as git truth",
   "board state" in PAGE and "Not read from git." in PAGE)
ok("run_history.pushed_branch is OPTIONAL in the type (un-migrated DB is fine)",
   "pushed_branch?: string | null" in SHARED)
ok("PR numbers are never scraped out of comment prose",
   "dev_task_comments" not in PAGE and "PR #" not in PAGE)

print("― Kevin 07-29: Staged must not be forgettable ―")
ok("Staged and Review both get a waiting bucket",
   "waitBucket('Staged'" in PAGE and "waitBucket('Review'" in PAGE)
ok("each bucket carries the AGE OF THE OLDEST, not an average",
   "oldest {ageShort(b.since)" in PAGE)
ok("the age's SOURCE is named in the tooltip (run vs board edit)",
   "its last completed run" in PAGE and "its last board edit" in PAGE)

print("― §10 poll shape: three list GETs, joined in memory ―")
gets = re.findall(r"apiFetch<[^>]*>\('([^']+)'", PAGE) + \
       re.findall(r"apiFetch<[^>]*>\(`([^`]+)`", PAGE)
loads = [g for g in gets if g.startswith("/api/")]
ok("exactly three list endpoints are read", len(set(
    g.split("?")[0] for g in loads)) == 3, loads)
ok("include_done=true (the run log references Done tasks)",
   any("include_done=true" in g for g in loads))
ok("run-history is bulk, not per-row", any("run-history?limit=" in g for g in loads))
# #148: the poll fetches THREE lists and joins them; the only per-task calls in
# the file are the two user-initiated WRITES (§9: "no new write path"), never a
# read fanned out over rows.
load_body = PAGE.split("const load = useCallback")[1].split("}, []);")[0]
ok("the poll body issues exactly three apiFetch calls (#148)",
   load_body.count("apiFetch") == 3, load_body.count("apiFetch"))
per_task = set(re.findall(r"apiFetch\(`(/api/[^`]*\$\{[^`]*)`", PAGE))
ok("the only per-task calls are the two writes, not a read fan-out",
   per_task == {"/api/dev-tasks/${id}", "/api/dev-tasks/${id}/run-request"}, per_task)
ok("auto-refresh pauses while the tab is hidden", "document.hidden" in PAGE)
ok("every timestamp renders UTC with a Z", "export function utcStamp" in SHARED
   and "Z`" in SHARED and "utcStamp(" in PAGE)

print("― §8 known gaps are PRINTED, and no liveness is claimed ―")
ok("the known-gaps footer is on the page", "KNOWN GAPS" in PAGE)
for gap in ("Dispatcher liveness is not observable",
            "PR / merge state is not tracked",
            "state.json"):
    ok(f"gap printed: {gap[:42]}", gap in PAGE)
ok("worded as 'last run event', never 'dispatcher alive'",
   "last run event" in PAGE
   and not re.search(r"dispatcher (is )?alive", PAGE.replace("never <i>dispatcher alive</i>", "")))
ok("the derived failure streak is labelled as an approximation",
   "consecutiveFailures" in PAGE and "state.json, which the app cannot read" in PAGE)

print("― §2/§11 wiring: route, sidebar, reciprocal links ―")
ok("the route is client-only (ssr: false), mirroring admin/agents",
   "ssr: false" in ROUTE and "AdminDispatchPage" in ROUTE)
ok("sidebar carries /admin/dispatch immediately after Agents",
   re.search(r"'/admin/agents', label: 'Agents' \},\s*\n\s*\{ href: '/admin/dispatch'", SIDEBAR)
   is not None)
ok("the Agents page links back to the dispatch page",
   '"/admin/dispatch"' in AGENTS or "'/admin/dispatch'" in AGENTS)
ok("dispatch rows deep-link into the board", "/admin/tasks?task=$" in PAGE)

print("― §9 non-goals: Half 2 is NOT built here ―")
# Functional markers only — the page is ALLOWED to name the Half-2 motivation in
# prose and tooltips (it must, to explain why PR'd is a dash); what it must not
# do is reach GitHub or offer a push button.
for banned in ("octokit", "api.github.com", "github.com/", "gh pr ", "pr_number:"):
    ok(f"no train-builder surface: {banned}", banned.lower() not in PAGE.lower())
ok("no 'push a train to dev' action", re.search(r"onClick=\{[^}]*(push|merge|train)", PAGE, re.I) is None)
ok("the page writes nothing new: only the ai_eligible PATCH and the run POST",
   sorted(set(re.findall(r"method: '(\w+)'", PAGE))) == ["PATCH", "POST"])

print("― CLAUDE.md production-bundle rails ―")
body = PAGE.split("export default function AdminDispatchPage")[1]
m = re.search(r"^  return \(", body, re.M)          # the component's own return
ok("the component has a render return", bool(m))
hooks_after_return = re.search(r"use(Memo|State|Effect|Callback)\(", body[m.start():])
ok("every hook precedes the render return", hooks_after_return is None)
ok("no IIFE initialisation", re.search(r"=\s*\(\(\)\s*=>\s*\{", PAGE) is None)

print(f"\nALL PASS ({PASS} checks)")
