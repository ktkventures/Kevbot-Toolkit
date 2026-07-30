"""Acceptance test — board #245: the shipping lane keys off an UNMERGED BRANCH.

WHAT WAS WRONG
--------------
`AdminDispatchPage.shippingRows` admitted a task on `rows.find(r => r.outcome
=== 'ok')` — *any run that finished successfully*. That is not a statement about
code. Measured on the live board: a Kevin DECISION (#133), a SPEC (#184) and
three E-lane DIAGNOSES (#125, #121, #31) all rendered as "built" and sat in a
lane about shipping, forever, because an agent had once analysed them.

And the lane's second signal was not merely weak, it was GONE. `pushed` reads
`run_history.pushed_branch`, which V4.23 wrote only on the dispatcher backstop's
OWN successful push. V4.26 (board #218) then told agents to push their own
branches — after which `push_worktree_branch` correctly refuses every one of them
(`exists-on-origin`) and recorded nothing. Every row showed `? pushed / unknown`.

WHAT THIS PINS — four rails, each with a check that FAILS without it
-------------------------------------------------------------------
  1. a task with an `ok` run but NO branch is ABSENT from the lane
  2. a task with an unmerged branch is PRESENT
  3. a task whose branch has MERGED drops off
  4. an unreadable merge source renders a STATED WARNING, never an empty list

...plus the writer half, on the Python side:
  5. an `exists-on-origin` refusal on a branch this run OWNS is RECORDED as
     published (the repair), while every other refusal still records nothing
  6. the refusal to ADVANCE a published branch is unchanged — the loop still
     never force-pushes or moves someone else's history

HOW THE UI RAILS ARE EXERCISED
------------------------------
The frontend has no JS test runner, so `mergeStateOf` and `shippingRows` are
LIFTED OUT of the .tsx and EXECUTED in node — the same technique board #228's
test uses. Only the SIGNATURE is rewritten (params de-typed); the body is copied
verbatim, so what runs here is what ships. That is the point: a static substring
assertion would pass against a lane that returns the wrong rows.

Run:  .venv/bin/python src/test_shipping_lane_unmerged_branch_245.py
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
sys.path.insert(0, os.path.join(ROOT, "tools", "team_dispatcher"))
import dispatcher as D  # noqa: E402

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


# ── lifting the pure TS out of the page so it can actually be RUN ───────────

def _split_params(params):
    """Split a parameter list on TOP-LEVEL commas only.

    Board #228's lifter split on every comma, which is fine until a parameter is
    generic: `runsByTask: Map<number, RunRow[]>` would become two parameters and
    silently shift every argument by one. Depth-aware, so it cannot.
    """
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
    """Return `function name(args) { ... }` as runnable JS. Signature rewritten,
    body copied VERBATIM (brace-matched), so the test runs the shipped code."""
    m = re.search(rf"^export function {name}\(", src, re.M)
    assert m, f"{name} not found in the page — the derivation is missing"
    line_end = src.index("\n", m.start())
    sig = src[m.start():line_end]
    params = sig[sig.index("(") + 1:sig.rindex(")")]
    i, depth = line_end, 1                      # the signature's own `{`
    while depth:
        i += 1
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
    return f"function {name}({', '.join(_split_params(params))}) {{\n" + src[line_end + 1:i + 1]


def js_const(src, name):
    """Lift `export const NAME = <literal>;` as runnable JS."""
    m = re.search(rf"^export const {name} = ([^;]+);", src, re.M)
    assert m, f"{name} not found"
    return f"const {name} = {m.group(1)};\n"


NODE = shutil.which("node")
ok("node is available to execute the lifted derivations", bool(NODE))

# The two shared helpers the page imports. Lifted from taskBoardShared so the
# harness uses the ONE implementation, never a re-statement of it.
FINISHED = re.search(r"export const isFinished = [^;]+;", SHARED)
ok("taskBoardShared exports isFinished (the #232 finished SET)", bool(FINISHED))
FINISHED_SET = re.search(r"export const FINISHED_STATUSES = [^;]+;", SHARED)

ok("taskBoardShared exports the FINISHED_STATUSES set it is built from", bool(FINISHED_SET))
# The SET is lifted verbatim (that is the thing that can drift); only the
# arrow's SIGNATURE is de-typed, exactly as js_body does for a function.
PRELUDE = (
    FINISHED_SET.group(0).replace("export ", "") + "\n"
    + re.sub(r"\(status\?[^)]*\):\s*boolean\s*=>", "(status) =>",
             FINISHED.group(0).replace("export ", "")) + "\n"
    + "const stepOwner = (s) => (s.owner !== undefined && s.owner !== null) ? s.owner : (s.role ?? null);\n"
    + "const stepTitle = (s) => s.title || s.text || '';\n"
    # hoursSince: only the ORDERING and the age basis matter here, so a fixed
    # `now` keeps the test deterministic.
    + "const NOW = Date.parse('2026-07-30T22:00:00Z');\n"
    + "const hoursSince = (t) => t ? (NOW - Date.parse(t)) / 3600000 : 0;\n"
)


def run_js(tasks, runs_by_task, pushed_known):
    """Execute the PAGE'S OWN shippingRows/mergeStateOf on a board fixture."""
    harness = (
        PRELUDE
        + js_const(PAGE, "SHIP_STEP_OWNERS")
        + js_body(PAGE, "mergeStateOf") + "\n"
        + js_body(PAGE, "shippingRows") + "\n"
        + "const a = JSON.parse(process.argv[2]);\n"
        + "const m = new Map(Object.entries(a.runs).map(([k, v]) => [Number(k), v]));\n"
        + "const lane = shippingRows(a.tasks, m, a.pushedKnown);\n"
        + "console.log(JSON.stringify({\n"
        + "  ids: lane.rows.map(r => r.task.id), basis: lane.basis,\n"
        + "  unknownMerges: lane.unknownMerges, warning: lane.warning,\n"
        + "  rows: lane.rows.map(r => ({id: r.task.id, branch: r.pushedBranch,\n"
        + "    merge: r.merge.state, why: r.merge.why, ageFrom: r.ageFrom, ageH: r.ageH})),\n"
        + "}));\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        out = subprocess.run(
            [NODE, path, json.dumps({"tasks": tasks, "runs": runs_by_task,
                                     "pushedKnown": pushed_known})],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:600]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


# ── the fixture: the five REAL shapes that provoked this task ──────────────
SHIP_STEP = {"owner": "R-A", "title": "Ship via a release train", "done": False,
             "mode": "execute"}
CHAIN = [{"owner": "F", "title": "build it", "done": True, "mode": "execute"}]


def task(tid, status="Review", chain=None):
    return {"id": tid, "title": f"task {tid}", "status": status,
            "checklist": chain if chain is not None else list(CHAIN) + [dict(SHIP_STEP)],
            "assignee": "M", "updated_at": "2026-07-30T12:00:00Z"}


def run(tid, outcome="ok", branch=None, pushed_at=None, finished="2026-07-30T10:00:00Z"):
    r = {"task_id": tid, "run_id": f"r1-{tid}", "agent_letter": "F",
         "outcome": outcome, "finished_at": finished, "requested_at": finished,
         "pushed_branch": branch, "pushed_at": pushed_at}
    return r


# #133 (a Kevin DECISION) and #184 (a SPEC): one ok run each, no code.
# #240 has a branch and an open ship step. #232 merged (its ship step is ticked).
# #999 has a branch but NO release-lane step at all → merge state UNREADABLE.
TASKS = [task(133), task(184), task(240), task(232),
         task(999, chain=list(CHAIN))]
MERGED_CHAIN = [{"owner": "F", "title": "build it", "done": True, "mode": "execute"},
                {"owner": "R-A", "title": "Ship via a release train", "done": True,
                 "mode": "execute"}]
TASKS[3]["checklist"] = MERGED_CHAIN
RUNS = {
    "133": [run(133)],
    "184": [run(184)],
    "240": [run(240, branch="fix/thing-240", pushed_at="2026-07-29T10:00:00Z")],
    "232": [run(232, branch="feat/statuses-232", pushed_at="2026-07-28T10:00:00Z")],
    "999": [run(999, branch="fix/orphan-999", pushed_at="2026-07-30T20:00:00Z")],
}

print("― RAIL 1: a successful run is NOT an artifact ―")
lane = run_js(TASKS, RUNS, True)
ok("#133 (a Kevin DECISION with an ok run, no branch) is ABSENT from the lane",
   133 not in lane["ids"], lane["ids"])
ok("#184 (a SPEC with an ok run, no branch) is ABSENT from the lane",
   184 not in lane["ids"], lane["ids"])
ok("membership no longer keys off outcome==='ok'",
   "rows.find((r) => r.outcome === 'ok')" not in PAGE
   or "pushedKnown" in PAGE.split("export function shippingRows")[1][:2000])
ok("the page states the rule in words, so the next reader cannot mistake it",
   "recorded pushed branch that is not merged" in PAGE
   or "recorded pushed branch that has not merged" in PAGE)

def row_of(res, tid):
    """The lane row for `tid`, or a placeholder — so a rail that DROPS a row it
    should keep reports a FAIL rather than crashing the run."""
    hit = [r for r in res["rows"] if r["id"] == tid]
    return hit[0] if hit else {"id": tid, "branch": None, "merge": "ABSENT",
                               "why": "", "ageFrom": "", "ageH": -1}


print("― RAIL 2: an UNMERGED branch is present ―")
ok("#240 (branch, ship step still open) IS in the lane", 240 in lane["ids"], lane["ids"])
row240 = row_of(lane, 240)
ok("...carrying its branch name", row240["branch"] == "fix/thing-240", row240)
ok("...with merge state UNMERGED", row240["merge"] == "unmerged", row240)
ok("...and its age measured from the BRANCH, not the run",
   row240["ageFrom"] == "branch", row240)
ok("age-from-branch is the pushed_at delta, not the run's finished_at",
   abs(row240["ageH"] - 36.0) < 0.01, row240)

print("― RAIL 3: a MERGED branch drops off ―")
ok("#232 (ship step ticked) is ABSENT", 232 not in lane["ids"], lane["ids"])
merged_state = run_js([TASKS[3]], {"232": RUNS["232"]}, True)
ok("...and the lane is empty for it, on the BRANCH basis (not a fallback)",
   merged_state["ids"] == [] and merged_state["basis"] == "branch", merged_state)
ok("a Done/Closed task drops off too (the #232 finished SET, unchanged)",
   run_js([task(240, status="Closed")], {"240": RUNS["240"]}, True)["ids"] == [])
ok("the merge source is the release-lane step, named in the page",
   "car_ship_tick" in PAGE and "merge-base --is-ancestor" in PAGE)
ok("the ship-step owner list MIRRORS car_ship_tick.py's SHIP_OWNERS",
   sorted(re.findall(r"'([^']+)'", re.search(r"export const SHIP_STEP_OWNERS = \[([^\]]*)\]", PAGE).group(1)))
   == sorted(re.findall(r"'([^']+)'", re.search(r"^SHIP_OWNERS = \(([^)]*)\)",
             read(ROOT, "tools", "release_brief", "car_ship_tick.py"), re.M).group(1)))
   if os.path.exists(os.path.join(ROOT, "tools", "release_brief", "car_ship_tick.py"))
   else "SHIP_STEP_OWNERS" in PAGE,
   "car_ship_tick.py is board #242 and may not have merged yet")

print("― RAIL 4: an unreadable source WARNS; it never renders as an empty lane ―")
# 4a — merge state unreadable for one row (no release-lane step in its chain).
ok("#999 (branch, but NO R/R-A step) is KEPT, not dropped", 999 in lane["ids"], lane["ids"])
row999 = row_of(lane, 999)
ok("...and labelled merge=unknown with a reason", row999["merge"] == "unknown"
   and "release-lane" in row999["why"], row999)
ok("...and counted on the lane, so the panel can say so", lane["unknownMerges"] == 1, lane)
ok("...and the lane carries a warning naming it",
   lane["warning"] and "release-lane" in lane["warning"], lane["warning"])

# 4b — the whole `pushed_branch` signal unreadable (un-migrated DB / API not
# serving the column). The realistic shape of that is the key being ABSENT from
# every run row — which is exactly when filtering on it would empty the lane and
# read as "nothing is in flight". So the fixture drops the key entirely.
LEGACY_RUNS = {k: [{kk: vv for kk, vv in r.items()
                    if kk not in ("pushed_branch", "pushed_at")} for r in v]
               for k, v in RUNS.items()}
legacy = run_js(TASKS, LEGACY_RUNS, False)
ok("with pushed_branch unreadable the lane does NOT go empty",
   len(legacy["ids"]) > 0, legacy)
ok("...it falls back to the legacy any-ok-run rule and SAYS so (basis)",
   legacy["basis"] == "legacy", legacy)
ok("...the warning states what cannot be answered",
   legacy["warning"] and "pushed_branch" in legacy["warning"], legacy["warning"])
ok("...and names the fix", legacy["warning"] and "run_history_pushed_branch.sql" in legacy["warning"])
ok("the warning is RENDERED above the list, not only computed",
   "lane.warning &&" in PAGE and "this lane is not fully readable" in PAGE)
ok("the empty-state sentence only claims the good news on the branch basis",
   "lane.basis === 'branch'" in PAGE and "nothing pushed and unmerged" in PAGE)
ok("...and says so explicitly when it is the legacy basis",
   "NOT \\n" not in PAGE and "legacy basis" in PAGE)

print("― the lane's OWN shape: still oldest-first, still a pure read ―")
ok("rows come back oldest-first", lane["ids"] == sorted(
    lane["ids"], key=lambda i: -[r for r in lane["rows"] if r["id"] == i][0]["ageH"]),
   lane["rows"])
ok("shippingRows takes pushedKnown — the degradation is an INPUT, not a guess",
   "shippingRows(tasks, runsByTask, pushedKnown)" in PAGE)
ok("the page still performs no new fetch for merge state — it is READ off the "
   "payload it already has (spec §3: no new endpoint, no new capture)",
   "/api/git" not in PAGE and "api.github.com" not in PAGE
   and not re.search(r"apiFetch[^\n]*(merge|git|github)", PAGE, re.I))

print("― RAIL 5 (the WRITER): an agent's own push is recorded on the run ―")
SRC_DISP = read(ROOT, "tools", "team_dispatcher", "dispatcher.py")
ok("dispatcher declares V4.29+", re.search(r"Team dispatcher \(V4\.(\d+)\)", SRC_DISP)
   and int(re.search(r"Team dispatcher \(V4\.(\d+)\)", SRC_DISP).group(1)) >= 29)
ok("exists-on-origin now returns a `published` branch",
   re.search(r'"reason": "exists-on-origin", "published": branch', SRC_DISP) is not None)
ok("_aggregate carries published_branch up to reap",
   '"published_branch": published or None' in SRC_DISP)
ok("reap records it through the SAME rh_record_push as its own push",
   re.search(r'elif res\.get\("published_branch"\):\s*(?:#[^\n]*\n\s*)*'
             r'r\["pushed_branch"\] = res\["published_branch"\]\s*\n\s*'
             r'rh_record_push\(r\["run_id"\], res\["published_branch"\]\)', SRC_DISP) is not None)
ok("the WHY (reaper, not agent-side) is written down where it is done",
   "prose contract" in SRC_DISP and "CHECKED fact" in SRC_DISP)
ok("no new column is introduced — pushed_branch/pushed_at only",
   "merged_at" not in SRC_DISP and "pushed_branch" in SRC_DISP)


def _fake_git(script):
    """Drive push_worktree_branch with a scripted _git()."""
    calls = []

    def fake(wt, *args):
        calls.append(args)
        for pat, res in script:
            if args[:len(pat)] == pat:
                return res
        return (0, "", "")
    return fake, calls


BASE = D.PUSH_BASE
ALREADY = [
    (("rev-parse", "--abbrev-ref", "HEAD"), (0, "feat/mine-245", "")),
    (("status", "--porcelain"), (0, "", "")),
    (("rev-list", "--count", f"{BASE}..feat/mine-245"), (0, "3", "")),
    (("ls-remote", "--heads", "origin", "feat/mine-245"),
     (0, "abc123\trefs/heads/feat/mine-245", "")),
    (("rev-parse", "feat/mine-245"), (0, "abc123", "")),
]
real = D._git
try:
    D._git, calls = _fake_git(ALREADY)
    res = D.push_worktree_branch("/tmp/wt-245", "push[t]")
finally:
    D._git = real
ok("5a an already-published branch this run owns reports published=<branch>",
   res.get("published") == "feat/mine-245", res)
ok("5a ...and STILL does not push it (the refusal is intact)",
   res["pushed"] is False and ("push", "-u", "origin", "feat/mine-245") not in calls, res)
ok("5a ...and never force-pushes anything", not any("--force" in c for c in calls))
ok("5a ...noting whether the remote matches local HEAD",
   res.get("published_head_matches") is True, res)

# The same branch, but the remote is AHEAD/behind local: still published.
DIVERGED = list(ALREADY[:-1]) + [(("rev-parse", "feat/mine-245"), (0, "zzz999", ""))]
try:
    D._git, calls = _fake_git(DIVERGED)
    res2 = D.push_worktree_branch("/tmp/wt-245", "push[t]")
finally:
    D._git = real
ok("5b a diverged remote is still a PUBLISHED branch (there is code on origin)",
   res2.get("published") == "feat/mine-245" and res2.get("published_head_matches") is False, res2)

# Every OTHER refusal must still record nothing — this is the rail that keeps
# the repair from becoming "record any branch we happen to see".
for label, script in (
    ("protected", [(("rev-parse", "--abbrev-ref", "HEAD"), (0, "dev", ""))]),
    ("dirty", [(("rev-parse", "--abbrev-ref", "HEAD"), (0, "feat/x", "")),
               (("status", "--porcelain"), (0, " M a.py", ""))]),
    ("nothing-ahead", [(("rev-parse", "--abbrev-ref", "HEAD"), (0, "feat/x", "")),
                       (("status", "--porcelain"), (0, "", "")),
                       (("rev-list", "--count", f"{BASE}..feat/x"), (0, "0", ""))]),
    ("detached", [(("rev-parse", "--abbrev-ref", "HEAD"), (0, "HEAD", ""))]),
):
    try:
        D._git, _ = _fake_git(script)
        r = D.push_worktree_branch("/tmp/wt-245", "push[t]")
    finally:
        D._git = real
    ok(f"5c a `{label}` refusal records NO published branch", not r.get("published"), r)

# ...and the aggregate/reap path only records when there IS one.
agg = D._aggregate("push[t]", [dict(res)])
ok("5d _aggregate surfaces the published branch with pushed=False",
   agg["pushed"] is False and agg["published_branch"] == "feat/mine-245", agg)
agg2 = D._aggregate("push[t]", [{"pushed": False, "branch": None, "worktree": "/w",
                                 "reason": "dirty"}])
ok("5d ...and reports None when nothing was published", agg2["published_branch"] is None, agg2)

print()
if FAILED:
    print(f"FAILURES ({PASS + len(FAILED)} checks, {len(FAILED)} FAILED)")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL {PASS} CHECKS PASSED — board #245 shipping lane keys off an unmerged branch")
