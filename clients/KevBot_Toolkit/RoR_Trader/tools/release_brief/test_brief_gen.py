#!/usr/bin/env python3
"""Rails for the release-brief generator (board #211).

Hermetic: no database, no git, no network. `build_train()` takes its three
external facts as callables, so every rail here is exercised against fixtures.

Every rail has a test that FAILS if the rail is removed -- the `_negative_*`
tests are the proof: each one deletes the rail's precondition and asserts the
generator's answer CHANGES.

Run:  python3 tools/release_brief/test_brief_gen.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brief_gen as B  # noqa: E402

PASS = FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}{(' -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def step(owner, title, done=True, mode="execute"):
    return {"owner": owner, "title": title, "done": done, "mode": mode,
            "id": f"s{abs(hash((owner, title))) % 10 ** 6}"}


def task(tid, title, steps):
    return {"id": tid, "title": title, "status": "Staged",
            "assignee": "M", "checklist": steps}


REVIEWED_CHAIN = [
    step("kevin", "Approve the approach — STAMP"),
    step("M", "Build: the thing"),
    step("M", "M review — full suite from a clean worktree"),
    step("R", "Ship with a release train", done=False),
    step("M", "Restart, verify, close", done=False),
]

UNREVIEWED_CHAIN = [
    step("kevin", "Approve the approach — STAMP"),
    step("M", "Build: the thing"),
    step("R", "Ship with a release train", done=False),
]


def fakes(branches, diffs, behind=None):
    behind = behind or {}
    return (lambda t: (branches.get(t["id"]), "fixture"),
            lambda b: list(diffs.get(b, [])),
            lambda b: behind.get(b, 0))


APP = B.APP_PREFIX


def train(tasks, branches, diffs, behind=None, owed=(), merged=()):
    rb, df, fb = fakes(branches, diffs, behind)
    return B.build_train(tasks, rb, df, fb, owed_logs=owed, wave=99,
                         base_sha="deadbee", date="2026-07-30", session="test",
                         already_merged=lambda b: b in merged)


# ---------------------------------------------------------------------------
# Car selection: "current chain step is R-owned"
# ---------------------------------------------------------------------------

def t_car_selection():
    t = task(202, "Step-tick contract", REVIEWED_CHAIN)
    p = train([t], {202: "feat/x-202"}, {"feat/x-202": [APP + "src/a.py"]})
    check("car selection: reviewed R-step task becomes a car", len(p["cars"]) == 1,
          str(p["refusals"]))
    check("car selection: car carries its branch",
          p["cars"] and p["cars"][0]["branch"] == "feat/x-202")

    # current step owned by M -> not waiting on a train
    mid = [step("M", "Build"), step("M", "M review", done=False),
           step("R", "Ship", done=False)]
    p = train([task(1, "mid-chain", mid)], {1: "feat/x-1"},
              {"feat/x-1": [APP + "src/a.py"]})
    check("car selection: M-owned current step is not a car", not p["cars"])

    # chain finished -> not a car
    done = [step("M", "M review"), step("R", "Ship"), step("M", "Close")]
    p = train([task(2, "finished", done)], {2: "feat/x-2"},
              {"feat/x-2": [APP + "src/a.py"]})
    check("car selection: finished chain is not a car", not p["cars"])

    # no chain at all -> not a car (legacy checklist / empty)
    p = train([task(3, "no chain", [])], {3: "feat/x-3"},
              {"feat/x-3": [APP + "src/a.py"]})
    check("car selection: chainless Staged task is not a car", not p["cars"])


# ---------------------------------------------------------------------------
# RAIL 1 -- an unreviewed car is REFUSED
# ---------------------------------------------------------------------------

def t_rail_unreviewed():
    t = task(300, "no review step", UNREVIEWED_CHAIN)
    p = train([t], {300: "feat/x-300"}, {"feat/x-300": [APP + "src/a.py"]})
    check("RAIL 1: chain with no M-review step is refused", not p["cars"],
          f"cars={[c['label'] for c in p['cars']]}")
    check("RAIL 1: refusal names the task and the reason",
          any(w == "#300" and "M-review" in why for w, why in p["refusals"]),
          str(p["refusals"]))

    # an M step that is not a REVIEW does not satisfy the rail
    sneaky = [step("M", "Build: the thing"), step("M", "M writes the docs"),
              step("R", "Ship", done=False)]
    p = train([task(301, "no review", sneaky)], {301: "feat/x-301"},
              {"feat/x-301": [APP + "src/a.py"]})
    check("RAIL 1: a non-review M step does not count as review", not p["cars"])

    # NEGATIVE PROOF: add the review step and the same task ships
    p = train([task(300, "now reviewed", REVIEWED_CHAIN)], {300: "feat/x-300"},
              {"feat/x-300": [APP + "src/a.py"]})
    check("RAIL 1 negative proof: adding the ticked review step admits the car",
          len(p["cars"]) == 1)


def t_rail_unreviewed_taskless():
    """A branch with no board task is refused unless it is an owed deploy-log."""
    p = train([], {}, {"feat/rogue": [APP + "src/a.py"]}, owed=["feat/rogue"])
    check("RAIL 1: taskless non-log branch is refused", not p["cars"], str(p["cars"]))
    p = train([], {}, {"docs/deploy-log-wave16-0730": [APP + "docs/Deploy_Log.md"]},
              owed=["docs/deploy-log-wave16-0730"])
    check("RAIL 1 negative proof: an owed deploy-log branch IS admitted",
          len(p["cars"]) == 1, str(p["refusals"]))


# ---------------------------------------------------------------------------
# RAIL 2 -- overlapping paths
# ---------------------------------------------------------------------------

def t_rail_overlap():
    a = task(401, "car A", REVIEWED_CHAIN)
    b = task(402, "car B", REVIEWED_CHAIN)
    dfile = APP + B.DISPATCHER_FILE
    p = train([a, b], {401: "feat/a-401", 402: "feat/b-402"},
              {"feat/a-401": [dfile], "feat/b-402": [dfile]})
    check("RAIL 2: two cars touching dispatcher.py -> HOLD", p["holds"], "no hold raised")
    check("RAIL 2: hold refuses auto-dispatch", p["auto_dispatch"] is False)
    check("RAIL 2: both cars still in the brief (hold != refuse)", len(p["cars"]) == 2)
    check("RAIL 2: hold text names the overlapping path",
          any(B.DISPATCHER_FILE in h for h in p["holds"]), str(p["holds"]))
    check("RAIL 2: rendered brief carries the HELD banner",
          "HELD FOR M" in B.render(p))

    # NEGATIVE PROOF: disjoint paths -> no hold, auto-dispatch
    p = train([a, b], {401: "feat/a-401", 402: "feat/b-402"},
              {"feat/a-401": [APP + "src/a.py"], "feat/b-402": [APP + "src/b.py"]})
    check("RAIL 2 negative proof: disjoint paths auto-dispatch",
          p["auto_dispatch"] is True and not p["holds"], str(p["holds"]))
    check("RAIL 2 negative proof: brief states zero overlap",
          "Zero file overlap" in B.render(p))


def t_rail_overlap_append_only():
    """Deploy_Log overlap is the ONE resolved shape: warn oldest-first, do not hold.
    This is the Wave-14 precedent (three owed logs merged 11 -> 12 -> 13)."""
    logs = ["docs/deploy-log-wave11-0730", "docs/deploy-log-wave12-0730"]
    p = train([], {}, {b: [APP + "docs/Deploy_Log.md"] for b in logs}, owed=logs)
    check("RAIL 2: append-only overlap warns instead of holding",
          p["warns"] and not p["holds"], f"holds={p['holds']}")
    check("RAIL 2: append-only overlap still auto-dispatches", p["auto_dispatch"] is True)
    body = B.render(p)
    check("RAIL 2: the warning tells R oldest-first", "oldest-first" in body)
    check("RAIL 2: the warning tells R when to abort", "ABORT and hand to M" in body)


# ---------------------------------------------------------------------------
# RAIL 3 -- a dispatcher.py car pulls the FULL suite list
# ---------------------------------------------------------------------------

def t_rail_dispatcher_suites():
    t = task(500, "dispatcher change", REVIEWED_CHAIN)
    p = train([t], {500: "feat/d-500"},
              {"feat/d-500": [APP + B.DISPATCHER_FILE,
                              APP + "src/test_dispatcher_step_tick_202.py"]})
    gates = p["gates"]
    for suite in B.DISPATCHER_SUITES:
        check(f"RAIL 3: full suite includes {os.path.basename(suite)}",
              f"python3 {suite}" in gates, str(gates))
    check("RAIL 3: the whole suite set is pulled, not just the touched test",
          len(gates) >= len(B.DISPATCHER_SUITES))
    check("RAIL 3: no duplicate gate command", len(gates) == len(set(gates)))
    check("RAIL 3: gate paths are app-relative, not git-root-relative",
          all(not g.startswith("python3 " + APP) for g in gates), str(gates))

    # NEGATIVE PROOF: same branch WITHOUT dispatcher.py -> only its own test
    p = train([t], {500: "feat/d-500"},
              {"feat/d-500": [APP + "src/other.py",
                              APP + "src/test_dispatcher_step_tick_202.py"]})
    check("RAIL 3 negative proof: no dispatcher.py -> only the touched test",
          p["gates"] == ["python3 src/test_dispatcher_step_tick_202.py"], str(p["gates"]))


# ---------------------------------------------------------------------------
# RAIL 4 -- a stale base is flagged AND holds
# ---------------------------------------------------------------------------

def t_rail_stale_base():
    t = task(600, "stale car", REVIEWED_CHAIN)
    p = train([t], {600: "feat/s-600"}, {"feat/s-600": [APP + "src/a.py"]},
              behind={"feat/s-600": 22})
    check("RAIL 4: fork-behind branch raises a hold", p["holds"], "no hold")
    check("RAIL 4: stale base refuses auto-dispatch", p["auto_dispatch"] is False)
    check("RAIL 4: hold reports the fork-behind count",
          any("22" in h for h in p["holds"]), str(p["holds"]))
    check("RAIL 4: hold warns about phantom deletions",
          any("phantom" in h for h in p["holds"]), str(p["holds"]))
    check("RAIL 4: car records fork_behind", p["cars"][0]["fork_behind"] == 22)

    # NEGATIVE PROOF: fresh base -> ships
    p = train([t], {600: "feat/s-600"}, {"feat/s-600": [APP + "src/a.py"]},
              behind={"feat/s-600": 0})
    check("RAIL 4 negative proof: fresh base auto-dispatches",
          p["auto_dispatch"] is True and not p["holds"], str(p["holds"]))


# ---------------------------------------------------------------------------
# RAIL 5 -- a car with no resolvable branch, or no diff, is refused
# ---------------------------------------------------------------------------

def t_rail_no_branch():
    t = task(700, "branchless", REVIEWED_CHAIN)
    p = train([t], {}, {})
    check("RAIL 5: unresolvable branch is refused", not p["cars"])
    check("RAIL 5: refusal explains where it looked",
          any("run_history" in why for _, why in p["refusals"]), str(p["refusals"]))

    p = train([t], {700: "feat/empty-700"}, {"feat/empty-700": []})
    check("RAIL 5: branch with no diff vs dev is refused", not p["cars"])
    check("RAIL 5: empty-diff refusal says nothing to ship",
          any("nothing to ship" in why for _, why in p["refusals"]), str(p["refusals"]))

    # NEGATIVE PROOF
    p = train([t], {700: "feat/ok-700"}, {"feat/ok-700": [APP + "src/a.py"]})
    check("RAIL 5 negative proof: resolvable branch with a diff ships",
          len(p["cars"]) == 1)


def t_rail_already_merged():
    """RAIL 6 -- a Staged car whose branch is already on dev is the #202 bug
    (shipped, R step never ticked), NOT a release. The refusal must say which."""
    t = task(750, "shipped but not ticked", REVIEWED_CHAIN)
    p = train([t], {750: "feat/shipped-750"}, {"feat/shipped-750": []},
              merged={"feat/shipped-750"})
    check("RAIL 6: already-merged branch is not shipped again", not p["cars"])
    why = dict(p["refusals"]).get("#750", "")
    check("RAIL 6: refusal says ALREADY MERGED", "ALREADY MERGED" in why, why)
    check("RAIL 6: refusal tells M to tick the step, not re-ship",
          "Tick step 4" in why and "do not re-ship" in why, why)

    # NEGATIVE PROOF: same empty diff, NOT merged -> the generic refusal
    p = train([t], {750: "feat/shipped-750"}, {"feat/shipped-750": []})
    why = dict(p["refusals"]).get("#750", "")
    check("RAIL 6 negative proof: unmerged empty diff gets the generic reason",
          "ALREADY MERGED" not in why and "nothing to ship" in why, why)


# ---------------------------------------------------------------------------
# ORDER -- docs first, code last
# ---------------------------------------------------------------------------

def t_order():
    code = task(801, "code car", REVIEWED_CHAIN)
    docs = task(802, "docs car", REVIEWED_CHAIN)
    p = train([code, docs], {801: "feat/code-801", 802: "docs/thing-802"},
              {"feat/code-801": [APP + "src/a.py"],
               "docs/thing-802": [APP + "docs/Plan.md"]})
    kinds = [c["kind"] for c in p["cars"]]
    check("ORDER: docs car sorts before the code car", kinds == ["docs", "code"], str(kinds))
    check("ORDER: classify() reads .md as docs",
          B.classify([APP + "docs/x.md"]) == "docs")
    check("ORDER: classify() reads a skill file as docs",
          B.classify([".claude/skills/preflight/SKILL.md"]) == "docs")
    check("ORDER: classify() reads a mixed diff as code",
          B.classify([APP + "docs/x.md", APP + "src/a.py"]) == "code")
    body = B.render(p)
    check("ORDER: brief table lists docs above code",
          body.index("docs/thing-802") < body.index("feat/code-801"))


# ---------------------------------------------------------------------------
# HARD LIMITS -- the constant block, and its two derived slots
# ---------------------------------------------------------------------------

def t_hard_limits():
    t = task(901, "code car", REVIEWED_CHAIN)
    p = train([t], {901: "feat/c-901"}, {"feat/c-901": [APP + "src/a.py"]})
    p["other_branches"] = ["merge/mrs5a-plus-dev-0722", "fix/mass-search-heartbeat-31"]
    body = B.render(p)
    for line in ("No rebase, no force-push, no reset, no flag flips, no `railway`",
                 "Do not arm or disarm any task",
                 "Do NOT restart the dispatcher loop",
                 "abort, comment, reassign M, stop"):
        check(f"HARD LIMITS: constant present -- {line[:38]!r}", line in body)
    check("HARD LIMITS: other worktree branches are named as hands-off",
          "merge/mrs5a-plus-dev-0722" in body and "fix/mass-search-heartbeat-31" in body)
    check("HARD LIMITS: main checkout is named as off-limits",
          "main checkout" in body)
    check("HARD LIMITS: non-dispatcher train says the loop is on the running version",
          "correctly on the running version" in body, body)

    # derived slot flips when a car touches the loop
    p = train([t], {901: "feat/c-901"}, {"feat/c-901": [APP + B.DISPATCHER_FILE]})
    p["loop_version"] = "4.25"
    body = B.render(p)
    check("HARD LIMITS: dispatcher train tells M to restart and verify",
          "M restarts and verifies" in body, body)
    check("HARD LIMITS: dispatcher train names the running version",
          "V4.25" in body)


# ---------------------------------------------------------------------------
# PROCEDURE + header shape R already executes
# ---------------------------------------------------------------------------

def t_procedure_shape():
    t = task(1001, "code car", REVIEWED_CHAIN)
    p = train([t], {1001: "feat/c-1001"}, {"feat/c-1001": [APP + "src/a.py"]})
    body = B.render(p)
    for frag in ("RELEASE BRIEF -- Wave 99, 2026-07-30",
                 "## Merge in this order",
                 "## GATING",
                 "## Procedure",
                 "## DO NOT",
                 "backup/dev-pre-wave99-0730",
                 "deadbee",
                 "all 7 services to `success`",
                 "docs/deploy-log-wave99-0730",
                 "set `assignee` to `M`, stop"):
        check(f"PROCEDURE: brief contains {frag!r}", frag in body, body[:400])
    check("PROCEDURE: one car reads 'ONE CAR'", "ONE CAR" in body)

    two = train([t, task(1002, "b", REVIEWED_CHAIN)],
                {1001: "feat/c-1001", 1002: "feat/d-1002"},
                {"feat/c-1001": [APP + "src/a.py"], "feat/d-1002": [APP + "src/b.py"]})
    check("PROCEDURE: two cars read '2 CARS'", "2 CARS" in B.render(two))


def t_frontend_note():
    t = task(1101, "frontend car", REVIEWED_CHAIN)
    p = train([t], {1101: "feat/ui-1101"},
              {"feat/ui-1101": [APP + "web/app/admin/page.tsx"]})
    check("frontend car is flagged", p["cars"][0]["frontend"] is True)
    check("frontend car gets the build-watch note",
          "frontend build goes `success`" in B.render(p))
    p = train([t], {1101: "feat/x-1101"}, {"feat/x-1101": [APP + "src/a.py"]})
    check("non-frontend car gets no build-watch note",
          "frontend build goes" not in B.render(p))


# ---------------------------------------------------------------------------
# The R task row -- AUTO vs HELD routing (Kevin's ruling, step 1)
# ---------------------------------------------------------------------------

def t_r_task_row():
    t = task(1201, "code car", REVIEWED_CHAIN)
    p = train([t], {1201: "feat/c-1201"}, {"feat/c-1201": [APP + "src/a.py"]})
    row = B.r_task_row(p, "BRIEF")
    check("R task: auto train is assigned to R", row["assignee"] == "R")
    check("R task: auto train is armed", row["ai_eligible"] is True)
    check("R task: auto train is Todo so the loop picks it up", row["status"] == "Todo")
    check("R task: description is the brief", row["description"] == "BRIEF")
    check("R task: title names the wave and the cars",
          "Wave 99" in row["title"] and "#1201" in row["title"], row["title"])

    p2 = train([t], {1201: "feat/c-1201"}, {"feat/c-1201": [APP + "src/a.py"]},
               behind={"feat/c-1201": 5})
    row = B.r_task_row(p2, "BRIEF")
    check("R task: HELD train is assigned to M", row["assignee"] == "M")
    check("R task: HELD train is NOT armed", row["ai_eligible"] is False)
    check("R task: HELD train is not Todo, so no agent dispatches it",
          row["status"] != "Todo", row["status"])


def t_empty_train():
    p = train([], {}, {})
    check("empty: no cars -> no auto-dispatch", p["auto_dispatch"] is False)
    check("empty: renders without crashing", "NO CARS" in B.render(p))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def t_paths():
    check("app_relative strips the app prefix",
          B.app_relative(APP + "src/a.py") == "src/a.py")
    check("app_relative leaves a repo-root path alone",
          B.app_relative(".claude/skills/x/SKILL.md") == ".claude/skills/x/SKILL.md")
    check("gates_for ignores non-test python",
          B.gates_for([APP + "src/engine.py"]) == [])
    check("gates_for picks up a tools/ test",
          B.gates_for([APP + "tools/preflight/test_preflight_hook_sim_194.py"])
          == ["python3 tools/preflight/test_preflight_hook_sim_194.py"])


def t_wave_guard():
    """An undecidable wave number must fail loud, never render `Wave-None`."""
    p = train([task(1301, "c", REVIEWED_CHAIN)], {1301: "feat/c-1301"},
              {"feat/c-1301": [APP + "src/a.py"]})
    p["wave"] = None
    body = B.render(p)
    check("wave guard: render() would produce Wave-None (so main() must refuse)",
          "Wave None" in body or "waveNone" in body)
    check("wave guard: main() exits 2 on an underivable wave",
          _main_with_no_wave() == 2)


def _main_with_no_wave():
    """Drive main() with every live read stubbed and the wave unknowable."""
    saved = {n: getattr(B, n) for n in
             ("git", "staged_tasks", "live_owed_logs", "live_wave",
              "live_other_branches", "live_loop_version")}
    try:
        B.git = lambda *a, **k: (0, "", "")
        B.staged_tasks = lambda: [task(1, "c", REVIEWED_CHAIN)]
        B.live_owed_logs = lambda: []
        B.live_wave = lambda: None
        B.live_other_branches = lambda c: []
        B.live_loop_version = lambda: None
        return B.main([])
    finally:
        for n, v in saved.items():
            setattr(B, n, v)


TESTS = [t_car_selection, t_rail_unreviewed, t_rail_unreviewed_taskless,
         t_rail_overlap, t_rail_overlap_append_only, t_rail_dispatcher_suites,
         t_rail_stale_base, t_rail_no_branch, t_rail_already_merged, t_order,
         t_hard_limits, t_procedure_shape, t_frontend_note, t_r_task_row,
         t_empty_train, t_paths, t_wave_guard]


def run():
    for t in TESTS:
        try:
            t()
        except Exception as e:  # a crashing rail is a failing rail
            check(f"{t.__name__} raised", False, f"{type(e).__name__}: {e}")
    print(f"\n{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print("  FAIL:", f)
    if FAIL == 0:
        print("ALL PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
