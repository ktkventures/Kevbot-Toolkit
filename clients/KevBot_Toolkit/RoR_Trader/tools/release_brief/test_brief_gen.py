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


def pr(number, head, title="a change", mergeable="MERGEABLE", draft=False):
    return {"number": number, "headRefName": head, "title": title,
            "mergeable": mergeable, "isDraft": draft}


def train(tasks, branches, diffs, behind=None, owed=(), merged=(),
          commits=None, open_prs=None, loop_staleness=None):
    rb, df, fb = fakes(branches, diffs, behind)
    return B.build_train(tasks, rb, df, fb, owed_logs=owed, wave=99,
                         base_sha="deadbee", date="2026-07-30", session="test",
                         already_merged=lambda b: b in merged,
                         commits=(lambda b: commits.get(b)) if commits else None,
                         open_prs=open_prs, loop_staleness=loop_staleness)


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
          "tick step 4" in why and "do not re-ship" in why, why)
    # #242: the refusal used to leave "tick it" as a manual instruction. It now
    # names the tool that does it with the merge verified, scoped to this one
    # car -- so the LATE safety net and the in-train sweep are one mechanism.
    check("RAIL 6: refusal names car_ship_tick with --only for this car",
          "car_ship_tick.py --only 750" in why, why)

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
    check("HARD LIMITS: non-dispatcher train says no restart is owed",
          "no restart is owed" in body, body)

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


def t_242_procedure_ticks_the_cars():
    """BOARD #242 -- the brief must tell the train to tick its CARS' ship steps.

    A train ticks its OWN chain and nothing ticks the cars', so after Waves 23,
    24 and 25 the shipped cars sat on open `Ship via a release train` steps and
    kept rendering in the shipping lane. The instruction lives in the generated
    PROCEDURE because that is the SOP R-A actually executes -- if it is not in
    the brief, it does not happen."""
    t = task(1201, "code car", REVIEWED_CHAIN)
    p = train([t], {1201: "feat/c-1201"}, {"feat/c-1201": [APP + "src/a.py"]})
    body = B.render(p)
    check("#242: the procedure names the sweep tool",
          "tools/release_brief/car_ship_tick.py" in body, body)
    check("#242: it runs AFTER the merges and BEFORE deploy-watch",
          body.index("car_ship_tick.py") < body.index("Deploy-watch"), body)
    check("#242: the brief says why (a train never ticks its cars')",
          "ticks its OWN chain" in body, body)
    check("#242: a non-zero exit is not allowed to pass quietly",
          "could not be verified" in body, body)
    # The ABORT path is the one that would otherwise leave a PARTIAL train's
    # merged cars stale -- R-A hands back on a red gate and never reaches the
    # end of the procedure, so the hard limits must send it through the sweep.
    check("#242: the abort path runs the sweep when a merge already landed",
          "if ANY merge already landed" in body, body)


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
              "live_other_branches", "live_loop_version", "live_open_prs",
              "live_loop_staleness")}
    try:
        B.git = lambda *a, **k: (0, "", "")
        B.staged_tasks = lambda: [task(1, "c", REVIEWED_CHAIN)]
        B.live_owed_logs = lambda: []
        B.live_wave = lambda: None
        B.live_other_branches = lambda c: []
        B.live_loop_version = lambda: None
        B.live_open_prs = lambda: None            # no `gh` in the test process
        B.live_loop_staleness = lambda: None      # no live loop either
        return B.main([])
    finally:
        for n, v in saved.items():
            setattr(B, n, v)


# ---------------------------------------------------------------------------
# M-REVIEW GAPS (07-30) -- each of these was a real miss when the generator was
# replayed against the hand-written Wave 14 / Wave 16 briefs. Every check below
# FAILS against the pre-review generator.
# ---------------------------------------------------------------------------

NO_R_CHAIN = [
    step("kevin", "Approve — STAMP"),
    step("M", "Build the thing"),
    step("M", "M review — judge the rails"),
    step("M", "Prove it on a real task, then ship", done=False),
]


def t_gap_no_r_step_is_loud():
    """GAP 1: Waves 14 and 16 each carried a car whose chain named no R step
    (tasks #200 and #194). The generator dropped both SILENTLY."""
    t = task(200, "task-chain skill", NO_R_CHAIN)
    p = train([t], {200: "docs/skills-200"},
              {"docs/skills-200": [APP + "docs/a.md"]})
    check("GAP 1: a Staged task with no R step is not a car", not p["cars"])
    check("GAP 1: ...and is REFUSED OUT LOUD, not dropped silently",
          any(who == "#200" for who, _ in p["refusals"]), str(p["refusals"]))
    check("GAP 1: the refusal says what to do about it",
          any("NO R step" in why and "task-chain" in why for _, why in p["refusals"]),
          str(p["refusals"]))
    check("GAP 1: the refusal is rendered into the brief", "#200" in B.render(p))

    # NEGATIVE PROOF: a chain that DOES name R, still mid-chain, stays quiet --
    # otherwise every in-flight task would spam the brief.
    mid = [step("kevin", "Approve — STAMP"), step("M", "Build", done=False),
           step("R", "Ship with a release train", done=False)]
    p = train([task(201, "mid-chain", mid)], {201: "feat/x-201"},
              {"feat/x-201": [APP + "src/a.py"]})
    check("GAP 1 negative proof: a mid-chain task with an R step stays quiet",
          not p["cars"] and not p["refusals"], str(p["refusals"]))


def t_gap_suite_list_tracks_reality():
    """GAP 2: DISPATCHER_SUITES is a constant. Replaying Wave 14 named
    `test_dispatcher_step_tick_202.py`, which #202 had not yet added -- R would
    have aborted on a missing file."""
    gone = B.DISPATCHER_SUITES[0]
    live = B.resolve_dispatcher_suites(lambda p: p != gone)
    check("GAP 2: a suite that no longer exists is dropped", gone not in live)
    check("GAP 2: the surviving suites keep the constant's order",
          live == [s for s in B.DISPATCHER_SUITES if s != gone], str(live))

    newcomer = "src/test_dispatcher_brand_new_999.py"
    live = B.resolve_dispatcher_suites(lambda p: True, discovered={newcomer})
    check("GAP 2: a suite the constant does not name is appended",
          newcomer in live and live[-1] == newcomer, str(live))
    check("GAP 2: a non-suite file is never appended as a gate",
          B.resolve_dispatcher_suites(lambda p: True,
                                      discovered={"src/local_update.py"})
          == list(B.DISPATCHER_SUITES))

    # ...and the resolved set is what a dispatcher car actually gates on.
    t = task(500, "dispatcher change", REVIEWED_CHAIN)
    rb, df, fb = fakes({500: "feat/d-500"},
                       {"feat/d-500": [APP + B.DISPATCHER_FILE]})
    p = B.build_train([t], rb, df, fb, wave=99, base_sha="deadbee",
                      date="2026-07-30", suites=["src/test_only_this.py"])
    check("GAP 2: build_train gates on the RESOLVED set, not the constant",
          p["gates"] == ["python3 src/test_only_this.py"], str(p["gates"]))


def t_gap_owed_log_stale_base_warns():
    """GAP 3: an owed deploy-log is stale BY CONSTRUCTION -- that is what owed
    means. Holding on it made the generator refuse the Wave-14 backlog case it
    exists to clear."""
    log = "docs/deploy-log-wave11-0730"
    rb, df, fb = fakes({}, {log: [APP + "docs/Deploy_Log.md"]}, behind={log: 8})
    p = B.build_train([], rb, df, fb, owed_logs=[log], wave=99,
                      base_sha="deadbee", date="2026-07-30")
    check("GAP 3: a stale owed deploy-log does NOT hold the train",
          not p["holds"], str(p["holds"]))
    check("GAP 3: it still auto-dispatches", p["auto_dispatch"] is True)
    check("GAP 3: the staleness is still stated, as a warning",
          any("fork-behind 8" in w for w in p["warns"]), str(p["warns"]))
    check("GAP 3: the warning still tells R to abort on a real conflict",
          any("ABORT" in w for w in p["warns"]), str(p["warns"]))

    # NEGATIVE PROOF: the exemption is scoped to append-only log paths only. A
    # deploy-log branch that also touches code holds like anything else.
    rb, df, fb = fakes({}, {log: [APP + "docs/Deploy_Log.md", APP + "src/a.py"]},
                       behind={log: 8})
    p = B.build_train([], rb, df, fb, owed_logs=[log], wave=99,
                      base_sha="deadbee", date="2026-07-30")
    check("GAP 3 negative proof: a log branch that also touches code still holds",
          p["holds"] and p["auto_dispatch"] is False, str(p["holds"]))


def t_gap_loop_version_is_read():
    """GAP 4: `live_loop_version()` returned None against the REAL banner -- the
    old regex required the word "version" on the line, and the banner reads
    `Team dispatcher (V4.25) - dispatches ...`. Every brief silently lost it."""
    real = '"""Team dispatcher (V4.25) — dispatches board tasks to headless agents.\n'
    check("GAP 4: the real banner shape parses", B._banner_version(real) == "4.25",
          repr(B._banner_version(real)))
    check("GAP 4: a file with no banner returns None",
          B._banner_version("import os\n") is None)
    check("GAP 4: only the HEAD of the file is scanned",
          B._banner_version("\n" * 40 + "V9.9\n") is None)

    # The version is still READ (that was gap 4's defect: it was silently lost)
    # but it is INFORMATIONAL only -- see t_gap224_restart_trigger for why a
    # comparison between banners can never be the restart trigger.
    t = task(902, "loop change", REVIEWED_CHAIN)
    p = train([t], {902: "feat/l-902"}, {"feat/l-902": [APP + B.DISPATCHER_FILE]})
    p["loop_version"] = "4.24"
    body = B.render(p)
    check("GAP 4: the brief still states the running loop's version",
          "V4.24" in body, body)


def t_gap_armed_state_is_checked():
    """GAP 5: the brief asserted "nothing in this train needs arming" without
    looking. Both hand-written briefs named the armed task explicitly."""
    t = dict(task(903, "armed car", REVIEWED_CHAIN), ai_eligible=True)
    p = train([t], {903: "feat/a-903"}, {"feat/a-903": [APP + "src/a.py"]})
    check("GAP 5: the car records its armed state", p["cars"][0]["armed"] is True)
    body = B.render(p)
    check("GAP 5: an armed car is named in DO NOT",
          "#903" in body.split("## DO NOT")[1], body)
    check("GAP 5: ...and R is told not to disarm it",
          "ARMED" in body and "disarm" in body)

    # NEGATIVE PROOF: unarmed car -> the flat constant, no false claim either way
    t = dict(task(903, "unarmed car", REVIEWED_CHAIN), ai_eligible=False)
    p = train([t], {903: "feat/a-903"}, {"feat/a-903": [APP + "src/a.py"]})
    body = B.render(p)
    check("GAP 5 negative proof: unarmed train says no car is armed",
          "No car in this train is armed" in body and "ARMED**" not in body, body)


def t_gap_nonoverlapping_paths_still_stated():
    """GAP 6: the per-car path summary was suppressed whenever ANY overlap
    existed -- i.e. exactly the briefs that needed it (Wave 14 spelled out that
    #141 and #142 did not touch the logs)."""
    logs = ["docs/deploy-log-wave11-0730", "docs/deploy-log-wave12-0730"]
    t = task(197, "reaper fix", REVIEWED_CHAIN)
    rb, df, fb = fakes({197: "fix/r-197"},
                       {logs[0]: [APP + "docs/Deploy_Log.md"],
                        logs[1]: [APP + "docs/Deploy_Log.md"],
                        "fix/r-197": [APP + B.DISPATCHER_FILE]})
    p = B.build_train([t], rb, df, fb, owed_logs=logs, wave=99,
                      base_sha="deadbee", date="2026-07-30")
    body = B.render(p)
    check("GAP 6: overlapping cars are still called out",
          "WILL conflict with each other" in body, body)
    check("GAP 6: the NON-overlapping car's paths are stated too",
          "do **not** overlap" in body and "#197" in body.split("## GATING")[0], body)


# ---------------------------------------------------------------------------
# BOARD #224 (07-30) -- three gaps found by running the generator on the two real
# trains of Waves 19 and 20. Each check below FAILS against the pre-#224
# generator; each is the generator REASONING about a change instead of READING it.
# ---------------------------------------------------------------------------

def t_gap224_prose_from_the_diff():
    """GAP 1: per-car prose came from the board task TITLE. Wave 19's cargo line
    named `DAILY_CAP` and shipped `CONCURRENCY 3->4` UNDESCRIBED -- the change
    was in the diff and gated correctly; only the summary a reviewer trusts was
    wrong."""
    t = task(219, "Raise the dispatcher DAILY_CAP to 40", REVIEWED_CHAIN)
    wave19 = ["f4e8a63 chore(dispatcher): DAILY_CAP 20 -> 40",
              "61dac54 chore(dispatcher): CONCURRENCY 3 -> 4"]
    p = train([t], {219: "chore/daily-cap-40-219"},
              {"chore/daily-cap-40-219": [APP + B.DISPATCHER_FILE]},
              commits={"chore/daily-cap-40-219": wave19})
    body = B.render(p)
    check("GAP 1: the car records the commits it carries",
          p["cars"][0]["commits"] == wave19, str(p["cars"][0]["commits"]))
    check("GAP 1: a commit absent from the task title is STILL described",
          "CONCURRENCY 3 -> 4" in body, body)
    check("GAP 1: the task title is still the header",
          "board #219 -- Raise the dispatcher DAILY_CAP to 40" in body, body)
    check("GAP 1: a multi-commit car is flagged as multi-commit",
          p["cars"][0]["multi_commit"] is True)
    check("GAP 1: ...and the flag is visible to a reader, not just in the plan",
          "2 commits" in body, body)
    check("GAP 1: the flag says the car carries more than its title",
          "more than its title says" in body, body)

    # NEGATIVE PROOF: one commit -> no multi-commit alarm, but still described.
    p = train([t], {219: "chore/daily-cap-40-219"},
              {"chore/daily-cap-40-219": [APP + B.DISPATCHER_FILE]},
              commits={"chore/daily-cap-40-219": [wave19[0]]})
    body = B.render(p)
    check("GAP 1 negative proof: a single-commit car is not flagged",
          p["cars"][0]["multi_commit"] is False and "2 commits" not in body, body)
    check("GAP 1 negative proof: its one commit is still spelled out",
          "DAILY_CAP 20 -> 40" in body, body)

    # UNKNOWN != empty: a failed `git log` must not read as "carries nothing".
    p = train([t], {219: "chore/daily-cap-40-219"},
              {"chore/daily-cap-40-219": [APP + B.DISPATCHER_FILE]},
              commits={"chore/daily-cap-40-219": None})
    check("GAP 1: an unreadable commit list says so instead of claiming empty",
          p["cars"][0]["commits"] is None)

    # ...and with no commit facts injected at all the section is simply absent.
    p = train([t], {219: "chore/daily-cap-40-219"},
              {"chore/daily-cap-40-219": [APP + "src/a.py"]})
    check("GAP 1: no cargo section when commits were never read",
          "## What each car carries" not in B.render(p))


def t_gap224_restart_trigger_is_dispatcher_touch():
    """GAP 2: the restart trigger compared version banners. Wave 19 carried
    `V4.25` on BOTH sides of the merge, so no comparison could ever have exposed
    the drift. The trigger is "a car touched `dispatcher.py`"."""
    t = task(219, "constant-only dispatcher change", REVIEWED_CHAIN)

    # THE WAVE 19 SHAPE: a real dispatcher change, identical banner both sides.
    p = train([t], {219: "chore/daily-cap-40-219"},
              {"chore/daily-cap-40-219": [APP + B.DISPATCHER_FILE]})
    p["loop_version"] = "4.25"          # and the branch's banner is 4.25 too
    body = B.render(p)
    check("GAP 2: a dispatcher-touching car demands a restart even when the "
          "banner did not move", "M restarts and verifies" in body, body)
    check("GAP 2: the brief says the trigger is the touch, not a version delta",
          "NEVER a version comparison" in body, body)
    check("GAP 2: the generator no longer computes a target version",
          not hasattr(B, "live_target_version"))
    check("GAP 2: ...and no plan carries one", "target_version" not in p)

    # NEGATIVE PROOF: a same-banner merge that touches NO dispatcher file must
    # NOT set the restart flag -- otherwise every train would claim a restart.
    p = train([t], {219: "feat/elsewhere-219"},
              {"feat/elsewhere-219": [APP + "src/a.py"]})
    p["loop_version"] = "4.25"
    body = B.render(p)
    check("GAP 2 negative proof: a non-dispatcher car owes no restart",
          "no restart is owed" in body and "M restarts and verifies" not in body, body)

    # The check that DOES work for a constant-only train: process start time vs.
    # the time the checkout last wrote the file the process loaded.
    stale = {"loop_started": 1_000_000, "file_mtime": 1_014_400, "stale": True}
    p = train([t], {219: "feat/elsewhere-219"},
              {"feat/elsewhere-219": [APP + "src/a.py"]}, loop_staleness=stale)
    body = B.render(p)
    check("GAP 2: an ALREADY-stale running loop is called out even with no "
          "dispatcher car", "ALREADY stale" in body, body)
    check("GAP 2: ...and the staleness is stated as a time delta",
          "4h" in body, body)
    fresh = {"loop_started": 1_014_400, "file_mtime": 1_000_000, "stale": False}
    p = train([t], {219: "feat/elsewhere-219"},
              {"feat/elsewhere-219": [APP + "src/a.py"]}, loop_staleness=fresh)
    check("GAP 2 negative proof: a loop started AFTER its file is not flagged",
          "ALREADY stale" not in B.render(p))

    # A pull landing SECONDS after the process started (measured live 07-30: 36s)
    # is as likely to BE the restart as to have missed it. Report the two
    # timestamps; do not manufacture a staleness claim out of race noise.
    noise = {"loop_started": 1_000_000, "file_mtime": 1_000_036, "stale": True}
    p = train([t], {219: "feat/elsewhere-219"},
              {"feat/elsewhere-219": [APP + "src/a.py"]}, loop_staleness=noise)
    body = B.render(p)
    check("GAP 2: a sub-margin delta is NOT asserted as staleness",
          "ALREADY stale" not in body and "NOT a staleness claim" in body, body)
    check("GAP 2: a 36s delta renders as seconds, never as '0m'",
          "36s" in body and "0m" not in body, body)


def t_gap224_zero_cars_are_loud():
    """GAP 3, the worst of the three: Wave 20 returned a clean `0 car(s)` with
    two mergeable PRs open (#154, #155). `0 car(s)` reads identically to
    "nothing to ship" -- the same silent shape #211 was built to fix."""
    wave20 = [pr(154, "feat/m-session-dashboard-220", "M session dashboard"),
              pr(155, "feat/agent-registry-m-split-222", "agent registry M split")]
    p = train([], {}, {}, open_prs=wave20)
    check("GAP 3: zero cars + unclaimed mergeable PRs is SUSPICIOUS",
          p["zero_car_verdict"] == "SUSPICIOUS", str(p["zero_car_verdict"]))
    check("GAP 3: both unclaimed PRs are carried in the plan",
          len(p["unclaimed_prs"]) == 2)
    body = B.render(p)
    check("GAP 3: the verdict is LOUD, not a bare zero",
          "ZERO CARS -- AND THAT IS SUSPICIOUS" in body, body)
    check("GAP 3: it explicitly refuses the 'nothing to ship' reading",
          "NOTHING TO SHIP" in body, body)
    check("GAP 3: it NAMES the PRs by number", "PR #154" in body and "PR #155" in body,
          body)
    check("GAP 3: it names their branches too",
          "feat/m-session-dashboard-220" in body, body)
    check("GAP 3: it explains WHY they were invisible",
          "status=eq.Staged" in body, body)

    # NEGATIVE PROOF 1: a claimed PR is not unclaimed.
    t = task(220, "M session dashboard", REVIEWED_CHAIN)
    p = train([t], {220: "feat/m-session-dashboard-220"},
              {"feat/m-session-dashboard-220": [APP + "src/a.py"]},
              open_prs=[wave20[0]])
    check("GAP 3 negative proof: a PR whose branch IS a car is claimed",
          not p["unclaimed_prs"] and p["zero_car_verdict"] is None,
          str(p["unclaimed_prs"]))

    # NEGATIVE PROOF 2: PRs listed, none shippable -> a genuinely clean zero.
    p = train([], {}, {}, open_prs=[pr(9, "feat/x", mergeable="CONFLICTING"),
                                    pr(10, "feat/y", draft=True)])
    check("GAP 3 negative proof: conflicting and draft PRs do not raise the alarm",
          p["zero_car_verdict"] == "CLEAN" and not p["unclaimed_prs"],
          str(p["zero_car_verdict"]))
    check("GAP 3 negative proof: a checked zero says it was CHECKED",
          "ZERO CARS -- checked" in B.render(p))
    p = train([], {}, {}, open_prs=[])
    check("GAP 3 negative proof: no open PRs at all is also a clean zero",
          p["zero_car_verdict"] == "CLEAN")

    # NEGATIVE PROOF 3: the check itself failing must NOT read as clean.
    p = train([], {}, {})                       # open_prs=None == never asked
    check("GAP 3: an unrun PR check is UNVERIFIED, never CLEAN",
          p["zero_car_verdict"] == "UNVERIFIED" and p["pr_check"] == "unavailable")
    check("GAP 3: ...and the brief says the emptiness is unproven",
          "UNPROVEN" in B.render(p), B.render(p))
    check("GAP 3: mergeability UNKNOWN is kept, not silently dropped",
          len(B.unclaimed_prs([pr(11, "feat/z", mergeable="UNKNOWN")], set())) == 1)

    # ...and unclaimed PRs are surfaced even when the train is NOT empty.
    p = train([t], {220: "feat/m-session-dashboard-220"},
              {"feat/m-session-dashboard-220": [APP + "src/a.py"]},
              open_prs=wave20)
    body = B.render(p)
    check("GAP 3: an unclaimed PR is surfaced on a non-empty train too",
          "PR #155" in body and "NOT in this train" in body, body)


TESTS = [t_car_selection, t_rail_unreviewed, t_rail_unreviewed_taskless,
         t_rail_overlap, t_rail_overlap_append_only, t_rail_dispatcher_suites,
         t_rail_stale_base, t_rail_no_branch, t_rail_already_merged, t_order,
         t_hard_limits, t_procedure_shape, t_242_procedure_ticks_the_cars,
         t_frontend_note, t_r_task_row,
         t_empty_train, t_paths, t_wave_guard,
         t_gap_no_r_step_is_loud, t_gap_suite_list_tracks_reality,
         t_gap_owed_log_stale_base_warns, t_gap_loop_version_is_read,
         t_gap_armed_state_is_checked, t_gap_nonoverlapping_paths_still_stated,
         t_gap224_prose_from_the_diff, t_gap224_restart_trigger_is_dispatcher_touch,
         t_gap224_zero_cars_are_loud]


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
