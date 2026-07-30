#!/usr/bin/env python3
"""Release-brief GENERATOR — board #211.

`Staged -> shipped` was the last hand-authored hop in the pipeline. Everything a
release brief needs is already structured (the board) or derivable (git), so this
generates the brief M used to type: cars, order, gate commands, conflict
warnings, procedure, hard limits.

KEVIN'S RULING (task #211 step 1, 07-30): *generate + auto-dispatch*, with two
constraints enforced mechanically rather than by attention:

  1. a car whose M-review step is not ticked can NEVER enter a train (REFUSED);
  2. a risk the generator cannot resolve -- overlapping paths between cars, a
     stale base, two cars touching `dispatcher.py` -- REFUSES AUTO-DISPATCH and
     hands the brief to M instead.

So there are two distinct verdicts and they are not the same thing:

  * a **refusal** drops one car out of the train (the rest still ship);
  * a **hold** keeps the whole train, brief and all, but routes it to M.

WHAT THIS TOOL WILL NEVER DO: merge, push, rebase, flip a flag, run a gate, or
touch a worktree. It reads the board, reads git, prints a brief, and -- only with
`--create-task` -- writes ONE `dev_tasks` row. Everything else is R's job, and R
is the only thing that touches `dev`.

Usage (from the app root, `clients/KevBot_Toolkit/RoR_Trader`):

    python3 tools/release_brief/brief_gen.py                 # print the brief + verdict
    python3 tools/release_brief/brief_gen.py --json          # machine-readable plan
    python3 tools/release_brief/brief_gen.py --create-task   # also create the R task

The core is pure: `build_train()` takes the board rows and three injected
callables (branch resolver / differ / base checker) so every rail is testable
without a database or a git repo. See `test_brief_gen.py`.

M REVIEW, 07-30 (task #211 step 3). The generator was replayed against the real
board + git state of Waves 14 and 16 and diffed line-by-line against the briefs M
hand-wrote for them. Six gaps were found and FIXED -- see the `t_gap_*` rails, each
of which fails against the pre-review generator:

  1. a `Staged` task whose chain names no R step was dropped SILENTLY (2 of the 5
     real cars across the two waves: #200, #194) -> now refused OUT LOUD;
  2. `DISPATCHER_SUITES` was a constant, so replaying Wave 14 gated on a test #202
     had not yet added -> resolved against the merged tip instead;
  3. an owed deploy-log is stale BY CONSTRUCTION, so the stale-base HOLD fired on
     the very backlog case this tool exists to clear -> warns for append-only logs;
  4. `live_loop_version()` returned None against the real banner, so every brief
     silently lost the version -> fixed (the version is INFORMATIONAL only; see
     board #224 gap 2 for why it can never be the restart trigger);
  5. "nothing needs arming" was asserted without looking -> reads `ai_eligible`;
  6. the per-car path summary vanished whenever any overlap existed -> always shown.

After the fixes the derived GATE BLOCK is byte-identical (commands and order) to
the hand-written one for both waves.

BOARD #224, 07-30 -- three more gaps, found by running the generator on Waves 19
and 20. None broke a train; all three are the generator REASONING about a change
instead of READING it:

  1. per-car prose came from the board task TITLE, so Wave 19's cargo line named
     `DAILY_CAP` and shipped `CONCURRENCY 3->4` UNDESCRIBED -> the prose is now
     derived from `git log --oneline origin/dev..<branch>`: the task title is the
     header, the commit subjects are the contents, and a multi-commit car is
     flagged so a reviewer sees there is more than one thing in it;
  2. the restart trigger compared version banners. Wave 19 carried `V4.25` on
     BOTH sides of the merge, so no comparison could ever have exposed the drift
     -> the trigger is now "a car touched `dispatcher.py`", full stop, and the
     check that works for a constant-only train is process-start-time vs.
     checkout-pull-time (`live_loop_staleness()`);
  3. zero cars were SILENT. Wave 20 returned a clean `0 car(s)` with two
     mergeable PRs open, because the entry point is `dev_tasks?status=eq.Staged`
     and a branch with no board task never enters the candidate set -- which is
     exactly the class of work M authors directly. `0 car(s)` read identically to
     "nothing to ship" -> a zero-car verdict is now UNVERIFIED (open PRs could
     not be listed), SUSPICIOUS (open mergeable PRs no car claims -- named), or
     CLEAN, and never a bare zero.

ACCEPTED LIMITATIONS -- known, deliberate, and M's job to fill in when they matter:

  * per-gate RESULTS (`# 74 checks passed`). The generator must never run a gate,
    so it says "all green at M's review" instead of quoting numbers.
  * the `#` column is the BOARD task id, not the GitHub PR number the hand-written
    briefs used. The Kind column disambiguates (`board #197 -- ...`). Resolving real
    PR numbers would add `gh` as a third data source; branch + board is enough for
    R, which merges by branch.
  * narrative context ("three logs are owed because trains ran back-to-back",
    "#195's and #198's suites are the load-bearing ones"). Judgement, not data.
  * armed NON-car tasks. Wave 14's brief warned that #202 -- not a car -- was armed.
    Only the cars' `ai_eligible` is read here; a fleet-wide armed inventory is
    `/preflight`'s job, not the brief's.
  * whether a car already HAS a PR. Task cars always print their board id; R opens
    a PR if one is missing, which its own SOP already covers.
  * ASCII punctuation (`--`, `->`) rather than em-dashes and arrows.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(os.path.dirname(HERE))          # .../RoR_Trader
APP_PREFIX = "clients/KevBot_Toolkit/RoR_Trader/"
ENV = f"{APP_ROOT}/src/.env"

# Git reads run against WHICHEVER checkout this script was invoked from -- every
# worktree shares the same object store and refs, so branch/diff facts are
# identical, and reading from here means the generator never has to reach into
# the main checkout (a live dispatcher loop runs from it).
GIT_ROOT = subprocess.run(["git", "-C", APP_ROOT, "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip() or APP_ROOT
# The RUNNING loop's version, on the other hand, is only knowable from the main
# checkout -- that is where the live loop executes from.
MAIN_CHECKOUT = "/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader"

API_TIMEOUT_S = 15
GIT_TIMEOUT_S = 120
GH_TIMEOUT_S = 45
BASE = "origin/dev"

# ---------------------------------------------------------------------------
# Constants that were byte-identical in every brief M hand-wrote on 07-29/30.
# ---------------------------------------------------------------------------

DISPATCHER_FILE = "tools/team_dispatcher/dispatcher.py"

# The deployed API the release brief tells R to PROBE after deploy-watch. Read from
# the dispatcher's own constant so the two cannot drift; the literal is only the
# fallback for a checkout where the dispatcher cannot be parsed. Board #237.
def _board_api_host():
    try:
        src = open(os.path.join(APP_ROOT, DISPATCHER_FILE), encoding="utf-8").read()
        m = re.search(r'BOARD_API_URL\s*=\s*os\.getenv\([^)]*?"(https://[^"]+)"', src) \
            or re.search(r'BOARD_API_URL\s*=\s*"(https://[^"]+)"', src)
        if m:
            return m.group(1).rstrip("/")
    except Exception:
        pass
    return "https://api-dev-2c9d.up.railway.app"


API_HOST = _board_api_host()

# The full suite set, in the order M ran it. Any car that touches the dispatcher
# pulls ALL of these, not just its own test: the dispatcher is one module that
# four branches edited in a week, and #195's/#198's suites are the ones that
# prove a union merge dropped neither feature.
DISPATCHER_SUITES = (
    "src/test_dispatcher_step_tick_202.py",
    "src/test_dispatcher_failed_run_status_197.py",
    "src/test_dispatcher_push_at_run_end_195.py",
    "src/test_dispatcher_ai_eligible_gate_198.py",
    "src/test_dispatcher_inbound_check_182.py",
    "src/test_dispatcher_step_level_182.py",
    "src/test_dispatcher_loud_skips_171.py",
    "src/test_mentions_agent_side_143.py",
    "tools/team_dispatcher/test_dispatcher_reap.py",
)

# Files whose "conflict" is always the same shape -- two dated entries wanting
# the same spot -- and therefore resolvable oldest-first without judgement.
# This is the Wave-14 precedent (three owed Deploy_Log cars, merged 11->12->13).
# Overlap on ANYTHING ELSE is a hold, because M has no recorded resolution for it.
APPEND_ONLY_PATHS = ("docs/Deploy_Log.md",)

# A taskless car is normally refused -- an unreviewed branch must not ship. The
# ONE exemption is the deploy-log R leaves behind on purpose: R writes it, opens
# a PR and deliberately leaves it unmerged, so the next train owes it. It has no
# board task by design.
OWED_LOG_RE = re.compile(r"^docs/deploy-log-wave\d+-\d{4}$")

DOCS_PREFIXES = ("docs/", ".claude/", "clients/KevBot_Toolkit/RoR_Trader/docs/")

# How far apart the loop's process-start and its `dispatcher.py` mtime must be
# before "the file is newer" means anything. Measured live on 07-30: 36s, which
# is a restart landing next to a pull, not a stale process. Inside the margin the
# brief reports both timestamps and asks M to confirm; outside it, it claims.
LOOP_STALE_MARGIN_S = 120

BRANCH_TOKEN_RE = re.compile(
    r"\b((?:feat|fix|docs|chore|refactor|test|perf|merge|verify|spec)/[A-Za-z0-9._\-/]+)")

# Every dispatcher/mentions suite lives under one of these two dirs with this name
# shape. Used to REPLACE the constant above with what actually exists at the merged
# tip -- the constant went stale the moment #202 added a tenth suite, and a brief
# that names a file R cannot run is a false abort.
SUITE_RE = re.compile(
    r"^(?:src|tools/team_dispatcher)/test_(?:dispatcher|mentions_agent)[A-Za-z0-9_]*\.py$")

# The hard-limits block. Three derived slots; the rest is the constant.
HARD_LIMITS = """## DO NOT
- **Do NOT restart the dispatcher loop.** {loop_note}
- {arm_note}
- Do not touch {other_branches}.
- No rebase, no force-push, no reset, no flag flips, no `railway`.
- Do not work in the main checkout (`{app_root}`) -- a live dispatcher loop runs from it.

Any gate red / CI red / unexpected conflict -> abort, comment, reassign M, stop."""

PROCEDURE = """## Procedure
1. Backup branch `backup/dev-pre-wave{wave}-{mmdd}` from current `origin/dev` (`{base_sha}`).
2. Breadcrumb before AND after every merge.
3. Merge in the order above, verifying each lands before the next.
4. **Deploy-watch:** all 7 services to `success` on the final head. Do not stop at `pending`.
5. **PROBE THE SERVICE, do not trust the deploy status.** `GET {api_host}/health` must return
   **200**, and `{api_host}/api/openapi.json` must return **200**. A 401 from `/api/dev-tasks`
   is CORRECT (router loaded, admin-gated); a 5xx or a timeout is not.
   **This step is not a formality.** On Wave 24 (07-30) deploy-watch reported **7/7 `success`
   while the api was serving 502** -- a container that Railway launched successfully and that
   then crash-looped on import. Deploy status answers "did Railway accept it"; only a probe
   answers "does the service work". If the probe fails, say so LOUDLY and hand back to M --
   the train is not done, whatever the deploy dashboard says.
6. Wave-{wave} `Deploy_Log.md` entry on `docs/deploy-log-wave{wave}-{mmdd}`, PR opened, left UNMERGED.
7. Report, set `assignee` to `M`, stop."""

HEADER = """**RELEASE BRIEF -- Wave {wave}, {date}. {car_count}.**

Generated by `tools/release_brief/brief_gen.py` (board #211) for M ({session}). \
Charter §8 default cadence: everything here is reviewed and ready NOW."""


# ---------------------------------------------------------------------------
# Chain reading -- the same shape the dispatcher reads (#182 process chains).
# ---------------------------------------------------------------------------

def _step_owner(step):
    o = step.get("owner")
    return o if o is not None else step.get("role")


def _step_title(step):
    t = step.get("title")
    return t if t is not None else (step.get("text") or "")


def r_step_index(task):
    """Index of the first incomplete step when that step is R-owned, else None.

    "Current chain step is R-owned" is the car definition from the SOP. Reading
    the FIRST incomplete step (not "any R step") is what makes it mean
    `reviewed, waiting for a train` -- every earlier step is ticked by
    construction."""
    cl = task.get("checklist") or []
    if not isinstance(cl, list):
        return None
    for i, s in enumerate(cl):
        if not isinstance(s, dict):
            continue
        if not s.get("done"):
            return i if (_step_owner(s) or "").upper() == "R" else None
    return None


def has_r_step(task):
    """True when the chain names an R step at all, ticked or not.

    A `Staged` task with NO R step anywhere is the gap M review found on 07-30:
    two of the five real cars across Waves 14 and 16 (task #200's skills branch,
    task #194's preflight branch) were exactly this, and `r_step_index()` returns
    None for them — indistinguishable from a task still mid-chain. 44 of the 52
    chained tasks on the board are in this class, so it is the common case, not
    an edge one. It must be REFUSED OUT LOUD, never dropped silently."""
    for s in task.get("checklist") or []:
        if isinstance(s, dict) and (_step_owner(s) or "").upper() == "R":
            return True
    return False


def review_step(task, r_idx):
    """The ticked M-owned review step in front of the R step, or None.

    RAIL 1 (Kevin's constraint 1). `r_step_index()` already implies every
    earlier step is ticked, so the teeth here are on chains that have NO M
    review step at all: those are exactly the ones where nobody read the build,
    and they must not enter a train."""
    cl = task.get("checklist") or []
    for s in cl[:r_idx]:
        if not isinstance(s, dict):
            continue
        if (_step_owner(s) or "").upper() != "M":
            continue
        if "review" not in _step_title(s).lower():
            continue
        if s.get("done"):
            return s
    return None


# ---------------------------------------------------------------------------
# Derivation from the diff
# ---------------------------------------------------------------------------

def app_relative(path):
    """Gate commands run from the app root; git reports paths from the git root."""
    return path[len(APP_PREFIX):] if path.startswith(APP_PREFIX) else path


def is_docs_path(path):
    return path.startswith(DOCS_PREFIXES) or path.endswith(".md")


def classify(paths):
    """`docs` when nothing in the diff can break at runtime, else `code`.

    Drives merge ORDER: docs first, code last, so a code failure leaves the docs
    landed instead of rolling the whole train back."""
    return "docs" if paths and all(is_docs_path(p) for p in paths) else "code"


def resolve_dispatcher_suites(exists, discovered=()):
    """The suite set as it exists AT THE MERGED TIP, not as of the day the
    constant was typed.

    `DISPATCHER_SUITES` supplies the ORDER M ran them in and nothing else: any
    entry that no longer exists is dropped (R would abort on a missing file --
    replaying Wave 14 named `test_dispatcher_step_tick_202.py`, which #202 had
    not yet added), and any suite present at the tip that the constant does not
    name is appended (so the next dispatcher test is gated the day it lands)."""
    out = [t for t in DISPATCHER_SUITES if exists(t)]
    for t in sorted(discovered):
        if t not in out and SUITE_RE.match(t) and exists(t):
            out.append(t)
    return out


def gates_for(paths, suites=DISPATCHER_SUITES):
    """Gate commands derivable from the diff.

    Two sources, in order: the full dispatcher suite set whenever
    `dispatcher.py` is touched (RAIL 3), then any test file in the diff that the
    suite list did not already cover."""
    cmds = []
    rel = [app_relative(p) for p in paths]
    if DISPATCHER_FILE in rel:
        cmds.extend(f"python3 {t}" for t in suites)
    for p in rel:
        base = os.path.basename(p)
        if not (base.startswith("test_") and base.endswith(".py")):
            continue
        cmd = f"python3 {p}"
        if cmd not in cmds:
            cmds.append(cmd)
    return cmds


def frontend_touched(paths):
    return any(p.endswith((".tsx", ".ts", ".jsx", ".css")) for p in paths)


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

def overlaps(cars):
    """{path: [car labels]} for every path more than one car touches."""
    seen = {}
    for c in cars:
        for p in c["paths"]:
            seen.setdefault(p, []).append(c["label"])
    return {p: labels for p, labels in sorted(seen.items()) if len(labels) > 1}


def assess(cars):
    """(holds, warns) -- RAIL 2 and RAIL 4.

    A HOLD is a risk with no recorded resolution, so the train still gets a
    brief but M reads it before R runs. A WARN is a risk M has already resolved
    once and written down (append-only log conflicts), so it goes in the brief
    as an instruction and does not block."""
    holds, warns = [], []

    ov = overlaps(cars)
    for path, labels in ov.items():
        who = " + ".join(labels)
        if app_relative(path) in APPEND_ONLY_PATHS:
            warns.append(
                f"**{who} all touch `{app_relative(path)}` -- they WILL conflict with each "
                f"other.** Merge them oldest-first; each appends its own dated entry, so "
                f"resolution is append-order only. **If a conflict is anything other than "
                f"two entries wanting the same spot, ABORT and hand to M.**")
        else:
            holds.append(
                f"{who} both touch `{app_relative(path)}` -- overlapping paths with no "
                f"recorded resolution. M reconciles before this train runs.")

    for c in cars:
        if not c["fork_behind"]:
            continue
        msg = (f"{c['label']} (`{c['branch']}`) is cut from a STALE base (fork-behind "
               f"{c['fork_behind']}). Its `git diff origin/dev` shows phantom deletions; "
               f"verify with `git show --stat`, not a two-dot diff.")
        if _append_only_log(c):
            # An owed deploy-log that has SAT is stale BY CONSTRUCTION -- that is
            # what "owed" means, and it is precisely the backlog this generator
            # exists to clear (Wave 14 cleared three at fork-behind 8). Holding on
            # it would make the tool refuse its own core case, and the resolution
            # is already recorded: append oldest-first, abort on anything else.
            warns.append(msg + " It is an append-only deploy-log entry, so this is "
                               "expected -- merge it oldest-first and **ABORT if the "
                               "conflict is anything other than two dated entries "
                               "wanting the same spot.**")
        else:
            holds.append(msg + " M rebases or confirms the merge is clean before this "
                               "train runs.")

    return holds, warns


def _append_only_log(car):
    """An owed-deploy-log car touching nothing but append-only paths."""
    return (OWED_LOG_RE.match(car["branch"] or "")
            and bool(car["paths"])
            and all(app_relative(p) in APPEND_ONLY_PATHS for p in car["paths"]))


# ---------------------------------------------------------------------------
# #224 GAP 3 -- zero cars must never be quiet.
#
# The candidate set starts at `dev_tasks?status=eq.Staged`, so a branch with NO
# board task never enters it and branch derivation's three fallbacks never even
# run. Everything a dispatched agent produces has a task; everything M authors
# directly does not -- so M's own output is the one class of work the generator
# structurally cannot see. Wave 20 returned a clean `0 car(s)` with two mergeable
# PRs open and was hand-authored as a result.
#
# The fix is not to widen the candidate set (auto-creating tasks for orphan
# branches is a design question, not this one); it is to make the ABSENCE
# checkable: read the open PRs, and say out loud which ones no car claims.
# ---------------------------------------------------------------------------

def unclaimed_prs(open_prs, car_branches):
    """Open PRs that could merge and that no car in this train accounts for.

    Drafts and PRs GitHub already knows are CONFLICTING are excluded -- neither
    is a candidate for a train. `UNKNOWN` mergeability is KEPT: gh reports it
    whenever the merge commit has not been computed yet, and treating unknown as
    "not mergeable" would reintroduce the silence this rail exists to break."""
    out = []
    for pr in open_prs or []:
        if pr.get("isDraft"):
            continue
        if (pr.get("headRefName") or "") in car_branches:
            continue
        if (pr.get("mergeable") or "").upper() == "CONFLICTING":
            continue
        out.append(pr)
    return out


def zero_car_verdict(cars, open_prs, unclaimed):
    """None when there ARE cars; else which KIND of zero this is.

    `UNVERIFIED` -- the PR check could not run, so "nothing to ship" is unproven.
    `SUSPICIOUS` -- open mergeable PRs exist that no car claims (the Wave-20 shape).
    `CLEAN`      -- the PRs were listed and every one is claimed or unmergeable.
    """
    if cars:
        return None
    if open_prs is None:
        return "UNVERIFIED"
    return "SUSPICIOUS" if unclaimed else "CLEAN"


def _pr_line(pr):
    num = pr.get("number")
    head = pr.get("headRefName") or "?"
    title = (pr.get("title") or "").strip()
    state = (pr.get("mergeable") or "UNKNOWN").upper()
    return f"PR #{num} `{head}` -- {title} (mergeable: {state})"


# ---------------------------------------------------------------------------
# Train assembly (pure -- everything external is injected)
# ---------------------------------------------------------------------------

def build_train(tasks, resolve_branch, diff_files, fork_behind,
                owed_logs=(), wave=None, base_sha="", date=None,
                session="headless", other_branches=(), loop_version=None,
                already_merged=None, suites=DISPATCHER_SUITES,
                commits=None, open_prs=None, loop_staleness=None):
    """Assemble a train plan from board rows + injected git facts.

    resolve_branch(task) -> (branch|None, source)
    diff_files(branch)   -> [git-root-relative path, ...]
    fork_behind(branch)  -> int commits of origin/dev the branch does not have
    already_merged(br)   -> True when the branch tip is already an ancestor of dev
    suites               -> the dispatcher suite set AS IT EXISTS at the merged tip
    commits(branch)      -> ["<sha> <subject>", ...] from `git log --oneline
                            origin/dev..<branch>`, or None when unknown (#224 gap 1)
    open_prs             -> `gh pr list` rows, or None when the check could not run
                            (#224 gap 3 -- None is UNVERIFIED, never a clean zero)
    loop_staleness       -> {"stale": bool, ...} from process-start vs pull time
    """
    already_merged = already_merged or (lambda b: False)
    commits = commits or (lambda b: None)
    cars, refusals = [], []

    for br in owed_logs:
        if not OWED_LOG_RE.match(br):
            refusals.append((br, "taskless branch that is not an owed deploy-log "
                                 "(`docs/deploy-log-waveN-MMDD`) -- nothing reviewed it"))
            continue
        cars.append(_car(None, "(open a PR)", br, "docs -- owed deploy-log entry",
                         diff_files(br), fork_behind(br), suites=suites,
                         commits=commits(br)))

    for t in sorted(tasks, key=lambda t: t["id"]):
        idx = r_step_index(t)
        if idx is None:
            if not has_r_step(t):
                # Loud, not silent: a Staged task whose chain never names R is
                # invisible to the car rule but VERY visible to Kevin, who staged
                # it expecting it to ship.
                refusals.append((f"#{t['id']}", "`Staged` but its chain names NO R step, "
                                 "so the car rule (current step is R-owned) cannot see "
                                 "it. Retrofit the chain with `/task-chain`, or dispatch "
                                 "R by hand for it."))
            continue                      # otherwise: still mid-chain, correctly quiet
        if review_step(t, idx) is None:
            refusals.append((f"#{t['id']}", "no TICKED M-review step in front of the "
                                            "R step -- unreviewed work never enters a train"))
            continue
        branch, src = resolve_branch(t)
        if not branch:
            refusals.append((f"#{t['id']}", "no branch found in run_history.pushed_branch, "
                                            "the task thread, or local branches"))
            continue
        paths = diff_files(branch)
        if not paths:
            # A Staged task whose branch is ALREADY on dev is the #202 bug, not a
            # release: the code shipped and nobody ticked the R step. Say so, so M
            # ticks and closes instead of hunting for a missing branch.
            if already_merged(branch):
                refusals.append((f"#{t['id']}", f"`{branch}` is ALREADY MERGED into "
                                 f"{BASE} -- it shipped on an earlier train and its R "
                                 f"step was never ticked. Tick step {idx + 1} and close; "
                                 f"do not re-ship."))
            else:
                refusals.append((f"#{t['id']}", f"`{branch}` has no diff against {BASE} "
                                                "-- nothing to ship"))
            continue
        cars.append(_car(t["id"], f"#{t['id']}", branch,
                         f"board #{t['id']} -- {t['title']}", paths, fork_behind(branch),
                         branch_source=src, armed=bool(t.get("ai_eligible")),
                         suites=suites, commits=commits(branch)))

    # RAIL: docs first, code last. Stable within a kind (owed logs, then id order).
    cars.sort(key=lambda c: 0 if c["kind"] == "docs" else 1)

    holds, warns = assess(cars)
    unclaimed = unclaimed_prs(open_prs, {c["branch"] for c in cars})

    return {
        "wave": wave,
        "date": date,
        "base_sha": base_sha,
        "session": session,
        "cars": cars,
        "refusals": refusals,
        "holds": holds,
        "warns": warns,
        "auto_dispatch": bool(cars) and not holds,
        "other_branches": list(other_branches),
        "loop_version": loop_version,
        "loop_staleness": loop_staleness,
        "pr_check": "unavailable" if open_prs is None else "ok",
        "unclaimed_prs": unclaimed,
        "zero_car_verdict": zero_car_verdict(cars, open_prs, unclaimed),
        "gates": _merged_gates(cars),
    }


def _car(task_id, pr_label, branch, what, paths, behind, branch_source="owed-log",
         armed=False, suites=DISPATCHER_SUITES, commits=None):
    return {
        "task_id": task_id,
        "label": pr_label if task_id else f"`{branch}`",
        "pr": pr_label,
        "branch": branch,
        "what": what,
        "paths": sorted(paths),
        "kind": classify(paths),
        "fork_behind": behind,
        "gates": gates_for(paths, suites),
        "frontend": frontend_touched(paths),
        "branch_source": branch_source,
        "armed": armed,
        # #224 gap 1: what the car CARRIES, read from the commits rather than
        # inferred from the task title. None = the commit list was not injected.
        "commits": list(commits) if commits is not None else None,
        "multi_commit": bool(commits is not None and len(commits) > 1),
    }


def _merged_gates(cars):
    out = []
    for c in cars:
        for g in c["gates"]:
            if g not in out:
                out.append(g)
    return out


# ---------------------------------------------------------------------------
# Rendering -- matches the format R already executes reliably. Do not redesign.
# ---------------------------------------------------------------------------

def render(plan):
    n = len(plan["cars"])
    car_count = {0: "NO CARS", 1: "ONE CAR"}.get(n, f"{n} CARS")
    out = [HEADER.format(wave=plan["wave"], date=plan["date"],
                         car_count=car_count, session=plan["session"]), ""]

    out += _coverage_block(plan)

    if plan["holds"]:
        out += ["> ⚠️ **HELD FOR M -- NOT AUTO-DISPATCHED.** The generator found "
                "risk it cannot resolve:", ""]
        out += [f"> - {h}" for h in plan["holds"]]
        out += ["", "> M reconciles, then re-runs the generator or dispatches R by hand.", ""]

    out += ["## Merge in this order",
            "| # | Branch | Kind |", "|---|--------|------|"]
    for c in plan["cars"]:
        flag = f" · **{len(c['commits'])} commits**" if c["multi_commit"] else ""
        out.append(f"| **{c['pr']}** | `{c['branch']}` | {c['what']}{flag} |")
    out.append("")

    out += _cargo_block(plan)

    ov = overlaps(plan["cars"])
    if not ov:
        out.append("**Zero file overlap** between cars: " + " · ".join(
            f"{c['label']} = {_paths_summary(c['paths'])}" for c in plan["cars"]) + ".")
        out.append("")
    else:
        # The hand-written briefs ALWAYS said where the non-conflicting cars live
        # ("#141 touches only `.claude/skills/**` ... neither overlaps the logs").
        # Suppressing that whenever ANY overlap existed dropped it from exactly the
        # briefs that needed it most.
        clean = [c for c in plan["cars"] if not any(p in ov for p in c["paths"])]
        if clean:
            out.append("The other cars do **not** overlap: " + " · ".join(
                f"{c['label']} = {_paths_summary(c['paths'])}" for c in clean) + ".")
            out.append("")
    for w in plan["warns"]:
        out += [w, ""]

    if plan["gates"]:
        out += ["## GATING -- from a CLEAN `origin/dev` worktree, never the main checkout",
                "```"]
        out += list(plan["gates"])
        out += ["```",
                "All green at M's review of the branch tips above. **Re-run against the "
                "MERGED tip.** Abort loudly on any failure.", ""]
        docs_cars = [c for c in plan["cars"] if not c["gates"]]
        if docs_cars:
            out += ["The remaining cars ("
                    + ", ".join(c["label"] for c in docs_cars)
                    + ") need no gates; confirm CI green on each PR.", ""]
    else:
        out += ["## GATING", "No gates derivable -- every car is docs-only. Confirm CI "
                "green on each PR before merging.", ""]

    if any(c["frontend"] for c in plan["cars"]):
        out += ["**Frontend cars in this train** ("
                + ", ".join(c["label"] for c in plan["cars"] if c["frontend"])
                + "): confirm the Vercel/Railway frontend build goes `success`, not just "
                  "CI green.", ""]

    out += [PROCEDURE.format(wave=plan["wave"], mmdd=plan["date"][5:].replace("-", ""),
                             base_sha=plan["base_sha"], api_host=API_HOST), ""]

    loop_note = _loop_note(plan)
    armed = [c["label"] for c in plan["cars"] if c["armed"]]
    if armed:
        # Asserting "nothing needs arming" without looking was a claim, not a fact.
        arm_note = (f"**{', '.join(armed)} {'is' if len(armed) == 1 else 'are'} ARMED** "
                    f"(`ai_eligible=true`). Do **not** disarm, and do not arm anything else.")
    else:
        arm_note = "**Do not arm or disarm any task.** No car in this train is armed."
    others = ", ".join(f"`{b}`" for b in plan["other_branches"]) or "any branch not listed above"
    out.append(HARD_LIMITS.format(loop_note=loop_note, arm_note=arm_note,
                                  other_branches=others, app_root=MAIN_CHECKOUT))

    if plan["refusals"]:
        out += ["", "## Refused cars (NOT in this train)"]
        out += [f"- **{who}** -- {why}" for who, why in plan["refusals"]]

    return "\n".join(out) + "\n"


def _loop_note(plan):
    """#224 GAP 2 -- the restart trigger is "a car touched `dispatcher.py`".

    It used to be phrased as a version delta ("a car makes it V4.25; the loop is
    V4.24"), which cannot work: Wave 19 carried **V4.25 on both sides of the
    merge** and shipped a real dispatcher change, so no banner comparison could
    ever have exposed the drift. A constant-only edit never moves the banner.
    The banner is now INFORMATIONAL only, and the check that actually decides
    whether the RUNNING process is stale is process-start-time vs. the time the
    checkout last pulled that file (`live_loop_staleness()`)."""
    touches_loop = any(app_relative(p) == DISPATCHER_FILE
                       for c in plan["cars"] for p in c["paths"])
    who = [c["label"] for c in plan["cars"]
           if any(app_relative(p) == DISPATCHER_FILE for p in c["paths"])]
    st = plan.get("loop_staleness") or {}

    if not touches_loop:
        note = "No car touches `dispatcher.py`, so no restart is owed for this train."
        if st.get("stale"):
            delta = _ago(st.get("file_mtime"), st.get("loop_started"))
            if _delta_s(st) < LOOP_STALE_MARGIN_S:
                # A pull landing seconds after the process started is just as
                # likely to BE the restart as to have missed it. Say which fact
                # we have (two timestamps, this close) rather than pick a story.
                note += (f" Its `dispatcher.py` was written {delta} after the process "
                         "started -- inside restart/pull race noise, so this is NOT a "
                         "staleness claim. **M confirms which came first.**")
            else:
                note += (" **But the RUNNING loop is ALREADY stale** -- its "
                         f"`dispatcher.py` was written {delta} AFTER the process "
                         "started, so the process is executing code the checkout no "
                         "longer has. Restart it anyway, and tell M.")
        return note

    run_v = plan.get("loop_version")
    vers = f" (the running loop reports V{run_v}; the banner is informational -- " \
           f"a constant-only change does not move it)" if run_v else ""
    note = (f"{', '.join(who)} changes `dispatcher.py`, so the merged tip is code the "
            f"running loop is not executing{vers}. **M restarts and verifies.** "
            "The trigger is the dispatcher-touch itself, NEVER a version comparison "
            "-- Wave 19 carried the same banner on both sides of a real change.")
    if st.get("stale"):
        note += (" Confirmed: the running process started BEFORE its current "
                 "`dispatcher.py` was pulled.")
    elif st:
        note += (" Verify the restart by process start time vs. the checkout's "
                 "`dispatcher.py` mtime, not by the banner.")
    return note


def _delta_s(st):
    try:
        return max(0.0, float(st.get("file_mtime")) - float(st.get("loop_started")))
    except (TypeError, ValueError):
        return 0.0


def _ago(later, earlier):
    try:
        secs = max(0, int(float(later) - float(earlier)))
    except (TypeError, ValueError):
        return "some time"
    if secs < 60:
        return f"{secs}s"                  # NOT "0m" -- a 36s delta is not zero
    return f"{secs // 60}m" if secs < 7200 else f"{secs // 3600}h"


def _coverage_block(plan):
    """#224 GAP 3 -- the block that makes a zero-car train impossible to misread.

    `0 car(s)` and "nothing to ship" are the same sentence to a reader, and only
    one of them is a fact. Whichever it is, say which."""
    v = plan.get("zero_car_verdict")
    unclaimed = plan.get("unclaimed_prs") or []
    if v == "SUSPICIOUS":
        out = ["> 🚨 **ZERO CARS -- AND THAT IS SUSPICIOUS. DO NOT READ THIS AS "
               "\"NOTHING TO SHIP\".** No `Staged` board task has a current R "
               f"step, yet {len(unclaimed)} open PR(s) are mergeable and no car "
               "claims them:", ""]
        out += [f"> - {_pr_line(p)}" for p in unclaimed]
        out += ["",
                "> The generator's entry point is `dev_tasks?status=eq.Staged`, so a "
                "branch with no board task never enters the candidate set -- which is "
                "exactly the class of work M authors directly (session docs, handoffs, "
                "notes). **M reviews the PRs above by hand before concluding this train "
                "is empty.**", ""]
        return out
    if v == "UNVERIFIED":
        return ["> ⚠️ **ZERO CARS -- UNVERIFIED.** No `Staged` board task has a current "
                "R step, AND the open-PR cross-check could not run (`gh` unavailable or "
                "failed). \"Nothing to ship\" is therefore UNPROVEN -- M checks "
                "`gh pr list --state open` by hand before closing this out.", ""]
    if v == "CLEAN":
        return ["> **ZERO CARS -- checked.** No `Staged` board task has a current R step, "
                "and every open PR is either already a car or not mergeable. This one is "
                "a real empty train.", ""]
    if unclaimed:
        # Cars exist, so the train is not empty -- but an unclaimed mergeable PR
        # is still work the board cannot see, and the same silence hid Wave 20.
        out = [f"> ℹ️ **{len(unclaimed)} open PR(s) are NOT in this train** and no car "
               "claims them. Not a blocker; M decides whether they belong here:", ""]
        out += [f"> - {_pr_line(p)}" for p in unclaimed]
        out += [""]
        return out
    return []


def _cargo_block(plan):
    """#224 GAP 1 -- what each car CARRIES, read from its commits.

    Wave 19's cargo line named `DAILY_CAP` (the task title) and shipped
    `CONCURRENCY 3->4` undescribed. The change was in the diff and gated
    correctly; only the human-readable summary -- the part a reviewer trusts --
    was wrong. Task title stays the header; the commit subjects are the contents."""
    if not any(c["commits"] is not None for c in plan["cars"]):
        return []                      # commit facts were not injected -- say nothing
    out = ["## What each car carries", ""]
    for c in plan["cars"]:
        if c["commits"] is None:
            out.append(f"- **{c['label']}** `{c['branch']}` -- {c['what']} "
                       f"(commit list unavailable -- R reads the PR)")
            continue
        n = len(c["commits"])
        if n == 1:
            out.append(f"- **{c['label']}** `{c['branch']}` -- {c['what']}")
        elif n == 0:
            out.append(f"- **{c['label']}** `{c['branch']}` -- {c['what']} "
                       f"(no commits ahead of {BASE})")
        else:
            out.append(f"- **{c['label']}** `{c['branch']}` -- {c['what']} "
                       f"-- **{n} commits: this car carries more than its title says.**")
        for line in c["commits"]:
            out.append(f"  - `{line}`")
    out.append("")
    return out


def _paths_summary(paths, limit=2):
    rel = [app_relative(p) for p in paths]
    dirs = []
    for p in rel:
        d = os.path.dirname(p) + "/**" if os.path.dirname(p) else p
        if d not in dirs:
            dirs.append(d)
    if len(dirs) <= limit:
        return " + ".join(f"`{d}`" for d in dirs)
    return " + ".join(f"`{d}`" for d in dirs[:limit]) + f" (+{len(dirs) - limit} more)"


# ---------------------------------------------------------------------------
# Live wiring -- board reads and git reads. Nothing here mutates a worktree.
# ---------------------------------------------------------------------------

def _env_path():
    """`src/.env` is untracked, so a fresh worktree does not have one -- fall back
    to the main checkout's copy (read-only). Fail LOUD if neither exists rather
    than silently producing a brief off no board data."""
    for p in (ENV, f"{MAIN_CHECKOUT}/src/.env"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no src/.env found at {ENV} or {MAIN_CHECKOUT}/src/.env -- "
        "cannot read the board")


def _creds():
    url = key = None
    for line in open(_env_path(), encoding="utf-8"):
        line = line.strip()
        if line.startswith("SUPABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
            key = line.split("=", 1)[1].strip().strip('"')
    return url.rstrip("/"), key


def api(method, path, body=None, prefer=None):
    import urllib.request
    url, key = _creds()
    headers = {"apikey": key, "Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{url}/rest/v1/{path}", data=data,
                                 headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as r:
        txt = r.read().decode()
        return json.loads(txt) if txt.strip() else None


def git(*args, timeout=GIT_TIMEOUT_S):
    p = subprocess.run(("git", "-C", GIT_ROOT) + args, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def live_resolve_branch(task):
    """run_history.pushed_branch -> task thread -> local branch named `*-<id>`."""
    tid = task["id"]
    try:
        rows = api("GET", f"run_history?task_id=eq.{tid}&pushed_branch=not.is.null"
                          f"&select=pushed_branch&order=id.desc&limit=1")
        if rows:
            return rows[0]["pushed_branch"], "run_history.pushed_branch"
    except Exception:
        pass
    try:
        rows = api("GET", f"dev_task_comments?task_id=eq.{tid}&select=body"
                          f"&order=id.desc&limit=40")
        for row in rows or []:
            found = BRANCH_TOKEN_RE.findall(row.get("body") or "")
            for b in found:
                b = b.rstrip("`.,;:)")
                if git("rev-parse", "--verify", b)[0] == 0:
                    return b, "task thread"
    except Exception:
        pass
    rc, out, _ = git("for-each-ref", "--format=%(refname:short)", "refs/heads")
    if rc == 0:
        for b in out.splitlines():
            if b.endswith(f"-{tid}"):
                return b, f"local branch matching `-{tid}`"
    return None, "unresolved"


def live_diff_files(branch):
    """Three-dot diff -- against the MERGE BASE, so a stale base does not
    manufacture phantom deletions (the Wave-16 trap)."""
    rc, out, _ = git("diff", "--name-only", f"{BASE}...{branch}")
    return out.splitlines() if rc == 0 else []


def live_commits(branch):
    """`git log --oneline origin/dev..<branch>` -- what the car actually carries.

    Two-dot on purpose: the commits the branch has that dev does not, which is
    the cargo. (The DIFF is read three-dot, against the merge base, so a stale
    base cannot manufacture phantom deletions -- different question, different
    operator.) None on failure, so the brief says "unavailable" instead of
    silently claiming a car is empty."""
    rc, out, _ = git("log", "--oneline", "--no-decorate", f"{BASE}..{branch}")
    if rc != 0:
        return None
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def live_open_prs():
    """Open PRs via `gh`, or None when the check could not run (#224 gap 3).

    None and `[]` mean very different things here and must not collapse: `[]` is
    "GitHub says there are no open PRs", None is "nobody asked". Only the first
    can license a clean zero-car verdict."""
    try:
        p = subprocess.run(
            ["gh", "pr", "list", "--state", "open", "--limit", "60",
             "--json", "number,title,headRefName,mergeable,isDraft,url"],
            cwd=GIT_ROOT, capture_output=True, text=True, timeout=GH_TIMEOUT_S)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    try:
        rows = json.loads(p.stdout)
    except ValueError:
        return None
    return rows if isinstance(rows, list) else None


def live_loop_staleness():
    """Is the RUNNING loop older than the `dispatcher.py` it runs from?

    The sound restart check (#224 gap 2). A version banner cannot answer this --
    Wave 19 carried V4.25 on both sides of a real change -- but two timestamps
    can: when the process started, and when the checkout last wrote the file it
    loaded. Returns None when the loop cannot be found (nothing to claim)."""
    path = f"{MAIN_CHECKOUT}/{DISPATCHER_FILE}"
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    try:
        p = subprocess.run(["pgrep", "-f", DISPATCHER_FILE],
                           capture_output=True, text=True, timeout=10)
        pids = [x for x in p.stdout.split() if x.isdigit()]
        started = None
        for pid in pids:
            q = subprocess.run(["ps", "-o", "etimes=", "-p", pid],
                               capture_output=True, text=True, timeout=10)
            el = q.stdout.strip()
            if not el.isdigit():
                continue
            cand = time.time() - int(el)
            started = cand if started is None else min(started, cand)
        if started is None:
            return None
    except Exception:
        return None
    return {"loop_started": started, "file_mtime": mtime, "stale": mtime > started}


def live_already_merged(branch):
    return git("merge-base", "--is-ancestor", branch, BASE)[0] == 0


def live_fork_behind(branch):
    rc, mb, _ = git("merge-base", BASE, branch)
    if rc != 0:
        return 0
    rc, out, _ = git("rev-list", "--count", f"{mb}..{BASE}")
    return int(out) if rc == 0 and out.isdigit() else 0


def live_owed_logs():
    """Local `docs/deploy-log-waveN-MMDD` branches not yet merged into dev."""
    rc, out, _ = git("for-each-ref", "--format=%(refname:short)", "refs/heads")
    if rc != 0:
        return []
    owed = []
    for b in out.splitlines():
        if not OWED_LOG_RE.match(b):
            continue
        if git("merge-base", "--is-ancestor", b, BASE)[0] == 0:
            continue                      # already on dev
        owed.append(b)
    return sorted(owed)


OTHER_BRANCH_MAX_AGE_S = 21 * 86400
OTHER_BRANCH_CAP = 10


def live_other_branches(car_branches):
    """Branches R must leave alone -- IN FLIGHT ones only.

    Every other worktree's checked-out branch is a candidate, but the hand-written
    briefs named 3-5, not 27. Two filters make the derived list match: drop
    anything already merged into dev (leaving it alone is moot) and anything not
    committed to in ~3 weeks (long-dead worktrees). Newest first, capped."""
    rc, out, _ = git("worktree", "list", "--porcelain")
    if rc != 0:
        return []
    now = time.time()
    cands = []
    for line in out.splitlines():
        if not line.startswith("branch refs/heads/"):
            continue
        b = line.split("refs/heads/", 1)[1]
        if b in ("dev", "main", "master") or b in car_branches:
            continue
        if git("merge-base", "--is-ancestor", b, BASE)[0] == 0:
            continue                              # already on dev
        rc2, ts, _ = git("log", "-1", "--format=%ct", b)
        age = now - int(ts) if (rc2 == 0 and ts.isdigit()) else 0
        if age > OTHER_BRANCH_MAX_AGE_S:
            continue
        if b not in [c[1] for c in cands]:
            cands.append((age, b))
    cands.sort()
    return [b for _, b in cands[:OTHER_BRANCH_CAP]]


def live_wave():
    """Next wave number = highest `R · Release train Wave N` on the board + 1."""
    try:
        rows = api("GET", "dev_tasks?title=like.*Release%20train%20Wave*&select=title")
    except Exception:
        return None
    n = 0
    for r in rows or []:
        m = re.search(r"Wave (\d+)", r["title"])
        if m:
            n = max(n, int(m.group(1)))
    return n + 1 if n else None


def _banner_version(text):
    """The dispatcher's version from its module docstring.

    The banner reads `Team dispatcher (V4.25) - dispatches board tasks ...`, so the
    old `"version" in line` guard never matched and every brief silently lost the
    version it was supposed to state. Take the first `V<maj>.<min>` in the head of
    the file, which is the banner by construction."""
    for line in text.splitlines()[:20]:
        m = re.search(r"\bV(\d+\.\d+)", line)
        if m:
            return m.group(1)
    return None


def live_loop_version():
    """Version banner of the RUNNING loop, read from the main checkout (read-only)."""
    try:
        with open(f"{MAIN_CHECKOUT}/{DISPATCHER_FILE}", encoding="utf-8") as fh:
            return _banner_version(fh.read())
    except Exception:
        return None


# NOTE (#224 gap 2): there used to be a `live_target_version()` here, reading the
# banner off whichever car edits the loop so the brief could print "a car makes it
# V4.25; the loop is V4.24". It is DELETED, not fixed. The delta it computed was
# never the restart trigger and could not be: Wave 19 shipped a real dispatcher
# change with V4.25 on both sides, and any constant-only edit does the same. The
# trigger is the dispatcher-touch; the staleness check is `live_loop_staleness()`.


def live_suite_env(car_paths):
    """(exists, discovered) for the MERGED tip = `origin/dev` plus what cars add."""
    rc, out, _ = git("ls-tree", "-r", "--name-only", BASE)
    present = {app_relative(p) for p in out.splitlines()} if rc == 0 else set()
    present |= {app_relative(p) for p in car_paths}
    return (lambda p: p in present), {p for p in present if SUITE_RE.match(p)}


def staged_tasks():
    return api("GET", "dev_tasks?status=eq.Staged"
                      "&select=id,title,status,assignee,checklist,ai_eligible") or []


# ---------------------------------------------------------------------------
# R-task creation -- the ONE write this tool may make.
# ---------------------------------------------------------------------------

def r_task_row(plan, brief):
    """The dev_tasks row for the train. AUTO -> assignee R + armed; HELD -> M,
    status Review, unarmed (visible, never dispatched)."""
    cars = ", ".join(
        (f"#{c['task_id']}" if c["task_id"] else c["branch"].split("/")[-1])
        for c in plan["cars"])
    return {
        "title": f"R · Release train Wave {plan['wave']} ({plan['date'][5:]}) — {cars}",
        "description": brief,
        "status": "Todo" if plan["auto_dispatch"] else "Review",
        "assignee": "R" if plan["auto_dispatch"] else "M",
        "ai_eligible": bool(plan["auto_dispatch"]),
        "area": "other",
        "impact": "contained",
        "origin": "discovered",
        "priority_phase": 1,
        "priority_seq": 1,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="emit the plan, not the brief")
    ap.add_argument("--create-task", action="store_true",
                    help="also create the R release-train task on the board")
    ap.add_argument("--session", default="headless", help="authoring session id")
    ap.add_argument("--wave", type=int, default=None,
                    help="override the derived wave number")
    ap.add_argument("--no-owed-logs", action="store_true",
                    help="skip auto-detected owed deploy-log branches")
    args = ap.parse_args(argv)

    git("fetch", "origin", "--quiet")
    tasks = staged_tasks()
    owed = () if args.no_owed_logs else live_owed_logs()
    _, base_sha, _ = git("rev-parse", "--short", BASE)

    diff_cache, commit_cache = {}, {}

    def cached_diff(branch):
        if branch not in diff_cache:
            diff_cache[branch] = live_diff_files(branch)
        return diff_cache[branch]

    def cached_commits(branch):
        if branch not in commit_cache:
            commit_cache[branch] = live_commits(branch)
        return commit_cache[branch]

    open_prs = live_open_prs()

    def assemble(**kw):
        return build_train(
            tasks, live_resolve_branch, cached_diff, live_fork_behind,
            already_merged=live_already_merged, commits=cached_commits,
            open_prs=open_prs, loop_staleness=live_loop_staleness(),
            owed_logs=owed, wave=args.wave or live_wave(), base_sha=base_sha,
            date=datetime.now(timezone.utc).date().isoformat(),
            session=args.session, loop_version=live_loop_version(), **kw)

    # Pass 1 discovers the cars; the suite set is a function of what those cars
    # touch, so pass 2 assembles the real plan. The caches keep this to one
    # `git diff` and one `git log` per branch.
    scout = assemble()
    car_paths = [p for c in scout["cars"] for p in c["paths"]]
    exists, discovered = live_suite_env(car_paths)
    plan = assemble(suites=resolve_dispatcher_suites(exists, discovered))
    plan["other_branches"] = live_other_branches({c["branch"] for c in plan["cars"]})

    if plan["wave"] is None:
        # Fail loud rather than emit a "Wave-None" brief that R would then execute.
        print("could not derive the wave number from the board -- pass --wave N",
              file=sys.stderr)
        return 2

    brief = render(plan)

    if args.json:
        print(json.dumps({k: v for k, v in plan.items()}, indent=1))
    else:
        print(brief)

    # #224 gap 3: the one-line verdict is what a reader actually reads. A bare
    # `0 car(s)` is indistinguishable from "nothing to ship", so a zero-car train
    # never gets one -- it gets its KIND, and the unclaimed PRs by number.
    zero = plan["zero_car_verdict"]
    verdict = ("AUTO-DISPATCH" if plan["auto_dispatch"]
               else f"ZERO CARS -- {zero}" if zero else "HELD FOR M")
    print(f"\n--- verdict: {verdict} · {len(plan['cars'])} car(s) · "
          f"{len(plan['refusals'])} refused · {len(plan['holds'])} hold(s) ---",
          file=sys.stderr)
    if plan["unclaimed_prs"]:
        print(f"--- {len(plan['unclaimed_prs'])} open PR(s) NO CAR CLAIMS: "
              + ", ".join(f"#{p.get('number')} {p.get('headRefName')}"
                          for p in plan["unclaimed_prs"]) + " ---", file=sys.stderr)
    elif zero == "UNVERIFIED":
        print("--- open-PR cross-check did NOT run (`gh` unavailable): "
              "\"nothing to ship\" is UNPROVEN ---", file=sys.stderr)

    if args.create_task:
        if not plan["cars"]:
            print("no cars -- nothing to create", file=sys.stderr)
            return 1
        row = api("POST", "dev_tasks", r_task_row(plan, brief),
                  prefer="return=representation")
        print(f"created #{row[0]['id']} assignee={row[0]['assignee']} "
              f"ai_eligible={row[0]['ai_eligible']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
