#!/usr/bin/env python3
"""Acceptance test — board #195: the dispatcher pushes the agent branch at run end.

A headless agent can build, test and COMMIT but structurally CANNOT push, so
before this every successful run ended with its work on local disk only, visible
to whoever happened to read preflight invariant (2). reap() now pushes the run's
own branch.

EVERY RAIL HAS A TEST THAT FAILS WITHOUT IT — a rail with no failing test is a
comment, not a rail. Each case asserts the specific decline REASON, not merely
"nothing was pushed": several rails would incidentally cover for each other
(refusing `dev` also trips "already exists on origin"), and a test that passes
for the wrong reason cannot notice its rail being deleted.

  1  happy path            branch is created on origin, upstream set
  2  refuses dev           protected, and origin/dev does NOT move
  3  refuses main          protected — and `main` is absent from origin here, so
                           ONLY the protected rail can decline it
  4  refuses dirty         a modified tracked file, and an untracked file
  5  refuses not-ours (a)  HEAD never moved off the branch the run started on
  6  refuses not-ours (b)  the branch already exists on origin
  7  nothing ahead         no commits vs origin/dev → nothing to push
  8  kill switch           tools/team_dispatcher/NO_PUSH beats a perfect candidate
  9  pre-#195 run records   no worktree / no start branch → declines, never crashes
 10  survives a push error  bad remote → reap COMPLETES, reports, comments LOUD
 11  survives an exception  git itself blowing up cannot fail a reap
 12  never --force          asserted against the real argv git was handed
 13  reap integration       report + push + run_history.pushed_branch, one pass
 14  dispatch-side capture  one_pass() records worktree + branch0 on the run

Cases 15-21 are THE REGRESSION SET for the first cut of this feature, which
passed all of 1-14 and never once pushed anything. It anchored the push to the
agent's REGISTERED worktree; charter §4 has each lane work in a FRESH worktree
cut after dispatch, leaving the registered one parked on dev — so every real run
declined as `protected`. Cases 1-14 could not see it: they handed
push_run_branch() a worktree that was ALREADY on the feature branch, a shape no
real dispatch produces. These build the real one — a lane tree on dev, plus a
`git worktree add` after the dispatch snapshot.

 15  THE BUG           §4 shape — work in a post-dispatch worktree IS pushed
 16  rails still hold  every rail re-applied to a DISCOVERED worktree
 17  not ours          a worktree another run DECLARED (board #218) is left
                      alone — alive or dead — and our own is taken regardless
 18  reap, §4 shape    end-to-end: reap of a lane-on-dev run pushes + records
 19  reap wires runs   reap supplies the run set; a live peer no longer blocks
                      our push, and its own tree is still untouched (#218)
 20  both legs         registered tree moved AND a fresh worktree → both pushed
 21  discovery is off  no snapshot / unlistable worktrees → decline, no crash

Cases 17 and 19 were rewritten by board #218: V4.23 decided ownership by
ELIMINATING the other in-flight runs, which declined whenever any older peer was
alive (the normal case at CONCURRENCY=4) and ADOPTED a worktree once its owner
DIED. Ownership is now DECLARED in the worktree's directory name, so add_worktree()
takes an `owner`. The principle both cases defend is unchanged; the full #218 set
lives in src/test_dispatcher_worktree_ownership_218.py.

Hermetic: real git against a LOCAL bare repo (no network, no GitHub), the
Supabase api() replaced with a recorder (ZERO board writes).

Run:  .venv/bin/python src/test_dispatcher_push_at_run_end_195.py   (from RoR_Trader root)
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


# ── load the real dispatcher module ────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "team_dispatcher",
    os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)

TMP = tempfile.mkdtemp(prefix="push195-")
disp.STATE_FILE = f"{TMP}/state.json"
disp.LOG_DIR = f"{TMP}/logs"
os.makedirs(disp.LOG_DIR, exist_ok=True)
# The kill-switch and PAUSE files must resolve inside the sandbox, never to the
# live dispatcher directory (a stray NO_PUSH there would silently pass case 8).
disp.NO_PUSH_FILE = f"{TMP}/NO_PUSH"
disp.PAUSE_FILE = f"{TMP}/PAUSE"

CALLS = []


def fake_api(method, path, body=None, prefer=None):
    CALLS.append((method, path, body))
    if method == "GET" and path.startswith("dev_task_comments"):
        return []
    if method == "GET" and path.startswith("dev_tasks?status=eq.Done"):
        return []
    if method == "GET" and path.startswith("run_history"):
        return []
    return None


disp.api = fake_api

GIT_ENV = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
               GIT_TERMINAL_PROMPT="0")


def git(cwd, *args, check=True):
    p = subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV,
                       capture_output=True, text=True)
    if check and p.returncode != 0:
        print(f"FAIL: git {' '.join(args)} in {cwd}\n{p.stdout}{p.stderr}")
        sys.exit(1)
    return p.stdout.strip()


# ── a throwaway "GitHub": one bare repo with a dev branch ──────────────────
ORIGIN = f"{TMP}/origin.git"
git(TMP, "init", "--quiet", "--bare", ORIGIN)
git(ORIGIN, "symbolic-ref", "HEAD", "refs/heads/dev")
SEED = f"{TMP}/seed"
git(TMP, "clone", "--quiet", ORIGIN, SEED)
git(SEED, "checkout", "--quiet", "-b", "dev")
open(f"{SEED}/README.md", "w").write("seed\n")
git(SEED, "add", "-A")
git(SEED, "commit", "--quiet", "-m", "seed")
git(SEED, "push", "--quiet", "-u", "origin", "dev")

TREES = [0]


def agent_tree(branch=None, commits=1, dirty=None, remote=ORIGIN):
    """A fresh clone standing in for an agent worktree.

    branch=None leaves it on dev; otherwise a branch cut from origin/dev (the
    #153 contract) with `commits` commits on it. dirty: 'modified' | 'untracked'.
    """
    TREES[0] += 1
    wt = f"{TMP}/wt{TREES[0]}"
    git(TMP, "clone", "--quiet", remote, wt)
    if branch:
        git(wt, "checkout", "--quiet", "-b", branch, "origin/dev")
    for i in range(commits):
        open(f"{wt}/work{i}.txt", "w").write(f"work {i}\n")
        git(wt, "add", "-A")
        git(wt, "commit", "--quiet", "-m", f"work {i}")
    if dirty == "modified":
        open(f"{wt}/README.md", "a").write("half-finished edit\n")
    elif dirty == "untracked":
        open(f"{wt}/scratch.txt", "w").write("left behind\n")
    return wt


def origin_has(branch):
    return bool(git(SEED, "ls-remote", "--heads", "origin", branch))


def run_rec(wt, branch0, run_id="rTEST-195", task_id=195):
    return {"run_id": run_id, "task_id": task_id, "agent": "M",
            "worktree": wt, "branch0": branch0}


print("board #195 — push at run end\n")

# 1 ── happy path ───────────────────────────────────────────────────────────
wt = agent_tree(branch="feat/happy-195")
res = disp.push_run_branch(run_rec(wt, "dev"))
ok("1 happy: pushed", res["pushed"] and res["reason"] == "ok", res)
ok("1 happy: branch is on origin", origin_has("feat/happy-195"))
ok("1 happy: branch reported back", res["branch"] == "feat/happy-195", res)
ok("1 happy: upstream tracking set",
   git(wt, "rev-parse", "--abbrev-ref", "feat/happy-195@{upstream}")
   == "origin/feat/happy-195")

# 2 ── refuses dev ──────────────────────────────────────────────────────────
# branch0 is deliberately something else, so ONLY the protected rail is in play.
wt = agent_tree(branch=None, commits=1)          # a commit sitting on dev
dev_before = git(SEED, "rev-parse", "origin/dev")
res = disp.push_run_branch(run_rec(wt, "feat/somewhere-else"))
ok("2 dev: refused", not res["pushed"])
ok("2 dev: refused AS PROTECTED", res["reason"] == "protected", res)
ok("2 dev: origin/dev did not move",
   git(SEED, "ls-remote", ORIGIN, "refs/heads/dev").split()[0] == dev_before)

# 3 ── refuses main (absent from origin — nothing else can decline it) ──────
wt = agent_tree(branch="main")
res = disp.push_run_branch(run_rec(wt, "dev"))
ok("3 main: refused AS PROTECTED", res["reason"] == "protected", res)
ok("3 main: not on origin", not origin_has("main"))
wt = agent_tree(branch="master")
ok("3 master: refused AS PROTECTED",
   disp.push_run_branch(run_rec(wt, "dev"))["reason"] == "protected")

# 4 ── refuses a dirty worktree ─────────────────────────────────────────────
wt = agent_tree(branch="feat/dirty-mod-195", dirty="modified")
res = disp.push_run_branch(run_rec(wt, "dev"))
ok("4 dirty(modified): refused AS DIRTY", res["reason"] == "dirty", res)
ok("4 dirty(modified): not on origin", not origin_has("feat/dirty-mod-195"))
wt = agent_tree(branch="feat/dirty-untracked-195", dirty="untracked")
res = disp.push_run_branch(run_rec(wt, "dev"))
ok("4 dirty(untracked): refused AS DIRTY", res["reason"] == "dirty", res)
ok("4 dirty(untracked): not on origin", not origin_has("feat/dirty-untracked-195"))

# 5 ── refuses a branch the run did not create (a): HEAD never moved ────────
wt = agent_tree(branch="feat/preexisting-195")
res = disp.push_run_branch(run_rec(wt, "feat/preexisting-195"))
ok("5 not-ours(a): refused AS NOT-THIS-RUNS-BRANCH",
   res["reason"] == "not-this-runs-branch", res)
ok("5 not-ours(a): not on origin", not origin_has("feat/preexisting-195"))

# 6 ── refuses a branch the run did not create (b): already on origin ───────
wt = agent_tree(branch="feat/published-195")
git(wt, "push", "--quiet", "origin", "feat/published-195")
head_before = git(SEED, "ls-remote", ORIGIN, "refs/heads/feat/published-195").split()[0]
open(f"{wt}/extra.txt", "w").write("more\n")
git(wt, "add", "-A")
git(wt, "commit", "--quiet", "-m", "extra")
res = disp.push_run_branch(run_rec(wt, "dev"))
ok("6 not-ours(b): refused AS EXISTS-ON-ORIGIN",
   res["reason"] == "exists-on-origin", res)
ok("6 not-ours(b): published branch did not advance",
   git(SEED, "ls-remote", ORIGIN, "refs/heads/feat/published-195").split()[0]
   == head_before)

# 7 ── nothing ahead of origin/dev ──────────────────────────────────────────
wt = agent_tree(branch="feat/empty-195", commits=0)
res = disp.push_run_branch(run_rec(wt, "dev"))
ok("7 empty: declined AS NOTHING-AHEAD", res["reason"] == "nothing-ahead", res)
ok("7 empty: not on origin", not origin_has("feat/empty-195"))

# 8 ── kill switch ──────────────────────────────────────────────────────────
wt = agent_tree(branch="feat/killswitch-195")
open(disp.NO_PUSH_FILE, "w").close()
res = disp.push_run_branch(run_rec(wt, "dev"))
os.remove(disp.NO_PUSH_FILE)
ok("8 NO_PUSH: declined AS KILL-SWITCH", res["reason"] == "kill-switch", res)
ok("8 NO_PUSH: nothing pushed", not origin_has("feat/killswitch-195"))
# ...and the very same run pushes once the switch is gone (proves case 8 was the
# switch talking, not an unrelated refusal).
ok("8 NO_PUSH: pushes after removal", disp.push_run_branch(run_rec(wt, "dev"))["pushed"])

# 9 ── pre-#195 run records (state.json written by the old dispatcher) ──────
ok("9 legacy: no worktree key → declines",
   disp.push_run_branch({"run_id": "rOLD", "task_id": 1})["reason"] == "no-worktree")
ok("9 legacy: worktree gone from disk → declines",
   disp.push_run_branch({"run_id": "rOLD", "task_id": 1,
                         "worktree": f"{TMP}/vanished", "branch0": "dev"}
                        )["reason"] == "no-worktree")
ok("9 legacy: no branch0 key → declines",
   disp.push_run_branch({"run_id": "rOLD", "task_id": 1,
                         "worktree": agent_tree(branch="feat/legacy-195")}
                        )["reason"] == "no-start-branch")

# 10 ── survives a push that ERRORS (the fail-open + loud rail) ─────────────
# A remote that is REACHABLE but rejects the push — i.e. the branch passed every
# rail and the push itself failed, which is the case the loud half exists for.
# (An unreachable remote is a different, earlier decline: ls-remote-failed.)
REJECT = f"{TMP}/reject.git"
git(TMP, "clone", "--quiet", "--bare", ORIGIN, REJECT)
hook = f"{REJECT}/hooks/pre-receive"
open(hook, "w").write("#!/bin/sh\necho 'protected branch: push rejected' >&2\nexit 1\n")
os.chmod(hook, 0o755)
wt = agent_tree(branch="feat/rejected-195", remote=REJECT)
res = disp.try_push_run_branch(run_rec(wt, "dev"))
ok("10 push error: not pushed, reason names the failure",
   not res["pushed"] and res["reason"].startswith("push-failed"), res)


def dead_pid():
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def finished_run(wt, branch0, run_id, task_id=195, result="done"):
    """A run record whose process is gone and whose log carries terminal JSON."""
    log = f"{disp.LOG_DIR}/{run_id}.log"
    open(log, "w").write(json.dumps(
        {"type": "result", "is_error": False, "result": result}) + "\n")
    r = run_rec(wt, branch0, run_id=run_id, task_id=task_id)
    r.update({"pid": dead_pid(), "t0": disp.time.time(), "log": log, "active": True})
    return r


CALLS.clear()
st = {"runs": [finished_run(wt, "dev", "rBAD-195")]}
disp.reap(st)                                    # must NOT raise
ok("10 reap survives a failed push", not st["runs"][0]["active"])
ok("10 reap still reported the result",
   any(m == "POST" and p == "dev_task_comments" and b["author"] == "M·auto"
       for m, p, b in CALLS))
ok("10 reap still moved the task to Review",
   any(m == "PATCH" and p.startswith("dev_tasks?") and b.get("status") == "Review"
       for m, p, b in CALLS))
ok("10 reap still finished run_history",
   any(m == "PATCH" and p.startswith("run_history?") and b.get("outcome") == "ok"
       for m, p, b in CALLS))
ok("10 push failure is LOUD on the thread",
   any(m == "POST" and p == "dev_task_comments"
       and "auto-push FAILED" in (b.get("body") or "") for m, p, b in CALLS))
ok("10 a FAILED push records no pushed_branch",
   not any("pushed_branch" in (b or {}) for m, p, b in CALLS))

# 11 ── survives git itself blowing up ──────────────────────────────────────
real_git = disp._git


def exploding_git(wt, *a, **k):
    # rev-parse still works: current_branch() has its own guard (it runs at
    # DISPATCH time, where it must never block a spawn), so the outer fail-open
    # wrapper is only reachable if something LATER in the leg throws.
    if a and a[0] == "rev-parse":
        return real_git(wt, *a, **k)
    raise RuntimeError("git exploded")


disp._git = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("git exploded"))
ok("11 exception: current_branch() swallows it (it runs at dispatch time)",
   disp.current_branch(f"{TMP}/nope") is None)
disp._git = exploding_git
res = disp.try_push_run_branch(run_rec(agent_tree(branch="feat/boom-195"), "dev"))
ok("11 exception: swallowed, named", not res["pushed"]
   and res["reason"].startswith("error:"), res)
CALLS.clear()
st = {"runs": [finished_run(agent_tree(branch="feat/boom2-195"), "dev", "rBOOM-195")]}
disp.reap(st)                                    # must NOT raise
ok("11 exception: reap completed anyway", not st["runs"][0]["active"])
ok("11 exception: result still reported",
   any(m == "PATCH" and p.startswith("run_history?") and b.get("outcome") == "ok"
       for m, p, b in CALLS))
disp._git = real_git

# 12 ── never --force (asserted on the argv git actually receives) ──────────
ARGV = []
disp._git = lambda wt, *a, **k: (ARGV.append(list(a)), real_git(wt, *a, **k))[1]
wt = agent_tree(branch="feat/argv-195")
disp.push_run_branch(run_rec(wt, "dev"))
disp._git = real_git
pushes = [a for a in ARGV if a and a[0] == "push"]
ok("12 exactly one push", len(pushes) == 1, pushes)
ok("12 push is `push -u origin <branch>`",
   pushes[0] == ["push", "-u", "origin", "feat/argv-195"], pushes[0])
ok("12 no --force anywhere in the leg",
   not any(x in ("-f", "--force", "--force-with-lease")
           for a in ARGV for x in a), ARGV)

# 13 ── reap integration: report AND push AND record, in one pass ───────────
CALLS.clear()
wt = agent_tree(branch="feat/reap-195")
st = {"runs": [finished_run(wt, "dev", "rOK-195", task_id=195)]}
disp.reap(st)
ok("13 reap: branch reached origin", origin_has("feat/reap-195"))
ok("13 reap: run marked collected", not st["runs"][0]["active"])
ok("13 reap: pushed branch recorded on the run record",
   st["runs"][0].get("pushed_branch") == "feat/reap-195")
rh = [b for m, p, b in CALLS if m == "PATCH" and p == "run_history?run_id=eq.rOK-195"]
ok("13 reap: run_history.pushed_branch recorded (dashboard #193 input)",
   any(b.get("pushed_branch") == "feat/reap-195" and b.get("pushed_at") for b in rh), rh)
ok("13 reap: the finish PATCH is still separate + terminal",
   any(b.get("outcome") == "ok" and "pushed_branch" not in b for b in rh), rh)
ok("13 reap: agent result still posted",
   any(m == "POST" and p == "dev_task_comments" and b["author"] == "M·auto"
       for m, p, b in CALLS))
ok("13 reap: no push-failure noise on a clean push",
   not any("auto-push FAILED" in (b.get("body") or "")
           for m, p, b in CALLS if m == "POST" and p == "dev_task_comments"))

# ...and an un-migrated DB (no pushed_branch column) cannot break the reap.
CALLS.clear()
wt = agent_tree(branch="feat/unmigrated-195")
st = {"runs": [finished_run(wt, "dev", "rUNMIG-195")]}


def api_rejecting_new_columns(method, path, body=None, prefer=None):
    if body and "pushed_branch" in body:
        raise RuntimeError("PGRST204: column run_history.pushed_branch does not exist")
    return fake_api(method, path, body, prefer)


disp.api = api_rejecting_new_columns
disp.reap(st)                                    # must NOT raise
disp.api = fake_api
ok("13 un-migrated DB: reap completed", not st["runs"][0]["active"])
ok("13 un-migrated DB: branch still pushed", origin_has("feat/unmigrated-195"))

# 14 ── dispatch side records worktree + branch0 ────────────────────────────
# Without this, reap() cannot answer "did this run create the branch it is on?"
# — the question is unanswerable after the fact, so it is captured at dispatch.
LANE = agent_tree(branch=None, commits=0)        # an agent worktree sitting on dev
TASK = {"id": 901, "title": "t", "description": "d", "assignee": "M",
        "status": "Todo", "tags": [], "checklist": []}
disp.headless_agents = lambda: ({"M": {"letter": "M", "status": "headless",
                                       "worktree": LANE, "scope": "", "boundaries": ""}},
                                "test")
disp.triage_todo = lambda agents, done, st=None: ([TASK], [])  # st arg added by #198 (V4.21); the stub ignores it
disp.run_requested = lambda agents, done: ([], [])
disp.mentions_for = lambda role: ("", [])
disp.mentions_delivered = lambda role, ids: 0
disp.spawn = lambda agent, task, prompt, log: subprocess.Popen(["sleep", "30"],
                                                               start_new_session=True)
CALLS.clear()
json.dump({"runs": []}, open(disp.STATE_FILE, "w"))
disp.one_pass(True, only_task=901)
rec = json.load(open(disp.STATE_FILE))["runs"][-1]
disp.kill_run_group(rec["pid"])
ok("14 dispatch records the worktree", rec.get("worktree") == LANE, rec)
ok("14 dispatch records the START branch", rec.get("branch0") == "dev", rec)
prompt_calls = [b for m, p, b in CALLS if m == "POST" and p == "dev_task_comments"]
ok("14 dispatch still announced the run",
   any("dispatched to M·auto" in (b.get("body") or "") for b in prompt_calls))

# ── the REAL shape: charter §4 fresh worktrees (cases 15-21) ───────────────
# Everything above hands push_run_branch() a worktree already sitting on the
# feature branch. No dispatch produces that. What a dispatch produces is a lane
# worktree parked on dev, and — some minutes later — a SECOND directory the
# agent cut with `git worktree add`. These helpers build exactly that.

def lane_tree():
    """An agent's REGISTERED worktree, as the registry hands it to the loop and
    as the charter leaves it between runs: on dev, nothing local."""
    return agent_tree(branch=None, commits=0)


def snapshot(lane):
    """What the dispatcher records at spawn time."""
    return disp.list_worktrees(lane)


def add_worktree(lane, branch, commits=1, dirty=None, base="origin/dev",
                 owner="rWT-195"):
    """The charter §4 move, performed AFTER the dispatch snapshot:
    `git worktree add ../Kevbot-wt-<slug>-<run-id> -b <branch> origin/dev`.

    `owner` is the run id carried in the DIRECTORY NAME. Board #218 made that
    token the ownership discriminator — the elimination test these cases were
    first written against ("is another ACTIVE run's snapshot also missing this
    tree?") was wrong in both directions and is gone — so a §4 worktree now has
    to declare whose it is, exactly as the WORKTREE CONTRACT tells the agent to.
    """
    TREES[0] += 1
    path = f"{TMP}/linked{TREES[0]}-{owner}" if owner else f"{TMP}/linked{TREES[0]}"
    git(lane, "worktree", "add", "--quiet", "-b", branch, path, base)
    for i in range(commits):
        open(f"{path}/work{i}.txt", "w").write(f"work {i}\n")
        git(path, "add", "-A")
        git(path, "commit", "--quiet", "-m", f"work {i}")
    if dirty == "modified":
        open(f"{path}/README.md", "a").write("half-finished edit\n")
    elif dirty == "untracked":
        open(f"{path}/scratch.txt", "w").write("left behind\n")
    return os.path.realpath(path)


def dispatched(lane, run_id="rWT-195", task_id=195, branch0="dev"):
    """A run record as one_pass() writes it: worktree + branch0 + worktrees0."""
    r = run_rec(lane, branch0, run_id=run_id, task_id=task_id)
    r["worktrees0"] = snapshot(lane)
    return r


# 15 ── THE BUG: the work is in a worktree created after dispatch ───────────
# Under the first cut this pushed NOTHING: the registered tree is on dev, so the
# protected rail declined and the leg ended there. The branch has to be
# DISCOVERED. This case is the whole reason for the rework — it must fail if the
# push ever goes back to assuming r["worktree"] holds the work.
lane = lane_tree()
r = dispatched(lane)                             # snapshot taken FIRST...
FRESH = add_worktree(lane, "feat/fresh-195")     # ...then the agent cuts its tree
dev_before = git(SEED, "ls-remote", ORIGIN, "refs/heads/dev").split()[0]
res = disp.push_run_branch(r)
ok("15 §4 shape: PUSHED", res["pushed"], res)
ok("15 §4 shape: the fresh worktree's branch is on origin",
   origin_has("feat/fresh-195"))
ok("15 §4 shape: branch reported back", res["branch"] == "feat/fresh-195", res)
ok("15 §4 shape: it was the DISCOVERED tree that pushed",
   any(x["pushed"] and x["worktree"] == FRESH for x in res["results"]), res)
# ...and the registered tree still declined, which is precisely what used to end
# the leg. Both facts in one run: the old anchor is wrong AND still guarded.
ok("15 §4 shape: the registered tree declined AS PROTECTED",
   any(x["worktree"] == lane and x["reason"] == "protected"
       for x in res["results"]), res)
ok("15 §4 shape: origin/dev did not move",
   git(SEED, "ls-remote", ORIGIN, "refs/heads/dev").split()[0] == dev_before)

# 16 ── every rail re-applied to a DISCOVERED worktree ──────────────────────
# A discovered target is not a trusted target. Same rails, same named reasons.
def fresh_case(branch, **kw):
    lane = lane_tree()
    r = dispatched(lane)
    add_worktree(lane, branch, **kw)
    return disp.push_run_branch(r), r


def reasons(res, exclude_lane):
    return [x["reason"] for x in res["results"] if x["worktree"] != exclude_lane]


res, r = fresh_case("feat/fresh-dirty-195", dirty="modified")
ok("16 discovered+dirty: refused AS DIRTY",
   reasons(res, r["worktree"]) == ["dirty"], res)
ok("16 discovered+dirty: not on origin", not origin_has("feat/fresh-dirty-195"))
res, r = fresh_case("feat/fresh-untracked-195", dirty="untracked")
ok("16 discovered+untracked: refused AS DIRTY",
   reasons(res, r["worktree"]) == ["dirty"], res)
res, r = fresh_case("main")                      # absent from origin: only the
ok("16 discovered `main`: refused AS PROTECTED",  # protected rail can decline it
   reasons(res, r["worktree"]) == ["protected"], res)
ok("16 discovered `main`: not on origin", not origin_has("main"))
res, r = fresh_case("feat/fresh-empty-195", commits=0)
ok("16 discovered+no commits: declined AS NOTHING-AHEAD",
   reasons(res, r["worktree"]) == ["nothing-ahead"], res)
ok("16 discovered+no commits: not on origin", not origin_has("feat/fresh-empty-195"))
# already published → advancing it is not this loop's job
lane = lane_tree()
r = dispatched(lane)
pub = add_worktree(lane, "feat/fresh-published-195")
git(pub, "push", "--quiet", "origin", "feat/fresh-published-195")
head_before = git(SEED, "ls-remote", ORIGIN,
                  "refs/heads/feat/fresh-published-195").split()[0]
open(f"{pub}/extra.txt", "w").write("more\n")
git(pub, "add", "-A")
git(pub, "commit", "--quiet", "-m", "extra")
res = disp.push_run_branch(r)
ok("16 discovered+published: refused AS EXISTS-ON-ORIGIN",
   reasons(res, lane) == ["exists-on-origin"], res)
ok("16 discovered+published: it did not advance",
   git(SEED, "ls-remote", ORIGIN,
       "refs/heads/feat/fresh-published-195").split()[0] == head_before)
# the kill switch beats discovery too
lane = lane_tree()
r = dispatched(lane)
add_worktree(lane, "feat/fresh-killswitch-195")
open(disp.NO_PUSH_FILE, "w").close()
res = disp.push_run_branch(r)
os.remove(disp.NO_PUSH_FILE)
ok("16 discovered+NO_PUSH: declined AS KILL-SWITCH", res["reason"] == "kill-switch", res)
ok("16 discovered+NO_PUSH: nothing pushed", not origin_has("feat/fresh-killswitch-195"))
ok("16 discovered: pushes once the switch is gone",
   disp.push_run_branch(r)["pushed"] and origin_has("feat/fresh-killswitch-195"))

# 17 ── a worktree that is not ours is not ours ─────────────────────────────
# SUPERSEDED BY BOARD #218 — and kept, because the PRINCIPLE it defends is
# unchanged: pushing a peer's branch is the one genuinely harmful outcome
# available here (it publishes unfinished work AND poisons that peer's own push,
# which would then decline exists-on-origin). What changed is how ownership is
# decided. V4.23 inferred it by ELIMINATION from the other in-flight runs, which
# declined whenever ANY older peer was alive and ADOPTED the tree once its owner
# DIED. V4.27 reads the run id the agent put in the directory name, so the answer
# no longer depends on who else is running. #218's own file carries the full set;
# these two assert that this file's rail did not merely move.
lane = lane_tree()
before = snapshot(lane)
mine = dispatched(lane, run_id="rMINE-195")
peer_older = {"run_id": "rPEER-OLD", "task_id": 196, "worktrees0": before}
THEIRS = add_worktree(lane, "feat/peers-work-195", owner="rPEER-OLD")
ok("17 a peer's declared worktree is not ours",
   disp.fresh_worktrees(mine, lane, "t", [peer_older]) == [], THEIRS)
res = disp.push_run_branch(mine, [peer_older])
ok("17 nothing pushed", not res["pushed"], res)
ok("17 the peer's branch did NOT reach origin", not origin_has("feat/peers-work-195"))
ok("17 ...and it is STILL not ours after that peer ends (the #218 orphan rail)",
   not disp.push_run_branch(mine, [])["pushed"] and
   not origin_has("feat/peers-work-195"))
# ...while our OWN declared worktree pushes with that same peer in flight — the
# false decline #218 fixed, asserted here so this file cannot regress it either.
MINE_WT = add_worktree(lane, "feat/mine-195", owner="rMINE-195")
ok("17 our own declared worktree IS ours, peer or no peer",
   disp.fresh_worktrees(mine, lane, "t", [peer_older]) == [MINE_WT])
ok("17 ...and it pushes", disp.push_run_branch(mine, [peer_older])["pushed"]
   and origin_has("feat/mine-195"))

# 18 ── reap end-to-end in the §4 shape ─────────────────────────────────────
CALLS.clear()
lane = lane_tree()
run = finished_run(lane, "dev", "rS4-195")
run["worktrees0"] = snapshot(lane)
add_worktree(lane, "feat/reap-fresh-195", owner="rS4-195")
st = {"runs": [run]}
disp.reap(st)
ok("18 reap §4: the discovered branch reached origin", origin_has("feat/reap-fresh-195"))
ok("18 reap §4: recorded on the run record",
   st["runs"][0].get("pushed_branch") == "feat/reap-fresh-195", st["runs"][0])
ok("18 reap §4: run_history.pushed_branch recorded (dashboard #193 input)",
   any(b.get("pushed_branch") == "feat/reap-fresh-195" and b.get("pushed_at")
       for m, p, b in CALLS if m == "PATCH" and p.startswith("run_history?")), CALLS)
ok("18 reap §4: the agent result was still reported",
   any(m == "POST" and p == "dev_task_comments" and b["author"] == "M·auto"
       for m, p, b in CALLS))
ok("18 reap §4: no push-failure noise", not any(
    "auto-push FAILED" in (b.get("body") or "") for m, p, b in CALLS if m == "POST"))

# 19 ── reap() itself supplies the other runs, and a live peer does not block ─
# The rail is only real if reap wires the run set; passing it by hand (case 17)
# would not notice reap forgetting to. SUPERSEDED BY #218 in its VERDICT: under
# V4.23 an in-flight peer blocked this push, and that is the false decline that
# stranded a real commit on 07-30 at CONCURRENCY=4. The finishing run now
# publishes its own declared worktree while the peer keeps working, and the
# peer's tree is left untouched.
CALLS.clear()
lane = lane_tree()
snap = snapshot(lane)
run = finished_run(lane, "dev", "rRIVAL-195")
run["worktrees0"] = snap
alive = subprocess.Popen(["sleep", "60"], start_new_session=True)
peer_log = f"{disp.LOG_DIR}/rPEER-195.log"
open(peer_log, "w").write("still working\n")     # no terminal JSON → still running
peer = {"run_id": "rPEER-195", "task_id": 196, "agent": "E", "worktree": lane,
        "branch0": "dev", "worktrees0": snap, "pid": alive.pid,
        "t0": disp.time.time(), "log": peer_log, "active": True}
add_worktree(lane, "feat/rival-195", owner="rRIVAL-195")
add_worktree(lane, "feat/peer-still-working-195", owner="rPEER-195")
st = {"runs": [run, peer]}
disp.reap(st)
alive.kill()
ok("19 reap: the finishing run published its OWN worktree", origin_has("feat/rival-195"))
ok("19 reap: the in-flight peer's worktree was left alone",
   not origin_has("feat/peer-still-working-195"))
ok("19 reap: the finished run was still collected", not st["runs"][0]["active"])
ok("19 reap: the peer is still in flight", st["runs"][1]["active"])
ok("19 reap: the finished run still reported",
   any(m == "PATCH" and p.startswith("run_history?") and b.get("outcome") == "ok"
       for m, p, b in CALLS))

# 20 ── both legs at once: it moved its own tree AND cut a fresh one ────────
lane = agent_tree(branch="feat/inplace-195")     # registered tree moved off dev
r = run_rec(lane, "dev")
r["worktrees0"] = snapshot(lane)
add_worktree(lane, "feat/alongside-195", owner="rTEST-195")
res = disp.push_run_branch(r)
ok("20 both: two targets considered", len(res["results"]) == 2, res)
ok("20 both: in-place branch on origin", origin_has("feat/inplace-195"))
ok("20 both: fresh-worktree branch on origin", origin_has("feat/alongside-195"))
ok("20 both: both reported, comma-joined for run_history",
   res["branch"] == "feat/inplace-195,feat/alongside-195", res)

# 21 ── discovery unavailable → decline, never crash, never guess ───────────
# A pre-#195 run record has no snapshot, so "which worktree did this run create?"
# has no answer and the fresh leg must stay OFF rather than push what it finds.
lane = lane_tree()
add_worktree(lane, "feat/legacy-fresh-195")
res = disp.push_run_branch({"run_id": "rOLD2", "task_id": 1,
                            "worktree": lane, "branch0": "dev"})
ok("21 no snapshot: declines on the registered tree only",
   res["reason"] == "protected" and len(res["results"]) == 1, res)
ok("21 no snapshot: did NOT push the worktree it happened to find",
   not origin_has("feat/legacy-fresh-195"))
# snapshot but no branch0 (a run that started detached): fresh leg still works,
# and with nothing to push the run says so by name rather than silently.
lane = lane_tree()
r = {"run_id": "rDETACHED-195", "task_id": 1, "worktree": lane,
     "worktrees0": snapshot(lane)}
ok("21 no branch0: no candidate at all is NAMED",
   disp.push_run_branch(r)["reason"] == "no-candidate", r)
add_worktree(lane, "feat/detached-run-195", owner="rDETACHED-195")
ok("21 no branch0: the fresh leg still pushes",
   disp.push_run_branch(r)["pushed"] and origin_has("feat/detached-run-195"))
# git worktree list blowing up must disable the leg, not the reap
lane = lane_tree()
r = dispatched(lane)
add_worktree(lane, "feat/unlistable-195")
real_git = disp._git


def no_worktree_list(wt, *a, **k):
    if a and a[0] == "worktree":
        raise RuntimeError("git worktree exploded")
    return real_git(wt, *a, **k)


disp._git = no_worktree_list
res = disp.try_push_run_branch(r)
disp._git = real_git
ok("21 unlistable: leg went quiet, run did not crash", not res["pushed"], res)
ok("21 unlistable: nothing pushed", not origin_has("feat/unlistable-195"))
ok("21 unlistable: list_worktrees() itself is total",
   disp.list_worktrees(f"{TMP}/does-not-exist") == [])

shutil.rmtree(TMP, ignore_errors=True)
print(f"\nALL {PASS} CHECKS PASSED — board #195 push-at-run-end")
