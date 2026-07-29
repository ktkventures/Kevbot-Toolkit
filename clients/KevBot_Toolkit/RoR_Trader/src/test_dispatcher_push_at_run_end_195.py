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
disp.triage_todo = lambda agents, done: ([TASK], [])
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

shutil.rmtree(TMP, ignore_errors=True)
print(f"\nALL {PASS} CHECKS PASSED — board #195 push-at-run-end")
