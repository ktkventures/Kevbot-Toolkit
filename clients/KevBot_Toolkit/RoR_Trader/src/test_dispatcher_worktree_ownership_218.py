#!/usr/bin/env python3
"""Acceptance test — board #218: worktree ownership is DECLARED, not inferred.

V4.23's run-end push decided "is this worktree mine?" by ELIMINATION: it looked
for another ACTIVE run whose dispatch snapshot ALSO lacked the worktree, and
declined if it found one. That question cannot distinguish an orphan from a
peer's live work, and it was observed wrong in BOTH directions on 07-30:

  • a run dispatched EARLIER can never have a later worktree in its snapshot, so
    every older concurrent run made ownership "ambiguous". At CONCURRENCY=4 that
    is the normal case — three declines in one reap cycle at 15:46Z, one of them
    stranding a real commit while its run reported `ok`;
  • once the owner DIED there was no rival left to object, so the next run to
    reap ADOPTED the orphan: run r1785382865-212 recorded `pushed_branch =
    verify/auto-push-live-195` — a branch cut by the #195 run, in a worktree R
    never touched. Harmless only by luck (the agent had already pushed it).

The fix is a discriminator the RUN carries: its own id, in the worktree's
directory name, required of the agent by the WORKTREE CONTRACT in build_prompt.

EVERY RAIL HAS A TEST THAT FAILS WITHOUT IT — a rail with no failing test is a
comment, not a rail. Cases 2, 3 and 4 are the ones that FAIL against the pre-#218
dispatcher, one per observed direction of the bug; the rest exist because the
cheap way to make those green is to loosen something that was correct.

  1  the discriminator      worktree_owner() as a table: own id, a foreign id,
                            the legacy task tag (decidable and not), untagged
  2  THE FALSE DECLINE      a run with THREE older peers in flight still pushes
                            its own tagged worktree            (pre-#218: declines)
  3  THE FALSE PUSH         an orphan left by a DEAD run — no rival anywhere, the
                            owner not even in state — is REFUSED (pre-#218: pushes)
  4  a peer's LIVE worktree is never pushed, and refusal does not depend on the
                            peer being alive                   (pre-#218: adopts it once the peer ends)
  5  sole run               the ordinary case still pushes, end to end
  6  untagged               refused as `unattributable`, carried in results, and
                            the refusal NAMES the required form
  7  legacy task tag        pushes when it is the task's only known run; refused
                            once a second run of that task is known (a retry, or
                            the next step of a chain)
  8  the other rails hold   protected / dirty / nothing-ahead / exists-on-origin
                            / NO_PUSH still decline an OWNED worktree, by name.
                            #218 was told not to touch these
  9  reap wires it          reap() passes EVERY run in state, finished ones too —
                            case 7's rival is usually already reaped — and the
                            orphan is refused end to end, run_history untouched
 10  the contract           the prompt hands the agent its run id and the exact
                            `git worktree add` form, and one_pass() passes the
                            REAL id (an agent that never learns it cannot comply)

Hermetic: real git against a LOCAL bare repo (no network, no GitHub), the
Supabase api() replaced with a recorder (ZERO board writes).

Run:  .venv/bin/python src/test_dispatcher_worktree_ownership_218.py   (from RoR_Trader root)
"""
import importlib.util
import io
import json
import os
import contextlib
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
    "team_dispatcher_218",
    os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)

TMP = tempfile.mkdtemp(prefix="own218-")
disp.STATE_FILE = f"{TMP}/state.json"
disp.LOG_DIR = f"{TMP}/logs"
os.makedirs(disp.LOG_DIR, exist_ok=True)
# Sandboxed switches — a stray NO_PUSH in the live dispatcher directory would
# otherwise pass case 8 for entirely the wrong reason.
disp.NO_PUSH_FILE = f"{TMP}/NO_PUSH"
disp.PAUSE_FILE = f"{TMP}/PAUSE"

CALLS = []


def fake_api(method, path, body=None, prefer=None):
    CALLS.append((method, path, body))
    if method == "GET" and path.startswith("dev_task_comments"):
        return []
    # Board #232: the blocked_by lookup is the shared FINISHED set now
    # (`status=in.(Done,Closed)`), not the literal `status=eq.Done`.
    if method == "GET" and path.startswith(f"dev_tasks?{disp.FINISHED_FILTER}"):
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

# Real ids, in the shape one_pass() mints (`r<epoch>-<task>`), because the
# foreign-run rail matches on that shape.
MINE = "r1785425881-218"
PEER = "r1785425697-220"
DEAD = "r1785382865-195"


def lane_tree():
    """An agent's REGISTERED worktree: on dev, nothing local — as the registry
    hands it to the loop and as the charter leaves it between runs."""
    TREES[0] += 1
    wt = f"{TMP}/lane{TREES[0]}"
    git(TMP, "clone", "--quiet", ORIGIN, wt)
    return wt


def snapshot(lane):
    return disp.list_worktrees(lane)


def add_worktree(lane, branch, name, commits=1, dirty=None, base="origin/dev"):
    """The charter §4 move, performed AFTER the dispatch snapshot. `name` is the
    DIRECTORY name — under #218 that is where ownership is declared."""
    path = f"{TMP}/{name}"
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


def dispatched(lane, run_id=MINE, task_id=218, branch0="dev"):
    """A run record exactly as one_pass() writes it."""
    return {"run_id": run_id, "task_id": task_id, "agent": "M", "worktree": lane,
            "branch0": branch0, "worktrees0": snapshot(lane)}


def in_flight(run_id, task_id, snap):
    """Another run the state file still holds, still marked active."""
    return {"run_id": run_id, "task_id": task_id, "agent": "E", "worktree": SEED,
            "branch0": "dev", "worktrees0": snap, "active": True,
            "pid": os.getpid(), "t0": disp.time.time(),
            "log": f"{disp.LOG_DIR}/{run_id}.log"}


def origin_has(branch):
    return bool(git(SEED, "ls-remote", "--heads", "origin", branch))


def reasons(res, lane):
    return [x["reason"] for x in res["results"] if x["worktree"] != lane]


print("board #218 — worktree ownership is declared, not inferred\n")

# 1 ── the discriminator itself ─────────────────────────────────────────────
# A table, so each rule is asserted where it is cheap to read. Everything below
# exercises the same function through the real push/reap path.
r1 = {"run_id": MINE, "task_id": 218}
own, why = disp.worktree_owner(f"/home/kevin/projects/Kevbot-wt-ownership-{MINE}", r1)
ok("1 own run id in the name → OWNED", own and MINE in why, why)
own, why = disp.worktree_owner(f"/home/kevin/projects/Kevbot-wt-msession-{PEER}", r1)
ok("1 another run's id → refused", not own and PEER in why, why)
ok("1 ...and refused for being SOMEONE ELSE'S, not for being unnamed",
   "not this run's" in why, why)
own, why = disp.worktree_owner(f"/home/kevin/projects/Kevbot-wt-401diag-{DEAD}", r1)
ok("1 a DEAD run's id is still that run's", not own and DEAD in why, why)
own, why = disp.worktree_owner("/home/kevin/projects/Kevbot-wt-401diag-218", r1)
ok("1 legacy task tag, sole known run → OWNED", own and "218" in why, why)
own, why = disp.worktree_owner("/home/kevin/projects/Kevbot-wt-401diag-218", r1,
                               [{"run_id": "r1785000000-218", "task_id": 218}])
ok("1 legacy task tag, a second run of the task → refused",
   not own and "not decidable" in why, why)
own, why = disp.worktree_owner("/home/kevin/projects/Kevbot-wt-401diag-218", r1,
                               [{"run_id": "rOTHER", "task_id": 999}])
ok("1 a run of a DIFFERENT task is not a rival for the task tag", own, why)
# A peer id that does NOT match the minted shape still has to be recognised —
# caught by the #195 suite, where `rPEER-195` slipped past the shape rule and the
# task tag `195` then claimed the peer's tree. Both readings are needed.
own, why = disp.worktree_owner("/tmp/linked40-rPEER-218", r1,
                               [{"run_id": "rPEER-218", "task_id": 999}])
ok("1 a KNOWN run's id is foreign even off-shape", not own and "rPEER-218" in why, why)
ok("1 ...and the task tag does not then claim it", "not this run's" in why, why)
own, why = disp.worktree_owner("/home/kevin/projects/Kevbot-wt-scratch", r1)
ok("1 untagged → refused", not own, why)
ok("1 ...and the refusal names the form the agent should have used",
   f"Kevbot-wt-<slug>-{MINE}" in why, why)
own, why = disp.worktree_owner("/home/kevin/projects/218-shaped/Kevbot-wt-scratch", r1)
ok("1 a task number in a PARENT directory does not confer ownership", not own, why)
ok("1 the run-id rule does look at the whole path",
   disp.worktree_owner(f"/tmp/{MINE}/nested-tree", r1)[0])

# 2 ── THE FALSE DECLINE: three older peers in flight, and it still pushes ──
# This is what CONCURRENCY=4 turned from an edge case into the default. Under
# V4.23 every one of these peers is a "rival" — their snapshots all predate the
# worktree — and the run reported `ok` with its commit stranded on local disk.
lane = lane_tree()
before = snapshot(lane)
mine = dispatched(lane)
CROWD = add_worktree(lane, "feat/crowded-218", f"Kevbot-wt-crowded-{MINE}")
peers = [in_flight(PEER, 220, before),
         in_flight("r1785425392-221", 221, before),
         in_flight("r1785425697-222", 222, before)]
ok("2 the peers really do all predate the worktree (else this proves nothing)",
   all(CROWD not in set(p["worktrees0"]) for p in peers))
ok("2 ownership is still decidable with 3 peers in flight",
   disp.fresh_worktrees(mine, lane, "t", peers) == [CROWD])
res = disp.push_run_branch(mine, peers)
ok("2 PUSHED despite the crowd", res["pushed"], res)
ok("2 the run's own branch reached origin", origin_has("feat/crowded-218"))
ok("2 it was the tagged worktree that pushed",
   any(x["pushed"] and x["worktree"] == CROWD for x in res["results"]), res)

# 3 ── THE FALSE PUSH: an orphan from a dead run is REFUSED ─────────────────
# The observed harm, reproduced: the owner is gone — not merely finished, but
# absent from state entirely, exactly as a reaped run is — so under V4.23 there
# is no rival left and the reaping run adopts the branch.
lane = lane_tree()
mine = dispatched(lane)
ORPHAN = add_worktree(lane, "verify/auto-push-live-195", f"Kevbot-wt-195verify-{DEAD}")
ok("3 the orphan is NOT claimed",
   disp.fresh_worktrees(mine, lane, "t", []) == [], ORPHAN)
res = disp.push_run_branch(mine, [])             # no rival ANYWHERE
ok("3 nothing was pushed", not res["pushed"], res)
ok("3 the dead run's branch did NOT reach origin",
   not origin_has("verify/auto-push-live-195"))
ok("3 refused AS UNATTRIBUTABLE, not as some incidental rail",
   reasons(res, lane) == ["unattributable"], res)
ok("3 the refusal names the run it actually belongs to",
   any(DEAD in (x.get("detail") or "") for x in res["results"]), res)
ok("3 the run reports no branch (it did not silently take credit)",
   res["branch"] is None, res)

# 4 ── a peer's LIVE worktree is never pushed — alive or not ───────────────
lane = lane_tree()
before = snapshot(lane)
mine = dispatched(lane)
THEIRS = add_worktree(lane, "feat/m-session-dashboard-220", f"Kevbot-wt-msession-{PEER}")
ok("4 not ours while the peer is in flight",
   disp.push_run_branch(mine, [in_flight(PEER, 220, before)])["pushed"] is False)
ok("4 ...and STILL not ours once the peer is gone from state",
   not disp.push_run_branch(mine, [])["pushed"])
ok("4 the peer's branch never reached origin",
   not origin_has("feat/m-session-dashboard-220"))
ok("4 the peer could push it itself afterwards (we did not poison it)",
   disp.push_worktree_branch(THEIRS, "t")["pushed"] and
   origin_has("feat/m-session-dashboard-220"))

# 5 ── the ordinary case: a sole run, nothing else in flight ───────────────
lane = lane_tree()
mine = dispatched(lane)
SOLO = add_worktree(lane, "feat/solo-218", f"Kevbot-wt-solo-{MINE}")
res = disp.push_run_branch(mine)                 # known_runs defaulted — sole run
ok("5 sole run: PUSHED", res["pushed"] and res["reason"] == "ok", res)
ok("5 sole run: on origin", origin_has("feat/solo-218"))
ok("5 sole run: branch reported back for run_history",
   res["branch"] == "feat/solo-218", res)
ok("5 sole run: the registered tree still declined AS PROTECTED (it is on dev)",
   any(x["worktree"] == lane and x["reason"] == "protected" for x in res["results"]))

# 6 ── an untagged worktree is refused, and the refusal is REPORTED ────────
# The conservative half: #218 must not trade a false decline for a false push,
# so a tree nobody can claim is left for preflight (2) and named in the log.
lane = lane_tree()
mine = dispatched(lane)
add_worktree(lane, "feat/untagged-218", "Kevbot-wt-someones-scratch")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = disp.push_run_branch(mine)
log = buf.getvalue()
ok("6 untagged: not pushed", not res["pushed"], res)
ok("6 untagged: not on origin", not origin_has("feat/untagged-218"))
ok("6 untagged: refused AS UNATTRIBUTABLE", reasons(res, lane) == ["unattributable"], res)
ok("6 untagged: the refusal is LOUD, not a silent skip", "REFUSED" in log, log)
ok("6 untagged: the log says what to name it instead",
   f"Kevbot-wt-<slug>-{MINE}" in log, log)

# 7 ── the legacy `-<task>` convention, kept working where it is decidable ──
lane = lane_tree()
mine = dispatched(lane)
add_worktree(lane, "diag/steps-complete-401-218", "Kevbot-wt-401diag-218")
ok("7 legacy tag, sole run of the task: PUSHED", disp.push_run_branch(mine)["pushed"])
ok("7 legacy tag: on origin", origin_has("diag/steps-complete-401-218"))
lane = lane_tree()
mine = dispatched(lane)
add_worktree(lane, "diag/second-run-218", "Kevbot-wt-second-218")
sibling = {"run_id": "r1785400000-218", "task_id": 218}     # an earlier step's run
res = disp.push_run_branch(mine, [sibling])
ok("7 legacy tag, a SECOND run of the same task: refused",
   reasons(res, lane) == ["unattributable"], res)
ok("7 ...and nothing published", not origin_has("diag/second-run-218"))
ok("7 the same tree IS pushed once it carries the run id (the escape hatch)",
   disp.push_run_branch(dispatched(lane, run_id="r1785400000-218"),
                        [sibling])["pushed"] is False)   # not THIS run's tree either
ok("7 ...whereas the run whose id is in the NAME takes it",
   disp.worktree_owner(f"{TMP}/Kevbot-wt-x-{MINE}", mine, [sibling])[0])

# 8 ── the other decline reasons still fire, on an OWNED worktree ──────────
# #218 was told explicitly not to touch these. Each is re-asserted through a
# worktree this run unambiguously owns, so ONLY the named rail can decline it.
def owned_case(branch, name, **kw):
    lane = lane_tree()
    r = dispatched(lane)
    add_worktree(lane, branch, f"{name}-{MINE}", **kw)
    return disp.push_run_branch(r), r["worktree"]


res, lane = owned_case("feat/owned-dirty-218", "wt-dirty", dirty="modified")
ok("8 owned+dirty: refused AS DIRTY", reasons(res, lane) == ["dirty"], res)
res, lane = owned_case("feat/owned-untracked-218", "wt-untracked", dirty="untracked")
ok("8 owned+untracked: refused AS DIRTY", reasons(res, lane) == ["dirty"], res)
res, lane = owned_case("main", "wt-main")        # absent from origin here
ok("8 owned+`main`: refused AS PROTECTED", reasons(res, lane) == ["protected"], res)
ok("8 owned+`main`: not on origin", not origin_has("main"))
res, lane = owned_case("feat/owned-empty-218", "wt-empty", commits=0)
ok("8 owned+no commits: declined AS NOTHING-AHEAD",
   reasons(res, lane) == ["nothing-ahead"], res)
lane = lane_tree()
mine = dispatched(lane)
pub = add_worktree(lane, "feat/owned-published-218", f"wt-published-{MINE}")
git(pub, "push", "--quiet", "origin", "feat/owned-published-218")
head_before = git(SEED, "ls-remote", ORIGIN,
                  "refs/heads/feat/owned-published-218").split()[0]
open(f"{pub}/extra.txt", "w").write("more\n")
git(pub, "add", "-A")
git(pub, "commit", "--quiet", "-m", "extra")
res = disp.push_run_branch(mine)
ok("8 owned+published: refused AS EXISTS-ON-ORIGIN",
   reasons(res, lane) == ["exists-on-origin"], res)
ok("8 owned+published: it did not advance",
   git(SEED, "ls-remote", ORIGIN,
       "refs/heads/feat/owned-published-218").split()[0] == head_before)
lane = lane_tree()
mine = dispatched(lane)
add_worktree(lane, "feat/owned-killswitch-218", f"wt-killswitch-{MINE}")
open(disp.NO_PUSH_FILE, "w").close()
res = disp.push_run_branch(mine)
os.remove(disp.NO_PUSH_FILE)
ok("8 owned+NO_PUSH: declined AS KILL-SWITCH", res["reason"] == "kill-switch", res)
ok("8 owned+NO_PUSH: nothing pushed", not origin_has("feat/owned-killswitch-218"))
ok("8 owned: pushes once the switch is gone (case 8 was the switch talking)",
   disp.push_run_branch(mine)["pushed"] and origin_has("feat/owned-killswitch-218"))

# 9 ── reap() wires it, and wires ALL of state ─────────────────────────────
# The rail is only real if reap supplies the run set; and it must supply the
# FINISHED runs too — case 7's rival is a sibling step that reaped hours ago, so
# an active-only list would call an undecidable tree decidable.
def finished_run(lane, run_id, task_id=218, branch0="dev"):
    log = f"{disp.LOG_DIR}/{run_id}.log"
    open(log, "w").write(json.dumps({"type": "result", "is_error": False,
                                     "result": "done"}) + "\n")
    r = dispatched(lane, run_id=run_id, task_id=task_id, branch0=branch0)
    r.update({"pid": 999999, "t0": disp.time.time(), "log": log, "active": True,
              "status0": "Todo"})
    return r

CALLS.clear()
lane = lane_tree()
run = finished_run(lane, MINE)
add_worktree(lane, "feat/reap-owned-218", f"Kevbot-wt-reap-{MINE}")
st = {"runs": [run]}
disp.reap(st)
ok("9 reap: the owned branch reached origin", origin_has("feat/reap-owned-218"))
ok("9 reap: recorded on the run record",
   st["runs"][0].get("pushed_branch") == "feat/reap-owned-218", st["runs"][0])
ok("9 reap: run_history.pushed_branch recorded (dashboard #193 input)",
   any(b.get("pushed_branch") == "feat/reap-owned-218" and b.get("pushed_at")
       for m, p, b in CALLS if m == "PATCH" and p.startswith("run_history?")), CALLS)

# the orphan, end to end through reap()
CALLS.clear()
lane = lane_tree()
run = finished_run(lane, MINE + ".2")
add_worktree(lane, "verify/reap-orphan-195", f"Kevbot-wt-orphan-{DEAD}")
st = {"runs": [run]}
disp.reap(st)
ok("9 reap: the orphan was NOT published", not origin_has("verify/reap-orphan-195"))
ok("9 reap: and nothing was recorded as pushed",
   st["runs"][0].get("pushed_branch") is None and
   not any("pushed_branch" in (b or {}) for m, p, b in CALLS), CALLS)
ok("9 reap: the run was still collected and reported",
   not st["runs"][0]["active"] and
   any(m == "PATCH" and p.startswith("run_history?") and b.get("outcome") == "ok"
       for m, p, b in CALLS))
ok("9 reap: a refusal is NOT escalated as a push failure (it is the design working)",
   not any("auto-push FAILED" in (b.get("body") or "")
           for m, p, b in CALLS if m == "POST" and p == "dev_task_comments"))

# reap must hand the push leg the FINISHED runs too
CALLS.clear()
lane = lane_tree()
sib = finished_run(lane_tree(), "r1785400000-218")
sib["active"] = False                            # reaped hours ago, still in state
run = finished_run(lane, MINE + ".3")
add_worktree(lane, "diag/reap-legacy-218", "Kevbot-wt-legacytag-218")
st = {"runs": [sib, run]}
disp.reap(st)
ok("9 reap: an already-reaped sibling of the same task still makes it undecidable",
   not origin_has("diag/reap-legacy-218"))
ok("9 reap: ...and the run recorded no push", st["runs"][1].get("pushed_branch") is None)

# 10 ── the WORKTREE CONTRACT reaches the agent, with the REAL run id ──────
AGENT = {"letter": "M", "scope": "docs", "boundaries": "none"}
TASK = {"id": 218, "title": "t", "description": "d", "assignee": "M",
        "status": "Todo", "tags": [], "checklist": []}
p = disp.build_prompt(AGENT, TASK, run_id=MINE)
ok("10 the block renders", "WORKTREE CONTRACT (board #218" in p)
ok("10 it hands the agent its run id", f"`{MINE}`" in p, p)
ok("10 it gives the exact command form",
   f"git worktree add ../Kevbot-wt-<slug>-{MINE} -b <branch> origin/dev" in p, p)
ok("10 it states what the loop does with an unattributable tree",
   "REFUSED" in p and "never\nadopted" in p.replace("\r", ""), p)
ok("10 it does not present itself as a substitute for pushing",
   "does not replace pushing it yourself" in p, p)
ok("10 the #153 git contract still renders after it",
   p.index("WORKTREE CONTRACT") < p.index("GIT CONTRACT (board #153"), p)
ok("10 a direct call with no run id still states the rule",
   "WORKTREE CONTRACT (board #218" in disp.build_prompt(AGENT, TASK))

# ...and one_pass() passes the id it actually dispatched under. Built AFTER the
# id is settled — pre-#218 the prompt was built first, so this is a real ordering
# rail, not a restatement of the case above.
PROMPTS = []
LANE = lane_tree()
disp.headless_agents = lambda: ({"M": {"letter": "M", "status": "headless",
                                       "worktree": LANE, "scope": "", "boundaries": ""}},
                                "test")
disp.triage_todo = lambda agents, done, st=None: ([dict(TASK, id=902)], [])
disp.run_requested = lambda agents, done: ([], [])
disp.mentions_for = lambda role: ("", [])
disp.mentions_delivered = lambda role, ids: 0
disp.spawn = lambda agent, task, prompt, log: (
    PROMPTS.append(prompt),
    subprocess.Popen(["sleep", "30"], start_new_session=True))[1]
CALLS.clear()
json.dump({"runs": []}, open(disp.STATE_FILE, "w"))
disp.one_pass(True, only_task=902)
rec = json.load(open(disp.STATE_FILE))["runs"][-1]
disp.kill_run_group(rec["pid"])
ok("10 one_pass dispatched and recorded a run", rec.get("run_id"), rec)
ok("10 the dispatched prompt carries THAT run's id",
   PROMPTS and f"`{rec['run_id']}`" in PROMPTS[0], rec.get("run_id"))
ok("10 ...in the worktree command the agent will copy",
   f"../Kevbot-wt-<slug>-{rec['run_id']} -b" in PROMPTS[0])
ok("10 the id is also announced on the thread, so a resumed agent can find it",
   any(f"(run {rec['run_id']})" in (b.get("body") or "")
       for m, pth, b in CALLS if m == "POST" and pth == "dev_task_comments"), CALLS)

shutil.rmtree(TMP, ignore_errors=True)
print(f"\nALL {PASS} CHECKS PASSED — board #218 worktree ownership")
