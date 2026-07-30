#!/usr/bin/env python3
"""Acceptance test — board #197: a FAILED run stops lying about its status.

Before V4.20 `reap()` moved EVERY finished run's task to `Review` with the system
comment "output awaiting sign-off" — including runs that died mid-stream (API 500,
crash, kill), where no output exists. Measured twice on 07-29: r1785350077-191 and
r1785354198-195. Three costs: the status lied; `run_history.outcome='error'` was
correct but invisible; and `Review` is not dispatch-eligible, so the task silently
stalled until a human noticed (the #171 41-hour shape).

What must hold now:
  • outcome='error'  → task returns to its PRE-DISPATCH status (recorded on the run
                       at claim time), retry counter bumped, comment says FAILED
  • after MAX_AUTO_RETRIES → `Blocked`, blocker NAMED (matching the lease leg)
  • "output awaiting sign-off" is NEVER emitted for a failed run
  • outcome='ok'     → unchanged: Review + the agent's result + sign-off comment
  • §8 AUDIBLE       → if a HUMAN moved the task off `In Progress` during the run,
                       neither leg fires: no status write, no retry spent, and the
                       thread says so. A restored status that is wrong is worse
                       than `Review` — reverting a Todo → Approval move would
                       re-dispatch, on the next poll, work Kevin pulled back.

§7 IS THE POINT. #195's step 2 shipped 52 green checks over a feature that never
fired, because every case handed the function a hand-built record instead of the
shape a real dispatch produces. §7 therefore drives the REAL topology: one_pass()
dispatches, spawn() launches an actual process that fails the way `claude -p` fails,
state.json round-trips through disk, and reap() collects it — no synthetic record
anywhere. §1-§6 are the unit ladder underneath it.

Hermetic: zero network (api() is an in-memory board that APPLIES its PATCHes, so a
later pass sees what an earlier pass wrote), zero board writes, no real `claude`.

Run:  .venv/bin/python src/test_dispatcher_failed_run_status_197.py   (from RoR_Trader root)
Pre-fix proof:  DISPATCHER_PY=<path to a pre-V4.20 dispatcher.py> .venv/bin/python \
                  src/test_dispatcher_failed_run_status_197.py   → must FAIL
"""
import importlib.util
import json
import os
import stat
import sys
import tempfile
import time

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

PASS = 0
FAILS = []


def ok(label, cond, detail=""):
    """Collect (not exit) — a pre-fix run should show EVERY behaviour that is
    missing, not just the first one."""
    global PASS
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAILS.append(label)
        print(f"  FAIL: {label} {detail}")


# ── load the real dispatcher module ────────────────────────────────────────
DISPATCHER_PY = os.environ.get(
    "DISPATCHER_PY", os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
spec = importlib.util.spec_from_file_location("team_dispatcher", DISPATCHER_PY)
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)
print(f"dispatcher under test: {DISPATCHER_PY}")

TMP = tempfile.mkdtemp(prefix="test197-")
disp.STATE_FILE = f"{TMP}/state.json"
disp.LOG_DIR = f"{TMP}/logs"
disp.PAUSE_FILE = f"{TMP}/PAUSE"        # never the live loop's PAUSE file
disp.mentions = None                     # supported disabled state — no network

# Pre-V4.20 dispatchers have no cap; treat the ruling (2) as the expectation.
MAX_RETRIES = getattr(disp, "MAX_AUTO_RETRIES", 2)


# ── in-memory board ────────────────────────────────────────────────────────
class Board:
    """The two tables reap()/one_pass() touch. PATCHes are APPLIED so passes see
    each other's writes — that is what makes §7's self-heal assertion real."""

    def __init__(self, tasks=(), agent_letter="Z"):
        self.tasks = {t["id"]: dict(t) for t in tasks}
        self.comments, self.run_history, self.calls = [], [], []
        self.agent = {"letter": agent_letter, "status": "headless", "worktree": TMP,
                      "scope": "test lane", "boundaries": "none — hermetic test"}
        # §8: make the status re-read fail, to exercise the fail-open leg.
        self.status_get_raises = False

    # -- helpers the assertions read ---------------------------------------
    def statuses(self, task_id):
        """Every status this task was PATCHed to, in order."""
        return [b.get("status") for m, p, b in self.calls
                if m == "PATCH" and p.startswith(f"dev_tasks?id=eq.{task_id}")
                and isinstance(b, dict) and b.get("status")]

    def bodies(self, task_id=None):
        return [c["body"] for c in self.comments
                if task_id is None or c["task_id"] == task_id]

    def outcomes(self):
        return [r.get("outcome") for r in self.run_history]

    # -- the api() the dispatcher calls ------------------------------------
    def api(self, method, path, body=None, prefer=None):
        self.calls.append((method, path, body))
        if method == "GET":
            if path.startswith("agents?"):
                return [dict(self.agent)]
            # Board #232: the blocked_by lookup is the shared FINISHED set
            # (`status=in.(Done,Closed)`), not the literal `status=eq.Done`.
            if path.startswith(f"dev_tasks?{disp.FINISHED_FILTER}"):
                return []
            if path.startswith(f"dev_tasks?{disp.GATE_FILTER}"):
                # V4.21 (#198) replaced the `status=eq.Todo` queue with
                # `ai_eligible OR status=Todo`. Emulated from the SAME dict the
                # PATCHes land in, and keyed off the shipped GATE_FILTER so the
                # stub cannot drift from the gate: §7's self-heal has to re-enter
                # the queue through the real gate to prove anything.
                return [dict(t) for t in self.tasks.values()
                        if t.get("ai_eligible") or t.get("status") == "Todo"]
            if path.startswith("dev_tasks?status=eq.Todo"):    # pre-V4.21 gate
                return [dict(t) for t in self.tasks.values() if t["status"] == "Todo"]
            if path.startswith("dev_tasks?id=eq."):
                # The audible's re-read (status_now). Served from the SAME dict
                # the PATCHes land in, so "what the board says now" is real: a
                # test that moves a task mid-run is seen exactly as a human's
                # move would be. Returning [] here instead would silently stub
                # the guard out and green-light a rail that never runs.
                if self.status_get_raises:
                    raise RuntimeError("simulated board read failure")
                tid = int(path.split("id=eq.")[1].split("&")[0])
                t = self.tasks.get(tid)
                # V4.24 serves `ai_eligible` alongside the status: post-#198 the
                # queue is `ai_eligible OR Todo`, so "will this restore actually
                # re-dispatch?" needs both.
                return ([{"status": t["status"],
                          "ai_eligible": bool(t.get("ai_eligible"))}] if t else [])
            return []                       # run-requests, comments, run_history
        if method == "POST":
            if path == "dev_task_comments":
                self.comments.append(dict(body))
            elif path == "run_history":
                self.run_history.append(dict(body))
            return None
        if method == "PATCH":
            if path.startswith("dev_tasks?id=eq."):
                tid = int(path.split("id=eq.")[1].split("&")[0])
                self.tasks.setdefault(tid, {"id": tid}).update(body or {})
            elif path.startswith("run_history?run_id=eq."):
                rid = path.split("run_id=eq.")[1].split("&")[0]
                for row in self.run_history:
                    if row.get("run_id") == rid:
                        row.update(body or {})
            return None
        return None


def use(board):
    disp.api = board.api
    return board


def mktask(tid, assignee="Z", status="Todo", ai_eligible=False):
    return {"id": tid, "title": f"test task {tid}", "assignee": assignee,
            "status": status, "description": "a scoped description", "tags": [],
            "blocked_by": [], "checklist": [], "ai_eligible": ai_eligible}


def write_log(name, text):
    path = f"{TMP}/{name}.log"
    os.makedirs(TMP, exist_ok=True)
    open(path, "w").write(text)
    return path


def terminal(result, is_error):
    return json.dumps({"type": "result", "subtype":
                       "error_during_execution" if is_error else "success",
                       "result": result, "is_error": is_error})


DEAD_PID = 999999999   # nothing can be running here (> pid_max)


def mkrun(run_id, task_id, log, status0="Todo", pid=DEAD_PID, t0=None):
    r = {"run_id": run_id, "task_id": task_id, "agent": "Z", "pid": pid,
         "t0": time.time() - 60 if t0 is None else t0, "log": log,
         "ts": "2026-07-29T00:00:00+00:00", "active": True}
    if status0 is not None:
        r["status0"] = status0
    return r


def reap_one(board, run, st=None):
    """Reap ONE run, with the run_history row rh_claim() would have inserted at
    dispatch already present — reap only ever PATCHes it."""
    st = st if st is not None else {"runs": []}
    board.run_history.append({"task_id": run["task_id"], "run_id": run["run_id"],
                              "agent_letter": run["agent"], "outcome": "running"})
    st["runs"].append(run)
    disp.reap(st)
    return st


SIGNOFF = "output awaiting sign-off"


# ══ §1 — a failed run (is_error terminal JSON) ════════════════════════════
print("\n[§1] reap of an is_error run — the r1785350077-191 shape")
# status="In Progress" is what a real claim leaves on the board (§7 proves it),
# and since the audible the reaper READS that value back. A fixture sitting in
# Todo would be indistinguishable from a human having moved the task mid-run.
b = use(Board([mktask(500, status="In Progress")]))
st = reap_one(b, mkrun("r-err", 500, write_log(
    "err", terminal("API Error: 500 · Internal server error", True))))
ok("§1a task is NOT moved to Review", "Review" not in b.statuses(500),
   f"got {b.statuses(500)}")
ok("§1b task restored to its pre-dispatch status (Todo)",
   b.tasks[500]["status"] == "Todo", f"got {b.tasks[500]['status']}")
ok("§1c a comment says the run FAILED",
   any("FAILED" in x for x in b.bodies(500)))
ok(f"§1d no comment claims '{SIGNOFF}'",
   not any(SIGNOFF in x for x in b.bodies(500)))
ok("§1e the agent is NOT credited with a result comment",
   not any(c["author"] == "Z·auto" for c in b.comments))
ok("§1f the failure text is on the thread (diagnosable)",
   any("API Error: 500" in x for x in b.bodies(500)))
ok("§1g run_history outcome stays 'error' (the truth was always right)",
   "error" in b.outcomes())
ok("§1h retry counter bumped to 1",
   disp.retries_used(st, 500) == 1 if hasattr(disp, "retries_used") else False)

# ══ §2 — died mid-stream: no terminal JSON at all ═════════════════════════
print("\n[§2] reap of a run that died mid-stream — the r1785354198-195 shape")
b = use(Board([mktask(501, status="In Progress")]))
st = reap_one(b, mkrun("r-dead", 501, write_log("dead", "half a log line, then noth")))
ok("§2a not Review", "Review" not in b.statuses(501), f"got {b.statuses(501)}")
ok("§2b restored to Todo", b.tasks[501]["status"] == "Todo")
ok("§2c comment says FAILED", any("FAILED" in x for x in b.bodies(501)))
ok(f"§2d never '{SIGNOFF}'", not any(SIGNOFF in x for x in b.bodies(501)))
ok("§2e outcome=error", "error" in b.outcomes())

# ══ §3 — the retry ladder, then escalation ════════════════════════════════
print(f"\n[§3] retry ladder: {MAX_RETRIES} auto-retries, then Blocked")
b = use(Board([mktask(502)]))
st = {"runs": []}
for i in range(MAX_RETRIES + 1):
    b.tasks[502]["status"] = "In Progress"        # the claim, as a real dispatch does
    reap_one(b, mkrun(f"r-lad{i}", 502, write_log(
        f"lad{i}", terminal(f"boom {i}", True))), st)
seq = b.statuses(502)
ok(f"§3a the first {MAX_RETRIES} failures restore Todo",
   seq[:MAX_RETRIES] == ["Todo"] * MAX_RETRIES, f"got {seq}")
ok(f"§3b failure {MAX_RETRIES + 1} escalates to Blocked",
   seq[MAX_RETRIES] == "Blocked" if len(seq) > MAX_RETRIES else False, f"got {seq}")
ok("§3c the blocker is NAMED in notes",
   "dispatch failure" in (b.tasks[502].get("notes") or ""),
   f"got {b.tasks[502].get('notes')!r}")
ok("§3d the escalation comment names the blocker + the spent retries",
   any("Blocked" in x and "retries spent" in x for x in b.bodies(502)))
ok("§3e retry counter cleared on escalation (a human unblock gets a fresh budget)",
   disp.retries_used(st, 502) == 0 if hasattr(disp, "retries_used") else False)
ok("§3f Review never appeared anywhere in the ladder", "Review" not in seq)
ok(f"§3g every failed run recorded outcome=error ({MAX_RETRIES + 1})",
   b.outcomes().count("error") == MAX_RETRIES + 1, f"got {b.outcomes()}")

# ══ §4 — an ok run is untouched by all of this ════════════════════════════
print("\n[§4] a SUCCESSFUL run still goes to Review, sign-off comment intact")
b = use(Board([mktask(503, status="In Progress")]))
st = {"runs": [], "retries": {"503": 1}}          # a prior failure on the same task
reap_one(b, mkrun("r-ok", 503, write_log("okrun", terminal("did the step. report.", False))), st)
ok("§4a task moved to Review", b.tasks[503]["status"] == "Review")
ok("§4b the agent's result is posted as the agent",
   any(c["author"] == "Z·auto" and "did the step" in c["body"] for c in b.comments))
ok(f"§4c the '{SIGNOFF}' comment is still emitted for a GOOD run",
   any(SIGNOFF in x for x in b.bodies(503)))
ok("§4d run_history outcome=ok", "ok" in b.outcomes())
ok("§4e a good run clears the failure streak",
   disp.retries_used(st, 503) == 0 if hasattr(disp, "retries_used") else False)

# ══ §5 — the lease leg is unchanged (regression guard) ════════════════════
print("\n[§5] lease-expired still Blocked with its own comment")
b = use(Board([mktask(504, status="In Progress")]))
os.makedirs(disp.LOG_DIR, exist_ok=True)
import subprocess  # noqa: E402 — only §5/§7 need a real process
proc = subprocess.Popen(["sleep", "120"], start_new_session=True)
time.sleep(0.1)
reap_one(b, mkrun("r-lease", 504, write_log("lease", "no terminal json\n"),
                  pid=proc.pid, t0=time.time() - disp.RUN_TIMEOUT_S - 60))
ok("§5a lease-expired → Blocked", b.tasks[504]["status"] == "Blocked")
ok("§5b LEASE EXPIRED comment", any("LEASE EXPIRED" in x for x in b.bodies(504)))
ok("§5c run_history outcome=lease-expired", "lease-expired" in b.outcomes())
try:
    os.killpg(proc.pid, 9)
except ProcessLookupError:
    pass

# ══ §6 — a run claimed before V4.20 has no status0 ════════════════════════
print("\n[§6] pre-V4.20 run (no status0): Todo fallback, stated LOUDLY")
b = use(Board([mktask(505, status="In Progress")]))
reap_one(b, mkrun("r-old", 505, write_log("old", terminal("boom", True)), status0=None))
ok("§6a still not Review", "Review" not in b.statuses(505))
ok("§6b falls back to Todo", b.tasks[505]["status"] == "Todo")
ok("§6c the fallback is announced, not silent",
   any("no pre-dispatch status" in x for x in b.bodies(505)))

# ══ §6-bis — a button run claimed from Backlog restores Backlog ══════════
print("\n[§6-bis] non-Todo pre-dispatch status: restored, and NO retry is promised")
b = use(Board([mktask(506, status="In Progress")]))
reap_one(b, mkrun("r-btn", 506, write_log("btn", terminal("boom", True)),
                  status0="Backlog"))
ok("§6d restored to Backlog, not Todo and not Review",
   b.tasks[506]["status"] == "Backlog", f"got {b.statuses(506)}")
ok("§6e the comment does NOT promise an automatic retry that cannot happen "
   "(Backlog is not in the dispatch queue)",
   not any("automatic retry" in x for x in b.bodies(506))
   and any("NOT auto-retried" in x for x in b.bodies(506)), f"got {b.bodies(506)}")

# ══ §6-ter — V4.24: an ARMED non-Todo task IS re-dispatched, so say so ════
# The mirror image of §6-bis, and a merge consequence rather than a new idea:
# #197 wrote "NOT auto-retried from X" when the queue was `status=eq.Todo`.
# V4.21 (#198) widened it to `ai_eligible OR Todo`, so for an ARMED task that
# sentence became false — the dispatcher denying a retry it then performs is
# #197's own defect wearing a different hat.
print("\n[§6-ter] an ARMED non-Todo task: the retry IS promised, because it happens")
b = use(Board([mktask(510, status="In Progress", ai_eligible=True)]))  # the claim
reap_one(b, mkrun("r-armed", 510, write_log("armed", terminal("boom", True)),
                  status0="Approval"))
ok("§6f restored to Approval (pre-dispatch status, not Todo, not Review)",
   b.tasks[510]["status"] == "Approval", f"got {b.statuses(510)}")
ok("§6g the comment promises the retry AND names why it happens (ai_eligible)",
   any("automatic retry 1 of" in x and "ai_eligible" in x for x in b.bodies(510))
   and not any("NOT auto-retried" in x for x in b.bodies(510)), f"got {b.bodies(510)}")


# ══ §7 — REAL TOPOLOGY: dispatch → real process → real reap ═══════════════
print("\n[§7] end-to-end: one_pass() dispatches a run that really fails, then reaps it")


def fake_claude(name, result, is_error, exit_code):
    """A stand-in for `claude -p --output-format json`: same argv, same stdout
    contract (a terminal result object on the last line), same exit behaviour."""
    path = f"{TMP}/{name}"
    open(path, "w").write(
        "#!/bin/bash\n"
        "echo '{\"type\":\"system\",\"subtype\":\"init\"}'\n"
        f"echo '{terminal(result, is_error)}'\n"
        f"exit {exit_code}\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def wait_for_terminal(timeout=30):
    """Wait for the newest run's log to carry its terminal JSON — the exact
    condition reap() collects on."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = json.load(open(disp.STATE_FILE))
        runs = [r for r in st["runs"] if r.get("active")]
        if runs and os.path.exists(runs[-1]["log"]):
            if disp.parse_terminal(open(runs[-1]["log"]).read()) is not None:
                return runs[-1]
        time.sleep(0.05)
    raise AssertionError("§7 fixture: spawned run never produced a terminal result")


TID = 197
b = use(Board([mktask(TID)]))
if os.path.exists(disp.STATE_FILE):
    os.remove(disp.STATE_FILE)
disp.CLAUDE_BIN = fake_claude("claude_fail.sh", "API Error: 500 · run died", True, 1)

disp.one_pass(live=True)                                    # pass 1 — dispatch
first = wait_for_terminal()
ok("§7a a real run was dispatched and recorded", first["task_id"] == TID)
ok("§7b the dispatch recorded the PRE-DISPATCH status on the run",
   first.get("status0") == "Todo", f"got {first.get('status0')!r}")
ok("§7c the board shows the claim (In Progress)", b.tasks[TID]["status"] == "In Progress")

# V4.24 — every run this ladder produces is collected AS IT HAPPENS. The merge
# with #198 made that necessary: the retry leg calls forget_task_runs(), so a
# failed run is dropped from state on the pass that retries it and the final
# state.json is no longer the run log. Reading it after the fact would silently
# weaken §7g/§7l/§7m into assertions about ONE run.
all_runs = [first]
seen = {first["run_id"]}
for _ in range(MAX_RETRIES + 1):                            # passes 2..N — reap + re-arm
    disp.one_pass(live=True)
    for r in json.load(open(disp.STATE_FILE))["runs"]:
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            all_runs.append(r)
    if b.tasks[TID]["status"] == "In Progress":
        wait_for_terminal()

seq = b.statuses(TID)
expected = ["In Progress"] + ["Todo", "In Progress"] * MAX_RETRIES + ["Blocked"]
ok("§7d no failed run EVER reached Review through the real path", "Review" not in seq,
   f"got {seq}")
ok("§7e the whole real-topology status walk is exactly claim → restore → "
   "re-dispatch ×N → Blocked", seq == expected, f"got {seq} want {expected}")
ok("§7f the failure SELF-HEALED — it was re-dispatched with no human touch",
   seq.count("In Progress") == MAX_RETRIES + 1, f"got {seq}")
ok(f"§7g exactly {MAX_RETRIES + 1} runs, then it stopped re-arming",
   len(all_runs) == MAX_RETRIES + 1, f"got {[r['run_id'] for r in all_runs]}")
# ── the #198 coupling, proved through the real path ──────────────────────────
# step_already_ran() refuses a step a FINISHED run already covered, and a run
# that DIED is finished. Without report_failed_run()'s forget_task_runs() call
# the retry granted above is refused on the very next poll: the ladder stops
# dead at ["In Progress", "Todo"] and the task sits armed and silent — #197's
# stall relocated. §7e/§7f fail without it; this names WHY they do.
ok("§7g-bis the retry was NOT refused by the #198 step-guard (forget_task_runs)",
   not disp.step_already_ran(json.load(open(disp.STATE_FILE)), mktask(TID))
   or b.tasks[TID]["status"] == "Blocked")
ok("§7g-ter the escalated run is STILL in state (only retried runs are forgotten)",
   [r["run_id"] for r in json.load(open(disp.STATE_FILE))["runs"]]
   == [all_runs[-1]["run_id"]],
   f"got {json.load(open(disp.STATE_FILE))['runs']}")
ok(f"§7h '{SIGNOFF}' was never posted for this task",
   not any(SIGNOFF in x for x in b.bodies(TID)))
ok("§7i every real run reported FAILED on the thread",
   sum("FAILED" in x for x in b.bodies(TID)) == MAX_RETRIES + 1)
ok("§7j run_history recorded error for every real run",
   b.outcomes().count("error") == MAX_RETRIES + 1, f"got {b.outcomes()}")
ok("§7k a Blocked task is no longer dispatch-eligible (queue leaves it alone)",
   b.tasks[TID]["status"] == "Blocked")
# Self-heal made same-SECOND re-dispatch reachable, and run_id is second-resolution:
# without the uniqueness suffix all three runs share an id, so they overwrite each
# other's log and rh_finish() patches all three run_history rows at once.
runs = all_runs
ok("§7l each re-dispatch got its own run_id (no same-second collision)",
   len({r["run_id"] for r in runs}) == len(runs), f"got {[r['run_id'] for r in runs]}")
ok("§7m each run kept its own log file (a collision would overwrite the evidence)",
   len({r["log"] for r in runs}) == len(runs))

print("\n[§7-ok] end-to-end again, but the run SUCCEEDS → Review through the real path")
TID2 = 198
b = use(Board([mktask(TID2)]))
os.remove(disp.STATE_FILE)
disp.CLAUDE_BIN = fake_claude("claude_ok.sh", "step done; here is the report", False, 0)
disp.one_pass(live=True)
wait_for_terminal()
disp.one_pass(live=True)
ok("§7n a successful real run lands in Review", b.tasks[TID2]["status"] == "Review",
   f"got {b.statuses(TID2)}")
ok(f"§7o and DOES carry the '{SIGNOFF}' comment",
   any(SIGNOFF in x for x in b.bodies(TID2)))
ok("§7p the agent's real result is on the thread",
   any(c["author"] == "Z·auto" and "here is the report" in c["body"] for c in b.comments))


# ══ §7-armed — THE #198 COUPLING, through the real path ═══════════════════
# The legacy Todo arm has no step-guard, so §7 above cannot exercise the
# coupling #198's step_already_ran() docstring wrote down. An ARMED task can:
# it never falls out of the queue, so the guard is the ONLY thing between a
# granted retry and a re-dispatch. Without report_failed_run()'s
# forget_task_runs() call the ladder stops dead after ONE restore and the task
# sits armed, In-Progress-free and silent — #197's stall in a new place.
print("\n[§7-armed] an ARMED task's granted retry is not refused by the #198 step-guard")
TID4 = 200
b = use(Board([mktask(TID4, status="Approval", ai_eligible=True)]))
os.remove(disp.STATE_FILE)
disp.CLAUDE_BIN = fake_claude("claude_fail3.sh", "API Error: 500 · run died", True, 1)
disp.one_pass(live=True)
wait_for_terminal()
for _ in range(MAX_RETRIES + 1):
    disp.one_pass(live=True)
    if b.tasks[TID4]["status"] == "In Progress":
        wait_for_terminal()
seq4 = b.statuses(TID4)
ok("§7q the armed ladder really re-dispatched (step-guard did NOT eat the retry)",
   seq4.count("In Progress") == MAX_RETRIES + 1, f"got {seq4}")
ok("§7r ...restoring Approval each time, never Todo and never Review",
   seq4 == ["In Progress"] + ["Approval", "In Progress"] * MAX_RETRIES + ["Blocked"],
   f"got {seq4}")
ok("§7s and it still escalates once the cap is spent",
   b.tasks[TID4]["status"] == "Blocked")


# ══ §8 — AUDIBLE: a human's mid-run status change is never reverted ═══════
# Runs last minutes against a 45-minute lease, so the board can move underneath
# one. If Kevin pulls a task Todo → Approval while it runs and the run then dies,
# writing status0 back would silently undo his decision AND re-dispatch, on the
# very next poll, work he had deliberately pulled back. That is a worse lie than
# the `Review` this whole task exists to kill, so the reaper re-reads first.
print("\n[§8] a human moved the task DURING the run — the reaper keeps its hands off")
b = use(Board([mktask(507, status="Approval")]))    # moved off In Progress mid-run
st = reap_one(b, mkrun("r-moved", 507, write_log("moved", terminal("boom", True)),
                       status0="Todo"))
ok("§8a the dispatcher wrote NO status at all", b.statuses(507) == [],
   f"got {b.statuses(507)}")
ok("§8b the task is left exactly where the human put it",
   b.tasks[507]["status"] == "Approval", f"got {b.tasks[507]['status']}")
ok("§8c the comment SAYS the status was changed during the run and names it",
   any("CHANGED during the run" in x and "Approval" in x for x in b.bodies(507)),
   f"got {b.bodies(507)}")
ok("§8d the comment says the pre-dispatch status was NOT restored",
   any("Todo NOT restored" in x for x in b.bodies(507)))
ok("§8e the comment still says the run FAILED (the #197 rail holds)",
   any("FAILED" in x for x in b.bodies(507))
   and not any(SIGNOFF in x for x in b.bodies(507)))
ok("§8f the retry budget is INTACT — nothing was retried",
   disp.retries_used(st, 507) == 0, f"got {disp.retries_used(st, 507)}")
ok("§8g run_history still records the truth (outcome=error)", "error" in b.outcomes())

print("\n[§8-cap] cap already spent + a human move → still hands off, no Blocked")
b = use(Board([mktask(508, status="Approval")]))
st = {"runs": [], "retries": {"508": MAX_RETRIES}}
reap_one(b, mkrun("r-moved2", 508, write_log("moved2", terminal("boom", True)),
                  status0="Todo"), st)
ok("§8h a spent cap does NOT let the escalation overwrite the human either",
   b.statuses(508) == [] and b.tasks[508]["status"] == "Approval",
   f"got {b.statuses(508)}")
ok("§8i no Blocked comment was posted",
   not any("Blocked" in x for x in b.bodies(508)), f"got {b.bodies(508)}")
ok("§8j the spent counter is left as it was (not reset by a non-escalation)",
   disp.retries_used(st, 508) == MAX_RETRIES)

print("\n[§8-open] the guard FAILS OPEN — an unreadable board must not park a task")
b = use(Board([mktask(509, status="In Progress")]))
b.status_get_raises = True
st = reap_one(b, mkrun("r-blind", 509, write_log("blind", terminal("boom", True)),
                       status0="Todo"))
ok("§8k an unreadable status still restores (no silent stall in In Progress)",
   b.tasks[509]["status"] == "Todo", f"got {b.tasks[509]['status']}")
ok("§8l and the comment SAYS the current status could not be verified",
   any("could not be read back" in x for x in b.bodies(509)), f"got {b.bodies(509)}")
ok("§8m that path does spend a retry (it really did re-arm the task)",
   disp.retries_used(st, 509) == 1)

print("\n[§8-real] REAL TOPOLOGY: dispatch → run dies → human moved it → reap")
TID3 = 199
b = use(Board([mktask(TID3)]))
os.remove(disp.STATE_FILE)
disp.CLAUDE_BIN = fake_claude("claude_fail2.sh", "API Error: 500 · run died", True, 1)
disp.one_pass(live=True)                       # real claim → In Progress
wait_for_terminal()
b.tasks[TID3]["status"] = "Approval"           # the human, mid-run, on the board
disp.one_pass(live=True)                       # reap + the next dispatch decision
ok("§8n the real reap left the human's status alone",
   b.tasks[TID3]["status"] == "Approval", f"got {b.statuses(TID3)}")
ok("§8o only the claim was ever PATCHed — no revert",
   b.statuses(TID3) == ["In Progress"], f"got {b.statuses(TID3)}")
ok("§8p the pulled-back task was NOT re-dispatched",
   len(json.load(open(disp.STATE_FILE))["runs"]) == 1)
ok("§8q Review never appeared", "Review" not in b.statuses(TID3))
ok("§8r the thread explains why it stopped here",
   any("CHANGED during the run" in x for x in b.bodies(TID3)))

print(f"\n{PASS} checks passed"
      + (f", {len(FAILS)} FAILED: {FAILS}" if FAILS else " — ALL PASS"))
sys.exit(1 if FAILS else 0)
