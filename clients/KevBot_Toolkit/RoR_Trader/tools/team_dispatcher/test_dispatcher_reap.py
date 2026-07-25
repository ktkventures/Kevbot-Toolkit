#!/usr/bin/env python3
"""Acceptance tests for the board #131 dispatcher shakedown fixes.

Standalone (no pytest, no network): `python3 test_dispatcher_reap.py`.
Spawns real throwaway `sleep` process groups to prove kill behavior; the
Supabase api() is replaced with a recorder, so ZERO board writes happen.

Covers, one case per #131 fix:
  1. lease path REACHABLE  — a run past RUN_TIMEOUT_S is collected by reap()
                             (old active_runs() filter excluded it forever)
  2. actual kill on expiry — the run's whole session group is dead afterward
  3. lingering-completed   — terminal result JSON in the log + process still
                             alive (child kept it open, run r1785000766-122)
                             → reported as a normal completion, leftovers killed;
                             stray child output AFTER the JSON still parses
  4. loop observability    — every print() in dispatcher.py carries flush=True
Plus regressions kept from the #109 review fixes: dead proc with is_error
terminal JSON → outcome=error; dead proc with garbage log → outcome=error;
healthy in-lease run is left alone.
"""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dispatcher", f"{HERE}/dispatcher.py")
disp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disp)

TMP = tempfile.mkdtemp(prefix="dispatch-test-")
disp.STATE_FILE = f"{TMP}/state.json"

CALLS = []


def fake_api(method, path, body=None, prefer=None):
    CALLS.append((method, path, body))
    if method == "GET" and path.startswith("dev_tasks?id=eq.") and "select=tags" in path:
        return [{"tags": ["old-tag"]}]
    return None


disp.api = fake_api


def spawn_group():
    """A session-leader bash with a background child — same process topology
    spawn() creates (start_new_session) and the lingering-child failure mode."""
    p = subprocess.Popen(["bash", "-c", "sleep 300 & exec sleep 300"],
                         start_new_session=True)
    time.sleep(0.1)
    return p.pid


def group_dead(pid, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def write_log(name, text):
    path = f"{TMP}/{name}.log"
    open(path, "w").write(text)
    return path


def run_reap(run):
    del CALLS[:]
    st = {"runs": [run]}
    disp.reap(st)
    return run


def calls(method, frag):
    return [c for c in CALLS if c[0] == method and frag in c[1]]


def mkrun(run_id, pid, t0, log):
    return {"run_id": run_id, "task_id": 999, "agent": "F", "pid": pid,
            "t0": t0, "log": log, "ts": "2026-07-25T00:00:00+00:00", "active": True}


fails = 0


def check(name, cond):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'} — {name}")
    if not cond:
        fails += 1


# ---- Case 1+2: lease expired, group still alive → collected AND killed -----
print("[1+2] lease-expired run: reachable branch + real kill")
pid = spawn_group()
expired_t0 = time.time() - disp.RUN_TIMEOUT_S - 60
r = run_reap(mkrun("r-lease", pid, expired_t0,
                   write_log("lease", "partial output, no terminal JSON\n")))
check("run collected (active flipped False) — unreachable before #131", r["active"] is False)
lease_comments = [c for c in calls("POST", "dev_task_comments")
                  if "LEASE EXPIRED" in c[2]["body"]]
check("system LEASE EXPIRED comment posted", len(lease_comments) == 1)
check("task PATCHed to Blocked",
      any(c[2].get("status") == "Blocked" for c in calls("PATCH", "dev_tasks")))
check("run_history outcome=lease-expired",
      any(c[2].get("outcome") == "lease-expired" for c in calls("PATCH", "run_history")))
check("whole session group actually dead (child included)", group_dead(pid))

# ---- Case 3: completed-but-lingering → normal report + leftover kill -------
print("[3] lingering-completed run (terminal JSON, proc alive, within lease)")
pid = spawn_group()
terminal = json.dumps({"type": "result", "result": "did the task. report here.",
                       "is_error": False})
# stray child output AFTER the terminal JSON — last-line parse would miss it
log = write_log("linger", f"noise line\n{terminal}\nstray child output\n")
r = run_reap(mkrun("r-linger", pid, time.time() - 60, log))
check("run collected despite live process", r["active"] is False)
agent_comments = [c for c in calls("POST", "dev_task_comments")
                  if c[2]["author"] == "F·auto"]
check("result posted as F·auto comment",
      len(agent_comments) == 1 and agent_comments[0][2]["body"] == "did the task. report here.")
check("task PATCHed to Review (board #136 — status replaced the needs-review tag)",
      any(c[2].get("status") == "Review" for c in calls("PATCH", "dev_tasks")))
check("system status-transition comment posted",
      any(c[2]["author"] == "system" and "In Progress → Review" in c[2]["body"]
          for c in calls("POST", "dev_task_comments")))
check("run_history outcome=ok",
      any(c[2].get("outcome") == "ok" for c in calls("PATCH", "run_history")))
check("NOT treated as lease-expired",
      not any("LEASE EXPIRED" in c[2]["body"] for c in calls("POST", "dev_task_comments")))
check("leftover session group killed", group_dead(pid))

# ---- Case 3-guard: healthy in-lease run is left alone ----------------------
print("[3-guard] mid-stream run (no terminal JSON, alive, within lease)")
pid = spawn_group()
r = run_reap(mkrun("r-live", pid, time.time() - 60, write_log("live", "still working...\n")))
check("run left active", r["active"] is True)
check("no board writes", len(CALLS) == 0)
check("process group untouched", not group_dead(pid, timeout=0.3))
os.killpg(pid, 9)  # test cleanup

# ---- Regressions kept from the #109 review fixes ---------------------------
print("[reg] dead proc + is_error terminal JSON → outcome=error")
log = write_log("err", json.dumps({"type": "result", "result": "boom", "is_error": True}) + "\n")
r = run_reap(mkrun("r-err", 999999999, time.time() - 60, log))
check("collected, outcome=error",
      r["active"] is False and any(c[2].get("outcome") == "error"
                                   for c in calls("PATCH", "run_history")))

print("[reg] dead proc + garbage log (died mid-stream) → outcome=error, tail posted")
r = run_reap(mkrun("r-garbage", 999999999, time.time() - 60,
                   write_log("garbage", "not json at all\n")))
check("collected, outcome=error",
      r["active"] is False and any(c[2].get("outcome") == "error"
                                   for c in calls("PATCH", "run_history")))
check("log tail posted as the comment",
      any("not json at all" in c[2]["body"] for c in calls("POST", "dev_task_comments")))

# ---- Case 4: every print() flushes (loop log observability) ----------------
print("[4] all print() calls carry flush=True")
tree = ast.parse(open(f"{HERE}/dispatcher.py").read())
prints = [n for n in ast.walk(tree)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
unflushed = [n.lineno for n in prints
             if not any(k.arg == "flush" and getattr(k.value, "value", None) is True
                        for k in n.keywords)]
check(f"{len(prints)} print calls, unflushed lines={unflushed or 'none'}", not unflushed)

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
sys.exit(1 if fails else 0)
