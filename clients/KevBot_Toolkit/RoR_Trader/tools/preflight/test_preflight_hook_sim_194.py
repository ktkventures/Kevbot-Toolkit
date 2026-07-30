#!/usr/bin/env python3
"""Acceptance tests for preflight invariants (8) and (9) — board #194.

Standalone (no pytest, no network, no board writes):
    python3 test_preflight_hook_sim_194.py

Every case here FAILS without the invariant it covers — a rail with no failing
test is a comment, not a rail. The two things the task demands proof of:

  * each invariant can actually go RED on deliberately drifted / missing state,
    and GREEN again once reconciled (an invariant that cannot go red is decoration)
  * both are ADVISORY — a crash inside either one degrades to a `--` note and
    main() still completes; the script still exits 0 and can never wedge a session.

  (8) live SessionStart hook == tracked canonical
      1  identical            -> OK
      2  drifted              -> !! naming the exact `cp` that reconciles them
      3  drift then real cp   -> RED -> GREEN (the reconcile actually works)
      4  live copy missing    -> !! naming the same `cp`
      5  canonical unreadable -> `--` note, never a false RED
  (9) SIM poller alive
      6  real process whose cmdline contains replay_sim_poller -> OK
         (proves the pgrep pattern matches the REAL thing, not a mock)
      7  down                 -> !! whose fix uses .venv/bin/python, NOT bare python3
      8  down + PAUSE present -> `--` note (documented kill switch, not an alarm)
  advisory posture
      9  both checks raising  -> main() completes, prints `--` crash notes
     10  the __main__ block still ends in sys.exit(0)
"""
import ast
import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PREFLIGHT_PY = f"{HERE}/preflight.py"
spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT_PY)
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

TMP = tempfile.mkdtemp(prefix="preflight-194-")
FAILURES = []
_PROCS = []


# --- harness ---------------------------------------------------------------
def run(fn):
    """Call a check, return its printed output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if not cond else ""))
    if not cond:
        FAILURES.append(name)


def stub_repo(hook_body=None, live_body=None, pause=False):
    """A throwaway REPO tree; returns (canon_path, live_path)."""
    root = tempfile.mkdtemp(dir=TMP)
    pf.REPO = root
    canon = f"{root}/{pf.HOOK_REL}"
    os.makedirs(os.path.dirname(canon), exist_ok=True)
    if hook_body is not None:
        open(canon, "w", encoding="utf-8").write(hook_body)
    pf.LIVE_HOOK = f"{root}/.claude/hooks/team_board_context.py"
    os.makedirs(os.path.dirname(pf.LIVE_HOOK), exist_ok=True)
    if live_body is not None:
        open(pf.LIVE_HOOK, "w", encoding="utf-8").write(live_body)
    if pause:
        os.makedirs(os.path.dirname(f"{root}/{pf.SIM_PAUSE_REL}"), exist_ok=True)
        open(f"{root}/{pf.SIM_PAUSE_REL}", "w", encoding="utf-8").write("")
    return canon, pf.LIVE_HOOK


def poller_down(*a, **kw):
    return subprocess.CompletedProcess(a, 1, stdout="", stderr="")


CANON = '#!/usr/bin/env python3\n"""hook."""\nimport sys\nprint_mentions()\n'


# --- (8) hook copy ----------------------------------------------------------
def t1_hook_identical():
    stub_repo(hook_body=CANON, live_body=CANON)
    out = run(pf.check_8_hook_copy)
    check("(8) identical copies read GREEN", "  OK  (8)" in out and "!!" not in out, out)


def t2_hook_drifted():
    canon, live = stub_repo(hook_body=CANON, live_body=CANON + "print('local hack')\n")
    out = run(pf.check_8_hook_copy)
    check("(8) drifted live copy goes RED", "  !!  (8)" in out, out)
    check("(8) RED names the exact cp that reconciles", f"cp {canon} {live}" in out, out)
    check("(8) RED shows the delta shape", "+1/-0 lines" in out, out)


def t3_hook_red_then_green():
    canon, live = stub_repo(hook_body=CANON, live_body=CANON + "drift\n")
    red = run(pf.check_8_hook_copy)
    shutil.copyfile(canon, live)          # the very `cp` the RED line prints
    green = run(pf.check_8_hook_copy)
    check("(8) RED -> cp -> GREEN (reconcile actually works)",
          "  !!  (8)" in red and "  OK  (8)" in green and "!!" not in green,
          f"red={red!r} green={green!r}")


def t4_hook_live_missing():
    canon, live = stub_repo(hook_body=CANON, live_body=None)
    out = run(pf.check_8_hook_copy)
    check("(8) missing live hook goes RED", "  !!  (8)" in out and "MISSING" in out, out)
    check("(8) missing-live RED names the cp", f"cp {canon} {live}" in out, out)


def t5_hook_canon_unreadable():
    stub_repo(hook_body=None, live_body=CANON)
    out = run(pf.check_8_hook_copy)
    check("(8) unreadable canonical degrades to a note, no false RED",
          "  --  (8)" in out and "!!" not in out, out)


# --- (9) SIM poller ---------------------------------------------------------
def t6_sim_alive_real_process():
    """A REAL process whose cmdline carries the name — proves the pattern matches."""
    stub_repo(hook_body=CANON, live_body=CANON)
    fake = f"{TMP}/replay_sim_poller_TESTFAKE"
    shutil.copyfile("/bin/sleep", fake)
    os.chmod(fake, 0o755)
    p = subprocess.Popen([fake, "45"])
    _PROCS.append(p)
    time.sleep(0.4)                       # let it appear in the process table
    out = run(pf.check_9_sim_poller)
    check("(9) live poller process reads GREEN via real pgrep",
          "  OK  (9)" in out and "!!" not in out, out)
    p.kill()
    p.wait()
    time.sleep(0.4)
    out2 = run(pf.check_9_sim_poller)     # GREEN -> RED once it dies
    check("(9) GREEN -> RED the moment the real process dies", "  !!  (9)" in out2, out2)


def t7_sim_down():
    stub_repo(hook_body=CANON, live_body=CANON)
    real_sh, pf.sh = pf.sh, poller_down
    try:
        out = run(pf.check_9_sim_poller)
    finally:
        pf.sh = real_sh
    check("(9) dead poller goes RED", "  !!  (9)" in out, out)
    check("(9) RED fix uses .venv/bin/python", pf.VENV_PY_REL in out, out)
    check("(9) RED fix never hands over bare `python3 tools/`",
          "python3 tools/" not in out, out)
    check("(9) RED explains the silent failure (button no-ops)",
          "no-op" in out or "silently" in out, out)


def t8_sim_paused():
    stub_repo(hook_body=CANON, live_body=CANON, pause=True)
    real_sh, pf.sh = pf.sh, poller_down
    try:
        out = run(pf.check_9_sim_poller)
    finally:
        pf.sh = real_sh
    check("(9) documented kill switch = note, not an alarm",
          "  --  (9)" in out and "!!" not in out, out)


# --- advisory posture -------------------------------------------------------
def t9_advisory_never_blocks():
    """Both new checks blow up -> main() still completes, one `--` note each."""
    def boom():
        raise RuntimeError("deliberate")

    saved = {n: getattr(pf, n) for n in
             ("check_1_main_on_dev", "_worktree_stats", "check_2_unpushed",
              "check_3_stale_base", "check_4_docs", "check_5_dispatcher",
              "check_6_flags", "check_7_todo_kevin", "check_8_hook_copy",
              "check_9_sim_poller", "git", "sh")}
    pf.check_1_main_on_dev = lambda *a, **k: None
    pf._worktree_stats = lambda *a, **k: []
    for n in ("check_2_unpushed", "check_3_stale_base", "check_4_docs",
              "check_5_dispatcher", "check_6_flags", "check_7_todo_kevin"):
        setattr(pf, n, lambda *a, **k: None)
    pf.check_8_hook_copy = boom
    pf.check_9_sim_poller = boom
    pf.git = lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr="")
    pf.sh = lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr="")
    try:
        out = run(pf.main)
        crashed = False
    except Exception:
        out, crashed = "", True
    finally:
        for n, v in saved.items():
            setattr(pf, n, v)
    check("advisory: a crash in (8)/(9) does not propagate out of main()", not crashed)
    check("advisory: (8) crash degrades to a note", "  --  (8) check crashed" in out, out)
    check("advisory: (9) crash degrades to a note", "  --  (9) check crashed" in out, out)


def t10_exit_zero_contract():
    tree = ast.parse(open(PREFLIGHT_PY, encoding="utf-8").read())
    exits = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "exit" and getattr(n.func.value, "id", "") == "sys"
             and n.args and getattr(n.args[0], "value", None) == 0]
    check("advisory: script still ends in sys.exit(0) — never wedges SessionStart",
          len(exits) >= 1)
    main_fn = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    inside = {c.id for c in ast.walk(main_fn) if isinstance(c, ast.Name)}
    check("(8)+(9) are wired into main()'s guarded dispatch",
          {"check_8_hook_copy", "check_9_sim_poller"} <= inside,
          f"names referenced in main(): {sorted(inside)}")


def main():
    print(f"Preflight #194 invariants (8) hook-copy + (9) SIM poller — {PREFLIGHT_PY}")
    for t in (t1_hook_identical, t2_hook_drifted, t3_hook_red_then_green,
              t4_hook_live_missing, t5_hook_canon_unreadable,
              t6_sim_alive_real_process, t7_sim_down, t8_sim_paused,
              t9_advisory_never_blocks, t10_exit_zero_contract):
        try:
            t()
        except Exception as e:
            check(t.__name__, False, f"threw {type(e).__name__}: {e}")
    for p in _PROCS:
        try:
            p.kill()
        except Exception:
            pass
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
