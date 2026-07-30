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
  (9) SIM poller alive — HERMETIC (board #194 step 3). The check's process lister is
      injectable, so these pass whether or not the real poller is running on this box.
      The first cut of this test spawned a decoy, killed it, and asserted RED — which
      only held while the REAL poller was down, i.e. only on a broken machine.
      6  the PRODUCTION path (real pgrep + bracket pattern + invocation filter)
         matches a REAL `python .../replay_sim_poller.py` process     -> OK
      6b the production lister returns nothing for an absent pattern (the RED input),
         proving the RED branch is reachable without killing the real poller
      6c injected lister: RED (empty table) and GREEN (a poller line) back-to-back in
         ONE run, asserted to hold under BOTH real-poller states
      6d a dispatched agent's own argv MENTIONING the poller is not a false GREEN
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


def lister(*lines):
    """An injected process lister — the test never consults the real process table."""
    return lambda: "".join(f"{ln}\n" for ln in lines)


POLLER_LINE = "4242 .venv/bin/python tools/team_sim/replay_sim_poller.py --live --loop --poll 30"
# A dispatched agent's cmdline carries its whole prompt, and #194's prompt quotes the
# poller command verbatim — this is the real false-GREEN that shipped in step 2.
AGENT_LINE = ("209506 claude -p You are M·auto ... nohup .venv/bin/python "
              "tools/team_sim/replay_sim_poller.py --live --loop --poll 30 ... END")


def real_poller_running():
    """Ground truth for THIS box, so the test can assert it does not depend on it."""
    return any(pf._is_poller_invocation(ln) for ln in pf.pgrep_lines().splitlines())


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
def t6_production_path_matches_a_real_process():
    """The PRODUCTION path end-to-end: real pgrep + bracket pattern + invocation filter.

    Spawns a genuine `python .../replay_sim_poller.py` and asserts the default
    (uninjected) check sees it. This is additive to the real process table, so it
    holds whether or not the real poller is up.
    """
    stub_repo(hook_body=CANON, live_body=CANON)
    d = tempfile.mkdtemp(dir=TMP)
    fake = f"{d}/{pf.SIM_POLLER_SCRIPT}"
    open(fake, "w", encoding="utf-8").write("import time\ntime.sleep(45)\n")
    p = subprocess.Popen([sys.executable, fake])
    _PROCS.append(p)
    time.sleep(0.5)                       # let it appear in the process table
    out = run(pf.check_9_sim_poller)      # no injection: the real pgrep runs
    check("(9) production path (real pgrep) sees a real poller invocation -> GREEN",
          "  OK  (9)" in out and "!!" not in out, out)
    matched = [ln for ln in pf.pgrep_lines().splitlines() if fake in ln]
    check("(9) the bracket pattern itself matched that real process",
          len(matched) == 1 and pf._is_poller_invocation(matched[0]),
          f"pgrep lines mentioning the spawned script: {matched}")
    p.kill()
    p.wait()


def t6b_production_lister_can_return_empty():
    """The RED input is reachable through the REAL lister — no need to kill the poller."""
    out = pf.pgrep_lines("[z]zz_absent_pattern_194")
    check("(9) real lister returns nothing for an absent pattern (RED branch reachable)",
          out.strip() == "", repr(out))


def t6c_red_and_green_back_to_back():
    """RED then GREEN in ONE run, asserted independent of this box's poller state."""
    was_running = real_poller_running()
    stub_repo(hook_body=CANON, live_body=CANON)
    red = run(lambda: pf.check_9_sim_poller(lister()))            # empty process table
    green = run(lambda: pf.check_9_sim_poller(lister(POLLER_LINE)))
    still_running = real_poller_running()
    check("(9) injected empty table -> RED", "  !!  (9)" in red, red)
    check("(9) injected poller line -> GREEN", "  OK  (9)" in green and "!!" not in green, green)
    check("(9) RED and GREEN both held in ONE run, real poller untouched",
          "  !!  (9)" in red and "  OK  (9)" in green and was_running == still_running,
          f"real poller running: before={was_running} after={still_running}")
    print(f"        (real SIM poller was {'UP' if was_running else 'DOWN'} during this run "
          f"— the two cases above must pass either way)")


def t6d_agent_prompt_is_not_a_false_green():
    """A process that merely MENTIONS the poller (an agent's own prompt) is not alive."""
    stub_repo(hook_body=CANON, live_body=CANON)
    out = run(lambda: pf.check_9_sim_poller(lister(AGENT_LINE)))
    check("(9) an agent prompt quoting the poller command is NOT a false GREEN",
          "  !!  (9)" in out, out)
    both = run(lambda: pf.check_9_sim_poller(lister(AGENT_LINE, POLLER_LINE)))
    check("(9) a real invocation alongside that noise still reads GREEN",
          "  OK  (9)" in both and "!!" not in both, both)


def t7_sim_down():
    stub_repo(hook_body=CANON, live_body=CANON)
    out = run(lambda: pf.check_9_sim_poller(lister()))
    check("(9) dead poller goes RED", "  !!  (9)" in out, out)
    check("(9) RED fix uses .venv/bin/python", pf.VENV_PY_REL in out, out)
    check("(9) RED fix never hands over bare `python3 tools/`",
          "python3 tools/" not in out, out)
    check("(9) RED explains the silent failure (button no-ops)",
          "no-op" in out or "silently" in out, out)


def t8_sim_paused():
    stub_repo(hook_body=CANON, live_body=CANON, pause=True)
    out = run(lambda: pf.check_9_sim_poller(lister()))
    check("(9) documented kill switch = note, not an alarm",
          "  --  (9)" in out and "!!" not in out, out)


def t8b_lister_crash_is_advisory():
    """A lister that blows up degrades to a `--` note, never a false GREEN or a raise."""
    stub_repo(hook_body=CANON, live_body=CANON)
    def boom():
        raise OSError("pgrep unavailable")
    out = run(lambda: pf.check_9_sim_poller(boom))
    check("(9) lister crash -> `--` note, no false GREEN",
          "  --  (9)" in out and "  OK  (9)" not in out, out)


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
              t6_production_path_matches_a_real_process,
              t6b_production_lister_can_return_empty,
              t6c_red_and_green_back_to_back,
              t6d_agent_prompt_is_not_a_false_green,
              t7_sim_down, t8_sim_paused, t8b_lister_crash_is_advisory,
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
