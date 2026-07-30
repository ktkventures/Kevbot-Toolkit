"""Acceptance suite for the pytest false-green guard (board #239).

The hazard: our script-style acceptance suites define no `test_` functions, so pytest
collects zero items from them and reports success for a file it never executed. A suite
that exits 1 when run directly reports `passed` under pytest.

The guard is `conftest.py` at the RoR_Trader project root (plus `pytest.ini`, which
anchors the rootdir so the conftest is loaded even when pytest starts inside `src/`).

Every rail below FAILS without the guard. The rails that matter most are rail 2 (a
suite whose script FAILS must not yield a passing pytest result) and rail 4 (the guard
must not be defeated by a suite that exits at import).

Run:  .venv/bin/python src/test_pytest_false_green_guard_239.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFTEST = os.path.join(ROOT, "conftest.py")
PYTEST_INI = os.path.join(ROOT, "pytest.ini")

PASS = FAIL = 0
FAILURES = []


def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        FAILURES.append(f"{label}{(' — ' + detail) if detail else ''}")
        print(f"  FAIL: {label} — {detail}")


# ── fixtures ────────────────────────────────────────────────────────────────
# A script-style suite that GENUINELY FAILS when run directly (exits 1).
BROKEN_SCRIPT_SUITE = '''\
import sys
PASS = FAIL = 0
def ok(name, cond):
    global PASS, FAIL
    if cond: PASS += 1
    else:    FAIL += 1; print("  FAIL:", name)
def main():
    ok("this rail is broken on purpose", 1 == 2)
    return 1 if FAIL else 0
if __name__ == "__main__":
    sys.exit(main())
'''

# A script-style suite that calls sys.exit() AT IMPORT (the INTERNALERROR shape).
EXITING_SCRIPT_SUITE = '''\
import sys
print("running at import")
FAIL = 1
sys.exit(1 if FAIL else 0)
'''

# A genuine pytest suite -- must keep working untouched.
REAL_PYTEST_SUITE = '''\
def test_a_real_one():
    assert True
'''

# Genuine pytest suites in less common shapes -- must NOT be hijacked.
CLASS_PYTEST_SUITE = '''\
class TestThing:
    def test_inside_class(self):
        assert True
'''
ASYNC_PYTEST_SUITE = '''\
async def test_async_one():
    assert True
'''


def make_tree(with_guard, files):
    """Build a temp dir of fixture files, optionally with the real guard installed."""
    d = tempfile.mkdtemp(prefix="pytest_guard_239_")
    if with_guard:
        shutil.copy(CONFTEST, os.path.join(d, "conftest.py"))
        shutil.copy(PYTEST_INI, os.path.join(d, "pytest.ini"))
    for name, body in files.items():
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    return d


def run_pytest(d, *args, cwd=None):
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args],
        cwd=cwd or d, capture_output=True, text=True, timeout=180,
    )
    return p.returncode, p.stdout + p.stderr


print("― rail 1: the hazard is real — without the guard, a FAILING suite goes green ―")
d = make_tree(False, {
    "test_broken_suite.py": BROKEN_SCRIPT_SUITE,
    "test_normal.py": REAL_PYTEST_SUITE,
})
rc_direct = subprocess.run(
    [sys.executable, os.path.join(d, "test_broken_suite.py")],
    capture_output=True, text=True,
).returncode
ok("the fixture suite genuinely FAILS when run directly (exit 1)",
   rc_direct == 1, f"got exit {rc_direct}")
rc, out = run_pytest(d)
ok("WITHOUT the guard pytest exits 0 over that same failing suite",
   rc == 0, f"got exit {rc}")
ok("WITHOUT the guard pytest reports a pass and never mentions the broken suite",
   "1 passed" in out and "broken_suite" not in out, out[-300:])
shutil.rmtree(d, ignore_errors=True)


print("\n― rail 2: THE RAIL THAT MATTERS — with the guard, that run cannot be green ―")
d = make_tree(True, {
    "test_broken_suite.py": BROKEN_SCRIPT_SUITE,
    "test_normal.py": REAL_PYTEST_SUITE,
})
rc, out = run_pytest(d)
ok("WITH the guard pytest exits NON-ZERO", rc != 0, f"got exit {rc}")
ok("the failure is attributed to the script-style suite by name",
   "test_broken_suite.py" in out, out[-300:])
ok("it is reported as a FAILURE, not an error or a skip",
   "1 failed" in out, out[-300:])
shutil.rmtree(d, ignore_errors=True)


print("\n― rail 3: no collateral damage — genuine pytest suites still run ―")
d = make_tree(True, {"test_normal.py": REAL_PYTEST_SUITE})
rc, out = run_pytest(d)
ok("a real pytest suite alone still passes (exit 0)", rc == 0, f"exit {rc}: {out[-300:]}")
ok("its test actually executed", "1 passed" in out, out[-300:])
ok("the guard did not hijack it",
   "script_style_suite_not_executed" not in out, out[-300:])
shutil.rmtree(d, ignore_errors=True)

for label, body in (("Test* class", CLASS_PYTEST_SUITE), ("async def test_", ASYNC_PYTEST_SUITE)):
    d = make_tree(True, {"test_shape.py": body})
    rc, out = run_pytest(d)
    ok(f"a suite using a {label} is NOT hijacked by the guard",
       "script_style_suite_not_executed" not in out, out[-300:])
    shutil.rmtree(d, ignore_errors=True)


print("\n― rail 4: a suite that exits AT IMPORT must not defeat the guard ―")
d = make_tree(False, {"test_exiting.py": EXITING_SCRIPT_SUITE})
rc, out = run_pytest(d)
ok("WITHOUT the guard an import-time sys.exit blows the session up (INTERNALERROR)",
   "INTERNALERROR" in out, out[-300:])
shutil.rmtree(d, ignore_errors=True)

d = make_tree(True, {
    "test_exiting.py": EXITING_SCRIPT_SUITE,
    "test_normal.py": REAL_PYTEST_SUITE,
})
rc, out = run_pytest(d)
ok("WITH the guard there is no INTERNALERROR", "INTERNALERROR" not in out, out[-400:])
ok("the file was never imported (its import-time print never ran)",
   "running at import" not in out, out[-400:])
ok("it is reported as a clean failure instead", rc != 0 and "1 failed" in out, out[-400:])
ok("and it no longer takes the other suite down with it",
   "1 passed" in out, out[-400:])
shutil.rmtree(d, ignore_errors=True)


print("\n― rail 5: the refusal is ACTIONABLE, not just red ―")
d = make_tree(True, {"test_broken_suite.py": BROKEN_SCRIPT_SUITE})
rc, out = run_pytest(d)
ok("the message says pytest did not execute the file",
   "cannot execute this suite" in out, out[-400:])
ok("the message gives the direct-run command that DOES gate correctly",
   "python test_broken_suite.py" in out, out[-400:])
ok("the message points at the board task for the rationale",
   "#239" in out, out[-400:])
shutil.rmtree(d, ignore_errors=True)


print("\n― rail 6: the rootdir anchor — the guard survives being run from a subdir ―")
d = make_tree(True, {})
os.mkdir(os.path.join(d, "sub"))
with open(os.path.join(d, "sub", "test_broken_suite.py"), "w", encoding="utf-8") as fh:
    fh.write(BROKEN_SCRIPT_SUITE)
rc, out = run_pytest(d, "test_broken_suite.py", cwd=os.path.join(d, "sub"))
ok("invoked from INSIDE the subdir, pytest.ini still anchors rootdir so conftest loads",
   rc != 0 and "script_style_suite_not_executed" in out, f"exit {rc}: {out[-400:]}")
shutil.rmtree(d, ignore_errors=True)


print("\n― rail 7: the guard is actually installed in this repo ―")
ok("conftest.py exists at the RoR_Trader project root", os.path.exists(CONFTEST))
ok("pytest.ini exists at the RoR_Trader project root", os.path.exists(PYTEST_INI))
CONF_SRC = open(CONFTEST, encoding="utf-8").read()
ok("it hooks pytest_pycollect_makemodule (the firstresult hook that prevents the import)",
   "def pytest_pycollect_makemodule" in CONF_SRC,
   "pytest_collect_file alone is NOT firstresult — pytest would import the file anyway")
ok("detection is by AST, so the file is never imported to classify it",
   "ast.parse" in CONF_SRC)


print("\n― rail 8: classification matches the real suites in this repo ―")
sys.path.insert(0, ROOT)
import conftest as guard  # noqa: E402

ok("a top-level `ok(...)` script body defines no pytest targets",
   guard.defines_pytest_targets(BROKEN_SCRIPT_SUITE) is False)
ok("a `def test_` module defines pytest targets",
   guard.defines_pytest_targets(REAL_PYTEST_SUITE) is True)
ok("a `Test*` class defines pytest targets",
   guard.defines_pytest_targets(CLASS_PYTEST_SUITE) is True)
ok("a syntax-error file is left to pytest to report (loud on its own)",
   guard.defines_pytest_targets("def (((") is True)

# Spot-check real files, both shapes, so the classifier is pinned to reality.
for rel, expect in (
    ("src/test_board_statuses_232.py", False),      # script-style, exits at import
    ("src/test_dispatcher_loud_skips_171.py", False),  # script-style, silent-zero shape
    ("tools/release_brief/test_brief_gen.py", False),  # script-style under a main guard
    ("src/test_market_calendar.py", True),          # genuine pytest suite
    ("src/test_uad_empty_lane_guard.py", True),     # genuine pytest suite
):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        ok(f"{rel} present", False, "file missing — update this rail")
        continue
    got = guard.defines_pytest_targets(open(path, encoding="utf-8").read())
    ok(f"{rel} classified as {'pytest' if expect else 'script-style'}",
       got is expect, f"got defines_pytest_targets={got}")


print()
print("― rail 8: `pytest src/` collects the REAL tree cleanly (pythonpath = src) ―")
# Suites are normally run directly, which puts src/ on sys.path. Under pytest the rootdir
# is the project root, so without `pythonpath = src` in pytest.ini 17 suites die at import
# with ModuleNotFoundError. Those errors are collection noise, not real failures — but a
# run that always ends in errors is exactly what gets the guard above deleted. Delete the
# `pythonpath` line from pytest.ini and this rail fails.
ok("pytest.ini puts src/ on the path the way a direct run does",
   "pythonpath = src" in open(PYTEST_INI, encoding="utf-8").read())
p = subprocess.run(
    [sys.executable, "-m", "pytest", "src/", "--collect-only", "-q"],
    cwd=ROOT, capture_output=True, text=True, timeout=600,
)
collect_out = p.stdout + p.stderr
ok("`pytest src/ --collect-only` reports ZERO collection errors",
   "errors during collection" not in collect_out
   and "error during collection" not in collect_out
   and not any(ln.startswith("ERROR ") for ln in collect_out.splitlines()),
   collect_out[-800:])
ok("and it exits 0 rather than being Interrupted",
   p.returncode == 0, f"exit {p.returncode}: {collect_out[-500:]}")
ok("ModuleNotFoundError for engine modules is gone",
   "ModuleNotFoundError" not in collect_out, collect_out[-500:])

print()
print(f"{PASS} passed, {FAIL} failed")
for f in FAILURES:
    print("  FAIL:", f)
print("ALL PASS" if not FAIL else "FAILURES")
sys.exit(1 if FAIL else 0)
