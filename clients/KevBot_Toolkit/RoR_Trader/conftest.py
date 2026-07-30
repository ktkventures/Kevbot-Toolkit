"""pytest guard — never report success for a suite pytest did not execute (board #239).

Most of this project's acceptance suites are *script-style*: they do their work at
import, print `ok:` / `FAIL:` lines, and are invoked directly by name --

    .venv/bin/python src/test_board_statuses_232.py

They deliberately define no `test_` functions and depend on nothing but the stdlib,
which is what makes them reliable for an agent to run with one command. That is a
feature and this guard does not try to change it.

The hazard is what pytest does with them. Measured on 2026-07-30 against the 80
`test_*.py` files in `src/` and `tools/`:

  * 21 suites define no `test_` function and do not exit at import. pytest collects
    ZERO items from them and says nothing. Run `pytest src/`, and the 43 genuine
    pytest suites collect and pass, pytest exits 0, and those 21 were never executed
    -- a green bar over an unmeasured rail. A suite that exits 1 when run directly
    reports `1 passed` under pytest.
  * 15 suites call `sys.exit()` at module scope. pytest raises SystemExit *during
    collection*, which surfaces as an INTERNALERROR and aborts the whole session --
    loud, but it takes every other suite down with it.

So this hook refuses to let pytest silently own a file it cannot run. For any
`test_*.py` with no collectable `test_` function or `Test*` class, it emits one
FAILING item naming the file and the exact command to run it properly.

Deliberately a refusal and not a subprocess runner: several of these suites hit the
production database or the network (`test_c_type_real_backtest`, `test_hifi_pack_smoke`,
`test_overnight_gap_full_cycle`). Auto-executing them on every `pytest` invocation
would quietly turn a test command into heavy prod load. A loud refusal is the safe
outcome; the human or agent then runs the suite the way it was built to be run.

Detection is by AST and never imports the file -- importing is the thing that
triggers the INTERNALERROR this guard exists to prevent.
"""

from __future__ import annotations

import ast
import os

import pytest

# Directories that never hold our acceptance suites.
_SKIP_DIRS = {".venv", "node_modules", ".git", "__pycache__", ".next", "build", "dist"}


def defines_pytest_targets(source: str) -> bool:
    """True if pytest would collect at least one item from this source.

    Mirrors pytest's default `python_functions`/`python_classes` rules: a module-level
    `test_*` function (sync or async), or a `Test*` class. Anything else -- including a
    file whose entire body is top-level `ok(...)` calls -- collects nothing.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Let pytest report the syntax error itself; it is loud on its own.
        return True

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                return True
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                return True
    return False


class UnexecutedSuiteError(Exception):
    """Raised for a script-style suite pytest cannot run."""


class ScriptStyleItem(pytest.Item):
    """One failing item standing in for a suite pytest refuses to pretend it ran."""

    def __init__(self, *, rel_path: str, **kwargs):
        super().__init__(**kwargs)
        self.rel_path = rel_path

    def runtest(self):
        raise UnexecutedSuiteError(self.rel_path)

    def repr_failure(self, excinfo, style=None):
        return (
            f"pytest cannot execute this suite, so it must not report success for it.\n"
            f"\n"
            f"    {self.rel_path}\n"
            f"\n"
            f"It is a script-style acceptance suite: it runs its checks at import and\n"
            f"defines no `test_` function, so pytest collects zero items from it. Left\n"
            f"unguarded that is a false green -- a suite that FAILS when run directly\n"
            f"still reports `passed` under pytest.\n"
            f"\n"
            f"Run it the way it was built to be run:\n"
            f"\n"
            f"    .venv/bin/python {self.rel_path}\n"
            f"\n"
            f"It exits 0 on pass and 1 on failure, so it gates correctly on its own.\n"
            f"Do not wire these into a pytest run (see conftest.py / board #239)."
        )

    def reportinfo(self):
        return self.path, 0, f"script-style suite not executed by pytest: {self.rel_path}"


class ScriptStyleFile(pytest.File):
    def collect(self):
        try:
            rel = os.path.relpath(str(self.path), str(self.config.rootpath))
        except ValueError:
            rel = str(self.path)
        yield ScriptStyleItem.from_parent(
            self, name="script_style_suite_not_executed", rel_path=rel
        )


def _script_style_collector(file_path, parent):
    """Return a refusing collector for a script-style suite, else None."""
    if file_path.suffix != ".py":
        return None
    if not file_path.name.startswith("test_"):
        return None
    if _SKIP_DIRS.intersection(file_path.parts):
        return None

    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if defines_pytest_targets(source):
        return None  # a genuine pytest suite -- let pytest handle it normally

    return ScriptStyleFile.from_parent(parent, path=file_path)


def pytest_pycollect_makemodule(module_path, parent):
    """Substitute our refusing collector for pytest's Module -- before any import.

    This is the hook that has to do the work, not `pytest_collect_file`. That one is
    NOT `firstresult`: pytest gathers every non-None result, so returning a collector
    from it leaves pytest's own python plugin free to ALSO build a Module and import
    the file -- which is precisely the import that raises SystemExit at module scope
    and blows the session up with an INTERNALERROR. `pytest_pycollect_makemodule` IS
    `firstresult`, so returning here replaces the Module outright and the file is
    never imported.
    """
    return _script_style_collector(module_path, parent)
