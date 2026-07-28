"""Env-parity gates for local_update.py (board #161).

local_update.py writes to the PRODUCTION DB. Its whole value proposition —
"byte-identical to Railway" — is only true when the local environment matches
the batch-worker compute service. On 2026-07-27 it held 6 of 26 RORT_ flags and
a major-version-stale pandas, and the behavioral smoke test passed green through
both (a known-answer canary only detects faults its strategy is sensitive to).

These tests pin the two direct assertions that replace that false confidence:
  - _assert_flag_parity  — .env mirrors ALL of batch-worker's RORT_ flags,
    fails closed on MISSING / WRONG / STRAY / FORBIDDEN drift.
  - _assert_dependency_parity — venv pandas/numpy/pyarrow vs prod requirements
    pins; MAJOR skew aborts, minor/patch warns.

No DB / network: local_update defers all heavy imports into functions, so the
module imports clean and we drive the pure assertion logic with fakes.
"""
import importlib.metadata
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parent)
sys.path.insert(0, SRC)

import local_update as lu  # noqa: E402


# --- helpers -----------------------------------------------------------------
def _good_env_flags():
    """The exactly-correct set of RORT_ flags src/.env must contain: every
    MIRROR flag at its prod value + every OVERRIDE flag at its local value.
    Built from the manifest so it can never silently drift from it."""
    d = dict(lu.BATCH_WORKER_MIRROR_FLAGS)
    for name, (local_val, _prod, _why) in lu.BATCH_WORKER_OVERRIDE_FLAGS.items():
        d[name] = local_val
    return d


# --- manifest shape ----------------------------------------------------------
def test_manifest_accounts_for_all_26_batch_worker_flags():
    assert lu.BATCH_WORKER_FLAG_COUNT == 26, (
        "batch-worker had 26 RORT_ flags on 2026-07-27; the manifest must "
        "account for every one (mirror + override + omit)."
    )
    # No name may be double-classified.
    mirror = set(lu.BATCH_WORKER_MIRROR_FLAGS)
    override = set(lu.BATCH_WORKER_OVERRIDE_FLAGS)
    omit = set(lu.BATCH_WORKER_OMIT_FLAGS)
    assert mirror.isdisjoint(override)
    assert mirror.isdisjoint(omit)
    assert override.isdisjoint(omit)


# --- flag parity: verify E3's real .env mirror is complete & correct ---------
def test_real_env_file_mirrors_manifest_exactly():
    """The strong check the task asks for: verify src/.env covers all 26 and is
    exactly right — the 22 required flags present at the right values, none of
    the 4 scheduler-only flags leaking in."""
    env_flags = lu._parse_env_rort_flags(lu._find_env_path())
    assert env_flags == _good_env_flags(), (
        "src/.env RORT_ block has drifted from the batch-worker manifest — "
        "this is the exact #161 failure mode."
    )
    for forbidden in lu.BATCH_WORKER_OMIT_FLAGS:
        assert forbidden not in env_flags


def test_flag_parity_passes_on_correct_env(monkeypatch):
    monkeypatch.setattr(lu, "_parse_env_rort_flags", lambda _p: _good_env_flags())
    lu._assert_flag_parity()  # must not raise


def test_flag_parity_fails_on_missing_flag(monkeypatch):
    bad = _good_env_flags()
    bad.pop("RORT_SUPPRESS_EOD_REENTRY")  # the exact flag 338 was blind to
    monkeypatch.setattr(lu, "_parse_env_rort_flags", lambda _p: bad)
    with pytest.raises(SystemExit) as e:
        lu._assert_flag_parity()
    assert "MISSING" in str(e.value)


def test_flag_parity_fails_on_wrong_value(monkeypatch):
    bad = _good_env_flags()
    bad["RORT_ENFORCE_1MIN_GATE"] = "0"
    monkeypatch.setattr(lu, "_parse_env_rort_flags", lambda _p: bad)
    with pytest.raises(SystemExit) as e:
        lu._assert_flag_parity()
    assert "WRONG" in str(e.value)


def test_flag_parity_fails_on_stray_flag(monkeypatch):
    bad = _good_env_flags()
    bad["RORT_SOME_NEW_PROD_FLAG"] = "1"
    monkeypatch.setattr(lu, "_parse_env_rort_flags", lambda _p: bad)
    with pytest.raises(SystemExit) as e:
        lu._assert_flag_parity()
    assert "STRAY" in str(e.value)


def test_flag_parity_fails_on_forbidden_scheduler_flag(monkeypatch):
    bad = _good_env_flags()
    bad["RORT_NIGHTLY_RECOMPUTE"] = "1"  # would fire recomputes on a workstation
    monkeypatch.setattr(lu, "_parse_env_rort_flags", lambda _p: bad)
    with pytest.raises(SystemExit) as e:
        lu._assert_flag_parity()
    assert "FORBIDDEN" in str(e.value)


def test_flag_parity_fails_on_forbidden_flag_in_environment(monkeypatch):
    """A scheduler flag exported in the shell (not .env) must also abort —
    load_dotenv(override=True) would NOT clear it."""
    monkeypatch.setattr(lu, "_parse_env_rort_flags", lambda _p: _good_env_flags())
    monkeypatch.setenv("RORT_NIGHTLY_RECOMPUTE", "1")
    with pytest.raises(SystemExit) as e:
        lu._assert_flag_parity()
    assert "FORBIDDEN" in str(e.value)


# --- dependency parity -------------------------------------------------------
def test_requirements_pins_parse():
    pins = lu._parse_requirements_pins(lu._find_requirements_path(), lu.NUMERIC_PACKAGES)
    assert pins["pandas"] == ("==", "3.0.3")
    assert pins["numpy"] == ("==", "2.5.1")
    assert pins["pyarrow"] == ("==", "24.0.0")


def _fake_versions(mapping):
    def _v(pkg):
        if pkg in mapping:
            return mapping[pkg]
        raise importlib.metadata.PackageNotFoundError(pkg)
    return _v


def test_dep_parity_passes_on_exact_match(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version",
                        _fake_versions({"pandas": "3.0.3", "numpy": "2.5.1", "pyarrow": "24.0.0"}))
    lu._assert_dependency_parity()  # must not raise


def test_dep_parity_fails_on_major_skew(monkeypatch):
    # the actual #161 bug: venv pandas 2.x against prod 3.x
    monkeypatch.setattr(importlib.metadata, "version",
                        _fake_versions({"pandas": "2.3.3", "numpy": "2.5.1", "pyarrow": "24.0.0"}))
    with pytest.raises(SystemExit) as e:
        lu._assert_dependency_parity()
    assert "MAJOR skew" in str(e.value)


def test_dep_parity_warns_but_passes_on_minor_skew(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version",
                        _fake_versions({"pandas": "3.0.1", "numpy": "2.5.1", "pyarrow": "24.0.0"}))
    lu._assert_dependency_parity()  # minor skew warns loudly, does NOT abort


def test_dep_parity_fails_on_missing_package(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version",
                        _fake_versions({"numpy": "2.5.1", "pyarrow": "24.0.0"}))  # pandas absent
    with pytest.raises(SystemExit) as e:
        lu._assert_dependency_parity()
    assert "NOT INSTALLED" in str(e.value)


def test_major_helper():
    assert lu._major("3.0.3") == 3
    assert lu._major("24.0.0") == 24
    assert lu._major("2.3.3") == 2
    assert lu._major("garbage") == -1
