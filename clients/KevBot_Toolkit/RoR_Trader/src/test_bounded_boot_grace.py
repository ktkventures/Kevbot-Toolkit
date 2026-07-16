"""RORT_HEALTHCHECK_BOUNDED_BOOT (Brandon audit P2) — flag-gated regression
gate locking BOTH states:
  * flag ON  → boot grace is BOUNDED by the write-once boot marker's age; a
               fresh manager file alone cannot grant unlimited grace
  * flag OFF → legacy byte-identical (manager fallback, unbounded)

⚠️ The check gates Docker restarts. These tests are the pre-arm proof.

    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_bounded_boot_grace.py
"""
from __future__ import annotations

import os

import engine_health as EH

FLAG = "RORT_HEALTHCHECK_BOUNDED_BOOT"
T0 = 1_000_000.0


def _files(tmp_path, engine=None, manager=None, marker=None):
    """Create heartbeat files with given mtimes (None = absent)."""
    paths = {}
    for name, mtime in (("engine", engine), ("manager", manager),
                        ("engine_boot_marker", marker)):
        p = tmp_path / name
        paths[name] = str(p)
        if mtime is not None:
            p.write_text("x")
            os.utime(p, (mtime, mtime))
    return paths


def _check(tmp_path, now, **mtimes):
    p = _files(tmp_path, **mtimes)
    return EH.check(now=now, engine_file=p["engine"],
                    manager_file=p["manager"], max_age_s=300,
                    boot_marker_file=p["engine_boot_marker"],
                    boot_deadline_s=600)


def test_flag_helper_runtime(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert EH._bounded_boot() is False
    monkeypatch.setenv(FLAG, "1")
    assert EH._bounded_boot() is True


def test_flag_on_fresh_boot_within_deadline_is_healthy(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, reason = _check(tmp_path, now=T0 + 120,
                             manager=T0 + 119, marker=T0)
    assert healthy, f"120s into a 600s boot deadline must be grace: {reason}"


def test_flag_on_deadline_exceeded_is_unhealthy(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, reason = _check(tmp_path, now=T0 + 700,
                             manager=T0 + 699, marker=T0)
    assert not healthy and "NEVER STARTED" in reason, (
        f"700s past boot with no engine heartbeat must fail: {reason}")


def test_flag_on_missing_marker_is_unhealthy(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, reason = _check(tmp_path, now=T0 + 120, manager=T0 + 119)
    assert not healthy and "boot marker" in reason, (
        f"no marker → cannot bound grace → unhealthy: {reason}")


def test_flag_on_engine_heartbeat_still_primary(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    # Engine present + fresh → healthy regardless of marker age (normal path).
    healthy, _ = _check(tmp_path, now=T0 + 10_000, engine=T0 + 9_990,
                        manager=T0 + 9_990, marker=T0)
    assert healthy
    # Engine present + stale → ENGINE STALLED (unchanged).
    healthy, reason = _check(tmp_path, now=T0 + 10_000, engine=T0 + 9_000,
                             manager=T0 + 9_999, marker=T0)
    assert not healthy and "STALLED" in reason


def test_flag_off_legacy_unbounded_grace(monkeypatch, tmp_path):
    # Brandon's exact scenario stays HEALTHY flag-OFF (documents the legacy
    # gap; proves the flag is the only switch).
    monkeypatch.delenv(FLAG, raising=False)
    healthy, reason = _check(tmp_path, now=T0 + 86_400,
                             manager=T0 + 86_399)
    assert healthy, f"flag OFF must preserve legacy unbounded grace: {reason}"


def test_flag_on_marker_default_derives_from_engine_dir(monkeypatch, tmp_path):
    # When boot_marker_file isn't passed, it derives from the engine file's
    # directory — so an isolated tmp dir with no marker is deterministic.
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.delenv("RORT_ENGINE_BOOT_MARKER_FILE", raising=False)
    p = _files(tmp_path, manager=T0 + 86_399)
    healthy, reason = EH.check(now=T0 + 86_400, engine_file=p["engine"],
                               manager_file=p["manager"], max_age_s=300)
    assert not healthy, (
        f"fresh manager must not grant unlimited grace flag-ON: {reason}")
