"""RORT_HEALTHCHECK_FEED_FRESHNESS (Brandon audit P2(b)) — flag-gated:
  * flag ON  → a fresh engine heartbeat with a DEAD data feed fails the check,
               but ONLY inside the band [stale_s, cap_s] and the Mon-Fri
               14:35-19:55 UTC window (holiday / closed-day / restart safe)
  * flag OFF → legacy byte-identical (feed never consulted)

    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_feed_freshness.py
"""
from __future__ import annotations

import calendar
import os

import engine_health as EH

FLAG = "RORT_HEALTHCHECK_FEED_FRESHNESS"
# Wed 2026-07-15 15:00:00 UTC — inside the enforcement window.
RTH_NOW = calendar.timegm((2026, 7, 15, 15, 0, 0))
# Wed 2026-07-15 02:00:00 UTC — outside it (overnight).
NIGHT_NOW = calendar.timegm((2026, 7, 15, 2, 0, 0))
# Sat 2026-07-18 15:00:00 UTC — weekend.
SAT_NOW = calendar.timegm((2026, 7, 18, 15, 0, 0))


def _files(tmp_path, engine_age=10, feed_age=None, now=RTH_NOW):
    e = tmp_path / "engine"
    e.write_text("x")
    os.utime(e, (now - engine_age, now - engine_age))
    f = tmp_path / "engine_last_tick"
    if feed_age is not None:
        f.write_text("x")
        os.utime(f, (now - feed_age, now - feed_age))
    m = tmp_path / "manager"
    m.write_text("x")
    os.utime(m, (now - 5, now - 5))
    return str(e), str(m), str(f)


def _check(tmp_path, now=RTH_NOW, **kw):
    e, m, f = _files(tmp_path, now=now, **kw)
    return EH.check(now=now, engine_file=e, manager_file=m, max_age_s=300,
                    feed_file=f)


def test_flag_off_feed_never_consulted(monkeypatch, tmp_path):
    monkeypatch.delenv(FLAG, raising=False)
    healthy, _ = _check(tmp_path, feed_age=3000)  # dead feed, flag off
    assert healthy, "flag OFF must be legacy (engine heartbeat alone decides)"


def test_flag_on_dead_feed_in_band_fails(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, reason = _check(tmp_path, feed_age=1200)  # in [900, 7200]
    assert not healthy and "FEED DEAD" in reason


def test_flag_on_fresh_feed_healthy(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, _ = _check(tmp_path, feed_age=30)
    assert healthy


def test_flag_on_beyond_cap_is_closed_day_safe(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, _ = _check(tmp_path, feed_age=30_000)  # > cap → holiday-safe
    assert healthy, "age past the cap = closed day, must not restart-loop"


def test_flag_on_marker_absent_safe(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, _ = _check(tmp_path, feed_age=None)  # never ticked (closed-day boot)
    assert healthy, "absent marker must never alarm"


def test_flag_on_outside_window_safe(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    for now in (NIGHT_NOW, SAT_NOW):
        healthy, _ = _check(tmp_path, now=now, feed_age=1200)
        assert healthy, "overnight/weekend must not enforce"


def test_engine_stall_still_primary(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    healthy, reason = _check(tmp_path, engine_age=9999, feed_age=30)
    assert not healthy and "STALLED" in reason