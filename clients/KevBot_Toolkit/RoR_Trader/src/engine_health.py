"""Worker liveness check — the Docker HEALTHCHECK entrypoint (2026-07-14).

WHY THIS EXISTS: the old healthcheck was `test -f /tmp/worker_alive` — it tested
that a file EXISTS. That file is touched by the worker MANAGER thread, which
keeps ticking happily even when the asyncio ENGINE (the thing that closes bars
and fires alerts) is starved. On 2026-07-14 the engine was stalled ~9 minutes at
a time by a periodic in-process rebuild: alerts dispatched 5-10 MINUTES late,
bar closes skipped outright, a real live entry lost — and the healthcheck stayed
green through all of it (memory: project_primary_resync_engine_starvation).

Liveness is now the ENGINE's pulse: `ralph_engine._periodic_tasks_loop` touches
ENGINE_HEARTBEAT_FILE every ~2s. We fail when that heartbeat AGES OUT.

Boot tolerance: the engine warms up (REST loads, indicator warmup) for minutes
before its periodic loop starts, so the heartbeat file may not exist yet. While
it is absent we fall back to the manager's file — but only for a bounded window,
so "engine never started" still fails eventually.

Exit 0 = healthy, exit 1 = unhealthy (Docker restarts / marks unhealthy).
"""
import os
import sys
import time

ENGINE_FILE = os.getenv("RORT_ENGINE_HEARTBEAT_FILE", "/tmp/engine_alive")
MANAGER_FILE = os.getenv("RORT_WORKER_HEARTBEAT_FILE", "/tmp/worker_alive")
MAX_AGE_S = float(os.getenv("RORT_ENGINE_HEARTBEAT_MAX_AGE_S", "300"))
# Brandon P2 (2026-07-16): boot grace must be BOUNDED. worker.main() writes a
# WRITE-ONCE-PER-BOOT marker; when the engine heartbeat is absent we grant
# grace only while that marker is younger than the deadline. Without the bound,
# the continuously-refreshed MANAGER file grants unlimited grace — an engine
# that NEVER starts looks healthy forever.
# Default 1200s: generous vs observed fleet warmups (minutes) because the
# failure mode of a TOO-TIGHT deadline is a Docker restart LOOP; the bound's
# purpose is catching "never starts" (infinite), so 20min loses nothing.
BOOT_DEADLINE_S = float(os.getenv("RORT_ENGINE_BOOT_DEADLINE_S", "1200"))


def _bounded_boot() -> bool:
    """RORT_HEALTHCHECK_BOUNDED_BOOT (default OFF). Read at runtime so tests
    toggle without reload. ⚠️ This check GATES Docker restarts — arm only after
    verifying in logs that the boot marker is written at startup."""
    return os.getenv(
        "RORT_HEALTHCHECK_BOUNDED_BOOT", "0").strip().lower() in (
        "1", "true", "yes", "on")


# Brandon P2(b) — feed freshness (2026-07-16): the engine heartbeat proves the
# LOOP is alive, not that DATA is flowing; a silently-dead WS feed stays
# Docker-healthy forever. Flag ON: when the engine is otherwise healthy, a
# LAST-TICK marker (touched from the per-second path) must not be
# [stale_s, cap_s] old during the enforcement window. HOLIDAY/RESTART-LOOP
# SAFETY: enforce ONLY when the marker EXISTS and its age is INSIDE the band
# (a closed day drifts past cap_s and stops alarming; a missing marker —
# post-restart on a closed day — never alarms), and only inside 14:35-19:55
# UTC Mon-Fri (inside RTH in BOTH DST regimes — no season bugs).
FEED_STALE_S = float(os.getenv("RORT_FEED_STALE_S", "900"))
FEED_STALE_CAP_S = float(os.getenv("RORT_FEED_STALE_CAP_S", "7200"))


def _feed_freshness() -> bool:
    return os.getenv(
        "RORT_HEALTHCHECK_FEED_FRESHNESS", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _in_enforce_window(now: float) -> bool:
    """Mon-Fri 14:35-19:55 UTC — strictly inside RTH under both EDT and EST."""
    t = time.gmtime(now)
    if t.tm_wday > 4:  # Sat/Sun
        return False
    minutes = t.tm_hour * 60 + t.tm_min
    return (14 * 60 + 35) <= minutes <= (19 * 60 + 55)


def _feed_check(now: float, feed_file: str,
                stale_s: float, cap_s: float):
    """None = no objection; (False, reason) = feed dead. Band + window rules
    per the safety note above."""
    if not _in_enforce_window(now):
        return None
    feed_age = _age(feed_file, now)
    if feed_age is None:
        return None  # marker absent (fresh boot on a closed day) — never alarm
    if stale_s <= feed_age <= cap_s:
        return False, (
            f"FEED DEAD: engine loop alive but no market data for "
            f"{feed_age:.0f}s during market hours (stale>{stale_s:.0f}s; "
            f"WS feed / subscription likely down)")
    return None


def _age(path: str, now: float) -> float | None:
    """Seconds since `path` was last touched, or None if it doesn't exist."""
    try:
        return now - os.path.getmtime(path)
    except OSError:
        return None


def check(now: float | None = None,
          engine_file: str = ENGINE_FILE,
          manager_file: str = MANAGER_FILE,
          max_age_s: float = MAX_AGE_S,
          boot_marker_file: str | None = None,
          boot_deadline_s: float | None = None,
          feed_file: str | None = None) -> tuple[bool, str]:
    """(healthy, reason). Pure + injectable so it is unit-testable."""
    now = time.time() if now is None else now
    engine_age = _age(engine_file, now)
    if engine_age is not None:
        if engine_age < max_age_s:
            # Brandon P2(b): loop-alive is not feed-alive. Flag ON: a dead
            # market-data feed during market hours fails the check even with
            # a fresh engine heartbeat.
            if _feed_freshness():
                if feed_file is None:
                    feed_file = (os.getenv("RORT_ENGINE_FEED_MARKER_FILE")
                                 or os.path.join(
                                     os.path.dirname(engine_file) or ".",
                                     "engine_last_tick"))
                verdict = _feed_check(now, feed_file, FEED_STALE_S,
                                      FEED_STALE_CAP_S)
                if verdict is not None:
                    return verdict
            return True, f"engine heartbeat {engine_age:.0f}s old"
        return False, (
            f"ENGINE STALLED: heartbeat {engine_age:.0f}s old "
            f"(max {max_age_s:.0f}s) — bar closes/alerts are not being "
            f"processed")
    # Engine heartbeat absent: still booting (warmup).
    manager_age = _age(manager_file, now)
    if _bounded_boot():
        # Brandon P2 (RORT_HEALTHCHECK_BOUNDED_BOOT=1): grace is bounded by the
        # WRITE-ONCE boot marker's age — a fresh MANAGER file alone can no
        # longer grant unlimited grace to an engine that never starts.
        if boot_marker_file is None:
            boot_marker_file = (os.getenv("RORT_ENGINE_BOOT_MARKER_FILE")
                                or os.path.join(
                                    os.path.dirname(engine_file) or ".",
                                    "engine_boot_marker"))
        if boot_deadline_s is None:
            boot_deadline_s = BOOT_DEADLINE_S
        boot_age = _age(boot_marker_file, now)
        if boot_age is None:
            return False, (
                "engine heartbeat absent and no boot marker — cannot bound "
                "boot grace (worker.main() writes it at startup)")
        if boot_age >= boot_deadline_s:
            return False, (
                f"ENGINE NEVER STARTED: {boot_age:.0f}s since boot "
                f"(deadline {boot_deadline_s:.0f}s) and no engine heartbeat")
        if manager_age is not None and manager_age < max_age_s:
            return True, (
                f"engine not up yet; {boot_age:.0f}s into boot "
                f"(deadline {boot_deadline_s:.0f}s; manager alive)")
        return False, (
            f"engine absent during boot and manager heartbeat "
            f"{'missing' if manager_age is None else f'{manager_age:.0f}s old'}")
    # Legacy (flag OFF): trust the manager file — UNBOUNDED grace (Brandon P2
    # documents the gap; the flag above is the fix).
    if manager_age is None:
        return False, "no heartbeat files at all (process not up)"
    if manager_age < max_age_s:
        return True, (
            f"engine not up yet; manager heartbeat {manager_age:.0f}s old "
            f"(boot grace)")
    return False, (
        f"no engine heartbeat and manager heartbeat is {manager_age:.0f}s old")


def main() -> int:
    healthy, reason = check()
    print(("HEALTHY: " if healthy else "UNHEALTHY: ") + reason)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
