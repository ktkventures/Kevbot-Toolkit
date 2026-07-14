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


def _age(path: str, now: float) -> float | None:
    """Seconds since `path` was last touched, or None if it doesn't exist."""
    try:
        return now - os.path.getmtime(path)
    except OSError:
        return None


def check(now: float | None = None,
          engine_file: str = ENGINE_FILE,
          manager_file: str = MANAGER_FILE,
          max_age_s: float = MAX_AGE_S) -> tuple[bool, str]:
    """(healthy, reason). Pure + injectable so it is unit-testable."""
    now = time.time() if now is None else now
    engine_age = _age(engine_file, now)
    if engine_age is not None:
        if engine_age < max_age_s:
            return True, f"engine heartbeat {engine_age:.0f}s old"
        return False, (
            f"ENGINE STALLED: heartbeat {engine_age:.0f}s old "
            f"(max {max_age_s:.0f}s) — bar closes/alerts are not being "
            f"processed")
    # Engine heartbeat absent: still booting (warmup) — trust the manager file
    # for a bounded window only.
    manager_age = _age(manager_file, now)
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
