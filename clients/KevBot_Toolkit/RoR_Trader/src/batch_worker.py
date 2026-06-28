#!/usr/bin/env python3
"""M-RS3 Phase 2 — dedicated batch-worker service.

Claims `compute_jobs` rows (Update-All `full_recompute` / Append-New
`append_recent`, enqueued by the API when RORT_COMPUTE_REMOTE=1) and runs each
through `recompute_jobs.run_remote_job`, which fans strategies across a process
pool (RORT_RECOMPUTE_PARALLELISM). Runs on its OWN Railway service so the CPU
burn never contends with the live API/worker (sub-second latency is a hard req).

Mirrors data_worker.py's shape: SIGTERM/SIGINT graceful stop, /tmp health file
(touched on a SEPARATE daemon thread so a multi-minute job doesn't look
unhealthy), BATCH_WORKER_DISABLED kill-switch. One job at a time per replica —
the pool gives intra-job parallelism; add replicas for cross-job throughput
(the optimistic claim is replica-safe).
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("batch_worker")

HEALTH_FILE = "/tmp/batch_worker_alive"
POLL_INTERVAL_S = int(os.environ.get("BATCH_WORKER_POLL_S", "5"))
STALE_RECLAIM_S = int(os.environ.get("BATCH_WORKER_STALE_RECLAIM_S", "1800"))
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Received signal %d — finishing current job then exiting",
                signum)
    _running = False


def _touch_health():
    try:
        Path(HEALTH_FILE).touch()
    except Exception as e:  # noqa: BLE001
        logger.warning("health touch failed: %s", e)


def _start_health_thread(stop_evt: threading.Event) -> threading.Thread:
    """Touch the health file every few seconds, independent of job duration —
    a synchronous multi-minute recompute must not trip the container HEALTHCHECK."""
    def loop():
        while not stop_evt.is_set():
            _touch_health()
            stop_evt.wait(5)
    t = threading.Thread(target=loop, name="batch-health", daemon=True)
    t.start()
    return t


def main():
    global _running

    if os.environ.get("BATCH_WORKER_DISABLED", "").strip().lower() in (
            "1", "true", "yes", "on"):
        logger.warning("BATCH_WORKER_DISABLED set — exiting without starting.")
        return

    # Local dev loads src/.env; Railway injects env vars directly (no .env).
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
    except Exception:  # noqa: BLE001
        pass

    os.environ.setdefault("USE_DB", "true")
    # This service IS the remote runner — make recompute_jobs DB-aware here even
    # if the var wasn't set explicitly on the service.
    os.environ.setdefault("RORT_COMPUTE_REMOTE", "1")

    # Registry once at startup: the N=1 in-parent path + _lookup_strat_name need
    # it (pool workers re-load their own in _pool_init). Fail loud if empty.
    import pack_registry
    npacks = len(pack_registry.scan_and_load_all())
    if not npacks:
        raise SystemExit("[batch_worker] pack registry empty — refusing to "
                         "start (every recompute would return 0 trades)")
    parallelism = os.environ.get("RORT_RECOMPUTE_PARALLELISM", "1")
    # Resources. os.cpu_count()/psutil report the HOST (overstated under a
    # container). The CONTAINER cgroup quota/limit is the real parallelism
    # ceiling: CPU quota = usable cores; RAM ÷ per-worker footprint (~1.5-2.5GB
    # on heavy 10Sec strats) caps safe N.
    def _cgroup_limits():
        cpu = ram = None
        try:  # cgroup v2
            q, p = open("/sys/fs/cgroup/cpu.max").read().split()
            if q != "max":
                cpu = int(q) / int(p)
        except Exception:  # noqa: BLE001
            pass
        try:
            v = open("/sys/fs/cgroup/memory.max").read().strip()
            if v != "max":
                ram = int(v) / 1e9
        except Exception:  # noqa: BLE001
            pass
        if cpu is None:  # cgroup v1
            try:
                q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
                p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
                if q > 0:
                    cpu = q / p
            except Exception:  # noqa: BLE001
                pass
        if ram is None:
            try:
                v = int(open("/sys/fs/cgroup/memory/memory.limit_in_bytes").read())
                if v < 1e15:
                    ram = v / 1e9
            except Exception:  # noqa: BLE001
                pass
        return cpu, ram
    _host_ram = "?"
    try:
        import psutil
        _host_ram = f"{psutil.virtual_memory().total / 1e9:.0f}GB"
    except Exception:  # noqa: BLE001
        pass
    _cg_cpu, _cg_ram = _cgroup_limits()
    logger.info(
        "[batch_worker] RESOURCES host(cpu=%s ram=%s) CONTAINER(cpu_limit=%s "
        "ram_limit=%s) — N should be <= min(cpu_limit, ram_limit/~2.5GB)",
        os.cpu_count(), _host_ram,
        f"{_cg_cpu:.1f}" if _cg_cpu else "unlimited",
        f"{_cg_ram:.1f}GB" if _cg_ram else "unlimited")
    logger.info("[batch_worker] up id=%s packs=%d parallelism=%s poll=%ss "
                "stale_reclaim=%ss", WORKER_ID, npacks, parallelism,
                POLL_INTERVAL_S, STALE_RECLAIM_S)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    stop_evt = threading.Event()
    _start_health_thread(stop_evt)

    import compute_jobs_store
    import recompute_jobs

    last_reclaim = 0.0
    idle_logged = False
    while _running:
        # Periodic janitor: requeue jobs a dead replica left 'running'.
        now = time.time()
        if STALE_RECLAIM_S and now - last_reclaim > STALE_RECLAIM_S:
            try:
                compute_jobs_store.reclaim_stale(STALE_RECLAIM_S)
            except Exception as e:  # noqa: BLE001
                logger.warning("reclaim_stale failed: %s", e)
            last_reclaim = now

        try:
            row = compute_jobs_store.claim_one(WORKER_ID)
        except Exception as e:  # noqa: BLE001
            logger.warning("claim failed: %s", e)
            row = None

        if row is None:
            if not idle_logged:
                logger.info("[batch_worker] queue empty — idling")
                idle_logged = True
            for _ in range(POLL_INTERVAL_S):
                if not _running:
                    break
                time.sleep(1)
            continue

        idle_logged = False
        jid = str(row.get("id"))[:8]
        t0 = time.time()
        logger.info("[batch_worker] running job=%s type=%s n=%d", jid,
                    row.get("job_type"), len(row.get("strategy_ids") or []))
        try:
            recompute_jobs.run_remote_job(row)
            logger.info("[batch_worker] job=%s done in %.1fs",
                        jid, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            logger.exception("[batch_worker] job=%s crashed: %s", jid, e)
            try:
                compute_jobs_store.update_row(
                    row["id"], status="failed",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error=f"{type(e).__name__}: {e}")
            except Exception:  # noqa: BLE001
                pass

    stop_evt.set()
    logger.info("[batch_worker] stopped")


if __name__ == "__main__":
    main()
