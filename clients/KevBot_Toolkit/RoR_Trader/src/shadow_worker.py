#!/usr/bin/env python3
"""M-RS4 Phase 3 — dedicated shadow-worker service (Step B: INERT scaffold).

The continuous-resident backtest engine (Plan_M-RS4_Phase3_ContinuousBacktestEngine.md).
Its job — landing across Steps C–F — is to keep each strategy's `backtest_<model>`
lane continuously, byte-identically fresh off the REST Bar Cache (`bar_cache`), so
"Append-New" is never needed again. The byte-identity premise is PROVEN offline by
`_resident_replay_harness.py` (Step A, GREEN): one warmed engine fed bar-by-bar ==
from-cold recompute, vs snapshot-resume's ~4% boundary loss.

STEP B (this commit) ships the service INERT. It mirrors batch_worker.py's shape
(SIGTERM/SIGINT graceful stop, /tmp health file on a SEPARATE daemon thread,
*_DISABLED kill-switch) but does NOTHING to the data until the lane-mode flag flips
to `shadow`. Until then it idles (heartbeat only). Even in `shadow` mode it is a
no-op in Step B — the resident engine manager arrives in Step C. This lets us deploy
the service to dev and prove liveness/crash-isolation with zero fidelity risk.

Gating: `RORT_BACKTEST_LANE_MODE` ∈ {button, cron, shadow}, default `button` (today's
behavior — the backtest lane is populated by the manual Append / cron append). The
flag is read at startup; Kevin / the backend agent sequence the flip relative to a
data pass — this service never flips it itself.

  button  the lane is driven by manual Append-New / Update-Backtest (today)
  cron    the lane is driven by the scheduled append (today's automated path)
  shadow  the lane is driven by THIS continuous-resident service (Step C+)
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("shadow_worker")

HEALTH_FILE = "/tmp/shadow_worker_alive"
POLL_INTERVAL_S = int(os.environ.get("SHADOW_WORKER_POLL_S", "5"))
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
VALID_MODES = ("button", "cron", "shadow")

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Received signal %d — stopping", signum)
    _running = False


def _lane_mode() -> str:
    """Read RORT_BACKTEST_LANE_MODE (default 'button'). Fail loud on an
    unrecognized value — a typo'd mode must never silently fall back to a
    different lane source (fidelity-critical)."""
    raw = os.environ.get("RORT_BACKTEST_LANE_MODE", "").strip().lower()
    if not raw:
        return "button"
    if raw not in VALID_MODES:
        raise SystemExit(
            f"[shadow_worker] RORT_BACKTEST_LANE_MODE={raw!r} is invalid — "
            f"must be one of {VALID_MODES}. Refusing to start.")
    return raw


def _touch_health():
    try:
        Path(HEALTH_FILE).touch()
    except Exception as e:  # noqa: BLE001
        logger.warning("health touch failed: %s", e)


def _start_health_thread(stop_evt: threading.Event) -> threading.Thread:
    """Touch the health file every few seconds, independent of work duration —
    so the container HEALTHCHECK stays green even mid-tick (batch-worker mold)."""
    def loop():
        while not stop_evt.is_set():
            _touch_health()
            stop_evt.wait(5)
    t = threading.Thread(target=loop, name="shadow-health", daemon=True)
    t.start()
    return t


def _idle_loop(stop_evt: threading.Event):
    """Inert heartbeat loop: prove liveness/crash-isolation, touch nothing.
    Used in {button, cron} mode (the lane is owned by another path)."""
    idle_logged = False
    while _running:
        if not idle_logged:
            logger.info("[shadow_worker] inert — idling (lane owned elsewhere)")
            idle_logged = True
        for _ in range(POLL_INTERVAL_S):
            if not _running:
                break
            time.sleep(1)


def _dry_run_default() -> bool:
    """Default ON — shadow mode COMPUTES but writes nothing until explicitly armed via
    RORT_SHADOW_DRY_RUN=0. A second safety gate beyond the lane-mode flag."""
    return os.environ.get("RORT_SHADOW_DRY_RUN", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _kpi_drain_loop(mgr):
    """Fix 2b: run the KPI + equity + Hi-Fi recompute OFF the hot poll path. Drains the
    manager's KPI queue (sids that traded) and recomputes each on its own cadence
    (`maybe_recompute_kpis` is debounced + resets the has-traded flag on success). Keeps
    the poll loop O(new bars) so no single slot's reload-all recompute starves the tail."""
    import time as _t
    drain_s = int(os.environ.get("SHADOW_KPI_DRAIN_S", "10"))
    while _running:
        try:
            for sid in mgr.drain_kpi():
                if not _running:
                    break
                slot = mgr.slots.get(sid)
                if slot is not None:
                    try:
                        mgr.maybe_recompute_kpis(slot)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[shadow_worker] KPI recompute sid=%s failed: %s",
                                       sid, e)
        except Exception as e:  # noqa: BLE001
            logger.warning("[shadow_worker] KPI drain loop error: %s", e)
        for _ in range(drain_s):
            if not _running:
                break
            _t.sleep(1)


_poll_executor = None
_poll_executor_workers = 0


def _get_poll_executor(workers: int):
    """Lazy, reused ThreadPoolExecutor for parallel slot polls (M-RS4 Fix 2c).

    Mirrors live_bars_writer.py's pool pattern. Sized to RORT_SHADOW_POLL_WORKERS;
    rebuilt only if the worker count changes. Safe to share: get_admin_client() is a
    thread-safe singleton, pack_registry is read-only post-startup, the KPI queue is
    lock-guarded, and each poll sets its own per-thread user context via _UserContext.
    """
    global _poll_executor, _poll_executor_workers
    if _poll_executor is None or _poll_executor_workers != workers:
        from concurrent.futures import ThreadPoolExecutor
        _poll_executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="shadow-poll")
        _poll_executor_workers = workers
    return _poll_executor


def _resident_loop(stop_evt: threading.Event):
    """The continuous-resident backtest lane (Step C). Per cycle: refresh the slot set,
    then poll each eligible slot (advance its resident engine to now-LAG, write new
    settled trades). dry_run computes + logs but writes nothing."""
    import pack_registry
    npacks = len(pack_registry.scan_and_load_all())
    if not npacks:
        raise SystemExit("[shadow_worker] pack registry empty — refusing to start "
                         "(every engine would return 0 trades)")

    from shadow_manager import ResidentEngineManager
    shard = os.environ.get("RORT_SHADOW_SHARD", "").strip()
    symbols = {s.strip() for s in shard.split(",") if s.strip()} or None
    sid_raw = os.environ.get("RORT_SHADOW_SIDS", "").strip()
    sids = {int(s) for s in sid_raw.split(",") if s.strip().isdigit()} or None
    dry = _dry_run_default()
    mgr = ResidentEngineManager(shard_symbols=symbols, dry_run=dry, shard_sids=sids)
    logger.warning("[shadow_worker] RESIDENT lane active — dry_run=%s shard=%s sids=%s "
                   "packs=%d (dry_run writes NOTHING; arm with RORT_SHADOW_DRY_RUN=0)",
                   dry, symbols or "(all)", f"{len(sids)} ids" if sids else "(none)", npacks)

    # Fix 2b: KPI/Hi-Fi recompute on a separate thread (off the poll path).
    from shadow_manager import KPI_ASYNC
    if KPI_ASYNC:
        threading.Thread(target=_kpi_drain_loop, args=(mgr,), daemon=True,
                         name="shadow-kpi").start()
        logger.warning("[shadow_worker] Fix 2b: async KPI worker thread started")
    # Fix 2a: process slots oldest-polled-first so no slot can starve (default on).
    fair_order = os.environ.get("RORT_SHADOW_FAIR_ORDER", "1").strip().lower() in (
        "1", "true", "yes", "on")
    # Fix 2c: parallelize the per-slot polls across a thread pool so the serial
    # bar_cache read tax is overlapped (default 1 = serial = today's behavior).
    try:
        poll_workers = max(1, int(os.environ.get("RORT_SHADOW_POLL_WORKERS", "1")))
    except (TypeError, ValueError):
        poll_workers = 1
    if poll_workers > 1:
        logger.warning("[shadow_worker] Fix 2c: parallel polls across %d workers",
                       poll_workers)

    def _poll_one(slot):
        """Poll one slot, returning trades inserted. Exceptions are contained so one
        bad slot never aborts the cycle (mirrors the serial path's try/except)."""
        if not _running:
            return 0
        try:
            return mgr.poll(slot).get('inserted', 0) or 0
        except Exception as e:  # noqa: BLE001
            logger.warning("[shadow_worker] sid=%s poll failed: %s", slot.sid, e,
                           exc_info=True)
            return 0

    reload_every = int(os.environ.get("RORT_SHADOW_RELOAD_S", "300"))
    last_discover = 0.0
    while _running:
        now = time.monotonic()
        if now - last_discover > reload_every:
            try:
                slots = mgr.discover()
                elig = sum(1 for s in slots.values() if s.eligible)
                logger.info("[shadow_worker] tracking %d strategies (%d eligible)",
                            len(slots), elig)
            except Exception as e:  # noqa: BLE001
                logger.warning("[shadow_worker] discover failed: %s", e)
            last_discover = now

        wrote = 0
        slots_iter = list(mgr.slots.values())
        if fair_order:
            # oldest last_tick_at first (never-polled = -1 → front). No slot starves.
            slots_iter.sort(
                key=lambda s: s.last_tick_at if s.last_tick_at is not None else -1.0)
        eligible = [s for s in slots_iter if s.eligible]

        if poll_workers > 1 and len(eligible) > 1:
            # Submit in fair-order; barrier-join before the next pass so a slot is
            # never polled by two workers at once and discover() never races with an
            # in-flight poll (both run only between passes on this thread).
            ex = _get_poll_executor(poll_workers)
            futures = [ex.submit(_poll_one, s) for s in eligible]
            for f in futures:
                wrote += f.result()
        else:
            for slot in eligible:
                if not _running:
                    break
                wrote += _poll_one(slot)
        if wrote:
            logger.info("[shadow_worker] cycle %s %d trades",
                        "would-write" if dry else "wrote", wrote)

        for _ in range(POLL_INTERVAL_S):
            if not _running:
                break
            time.sleep(1)


def main():
    if os.environ.get("SHADOW_WORKER_DISABLED", "").strip().lower() in (
            "1", "true", "yes", "on"):
        logger.warning("SHADOW_WORKER_DISABLED set — exiting without starting.")
        return

    # Local dev loads src/.env; Railway injects env vars directly (no .env).
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
    except Exception:  # noqa: BLE001
        pass

    os.environ.setdefault("USE_DB", "true")

    mode = _lane_mode()
    shard = os.environ.get("RORT_SHADOW_SHARD", "").strip() or "(all)"
    logger.info("[shadow_worker] up id=%s lane_mode=%s shard=%s poll=%ss",
                WORKER_ID, mode, shard, POLL_INTERVAL_S)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    stop_evt = threading.Event()
    _start_health_thread(stop_evt)

    if mode != "shadow":
        logger.info("[shadow_worker] lane_mode=%s — backtest lane is owned by the "
                    "%s path; this service stays inert.", mode, mode)
        _idle_loop(stop_evt)
    else:
        # Resident backtest lane (Step C): warm a resident engine per strategy on this
        # symbol shard, poll bar_cache for new SETTLED bars, apply incrementally (never
        # re-window), emit backtest_<model> trades. dry_run (default) computes only.
        # Unsettled-tail provisional emission is a follow-up increment.
        _resident_loop(stop_evt)

    stop_evt.set()
    logger.info("[shadow_worker] stopped")


if __name__ == "__main__":
    main()
