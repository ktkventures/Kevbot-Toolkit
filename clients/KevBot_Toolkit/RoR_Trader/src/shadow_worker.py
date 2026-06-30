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
    Used in {button, cron} mode always, and in {shadow} mode until Step C lands."""
    idle_logged = False
    while _running:
        if not idle_logged:
            logger.info("[shadow_worker] inert — idling (no engine manager yet)")
            idle_logged = True
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
        # Step C will: load pack registry (fail loud), warm a resident engine per
        # strategy on this symbol shard, poll bar_cache for new SETTLED bars, apply
        # them incrementally (never re-window), and emit backtest_<model> trades
        # (settled committed, unsettled-tail provisional). Inert until then.
        logger.warning("[shadow_worker] lane_mode=shadow but the resident engine "
                        "manager is not implemented yet (Step C) — staying inert.")
        _idle_loop(stop_evt)

    stop_evt.set()
    logger.info("[shadow_worker] stopped")


if __name__ == "__main__":
    main()
