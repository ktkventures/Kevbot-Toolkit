"""DB circuit breaker for the data-worker streaming loop.

`insert_trade_admin` / `update_strategy_admin` already retry individual
transient Supabase errors (`db._execute_with_retry`). This is the
*coarse* layer above that: if write batches keep failing anyway —
Supabase genuinely down — the breaker OPENs and the streaming +
snapshot-flush loops stop issuing DB writes entirely, backing off
exponentially, until a single HALF_OPEN probe succeeds.

It is the explicit guarantee of `docs/Spec_Data_Worker_Service.md` §5.1:
the 2026-05-22 self-DDoS-into-522 outage must be structurally
impossible. The steady single-row trickle is not the bulk-update death
spiral, but the breaker makes "hammer Supabase while it is down"
unreachable regardless.

Usage (the loop checks the gate, then reports the outcome):

    if circuit.allow():
        try:
            ...do DB writes...
            circuit.record_success()
        except Exception:
            circuit.record_failure()
            ...leave snapshot un-advanced; retry next tick...
    else:
        ...skip the write batch this tick...
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("data_worker")

CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"


class DBCircuitBreaker:
    """Three-state breaker — CLOSED / OPEN / HALF_OPEN.

    CLOSED      — normal; calls allowed.
    OPEN        — too many consecutive failures; calls blocked until the
                  cooldown elapses, then one probe is allowed (HALF_OPEN).
    HALF_OPEN   — a single probe call is in flight; success → CLOSED,
                  failure → OPEN again with a doubled cooldown.
    """

    def __init__(self, failure_threshold: int = 5,
                 base_cooldown_s: float = 30.0,
                 max_cooldown_s: float = 300.0):
        self._lock = threading.Lock()
        self._failure_threshold = max(1, failure_threshold)
        self._base_cooldown = base_cooldown_s
        self._max_cooldown = max_cooldown_s
        self._state = CLOSED
        self._consecutive_failures = 0
        self._open_count = 0          # times tripped (instrumentation)
        self._opened_at = 0.0          # time.monotonic() when OPENed
        self._cooldown = base_cooldown_s

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def open_count(self) -> int:
        with self._lock:
            return self._open_count

    def allow(self) -> bool:
        """True if a DB call should be attempted now.

        CLOSED → always. HALF_OPEN → yes (the probe). OPEN → only once
        the cooldown has elapsed, transitioning OPEN→HALF_OPEN so the
        next call is the single probe.
        """
        with self._lock:
            if self._state == CLOSED or self._state == HALF_OPEN:
                return True
            # OPEN — allow a probe only after the cooldown.
            if time.monotonic() - self._opened_at >= self._cooldown:
                self._state = HALF_OPEN
                logger.info("[circuit] cooldown elapsed — HALF_OPEN probe")
                return True
            return False

    def record_success(self) -> None:
        """A DB batch succeeded — reset to CLOSED."""
        with self._lock:
            if self._state != CLOSED:
                logger.info("[circuit] DB write succeeded — CLOSED")
            self._state = CLOSED
            self._consecutive_failures = 0
            self._cooldown = self._base_cooldown

    def record_failure(self) -> None:
        """A DB batch failed — count it; trip OPEN past the threshold."""
        with self._lock:
            self._consecutive_failures += 1
            if self._state == HALF_OPEN:
                # The probe failed — re-OPEN with a longer cooldown.
                self._cooldown = min(self._cooldown * 2, self._max_cooldown)
                self._state = OPEN
                self._opened_at = time.monotonic()
                logger.warning("[circuit] HALF_OPEN probe failed — OPEN "
                               "(cooldown %.0fs)", self._cooldown)
                return
            if (self._state == CLOSED
                    and self._consecutive_failures >= self._failure_threshold):
                self._state = OPEN
                self._open_count += 1
                self._opened_at = time.monotonic()
                self._cooldown = self._base_cooldown
                logger.warning("[circuit] %d consecutive DB failures — OPEN "
                               "(cooldown %.0fs); streaming writes paused",
                               self._consecutive_failures, self._cooldown)

    def snapshot(self) -> dict:
        """State for the metrics log."""
        with self._lock:
            return {
                'state': self._state,
                'consecutive_failures': self._consecutive_failures,
                'open_count': self._open_count,
            }
