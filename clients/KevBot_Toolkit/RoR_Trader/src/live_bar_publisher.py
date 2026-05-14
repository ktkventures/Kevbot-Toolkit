"""Live-bar publisher for Ralph engine.

Broadcasts forming-bar and bar-close snapshots to Supabase Realtime so the
frontend chart can render a TradingView-style live candle. Runs as a
fire-and-forget side effect off Ralph's tick handler — never blocks, never
raises into the alert path.

Channel convention:
    Topic:   live_bars:{user_id}
    Event:   bar_update
    Payload: {symbol, tf_seconds, bar: {open, high, low, close, volume, timestamp},
              is_forming: bool}

Throttle design (Phase A, 2026-05-14):
    1Hz per-user budget (preserves Supabase load characteristics of 738ab38)
    + round-robin distribution across (symbol, tf) tuples within that budget,
    so each active tuple gets ~1/N of the budget instead of one winning the
    deterministic race every second.

    See `docs/Parity_Plan_2026-05-14.md` Phase A for design context. Rollback
    by env: LIVE_BAR_THROTTLE_DISABLE_ROUNDROBIN=true reverts to pure
    per-user 1Hz throttle without redeploy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("ralph.live_bar")


def _roundrobin_disabled() -> bool:
    val = os.environ.get(
        "LIVE_BAR_THROTTLE_DISABLE_ROUNDROBIN", "").strip().lower()
    return val in ("1", "true", "yes", "on")


class LiveBarPublisher:
    """HTTP-based Supabase Realtime broadcast publisher.

    Uses the Realtime broadcast REST endpoint rather than the async Supabase
    Python client to avoid introducing a second event-loop dependency inside
    Ralph's tick handler. httpx.AsyncClient reuses a connection pool.
    """

    # Window during which a tuple counts as "active" for the
    # round-robin denominator. Tuples that haven't attempted to
    # publish in this window are excluded from the per-user N.
    _ACTIVE_WINDOW_MS = 5_000
    # Stale-entry retention. Keys older than this are dropped on
    # the next periodic prune.
    _STALE_MS = 60_000
    # How often to run the prune scan.
    _PRUNE_INTERVAL_MS = 30_000

    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        throttle_ms: int = 1000,
        request_timeout_s: float = 0.5,
    ):
        self._url = (supabase_url or "").rstrip("/")
        self._key = service_role_key or ""
        self._throttle_ms = throttle_ms
        self._timeout = request_timeout_s
        self._client: Optional[httpx.AsyncClient] = None
        # user_id -> last-successful-publish epoch_ms (enforces 1Hz user cap)
        self._last_published: dict[str, float] = {}
        # (user_id, symbol, tf_seconds) -> last-successful-publish epoch_ms.
        # Round-robin uses this to skip tuples that recently won a slot.
        self._tuple_last_published: dict[tuple, float] = {}
        # (user_id, symbol, tf_seconds) -> last-attempt epoch_ms.
        # Used to count "active" tuples per user (the denominator for the
        # per-tuple gap).
        self._tuple_last_attempt: dict[tuple, float] = {}
        # Last time we pruned stale entries.
        self._last_prune_ms: float = 0.0
        # Cached env flag — re-read on init.
        self._roundrobin_disabled = _roundrobin_disabled()

        self._enabled = bool(self._url and self._key)
        if not self._enabled:
            logger.warning(
                "LiveBarPublisher disabled — SUPABASE_URL or SERVICE_ROLE_KEY missing."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init httpx client so we don't pay the cost until first publish."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "apikey": self._key,
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def _prune_stale(self, now_ms: float) -> None:
        cutoff = now_ms - self._STALE_MS
        # Prune attempt + publish dicts. We DO NOT prune
        # `_last_published[user_id]` itself — a user might publish
        # less than once per minute on a quiet TF and we still want
        # the 1Hz cap to apply.
        stale_attempts = [
            k for k, ts in self._tuple_last_attempt.items() if ts < cutoff]
        for k in stale_attempts:
            self._tuple_last_attempt.pop(k, None)
        stale_pubs = [
            k for k, ts in self._tuple_last_published.items() if ts < cutoff]
        for k in stale_pubs:
            self._tuple_last_published.pop(k, None)

    def _should_throttle(
        self, user_id: str, symbol: str, tf_seconds: int
    ) -> bool:
        """Return True if this update should be dropped.

        Two-tier throttle:
        1. **1Hz user cap** (preserves Supabase load — never changed)
        2. **Per-tuple round-robin** — within the 1Hz user slot, each
           active (symbol, tf) gets ~1/N of the budget by requiring a
           gap of `throttle_ms × N` between consecutive publishes
           for the same tuple.

        First-time tuples bypass the per-tuple gap so a freshly-opened
        chart starts showing data immediately (only the user cap
        applies).

        Rollback: env `LIVE_BAR_THROTTLE_DISABLE_ROUNDROBIN=true`
        falls back to pure per-user 1Hz throttle (the 738ab38 behavior
        that caused starvation).
        """
        now_ms = time.time() * 1000
        key = (user_id, symbol, tf_seconds)
        self._tuple_last_attempt[key] = now_ms

        # Periodic memory hygiene.
        if (now_ms - self._last_prune_ms) > self._PRUNE_INTERVAL_MS:
            self._prune_stale(now_ms)
            self._last_prune_ms = now_ms

        # Tier 1: per-user 1Hz cap.
        last_user = self._last_published.get(user_id)
        if last_user is not None and (now_ms - last_user) < self._throttle_ms:
            return True

        # Rollback path — original per-user behavior.
        if self._roundrobin_disabled:
            self._last_published[user_id] = now_ms
            self._tuple_last_published[key] = now_ms
            return False

        # Tier 2: per-tuple round-robin gap (only applied to tuples
        # that have published before — first-time publishers fall
        # through and are allowed if Tier 1 passed).
        last_tuple = self._tuple_last_published.get(key)
        if last_tuple is not None:
            active_cutoff = now_ms - self._ACTIVE_WINDOW_MS
            # Count distinct active tuples for this user only.
            active_n = sum(
                1 for k, ts in self._tuple_last_attempt.items()
                if k[0] == user_id and ts > active_cutoff
            )
            if active_n < 1:
                active_n = 1
            required_gap_ms = self._throttle_ms * active_n
            if (now_ms - last_tuple) < required_gap_ms:
                return True

        # Allow — record both timestamps.
        self._last_published[user_id] = now_ms
        self._tuple_last_published[key] = now_ms
        return False

    async def publish_async(
        self,
        user_id: str,
        symbol: str,
        tf_seconds: int,
        bar: dict,
        is_forming: bool,
        extras: Optional[dict] = None,
    ) -> None:
        """Publish a bar update. Never raises, never blocks meaningfully.

        Forming-bar updates are silently dropped inside the throttle window.
        Completed bars share the same throttle path; in practice a 1-minute
        close fires once per minute and is rarely in conflict with the 1Hz
        cap, but the throttle does apply.

        `extras` (when provided) is merged into the broadcast payload. Used
        by the Ralph engine to carry tentative indicator values, interpreter
        states, and cross-TF state snapshots so the frontend can animate
        heatmap / oscillator / overlay panes in sync with the price candle.
        """
        if not self._enabled or not user_id:
            return

        if self._should_throttle(user_id, symbol, tf_seconds):
            return

        endpoint = f"{self._url}/realtime/v1/api/broadcast"
        inner_payload: dict = {
            "symbol": symbol,
            "tf_seconds": tf_seconds,
            "bar": bar,
            "is_forming": is_forming,
        }
        if extras:
            inner_payload.update(extras)

        payload = {
            "messages": [
                {
                    "topic": f"live_bars:{user_id}",
                    "event": "bar_update",
                    "payload": inner_payload,
                }
            ]
        }

        try:
            client = await self._get_client()
            resp = await client.post(endpoint, json=payload)
            if resp.status_code >= 300:
                logger.debug(
                    "LiveBar broadcast %d for %s %ds (forming=%s): %s",
                    resp.status_code, symbol, tf_seconds, is_forming, resp.text[:200],
                )
        except Exception as e:
            logger.debug(
                "LiveBar publish failed for %s %ds (forming=%s): %s",
                symbol, tf_seconds, is_forming, e,
            )

    async def aclose(self) -> None:
        """Close the underlying httpx client. Call at shutdown if desired."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None


def make_publisher_from_env() -> LiveBarPublisher:
    """Convenience factory that pulls credentials from db module."""
    from db import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    return LiveBarPublisher(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
