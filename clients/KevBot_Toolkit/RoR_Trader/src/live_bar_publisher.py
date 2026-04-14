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

Forming-bar updates are throttled per (symbol, tf_seconds) to the publisher's
`throttle_ms` setting (default 250ms). Bar-close updates are never throttled —
a completed bar must always land on the frontend immediately.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger("ralph.live_bar")


class LiveBarPublisher:
    """HTTP-based Supabase Realtime broadcast publisher.

    Uses the Realtime broadcast REST endpoint rather than the async Supabase
    Python client to avoid introducing a second event-loop dependency inside
    Ralph's tick handler. httpx.AsyncClient reuses a connection pool.
    """

    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        throttle_ms: int = 250,
        request_timeout_s: float = 0.5,
    ):
        self._url = (supabase_url or "").rstrip("/")
        self._key = service_role_key or ""
        self._throttle_ms = throttle_ms
        self._timeout = request_timeout_s
        self._client: Optional[httpx.AsyncClient] = None
        # (symbol, tf_seconds) -> last-published epoch ms
        self._last_published: dict[tuple[str, int], float] = {}

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

    def _should_throttle(self, symbol: str, tf_seconds: int) -> bool:
        """Return True if this forming-bar update is within the throttle window."""
        key = (symbol, tf_seconds)
        now_ms = time.time() * 1000
        last = self._last_published.get(key)
        if last is not None and (now_ms - last) < self._throttle_ms:
            return True
        self._last_published[key] = now_ms
        return False

    def _mark_published(self, symbol: str, tf_seconds: int) -> None:
        """Reset the throttle timestamp for a completed bar."""
        self._last_published[(symbol, tf_seconds)] = time.time() * 1000

    async def publish_async(
        self,
        user_id: str,
        symbol: str,
        tf_seconds: int,
        bar: dict,
        is_forming: bool,
    ) -> None:
        """Publish a bar update. Never raises, never blocks meaningfully.

        Forming-bar updates are silently dropped inside the throttle window.
        Completed bars are always sent and reset the throttle timer.
        """
        if not self._enabled or not user_id:
            return

        if is_forming:
            if self._should_throttle(symbol, tf_seconds):
                return
        else:
            self._mark_published(symbol, tf_seconds)

        endpoint = f"{self._url}/realtime/v1/api/broadcast"
        payload = {
            "messages": [
                {
                    "topic": f"live_bars:{user_id}",
                    "event": "bar_update",
                    "payload": {
                        "symbol": symbol,
                        "tf_seconds": tf_seconds,
                        "bar": bar,
                        "is_forming": is_forming,
                    },
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
