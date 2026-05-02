"""Fire-and-forget writer for the bar_engine_states cache (M8.7 M6).

The worker calls write_state() each time a strategy monitor's bar-close
processing succeeds (not on duplicate rebroadcasts — the original write
covered that bar's state). Writes happen on a small background thread
pool so Supabase latency never blocks the bar-processing path.

Today's scope: write-only, data-capture foundation. No read path / UI.
The point is to start collecting now so we have data to draw from
when/if engine-truth replay becomes a real feature.

Feature flag: BAR_ENGINE_STATE_WRITE_ENABLED (default off).
- Flip on in the Railway worker env to start recording.
- Flip off to stop recording with zero code change.

Schema: see src/migrations/bar_engine_states_table.sql

Mirrors the live_bars_writer.py pattern verbatim — same shape so any
fix to one easily transfers to the other.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from typing import Optional

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    val = os.environ.get("BAR_ENGINE_STATE_WRITE_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


_executor: Optional[ThreadPoolExecutor] = None
_warned_disabled = False


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="bar_engine_state_writer")
    return _executor


def _ensure_iso_utc(ts) -> str:
    if isinstance(ts, str):
        return ts
    if hasattr(ts, "isoformat"):
        if getattr(ts, "tzinfo", None) is None:
            try:
                ts = ts.tz_localize("UTC")
            except (AttributeError, TypeError):
                ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return str(ts)


def _coerce_state(state: dict) -> dict:
    """Coerce indicator state dict to JSON-safe scalars.

    state.current values are typically Dict[str, float|bool] but
    user-pack engines may emit other types. Defensive coercion so
    json.dumps never trips.
    """
    out: dict = {}
    if not state:
        return out
    for k, v in state.items():
        if isinstance(v, bool):
            out[k] = bool(v)
        elif isinstance(v, (int, float)):
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                continue
        elif v is None:
            continue
        else:
            try:
                out[k] = str(v)
            except Exception:
                continue
    return out


def _write_sync(strategy_id: int, bar_start, indicator_state: dict,
                source: str) -> None:
    """Insert/upsert one row into bar_engine_states. Idempotent on PK."""
    try:
        from db import get_admin_client
    except Exception as e:
        logger.warning("bar_engine_states: db import failed: %s", e)
        return

    payload = {
        "strategy_id": int(strategy_id),
        "bar_start": _ensure_iso_utc(bar_start),
        "indicator_state": _coerce_state(indicator_state),
        "source": source,
    }

    try:
        client = get_admin_client()
        client.table("bar_engine_states").upsert(
            payload,
            on_conflict="strategy_id,bar_start",
        ).execute()
    except Exception as e:
        logger.warning(
            "bar_engine_states: write failed sid=%s bar=%s: %s",
            strategy_id, payload.get("bar_start"), e)


def write_state(strategy_id: int, bar_start, indicator_state: dict,
                source: str = "ws") -> None:
    """Submit a state snapshot to the background pool. No-op when disabled."""
    global _warned_disabled
    if not is_enabled():
        if not _warned_disabled:
            logger.info(
                "bar_engine_states: writer disabled "
                "(BAR_ENGINE_STATE_WRITE_ENABLED unset)")
            _warned_disabled = True
        return
    if not strategy_id or bar_start is None:
        return
    try:
        _get_executor().submit(
            _write_sync, strategy_id, bar_start,
            indicator_state or {}, source)
    except Exception as e:
        logger.warning("bar_engine_states: submit failed: %s", e)


def shutdown(wait: bool = True) -> None:
    """Drain the pool on worker shutdown so in-flight writes finish."""
    global _executor
    if _executor is None:
        return
    try:
        _executor.shutdown(wait=wait)
    except Exception as e:
        logger.warning("bar_engine_states: shutdown error: %s", e)
    _executor = None
