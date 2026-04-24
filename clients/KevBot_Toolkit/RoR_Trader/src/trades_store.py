"""Feature-flagged abstraction over the trades storage layer.

Before Phase 40 (2026-04-24): trades lived in a ``strategies.stored_trades``
JSONB column. Every exit rewrote the entire (potentially megabyte-sized)
column, causing statement-timeout cascades.

After Phase 40: trades live in a proper ``trades`` table. Exits do a
single-row INSERT; recomputes do a scoped DELETE + bulk INSERT.

This module guards the cutover with a ``USE_TRADES_TABLE`` env var:

- OFF (default): every call is a thin passthrough to the legacy JSONB
  behaviour. Zero runtime change.
- ON: reads/writes go to the new ``trades`` table. The hydrate helper
  back-fills ``strategy['stored_trades']`` at API response boundaries so
  the frontend DTO shape stays identical.

Flipping the flag back OFF is safe — historical JSONB rows are untouched
by this module. Trades written during an ON window need a backtest
recompute to reseed JSONB (Refresh button).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Short-TTL read cache
# ---------------------------------------------------------------------------
# Strategy Detail page fires 5-6 endpoints in parallel on load (main,
# /trades, /forward-test, /kpis, /chart-data, /trigger-analysis). Each one
# calls get_strategy_by_id_db → _row_to_strategy → hydrates trades. For a
# strategy with 1000+ trades, each hydration paginates Supabase 2+ times.
# Without caching, a single page load = 10-15 paginated queries.
#
# 8-second TTL keyed on (strategy_id, user_id). Long enough to dedupe the
# parallel fan-out of one page load, short enough that Strategy Detail
# showing "live" data after a fresh exit stays current (worker dispatch
# also invalidates the cache on insert/replace below).
_CACHE_TTL_S = 8.0
_cache: dict[tuple, tuple[float, list]] = {}
_cache_lock = threading.Lock()


def _cache_key(strategy_id: int, user_id: Optional[str]) -> tuple:
    return (int(strategy_id), user_id or "")


def _cache_get(key: tuple) -> Optional[list]:
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        ts, trades = hit
        if time.time() - ts > _CACHE_TTL_S:
            _cache.pop(key, None)
            return None
        return trades


def _cache_set(key: tuple, trades: list) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), trades)


def _cache_invalidate(strategy_id: int, user_id: Optional[str] = None) -> None:
    """Invalidate cache entries for a strategy. Called after writes."""
    with _cache_lock:
        # Match any user_id for this strategy (defensive — we usually
        # write with a specific user, but invalidation is cheap).
        sid = int(strategy_id)
        for k in [k for k in _cache if k[0] == sid]:
            _cache.pop(k, None)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def _flag_on() -> bool:
    """Return True when the trades table should be used as the source of truth.

    Per-call env lookup so the flag can be flipped on Railway and picked up
    on the next request without a fresh process. The lookup cost is trivial.
    """
    val = os.environ.get("USE_TRADES_TABLE", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def load_trades_for_strategy(
    strategy_id: int,
    user_id: Optional[str] = None,
) -> list[dict]:
    """Return the trades list for a strategy.

    When the flag is ON, queries the ``trades`` table via
    ``db.load_trades_admin`` and reconstructs the legacy stored_trades
    element shape (flat dict with every field at the top level, using the
    canonical *_fill_ts field names).

    When the flag is OFF, callers should not reach this function — they
    should continue to read ``strategy['stored_trades']`` directly. Kept
    as a safe passthrough (returns []) for any caller that wires through
    us unconditionally.
    """
    if not _flag_on():
        return []
    key = _cache_key(strategy_id, user_id)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        from db import load_trades_admin
        trades = load_trades_admin(strategy_id, user_id)
        _cache_set(key, trades)
        return trades
    except Exception as e:
        logger.warning(
            "trades_store.load_trades_for_strategy failed for strategy %s: %s",
            strategy_id, e,
        )
        return []


def get_stored_trades(strategy: dict) -> list[dict]:
    """Return the effective stored_trades for a strategy dict.

    Flag OFF: returns ``strategy['stored_trades']`` as today.
    Flag ON:  fetches from the trades table.

    Use this helper in services/code paths that need the trades list but
    don't want to care which backend is authoritative.
    """
    if _flag_on():
        sid = strategy.get("id")
        uid = strategy.get("user_id")
        if sid is None:
            return strategy.get("stored_trades", []) or []
        return load_trades_for_strategy(sid, uid)
    return strategy.get("stored_trades", []) or []


def hydrate_strategy_trades(strategy: dict) -> dict:
    """Populate ``strategy['stored_trades']`` from the trades table (flag ON).

    No-op when the flag is OFF. Mutates in place and returns the dict for
    chaining.

    Call this at API response boundaries on endpoints that expose a full
    strategy object (detail, hydrate, refresh, trades endpoint, portfolio
    combined view). Keeps the frontend DTO shape identical regardless of
    storage backend.
    """
    if not _flag_on() or not isinstance(strategy, dict):
        return strategy
    sid = strategy.get("id")
    uid = strategy.get("user_id")
    if sid is None:
        return strategy
    try:
        strategy["stored_trades"] = load_trades_for_strategy(sid, uid)
    except Exception as e:
        logger.warning(
            "trades_store.hydrate failed for strategy %s: %s — leaving "
            "stored_trades as-is", sid, e,
        )
    return strategy


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def insert_trade(
    strategy_id: int,
    user_id: str,
    trade_record: dict,
) -> bool:
    """Append a single trade row to the trades table (flag ON only).

    Returns True on success, False otherwise (or when flag OFF). Caller is
    responsible for also writing to the legacy JSONB path when flag OFF —
    this module does not double-write.
    """
    if not _flag_on():
        return False
    try:
        from db import insert_trade_admin
        insert_trade_admin(strategy_id, user_id, trade_record)
        _cache_invalidate(strategy_id, user_id)
        return True
    except Exception as e:
        logger.error(
            "trades_store.insert_trade failed for strategy %s: %s",
            strategy_id, e,
        )
        return False


def replace_trades_for_strategy(
    strategy_id: int,
    user_id: str,
    trade_records: Iterable[dict],
) -> bool:
    """Atomically replace a strategy's trades (flag ON only).

    DELETEs all current rows for ``strategy_id`` and bulk-INSERTs the new
    list. Used by ``forward_test_service.recompute_and_persist_stored_trades``
    and the Mass Builder backtest-save paths, which need to rewrite the
    full trade history.

    Returns True on success, False on error or when flag OFF. When flag OFF,
    callers should keep writing to ``strategies.stored_trades`` as today.
    """
    if not _flag_on():
        return False
    try:
        from db import replace_trades_admin
        replace_trades_admin(strategy_id, user_id, list(trade_records))
        _cache_invalidate(strategy_id, user_id)
        return True
    except Exception as e:
        logger.error(
            "trades_store.replace_trades failed for strategy %s: %s",
            strategy_id, e,
        )
        return False
