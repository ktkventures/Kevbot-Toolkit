"""Process-scoped, per-user TTL memo for hot-path config loaders (M-RS4 Fix 1b).

The continuous backtest engine (`shadow-worker`) polls N strategy slots in a tight
loop; each poll runs `services.prepare_data_with_indicators`, which re-loads per-user
config from the DB every call — `load_confluence_groups()` and `load_general_packs()`.
With 8 same-user SPY slots polled every ~5 s that is ~4.5 confluence reads/sec (~100 ms
each, latency-bound), which starves the compute+write stage (root-caused 2026-07-01,
`Plan_M-RS4_Phase3_Scheduling_Fixes.md` §0-JUL01-PM).

This memo lets all slots of one user share a single config load per TTL window.

Live-lane safety: gated by `RORT_SHADOW_CONFIG_CACHE_TTL_S` (int seconds, default 0 =
OFF = byte-identical to the uncached path). Only the `shadow-worker` process sets the
flag, so the live lane (`api` / `Worker` / `data-worker` — separate processes) never
memoizes and its `updated_at` hot-reload is untouched. Within the shadow process the
staleness bound is the TTL, which the backtest lane tolerates.
"""

import os
import time
import threading

# (namespace, user_id) -> (expires_at_monotonic, value)
_store: dict = {}
_lock = threading.Lock()


def _ttl_seconds() -> float:
    """Cache TTL from env; <= 0 (default) disables the memo entirely."""
    try:
        return float(os.getenv("RORT_SHADOW_CONFIG_CACHE_TTL_S", "0"))
    except (TypeError, ValueError):
        return 0.0


def cached(namespace: str, loader, uid=None):
    """Return `loader()` memoized per `(namespace, user_id)` for TTL seconds.

    `namespace` distinguishes concurrent caches (e.g. 'confluence_groups' vs
    'general_packs'). The key also includes the user id: `uid` when given (the
    admin loaders scope by an explicit `user_id` arg), otherwise the *current*
    context user via `db.get_current_user_id()` (the context-scoped loaders). Either
    way user A never receives user B's config. Use a distinct `namespace` per
    loader return-shape (parsed vs raw) so a cross-hit can't return the wrong type.

    TTL <= 0 -> bypass the cache and call `loader()` directly (byte-identical to
    the pre-Fix-1b path). The returned object is shared read-only, exactly like
    `shadow_manager._gen_packs`'s cached `_enabled_gen` — callers must not mutate it.
    """
    ttl = _ttl_seconds()
    if ttl <= 0:
        return loader()

    if uid is None:
        from db import get_current_user_id
        uid = get_current_user_id()
    key = (namespace, uid)
    now = time.monotonic()

    with _lock:
        hit = _store.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]

    # Load OUTSIDE the lock so a slow DB read for one user doesn't serialize polls
    # for others; a rare duplicate load on a cold same-key race is harmless.
    value = loader()

    with _lock:
        _store[key] = (now + ttl, value)
    return value


def clear() -> None:
    """Drop all memoized config (test hook)."""
    with _lock:
        _store.clear()
