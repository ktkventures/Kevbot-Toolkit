"""Gap healer — inserts WS-missed bars into the live engine (2026-06-11).

THE PROBLEM: the Polygon WebSocket genuinely misses bar windows (~14% of
RTH 10-second windows on SPY measured 2026-06-10 — real, high-volume
bars). The REST verifier (rest_verifier.py) can only CORRECT values of
bars already in BarBuilder.history; if WS missed the bar entirely,
apply_rest_correction logs "bar not in history — skipping" and the
engine's incremental indicators run over a sparser series than the
backtest (REST) forever after → indicator-state drift → phantom /
missed / late entries vs backtest.

THE FIX: detect missing bar windows (at the next bar close + a periodic
sweeper backstop), fetch the REAL bars from Polygon REST, insert them
into builder.history in chronological position, recompute indicator
state forward (SymbolHub.apply_rest_insert), and persist the healed bar
to live_bars with source='rest_insert' so faithful "what the engine
consumed" views (ENGINE_CONSUMED_SOURCES) include it.

NEVER synthesizes bars: a window REST has no trades for is genuinely
tradeless and stays absent — same as the backtest's resample dropna.
(Synthetic flat-bar gap-fill was Bug A, removed 0d3affb.)

Mirrors rest_verifier.py's architecture: module-level env flag, small
dedicated thread pool, per-bar state dict with GC, background sweeper,
and the SAME per-symbol mutex (rest_verifier._get_symbol_lock) so heals
serialize against verifier corrections.

Feature flags:
- GAP_HEAL_ENABLED env — default ON ("false"/"0"/"no"/"off" disables;
  no redeploy needed beyond Railway's env-change restart).
- GAP_HEAL_FORCE_DISABLED module constant — one-flip code rollback
  (same pattern as ralph_engine.BAR_GAP_FILL_ENABLED).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

logger = logging.getLogger("gap_healer")

# One-flip code rollback. When True, every entry point is a no-op
# regardless of env. Flip + redeploy to kill the healer if it
# misbehaves in a way an env change can't reach fast enough.
GAP_HEAL_FORCE_DISABLED = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (ValueError, TypeError):
        return default


# Tunables (env-overridable).
GAP_HEAL_LOOKBACK_SEC = _env_int("GAP_HEAL_LOOKBACK_SEC", 1800)
GAP_HEAL_MAX_WINDOWS_PER_DETECT = _env_int(
    "GAP_HEAL_MAX_WINDOWS_PER_DETECT", 30)
GAP_HEAL_MAX_PER_SWEEP = _env_int("GAP_HEAL_MAX_PER_SWEEP", 50)
GAP_HEAL_MAX_WAIT_S = _env_int("GAP_HEAL_MAX_WAIT_S", 30)
# A window REST still has no trades for after this age is genuinely
# tradeless — mark it 'empty' permanently. 900s = Polygon's documented
# worst-case REST delivery delay (see rest_verifier sweeper notes).
GAP_HEAL_EMPTY_FINAL_AGE_S = _env_int("GAP_HEAL_EMPTY_FINAL_AGE_S", 900)
# Retry cap per window before marking 'failed' permanently (protects
# REST quota from a pathological window).
GAP_HEAL_MAX_ATTEMPTS = _env_int("GAP_HEAL_MAX_ATTEMPTS", 10)
# Full-replay budget per sweep cycle: each insert whose snapshot replay
# misses falls back to an O(N) full replay (~710ms under RTH load).
# After this many in one sweep, remaining windows stay pending for the
# next cycle so a backlog never stalls the worker.
GAP_HEAL_FULL_REPLAY_BUDGET = _env_int("GAP_HEAL_FULL_REPLAY_BUDGET", 5)

# Settle grace before first REST poll (mirrors the per-bar verifier
# dispatch constants at ralph_engine.on_bar_close).
_SETTLE_GRACE_SUBMINUTE_S = 5.0
_SETTLE_GRACE_MINUTE_S = 30.0


def is_enabled() -> bool:
    if GAP_HEAL_FORCE_DISABLED:
        return False
    val = os.environ.get("GAP_HEAL_ENABLED", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True  # default ON


# ---------------------------------------------------------------------------
# Callbacks (set once at worker startup via configure()).
# ---------------------------------------------------------------------------
# insert_callback(symbol, tf_seconds, rest_bar_dict, rest_per_second_bars)
#   -> str status: 'inserted' | 'inserted_full' | 'exists' |
#                  'rejected_forming' | 'no_builder' | 'no_hub' | 'failed'
# scan_provider() -> list[(symbol, tf_seconds, [missing_bar_start_dt])]
_insert_callback: Optional[Callable] = None
_scan_provider: Optional[Callable] = None


def configure(*, insert_callback: Optional[Callable] = None,
              scan_provider: Optional[Callable] = None) -> None:
    global _insert_callback, _scan_provider
    _insert_callback = insert_callback
    _scan_provider = scan_provider


# ---------------------------------------------------------------------------
# Per-window heal state (dedup + retry/terminal tracking). Keyed by
# (symbol, tf_seconds, bar_start_iso). Same shape as rest_verifier's
# _bar_verify_state. Statuses:
#   pending     — known gap, REST had no data yet (sweeper retries)
#   in_flight   — a heal run currently owns it
#   healed      — inserted into engine history (terminal)
#   empty       — REST confirms genuinely tradeless (terminal)
#   exists      — bar appeared in history before we inserted (terminal)
#   failed      — attempts exhausted or insert failed permanently (terminal)
# ---------------------------------------------------------------------------
_heal_state: dict[tuple, dict] = {}
_heal_state_lock = threading.Lock()
_TERMINAL = ("healed", "empty", "exists", "failed")

# Aggregate counters for logs / get_stats().
_stats: dict = {
    "queued": 0, "healed": 0, "healed_full": 0, "empty": 0,
    "exists": 0, "failed": 0, "rejected": 0, "rest_pending": 0,
}
_stats_per_tf: dict[tuple, int] = {}  # (symbol, tf) -> healed count
_stats_lock = threading.Lock()


def _state_key(symbol: str, tf_seconds: int, bar_start: datetime) -> tuple:
    return (symbol, tf_seconds, bar_start.isoformat())


def _get_state(key: tuple) -> dict:
    with _heal_state_lock:
        return dict(_heal_state.get(key, {}))


def _set_state(key: tuple, **updates) -> None:
    with _heal_state_lock:
        entry = _heal_state.setdefault(key, {})
        entry.update(updates)


def _bump(stat: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[stat] = _stats.get(stat, 0) + n


def _gc_heal_state(cutoff_age_seconds: int = 3600) -> int:
    """Drop entries whose bar_start is older than cutoff. Terminal
    'empty'/'failed' entries are kept for 4x the cutoff so a sweeper
    rescan doesn't re-fetch a known-tradeless window every cycle."""
    now = datetime.now(timezone.utc)
    cutoff_iso = (now - timedelta(seconds=cutoff_age_seconds)).isoformat()
    long_cutoff_iso = (
        now - timedelta(seconds=cutoff_age_seconds * 4)).isoformat()
    dropped = 0
    with _heal_state_lock:
        for k in list(_heal_state.keys()):
            status = _heal_state[k].get("status")
            limit = (long_cutoff_iso if status in ("empty", "failed")
                     else cutoff_iso)
            if k[2] < limit:
                _heal_state.pop(k, None)
                dropped += 1
    return dropped


def get_stats() -> dict:
    with _stats_lock:
        out = dict(_stats)
        out["per_tf_healed"] = {
            f"{sym}:{tf}s": n for (sym, tf), n in _stats_per_tf.items()}
    with _heal_state_lock:
        out["tracked_windows"] = len(_heal_state)
    return out


# ---------------------------------------------------------------------------
# Executor (own pool — a heal burst must not starve verifier passes).
# ---------------------------------------------------------------------------
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gap_healer")
    return _executor


# ---------------------------------------------------------------------------
# REST fetch — one call per contiguous run of missing windows.
# ---------------------------------------------------------------------------
def _fetch_rest_windows(symbol: str, tf_seconds: int,
                        bar_starts: list) -> Optional[dict]:
    """Fetch REST data covering [min(bar_starts), max(bar_starts)+tf) and
    bucket it per missing window.

    Returns {bar_start_dt: (bar_dict, per_second_list_or_None)} for
    windows that have REAL trades; windows with no trades are absent
    from the dict. Returns None when the REST call itself failed or
    returned nothing for the whole span (caller treats as
    "REST not settled yet" and retries / leaves pending).

    tf < 60  → 1-second aggregates (per-second lists survive so the
               verifier can splice later passes at sub-bar granularity).
    tf >= 60 → 1-minute aggregates rolled up per window (repo
               convention: always resample from 1Min, never native
               coarse bars — see CLAUDE.md).
    """
    try:
        from data_loader import (
            _polygon_fetch_bars, _polygon_bars_to_df, _to_polygon_ticker)
        import pandas as pd

        starts = sorted(bar_starts)
        span_start = starts[0] - timedelta(seconds=2)
        span_end = (starts[-1] + timedelta(seconds=tf_seconds)
                    + timedelta(seconds=2))
        from_ms = str(int(span_start.timestamp() * 1000))
        to_ms = str(int(span_end.timestamp() * 1000))
        poly_ticker = _to_polygon_ticker(symbol)
        timespan = "second" if tf_seconds < 60 else "minute"
        results = _polygon_fetch_bars(
            poly_ticker, 1, timespan, from_ms, to_ms)
        if not results:
            return None
        df = _polygon_bars_to_df(results)
        if df is None or len(df) == 0:
            return None

        out: dict = {}
        for bar_start in starts:
            bs = pd.Timestamp(bar_start)
            if bs.tzinfo is None:
                bs = bs.tz_localize("UTC")
            be = bs + pd.Timedelta(seconds=tf_seconds)
            in_window = df[(df.index >= bs) & (df.index < be)]
            if len(in_window) == 0:
                continue  # genuinely tradeless (or not settled yet)
            rows = []
            for ts, row in in_window.iterrows():
                rows.append({
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                })
            bar_dict = {
                "timestamp": bar_start,
                "open": rows[0]["open"],
                "high": max(r["high"] for r in rows),
                "low": min(r["low"] for r in rows),
                "close": rows[-1]["close"],
                "volume": sum(r["volume"] for r in rows),
            }
            per_sec = rows if tf_seconds < 60 else None
            out[bar_start] = (bar_dict, per_sec)
        return out
    except Exception as e:
        logger.warning(
            "gap_heal: REST fetch failed sym=%s tf=%ss span=[%s..%s]: %s",
            symbol, tf_seconds,
            min(bar_starts).isoformat(), max(bar_starts).isoformat(), e)
        return None


# ---------------------------------------------------------------------------
# Queueing + the heal run.
# ---------------------------------------------------------------------------
def queue_heal(symbol: str, tf_seconds: int, missing_bar_starts: list,
               detected_via: str = "bar_close") -> int:
    """Queue a heal for the given missing bar windows. Dedups against
    in-flight and terminal windows; retries 'pending' ones. Returns the
    number of windows actually queued. No-op when disabled."""
    if not is_enabled() or _insert_callback is None:
        return 0
    if not missing_bar_starts:
        return 0

    now = datetime.now(timezone.utc)
    accepted: list = []
    for bar_start in sorted(missing_bar_starts):
        if len(accepted) >= GAP_HEAL_MAX_WINDOWS_PER_DETECT:
            break
        if bar_start.tzinfo is None:
            bar_start = bar_start.replace(tzinfo=timezone.utc)
        # Never queue inside the lookback=0 zone: the forming bar and
        # the most recent closed period belong to the live WS path
        # (apply_rest_insert re-checks; this just avoids busywork).
        if bar_start > now - timedelta(seconds=2 * tf_seconds):
            continue
        if bar_start < now - timedelta(seconds=GAP_HEAL_LOOKBACK_SEC):
            continue
        key = _state_key(symbol, tf_seconds, bar_start)
        st = _get_state(key)
        if st.get("status") in _TERMINAL or st.get("status") == "in_flight":
            continue
        if st.get("attempts", 0) >= GAP_HEAL_MAX_ATTEMPTS:
            _set_state(key, status="failed")
            _bump("failed")
            continue
        _set_state(key, status="in_flight",
                   attempts=st.get("attempts", 0) + 1,
                   detected_via=detected_via)
        accepted.append(bar_start)

    if not accepted:
        return 0
    _bump("queued", len(accepted))
    try:
        _get_executor().submit(
            _heal_run_sync, symbol, tf_seconds, accepted, detected_via)
    except Exception as e:
        logger.warning("gap_heal: submit failed sym=%s tf=%ss: %s",
                       symbol, tf_seconds, e)
        for bar_start in accepted:
            _set_state(_state_key(symbol, tf_seconds, bar_start),
                       status="pending")
        return 0
    logger.info(
        "gap_heal: DETECTED sym=%s tf=%ss missing=%d window=[%s..%s] via=%s",
        symbol, tf_seconds, len(accepted),
        accepted[0].isoformat(), accepted[-1].isoformat(), detected_via)
    return len(accepted)


def _heal_run_sync(symbol: str, tf_seconds: int, bar_starts: list,
                   detected_via: str) -> None:
    """Background worker: settle-wait, fetch REST, insert oldest-first
    under the shared per-symbol lock."""
    try:
        _heal_run_inner(symbol, tf_seconds, bar_starts, detected_via)
    except Exception as e:
        logger.warning(
            "gap_heal: heal run crashed sym=%s tf=%ss: %s",
            symbol, tf_seconds, e, exc_info=True)
        for bar_start in bar_starts:
            _set_state(_state_key(symbol, tf_seconds, bar_start),
                       status="pending")


def _heal_run_inner(symbol: str, tf_seconds: int, bar_starts: list,
                    detected_via: str) -> None:
    from rest_verifier import _get_symbol_lock

    bar_starts = sorted(bar_starts)
    now = datetime.now(timezone.utc)

    # Settle wait: REST needs a few seconds after a window closes before
    # its aggregates exist. Wait relative to the NEWEST window's end.
    settle = (_SETTLE_GRACE_SUBMINUTE_S if tf_seconds < 60
              else _SETTLE_GRACE_MINUTE_S)
    newest_end = bar_starts[-1] + timedelta(seconds=tf_seconds)
    sleep_for = (newest_end + timedelta(seconds=settle)
                 - now).total_seconds()
    if sleep_for > 0:
        time.sleep(min(sleep_for, settle + 2 * tf_seconds))

    # Poll REST until data appears or the wait budget is spent.
    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=GAP_HEAL_MAX_WAIT_S)
    fetched: Optional[dict] = None
    while True:
        fetched = _fetch_rest_windows(symbol, tf_seconds, bar_starts)
        if fetched:
            break
        if datetime.now(timezone.utc) >= deadline:
            break
        time.sleep(2.0)

    now = datetime.now(timezone.utc)
    if not fetched:
        # Whole span returned nothing. Either REST hasn't settled, or
        # every window is genuinely tradeless. Finalize 'empty' only
        # past the final age; otherwise leave pending for the sweeper.
        n_empty = n_pending = 0
        for bar_start in bar_starts:
            key = _state_key(symbol, tf_seconds, bar_start)
            bar_end = bar_start + timedelta(seconds=tf_seconds)
            age = (now - bar_end).total_seconds()
            if age >= GAP_HEAL_EMPTY_FINAL_AGE_S:
                _set_state(key, status="empty")
                n_empty += 1
            else:
                _set_state(key, status="pending")
                n_pending += 1
        if n_empty:
            _bump("empty", n_empty)
            logger.info(
                "gap_heal: EMPTY_WINDOW sym=%s tf=%ss n=%d (REST confirms "
                "tradeless — staying absent, matches backtest dropna)",
                symbol, tf_seconds, n_empty)
        if n_pending:
            _bump("rest_pending", n_pending)
        return

    sym_lock = _get_symbol_lock(symbol)
    n_healed = n_full = 0
    for bar_start in bar_starts:  # oldest-first — keeps replays cheap
        key = _state_key(symbol, tf_seconds, bar_start)
        entry = fetched.get(bar_start)
        if entry is None:
            # Span had data but this window didn't: tradeless once old
            # enough to be definitive, else pending for retry.
            bar_end = bar_start + timedelta(seconds=tf_seconds)
            age = (now - bar_end).total_seconds()
            if age >= GAP_HEAL_EMPTY_FINAL_AGE_S:
                _set_state(key, status="empty")
                _bump("empty")
            else:
                _set_state(key, status="pending")
                _bump("rest_pending")
            continue
        bar_dict, per_sec = entry
        status = "failed"
        try:
            with sym_lock:
                status = _insert_callback(
                    symbol, tf_seconds, bar_dict,
                    rest_per_second_bars=per_sec)
        except Exception as e:
            logger.warning(
                "gap_heal: insert callback raised sym=%s tf=%ss bar=%s: %s",
                symbol, tf_seconds, bar_start.isoformat(), e)
        if status in ("inserted", "inserted_full"):
            _set_state(key, status="healed")
            _bump("healed")
            n_healed += 1
            if status == "inserted_full":
                _bump("healed_full")
                n_full += 1
            with _stats_lock:
                _stats_per_tf[(symbol, tf_seconds)] = \
                    _stats_per_tf.get((symbol, tf_seconds), 0) + 1
        elif status == "exists":
            _set_state(key, status="exists")
            _bump("exists")
        elif status in ("rejected_forming",):
            # Too close to the live edge — retry next cycle.
            _set_state(key, status="pending")
            _bump("rejected")
        else:  # 'no_builder' / 'no_hub' / 'failed'
            _set_state(key, status="pending")
            _bump("rejected")

    if n_healed:
        logger.info(
            "gap_heal: HEALED sym=%s tf=%ss n=%d (full_replays=%d) "
            "window=[%s..%s] via=%s",
            symbol, tf_seconds, n_healed, n_full,
            bar_starts[0].isoformat(), bar_starts[-1].isoformat(),
            detected_via)


# ---------------------------------------------------------------------------
# Sweeper — periodic backstop for gaps the at-close hook can't see
# (WS silent for a stretch, force_close_stale_bar path, REST-settle
# stragglers left 'pending').
# ---------------------------------------------------------------------------
_sweeper_thread: Optional[threading.Thread] = None
_sweeper_stop = threading.Event()


def sweep_gaps(max_per_sweep: int = None) -> int:
    """One sweeper cycle: ask the engine for current gap candidates and
    queue heals, capped. Returns windows queued."""
    if not is_enabled() or _scan_provider is None:
        return 0
    cap = max_per_sweep or GAP_HEAL_MAX_PER_SWEEP
    queued = 0
    try:
        candidates = _scan_provider() or []
    except Exception as e:
        logger.warning("gap_heal: scan provider failed: %s", e)
        return 0
    for symbol, tf_seconds, starts in candidates:
        if queued >= cap:
            break
        budget = cap - queued
        queued += queue_heal(symbol, tf_seconds, starts[:budget],
                             detected_via="sweep")
    if len(_heal_state) > 2000:
        _gc_heal_state()
    if queued:
        stats = get_stats()
        logger.info(
            "gap_heal: sweep queued=%d totals healed=%d (full=%d) "
            "empty=%d exists=%d failed=%d per_tf=%s",
            queued, stats["healed"], stats["healed_full"], stats["empty"],
            stats["exists"], stats["failed"], stats["per_tf_healed"])
    return queued


def start_sweeper(interval_seconds: int = 120) -> None:
    """Launch the background sweeper. Safe to call once at worker
    startup; no-op when already running or when disabled."""
    global _sweeper_thread
    if not is_enabled():
        logger.info("gap_heal: disabled (GAP_HEAL_ENABLED off or "
                    "FORCE_DISABLED) — sweeper not started")
        return
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_stop.clear()

    def _run():
        logger.info("gap_heal: sweeper started (interval=%ds)",
                    interval_seconds)
        while not _sweeper_stop.wait(interval_seconds):
            try:
                sweep_gaps()
            except Exception as e:
                logger.warning("gap_heal: sweep cycle failed: %s", e)

    _sweeper_thread = threading.Thread(
        target=_run, name="gap_healer_sweeper", daemon=True)
    _sweeper_thread.start()


def stop_sweeper() -> None:
    _sweeper_stop.set()


def shutdown(wait: bool = True) -> None:
    global _executor
    stop_sweeper()
    if _executor is not None:
        try:
            _executor.shutdown(wait=wait)
        except Exception:
            pass
        _executor = None
