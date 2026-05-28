"""REST verifier for the ws_rest_spliced live model.

After Ralph fires an alert on WS-aggregated data, this verifier
asynchronously fetches Polygon REST per-second aggregates for the
same bar and:

1. If WS close matches REST close (within correction_threshold):
   UPDATE alert with verification_status='verified', close_delta=Δ
2. If they differ AND a correction callback is configured AND bar is
   still latest in BarBuilder.history (callback checks this):
   - Pass a corrected bar dict (full OHLCV from REST) to the callback
   - Callback wires through Ralph's existing rebroadcast path →
     apply_last_bar_correction → indicator state convergence
   - UPDATE alert with verification_status='corrected', close_delta=Δ
3. If REST never returns within max_wait_seconds:
   UPDATE alert with verification_status='rest_unavailable'

Feature-flagged via REST_VERIFY_ENABLED env var (default off).

Thread pool pattern mirrors src/live_bars_writer.py (max_workers=2,
fire-and-forget submit, never blocks the bar-processing thread).

Per-symbol mutex serializes the correction path so two concurrent
corrections for the same symbol can't race. The verifier and the
WS-tick thread should never both be calling accept_bar on the same
BarBuilder simultaneously — the rebroadcast path in
ralph_engine.BarBuilder is not designed to be reentrant.

See /home/kevin/.claude/plans/breezy-dreaming-umbrella.md
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    val = os.environ.get("REST_VERIFY_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


_executor: Optional[ThreadPoolExecutor] = None
_warned_disabled = False
_symbol_locks: dict[str, threading.Lock] = {}
_symbol_locks_mutex = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="rest_verifier",
        )
    return _executor


def _get_symbol_lock(symbol: str) -> threading.Lock:
    """Per-symbol mutex for serializing corrections."""
    with _symbol_locks_mutex:
        if symbol not in _symbol_locks:
            _symbol_locks[symbol] = threading.Lock()
        return _symbol_locks[symbol]


# Correction callback invoked when REST differs from WS by ≥ threshold.
# Signature: callback(symbol, tf_seconds, rest_bar_dict) -> bool
#   Returns True if correction was applied (bar still latest in history,
#   accept_bar succeeded). Returns False on no-op (stale bar, or any
#   internal check failed).
# Set at module init via configure(). When None, verifier logs drift but
# never applies corrections — useful for verification-only mode.
_correction_callback: Optional[Callable[[str, int, dict], bool]] = None


def configure(
    *, correction_callback: Optional[Callable[[str, int, dict], bool]] = None
) -> None:
    """Configure module-level callback for corrections.

    Called once at worker startup by Ralph. The callback receives the
    symbol, tf_seconds, and a full OHLCV bar dict constructed from
    REST 1-sec aggregates. The callback is responsible for:

    1. Checking that the bar is still the latest in BarBuilder.history
       for that (symbol, tf_seconds). If not, return False (stale —
       a newer bar has already arrived, no point correcting an old one).
    2. Calling BarBuilder.accept_bar(rest_bar_dict) — triggers the
       existing rebroadcast handling path which sets
       last_was_correction=True and runs recompute_from_history.
    3. Returning True on success, False on no-op.
    """
    global _correction_callback
    _correction_callback = correction_callback


def queue_verify(
    symbol: str,
    tf_seconds: int,
    bar_start: datetime,
    ws_close: float,
    alert_ids: list[int],
    grace_seconds: float,
    max_wait_seconds: float = 60.0,
    correction_threshold: float = 0.01,
) -> None:
    """Submit a REST verification task to the background pool.

    Schedules an async comparison between this WS bar's close and the
    REST settled close. Updates each alert in alert_ids with
    verification_* columns. Optionally triggers indicator state
    correction via the configured callback.

    No-op when REST_VERIFY_ENABLED is unset (saves the resource cost
    until canary).
    """
    global _warned_disabled
    if not is_enabled():
        if not _warned_disabled:
            logger.info(
                "rest_verifier disabled (REST_VERIFY_ENABLED unset)")
            _warned_disabled = True
        return
    if not symbol or not alert_ids:
        return
    try:
        _get_executor().submit(
            _verify_sync,
            symbol, tf_seconds, bar_start, ws_close,
            list(alert_ids), grace_seconds, max_wait_seconds,
            correction_threshold,
        )
    except Exception as e:
        logger.warning("rest_verifier: submit failed: %s", e)


def _verify_sync(
    symbol: str,
    tf_seconds: int,
    bar_start: datetime,
    ws_close: float,
    alert_ids: list[int],
    grace_seconds: float,
    max_wait_seconds: float,
    correction_threshold: float,
) -> None:
    """Background worker: sleep until grace, poll REST, compare, update DB.

    Never raises — all exceptions logged at WARNING and the alert row
    is left untouched (caller can re-queue if needed).
    """
    try:
        bar_close = bar_start + timedelta(seconds=tf_seconds)

        # Sleep until grace_seconds past bar close. Account for time
        # already elapsed since bar_close (the verifier might have been
        # queued late if Ralph was busy).
        sleep_for = (
            (bar_close - datetime.now(timezone.utc)).total_seconds()
            + grace_seconds
        )
        if sleep_for > 0:
            time.sleep(sleep_for)

        # Poll REST every 2s until data appears or max_wait elapses.
        deadline = bar_close + timedelta(seconds=max_wait_seconds)
        rest_bar: Optional[dict] = None
        while datetime.now(timezone.utc) < deadline:
            rest_bar = _fetch_rest_bar(symbol, bar_start, tf_seconds)
            if rest_bar is not None:
                break
            time.sleep(2.0)

        now_iso = datetime.now(timezone.utc).isoformat()

        if rest_bar is None:
            _update_alerts(alert_ids, {
                "verification_status": "rest_unavailable",
                "verification_completed_at": now_iso,
            })
            logger.info(
                "rest_verifier: sym=%s bar=%s — REST unavailable after %.0fs "
                "(ws_close=%s)",
                symbol, bar_start.isoformat(), max_wait_seconds, ws_close,
            )
            return

        rest_close = float(rest_bar["close"])
        delta = ws_close - rest_close

        if abs(delta) < correction_threshold:
            _update_alerts(alert_ids, {
                "verification_status": "verified",
                "verification_close_delta": delta,
                "verification_completed_at": now_iso,
            })
            return

        # Drift exceeds threshold — try correction if callback configured.
        correction_applied = False
        if _correction_callback is not None:
            with _get_symbol_lock(symbol):
                try:
                    correction_applied = bool(_correction_callback(
                        symbol, tf_seconds, rest_bar))
                except Exception as e:
                    logger.warning(
                        "rest_verifier: correction callback raised sym=%s "
                        "bar=%s: %s",
                        symbol, bar_start.isoformat(), e,
                    )

        # Distinct status when drift exceeded threshold but correction
        # didn't land: either no callback was wired, the callback raised,
        # or the bar was no longer the latest in BarBuilder.history (a
        # newer bar arrived before REST settled — common on sub-minute
        # TFs). Marking these "verified" would conflate them with true
        # small-drift verifications. Treat as their own bucket so the
        # canary dashboard can surface the staleness rate honestly.
        status = "corrected" if correction_applied else "drift_uncorrected"
        _update_alerts(alert_ids, {
            "verification_status": status,
            "verification_close_delta": delta,
            "verification_completed_at": now_iso,
        })
        if correction_applied:
            logger.info(
                "rest_verifier: sym=%s bar=%s CORRECTED "
                "ws_close=%s rest_close=%s Δ=%+.4f",
                symbol, bar_start.isoformat(), ws_close, rest_close, delta,
            )
        else:
            logger.info(
                "rest_verifier: sym=%s bar=%s DRIFT_UNCORRECTED "
                "ws_close=%s rest_close=%s Δ=%+.4f "
                "(callback returned False — likely stale bar)",
                symbol, bar_start.isoformat(), ws_close, rest_close, delta,
            )
    except Exception as e:
        logger.warning(
            "rest_verifier: _verify_sync crashed sym=%s bar=%s: %s",
            symbol, bar_start.isoformat() if bar_start else "?", e,
            exc_info=True,
        )


def _fetch_rest_bar(
    symbol: str, bar_start: datetime, tf_seconds: int
) -> Optional[dict]:
    """Fetch REST per-second aggregates for the bar window and aggregate
    them to a single OHLCV bar dict matching BarBuilder's accept_bar
    shape.

    Returns None if no REST data is available yet for this window
    (verifier will retry).
    """
    try:
        from data_loader import fetch_1s_bars_for_window
        bar_end = bar_start + timedelta(seconds=tf_seconds)
        # padding_seconds=2 covers the ~2s Polygon per-sec settle floor
        bars_1s = fetch_1s_bars_for_window(
            symbol, bar_start, bar_end, padding_seconds=2)
        if bars_1s is None or len(bars_1s) == 0:
            return None
        # Filter to bars strictly inside [bar_start, bar_end) — the
        # padding above could have pulled in adjacent seconds.
        import pandas as pd
        bar_start_ts = pd.Timestamp(bar_start).tz_convert("UTC")
        bar_end_ts = pd.Timestamp(bar_end).tz_convert("UTC")
        in_window = bars_1s[
            (bars_1s.index >= bar_start_ts) & (bars_1s.index < bar_end_ts)
        ]
        if len(in_window) == 0:
            return None
        return {
            "timestamp": bar_start,
            "open": float(in_window["open"].iloc[0]),
            "high": float(in_window["high"].max()),
            "low": float(in_window["low"].min()),
            "close": float(in_window["close"].iloc[-1]),
            "volume": float(in_window["volume"].sum()),
        }
    except Exception as e:
        logger.warning(
            "rest_verifier: REST fetch failed sym=%s bar=%s: %s",
            symbol, bar_start.isoformat(), e,
        )
        return None


def _update_alerts(alert_ids: list[int], updates: dict) -> None:
    """Apply a partial UPDATE to multiple alerts.

    Per-row PATCH via the admin client (no bulk UPDATE-by-id-list in
    Supabase REST). For typical "entry+exit pair on same bar" → 2
    UPDATEs. Acceptable cost on the verifier thread (not on the
    bar-processing critical path).
    """
    if not alert_ids:
        return
    try:
        from db import get_admin_client
        client = get_admin_client()
        for aid in alert_ids:
            try:
                client.table("alerts").update(updates).eq("id", aid).execute()
            except Exception as e:
                logger.warning(
                    "rest_verifier: alert UPDATE failed id=%s: %s", aid, e)
    except Exception as e:
        logger.warning(
            "rest_verifier: alert UPDATE setup failed ids=%s: %s",
            alert_ids, e,
        )


def queue_verify_for_alert(
    alert: dict, signal_data: dict, strategy: dict
) -> None:
    """Convenience wrapper called from alert dispatchers right after
    save_alert succeeds.

    Extracts bar identity (symbol, tf_seconds, bar_start) and ws_close
    from the saved alert + signal + strategy. Gates on live_model ==
    'ws_rest_spliced' and a non-null alert id. Silent no-op on any
    error (logs at WARNING).

    The dispatcher caller doesn't need to know any of the verifier's
    internals — just call this and move on.
    """
    if alert.get("live_model") != "ws_rest_spliced":
        return
    if not alert.get("id"):
        return
    try:
        from unified_engine import TIMEFRAME_SECONDS
        from datetime import datetime, timedelta, timezone

        tf_label = strategy.get("timeframe", "1Min")
        tf_seconds = TIMEFRAME_SECONDS.get(tf_label)
        if tf_seconds is None:
            logger.warning(
                "rest_verifier: unknown timeframe %r on sid=%s — skip queue",
                tf_label, strategy.get("id"))
            return

        # bar_start = the bar_start timestamp of the bar whose CLOSE
        # produced the price we want REST to verify. Prefer signal_data
        # 'bar_time' directly — Ralph stamps it from the closed bar's
        # bar_start unconditionally. Deriving from fill_ts is unsafe:
        # for non-C-type triggers (max_hold_bars, time-based exits) the
        # signal_data fill_ts can carry the ENTRY fill_ts even though
        # the alert row's fill_ts is the exit time, which would point
        # the verifier at the wrong bar (caught 2026-05-28 canary —
        # alert 142414 max_hold exit verified against the entry bar
        # 40s upstream, producing a phantom drift).
        bar_time_iso = signal_data.get("bar_time")
        if bar_time_iso:
            bar_start_dt = datetime.fromisoformat(
                str(bar_time_iso).replace("Z", "+00:00"))
            if bar_start_dt.tzinfo is None:
                bar_start_dt = bar_start_dt.replace(tzinfo=timezone.utc)
            bar_start = bar_start_dt
        else:
            # Fallback for callers that don't supply bar_time. Use
            # fill_ts arithmetic; correct for C-type bar-close triggers
            # but unreliable for time-based exits.
            fill_ts_iso = (
                signal_data.get("fill_ts")
                or signal_data.get("entry_fill_ts")
                or signal_data.get("exit_fill_ts")
            )
            if not fill_ts_iso:
                return
            bar_close_dt = datetime.fromisoformat(
                str(fill_ts_iso).replace("Z", "+00:00"))
            if bar_close_dt.tzinfo is None:
                bar_close_dt = bar_close_dt.replace(tzinfo=timezone.utc)
            bar_start = bar_close_dt - timedelta(seconds=tf_seconds)

        # ws_close = engine's view of bar close. The indicator_snapshot
        # carries this verbatim from the WS-aggregated bar Ralph processed.
        # Fall back to signal price for L-type triggers where the
        # snapshot might be absent.
        snap = signal_data.get("indicator_snapshot") or {}
        ws_close = snap.get("close") if isinstance(snap, dict) else None
        if ws_close is None:
            ws_close = signal_data.get("price")
        if ws_close is None:
            return

        # Per-TF grace defaults. Per-strategy override via
        # config.grace_seconds.
        cfg = strategy.get("config") or {}
        grace = cfg.get("grace_seconds")
        if grace is None:
            if tf_seconds >= 60:
                grace = 2.0
            elif tf_seconds >= 30:
                grace = 3.0
            elif tf_seconds >= 10:
                grace = 4.0
            else:
                grace = 5.0

        # Correction threshold + max_wait — defaults from the plan,
        # overridable via config.
        correction_threshold = float(
            cfg.get("correction_threshold_dollars") or 0.01)
        # 2026-05-28 canary on sid 151 observed ~40% rest_unavailable
        # rate at max_wait=60s — Polygon 1-sec aggregate cache has a
        # long settle tail. Direct REST probes 5 min after a "timed
        # out" verifier attempt routinely return clean data, so the
        # bars settle eventually; 60s just isn't long enough to catch
        # the tail. Bumped to 120s here for the initial polling window;
        # the sweeper (sweep_rest_unavailable below) handles anything
        # still stuck after that, covering the documented Polygon
        # worst-case of ~15 min REST delivery delay.
        max_wait = float(cfg.get("rest_max_wait_seconds") or 120.0)

        queue_verify(
            symbol=strategy.get("symbol") or "?",
            tf_seconds=int(tf_seconds),
            bar_start=bar_start,
            ws_close=float(ws_close),
            alert_ids=[int(alert["id"])],
            grace_seconds=float(grace),
            max_wait_seconds=max_wait,
            correction_threshold=correction_threshold,
        )
    except Exception as e:
        logger.warning(
            "rest_verifier: queue_verify_for_alert failed alert=%s sid=%s: %s",
            alert.get("id"), strategy.get("id"), e,
        )


def shutdown(wait: bool = True) -> None:
    """Drain the pool on worker shutdown so in-flight verifications
    finish writing to the DB."""
    global _executor
    if _executor is None:
        return
    try:
        _executor.shutdown(wait=wait)
    except Exception as e:
        logger.warning("rest_verifier: shutdown error: %s", e)
    _executor = None


# ---------------------------------------------------------------------------
# rest_unavailable re-sweep
#
# Polygon's documented worst-case REST delivery delay is ~15 min (rare,
# but real). Even max_wait=120s on the initial verifier pass misses
# bars that fall into the long tail. The sweeper is a periodic
# background pass that picks up alerts stamped 'rest_unavailable' a
# few minutes ago and re-queues a verification task for each — most
# will settle on the second attempt because Polygon has had more time.
#
# Strict scoping to keep this safe:
#   - Only revisits 'rest_unavailable' alerts (never re-stamps
#     verified / corrected / drift_uncorrected).
#   - Looks back a bounded window (default 30 min from fill_ts) so we
#     don't replay history forever — a bar that hasn't settled in
#     30 min is genuinely lost.
#   - Caps per-cycle batch size so a backlog doesn't blow REST quota.
#   - Uses bar_time from the alert row (post-541d78c canonical bar
#     identity) so even pre-fix mis-stamped alerts get the right bar.
# ---------------------------------------------------------------------------

_sweeper_thread: Optional[threading.Thread] = None
_sweeper_stop = threading.Event()


def sweep_rest_unavailable(
    lookback_minutes: int = 30,
    max_alerts: int = 50,
) -> int:
    """Re-queue verification for recent alerts stamped 'rest_unavailable'.

    Returns the number of alerts re-queued. Safe to call from any thread.
    No-op when REST_VERIFY_ENABLED is unset.
    """
    if not is_enabled():
        return 0
    try:
        from db import get_admin_client
        from unified_engine import TIMEFRAME_SECONDS
        client = get_admin_client()
        cutoff_dt = datetime.now(timezone.utc) - timedelta(
            minutes=lookback_minutes)
        cutoff_iso = cutoff_dt.isoformat()
        rows = (client.table("alerts")
                .select("id,symbol,timeframe,bar_time,price")
                .eq("verification_status", "rest_unavailable")
                .gte("fill_ts", cutoff_iso)
                .limit(max_alerts)
                .execute()).data or []
        n_queued = 0
        for a in rows:
            try:
                tf_s = TIMEFRAME_SECONDS.get(a.get("timeframe"))
                bar_time_iso = a.get("bar_time")
                if not tf_s or not bar_time_iso:
                    continue
                bt = datetime.fromisoformat(
                    str(bar_time_iso).replace("Z", "+00:00"))
                if bt.tzinfo is None:
                    bt = bt.replace(tzinfo=timezone.utc)
                queue_verify(
                    symbol=a.get("symbol") or "?",
                    tf_seconds=int(tf_s),
                    bar_start=bt,
                    ws_close=float(a.get("price") or 0),
                    alert_ids=[int(a["id"])],
                    # grace=0 — we're already past the original grace
                    # window, no point sleeping again. Re-poll
                    # immediately and use a longer max_wait for the
                    # second attempt.
                    grace_seconds=0.0,
                    max_wait_seconds=180.0,
                )
                n_queued += 1
            except Exception as e:
                logger.warning(
                    "sweep_rest_unavailable: skip id=%s: %s",
                    a.get("id"), e)
        if n_queued > 0:
            logger.info(
                "sweep_rest_unavailable: re-queued %d alert(s) "
                "(cutoff=%s, lookback=%dmin)",
                n_queued, cutoff_iso[:19], lookback_minutes)
        return n_queued
    except Exception as e:
        logger.warning("sweep_rest_unavailable failed: %s", e)
        return 0


def start_sweeper(interval_seconds: int = 300) -> None:
    """Launch a background thread that runs sweep_rest_unavailable on a
    cadence. Safe to call once at worker startup. Subsequent calls are
    no-ops while a sweeper is already running.
    """
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return

    def _run():
        # Stagger first sweep so it doesn't fire concurrently with the
        # worker's own warmup load.
        if _sweeper_stop.wait(interval_seconds):
            return
        while not _sweeper_stop.is_set():
            try:
                sweep_rest_unavailable()
            except Exception as e:
                logger.warning("sweeper loop iter raised: %s", e)
            if _sweeper_stop.wait(interval_seconds):
                return

    _sweeper_stop.clear()
    _sweeper_thread = threading.Thread(
        target=_run, name="rest_verifier_sweeper", daemon=True)
    _sweeper_thread.start()
    logger.info(
        "rest_verifier: sweeper started (interval=%ds)", interval_seconds)


def stop_sweeper() -> None:
    """Signal the sweeper to exit on its next wake. Idempotent."""
    _sweeper_stop.set()
