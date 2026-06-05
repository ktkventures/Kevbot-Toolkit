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

# Per-bar verify schedule (2026-06-05, #57E+F). Each closed bar gets
# verified at these offsets past bar close (not bar start). Each pass
# is skipped early if the prior pass's REST close matched the current
# bar's OHLCV — i.e., the bar has settled and no more drift is expected.
# Polygon REST typically settles within seconds; the longer-tail
# adjustments (odd lots, corrections) can take up to ~15 min.
_BAR_VERIFY_SCHEDULE_S = (None, 60, 300, 900)  # 1st = grace_seconds (TF-aware)

# In-memory per-bar verify state. Keyed by (symbol, tf_seconds, bar_start_iso).
# Used to (a) dedup concurrent passes, (b) detect when the bar has
# settled (REST close stable across passes), (c) decide whether to
# skip a pass because the bar was evicted from BarBuilder history.
# GC'd in _gc_bar_verify_state when entries exceed ~1h age.
_bar_verify_state: dict[tuple, dict] = {}
_bar_verify_state_lock = threading.Lock()


def _bar_state_key(symbol: str, tf_seconds: int, bar_start: datetime) -> tuple:
    return (symbol, tf_seconds, bar_start.isoformat())


def _get_bar_verify_state(key: tuple) -> dict:
    with _bar_verify_state_lock:
        return dict(_bar_verify_state.get(key, {}))


def _set_bar_verify_state(key: tuple, **updates) -> None:
    with _bar_verify_state_lock:
        entry = _bar_verify_state.setdefault(key, {})
        entry.update(updates)


def _gc_bar_verify_state(cutoff_age_seconds: int = 3600) -> int:
    """Drop _bar_verify_state entries whose bar_start is older than
    cutoff_age_seconds. Called opportunistically. Returns count dropped."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=cutoff_age_seconds)
    cutoff_iso = cutoff.isoformat()
    dropped = 0
    with _bar_verify_state_lock:
        keys_to_drop = [k for k in _bar_verify_state if k[2] < cutoff_iso]
        for k in keys_to_drop:
            _bar_verify_state.pop(k, None)
            dropped += 1
    return dropped


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


def queue_verify_for_bar(
    symbol: str,
    tf_seconds: int,
    bar_start: datetime,
    ws_close: float,
    grace_seconds: float,
    pass_num: int = 1,
    max_wait_seconds: float = 60.0,
    correction_threshold: float = 0.01,
) -> None:
    """Bar-scoped verification (no alert_ids — runs on every closed bar).

    Schedules pass `pass_num` of the multi-pass verify cadence
    (T+grace, T+60s, T+5min, T+15min). Each pass:
      1. Sleeps until its scheduled time relative to bar_close.
      2. Fetches REST and compares to the bar's current OHLCV (which
         may have been spliced by an earlier pass).
      3. If drift > correction_threshold AND splice succeeds, marks
         this pass as having corrected.
      4. Updates _bar_verify_state with the REST close from this pass.
      5. If the REST close didn't change since the last pass (settled)
         OR pass_num >= 4, doesn't schedule the next pass.
      6. Otherwise queues pass_num+1 with the next scheduled offset.

    No-op when REST_VERIFY_ENABLED unset. Per-symbol mutex prevents
    concurrent corrections. Never raises.

    Added 2026-06-05 (#57E+F) — closes the WS-bar-drift gap on bars
    that don't trigger alerts. Architecture: see queue_verify (which
    handles alert-triggered verification) — they share splice
    machinery, _verify_sync_core does the work.
    """
    if not is_enabled():
        return
    if not symbol or bar_start is None:
        return
    try:
        _get_executor().submit(
            _verify_bar_sync,
            symbol, tf_seconds, bar_start, ws_close,
            grace_seconds, pass_num,
            max_wait_seconds, correction_threshold,
        )
    except Exception as e:
        logger.warning("queue_verify_for_bar: submit failed: %s", e)


def _verify_bar_sync(
    symbol: str,
    tf_seconds: int,
    bar_start: datetime,
    ws_close: float,
    grace_seconds: float,
    pass_num: int,
    max_wait_seconds: float,
    correction_threshold: float,
) -> None:
    """Background worker for a single pass of per-bar verification.

    Mirrors `_verify_sync`'s flow but without the alert_ids parameter
    (bar-scoped, not alert-scoped) and with multi-pass scheduling +
    settle detection.
    """
    key = _bar_state_key(symbol, tf_seconds, bar_start)
    try:
        bar_close = bar_start + timedelta(seconds=tf_seconds)

        # Pass-specific delay. Pass 1 uses grace_seconds (TF-aware);
        # subsequent passes use _BAR_VERIFY_SCHEDULE_S offsets.
        if pass_num == 1:
            target_delay = grace_seconds
        elif 2 <= pass_num <= len(_BAR_VERIFY_SCHEDULE_S):
            target_delay = _BAR_VERIFY_SCHEDULE_S[pass_num - 1]
        else:
            return  # out of range
        if target_delay is None:
            return

        sleep_for = (
            (bar_close - datetime.now(timezone.utc)).total_seconds()
            + target_delay
        )
        if sleep_for > 0:
            time.sleep(sleep_for)

        # Check if a prior pass settled this bar. If so, skip.
        state = _get_bar_verify_state(key)
        if state.get('settled'):
            return

        # Poll REST for this bar. Aggregates per-second to OHLCV.
        deadline = bar_close + timedelta(seconds=max_wait_seconds + target_delay)
        rest_per_second: Optional[list] = None
        rest_bar: Optional[dict] = None
        while datetime.now(timezone.utc) < deadline:
            rest_per_second = _fetch_rest_bar_window(
                symbol, bar_start, tf_seconds)
            if rest_per_second:
                rest_bar = _aggregate_rest_bar(rest_per_second, bar_start)
                if rest_bar is not None:
                    break
            time.sleep(2.0)

        if rest_bar is None:
            logger.debug(
                "bar_verify: sym=%s bar=%s pass=%d — REST unavailable",
                symbol, bar_start.isoformat(), pass_num)
            # Don't mark settled — let later passes try again. But cap
            # at pass_num=4 by not scheduling further.
            return

        rest_close = float(rest_bar["close"])
        prior_rest_close = state.get('last_rest_close')

        # Settle detection: if this pass's REST matches the prior pass's
        # REST within tolerance, the bar has stopped moving — no more
        # passes needed.
        settled = (
            prior_rest_close is not None
            and abs(rest_close - prior_rest_close) < correction_threshold
        )

        # Compare current bar state in BarBuilder.history (which may
        # have already been spliced) vs latest REST. If drift detected,
        # trigger splice via the existing callback.
        delta = ws_close - rest_close
        correction_applied = False
        if abs(delta) >= correction_threshold and _correction_callback is not None:
            with _get_symbol_lock(symbol):
                try:
                    try:
                        correction_applied = bool(_correction_callback(
                            symbol, tf_seconds, rest_bar,
                            rest_per_second_bars=rest_per_second))
                    except TypeError:
                        correction_applied = bool(_correction_callback(
                            symbol, tf_seconds, rest_bar))
                except Exception as e:
                    logger.warning(
                        "bar_verify: correction callback raised sym=%s "
                        "bar=%s pass=%d: %s",
                        symbol, bar_start.isoformat(), pass_num, e)

        # Persist pass result
        _set_bar_verify_state(
            key,
            pass_num=pass_num,
            last_rest_close=rest_close,
            last_delta=delta,
            last_correction=correction_applied,
            settled=settled,
        )

        if correction_applied:
            logger.info(
                "bar_verify: sym=%s bar=%s pass=%d CORRECTED "
                "ws_close=%s rest_close=%s Δ=%+.4f",
                symbol, bar_start.isoformat(), pass_num,
                ws_close, rest_close, delta)
        elif abs(delta) < correction_threshold:
            logger.debug(
                "bar_verify: sym=%s bar=%s pass=%d VERIFIED Δ=%+.4f",
                symbol, bar_start.isoformat(), pass_num, delta)
        else:
            logger.info(
                "bar_verify: sym=%s bar=%s pass=%d DRIFT_UNCORRECTED Δ=%+.4f "
                "(callback returned False — likely bar evicted from history)",
                symbol, bar_start.isoformat(), pass_num, delta)

        # Schedule next pass if not settled and not at the last pass.
        if not settled and pass_num < len(_BAR_VERIFY_SCHEDULE_S):
            queue_verify_for_bar(
                symbol, tf_seconds, bar_start, ws_close,
                grace_seconds=grace_seconds,
                pass_num=pass_num + 1,
                max_wait_seconds=max_wait_seconds,
                correction_threshold=correction_threshold,
            )

        # Opportunistic GC: every ~100 bars trim old entries.
        if len(_bar_verify_state) > 1000:
            _gc_bar_verify_state()

    except Exception as e:
        logger.warning(
            "bar_verify: _verify_bar_sync crashed sym=%s bar=%s pass=%d: %s",
            symbol, bar_start.isoformat() if bar_start else "?", pass_num, e,
            exc_info=True)


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
        # Phase C (2026-05-28): keep the per-second list around for
        # downstream splice; aggregate to a single OHLCV for the
        # verification comparison.
        deadline = bar_close + timedelta(seconds=max_wait_seconds)
        rest_per_second: Optional[list] = None
        rest_bar: Optional[dict] = None
        while datetime.now(timezone.utc) < deadline:
            rest_per_second = _fetch_rest_bar_window(
                symbol, bar_start, tf_seconds)
            if rest_per_second:
                rest_bar = _aggregate_rest_bar(rest_per_second, bar_start)
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
        # Phase C (2026-05-28): pass per-second list as a kwarg so the
        # engine can splice at sub-bar granularity. Older callbacks that
        # don't accept the kwarg get the old (symbol, tf_seconds, rest_bar)
        # signature via fallback.
        correction_applied = False
        if _correction_callback is not None:
            with _get_symbol_lock(symbol):
                try:
                    try:
                        correction_applied = bool(_correction_callback(
                            symbol, tf_seconds, rest_bar,
                            rest_per_second_bars=rest_per_second))
                    except TypeError:
                        # Callback doesn't accept the new kwarg —
                        # backwards-compatible call.
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


def _fetch_rest_bar_window(
    symbol: str, bar_start: datetime, tf_seconds: int
) -> Optional[list]:
    # Fetch REST per-second aggregates for [bar_start, bar_end) and
    # return them as a list of OHLCV dicts (sorted ascending by ts).
    # Returns None when REST has no data yet for this window (verifier
    # retries). Bypasses data_loader's day-level cache: caches there
    # capture whatever Polygon returned at first-fetch time and never
    # refresh, so long-lived verifier threads keep reading stale
    # snapshots. Caught 2026-05-28 canary.
    try:
        from data_loader import (
            _polygon_fetch_bars, _polygon_bars_to_df, _to_polygon_ticker)
        import pandas as pd
        bar_end = bar_start + timedelta(seconds=tf_seconds)
        padded_start = bar_start - timedelta(seconds=2)
        padded_end = bar_end + timedelta(seconds=2)
        from_ms = str(int(padded_start.timestamp() * 1000))
        to_ms = str(int(padded_end.timestamp() * 1000))
        poly_ticker = _to_polygon_ticker(symbol)
        results = _polygon_fetch_bars(
            poly_ticker, 1, "second", from_ms, to_ms)
        if not results:
            logger.info(
                "rest_verifier: _polygon_fetch_bars returned 0 results "
                "sym=%s poly=%s window_ms=[%s, %s]",
                symbol, poly_ticker, from_ms, to_ms)
            return None
        bars_1s = _polygon_bars_to_df(results)
        if bars_1s is None or len(bars_1s) == 0:
            return None
        bar_start_ts = pd.Timestamp(bar_start).tz_convert("UTC")
        bar_end_ts = pd.Timestamp(bar_end).tz_convert("UTC")
        in_window = bars_1s[
            (bars_1s.index >= bar_start_ts) & (bars_1s.index < bar_end_ts)
        ]
        if len(in_window) == 0:
            return None
        out: list = []
        for ts, row in in_window.iterrows():
            out.append({
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0) or 0),
            })
        return out
    except Exception as e:
        logger.warning(
            "rest_verifier: REST fetch failed sym=%s bar=%s: %s",
            symbol, bar_start.isoformat(), e,
        )
        return None


def _aggregate_rest_bar(per_second: list,
                         bar_start: datetime) -> Optional[dict]:
    # Aggregate a per-second list into a single OHLCV bar dict using
    # the same rules as BarBuilder.accept_second_bar: open=first,
    # high=max, low=min, close=last, volume=sum. Returns None on empty.
    if not per_second:
        return None
    ordered = sorted(per_second,
                     key=lambda b: b.get("timestamp"))
    return {
        "timestamp": bar_start,
        "open": float(ordered[0]["open"]),
        "high": max(float(b["high"]) for b in ordered),
        "low": min(float(b["low"]) for b in ordered),
        "close": float(ordered[-1]["close"]),
        "volume": sum(float(b.get("volume", 0) or 0) for b in ordered),
    }


def _fetch_rest_bar(
    symbol: str, bar_start: datetime, tf_seconds: int
) -> Optional[dict]:
    # Backwards-compatible wrapper that returns just the aggregated
    # OHLCV dict. Kept for any existing callers; the new code path
    # (Phase C, 2026-05-28) prefers _fetch_rest_bar_window +
    # _aggregate_rest_bar so the per-second list survives for splicing.
    per_sec = _fetch_rest_bar_window(symbol, bar_start, tf_seconds)
    if not per_sec:
        return None
    return _aggregate_rest_bar(per_sec, bar_start)


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
            # 2026-05-28: align to tf_seconds bar boundary. For L-type
            # intra-bar alerts (stop_loss, target firing on per-second
            # price action), Ralph stamps bar_time = fill_ts (the trigger
            # second), which is NOT a bar boundary. The verifier must
            # look up the CONTAINING bar, not the intra-bar second. For
            # C-type bar-close alerts, bar_time is already at the
            # boundary so the alignment is a no-op.
            _epoch = int(bar_start_dt.timestamp())
            _aligned_epoch = (_epoch // tf_seconds) * tf_seconds
            bar_start = datetime.fromtimestamp(
                _aligned_epoch, tz=timezone.utc)
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

        # 2026-05-28: gap-fill detection. BarBuilder.accept_bar gap-fills
        # empty 10-sec windows with prev_close and volume=0 (necessary so
        # max_hold_bars and other calendar-bar triggers work consistently).
        # Polygon REST has no aggregate for those windows since no trades
        # happened — so the verifier would always stamp them
        # `rest_unavailable`, conflating "REST is broken" with "no trades
        # in window." Detect at queue time and stamp `gap_fill_unverified`
        # synchronously instead so the EH/AH signal is interpretable.
        # Background: live and backtest diverge structurally on gap-fill
        # bars (backtest skips them entirely via dropna). Documented in
        # Known_Bugs.md; structural fix is a separate decision.
        ws_volume = (snap.get("volume")
                     if isinstance(snap, dict) else None)
        if ws_volume is not None and float(ws_volume) == 0:
            try:
                _update_alerts([int(alert["id"])], {
                    "verification_status": "gap_fill_unverified",
                    "verification_completed_at": datetime.now(
                        timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning(
                    "rest_verifier: gap_fill stamp failed alert=%s: %s",
                    alert.get("id"), e)
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
