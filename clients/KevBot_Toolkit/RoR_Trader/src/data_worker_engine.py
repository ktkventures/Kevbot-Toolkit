"""Per-strategy streaming backtest engine for the data-worker — Phase 2.

`docs/Spec_Data_Worker_Service.md` §9 Phase 2. Each TSLA strategy ticks
at its bar-close cadence, reads bars from the shared `SymbolBarStore`
(no Polygon re-fetch in the steady path), runs the unified engine
incrementally via snapshot resume, and writes new closed trades to the
`trades` table under `backtest_<model>`.

This is the existing `append_new_backtest_trades_for_strategy` pattern
(forward_test_service.py) made continuous and store-fed:

  - catch-up:  one-time REST-fed `append_new_backtest_trades_for_strategy`
               brings a stale strategy current and persists a snapshot.
  - tick:      store-fed windowed engine run, resume from the snapshot,
               write new closed trades, advance the in-memory snapshot.

WINDOW / LAG CONTRACT
---------------------
The streaming tick processes the window `[snapshot.last_bar_ts,
now - LAG]` where LAG = `_ALGO_HISTORY_LAG_MINUTES` (15 min — the
trade-commit boundary). The snapshot advances only to `now - LAG`, so
the snapshot boundary always equals the commit boundary: the engine
never advances past an un-settled / un-committable trade, and nothing
is dropped at the lag edge. Tier 3 "always start flat"
(`Spec_Tier3_Always_Start_Flat.md`) means each run starts FLAT — an
open position is owned by the window it opened in; the next window
re-derives from the snapshot's restored indicator state.

SCOPE
-----
REST-based backtest models only (`rest_only` / `rest_hifi` / unset) —
the store serves Polygon REST 1-second bars. `cache_locked` /
`cache_corrected` strategies are marked ineligible (their bars come
from the `live_bars` cache, a different source). Hi-Fi Pass 2 timestamp
refinement for `rest_hifi` runs in the debounced KPI loop, not the tick.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("data_worker")

# Trade-commit lag — the window end is `now - LAG`. Mirrors the cron's
# `_ALGO_HISTORY_LAG_MINUTES` (forward_test_service.py) so the streaming
# lane and the legacy append lane agree on what "settled" means.
LAG_MINUTES = int(os.getenv("ALGO_HISTORY_LAG_MINUTES", "15"))

# Backtest models the store (Polygon REST 1s bars) can correctly serve.
REST_BACKTEST_MODELS = {None, "", "rest_only", "rest_hifi"}

# Models whose KPI recompute also runs a Hi-Fi Pass 2 (mirrors the cron).
HIFI_BACKTEST_MODELS = {"rest_hifi", "cache_locked", "cache_corrected"}

_NEW_STRATEGY_GRACE = timedelta(seconds=15)


def _tf_seconds(timeframe: str) -> int:
    """Seconds per bar for a timeframe label. Falls back to 60 (1Min)."""
    try:
        from unified_engine import TIMEFRAME_SECONDS
        return int(TIMEFRAME_SECONDS.get(timeframe, 60)) or 60
    except Exception:
        return 60


class _UserContext:
    """Set the admin user context for the duration of an engine call.

    The engine path (`prepare_data_with_indicators` → `load_confluence_groups`
    / general packs) loads *per-user* config — without a thread-local user
    context it queries `user_id=None` and silently drops the strategy's
    confluence groups. The cron sets this via `set_admin_user_context`
    (forward_test_service.py:1353-1357); the streaming tick must too.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._swap = False

    def __enter__(self):
        from db import set_admin_user_context, get_current_user_id
        if get_current_user_id() != self.user_id:
            set_admin_user_context(self.user_id)
            self._swap = True
        return self

    def __exit__(self, *exc):
        if self._swap:
            from db import clear_current_user
            clear_current_user()
        return False


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

class StreamingMetrics:
    """Thread-safe counters for the streaming loop — logged every cycle."""

    def __init__(self):
        self._lock = threading.Lock()
        self.ticks_total = 0
        self.ticks_errored = 0
        self.ticks_no_new_bar = 0
        self.ticks_store_gap = 0
        self.trades_written_total = 0
        self.trades_written_last_tick = 0
        self.snapshot_flushes_total = 0
        self.snapshot_flush_errors = 0
        self.catchups_run = 0
        self.catchups_skipped_no_baseline = 0
        self.kpi_recomputes_total = 0
        self.circuit_skips = 0
        self._latencies: list = []

    def record_tick(self, errored=False, latency=None, trades=0):
        with self._lock:
            self.ticks_total += 1
            if errored:
                self.ticks_errored += 1
            if latency is not None:
                self._latencies.append(latency)
                if len(self._latencies) > 200:
                    self._latencies.pop(0)
            self.trades_written_total += trades
            self.trades_written_last_tick = trades

    def record_no_new_bar(self):
        with self._lock:
            self.ticks_no_new_bar += 1

    def record_store_gap(self):
        with self._lock:
            self.ticks_store_gap += 1

    def record_flush(self, ok=True):
        with self._lock:
            self.snapshot_flushes_total += 1
            if not ok:
                self.snapshot_flush_errors += 1

    def record_catchup(self, no_baseline=False):
        with self._lock:
            self.catchups_run += 1
            if no_baseline:
                self.catchups_skipped_no_baseline += 1

    def record_kpi_recompute(self):
        with self._lock:
            self.kpi_recomputes_total += 1

    def record_circuit_skip(self):
        with self._lock:
            self.circuit_skips += 1

    @staticmethod
    def _p(values, q):
        if not values:
            return None
        s = sorted(values)
        return round(s[min(len(s) - 1, int(len(s) * q))], 3)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'ticks_total': self.ticks_total,
                'ticks_errored': self.ticks_errored,
                'ticks_no_new_bar': self.ticks_no_new_bar,
                'ticks_store_gap': self.ticks_store_gap,
                'trades_written_total': self.trades_written_total,
                'trades_written_last_tick': self.trades_written_last_tick,
                'tick_latency_p50': self._p(self._latencies, 0.50),
                'tick_latency_p95': self._p(self._latencies, 0.95),
                'snapshot_flushes_total': self.snapshot_flushes_total,
                'snapshot_flush_errors': self.snapshot_flush_errors,
                'catchups_run': self.catchups_run,
                'catchups_skipped_no_baseline': self.catchups_skipped_no_baseline,
                'kpi_recomputes_total': self.kpi_recomputes_total,
                'circuit_skips': self.circuit_skips,
            }


# ─────────────────────────────────────────────────────────────────────
# Per-strategy engine state
# ─────────────────────────────────────────────────────────────────────

class StrategyEngineState:
    """In-memory streaming state for one strategy.

    The engine's cross-tick state lives entirely in `snapshot_b64` (the
    serialized indicator state). It is kept in memory and flushed to
    `config.engine_snapshot_b64` periodically (never per tick).
    """

    def __init__(self, strategy_id: int, user_id: str, strat: dict):
        self.strategy_id = strategy_id
        self.user_id = user_id
        self.strat = strat

        # Derived fields — authoritatively (re)set by classify_strategy(),
        # which is called once after construction and on every reload.
        # Safe defaults here so attribute access never raises.
        self.symbol = strat.get('symbol')
        self.timeframe = strat.get('timeframe', '1Min')
        self.tf_seconds = 60
        self.bt_model = 'rest_hifi'
        self.data_source = 'backtest_rest_hifi'  # also the snapshot model_id
        self.sec_tfs: tuple = ()
        self.fingerprint: Optional[str] = None
        self.streaming_eligible = True
        self.ineligible_reason: Optional[str] = None

        # Live engine state.
        self.snapshot_b64: Optional[str] = None
        self.last_bar_ts = None                # tz-aware Timestamp
        self.last_entry_ts_written: Optional[str] = None
        self.snapshot_dirty = False
        self.snapshot_persisted_at: Optional[datetime] = None
        self.catchup_done = False

        # Scheduling / bookkeeping.
        self.next_due_at = datetime.now(timezone.utc)
        self.last_tick_at: Optional[datetime] = None
        self.has_traded_since_kpi = False
        self._no_baseline_logged = False

    def lag_seconds(self, now: datetime) -> Optional[float]:
        """How far behind real time the snapshot is — the health metric."""
        if self.last_bar_ts is None:
            return None
        import pandas as pd
        return (now - self.last_bar_ts.to_pydatetime()).total_seconds() \
            if hasattr(self.last_bar_ts, 'to_pydatetime') \
            else (now - pd.Timestamp(self.last_bar_ts).to_pydatetime()).total_seconds()


def classify_strategy(state: StrategyEngineState) -> None:
    """(Re)derive all engine fields from `state.strat` and decide
    streaming eligibility. Idempotent — called after construction and on
    every 5-min reload (a config edit changes the fingerprint).

    Ineligible (logged, never ticked):
      - non-REST backtest model — the store serves Polygon REST bars only.
      - a secondary TF >= 1Hour — needs more warmup history than the
        90-min store window holds (services.py:556-561).
    """
    from data_loader import get_required_tfs_from_confluence, get_tf_from_label
    from unified_engine import compute_backtest_fingerprint, TIMEFRAME_SECONDS

    strat = state.strat
    cfg = strat.get('config') if isinstance(strat.get('config'), dict) else {}
    state.symbol = strat.get('symbol')
    state.timeframe = strat.get('timeframe', '1Min')
    state.tf_seconds = _tf_seconds(state.timeframe)
    # Must mirror append_new_backtest_trades_for_strategy
    # (forward_test_service.py:1318) — divergent defaults mean the
    # saved snapshot's `model_id` won't match on resume and the tick
    # snapshot_invalids → re-catchups forever.
    state.bt_model = (cfg.get('backtest_model')
                      or strat.get('backtest_model'))
    state.data_source = f"backtest_{state.bt_model}"
    state.fingerprint = compute_backtest_fingerprint(strat)
    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    state.sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))

    # Reset eligibility — a reload may flip a strategy back to eligible.
    state.streaming_eligible = True
    state.ineligible_reason = None

    if state.bt_model not in REST_BACKTEST_MODELS:
        state.streaming_eligible = False
        state.ineligible_reason = f"non-REST backtest_model={state.bt_model}"
        return

    # Phase 2.5: the Tier-2 coarse-bar store (~150-day 1Min) covers
    # secondary TFs up through 1Day with full warmup. No length-based
    # ineligibility — TIMEFRAME_SECONDS isn't even used here anymore.
    # (Pre-2.5 we marked >=1Hour ineligible because Tier-1's 90-min 1s
    # store couldn't warm them; Tier-2 makes that block obsolete.)
    _ = TIMEFRAME_SECONDS  # imported for any future per-TF guards


# ─────────────────────────────────────────────────────────────────────
# Store-fed engine window — the steady-tick core
# ─────────────────────────────────────────────────────────────────────

def build_injected_frames(store, timeframe: str, sec_tfs: tuple, until_dt: datetime):
    """Resample the shared 1s store to the primary + each secondary TF.

    Returns `(primary_df, {sec_tf: df})` sliced to bars at or before
    `until_dt` (only fully-settled bars feed the engine). `primary_df`
    is None when the store has nothing in range.
    """
    import pandas as pd
    until_ts = pd.Timestamp(until_dt)
    if until_ts.tz is None:
        until_ts = until_ts.tz_localize('UTC')

    primary = store.get_timeframe(timeframe)
    if primary is None or len(primary) == 0:
        return None, {}
    primary = primary[primary.index <= until_ts]
    if len(primary) == 0:
        return None, {}

    sec_dfs: dict = {}
    for stf in sec_tfs:
        s = store.get_timeframe(stf)
        if s is not None and len(s) > 0:
            s = s[s.index <= until_ts]
            if len(s) > 0:
                sec_dfs[stf] = s
    return primary, sec_dfs


def run_store_fed_window(state: StrategyEngineState, store, until_dt: datetime):
    """Run the unified engine over `(snapshot.last_bar_ts, until_dt]`.

    Store-fed equivalent of `services.get_strategy_trades_for_window`'s
    snapshot-resume core (services.py:584-697) — same engine, same
    snapshot primitives, only the bar source differs.

    Returns `(trades_df, new_snapshot_b64, last_bar_ts, status)`.
    status ∈ {ok, no_new_bars, no_bars, snapshot_invalid}.
    """
    import pandas as pd
    from unified_engine import deserialize_backtest_snapshot, run_unified_backtest
    from services import prepare_data_with_indicators, get_secondary_tf_map
    import general_packs as gp_module

    envelope = deserialize_backtest_snapshot(
        state.snapshot_b64,
        expected_fingerprint=state.fingerprint,
        expected_model_id=state.data_source,
    )
    if envelope is None:
        return pd.DataFrame(), state.snapshot_b64, None, 'snapshot_invalid'

    primary_df, sec_dfs = build_injected_frames(
        store, state.timeframe, state.sec_tfs, until_dt)
    if primary_df is None:
        return pd.DataFrame(), state.snapshot_b64, None, 'no_bars'

    df = prepare_data_with_indicators(
        state.symbol,
        timeframe=state.timeframe,
        session=state.strat.get('trading_session', 'RTH'),
        secondary_tfs=state.sec_tfs,
        primary_df=primary_df,
        secondary_tf_dfs=sec_dfs if sec_dfs else None,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame(), state.snapshot_b64, None, 'no_bars'

    # Strip bars at/before the snapshot — already absorbed by a prior
    # run (mirror services.py:642-652).
    last_bar_for_filter = pd.Timestamp(envelope['last_bar_ts'])
    if last_bar_for_filter.tz is None:
        last_bar_for_filter = last_bar_for_filter.tz_localize('UTC')
    if df.index.tz is None:
        last_bar_for_filter = last_bar_for_filter.tz_localize(None)
    df = df[df.index > last_bar_for_filter]
    if len(df) == 0:
        return pd.DataFrame(), state.snapshot_b64, None, 'no_new_bars'

    enabled_gen = gp_module.get_enabled_general_packs(
        gp_module.load_general_packs())
    sec_tf_map = get_secondary_tf_map(df)
    trades, _enriched, captured_b64 = run_unified_backtest(
        df, state.strat,
        general_packs=enabled_gen,
        secondary_tf_map=sec_tf_map if sec_tf_map else None,
        include_open_position=False,
        last_bar_partial=False,
        resume_snapshot=envelope,
        return_snapshot=True,
        snapshot_model_id=state.data_source,
    )
    new_b64 = captured_b64 if captured_b64 is not None else state.snapshot_b64
    last_bar_ts = df.index[-1]
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame()
    return trades, new_b64, last_bar_ts, 'ok'


def _filter_new_closed_trades(trades_df, last_entry_iso: Optional[str]):
    """Closed trades whose entry is strictly newer than the DB anchor.

    Mirrors forward_test_service.py:1406-1448. The whole tick window is
    already past the commit lag, so no separate exit-cutoff filter is
    needed — every closed trade in it is committable.
    """
    import pandas as pd
    if not isinstance(trades_df, pd.DataFrame) or len(trades_df) == 0:
        return pd.DataFrame()
    if 'exit_fill_ts' not in trades_df.columns:
        return pd.DataFrame()
    closed = trades_df[trades_df['exit_fill_ts'].notna()].copy()
    if len(closed) == 0:
        return pd.DataFrame()
    if last_entry_iso:
        last_ts = pd.Timestamp(last_entry_iso)
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize('UTC')
        ent = pd.to_datetime(closed['entry_fill_ts'], utc=True, errors='coerce')
        closed = closed[ent > last_ts]
    return closed


def _write_trades(state: StrategyEngineState, closed_df) -> int:
    """Insert new closed trades. Raises on a transient DB failure so the
    caller's circuit breaker can trip; partial inserts are idempotent via
    the `(strategy_id, entry_fill_ts, exit_fill_ts)` unique index.

    `last_entry_ts_written` advances only after the whole batch succeeds —
    a mid-batch failure leaves it untouched so the next tick re-filters
    from the same anchor and re-attempts every row.
    """
    from api.services.forward_test_service import _serialize_trades
    from db import insert_trade_admin

    records = _serialize_trades(closed_df)
    inserted = 0
    max_entry = state.last_entry_ts_written
    for rec in records:
        rec = dict(rec)
        rec['data_source'] = state.data_source
        row = insert_trade_admin(state.strategy_id, state.user_id, rec)
        if row is not None:
            inserted += 1
        ek = rec.get('entry_fill_ts')
        if ek and (max_entry is None or str(ek) > str(max_entry)):
            max_entry = str(ek)
    state.last_entry_ts_written = max_entry
    return inserted


# ─────────────────────────────────────────────────────────────────────
# The tick
# ─────────────────────────────────────────────────────────────────────

def tick_strategy(state: StrategyEngineState, store, circuit, metrics: StreamingMetrics) -> dict:
    """One streaming tick for a strategy. Cadence-gated, store-fed, O(1)."""
    now = datetime.now(timezone.utc)

    if not state.streaming_eligible:
        return {'status': 'ineligible'}
    if now < state.next_due_at:
        return {'status': 'not_due'}
    if state.snapshot_b64 is None:
        # No baseline snapshot — the streaming loop runs catch-up
        # out-of-band; just reschedule.
        state.next_due_at = now + timedelta(seconds=30)
        return {'status': 'needs_catchup'}

    until_dt = now - timedelta(minutes=LAG_MINUTES)
    tf_delta = timedelta(seconds=state.tf_seconds)

    # No full new committable bar has formed since the snapshot.
    if state.last_bar_ts is not None:
        import pandas as pd
        last_dt = pd.Timestamp(state.last_bar_ts).to_pydatetime()
        if until_dt <= last_dt + tf_delta:
            state.next_due_at = (last_dt + tf_delta
                                 + timedelta(minutes=LAG_MINUTES, seconds=5))
            metrics.record_no_new_bar()
            return {'status': 'no_new_bar'}

    # The store must span back to the resume point.
    cov = store.coverage()
    if cov['count'] == 0 or cov['first_ts'] is None:
        state.next_due_at = now + timedelta(seconds=15)
        metrics.record_store_gap()
        return {'status': 'store_empty'}
    if state.last_bar_ts is not None:
        import pandas as pd
        if pd.Timestamp(cov['first_ts']) > pd.Timestamp(state.last_bar_ts):
            # Store doesn't reach back to the resume point. If the
            # snapshot is hours stale (weekend gap, container restart,
            # extended off-hours), the rolling store will NEVER close
            # the gap on its own — trigger a fresh REST-fed catch-up to
            # re-anchor the snapshot. Threshold 1h is conservative:
            # transient blips don't trip it; weekend gaps (60h+) do.
            snap_dt = pd.Timestamp(state.last_bar_ts).to_pydatetime()
            gap_h = (now - snap_dt).total_seconds() / 3600
            if gap_h > 1.0:
                logger.info("[stream] sid=%s store_gap with %.1fh-old "
                            "snapshot — queueing re-catchup",
                            state.strategy_id, gap_h)
                state.catchup_done = False
                state.next_due_at = now + timedelta(seconds=5)
                return {'status': 'store_gap_recatchup_queued'}
            state.next_due_at = now + timedelta(seconds=15)
            metrics.record_store_gap()
            return {'status': 'store_gap'}

    t0 = time.monotonic()
    try:
        with _UserContext(state.user_id):
            trades_df, new_b64, last_bar_ts, run_status = run_store_fed_window(
                state, store, until_dt)
    except Exception as e:
        metrics.record_tick(errored=True)
        logger.warning("[stream] sid=%s engine run failed: %s",
                        state.strategy_id, e, exc_info=True)
        state.next_due_at = now + timedelta(seconds=60)
        return {'status': 'engine_error', 'error': str(e)}
    latency = time.monotonic() - t0

    if run_status == 'snapshot_invalid':
        # Config changed under us — drop the snapshot, force a re-catchup.
        logger.info("[stream] sid=%s snapshot invalidated — re-catchup queued",
                    state.strategy_id)
        state.snapshot_b64 = None
        state.catchup_done = False
        state.next_due_at = now + timedelta(seconds=10)
        return {'status': 'snapshot_invalid'}

    if run_status in ('no_bars', 'no_new_bars'):
        # Snapshot still valid; just wait for the next bar.
        anchor = (last_bar_ts if last_bar_ts is not None else state.last_bar_ts)
        if anchor is not None:
            import pandas as pd
            state.next_due_at = (pd.Timestamp(anchor).to_pydatetime()
                                 + tf_delta
                                 + timedelta(minutes=LAG_MINUTES, seconds=5))
        else:
            state.next_due_at = now + timedelta(seconds=30)
        metrics.record_no_new_bar()
        return {'status': run_status}

    # run_status == 'ok' — process trades.
    new_closed = _filter_new_closed_trades(trades_df, state.last_entry_ts_written)
    inserted = 0
    write_failed = False
    if len(new_closed) > 0:
        if circuit.allow():
            try:
                inserted = _write_trades(state, new_closed)
                circuit.record_success()
            except Exception as e:
                circuit.record_failure()
                write_failed = True
                logger.warning("[stream] sid=%s trade write failed: %s",
                                state.strategy_id, e)
        else:
            write_failed = True
            metrics.record_circuit_skip()

    if write_failed:
        # Leave the snapshot un-advanced — the window re-runs next tick
        # and re-emits the same trades (idempotent via the unique index).
        state.next_due_at = now + timedelta(seconds=30)
        metrics.record_tick(errored=False, latency=latency, trades=0)
        return {'status': 'write_deferred', 'inserted': 0}

    # Advance the in-memory snapshot to the new commit boundary.
    if new_b64 is not None and new_b64 != state.snapshot_b64:
        state.snapshot_b64 = new_b64
        state.snapshot_dirty = True
    if last_bar_ts is not None:
        state.last_bar_ts = last_bar_ts
    if inserted > 0:
        state.has_traded_since_kpi = True

    import pandas as pd
    anchor = (pd.Timestamp(state.last_bar_ts).to_pydatetime()
              if state.last_bar_ts is not None else until_dt)
    state.next_due_at = anchor + tf_delta + timedelta(minutes=LAG_MINUTES, seconds=5)
    state.last_tick_at = now
    metrics.record_tick(errored=False, latency=latency, trades=inserted)
    return {'status': 'ok', 'inserted': inserted, 'latency_s': round(latency, 3)}


# ─────────────────────────────────────────────────────────────────────
# Catch-up + snapshot hydration / flush
# ─────────────────────────────────────────────────────────────────────

def _hydrate_snapshot_from_db(state: StrategyEngineState) -> None:
    """Read the fresh `engine_snapshot_b64` + trade anchor from the DB.

    Uses a raw config read (the documented forward_test_service.py:1313
    pattern) — `get_strategy_by_id_admin` flattens config so a direct
    `.get('config')` would be None.
    """
    import pandas as pd
    from db import get_admin_client, get_max_entry_ts_admin
    from unified_engine import deserialize_backtest_snapshot

    client = get_admin_client()
    resp = client.table('strategies').select('config') \
        .eq('id', state.strategy_id).eq('user_id', state.user_id) \
        .single().execute()
    cfg = dict((resp.data or {}).get('config') or {}) if resp.data else {}
    state.snapshot_b64 = cfg.get('engine_snapshot_b64')
    if state.snapshot_b64:
        env = deserialize_backtest_snapshot(state.snapshot_b64)
        if env and env.get('last_bar_ts'):
            ts = pd.Timestamp(env['last_bar_ts'])
            state.last_bar_ts = ts.tz_localize('UTC') if ts.tz is None else ts
    state.last_entry_ts_written = get_max_entry_ts_admin(
        state.strategy_id, state.user_id, data_source_filter='backtest_%')
    state.snapshot_dirty = False


def run_startup_catchup(state: StrategyEngineState, metrics: StreamingMetrics) -> dict:
    """One-time REST-fed catch-up — bring a stale strategy current.

    Reuses the proven `append_new_backtest_trades_for_strategy`
    (forward_test_service.py:1256): resumes from the stored snapshot,
    re-fetches bars via Polygon REST, inserts new trades, persists a
    fresh ~current snapshot. A deliberate one-time REST hit, off the
    steady store-fed path.
    """
    from api.services.forward_test_service import append_new_backtest_trades_for_strategy

    state.catchup_done = True
    try:
        result = append_new_backtest_trades_for_strategy(
            state.strategy_id, state.user_id, force=True)
    except Exception as e:
        logger.warning("[stream] sid=%s catch-up failed: %s",
                        state.strategy_id, e, exc_info=True)
        metrics.record_catchup()
        # Retry catch-up on the next pass.
        state.catchup_done = False
        state.next_due_at = datetime.now(timezone.utc) + timedelta(seconds=120)
        return {'status': 'error', 'reason': str(e)}

    status = result.get('status')
    if status == 'skipped':
        # No baseline trades — Phase 4 cold-start scope, not Phase 2.
        state.streaming_eligible = False
        state.ineligible_reason = 'no_baseline (Phase 4 cold-start)'
        metrics.record_catchup(no_baseline=True)
        if not state._no_baseline_logged:
            logger.warning("[stream] sid=%s no baseline — full recompute "
                            "needed first; not streaming", state.strategy_id)
            state._no_baseline_logged = True
        return result

    _hydrate_snapshot_from_db(state)
    metrics.record_catchup()
    logger.info("[stream] sid=%s catch-up done (%s, inserted=%s) — "
                "snapshot last_bar=%s", state.strategy_id, status,
                result.get('inserted', 0), state.last_bar_ts)
    return result


def flush_snapshot(state: StrategyEngineState, circuit, metrics: StreamingMetrics) -> bool:
    """Persist the in-memory snapshot to `config.engine_snapshot_b64`.

    Periodic only (never per tick) — spec §10.4. Uses the
    raw-config-read-then-merge + `_stamp_config` pattern
    (forward_test_service.py:1313-1396): NEVER passes a partial config
    dict, which would wipe the JSONB column.
    """
    if not state.snapshot_dirty or state.snapshot_b64 is None:
        return True
    if not circuit.allow():
        metrics.record_circuit_skip()
        return False
    try:
        from db import get_admin_client
        from api.services.forward_test_service import _stamp_config

        client = get_admin_client()
        resp = client.table('strategies').select('config') \
            .eq('id', state.strategy_id).eq('user_id', state.user_id) \
            .single().execute()
        cfg = dict((resp.data or {}).get('config') or {}) if resp.data else {}
        if not cfg:
            logger.warning("[stream] sid=%s snapshot flush skipped — empty "
                            "config read", state.strategy_id)
            return False
        cfg['engine_snapshot_b64'] = state.snapshot_b64
        cfg['engine_snapshot_at'] = datetime.now(timezone.utc).isoformat()
        _stamp_config(state.strategy_id, state.user_id, cfg)
        state.snapshot_dirty = False
        state.snapshot_persisted_at = datetime.now(timezone.utc)
        circuit.record_success()
        metrics.record_flush(ok=True)
        return True
    except Exception as e:
        circuit.record_failure()
        metrics.record_flush(ok=False)
        logger.warning("[stream] sid=%s snapshot flush failed: %s",
                        state.strategy_id, e)
        return False


# ─────────────────────────────────────────────────────────────────────
# Debounced KPI recompute (+ Hi-Fi pass)
# ─────────────────────────────────────────────────────────────────────

def recompute_kpis_for_strategy(state: StrategyEngineState) -> dict:
    """Recompute KPIs + equity curve from the strategy's backtest trades.

    Replicates forward_test_service.py:1488-1549 (the cron's tail). Kept
    out of the per-tick path so the tick stays O(1); the streaming loop
    runs this on a ~5-min debounce only for strategies that traded.
    """
    import services as svc
    from db import load_trades_kpi_fields_admin, update_strategy_admin
    from api.services.backtest_service import _build_equity_curve

    sid, uid = state.strategy_id, state.user_id
    all_bt = load_trades_kpi_fields_admin(
        sid, uid, data_source_filter='backtest_%') or []
    all_bt_sorted = sorted(all_bt, key=lambda t: str(t.get('entry_fill_ts') or ''))
    merged_df = svc.trades_df_from_stored(all_bt_sorted)

    trading_days = (svc.count_trading_days(merged_df)
                    if hasattr(merged_df, 'index') and len(merged_df) > 0 else 1)
    kpis = svc.calculate_kpis(merged_df, total_trading_days=trading_days)

    eq_data = _build_equity_curve(merged_df)
    boundary_index = None
    fwd = state.strat.get('forward_test_start')
    if fwd:
        try:
            fwd_start = datetime.fromisoformat(fwd)
            bt_portion, _ = svc.split_trades_at_boundary(merged_df, fwd_start)
            boundary_index = len(bt_portion) if len(bt_portion) > 0 else None
        except Exception:
            boundary_index = None
    equity_curve_data = {
        'exit_times': [p.get('timestamp', '') for p in eq_data],
        'cumulative_r': [p.get('cumulative_r', 0) for p in eq_data],
        'boundary_index': boundary_index,
    }
    refreshed_at = datetime.now(timezone.utc).isoformat()
    update_strategy_admin(sid, uid, {
        'kpis': kpis,
        'equity_curve_data': equity_curve_data,
        'data_refreshed_at': refreshed_at,
        'kpis_computed_at': refreshed_at,
        'kpis_stale_since': None,
    })

    # Hi-Fi Pass 2 — refine entry/exit timestamps (mirrors the cron).
    # Needs the user context (it re-runs the per-user engine path).
    hifi = None
    if state.bt_model in HIFI_BACKTEST_MODELS:
        try:
            from api.routers.strategies import run_hifi_pass2
            with _UserContext(uid):
                hifi = run_hifi_pass2(sid, data_source_filter='backtest_%',
                                      user={'id': uid})
        except Exception as e:
            logger.warning("[stream] sid=%s Hi-Fi pass failed: %s", sid, e)

    state.has_traded_since_kpi = False
    return {'status': 'ok', 'trades': len(all_bt_sorted), 'hifi': hifi}
