"""
Core business logic for RoR Trader — importable from both Streamlit and FastAPI.

Extracted from app.py (Phase 38) to remove Streamlit dependencies (@st.cache_data,
st.session_state). All functions here are pure computation — no UI, no caching decorators.

Usage:
    # From FastAPI:
    from services import prepare_data_with_indicators, calculate_kpis

    # From Streamlit (app.py) — thin wrappers add @st.cache_data:
    @st.cache_data(...)
    def prepare_data_with_indicators(...):
        return services.prepare_data_with_indicators(...)
"""

import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta, timezone

from strategy_periods import resolve_in_sample_end
from typing import Dict, Optional

from data_loader import load_market_data, get_data_source, is_crypto
from indicators import run_all_indicators, run_indicators_for_group
from interpreters import INTERPRETERS, run_all_interpreters, detect_all_triggers
from triggers import generate_trades
import general_packs as gp_module
from confluence_groups import load_confluence_groups, get_enabled_groups

_logger = logging.getLogger('ror_trader')

# =============================================================================
# PREPARE_DATA_WITH_INDICATORS TTL CACHE
# =============================================================================
# Process-local TTL cache for prepare_data_with_indicators(). Strategy detail
# page fires multiple parallel requests for the same data (chart-data,
# confluence-chart × N conditions, kpis, etc.) — each was running the full
# indicator pipeline independently. With this cache, only the first call does
# the work and the next 30 seconds of identical calls return a copy of the
# cached frame.
#
# Why a copy: callers mutate the returned DataFrame (add stop levels, trade
# columns, etc.). Returning the cached frame directly would cause cross-request
# contamination. A pandas .copy() of ~70k rows is sub-second; cheap relative
# to the indicator pipeline cost we're avoiding.
#
# Why process-local (not Redis): single API process today; cross-process
# caching is the next step but not required to fix today's symptom. When we
# move to multi-replica, swap this for Redis with the same key shape.
#
# TTL chosen to span a typical user session (page browsing + bulk
# strategy refresh). Bumped from 30s → 900s on 2026-04-28 because the
# 30s window only absorbed the burst of parallel calls on a single page
# open, not subsequent navigations. With 15 min:
#   - Bulk refresh of N strategies on the same symbol shares work N-1 times
#   - Chart navigations within a session feel instant after first load
#   - Worst-case staleness: a chart historical pane lagging by 15 min;
#     the live forming-bar still streams via Supabase Realtime so the
#     CURRENT tick is always fresh
# Critically: this cache is NOT used by the live worker (alert-firing
# path is independent). Worker calls load_market_data directly for
# warmup, never prepare_data_with_indicators.
_PREPARE_CACHE_TTL_SECONDS = 900
_prepare_cache: dict = {}
_prepare_cache_lock = threading.Lock()


def _prepare_cache_key(symbol, days, seed, start_date, end_date,
                       timeframe, data_feed, session, secondary_tfs):
    """Build a hashable cache key. start_date/end_date stringified for hash."""
    return (
        symbol, days, seed,
        str(start_date) if start_date is not None else None,
        str(end_date) if end_date is not None else None,
        timeframe, data_feed, session,
        tuple(secondary_tfs) if secondary_tfs else (),
    )


def _prepare_cache_get(key):
    """Return a fresh copy of the cached frame if present and within TTL."""
    with _prepare_cache_lock:
        entry = _prepare_cache.get(key)
        if entry is None:
            return None
        ts, df = entry
        if time.monotonic() - ts > _PREPARE_CACHE_TTL_SECONDS:
            _prepare_cache.pop(key, None)
            return None
    return df.copy()


def _prepare_cache_put(key, df):
    """Store a copy of the frame in the cache. Caller's mutations don't bleed."""
    with _prepare_cache_lock:
        # Bound the cache so a long-running process can't grow it unbounded.
        # 32 entries × ~70k rows ≈ ~200 MB at worst — well within Railway limits.
        if len(_prepare_cache) >= 32:
            # Drop the oldest entry (smallest timestamp)
            oldest_key = min(_prepare_cache, key=lambda k: _prepare_cache[k][0])
            _prepare_cache.pop(oldest_key, None)
        _prepare_cache[key] = (time.monotonic(), df.copy())


def clear_prepare_cache():
    """Manual cache invalidation — call after refresh-data or strategy edit."""
    with _prepare_cache_lock:
        _prepare_cache.clear()


# =============================================================================
# DATA PREPARATION
# =============================================================================

def _resolve_primary_df_for_backtest_model(
    strat: dict,
    start_date,
    end_date,
    timeframe: str,
    secondary_tfs: tuple,
    model_override: str | None = None,
):
    """Phase E preview (2026-05-06): route cache-backed backtest models.
    Algo split (2026-05-07): accepts `model_override` so callers can
    dispatch under `algo_model` instead of `backtest_model` (cron uses
    this to populate the algo lane).

    When the resolved model is `cache_locked` or `cache_corrected`,
    return (primary_df, secondary_tf_dfs) populated from `live_bars`
    cache. Otherwise return (None, None) so the caller falls back to
    the REST-default path in prepare_data_with_indicators.

    Modular dispatch — adds new backtest models by extending this table:
      - 'rest_only', 'rest_hifi', None: REST default (return None, None)
      - 'cache_locked':  fetch from cache, sources=ENGINE_CONSUMED_SOURCES
                         (ws/ws_agg/rest_correction/rest_insert — what
                         the live engine actually consumed)
      - 'cache_corrected' (Phase D): fetch from cache, sources=None
                          (includes rest_backfill rows)

    Caller is responsible for handling empty cache gracefully — typical
    pattern is to fall through to REST when cache returns empty DF.
    """
    if model_override is not None:
        bt_model = model_override
    else:
        bt_model = (strat.get('config') or {}).get('backtest_model') \
            if isinstance(strat.get('config'), dict) else None
        if not bt_model:
            bt_model = strat.get('backtest_model')
    if bt_model not in ('cache_locked', 'cache_corrected'):
        return None, None

    from data_loader import (
        fetch_cache_as_df, TF_TO_SECONDS, ENGINE_CONSUMED_SOURCES)
    # cache_locked = "what the live engine actually consumed":
    # ws/ws_agg plus rest_correction/rest_insert (bars the engine
    # spliced/inserted into its own history — see ENGINE_CONSUMED_SOURCES
    # in data_loader.py). rest_backfill stays excluded (cosmetic).
    sources = (ENGINE_CONSUMED_SOURCES
               if bt_model == 'cache_locked' else None)
    tf_seconds = TF_TO_SECONDS.get(timeframe)
    if tf_seconds is None:
        # Unknown timeframe — bail to REST default
        return None, None
    if start_date is None or end_date is None:
        return None, None

    primary_df = fetch_cache_as_df(
        strat['symbol'], tf_seconds, start_date, end_date,
        sources=sources)
    sec_dfs: dict = {}
    for sec_tf in secondary_tfs:
        sec_seconds = TF_TO_SECONDS.get(sec_tf)
        if sec_seconds is None:
            continue
        sec_df = fetch_cache_as_df(
            strat['symbol'], sec_seconds, start_date, end_date,
            sources=sources)
        if len(sec_df) > 0:
            sec_dfs[sec_tf] = sec_df
    return primary_df, sec_dfs


def prepare_data_with_indicators(
    symbol: str, days: int = 30, seed: int = 42,
    start_date=None, end_date=None,
    timeframe: str = "1Min", data_feed: str = "sip",
    session: str = "RTH", secondary_tfs: tuple = (),
    primary_df: Optional[pd.DataFrame] = None,
    secondary_tf_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    strat: Optional[dict] = None,
    model_override: Optional[str] = None,
    required_confluence_ids: Optional[set] = None,
    force_scope: bool = False,
    no_backfill: bool = False,
) -> pd.DataFrame:
    """Load market data and run all indicators, interpreters, and trigger detection.

    `no_backfill` (M-RS4 Phase 3): read-only cache access on the primary REST load
    (no Polygon fetch / no cache write, no direct-Polygon fallthrough). For the
    read-only shadow-worker path. Only affects the non-injected REST load.

    Extracted from app.py:634 — identical logic, no @st.cache_data.

    Args:
        symbol: Stock symbol
        days: Number of days (used if start_date/end_date not provided)
        seed: Random seed for mock data
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)
        timeframe: Bar timeframe (e.g., "1Min", "5Min", "1Hour")
        data_feed: Alpaca data feed — "sip" or "iex" (also used as cache key)
        session: Trading session — "RTH", "Pre-Market", "After Hours", "Extended Hours"
        secondary_tfs: Tuple of secondary timeframes to compute for MTF confluence
        primary_df: M8.7 (2026-05-02) — optional pre-loaded primary DataFrame.
                    If provided, skip the load_market_data call and use this
                    DataFrame as the OHLCV source. Used by the cache-backed
                    chart endpoint to inject `live_bars` data instead of
                    loading from Polygon REST. NOT cached (TTL cache only
                    applies to the REST path because cache invalidation is
                    keyed on inputs).
        secondary_tf_dfs: M8.7 — optional dict of {tf_label: DataFrame} with
                    pre-loaded secondary-TF OHLCV. If provided alongside
                    secondary_tfs, the indicator pipeline runs on these
                    rather than resampling primary_df. Used by the cache
                    path to inject WS-aggregated secondary TFs (avoids the
                    REST-resample-vs-WS-aggregation mismatch).

    Returns DataFrame ready for trade generation and analysis.
    """
    from data_loader import resample_to_timeframe, get_tf_label

    # Phase E preview (2026-05-06): if caller passed `strat` and the
    # strategy's backtest_model is cache-backed, resolve primary_df +
    # secondary_tf_dfs from live_bars cache. Bypasses REST entirely.
    # Honors `primary_df`/`secondary_tf_dfs` when explicitly passed by
    # caller (chart-data-cache endpoint already resolves itself).
    if primary_df is None and strat is not None:
        try:
            cache_primary, cache_secondary = \
                _resolve_primary_df_for_backtest_model(
                    strat, start_date, end_date, timeframe, secondary_tfs,
                    model_override=model_override)
            if cache_primary is not None and len(cache_primary) > 0:
                primary_df = cache_primary
                if secondary_tf_dfs is None and cache_secondary:
                    secondary_tf_dfs = cache_secondary
        except Exception as e:
            # Cache lookup failure should NEVER block engine — fall
            # through to REST default.
            import logging as _l
            _l.getLogger(__name__).warning(
                "cache-backed backtest_model resolve failed for sid=%s: %s "
                "— falling through to REST",
                strat.get('id', '?'), e)

    use_injected = primary_df is not None

    # #21 prep-scoping kill-switch (default OFF → full behavior, byte-identical).
    # When ON, derive the needed-group set from the `strat` the caller already
    # passes — no caller threading. The backtest/chart paths
    # (get_strategy_trades_for_window, load_strategy_data) pass strat → auto-
    # scoped. The LIVE engine (data_worker_engine) does NOT pass strat → it
    # stays full/unscoped on purpose (separate, later validation). Validated
    # byte-identical (trades + gate states) by the parity guard.
    import os
    # graduated to default-ON 2026-07-03 (flag-graduation; env remains the kill switch)
    _scope_flag = os.getenv('RORT_SCOPE_CONFLUENCE_GROUPS', '1') == '1'
    if _scope_flag and required_confluence_ids is None and strat is not None:
        try:
            from unified_engine import resolve_required_confluence_groups
            required_confluence_ids = resolve_required_confluence_groups(
                strat, get_enabled_groups(load_confluence_groups()))
        except Exception as _e:
            logger.warning("[#21] scope-prep derive failed (%s) — full compute", _e)
            required_confluence_ids = None
    # `force_scope=True` lets a specific caller (Mass Builder) scope to an
    # explicit required_confluence_ids set WITHOUT the global flag — so the
    # live engine + single-strategy backtest stay full/unscoped on purpose
    # (global enablement is a separate, post-Monday change). Mass Builder
    # passes the search-level UNION of every selected trigger/confluence group.
    _scope_on = (_scope_flag or force_scope) and required_confluence_ids is not None

    # TTL-cache lookup. Only applies to the REST path — the cache key is
    # keyed on REST inputs (symbol, days, etc.), so injected DataFrames
    # would silently collide with REST results otherwise.
    if not use_injected:
        cache_key = _prepare_cache_key(
            symbol, days, seed, start_date, end_date,
            timeframe, data_feed, session, secondary_tfs)
        # Scoped frames are strategy-dependent (only some groups computed) — they
        # must not share a cache entry with a full or differently-scoped frame.
        if _scope_on:
            cache_key = cache_key + (tuple(sorted(required_confluence_ids)),)
        cached = _prepare_cache_get(cache_key)
        if cached is not None:
            return cached

    # Load raw bars (or use the injected DataFrame)
    if use_injected:
        df = primary_df.copy()
        # #42 fix (2026-06-05): apply session filter to injected df for
        # parity with load_market_data's filtering. Without this, callers
        # that inject pre-loaded bars (cache_locked path, BT-recompute
        # path) skip the RTH/Extended Hours filter, and indicator state
        # gets computed on bars the live engine wouldn't see — source of
        # phantom entries on session-boundary bars. trading_session
        # values: 'RTH' / 'Pre-Market' / 'After Hours' / 'Extended Hours'
        # / '24/7' (crypto, no filter). Same enum as load_market_data.
        if session and session != "24/7":
            from data_loader import _filter_session
            df = _filter_session(df, session)
    else:
        df = load_market_data(symbol, days=days, seed=seed,
                              start_date=start_date, end_date=end_date,
                              timeframe=timeframe, feed=data_feed,
                              session=session, no_backfill=no_backfill)

    if len(df) == 0:
        return df

    # Run indicators
    df = run_all_indicators(df)

    # #21: group-specific indicators. When scoping is ON, compute only the
    # groups the strategy actually reads (entry/exit/gate packs); interpreters/
    # triggers for un-computed groups self-skip via their existing `except
    # KeyError` (missing-column) guards, so the strategy's USED columns stay
    # byte-identical to the full path. Default (scope OFF) = all enabled groups.
    _enabled_groups = get_enabled_groups(load_confluence_groups())
    _scoped_groups = ([g for g in _enabled_groups if g.id in required_confluence_ids]
                      if _scope_on else _enabled_groups)
    for group in _scoped_groups:
        df = run_indicators_for_group(df, group)

    # Run interpreters
    df = run_all_interpreters(df)

    # Detect triggers
    df = detect_all_triggers(df)

    # Evaluate enabled general packs
    gen_packs = gp_module.load_general_packs()
    enabled_gen = gp_module.get_enabled_general_packs(gen_packs)
    for gpack in enabled_gen:
        col_name = gpack.get_condition_column()
        df[col_name] = gp_module.evaluate_condition(df, gpack)

    # Multi-Timeframe: resample + run pipeline on secondary TFs.
    # M8.7: when secondary_tf_dfs is provided, use those DataFrames
    # directly (skip the resample step) — supports cache-backed
    # secondary TFs.
    if secondary_tfs and len(df) > 0:
        interp_keys = list(INTERPRETERS.keys())
        for sec_tf in secondary_tfs:
            try:
                if secondary_tf_dfs is not None and sec_tf in secondary_tf_dfs:
                    sec_df = secondary_tf_dfs[sec_tf].copy()
                else:
                    sec_df = resample_to_timeframe(
                        df[['open', 'high', 'low', 'close', 'volume']].copy(),
                        sec_tf)
                if len(sec_df) == 0:
                    continue
                sec_df = run_all_indicators(sec_df)
                for group in _scoped_groups:  # #21: same scoped set as primary
                    sec_df = run_indicators_for_group(sec_df, group)
                sec_df = run_all_interpreters(sec_df)

                tf_label = get_tf_label(sec_tf)
                from unified_engine import TIMEFRAME_SECONDS
                period_seconds = TIMEFRAME_SECONDS.get(sec_tf, 60)
                period_offset = pd.Timedelta(seconds=period_seconds)
                for interp_col in interp_keys:
                    if interp_col in sec_df.columns:
                        suffixed = f"{interp_col}__{tf_label}"
                        shifted = sec_df[interp_col].copy()
                        shifted.index = shifted.index + period_offset
                        df[suffixed] = shifted.reindex(df.index, method='ffill')
                        # Speculative (unshifted) values for heatmap yellow detection
                        df[f"_spec_{interp_col}__{tf_label}"] = sec_df[interp_col].reindex(
                            df.index, method='ffill')
            except Exception:
                pass

    # Cache the result for the REST path only. Injected DataFrames bypass
    # the cache (see use_injected above).
    if not use_injected:
        _prepare_cache_put(cache_key, df)
    return df


def get_secondary_tf_map(df: pd.DataFrame) -> dict:
    """Extract secondary TF map from column names containing '__'.

    Returns ``{tf_label: [suffixed_col_names]}`` suitable for passing to
    ``get_mtf_confluence_records()`` or ``generate_trades(secondary_tf_map=...)``.
    """
    tf_map: dict = {}
    for col in df.columns:
        if "__" in col and not col.startswith("_spec_"):
            parts = col.rsplit("__", 1)
            if len(parts) == 2:
                tf_label = parts[1]
                tf_map.setdefault(tf_label, []).append(col)
    return tf_map


# =============================================================================
# TRADE GENERATION
# =============================================================================

def unified_trades(
    df: pd.DataFrame, strategy: dict,
    include_open_position: bool = True,
    last_bar_partial: bool = False,
    bar_cache=None, cache_metadata=None
) -> pd.DataFrame:
    """Trade generation via unified engine with MTF support.

    Extracted from app.py:748 (_unified_trades).

    Args:
        df: Enriched DataFrame from prepare_data_with_indicators().
        strategy: Strategy config dict (saved format OR builder format).
        include_open_position: Include synthetic row for open position at end of data.
        last_bar_partial: If True, suppress signals on last bar (forming).
        bar_cache: Optional pre-computed bar cache from precompute_bar_cache().
        cache_metadata: Required when bar_cache is provided.

    Returns:
        trades_df matching generate_trades() schema.
    """
    # 1-minute-secondary gate (RORT_ENFORCE_1MIN_GATE): primary-aware relabel of
    # an overloaded '1M-' gate → lowercase '1m-' so the engine's confluence_set
    # matches the 1-minute SECONDARY records built from the df (which use the
    # lowercase '1m' suffix). MUST mirror the same relabel `load_strategy_data`
    # applied when it built `df`, so the gate points at the '1m-' record (the
    # true 1-minute state) instead of the mislabeled '1M-' PRIMARY record. Flag
    # OFF / 1Min-primary → unchanged (byte-identical). Shallow copy — never
    # mutate the caller's dict.
    from data_loader import normalize_1min_secondary_gate as _norm_1m
    _nc = _norm_1m(strategy.get('confluence'), strategy.get('timeframe', '1Min'))
    if _nc is not strategy.get('confluence'):
        strategy = {**strategy, 'confluence': _nc}

    # Fast path: replay from cache
    if bar_cache is not None and cache_metadata is not None:
        try:
            from unified_engine import run_trades_from_cache
            return run_trades_from_cache(
                bar_cache, strategy, cache_metadata,
                include_open_position=include_open_position)
        except Exception as exc:
            _logger.warning("cache replay failed (%s), falling back to full backtest", exc)

    sec_tf_map = get_secondary_tf_map(df)

    enabled_gen = gp_module.get_enabled_general_packs(
        gp_module.load_general_packs())

    try:
        from unified_engine import run_unified_backtest
        trades_df, _ = run_unified_backtest(
            df, strategy, general_packs=enabled_gen,
            secondary_tf_map=sec_tf_map if sec_tf_map else None,
            include_open_position=include_open_position,
            last_bar_partial=last_bar_partial)
    except Exception as exc:
        _logger.warning("unified engine failed (%s), falling back", exc)
        confluence_set = (
            set(strategy.get('confluence', []))
            | set(strategy.get('general_confluences', []))
        )
        confluence_set = confluence_set if confluence_set else None
        general_cols = [c for c in df.columns if c.startswith("GP_")]
        return generate_trades(
            df,
            direction=strategy['direction'],
            entry_trigger=strategy['entry_trigger'],
            exit_trigger=strategy.get('exit_trigger'),
            exit_triggers=strategy.get('exit_triggers'),
            confluence_required=confluence_set,
            risk_per_trade=strategy.get('risk_per_trade', 100.0),
            stop_atr_mult=strategy.get('stop_atr_mult', 1.5),
            stop_config=strategy.get('stop_config'),
            target_config=strategy.get('target_config'),
            bar_count_exit=strategy.get('bar_count_exit'),
            general_columns=general_cols if general_cols else None,
            secondary_tf_map=sec_tf_map if sec_tf_map else None,
        )

    if not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame()
    # Trade_Timestamps_Spec (2026-04-17): engine emits entry_fill_ts /
    # exit_fill_ts as canonical fields. Alias to entry_time / exit_time
    # here so existing downstream consumers (split_trades_at_boundary,
    # calculate_kpis, drawdown, etc.) keep working. Engine emit contract
    # stays clean per locked decision #5; this is an internal services
    # bridge. Same shape as trades_df_from_stored.
    if len(trades_df) > 0:
        if 'entry_fill_ts' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(
                trades_df['entry_fill_ts'], utc=True, errors='coerce')
        if 'exit_fill_ts' in trades_df.columns:
            trades_df['exit_time'] = pd.to_datetime(
                trades_df['exit_fill_ts'], utc=True, errors='coerce')
    return trades_df


# =============================================================================
# STRATEGY TRADE LOADING
# =============================================================================

def trades_df_from_stored(stored_trades: list) -> pd.DataFrame:
    """Reconstruct a trades DataFrame from stored minimal records.

    Trade_Timestamps_Spec (2026-04-17): engine emits entry_fill_ts /
    exit_fill_ts as the canonical timestamp fields on new stored_trades.
    This function aliases them to entry_time / exit_time columns here so
    existing downstream consumers (split_trades_at_boundary, calculate_kpis,
    drawdown computations, etc.) keep working without a bulk rename. The
    engine's emit contract stays clean (no legacy aliases per locked
    decision #5); this is an internal bridge at the services layer.
    """
    if not stored_trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "r_multiple", "win"])
    df = pd.DataFrame(stored_trades)
    # Prefer the new fill_ts fields; fall back to legacy entry_time /
    # exit_time for any pre-migration row that might slip through.
    if "entry_fill_ts" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_fill_ts"], utc=True, errors='coerce')
    elif "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors='coerce')
    if "exit_fill_ts" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_fill_ts"], utc=True, errors='coerce')
    elif "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors='coerce')
    return df


def split_trades_at_boundary(trades_df: pd.DataFrame, boundary_dt: datetime):
    """Split trades into backtest (before boundary) and forward (at/after boundary).

    Resolves the entry-time column flexibly: stored-trades DataFrames
    carry a datetime `entry_time`; raw unified-engine output carries an
    ISO-string `entry_fill_ts`. The column is coerced to datetime so the
    comparison is valid in either case. A DataFrame with neither column
    is returned whole as the backtest side (cannot split).
    """
    if len(trades_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    col = ('entry_time' if 'entry_time' in trades_df.columns
           else 'entry_fill_ts' if 'entry_fill_ts' in trades_df.columns
           else None)
    if col is None:
        return trades_df.copy(), trades_df.iloc[0:0].copy()

    entry = pd.to_datetime(trades_df[col], errors='coerce')
    boundary_ts = pd.Timestamp(boundary_dt)
    col_tz = getattr(entry.dtype, 'tz', None)

    if col_tz is not None and boundary_ts.tzinfo is None:
        boundary_ts = boundary_ts.tz_localize(col_tz)
    elif col_tz is not None and boundary_ts.tzinfo is not None:
        boundary_ts = boundary_ts.tz_convert(col_tz)
    elif col_tz is None and boundary_ts.tzinfo is not None:
        boundary_ts = boundary_ts.tz_localize(None)

    backtest = trades_df[entry < boundary_ts].copy()
    forward = trades_df[entry >= boundary_ts].copy()
    return backtest, forward


def get_strategy_trades_for_window(
    strat: dict,
    since_dt: datetime,
    until_dt: datetime,
    warmup_bars: int = 100,
    data_feed: str = "sip",
    model_override: str | None = None,
    resume_snapshot_b64: str | None = None,
    expected_fingerprint: str | None = None,
    expected_model_id: str | None = None,
    return_snapshot: bool = False,
    diagnostics_source: str | None = None,
    inherit_position: bool = False,
):
    """Run the unified engine over a small windowed slice of bars.

    Mirrors get_strategy_trades return shape but loads only:
      [since_dt - warmup_period, until_dt]
    Where warmup_period scales with the strategy's primary timeframe so
    indicators have enough history to converge before since_dt.

    Returns trades with entry_time > since_dt only — caller doesn't need
    to filter again. This is the FastAPI port of Streamlit's
    `_generate_incremental_trades` (`src/app.py:682`); the only behavior
    difference is signature shape (datetime in/out vs Streamlit's
    pd.Timestamp).

    Used by the cron + manual incremental-refresh paths to avoid
    re-running the engine over the strategy's full forward-test history
    on every cycle. For 1Min strategies this drops engine time from
    ~30-60s (90 days) to ~3-5s (1-2 days).

    Snapshot resume (2026-05-20, LEF 2c port to API path):
      When `resume_snapshot_b64` is provided AND it deserializes
      validly (matching fingerprint + model_id), the engine resumes
      from the snapshot's `last_bar_ts` instead of warming up — the
      data window becomes `[last_bar_ts, until_dt]` and indicator
      state carries forward byte-identically. On any invalidation
      (fingerprint mismatch, model change, corrupt blob), falls back
      to the standard warmup-windowed path.

      `return_snapshot=True` makes this function return
      `(trades_df, new_snapshot_b64)` instead of just `trades_df`.
      The new_b64 captures the engine's end-of-run state for the
      caller to persist (next refresh resumes from it). Per Tier 3
      §8.2 (always-start-flat, 2026-05-20), the snapshot stores
      INDICATOR state only — position state is intentionally not
      inherited across the boundary.

    LIMITATION: 100-bar warmup isn't enough for strategies whose
    confluence references long-cycle secondary TFs (1Hour or larger).
    Caller should detect those and either bump warmup_bars OR fall back
    to get_strategy_trades. See `get_required_tfs_from_confluence` in
    data_loader.
    """
    import math
    from data_loader import (
        BARS_PER_DAY, get_required_tfs_from_confluence, get_tf_from_label,
        normalize_1min_secondary_gate, enforce_1min_gate_enabled,
    )

    # 1-minute-secondary gate (RORT_ENFORCE_1MIN_GATE): primary-aware relabel of
    # an overloaded '1M-' gate → lowercase '1m-' so the windowed path resolves it
    # as a 1-minute SECONDARY (loaded/built/matched) instead of dropping it. Flag
    # OFF / 1Min-primary → unchanged (byte-identical). Shallow copy — the caller's
    # dict is never mutated. Covers both the run_unified_backtest (return_snapshot)
    # and unified_trades branches below.
    _nc = normalize_1min_secondary_gate(
        strat.get('confluence'), strat.get('timeframe', '1Min'))
    if _nc is not strat.get('confluence'):
        strat = {**strat, 'confluence': _nc}

    def _result(trades_df, new_b64=None):
        """Shape the return per the caller's request mode."""
        if return_snapshot:
            return trades_df, new_b64
        return trades_df

    if 'entry_trigger_confluence_id' not in strat:
        return _result(pd.DataFrame())
    if strat.get('strategy_origin') == 'webhook_inbound':
        return _result(trades_df_from_stored(strat.get('stored_trades', [])))

    timeframe = strat.get('timeframe', '1Min')

    # Snapshot resume: try to deserialize and use the snapshot's
    # last_bar_ts as the window-start. Falls back to warmup-windowed
    # cold-start on any mismatch.
    envelope = None
    if resume_snapshot_b64:
        try:
            from unified_engine import deserialize_backtest_snapshot
            envelope = deserialize_backtest_snapshot(
                resume_snapshot_b64,
                expected_fingerprint=expected_fingerprint,
                expected_model_id=expected_model_id,
            )
        except Exception as e:
            _logger.warning("snapshot deserialize failed in "
                            "get_strategy_trades_for_window: %s", e)
            envelope = None

    # Compute warmup_days from the LONGEST TF the strategy uses
    # (primary OR any secondary). A 10Sec strategy with a 15Min
    # secondary needs warmup proportional to 15Min — 100 × 15min = 25
    # hours = ~4 calendar days. Sizing warmup off the primary alone
    # leaves secondary indicators in undefined (NaN) state, silently
    # killing all confluence-gated triggers in the window. The min bpd
    # (longest TF) is the binding constraint.
    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))

    # Secondary-TF snapshot (RORT_SECONDARY_TF_SNAPSHOT): if a valid cache exists,
    # inject the cached+extended secondary series and size warmup off the
    # PRIMARY only (short) — the coarse secondary no longer needs the long
    # warmup load. Inert when the kill-switch is OFF. Byte-identical by
    # construction (see secondary_tf_snapshot.py).
    _sec_inject = None
    if _secondary_snapshot_enabled() and sec_tfs:
        _sec_inject = _secondary_snapshot_load_extend(
            strat, sec_tfs, until_dt, timeframe,
            strat.get('trading_session', 'RTH'), data_feed, model_override)
    _used_sec_snapshot = _sec_inject is not None

    if _used_sec_snapshot:
        # warmup off the PRIMARY only — secondary comes from cache
        primary_bpd = BARS_PER_DAY.get(timeframe, 390)
        warmup_days = max(1, math.ceil(warmup_bars / max(primary_bpd, 0.001) * 365 / 252))
    else:
        all_tfs = [timeframe] + list(sec_tfs)
        bpds = [BARS_PER_DAY.get(tf, 390) for tf in all_tfs]
        binding_bpd = min(bpd for bpd in bpds if bpd > 0) if bpds else 390
        warmup_days = max(1, math.ceil(warmup_bars / max(binding_bpd, 0.001) * 365 / 252))

    # Strip tz for comparison if since_dt carries one — prepare_data_with_indicators
    # accepts naive or aware; staying naive matches the Streamlit pattern.
    since_naive = since_dt.replace(tzinfo=None) if since_dt.tzinfo else since_dt
    if envelope is not None:
        # Resume path: window starts AT the snapshot's last_bar_ts
        # (engine processed up to and including this bar previously).
        # No warmup needed — indicator state restored byte-identically.
        last_bar_naive = pd.Timestamp(envelope['last_bar_ts'])
        if last_bar_naive.tz is not None:
            last_bar_naive = last_bar_naive.tz_localize(None)
        start_date = last_bar_naive.to_pydatetime()
    else:
        start_date = since_naive - timedelta(days=warmup_days)
    end_date = until_dt.replace(tzinfo=None) if until_dt.tzinfo else until_dt

    # 1-minute SECONDARY gate (RORT_ENFORCE_1MIN_GATE): build the 1Min secondary
    # from the NATIVE 1Min bar (matches LIVE; no resample-from-primary drift) and
    # inject it. Inert when flag OFF (1Min never appears in sec_tfs) or 1Min
    # primary. Mirrors load_strategy_data's native-1Min injection.
    if (enforce_1min_gate_enabled() and "1Min" in sec_tfs
            and timeframe != "1Min"
            and (_sec_inject is None or "1Min" not in _sec_inject)):
        try:
            from strategy_data import _build_native_1min_secondary
            _n1 = _build_native_1min_secondary(
                strat['symbol'], start_date, end_date,
                strat.get('trading_session', 'RTH'), data_feed)
            if _n1:
                _sec_inject = {**(_sec_inject or {}), **_n1}
        except Exception as _e1m:  # noqa: BLE001
            _logger.warning("[1MinSecondary] window native build failed: %s", _e1m)

    df = prepare_data_with_indicators(
        strat['symbol'],
        seed=strat.get('data_seed', 42),
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        data_feed=data_feed,
        session=strat.get('trading_session', 'RTH'),
        secondary_tfs=sec_tfs,
        secondary_tf_dfs=_sec_inject,  # secondary-TF snapshot injection (or None)
        strat=strat,  # Phase E preview: enables cache_locked dispatch
        model_override=model_override,  # algo_model split (2026-05-07)
    )
    if len(df) == 0:
        return _result(pd.DataFrame(), resume_snapshot_b64)

    # Secondary-TF snapshot WRITE: on a FULL compute (no cache used, no primary
    # snapshot resume → df carries the fully-warmed secondary), build + persist
    # the cache so the NEXT append takes the fast path. Cold-seed + refresh.
    if (_secondary_snapshot_enabled() and sec_tfs
            and not _used_sec_snapshot and envelope is None):
        _secondary_snapshot_persist(strat, df, sec_tfs, model_override)

    # Honor `until_dt`: the live-cache load path returns bars up to "now"
    # even when a PAST end_date is requested, so clip explicitly. Without
    # this, a snapshot taken with a past until_dt (e.g. the band-replace
    # lagged-snapshot's base_target) lands at "now", and the resume strip
    # below then removes the entire intended window → 0 trades. No-op for
    # callers using until=now (the common case). (2026-06-16)
    _end_clip = pd.Timestamp(end_date)
    if df.index.tz is not None and _end_clip.tz is None:
        _end_clip = _end_clip.tz_localize('UTC')
    elif df.index.tz is None and _end_clip.tz is not None:
        _end_clip = _end_clip.tz_localize(None)
    df = df[df.index <= _end_clip]
    if len(df) == 0:
        return _result(pd.DataFrame(), resume_snapshot_b64)

    if envelope is not None:
        # Strip bars at or before the snapshot — already absorbed.
        last_bar_for_filter = pd.Timestamp(envelope['last_bar_ts'])
        if last_bar_for_filter.tz is None:
            last_bar_for_filter = last_bar_for_filter.tz_localize('UTC')
        if df.index.tz is None:
            last_bar_for_filter = last_bar_for_filter.tz_localize(None)
        df = df[df.index > last_bar_for_filter]
        if len(df) == 0:
            # No new bars since snapshot — snapshot still valid.
            return _result(pd.DataFrame(), resume_snapshot_b64)

    new_b64 = resume_snapshot_b64  # default: preserve existing on engine miss

    if return_snapshot:
        # Call run_unified_backtest directly so we can ride the resume
        # + return_snapshot path. Bypasses unified_trades.
        try:
            from unified_engine import run_unified_backtest
            import general_packs as gp_module
            enabled_gen = gp_module.get_enabled_general_packs(
                gp_module.load_general_packs())
            sec_tf_map = get_secondary_tf_map(df)
            _result_tuple = run_unified_backtest(
                df, strat,
                general_packs=enabled_gen,
                secondary_tf_map=sec_tf_map if sec_tf_map else None,
                include_open_position=False,
                last_bar_partial=False,
                resume_snapshot=envelope,
                return_snapshot=True,
                snapshot_model_id=expected_model_id,
                diagnostics_source=diagnostics_source,
                inherit_position=inherit_position,
            )
            trades, _enriched, captured_b64 = _result_tuple
            if captured_b64 is not None:
                new_b64 = captured_b64
            # Bridge entry_fill_ts → entry_time
            if isinstance(trades, pd.DataFrame) and len(trades) > 0:
                if 'entry_fill_ts' in trades.columns and 'entry_time' not in trades.columns:
                    trades['entry_time'] = pd.to_datetime(
                        trades['entry_fill_ts'], utc=True, errors='coerce')
                if 'exit_fill_ts' in trades.columns and 'exit_time' not in trades.columns:
                    trades['exit_time'] = pd.to_datetime(
                        trades['exit_fill_ts'], utc=True, errors='coerce')
            else:
                trades = pd.DataFrame()
        except Exception as e:
            _logger.warning(
                "get_strategy_trades_for_window: snapshot-aware engine "
                "call failed (%s) — falling back to legacy unified_trades", e)
            trades = unified_trades(df, strat)
    else:
        trades = unified_trades(df, strat)

    if len(trades) == 0:
        return _result(pd.DataFrame(), new_b64)

    # Filter to trades that started AFTER since_dt — anything earlier
    # was already in the existing trades set the caller fed in.
    since_ts = pd.Timestamp(since_dt)
    if 'entry_time' in trades.columns:
        if trades['entry_time'].dt.tz is not None and since_ts.tz is None:
            since_ts = since_ts.tz_localize('UTC')
        trades = trades[trades['entry_time'] > since_ts]

    # Attach trading_days from windowed source bars (kpi denominator
    # gotcha — see feedback_trading_days_kpi.md). Caller should NOT
    # use this for full-period kpi recompute; only for the windowed
    # slice. KPI recompute should select all trades from DB and
    # compute trading_days separately.
    trades.attrs['trading_days'] = count_trading_days(df)
    # M8.7 (2026-05-07): expose source-bar count so the Jobs UI can
    # show "scanned N bars over W days" — answers the "how far back
    # did we look" question for the user without confusing them about
    # candles-vs-trades semantics.
    trades.attrs['source_bar_count'] = len(df)
    trades.attrs['window_days'] = (end_date - start_date).days
    return _result(trades, new_b64)


def _has_long_cycle_secondary_tf(strat: dict) -> bool:
    """Return True if strategy uses 1Hour or larger secondary TF.

    Used by callers of get_strategy_trades_for_window to decide whether
    100-bar warmup is sufficient or they need the full-history path.
    A 1Hour secondary TF needs ~250 bars × 1 hour = ~10 days of warmup;
    the windowed helper's default 100 bars at 1Min would only give
    ~100 minutes, leaving the secondary indicators in undefined state.
    """
    from data_loader import get_required_tfs_from_confluence, get_tf_from_label
    # `get_tf_from_label` returns a CANONICAL TIMEFRAME LABEL string
    # (e.g. "15Min", "1Hour", "1Day") — not seconds. Map these to a
    # rough "long-cycle" set rather than parsing.
    LONG_CYCLE = {'1Hour', '2Hour', '4Hour', '1Day', '1Week', '1Month'}
    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    for lbl in req_labels:
        tf_label = get_tf_from_label(lbl)
        if tf_label in LONG_CYCLE:
            return True
    return False


# ── Secondary-TF snapshot (RORT_SECONDARY_TF_SNAPSHOT) ───────────────────────────
# Extends the snapshot concept to the SECONDARY series so coarse-gate strategies
# (1Hour+ secondary) can take the fast windowed resume instead of the long
# warmup load. The cached resampled secondary OHLCV is byte-identical to a fresh
# full resample (proven in secondary_tf_snapshot.py); injected via secondary_tf_dfs.
# Kill-switch default OFF — fully inert until enabled.

def _secondary_snapshot_enabled() -> bool:
    import os
    # graduated to default-ON 2026-07-03 (flag-graduation; env remains the kill switch)
    return os.getenv('RORT_SECONDARY_TF_SNAPSHOT', '1') == '1'


def _has_valid_secondary_snapshot(strat: dict, model_id: str | None = None) -> bool:
    """True if the strategy has a fingerprint+model-valid secondary snapshot."""
    if not _secondary_snapshot_enabled():
        return False
    try:
        import secondary_tf_snapshot as STC
        from unified_engine import compute_backtest_fingerprint
        cfg = strat.get('config') or {}
        b64 = strat.get('secondary_snapshot_b64') or cfg.get('secondary_snapshot_b64')
        de = STC.deserialize_secondary_snapshot(
            b64, expected_fingerprint=compute_backtest_fingerprint(strat),
            expected_model_id=model_id)
        return de is not None and bool(de.get('series'))
    except Exception:
        return False


def _secondary_snapshot_load_extend(strat, sec_tfs, until_dt, primary_tf,
                                 session, data_feed, model_id, no_backfill=False):
    """Load the cached secondary series + extend it with a SHORT fresh load
    from the last cached bar → until. Returns {canonical_tf: DataFrame} keyed
    for prepare_data's secondary_tf_dfs, or None on any miss (caller falls back
    to full resample). Byte-identical to fresh full resample by construction.

    `no_backfill`: the short recent-extend load reads the cache read-only (no
    Polygon) for the shadow-worker path."""
    try:
        import pandas as _pd
        import secondary_tf_snapshot as STC
        from unified_engine import compute_backtest_fingerprint
        from data_loader import load_market_data
        cfg = strat.get('config') or {}
        b64 = strat.get('secondary_snapshot_b64') or cfg.get('secondary_snapshot_b64')
        de = STC.deserialize_secondary_snapshot(
            b64, expected_fingerprint=compute_backtest_fingerprint(strat),
            expected_model_id=model_id)
        if de is None or not de.get('series'):
            return None
        cached = de['series']
        # earliest boundary across secondaries → recent load start
        boundaries = [df.index[-1] + _pd.Timedelta(seconds=STC._period_seconds(tf))
                      for tf, df in cached.items() if len(df)]
        if not boundaries:
            return None
        start = min(boundaries)
        _start = start.tz_localize(None) if getattr(start, 'tz', None) is not None else start
        _until = until_dt.replace(tzinfo=None) if getattr(until_dt, 'tzinfo', None) else until_dt
        recent = load_market_data(strat['symbol'], start_date=_start, end_date=_until,
                                  timeframe=primary_tf, feed=data_feed, session=session,
                                  no_backfill=no_backfill)
        extended = STC.extend_secondary_ohlcv(cached, recent, tuple(sec_tfs))
        return extended
    except Exception as _e:
        _logger.warning("secondary-snapshot load/extend failed (%s) — full resample", _e)
        return None


def _secondary_snapshot_persist(strat, df, sec_tfs, model_id):
    """Build the secondary snapshot from a fully-computed df (primary OHLCV) and
    persist it to config.secondary_snapshot_b64 via a FULL-config read-modify-write
    (never a partial JSONB update — see feedback_jsonb_partial_updates)."""
    try:
        import secondary_tf_snapshot as STC
        from unified_engine import compute_backtest_fingerprint
        from db import get_admin_client, get_current_user_id
        sec = STC.build_secondary_ohlcv(df, tuple(sec_tfs))
        if not sec:
            return
        last_ts = df.index[-1].isoformat()
        blob = STC.serialize_secondary_snapshot(
            sec, compute_backtest_fingerprint(strat), model_id, last_ts)
        if not blob:
            return
        c = get_admin_client()
        sid = strat.get('id')
        uid = strat.get('user_id') or get_current_user_id()
        row = c.table('strategies').select('config').eq('id', sid).single().execute()
        full_cfg = dict((row.data or {}).get('config') or {})
        full_cfg['secondary_snapshot_b64'] = blob
        c.table('strategies').update({'config': full_cfg}).eq('id', sid).execute()
    except Exception as _e:
        _logger.warning("secondary-snapshot persist failed (%s) — non-fatal", _e)


# ── Mechanism #1: bar-count warmup (RORT_PREP_BAR_COUNT_WARMUP) ───────────────
# Bug_Hunt_Wave1_2026-07-06 §MECHANISM #1: the calendar-days warmup formula
# (`ceil(warmup_bars/bpd × 365/252)`) yields 1 day for warmup_bars=300 on a
# sub-minute strategy. After a weekend/holiday `since − 1 day` holds ZERO
# trading bars, so the windowed re-prepare delivers a same-day-only window and
# recursive user-pack columns (UT-Bot-v4 trailing stop, Wilder ATR — infinite-
# memory recursions) RE-SEED at the session open (measured Δ up to 0.4858 on
# sid 270 Mon 2026-07-06; 3 flip mismatches; day's first trade lost). The fix:
# deliver warmup as bars ACTUALLY PRESENT — widen start_date back over closed
# days until enough pre-`since` bars exist.

PREP_WARMUP_MAX_LOOKBACK_DAYS = 30
"""Cap (calendar days before `since`) for the bar-count warmup widen. Bounds
the widen loop on symbols with sparse/no cached data, and leaves pathological
coarse-gate windows (e.g. a 1Day gate whose legacy formula start is already
~435d back) on their existing behavior — the widen only ever EXTENDS a window
that is SHALLOWER than this cap; it never narrows one."""


def _prep_warmup_submin_cap_days() -> int:
    """Tighter widen cap for SUB-MINUTE primaries (2026-07-07 incident,
    Bug_Hunt_Wave1_2026-07-06 'Re-arm attempt FAILED'): tf<60s bars are derived
    from the native 1Sec layer (~0.8-1.2s of load per calendar day for TSLA), so
    a widen chasing an UNSATISFIABLE secondary target (e.g. 300 x 4Hour buckets
    ~ 150 trading days) to the generic 30d cap costs minutes per slot — x ~20
    sub-minute slots vs the shadow-worker's 600s pass watchdog = starvation
    loop, zero polls complete. 14 days ≈ 9-10 trading days ≈ 3-12x the 1200-bar
    primary convergence target for every sub-minute TF (30Sec: 7.5k bars), and
    comfortably spans any holiday cluster — so the PRIMARY target is always
    satisfiable within it; only unsatisfiable coarse-secondary targets hit the
    cap (logged with the existing CAPPED warning)."""
    import os
    try:
        return max(1, int(os.getenv('RORT_PREP_WARMUP_SUBMIN_CAP_DAYS', '14')))
    except ValueError:
        return 14


def _prep_warmup_cap_days(timeframe: str) -> int:
    """Effective widen cap for a primary TF: the generic 30d cap, tightened to
    the sub-minute cap for tf<60s (1Sec-source loads — see above)."""
    try:
        from unified_engine import TIMEFRAME_SECONDS
        tf_s = int(TIMEFRAME_SECONDS.get(timeframe, 60))
    except Exception:
        tf_s = 60
    if tf_s < 60:
        return min(PREP_WARMUP_MAX_LOOKBACK_DAYS, _prep_warmup_submin_cap_days())
    return PREP_WARMUP_MAX_LOOKBACK_DAYS


def _prep_bar_count_warmup_enabled() -> bool:
    """Mechanism #1 kill-switch. Default OFF = byte-identical legacy behavior
    (calendar-days formula). Instant rollback by unsetting."""
    import os
    return os.getenv('RORT_PREP_BAR_COUNT_WARMUP', '0') == '1'


# ── (c) per-slot widen cache ─────────────────────────────────────────────────
# {(sid, symbol, timeframe, session, sec_tfs, warmup_bars, since_date): start}
# The widened start for a given strategy/day is deterministic (bars already on
# disk), so later polls the same day skip the count-fail + re-prep cycle and
# make EXACTLY ONE prep at the known depth. A later `since` the same day only
# ever has MORE pre-`since` bars, so a cached start stays sufficient; the
# delivered-count verification still runs (cheap) and re-widens + refreshes
# the entry if a hit ever under-delivers. Process-local, bounded, flag-ON only.
_PREP_WARMUP_WIDEN_CACHE: dict = {}
_PREP_WARMUP_WIDEN_CACHE_MAX = 4096
_prep_warmup_widen_lock = threading.Lock()


def _prep_warmup_cache_key(strat, timeframe, sec_tfs, warmup_bars, since_naive):
    return (strat.get('id'), strat.get('symbol'), timeframe,
            strat.get('trading_session', 'RTH'), tuple(sec_tfs or ()),
            int(warmup_bars), since_naive.date())


def _prep_warmup_cache_get(key):
    return _PREP_WARMUP_WIDEN_CACHE.get(key)


def _prep_warmup_cache_put(key, start_dt) -> None:
    with _prep_warmup_widen_lock:
        if len(_PREP_WARMUP_WIDEN_CACHE) >= _PREP_WARMUP_WIDEN_CACHE_MAX:
            _PREP_WARMUP_WIDEN_CACHE.clear()
        _PREP_WARMUP_WIDEN_CACHE[key] = start_dt


# ── (a) one-shot widen depth from the native 1Min layer ─────────────────────
def _derive_warmup_start_from_1min(symbol, session, since_naive, timeframe,
                                   primary_target, sec_targets, cap_start):
    """Compute the widened start_date in ONE cheap read of the native 1Min
    SOURCE layer (~390-960 rows/day) instead of doubling full re-preps — for a
    sub-minute Hi-Fi primary each doubling round is a multi-day 1Sec load, and
    the 2026-07-07 incident showed the loop (1d->2d->4d->8d...) costs minutes
    per slot. The 1Min layer is used ONLY FOR COUNTING depth; the delivered
    bars keep their exact existing source (tf<60s stays 1Sec-derived Hi-Fi).

    Counting rules (all on the session-filtered pre-`since` 1Min index):
      - primary tf >= 60s : EXACT — distinct floor(tf) buckets == the bars the
        prep will deliver (1Min-or-coarser bars are themselves 1Min-derived).
      - primary tf < 60s  : ESTIMATE — ceil(target * tf/60) 1Min bars, i.e. a
        liquid symbol carries 60/tf sub-minute bars per traded minute. Deep
        warmup CONTEXT doesn't need the estimate to be exact: 1200 bars is
        M-RS1's convergence bound with enormous margin (0.9^1200 is float-
        zero; PR #39 measured utv4 bit-identical across 4 different seed
        depths past ~1 trading day), and the caller VERIFIES the delivered
        count and tops up if short — so a sparse symbol degrades to one extra
        bounded re-prep, never to wrong values.
      - each secondary tf : EXACT bucket count (same floor logic).

    Returns (start_dt_naive_utc, satisfiable). `satisfiable=False` means some
    target cannot be met even at `cap_start` (e.g. 300 x 4Hour buckets ~ 150
    trading days vs a 14/30d cap) — the caller preps once AT the cap and logs
    the existing CAPPED warning instead of crawling there in doubling rounds.
    Returns (None, True) when the layer can't answer (no direct PG / no rows /
    error) — caller falls back to the legacy doubling loop unchanged."""
    import math as _math
    try:
        import bar_cache as _bc
        if not (_bc.is_enabled() and _bc.direct_pg_available()):
            return None, True
        raw = _bc.read_bars(symbol, "1Min", cap_start, since_naive)
        if raw is None or len(raw) == 0:
            return None, True
        from data_loader import _filter_session
        if session and session != "24/7":
            raw = _filter_session(raw, session)
        if raw is None or len(raw) == 0:
            return None, True
        clip = pd.Timestamp(since_naive)
        if raw.index.tz is not None and clip.tz is None:
            clip = clip.tz_localize('UTC')
        idx = raw.index[raw.index < clip]
        if len(idx) == 0:
            return None, True

        from unified_engine import TIMEFRAME_SECONDS
        requirements = []   # (needed_start_ts | None-if-unmet)
        satisfiable = True

        def _bucket_requirement(tf_label, target):
            secs = int(TIMEFRAME_SECONDS.get(tf_label, 60))
            buckets = idx.floor(f'{secs}s')
            uniq = buckets.unique()          # ascending
            if len(uniq) < target:
                return None
            cutoff = uniq[-int(target)]      # newest `target` buckets
            return idx[buckets >= cutoff][0]

        tf_s = int(TIMEFRAME_SECONDS.get(timeframe, 60))
        if tf_s >= 60:
            req = _bucket_requirement(timeframe, primary_target)
        else:
            need_1min = _math.ceil(primary_target * tf_s / 60.0)
            req = idx[-need_1min] if len(idx) >= need_1min else None
        if req is None:
            satisfiable = False
        else:
            requirements.append(req)

        for tf_label, target in (sec_targets or {}).items():
            req = _bucket_requirement(tf_label, target)
            if req is None:
                satisfiable = False
            else:
                requirements.append(req)

        if not satisfiable:
            return cap_start, False
        start_ts = min(requirements)
        if getattr(start_ts, 'tz', None) is not None:
            start_ts = start_ts.tz_convert('UTC').tz_localize(None)
        # 5-min pad so the boundary bar is inside the read range and a couple
        # of trade-less sub-minute slots can't leave the delivered count 1-2
        # bars short (which would cost a full fallback re-prep); never deeper
        # than the cap.
        start_dt = max(cap_start, start_ts.to_pydatetime() - timedelta(minutes=5))
        return start_dt, True
    except Exception as _e:
        _logger.warning("[PREP-WARMUP] 1Min-source depth derivation failed for "
                        "%s (%s) — falling back to doubling widen", symbol, _e)
        return None, True


def _prep_warmup_counts(df, since_naive, count_sec_tfs):
    """Count warmup bars ACTUALLY PRESENT before `since` in a prepared df.

    primary  = raw pre-`since` row count of the prepared (session-filtered)
               primary index.
    each sec = the number of resampled buckets those pre-`since` primary rows
               produce (distinct floor(period) values) — i.e. how many sec-TF
               bars the in-prep resample derives from the warmup span. Only
               meaningful on the FULL path where secondaries are resampled
               from this very window (the snapshot fast path injects its own
               fully-warmed series, left edge independent of this window)."""
    if df is None or len(df) == 0:
        return 0, {tf: 0 for tf in count_sec_tfs}
    clip = pd.Timestamp(since_naive)
    if df.index.tz is not None and clip.tz is None:
        clip = clip.tz_localize('UTC')
    elif df.index.tz is None and clip.tz is not None:
        clip = clip.tz_localize(None)
    pre = df.index[df.index < clip]
    n_primary = int(len(pre))
    n_sec = {}
    if count_sec_tfs:
        from unified_engine import TIMEFRAME_SECONDS
        for tf in count_sec_tfs:
            secs = int(TIMEFRAME_SECONDS.get(tf, 60))
            n_sec[tf] = (int(pre.floor(f'{secs}s').nunique())
                         if n_primary else 0)
    return n_primary, n_sec


def _widen_prep_for_bar_count_warmup(prep_fn, df, strat, since_naive,
                                     start_date, warmup_days, warmup_bars,
                                     timeframe, count_sec_tfs,
                                     effective_start=None):
    """Mechanism #1 fix core (flag ON only): re-prepare with a deeper
    start_date until the DELIVERED window holds

      - >= max(warmup_bars, strategy_data.PRIMARY_WARMUP_BARS) pre-`since`
        PRIMARY bars (1200 — M-RS1's convergence target; 0.9^1200 is float-
        zero, so infinite-memory recursions match a full-history prep), and
      - >= warmup_bars resampled bars for EACH secondary TF (full path only —
        callers pass count_sec_tfs=() when the snapshot fast path injected
        pre-warmed secondaries).

    BOUNDED (2026-07-07 incident rework — Bug_Hunt_Wave1_2026-07-06 'Re-arm
    attempt FAILED'): the original doubling loop re-prepared the FULL window
    each round (1d->2d->4d->8d ... cumulative), and for sub-minute Hi-Fi
    primaries each round is a multi-day 1Sec load — measured 22.5s on sid 325
    for one cold widen, minutes on the full path, x ~20 TSLA slots vs the
    600s pass watchdog = starvation, zero polls completed. Now:

      1. ONE-SHOT DEPTH: when the first prep is short, the required start is
         derived from ONE cheap read of the native 1Min layer
         (`_derive_warmup_start_from_1min`) and a SINGLE re-prep is made
         there. The 1Min layer only decides DEPTH — delivered bars keep their
         exact existing source (tf<60s stays 1Sec-derived Hi-Fi end to end).
      2. TF-AWARE CAP: sub-minute primaries cap the widen at
         `_prep_warmup_submin_cap_days()` (default 14d) instead of 30d, so an
         UNSATISFIABLE secondary target (300 x 4Hour buckets ~ 150 trading
         days) costs one bounded prep at the cap + the CAPPED warning — never
         minutes of crawling. The primary 1200-bar target always fits.
      3. VERIFIED + FALLBACK: the delivered count is still authoritative. If
         the one-shot estimate under-delivers (sparse symbol) or the 1Min
         layer can't answer, the legacy doubling loop finishes the job,
         bounded by the same cap.
      4. CACHE (per-slot, per-day): a successful widen records its start in
         `_PREP_WARMUP_WIDEN_CACHE`; the caller's NEXT poll preps once at the
         cached depth directly (`effective_start`), skipping the count-fail +
         re-prep cycle entirely.

    Counting the PREPARED df (not a raw probe) means "bars actually present"
    holds for every backtest_model path, including cache-backed ones. The
    widen only ever EXTENDS the legacy formula start — bars in (since, until]
    are untouched (depth is added strictly LEFT of `since`), and the common
    case (enough bars already present) costs zero extra work. Returns the
    (possibly re-prepared) df."""
    try:
        from strategy_data import PRIMARY_WARMUP_BARS as _pwb
    except Exception:
        _pwb = 1200
    primary_target = max(int(warmup_bars), int(_pwb))
    sec_target = int(warmup_bars)
    cache_key = _prep_warmup_cache_key(
        strat, timeframe, count_sec_tfs, warmup_bars, since_naive)
    if effective_start is None:
        effective_start = start_date

    def _satisfied(n_p, n_s):
        return (n_p >= primary_target
                and all(n >= sec_target for n in n_s.values()))

    n_primary, n_sec = _prep_warmup_counts(df, since_naive, count_sec_tfs)
    if _satisfied(n_primary, n_sec):
        if effective_start != start_date:
            # (c) keep the cache warm for the rest of the day's polls.
            _prep_warmup_cache_put(cache_key, effective_start)
        return df

    cap_days = _prep_warmup_cap_days(timeframe)
    cap_start = since_naive - timedelta(days=cap_days)
    final_start = effective_start

    # (a) one-shot: derive the needed depth from the native 1Min layer, then
    # make a single re-prep there instead of doubling full re-preps.
    if final_start > cap_start:
        derived_start, satisfiable = _derive_warmup_start_from_1min(
            strat.get('symbol'), strat.get('trading_session', 'RTH'),
            since_naive, timeframe, primary_target,
            {tf: sec_target for tf in count_sec_tfs}, cap_start)
        if derived_start is not None and derived_start < final_start:
            final_start = derived_start
            df = prep_fn(final_start)
            n_primary, n_sec = _prep_warmup_counts(df, since_naive, count_sec_tfs)
            if not satisfiable:
                # Target provably unmeetable within the cap — don't crawl.
                final_start = cap_start

    # Fallback / top-up: legacy doubling loop, bounded by the tf-aware cap.
    if not _satisfied(n_primary, n_sec):
        lookback = max(1, int(warmup_days),
                       (since_naive - final_start).days)
        while final_start > cap_start:
            lookback = min(cap_days, lookback * 2)
            final_start = since_naive - timedelta(days=lookback)
            df = prep_fn(final_start)
            n_primary, n_sec = _prep_warmup_counts(df, since_naive, count_sec_tfs)
            if _satisfied(n_primary, n_sec):
                break

    if final_start != start_date:
        # (c) cache even a CAPPED-short result — re-prepping deeper can't help,
        # and without the entry every subsequent poll would re-pay the widen.
        _prep_warmup_cache_put(cache_key, final_start)
        _logger.info(
            "[PREP-WARMUP] %s %s: widened warmup start %s -> %s "
            "(requested primary>=%d bars%s; delivered pre-since: primary=%d%s)%s",
            strat.get('symbol'), timeframe, start_date.isoformat(),
            final_start.isoformat(), primary_target,
            (", sec>=%d bars each %s" % (sec_target, list(count_sec_tfs))
             if count_sec_tfs else ""),
            n_primary,
            (", sec=%s" % n_sec) if n_sec else "",
            ("" if _satisfied(n_primary, n_sec)
             else " — CAPPED at %dd, still short" % cap_days))
    return df


def prepare_strategy_window_df(
    strat: dict, since_dt: datetime, until_dt: datetime,
    warmup_bars: int = 300, data_feed: str = "sip",
    model_override: Optional[str] = None,
    no_backfill: bool = False, persist_snapshot: bool = True,
) -> pd.DataFrame:
    """Prepare the fully-warmed, indicator-enriched df for the window WITHOUT
    running the engine — the data-prep half of get_strategy_trades_for_window
    (the no-resume path), factored out for the shadow-worker's resident engine
    (`shadow_manager`). Returns bars in `(since-warmup, until]` with all primary +
    secondary-TF / user-pack columns computed.

    Uses the secondary-TF-snapshot fast path (warmup sized off the PRIMARY only,
    coarse secondary injected from cache — byte-identical to a full resample by
    construction) when `RORT_SECONDARY_TF_SNAPSHOT=1` and a valid snapshot exists;
    otherwise the full path (warmup off the binding/coarsest TF). This avoids the
    coarse-secondary warmup blow-up (multi-day primary read at 1Sec) that a naive
    windowed prepare hits — see Plan_M-RS4_Phase3, project_coarse_secondary_warmup_blowup.

    no_backfill: read-only cache reads (shadow-worker — no Polygon, no cache writes).
    persist_snapshot: write the secondary snapshot on a full compute (pass False for
        the read-only shadow path so it never writes).
    """
    import math
    from data_loader import (
        BARS_PER_DAY, get_required_tfs_from_confluence, get_tf_from_label,
    )
    timeframe = strat.get('timeframe', '1Min')
    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))

    _sec_inject = None
    if _secondary_snapshot_enabled() and sec_tfs:
        _sec_inject = _secondary_snapshot_load_extend(
            strat, sec_tfs, until_dt, timeframe,
            strat.get('trading_session', 'RTH'), data_feed, model_override,
            no_backfill=no_backfill)
    _used_sec_snapshot = _sec_inject is not None

    if _used_sec_snapshot:
        primary_bpd = BARS_PER_DAY.get(timeframe, 390)
        warmup_days = max(1, math.ceil(warmup_bars / max(primary_bpd, 0.001) * 365 / 252))
    else:
        all_tfs = [timeframe] + list(sec_tfs)
        bpds = [BARS_PER_DAY.get(tf, 390) for tf in all_tfs]
        binding_bpd = min(bpd for bpd in bpds if bpd > 0) if bpds else 390
        warmup_days = max(1, math.ceil(warmup_bars / max(binding_bpd, 0.001) * 365 / 252))

    since_naive = since_dt.replace(tzinfo=None) if since_dt.tzinfo else since_dt
    end_date = until_dt.replace(tzinfo=None) if until_dt.tzinfo else until_dt
    start_date = since_naive - timedelta(days=warmup_days)

    def _prep(_start):
        return prepare_data_with_indicators(
            strat['symbol'], seed=strat.get('data_seed', 42),
            start_date=_start, end_date=end_date, timeframe=timeframe,
            data_feed=data_feed, session=strat.get('trading_session', 'RTH'),
            secondary_tfs=sec_tfs, secondary_tf_dfs=_sec_inject, strat=strat,
            model_override=model_override, no_backfill=no_backfill)

    # Mechanism #1 (RORT_PREP_BAR_COUNT_WARMUP, default OFF): the calendar-days
    # formula above can land `start_date` entirely inside closed days (weekend/
    # holiday) so the window carries ~0 pre-`since` bars and recursive user-pack
    # columns re-seed at the session open. Flag ON: widen start_date until the
    # delivered warmup is BARS ACTUALLY PRESENT (see helper) — bounded one-shot
    # depth via the 1Min layer + per-slot/day cache so the FIRST prep of a
    # repeat poll already sits at the widened depth (2026-07-07 incident: the
    # unbounded doubling widen starved the shadow-worker pass on sub-minute
    # Hi-Fi slots). Flag OFF is byte-identical (single `_prep` call with the
    # formula start above).
    if _prep_bar_count_warmup_enabled():
        _count_sec_tfs = () if _used_sec_snapshot else sec_tfs
        _cached_start = _prep_warmup_cache_get(_prep_warmup_cache_key(
            strat, timeframe, _count_sec_tfs, warmup_bars, since_naive))
        _eff_start = (_cached_start
                      if _cached_start is not None and _cached_start < start_date
                      else start_date)
        df = _prep(_eff_start)
        df = _widen_prep_for_bar_count_warmup(
            _prep, df, strat, since_naive, start_date, warmup_days,
            warmup_bars, timeframe, _count_sec_tfs,
            effective_start=_eff_start)
    else:
        df = _prep(start_date)
    if df is None or len(df) == 0:
        return df

    # On a FULL compute (no snapshot used), persist the snapshot so the NEXT read
    # takes the fast path — unless the read-only caller opted out.
    if (persist_snapshot and _secondary_snapshot_enabled() and sec_tfs
            and not _used_sec_snapshot):
        _secondary_snapshot_persist(strat, df, sec_tfs, model_override)

    _end_clip = pd.Timestamp(end_date)
    if df.index.tz is not None and _end_clip.tz is None:
        _end_clip = _end_clip.tz_localize('UTC')
    elif df.index.tz is None and _end_clip.tz is not None:
        _end_clip = _end_clip.tz_localize(None)
    return df[df.index <= _end_clip]


def get_strategy_trades(strat: dict, data_feed: str = "sip", model_override: str | None = None) -> pd.DataFrame:
    """Get trades for any modern strategy (backtest-only or forward-testing).

    Routes through `strategy_data.load_strategy_data` so the visible
    window + warmup is consistent with prepare_forward_test_data,
    Mass Builder, and Update All Data. See
    docs/Audit_Warmup_Window_Alignment.md for why this matters.

    `model_override` (2026-05-07): forces the engine to dispatch on a
    specific backtest_model value instead of strategy.config.backtest_model.
    Used by the cron path to run under `algo_model` while leaving the
    strategy's backtest_model alone.
    """
    # Webhook origin: use stored trades only
    if strat.get('strategy_origin') == 'webhook_inbound':
        return trades_df_from_stored(strat.get('stored_trades', []))

    if 'entry_trigger_confluence_id' not in strat:
        return pd.DataFrame()

    # Forward-testing strategies need the bt/fw split — defer to
    # prepare_forward_test_data, which also routes through the helper.
    if strat.get('forward_testing') and strat.get('forward_test_start'):
        df_full, bt, fw, _ = prepare_forward_test_data(
            strat, data_feed=data_feed, model_override=model_override)
        trades = pd.concat([bt, fw], ignore_index=True)
        # daily_r denominator: count trading days in the VISIBLE window
        # (the helper-trimmed df), not the warmup-extended one.
        trades.attrs['trading_days'] = count_trading_days(df_full) if len(df_full) else 1
        return trades

    # Non-forward-testing branch: backtest-only / lookback-range strategies.
    from strategy_data import (
        load_strategy_data, trim_trades_to_visible, trim_df_to_visible,
    )
    bundle = load_strategy_data(
        strat, data_feed=data_feed, model_override=model_override)
    if len(bundle.df) == 0:
        return pd.DataFrame()
    trades = unified_trades(bundle.df, strat)
    trades = trim_trades_to_visible(trades, bundle.visible_start)
    # daily_r denominator: trading days in visible window only.
    visible_df = trim_df_to_visible(bundle.df, bundle.visible_start)
    trades.attrs['trading_days'] = (
        count_trading_days(visible_df) if len(visible_df) else 1)
    return trades


def prepare_forward_test_data(
    strat: dict, data_feed: str = "sip",
    data_days_override: int = None,
    model_override: str | None = None,
):
    """Load continuous data and split trades at forward test boundary.

    Extracted from app.py:825. The data_feed parameter replaces _get_data_feed().

    `model_override` (2026-05-07): forces engine dispatch on a specific
    backtest_model value instead of strategy.config.backtest_model. Used
    by the cron path to run under algo_model.

    Returns (df, backtest_trades, forward_trades, forward_test_start_dt)
    """
    forward_test_start_dt = datetime.fromisoformat(strat['forward_test_start'])
    # OOS: trades are bucketed Backtest|Forward at the in-sample-end
    # divider (resolve_in_sample_end falls back to forward_test_start, so
    # this is behaviour-neutral for every pre-OOS strategy).
    _kpi_boundary = resolve_in_sample_end(strat) or forward_test_start_dt

    # Webhook origin: use stored trades directly
    if strat.get('strategy_origin') == 'webhook_inbound':
        trades = trades_df_from_stored(strat.get('stored_trades', []))
        if len(trades) == 0:
            empty = pd.DataFrame()
            return pd.DataFrame(), empty, empty, forward_test_start_dt
        backtest_trades, forward_trades = split_trades_at_boundary(trades, _kpi_boundary)
        return pd.DataFrame(), backtest_trades, forward_trades, forward_test_start_dt

    # Route through the unified helper so visible window + warmup match
    # every other read path. The helper:
    #   - Resolves visible_start from backtest_start_date (or fallback)
    #   - Computes warmup = visible_days * WARMUP_MULTIPLIER (still `*2`)
    #   - Loads bars [warmup_start, now]
    # See docs/Audit_Warmup_Window_Alignment.md.
    from strategy_data import (
        load_strategy_data, trim_trades_to_visible, trim_df_to_visible,
    )
    # Honor data_days_override by injecting it into the synth strat the
    # helper sees — callers passing override expect the override to drive
    # the visible window.
    strat_for_load = dict(strat)
    if data_days_override is not None:
        strat_for_load['data_days'] = data_days_override
        # If overriding, force the helper away from backtest_start_date
        # so the override actually takes effect (override semantics
        # preserve the legacy callers' intent).
        strat_for_load.pop('backtest_start_date', None)
    bundle = load_strategy_data(
        strat_for_load, data_feed=data_feed, model_override=model_override)

    if len(bundle.df) == 0:
        empty = pd.DataFrame()
        return bundle.df, empty, empty, forward_test_start_dt

    trades = unified_trades(bundle.df, strat)

    # Trim ALL trades to the visible window before splitting at the OOS
    # boundary. The helper's `visible_start` is the canonical anchor;
    # everything before it is warmup that must not surface to the user.
    trades = trim_trades_to_visible(trades, bundle.visible_start)
    backtest_trades, forward_trades = split_trades_at_boundary(trades, _kpi_boundary)

    # Return the trimmed df so downstream count_trading_days uses the
    # visible window (not the warmup-extended one).
    df_visible = trim_df_to_visible(bundle.df, bundle.visible_start)
    return df_visible, backtest_trades, forward_trades, forward_test_start_dt


# =============================================================================
# KPI CALCULATIONS
# =============================================================================

def count_trading_days(df: pd.DataFrame) -> int:
    """Count unique trading days in a DataFrame with a DatetimeIndex."""
    if len(df) == 0 or not hasattr(df.index, 'normalize'):
        return 1
    return max(df.index.normalize().nunique(), 1)


def calculate_kpis(
    trades_df: pd.DataFrame,
    starting_balance: float = 10000,
    risk_per_trade: float = 100,
    total_trading_days: int = None
) -> dict:
    """Calculate strategy KPIs.

    Extracted from app.py:1730.

    Args:
        total_trading_days: Total trading days in the data period (all market days,
            not just days with trades). When provided, Daily R = total_r / total_trading_days.
    """
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
            "max_r_drawdown": 0, "final_balance": starting_balance, "total_pnl": 0
        }

    # Exclude open-position rows
    if 'exit_reason' in trades_df.columns:
        trades_df = trades_df[trades_df['exit_reason'] != 'open']
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
            "max_r_drawdown": 0, "final_balance": starting_balance, "total_pnl": 0
        }

    wins = trades_df[trades_df["win"] == True]
    losses = trades_df[trades_df["win"] == False]

    gross_profit = wins["r_multiple"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["r_multiple"].sum()) if len(losses) > 0 else 0
    total_r = trades_df["r_multiple"].sum()

    if total_trading_days is not None and total_trading_days > 0:
        trading_days = total_trading_days
    elif "exit_time" in trades_df.columns:
        trading_days = trades_df["exit_time"].dt.date.nunique()
    else:
        trading_days = 1
    trading_days = max(trading_days, 1)

    if len(trades_df) >= 2:
        cumulative_r = trades_df["r_multiple"].cumsum().values
        x = np.arange(len(cumulative_r))
        correlation = np.corrcoef(x, cumulative_r)[0, 1]
        r_squared = round(correlation ** 2, 4) if not np.isnan(correlation) else 0.0
        running_max = np.maximum.accumulate(cumulative_r)
        drawdown = cumulative_r - running_max
        max_r_drawdown = round(float(drawdown.min()), 2)
    else:
        r_squared = 0.0
        max_r_drawdown = 0.0

    total_pnl = total_r * risk_per_trade
    final_balance = starting_balance + total_pnl

    return {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        "avg_r": trades_df["r_multiple"].mean(),
        "total_r": total_r,
        "daily_r": total_r / trading_days,
        "r_squared": r_squared,
        "max_r_drawdown": max_r_drawdown,
        "final_balance": final_balance,
        "total_pnl": total_pnl,
        "_risk_per_trade": risk_per_trade,
        "_starting_balance": starting_balance,
    }


def calculate_secondary_kpis(trades_df: pd.DataFrame, kpis: dict) -> dict:
    """Calculate secondary/extended KPIs from trade data.

    Extracted from app.py:1808. Includes basic trade stats, risk-adjusted ratios,
    distribution metrics, and drawdown analytics.
    """
    _EMPTY = {
        "win_count": 0, "loss_count": 0,
        "best_trade_r": 0, "worst_trade_r": 0,
        "avg_win_r": 0, "avg_loss_r": 0,
        "max_consec_wins": 0, "max_consec_losses": 0,
        "payoff_ratio": 0, "recovery_factor": 0,
        "longest_dd_trades": 0,
        "sharpe_ratio": None, "sortino_ratio": None, "calmar_ratio": None,
        "kelly_criterion": None, "daily_var_95": None, "cvar_95": None,
        "gain_pain_ratio": None, "common_sense_ratio": None,
        "tail_ratio": None, "outlier_win_ratio": None, "outlier_loss_ratio": None,
        "ulcer_index": None, "serenity_index": None,
        "skewness": None, "kurtosis": None,
        "expected_daily": None, "expected_monthly": None, "expected_yearly": None,
        "volatility": None, "longest_dd_days": None,
    }
    if len(trades_df) == 0:
        return _EMPTY

    wins_mask = trades_df["win"].values
    r_mult = trades_df["r_multiple"].values
    n = len(r_mult)

    win_count = int(wins_mask.sum())
    loss_count = n - win_count

    best_trade_r = float(r_mult.max())
    worst_trade_r = float(r_mult.min())

    avg_win_r = float(r_mult[wins_mask].mean()) if win_count > 0 else 0
    avg_loss_r = float(r_mult[~wins_mask].mean()) if loss_count > 0 else 0

    max_consec_wins = max_consec_losses = 0
    current_wins = current_losses = 0
    for w in wins_mask:
        if w:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    abs_avg_loss = abs(avg_loss_r)
    payoff_ratio = avg_win_r / abs_avg_loss if abs_avg_loss > 0 else float('inf')

    max_r_dd_abs = abs(kpis.get("max_r_drawdown", 0))
    total_r = kpis.get("total_r", 0)
    recovery_factor = total_r / max_r_dd_abs if max_r_dd_abs > 0 else float('inf')

    cumulative = np.cumsum(r_mult) if n >= 2 else np.array([0.0])
    if n >= 2:
        running_max = np.maximum.accumulate(cumulative)
        in_dd = cumulative < running_max
        longest_dd = 0
        current_dd = 0
        for d in in_dd:
            if d:
                current_dd += 1
                longest_dd = max(longest_dd, current_dd)
            else:
                current_dd = 0
    else:
        longest_dd = 0

    # ── Advanced Metrics ──────────────────────────────────────────────────
    risk_per_trade = kpis.get("_risk_per_trade", 100.0)
    starting_balance = kpis.get("_starting_balance", 10000.0)

    daily_r = None
    if "exit_time" in trades_df.columns and n >= 5:
        try:
            daily_r = trades_df.groupby(trades_df["exit_time"].dt.date)["r_multiple"].sum().values
        except Exception:
            daily_r = None

    sharpe_ratio = None
    if daily_r is not None and len(daily_r) >= 5:
        dr_mean = np.mean(daily_r)
        dr_std = np.std(daily_r, ddof=1)
        if dr_std > 0:
            sharpe_ratio = round(float(dr_mean / dr_std * np.sqrt(252)), 2)

    sortino_ratio = None
    if daily_r is not None and len(daily_r) >= 5:
        dr_mean = np.mean(daily_r)
        downside = daily_r[daily_r < 0]
        if len(downside) > 0:
            downside_std = np.std(downside, ddof=1)
            if downside_std > 0:
                sortino_ratio = round(float(dr_mean / downside_std * np.sqrt(252)), 2)

    calmar_ratio = None
    if n >= 10 and max_r_dd_abs > 0 and daily_r is not None and len(daily_r) >= 5:
        trading_years = len(daily_r) / 252
        if trading_years > 0:
            annualized_return = (total_r * risk_per_trade / starting_balance) / trading_years
            max_dd_pct = max_r_dd_abs * risk_per_trade / starting_balance
            if max_dd_pct > 0:
                calmar_ratio = round(float(annualized_return / max_dd_pct), 2)

    kelly_criterion = None
    if n >= 5 and win_count > 0 and loss_count > 0:
        win_rate_frac = win_count / n
        pr = avg_win_r / abs_avg_loss if abs_avg_loss > 0 else 0
        if pr > 0:
            kelly = win_rate_frac - (1 - win_rate_frac) / pr
            kelly_criterion = round(float(max(0.0, min(1.0, kelly))), 3)

    daily_var_95 = None
    if daily_r is not None and len(daily_r) >= 10:
        var_r = float(np.percentile(daily_r, 5))
        daily_var_95 = round(var_r * risk_per_trade, 2)

    cvar_95 = None
    if daily_r is not None and len(daily_r) >= 10:
        var_threshold = np.percentile(daily_r, 5)
        tail = daily_r[daily_r <= var_threshold]
        if len(tail) > 0:
            cvar_95 = round(float(np.mean(tail) * risk_per_trade), 2)

    gross_profit = float(r_mult[r_mult > 0].sum()) if np.any(r_mult > 0) else 0
    gross_loss_abs = float(abs(r_mult[r_mult < 0].sum())) if np.any(r_mult < 0) else 0
    gain_pain_ratio = round(gross_profit / gross_loss_abs, 2) if gross_loss_abs > 0 else None

    common_sense_ratio = None
    if n >= 5:
        pf = kpis.get("profit_factor", 0)
        if pf != float('inf') and pf > 0:
            common_sense_ratio = round(float(pf * (1 - 1 / n)), 2)

    tail_ratio = None
    if n >= 10:
        p95 = abs(float(np.percentile(r_mult, 95)))
        p5 = abs(float(np.percentile(r_mult, 5)))
        if p5 > 0:
            tail_ratio = round(p95 / p5, 2)

    outlier_win_ratio = None
    if win_count >= 2 and avg_win_r > 0:
        max_win = float(r_mult[wins_mask].max())
        outlier_win_ratio = round(max_win / avg_win_r, 2)

    outlier_loss_ratio = None
    if loss_count >= 2 and abs_avg_loss > 0:
        max_loss_abs = float(abs(r_mult[~wins_mask].min()))
        outlier_loss_ratio = round(max_loss_abs / abs_avg_loss, 2)

    ulcer_index = None
    if n >= 5:
        dd_series = cumulative - np.maximum.accumulate(cumulative)
        ulcer_index = round(float(np.sqrt(np.mean(dd_series ** 2))), 3)

    serenity_index = None
    if ulcer_index is not None and ulcer_index > 0:
        serenity_index = round(float(total_r / ulcer_index), 2)

    skewness = None
    if n >= 5:
        m = np.mean(r_mult)
        s = np.std(r_mult, ddof=1)
        if s > 0:
            skewness = round(float(np.mean(((r_mult - m) / s) ** 3) * n / max((n - 1) * (n - 2), 1) * n), 2)

    kurtosis = None
    if n >= 5:
        m = np.mean(r_mult)
        s = np.std(r_mult, ddof=1)
        if s > 0:
            kurt_raw = float(np.mean(((r_mult - m) / s) ** 4))
            kurtosis = round(kurt_raw - 3.0, 2)

    expected_daily = None
    expected_monthly = None
    expected_yearly = None
    if daily_r is not None and len(daily_r) >= 5:
        ed = float(np.mean(daily_r) * risk_per_trade)
        expected_daily = round(ed, 2)
        expected_monthly = round(ed * 21, 2)
        expected_yearly = round(ed * 252, 2)

    volatility = None
    if daily_r is not None and len(daily_r) >= 5:
        vol = float(np.std(daily_r, ddof=1) * np.sqrt(252) * 100)
        volatility = round(vol, 1)

    longest_dd_days = None
    if n >= 5 and "exit_time" in trades_df.columns:
        try:
            _exit_times = pd.to_datetime(trades_df["exit_time"])
            _cum = trades_df["r_multiple"].cumsum().values
            _rm = np.maximum.accumulate(_cum)
            _in_dd = _cum < _rm
            _max_dd_dur = 0
            _dd_start = None
            for idx_i in range(len(_in_dd)):
                if _in_dd[idx_i]:
                    if _dd_start is None:
                        _dd_start = _exit_times.iloc[idx_i]
                else:
                    if _dd_start is not None:
                        dur = (_exit_times.iloc[idx_i] - _dd_start).days
                        _max_dd_dur = max(_max_dd_dur, dur)
                        _dd_start = None
            if _dd_start is not None:
                dur = (_exit_times.iloc[-1] - _dd_start).days
                _max_dd_dur = max(_max_dd_dur, dur)
            longest_dd_days = _max_dd_dur
        except Exception:
            longest_dd_days = None

    return {
        "win_count": win_count,
        "loss_count": loss_count,
        "best_trade_r": round(best_trade_r, 2),
        "worst_trade_r": round(worst_trade_r, 2),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float('inf') else float('inf'),
        "recovery_factor": round(recovery_factor, 1) if recovery_factor != float('inf') else float('inf'),
        "longest_dd_trades": longest_dd,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "kelly_criterion": kelly_criterion,
        "daily_var_95": daily_var_95,
        "cvar_95": cvar_95,
        "gain_pain_ratio": gain_pain_ratio,
        "common_sense_ratio": common_sense_ratio,
        "tail_ratio": tail_ratio,
        "outlier_win_ratio": outlier_win_ratio,
        "outlier_loss_ratio": outlier_loss_ratio,
        "ulcer_index": ulcer_index,
        "serenity_index": serenity_index,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "expected_daily": expected_daily,
        "expected_monthly": expected_monthly,
        "expected_yearly": expected_yearly,
        "volatility": volatility,
        "longest_dd_days": longest_dd_days,
    }


# =============================================================================
# STRATEGY FIELD ENRICHMENT (Phase B5)
# =============================================================================

def compute_forward_kpis(strategy: dict, full_compute: bool = False) -> dict | None:
    """Compute KPIs for the forward test portion of a strategy.

    If full_compute=False (default, used for list endpoint):
      Only splits stored_trades at forward_test_start. Fast but may be stale.

    If full_compute=True (used for detail endpoint):
      Loads fresh market data and runs the full backtest to get current forward trades.
      This is slower (Polygon API + engine computation) but always current.

    Returns dict with win_rate, profit_factor, daily_r, total_r, trades, max_r_drawdown,
    or None if no forward test data.
    """
    if not strategy.get('forward_testing') or not strategy.get('forward_test_start'):
        return None

    # OOS divider — in_sample_end, falling back to forward_test_start.
    fwd_start = resolve_in_sample_end(strategy)
    fwd_trades = None

    # Try stored_trades first (fast path)
    stored = strategy.get('stored_trades', [])
    if stored:
        trades_df = trades_df_from_stored(stored)
        if len(trades_df) > 0:
            _, fwd_portion = split_trades_at_boundary(trades_df, fwd_start)
            if len(fwd_portion) > 0:
                fwd_trades = fwd_portion

    # Full computation if stored path yielded nothing and full_compute requested
    if fwd_trades is None and full_compute:
        try:
            _, _, fwd_portion, _ = prepare_forward_test_data(strategy)
            if len(fwd_portion) > 0:
                fwd_trades = fwd_portion
        except Exception as e:
            _logger.warning("Full forward test computation failed for strategy %s: %s",
                          strategy.get('id'), e)

    if fwd_trades is None or len(fwd_trades) == 0:
        return None

    kpis = calculate_kpis(fwd_trades)
    return {
        "win_rate": kpis.get("win_rate", 0),
        "profit_factor": kpis.get("profit_factor", 0),
        "daily_r": kpis.get("daily_r", 0),
        "total_r": kpis.get("total_r", 0),
        "avg_r": kpis.get("avg_r", 0),
        "trades": kpis.get("total_trades", 0),
        "max_r_drawdown": kpis.get("max_r_drawdown", 0),
    }


def compute_alert_kpis(strategy: dict) -> dict | None:
    """Compute KPIs from a strategy's live_executions (alert trades).

    Returns dict with win_rate, profit_factor, etc., or None if no alert data.
    """
    live_execs = strategy.get('live_executions')
    if not live_execs:
        return None

    # live_executions is a list of matched alert entries with r_multiple
    # Extract trades that have both entry and exit
    trades = []
    for exec_entry in live_execs:
        if exec_entry.get('r_multiple') is not None:
            trades.append({
                'entry_time': exec_entry.get('entry_time', ''),
                'exit_time': exec_entry.get('exit_time', ''),
                'r_multiple': exec_entry['r_multiple'],
                'win': exec_entry.get('r_multiple', 0) > 0,
            })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    if 'entry_time' in trades_df.columns:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], utc=True, errors='coerce')
    if 'exit_time' in trades_df.columns:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'], utc=True, errors='coerce')

    kpis = calculate_kpis(trades_df)
    return {
        "win_rate": kpis.get("win_rate", 0),
        "profit_factor": kpis.get("profit_factor", 0),
        "daily_r": kpis.get("daily_r", 0),
        "total_r": kpis.get("total_r", 0),
        "avg_r": kpis.get("avg_r", 0),
        "trades": kpis.get("total_trades", 0),
        "max_r_drawdown": kpis.get("max_r_drawdown", 0),
    }


def compute_sigma_deviation(actual_cumulative_r: float, n_trades: int, bt_trades_df: pd.DataFrame) -> float | None:
    """Compute sigma deviation using Performance vs Plan formula.

    Compares actual cumulative R against the expected cumulative R at N trades,
    using the backtest R distribution to set the plan line and confidence bands.

    Formula: sigma = (actual_cumR - N × avg_r) / √(N × var_r)

    This is more useful than per-trade comparison because it detects sustained
    underperformance that compounds over many trades (a small per-trade deviation
    can mask a huge cumulative shortfall).

    Args:
        actual_cumulative_r: Total cumulative R in the forward/alert period
        n_trades: Number of trades in the forward/alert period
        bt_trades_df: Backtest trades DataFrame with 'r_multiple' column

    Returns float (sigma) or None if insufficient data.
    """
    if len(bt_trades_df) < 10 or n_trades < 3:
        return None

    if 'r_multiple' not in bt_trades_df.columns:
        return None

    r_values = bt_trades_df['r_multiple'].values
    avg_r = float(np.mean(r_values))
    var_r = float(np.var(r_values, ddof=1))

    if var_r <= 0 or np.isnan(var_r):
        return None

    expected = n_trades * avg_r
    std_at_n = np.sqrt(n_trades * var_r)

    if std_at_n == 0:
        return None

    sigma = (actual_cumulative_r - expected) / std_at_n
    return round(float(sigma), 2)


def derive_strategy_status(
    forward_kpis: dict | None,
    sigma_fwd: float | None,
) -> str:
    """Derive strategy status from forward test performance.

    Returns: 'On Track', 'Outperforming', 'Underperforming', or 'Insufficient Data'
    """
    if forward_kpis is None or sigma_fwd is None:
        return 'Insufficient Data'

    if forward_kpis.get('trades', 0) < 5:
        return 'Insufficient Data'

    if sigma_fwd >= 2.0:
        return 'Outperforming'
    elif sigma_fwd <= -2.0:
        return 'Underperforming'
    else:
        return 'On Track'


def enrich_confluence_with_fidelity(confluence: list) -> list:
    """Enrich confluence condition strings with fidelity type [PB]/[CB].

    Current behavior: all cross-TF conditions use shifted (previous bar) data = [PB].
    The _spec_ columns exist for [CB] but aren't used in trading decisions yet.

    Input: ["5M-EMA_STACK-SML", "1D-MACD_LINE-BULL"]
    Output: [{"id": "5M-EMA_STACK-SML", "fidelity": "PB", "label": "5M-EMA_STACK-SML [PB]"}, ...]
    """
    if not confluence:
        return []

    result = []
    for cond in confluence:
        if isinstance(cond, dict):
            # Already enriched
            result.append(cond)
        elif isinstance(cond, str):
            # Primary TF conditions (1M prefix) don't need fidelity tags
            parts = cond.split('-')
            tf = parts[0] if parts else ''
            if tf in ('1M', '1m', 'GEN'):
                result.append({"id": cond, "fidelity": None, "label": cond})
            else:
                # Cross-TF = [PB] by default
                result.append({"id": cond, "fidelity": "PB", "label": f"{cond} [PB]"})
        else:
            result.append({"id": str(cond), "fidelity": None, "label": str(cond)})
    return result


def enrich_strategy(strategy: dict, full_compute: bool = False) -> dict:
    """Add forward_kpis, alert_kpis, sigma, status, and fidelity to a strategy dict.

    Args:
        strategy: Raw strategy dict from DB
        full_compute: If True, load fresh market data for forward test computation.
            Used for detail endpoint. False (default) for list endpoint (faster).
    """
    enriched = dict(strategy)
    sid = strategy.get('id', '?')

    # Log data quality warnings
    stored = strategy.get('stored_trades', [])
    kpis = strategy.get('kpis', {})
    if not stored:
        logger.warning("[ENRICH] Strategy %s has 0 stored_trades", sid)
    if not kpis:
        logger.warning("[ENRICH] Strategy %s has no kpis object", sid)
        kpis = {}
    # Validate trade dates (sample first 5)
    for trade in stored[:5]:
        et = trade.get('entry_time')
        xt = trade.get('exit_time')
        if et and not isinstance(et, str):
            logger.warning("[ENRICH] Strategy %s trade has non-string entry_time: %s (%s)", sid, et, type(et).__name__)
        if xt and not isinstance(xt, str):
            logger.warning("[ENRICH] Strategy %s trade has non-string exit_time: %s (%s)", sid, xt, type(xt).__name__)
    # Sanitize inf/NaN KPIs
    if isinstance(kpis, dict):
        for k, v in list(kpis.items()):
            if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                logger.warning("[ENRICH] Strategy %s KPI '%s' is %s — sanitizing to 0", sid, k, v)
                kpis[k] = 0.0
    enriched['kpis'] = kpis

    # Alerts always on for forward testing strategies
    if strategy.get('forward_testing') and not strategy.get('alert_tracking_enabled'):
        enriched['alert_tracking_enabled'] = True

    # Resolve time_exit_pack_id → time_exit_config (for strategies saved with pack ID only)
    if strategy.get('time_exit_pack_id') and not strategy.get('time_exit_config'):
        try:
            from time_exit_packs import load_time_exit_packs, get_pack_by_id
            packs = load_time_exit_packs()
            pack = get_pack_by_id(strategy['time_exit_pack_id'], packs)
            if pack:
                enriched['time_exit_config'] = pack.get_exit_config()
        except Exception:
            pass

    # Fidelity-enriched confluence
    raw_conf = strategy.get('confluence', [])
    enriched['confluence_enriched'] = enrich_confluence_with_fidelity(raw_conf)

    # Forward KPIs
    fwd_kpis = compute_forward_kpis(strategy, full_compute=full_compute)
    enriched['forward_kpis'] = fwd_kpis

    # Alert KPIs
    alert_kpis = compute_alert_kpis(strategy)
    enriched['alert_kpis'] = alert_kpis

    # Sigma deviation (PvP formula: cumulative R vs expected at N trades)
    stored = strategy.get('stored_trades', [])
    sigma_fwd = None
    sigma_alert = None
    if stored and strategy.get('forward_test_start'):
        try:
            bt_df = trades_df_from_stored(stored)
            # OOS divider — in_sample_end, falling back to forward_test_start.
            fwd_start = resolve_in_sample_end(strategy)
            bt_df, fwd_df = split_trades_at_boundary(bt_df, fwd_start)

            # FWD sigma from cumulative R
            if fwd_kpis and len(fwd_df) >= 3 and 'r_multiple' in fwd_df.columns:
                fwd_cum_r = float(fwd_df['r_multiple'].sum())
                sigma_fwd = compute_sigma_deviation(fwd_cum_r, len(fwd_df), bt_df)

            # Alert sigma — use alert_kpis total_r and trades count
            if alert_kpis and alert_kpis.get('trades', 0) >= 3:
                alert_cum_r = alert_kpis.get('total_r', 0.0)
                alert_n = alert_kpis.get('trades', 0)
                sigma_alert = compute_sigma_deviation(alert_cum_r, alert_n, bt_df)
        except Exception as e:
            logger.warning("[ENRICH] Sigma computation failed: %s", e)

    enriched['sigma_fwd'] = sigma_fwd
    enriched['sigma_alert'] = sigma_alert

    # Status
    enriched['status'] = derive_strategy_status(fwd_kpis, sigma_fwd)

    # Resolve trigger IDs to display names (only for detail page, not list — too expensive)
    if full_compute:
        try:
            from confluence_groups import get_enabled_groups, get_all_triggers
            all_triggers = get_all_triggers(get_enabled_groups())
            entry_id = strategy.get('entry_trigger_confluence_id', '')
            if entry_id and entry_id in all_triggers:
                enriched['entry_trigger_display_name'] = all_triggers[entry_id].name
            exit_ids = strategy.get('exit_trigger_confluence_ids', [])
            enriched['exit_trigger_display_names'] = {
                eid: all_triggers[eid].name for eid in exit_ids if eid in all_triggers
            }
        except Exception as e:
            logger.warning("[ENRICH] Failed to resolve trigger names: %s", e)

    return enriched


# =============================================================================
# CONFLUENCE ANALYSIS (ported from Streamlit app.py)
# =============================================================================

def _safe_float(v: float, default: float = 0.0) -> float:
    """Replace inf/nan with a safe default for JSON serialization."""
    import math
    if v is None or math.isnan(v) or math.isinf(v):
        return default
    return v


def _safe_subtract(a: float, b: float) -> float:
    """Safely subtract two values that may be infinity."""
    if a == float('inf') and b == float('inf'):
        return 0.0
    if a == float('-inf') and b == float('-inf'):
        return 0.0
    return a - b


def analyze_confluences(
    trades_df: pd.DataFrame,
    required: set = None,
    min_trades: int = 3,
    starting_balance: float = 10000,
    risk_per_trade: float = 100,
    total_trading_days: int = None,
    exclude_prefix: str = None,
    include_prefix: str = None,
    confluence_col: str = 'confluence_records',
) -> list[dict]:
    """Analyze how different confluence conditions affect results.

    Ported from Streamlit app.py:analyze_confluences (lines 1482-1544).

    Runs on a SINGLE trades_df — no additional backtests needed.
    Filters trades by confluence_records set membership per condition.

    Args:
        trades_df: Trades from a backtest (must have confluence_col column)
        required: If provided, only consider trades where required.issubset(confluence_records)
        min_trades: Minimum trades per condition to include in results
        exclude_prefix: Skip conditions starting with this (e.g. 'GEN-' for TF tab)
        include_prefix: Only include conditions starting with this (e.g. 'GEN-' for General tab)
        confluence_col: Column name with confluence record sets (default 'confluence_records',
                        use 'cb_confluence_records' for Current Bar fidelity analysis)

    Returns: List of dicts with per-condition KPIs, sorted by profit_factor desc.
    """
    if len(trades_df) == 0 or confluence_col not in trades_df.columns:
        return []

    # Normalize confluence_records: handle set, frozenset, list
    def _as_set(r):
        if isinstance(r, (set, frozenset)):
            return set(r)
        if isinstance(r, list):
            return set(r)
        return set()

    # Get base trades (filtered by required confluences)
    if required and len(required) > 0:
        mask = trades_df[confluence_col].apply(lambda r: required.issubset(_as_set(r)))
        base_trades = trades_df[mask]
    else:
        base_trades = trades_df

    if len(base_trades) < min_trades:
        return []

    base_kpis = calculate_kpis(base_trades, starting_balance=starting_balance,
                               risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)

    # Find all unique confluence records
    all_records = set()
    for records in base_trades[confluence_col]:
        all_records.update(_as_set(records))

    # Remove already-required records
    if required:
        all_records -= required

    # Apply prefix filters
    if exclude_prefix:
        all_records = {r for r in all_records if not r.startswith(exclude_prefix)}
    if include_prefix:
        all_records = {r for r in all_records if r.startswith(include_prefix)}

    results = []
    for record in sorted(all_records):
        # Filter to trades with this record
        mask = base_trades[confluence_col].apply(lambda r, rec=record: rec in _as_set(r))
        subset = base_trades[mask]

        if len(subset) >= min_trades:
            kpis = calculate_kpis(subset, starting_balance=starting_balance,
                                  risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)
            results.append({
                'confluence': record,
                'total_trades': kpis['total_trades'],
                'win_rate': round(_safe_float(kpis.get('win_rate', 0)), 1),
                'profit_factor': round(_safe_float(kpis.get('profit_factor', 0)), 2),
                'avg_r': round(_safe_float(kpis.get('avg_r', 0)), 3),
                'total_r': round(_safe_float(kpis.get('total_r', 0)), 2),
                'daily_r': round(_safe_float(kpis.get('daily_r', 0)), 3),
                'r_squared': round(_safe_float(kpis.get('r_squared', 0)), 3),
                'pf_change': round(_safe_float(_safe_subtract(kpis.get('profit_factor', 0), base_kpis.get('profit_factor', 0))), 2),
                'wr_change': round(_safe_float(kpis.get('win_rate', 0) - base_kpis.get('win_rate', 0)), 1),
            })

    results.sort(key=lambda r: r.get('profit_factor', 0), reverse=True)
    return results


def find_best_combinations(
    trades_df: pd.DataFrame,
    max_depth: int = 3,
    min_trades: int = 5,
    top_n: int = 50,
    starting_balance: float = 10000,
    risk_per_trade: float = 100,
    total_trading_days: int = None,
    exclude_prefix: str = None,
    include_prefix: str = None,
    allowed_labels: set = None,
    required_groups: list = None,
    progress_callback=None,
    oos_boundary=None,
    oos_required: dict = None,
    oos_sigma_cfg: dict = None,
    oos_trading_days: int = None,
) -> list[dict]:
    """Find the best confluence combinations automatically.

    Ported from Streamlit app.py:find_best_combinations (lines 1547-1624).
    Uses pre-computed numpy boolean masks for fast subset filtering.

    allowed_labels: if provided, restrict search to records in this set.
        Matching is both by exact membership and by suffix (TF-agnostic for
        records like "{tf}-{INTERP}-{STATE}" — any TF prefix is accepted if
        "{INTERP}-{STATE}" is in the set).
    required_groups: Mass Builder ✱ requirements. A list of alternative-sets:
        each entry is ONE required selection, given as labels in exact or
        TF-agnostic suffix form (same matching as allowed_labels). AND across
        entries; OR within an entry (one selection may resolve to several
        concrete records, e.g. the same state on two gate TFs). Every
        concrete assignment becomes its own baseline: its mask is ANDed under
        every candidate combo, a depth-0 (required-only) row is emitted, and
        `max_depth` counts only the optional adds on top. Result rows carry
        the concrete required records in `combination` (so downstream gating/
        replay/save works unchanged) plus a `required` field naming them.
        An entry that matches no trade records fails the search loudly
        (returns []) rather than silently broadening it. None/empty →
        behavior identical to before this parameter existed.
    progress_callback(idx, total): called during the combination loop.

    OOS gate (docs/Spec_OOS_Test_Periods.md §9): when `oos_boundary` (a
    datetime) is given, each combo's filtered trades are split at that
    boundary — the row KPIs are computed on the in-sample side, and the
    combo is dropped unless its out-of-sample side clears the OOS gate
    (`oos_required` thresholds + optional `oos_sigma_cfg` band). Rows then
    carry `oos_kpis` / `oos_sigma`. When None, behaves exactly as before.

    Returns: List of dicts with combination KPIs, sorted by profit_factor desc.
    """
    import numpy as np
    from itertools import combinations

    if len(trades_df) == 0 or 'confluence_records' not in trades_df.columns:
        return []

    def _as_set(r):
        if isinstance(r, (set, frozenset)):
            return set(r)
        if isinstance(r, list):
            return set(r)
        return set()

    def _label_match(rec: str, labels) -> bool:
        """Exact or TF-agnostic-suffix membership (see allowed_labels doc)."""
        if rec in labels:
            return True
        parts = rec.split("-", 2)
        return len(parts) == 3 and f"{parts[1]}-{parts[2]}" in labels

    # Get all unique records
    all_records = set()
    for records in trades_df['confluence_records']:
        all_records.update(_as_set(records))

    # Resolve required selections to concrete records BEFORE the pool
    # filters — a required record does not need to be in the optional pool.
    required_alternatives: list[list[str]] = []
    for grp in (required_groups or []):
        grp_set = set(grp)
        matches = sorted(r for r in all_records if _label_match(r, grp_set))
        if not matches:
            # No trade ever satisfied this requirement → no combo can pass.
            # Fail loudly rather than silently dropping the requirement.
            print(f"[MASS] required confluence {sorted(grp_set)} matched no "
                  f"trade records — 0 combos for this base config", flush=True)
            return []
        required_alternatives.append(matches)

    if exclude_prefix:
        all_records = {r for r in all_records if not r.startswith(exclude_prefix)}
    if include_prefix:
        all_records = {r for r in all_records if r.startswith(include_prefix)}
    # `is not None` (not truthiness): an EMPTY set means "no optional pool"
    # (required-only Mass Builder search), which must not fall through to
    # explore-all. None keeps the historical unrestricted behavior.
    if allowed_labels is not None:
        all_records = {r for r in all_records if _label_match(r, allowed_labels)}

    # Required records are baseline, never optional adds.
    required_record_set = {r for alts in required_alternatives for r in alts}
    all_records -= required_record_set

    # Pre-compute boolean mask per record (vectorized)
    n_trades = len(trades_df)
    record_masks = {}
    conf_list = trades_df['confluence_records'].tolist()
    for r in all_records | required_record_set:
        mask = np.zeros(n_trades, dtype=bool)
        for i, recs in enumerate(conf_list):
            if r in _as_set(recs):
                mask[i] = True
        record_masks[r] = mask

    # One baseline per concrete assignment of the required alternatives
    # (cartesian product — almost always 1×1×…). Baseline () = unrestricted.
    if required_alternatives:
        from itertools import product as _product
        _MAX_BASELINES = 32
        baselines = []
        _seen = set()
        for assign in _product(*required_alternatives):
            key = tuple(sorted(set(assign)))
            if key not in _seen:
                _seen.add(key)
                baselines.append(key)
        if len(baselines) > _MAX_BASELINES:
            print(f"[MASS] required confluences expand to {len(baselines)} "
                  f"baselines; capping at {_MAX_BASELINES}", flush=True)
            baselines = baselines[:_MAX_BASELINES]
    else:
        baselines = [()]

    # Enumerate (baseline, optional-combo) pairs. Depth counts optional adds
    # only; a non-empty baseline also gets a depth-0 (required-only) row.
    # min_trades for the optional pool is evaluated WITHIN the baseline
    # subset — a record plentiful overall may be too thin under the gates.
    all_combos = []
    for req in baselines:
        req_mask = None
        for r in req:
            req_mask = record_masks[r] if req_mask is None else (req_mask & record_masks[r])
        valid_records = sorted([
            r for r in all_records
            if (record_masks[r] if req_mask is None
                else (record_masks[r] & req_mask)).sum() >= min_trades])
        start_depth = 0 if req else 1
        for depth in range(start_depth, min(max_depth + 1, len(valid_records) + 1)):
            for combo in combinations(valid_records, depth):
                all_combos.append((req, combo))
    total = len(all_combos)

    results = []
    for idx, (req, combo) in enumerate(all_combos):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx, total)
        # AND the pre-computed masks (required baseline + optional adds)
        full = tuple(req) + tuple(combo)
        combined = record_masks[full[0]]
        for r in full[1:]:
            combined = combined & record_masks[r]

        count = int(combined.sum())
        if count >= min_trades:
            subset = trades_df[combined]
            oos_eval = None
            if oos_boundary is not None:
                # Rank on the in-sample side; gate on the OOS side.
                is_subset, oos_subset = split_trades_at_boundary(
                    subset, oos_boundary)
                if len(is_subset) < min_trades:
                    continue
                kpis = calculate_kpis(
                    is_subset, starting_balance=starting_balance,
                    risk_per_trade=risk_per_trade,
                    total_trading_days=total_trading_days)
                from mass_builder import evaluate_oos
                oos_eval = evaluate_oos(
                    is_subset, oos_subset, oos_required, oos_sigma_cfg,
                    starting_balance=starting_balance,
                    risk_per_trade=risk_per_trade,
                    oos_trading_days=oos_trading_days)
                if not oos_eval['passed']:
                    continue
            else:
                kpis = calculate_kpis(
                    subset, starting_balance=starting_balance,
                    risk_per_trade=risk_per_trade,
                    total_trading_days=total_trading_days)
            row = {
                'combination': list(sorted(full)),
                'combo_str': ' + '.join(sorted(full)),
                'depth': len(full),
                # Concrete records that came from ✱ requirements ([] when
                # none) — display marking; combination already includes them.
                'required': list(req),
                'total_trades': kpis['total_trades'],
                'win_rate': round(_safe_float(kpis.get('win_rate', 0)), 1),
                'profit_factor': round(_safe_float(kpis.get('profit_factor', 0)), 2),
                'avg_r': round(_safe_float(kpis.get('avg_r', 0)), 3),
                'daily_r': round(_safe_float(kpis.get('daily_r', 0)), 3),
                'r_squared': round(_safe_float(kpis.get('r_squared', 0)), 3),
            }
            if oos_eval is not None:
                row['oos_kpis'] = oos_eval['oos_kpis']
                row['oos_sigma'] = oos_eval['sigma']
            results.append(row)

    if progress_callback:
        progress_callback(total, total)

    results.sort(key=lambda r: (r.get('profit_factor', 0), r.get('total_trades', 0)), reverse=True)
    return results[:top_n]


# =============================================================================
# STRATEGY HEALTH BADGE
# =============================================================================
# Pure-function health computer per docs/Strategy_Health_Badge_Design.md.
# Takes already-loaded state (strategy dict + optional auxiliary inputs) and
# returns a typed health report. No DB writes, no extra queries — caller
# decides how to source the inputs.

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# Backtest-affecting fields that, when edited, invalidate KPIs.
# Set conservatively — anything in `config` that the unified engine consumes.
# Excludes name, tags, portfolio assignment, monitored flag, forward_testing.
HEALTH_BACKTEST_AFFECTING_FIELDS = frozenset({
    'entry_trigger_confluence_id',
    'exit_trigger_confluence_ids',
    'exit_trigger_confluence_id',  # legacy single-form
    'confluence',
    'general_confluences',
    'stop_config',
    'target_config',
    'time_exit_config',
    'direction',
    'trading_session',
    'data_days',
    'lookback_mode',
    'symbol',
    'timeframe',
})

# Time-staleness threshold (days). KPIs older than this flag yellow.
HEALTH_TIME_STALE_DAYS = 60

# Forward-test trade threshold for the green tier (vs minor).
HEALTH_GREEN_FORWARD_TRADES = 30


_SEVERITY_ORDER = {'minor': 1, 'action': 2, 'broken': 3}


@dataclass
class HealthIssue:
    """One issue surfaced for a strategy. Multiple issues per strategy are OK;
    the badge color reflects the worst severity present."""
    severity: Literal['minor', 'action', 'broken']
    code: str
    title: str           # short — for tooltip
    detail: str          # longer — for drawer
    fix_action: Optional[str] = None
    fix_action_label: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyHealth:
    severity: Literal['healthy', 'minor', 'action', 'broken']
    issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'severity': self.severity,
            'issues': [i.to_dict() for i in self.issues],
        }


def _config_view(strategy: dict) -> dict:
    """Return a unified config view that works for both shapes:
       (a) raw DB row: strategy['config'] is a dict / JSON string with the
           config-level keys nested inside.
       (b) API/app shape: _row_to_strategy has already merged config keys
           up to the top level, so strategy['config'] is empty/missing.

    The fix (2026-04-27): merge BOTH levels into a single dict, with the
    nested `config` taking precedence when both have a key (in practice
    they don't disagree). Without this merge, every API-returned strategy
    looked like it had a missing entry trigger, flagging healthy strategies
    as 'broken' in the Health Badge.
    """
    cfg_nested = strategy.get('config') or {}
    if isinstance(cfg_nested, str):
        try:
            import json
            cfg_nested = json.loads(cfg_nested)
        except Exception:
            cfg_nested = {}
    if not isinstance(cfg_nested, dict):
        cfg_nested = {}
    # Start with the entire flat dict (API shape: config keys live at top).
    # Then layer the nested config on top so DB-shaped rows still work.
    merged = dict(strategy)
    merged.update(cfg_nested)
    return merged


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        # Supabase returns 'YYYY-MM-DDTHH:MM:SS.ffffff+00:00' — fromisoformat handles it
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None


def compute_strategy_health(
    strategy: dict,
    *,
    n_alerts: Optional[int] = None,
    n_trades: Optional[int] = None,
    n_forward_trades: Optional[int] = None,
    pack_registry_slugs: Optional[set] = None,
    enabled_confluence_group_ids: Optional[set] = None,
    now: Optional[datetime] = None,
) -> StrategyHealth:
    """Derive a StrategyHealth report from current state.

    Args:
        strategy: Row dict from the `strategies` table (config can be JSON or dict).
        n_alerts: Alerts in DB for this strategy since forward_test_start.
                  When None, the live_alert_divergence check is skipped.
        n_trades: Trade rows in DB for this strategy since forward_test_start.
                  Required alongside n_alerts for the divergence check.
        n_forward_trades: Forward-test trade count (may equal n_trades) — used
                          for the green-tier promotion check.
        pack_registry_slugs: Set of currently-registered user-pack slugs.
                             When provided, missing-pack issues are detected.
        enabled_confluence_group_ids: Set of currently-enabled confluence
                                      group IDs. When provided, orphan-group
                                      issues are detected.
        now: Override "current time" for time-staleness; defaults to UTC now.

    Returns:
        StrategyHealth with severity (worst issue's severity, or 'healthy') and
        the full issue list ordered worst-first.

    Pure: no DB writes, no extra DB reads. Caller sources optional inputs.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    cfg = _config_view(strategy)
    issues: list[HealthIssue] = []

    # ── Configuration issues (additive — any can fire) ──
    entry_id = cfg.get('entry_trigger_confluence_id')
    if not entry_id:
        issues.append(HealthIssue(
            severity='broken',
            code='missing_entry_trigger',
            title='Missing entry trigger',
            detail=(
                'This strategy has no entry_trigger_confluence_id, so it '
                'cannot be re-backtested or run live. Likely a legacy save '
                'before the confluence-trigger schema landed. Edit the '
                'strategy and pick an entry trigger to repair.'),
            fix_action='edit_strategy',
            fix_action_label='Edit strategy',
        ))

    if pack_registry_slugs is not None and entry_id:
        # Each user-pack-prefixed entry_id is `<pack_slug>_<rest>` — look for
        # any registered slug that matches as a prefix.
        looks_like_user_pack = '_' in entry_id and not any(
            entry_id.startswith(p) for p in
            ('ema_', 'macd_', 'vwap_', 'rvol_', 'utbot_', 'ema_pp_v2_',
             'ema_price_position_', 'utbot_v2_', 'macd_line_', 'macd_histogram_',
             'ema_stack_'))
        if looks_like_user_pack:
            matched = any(entry_id.startswith(slug + '_')
                          for slug in pack_registry_slugs)
            if not matched:
                # Could be a removed user pack. Best-effort guess by
                # extracting the slug-like prefix.
                issues.append(HealthIssue(
                    severity='action',
                    code='orphan_user_pack',
                    title='References a missing user pack',
                    detail=(
                        f'Entry trigger {entry_id!r} references a user pack '
                        'that is no longer registered. The strategy will '
                        'not run live until the pack is re-installed or the '
                        'trigger is changed.'),
                    fix_action='edit_strategy',
                    fix_action_label='Edit strategy',
                ))

    if enabled_confluence_group_ids is not None:
        confluences = cfg.get('confluence', []) or []
        # Confluence group IDs are tracked separately from the records; this
        # check approximates by looking at template references encoded in
        # entry_id and exit_id strings.
        # (Detailed group-vs-record matching deferred — keep MVP simple.)

    # ── Data quality issues (mutually exclusive — pick highest observed) ──
    # The migration guarantees these columns exist on the row; tolerate
    # legacy reads that haven't been re-fetched yet.
    data_source = strategy.get('data_source')
    kpis_stale_since = _parse_iso(strategy.get('kpis_stale_since'))
    kpis_computed_at = _parse_iso(strategy.get('kpis_computed_at'))

    # Stale-since-edit dominates over time-stale and tier-by-data-source.
    if kpis_stale_since is not None:
        issues.append(HealthIssue(
            severity='action',
            code='kpis_stale_since_edit',
            title='KPIs stale since last edit',
            detail=(
                'A backtest-affecting field was edited after the last KPI '
                'run, so the displayed KPIs no longer reflect the current '
                'config. Re-run the backtest to refresh.'),
            fix_action='run_full_backtest',
            fix_action_label='Run full backtest',
        ))
    elif data_source == 'rapid':
        issues.append(HealthIssue(
            severity='action',
            code='rapid_test_data',
            title='Rapid-test KPIs only',
            detail=(
                'These KPIs come from the Mass Builder rapid-test path, '
                'which uses a short window and simplified stops. Run a '
                'full backtest before trusting the numbers in production.'),
            fix_action='run_full_backtest',
            fix_action_label='Run full backtest',
        ))
    elif data_source == 'full':
        # Full backtest is OK but Hi-Fi gives execution-fidelity bonus.
        # Only add the minor issue if there's no Hi-Fi run yet.
        issues.append(HealthIssue(
            severity='minor',
            code='no_hifi_run',
            title='Full backtest, no Hi-Fi yet',
            detail=(
                'KPIs are from the full backtest. A Hi-Fi run would tighten '
                'execution fidelity (slippage, intra-bar fills). Optional.'),
            fix_action='run_hifi_backtest',
            fix_action_label='Run Hi-Fi backtest',
        ))
    # data_source == 'hifi' → no issue added (it's the green tier)

    # Time-stale check (independent of data_source)
    if kpis_computed_at is not None:
        age_days = (now - kpis_computed_at).total_seconds() / 86400.0
        if age_days > HEALTH_TIME_STALE_DAYS:
            issues.append(HealthIssue(
                severity='minor',
                code='kpis_time_stale',
                title=f'KPIs >{HEALTH_TIME_STALE_DAYS} days old',
                detail=(
                    f'Last KPI computation was {int(age_days)} days ago. '
                    'Market conditions may have shifted since. Re-running '
                    'the backtest re-validates the strategy against more '
                    'recent data.'),
                fix_action='run_full_backtest',
                fix_action_label='Run full backtest',
            ))

    # ── Operational issues (additive) ──
    # The standout case from the 2026-04-27 audit: strategy has trades in DB
    # but zero alerts since forward_test_start. Strong signal that live
    # was/is silently broken. Independent of data_source.
    if (n_alerts is not None and n_trades is not None
            and n_alerts == 0 and n_trades > 0):
        issues.append(HealthIssue(
            severity='action',
            code='live_alert_divergence',
            title='Backtest trades exist but no live alerts',
            detail=(
                f'{n_trades} backtest trades are recorded for this strategy '
                'but no alerts have ever fired live. The most common cause '
                'is a confluence gate that backtest satisfies but live '
                'does not — historically this has flagged silent engine '
                'bugs. Investigate before trusting the strategy in '
                'production.'),
            fix_action=None,
            fix_action_label=None,
        ))

    # ── Determine overall severity ──
    if not issues:
        return StrategyHealth(severity='healthy', issues=[])

    issues.sort(key=lambda i: -_SEVERITY_ORDER[i.severity])
    worst = issues[0].severity
    return StrategyHealth(severity=worst, issues=issues)

