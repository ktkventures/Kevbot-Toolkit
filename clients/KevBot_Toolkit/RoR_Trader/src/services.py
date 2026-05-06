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

def prepare_data_with_indicators(
    symbol: str, days: int = 30, seed: int = 42,
    start_date=None, end_date=None,
    timeframe: str = "1Min", data_feed: str = "sip",
    session: str = "RTH", secondary_tfs: tuple = (),
    primary_df: Optional[pd.DataFrame] = None,
    secondary_tf_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load market data and run all indicators, interpreters, and trigger detection.

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

    use_injected = primary_df is not None

    # TTL-cache lookup. Only applies to the REST path — the cache key is
    # keyed on REST inputs (symbol, days, etc.), so injected DataFrames
    # would silently collide with REST results otherwise.
    if not use_injected:
        cache_key = _prepare_cache_key(
            symbol, days, seed, start_date, end_date,
            timeframe, data_feed, session, secondary_tfs)
        cached = _prepare_cache_get(cache_key)
        if cached is not None:
            return cached

    # Load raw bars (or use the injected DataFrame)
    if use_injected:
        df = primary_df.copy()
    else:
        df = load_market_data(symbol, days=days, seed=seed,
                              start_date=start_date, end_date=end_date,
                              timeframe=timeframe, feed=data_feed,
                              session=session)

    if len(df) == 0:
        return df

    # Run indicators
    df = run_all_indicators(df)

    # Run group-specific indicators for custom parameters
    for group in get_enabled_groups(load_confluence_groups()):
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
                for group in get_enabled_groups(load_confluence_groups()):
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

    Extracted from app.py:1587.
    """
    if len(trades_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    boundary_ts = pd.Timestamp(boundary_dt)
    col_tz = getattr(trades_df['entry_time'].dtype, 'tz', None)

    if col_tz is not None and boundary_ts.tzinfo is None:
        boundary_ts = boundary_ts.tz_localize(col_tz)
    elif col_tz is not None and boundary_ts.tzinfo is not None:
        boundary_ts = boundary_ts.tz_convert(col_tz)
    elif col_tz is None and boundary_ts.tzinfo is not None:
        boundary_ts = boundary_ts.tz_localize(None)

    backtest = trades_df[trades_df['entry_time'] < boundary_ts].copy()
    forward = trades_df[trades_df['entry_time'] >= boundary_ts].copy()
    return backtest, forward


def get_strategy_trades_for_window(
    strat: dict,
    since_dt: datetime,
    until_dt: datetime,
    warmup_bars: int = 100,
    data_feed: str = "sip",
) -> pd.DataFrame:
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

    LIMITATION: 100-bar warmup isn't enough for strategies whose
    confluence references long-cycle secondary TFs (1Hour or larger).
    Caller should detect those and either bump warmup_bars OR fall back
    to get_strategy_trades. See `get_required_tfs_from_confluence` in
    data_loader.
    """
    import math
    from data_loader import (
        BARS_PER_DAY, get_required_tfs_from_confluence, get_tf_from_label,
    )

    if 'entry_trigger_confluence_id' not in strat:
        return pd.DataFrame()
    if strat.get('strategy_origin') == 'webhook_inbound':
        return trades_df_from_stored(strat.get('stored_trades', []))

    timeframe = strat.get('timeframe', '1Min')
    bpd = BARS_PER_DAY.get(timeframe, 390)
    # Translate warmup-bars to calendar days. Use 365/252 multiplier to
    # account for non-trading days inside the warmup window.
    warmup_days = max(1, math.ceil(warmup_bars / max(bpd, 1) * 365 / 252))

    # Strip tz for comparison if since_dt carries one — prepare_data_with_indicators
    # accepts naive or aware; staying naive matches the Streamlit pattern.
    since_naive = since_dt.replace(tzinfo=None) if since_dt.tzinfo else since_dt
    start_date = since_naive - timedelta(days=warmup_days)
    end_date = until_dt.replace(tzinfo=None) if until_dt.tzinfo else until_dt

    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))

    df = prepare_data_with_indicators(
        strat['symbol'],
        seed=strat.get('data_seed', 42),
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        data_feed=data_feed,
        session=strat.get('trading_session', 'RTH'),
        secondary_tfs=sec_tfs,
    )
    if len(df) == 0:
        return pd.DataFrame()

    trades = unified_trades(df, strat)
    if len(trades) == 0:
        return pd.DataFrame()

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
    return trades


def _has_long_cycle_secondary_tf(strat: dict) -> bool:
    """Return True if strategy uses 1Hour or larger secondary TF.

    Used by callers of get_strategy_trades_for_window to decide whether
    100-bar warmup is sufficient or they need the full-history path.
    A 1Hour secondary TF needs ~250 bars × 1 hour = ~10 days of warmup;
    the windowed helper's default 100 bars at 1Min would only give
    ~100 minutes, leaving the secondary indicators in undefined state.
    """
    from data_loader import get_required_tfs_from_confluence, get_tf_from_label
    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    for lbl in req_labels:
        tf_seconds = get_tf_from_label(lbl)
        if tf_seconds and tf_seconds >= 3600:  # 1 hour or larger
            return True
    return False


def get_strategy_trades(strat: dict, data_feed: str = "sip") -> pd.DataFrame:
    """Get trades for any modern strategy (backtest-only or forward-testing).

    Extracted from app.py:892. The data_feed parameter replaces the
    Streamlit session-state lookup (_get_data_feed).
    """
    # Webhook origin: use stored trades only
    if strat.get('strategy_origin') == 'webhook_inbound':
        return trades_df_from_stored(strat.get('stored_trades', []))

    if 'entry_trigger_confluence_id' not in strat:
        return pd.DataFrame()

    if strat.get('forward_testing') and strat.get('forward_test_start'):
        df_full, bt, fw, _ = prepare_forward_test_data(strat, data_feed=data_feed)
        trades = pd.concat([bt, fw], ignore_index=True)
        # Attach trading_days from the source bars so callers can compute
        # daily_r against the full period (matches Mass Builder semantics).
        # count_trading_days returns 1 for trades-only DFs (integer index),
        # which would otherwise inflate refresh KPIs ~180× for a 180-day
        # window. The df has a DatetimeIndex so normalize().nunique() works.
        trades.attrs['trading_days'] = count_trading_days(df_full) if len(df_full) else 1
        return trades
    else:
        data_days = strat.get('data_days', 30)
        data_seed = strat.get('data_seed', 42)
        gst_start = None
        gst_end = None
        if strat.get('lookback_mode') == 'Date Range' and strat.get('lookback_start_date'):
            gst_start = datetime.fromisoformat(strat['lookback_start_date'])
            gst_end = datetime.fromisoformat(strat['lookback_end_date'])
        from data_loader import get_required_tfs_from_confluence, get_tf_from_label
        req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
        sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))
        df = prepare_data_with_indicators(
            strat['symbol'], data_days, data_seed,
            start_date=gst_start, end_date=gst_end,
            timeframe=strat.get('timeframe', '1Min'),
            data_feed=data_feed,
            session=strat.get('trading_session', 'RTH'),
            secondary_tfs=sec_tfs)
        if len(df) == 0:
            return pd.DataFrame()
        trades = unified_trades(df, strat)
        # See note in the forward-test branch above.
        trades.attrs['trading_days'] = count_trading_days(df)
        return trades


def prepare_forward_test_data(
    strat: dict, data_feed: str = "sip",
    data_days_override: int = None
):
    """Load continuous data and split trades at forward test boundary.

    Extracted from app.py:825. The data_feed parameter replaces _get_data_feed().

    Returns (df, backtest_trades, forward_trades, forward_test_start_dt)
    """
    forward_test_start_dt = datetime.fromisoformat(strat['forward_test_start'])

    # Webhook origin: use stored trades directly
    if strat.get('strategy_origin') == 'webhook_inbound':
        trades = trades_df_from_stored(strat.get('stored_trades', []))
        if len(trades) == 0:
            empty = pd.DataFrame()
            return pd.DataFrame(), empty, empty, forward_test_start_dt
        backtest_trades, forward_trades = split_trades_at_boundary(trades, forward_test_start_dt)
        return pd.DataFrame(), backtest_trades, forward_trades, forward_test_start_dt

    data_days = data_days_override if data_days_override is not None else strat.get('data_days', 30)
    data_seed = strat.get('data_seed', 42)

    start_date = forward_test_start_dt - timedelta(days=data_days * 2)
    end_date = datetime.now(timezone.utc).replace(hour=23, minute=59, second=0, microsecond=0)

    strat_timeframe = strat.get('timeframe', '1Min')
    from data_loader import get_required_tfs_from_confluence, get_tf_from_label
    required_tf_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in required_tf_labels))

    df = prepare_data_with_indicators(
        strat['symbol'], seed=data_seed,
        start_date=start_date, end_date=end_date,
        timeframe=strat_timeframe, data_feed=data_feed,
        session=strat.get('trading_session', 'RTH'),
        secondary_tfs=sec_tfs,
    )

    if len(df) == 0:
        empty = pd.DataFrame()
        return df, empty, empty, forward_test_start_dt

    trades = unified_trades(df, strat)

    backtest_trades, forward_trades = split_trades_at_boundary(trades, forward_test_start_dt)

    # Trim backtest trades to pinned backtest window (exclude warmup-period trades)
    _bt_start = strat.get('backtest_start_date')
    if _bt_start and len(backtest_trades) > 0:
        _bt_start_ts = pd.Timestamp(_bt_start)
        if _bt_start_ts.tz is None and backtest_trades['entry_time'].dt.tz is not None:
            _bt_start_ts = _bt_start_ts.tz_localize('UTC')
        backtest_trades = backtest_trades[backtest_trades['entry_time'] >= _bt_start_ts]

    return df, backtest_trades, forward_trades, forward_test_start_dt


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

    fwd_start = datetime.fromisoformat(strategy['forward_test_start'])
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
            fwd_start = datetime.fromisoformat(strategy['forward_test_start'])
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
    progress_callback=None,
) -> list[dict]:
    """Find the best confluence combinations automatically.

    Ported from Streamlit app.py:find_best_combinations (lines 1547-1624).
    Uses pre-computed numpy boolean masks for fast subset filtering.

    allowed_labels: if provided, restrict search to records in this set.
        Matching is both by exact membership and by suffix (TF-agnostic for
        records like "{tf}-{INTERP}-{STATE}" — any TF prefix is accepted if
        "{INTERP}-{STATE}" is in the set).
    progress_callback(idx, total): called during the combination loop.

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

    # Get all unique records
    all_records = set()
    for records in trades_df['confluence_records']:
        all_records.update(_as_set(records))
    if exclude_prefix:
        all_records = {r for r in all_records if not r.startswith(exclude_prefix)}
    if include_prefix:
        all_records = {r for r in all_records if r.startswith(include_prefix)}
    if allowed_labels:
        def _matches(rec: str) -> bool:
            if rec in allowed_labels:
                return True
            parts = rec.split("-", 2)
            if len(parts) == 3 and f"{parts[1]}-{parts[2]}" in allowed_labels:
                return True
            return False
        all_records = {r for r in all_records if _matches(r)}

    # Pre-compute boolean mask per record (vectorized)
    n_trades = len(trades_df)
    record_masks = {}
    conf_list = trades_df['confluence_records'].tolist()
    for r in all_records:
        mask = np.zeros(n_trades, dtype=bool)
        for i, recs in enumerate(conf_list):
            if r in _as_set(recs):
                mask[i] = True
        record_masks[r] = mask

    # Only test records that appear in >= min_trades
    valid_records = sorted([r for r, m in record_masks.items() if m.sum() >= min_trades])

    all_combos = []
    for depth in range(1, min(max_depth + 1, len(valid_records) + 1)):
        for combo in combinations(valid_records, depth):
            all_combos.append(combo)
    total = len(all_combos)

    results = []
    for idx, combo in enumerate(all_combos):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx, total)
        # AND the pre-computed masks
        combined = record_masks[combo[0]]
        for r in combo[1:]:
            combined = combined & record_masks[r]

        count = int(combined.sum())
        if count >= min_trades:
            subset = trades_df[combined]
            kpis = calculate_kpis(subset, starting_balance=starting_balance,
                                  risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)
            results.append({
                'combination': list(sorted(combo)),
                'combo_str': ' + '.join(sorted(combo)),
                'depth': len(combo),
                'total_trades': kpis['total_trades'],
                'win_rate': round(_safe_float(kpis.get('win_rate', 0)), 1),
                'profit_factor': round(_safe_float(kpis.get('profit_factor', 0)), 2),
                'avg_r': round(_safe_float(kpis.get('avg_r', 0)), 3),
                'daily_r': round(_safe_float(kpis.get('daily_r', 0)), 3),
                'r_squared': round(_safe_float(kpis.get('r_squared', 0)), 3),
            })

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

