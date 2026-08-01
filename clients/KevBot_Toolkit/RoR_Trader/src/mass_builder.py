"""
Mass Strategy Builder — combination engine and helpers for Phase 33.

Enumerates strategy combinations across tickers, timeframes, directions,
triggers, confluences, and risk management packs.  Reuses the same unified
engine pipeline as the Strategy Builder.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CACHE — avoids re-running prepare_data_with_indicators for the same
# (symbol, TF, days, session) across mass search runs.  Equivalent of
# Streamlit's @st.cache_data for the API context.
# ═══════════════════════════════════════════════════════════════════════════

_data_cache: Dict[tuple, tuple] = {}   # key → (df, sec_tf_map, trading_days, visible_start)
_DATA_CACHE_MAX = 20                    # max entries (LRU eviction)


def _cache_key(symbol, tf, days, session, start_date, end_date) -> tuple:
    return (symbol, tf, days, session, str(start_date), str(end_date))


def _cache_put(key: tuple, df, sec_tf_map, trading_days,
                visible_start=None):
    if len(_data_cache) >= _DATA_CACHE_MAX:
        # evict oldest entry
        _data_cache.pop(next(iter(_data_cache)))
    _data_cache[key] = (df, sec_tf_map, trading_days, visible_start)


def clear_data_cache():
    """Clear the prepared-data cache (called from tests or admin endpoints)."""
    _data_cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def estimate_combinations(config: dict) -> dict:
    """Estimate total combination counts and rough execution time.

    Returns dict with:
        base_configs: int — (tickers × TFs × directions × entries × exit_combos × RM_packs)
        confluence_combos_per_base: int — sum of C(n,d) for d=1..depth
        total_evaluations: int — base_configs × confluence_combos_per_base
        est_seconds: float — rough time estimate
    """
    n_tickers = max(len(config.get('tickers', [])), 1)
    n_tfs = max(len(config.get('timeframes', [])), 1)
    n_dirs = max(len(config.get('directions', [])), 1)
    n_entries = max(len(config.get('entry_triggers', [])), 1)

    # Exit combinations
    n_exits_raw = len(config.get('exit_triggers', []))
    exit_depth = config.get('exit_depth', 1)
    n_exit_combos = _n_choose_up_to(n_exits_raw, exit_depth)

    base_configs = n_tickers * n_tfs * n_dirs * n_entries * n_exit_combos

    # Confluence combinations per base config. ✱ requirements don't multiply
    # the count — they sit under every combo — but a required-only search
    # still evaluates its depth-0 baseline row.
    n_tf_conf = len(config.get('tf_confluences', []))
    tf_depth = config.get('tf_confluence_depth', 2)
    n_gen_conf = len(config.get('general_confluences', []))
    gen_depth = config.get('general_confluence_depth', 1)
    n_required = (len(config.get('required_tf_confluences', []) or []) +
                  len(config.get('required_general_confluences', []) or []))

    tf_combos = _n_choose_up_to(n_tf_conf, tf_depth) if n_tf_conf > 0 else 1
    gen_combos = _n_choose_up_to(n_gen_conf, gen_depth) if n_gen_conf > 0 else 1
    confluence_combos = tf_combos * gen_combos
    if n_required > 0:
        confluence_combos += 1  # the depth-0 required-baseline row

    total = base_configs * confluence_combos

    # Rough time estimates:
    # - ~200ms per base config (bar_cache replay)
    # - ~2ms per confluence filter
    data_groups = n_tickers * n_tfs
    est_data_load = data_groups * 3.0          # ~3s per data load (cached after first)
    est_base = base_configs * 0.2               # ~200ms per bar_cache replay
    est_conf = total * 0.002                    # ~2ms per confluence filter
    est_seconds = est_data_load + est_base + est_conf

    return {
        'n_tickers': n_tickers,
        'n_timeframes': n_tfs,
        'n_directions': n_dirs,
        'n_entries': n_entries,
        'n_exit_combos': n_exit_combos,
        'base_configs': base_configs,
        'confluence_combos_per_base': confluence_combos,
        'total_evaluations': total,
        'est_seconds': est_seconds,
    }


def _n_choose_up_to(n: int, max_depth: int) -> int:
    """Sum of C(n, d) for d = 1 .. min(max_depth, n).  Returns at least 1."""
    if n <= 0:
        return 1
    total = 0
    for d in range(1, min(max_depth, n) + 1):
        total += math.comb(n, d)
    return max(total, 1)


def generate_exit_combos(exit_triggers: list, depth: int) -> list:
    """Generate exit trigger combinations up to given depth.

    Returns list of lists.  Each inner list is a combination of exit trigger IDs.
    E.g., depth=1: [[a], [b], [c]], depth=2: [[a], [b], [c], [a,b], [a,c], [b,c]]
    """
    combos = []
    for d in range(1, min(depth, len(exit_triggers)) + 1):
        for combo in itertools.combinations(exit_triggers, d):
            combos.append(list(combo))
    return combos if combos else [[]]


def meets_required_performance(kpis: dict, required: dict) -> bool:
    """Check if KPIs meet all required performance thresholds.

    None/null thresholds are treated as no constraint.
    """
    if not required:
        return True
    min_trades = required.get('min_trades')
    if min_trades and kpis.get('total_trades', 0) < min_trades:
        return False
    min_wr = required.get('min_win_rate')
    if min_wr is not None and kpis.get('win_rate', 0) < min_wr:
        return False
    min_pf = required.get('min_profit_factor')
    if min_pf is not None and kpis.get('profit_factor', 0) < min_pf:
        return False
    min_dr = required.get('min_daily_r')
    if min_dr is not None and kpis.get('daily_r', 0) < min_dr:
        return False
    min_r2 = required.get('min_r_squared')
    if min_r2 is not None and kpis.get('r_squared', 0) < min_r2:
        return False
    return True


# =============================================================================
# OUT-OF-SAMPLE GATE  (docs/Spec_OOS_Test_Periods.md §9)
# =============================================================================
# A combo is kept only if it clears the in-sample Required-Performance AND
# the out-of-sample gate. The OOS gate has two combinable parts:
#   - raw thresholds  — plain minimums on OOS KPIs (meets_required_performance)
#   - sigma band      — the OOS cumulative R must stay within N sigma of the
#                       projection from the in-sample per-trade R stats.

def _r_series(trades_df):
    """Closed-trade r_multiple values as a clean float Series."""
    import pandas as pd
    if trades_df is None or len(trades_df) == 0 \
            or 'r_multiple' not in trades_df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(trades_df['r_multiple'], errors='coerce').dropna()


def oos_sigma_status(is_trades_df, oos_trades_df) -> dict:
    """Compare OOS cumulative R to the in-sample projection.

    From the in-sample per-trade R stats (mean mu, std sigma) the expected
    OOS cumulative R over n_oos trades is n_oos*mu, with a 1-sigma spread
    of sigma*sqrt(n_oos). Reports how many sigma the actual OOS result
    sits from that expectation.

    status: 'green' z >= -1, 'amber' z >= -2, 'red' z < -2. z is None
    when not computable (too few trades or zero in-sample variance).
    """
    import math
    is_r = _r_series(is_trades_df)
    oos_r = _r_series(oos_trades_df)
    n_oos = len(oos_r)
    actual = round(float(oos_r.sum()), 3) if n_oos else 0.0
    out = {'z': None, 'status': 'unknown', 'expected_r': None,
           'actual_r': actual, 'n_oos': n_oos,
           'lower_1s': None, 'lower_2s': None}
    if len(is_r) < 2 or n_oos < 1:
        return out
    mu = float(is_r.mean())
    sigma = float(is_r.std(ddof=1))
    expected = n_oos * mu
    out['expected_r'] = round(expected, 3)
    if sigma <= 0:
        out['status'] = 'green' if actual >= expected else 'red'
        return out
    std_k = sigma * math.sqrt(n_oos)
    z = (actual - expected) / std_k
    out['z'] = round(z, 2)
    out['lower_1s'] = round(expected - std_k, 3)
    out['lower_2s'] = round(expected - 2 * std_k, 3)
    out['status'] = 'green' if z >= -1 else ('amber' if z >= -2 else 'red')
    return out


def evaluate_oos(is_trades_df, oos_trades_df, oos_required, oos_sigma_cfg,
                 starting_balance=10000, risk_per_trade=100,
                 oos_trading_days=None) -> dict:
    """Compute OOS KPIs + sigma status and decide the out-of-sample gate.

    Gate = raw OOS Required-Performance thresholds AND, when the sigma
    band is enabled, z >= -N. Either side may be left empty.
    Returns {oos_kpis, sigma, passed}.
    """
    from services import calculate_kpis
    oos_kpis = calculate_kpis(
        oos_trades_df, starting_balance=starting_balance,
        risk_per_trade=risk_per_trade, total_trading_days=oos_trading_days)
    sigma = oos_sigma_status(is_trades_df, oos_trades_df)

    raw_ok = meets_required_performance(oos_kpis, oos_required or {})
    sigma_ok = True
    if oos_sigma_cfg and oos_sigma_cfg.get('enabled'):
        n_sigma = float(oos_sigma_cfg.get('n', 2.0) or 2.0)
        if sigma['z'] is not None:
            sigma_ok = sigma['z'] >= -n_sigma
        elif sigma['status'] == 'red':
            sigma_ok = False
    return {
        'oos_kpis': oos_kpis,
        'sigma': sigma,
        'passed': bool(raw_ok and sigma_ok),
    }


def build_strategy_config(
    symbol: str,
    timeframe: str,
    direction: str,
    session: str,
    entry_cid: str,
    entry_base: str,
    entry_name: str,
    exit_cids: list,
    exit_bases: list,
    exit_names: list,
    bar_count_exit: Optional[int],
    stop_config: dict,
    target_config: Optional[dict],
    date_range: dict,
    asset_type: str = 'equity',
    risk_per_trade: float = 100.0,
    starting_balance: float = 10000.0,
    backtest_start_date: Optional[str] = None,
    backtest_end_date: Optional[str] = None,
    time_exit_config: Optional[dict] = None,
    backtest_model: Optional[str] = None,
    algo_model: Optional[str] = None,
    live_model: Optional[str] = None,
    in_sample_start: Optional[str] = None,
    in_sample_end: Optional[str] = None,
) -> dict:
    """Build a strategy config dict compatible with _unified_trades() and save flow.

    Model fields (algo_model split — 2026-05-08): if any are None, the
    GET-side enrichment in `api/routers/strategies.py` fills the
    registry default. Mass Builder propagates explicit user choices
    here so spawned strategies don't silently inherit defaults.
    """
    lookback_mode = date_range.get('mode', 'days')
    data_days = date_range.get('days', 90)
    start_date = date_range.get('start')
    end_date = date_range.get('end')

    cfg = {
        'symbol': symbol,
        'asset_type': asset_type,
        'direction': direction,
        'timeframe': timeframe,
        'trading_session': session,
        'entry_trigger': entry_base,
        'entry_trigger_confluence_id': entry_cid,
        'entry_trigger_name': entry_name,
        'exit_triggers': exit_bases,
        'exit_trigger_confluence_ids': exit_cids,
        'exit_trigger_names': exit_names,
        'exit_trigger': exit_bases[0] if exit_bases else None,
        'exit_trigger_confluence_id': exit_cids[0] if exit_cids else None,
        'exit_trigger_name': exit_names[0] if exit_names else None,
        'bar_count_exit': bar_count_exit,
        'risk_per_trade': risk_per_trade,
        'stop_config': stop_config,
        'target_config': target_config,
        'starting_balance': starting_balance,
        'data_days': data_days,
        'lookback_mode': lookback_mode,
        'lookback_start_date': start_date,
        'lookback_end_date': end_date,
        'backtest_start_date': backtest_start_date,
        'backtest_end_date': backtest_end_date,
        'time_exit_config': time_exit_config,
        'strategy_origin': 'standard_mass_builder',
    }
    if backtest_model:
        cfg['backtest_model'] = backtest_model
    if algo_model:
        cfg['algo_model'] = algo_model
    if live_model:
        cfg['live_model'] = live_model
    # OOS: stamp the in-sample window so the strategy's charts/KPIs know
    # where the held-out band begins (docs/Spec_OOS_Test_Periods.md §4).
    if in_sample_start:
        cfg['in_sample_start'] = in_sample_start
    if in_sample_end:
        cfg['in_sample_end'] = in_sample_end
    return cfg


def _serialize_trades(trades_df, *, backtest_model: str = 'rest_hifi') -> list:
    """Serialize trades DataFrame to list of dicts for storage.

    Stamps `data_source = 'backtest_<backtest_model>'` on every record so
    that the trades-table insert path (Phase 40) tags rows correctly.
    Without this stamp, saved trades land with `data_source=NULL` and
    the `data_source LIKE 'backtest_%'` filter used by chart loaders +
    Update All Data wouldn't see them — leading to invisible trades and
    duplicate inserts on the next recompute.
    """
    import pandas as pd
    if trades_df is None or not isinstance(trades_df, pd.DataFrame) or len(trades_df) == 0:
        return []
    ds_tag = f'backtest_{backtest_model or "rest_hifi"}'
    cols = ['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price',
            'stop_price', 'target_price', 'r_multiple', 'win', 'exit_reason',
            'exec_type', 'bars_held', 'hold_time_seconds', 'entry_trigger',
            # Canonical Trade_Timestamps_Spec fields. _trade_to_row already
            # back-fills entry_fill_ts from entry_time aliases, but emit
            # both so callers that read stored_trades JSONB see canonical
            # field names directly.
            'entry_trigger_ts', 'entry_fill_ts',
            'exit_trigger_ts', 'exit_fill_ts',
            'hold_duration_s', 'behavior']
    records = []
    for _, row in trades_df.iterrows():
        d: dict = {'data_source': ds_tag}
        for c in cols:
            val = row.get(c)
            if val is not None and pd.notna(val):
                if hasattr(val, 'isoformat'):
                    d[c] = val.isoformat()
                elif isinstance(val, (bool,)):
                    d[c] = bool(val)
                else:
                    try:
                        d[c] = float(val) if isinstance(val, (int, float)) else str(val)
                    except (TypeError, ValueError):
                        d[c] = str(val)
        records.append(d)
    return records


def build_equity_curve(trades_df) -> list:
    """Extract cumulative R series from trades DataFrame for sparkline display."""
    import pandas as pd
    if trades_df is None or not isinstance(trades_df, pd.DataFrame) or len(trades_df) == 0:
        return []
    if 'r_multiple' not in trades_df.columns:
        return []
    closed = trades_df[trades_df['exit_time'].notna()] if 'exit_time' in trades_df.columns else trades_df
    if len(closed) == 0:
        return []
    cumulative = closed['r_multiple'].cumsum().tolist()
    return [round(v, 3) for v in cumulative]


def _hifi_refine_result(result: dict, *, bar_df, visible_start,
                         trading_days: int) -> dict:
    """Mass Builder §8.5 final step: re-run Hi-Fi Pass 2 on a search
    result's serialized trades, refresh KPIs + equity curve, and stamp
    a `hifi` summary so the UI can badge cards where Hi-Fi moved trades.

    Mutates `result` in place and returns it. On failure (logged
    warning), returns the result unchanged. Per-candidate try/except
    keeps a single bad refinement from killing the whole search.

    `bar_df` is the (symbol, tf) group's prepared df with indicator
    columns — needed for signal-exit refinement (Phase 1 of Hi-Fi,
    `_hifi_resolve_trades` line 740). For stop/target-only refinement
    `bar_df=None` would work, but most strategies don't know that in
    advance — always pass it.
    """
    import pandas as pd
    if result is None:
        return result
    cfg = result.get('config') or {}
    stored = result.get('stored_trades') or []
    if not stored:
        return result
    symbol = cfg.get('symbol')
    timeframe = cfg.get('timeframe', '1Min')
    try:
        trades_df = pd.DataFrame(stored)

        # Snapshot original timestamps for delta counting
        def _col(name):
            if name in trades_df.columns:
                return trades_df[name].astype(str).copy()
            return pd.Series([''] * len(trades_df))
        orig_entry = _col('entry_fill_ts')
        orig_exit = _col('exit_fill_ts')

        # Pass 2
        from api.services.backtest_service import _hifi_resolve_trades
        refined = _hifi_resolve_trades(
            trades_df, symbol, timeframe, bar_df=bar_df)

        # Trim to visible window — Pass 2 doesn't change the window but
        # the canonical post-refinement step trims for safety.
        if visible_start is not None:
            from strategy_data import trim_trades_to_visible
            refined = trim_trades_to_visible(refined, visible_start)

        # Count refinement deltas (compare strings, since timestamps
        # round-trip through ISO strings in stored_trades).
        entries_refined = 0
        exits_refined = 0
        signal_exits_refined = 0
        for i in range(min(len(refined), len(orig_entry))):
            row = refined.iloc[i]
            new_entry = str(row.get('entry_fill_ts') or '')
            new_exit = str(row.get('exit_fill_ts') or '')
            new_reason = str(row.get('exit_reason') or '')
            if new_entry and new_entry != orig_entry.iloc[i]:
                entries_refined += 1
            if new_exit and new_exit != orig_exit.iloc[i]:
                if new_reason in ('stop_loss', 'stop', 'target'):
                    exits_refined += 1
                else:
                    signal_exits_refined += 1
        total_refined = entries_refined + exits_refined + signal_exits_refined

        # Recompute KPIs from refined trades, using the visible-window
        # trading_days denominator (passed in by caller).
        from services import calculate_kpis
        new_kpis = calculate_kpis(
            refined,
            starting_balance=cfg.get('starting_balance', 10000),
            risk_per_trade=cfg.get('risk_per_trade', 100),
            total_trading_days=trading_days)

        bt_model = cfg.get('backtest_model') or 'rest_hifi'
        new_stored = _serialize_trades(refined, backtest_model=bt_model)
        new_eq = build_equity_curve(refined)

        result['kpis'] = new_kpis
        result['stored_trades'] = new_stored
        result['equity_curve'] = new_eq
        result['hifi'] = {
            'entries_refined': entries_refined,
            'exits_refined': exits_refined,
            'signal_exits_refined': signal_exits_refined,
            'total_refined': total_refined,
        }
    except Exception as e:
        logger.warning(
            "[MASS-HIFI] refinement failed for symbol=%s tf=%s: %s",
            symbol, timeframe, e, exc_info=True)
    return result


def format_time_estimate(seconds: float) -> str:
    """Format seconds into a human-readable time estimate."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def new_search_id() -> str:
    """Generate a unique mass search ID."""
    import uuid
    return f"ms_{uuid.uuid4().hex[:10]}"


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT TICKER LISTS (for quick-add buttons)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# COMBINATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_mass_search(
    search_config: dict,
    progress_callback: Callable[[int, int, str], None] = None,
    resume_checkpoint: Optional[dict] = None,
    checkpoint_callback: Optional[Callable[[list, list, dict], None]] = None,
    partial_callback: Optional[Callable[[list], None]] = None,
) -> list:
    """Execute a mass strategy search.

    Groups by (symbol, timeframe, session) to minimize expensive data loads.
    Within each group, iterates over (direction, entry, exit_combo, RM_pack)
    and runs confluence auto-search on each base config.

    Args:
        search_config: Mass search configuration dict.
        progress_callback: Called with (current_step, total_steps, label).
        resume_checkpoint: Optional dict with 'completed_symbol_tfs',
            'partial_results', 'diagnostics_so_far' to skip already-done
            work. Produced by checkpoint_callback on a prior interrupted
            run. See Roadmap 9s.
        checkpoint_callback: Called at the end of each (symbol, tf) group
            with (completed_symbol_tfs, results_snapshot, diag_snapshot)
            so the caller can persist a resumable snapshot to DB. No-op
            if None.

    Returns:
        List of result dicts sorted by daily_r descending.
    """
    import os
    import pandas as pd
    from services import (
        prepare_data_with_indicators, get_secondary_tf_map,
        calculate_kpis, count_trading_days, find_best_combinations,
        split_trades_at_boundary,
    )
    from unified_engine import precompute_bar_cache, run_trades_from_cache
    from alerts import _get_base_trigger_id as get_base_trigger_id

    # ── Search fidelity mode (Mass Builder §8.5) ──────────────────────
    # Rapid (default, current behavior): one engine run per
    #   (entry × exit × stop × target) combo, then post-hoc mask
    #   confluence candidates against that run's trades. Fast but
    #   structurally divergent from "Update All Data".
    # HighFidelity: build a per-(symbol, tf) cache once, replay the
    #   state machine per candidate with confluence inline. Each
    #   confluence combo gets its own engine-gated trades_df, bit-
    #   identical to a full `run_unified_backtest` with that
    #   confluence set. See plan: piped-wondering-tower.md.
    _search_fidelity = str(search_config.get('searchFidelity')
                            or search_config.get('search_fidelity')
                            or 'Rapid').strip()
    _hifi_mode = _search_fidelity == 'HighFidelity'
    # Hi-Fi Pass 2 timestamp refinement on top results. Plumbed through
    # now; the per-search top-K refinement is deferred to a follow-up
    # (see plan: piped-wondering-tower.md → "Hi-Fi Pass 2 — default
    # on, top-K only"). The structural fidelity win in this PR is the
    # cache-replay branch above; Hi-Fi only refines sub-second
    # L-type timestamps on top of that.
    _skip_hifi = bool(search_config.get('skipHifi', False))
    _hifi_buffer_mult = max(1, min(10, int(
        search_config.get('hifiBufferMultiplier', 3))))
    if _hifi_mode:
        logger.info("Mass search: HighFidelity mode (cache+replay per "
                    "confluence combo); skipHifi=%s hifi_buffer×=%d "
                    "(Hi-Fi top-K refinement: follow-up)",
                    _skip_hifi, _hifi_buffer_mult)
    # Per-(symbol, tf) cache holder. Lazily populated on first
    # HighFidelity request inside the group; freed when the group
    # ends.
    _group_cache_holder: dict = {}
    # Hi-Fi top-K context (Mass Builder §8.5). Populated alongside
    # _data_cache; read at the end of the search by _hifi_refine_result.
    # Tuple: (bar_df, visible_df, visible_start, trading_days). Keyed
    # by (symbol, tf).
    _hifi_group_ctx: dict = {}
    from data_loader import get_data_source
    from confluence_groups import (
        get_enabled_groups, get_all_triggers, load_confluence_groups,
        TEMPLATES as _CONF_TEMPLATES, expand_direction_to_states,
    )
    from risk_management_packs import load_risk_management_packs
    from unified_engine import run_unified_backtest
    import general_packs as gp_module

    enabled_groups = get_enabled_groups(load_confluence_groups())
    all_trigger_defs = get_all_triggers(enabled_groups)
    all_rm_packs = load_risk_management_packs()
    rm_pack_map = {p.id: p for p in all_rm_packs}
    enabled_gen = gp_module.get_enabled_general_packs(gp_module.load_general_packs())
    data_feed = os.getenv('ALPACA_DATA_FEED', 'sip')

    tickers = search_config.get('tickers', [])
    timeframes = search_config.get('timeframes', ['1Min'])
    directions = search_config.get('directions', ['LONG'])
    entry_cids = search_config.get('entry_triggers', [])
    exit_cids_raw = search_config.get('exit_triggers', [])
    exit_depth = search_config.get('exit_depth', 1)
    date_range = search_config.get('date_range', {'mode': 'days', 'days': 90})
    required_perf = search_config.get('required_performance', {})
    max_results = search_config.get('max_results', 500)
    tf_conf_depth = search_config.get('tf_confluence_depth', 2)
    gen_conf_depth = search_config.get('general_confluence_depth', 1)
    tf_conf_ids = search_config.get('tf_confluences', []) or []
    gen_conf_ids = search_config.get('general_confluences', []) or []
    # ✱ requirements (2026-07-02): confluences every combo must satisfy.
    # The frontend keeps these disjoint from the optional pools above.
    required_tf_ids = search_config.get('required_tf_confluences', []) or []
    required_gen_ids = search_config.get('required_general_confluences', []) or []

    # ── Out-of-Sample gate config (docs/Spec_OOS_Test_Periods.md §9) ──
    # When enabled, every combo is backtested through today and split at
    # `in_sample_end`: ranked on the in-sample window, gated additionally
    # on the OOS window. Disabled (or no in_sample_end) → behaves as before.
    oos_cfg = search_config.get('oos', {}) or {}
    oos_in_sample_end = oos_cfg.get('in_sample_end')
    oos_enabled = bool(oos_cfg.get('enabled')) and bool(oos_in_sample_end)
    oos_required = oos_cfg.get('required_performance', {}) or {}
    oos_sigma_cfg = oos_cfg.get('sigma', {}) or {}
    oos_in_sample_end_dt = None
    if oos_enabled:
        try:
            oos_in_sample_end_dt = datetime.fromisoformat(
                str(oos_in_sample_end).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            logger.warning("Mass search: bad oos.in_sample_end=%r — OOS off",
                            oos_in_sample_end)
            oos_enabled = False

    # Resolve synthetic confluence IDs into label suffixes for Layer 2
    # filtering. TF IDs have format "_TF_-{GROUP_ID}-{BULL|BEAR}-{fidelity}";
    # expand each to the real interpreter states via template outputs.
    # General IDs have format "GEN-{PACK_ID}-{STATE}" and match records directly.
    # Empty selections → skip confluence search entirely (no Layer 2).
    group_by_upper_id = {g.id.upper(): g for g in enabled_groups}

    def _resolve_tf_id_labels(syn) -> set:
        """One `_TF_-…` UI id → the set of `{interp}-{state}` label suffixes."""
        labels: set = set()
        if not isinstance(syn, str) or not syn.startswith('_TF_-'):
            return labels
        # Strip prefix and optional trailing fidelity (e.g. "-PB", "-CB")
        body = syn[len('_TF_-'):]
        parts = body.rsplit('-', 2) if body.count('-') >= 2 else body.rsplit('-', 1)
        # body forms: "GROUP_ID-DIR" or "GROUP_ID-DIR-FIDELITY"
        if len(parts) == 3:
            group_id_up, direction, _fidelity = parts
        elif len(parts) == 2:
            group_id_up, direction = parts
        else:
            return labels
        group = group_by_upper_id.get(group_id_up)
        if not group:
            return labels
        tmpl = _CONF_TEMPLATES.get(group.base_template)
        if not tmpl:
            return labels
        outputs = tmpl.get('outputs', [])
        for interp in tmpl.get('interpreters', []):
            # `direction` is either a direction keyword (BULL/BEAR/NEUTRAL)
            # or a real state code (e.g. "SML", "H+up"). Treat as a state
            # when it's one of the template's outputs, else expand via the
            # direction map.
            if direction in outputs:
                labels.add(f"{interp}-{direction}")
            else:
                for state in expand_direction_to_states(interp, direction):
                    labels.add(f"{interp}-{state}")
        return labels

    allowed_labels: Optional[set] = None
    skip_confluence_search = not (tf_conf_ids or gen_conf_ids
                                  or required_tf_ids or required_gen_ids)
    # ✱ requirements → one alternative-set per selection (AND across
    # selections; OR within — find_best_combinations resolves each set to
    # concrete records). An id that resolves to nothing becomes an empty
    # set, which find_best_combinations fails loudly on — never silently
    # broader.
    required_groups: list = []
    for syn in required_tf_ids:
        grp = _resolve_tf_id_labels(syn)
        if not grp:
            print(f"[MASS] required TF confluence {syn!r} did not resolve — "
                  f"combos for affected configs will fail loudly", flush=True)
        required_groups.append(grp)
    for syn in required_gen_ids:
        if isinstance(syn, str) and syn.startswith('GEN-'):
            required_groups.append({syn})
        else:
            print(f"[MASS] required general confluence {syn!r} invalid — "
                  f"combos for affected configs will fail loudly", flush=True)
            required_groups.append(set())
    if not skip_confluence_search:
        allowed_labels = set()
        for syn in tf_conf_ids:
            allowed_labels |= _resolve_tf_id_labels(syn)
        for syn in gen_conf_ids:
            if isinstance(syn, str) and syn.startswith('GEN-'):
                allowed_labels.add(syn)
        # Safety: user selected items but nothing resolved (e.g., stale group IDs).
        # Treat as empty selection → skip confluence search rather than falling
        # back to explore-all (which would surprise the user with a huge run).
        # (Required-only searches legitimately have an empty optional pool —
        # the depth-0 required baseline row is still produced.)
        if not allowed_labels and not required_groups:
            skip_confluence_search = True
            allowed_labels = None
            print("[MASS] confluence selections did not resolve to any labels; "
                  "skipping confluence search", flush=True)
        else:
            # NOTE: an EMPTY allowed set (required-only search) is passed
            # through as-is — find_best_combinations treats it as an empty
            # optional pool (only the required baseline row), NOT explore-all.
            print(f"[MASS] confluence filter active with "
                  f"{len(allowed_labels or set())} allowed labels "
                  f"(tf_selections={len(tf_conf_ids)}, "
                  f"gen_selections={len(gen_conf_ids)}, "
                  f"required={len(required_groups)})", flush=True)

    # Resolve stop/target/time_exit packs into config lists
    # If pack IDs are provided, resolve each to its config. Otherwise use default.
    stop_configs = [{"method": "atr", "atr_mult": 1.5}]  # default
    target_configs = [None]  # default (signal exit only)
    time_exit_configs = [None]  # default (no time exit)

    stop_pack_ids = search_config.get('stop_packs', [])
    target_pack_ids = search_config.get('target_packs', [])
    time_exit_pack_ids = search_config.get('time_exit_packs', [])

    if stop_pack_ids:
        stop_configs = []
        for pid in stop_pack_ids:
            p = rm_pack_map.get(pid)
            if p:
                stop_configs.append(p.get_stop_config())
        if not stop_configs:
            stop_configs = [{"method": "atr", "atr_mult": 1.5}]

    if target_pack_ids:
        target_configs = []
        for pid in target_pack_ids:
            p = rm_pack_map.get(pid)
            if p:
                tc = p.get_target_config()
                if tc:
                    target_configs.append(tc)
        if not target_configs:
            target_configs = [None]

    if time_exit_pack_ids:
        from time_exit_packs import load_time_exit_packs
        te_packs = load_time_exit_packs()
        te_map = {p.id: p for p in te_packs}
        time_exit_configs = []
        for pid in time_exit_pack_ids:
            p = te_map.get(pid)
            if p:
                time_exit_configs.append(p.get_exit_config())
        if not time_exit_configs:
            time_exit_configs = [None]

    # For backward compat, also accept single stop/target config
    if not stop_pack_ids and search_config.get('stop_config'):
        stop_configs = [search_config['stop_config']]
    if not target_pack_ids and search_config.get('target_config'):
        target_configs = [search_config['target_config']]

    # Resolve date range
    data_days = date_range.get('days', 90)
    start_date = date_range.get('start')
    end_date = date_range.get('end')
    data_seed = 42

    # OOS: the out-of-sample band must have bars. In days-mode (the
    # Mass Builder default — end_date is None) the loader already loads
    # the last `data_days` days through today, so the OOS window
    # [in_sample_end, today] is already covered. Only when an explicit
    # end_date predates today do we extend it — never set end_date while
    # start_date is None (that breaks the days-mode load → 0 bars).
    if oos_enabled and end_date:
        _today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if str(end_date)[:10] < _today:
            logger.info("Mass search OOS: extending end_date %s -> %s",
                        end_date, _today)
            end_date = _today
    if oos_enabled:
        logger.info("Mass search OOS: in_sample_end=%s", oos_in_sample_end)

    logger.info("Mass search config: data_days=%d, date_range=%s, start=%s, end=%s",
                data_days, date_range, start_date, end_date)

    # Resolve session
    session = search_config.get('session', 'RTH')

    # Build exit trigger combinations
    exit_combos = generate_exit_combos(exit_cids_raw, exit_depth)
    if not exit_combos or exit_combos == [[]]:
        exit_combos = [[]]

    # Pre-resolve ALL trigger base IDs for the mega-config
    print(f"[MASS] entry_cids={entry_cids}, exit_cids_raw={exit_cids_raw}")
    # Debug logging to file (daemon threads don't reliably flush stdout)
    with open('/tmp/mass_debug.log', 'w') as _dbg:
        _dbg.write(f"entry_cids={entry_cids}\n")
        _dbg.write(f"exit_cids_raw={exit_cids_raw}\n")
        _dbg.write(f"trigger_count={len(all_trigger_defs)}\n")
        _dbg.write(f"enabled_groups={[g.id for g in enabled_groups]}\n")
        for cid in entry_cids:
            _dbg.write(f"entry '{cid}' in triggers: {cid in all_trigger_defs}\n")
        for cid in exit_cids_raw:
            _dbg.write(f"exit '{cid}' in triggers: {cid in all_trigger_defs}\n")
    all_entry_bases = {}
    all_entry_names = {}
    for cid in entry_cids:
        tdef = all_trigger_defs.get(cid)
        if tdef:
            all_entry_bases[cid] = get_base_trigger_id(cid)
            all_entry_names[cid] = tdef.name

    all_exit_bases = {}
    all_exit_names = {}
    for cid in exit_cids_raw:
        tdef = all_trigger_defs.get(cid)
        if tdef:
            all_exit_bases[cid] = get_base_trigger_id(cid)
            all_exit_names[cid] = tdef.name

    # Collect ALL unique base trigger IDs for the mega bar_cache
    _all_base_triggers = set(all_entry_bases.values()) | set(all_exit_bases.values())

    # ── #21 prep-scoping for Mass Builder (memory hardening) ──────────────
    # The per-(symbol,tf) df is otherwise baked with ALL enabled confluence
    # groups (e.g. 851 cols) regardless of what this search uses → the dough
    # copies that wide per-bar state into one object per bar → multi-trigger
    # / long-window runs blow past RAM and get OOM-killed (losing the whole
    # run). Scope prep to the UNION of groups any selected trigger/confluence
    # needs — a provable superset, byte-identical (validated by the Fidelity
    # Gate + scoped-vs-unscoped repro). Mass-builder-only for now via
    # force_scope; global enablement is a separate post-Monday change.
    # Kill-switch: RORT_MASS_SCOPE_CONFLUENCE=0 → full/unscoped (old behavior).
    _mass_scope_on = os.getenv('RORT_MASS_SCOPE_CONFLUENCE', '1') == '1'
    _search_required_groups = None
    if _mass_scope_on:
        try:
            from unified_engine import resolve_required_confluence_groups
            _union_strat = {
                'symbol': tickers[0] if tickers else None,
                'timeframe': timeframes[0] if timeframes else '1Min',
                'entry_trigger': next(iter(_all_base_triggers), None),
                'exit_triggers': sorted(_all_base_triggers),
                # ✱-required general confluences must be baked into prep
                # too, or their records never exist and every combo fails.
                'confluence': [], 'general_confluences': list(gen_conf_ids) + list(required_gen_ids),
            }
            _req = set(resolve_required_confluence_groups(_union_strat, enabled_groups))
            # Selected TF-confluence groups: their synthetic _TF_- IDs don't
            # flow through resolve_strategy_requirements, so add them directly.
            # (✱-required TF ids included for the same reason as above.)
            _by_upper = {g.id.upper(): g for g in enabled_groups}
            for syn in list(tf_conf_ids) + list(required_tf_ids):
                if isinstance(syn, str) and syn.startswith('_TF_-'):
                    body = syn[len('_TF_-'):]
                    gid_up = (body.rsplit('-', 2) if body.count('-') >= 2
                              else body.rsplit('-', 1))[0]
                    g = _by_upper.get(gid_up)
                    if g:
                        _req.add(g.id)
            _search_required_groups = _req
            print(f"[MASS] prep-scope ON: {len(_search_required_groups)}/"
                  f"{len(enabled_groups)} groups -> {sorted(_search_required_groups)}",
                  flush=True)
        except Exception as _exc:
            logger.warning("Mass search: prep-scope resolve failed (%s) — "
                           "full/unscoped", _exc)
            _search_required_groups = None

    # Build pack combos (stop × target × time_exit)
    pack_combos = [(sc, tc, tec)
                   for sc in stop_configs
                   for tc in target_configs
                   for tec in time_exit_configs]

    # Count total steps for progress: data load groups + backtest combos
    n_data_groups = len(tickers) * len(timeframes)
    n_backtest_combos = (len(tickers) * len(timeframes) * len(directions)
                         * len(entry_cids) * len(exit_combos) * len(pack_combos))
    _n_total_bts = n_backtest_combos  # used in phase labels
    total_steps = n_data_groups + n_backtest_combos
    step = 0
    results = []
    # 1.16 mid-flight persistence throttle: flush results-so-far to the DB at
    # most this often (env-tunable; 0 disables). Resume is unaffected — this
    # writes the `results` field, resume reads the separate `checkpoint`.
    _last_partial_flush = _time.monotonic()
    _partial_flush_sec = float(os.getenv('RORT_MASS_PARTIAL_FLUSH_SEC', '30'))
    min_trades = required_perf.get('min_trades', 10)
    print(f"[MASS] total_steps={total_steps}, tickers={len(tickers)}, tfs={len(timeframes)}, dirs={len(directions)}, entries={len(entry_cids)}, exits={len(exit_combos)}, packs={len(pack_combos)}")
    print(f"[MASS] all_entry_bases={all_entry_bases}")
    print(f"[MASS] all_exit_bases={all_exit_bases}")
    print(f"[MASS] data_days={data_days}, session={session}")
    # Diagnostics counters — includes per-phase timing for estimate calibration
    _diag = {
        'data_loads': 0, 'data_failures': 0,
        'backtests_run': 0, 'backtests_failed': 0,
        'trigger_bt_total_sec': 0.0,   # cumulative wall time in run_unified_backtest
        'conf_search_total_sec': 0.0,  # cumulative wall time in find_best_combinations
        'conf_combos_total': 0,        # total confluence combinations evaluated
        'overall_start': _time.monotonic(),
        'combos_with_trades': 0, 'combos_zero_trades': 0,
        'combos_below_min': 0, 'combos_passed_perf': 0,
        'confluence_results': 0, 'direction_skips': 0,
    }

    # Resume-from-checkpoint bookkeeping (Roadmap 9s). When resume_checkpoint
    # is provided, seed the completed-groups set and restore prior results +
    # diagnostics so the overall run appears continuous.
    completed_symbol_tfs: set = set()
    _completed_symbol_tfs_list: list = []
    _combos_per_group = (len(directions) * len(entry_cids)
                         * len(exit_combos) * len(pack_combos))
    _steps_per_group = 1 + _combos_per_group  # 1 data-load step + combos
    if resume_checkpoint:
        raw_completed = resume_checkpoint.get('completed_symbol_tfs') or []
        for pair in raw_completed:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                completed_symbol_tfs.add((str(pair[0]), str(pair[1])))
                _completed_symbol_tfs_list.append([str(pair[0]), str(pair[1])])
        prior_results = resume_checkpoint.get('partial_results') or []
        if isinstance(prior_results, list):
            results.extend(prior_results)
        prior_diag = resume_checkpoint.get('diagnostics_so_far') or {}
        if isinstance(prior_diag, dict):
            for k, v in prior_diag.items():
                # Don't overwrite overall_start — wall_total calculated on
                # final diag will reflect only this process's runtime, which
                # is fine for calibration (resume-time != original run time).
                if k == 'overall_start':
                    continue
                _diag[k] = v
        print(f"[MASS] resume: {len(completed_symbol_tfs)} completed groups, "
              f"{len(results)} prior results, {_diag.get('backtests_run', 0)} "
              f"backtests already run")

    logger.info("Mass search: %d tickers × %d TFs × %d dirs × %d entries "
                "× %d exit_combos = %d base configs",
                len(tickers), len(timeframes), len(directions),
                len(entry_cids), len(exit_combos), total_steps)

    for symbol in tickers:
        asset_type = 'crypto' if '/' in symbol else 'equity'
        sym_session = '24/7' if asset_type == 'crypto' else session

        for tf in timeframes:
            # Resume: skip (symbol, tf) groups already completed on a prior
            # run. Fast-forward the step counter so progress bar math stays
            # correct.
            if (symbol, tf) in completed_symbol_tfs:
                step += _steps_per_group
                if progress_callback:
                    progress_callback(step, total_steps,
                                      f"{symbol} {tf} — resumed (skipped)",
                                      phase='load',
                                      phase_detail=f"{symbol} {tf} skipped (resume)",
                                      inner_step=1, inner_total=1)
                continue

            # ── Level 1: Load data (one per symbol+TF, cached) ──
            ck = _cache_key(symbol, tf, data_days, sym_session,
                            start_date, end_date)
            cached = _data_cache.get(ck)
            if cached:
                # Cache tuple was extended to (df, sec_tf_map,
                # period_trading_days, visible_start). Old 3-tuples
                # gracefully fall back to df.index[0].
                if len(cached) == 4:
                    df, sec_tf_map, period_trading_days, _search_visible_start = cached
                else:
                    df, sec_tf_map, period_trading_days = cached
                    _search_visible_start = (df.index[0]
                                              if len(df) > 0
                                              else None)
                # Also stash on the Hi-Fi context map for end-of-search
                # refinement. Recompute the visible df since cached
                # tuple doesn't carry it (avoids a memory dup; cheap).
                from strategy_data import trim_df_to_visible as _trim_vis
                _hifi_visible_df = (_trim_vis(df, _search_visible_start)
                                     if _search_visible_start is not None
                                     else df)
                _hifi_group_ctx[(symbol, tf)] = (
                    df, _hifi_visible_df, _search_visible_start,
                    period_trading_days,
                )
                logger.info("Mass search: %s/%s — cache hit (%d bars)",
                            symbol, tf, len(df))
                if progress_callback:
                    progress_callback(step, total_steps,
                                      f"{symbol} {tf} (cached)",
                                      phase='load',
                                      phase_detail=f"{symbol} {tf} (cached)",
                                      inner_step=1, inner_total=1)
            else:
                if progress_callback:
                    progress_callback(step, total_steps,
                                      f"Loading {symbol} {tf} data...",
                                      phase='load',
                                      phase_detail=f"Loading {symbol} {tf}",
                                      inner_step=0, inner_total=1)
                try:
                    from db import load_settings_db
                    from data_loader import SUB_MINUTE_TIMEFRAMES
                    from strategy_data import load_strategy_data
                    import pandas as _pd
                    _enabled_tfs = set(load_settings_db().get("enabled_timeframes", ["1Min"]))
                    sec_tfs = tuple(sorted(_enabled_tfs - {tf} - SUB_MINUTE_TIMEFRAMES))
                    # Route through the unified helper so Mass Builder's
                    # visible window + warmup match Update All Data /
                    # Strategy Detail by construction. The synth strat
                    # encodes the search's intended window so the helper
                    # resolves it as the canonical anchor — saved
                    # strategies will inherit this same window via
                    # `backtest_start_date` (stamped by
                    # build_strategy_config), so search KPIs == saved KPIs.
                    _synth_strat = {
                        'id': f'_search_{symbol}_{tf}',
                        'symbol': symbol, 'timeframe': tf,
                        'trading_session': sym_session,
                        'data_seed': data_seed,
                    }
                    if start_date and end_date:
                        _synth_strat['lookback_mode'] = 'Date Range'
                        _synth_strat['lookback_start_date'] = (
                            start_date.isoformat()
                            if hasattr(start_date, 'isoformat')
                            else str(start_date))
                        _synth_strat['lookback_end_date'] = (
                            end_date.isoformat()
                            if hasattr(end_date, 'isoformat')
                            else str(end_date))
                    else:
                        # Days-mode: synth's `backtest_start_date` =
                        # `now - data_days` so the helper resolves to
                        # the same window saved strategies use.
                        _synth_strat['backtest_start_date'] = (
                            _pd.Timestamp.now(tz='UTC')
                            - _pd.Timedelta(days=data_days)).isoformat()
                    _bundle = load_strategy_data(
                        _synth_strat,
                        data_feed=data_feed,
                        secondary_tfs_override=sec_tfs,
                        required_confluence_ids=_search_required_groups,
                        force_scope=(_search_required_groups is not None),
                    )
                    df = _bundle.df
                    # Stash visible_start for downstream trim — every
                    # trade emitted by this search must be trimmed to
                    # this anchor so search KPIs match the saved
                    # strategy's KPIs (which use the same anchor via
                    # backtest_start_date).
                    _search_visible_start = _bundle.visible_start
                except Exception as exc:
                    _diag['data_failures'] += 1
                    logger.warning("Mass search: data load failed %s/%s: %s",
                                   symbol, tf, exc)
                    # Skip data load step + all backtest combos for this group
                    step += 1 + (len(directions) * len(entry_cids)
                                 * len(exit_combos) * len(pack_combos))
                    if progress_callback:
                        progress_callback(step, total_steps,
                                          f"{symbol} {tf} — skipped",
                                          phase='load',
                                          phase_detail=f"{symbol} {tf} skipped")
                    continue

                if len(df) == 0:
                    logger.info("Mass search: %s/%s — 0 bars, skipping",
                                symbol, tf)
                    step += 1 + (len(directions) * len(entry_cids)
                                 * len(exit_combos) * len(pack_combos))
                    if progress_callback:
                        progress_callback(step, total_steps,
                                          f"{symbol} {tf} — no data",
                                          phase='load',
                                          phase_detail=f"{symbol} {tf} no data")
                    continue

                sec_tf_map = get_secondary_tf_map(df)
                # period_trading_days counts trading days in the VISIBLE
                # window only (excludes warmup). The full df spans
                # [warmup_start, visible_end] but the daily_r denominator
                # should reflect only the user-visible window.
                from strategy_data import trim_df_to_visible
                _visible_df = trim_df_to_visible(df, _search_visible_start)
                period_trading_days = count_trading_days(_visible_df)
                _cache_put(ck, df, sec_tf_map, period_trading_days,
                            _search_visible_start)
                # Stash for Hi-Fi top-K refinement at end-of-search.
                _hifi_group_ctx[(symbol, tf)] = (
                    df, _visible_df, _search_visible_start,
                    period_trading_days,
                )

            # Count data load as a progress step
            step += 1
            if progress_callback:
                progress_callback(step, total_steps,
                                  f"{symbol} {tf} data ready",
                                  phase='prep',
                                  phase_detail=f"Preparing indicators — {symbol} {tf}",
                                  inner_step=1, inner_total=1)
            _diag['data_loads'] += 1
            # Pin backtest dates from actual DataFrame
            _bt_start_iso = df.index[0].isoformat()
            _bt_end_iso = df.index[-1].isoformat()
            logger.info("Mass search: %s/%s — %d bars loaded (%s to %s)",
                        symbol, tf, len(df), _bt_start_iso[:10],
                        _bt_end_iso[:10])

            # HighFidelity: clear the per-group cache from the previous
            # (symbol, tf) group so the next group gets a fresh build.
            # The cache itself is built lazily on first use below.
            _group_cache_holder.clear()

            # OOS: split trading-day counts at the in-sample boundary so
            # each side's daily-R uses the right denominator.
            is_trading_days = period_trading_days
            oos_trading_days = None
            if oos_enabled:
                _b = pd.Timestamp(oos_in_sample_end_dt)
                _itz = getattr(df.index, 'tz', None)
                if _itz is not None and _b.tzinfo is None:
                    _b = _b.tz_localize(_itz)
                elif _itz is None and _b.tzinfo is not None:
                    _b = _b.tz_localize(None)
                is_trading_days = count_trading_days(df[df.index < _b])
                oos_trading_days = count_trading_days(df[df.index >= _b])

            for direction in directions:
                for entry_cid in entry_cids:
                    entry_tdef = all_trigger_defs.get(entry_cid)
                    if not entry_tdef:
                        step += len(exit_combos) * len(pack_combos)
                        continue
                    entry_base = all_entry_bases.get(entry_cid, '')
                    entry_name = all_entry_names.get(entry_cid, '?')

                    # Direction compatibility
                    if (entry_tdef.direction != 'BOTH'
                            and entry_tdef.direction != direction):
                        _diag['direction_skips'] += len(exit_combos) * len(pack_combos)
                        step += len(exit_combos) * len(pack_combos)
                        continue

                    for exit_combo in exit_combos:
                        exit_bases = []
                        exit_names = []
                        exit_cid_list = []
                        bar_count_exit_value = None
                        valid_exits = True
                        for ecid in exit_combo:
                            # Separate bar_count exits from signal exits
                            # (same logic as Strategy Builder lines 5088-5105)
                            _is_bar_count = False
                            for _g in enabled_groups:
                                if (_g.get_trigger_id("exit") == ecid
                                        and _g.base_template == "bar_count"):
                                    bar_count_exit_value = _g.parameters.get(
                                        "candle_count", 4)
                                    _is_bar_count = True
                                    break
                            if _is_bar_count:
                                continue  # Don't add to signal exits
                            eb = all_exit_bases.get(ecid)
                            en = all_exit_names.get(ecid)
                            if eb is None:
                                valid_exits = False
                                break
                            exit_bases.append(eb)
                            exit_names.append(en)
                            exit_cid_list.append(ecid)

                        # Need at least signal exits OR bar_count exit
                        if not valid_exits:
                            step += len(pack_combos)
                            continue
                        if not exit_bases and bar_count_exit_value is None:
                            step += len(pack_combos)
                            continue

                        for pack_sc, pack_tc, pack_tec in pack_combos:
                            step += 1
                            label = f"{symbol} {tf} {direction}"
                            _bt_idx = _diag['backtests_run'] + 1
                            _bt_detail = (f"BT {_bt_idx}/{_n_total_bts} · "
                                          f"{symbol} {tf} {direction} · "
                                          f"{entry_base} → "
                                          f"{'+'.join(exit_bases) if exit_bases else f'bar_count_{bar_count_exit_value}'}")

                            config = build_strategy_config(
                                symbol=symbol, timeframe=tf,
                                direction=direction, session=sym_session,
                                entry_cid=entry_cid, entry_base=entry_base,
                                entry_name=entry_name,
                                exit_cids=exit_cid_list,
                                exit_bases=exit_bases,
                                exit_names=exit_names,
                                bar_count_exit=bar_count_exit_value,
                                stop_config=pack_sc,
                                target_config=pack_tc,
                                date_range=date_range,
                                asset_type=asset_type,
                                backtest_start_date=_bt_start_iso,
                                backtest_end_date=_bt_end_iso,
                                time_exit_config=pack_tec,
                                # Model fields (algo_model split — 2026-05-08).
                                # Spawned strategies inherit explicit user
                                # choices from the Mass Builder spec form.
                                backtest_model=search_config.get('backtest_model'),
                                algo_model=search_config.get('algo_model'),
                                live_model=search_config.get('live_model'),
                                # OOS: stamp the in-sample window onto the
                                # spawned strategy so its chart bands + KPI
                                # split know where the held-out region is.
                                in_sample_start=(start_date if oos_enabled else None),
                                in_sample_end=(oos_in_sample_end if oos_enabled else None),
                            )

                            # ── Level 2: Run full backtest ──
                            _diag['backtests_run'] += 1
                            _bt_t0 = _time.monotonic()
                            # Bar-progress callback — fires every 1000 bars or
                            # 250ms. Feeds bottom bar fine-grain updates.
                            def _bar_progress(bar_n, total_bars,
                                              _s=step, _t=total_steps,
                                              _l=label, _d=_bt_detail):
                                if progress_callback:
                                    progress_callback(_s, _t, _l,
                                                      phase='backtest',
                                                      phase_detail=_d,
                                                      inner_step=bar_n,
                                                      inner_total=total_bars)
                            try:
                                if _hifi_mode:
                                    # HighFidelity §8.5: build per-group
                                    # cache once (superset of all triggers
                                    # used by any candidate in the group),
                                    # then run_trades_from_cache for this
                                    # specific candidate config. Indicator
                                    # work happens once, not N times.
                                    if 'cache' not in _group_cache_holder:
                                        _superset = dict(config)
                                        _all_entry_bases = sorted({
                                            all_entry_bases[cid]
                                            for cid in entry_cids
                                            if all_entry_bases.get(cid)
                                        })
                                        _all_exit_bases = sorted({
                                            all_exit_bases[ecid]
                                            for _combo in exit_combos
                                            for ecid in _combo
                                            if all_exit_bases.get(ecid)
                                        })
                                        # Cover every entry + every exit
                                        # by stuffing them into the
                                        # superset's exit_triggers — that
                                        # field drives the trigger
                                        # evaluator's required_triggers set
                                        # (resolve_strategy_requirements).
                                        _superset['exit_triggers'] = sorted(
                                            set(_all_entry_bases) |
                                            set(_all_exit_bases))
                                        # CRITICAL: _resolve_trigger_ids reads
                                        # exit_trigger_confluence_ids FIRST and
                                        # only falls back to exit_triggers when
                                        # it's empty. The combo config we copied
                                        # carries the FIRST combo's exit CIDs, so
                                        # without clearing them the dough would
                                        # track only the first combo's exit (+
                                        # entry) — every other pack's trigger
                                        # (rvol_v2, supertrend, …) silently
                                        # never gets cached -> "not in cache"
                                        # failures that drop whole entry/exit
                                        # families. Clear the CID fields so the
                                        # stuffed exit_triggers superset wins.
                                        _superset['exit_trigger_confluence_ids'] = []
                                        _superset['exit_trigger_confluence_id'] = None
                                        _superset['confluence'] = []
                                        _superset['general_confluences'] = []
                                        _cache, _meta = precompute_bar_cache(
                                            df, _superset,
                                            general_packs=enabled_gen,
                                            secondary_tf_map=sec_tf_map if sec_tf_map else None)
                                        _group_cache_holder['cache'] = _cache
                                        _group_cache_holder['meta'] = _meta
                                        # Tier 2: persist this group's dough so a
                                        # later save can re-bake the lane (gated
                                        # consumer in forward_test_service) instead
                                        # of a full recompute. One dough per
                                        # (symbol,tf,window); best-effort — a
                                        # failure never breaks the search.
                                        try:
                                            import bar_cache_store as _bcs
                                            from datetime import datetime as _dt, timezone as _tz
                                            _dkey = _bcs.dough_key(config)
                                            if _dkey:
                                                _m2 = dict(_meta or {})
                                                # Visible-window trading days
                                                # (line 1004) — NOT the full-df
                                                # warmup-inclusive count — so the
                                                # consumer's KPIs match.
                                                _m2['_trading_days'] = period_trading_days
                                                # Visible_start so the consumer can
                                                # trim the re-bake to the visible
                                                # window (dough covers warmup+visible;
                                                # the lane is visible-only).
                                                _m2['_visible_start'] = (
                                                    _search_visible_start.isoformat()
                                                    if _search_visible_start is not None
                                                    else None)
                                                _bcs.put_dough(_dkey, _cache, _m2,
                                                               _dt.now(_tz.utc).isoformat())
                                        except Exception:
                                            pass
                                    trades_df = run_trades_from_cache(
                                        _group_cache_holder['cache'],
                                        config,
                                        _group_cache_holder['meta'])
                                else:
                                    trades_df, _ = run_unified_backtest(
                                        df, config,
                                        general_packs=enabled_gen,
                                        secondary_tf_map=sec_tf_map if sec_tf_map else None,
                                        progress_cb=_bar_progress)
                                _diag['trigger_bt_total_sec'] += _time.monotonic() - _bt_t0
                            except _CancelledError:
                                # Cancellation must propagate — the outer handler
                                # in start_mass_search_async sets status=cancelled
                                # and exits the thread cleanly.
                                _diag['trigger_bt_total_sec'] += _time.monotonic() - _bt_t0
                                raise
                            except Exception as exc:
                                _diag['trigger_bt_total_sec'] += _time.monotonic() - _bt_t0
                                _diag['backtests_failed'] += 1
                                logger.warning("Mass search: backtest failed "
                                               "%s %s %s entry=%s exit=%s: %s",
                                               symbol, tf, direction,
                                               entry_base, exit_bases, exc)
                                if progress_callback:
                                    progress_callback(step, total_steps, label)
                                continue

                            if not isinstance(trades_df, pd.DataFrame):
                                trades_df = pd.DataFrame()
                            # Trim base trades to the visible window so
                            # (a) find_best_combinations only enumerates
                            # confluence records from visible-window
                            # bars, and (b) KPIs/equity downstream reflect
                            # only the user-visible window. The cache
                            # itself spans warmup+visible so indicators
                            # are properly warm.
                            if _search_visible_start is not None and len(trades_df) > 0:
                                from strategy_data import trim_trades_to_visible
                                trades_df = trim_trades_to_visible(
                                    trades_df, _search_visible_start)

                            n_trades = len(trades_df)

                            if step <= 3:
                                # Detailed debug for first few combos
                                logger.info(
                                    "Mass search: step %d CONFIG: entry=%s "
                                    "entry_cid=%s exit=%s exit_cids=%s "
                                    "stop=%s target=%s direction=%s",
                                    step, config.get('entry_trigger'),
                                    config.get('entry_trigger_confluence_id'),
                                    config.get('exit_triggers'),
                                    config.get('exit_trigger_confluence_ids'),
                                    config.get('stop_config'),
                                    config.get('target_config'),
                                    config.get('direction'))
                                if n_trades > 0:
                                    _sample = trades_df.head(3)
                                    for _, t in _sample.iterrows():
                                        logger.info(
                                            "  Trade: entry=%s r=%.2f win=%s "
                                            "exit_reason=%s",
                                            t.get('entry_trigger', '?'),
                                            t.get('r_multiple', 0),
                                            t.get('win', '?'),
                                            t.get('exit_reason', '?'))
                            if step <= 5 or (step % 50 == 0):
                                logger.info(
                                    "Mass search: step %d/%d %s %s %s "
                                    "entry=%s exit=%s → %d trades",
                                    step, total_steps, symbol, tf, direction,
                                    entry_base, exit_bases, n_trades)

                            if n_trades == 0:
                                _diag['combos_zero_trades'] += 1
                            elif n_trades < min_trades:
                                _diag['combos_below_min'] += 1
                            else:
                                _diag['combos_with_trades'] += 1

                            if n_trades < min_trades:
                                if progress_callback:
                                    progress_callback(step, total_steps, label)
                                continue

                            # KPIs on base trades (no confluence filter).
                            # OOS on: rank on the in-sample window, gate
                            # additionally on the out-of-sample window.
                            _start_bal = config.get('starting_balance', 10000)
                            _rpt = config.get('risk_per_trade', 100)
                            if oos_enabled:
                                _is_df, _oos_df = split_trades_at_boundary(
                                    trades_df, oos_in_sample_end_dt)
                                base_kpis = calculate_kpis(
                                    _is_df, starting_balance=_start_bal,
                                    risk_per_trade=_rpt,
                                    total_trading_days=is_trading_days)
                                _oos_eval = evaluate_oos(
                                    _is_df, _oos_df, oos_required, oos_sigma_cfg,
                                    starting_balance=_start_bal, risk_per_trade=_rpt,
                                    oos_trading_days=oos_trading_days)
                                _gate_ok = (meets_required_performance(
                                    base_kpis, required_perf)
                                    and _oos_eval['passed'])
                            else:
                                base_kpis = calculate_kpis(
                                    trades_df, starting_balance=_start_bal,
                                    risk_per_trade=_rpt,
                                    total_trading_days=period_trading_days)
                                _oos_eval = None
                                _gate_ok = meets_required_performance(
                                    base_kpis, required_perf)

                            _diag['combos_passed_perf'] += 1
                            # ✱ requirements suppress the ungated base row —
                            # a no-confluence strategy can't satisfy them;
                            # the depth-0 required-baseline row from Level 3
                            # is its replacement.
                            if _gate_ok and not required_groups:
                                _res = {
                                    'config': dict(config),
                                    'kpis': base_kpis,
                                    'equity_curve': build_equity_curve(trades_df),
                                    'stored_trades': _serialize_trades(
                                        trades_df,
                                        backtest_model=config.get('backtest_model') or 'rest_hifi'),
                                    'status': 'active',
                                    'confluence_str': 'None',
                                }
                                if _oos_eval is not None:
                                    _res['oos_kpis'] = _oos_eval['oos_kpis']
                                    _res['oos_sigma'] = _oos_eval['sigma']
                                    # Index into equity_curve where the OOS
                                    # band begins (= in-sample point count).
                                    _res['oos_boundary_index'] = len(
                                        build_equity_curve(_is_df))
                                results.append(_res)

                            # ── Level 3: Auto-search confluences ──
                            # Skip entirely when user selected no TF/General labels.
                            if (not skip_confluence_search and
                                    'confluence_records' in trades_df.columns
                                    and n_trades >= min_trades):
                                try:
                                    # top_n per base config scales with max_results
                                    _top_n_per_base = min(50, max(20, max_results // max(total_steps, 1)))

                                    # Sub-progress: confluence combo count as separate bar
                                    _combos_seen = {'n': 0}
                                    _conf_detail = f"Confluences — {_bt_detail}"
                                    def _conf_progress(idx, total_combos):
                                        _combos_seen['n'] = total_combos
                                        if progress_callback:
                                            progress_callback(
                                                step, total_steps, label,
                                                phase='confluence',
                                                phase_detail=_conf_detail,
                                                inner_step=idx,
                                                inner_total=total_combos)

                                    _conf_t0 = _time.monotonic()
                                    best = find_best_combinations(
                                        trades_df,
                                        max_depth=tf_conf_depth,
                                        min_trades=min_trades,
                                        top_n=_top_n_per_base,
                                        starting_balance=config.get(
                                            'starting_balance', 10000),
                                        risk_per_trade=config.get(
                                            'risk_per_trade', 100),
                                        total_trading_days=is_trading_days,
                                        allowed_labels=allowed_labels,
                                        required_groups=required_groups,
                                        progress_callback=_conf_progress,
                                        oos_boundary=(oos_in_sample_end_dt
                                                      if oos_enabled else None),
                                        oos_required=oos_required,
                                        oos_sigma_cfg=oos_sigma_cfg,
                                        oos_trading_days=oos_trading_days,
                                    )
                                    _diag['conf_search_total_sec'] += _time.monotonic() - _conf_t0
                                    _diag['conf_combos_total'] += _combos_seen['n']
                                    if best:
                                        for row in best:
                                            combo_kpis = {
                                                k: row[k] for k in base_kpis
                                                if k in row
                                            }
                                            if not meets_required_performance(
                                                    combo_kpis, required_perf):
                                                continue
                                            combo_set = set(row['combination'])
                                            conf_config = dict(config)
                                            tf_confs = [c for c in combo_set
                                                        if not c.startswith('GEN-')]
                                            gen_confs = [c for c in combo_set
                                                         if c.startswith('GEN-')]
                                            conf_config['confluence'] = tf_confs
                                            conf_config['general_confluences'] = gen_confs
                                            if _hifi_mode and 'cache' in _group_cache_holder:
                                                # HighFidelity §8.5: replay
                                                # the state machine with the
                                                # candidate's confluence set
                                                # gating entry inline. Trades
                                                # are bit-identical to what
                                                # a full backtest with this
                                                # combo's confluence would
                                                # produce — no post-hoc
                                                # mask drift.
                                                # NO fallback to mask: per
                                                # feedback_no_silent_defaults
                                                # — a silent downgrade to
                                                # the rapid path would mask
                                                # the very fidelity bug
                                                # we're fixing. Let the
                                                # exception propagate to
                                                # the outer combo handler
                                                # where it's logged as
                                                # "confluence search failed"
                                                # — surface, don't hide.
                                                filtered = run_trades_from_cache(
                                                    _group_cache_holder['cache'],
                                                    conf_config,
                                                    _group_cache_holder['meta'])
                                                # HighFidelity replay covers warmup+visible
                                                # bars; trim to visible.
                                                if _search_visible_start is not None and len(filtered) > 0:
                                                    from strategy_data import trim_trades_to_visible
                                                    filtered = trim_trades_to_visible(
                                                        filtered, _search_visible_start)
                                            else:
                                                mask = trades_df['confluence_records'].apply(
                                                    lambda r: isinstance(r, set)
                                                    and combo_set.issubset(r))
                                                filtered = trades_df[mask]
                                                # trades_df was already trimmed above; the
                                                # mask preserves that. Explicit trim is a
                                                # no-op safety net.
                                                if _search_visible_start is not None and len(filtered) > 0:
                                                    from strategy_data import trim_trades_to_visible
                                                    filtered = trim_trades_to_visible(
                                                        filtered, _search_visible_start)
                                            _conf_res = {
                                                'config': conf_config,
                                                'kpis': combo_kpis,
                                                'equity_curve': build_equity_curve(
                                                    filtered),
                                                'stored_trades': _serialize_trades(
                                                    filtered,
                                                    backtest_model=conf_config.get('backtest_model') or 'rest_hifi'),
                                                'status': 'active',
                                                # ✱-prefix the required
                                                # records so result cards
                                                # show what was pinned vs
                                                # discovered.
                                                'confluence_str': ' + '.join(
                                                    ('✱' + c) if c in set(row.get('required') or []) else c
                                                    for c in sorted(combo_set)),
                                            }
                                            # OOS gate fields (find_best_
                                            # combinations stamps these when
                                            # oos_boundary was supplied).
                                            if 'oos_kpis' in row:
                                                _conf_res['oos_kpis'] = row['oos_kpis']
                                                _conf_res['oos_sigma'] = row.get(
                                                    'oos_sigma')
                                                _filt_is, _ = split_trades_at_boundary(
                                                    filtered, oos_in_sample_end_dt)
                                                _conf_res['oos_boundary_index'] = len(
                                                    build_equity_curve(_filt_is))
                                            results.append(_conf_res)
                                except _CancelledError:
                                    raise
                                except Exception as exc:
                                    logger.warning(
                                        "Mass search: confluence search "
                                        "failed %s: %s", label, exc)

                            if progress_callback:
                                progress_callback(step, total_steps, label)

                            # 1.16: throttled mid-flight flush of results-so-far
                            # — crash-visible + viewable as each combo lands.
                            if (partial_callback and _partial_flush_sec > 0 and
                                    _time.monotonic() - _last_partial_flush
                                    > _partial_flush_sec):
                                _last_partial_flush = _time.monotonic()
                                try:
                                    partial_callback(list(results))
                                except Exception as _pf_err:
                                    logger.warning("Mass search: partial flush "
                                                   "failed (non-fatal): %s", _pf_err)

            logger.info("Mass search: %s/%s group complete, %d results so far",
                        symbol, tf, len(results))

            # Checkpoint emission (Roadmap 9s). End of a (symbol, tf) group
            # is our natural resume boundary — all downstream combos for
            # this group are done, next group starts a fresh data load.
            _completed_symbol_tfs_list.append([symbol, tf])
            completed_symbol_tfs.add((symbol, tf))
            if checkpoint_callback:
                try:
                    checkpoint_callback(
                        list(_completed_symbol_tfs_list),
                        list(results),
                        dict(_diag),
                    )
                except Exception as _cp_err:
                    logger.warning("Checkpoint callback failed (non-fatal): %s",
                                   _cp_err)

    # Sort by user's chosen metric, trim to max_results
    _sort_metric = required_perf.get('sort_by', 'daily_r')
    results.sort(key=lambda r: r.get('kpis', {}).get(_sort_metric, -999),
                 reverse=True)

    # ── Hi-Fi top-K refinement (Mass Builder §8.5 final step) ──────────
    # When in HighFidelity mode and skipHifi=False, run Hi-Fi Pass 2
    # on the top (hifi_buffer_mult × max_results) candidates. Refines
    # stop/target/signal-exit timestamps to sub-second via 1-sec
    # Polygon bars. Re-sort after refinement so any KPI shifts
    # propagate into the displayed top-N.
    if _hifi_mode and not _skip_hifi and results:
        _buffer_n = min(len(results), _hifi_buffer_mult * max_results)
        _hifi_start = _time.monotonic()
        _hifi_refined_count = 0
        _hifi_entries_total = 0
        _hifi_exits_total = 0
        _hifi_signal_exits_total = 0
        for _hifi_i, r in enumerate(results[:_buffer_n]):
            # Visible progress during the Hi-Fi pass — this phase runs AFTER
            # the main backtest/confluence bars complete, and before this it
            # reported nothing, so long refinements looked like a hung
            # "Processing" (2026-07-02).
            if progress_callback and _hifi_i % 5 == 0:
                progress_callback(
                    total_steps, total_steps, 'Hi-Fi Pass 2',
                    phase='save',
                    phase_detail=f'Hi-Fi refining top candidates '
                                 f'({_hifi_i + 1}/{_buffer_n})',
                    inner_step=_hifi_i + 1, inner_total=_buffer_n)
            _cfg = r.get('config') or {}
            _sym = _cfg.get('symbol')
            _tf = _cfg.get('timeframe')
            _ctx = _hifi_group_ctx.get((_sym, _tf))
            if _ctx is None:
                continue
            _bar_df, _visible_df, _visible_start, _trading_days = _ctx
            _hifi_refine_result(
                r,
                bar_df=_bar_df,
                visible_start=_visible_start,
                trading_days=_trading_days,
            )
            _h = r.get('hifi') or {}
            if _h.get('total_refined', 0) > 0:
                _hifi_refined_count += 1
                _hifi_entries_total += _h.get('entries_refined', 0)
                _hifi_exits_total += _h.get('exits_refined', 0)
                _hifi_signal_exits_total += _h.get('signal_exits_refined', 0)
        _hifi_wall = round(_time.monotonic() - _hifi_start, 2)
        _diag['hifi_topk_total_sec'] = _hifi_wall
        _diag['hifi_topk_candidates'] = _buffer_n
        _diag['hifi_topk_with_changes'] = _hifi_refined_count
        _diag['hifi_topk_entries_refined'] = _hifi_entries_total
        _diag['hifi_topk_exits_refined'] = _hifi_exits_total
        _diag['hifi_topk_signal_exits_refined'] = _hifi_signal_exits_total
        print(
            f"[MASS-HIFI] top-K refined: {_hifi_refined_count}/{_buffer_n} "
            f"candidates had changes (entries={_hifi_entries_total} "
            f"exits={_hifi_exits_total} signals={_hifi_signal_exits_total}) "
            f"wall={_hifi_wall}s",
            flush=True)
        # Re-sort post-Hi-Fi — KPI shifts may re-rank.
        results.sort(
            key=lambda r: r.get('kpis', {}).get(_sort_metric, -999),
            reverse=True)

    _diag['total_results_before_trim'] = len(results)
    # Compute per-unit timings for preview calibration.
    _wall_total = _time.monotonic() - _diag['overall_start']
    _diag['wall_total_sec'] = round(_wall_total, 3)
    _n_bt = max(_diag['backtests_run'], 1)
    _n_combos = max(_diag['conf_combos_total'], 1)
    _diag['trigger_bt_avg_ms'] = round(_diag['trigger_bt_total_sec'] / _n_bt * 1000, 2)
    _diag['conf_bt_avg_ms'] = round(_diag['conf_search_total_sec'] / _n_combos * 1000, 4)
    _overhead = _wall_total - _diag['trigger_bt_total_sec'] - _diag['conf_search_total_sec']
    print(
        f"[MASS-CALIBRATION] wall={_wall_total:.1f}s | "
        f"trigger_bts={_diag['backtests_run']} @ avg {_diag['trigger_bt_avg_ms']:.1f}ms "
        f"(sum {_diag['trigger_bt_total_sec']:.1f}s) | "
        f"confluence_bts={_diag['conf_combos_total']} @ avg {_diag['conf_bt_avg_ms']:.3f}ms "
        f"(sum {_diag['conf_search_total_sec']:.1f}s) | overhead={_overhead:.1f}s",
        flush=True)
    print(f"[MASS-CALIBRATION] params: data_days={data_days} session={session} "
          f"tickers={len(tickers)} tfs={timeframes} dirs={directions} "
          f"entries={len(entry_cids)} exits={len(exit_combos)} packs={len(pack_combos)}",
          flush=True)
    logger.info("Mass search complete: %s", _diag)

    trimmed = results[:max_results]
    # Attach diagnostics to the result set
    return trimmed, _diag


# ═══════════════════════════════════════════════════════════════════════════
# BACKGROUND EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

import threading
import time as _time

# Thread-safe shared state for progress reporting.
# Key: search_id → dict with progress info.
_active_searches: Dict[str, dict] = {}
_search_lock = threading.Lock()


_MAX_AUTO_RESUMES = 3  # retry cap: if a search fails to complete after
                        # this many auto-resumes, mark orphaned and require
                        # a manual Edit+Restart. Prevents infinite retry
                        # loops on fundamentally broken searches.


def cleanup_orphaned_mass_searches() -> dict:
    """Recover mass_searches rows left behind by a dead API process on startup.

    A fresh API process has no worker threads by definition, so any DB row
    still showing 'running' belongs to a prior process whose daemon thread
    died (pod restart, deploy, crash). With RORT_MASS_RECOVER_QUEUED=1 we also
    reclaim 'queued' rows — a job that OOM-restarted during the load phase,
    before the worker flushed 'running' or wrote its first checkpoint, is
    otherwise stuck 'queued' forever, invisible to a 'running'-only sweep
    (board #32). Two paths per row:

    * **Auto-resume** (Roadmap 9s Tier 2 follow-up, 2026-04-21): row has a
      checkpoint with ≥1 completed (symbol, tf) group AND resume_count <
      _MAX_AUTO_RESUMES. Relaunch the worker with the saved config +
      checkpoint; Kevin's overnight queued searches continue through
      deploys without intervention.

    * **Orphan**: no checkpoint (died before first group completed) OR
      retry cap exhausted. Mark 'orphaned' so the user can Edit+Restart
      or click Resume (which uses the same relaunch code but bumps
      resume_count and is a no-op if capped).

    Returns a summary dict: {resumed: int, orphaned: int, ids_resumed: [],
    ids_orphaned: [], queued_recovered: int}.
    """
    import os
    summary = {'resumed': 0, 'orphaned': 0, 'ids_resumed': [], 'ids_orphaned': [],
               'queued_recovered': 0}

    # Environment-role guard (board #286). This sweep uses the ADMIN client and
    # flips EVERY user's rows — per-user scoping does not contain it at all.
    # Under the shared-database dev environment (#165) it must not run, and the
    # guard lives HERE as well as at the api-boot call site so that no caller
    # (boot, the /cleanup-orphans route, app.py, a future one) can reach it by
    # accident. With RORT_ENV_ROLE unset this is one dict lookup and a False.
    from env_role import boot_sweep_blocked
    if boot_sweep_blocked('cleanup_orphaned_mass_searches'):
        summary['skipped_env_role'] = True
        return summary

    from db import USE_DB
    if not USE_DB:
        return summary

    # Kill-switch (board #32). OFF (default) → legacy behavior: only
    # status='running' rows are swept, and a 'running' row is auto-resumed
    # ONLY from a real checkpoint. ON → also reclaim status='queued' rows that
    # a dead process left stranded (see docstring), and skip any row whose
    # worker is still live in THIS process so the manual /cleanup-orphans
    # trigger can't reap a freshly-spawned search mid-run.
    _recover_queued = os.getenv('RORT_MASS_RECOVER_QUEUED', '0') == '1'

    try:
        from db import get_admin_client, update_mass_search
        client = get_admin_client()
        # Pull the full config_data with each row so we can decide per-row
        # whether to auto-resume or mark orphaned without extra round-trips.
        if _recover_queued:
            resp = client.table('mass_searches') \
                .select('id, config_data, user_id, status') \
                .in_('status', ['queued', 'running']) \
                .execute()
        else:
            resp = client.table('mass_searches') \
                .select('id, config_data, user_id') \
                .eq('status', 'running') \
                .execute()
        rows = resp.data if resp and resp.data else []
        if not rows:
            logger.info("Mass search startup cleanup: 0 recoverable rows found")
            return summary

        now_iso = datetime.now(timezone.utc).isoformat()

        for row in rows:
            sid = row.get('id')
            is_queued = _recover_queued and row.get('status') == 'queued'

            # Live-process guard (flag-gated). Never reap a search whose worker
            # is still running in THIS process. On API boot _active_searches is
            # empty, so every DB row is a genuine orphan and nothing is skipped;
            # this only matters for the manual /cleanup-orphans endpoint.
            if _recover_queued:
                with _search_lock:
                    _live = (_active_searches.get(str(sid))
                             or _active_searches.get(sid))
                if _live and _live.get('status') == 'running' \
                        and not _live.get('cancelled'):
                    continue

            cfg_data = row.get('config_data') or {}
            checkpoint = cfg_data.get('checkpoint') or {}
            completed = checkpoint.get('completed_symbol_tfs') or []
            resume_count = int(checkpoint.get('resume_count') or 0)
            saved_config = cfg_data.get('config') or {}

            has_checkpoint = isinstance(checkpoint, dict) and len(completed) > 0
            # Legacy ('running'): resume only from a real checkpoint. #32
            # ('queued'): no checkpoint exists yet, so resume from scratch; a
            # row already mid from-scratch-retry (resume_count>0, only ever
            # produced by our own queued recovery) keeps its full retry budget.
            # A first-run 'running' row that died before its first checkpoint
            # stays orphaned, exactly as before.
            can_resume = (
                resume_count < _MAX_AUTO_RESUMES
                and isinstance(saved_config, dict)
                and saved_config  # non-empty
                and (has_checkpoint
                     or is_queued
                     or (_recover_queued and resume_count > 0))
            )

            if can_resume:
                # Bump retry counter BEFORE launching so a crash loop is
                # bounded. Leave status='running' since the new worker
                # takes over immediately (for a reclaimed 'queued' row this
                # also flips it out of 'queued' so it stops looking fresh).
                new_checkpoint = dict(checkpoint)
                new_checkpoint['resume_count'] = resume_count + 1
                new_checkpoint['last_auto_resumed_at'] = now_iso
                try:
                    # Write the bumped checkpoint via admin client — the
                    # normal update_mass_search uses the user-scoped client
                    # which has no JWT at startup. We already have the row
                    # via admin so we know it exists; a direct update is
                    # safe.
                    _merged_cfg = dict(cfg_data)
                    _merged_cfg['checkpoint'] = new_checkpoint
                    _upd = {'config_data': _merged_cfg, 'updated_at': now_iso}
                    if _recover_queued:
                        _upd['status'] = 'running'
                    client.table('mass_searches').update(_upd) \
                        .eq('id', sid).execute()

                    # Relaunch worker in admin-mode context so user-scoped
                    # DB queries (user packs, strategies) still resolve to
                    # the original owner's data without needing a JWT.
                    row_user_id = row.get('user_id')
                    start_mass_search_async(
                        sid, saved_config,
                        # No completed groups (a reclaimed 'queued' row) →
                        # start from scratch; else resume from checkpoint.
                        resume_checkpoint=(new_checkpoint if has_checkpoint
                                           else None),
                        user_id_override=row_user_id,
                    )
                    summary['resumed'] += 1
                    summary['ids_resumed'].append(sid)
                    if is_queued:
                        summary['queued_recovered'] += 1
                    logger.info(
                        "Mass search auto-resumed id=%s at %d/%d groups "
                        "(attempt %d/%d)",
                        sid, len(completed),
                        len(saved_config.get('tickers') or []) *
                        len(saved_config.get('timeframes') or []) or len(completed),
                        resume_count + 1, _MAX_AUTO_RESUMES)
                except Exception as e:
                    logger.warning("Auto-resume failed for id=%s: %s — "
                                   "falling back to orphan", sid, e)
                    # Fall through to orphan handling for this row
                    client.table('mass_searches') \
                        .update({'status': 'orphaned', 'updated_at': now_iso}) \
                        .eq('id', sid).execute()
                    summary['orphaned'] += 1
                    summary['ids_orphaned'].append(sid)
            else:
                # Mark orphaned
                client.table('mass_searches') \
                    .update({'status': 'orphaned', 'updated_at': now_iso}) \
                    .eq('id', sid).execute()
                summary['orphaned'] += 1
                summary['ids_orphaned'].append(sid)

        logger.info(
            "Mass search startup cleanup: resumed=%d (queued_recovered=%d) "
            "orphaned=%d (ids_resumed=%s, ids_orphaned=%s)",
            summary['resumed'], summary['queued_recovered'], summary['orphaned'],
            summary['ids_resumed'], summary['ids_orphaned'])
        return summary
    except Exception as e:
        logger.warning("Mass search orphan cleanup failed: %s", e)
        return summary


def get_search_progress(search_id: str) -> dict:
    """Get current progress for a running search (thread-safe)."""
    with _search_lock:
        return dict(_active_searches.get(search_id, {}))


def is_search_running(search_id: str) -> bool:
    with _search_lock:
        info = _active_searches.get(search_id, {})
        return info.get('status') == 'running'


def cancel_search(search_id):
    """Signal a running search to stop."""
    search_id = str(search_id)
    with _search_lock:
        if search_id in _active_searches:
            _active_searches[search_id]['cancelled'] = True


def start_mass_search_async(search_id, search_config: dict,
                            resume_checkpoint: Optional[dict] = None,
                            user_id_override: Optional[str] = None):
    """Launch a mass search in a background daemon thread.

    Progress is written to _active_searches (in-memory, polled by UI)
    and periodically flushed to the database.
    """
    # Normalize to string so in-memory lookups match what the FastAPI
    # router passes (URL path params are always str). Supabase returns
    # the row id as int, which would otherwise mismatch the router's
    # str-keyed get_search_progress() lookups.
    search_id = str(search_id)

    # Capture the current user context so the background thread can
    # load user-specific packs from the database (confluence groups,
    # RM packs, general packs, strategies).
    #
    # The worker thread ALWAYS runs in admin-mode (service-role key, no
    # JWT) because mass searches routinely run for >60 min and a user
    # JWT expires in that window. Once expired, every subsequent
    # Supabase call fails with PGRST303 and the worker silently
    # bleeds — last hit was 2026-04-29 sid 1471921923 which lost 9 of
    # 10 (symbol, tf) data loads + the final results writeback after
    # 60 min. Admin mode + user_id WHERE filters preserve the same
    # row-scoping that RLS+JWT would have enforced.
    from db import get_current_user_id
    _user_id = user_id_override or get_current_user_id()

    with _search_lock:
        _active_searches[search_id] = {
            'status': 'running',
            'current_step': 0,
            'total_steps': 0,
            'current_label': 'Starting...',
            'results_so_far': 0,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'cancelled': False,
        }

    def _worker():
        try:
            # Always admin-mode on this thread. JWT-bearing user
            # context would expire mid-run on long searches; admin
            # context never expires. user_id is captured at request
            # time, so user-scoped WHERE filters still prevent cross-
            # user data exposure (same as worker.py).
            if _user_id:
                from db import set_admin_user_context
                set_admin_user_context(_user_id)

            from db import save_mass_search, update_mass_search

            last_db_flush = _time.monotonic()

            def _progress(step, total, label, conf_step=None, conf_total=None,
                          phase=None, phase_detail=None,
                          inner_step=None, inner_total=None):
                """Progress callback with two-tier info.

                step/total/label: overall search progress (top-bar fill + coarse label).
                phase: one of 'load', 'prep', 'backtest', 'confluence', 'save' — the
                    kind of activity happening RIGHT NOW.
                phase_detail: descriptive sub-info (e.g. "TSLA 1Min LONG · BT 3/8").
                inner_step/inner_total: fine-grain progress inside the current phase
                    (bars loaded, bar N of total, combo N of total).
                conf_step/conf_total: legacy fields kept for backward-compat; mirror
                    inner_step/inner_total when phase is 'confluence'.
                """
                with _search_lock:
                    info = _active_searches.get(search_id, {})
                    if info.get('cancelled'):
                        raise _CancelledError()
                    info['current_step'] = step
                    info['total_steps'] = total
                    info['current_label'] = label
                    if phase is not None:
                        info['phase'] = phase
                    if phase_detail is not None:
                        info['phase_detail'] = phase_detail
                    if inner_step is not None:
                        info['inner_step'] = inner_step
                        info['inner_total'] = inner_total or 0
                    # Maintain legacy conf_step/conf_total for consumers that
                    # still read them. Only populated during confluence phase.
                    if conf_step is not None:
                        info['conf_step'] = conf_step
                        info['conf_total'] = conf_total or 0
                    elif phase == 'confluence' and inner_step is not None:
                        info['conf_step'] = inner_step
                        info['conf_total'] = inner_total or 0
                    else:
                        info.pop('conf_step', None)
                        info.pop('conf_total', None)

                # Flush to DB every 60 seconds (was 10s before 2026-04-22).
                # Each flush rewrites the full config_data JSONB blob, which
                # includes the checkpoint payload — cheap on a small search,
                # expensive at 70+ strategy scale. Checkpoints land at the
                # (symbol, tf) boundary which typically takes >60s anyway,
                # so recovery granularity is unchanged; we just stop hammering
                # Supabase with redundant mid-group progress writes.
                nonlocal last_db_flush
                now = _time.monotonic()
                if now - last_db_flush > 60:
                    last_db_flush = now
                    try:
                        update_mass_search(search_id, {
                            'status': 'running',
                            'progress': {
                                'current_step': step,
                                'total_steps': total,
                                'current_label': label,
                                'phase': phase,
                                'phase_detail': phase_detail,
                            },
                        })
                    except Exception:
                        pass

            # Checkpoint callback — persists completed (symbol, tf) pairs +
            # partial results + running diagnostics to DB so this search can
            # be resumed from the last completed group if the worker thread
            # dies (API pod restart, etc.). See Roadmap 9s.
            def _checkpoint(completed_pairs, results_snapshot, diag_snapshot):
                # Strip stored_trades from partial_results (too large for
                # JSONB; matches the final flush strip behavior).
                slim_results = []
                for r in results_snapshot:
                    if isinstance(r, dict):
                        slim_results.append(
                            {k: v for k, v in r.items() if k != 'stored_trades'}
                        )
                    else:
                        slim_results.append(r)
                checkpoint_payload = {
                    'completed_symbol_tfs': completed_pairs,
                    'partial_results': slim_results,
                    'diagnostics_so_far': {
                        k: v for k, v in diag_snapshot.items()
                        # Skip the monotonic start marker; not useful on resume
                        if k != 'overall_start'
                    },
                    'last_checkpoint_at': datetime.now(timezone.utc).isoformat(),
                }
                with _search_lock:
                    if search_id in _active_searches:
                        _active_searches[search_id]['checkpoint'] = checkpoint_payload
                try:
                    update_mass_search(search_id, {'checkpoint': checkpoint_payload})
                except Exception as e:
                    logger.warning("Checkpoint DB flush failed (non-fatal): %s", e)

            # 1.16 mid-flight persistence — write the results-so-far to the DB
            # `results` field (slim, stored_trades stripped) + in-memory count
            # as each dough/combo completes. Makes a 20h run's progress
            # crash-visible AND viewable live (feeds the nested Mass-Results UI,
            # 1.17). Independent of `checkpoint` (resume) — no duplication risk.
            def _partial(results_snapshot):
                slim = [{k: v for k, v in r.items() if k != 'stored_trades'}
                        if isinstance(r, dict) else r
                        for r in results_snapshot]
                with _search_lock:
                    if search_id in _active_searches:
                        _active_searches[search_id]['results_so_far'] = len(slim)
                try:
                    update_mass_search(search_id, {
                        'status': 'running', 'results': slim})
                except Exception as e:
                    logger.warning("Partial results flush failed (non-fatal): %s", e)

            raw = run_mass_search(
                search_config,
                progress_callback=_progress,
                resume_checkpoint=resume_checkpoint,
                checkpoint_callback=_checkpoint,
                partial_callback=_partial,
            )

            # run_mass_search returns (results_list, diagnostics_dict)
            results, diagnostics = raw if isinstance(raw, tuple) else (raw, {})

            # Sanitize inf/nan in KPIs — JSON can't serialize them
            import math
            def _sanitize_floats(obj):
                if isinstance(obj, float):
                    if math.isnan(obj):
                        return 0
                    if math.isinf(obj):
                        return 9999.0 if obj > 0 else -9999.0
                    return obj
                if isinstance(obj, dict):
                    return {k: _sanitize_floats(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize_floats(v) for v in obj]
                return obj
            results = [_sanitize_floats(r) if isinstance(r, dict) else r
                       for r in results]

            # Save final results — keep full results (with stored_trades)
            # in memory so the frontend can read them immediately.
            with _search_lock:
                if search_id in _active_searches:
                    _active_searches[search_id]['status'] = 'completed'
                    _active_searches[search_id]['results_so_far'] = len(results)
                    _active_searches[search_id]['results'] = results

            # Strip stored_trades from DB payload — too large for JSONB.
            # The frontend re-runs a single backtest when user saves.
            db_results = []
            for r in results:
                if isinstance(r, dict):
                    slim = {k: v for k, v in r.items() if k != 'stored_trades'}
                    db_results.append(slim)
                else:
                    db_results.append(r)

            try:
                update_mass_search(search_id, {
                    'status': 'completed',
                    'results': db_results,
                    'progress': {},
                    'summary': {
                        'results_stored': len(db_results),
                        'best_daily_r': max(
                            (r.get('kpis', {}).get('daily_r', 0) for r in db_results
                             if isinstance(r, dict)),
                            default=0),
                        'diagnostics': diagnostics,
                    },
                })
            except Exception as _save_err:
                logger.error("Mass search %s DB save error: %s", search_id, _save_err)
            logger.info("Mass search %s completed: %d results", search_id, len(results))

        except _CancelledError:
            with _search_lock:
                if search_id in _active_searches:
                    _active_searches[search_id]['status'] = 'cancelled'
            try:
                from db import update_mass_search
                update_mass_search(search_id, {'status': 'cancelled'})
            except Exception:
                pass
            logger.info("Mass search %s cancelled", search_id)

        except Exception as exc:
            with _search_lock:
                if search_id in _active_searches:
                    _active_searches[search_id]['status'] = 'failed'
                    _active_searches[search_id]['error'] = str(exc)
            try:
                from db import update_mass_search
                update_mass_search(search_id, {'status': 'failed'})
            except Exception:
                pass
            logger.exception("Mass search %s failed", search_id)

        finally:
            # Clean up after a delay so the UI can read final status
            def _cleanup():
                _time.sleep(60)
                with _search_lock:
                    _active_searches.pop(search_id, None)
            threading.Thread(target=_cleanup, daemon=True).start()

    thread = threading.Thread(target=_worker, daemon=True, name=f"mass_search_{search_id}")
    thread.start()
    return thread


class _CancelledError(Exception):
    pass


TICKER_PRESETS = {
    "S&P Top 10": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK.B", "LLY", "AVGO", "JPM"],
    "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "CRM", "ORCL", "ADBE"],
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "TLT", "VXX"],
    "Crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "AVAX/USD", "DOGE/USD"],
    "Momentum": ["PLTR", "TSLA", "COIN", "MARA", "RIOT", "SQ", "SHOP", "SNOW", "NET", "DDOG"],
}
