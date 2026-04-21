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

_data_cache: Dict[tuple, tuple] = {}   # key → (df, sec_tf_map, trading_days)
_DATA_CACHE_MAX = 20                    # max entries (LRU eviction)


def _cache_key(symbol, tf, days, session, start_date, end_date) -> tuple:
    return (symbol, tf, days, session, str(start_date), str(end_date))


def _cache_put(key: tuple, df, sec_tf_map, trading_days):
    if len(_data_cache) >= _DATA_CACHE_MAX:
        # evict oldest entry
        _data_cache.pop(next(iter(_data_cache)))
    _data_cache[key] = (df, sec_tf_map, trading_days)


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

    # Confluence combinations per base config
    n_tf_conf = len(config.get('tf_confluences', []))
    tf_depth = config.get('tf_confluence_depth', 2)
    n_gen_conf = len(config.get('general_confluences', []))
    gen_depth = config.get('general_confluence_depth', 1)

    tf_combos = _n_choose_up_to(n_tf_conf, tf_depth) if n_tf_conf > 0 else 1
    gen_combos = _n_choose_up_to(n_gen_conf, gen_depth) if n_gen_conf > 0 else 1
    confluence_combos = tf_combos * gen_combos

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
) -> dict:
    """Build a strategy config dict compatible with _unified_trades() and save flow."""
    lookback_mode = date_range.get('mode', 'days')
    data_days = date_range.get('days', 90)
    start_date = date_range.get('start')
    end_date = date_range.get('end')

    return {
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
        'strategy_origin': 'standard',
    }


def _serialize_trades(trades_df) -> list:
    """Serialize trades DataFrame to list of dicts for storage (same format as backtest API)."""
    import pandas as pd
    if trades_df is None or not isinstance(trades_df, pd.DataFrame) or len(trades_df) == 0:
        return []
    cols = ['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price',
            'stop_price', 'target_price', 'r_multiple', 'win', 'exit_reason',
            'exec_type', 'bars_held', 'hold_time_seconds', 'entry_trigger']
    records = []
    for _, row in trades_df.iterrows():
        d = {}
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
) -> list:
    """Execute a mass strategy search.

    Groups by (symbol, timeframe, session) to minimize expensive data loads.
    Within each group, iterates over (direction, entry, exit_combo, RM_pack)
    and runs confluence auto-search on each base config.

    Args:
        search_config: Mass search configuration dict.
        progress_callback: Called with (current_step, total_steps, label).

    Returns:
        List of result dicts sorted by daily_r descending.
    """
    import os
    import pandas as pd
    from services import (
        prepare_data_with_indicators, get_secondary_tf_map,
        calculate_kpis, count_trading_days, find_best_combinations,
    )
    from alerts import _get_base_trigger_id as get_base_trigger_id
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

    # Resolve synthetic confluence IDs into allowed label suffixes for Layer 2
    # filtering. TF IDs have format "_TF_-{GROUP_ID}-{BULL|BEAR}-{fidelity}";
    # expand each to the real interpreter states via template outputs.
    # General IDs have format "GEN-{PACK_ID}-{STATE}" and match records directly.
    # Empty selections → skip confluence search entirely (no Layer 2).
    allowed_labels: Optional[set] = None
    skip_confluence_search = not (tf_conf_ids or gen_conf_ids)
    if not skip_confluence_search:
        allowed_labels = set()
        group_by_upper_id = {g.id.upper(): g for g in enabled_groups}
        for syn in tf_conf_ids:
            if not isinstance(syn, str) or not syn.startswith('_TF_-'):
                continue
            # Strip prefix and optional trailing fidelity (e.g. "-PB", "-CB")
            body = syn[len('_TF_-'):]
            parts = body.rsplit('-', 2) if body.count('-') >= 2 else body.rsplit('-', 1)
            # body forms: "GROUP_ID-DIR" or "GROUP_ID-DIR-FIDELITY"
            if len(parts) == 3:
                group_id_up, direction, _fidelity = parts
            elif len(parts) == 2:
                group_id_up, direction = parts
            else:
                continue
            group = group_by_upper_id.get(group_id_up)
            if not group:
                continue
            tmpl = _CONF_TEMPLATES.get(group.base_template)
            if not tmpl:
                continue
            outputs = tmpl.get('outputs', [])
            for interp in tmpl.get('interpreters', []):
                # `direction` is either a direction keyword (BULL/BEAR/NEUTRAL)
                # or a real state code (e.g. "SML", "H+up"). Treat as a state
                # when it's one of the template's outputs, else expand via the
                # direction map.
                if direction in outputs:
                    allowed_labels.add(f"{interp}-{direction}")
                else:
                    for state in expand_direction_to_states(interp, direction):
                        allowed_labels.add(f"{interp}-{state}")
        for syn in gen_conf_ids:
            if isinstance(syn, str) and syn.startswith('GEN-'):
                allowed_labels.add(syn)
        # Safety: user selected items but nothing resolved (e.g., stale group IDs).
        # Treat as empty selection → skip confluence search rather than falling
        # back to explore-all (which would surprise the user with a huge run).
        if not allowed_labels:
            skip_confluence_search = True
            allowed_labels = None
            print("[MASS] confluence selections did not resolve to any labels; "
                  "skipping confluence search", flush=True)
        else:
            print(f"[MASS] confluence filter active with {len(allowed_labels)} "
                  f"allowed labels (tf_selections={len(tf_conf_ids)}, "
                  f"gen_selections={len(gen_conf_ids)})", flush=True)

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

    logger.info("Mass search: %d tickers × %d TFs × %d dirs × %d entries "
                "× %d exit_combos = %d base configs",
                len(tickers), len(timeframes), len(directions),
                len(entry_cids), len(exit_combos), total_steps)

    for symbol in tickers:
        asset_type = 'crypto' if '/' in symbol else 'equity'
        sym_session = '24/7' if asset_type == 'crypto' else session

        for tf in timeframes:
            # ── Level 1: Load data (one per symbol+TF, cached) ──
            ck = _cache_key(symbol, tf, data_days, sym_session,
                            start_date, end_date)
            cached = _data_cache.get(ck)
            if cached:
                df, sec_tf_map, period_trading_days = cached
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
                    _enabled_tfs = set(load_settings_db().get("enabled_timeframes", ["1Min"]))
                    sec_tfs = tuple(sorted(_enabled_tfs - {tf} - SUB_MINUTE_TIMEFRAMES))
                    df = prepare_data_with_indicators(
                        symbol, data_days, data_seed,
                        start_date=start_date, end_date=end_date,
                        timeframe=tf, data_feed=data_feed,
                        session=sym_session, secondary_tfs=sec_tfs)
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
                period_trading_days = count_trading_days(df)
                _cache_put(ck, df, sec_tf_map, period_trading_days)

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

                            # KPIs on base trades (no confluence filter)
                            base_kpis = calculate_kpis(
                                trades_df,
                                starting_balance=config.get('starting_balance', 10000),
                                risk_per_trade=config.get('risk_per_trade', 100),
                                total_trading_days=period_trading_days)

                            _diag['combos_passed_perf'] += 1
                            if meets_required_performance(base_kpis, required_perf):
                                results.append({
                                    'config': dict(config),
                                    'kpis': base_kpis,
                                    'equity_curve': build_equity_curve(trades_df),
                                    'stored_trades': _serialize_trades(trades_df),
                                    'status': 'active',
                                    'confluence_str': 'None',
                                })

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
                                        total_trading_days=period_trading_days,
                                        allowed_labels=allowed_labels,
                                        progress_callback=_conf_progress,
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
                                            mask = trades_df['confluence_records'].apply(
                                                lambda r: isinstance(r, set)
                                                and combo_set.issubset(r))
                                            filtered = trades_df[mask]
                                            results.append({
                                                'config': conf_config,
                                                'kpis': combo_kpis,
                                                'equity_curve': build_equity_curve(
                                                    filtered),
                                                'stored_trades': _serialize_trades(
                                                    filtered),
                                                'status': 'active',
                                                'confluence_str': row.get(
                                                    'combo_str', ''),
                                            })
                                except _CancelledError:
                                    raise
                                except Exception as exc:
                                    logger.warning(
                                        "Mass search: confluence search "
                                        "failed %s: %s", label, exc)

                            if progress_callback:
                                progress_callback(step, total_steps, label)

            logger.info("Mass search: %s/%s group complete, %d results so far",
                        symbol, tf, len(results))

    # Sort by user's chosen metric, trim to max_results
    _sort_metric = required_perf.get('sort_by', 'daily_r')
    results.sort(key=lambda r: r.get('kpis', {}).get(_sort_metric, -999),
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


def start_mass_search_async(search_id, search_config: dict):
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
    from db import get_current_user_id, get_current_token
    _user_id = get_current_user_id()
    _token = get_current_token()

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
            # Re-establish user context on this thread so DB queries
            # (confluence groups, RM packs, etc.) return user's data.
            if _user_id and _token:
                from db import set_current_user
                set_current_user(_user_id, _token)

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

                # Flush to DB every 10 seconds
                nonlocal last_db_flush
                now = _time.monotonic()
                if now - last_db_flush > 10:
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

            raw = run_mass_search(search_config, progress_callback=_progress)

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
