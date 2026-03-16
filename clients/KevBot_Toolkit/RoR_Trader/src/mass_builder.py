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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


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
    n_rm = max(len(config.get('rm_packs', [])), 1)

    # Exit combinations
    n_exits_raw = len(config.get('exit_triggers', []))
    exit_depth = config.get('exit_depth', 1)
    n_exit_combos = _n_choose_up_to(n_exits_raw, exit_depth)

    base_configs = n_tickers * n_tfs * n_dirs * n_entries * n_exit_combos * n_rm

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
        'n_rm_packs': n_rm,
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
        'strategy_origin': 'standard',
    }


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
    import pandas as pd
    from app import (
        prepare_data_with_indicators, get_secondary_tf_map,
        get_base_trigger_id, calculate_kpis, count_trading_days,
        find_best_combinations, _get_data_feed, get_data_source,
    )
    from confluence_groups import (
        get_enabled_groups, get_all_triggers, load_confluence_groups,
    )
    from risk_management_packs import load_risk_management_packs
    from unified_engine import run_unified_backtest
    import general_packs as gp_module

    enabled_groups = get_enabled_groups(load_confluence_groups())
    all_trigger_defs = get_all_triggers(enabled_groups)
    all_rm_packs = load_risk_management_packs()
    rm_pack_map = {p.id: p for p in all_rm_packs}
    enabled_gen = gp_module.get_enabled_general_packs(gp_module.load_general_packs())
    data_feed = _get_data_feed()

    tickers = search_config.get('tickers', [])
    timeframes = search_config.get('timeframes', ['1Min'])
    directions = search_config.get('directions', ['LONG'])
    entry_cids = search_config.get('entry_triggers', [])
    exit_cids_raw = search_config.get('exit_triggers', [])
    exit_depth = search_config.get('exit_depth', 1)
    rm_pack_ids = search_config.get('rm_packs', [])
    date_range = search_config.get('date_range', {'mode': 'days', 'days': 90})
    required_perf = search_config.get('required_performance', {})
    max_results = search_config.get('max_results', 500)
    tf_conf_depth = search_config.get('tf_confluence_depth', 2)
    gen_conf_depth = search_config.get('general_confluence_depth', 1)

    # Resolve date range
    data_days = date_range.get('days', 90)
    start_date = date_range.get('start')
    end_date = date_range.get('end')
    data_seed = 42

    # Resolve session
    session = search_config.get('session', 'RTH')

    # Build exit trigger combinations
    exit_combos = generate_exit_combos(exit_cids_raw, exit_depth)
    if not exit_combos or exit_combos == [[]]:
        exit_combos = [[]]

    # If no RM packs selected, use a default
    if not rm_pack_ids:
        rm_pack_ids = ['_default']

    # Count total base configs for progress
    total_steps = (len(tickers) * len(timeframes) * len(directions)
                   * len(entry_cids) * len(exit_combos) * len(rm_pack_ids))
    step = 0
    results = []

    for symbol in tickers:
        # Detect asset type
        asset_type = 'crypto' if '/' in symbol else 'equity'
        sym_session = '24/7' if asset_type == 'crypto' else session

        for tf in timeframes:
            # Level 1: Load data (cached by @st.cache_data)
            try:
                from app import _get_secondary_tfs
                sec_tfs = _get_secondary_tfs(tf)
                df = prepare_data_with_indicators(
                    symbol, data_days, data_seed,
                    start_date=start_date, end_date=end_date,
                    timeframe=tf, data_feed=data_feed,
                    session=sym_session, secondary_tfs=sec_tfs)
            except Exception as exc:
                logger.warning("Mass search: failed to load %s/%s: %s",
                               symbol, tf, exc)
                step += (len(directions) * len(entry_cids)
                         * len(exit_combos) * len(rm_pack_ids))
                if progress_callback:
                    progress_callback(step, total_steps, f"{symbol} {tf} — skipped")
                continue

            if len(df) == 0:
                step += (len(directions) * len(entry_cids)
                         * len(exit_combos) * len(rm_pack_ids))
                if progress_callback:
                    progress_callback(step, total_steps, f"{symbol} {tf} — no data")
                continue

            sec_tf_map = get_secondary_tf_map(df)
            period_trading_days = count_trading_days(df)

            for direction in directions:
                for entry_cid in entry_cids:
                    entry_tdef = all_trigger_defs.get(entry_cid)
                    if not entry_tdef:
                        step += len(exit_combos) * len(rm_pack_ids)
                        continue
                    entry_base = get_base_trigger_id(entry_cid)
                    entry_name = entry_tdef.name

                    # Check direction compatibility
                    if (entry_tdef.direction != 'BOTH'
                            and entry_tdef.direction != direction):
                        step += len(exit_combos) * len(rm_pack_ids)
                        continue

                    for exit_combo in exit_combos:
                        # Resolve exit trigger details
                        exit_bases = []
                        exit_names = []
                        valid_exits = True
                        for ecid in exit_combo:
                            etdef = all_trigger_defs.get(ecid)
                            if not etdef:
                                valid_exits = False
                                break
                            exit_bases.append(get_base_trigger_id(ecid))
                            exit_names.append(etdef.name)
                        if not valid_exits:
                            step += len(rm_pack_ids)
                            continue

                        for rm_id in rm_pack_ids:
                            step += 1
                            label = f"{symbol} {tf} {direction}"

                            # Resolve stop/target from RM pack
                            rm_pack = rm_pack_map.get(rm_id)
                            if rm_pack:
                                stop_config = rm_pack.get_stop_config()
                                target_config = rm_pack.get_target_config()
                            else:
                                stop_config = {"method": "atr", "atr_mult": 1.5}
                                target_config = {"method": "risk_reward",
                                                 "rr_ratio": 2.0}

                            # Build strategy config (NO confluences — base run)
                            config = build_strategy_config(
                                symbol=symbol, timeframe=tf,
                                direction=direction, session=sym_session,
                                entry_cid=entry_cid, entry_base=entry_base,
                                entry_name=entry_name,
                                exit_cids=list(exit_combo),
                                exit_bases=exit_bases,
                                exit_names=exit_names,
                                bar_count_exit=None,
                                stop_config=stop_config,
                                target_config=target_config,
                                date_range=date_range,
                                asset_type=asset_type,
                            )

                            # Level 2: Run base backtest (no confluences)
                            try:
                                trades_df, _ = run_unified_backtest(
                                    df, config,
                                    general_packs=enabled_gen,
                                    secondary_tf_map=sec_tf_map if sec_tf_map else None)
                            except Exception as exc:
                                logger.warning("Mass search: backtest failed %s: %s",
                                               label, exc)
                                if progress_callback:
                                    progress_callback(step, total_steps, label)
                                continue

                            if not isinstance(trades_df, pd.DataFrame):
                                trades_df = pd.DataFrame()

                            n_trades = len(trades_df)
                            min_trades = required_perf.get('min_trades', 10)

                            if n_trades < min_trades:
                                if progress_callback:
                                    progress_callback(step, total_steps, label)
                                continue

                            # Check base performance (no confluences)
                            base_kpis = calculate_kpis(
                                trades_df,
                                starting_balance=config.get('starting_balance', 10000),
                                risk_per_trade=config.get('risk_per_trade', 100),
                                total_trading_days=period_trading_days)

                            # Store base result if it meets performance
                            if meets_required_performance(base_kpis, required_perf):
                                results.append({
                                    'config': dict(config),
                                    'kpis': base_kpis,
                                    'equity_curve': build_equity_curve(trades_df),
                                    'status': 'active',
                                    'confluence_str': 'None',
                                })

                            # Level 3: Auto-search confluences
                            if ('confluence_records' in trades_df.columns
                                    and n_trades >= min_trades):
                                try:
                                    best = find_best_combinations(
                                        trades_df,
                                        max_depth=tf_conf_depth,
                                        min_trades=min_trades,
                                        top_n=5,
                                        starting_balance=config.get(
                                            'starting_balance', 10000),
                                        risk_per_trade=config.get(
                                            'risk_per_trade', 100),
                                        total_trading_days=period_trading_days,
                                    )
                                    if len(best) > 0:
                                        for _, row in best.iterrows():
                                            combo_kpis = {
                                                k: row[k] for k in base_kpis
                                                if k in row.index
                                            }
                                            if not meets_required_performance(
                                                    combo_kpis, required_perf):
                                                continue
                                            combo_set = row['combination']
                                            conf_config = dict(config)
                                            # Split into TF and General
                                            tf_confs = [c for c in combo_set
                                                        if not c.startswith('GEN-')]
                                            gen_confs = [c for c in combo_set
                                                         if c.startswith('GEN-')]
                                            conf_config['confluence'] = tf_confs
                                            conf_config['general_confluences'] = gen_confs
                                            # Re-filter trades for equity curve
                                            mask = trades_df['confluence_records'].apply(
                                                lambda r: isinstance(r, set)
                                                and combo_set.issubset(r))
                                            filtered = trades_df[mask]
                                            results.append({
                                                'config': conf_config,
                                                'kpis': combo_kpis,
                                                'equity_curve': build_equity_curve(
                                                    filtered),
                                                'status': 'active',
                                                'confluence_str': row.get(
                                                    'combo_str', ''),
                                            })
                                except Exception as exc:
                                    logger.warning(
                                        "Mass search: confluence search "
                                        "failed %s: %s", label, exc)

                            if progress_callback:
                                progress_callback(step, total_steps, label)

    # Sort by daily_r descending, trim to max_results
    results.sort(key=lambda r: r.get('kpis', {}).get('daily_r', -999),
                 reverse=True)
    return results[:max_results]


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


def cancel_search(search_id: str):
    """Signal a running search to stop."""
    with _search_lock:
        if search_id in _active_searches:
            _active_searches[search_id]['cancelled'] = True


def start_mass_search_async(search_id: str, search_config: dict):
    """Launch a mass search in a background daemon thread.

    Progress is written to _active_searches (in-memory, polled by UI)
    and periodically flushed to the database.
    """
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
            from db import save_mass_search, update_mass_search

            last_db_flush = _time.monotonic()

            def _progress(step, total, label):
                with _search_lock:
                    info = _active_searches.get(search_id, {})
                    if info.get('cancelled'):
                        raise _CancelledError()
                    info['current_step'] = step
                    info['total_steps'] = total
                    info['current_label'] = label

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
                            },
                        })
                    except Exception:
                        pass

            results = run_mass_search(search_config, progress_callback=_progress)

            # Save final results
            with _search_lock:
                if search_id in _active_searches:
                    _active_searches[search_id]['status'] = 'completed'
                    _active_searches[search_id]['results_so_far'] = len(results)

            update_mass_search(search_id, {
                'status': 'completed',
                'results': results,
                'progress': {},
                'summary': {
                    'results_stored': len(results),
                    'best_daily_r': max(
                        (r['kpis'].get('daily_r', 0) for r in results),
                        default=0),
                },
            })
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
