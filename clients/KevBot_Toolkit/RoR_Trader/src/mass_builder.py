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

TICKER_PRESETS = {
    "S&P Top 10": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK.B", "LLY", "AVGO", "JPM"],
    "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "CRM", "ORCL", "ADBE"],
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "TLT", "VXX"],
    "Crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "AVAX/USD", "DOGE/USD"],
    "Momentum": ["PLTR", "TSLA", "COIN", "MARA", "RIOT", "SQ", "SHOP", "SNOW", "NET", "DDOG"],
}
