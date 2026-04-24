"""
Forward-test persistence service.

Re-runs the unified engine on a strategy's historical bars and persists
the result into `stored_trades` / `kpis` / `equity_curve_data` /
`data_refreshed_at` on the strategies row.

Single source of truth: both the /refresh API endpoint (on-demand user
click) and the worker's bar-close hook (automatic per-bar recompute)
go through this function. `stored_trades` is always backtest truth —
never a live append path — so parity with the alerts table becomes a
direct fidelity signal rather than a data-origin question.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


def recompute_and_persist_stored_trades(
    strategy_id: int,
    user_id: str,
) -> Dict[str, Any]:
    """Run unified engine on historical bars and persist results.

    Idempotent per strategy. Replaces `stored_trades` entirely (no
    append semantics). Uses the admin client so it works from worker
    threads that have no user JWT.

    Returns:
        {
          "status": "refreshed" | "no_trades",
          "trades": int,
          "kpis": dict,
          "refreshed_at": ISO8601 str,
        }

    Raises on unrecoverable failure (caller decides whether to surface
    as HTTP 500 or a worker log warning).
    """
    from db import (
        USE_DB,
        get_strategy_by_id_admin,
        update_strategy_admin,
        set_admin_user_context,
        get_current_user_id,
        clear_current_user,
    )
    import services as svc

    # Admin load — works from worker thread without user JWT. The API
    # endpoint path has already authenticated the user; here we still
    # gate by user_id so an admin call can't mutate a strategy that
    # belongs to a different user.
    strat = get_strategy_by_id_admin(strategy_id, user_id)
    if strat is None:
        raise RuntimeError(
            f"Strategy {strategy_id} not found for user {user_id}"
        )

    # services.get_strategy_trades ultimately calls load_confluence_groups
    # and load_general_packs, both of which read thread-local user context.
    # The API path already has a real session; if this function is being
    # called from a worker / background thread we set admin-mode context
    # so those helpers can run. Restore previous context in finally.
    _prev_user = get_current_user_id()
    _need_ctx_swap = _prev_user != user_id
    if _need_ctx_swap:
        set_admin_user_context(user_id)
    try:
        return _do_recompute(strategy_id, user_id, strat, svc, USE_DB,
                             update_strategy_admin)
    finally:
        if _need_ctx_swap:
            clear_current_user()


def _do_recompute(
    strategy_id: int,
    user_id: str,
    strat: Dict[str, Any],
    svc,
    USE_DB: bool,
    update_strategy_admin,
) -> Dict[str, Any]:

    confluence = strat.get('confluence', [])
    general_conf = strat.get('general_confluences', [])
    logger.info(
        "[FT-RECOMPUTE] strategy=%s (%s): confluence=%s, general=%s, entry=%s",
        strategy_id, strat.get('name', '?'),
        confluence, general_conf,
        strat.get('entry_trigger_confluence_id', '?'),
    )

    all_trades = svc.get_strategy_trades(strat)
    logger.info(
        "[FT-RECOMPUTE] strategy=%s: %d trades", strategy_id, len(all_trades),
    )

    refreshed_at = datetime.now(timezone.utc).isoformat()

    if len(all_trades) == 0:
        # No trades to persist, but still mark the timestamp so the
        # caller knows we ran. update_strategy_admin handles the
        # flag-aware trades-table redirect automatically.
        if USE_DB:
            update_strategy_admin(strategy_id, user_id, {
                'stored_trades': [],
                'data_refreshed_at': refreshed_at,
            })
        return {
            'status': 'no_trades',
            'trades': 0,
            'kpis': {},
            'refreshed_at': refreshed_at,
        }

    stored = _serialize_trades(all_trades)

    # Prefer trading_days attached by services.get_strategy_trades (counted
    # on the source bars DataFrame's DatetimeIndex). Falls back to
    # count_trading_days(trades_df) which returns 1 for an integer-indexed
    # trades DF — that fallback was the source of a ~180× inflated daily_r
    # before 2026-04-22 (trades DF has a RangeIndex, so .index.normalize
    # didn't exist → count_trading_days returned 1 → daily_r = total_r).
    trading_days = all_trades.attrs.get('trading_days') \
        if hasattr(all_trades, 'attrs') else None
    if not trading_days or trading_days < 1:
        trading_days = svc.count_trading_days(all_trades) \
            if hasattr(all_trades, 'index') else 1
    kpis = svc.calculate_kpis(all_trades, total_trading_days=trading_days)

    from api.services.backtest_service import _build_equity_curve
    eq_data = _build_equity_curve(all_trades)
    boundary_index = None
    if strat.get('forward_test_start'):
        fwd_start = datetime.fromisoformat(strat['forward_test_start'])
        bt_portion, _ = svc.split_trades_at_boundary(all_trades, fwd_start)
        boundary_index = len(bt_portion) if len(bt_portion) > 0 else None

    equity_curve_data = {
        'exit_times': [p.get('timestamp', '') for p in eq_data],
        'cumulative_r': [p.get('cumulative_r', 0) for p in eq_data],
        'boundary_index': boundary_index,
    }

    if USE_DB:
        # Flag-aware: update_strategy_admin redirects stored_trades to
        # the trades table automatically when USE_TRADES_TABLE is ON.
        # kpis + equity_curve_data stay as strategies-row columns either
        # way.
        update_strategy_admin(strategy_id, user_id, {
            'stored_trades': stored,
            'kpis': kpis,
            'equity_curve_data': equity_curve_data,
            'data_refreshed_at': refreshed_at,
        })

    return {
        'status': 'refreshed',
        'trades': len(stored),
        'kpis': kpis,
        'refreshed_at': refreshed_at,
    }


def _serialize_trades(trades_df) -> list[dict]:
    """Convert a trades DataFrame into JSON-safe records for Supabase.

    NaN / ±Inf are replaced with None — JSONB doesn't accept them and the
    open-trade row (still in-position at snapshot time) carries NaN in
    exit_time / exit_price / bars_held etc.

    Trade_Timestamps_Spec hotfix (2026-04-20): pandas NaT.isoformat()
    returns the literal string 'NaT' — downstream rendering then shows
    'Invalid Date'. Check pd.isna() BEFORE isoformat so NaT → None.
    """
    import math
    import pandas as pd
    records = []
    for _, row in trades_df.iterrows():
        record = {}
        for col in trades_df.columns:
            val = row[col]
            if hasattr(val, 'isoformat'):
                # NaT / NaN timestamps: isoformat returns 'NaT' string.
                # Convert to None so JSON consumers get a clean null.
                record[col] = None if pd.isna(val) else val.isoformat()
            elif hasattr(val, 'item'):  # numpy scalar types
                v = val.item()
                if isinstance(v, float) and not math.isfinite(v):
                    v = None
                record[col] = v
            elif isinstance(val, set):
                record[col] = list(val)
            elif isinstance(val, float) and not math.isfinite(val):
                record[col] = None
            else:
                record[col] = val
        records.append(record)
    return records
