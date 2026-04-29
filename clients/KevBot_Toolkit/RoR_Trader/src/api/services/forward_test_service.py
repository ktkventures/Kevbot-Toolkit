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
import threading
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


_AUTO_PARITY_LAST_N = 200


def recompute_and_persist_stored_trades(
    strategy_id: int,
    user_id: str,
    compute_parity: bool = True,
) -> Dict[str, Any]:
    """Run unified engine on historical bars and persist results.

    Idempotent per strategy. Replaces `stored_trades` entirely (no
    append semantics). Uses the admin client so it works from worker
    threads that have no user JWT.

    Args:
        compute_parity: when True (default), automatically replay the
            freshly-written trades through the live engine and persist
            a `parity_status` summary on the strategy row. Set False
            for bulk paths where the caller will batch-parity later
            (mass builder, migration scripts) — keeps the refresh fast
            when running over many strategies.

    Returns:
        {
          "status": "refreshed" | "no_trades",
          "trades": int,
          "kpis": dict,
          "refreshed_at": ISO8601 str,
          "parity": dict | None,    # only when compute_parity=True
        }

    Raises on unrecoverable failure of the trade-recompute step. Parity
    failures are isolated and surface as `parity.verdict == "ERROR"`.
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
                             update_strategy_admin, compute_parity)
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
    compute_parity: bool,
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
            # Strategy Health Badge: forward-test recompute is treated as a
            # KPI refresh — clear stale flag, stamp new computed timestamp.
            # data_source preserved (recompute uses whatever path saved it).
            'kpis_computed_at': refreshed_at,
            'kpis_stale_since': None,
        })

    parity_summary = None
    if compute_parity and USE_DB:
        # Kick parity off as a fire-and-forget background thread. A 10Sec
        # strategy with thousands of bars takes minutes to replay; if we
        # blocked the refresh response on it, the user would see a hung
        # spinner (and Railway/Cloudflare would 504 well before parity
        # finished). Instead the refresh returns immediately with a
        # PENDING parity_status, and the bg thread writes the real
        # verdict + score when the replay completes. The strategy card
        # picks up the new value on the next list refetch.
        pending_status = {
            'verdict': 'PENDING',
            'score': None,
            'stored_count': len(stored),
            'matched_count': 0,
            'computed_at': refreshed_at,
            'params': {
                'last_n': _AUTO_PARITY_LAST_N,
                'forward_test_only': False,
            },
        }
        _try_persist_parity(strategy_id, user_id, pending_status,
                            update_strategy_admin)
        parity_summary = pending_status

        threading.Thread(
            target=_bg_compute_parity,
            args=(strategy_id, user_id),
            name=f"parity-bg-{strategy_id}",
            daemon=True,
        ).start()

    result: Dict[str, Any] = {
        'status': 'refreshed',
        'trades': len(stored),
        'kpis': kpis,
        'refreshed_at': refreshed_at,
    }
    if parity_summary is not None:
        result['parity'] = parity_summary
    return result


def _bg_compute_parity(strategy_id: int, user_id: str) -> None:
    """Background-thread entry. Re-imports update_strategy_admin so the
    thread doesn't depend on closures from the originating request, and
    re-establishes admin user context (set_admin_user_context is
    thread-local; the request thread's context doesn't propagate)."""
    try:
        from db import (
            update_strategy_admin,
            set_admin_user_context,
            clear_current_user,
        )
        set_admin_user_context(user_id)
        try:
            _compute_and_persist_parity(
                strategy_id, user_id, update_strategy_admin)
        finally:
            clear_current_user()
    except Exception as e:
        logger.exception(
            "[FT-PARITY-BG] strategy=%s crashed: %s", strategy_id, e)


def _compute_and_persist_parity(
    strategy_id: int,
    user_id: str,
    update_strategy_admin,
) -> Dict[str, Any]:
    """Run parity replay against the just-written stored_trades and
    persist a compact summary on strategies.parity_status.

    Tolerant of failure: an exception inside the parity service never
    blocks the parent recompute path — the function records an ERROR
    verdict, attempts to persist it, and returns. The caller's trade
    write has already committed by the time we get here.

    Defaults: last_n=200, forward_test_only=False. The user's intent for
    auto-scoring is "does the live engine reproduce the trades I just
    backtested?" — both pre- and post-forward-test trades are valid
    inputs since they were all produced by the same engine that just
    ran. The Parity tab in the UI exposes the forward-test-only filter
    for users who want to scope to live-window-only.
    """
    import time as _time
    from datetime import datetime, timezone

    _t = _time.time()
    try:
        from api.services.parity_service import run_strategy_parity
        report = run_strategy_parity(
            strategy_id, user_id=user_id,
            last_n=_AUTO_PARITY_LAST_N, forward_test_only=False,
        )
    except Exception as e:
        logger.exception(
            "[FT-RECOMPUTE] auto-parity crashed for strategy %s: %s",
            strategy_id, e)
        status = {
            'verdict': 'ERROR',
            'score': None,
            'error': f'{type(e).__name__}: {e}',
            'computed_at': datetime.now(timezone.utc).isoformat(),
            'duration_s': round(_time.time() - _t, 2),
            'params': {
                'last_n': _AUTO_PARITY_LAST_N,
                'forward_test_only': False,
            },
        }
        _try_persist_parity(strategy_id, user_id, status,
                            update_strategy_admin)
        return status

    duration_s = _time.time() - _t
    stored_count = report.get('stored_count') or 0
    matched_count = report.get('matched_count') or 0
    score = report.get('parity_score')
    if score is None and stored_count > 0:
        # Defensive: parity_service computes score from
        # matched / (matched + stored_only + replay_only); if the upstream
        # ever drifts, surface a usable matched/stored ratio anyway.
        score = matched_count / stored_count if stored_count > 0 else None

    status = {
        'score': score,
        'verdict': report.get('verdict'),
        'stored_count': stored_count,
        'matched_count': matched_count,
        'stored_only_count': report.get('stored_only_count') or 0,
        'replay_only_count': report.get('replay_only_count') or 0,
        'most_common_failing_gate': report.get('most_common_failing_gate'),
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'duration_s': round(duration_s, 2),
        'params': {
            'last_n': _AUTO_PARITY_LAST_N,
            'forward_test_only': False,
        },
    }
    _try_persist_parity(strategy_id, user_id, status, update_strategy_admin)
    logger.warning(
        "[FT-RECOMPUTE] parity strategy=%s verdict=%s score=%s "
        "stored=%s matched=%s duration=%.2fs",
        strategy_id, status['verdict'], status['score'],
        status['stored_count'], status['matched_count'], duration_s)
    return status


def _try_persist_parity(
    strategy_id: int,
    user_id: str,
    parity_status: Dict[str, Any],
    update_strategy_admin,
) -> None:
    """Best-effort write of parity_status. Logs but never raises."""
    try:
        update_strategy_admin(strategy_id, user_id, {
            'parity_status': parity_status,
        })
    except Exception as e:
        logger.warning(
            "[FT-RECOMPUTE] failed to persist parity_status for %s: %s",
            strategy_id, e)


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
