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
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Serialize background parity work. Concurrent bg parity threads on a
# single API instance contend for CPU brutally — observed 04:32 UTC
# 2026-04-28: sid 143 replay loop dropped from ~1500 bars/s isolated
# to 83 bars/s under contention (18× slowdown), turning a 60s parity
# into 16.5 minutes. Semaphore(1) means at most one bg parity runs;
# subsequent ones queue. The user-facing refresh response stays fast
# (it returns before the semaphore is acquired); only the parity write
# completion is delayed.
_PARITY_BG_SEM = threading.Semaphore(1)


def _auto_parity_enabled() -> bool:
    """Allow ops to globally disable auto-parity via env var without a
    redeploy of code. Useful when running mass refreshes / mass-builder
    where the parity overhead isn't wanted, or as a kill-switch when
    something goes sideways."""
    val = os.environ.get("AUTO_PARITY_ENABLED", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    # Default on — explicit disable required.
    return True


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
        # Phase 41 (2026-05-11): backtest trades go to the trades table
        # with `data_source='backtest_<model>'` instead of the stored_trades
        # JSONB column. This unifies storage with algo trades (Phase 40)
        # and enables real REST↔CACHE divergence in the Divergence tab.
        #
        # The previous "stored_trades" JSONB write is now skipped entirely.
        # Readers (KPI compute, divergence endpoint, chart render) must
        # query the trades table with `data_source LIKE 'backtest_%'` filter.
        bt_model = (strat.get('config') or {}).get('backtest_model') \
            if isinstance(strat.get('config'), dict) else None
        if not bt_model:
            bt_model = strat.get('backtest_model') or 'rest_hifi'
        ds_tag = f'backtest_{bt_model}'
        # Stamp data_source on every trade record before persistence.
        tagged_stored = []
        for t in stored:
            t_copy = dict(t)
            t_copy['data_source'] = ds_tag
            tagged_stored.append(t_copy)

        # Write to trades table with backtest data_source filter so we
        # only DELETE existing backtest rows (NOT the algo rows).
        # 2026-05-11 (Phase 41 followup): call db.replace_trades_admin
        # directly instead of going through trades_store.replace_trades_for_strategy.
        # The trades_store wrapper checks USE_TRADES_TABLE env var and
        # falls back to a JSONB write when OFF — but the JSONB path is
        # exactly what Phase 41 obsoleted (large blob writes silently
        # fail under load). Phase 41 commits to the trades table being
        # canonical; the flag is vestigial. Direct call avoids the
        # broken fallback path in environments where the API service
        # doesn't have USE_TRADES_TABLE set.
        try:
            from db import replace_trades_admin
            replace_trades_admin(
                strategy_id, user_id, tagged_stored,
                data_source_filter='backtest_%')
            # Invalidate trades_store cache so future reads see fresh data
            try:
                from trades_store import _cache_invalidate
                _cache_invalidate(strategy_id, user_id)
            except Exception:
                pass
        except Exception as e:
            logger.exception(
                "[FT-RECOMPUTE] sid=%s trades-table write failed: %s",
                strategy_id, e)
            # No fallback to JSONB — Phase 41 made trades table canonical.
            # If the write fails, propagate so the caller can retry.
            raise

        # KPIs + equity curve stay as strategies-row columns (derived
        # data, fast access for page renders).
        update_strategy_admin(strategy_id, user_id, {
            'kpis': kpis,
            'equity_curve_data': equity_curve_data,
            'data_refreshed_at': refreshed_at,
            'kpis_computed_at': refreshed_at,
            'kpis_stale_since': None,
        })

        # Also stamp last_recompute_until_ts on the config so the cron's
        # "recently_processed" skip logic recognizes that this strategy
        # was just refreshed and doesn't redundantly re-process on the
        # next 5-min tick. Caller-side full-merge for partial JSONB
        # safety (feedback_jsonb_partial_updates.md).
        try:
            from db import get_admin_client
            _client = get_admin_client()
            _cur = _client.table('strategies').select('config') \
                .eq('id', strategy_id).eq('user_id', user_id) \
                .single().execute()
            _cfg = dict((_cur.data or {}).get('config') or {})
            if _cfg or len(_cfg) == 0:
                _cfg['last_recompute_until_ts'] = refreshed_at
                _client.table('strategies').update({'config': _cfg}) \
                    .eq('id', strategy_id).eq('user_id', user_id).execute()
        except Exception as e:
            logger.warning(
                "[FT-RECOMPUTE] strategy=%s last_recompute stamp failed: %s",
                strategy_id, e)

    # Phase-E preview (M8.7 — 2026-05-06; expanded 2026-05-07): when
    # strategy opted into a Hi-Fi-eligible backtest_model, auto-run
    # Hi-Fi Pass 2 after the recompute so manual Refresh stays
    # consistent with cron behavior. Same coverage as cron path:
    # rest_hifi (explicit opt-in) + cache_locked / cache_corrected
    # (cache-aligned models also benefit from sub-second L-type
    # refinement since cache_locked alone only fixes bar-parity, not
    # the bar-vs-tick exit timing gap).
    bt_model = (strat.get('config') or {}).get('backtest_model') \
        if isinstance(strat.get('config'), dict) else None
    if bt_model is None:
        bt_model = strat.get('backtest_model')
    HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
    hifi_summary = None
    if bt_model in HIFI_BACKTEST_MODELS and len(stored) > 0:
        try:
            from api.routers.strategies import run_hifi_pass2
            hifi_summary = run_hifi_pass2(
                strategy_id, user={'id': user_id})
            logger.info(
                "[FT-RECOMPUTE] strategy=%s bt=%s Hi-Fi pass: refined "
                "entries=%s exits=%s persisted=%s",
                strategy_id, bt_model,
                hifi_summary.get('entries_refined', 0),
                hifi_summary.get('exits_refined', 0),
                hifi_summary.get('persisted', 0))
        except Exception as e:
            logger.warning(
                "[FT-RECOMPUTE] strategy=%s Hi-Fi pass failed: %s",
                strategy_id, e)

    # Parity is now a distinct user-triggered action — mirrors the
    # Run Hi-Fi button pattern. Refresh writes stored_trades + KPIs and
    # returns. The user clicks "Run Parity Test" separately when they
    # want to see whether the backtest reproduces in the live engine.
    # The compute_parity arg is preserved for backwards compatibility
    # but is no longer auto-triggered here. queue_parity_for_strategy()
    # is the entrypoint for the new button.

    result = {
        'status': 'refreshed',
        'trades': len(stored),
        'kpis': kpis,
        'refreshed_at': refreshed_at,
    }
    if hifi_summary is not None:
        result['hifi'] = {
            'entries_refined': hifi_summary.get('entries_refined', 0),
            'exits_refined': hifi_summary.get('exits_refined', 0),
            'persisted': hifi_summary.get('persisted', 0),
        }
    return result


def queue_parity_for_strategy(strategy_id: int, user_id: str) -> Dict[str, Any]:
    """Public API: queue a background parity replay for the given
    strategy. Stamps parity_status=PENDING immediately and spawns a
    daemon thread that runs through the global semaphore.

    Returns immediately with the pending status. Caller doesn't wait
    for completion. The frontend polls or refreshes the strategy list
    to see the final verdict.

    Honors AUTO_PARITY_ENABLED env var as a kill-switch.
    """
    from db import USE_DB, update_strategy_admin
    if not USE_DB:
        return {
            'strategy_id': strategy_id,
            'status': 'skipped',
            'detail': 'USE_DB is False',
        }
    if not _auto_parity_enabled():
        return {
            'strategy_id': strategy_id,
            'status': 'disabled',
            'detail': 'AUTO_PARITY_ENABLED is false',
        }

    pending_status = {
        'verdict': 'PENDING',
        'score': None,
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'params': {
            'last_n': _AUTO_PARITY_LAST_N,
            'forward_test_only': False,
        },
    }
    _try_persist_parity(strategy_id, user_id, pending_status,
                        update_strategy_admin)

    threading.Thread(
        target=_bg_compute_parity,
        args=(strategy_id, user_id),
        name=f"parity-bg-{strategy_id}",
        daemon=True,
    ).start()

    return {
        'strategy_id': strategy_id,
        'status': 'queued',
        'parity_status': pending_status,
    }


def _bg_compute_parity(strategy_id: int, user_id: str) -> None:
    """Background-thread entry. Re-imports update_strategy_admin so the
    thread doesn't depend on closures from the originating request, and
    re-establishes admin user context (set_admin_user_context is
    thread-local; the request thread's context doesn't propagate).

    Acquires _PARITY_BG_SEM so concurrent bg parity threads serialize
    rather than thrashing the CPU. A blocked thread sits in semaphore
    wait — it doesn't consume CPU, just memory for its frame.
    """
    queued_ts = datetime.now(timezone.utc).isoformat()
    with _PARITY_BG_SEM:
        wait_s = (datetime.now(timezone.utc) -
                  datetime.fromisoformat(queued_ts)).total_seconds()
        if wait_s > 1.0:
            logger.warning(
                "[FT-PARITY-BG] strategy=%s queued for %.1fs before run",
                strategy_id, wait_s)
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


# ============================================================
# Algo-history incremental writer (M8.7 — 2026-05-06)
# ============================================================
#
# Replaces two legacy paths with one cron-driven append-only path:
#   1. DBAlertDispatcher._persist_algo_trade — synchronous single-trade
#      INSERT on every exit alert. Writes sub-second L-type fill ts,
#      which then drifts when manual Refresh overwrites with bar-aligned
#      ts — the source of "algo-history row a few seconds before the
#      alert" confusion.
#   2. Manual Refresh button (recompute_and_persist_stored_trades) —
#      DELETE+INSERT all trades. Wipes Hi-Fi pass refinements + slow at
#      scale.
#
# New cron path semantics:
#   - Runs every ALGO_HISTORY_CRON_INTERVAL_SECONDS (default 300 = 5min)
#   - Processes trades with exit_fill_ts <= now - ALGO_HISTORY_LAG_MINUTES
#     (default 15min). The lag protects against FINRA late-print
#     corrections — only commits trades after the bar can no longer
#     change.
#   - Set-diff against existing trades-table rows by
#     (entry_fill_ts, exit_fill_ts) tuple — INSERTs only what's new.
#     Idempotent. Hi-Fi-refined rows are preserved.
#   - Stamps strategy.config.last_recompute_until_ts so subsequent runs
#     can skip cheaply when the lag window hasn't advanced.

_ALGO_HISTORY_LAG_MINUTES = int(
    os.environ.get('ALGO_HISTORY_LAG_MINUTES', '15'))


def _algo_history_cron_enabled() -> bool:
    val = os.environ.get('ALGO_HISTORY_CRON_ENABLED', '').strip().lower()
    if val in ('0', 'false', 'no', 'off'):
        return False
    return True  # default on


def _stamp_config(strategy_id: int, user_id: str, cfg: dict) -> None:
    """Write a fully-merged config dict back via the raw admin client.

    Bypasses update_strategy_admin to avoid the partial-JSONB-wipe bug
    documented in feedback_jsonb_partial_updates.md.

    Defensive shrink-guard: re-fetches the current config from DB and
    REFUSES the write if the new dict has materially fewer keys than
    what's already stored. Caused a config wipe on 2026-05-06 when an
    upstream caller passed a tiny dict (because they read config from
    the flattened strat shape that returns None). Loud failure here
    would have prevented data loss.
    """
    try:
        from db import get_admin_client
        _client = get_admin_client()
        # Pre-check: read current config and bail if we'd be shrinking
        # the dict by more than 1 key (we expect to be ADDING
        # last_recompute_until_ts; loss of any other key is a bug).
        try:
            cur_resp = _client.table('strategies').select('config') \
                .eq('id', strategy_id).eq('user_id', user_id).single().execute()
            cur_cfg = (cur_resp.data or {}).get('config') or {}
        except Exception:
            cur_cfg = {}
        cur_keys = set(cur_cfg.keys())
        new_keys = set(cfg.keys())
        lost = cur_keys - new_keys
        if lost and len(cur_cfg) > 2:
            logger.error(
                "[ALGO-APPEND] strategy=%s REFUSING config write: would "
                "lose %d keys (%s) — caller built an incomplete merge. "
                "current=%d keys, new=%d keys.",
                strategy_id, len(lost), sorted(lost),
                len(cur_cfg), len(cfg))
            return
        _client.table('strategies').update({'config': cfg}) \
            .eq('id', strategy_id).eq('user_id', user_id).execute()
    except Exception as e:
        logger.warning(
            "[ALGO-APPEND] strategy=%s config stamp failed: %s",
            strategy_id, e)


def append_new_trades_for_strategy(
    strategy_id: int,
    user_id: str,
    lag_minutes: int = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the unified engine and INSERT only trades newer than what's
    already in the trades table for this strategy. Append-only.

    Idempotent. Safe to call repeatedly. Skips strategies whose
    ``config.last_recompute_until_ts`` is already past ``now - lag``.

    Returns:
        {"status": "skipped"|"appended"|"no_new_trades"|"first_run",
         "inserted": int, "cutoff": iso, "elapsed_s": float}
    """
    import time as _time
    import pandas as pd

    if lag_minutes is None:
        lag_minutes = _ALGO_HISTORY_LAG_MINUTES

    from db import (
        USE_DB,
        get_strategy_by_id_admin,
        update_strategy_admin,
        set_admin_user_context,
        get_current_user_id,
        clear_current_user,
    )
    import services as svc

    if not USE_DB:
        return {'status': 'skipped', 'reason': 'USE_DB is False',
                'inserted': 0}

    strat = get_strategy_by_id_admin(strategy_id, user_id)
    if strat is None:
        return {'status': 'skipped',
                'reason': f'strategy {strategy_id} not found',
                'inserted': 0}

    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    cutoff = now_dt - timedelta(minutes=lag_minutes)
    cutoff_iso = cutoff.isoformat()

    # CRITICAL: get_strategy_by_id_admin returns a FLATTENED strat dict
    # — it spreads config fields to top-level, so strat.get('config')
    # returns None even when the JSONB column has 20+ fields. Reading it
    # that way and merging back wipes the column. Read the raw JSONB
    # directly to get the actual current config dict.
    from db import get_admin_client
    _raw_client = get_admin_client()
    _raw_resp = _raw_client.table('strategies').select('config') \
        .eq('id', strategy_id).eq('user_id', user_id).single().execute()
    cfg = dict(_raw_resp.data.get('config') or {}) if _raw_resp.data else {}
    last_until = cfg.get('last_recompute_until_ts')
    is_first_run = not last_until

    # Skip if this strategy was processed within the last cron interval.
    # cutoff_iso is now-lag (advances every call), so comparing to it
    # would never trigger a skip. We want: "if I just ran on this
    # strategy within the cron interval, no point re-running yet."
    # `force=True` bypasses this gate (manual user click via Update New
    # Data button — they want fresh work even if cron just ran).
    if last_until and not force:
        try:
            last_dt = datetime.fromisoformat(last_until)
            recent_window = datetime.now(timezone.utc) - timedelta(
                seconds=int(os.environ.get(
                    'ALGO_HISTORY_CRON_INTERVAL_SECONDS', '300')))
            if last_dt >= recent_window:
                return {'status': 'skipped',
                        'reason': 'recently_processed',
                        'inserted': 0,
                        'last_recompute_until_ts': last_until,
                        'cutoff': cutoff_iso}
        except Exception:
            pass  # fall through to engine run on parse failure

    # Alerts-gate (M8.7 throughput fix, 2026-05-06): skip the expensive
    # engine run when no exit_signal alerts have fired for this strategy
    # since last_recompute_until_ts. Reasoning: closed trades only
    # materialize when the engine emits an exit signal — and the LIVE
    # engine writes that to the alerts table. If no exit alerts since
    # last cron, the engine has no new closed trades to discover, and
    # the ~30-40s engine replay would just redundantly produce the same
    # set we already have in the trades table.
    #
    # Tradeoff: this gate skips "missed trades" (algo-only trades the
    # live engine didn't emit). Manual Refresh button still picks those
    # up if user requests. Acceptable for v1 since missed trades are a
    # divergence-investigation concern, not a per-minute concern.
    #
    # Drops cycle time from ~25-30 min full-coverage to <1 min when most
    # strategies have no recent exits (typical case outside of high-vol
    # RTH windows).
    # Floor is the EARLIER of:
    #   - last_recompute_until_ts (when the cron last ran), and
    #   - max(trades.exit_fill_ts) + 1 second (the actual data baseline)
    # The latter catches the case where a previous run stamped
    # last_recompute but failed to insert trades it should have (engine
    # crash, warmup misconfigured, etc.). Without this, the alerts-gate
    # would falsely think "no new trades since stamp" and skip forever
    # — even though there are exit alerts past the actual data baseline
    # waiting to be picked up. Documented bug found 2026-05-07 on sid 154.
    since_check_iso = last_until or (
        datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    try:
        # Latest exit timestamp in the trades table for this strategy
        from trades_store import load_trades_for_strategy as _load_trades_floor
        _existing_for_floor = _load_trades_floor(
            strategy_id, user_id) or []
        if _existing_for_floor:
            max_exit_iso = max(
                str(t.get('exit_fill_ts'))
                for t in _existing_for_floor
                if t.get('exit_fill_ts')
            )
            # Use the EARLIER of last_until and max_exit. If max_exit is
            # earlier, that's the actual baseline we trust.
            if max_exit_iso < since_check_iso:
                since_check_iso = max_exit_iso
    except Exception as e:
        logger.warning(
            "[ALGO-APPEND] strategy=%s data-baseline lookup failed: %s",
            strategy_id, e)

    try:
        # alerts.side='exit' identifies exit alerts; event_type is always
        # 'fill' regardless of side. Verified 2026-05-06 against schema.
        _alerts_check = _raw_client.table('alerts').select('id') \
            .eq('strategy_id', strategy_id) \
            .eq('side', 'exit') \
            .gte('fill_ts', since_check_iso).limit(1).execute()
        recent_exits_count = len(_alerts_check.data or [])
    except Exception as e:
        logger.warning(
            "[ALGO-APPEND] strategy=%s alerts-gate query failed: %s — "
            "falling through to full engine run",
            strategy_id, e)
        recent_exits_count = -1  # gate disabled on error

    # `force=True` (manual user click) bypasses the alerts-gate so the
    # engine always runs — useful for catching missed trades (algo-only,
    # no live alert) or just confirming the system is current.
    if recent_exits_count == 0 and not force:
        # No new closed trades possible — stamp + skip.
        # Also bump data_refreshed_at on the strategies row so the
        # "Updated X ago" UI tag reflects cron activity (visual proof
        # the cron is alive even when no trades changed). This is a
        # column update — does NOT touch config JSONB.
        cfg['last_recompute_until_ts'] = now_iso
        _stamp_config(strategy_id, user_id, cfg)
        try:
            update_strategy_admin(strategy_id, user_id, {
                'data_refreshed_at': now_iso,
            })
        except Exception as e:
            logger.warning(
                "[ALGO-APPEND] strategy=%s data_refreshed_at bump failed: %s",
                strategy_id, e)
        return {'status': 'skipped',
                'reason': 'no_recent_exits',
                'inserted': 0,
                'cutoff': cutoff_iso,
                'since_check': since_check_iso,
                'elapsed_s': 0.0}

    # Engine context — same admin-mode swap as recompute_and_persist
    _prev_user = get_current_user_id()
    _need_ctx_swap = _prev_user != user_id
    if _need_ctx_swap:
        set_admin_user_context(user_id)

    t0 = _time.time()
    try:
        # Phase 1 (M8.7 — 2026-05-06): use the windowed engine helper
        # when existing trades give us a `since_dt` cutoff. Drops engine
        # cost from ~30-40s (full forward-test history) to ~3-5s
        # (1-2 day window). Falls back to full backtest only on cold
        # start (no existing trades to derive `since` from).
        #
        # Long-cycle secondary TFs (1Hour+) need more warmup than the
        # 100-bar default — fall back to full path for those.
        from trades_store import load_trades_for_strategy as _load_trades
        existing_trades_for_window = _load_trades(strategy_id, user_id) or []
        use_windowed = (
            len(existing_trades_for_window) > 0
            and not svc._has_long_cycle_secondary_tf(strat)
        )
        engine_path = 'windowed' if use_windowed else 'full'

        try:
            if use_windowed:
                # Derive since_dt from the latest existing trade's
                # entry_fill_ts (mirrors Streamlit's pattern at
                # app.py:3173).
                latest_entry_iso = max(
                    str(t.get('entry_fill_ts'))
                    for t in existing_trades_for_window
                    if t.get('entry_fill_ts')
                )
                since_dt = datetime.fromisoformat(latest_entry_iso)
                # Algo-model split (2026-05-07): cron dispatches under
                # algo_model so the algo-history lane reflects what the
                # live engine SHOULD have produced on the same data
                # (cache_locked by default). Falls back to backtest_model
                # for strategies that pre-date the algo_model field.
                algo_model = cfg.get('algo_model') or cfg.get('backtest_model')
                all_trades_df = svc.get_strategy_trades_for_window(
                    strat, since_dt=since_dt, until_dt=now_dt,
                    model_override=algo_model)
            else:
                algo_model = cfg.get('algo_model') or cfg.get('backtest_model')
                all_trades_df = svc.get_strategy_trades(
                    strat, model_override=algo_model)
        except Exception as e:
            logger.warning(
                "[ALGO-APPEND] strategy=%s engine run (%s) failed: %s",
                strategy_id, engine_path, e)
            return {'status': 'error', 'reason': f'engine: {e}',
                    'inserted': 0}

        if all_trades_df is None or len(all_trades_df) == 0:
            # Stamp last_recompute_until_ts so we don't re-run engine
            # next cycle for a strategy that has no trades.
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'cutoff': cutoff_iso, 'engine_path': engine_path,
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Filter: only commit closed trades whose exit_fill_ts is past
        # the lag boundary. Open trades stay live in engine; partially-
        # late-printed bars excluded.
        if 'exit_fill_ts' not in all_trades_df.columns:
            return {'status': 'error',
                    'reason': 'engine output missing exit_fill_ts column',
                    'inserted': 0}

        closed_mask = all_trades_df['exit_fill_ts'].notna()
        closed = all_trades_df[closed_mask].copy()
        if len(closed) == 0:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'cutoff': cutoff_iso, 'open_only': True,
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Coerce exit_fill_ts to UTC-aware Timestamp for comparison
        exit_ts_col = pd.to_datetime(closed['exit_fill_ts'], utc=True,
                                     errors='coerce')
        closed = closed[exit_ts_col <= pd.Timestamp(cutoff)]
        if len(closed) == 0:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'cutoff': cutoff_iso, 'reason': 'all_in_lag_window',
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Set-diff against existing DB trades — scope to cache_% so we
        # don't see backtest_<model> rows and accidentally consider them
        # already-inserted (which would suppress legitimate algo writes).
        # Phase 41 fix (2026-05-11 evening).
        from trades_store import load_trades_for_strategy, insert_trade
        existing = load_trades_for_strategy(
            strategy_id, user_id,
            data_source_filter='cache_%') or []
        existing_keys: set = set()
        for t in existing:
            ek = t.get('entry_fill_ts')
            xk = t.get('exit_fill_ts')
            if ek and xk:
                existing_keys.add((str(ek), str(xk)))

        new_records = []
        for _, row in closed.iterrows():
            ek_raw = row.get('entry_fill_ts')
            xk_raw = row.get('exit_fill_ts')
            if pd.isna(ek_raw) or pd.isna(xk_raw):
                continue
            ek = ek_raw.isoformat() if hasattr(ek_raw, 'isoformat') else str(ek_raw)
            xk = xk_raw.isoformat() if hasattr(xk_raw, 'isoformat') else str(xk_raw)
            if (ek, xk) in existing_keys:
                continue
            new_records.append(row)

        # Convert new records to JSON-safe dicts (re-uses existing
        # _serialize_trades for shape consistency with manual Refresh).
        # Tag with cache_<algo_model> so the divergence reader can find
        # them and so future recomputes scope their DELETE correctly.
        ds_tag = f'cache_{algo_model}'
        if new_records:
            new_df = pd.DataFrame(new_records)
            new_serialized = _serialize_trades(new_df)
        else:
            new_serialized = []

        inserted = 0
        for rec in new_serialized:
            try:
                rec['data_source'] = ds_tag
                if insert_trade(strategy_id, user_id, rec):
                    inserted += 1
            except Exception as e:
                logger.warning(
                    "[ALGO-APPEND] strategy=%s insert_trade failed: %s",
                    strategy_id, e)

        # Auto-Hi-Fi for cache-aligned and Hi-Fi-opted-in backtest
        # models. Refines L-type entry/exit timestamps from bar-level
        # to per-second precision so algo history matches live alerts.
        # _hifi_resolve_trades is idempotent (skips rows already
        # hifi_resolved=True), so calling it every cycle is cheap.
        # Failure NEVER blocks the cron — alert table is source of
        # truth for live; algo history is best-effort backtest mirror.
        #
        # Coverage (2026-05-07 expansion):
        # - rest_hifi:        explicit user opt-in
        # - cache_locked:     bar parity is the goal AND L-type
        #                     sub-second alignment is the win we want
        # - cache_corrected:  same logic when Phase D ships
        # - rest_only:        skip — fast bulk default
        # Hi-Fi gate uses algo_model (2026-05-07) — this path produces
        # the algo lane, so Hi-Fi eligibility follows that model not
        # the backtest model.
        HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
        hifi_summary = None
        algo_model_for_hifi = cfg.get('algo_model') or cfg.get('backtest_model')
        if inserted > 0 and algo_model_for_hifi in HIFI_BACKTEST_MODELS:
            try:
                from api.routers.strategies import run_hifi_pass2
                hifi_summary = run_hifi_pass2(
                    strategy_id, user={'id': user_id})
                logger.info(
                    "[ALGO-APPEND] strategy=%s algo=%s Hi-Fi pass: refined "
                    "entries=%s exits=%s persisted=%s",
                    strategy_id, algo_model_for_hifi,
                    hifi_summary.get('entries_refined', 0),
                    hifi_summary.get('exits_refined', 0),
                    hifi_summary.get('persisted', 0))
            except Exception as e:
                logger.warning(
                    "[ALGO-APPEND] strategy=%s Hi-Fi pass failed: %s",
                    strategy_id, e)

        # Update kpis + equity_curve. KPI source-of-truth differs by
        # engine_path:
        #   - 'full' (cold-start): all_trades_df IS the full backtest,
        #     use it directly.
        #   - 'windowed' (incremental): all_trades_df only contains the
        #     small windowed slice. Load full trades from DB (now
        #     including the inserted ones) and compute kpis from the
        #     merged set. Otherwise kpis would reflect only recent
        #     trades.
        do_full_update = (inserted > 0) or is_first_run
        update_payload: Dict[str, Any] = {}

        if do_full_update:
            try:
                if engine_path == 'windowed':
                    # Re-load all trades from DB for the strategy, build
                    # a kpi-ready DataFrame from them.
                    all_db_trades = load_trades_for_strategy(
                        strategy_id, user_id) or []
                    full_trades_df = svc.trades_df_from_stored(all_db_trades)
                    # Trading days for KPI denominator: derive from the
                    # FULL trades period (entry_time min → exit_time max),
                    # not the windowed source bars. Fallback to 1 if
                    # trades_df is empty / has no entry_time.
                    if (len(full_trades_df) > 0 and
                            'entry_time' in full_trades_df.columns):
                        first_entry = full_trades_df['entry_time'].min()
                        last_exit = full_trades_df.get(
                            'exit_time', full_trades_df['entry_time']).max()
                        if pd.notna(first_entry) and pd.notna(last_exit):
                            # Use trading-day count between min entry and
                            # max exit. svc.count_trading_days expects a
                            # DataFrame with DatetimeIndex; build a quick
                            # one for the date range.
                            try:
                                date_range = pd.date_range(
                                    first_entry.normalize(),
                                    last_exit.normalize(),
                                    freq='B')
                                trading_days = max(1, len(date_range))
                            except Exception:
                                trading_days = 1
                        else:
                            trading_days = 1
                    else:
                        trading_days = 1
                    kpis_df = full_trades_df
                else:
                    # Full-engine path: use all_trades_df as-is
                    trading_days = all_trades_df.attrs.get('trading_days') \
                        if hasattr(all_trades_df, 'attrs') else None
                    if not trading_days or trading_days < 1:
                        trading_days = 1
                    kpis_df = all_trades_df
            except Exception as e:
                logger.warning(
                    "[ALGO-APPEND] strategy=%s kpi-source build failed: %s — "
                    "falling back to engine output",
                    strategy_id, e)
                kpis_df = all_trades_df
                trading_days = 1

            try:
                kpis = svc.calculate_kpis(kpis_df,
                                          total_trading_days=trading_days)
            except Exception as e:
                logger.warning(
                    "[ALGO-APPEND] strategy=%s calc_kpis failed: %s",
                    strategy_id, e)
                kpis = None

            try:
                from api.services.backtest_service import _build_equity_curve
                eq_data = _build_equity_curve(kpis_df)
                boundary_index = None
                if strat.get('forward_test_start'):
                    fwd_start = datetime.fromisoformat(
                        strat['forward_test_start'])
                    bt_portion, _ = svc.split_trades_at_boundary(
                        kpis_df, fwd_start)
                    boundary_index = (len(bt_portion)
                                      if len(bt_portion) > 0 else None)
                equity_curve_data = {
                    'exit_times': [p.get('timestamp', '') for p in eq_data],
                    'cumulative_r': [p.get('cumulative_r', 0)
                                     for p in eq_data],
                    'boundary_index': boundary_index,
                }
            except Exception as e:
                logger.warning(
                    "[ALGO-APPEND] strategy=%s build_equity failed: %s",
                    strategy_id, e)
                equity_curve_data = None

            if kpis is not None:
                update_payload['kpis'] = kpis
                update_payload['kpis_computed_at'] = cutoff_iso
                update_payload['kpis_stale_since'] = None
            if equity_curve_data is not None:
                update_payload['equity_curve_data'] = equity_curve_data
            update_payload['data_refreshed_at'] = cutoff_iso

        # Column-only update via update_strategy_admin (handles trades-
        # table redirect, etc). Config update goes through raw admin
        # client below to dodge the partial-JSONB wipe bug
        # (feedback_jsonb_partial_updates.md): update_strategy_admin
        # treats unknown keys as a config REPLACEMENT, which would erase
        # live_model and other config fields. Caller-side full-merge is
        # the documented safe pattern.
        if update_payload:
            update_strategy_admin(strategy_id, user_id, update_payload)

        # Stamp last_recompute_until_ts via direct config merge.
        cfg['last_recompute_until_ts'] = now_iso
        _stamp_config(strategy_id, user_id, cfg)

        elapsed_s = round(_time.time() - t0, 2)
        # Engine scope: how many bars the engine ran over (from the
        # source DF in get_strategy_trades_for_window). Surfaced so the
        # Jobs UI can show "scanned X bars over W days" — answers the
        # "how far back did we look" question.
        bars_processed = None
        window_days = None
        if hasattr(all_trades_df, 'attrs'):
            bars_processed = all_trades_df.attrs.get('source_bar_count')
            window_days = all_trades_df.attrs.get('window_days')
        result = {
            'status': 'appended' if inserted > 0 else 'no_new_trades',
            'inserted': inserted,
            'cutoff': cutoff_iso,
            'elapsed_s': elapsed_s,
            'kpis_updated': bool(do_full_update),
            'engine_path': engine_path,
            'bars_processed': bars_processed,
            'window_days': window_days,
        }
        if hifi_summary is not None:
            result['hifi'] = {
                'entries_refined': hifi_summary.get('entries_refined', 0),
                'exits_refined': hifi_summary.get('exits_refined', 0),
                'persisted': hifi_summary.get('persisted', 0),
            }
        return result
    finally:
        if _need_ctx_swap:
            clear_current_user()


def append_recent_trades_for_user(
    user_id: str,
    lag_minutes: int = None,
    max_seconds: float = 240.0,
) -> Dict[str, Any]:
    """Iterate strategies for a user and append recent trades for each.

    ``max_seconds`` caps total wall time to avoid blocking the cron
    longer than one cycle. When exceeded, processing stops and remaining
    strategies are picked up next cycle (their last_recompute_until_ts
    won't advance, so they get priority).

    Returns a summary dict suitable for log inspection.
    """
    import time as _time

    if not _algo_history_cron_enabled():
        return {'status': 'disabled', 'processed': 0}

    if lag_minutes is None:
        lag_minutes = _ALGO_HISTORY_LAG_MINUTES

    from db import load_strategies_admin

    try:
        strategies = load_strategies_admin(user_id) or []
    except Exception as e:
        return {'status': 'error', 'reason': f'load_strategies: {e}',
                'processed': 0}

    # Skip strategies that obviously can't produce trades (no
    # entry_trigger_confluence_id) — same gate the engine enforces.
    eligible = [s for s in strategies
                if 'entry_trigger_confluence_id' in s and s.get('id')]

    # Sort by last_recompute_until_ts (oldest first → fairness across
    # cycles when budget runs out).
    def _sort_key(s):
        cfg = s.get('config') or {}
        return cfg.get('last_recompute_until_ts') or ''
    eligible.sort(key=_sort_key)

    summary: Dict[str, Any] = {
        'status': 'ok',
        'processed': 0,
        'inserted_total': 0,
        'skipped': 0,
        'errors': 0,
        'budget_exhausted': False,
        'detail': [],
    }
    t0 = _time.time()
    for s in eligible:
        if _time.time() - t0 > max_seconds:
            summary['budget_exhausted'] = True
            break
        sid = s.get('id')
        try:
            r = append_new_trades_for_strategy(sid, user_id, lag_minutes)
        except Exception as e:
            logger.exception(
                "[ALGO-APPEND-USER] strategy=%s crashed: %s", sid, e)
            r = {'status': 'error', 'reason': str(e), 'inserted': 0}
        summary['processed'] += 1
        summary['inserted_total'] += int(r.get('inserted') or 0)
        if r.get('status') == 'skipped':
            summary['skipped'] += 1
        elif r.get('status') == 'error':
            summary['errors'] += 1
        summary['detail'].append({'sid': sid, **r})

    summary['elapsed_s'] = round(_time.time() - t0, 2)
    return summary


# ============================================================
# Lane×mode matrix completers (algo_model split — 2026-05-08)
# ============================================================

def append_new_backtest_trades_for_strategy(
    strategy_id: int,
    user_id: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Forward-append on the BACKTEST lane.

    Mirrors `append_new_trades_for_strategy` (which writes to the trades
    table under `algo_model`) but writes to `stored_trades` JSONB under
    `backtest_model`. Extends the existing snapshot without a full
    recompute.

    Pattern:
      1. Load strategy + existing stored_trades
      2. Find latest entry_fill_ts in stored_trades
      3. Run engine on `[latest, now - lag]` window with
         `model_override=backtest_model`
      4. Filter closed trades past lag boundary, dedup vs existing
         stored_trades by (entry_fill_ts, exit_fill_ts)
      5. Concat new trades to stored_trades, write back
      6. Recompute KPIs from merged set
      7. Hi-Fi pass eligible if backtest_model in HIFI set
      8. Stamp `last_recompute_until_ts`

    `force=True` bypasses the recently-processed gate.

    Returns:
        {'status': 'appended'|'no_new_trades'|'error',
         'inserted': int, 'elapsed_s': float, ...}
    """
    import time as _time
    import pandas as pd
    import services as svc
    from db import (
        get_strategy_by_id_admin,
        update_strategy_admin,
        set_admin_user_context,
        get_current_user_id,
        clear_current_user,
    )

    t0 = _time.time()
    strat = get_strategy_by_id_admin(strategy_id, user_id)
    if strat is None:
        return {'status': 'error',
                'reason': f'strategy {strategy_id} not found',
                'inserted': 0}

    cfg = (strat.get('config') or {})
    bt_model = cfg.get('backtest_model') or strat.get('backtest_model')
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    cutoff = now_dt - timedelta(minutes=_ALGO_HISTORY_LAG_MINUTES)
    cutoff_iso = cutoff.isoformat()

    # W2 BT-APPEND optimization (2026-05-12): query the trades table
    # directly for the latest entry timestamp instead of paginating
    # through stored_trades. Avoids the 60s+ DELETE+INSERT-merged cycle
    # that the legacy path triggered every refresh; for a 6,000-trade
    # strategy with 5 new trades, this turns ~60s into ~5s of DB work.
    from db import get_max_entry_ts_admin, insert_trade_admin
    ds_tag = f'backtest_{bt_model}'
    latest_ts_iso = get_max_entry_ts_admin(
        strategy_id, user_id, data_source_filter='backtest_%')

    if latest_ts_iso is None:
        # No existing baseline in trades table — fall through to a full
        # recompute because forward append needs an anchor timestamp.
        return {
            'status': 'skipped',
            'reason': 'no_baseline_run_full_recompute_first',
            'inserted': 0,
            'elapsed_s': round(_time.time() - t0, 2),
        }

    try:
        since_dt = datetime.fromisoformat(latest_ts_iso)
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        return {'status': 'error',
                'reason': f'parse latest entry_fill_ts: {e}',
                'inserted': 0}

    # Set admin context if not already set (worker / cron callers)
    _prev_user = get_current_user_id()
    _need_ctx_swap = _prev_user != user_id
    if _need_ctx_swap:
        set_admin_user_context(user_id)

    try:
        try:
            all_trades_df = svc.get_strategy_trades_for_window(
                strat, since_dt=since_dt, until_dt=now_dt,
                model_override=bt_model)
        except Exception as e:
            logger.warning(
                "[BT-APPEND] strategy=%s engine run failed: %s",
                strategy_id, e)
            return {'status': 'error', 'reason': f'engine: {e}',
                    'inserted': 0}

        if all_trades_df is None or len(all_trades_df) == 0:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'cutoff': cutoff_iso,
                    'elapsed_s': round(_time.time() - t0, 2)}

        if 'exit_fill_ts' not in all_trades_df.columns:
            return {'status': 'error',
                    'reason': 'engine output missing exit_fill_ts column',
                    'inserted': 0}

        closed_mask = all_trades_df['exit_fill_ts'].notna()
        closed = all_trades_df[closed_mask].copy()
        if len(closed) == 0:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'cutoff': cutoff_iso, 'open_only': True,
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Filter to past lag boundary
        exit_ts_col = pd.to_datetime(closed['exit_fill_ts'], utc=True,
                                     errors='coerce')
        closed = closed[exit_ts_col <= pd.Timestamp(cutoff)]
        if len(closed) == 0:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'cutoff': cutoff_iso, 'reason': 'all_in_lag_window',
                    'elapsed_s': round(_time.time() - t0, 2)}

        # True-append: emit only trades strictly newer than the anchor
        # entry_ts. The engine windowed query started at since_dt =
        # latest_ts so trades AT that timestamp may collide — the unique
        # index will reject re-inserts naturally (insert_trade_admin
        # returns None on unique violation) but we filter explicitly to
        # avoid the wasted attempts.
        new_records = []
        for _, row in closed.iterrows():
            ek_raw = row.get('entry_fill_ts')
            xk_raw = row.get('exit_fill_ts')
            if pd.isna(ek_raw) or pd.isna(xk_raw):
                continue
            ek = ek_raw.isoformat() if hasattr(ek_raw, 'isoformat') else str(ek_raw)
            if ek <= latest_ts_iso:
                continue  # already in DB (boundary or older)
            new_records.append(row)

        if not new_records:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'cutoff': cutoff_iso, 'reason': 'all_already_stored',
                    'elapsed_s': round(_time.time() - t0, 2)}

        # INSERT only the new rows. Per-row insert_trade_admin handles
        # unique violations gracefully (returns None on collision) and
        # has retry baked in for transient Supabase errors. For typical
        # cron-cadence appends, new_records is 1-10 rows — single-row
        # inserts are appropriate. Bulk strategies might emit 50+ rows
        # but even then we're far below the legacy 6,000-row rewrite.
        new_df = pd.DataFrame(new_records)
        new_serialized = _serialize_trades(new_df)
        inserted_count = 0
        for rec in new_serialized:
            rec_copy = dict(rec)
            rec_copy['data_source'] = ds_tag
            try:
                row_saved = insert_trade_admin(
                    strategy_id, user_id, rec_copy)
                if row_saved is not None:
                    inserted_count += 1
            except Exception as e:
                logger.warning(
                    "[BT-APPEND] sid=%s insert_trade failed: %s",
                    strategy_id, e)

        try:
            from trades_store import _cache_invalidate
            _cache_invalidate(strategy_id, user_id)
        except Exception:
            pass

        refreshed_at = now_iso

        # Lazy KPI recompute: re-read all backtest_% rows for this
        # strategy and compute KPIs over the full set. Read is paginated
        # but cheap relative to the legacy DELETE+INSERT cycle. UPDATE
        # writes only KPIs + equity_curve_data on the strategy row.
        from db import load_trades_admin
        all_backtest = load_trades_admin(
            strategy_id, user_id, data_source_filter='backtest_%') or []
        # Sort chronologically for the KPI + equity curve math
        all_backtest_sorted = sorted(
            all_backtest,
            key=lambda t: str(t.get('entry_fill_ts') or ''))
        merged_df = svc.trades_df_from_stored(all_backtest_sorted)

        trading_days = svc.count_trading_days(merged_df) \
            if hasattr(merged_df, 'index') and len(merged_df) > 0 else 1
        kpis = svc.calculate_kpis(merged_df, total_trading_days=trading_days)

        from api.services.backtest_service import _build_equity_curve
        eq_data = _build_equity_curve(merged_df)
        boundary_index = None
        if strat.get('forward_test_start'):
            fwd_start = datetime.fromisoformat(strat['forward_test_start'])
            bt_portion, _ = svc.split_trades_at_boundary(merged_df, fwd_start)
            boundary_index = len(bt_portion) if len(bt_portion) > 0 else None
        equity_curve_data = {
            'exit_times': [p.get('timestamp', '') for p in eq_data],
            'cumulative_r': [p.get('cumulative_r', 0) for p in eq_data],
            'boundary_index': boundary_index,
        }

        # KPIs + equity curve stay as strategies-row columns.
        update_strategy_admin(strategy_id, user_id, {
            'kpis': kpis,
            'equity_curve_data': equity_curve_data,
            'data_refreshed_at': refreshed_at,
            'kpis_computed_at': refreshed_at,
            'kpis_stale_since': None,
        })

        cfg['last_recompute_until_ts'] = now_iso
        _stamp_config(strategy_id, user_id, cfg)

        # Hi-Fi pass on appended set (idempotent — skips already-resolved)
        HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
        hifi_summary = None
        if bt_model in HIFI_BACKTEST_MODELS:
            try:
                from api.routers.strategies import run_hifi_pass2
                hifi_summary = run_hifi_pass2(
                    strategy_id, user={'id': user_id})
                logger.info(
                    "[BT-APPEND] strategy=%s bt=%s Hi-Fi pass: refined "
                    "entries=%s exits=%s persisted=%s",
                    strategy_id, bt_model,
                    hifi_summary.get('entries_refined', 0),
                    hifi_summary.get('exits_refined', 0),
                    hifi_summary.get('persisted', 0))
            except Exception as e:
                logger.warning(
                    "[BT-APPEND] strategy=%s Hi-Fi pass failed: %s",
                    strategy_id, e)

        result = {
            'status': 'appended',
            'inserted': inserted_count,
            'cutoff': cutoff_iso,
            'elapsed_s': round(_time.time() - t0, 2),
        }
        if hifi_summary is not None:
            result['hifi'] = {
                'entries_refined': hifi_summary.get('entries_refined', 0),
                'exits_refined': hifi_summary.get('exits_refined', 0),
                'persisted': hifi_summary.get('persisted', 0),
            }
        return result
    finally:
        if _need_ctx_swap:
            clear_current_user()


def recompute_and_persist_algo_trades(
    strategy_id: int,
    user_id: str,
) -> Dict[str, Any]:
    """Full recompute on the ALGO lane.

    Mirrors `recompute_and_persist_stored_trades` (which writes to
    `stored_trades` JSONB under `backtest_model`) but writes to the
    trades table under `algo_model`. Wipes existing trades-table rows
    for the strategy and re-inserts the full set.

    Pattern:
      1. Load strategy
      2. Run engine on full strategy window with
         `model_override=algo_model`
      3. DELETE existing trades-table rows for strategy
      4. INSERT all generated trades (alert linkages reform via
         existing match-on-insert logic)
      5. Hi-Fi pass eligible if algo_model in HIFI set
      6. Stamp `last_recompute_until_ts`

    Returns:
        {'status': 'refreshed'|'no_trades'|'error',
         'inserted': int, 'elapsed_s': float, ...}
    """
    import time as _time
    import services as svc
    from db import (
        get_strategy_by_id_admin,
        get_admin_client,
        set_admin_user_context,
        get_current_user_id,
        clear_current_user,
    )

    t0 = _time.time()
    strat = get_strategy_by_id_admin(strategy_id, user_id)
    if strat is None:
        return {'status': 'error',
                'reason': f'strategy {strategy_id} not found',
                'inserted': 0}

    cfg = (strat.get('config') or {})
    algo_model = cfg.get('algo_model') or cfg.get('backtest_model') \
        or strat.get('algo_model') or strat.get('backtest_model')
    now_iso = datetime.now(timezone.utc).isoformat()

    _prev_user = get_current_user_id()
    _need_ctx_swap = _prev_user != user_id
    if _need_ctx_swap:
        set_admin_user_context(user_id)

    try:
        try:
            all_trades_df = svc.get_strategy_trades(
                strat, model_override=algo_model)
        except Exception as e:
            logger.warning(
                "[ALGO-RECOMPUTE] strategy=%s engine run failed: %s",
                strategy_id, e)
            return {'status': 'error', 'reason': f'engine: {e}',
                    'inserted': 0}

        if all_trades_df is None or len(all_trades_df) == 0:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_trades', 'inserted': 0,
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Filter to closed trades only — open positions can't be persisted
        # to the trades table (need exit_fill_ts).
        if 'exit_fill_ts' in all_trades_df.columns:
            closed_mask = all_trades_df['exit_fill_ts'].notna()
            closed_df = all_trades_df[closed_mask].copy()
        else:
            closed_df = all_trades_df

        if len(closed_df) == 0:
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_trades', 'inserted': 0,
                    'reason': 'all_open',
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Phase 41 fix (2026-05-11 evening): scope DELETE to cache_% rows
        # only. The previous unfiltered DELETE wiped the backtest_<model>
        # rows the backtest lane wrote moments earlier (Update All Data
        # runs backtest THEN algo back-to-back). That left every strategy
        # with all-NULL data_source after a bulk update, breaking the
        # Divergence tab. Sid 152 + sid 154 hit this state earlier today.
        ds_tag = f'cache_{algo_model}'
        try:
            from db import replace_trades_admin
            tagged_closed = []
            for rec in _serialize_trades(closed_df):
                rec_copy = dict(rec)
                rec_copy['data_source'] = ds_tag
                tagged_closed.append(rec_copy)
            # replace_trades_admin handles the DELETE+chunked INSERT with
            # retry-on-transient-error, scoped to cache_% so backtest rows
            # written earlier in the same bulk job survive.
            inserted = replace_trades_admin(
                strategy_id, user_id, tagged_closed,
                data_source_filter='cache_%')
            try:
                from trades_store import _cache_invalidate
                _cache_invalidate(strategy_id, user_id)
            except Exception:
                pass
        except Exception as e:
            logger.exception(
                "[ALGO-RECOMPUTE] strategy=%s trades-table write failed: %s",
                strategy_id, e)
            return {'status': 'error',
                    'reason': f'write: {e}',
                    'inserted': 0}

        cfg['last_recompute_until_ts'] = now_iso
        _stamp_config(strategy_id, user_id, cfg)

        # Hi-Fi pass on the trades table (idempotent)
        HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
        hifi_summary = None
        if inserted > 0 and algo_model in HIFI_BACKTEST_MODELS:
            try:
                from api.routers.strategies import run_hifi_pass2
                hifi_summary = run_hifi_pass2(
                    strategy_id, user={'id': user_id})
                logger.info(
                    "[ALGO-RECOMPUTE] strategy=%s algo=%s Hi-Fi pass: "
                    "refined entries=%s exits=%s persisted=%s",
                    strategy_id, algo_model,
                    hifi_summary.get('entries_refined', 0),
                    hifi_summary.get('exits_refined', 0),
                    hifi_summary.get('persisted', 0))
            except Exception as e:
                logger.warning(
                    "[ALGO-RECOMPUTE] strategy=%s Hi-Fi pass failed: %s",
                    strategy_id, e)

        result = {
            'status': 'refreshed',
            'inserted': inserted,
            'elapsed_s': round(_time.time() - t0, 2),
        }
        if hifi_summary is not None:
            result['hifi'] = {
                'entries_refined': hifi_summary.get('entries_refined', 0),
                'exits_refined': hifi_summary.get('exits_refined', 0),
                'persisted': hifi_summary.get('persisted', 0),
            }
        return result
    finally:
        if _need_ctx_swap:
            clear_current_user()
