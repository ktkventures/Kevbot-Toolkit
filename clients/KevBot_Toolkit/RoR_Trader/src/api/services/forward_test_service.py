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


def _os_env_skip_algo_cold_seed() -> bool:
    """1.5 kill-switch (default ON): skip the historical algo-lane cold-seed on
    a fresh strategy's first UND (the lane fills live instead). Set
    RORT_SKIP_ALGO_COLD_SEED=0 to restore the old full-backtest cold-seed."""
    import os
    return os.getenv('RORT_SKIP_ALGO_COLD_SEED', '1') == '1'


def _forward_divider_dt(strat: dict):
    """Forward-test divider (2026-06-18 decision): the equity-curve blue/yellow
    split anchors to `in_sample_end` (the out-of-sample start a trader means by
    "forward test"), falling back to `forward_test_start`. Returns a tz-aware
    UTC datetime or None. Keeps the equity boundary aligned with the Fwd count
    (`_count_forward_trades` uses the same divider). The cache/live lane + alerts
    are a SEPARATE live-monitoring overlay (see Mass_Builder_Forward_Test_Bugs)."""
    cfg = strat.get('config') if isinstance(strat.get('config'), dict) else {}
    raw = (strat.get('in_sample_end') or (cfg or {}).get('in_sample_end')
           or strat.get('forward_test_start'))
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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

    # Tier 2 (gated, default OFF via RORT_USE_DOUGH_CACHE): if a fresh dough
    # exists for this strategy's window, re-bake the lane from it
    # (run_trades_from_cache, 10-50x) instead of a full prepare+engine recompute.
    # Fallback-safe: any miss / stale / error / trigger-not-in-superset falls
    # through to the full svc.get_strategy_trades. CRITICAL: carry trading_days
    # from the dough metadata onto .attrs, else calculate_kpis falls back to
    # count_trading_days(trades_df)=1 and inflates daily_r ~180x (see line ~190).
    import os as _os
    all_trades = None
    if _os.getenv('RORT_USE_DOUGH_CACHE', '0') == '1':
        try:
            import bar_cache_store as _bcs
            from unified_engine import run_trades_from_cache as _rtfc
            _got = _bcs.get_dough(_bcs.dough_key(strat))
            if _got is not None:
                _cache, _meta = _got
                # The dough covers warmup+visible bars; the persisted lane is
                # visible-only (MB trims to visible_start). Require _visible_start
                # so we can trim the re-bake identically — without it we'd
                # over-count warmup trades, so treat as a miss (full recompute).
                _vs = (_meta or {}).get('_visible_start')
                if not _vs:
                    logger.info(
                        "[Tier2] strategy=%s dough lacks _visible_start "
                        "-> full recompute", strategy_id)
                else:
                    _reb = _rtfc(_cache, strat, _meta)
                    if _reb is not None and len(_reb) > 0:
                        from strategy_data import trim_trades_to_visible as _trim
                        import pandas as _pd
                        _reb = _trim(_reb, _pd.Timestamp(_vs))
                        _td = (_meta or {}).get('_trading_days')
                        if _td:
                            try:
                                _reb.attrs['trading_days'] = _td
                            except Exception:
                                pass
                        all_trades = _reb
                        logger.info(
                            "[Tier2] strategy=%s lane re-baked from dough "
                            "(%d trades trimmed to visible %s, trading_days=%s)",
                            strategy_id, len(all_trades), _vs, _td)
        except Exception as _e:
            logger.warning(
                "[Tier2] strategy=%s dough re-bake failed (%s) — full recompute",
                strategy_id, _e)
            all_trades = None

    if all_trades is None:
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
    _div = _forward_divider_dt(strat)
    if _div is not None:
        bt_portion, _ = svc.split_trades_at_boundary(all_trades, _div)
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
        bt_model = strat.get('backtest_model') or 'rest_hifi'
    HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
    hifi_summary = None
    if bt_model in HIFI_BACKTEST_MODELS and len(stored) > 0:
        try:
            from api.routers.strategies import run_hifi_pass2
            # Phase D (2026-05-14): FT-recompute writes backtest rows,
            # so scope Hi-Fi to the backtest lane.
            hifi_summary = run_hifi_pass2(
                strategy_id,
                data_source_filter='backtest_%',
                user={'id': user_id},
            )
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

# ── B2 fix (2026-06-15): append edge-band replace ──────────────────────
# Appends run a WINDOWED engine recompute that under-generates trades near
# the data edge (imperfect resume-snapshot warmup + unsettled REST), and the
# INSERT-only dedup on (strategy_id, entry_fill_ts, exit_fill_ts) means the
# fresh, settled recompute can never REPLACE the stale rows. The band
# fossilizes → live↔backtest pairing degrades the closer you measure to
# "now" (the "clean-then-messier" artifact; confirmed on sid 302 2026-06-15,
# afternoon 49%→98% after a full UAD). Fix: each append RECOMPUTES its
# trailing band with full UAD-parity warmup (NOT the windowed resume) and
# REPLACES it via db.replace_trades_in_window_admin. Band sits entirely
# behind the lag cutoff so it never touches unsettled bars.
_APPEND_EDGE_BAND_MINUTES = int(
    os.environ.get('APPEND_EDGE_BAND_MINUTES', '120'))
# Cron-cost throttle: the band recompute uses full UAD-parity warmup
# (~12s/strategy), so the periodic worker cron rewrites a strategy's band at
# most once per this interval. Manual "Update New Data" (force=True) always
# heals immediately. 30 min keeps the 120-min band re-settled ~4× before
# rows age out. (Future scale optimization: a convergence-sufficient warmup
# would cut the per-run cost ~5-10× and could relax this.)
_APPEND_EDGE_BAND_THROTTLE_SEC = int(
    os.environ.get('APPEND_EDGE_BAND_THROTTLE_SEC', '1800'))
# KILL-SWITCH (2026-06-15): the full-UAD-parity-warmup band recompute costs
# ~427s/strategy on a 10Sec strategy — not viable as-is. Default OFF so the
# code is dormant (appends behave exactly as before) until the approach is
# finalized (reduced warmup / snapshot-at-band-start). Flip to 'true' only
# with a cost-bounded recompute in place.
_APPEND_EDGE_BAND_ENABLED = (
    os.environ.get('APPEND_EDGE_BAND_ENABLED', 'false').lower() == 'true')
# Band-recompute warmup CAP (trading days). Full UAD warmup is ~20 days for a
# 10Sec strategy (≈48k bars → 427s) — overkill: indicators converge in a
# fraction of that. Cap the band recompute's warmup at this many days
# (validated to produce byte-identical band trades to full warmup). Far
# cheaper while honoring "never under-warm". Bump if a coarse secondary-TF
# strategy ever fails the identity check.
_EDGE_BAND_WARMUP_DAYS = float(
    os.environ.get('APPEND_EDGE_BAND_WARMUP_DAYS', '5'))
# Coarsest-TF convergence floor (bars). The warmup cap is scaled up so the
# COARSEST timeframe in the strategy always gets at least this many bars —
# makes the day-cap TF-safe (a 15m/1H gate auto-gets more calendar days; a
# 10Sec strategy stays at the validated floor). 300 ≈ generous for the
# indicator periods in use. (10Sec + a DAILY secondary is an extreme TF
# spread the cap can't serve cheaply — that's the snapshot-resume follow-up.)
_EDGE_BAND_CONV_BARS = int(
    os.environ.get('APPEND_EDGE_BAND_CONV_BARS', '300'))
# v2 (2026-06-16): band-recompute MODE. 'capped' = cold capped-warmup (proven,
# ~55-90s/strategy). 'snapshot' = resume from a rolling lagged base snapshot
# (~10s/strategy, ~6-9x faster → fleet appends + cron viable). Default
# 'capped' until snapshot mode is A/B-validated in prod; any snapshot miss/
# error falls back to capped, so snapshot mode can never produce worse data.
_APPEND_EDGE_BAND_MODE = os.environ.get(
    'APPEND_EDGE_BAND_MODE', 'capped').lower()
# Snapshot mode takes the lagged base snapshot this many minutes BEFORE
# edge_start, so Tier-3 always-start-flat's boundary miss (a trade signaled on
# the snapshot's last bar) lands in the buffer zone — OUTSIDE the replaced band
# → band stays at full UAD parity.
_EDGE_BAND_BUFFER_MIN = int(
    os.environ.get('APPEND_EDGE_BAND_BUFFER_MIN', '15'))


def _uad_warmup_bars(strat: dict, cap_days: float | None = None) -> int:
    """Cold-warmup `warmup_bars` sized to match a full UAD's natural
    warmup_days for this strategy, so indicator convergence at a window
    start matches Update-All-Data. Inverse of the warmup_days→bars mapping
    in services.get_strategy_trades_for_window.

    Single source of truth for the windowed backtest append (Tier 2) and
    the B2 edge-band replace in both append lanes — keeps their warmup
    identical so the band recompute is UAD-faithful.

    `cap_days` (B2 edge-band): cap the warmup at this many trading days to
    keep the band recompute cheap. TF-SAFE: the cap is scaled UP for coarse
    timeframes so the COARSEST TF always gets at least
    `_EDGE_BAND_CONV_BARS` bars to converge — i.e. a 10Sec strategy is
    capped hard (validated identical to full warmup) but a 15m/1H gate
    automatically gets more calendar days. The cap can only ever REDUCE
    warmup below the full requirement, never below the convergence floor,
    so it cannot under-warm. None = no cap (full UAD parity).
    """
    from strategy_data import resolve_visible_window, compute_warmup_days
    from data_loader import (
        BARS_PER_DAY, get_required_tfs_from_confluence, get_tf_from_label,
    )
    import math as _math
    uad_vs, uad_ve, _src = resolve_visible_window(strat)
    uad_visible_days = max(
        1.0, (uad_ve - uad_vs).total_seconds() / 86400.0)
    uad_warmup_days = compute_warmup_days(strat, uad_visible_days)
    req_labels = get_required_tfs_from_confluence(
        strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))
    all_tfs = [strat.get('timeframe', '1Min')] + list(sec_tfs)
    bpds = [BARS_PER_DAY.get(tf, 390) for tf in all_tfs]
    binding_bpd = min(bpd for bpd in bpds if bpd > 0) if bpds else 390
    if cap_days is not None:
        # TF-safe cap: floor at `cap_days`, but scale up so the coarsest TF
        # gets >= _EDGE_BAND_CONV_BARS bars (in trading days). For fine TFs
        # the floor dominates (validated); for coarse TFs the convergence
        # term dominates → more days. Never exceeds the full requirement.
        conv_days = (_EDGE_BAND_CONV_BARS / binding_bpd) if binding_bpd else 0
        eff_cap = max(cap_days, conv_days)
        uad_warmup_days = min(uad_warmup_days, eff_cap)
    return max(
        100, int(_math.ceil(uad_warmup_days * binding_bpd * 252 / 365)))


def _compute_band_df(svc, strat, cfg, lane, strategy_id, edge_start, edge_ceil,
                     model_override, diagnostics_source):
    """Recompute the band's trades and return a DataFrame.

    Mode 'capped' (default): cold capped-warmup recompute over
    [edge_start, edge_ceil] (proven, ~55-90s).

    Mode 'snapshot' (v2): resume from a rolling lagged base snapshot kept
    in cfg['band_base_snapshot_b64_{lane}'] at position B (~10s):
      1. Obtain a base snapshot AT base_target = edge_start - BUFFER:
         - warm: resume the existing base (B → base_target), capturing the
           new base snapshot (cheap, no warmup);
         - cold/stale/mismatch: seed it with one capped-warmup run to
           base_target (one-time, amortized).
      2. Resume base_target → edge_ceil for the band trades.
      3. Persist the base snapshot @ base_target into cfg for next append.
    The BUFFER keeps the always-start-flat boundary miss outside the band
    (caller filters to entry >= edge_start). Any miss/exception falls back
    to capped — snapshot mode can never produce worse data than capped.
    """
    cap_warmup = _uad_warmup_bars(strat, cap_days=_EDGE_BAND_WARMUP_DAYS)

    def _capped():
        return svc.get_strategy_trades_for_window(
            strat, since_dt=edge_start, until_dt=edge_ceil,
            model_override=model_override, resume_snapshot_b64=None,
            warmup_bars=cap_warmup, diagnostics_source=diagnostics_source)

    if _APPEND_EDGE_BAND_MODE != 'snapshot' or cfg is None or lane is None:
        return _capped()

    from unified_engine import (compute_backtest_fingerprint,
                                 deserialize_backtest_snapshot)
    fp = compute_backtest_fingerprint(strat)
    mid = f"{'backtest' if lane == 'bt' else 'algo'}_{model_override}"
    base_target = edge_start - timedelta(minutes=_EDGE_BAND_BUFFER_MIN)
    snap_key = f'band_base_snapshot_b64_{lane}'
    ts_key = f'band_base_snapshot_ts_{lane}'
    base_b64 = cfg.get(snap_key)
    base_dt = None
    if cfg.get(ts_key):
        try:
            base_dt = datetime.fromisoformat(cfg[ts_key])
            if base_dt.tzinfo is None:
                base_dt = base_dt.replace(tzinfo=timezone.utc)
        except Exception:
            base_dt = None
    try:
        valid_warm = (base_b64 and base_dt and base_dt <= base_target
                      and deserialize_backtest_snapshot(
                          base_b64, expected_fingerprint=fp,
                          expected_model_id=mid) is not None)
        if valid_warm:
            # Warm roll: resume existing base (B) forward to base_target.
            _roll_df, base_at_target = svc.get_strategy_trades_for_window(
                strat, since_dt=base_dt, until_dt=base_target,
                model_override=model_override, resume_snapshot_b64=base_b64,
                expected_fingerprint=fp, expected_model_id=mid,
                return_snapshot=True, diagnostics_source=diagnostics_source)
            _origin = f"rolled {base_dt.isoformat()[:19]}"
        else:
            # Cold seed: capped warmup up to base_target (one-time).
            _seed_df, base_at_target = svc.get_strategy_trades_for_window(
                strat, since_dt=base_target, until_dt=base_target,
                model_override=model_override, resume_snapshot_b64=None,
                warmup_bars=cap_warmup, expected_fingerprint=fp,
                expected_model_id=mid, return_snapshot=True,
                diagnostics_source=diagnostics_source)
            _origin = "cold-seed"
        if not base_at_target:
            raise ValueError("no base snapshot produced")
        # Band: resume base_target → edge_ceil (warmup-free).
        band_df, _ = svc.get_strategy_trades_for_window(
            strat, since_dt=base_target, until_dt=edge_ceil,
            model_override=model_override, resume_snapshot_b64=base_at_target,
            expected_fingerprint=fp, expected_model_id=mid,
            return_snapshot=True, diagnostics_source=diagnostics_source)
        # Persist the rolled/seeded base snapshot @ base_target for next time.
        cfg[snap_key] = base_at_target
        cfg[ts_key] = base_target.isoformat()
        logger.info(
            "[EDGE-BAND] sid=%s snapshot-mode band (%s → base %s)",
            strategy_id, _origin, base_target.isoformat()[:19])
        return band_df
    except Exception as e:
        logger.warning(
            "[EDGE-BAND] sid=%s snapshot mode failed (%s) — capped fallback",
            strategy_id, e)
        return _capped()


def _replace_edge_band(svc, strat, strategy_id, user_id, cutoff,
                       model_override, ds_tag, data_source_filter,
                       diagnostics_source, cfg=None, lane=None):
    """B2 fix — recompute the trailing edge band
    [cutoff - _APPEND_EDGE_BAND_MINUTES, cutoff] and REPLACE it in the
    trades table. Mode (capped warmup vs snapshot resume) is chosen by
    `_compute_band_df`; this fn owns the band filter, no-wipe guard, and
    the replace.

    The windowed append under-generates trades near the data edge
    (unsettled REST + imperfect resume warmup) and insert-only dedup
    never rewrites them, so the band fossilizes → live↔backtest pairing
    degrades the closer you measure to "now". This rewrites the band
    every append with a settled, full-warmup recompute. The band sits
    entirely behind the lag cutoff, so it never touches unsettled bars.

    Independent of the windowed append's new-trade detection so it heals
    prior fossils even on a "no new trades" append. Returns rows written.

    Fully guarded: a band-replace failure NEVER crashes the append (the
    band is a measurement-trust safety net, not the append's job).
    """
    import pandas as pd  # local import — module-level scope lacks pd
    edge_ceil = cutoff
    edge_start = cutoff - timedelta(minutes=_APPEND_EDGE_BAND_MINUTES)
    try:
        band_df = _compute_band_df(
            svc, strat, cfg, lane, strategy_id, edge_start, edge_ceil,
            model_override, diagnostics_source)
        band_records = []
        if (band_df is not None and len(band_df) > 0
                and 'exit_fill_ts' in band_df.columns):
            bxt = pd.to_datetime(band_df['exit_fill_ts'], utc=True,
                                 errors='coerce')
            bet = pd.to_datetime(band_df['entry_fill_ts'], utc=True,
                                 errors='coerce')
            band_df = band_df[band_df['exit_fill_ts'].notna()
                              & (bxt <= pd.Timestamp(edge_ceil))
                              & (bet >= pd.Timestamp(edge_start))]
            for rec in _serialize_trades(band_df):
                rec['data_source'] = ds_tag
                band_records.append(rec)
        # NO-WIPE GUARD (2026-06-16): if the recompute produced 0 band trades,
        # do NOT delete-then-insert-nothing — that would WIPE the band's
        # existing (correct) trades on a SILENT empty/failed fetch (e.g. a
        # Polygon read-timeout that returns no data instead of raising; the
        # full UAD hit exactly this on sid 303). Return -1 (same as a raised
        # failure) so the caller's insert loop handles band trades normally
        # (insert-only, no delete). The band self-heals on the next non-empty
        # append. Trade-off: a band that is LEGITIMATELY empty keeps any stale
        # rows until a non-empty recompute — rare, and far safer than a wipe.
        if not band_records:
            logger.info(
                "[EDGE-BAND] sid=%s recompute returned 0 band trades — "
                "skipping replace to avoid wiping the band (insert-only "
                "fallback; retries next append)", strategy_id)
            return -1
        from db import replace_trades_in_window_admin
        n = replace_trades_in_window_admin(
            strategy_id, user_id, band_records,
            data_source_filter=data_source_filter,
            entry_floor_iso=edge_start.isoformat(),
            entry_ceil_iso=edge_ceil.isoformat())
        logger.info(
            "[EDGE-BAND] sid=%s %s replace [%s..%s): %d trades",
            strategy_id, data_source_filter, edge_start.isoformat()[:19],
            edge_ceil.isoformat()[:19], n)
        return n
    except Exception as e:
        logger.warning(
            "[EDGE-BAND] sid=%s replace failed (append continues): %s",
            strategy_id, e)
        return -1  # sentinel: failed → caller must NOT skip band inserts

# ── Snapshot lineage guard (2026-06-11, iter 0611a) ─────────────────────
# Appends resume from persisted engine snapshots (engine_snapshot_b64 /
# _algo); FULL recompute (Update All Data) never rewrites them. So
# path-dependent indicator state that diverged in the past (e.g. UT Bot
# trail computed over pre-gap-healer polluted bars) is FOSSILIZED: every
# append inherits it, re-saves it, and the lineage never heals — stored
# backtest trades miss flips the engine's own batch data confirms
# (sid 302 case, 2026-06-11: append-created trades exited via stop while
# a fresh run + the live engine both exited via utv4_bear_flip).
#
# Fix: a snapshot may be resumed ONLY when its lineage was anchored
# (started fresh, with no snapshot resume) at/after this epoch. When the
# guard refuses, the run takes the warmup-windowed path — slower but it
# recomputes path-dependent state from bars and converges to truth — and
# the NEW snapshot it persists gets anchored `now`, so the strategy
# self-heals on its first post-deploy append, full UAD not required.
# Bump the epoch whenever a data/engine fix invalidates historical
# indicator state.
SNAPSHOT_LINEAGE_EPOCH = '2026-06-11T16:28:00+00:00'


def _snapshot_lineage_key(lane: str) -> str:
    return ('snapshot_lineage_anchor_at_algo' if lane == 'algo'
            else 'snapshot_lineage_anchor_at')


def _snapshot_lineage_ok(cfg: dict, lane: str) -> bool:
    anchor = cfg.get(_snapshot_lineage_key(lane))
    return bool(anchor) and str(anchor) >= SNAPSHOT_LINEAGE_EPOCH


def _guarded_snapshot(cfg: dict, lane: str, snapshot_key: str,
                      strategy_id) -> 'Optional[str]':
    """Return the persisted snapshot only if its lineage is anchored at/
    after SNAPSHOT_LINEAGE_EPOCH; else None (warmup-windowed fallback)."""
    snap = cfg.get(snapshot_key)
    if not snap:
        return None
    if _snapshot_lineage_ok(cfg, lane):
        return snap
    logger.info(
        "[SNAPSHOT-LINEAGE] sid=%s lane=%s: refusing snapshot resume — "
        "lineage anchor %r predates epoch %s; falling back to "
        "warmup-windowed (state re-converges, new lineage anchors fresh)",
        strategy_id, lane, cfg.get(_snapshot_lineage_key(lane)),
        SNAPSHOT_LINEAGE_EPOCH)
    return None


def _stamp_lineage_on_persist(cfg: dict, lane: str,
                              resumed_from_b64) -> None:
    """Call when persisting a NEW snapshot: a fresh (non-resumed) run
    anchors the lineage `now`; a resumed run keeps its existing anchor."""
    if resumed_from_b64 is None:
        cfg[_snapshot_lineage_key(lane)] = datetime.now(
            timezone.utc).isoformat()


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
    # `force=True` (manual user click) bypasses the alerts-gate entirely.
    # Skip the whole floor + alerts query when force=True so the user
    # path doesn't pay for cron's optimization work it doesn't need.
    # W2C optimization (2026-05-13): skipping this block alone removes
    # one full `load_trades_for_strategy` call per strategy on bulk
    # Update New Data runs.
    if force:
        recent_exits_count = -1  # treat as "alerts-gate disabled"
    else:
        # Floor is the EARLIER of:
        #   - last_recompute_until_ts (when the cron last ran), and
        #   - max(trades.exit_fill_ts) + 1 second (the actual data baseline)
        # The latter catches the case where a previous run stamped
        # last_recompute but failed to insert trades it should have
        # (engine crash, warmup misconfigured, etc.). Without this, the
        # alerts-gate would falsely think "no new trades since stamp"
        # and skip forever — even though there are exit alerts past the
        # actual data baseline waiting to be picked up. Documented bug
        # found 2026-05-07 on sid 154.
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
        #
        # W2C optimization (2026-05-13): use get_max_entry_ts_admin
        # instead of `load_trades_for_strategy` — one indexed
        # `ORDER BY entry_fill_ts DESC LIMIT 1` query instead of
        # downloading every cache_% row. Replaces a load that scales
        # O(N) on the strategy's algo history (could be 6000+ rows) with
        # an O(1) lookup. Mirrors the BT-APPEND optimization shipped
        # 2026-05-12.
        from db import get_max_entry_ts_admin
        latest_entry_iso = get_max_entry_ts_admin(
            strategy_id, user_id, data_source_filter='cache_%')
        # Secondary-TF snapshot (RORT_SECONDARY_TF_SNAPSHOT): a valid cache makes the
        # windowed path safe for coarse-gate strategies (the secondary comes
        # from the cache, not the short window), so lift the long-cycle guard.
        _algo_model_for_snap = cfg.get('algo_model') or cfg.get('backtest_model')
        use_windowed = (
            latest_entry_iso is not None
            and (not svc._has_long_cycle_secondary_tf(strat)
                 or svc._has_valid_secondary_snapshot(
                     strat, _algo_model_for_snap))
        )
        engine_path = 'windowed' if use_windowed else 'full'

        # LEF 2c port (2026-05-20): algo-lane snapshot resume. Uses
        # engine_snapshot_b64_algo (separate from backtest lane's
        # engine_snapshot_b64) since the two lanes may use different
        # models and produce different indicator-state trajectories.
        from unified_engine import compute_backtest_fingerprint
        algo_model = cfg.get('algo_model') or cfg.get('backtest_model')
        algo_snapshot_b64 = _guarded_snapshot(
            cfg, 'algo', 'engine_snapshot_b64_algo', strategy_id)
        algo_fingerprint = compute_backtest_fingerprint(strat)
        new_algo_snapshot_b64 = None

        # ── B2 edge-band replace (2026-06-15) ──────────────────────────
        # HOISTED above the windowed engine run + its early returns so it
        # heals the trailing band on EVERY append. Recompute the band with
        # full UAD-parity warmup (resume disabled) and REPLACE it.
        # Throttled on the cron; force=True (manual) always heals now.
        algo_ds_tag = f'cache_{algo_model}'
        edge_start_iso = (cutoff - timedelta(
            minutes=_APPEND_EDGE_BAND_MINUTES)).isoformat()
        _last_band_algo = cfg.get('last_edge_band_ts_algo')
        _do_band_algo = _APPEND_EDGE_BAND_ENABLED and (
            force or not _last_band_algo or (
                (now_dt - datetime.fromisoformat(_last_band_algo)).total_seconds()
                > _APPEND_EDGE_BAND_THROTTLE_SEC))
        band_replaced = 0
        if _do_band_algo:
            band_replaced = _replace_edge_band(
                svc, strat, strategy_id, user_id, cutoff,
                model_override=algo_model, ds_tag=algo_ds_tag,
                data_source_filter='cache_%', diagnostics_source='algo',
                cfg=cfg, lane='algo')
            cfg['last_edge_band_ts_algo'] = now_iso

        try:
            if use_windowed:
                since_dt = datetime.fromisoformat(latest_entry_iso)
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
                # Algo-model split (2026-05-07): cron dispatches under
                # algo_model so the algo-history lane reflects what the
                # live engine SHOULD have produced on the same data
                # (cache_locked by default). Falls back to backtest_model
                # for strategies that pre-date the algo_model field.
                all_trades_df, new_algo_snapshot_b64 = svc.get_strategy_trades_for_window(
                    strat, since_dt=since_dt, until_dt=now_dt,
                    model_override=algo_model,
                    resume_snapshot_b64=algo_snapshot_b64,
                    expected_fingerprint=algo_fingerprint,
                    expected_model_id=f"algo_{algo_model}",
                    return_snapshot=True,
                    diagnostics_source='algo',
                )
            elif (latest_entry_iso is None
                  and _os_env_skip_algo_cold_seed()):
                # 1.5 (2026-06-19): a fresh strategy's algo/cache lane starts
                # EMPTY and fills LIVE — the worker inserts each closed live
                # trade (worker.py "ALGO TRADE APPENDED"). The old cold-seed
                # ran a FULL algo-model backtest over all history (the ~17min
                # first-UND cost), which is redundant here: algo_model ==
                # backtest_model so it just duplicated the backtest lane, and
                # the pre-live portion has no live alerts to diverge against.
                # Skip it → fast first-UND; divergence still works from
                # live-start onward. RORT_SKIP_ALGO_COLD_SEED=0 restores it.
                logger.info(
                    "[1.5] strategy=%s algo lane empty → skip cold-seed "
                    "(lane fills live)", strategy_id)
                all_trades_df = pd.DataFrame()
            else:
                # Full-path: no snapshot benefit (this branch runs the
                # full strategy history; resume_snapshot would skip too
                # much). Snapshot is captured by the windowed path.
                all_trades_df = svc.get_strategy_trades(
                    strat, model_override=algo_model)
        except Exception as e:
            logger.warning(
                "[ALGO-APPEND] strategy=%s engine run (%s) failed: %s",
                strategy_id, engine_path, e)
            return {'status': 'error', 'reason': f'engine: {e}',
                    'inserted': 0}

        # Persist fresh algo-lane snapshot to config — same pattern as
        # backtest lane. Only update if engine produced a new one (the
        # full-path branch above doesn't return a snapshot).
        if (new_algo_snapshot_b64 is not None
                and new_algo_snapshot_b64 != algo_snapshot_b64):
            cfg['engine_snapshot_b64_algo'] = new_algo_snapshot_b64
            cfg['engine_snapshot_at_algo'] = now_iso
            _stamp_lineage_on_persist(cfg, 'algo', algo_snapshot_b64)

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

        # True-append filter: only commit trades strictly newer than the
        # anchor entry_ts. The engine windowed query started at since_dt
        # = latest_entry_iso, so trades AT that timestamp would collide
        # on the (strategy_id, entry_fill_ts, exit_fill_ts) unique index
        # — insert_trade returns None on collision but we filter to
        # avoid wasted round-trips.
        #
        # On cold start (latest_entry_iso=None), all trades are new.
        #
        # W2C optimization (2026-05-13): replaces the O(N) "load every
        # cache_% row, build a (entry_ts, exit_ts) set, filter against
        # it" pattern with an O(1) timestamp comparison. The
        # `get_max_entry_ts_admin` call above already supplies the
        # anchor — no second full load needed. Mirrors BT-APPEND.
        # `load_trades_for_strategy` is still needed for KPI recompute
        # below (full row set required to compute new KPIs accurately).
        from trades_store import insert_trade, load_trades_for_strategy
        new_records = []
        for _, row in closed.iterrows():
            ek_raw = row.get('entry_fill_ts')
            xk_raw = row.get('exit_fill_ts')
            if pd.isna(ek_raw) or pd.isna(xk_raw):
                continue
            ek = ek_raw.isoformat() if hasattr(ek_raw, 'isoformat') else str(ek_raw)
            if latest_entry_iso and ek <= latest_entry_iso:
                continue  # boundary or older — already in DB
            if _do_band_algo and band_replaced >= 0 and ek >= edge_start_iso:
                continue  # band owned by the edge-band replace (it succeeded)
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

        # (Edge-band replace already ran, hoisted above — its rows are in
        # the DB. band_replaced carries its count for the Hi-Fi gate below
        # so the fresh band rows get L-type refinement.)

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
        if ((inserted > 0 or band_replaced > 0)
                and algo_model_for_hifi in HIFI_BACKTEST_MODELS):
            try:
                from api.routers.strategies import run_hifi_pass2
                # Phase D (2026-05-14): ALGO-APPEND writes cache rows,
                # so scope Hi-Fi to the algo lane.
                # incremental=True (2026-06-04 fix): the BT lane already
                # had this since 2026-05-27 but the algo lane was missed.
                # Without it, every cron/Update-New-Data cycle walks every
                # cache_% trade in the strategy's history (1000+ rows for
                # active canaries), dominating the 180s+ per-strategy
                # runtime even when only a few new trades exist.
                hifi_summary = run_hifi_pass2(
                    strategy_id,
                    data_source_filter='cache_%',
                    incremental=True,
                    user={'id': user_id},
                )
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
                    # KPI recompute — Priority 1 perf fix (2026-05-21).
                    # Was: load_trades_for_strategy (full data JSONB per
                    # row). Now: load_trades_kpi_fields_admin projects to
                    # the 5 fields calculate_kpis uses (~15-20x lighter).
                    # Same scope (no data_source filter — all lanes, as
                    # the legacy call did), KPIs byte-identical.
                    # Supersedes the 2026-05-20 algo-lane OOM-guard.
                    from db import load_trades_kpi_fields_admin
                    all_db_trades = load_trades_kpi_fields_admin(
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
                _div = _forward_divider_dt(strat)
                if _div is not None:
                    bt_portion, _ = svc.split_trades_at_boundary(
                        kpis_df, _div)
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

    # BUG FIX 2026-05-20: get_strategy_by_id_admin returns a FLATTENED
    # strat dict, so strat.get('config') is None even when the JSONB
    # column has 20+ fields. Reading it via the flattened shape and
    # passing back through _stamp_config emits a near-empty dict that
    # the shrink-guard correctly refuses — silently dropping the
    # snapshot writes (and last_recompute_until_ts stamps) for every
    # backtest-lane call. The algo lane already does this raw read
    # (line 691); mirror it here. Surfaced by the LEF 2c port on
    # 2026-05-20 19:04 (sid 192 snapshot didn't persist).
    from db import get_admin_client as _get_admin_client_bt
    _raw_client_bt = _get_admin_client_bt()
    _raw_resp_bt = _raw_client_bt.table('strategies').select('config') \
        .eq('id', strategy_id).eq('user_id', user_id).single().execute()
    cfg = dict(_raw_resp_bt.data.get('config') or {}) if _raw_resp_bt.data else {}
    # System default is `rest_hifi` — must match data_worker_engine.classify_strategy
    # so saved snapshot model_id agrees with the streaming engine's expectation.
    bt_model = (cfg.get('backtest_model')
                or strat.get('backtest_model')
                or 'rest_hifi')
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

    # ── B2 edge-band replace (2026-06-15) ──────────────────────────────
    # HOISTED above the windowed-append logic so it heals the trailing
    # band on EVERY append — including the all-in-lag / all-already-stored
    # early-return paths where the windowed run produces nothing new but
    # the band still carries under-generated fossils. Recompute the band
    # with full UAD-parity warmup (resume disabled) and REPLACE it.
    # Throttled on the cron (cost ~12s/strategy); force=True heals now.
    edge_start_iso = (cutoff - timedelta(
        minutes=_APPEND_EDGE_BAND_MINUTES)).isoformat()
    _last_band_bt = cfg.get('last_edge_band_ts_bt')
    _do_band_bt = _APPEND_EDGE_BAND_ENABLED and (
        force or not _last_band_bt or (
            (now_dt - datetime.fromisoformat(_last_band_bt)).total_seconds()
            > _APPEND_EDGE_BAND_THROTTLE_SEC))
    bt_band_replaced = 0
    if _do_band_bt:
        bt_band_replaced = _replace_edge_band(
            svc, strat, strategy_id, user_id, cutoff,
            model_override=bt_model, ds_tag=ds_tag,
            data_source_filter='backtest_%',
            diagnostics_source='backtest', cfg=cfg, lane='bt')
        cfg['last_edge_band_ts_bt'] = now_iso

    # LEF 2c port (2026-05-20): snapshot resume on the backtest lane.
    # Read the stored snapshot from config (one-per-lane scheme:
    # engine_snapshot_b64 = backtest lane, engine_snapshot_b64_algo =
    # algo lane). Fingerprint + model_id are validated in the deserialize
    # path — any mismatch silently falls back to warmup-windowed.
    from unified_engine import compute_backtest_fingerprint
    bt_snapshot_b64 = _guarded_snapshot(
        cfg, 'bt', 'engine_snapshot_b64', strategy_id)
    bt_fingerprint = compute_backtest_fingerprint(strat)

    try:
        try:
            import os as _os_bt
            all_trades_df, new_bt_snapshot_b64 = svc.get_strategy_trades_for_window(
                strat, since_dt=since_dt, until_dt=now_dt,
                model_override=bt_model,
                resume_snapshot_b64=bt_snapshot_b64,
                expected_fingerprint=bt_fingerprint,
                expected_model_id=f"backtest_{bt_model}",
                return_snapshot=True,
                diagnostics_source='backtest',
                # 2026-06-23 fix: inherit the boundary-open position on the
                # BACKTEST lane so trades spanning an append boundary aren't
                # dropped (root cause of ~50% lane under-production). Backtest
                # lane ONLY — live/algo keep always-flat (§8.2). Kill-switch.
                inherit_position=(_os_bt.getenv(
                    'RORT_RESUME_INHERIT_POSITION', '0') == '1'),
            )
        except Exception as e:
            logger.warning(
                "[BT-APPEND] strategy=%s engine run failed: %s",
                strategy_id, e)
            return {'status': 'error', 'reason': f'engine: {e}',
                    'inserted': 0}

        # LEF 2c port: persist the fresh snapshot into cfg so all
        # subsequent paths (early-return + main-path) write it via
        # `_stamp_config`. Only update if engine actually produced a new
        # one (engine miss → keep prior snapshot to avoid invalidation
        # cascade).
        if new_bt_snapshot_b64 is not None and new_bt_snapshot_b64 != bt_snapshot_b64:
            cfg['engine_snapshot_b64'] = new_bt_snapshot_b64
            cfg['engine_snapshot_at'] = now_iso
            _stamp_lineage_on_persist(cfg, 'bt', bt_snapshot_b64)

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
            if _do_band_bt and bt_band_replaced >= 0 and ek >= edge_start_iso:
                continue  # band owned by the edge-band replace (it succeeded)
            new_records.append(row)

        if not new_records:
            # No new trades to append. The trailing band was already
            # healed by the hoisted edge-band replace above (B2 fix).
            cfg['last_recompute_until_ts'] = now_iso
            _stamp_config(strategy_id, user_id, cfg)
            return {'status': 'no_new_trades', 'inserted': 0,
                    'band_replaced': bt_band_replaced,
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

        # (Edge-band replace already ran, hoisted above — its rows are in
        # the DB, so the KPI recompute below reflects the corrected band.)
        try:
            from trades_store import _cache_invalidate
            _cache_invalidate(strategy_id, user_id)
        except Exception:
            pass

        refreshed_at = now_iso

        # KPI recompute — Priority 1 perf fix (2026-05-21).
        # Was: load_trades_admin (select '*' — full data JSONB per row,
        # ~2KB/row) which OOM-killed the API on sid 151 (8900+ trades).
        # Now: load_trades_kpi_fields_admin projects to the 5 fields
        # calculate_kpis actually uses (~120 bytes/row, ~15-20x lighter).
        # KPIs are byte-identical to the full-load path. Supersedes the
        # 2026-05-20 OOM-guard stopgap (skip-and-mark-stale) — no longer
        # needed, the projected load is light enough to always run.
        from db import load_trades_kpi_fields_admin
        all_backtest = load_trades_kpi_fields_admin(
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
        _div = _forward_divider_dt(strat)
        if _div is not None:
            bt_portion, _ = svc.split_trades_at_boundary(merged_df, _div)
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

        # Hi-Fi pass on appended set (idempotent — skips already-resolved).
        # Phase D (2026-05-14): scope to backtest_% so the BT recompute path
        # only refines BT rows, not algo rows that happen to coexist.
        HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
        hifi_summary = None
        if bt_model in HIFI_BACKTEST_MODELS:
            try:
                from api.routers.strategies import run_hifi_pass2
                # Incremental mode (2026-05-27): only walks trades created
                # since the last pass on this scope. New trades from this
                # forward-append are by definition newer than last_hifi_pass.
                hifi_summary = run_hifi_pass2(
                    strategy_id,
                    data_source_filter='backtest_%',
                    incremental=True,
                    user={'id': user_id},
                )
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


def append_windowed_backtest_trades_for_strategy(
    strategy_id: int,
    user_id: str,
    window_start: datetime,
    window_end: datetime,
) -> Dict[str, Any]:
    """Backfill a SPECIFIC time window of the BACKTEST lane.

    Variant of `append_new_backtest_trades_for_strategy` that takes an
    EXPLICIT [window_start, window_end] instead of computing the start
    from the latest existing trade. Used by the gap-detection cron and
    the manual "Window Backfill" UI to fill gaps in the BT lane that
    Update New Data won't touch (because it uses `max(entry_fill_ts)`
    and so skips over gaps that have trades on both sides).

    UAD-PARITY CONTRACT (2026-06-05): the trades emitted for [window_start,
    window_end] must match what `recompute_and_persist_stored_trades`
    (UAD) would emit for the same window. Two-tier strategy:

      Tier 1 — snapshot resume (preferred): if the strategy's stored
        backtest snapshot has `last_bar_ts <= window_start`, resume
        from it. Indicator + position state are bit-identical to the
        last UAD, so engine output IS UAD-parity by construction. Zero
        warmup-load cost. Common in MANUAL streaming mode (snapshots
        get stale as time passes).

      Tier 2 — UAD-equivalent cold warmup (fallback): when snapshot is
        missing / invalid / too recent (last_bar_ts > window_start),
        load the same warmup UAD would use for THIS strategy's natural
        visible window. That's `WARMUP_MULTIPLIER × visible_days` per
        `compute_warmup_days`. Solve back to `warmup_bars` and pass to
        `get_strategy_trades_for_window` so the engine loads
        [window_start - uad_warmup_days, window_end] instead of the
        100-bar default. Slower than Tier 1 but still UAD-parity.

    Pattern (post-fix):
      1. Load strategy + raw config
      2. Try Tier 1 snapshot resume; fall back to Tier 2
      3. Filter closed trades to ones within the window
      4. Dedup against existing backtest_* trades in the window
         (entry_fill_ts, exit_fill_ts) unique
      5. Insert new rows; per-row insert_trade_admin handles
         unique-constraint collisions gracefully
      6. Persist any fresh snapshot the engine emitted (extends the
         resume chain forward for future window-fills)
      7. Hi-Fi pass on the inserted set (if model in HIFI set)

    Does NOT update KPIs or last_recompute_until_ts — this is a
    targeted backfill, not a forward-progressing update. Snapshot
    persistence IS updated when the engine emits a newer one, since
    that's necessary for the resume chain to remain useful.

    Args:
      strategy_id: int
      user_id: str
      window_start: tz-aware datetime — earliest bar to recompute
      window_end: tz-aware datetime — latest bar to recompute (exclusive)

    Returns:
        {'status': 'window_filled' | 'no_new_trades' | 'error',
         'inserted': int, 'window_start': iso, 'window_end': iso,
         'elapsed_s': float, ...}
    """
    import time as _time
    import pandas as pd
    import services as svc
    from db import (
        get_strategy_by_id_admin,
        set_admin_user_context,
        get_current_user_id,
        clear_current_user,
        get_admin_client,
    )

    t0 = _time.time()
    if window_start.tzinfo is None:
        window_start = window_start.replace(tzinfo=timezone.utc)
    if window_end.tzinfo is None:
        window_end = window_end.replace(tzinfo=timezone.utc)
    if window_end <= window_start:
        return {'status': 'error',
                'reason': 'window_end must be after window_start',
                'inserted': 0}

    strat = get_strategy_by_id_admin(strategy_id, user_id)
    if strat is None:
        return {'status': 'error',
                'reason': f'strategy {strategy_id} not found',
                'inserted': 0}

    # Read raw config JSONB (avoids the flattened-shape gotcha that
    # bit append_new_backtest before the 2026-05-20 fix).
    _raw_client = get_admin_client()
    _raw_resp = (_raw_client.table('strategies').select('config')
                 .eq('id', strategy_id).eq('user_id', user_id).single().execute())
    cfg = dict(_raw_resp.data.get('config') or {}) if _raw_resp.data else {}
    bt_model = (cfg.get('backtest_model')
                or strat.get('backtest_model')
                or 'rest_hifi')

    _prev_user = get_current_user_id()
    _need_ctx_swap = _prev_user != user_id
    if _need_ctx_swap:
        set_admin_user_context(user_id)

    try:
        # ── UAD-parity warmup resolution ────────────────────────────────
        # Tier 1: try snapshot resume. If the strategy's stored backtest
        # snapshot has last_bar_ts <= window_start, the engine can resume
        # from there with bit-identical state — zero warmup load.
        # Tier 2 (fallback): cold start, but load UAD's full warmup so
        # indicator state at window_start matches what UAD would have.
        path_used = None  # for diagnostics in the return value
        resume_b64 = None
        resume_fingerprint = None
        resume_model_id = None

        bt_snapshot_b64 = _guarded_snapshot(
            cfg, 'bt', 'engine_snapshot_b64', strategy_id)
        snapshot_last_ts = None
        if bt_snapshot_b64:
            try:
                from unified_engine import (
                    compute_backtest_fingerprint,
                    deserialize_backtest_snapshot,
                )
                bt_fingerprint = compute_backtest_fingerprint(strat)
                envelope = deserialize_backtest_snapshot(
                    bt_snapshot_b64,
                    expected_fingerprint=bt_fingerprint,
                    expected_model_id=f"backtest_{bt_model}",
                )
                if envelope is not None and envelope.get('last_bar_ts'):
                    last_ts_raw = envelope.get('last_bar_ts')
                    # `last_bar_ts` is serialized as a string per
                    # unified_engine.serialize_backtest_snapshot.
                    if isinstance(last_ts_raw, str):
                        last_ts = datetime.fromisoformat(
                            last_ts_raw.replace('Z', '+00:00'))
                    elif hasattr(last_ts_raw, 'to_pydatetime'):
                        last_ts = last_ts_raw.to_pydatetime()
                    else:
                        last_ts = last_ts_raw
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    if last_ts <= window_start:
                        snapshot_last_ts = last_ts
                        resume_b64 = bt_snapshot_b64
                        resume_fingerprint = bt_fingerprint
                        resume_model_id = f"backtest_{bt_model}"
            except Exception as e:
                logger.warning(
                    "[BT-WINDOW] sid=%s snapshot deserialize failed, "
                    "falling back to cold warmup: %s", strategy_id, e)

        # Compute cold-warmup `warmup_bars` matching UAD's natural
        # warmup_days for this strategy (shared helper — single source of
        # truth with the B2 edge-band replace). Only USED on the fallback
        # path (snapshot resume ignores warmup_bars per services.py:627).
        uad_warmup_bars = _uad_warmup_bars(strat)

        try:
            if resume_b64:
                # Tier 1: snapshot resume — bit-identical to UAD state
                # at snapshot.last_bar_ts. Engine processes
                # [last_bar_ts, window_end].
                path_used = 'snapshot_resume'
                all_trades_df, new_bt_snapshot_b64 = svc.get_strategy_trades_for_window(
                    strat,
                    since_dt=snapshot_last_ts,
                    until_dt=window_end,
                    model_override=bt_model,
                    resume_snapshot_b64=resume_b64,
                    expected_fingerprint=resume_fingerprint,
                    expected_model_id=resume_model_id,
                    return_snapshot=True,
                    diagnostics_source='backtest',
                )
            else:
                # Tier 2: cold warmup, but sized to UAD's warmup_days so
                # indicator convergence at window_start matches UAD.
                path_used = 'cold_warmup_uad_matched'
                logger.info(
                    "[BT-WINDOW] sid=%s cold path: uad_warmup_days=%.1f "
                    "binding_bpd=%s → warmup_bars=%d",
                    strategy_id, uad_warmup_days, binding_bpd, uad_warmup_bars)
                all_trades_df, new_bt_snapshot_b64 = svc.get_strategy_trades_for_window(
                    strat,
                    since_dt=window_start,
                    until_dt=window_end,
                    warmup_bars=uad_warmup_bars,
                    model_override=bt_model,
                    return_snapshot=True,
                    diagnostics_source='backtest',
                )
        except Exception as e:
            logger.warning(
                "[BT-WINDOW] strategy=%s engine run failed (%s): %s",
                strategy_id, path_used, e)
            return {'status': 'error',
                    'reason': f'engine: {e}',
                    'inserted': 0,
                    'path': path_used}

        if all_trades_df is None or len(all_trades_df) == 0:
            return {'status': 'no_new_trades', 'inserted': 0,
                    'window_start': window_start.isoformat(),
                    'window_end': window_end.isoformat(),
                    'path': path_used,
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Only closed trades fully within the window
        if 'exit_fill_ts' not in all_trades_df.columns:
            return {'status': 'error',
                    'reason': 'engine output missing exit_fill_ts column',
                    'inserted': 0,
                    'path': path_used}
        closed_mask = all_trades_df['exit_fill_ts'].notna()
        closed = all_trades_df[closed_mask].copy()
        if len(closed) == 0:
            return {'status': 'no_new_trades', 'inserted': 0,
                    'reason': 'no_closed_trades_in_window',
                    'window_start': window_start.isoformat(),
                    'window_end': window_end.isoformat(),
                    'path': path_used,
                    'elapsed_s': round(_time.time() - t0, 2)}
        entry_ts_col = pd.to_datetime(closed['entry_fill_ts'], utc=True,
                                     errors='coerce')
        in_window = (entry_ts_col >= pd.Timestamp(window_start)) & \
                    (entry_ts_col < pd.Timestamp(window_end))
        closed = closed[in_window]
        if len(closed) == 0:
            return {'status': 'no_new_trades', 'inserted': 0,
                    'reason': 'all_outside_window',
                    'window_start': window_start.isoformat(),
                    'window_end': window_end.isoformat(),
                    'path': path_used,
                    'elapsed_s': round(_time.time() - t0, 2)}

        # Dedup against existing backtest_* trades in this window
        existing_q = (_raw_client.table('trades')
                      .select('entry_fill_ts,exit_fill_ts')
                      .eq('strategy_id', strategy_id)
                      .like('data_source', 'backtest_%')
                      .gte('entry_fill_ts', window_start.isoformat())
                      .lt('entry_fill_ts', window_end.isoformat())
                      .execute())
        existing_keys: set = set()
        for r in (existing_q.data or []):
            existing_keys.add((r.get('entry_fill_ts'), r.get('exit_fill_ts')))

        from db import insert_trade_admin
        ds_tag = f'backtest_{bt_model}'
        inserted = 0
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

        if not new_records:
            return {'status': 'no_new_trades', 'inserted': 0,
                    'reason': 'all_already_in_db',
                    'window_start': window_start.isoformat(),
                    'window_end': window_end.isoformat(),
                    'path': path_used,
                    'elapsed_s': round(_time.time() - t0, 2)}

        new_df = pd.DataFrame(new_records)
        new_serialized = _serialize_trades(new_df)
        for rec in new_serialized:
            rec_copy = dict(rec)
            rec_copy['data_source'] = ds_tag
            try:
                row_saved = insert_trade_admin(strategy_id, user_id, rec_copy)
                if row_saved is not None:
                    inserted += 1
            except Exception as e:
                logger.warning(
                    "[BT-WINDOW] sid=%s insert_trade failed: %s",
                    strategy_id, e)

        # Persist the fresh snapshot the engine emitted, when we have
        # a usable one. Only update if the engine produced a newer
        # snapshot — an engine miss / serialization failure leaves
        # `new_bt_snapshot_b64=None`, in which case keep the prior
        # snapshot to avoid invalidation cascade. The snapshot's
        # last_bar_ts will now be at-or-after this window's end, so
        # FUTURE window-fills inside this strategy's recent history can
        # take the Tier 1 resume path instead of paying cold warmup.
        if new_bt_snapshot_b64 is not None and \
                new_bt_snapshot_b64 != bt_snapshot_b64:
            try:
                cfg['engine_snapshot_b64'] = new_bt_snapshot_b64
                cfg['engine_snapshot_at'] = datetime.now(
                    timezone.utc).isoformat()
                _stamp_lineage_on_persist(cfg, 'bt', bt_snapshot_b64)
                _stamp_config(strategy_id, user_id, cfg)
            except Exception as e:
                logger.warning(
                    "[BT-WINDOW] sid=%s snapshot persist failed: %s",
                    strategy_id, e)

        # Invalidate cache so the next get_strategy_trades_for_window
        # call (e.g., a chart fetch) sees the inserts.
        try:
            from trades_store import _cache_invalidate
            _cache_invalidate(strategy_id, user_id)
        except Exception:
            pass

        # Hi-Fi pass on the just-inserted backtest_% rows
        HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
        hifi_summary = None
        if bt_model in HIFI_BACKTEST_MODELS and inserted > 0:
            try:
                from api.routers.strategies import run_hifi_pass2
                hifi_summary = run_hifi_pass2(
                    strategy_id,
                    data_source_filter='backtest_%',
                    incremental=True,
                    user={'id': user_id},
                )
                logger.info(
                    "[BT-WINDOW] strategy=%s window=[%s,%s] Hi-Fi: refined "
                    "entries=%s exits=%s persisted=%s",
                    strategy_id, window_start.isoformat()[:19],
                    window_end.isoformat()[:19],
                    hifi_summary.get('entries_refined', 0),
                    hifi_summary.get('exits_refined', 0),
                    hifi_summary.get('persisted', 0))
            except Exception as e:
                logger.warning(
                    "[BT-WINDOW] strategy=%s Hi-Fi pass failed: %s",
                    strategy_id, e)

        logger.info(
            "[BT-WINDOW] strategy=%s window=[%s,%s] inserted=%d path=%s",
            strategy_id, window_start.isoformat()[:19],
            window_end.isoformat()[:19], inserted, path_used)

        result = {
            'status': 'window_filled',
            'inserted': inserted,
            'window_start': window_start.isoformat(),
            'window_end': window_end.isoformat(),
            'path': path_used,
            'warmup': {
                # Diagnostic — surfaces which warmup the engine ran with
                # so the operator can sanity-check UAD parity.
                'uad_visible_days': round(uad_visible_days, 2),
                'uad_warmup_days': round(uad_warmup_days, 2),
                'binding_bpd': binding_bpd,
                'uad_warmup_bars': uad_warmup_bars,
                'snapshot_last_ts': (
                    snapshot_last_ts.isoformat()
                    if snapshot_last_ts is not None else None),
            },
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

        # Hi-Fi pass on the trades table (idempotent).
        # Phase D (2026-05-14): scope to cache_% so the algo recompute path
        # only refines algo rows, not backtest rows that happen to coexist.
        HIFI_BACKTEST_MODELS = {'rest_hifi', 'cache_locked', 'cache_corrected'}
        hifi_summary = None
        if inserted > 0 and algo_model in HIFI_BACKTEST_MODELS:
            try:
                from api.routers.strategies import run_hifi_pass2
                # Incremental mode (2026-05-27): same reasoning as
                # backtest-append above — new algo trades are by definition
                # newer than the last pass.
                hifi_summary = run_hifi_pass2(
                    strategy_id,
                    data_source_filter='cache_%',
                    incremental=True,
                    user={'id': user_id},
                )
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
