"""Async on-demand UAD jobs — background executor.

Mirrors `mass_builder.py`'s daemon-thread + progress-polling pattern.
The user clicks Update All Data → the API endpoint creates a row in
the `update_jobs` table, returns the job_id immediately, and spawns
a daemon thread to run `recompute_and_persist_stored_trades` +
`recompute_and_persist_algo_trades` (mode='all') OR the corresponding
append functions (mode='new') for each strategy in the list.

Frontend polls `/api/update-jobs/progress/{job_id}` to render status.

Survives browser close. Survives one worker restart via the
`cleanup_orphaned_update_jobs` boot routine which marks any
'running' rows as 'orphaned' so the user can retry.

Cancellation: user POSTs /cancel/{job_id} → sets an in-memory flag;
the worker thread checks it between strategies and exits cleanly.
"""
from __future__ import annotations

import logging
import threading
import time as _time
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory registry: maps job_id → {'cancelled': bool, ...}.
# Mirrors mass_builder._active_searches.
_active_jobs: dict = {}
_jobs_lock = threading.Lock()


def _set_active(job_id, **kwargs) -> None:
    with _jobs_lock:
        entry = _active_jobs.setdefault(job_id, {'cancelled': False})
        entry.update(kwargs)


def _get_active(job_id) -> dict:
    with _jobs_lock:
        return dict(_active_jobs.get(job_id, {}))


def _is_cancelled(job_id) -> bool:
    return _get_active(job_id).get('cancelled', False)


def cancel_update_job(job_id) -> bool:
    """Signal cancellation. Worker exits cleanly between strategies.

    Returns True if there's an in-process job to cancel, False if not
    (caller should still update DB status for the orphan case).
    """
    with _jobs_lock:
        if job_id in _active_jobs:
            _active_jobs[job_id]['cancelled'] = True
            return True
        # Try str key too (DB returns int, in-memory might be str)
        sid = str(job_id)
        if sid in _active_jobs:
            _active_jobs[sid]['cancelled'] = True
            return True
    return False


def is_job_running(job_id) -> bool:
    """Whether this job has an active in-process worker thread."""
    with _jobs_lock:
        return job_id in _active_jobs or str(job_id) in _active_jobs


def start_update_job_async(
    job_id: int,
    mode: str,
    strategy_ids: list[int],
    user_id: str,
    window_start: str | None = None,
    window_end: str | None = None,
) -> None:
    """Spawn a daemon thread to run UAD for the strategy_ids list.

    mode='all':    recompute_and_persist_stored_trades (full backtest)
                   + recompute_and_persist_algo_trades (full algo)
    mode='new':    append_new_backtest_trades_for_strategy
                   + append_new_trades_for_strategy
    mode='window': append_windowed_backtest_trades_for_strategy
                   (BT lane only; requires window_start + window_end)

    Per-strategy progress is written to the DB row's config_data.progress
    field every step. Cancellation is checked between strategies.

    window_* params are ISO timestamp strings; only used when mode='window'.
    """
    def _worker():
        # Capture admin user context for the lifetime of this thread so
        # downstream Supabase calls work even after the user's JWT
        # expires (3-5 min UAD per strategy can exceed JWT TTL). Same
        # pattern as mass_builder.start_mass_search_async.
        try:
            from db import (
                set_admin_user_context, save_update_job, update_update_job,
            )
        except Exception as e:
            logger.error("[update_job %s] failed import: %s", job_id, e)
            return

        set_admin_user_context(user_id)
        logger.info(
            "[update_job %s] starting mode=%s strategy_count=%d",
            job_id, mode, len(strategy_ids))

        try:
            from api.services.forward_test_service import (
                recompute_and_persist_stored_trades,
                recompute_and_persist_algo_trades,
                append_new_backtest_trades_for_strategy,
                append_new_trades_for_strategy,
                append_windowed_backtest_trades_for_strategy,
            )
        except Exception as e:
            logger.error("[update_job %s] forward_test_service import failed: %s",
                         job_id, e)
            update_update_job(job_id, {
                'status': 'failed',
                'error': f"import failed: {e}",
            })
            return

        per_strategy: dict[str, dict] = {}
        total = len(strategy_ids)
        succeeded = 0
        failed = 0
        total_trades_inserted = 0
        t_job0 = _time.monotonic()

        update_update_job(job_id, {
            'status': 'running',
            'progress': {
                'current_step': 0,
                'total_steps': total,
                'current_sid': None,
                'current_phase': 'starting',
                'per_strategy': {},
            },
        })

        for i, sid in enumerate(strategy_ids):
            if _is_cancelled(job_id):
                logger.info("[update_job %s] cancelled at step %d/%d",
                            job_id, i, total)
                break

            t_strat0 = _time.monotonic()
            update_update_job(job_id, {
                'progress': {
                    'current_step': i + 1,
                    'total_steps': total,
                    'current_sid': sid,
                    'current_phase': f'mode={mode}',
                    'per_strategy': per_strategy,
                },
            })

            try:
                algo = None  # window mode only touches BT lane
                if mode == 'all':
                    bt = recompute_and_persist_stored_trades(sid, user_id)
                    algo = recompute_and_persist_algo_trades(sid, user_id)
                elif mode == 'new':
                    bt = append_new_backtest_trades_for_strategy(
                        sid, user_id, force=True)
                    algo = append_new_trades_for_strategy(
                        sid, user_id, force=True)
                elif mode == 'window':
                    # Window backfill — BT lane only. Caller must pass
                    # ISO timestamps; convert here.
                    if not window_start or not window_end:
                        raise ValueError(
                            "mode='window' requires window_start and window_end")
                    from datetime import datetime as _dt
                    ws = _dt.fromisoformat(window_start.replace('Z', '+00:00'))
                    we = _dt.fromisoformat(window_end.replace('Z', '+00:00'))
                    bt = append_windowed_backtest_trades_for_strategy(
                        sid, user_id, ws, we)
                else:
                    raise ValueError(f"unknown mode: {mode!r}")

                # Try to extract inserted trade counts for the summary.
                bt_count = 0
                if isinstance(bt, dict):
                    bt_count = (bt.get('inserted') or bt.get('trades') or 0)
                algo_count = 0
                if isinstance(algo, dict):
                    algo_count = (algo.get('inserted') or algo.get('trades') or 0)
                total_trades_inserted += int(bt_count) + int(algo_count)

                per_strategy[str(sid)] = {
                    'status': 'ok',
                    'elapsed_s': round(_time.monotonic() - t_strat0, 2),
                    'backtest': {
                        'status': bt.get('status') if isinstance(bt, dict) else 'unknown',
                        'count': int(bt_count),
                    },
                    'algo': {
                        'status': algo.get('status') if isinstance(algo, dict) else (
                            'skipped' if mode == 'window' else 'unknown'),
                        'count': int(algo_count),
                    },
                }
                succeeded += 1
            except Exception as e:
                logger.warning(
                    "[update_job %s] sid=%s failed: %s",
                    job_id, sid, e, exc_info=True)
                per_strategy[str(sid)] = {
                    'status': 'error',
                    'elapsed_s': round(_time.monotonic() - t_strat0, 2),
                    'error': str(e),
                }
                failed += 1

        # Finalize
        cancelled = _is_cancelled(job_id)
        final_status = 'cancelled' if cancelled else (
            'failed' if failed > 0 and succeeded == 0 else 'completed'
        )
        elapsed = round(_time.monotonic() - t_job0, 2)
        update_update_job(job_id, {
            'status': final_status,
            'progress': {
                'current_step': total,
                'total_steps': total,
                'current_sid': None,
                'current_phase': 'done',
                'per_strategy': per_strategy,
            },
            'summary': {
                'succeeded': succeeded,
                'failed': failed,
                'cancelled': cancelled,
                'total_trades_inserted': total_trades_inserted,
                'total_elapsed_s': elapsed,
            },
        })
        logger.info(
            "[update_job %s] done status=%s ok=%d fail=%d elapsed=%.1fs",
            job_id, final_status, succeeded, failed, elapsed)

        # Spawn delayed cleanup so polling clients get one last fresh read.
        def _cleanup():
            _time.sleep(60)
            with _jobs_lock:
                _active_jobs.pop(job_id, None)
                _active_jobs.pop(str(job_id), None)
        threading.Thread(target=_cleanup, daemon=True).start()

    _set_active(job_id, cancelled=False)
    thread = threading.Thread(
        target=_worker, daemon=True, name=f"update_job_{job_id}")
    thread.start()


def cleanup_orphaned_update_jobs() -> dict:
    """Mark orphaned 'running' rows as 'orphaned'.

    Called once on api boot. Mirrors `mass_builder.cleanup_orphaned_mass_searches`.
    Any update_jobs row with status='running' is from a worker that died
    before completion — mark it 'orphaned' so the UI surfaces it and the
    user can retry. No auto-resume: UAD is bounded enough that re-running
    from scratch is acceptable (3-5 min per strategy).

    Returns {'orphaned': int, 'ids': [...]}.
    """
    try:
        from db import load_orphaned_update_jobs, mark_update_job_orphaned
    except Exception as e:
        logger.warning("cleanup_orphaned_update_jobs import failed: %s", e)
        return {'orphaned': 0, 'ids': []}

    rows = load_orphaned_update_jobs()
    if not rows:
        return {'orphaned': 0, 'ids': []}

    orphaned_ids = []
    for row in rows:
        jid = row.get('id')
        if jid is None:
            continue
        # If we still have it in _active_jobs (rare — same-process reboot
        # without container restart), skip cleanup. Almost always the
        # container restart means _active_jobs is empty.
        if is_job_running(jid):
            continue
        try:
            mark_update_job_orphaned(jid, error_msg="worker restarted mid-job")
            orphaned_ids.append(jid)
        except Exception as e:
            logger.warning("[update_jobs] failed to orphan %s: %s", jid, e)

    if orphaned_ids:
        logger.info(
            "[update_jobs] cleaned up %d orphan job(s): %s",
            len(orphaned_ids), orphaned_ids)
    return {'orphaned': len(orphaned_ids), 'ids': orphaned_ids}
