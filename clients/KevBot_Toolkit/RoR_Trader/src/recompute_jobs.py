"""Background-job orchestration for algo-history recompute operations.

Mirrors the in-memory pattern used by `mass_builder.py` (`_active_searches`
dict + `_search_lock`), scoped down for the simpler recompute case.

Two job types:

* `append_recent` — runs `forward_test_service.append_new_trades_for_strategy`
  per strategy. Uses the windowed engine helper (Phase 1) on strategies
  with existing trades; falls back to full backtest on cold-start. Fast
  for already-stamped strategies, slow on cold-start.

* `full_recompute` — runs `forward_test_service.recompute_and_persist_stored_trades`
  per strategy. Always full DELETE+INSERT backtest. Used by the
  "Update All Data" button.

Why background: Railway's Cloudflare edge times out HTTP requests at
~60-300s. A cold-start engine replay on a 10Sec strategy can take
5+ minutes, exceeding the timeout while the work continues server-side
unobserved by the user. Background jobs return a job_id immediately;
the frontend polls `/api/jobs/{id}` for progress.

State persistence: in-memory only for v1 (DECIDED 2026-05-06). Worker
restart loses in-flight jobs — user re-triggers manually. Suitable
because recompute jobs are short-lived (~5s typical).
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_active_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Per-strategy semaphore to prevent concurrent jobs racing on the same
# strategy_id. Cron + manual trigger could otherwise both engine-replay
# the same strategy at once (set-diff prevents duplicates but kpis would
# flap and Hi-Fi pass would race against the trade INSERTs).
_strategy_locks: Dict[int, threading.Lock] = {}
_strategy_locks_lock = threading.Lock()

# Cap retained completed jobs to avoid unbounded memory growth. FIFO eviction.
_MAX_RETAINED_JOBS = 200


def _get_strategy_lock(strategy_id: int) -> threading.Lock:
    """Get or create the per-strategy lock. Thread-safe."""
    with _strategy_locks_lock:
        if strategy_id not in _strategy_locks:
            _strategy_locks[strategy_id] = threading.Lock()
        return _strategy_locks[strategy_id]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evict_old_jobs() -> None:
    """If we exceed the retention cap, drop oldest completed/failed/cancelled jobs.

    Held by callers under _jobs_lock. Doesn't touch running/queued jobs
    (those would orphan the worker thread).
    """
    if len(_active_jobs) <= _MAX_RETAINED_JOBS:
        return
    finalized = [
        (jid, j) for jid, j in _active_jobs.items()
        if j.get('status') in ('completed', 'failed', 'cancelled')
    ]
    finalized.sort(key=lambda x: x[1].get('created_at', ''))
    overflow = len(_active_jobs) - _MAX_RETAINED_JOBS
    for jid, _ in finalized[:overflow]:
        _active_jobs.pop(jid, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit_recompute_job(
    user_id: str,
    strategy_ids: List[int],
    job_type: str,
) -> str:
    """Queue a recompute job and return its id immediately.

    Args:
        user_id: Owning user (for scoping queries via list_jobs).
        strategy_ids: Non-empty list of strategy IDs to process.
        job_type: 'append_recent' or 'full_recompute'.

    Returns:
        job_id (UUID string) — caller polls `get_job_status(job_id)`.

    Raises:
        ValueError on invalid input. Jobs are NOT persisted to DB; if the
        worker restarts the job is lost.
    """
    if job_type not in ('append_recent', 'full_recompute'):
        raise ValueError(f"Unknown job_type: {job_type!r}")
    if not strategy_ids:
        raise ValueError("strategy_ids must be non-empty")
    if not user_id:
        raise ValueError("user_id required")

    job_id = str(uuid.uuid4())
    now = _now_iso()
    job: Dict[str, Any] = {
        'id': job_id,
        'user_id': user_id,
        'strategy_ids': list(strategy_ids),
        'job_type': job_type,
        'status': 'queued',
        'created_at': now,
        'started_at': None,
        'completed_at': None,
        'cancelled': False,
        'current_strategy_idx': 0,
        'current_strategy_id': None,
        'current_strategy_name': None,
        'progress_label': 'queued',
        'per_strategy_results': [],
        'summary': {
            'total_strategies': len(strategy_ids),
            'total_inserted': 0,
            'total_skipped': 0,
            'total_failed': 0,
            'total_appended': 0,
        },
        'error': None,
    }
    with _jobs_lock:
        _active_jobs[job_id] = job
        _evict_old_jobs()

    threading.Thread(
        target=_run_job_worker,
        args=(job_id,),
        name=f"recompute-job-{job_id[:8]}",
        daemon=True,
    ).start()

    logger.info(
        "[RECOMPUTE-JOB] queued job=%s type=%s strategies=%d user=%s",
        job_id[:8], job_type, len(strategy_ids), user_id[:8])
    return job_id


def get_job_status(job_id: str) -> Optional[dict]:
    """Snapshot of current job state. Returns None if job_id unknown."""
    with _jobs_lock:
        job = _active_jobs.get(job_id)
        return dict(job) if job is not None else None


def cancel_job(job_id: str) -> bool:
    """Mark job for cancellation. Returns True if found and was running/queued.

    Worker thread checks the cancelled flag between strategies and bails
    cleanly. Already-running engine work for the current strategy
    completes before cancellation takes effect (we don't kill threads
    mid-engine — too risky for DB consistency).
    """
    with _jobs_lock:
        job = _active_jobs.get(job_id)
        if job is None:
            return False
        if job['status'] not in ('queued', 'running'):
            return False
        job['cancelled'] = True
        job['progress_label'] = 'cancelling…'
        return True


def list_jobs(
    user_id: str,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[dict]:
    """List recent jobs for a user, newest first.

    Args:
        user_id: Filter to this user only.
        status: Optional filter ('queued'|'running'|'completed'|'failed'|'cancelled').
        limit: Max results (newest first).
    """
    with _jobs_lock:
        snapshot = [
            dict(j) for j in _active_jobs.values()
            if j.get('user_id') == user_id
            and (status is None or j.get('status') == status)
        ]
    snapshot.sort(key=lambda j: j.get('created_at', ''), reverse=True)
    return snapshot[:limit]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _update_job(job_id: str, **kwargs) -> None:
    """Thread-safe partial update on the in-memory job state."""
    with _jobs_lock:
        job = _active_jobs.get(job_id)
        if job is None:
            return
        job.update(kwargs)


def _is_cancelled(job_id: str) -> bool:
    with _jobs_lock:
        job = _active_jobs.get(job_id)
        return bool(job and job.get('cancelled'))


def _run_job_worker(job_id: str) -> None:
    """Daemon thread entry. Iterates strategies, calls the right helper.

    Mirrors mass_builder's worker pattern but simpler — no checkpoint/
    resume, no combinatorial enumeration. Just sequential per-strategy
    processing with progress updates between each.
    """
    with _jobs_lock:
        job = dict(_active_jobs.get(job_id) or {})
    if not job:
        return

    user_id = job['user_id']
    strategy_ids = job['strategy_ids']
    job_type = job['job_type']

    _update_job(job_id, status='running', started_at=_now_iso(),
                progress_label='starting')

    # Lazy import to avoid circular at module-load time
    try:
        from api.services.forward_test_service import (
            append_new_trades_for_strategy,
            recompute_and_persist_stored_trades,
        )
        from db import get_strategy_by_id_admin
    except Exception as e:
        _update_job(job_id, status='failed', completed_at=_now_iso(),
                    error=f'import failed: {type(e).__name__}: {e}')
        logger.exception("[RECOMPUTE-JOB] %s import failed", job_id[:8])
        return

    per_results: List[dict] = []
    summary = dict(job['summary'])

    try:
        for idx, sid in enumerate(strategy_ids):
            if _is_cancelled(job_id):
                _update_job(job_id,
                            status='cancelled',
                            completed_at=_now_iso(),
                            progress_label='cancelled by user')
                logger.info(
                    "[RECOMPUTE-JOB] %s cancelled at idx=%s",
                    job_id[:8], idx)
                return

            # Look up strategy name for progress display
            strat_name = ''
            try:
                strat_meta = get_strategy_by_id_admin(sid, user_id)
                strat_name = (strat_meta or {}).get('name', f'sid {sid}')
            except Exception:
                strat_name = f'sid {sid}'

            _update_job(job_id,
                        current_strategy_idx=idx,
                        current_strategy_id=sid,
                        current_strategy_name=strat_name,
                        progress_label=f'running engine ({job_type})')

            t0 = time.time()
            r: Dict[str, Any]
            try:
                # Per-strategy lock prevents cron + manual jobs from
                # racing on the same strategy. If both try the same sid
                # at once, one waits for the other to release.
                with _get_strategy_lock(sid):
                    if job_type == 'append_recent':
                        r = append_new_trades_for_strategy(sid, user_id)
                    else:  # full_recompute
                        r = recompute_and_persist_stored_trades(
                            sid, user_id, compute_parity=False)
            except Exception as e:
                logger.exception(
                    "[RECOMPUTE-JOB] %s strategy=%s crashed: %s",
                    job_id[:8], sid, e)
                r = {'status': 'error', 'error': str(e), 'inserted': 0}

            elapsed = round(time.time() - t0, 2)
            per_entry = {
                'strategy_id': sid,
                'strategy_name': strat_name,
                'status': r.get('status'),
                'inserted': r.get('inserted', r.get('trades', 0)) or 0,
                'engine_path': r.get('engine_path'),
                'elapsed_s': elapsed,
                'error': r.get('error') or r.get('reason'),
            }
            per_results.append(per_entry)

            # Update summary counters
            summary['total_inserted'] += int(per_entry['inserted'] or 0)
            if per_entry['status'] == 'skipped':
                summary['total_skipped'] += 1
            elif per_entry['status'] in ('error', 'failed'):
                summary['total_failed'] += 1
            elif per_entry['status'] in ('appended', 'refreshed'):
                summary['total_appended'] += 1

            _update_job(job_id,
                        per_strategy_results=list(per_results),
                        summary=dict(summary))

        _update_job(job_id,
                    status='completed',
                    completed_at=_now_iso(),
                    progress_label='done',
                    per_strategy_results=per_results,
                    summary=summary)
        logger.info(
            "[RECOMPUTE-JOB] %s completed: %s",
            job_id[:8], summary)
    except Exception as e:
        logger.exception(
            "[RECOMPUTE-JOB] %s outer worker failure: %s",
            job_id[:8], e)
        _update_job(job_id,
                    status='failed',
                    completed_at=_now_iso(),
                    error=f'{type(e).__name__}: {e}',
                    per_strategy_results=per_results,
                    summary=summary)
