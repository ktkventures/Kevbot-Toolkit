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
    force: bool = False,
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
        'force': bool(force),
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


# ---------------------------------------------------------------------------
# Parallel recompute (M-RS3 Phase 1)
# ---------------------------------------------------------------------------
# RORT_RECOMPUTE_PARALLELISM (default 1 == today's exact sequential loop) runs
# the embarrassingly-parallel per-strategy unit across N processes. Threads do
# NOT help: the engine process_bar loop is GIL-bound (~70% of cost is Python
# CPU). Each worker is a fresh interpreter, so _pool_init re-arms the two
# silent-failure traps — an unloaded pack registry (=> every recompute returns
# 0 trades) and a fork-inherited Supabase admin client (=> shared socket). The
# per-strategy compute is byte-identical to the sequential path: parallelism
# only changes scheduling. See docs/_active/Design_M-RS3_Parallel_Recompute.md.

def _status_rank(s: Optional[str]) -> int:
    return {'error': 4, 'skipped': 3, 'appended': 2,
            'refreshed': 2, 'no_new_trades': 1,
            'no_trades': 1}.get(s or '', 0)


def _lookup_strat_name(sid: int, user_id: str) -> str:
    try:
        from db import get_strategy_by_id_admin
        meta = get_strategy_by_id_admin(sid, user_id)
        return (meta or {}).get('name', f'sid {sid}')
    except Exception:
        return f'sid {sid}'


def _pool_init(user_id: str) -> None:
    """ProcessPoolExecutor initializer — runs ONCE per worker process.

    A forked child inherits the parent's module globals, including the cached
    Supabase admin client (a live socket; sharing it across processes corrupts
    responses) and — under spawn — an unscanned pack registry. Re-arm both:
      1. Null db._admin_client / _anon_client so each process builds its OWN.
      2. scan_and_load_all() and assert packs>0 (the silent-0-trades trap;
         feedback_worker_pack_registry_init / feedback_local_script_pack_registry).
      3. set_admin_user_context so user-context helpers run without a JWT.
    """
    import db
    # Force a fresh per-process admin client (never reuse a forked socket).
    try:
        with db._client_lock:
            db._admin_client = None
            db._anon_client = None
    except Exception:
        db._admin_client = None
        db._anon_client = None
    import pack_registry
    loaded = pack_registry.scan_and_load_all()  # -> Dict[str, RegisteredPack]
    npacks = len(loaded) if hasattr(loaded, '__len__') else (loaded or 0)
    if not npacks:
        raise RuntimeError(
            f"[RECOMPUTE-POOL] pack registry empty (packs={npacks}) in worker — "
            f"every recompute would silently return 0 trades; aborting")
    db.set_admin_user_context(user_id)
    logger.info("[RECOMPUTE-POOL] worker initialized: %d packs, user=%s",
                npacks, (user_id or '')[:8])


def _recompute_one(sid: int, user_id: str, job_type: str,
                   force: bool) -> Dict[str, Any]:
    """Compute BOTH lanes for one strategy. Picklable, top-level — runs in a
    pool worker (parallel) or inline in the parent (N==1). Returns raw lane
    results; the PARENT owns all job-state writes (no cross-process race on the
    job row). Mirrors the original sequential loop body exactly, minus the
    in-process per-strategy lock (which can't span processes; see caller)."""
    t0 = time.time()
    from api.services.forward_test_service import (
        append_new_trades_for_strategy,
        recompute_and_persist_stored_trades,
        append_new_backtest_trades_for_strategy,
        recompute_and_persist_algo_trades,
    )
    bt_r: Dict[str, Any] = {}
    algo_r: Dict[str, Any] = {}
    try:
        if job_type == 'append_recent':
            try:
                bt_r = append_new_backtest_trades_for_strategy(
                    sid, user_id, force=force)
            except Exception as e:
                bt_r = {'status': 'error', 'reason': str(e)}
            try:
                algo_r = append_new_trades_for_strategy(
                    sid, user_id, force=force)
            except Exception as e:
                algo_r = {'status': 'error', 'reason': str(e)}
        else:  # full_recompute
            try:
                bt_r = recompute_and_persist_stored_trades(
                    sid, user_id, compute_parity=False)
            except Exception as e:
                bt_r = {'status': 'error', 'reason': str(e)}
            try:
                algo_r = recompute_and_persist_algo_trades(sid, user_id)
            except Exception as e:
                algo_r = {'status': 'error', 'reason': str(e)}
    except Exception as e:
        bt_r = {'status': 'error', 'error': str(e)}
        algo_r = {'status': 'error', 'error': str(e)}
    return {'sid': sid, 'bt_r': bt_r, 'algo_r': algo_r,
            'elapsed': round(time.time() - t0, 2)}


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
    force = bool(job.get('force'))

    _update_job(job_id, status='running', started_at=_now_iso(),
                progress_label='starting')

    # Lazy import to avoid circular at module-load time
    try:
        from api.services.forward_test_service import (
            append_new_trades_for_strategy,
            recompute_and_persist_stored_trades,
            append_new_backtest_trades_for_strategy,
            recompute_and_persist_algo_trades,
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
        # Assemble + publish one strategy's result. Closes over per_results /
        # summary / job_id so the sequential and parallel branches share
        # identical aggregation (the only difference between them is
        # *scheduling*, never the recorded numbers).
        def _publish_result(sid, strat_name, res):
            bt_r = res['bt_r']
            algo_r = res['algo_r']
            elapsed = res['elapsed']
            # Combined per-strategy entry — pick the more-informative status
            # for the row label, sum inserted counts across lanes.
            bt_inserted = bt_r.get('inserted', bt_r.get('trades', 0)) or 0
            algo_inserted = algo_r.get('inserted') or 0
            combined_inserted = int(bt_inserted) + int(algo_inserted)
            # Status priority: error > skipped > non-zero-insert > zero-insert
            primary_status = bt_r.get('status') if _status_rank(bt_r.get('status')) >= _status_rank(algo_r.get('status')) else algo_r.get('status')

            per_entry = {
                'strategy_id': sid,
                'strategy_name': strat_name,
                'status': primary_status,
                'inserted': combined_inserted,
                'engine_path': bt_r.get('engine_path') or algo_r.get('engine_path'),
                'bars_processed': bt_r.get('bars_processed') or algo_r.get('bars_processed'),
                'window_days': bt_r.get('window_days') or algo_r.get('window_days'),
                'reason': bt_r.get('reason') or algo_r.get('reason'),
                'elapsed_s': elapsed,
                'error': bt_r.get('error') or bt_r.get('reason') if bt_r.get('status') == 'error'
                         else (algo_r.get('error') or algo_r.get('reason') if algo_r.get('status') == 'error' else None),
                # Lane breakdown for detail UI
                'backtest_lane': {
                    'status': bt_r.get('status'),
                    'inserted': bt_inserted,
                    'reason': bt_r.get('reason'),
                },
                'algo_lane': {
                    'status': algo_r.get('status'),
                    'inserted': algo_inserted,
                    'reason': algo_r.get('reason'),
                },
            }
            per_results.append(per_entry)

            # Update summary counters
            summary['total_inserted'] += combined_inserted
            if primary_status == 'skipped':
                summary['total_skipped'] += 1
            elif primary_status in ('error', 'failed'):
                summary['total_failed'] += 1
            elif primary_status in ('appended', 'refreshed'):
                summary['total_appended'] += 1

            _update_job(job_id,
                        per_strategy_results=list(per_results),
                        summary=dict(summary))

        # M-RS3: parallelism degree. Default 1 == today's exact sequential
        # path (also the kill-switch). >1 fans the catalog across N processes.
        try:
            parallelism = max(1, int(
                os.getenv('RORT_RECOMPUTE_PARALLELISM', '1') or '1'))
        except (TypeError, ValueError):
            parallelism = 1

        if parallelism <= 1:
            # Sequential — byte-identical to the pre-M-RS3 loop. Keeps the
            # in-process per-strategy lock (cron × manual de-race).
            for idx, sid in enumerate(strategy_ids):
                if _is_cancelled(job_id):
                    _update_job(job_id, status='cancelled',
                                completed_at=_now_iso(),
                                progress_label='cancelled by user')
                    logger.info("[RECOMPUTE-JOB] %s cancelled at idx=%s",
                                job_id[:8], idx)
                    return
                strat_name = _lookup_strat_name(sid, user_id)
                _update_job(job_id, current_strategy_idx=idx,
                            current_strategy_id=sid,
                            current_strategy_name=strat_name,
                            progress_label=f'running engine ({job_type})')
                # Per-strategy lock prevents cron + manual jobs from racing on
                # the same sid. _recompute_one runs BOTH lanes (Option A,
                # 2026-05-08), mirroring the /update?mode= dispatch table.
                with _get_strategy_lock(sid):
                    res = _recompute_one(sid, user_id, job_type, force)
                _publish_result(sid, strat_name, res)
        else:
            # Parallel (M-RS3 Phase 1) — N worker processes over the
            # embarrassingly-parallel catalog. The PARENT owns every job-state
            # write (no cross-process race on the job row). The in-process
            # per-strategy lock can't span processes; each sid is submitted
            # once per job, and cross-job (cron) racing is a documented
            # Phase-1 limitation that the default-OFF flag keeps inert.
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            total = len(strategy_ids)
            logger.info(
                "[RECOMPUTE-JOB] %s PARALLEL N=%d over %d strategies (type=%s)",
                job_id[:8], parallelism, total, job_type)
            _update_job(
                job_id,
                progress_label=f'running engine ({job_type}) ×{parallelism}')
            done = 0
            # SPAWN context (not fork): the host (API/worker) is multithreaded,
            # and fork-from-threaded can deadlock if a child inherits a held
            # lock (e.g. db._client_lock). Spawn gives each worker a clean
            # interpreter — _pool_init then loads the registry + a fresh admin
            # client from scratch. Worker startup re-imports (~secs), amortized
            # against ~193s/strat.
            with ProcessPoolExecutor(max_workers=parallelism,
                                     mp_context=_mp.get_context('spawn'),
                                     initializer=_pool_init,
                                     initargs=(user_id,)) as ex:
                futs = {ex.submit(_recompute_one, sid, user_id, job_type,
                                  force): sid for sid in strategy_ids}
                for fut in as_completed(futs):
                    sid = futs[fut]
                    if _is_cancelled(job_id):
                        _update_job(job_id, status='cancelled',
                                    completed_at=_now_iso(),
                                    progress_label='cancelled by user')
                        logger.info(
                            "[RECOMPUTE-JOB] %s cancelled (parallel) after "
                            "%d/%d", job_id[:8], done, total)
                        ex.shutdown(wait=False, cancel_futures=True)
                        return
                    try:
                        res = fut.result()
                    except Exception as e:
                        logger.exception(
                            "[RECOMPUTE-JOB] %s strategy=%s worker crashed: %s",
                            job_id[:8], sid, e)
                        res = {'sid': sid,
                               'bt_r': {'status': 'error', 'error': str(e)},
                               'algo_r': {'status': 'error', 'error': str(e)},
                               'elapsed': 0}
                    done += 1
                    strat_name = _lookup_strat_name(sid, user_id)
                    _update_job(
                        job_id, current_strategy_idx=done,
                        current_strategy_id=sid,
                        current_strategy_name=strat_name,
                        progress_label=f'{done}/{total} ({job_type}, ×{parallelism})')
                    _publish_result(sid, strat_name, res)

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
