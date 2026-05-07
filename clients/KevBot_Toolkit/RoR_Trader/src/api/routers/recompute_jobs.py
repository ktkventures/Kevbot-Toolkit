"""Background-job endpoints for algo-history recompute operations.

Mirrors the Mass Builder router shape but scoped down. Job state lives
in-memory in `recompute_jobs._active_jobs`; no DB persistence in v1.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _resolve_user_id(user) -> str:
    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="missing user_id")
    return user_id


@router.post("/recompute")
def queue_recompute_job(
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Queue a recompute job. Returns job_id immediately.

    Body:
      {
        "strategy_ids": [int, ...],
        "job_type": "append_recent" | "full_recompute"
      }

    Worker thread runs in the API process; client polls
    GET /api/jobs/{id} for progress. No HTTP timeout concerns —
    the POST returns in <200ms regardless of engine cost.
    """
    from recompute_jobs import submit_recompute_job

    sids_raw = body.get('strategy_ids') or []
    job_type = (body.get('job_type') or '').strip()

    if not isinstance(sids_raw, list) or not sids_raw:
        raise HTTPException(
            status_code=400,
            detail="strategy_ids must be a non-empty list")
    try:
        strategy_ids = [int(x) for x in sids_raw]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail="strategy_ids must be integers")
    if job_type not in ('append_recent', 'full_recompute'):
        raise HTTPException(
            status_code=400,
            detail="job_type must be 'append_recent' or 'full_recompute'")

    user_id = _resolve_user_id(user)
    # `force=true` (default for the manual button) bypasses both the
    # recently_processed gate AND the alerts-gate inside the worker so
    # the engine ALWAYS runs. Catches missed trades (algo-only, no
    # live alert) and gives the user immediate feedback that work
    # happened. Cron path uses force=false (the default) for throughput.
    force = bool(body.get('force', True))

    # Defense-in-depth: filter to strategies actually owned by this user
    from db import get_admin_client
    client = get_admin_client()
    owned = client.table('strategies').select('id') \
        .in_('id', strategy_ids).eq('user_id', user_id).execute()
    owned_ids = [r['id'] for r in (owned.data or [])]
    if not owned_ids:
        raise HTTPException(
            status_code=404,
            detail="None of the requested strategies belong to this user")
    if len(owned_ids) != len(strategy_ids):
        rejected = sorted(set(strategy_ids) - set(owned_ids))
        logger.warning(
            "[RECOMPUTE-JOB] dropping non-owned strategy_ids=%s "
            "for user=%s",
            rejected, user_id[:8])

    try:
        job_id = submit_recompute_job(
            user_id, owned_ids, job_type, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[RECOMPUTE-JOB] submit failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        'job_id': job_id,
        'status': 'queued',
        'job_type': job_type,
        'strategy_count': len(owned_ids),
        'force': force,
    }


@router.get("/{job_id}")
def get_job(job_id: str, user=Depends(get_current_user)):
    """Snapshot of current job state."""
    from recompute_jobs import get_job_status
    user_id = _resolve_user_id(user)

    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    if job.get('user_id') != user_id:
        # Don't leak job existence cross-user
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@router.post("/{job_id}/cancel")
def cancel_job_endpoint(job_id: str, user=Depends(get_current_user)):
    """Mark a job for cancellation. Worker bails between strategies."""
    from recompute_jobs import cancel_job, get_job_status
    user_id = _resolve_user_id(user)

    job = get_job_status(job_id)
    if job is None or job.get('user_id') != user_id:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    ok = cancel_job(job_id)
    if not ok:
        return {'job_id': job_id, 'status': job.get('status'),
                'message': 'job already finalized; cancel had no effect'}
    return {'job_id': job_id, 'status': 'cancelling'}


@router.get("")
def list_jobs_endpoint(
    status: Optional[str] = Query(None, regex="^(queued|running|completed|failed|cancelled)$"),
    limit: int = Query(50, ge=1, le=200),
    user=Depends(get_current_user),
):
    """List recent jobs for the current user, newest first."""
    from recompute_jobs import list_jobs
    user_id = _resolve_user_id(user)
    return {
        'jobs': list_jobs(user_id, status=status, limit=limit),
        'total': None,  # not paginated; client knows from len(jobs)
    }
