"""Update Jobs router — async UAD execution + status polling.

Modeled directly on `src/api/routers/mass_builder.py`. Same async-by-job_id
pattern: kick off, poll progress, cancel/delete.

Replaces (or runs alongside) the synchronous `POST /api/strategies/{id}/update`
endpoint. Frontend prefers this when the user wants to walk away — the
job continues even if the browser closes.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/update-jobs", tags=["update-jobs"])


@router.post("/run")
def start_update_job(
    payload: dict = Body(...),
    user=Depends(get_current_user),
):
    """Start an async UAD job. Returns immediately with `job_id`.

    Request body:
      - mode: 'all' | 'new' (required)
      - strategy_ids: [int, int, ...] (required; can be length 1 for single-strategy)
      - name: optional display name

    Frontend then polls `/api/update-jobs/progress/{job_id}` to render status.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="update-jobs requires DB mode")

    mode = payload.get("mode")
    if mode not in ("all", "new"):
        raise HTTPException(
            status_code=400,
            detail="payload.mode must be 'all' or 'new'",
        )

    strategy_ids = payload.get("strategy_ids")
    if not isinstance(strategy_ids, list) or not strategy_ids:
        raise HTTPException(
            status_code=400,
            detail="payload.strategy_ids must be a non-empty list of ints",
        )
    try:
        strategy_ids = [int(x) for x in strategy_ids]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="payload.strategy_ids must be ints",
        )

    name = payload.get("name") or (
        f"Update {mode} ({len(strategy_ids)} strategies)"
    )
    scope = "single" if len(strategy_ids) == 1 else "bulk"
    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)

    from db import save_update_job
    job_id = save_update_job({
        "name": name,
        "mode": mode,
        "scope": scope,
        "strategy_ids": strategy_ids,
        "status": "queued",
        "progress": {
            "current_step": 0,
            "total_steps": len(strategy_ids),
            "current_sid": None,
            "current_phase": "queued",
            "per_strategy": {},
        },
    })
    if job_id is None:
        raise HTTPException(
            status_code=500,
            detail="failed to persist update job",
        )

    from update_jobs import start_update_job_async
    start_update_job_async(job_id, mode, strategy_ids, user_id)

    return {
        "job_id": job_id,
        "status": "running",
        "mode": mode,
        "scope": scope,
        "strategy_count": len(strategy_ids),
    }


@router.get("/progress/{job_id}")
def get_job_progress(job_id: int, user=Depends(get_current_user)):
    """Poll progress of an update job."""
    from db import USE_DB, get_update_job
    if not USE_DB:
        raise HTTPException(status_code=501, detail="update-jobs requires DB mode")

    job = get_update_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Shape mirrors mass_builder /progress — extract progress fields to top-level
    progress = job.get("progress") or {}
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "mode": job.get("mode"),
        "scope": job.get("scope"),
        "strategy_ids": job.get("strategy_ids") or [],
        "current_step": progress.get("current_step", 0),
        "total_steps": progress.get("total_steps", 0),
        "current_sid": progress.get("current_sid"),
        "current_phase": progress.get("current_phase"),
        "per_strategy": progress.get("per_strategy") or {},
        "summary": job.get("summary"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


@router.post("/cancel/{job_id}")
def cancel_job(job_id: int, user=Depends(get_current_user)):
    """Signal cancellation. Worker exits cleanly between strategies.

    For jobs that aren't actively running (orphaned, queued without a
    worker), this still flips the DB status so the UI updates.
    """
    from db import USE_DB, get_update_job, update_update_job
    if not USE_DB:
        raise HTTPException(status_code=501, detail="update-jobs requires DB mode")

    job = get_update_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    from update_jobs import cancel_update_job as _cancel_in_memory
    _cancel_in_memory(job_id)
    update_update_job(job_id, {"status": "cancelled"})
    return {"status": "cancelled"}


@router.get("")
def list_jobs(user=Depends(get_current_user)):
    """List recent update jobs for current user."""
    from db import USE_DB
    if not USE_DB:
        return []
    from db import load_update_jobs
    return load_update_jobs(limit=50)


@router.get("/{job_id}")
def get_job(job_id: int, user=Depends(get_current_user)):
    """Get full job record including config_data (results + per-strategy detail)."""
    from db import USE_DB, get_update_job
    if not USE_DB:
        raise HTTPException(status_code=501, detail="update-jobs requires DB mode")
    job = get_update_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{job_id}")
def delete_job(job_id: int, user=Depends(get_current_user)):
    """Delete a job. Cancels first if still running."""
    from db import USE_DB, get_update_job, delete_update_job
    if not USE_DB:
        raise HTTPException(status_code=501, detail="update-jobs requires DB mode")
    from update_jobs import cancel_update_job as _cancel_in_memory
    # Best-effort cancel — harmless if not running
    _cancel_in_memory(job_id)
    deleted = delete_update_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted"}


@router.post("/cleanup-orphans")
def cleanup_orphans_now(user=Depends(get_current_user)):
    """Manual orphan-cleanup trigger.

    Same logic as the api-boot startup cleanup, exposed for the rare case
    where a job hangs in 'running' after the orphan window. Returns
    `{orphaned, ids}`.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="update-jobs requires DB mode")
    from update_jobs import cleanup_orphaned_update_jobs
    return cleanup_orphaned_update_jobs()
