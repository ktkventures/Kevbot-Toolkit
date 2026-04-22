"""Mass builder router — async search execution, progress polling, results."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Body

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mass-builder", tags=["mass-builder"])


@router.post("/run")
def start_mass_search(config: dict = Body(...), user=Depends(get_current_user)):
    """Start an async mass backtest search.

    Returns immediately with a search_id. Frontend polls /progress/{id}.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import save_mass_search
    search = {
        "name": config.get("name", "Untitled Search"),
        "config": config,
        "status": "queued",
        "progress": 0,
        "total": 0,
        "results": [],
    }
    search_id = save_mass_search(search)

    # Launch background execution
    if search_id:
        from mass_builder import start_mass_search_async
        start_mass_search_async(search_id, config)

    return {"search_id": search_id, "status": "running"}


@router.get("/progress/{search_id}")
def get_progress(search_id: str, user=Depends(get_current_user)):
    """Poll progress of a running mass search."""
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    # Check in-memory state first (faster, updated in real-time by worker thread)
    from mass_builder import get_search_progress
    # Try both int and string keys (DB returns int, in-memory may store either)
    mem_progress = get_search_progress(search_id) or get_search_progress(str(search_id))
    if mem_progress:
        resp = {
            "search_id": search_id,
            "status": mem_progress.get("status", "running"),
            "progress": mem_progress.get("current_step", 0),
            "total": mem_progress.get("total_steps", 0),
            "current_label": mem_progress.get("current_label", ""),
            "phase": mem_progress.get("phase"),
            "phase_detail": mem_progress.get("phase_detail"),
            "inner_step": mem_progress.get("inner_step"),
            "inner_total": mem_progress.get("inner_total"),
        }
        # Confluence sub-progress (legacy field; mirrors inner_step during confluence phase)
        if "conf_step" in mem_progress:
            resp["conf_step"] = mem_progress["conf_step"]
            resp["conf_total"] = mem_progress.get("conf_total", 0)
        # Include results when completed (in-memory has full data incl. stored_trades)
        if mem_progress.get("status") == "completed" and "results" in mem_progress:
            resp["results"] = mem_progress["results"]
        return resp

    # Fall back to DB
    from db import get_mass_search
    search = get_mass_search(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")

    progress = search.get("progress", {})
    return {
        "search_id": search_id,
        "status": search.get("status", "unknown"),
        "progress": progress.get("current_step", 0) if isinstance(progress, dict) else 0,
        "total": progress.get("total_steps", 0) if isinstance(progress, dict) else 0,
        "current_label": progress.get("current_label", "") if isinstance(progress, dict) else "",
    }


@router.post("/resume/{search_id}")
def resume_mass_search(search_id: str, user=Depends(get_current_user)):
    """Resume an orphaned (or failed) mass search from the last checkpoint.

    Reads the saved config + checkpoint payload from the search row and
    relaunches the worker with resume_checkpoint set, so it skips
    (symbol, tf) groups that have already completed. See Roadmap 9s.

    Flow:
      1. Fetch the saved search.
      2. Validate status is 'orphaned' or 'failed' (can't resume a running
         or completed search).
      3. Read the `config` + `checkpoint` subkeys from config_data (both
         surfaced at top level by get_mass_search's merge).
      4. Flip DB status to 'running' and relaunch the worker with the
         checkpoint.

    If no checkpoint exists (e.g. orphan happened before the first group
    finished), resumes from scratch using the saved config — equivalent
    to clicking Edit → Analyze but keeps the same search_id.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import get_mass_search, update_mass_search
    from mass_builder import start_mass_search_async

    search = get_mass_search(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")

    status = search.get('status')
    if status not in ('orphaned', 'failed'):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume a search with status '{status}'. "
                   f"Only orphaned or failed searches can be resumed.",
        )

    # get_mass_search merges config_data into the top level, so both of
    # these are reachable directly.
    checkpoint = search.get('checkpoint')
    # The original run's config was stored on `config_data.config` by
    # the original POST /run handler.
    config = search.get('config') or {}
    if not config:
        raise HTTPException(
            status_code=400,
            detail="Search has no stored config — cannot resume.",
        )

    # Flip status first so any concurrent polls see 'running'.
    update_mass_search(search_id, {'status': 'running'})

    # Launch worker with checkpoint (may be None for empty-checkpoint case).
    start_mass_search_async(search_id, config, resume_checkpoint=checkpoint)

    completed_groups = 0
    if isinstance(checkpoint, dict):
        completed_groups = len(checkpoint.get('completed_symbol_tfs') or [])

    return {
        "search_id": search_id,
        "status": "running",
        "resumed_from_checkpoint": checkpoint is not None,
        "completed_groups": completed_groups,
    }


@router.post("/cancel/{search_id}")
def cancel_search(search_id: str, user=Depends(get_current_user)):
    """Cancel a running mass search.

    Sets the in-memory `cancelled` flag so the worker thread exits on its
    next progress check (typically within a few hundred ms). Also updates
    DB status to cancelled so the UI reflects the change immediately.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import get_mass_search, update_mass_search
    from mass_builder import cancel_search as _cancel_in_memory

    search = get_mass_search(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")

    # Signal the running thread (if any) to stop
    _cancel_in_memory(str(search_id))
    _cancel_in_memory(search_id if isinstance(search_id, str) else str(search_id))
    # Mark DB state
    update_mass_search(search_id, {'status': 'cancelled'})
    return {"status": "cancelled"}


@router.get("/results")
def list_results(user=Depends(get_current_user)):
    """List all saved mass search results."""
    from db import USE_DB
    if not USE_DB:
        return []

    from db import load_mass_searches
    return load_mass_searches() or []


@router.get("/results/{search_id}")
def get_result(search_id: str, user=Depends(get_current_user)):
    """Get a specific mass search result."""
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import get_mass_search
    search = get_mass_search(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")
    return search


@router.delete("/results/{search_id}")
def delete_result(search_id: str, user=Depends(get_current_user)):
    """Delete a mass search result.

    If the search is still running, signal cancellation first so the worker
    thread exits cleanly instead of writing results back to a deleted row.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import delete_mass_search
    from mass_builder import cancel_search as _cancel_in_memory, is_search_running

    # Signal cancellation regardless — harmless if not running.
    _cancel_in_memory(str(search_id))
    deleted = delete_mass_search(search_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Search not found")
    return {"status": "deleted"}
