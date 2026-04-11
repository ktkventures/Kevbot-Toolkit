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
        "config": config,
        "status": "queued",
        "progress": 0,
        "total": 0,
        "results": [],
    }
    saved = save_mass_search(search)
    search_id = saved.get("id") if isinstance(saved, dict) else None

    # Launch background execution
    if search_id:
        from mass_builder import start_mass_search_async
        start_mass_search_async(search_id, config)

    return {"search_id": search_id, "status": "running"}


@router.get("/progress/{search_id}")
def get_progress(search_id: int, user=Depends(get_current_user)):
    """Poll progress of a running mass search."""
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    # Check in-memory state first (faster, updated in real-time by worker thread)
    from mass_builder import get_search_progress
    mem_progress = get_search_progress(search_id)
    if mem_progress:
        return {
            "search_id": search_id,
            "status": mem_progress.get("status", "running"),
            "progress": mem_progress.get("current_step", 0),
            "total": mem_progress.get("total_steps", 0),
            "current_label": mem_progress.get("current_label", ""),
        }

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


@router.post("/cancel/{search_id}")
def cancel_search(search_id: int, user=Depends(get_current_user)):
    """Cancel a running mass search."""
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import get_mass_search, update_mass_search
    search = get_mass_search(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")

    search["status"] = "cancelled"
    update_mass_search(search_id, search)
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
def get_result(search_id: int, user=Depends(get_current_user)):
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
def delete_result(search_id: int, user=Depends(get_current_user)):
    """Delete a mass search result."""
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import delete_mass_search
    deleted = delete_mass_search(search_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Search not found")
    return {"status": "deleted"}
