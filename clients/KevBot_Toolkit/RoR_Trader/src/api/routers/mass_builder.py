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
    return {"search_id": search_id, "status": "queued"}


@router.get("/progress/{search_id}")
def get_progress(search_id: int, user=Depends(get_current_user)):
    """Poll progress of a running mass search."""
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Mass builder requires DB mode")

    from db import get_mass_search
    search = get_mass_search(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")

    return {
        "search_id": search_id,
        "status": search.get("status", "unknown"),
        "progress": search.get("progress", 0),
        "total": search.get("total", 0),
        "current_label": search.get("current_label", ""),
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
