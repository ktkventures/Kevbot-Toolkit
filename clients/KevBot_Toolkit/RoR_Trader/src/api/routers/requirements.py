"""Requirements router — CRUD for portfolio requirement sets."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Body

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/requirements", tags=["requirements"])


# =============================================================================
# CRUD
# =============================================================================

@router.get("")
def list_requirements(user=Depends(get_current_user)):
    """Load all requirement sets for the current user."""
    from portfolios import load_requirements
    return load_requirements()


@router.post("")
def create_requirement_set(req_set: dict = Body(...), user=Depends(get_current_user)):
    """Create a new requirement set."""
    from portfolios import save_requirement_set
    result = save_requirement_set(req_set)
    return result


@router.get("/{req_id}")
def get_requirement_set(req_id: int, user=Depends(get_current_user)):
    """Get a single requirement set by ID."""
    from portfolios import get_requirement_set_by_id
    req_set = get_requirement_set_by_id(req_id)
    if req_set is None:
        raise HTTPException(status_code=404, detail="Requirement set not found")
    return req_set


@router.put("/{req_id}")
def update_requirement_set(
    req_id: int,
    updated: dict = Body(...),
    user=Depends(get_current_user),
):
    """Update an existing requirement set."""
    from portfolios import update_requirement_set as _update
    success = _update(req_id, updated)
    if not success:
        raise HTTPException(status_code=404, detail="Requirement set not found")
    return {"status": "updated"}


@router.delete("/{req_id}")
def delete_requirement_set(req_id: int, user=Depends(get_current_user)):
    """Delete a requirement set by ID. Built-in sets cannot be deleted."""
    from portfolios import delete_requirement_set as _delete
    deleted = _delete(req_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Requirement set not found or is built-in")
    return {"status": "deleted"}
