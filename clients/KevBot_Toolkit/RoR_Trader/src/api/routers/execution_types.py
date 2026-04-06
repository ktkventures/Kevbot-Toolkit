"""Execution Types Router — list and inspect execution type modules."""

from fastapi import APIRouter, Depends, HTTPException
from api.deps import get_current_user

router = APIRouter(prefix="/api/execution-types", tags=["execution-types"])


@router.get("")
def list_execution_types(user=Depends(get_current_user)):
    """List all registered execution type modules."""
    from execution_types import list_modules
    return [m.to_dict() for m in list_modules()]


@router.get("/{slug}")
def get_execution_type(slug: str, user=Depends(get_current_user)):
    """Get a specific execution type module by slug."""
    from execution_types import list_modules
    for m in list_modules():
        if m.slug == slug:
            return m.to_dict()
    raise HTTPException(status_code=404, detail=f"Execution type '{slug}' not found")
