"""Agents Registry API — the multi-session team roster (Spec_Agents_Registry.md).

Admin-gated CRUD over `agents` (see migrations/agents_registry.sql), mirroring
the dev_tasks router pattern: service-role client, field whitelist, fail-loud
validation. Phase 1: the UI is read-only; the CRUD surface exists so M manages
rows programmatically, same as the board. Until the V4.9 dispatcher ships,
Session_Charters.md §1 stays authoritative and this table mirrors it.
"""
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Fields a caller may set on create / update (whitelist — ignore anything else).
_EDITABLE = {
    "letter", "name", "kind", "department", "status", "scope", "boundaries",
    "worktree", "context_docs", "prompt_template", "notes",
}

_KINDS = {"agent", "human"}
_DEPARTMENTS = {"dev", "builder", "marketing", "ops"}
_STATUSES = {"live-session", "headless", "dormant", "retired", "ephemeral"}


def _admin():
    from db import get_admin_client
    return get_admin_client()


def _validate(row: dict):
    if "kind" in row and row["kind"] not in _KINDS:
        raise HTTPException(
            status_code=400, detail=f"kind must be one of {sorted(_KINDS)}")
    if "department" in row and row["department"] not in _DEPARTMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"department must be one of {sorted(_DEPARTMENTS)}")
    if "status" in row and row["status"] not in _STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"status must be one of {sorted(_STATUSES)}")
    if "context_docs" in row and not (
            isinstance(row["context_docs"], list)
            and all(isinstance(d, str) for d in row["context_docs"])):
        raise HTTPException(
            status_code=400, detail="context_docs must be a list of strings")


@router.get("")
def list_agents(active: bool = False, user=Depends(get_current_user)):
    """All agents in seed order. `active=true` filters out retired rows —
    what the task UI's assignee/author dropdowns consume."""
    c = _admin()
    q = c.table("agents").select("*")
    if active:
        q = q.neq("status", "retired")
    return q.order("id").execute().data or []


@router.post("")
def create_agent(payload: dict = Body(...), user=Depends(get_current_user)):
    """Create an agent. Requires `letter` + `name`."""
    if not (payload.get("letter") or "").strip():
        raise HTTPException(status_code=400, detail="letter is required")
    if not (payload.get("name") or "").strip():
        raise HTTPException(status_code=400, detail="name is required")
    row = {k: v for k, v in payload.items() if k in _EDITABLE}
    _validate(row)
    res = _admin().table("agents").insert(row).execute()
    return res.data[0] if res.data else {"status": "created"}


@router.patch("/{agent_id}")
def update_agent(agent_id: int, payload: dict = Body(...),
                 user=Depends(get_current_user)):
    """Partial update — only whitelisted fields. Empty patch is a no-op read."""
    row = {k: v for k, v in payload.items() if k in _EDITABLE}
    c = _admin()
    if row:
        _validate(row)
        c.table("agents").update(row).eq("id", agent_id).execute()
    res = c.table("agents").select("*").eq("id", agent_id).execute().data
    if not res:
        raise HTTPException(status_code=404, detail="agent not found")
    return res[0]
