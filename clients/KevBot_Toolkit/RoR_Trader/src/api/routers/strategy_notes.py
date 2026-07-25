"""Strategy notes — board #70 Phase B (M spec 07-24 17:05Z).

Per-sid freeform notes canvas on the Strategy Health page: humans and the
nightly/AI writers leave explanations ("why is this sid red") that persist
across sessions. Mirrors the dev_task_comments pattern: service-role client,
admin-gated, newest first. Table `strategy_notes` was migrated + seeded by M
(sid, author, body, created_at; RLS service-role only).
"""
from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

router = APIRouter(prefix="/api/strategy-notes", tags=["strategy-health"])


def _admin():
    from db import get_admin_client
    return get_admin_client()


@router.get("")
def all_notes(user=Depends(get_current_user)):
    """Every note, newest first — the health page groups client-side for
    per-row counts (the table is tooling-sized; one read beats N)."""
    return _admin().table("strategy_notes").select("*") \
        .order("created_at", desc=True).limit(2000).execute().data or []


@router.get("/{sid}")
def notes_for_sid(sid: int, user=Depends(get_current_user)):
    return _admin().table("strategy_notes").select("*") \
        .eq("sid", sid).order("created_at", desc=True).execute().data or []


@router.post("/{sid}")
def add_note(sid: int, payload: dict = Body(...),
             user=Depends(get_current_user)):
    body = (payload.get("body") or "").strip()
    if not body:
        raise HTTPException(status_code=400, detail="body is required")
    author = payload.get("author") or "claude"
    res = _admin().table("strategy_notes").insert(
        {"sid": sid, "author": author, "body": body}).execute()
    return res.data[0] if res.data else {"status": "added"}
