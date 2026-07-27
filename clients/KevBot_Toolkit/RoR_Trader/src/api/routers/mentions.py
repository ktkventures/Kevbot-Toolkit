"""Mentions API — the @-mention inbox that closes the last cracks-through gap
(board #143, V2.12).

Rows are WRITTEN by the dev_tasks comment endpoint (parse @tokens at POST
time); this router only READS the inbox and marks mentions seen. Same posture
as dev_tasks / agents: admin-gated, service-role client, fail-loud validation.

Symmetry that makes it work: the Messages tab reads `?for=kevin&unseen=true`
so Kevin sees what the agents tagged him on; the SessionStart hook + dispatcher
read `?for=<role>&unseen=true` (and POST /seen-all) so an agent picking up work
sees "Kevin asked you X on #N" without anyone remembering to tell it.
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_current_user

router = APIRouter(prefix="/api/mentions", tags=["mentions"])


def _admin():
    from db import get_admin_client
    return get_admin_client()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cutoff_iso(days: int) -> str:
    # No microseconds / offset — keeps it clean inside a PostgREST or() filter.
    return (datetime.now(timezone.utc) - timedelta(days=days)) \
        .strftime("%Y-%m-%dT%H:%M:%S")


@router.get("")
def list_mentions(for_: str = Query(..., alias="for"),
                  unseen: bool = True, days: int = 7, limit: int = 200,
                  user=Depends(get_current_user)):
    """Mentions addressed to role `for`, newest first.

    unseen=true (default) → the inbox: only seen_at IS NULL.
    unseen=false → the "show already-seen" view: still-unseen PLUS anything
    seen or created within the last `days` (default 7) — so the toggle reveals
    recent history without ever hiding a stale unseen mention.

    Each row is enriched with the task title/status and the comment body
    (excerpt) via a manual join — robust vs PostgREST relationship detection.
    """
    role = (for_ or "").strip()
    if not role:
        raise HTTPException(status_code=400, detail="'for' is required")
    c = _admin()
    q = c.table("task_mentions").select("*").eq("mentioned", role)
    if unseen:
        q = q.is_("seen_at", "null")
    else:
        q = q.or_(f"seen_at.is.null,created_at.gte.{_cutoff_iso(days)}")
    rows = q.order("created_at", desc=True).limit(limit).execute().data or []

    task_ids = sorted({r["task_id"] for r in rows})
    comment_ids = sorted({r["comment_id"] for r in rows})
    tasks, comments = {}, {}
    if task_ids:
        for t in (c.table("dev_tasks").select("id,title,status")
                  .in_("id", task_ids).execute().data or []):
            tasks[t["id"]] = t
    if comment_ids:
        for cm in (c.table("dev_task_comments").select("id,body")
                   .in_("id", comment_ids).execute().data or []):
            comments[cm["id"]] = cm
    for r in rows:
        t = tasks.get(r["task_id"]) or {}
        r["task_title"] = t.get("title")
        r["task_status"] = t.get("status")
        r["excerpt"] = (comments.get(r["comment_id"]) or {}).get("body") or ""
    return rows


@router.post("/seen-all")
def mark_all_seen(for_: str = Query(..., alias="for"),
                  user=Depends(get_current_user)):
    """Mark every unseen mention for role `for` as seen. Returns the count."""
    role = (for_ or "").strip()
    if not role:
        raise HTTPException(status_code=400, detail="'for' is required")
    res = _admin().table("task_mentions").update({"seen_at": _now_iso()}) \
        .eq("mentioned", role).is_("seen_at", "null").execute()
    return {"status": "ok", "marked": len(res.data or [])}


@router.post("/{mention_id}/seen")
def mark_seen(mention_id: int, user=Depends(get_current_user)):
    """Mark one mention seen (idempotent — re-marking just re-stamps)."""
    res = _admin().table("task_mentions").update({"seen_at": _now_iso()}) \
        .eq("id", mention_id).execute()
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="mention not found")
    return rows[0]
