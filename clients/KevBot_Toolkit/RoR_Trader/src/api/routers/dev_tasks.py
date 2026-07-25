"""Dev Task Tracker API — a ClickUp-lite shared board for Kevin + Claude.

Admin-gated CRUD over `dev_tasks` + `dev_task_comments` (see
migrations/dev_tasks_table.sql). Uses the service-role admin client. Tasks sort
by priority (phase, seq) ascending so 1.1 = "do next" surfaces first.
"""
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

router = APIRouter(prefix="/api/dev-tasks", tags=["dev-tasks"])

# Fields a caller may set on create / update (whitelist — ignore anything else).
_EDITABLE = {
    "title", "description", "status", "priority_phase", "priority_seq",
    "is_urgent", "impacts_live", "needs_live_validation", "area",
    "assignee", "blocked_by", "tags", "notes", "parent_id", "origin",
    "checklist", "affected_sids", "impact", "kevin_final",
    "standing_approval",
}

# Blast-radius chip values (board #136) — see migrations/dev_tasks_lifecycle.sql.
_IMPACTS = {"contained", "app", "engine", "live"}

# Kanban pipeline stages that are NOT dispatchable work (board #136):
# Approval = awaiting Kevin's stamp; Review/Staged = the work already ran.
# The organic dispatcher queue is status=Todo only; these guard the
# run-request button lane.
_UNDISPATCHABLE_STAGES = ("Approval", "Review", "Staged")

# 'discovered' marks rabbit-hole work found mid-task, parented under the
# vision item that spawned it (Team Board convention).
_ORIGINS = {"planned", "discovered", "kevin"}

# Board #109 (Registry Phase 2): the Run button is DECLARATIVE — this tag is
# the request; the LOCAL dispatcher polls for it and executes (Railway cannot
# reach Kevin's machine). Cleared by the dispatcher on claim.
RUN_REQUESTED_TAG = "run-requested"


def _validate_team_fields(c, row: dict, task_id: Optional[int] = None):
    """Enforce origin values and ONE nesting level (vision → subtask).

    Deeper trees are rejected loudly rather than silently flattened, per
    the no-silent-defaults convention.
    """
    if "origin" in row and row["origin"] not in _ORIGINS:
        raise HTTPException(
            status_code=400,
            detail=f"origin must be one of {sorted(_ORIGINS)}")
    if "impact" in row and row["impact"] not in _IMPACTS:
        raise HTTPException(
            status_code=400,
            detail=f"impact must be one of {sorted(_IMPACTS)}")
    for f in ("kevin_final", "standing_approval"):
        if f in row and not isinstance(row[f], bool):
            raise HTTPException(
                status_code=400, detail=f"{f} must be a boolean")
    if "affected_sids" in row:
        sids = row["affected_sids"]
        if not (isinstance(sids, list)
                and all(isinstance(x, int) for x in sids)):
            raise HTTPException(
                status_code=400,
                detail="affected_sids must be a list of integers")
    if "checklist" in row:
        cl = row["checklist"]
        ok = isinstance(cl, list) and all(
            isinstance(s, dict)
            and isinstance(s.get("text"), str)
            and isinstance(s.get("done"), bool)
            and (s.get("role") is None or isinstance(s.get("role"), str))
            for s in cl)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail='checklist must be a list of '
                       '{"text": str, "done": bool, "role": str|null} '
                       '(send the WHOLE array — JSONB is replaced, not merged)')
    parent_id = row.get("parent_id")
    if parent_id is None:
        return
    if task_id is not None and parent_id == task_id:
        raise HTTPException(
            status_code=400, detail="a task cannot be its own parent")
    parent = c.table("dev_tasks").select("id,parent_id") \
        .eq("id", parent_id).execute().data
    if not parent:
        raise HTTPException(
            status_code=400,
            detail=f"parent task #{parent_id} does not exist")
    if parent[0].get("parent_id") is not None:
        raise HTTPException(
            status_code=400,
            detail=f"task #{parent_id} is itself a subtask — only one "
                   "nesting level (vision → subtask) is allowed")
    if task_id is not None:
        children = c.table("dev_tasks").select("id") \
            .eq("parent_id", task_id).limit(1).execute().data
        if children:
            raise HTTPException(
                status_code=400,
                detail=f"task #{task_id} has subtasks of its own — nesting "
                       "it under a parent would create two levels")


def _admin():
    from db import get_admin_client
    return get_admin_client()


@router.get("")
def list_tasks(include_done: bool = True, assignee: Optional[str] = None,
               user=Depends(get_current_user)):
    """All tasks, ordered by priority (phase, seq) ascending then created_at.

    `assignee` filters to one role — what role sessions poll at session start
    (`?assignee=E&include_done=false`).
    """
    c = _admin()
    q = c.table("dev_tasks").select("*")
    if not include_done:
        q = q.neq("status", "Done")
    if assignee:
        q = q.eq("assignee", assignee)
    rows = q.order("priority_phase").order("priority_seq") \
        .order("created_at").limit(1000).execute().data or []
    return rows


@router.post("")
def create_task(payload: dict = Body(...), user=Depends(get_current_user)):
    """Create a task. Requires `title`; everything else has DB defaults."""
    if not (payload.get("title") or "").strip():
        raise HTTPException(status_code=400, detail="title is required")
    row = {k: v for k, v in payload.items() if k in _EDITABLE}
    c = _admin()
    _validate_team_fields(c, row)
    res = c.table("dev_tasks").insert(row).execute()
    return res.data[0] if res.data else {"status": "created"}


@router.patch("/{task_id}")
def update_task(task_id: int, payload: dict = Body(...),
                user=Depends(get_current_user)):
    """Partial update — only whitelisted fields. Empty patch is a no-op read.

    Status/assignee changes auto-log a system activity entry on the comment
    thread ("status: Todo → In Progress (by F)") so handoffs stay traceable.
    The optional `actor` field names who made the change; it is not a task
    column and never lands on the row.
    """
    row = {k: v for k, v in payload.items() if k in _EDITABLE}
    c = _admin()
    if row:
        _validate_team_fields(c, row, task_id=task_id)
        tracked = {k: row[k] for k in ("status", "assignee") if k in row}
        prev = None
        if tracked:
            prev_rows = c.table("dev_tasks") \
                .select("status,assignee,kevin_final") \
                .eq("id", task_id).execute().data
            if not prev_rows:
                raise HTTPException(status_code=404, detail="task not found")
            prev = prev_rows[0]
        # Two-touch close guard (board #136): once Kevin stamps "Approve + I
        # review before Done", only Kevin's hand moves the task to Staged or
        # Done. Staged → Done is exempt — that transition is the R release
        # train shipping a task Kevin already signed off Review → Staged.
        if (prev is not None and prev.get("kevin_final")
                and row.get("status") in ("Staged", "Done")
                and prev.get("status") not in ("Staged", "Done")
                and (payload.get("actor") or "") != "kevin"):
            raise HTTPException(
                status_code=403,
                detail="two-touch task — Kevin stamped 'I review before "
                       "Done', so only actor=kevin moves it to Staged/Done "
                       "(relaying his verbal OK? send actor='kevin' and say "
                       "so on the thread)")
        c.table("dev_tasks").update(row).eq("id", task_id).execute()
        if prev:
            actor = payload.get("actor")
            by = f" (by {actor})" if actor else ""
            for field, new in tracked.items():
                old = prev.get(field)
                if (old or None) != (new or None):
                    c.table("dev_task_comments").insert({
                        "task_id": task_id, "author": "system",
                        "body": f"{field}: {old or '—'} → {new or '—'}{by}",
                    }).execute()
    res = c.table("dev_tasks").select("*").eq("id", task_id).execute().data
    if not res:
        raise HTTPException(status_code=404, detail="task not found")
    return res[0]


@router.delete("/{task_id}")
def delete_task(task_id: int, user=Depends(get_current_user)):
    _admin().table("dev_tasks").delete().eq("id", task_id).execute()
    return {"status": "deleted", "id": task_id}


@router.post("/{task_id}/run-request")
def request_run(task_id: int, payload: dict = Body(default={}),
                user=Depends(get_current_user)):
    """Request a dispatcher run of this task (board #109, Registry Phase 2).

    Declarative: adds the 'run-requested' tag + a system comment + a
    run_history row (outcome='requested'). The local dispatcher's --loop
    treats the tag as a priority-jump, re-gates eligibility at claim time,
    and owns the rest of the run lifecycle. 409 if a run is already
    requested or the task is already In Progress.
    """
    author = (payload.get("author") or "kevin").strip() or "kevin"
    c = _admin()
    rows = c.table("dev_tasks").select("*").eq("id", task_id).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="task not found")
    t = rows[0]
    tags = t.get("tags") or []
    if RUN_REQUESTED_TAG in tags:
        raise HTTPException(status_code=409, detail="run already requested")
    if t.get("status") == "In Progress":
        raise HTTPException(
            status_code=409, detail="task is already In Progress")
    if t.get("status") == "Done":
        raise HTTPException(status_code=400, detail="task is Done")
    # The button overrides queue ORDER, never "not workable yet" (M, #109
    # review) — mirror of the dispatcher's run_requested() refusals.
    if t.get("status") == "Scoping":
        raise HTTPException(
            status_code=400, detail="task is Scoping — not workable yet")
    # Board #136 pipeline stages: Approval isn't approved yet; Review/Staged
    # already ran. Same "never overrides 'not workable'" rule as Scoping.
    if t.get("status") in _UNDISPATCHABLE_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"task is {t['status']} — pipeline stage is not "
                   "dispatchable")
    if "needs-scoping" in tags:
        raise HTTPException(
            status_code=400,
            detail="tagged needs-scoping — not workable yet")
    if not (t.get("description") or "").strip():
        raise HTTPException(
            status_code=400,
            detail="task has no description — never dispatch unscoped work")
    if not (t.get("assignee") or "").strip():
        raise HTTPException(status_code=400, detail="task has no assignee")
    c.table("dev_tasks").update(
        {"tags": tags + [RUN_REQUESTED_TAG]}).eq("id", task_id).execute()
    c.table("dev_task_comments").insert({
        "task_id": task_id, "author": "system",
        "body": f"run requested by {author}"}).execute()
    res = c.table("run_history").insert({
        "task_id": task_id, "agent_letter": t["assignee"],
        "requested_by": author, "outcome": "requested"}).execute()
    return res.data[0] if res.data else {"status": "requested"}


# The two Approval-stage stamp buttons (board #136, Kevin+M 07-25).
# 'delegate' = "Approve — M closes" · 'final' = "Approve + I review before
# Done" (two-touch: kevin_final=TRUE, only Kevin signs off Review →
# Staged/Done). Both send the task to Todo and log ONE system comment.
_STAMP_MODES = {"delegate": False, "final": True}


@router.post("/{task_id}/stamp")
def stamp_approval(task_id: int, payload: dict = Body(...),
                   user=Depends(get_current_user)):
    """Record Kevin's approval stamp on a task sitting in Approval.

    Atomic replacement for the hand flow M ran on board #136 itself: sets
    kevin_final per the stamp mode, moves Approval → Todo, and writes one
    system comment naming the stamp and who recorded it (M relaying Kevin's
    verbal OK passes author='kevin' and says so on the thread).
    """
    mode = payload.get("mode")
    if mode not in _STAMP_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {sorted(_STAMP_MODES)}")
    author = (payload.get("author") or "kevin").strip() or "kevin"
    c = _admin()
    rows = c.table("dev_tasks").select("id,status").eq("id", task_id) \
        .execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="task not found")
    if rows[0].get("status") != "Approval":
        raise HTTPException(
            status_code=409,
            detail=f"task is {rows[0].get('status')} — stamps only apply "
                   "to tasks in Approval")
    kevin_final = _STAMP_MODES[mode]
    c.table("dev_tasks").update(
        {"status": "Todo", "kevin_final": kevin_final}) \
        .eq("id", task_id).execute()
    label = ("two-touch (Kevin reviews before Done)" if kevin_final
             else "M closes")
    c.table("dev_task_comments").insert({
        "task_id": task_id, "author": "system",
        "body": f"stamp: approved — {label} (by {author}) · "
                "status: Approval → Todo"}).execute()
    res = c.table("dev_tasks").select("*").eq("id", task_id).execute().data
    return res[0] if res else {"status": "stamped"}


@router.get("/{task_id}/comments")
def list_comments(task_id: int, user=Depends(get_current_user)):
    return _admin().table("dev_task_comments").select("*") \
        .eq("task_id", task_id).order("created_at").execute().data or []


@router.post("/{task_id}/comments")
def add_comment(task_id: int, payload: dict = Body(...),
                user=Depends(get_current_user)):
    body = (payload.get("body") or "").strip()
    if not body:
        raise HTTPException(status_code=400, detail="body is required")
    author = payload.get("author") or "claude"
    res = _admin().table("dev_task_comments").insert(
        {"task_id": task_id, "author": author, "body": body}).execute()
    return res.data[0] if res.data else {"status": "added"}
