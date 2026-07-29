"""Dev Task Tracker API — a ClickUp-lite shared board for Kevin + Claude.

Admin-gated CRUD over `dev_tasks` + `dev_task_comments` (see
migrations/dev_tasks_table.sql). Uses the service-role admin client. Tasks sort
by priority (phase, seq) ascending so 1.1 = "do next" surfaces first.
"""
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

router = APIRouter(prefix="/api/dev-tasks", tags=["dev-tasks"])

log = logging.getLogger(__name__)

# @-mention token: '@' followed by a letter then word chars. The negative
# lookbehind (?<![\w@]) requires the '@' to start a token — so an email local
# part like `ktkventures@gmail` (a word char precedes the '@') never matches,
# and `@@x` doesn't either. Board #143.
_MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9]*)")


def _parse_mentions(body: str, valid_by_lower: dict) -> list:
    """Distinct canonical mentions in `body`, in first-seen order.

    `valid_by_lower` maps lower(letter) → canonical registry letter, so
    `@f`/`@F` both resolve to 'F' and unknown tokens (`@gmail`, `@someone`)
    are dropped. Case-insensitive per the spec.
    """
    out, seen = [], set()
    for m in _MENTION_RE.finditer(body or ""):
        canon = valid_by_lower.get(m.group(1).lower())
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _write_mentions(c, comment_id: int, task_id: int, author: str, body: str):
    """Fan a posted comment's @tokens out into task_mentions rows (board #143).

    Best-effort by design: mentions are additive metadata, so a parse /
    registry / missing-table error (e.g. the API shipping before the migration
    lands) must NEVER break comment posting. The UNIQUE (comment_id, mentioned)
    constraint makes a re-POST a no-op. Only real registry letters + kevin
    become rows — unknown tokens are silently ignored (that's the spec, not a
    hidden default: they simply aren't mentions).
    """
    if "@" not in (body or ""):
        return  # common case — skip the registry round-trip entirely
    try:
        letters = c.table("agents").select("letter").execute().data or []
        valid = {r["letter"].lower(): r["letter"]
                 for r in letters if r.get("letter")}
        valid.setdefault("kevin", "kevin")  # guard if the human row is absent
        mentions = _parse_mentions(body, valid)
        if not mentions:
            return
        c.table("task_mentions").insert([
            {"comment_id": comment_id, "task_id": task_id,
             "mentioned": mtn, "author": author}
            for mtn in mentions
        ]).execute()
    except Exception as e:  # noqa: BLE001 — never let mentions break a comment
        log.warning("mention write failed for comment %s on task %s: %s",
                    comment_id, task_id, e)

# Fields a caller may set on create / update (whitelist — ignore anything else).
_EDITABLE = {
    "title", "description", "status", "priority_phase", "priority_seq",
    "is_urgent", "impacts_live", "needs_live_validation", "area",
    "assignee", "blocked_by", "tags", "notes", "parent_id", "origin",
    "checklist", "affected_sids", "impact", "kevin_final",
    "standing_approval", "ai_eligible",
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

# ── Process chain (board #182) ───────────────────────────────────────────────
# The checklist widens from legacy {role,text,done} into an ordered PROCESS
# CHAIN of richer step objects (see migrations/dev_tasks_process_chain.sql).
# A checklist is treated as a chain — with strict-order completion, per-step
# stamps and auto hand-off — only once a step carries one of the new keys.
# Legacy {role,text,done}-only checklists keep their exact pre-#182 behavior so
# the 42 existing tasks are untouched (retrofit is Step 9, deferred).
_STEP_MODES = {"execute", "discuss"}
_STEP_ORIGINS = {"planned", "audible"}
_STAMP_STATES = {"pending", "approved", "rejected"}
_NEW_SHAPE_KEYS = ("id", "owner", "title", "body", "mode", "origin", "stamp")
# Completion state is server-owned (managed by the /steps/* action endpoints);
# a generic PATCH may edit structure but never these fields on an existing step.
_STEP_COMPLETION_FIELDS = ("done", "completed_at", "completed_by")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_process_chain(checklist) -> bool:
    """True once ANY step carries a new-shape key — the opt-in signal that a
    checklist is a #182 process chain rather than a legacy checklist."""
    return isinstance(checklist, list) and any(
        isinstance(s, dict) and any(k in s for k in _NEW_SHAPE_KEYS)
        for s in checklist)


def _step_owner(step: dict):
    """Who acts on a step — new `owner`, falling back to legacy `role`."""
    owner = step.get("owner")
    return owner if owner is not None else step.get("role")


def _step_title(step: dict) -> str:
    """One line of substance — new `title`, falling back to legacy `text`."""
    return step.get("title") or step.get("text") or ""


def _current_step_index(checklist) -> Optional[int]:
    """Index of the first incomplete step (the only completable one — T3),
    or None when the whole chain is done."""
    for i, s in enumerate(checklist or []):
        if isinstance(s, dict) and not s.get("done"):
            return i
    return None


def _next_assignee(checklist, current):
    """T1/T2: owner of the first incomplete step; 'M' when the chain is
    complete (never kevin — closing is M's follow-through, per
    feedback_kevin_reviews_m_closes). An empty owner leaves assignee as-is."""
    idx = _current_step_index(checklist)
    if idx is None:
        return "M"
    owner = _step_owner(checklist[idx])
    return owner if owner else current


def _derive_kevin_final(checklist, existing) -> bool:
    """kevin_final is derived (spec #182): TRUE if any kevin-owned step carries
    a REQUIRED stamp. It NEVER clears an existing TRUE — the #136 two-touch
    guard set task-level by /stamp must survive on legacy tasks that have no
    per-step stamps (clearing it would silently drop a Review gate)."""
    if existing:
        return True
    for s in (checklist or []):
        if not isinstance(s, dict):
            continue
        stamp = s.get("stamp") or {}
        if (str(_step_owner(s) or "").strip().lower() == "kevin"
                and stamp.get("required")):
            return True
    return False


def _ensure_step_ids(checklist):
    """Assign a stable id to any chain step that lacks one, so reorder/insert
    keeps track of which steps are already complete. Legacy steps are only
    touched once the checklist has opted into the chain shape."""
    for s in (checklist or []):
        if isinstance(s, dict) and not s.get("id"):
            s["id"] = uuid.uuid4().hex[:12]
    return checklist


def _prepare_checklist_patch(c, task_id: int, row: dict):
    """Guard + normalize a PATCH that changes `checklist` (board #182).

    Legacy checklists pass straight through (pre-#182 free-toggle behavior).
    For a process chain the generic PATCH is STRUCTURE ONLY: it may add, edit,
    reorder or delete not-yet-done steps, but completion state (done /
    completed_* / stamp.state) is owned by the /steps/* action endpoints and
    cannot be changed here (T3 strict order + T4' immutability are API
    refusals, not UI niceties). Assignee is deliberately NOT recomputed on a
    structural edit — a manual override (T6) must survive an SOP edit, and the
    next /steps/complete recomputes it. kevin_final is re-derived."""
    incoming = row["checklist"]
    if not _is_process_chain(incoming):
        return  # legacy checklist — untouched, exactly as before #182
    stored_rows = c.table("dev_tasks").select("checklist,kevin_final") \
        .eq("id", task_id).execute().data
    stored = (stored_rows[0].get("checklist") if stored_rows else None) or []
    stored_by_id = {s["id"]: s for s in stored
                    if isinstance(s, dict) and s.get("id")}
    incoming_ids = set()
    for s in incoming:
        sid = s.get("id")
        prev = stored_by_id.get(sid) if sid else None
        if prev is not None:
            incoming_ids.add(sid)
            if bool(s.get("done")) != bool(prev.get("done")) or any(
                    s.get(f) != prev.get(f)
                    for f in _STEP_COMPLETION_FIELDS if f != "done"):
                raise HTTPException(
                    status_code=409,
                    detail="completion is managed by /steps/complete — a PATCH "
                           "cannot tick or untick a step (course-correct by "
                           "inserting an origin='audible' step instead)")
            if ((s.get("stamp") or {}).get("state")
                    != (prev.get("stamp") or {}).get("state")):
                raise HTTPException(
                    status_code=409,
                    detail="stamp state is managed by /steps/stamp — a PATCH "
                           "cannot approve or reject a step")
        elif s.get("done"):
            raise HTTPException(
                status_code=400,
                detail="a newly inserted step cannot start completed "
                       "(done must be false)")
    for sid, prev in stored_by_id.items():
        if sid not in incoming_ids and prev.get("done"):
            raise HTTPException(
                status_code=409,
                detail="cannot delete a completed step — it is part of the "
                       "audit trail (course-correct by inserting an "
                       "origin='audible' revision step, T4')")
    _ensure_step_ids(incoming)
    row["kevin_final"] = _derive_kevin_final(
        incoming, stored_rows[0].get("kevin_final") if stored_rows else False)


def _task_row(c, task_id: int) -> dict:
    res = c.table("dev_tasks").select("*").eq("id", task_id).execute().data
    if not res:
        raise HTTPException(status_code=404, detail="task not found")
    return res[0]


def _sys_comment(c, task_id: int, body: str, step_order: Optional[int] = None):
    c.table("dev_task_comments").insert({
        "task_id": task_id, "author": "system", "body": body,
        "step_order": step_order}).execute()


def _validate_step_shape(s) -> bool:
    """Accept both legacy {role,text,done} and #182 chain steps. Every new
    field is optional; a step just needs a string label (title or text)."""
    if not isinstance(s, dict):
        return False
    if not isinstance(s.get("title", s.get("text")), str):
        return False
    if "done" in s and not isinstance(s["done"], bool):
        return False
    owner = s.get("owner", s.get("role"))
    if owner is not None and not isinstance(owner, str):
        return False
    if "body" in s and not isinstance(s["body"], str):
        return False
    if "mode" in s and s["mode"] not in _STEP_MODES:
        return False
    if "origin" in s and s["origin"] not in _STEP_ORIGINS:
        return False
    stamp = s.get("stamp")
    if stamp is not None:
        if not isinstance(stamp, dict):
            return False
        if "required" in stamp and not isinstance(stamp["required"], bool):
            return False
        if stamp.get("state") is not None and stamp["state"] not in _STAMP_STATES:
            return False
    return True


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
    for f in ("kevin_final", "standing_approval", "ai_eligible"):
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
        ok = isinstance(cl, list) and all(_validate_step_shape(s) for s in cl)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail='checklist must be a list of process-chain steps — at '
                       'minimum a string title (or legacy "text"); optional '
                       'owner/body, mode (execute|discuss), origin '
                       '(planned|audible), stamp, done:bool '
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
    # Give a process chain stable step ids + derived kevin_final at birth
    # (board #182); legacy checklists are left untouched.
    if _is_process_chain(row.get("checklist")):
        _ensure_step_ids(row["checklist"])
        row["kevin_final"] = _derive_kevin_final(
            row["checklist"], row.get("kevin_final"))
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
        # Process-chain PATCH (board #182): keep completion server-owned and
        # re-derive kevin_final. Legacy checklists pass through untouched.
        if "checklist" in row:
            _prepare_checklist_patch(c, task_id, row)
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
    kevin_final per the stamp mode, arms `ai_eligible` (board #198), moves
    Approval → Todo, and writes one system comment naming the stamp and who
    recorded it (M relaying Kevin's verbal OK passes author='kevin' and says so
    on the thread).
    """
    mode = payload.get("mode")
    if mode not in _STAMP_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {sorted(_STAMP_MODES)}")
    author = (payload.get("author") or "kevin").strip() or "kevin"
    c = _admin()
    rows = c.table("dev_tasks").select("id,status,checklist").eq("id", task_id) \
        .execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="task not found")
    if rows[0].get("status") != "Approval":
        raise HTTPException(
            status_code=409,
            detail=f"task is {rows[0].get('status')} — stamps only apply "
                   "to tasks in Approval")
    kevin_final = _STAMP_MODES[mode]
    # The stamp ARMS the task for AI work (board #198). Kevin's review stays the
    # arming gate — nothing runs unapproved — but the guarantee now rides on
    # `ai_eligible` instead of on the task sitting in `Todo`, so the chain can
    # re-arm itself between steps without shoving the card back across the
    # board. Both stamp modes arm: 'final' vs 'delegate' decides who CLOSES the
    # task, not whether an agent may work it.
    update = {"status": "Todo", "kevin_final": kevin_final,
              "ai_eligible": True}

    # The stamp IS the completion of the first pending Kevin checklist step
    # (board #151). Without ticking it, next_actor() keeps returning 'kevin'
    # after the task lands in Todo; the dispatcher's eligibility gate
    # (next_actor == assignee) then skips it, and an approved task sits in
    # Todo forever (dispatchable=0) — looking approved but going nowhere.
    # Tick ONLY the first un-done kevin step: a later two-touch review step
    # (kevin_final) must stay open so its Review sign-off is still gated.
    checklist = rows[0].get("checklist") or []
    ticked = None
    for step in checklist:
        if (not step.get("done")
                and (step.get("role") or "").strip().lower() == "kevin"):
            step["done"] = True
            ticked = step.get("text")
            update["checklist"] = checklist  # JSONB is replaced whole
            break

    c.table("dev_tasks").update(update).eq("id", task_id).execute()
    label = ("two-touch (Kevin reviews before Done)" if kevin_final
             else "M closes")
    tick_note = f' · checklist: ticked "{ticked}"' if ticked else ""
    c.table("dev_task_comments").insert({
        "task_id": task_id, "author": "system",
        "body": f"stamp: approved — {label} (by {author}) · "
                f"status: Approval → Todo · ai_eligible: true"
                f"{tick_note}"}).execute()
    res = c.table("dev_tasks").select("*").eq("id", task_id).execute().data
    return res[0] if res else {"status": "stamped"}


# ── Process-chain step actions (board #182) ──────────────────────────────────
# The server side of the three step buttons (Step 5 UI). Completion state and
# hand-off live here — NOT in the generic checklist PATCH — so the transition
# rules (T1/T2/T3/T4'/T8/T9) are API refusals, not UI conventions.

@router.post("/{task_id}/steps/complete")
def complete_step(task_id: int, payload: dict = Body(default={}),
                  user=Depends(get_current_user)):
    """Complete the current step and hand off (T1/T2/T3/T8).

    Ticks the FIRST incomplete step only — you cannot complete a later step out
    of order (T3). A step that gates on a stamp cannot be completed until the
    stamp is approved (T8). Records completed_at/by and reassigns the task to
    the next incomplete step's owner — 'M' when the chain finishes, never kevin
    (T2). One action, one write.
    """
    actor = (payload.get("actor") or "").strip() or "system"
    c = _admin()
    task = _task_row(c, task_id)
    cl = task.get("checklist") or []
    idx = _current_step_index(cl)
    if idx is None:
        raise HTTPException(
            status_code=409, detail="every step is already complete")
    step = cl[idx]
    stamp = step.get("stamp") or {}
    if stamp.get("required") and stamp.get("state") != "approved":
        raise HTTPException(
            status_code=409,
            detail=f'step "{_step_title(step)}" needs an approved stamp before '
                   "it can be completed (T8) — use /steps/stamp")
    step["done"] = True
    step["completed_at"] = _now_iso()
    step["completed_by"] = actor
    _ensure_step_ids(cl)
    old_assignee = task.get("assignee")
    new_assignee = _next_assignee(cl, old_assignee)
    kevin_final = _derive_kevin_final(cl, task.get("kevin_final"))
    c.table("dev_tasks").update(
        {"checklist": cl, "assignee": new_assignee,
         "kevin_final": kevin_final}).eq("id", task_id).execute()
    if _current_step_index(cl) is None:
        tail = f"chain complete → assignee {new_assignee} (close)"
    elif (old_assignee or None) != (new_assignee or None):
        tail = f"handed off: {old_assignee or '—'} → {new_assignee}"
    else:
        tail = f"still on {new_assignee}"
    _sys_comment(
        c, task_id,
        f'step {idx + 1} "{_step_title(step)}" complete (by {actor}) · {tail}',
        step_order=idx)
    return _task_row(c, task_id)


@router.post("/{task_id}/steps/raise-issue")
def raise_issue(task_id: int, payload: dict = Body(...),
                user=Depends(get_current_user)):
    """Raise an issue on the current step and route to M (T9).

    Requires a comment explaining the problem, posts it (tagged to the current
    step), and reassigns the task to M — the router — WITHOUT ticking any step
    or altering the chain. M then decides whether to rewrite the summary,
    insert an origin='audible' revision step, reorder, or escalate to Kevin.
    """
    body = (payload.get("body") or "").strip()
    if not body:
        raise HTTPException(
            status_code=400,
            detail="raising an issue requires a comment explaining it")
    actor = (payload.get("actor") or "").strip() or "system"
    c = _admin()
    task = _task_row(c, task_id)
    idx = _current_step_index(task.get("checklist") or [])
    res = c.table("dev_task_comments").insert(
        {"task_id": task_id, "author": actor, "body": body,
         "step_order": idx}).execute()
    crow = res.data[0] if res.data else None
    if crow:
        _write_mentions(c, crow["id"], task_id, actor, body)
    old = task.get("assignee")
    if (old or None) != "M":
        c.table("dev_tasks").update({"assignee": "M"}).eq("id", task_id).execute()
        _sys_comment(c, task_id,
                     f"⚠️ issue raised (by {actor}) · assignee: "
                     f"{old or '—'} → M", step_order=idx)
    else:
        _sys_comment(c, task_id,
                     f"⚠️ issue raised (by {actor}) · already with M",
                     step_order=idx)
    return _task_row(c, task_id)


@router.post("/{task_id}/steps/stamp")
def stamp_step(task_id: int, payload: dict = Body(...),
               user=Depends(get_current_user)):
    """Approve or reject the current step's stamp (T8/T9).

    Approve unblocks completion (the owner then hits /steps/complete). Reject
    records the rejection and routes to M — the escalation path — without
    ticking the step. Only the current step, and only one that gates on a stamp.
    """
    decision = payload.get("decision")
    if decision not in ("approve", "reject"):
        raise HTTPException(
            status_code=400, detail="decision must be 'approve' or 'reject'")
    actor = (payload.get("actor") or "").strip() or "system"
    c = _admin()
    task = _task_row(c, task_id)
    cl = task.get("checklist") or []
    idx = _current_step_index(cl)
    if idx is None:
        raise HTTPException(
            status_code=409, detail="the chain is complete — no step to stamp")
    step = cl[idx]
    stamp = step.get("stamp") or {}
    if not stamp.get("required"):
        raise HTTPException(
            status_code=409,
            detail=f'step "{_step_title(step)}" does not gate on a stamp')
    stamp["state"] = "approved" if decision == "approve" else "rejected"
    stamp["by"] = actor
    stamp["at"] = _now_iso()
    step["stamp"] = stamp
    _ensure_step_ids(cl)
    update = {"checklist": cl}
    old = task.get("assignee")
    if decision == "reject" and (old or None) != "M":
        update["assignee"] = "M"  # T9 escalation
    c.table("dev_tasks").update(update).eq("id", task_id).execute()
    if decision == "approve":
        _sys_comment(
            c, task_id,
            f'stamp approved on step {idx + 1} "{_step_title(step)}" '
            f"(by {actor}) — owner may now complete it", step_order=idx)
    else:
        tail = "" if (old or None) == "M" else f" · assignee: {old or '—'} → M"
        _sys_comment(
            c, task_id,
            f'stamp REJECTED on step {idx + 1} "{_step_title(step)}" '
            f"(by {actor}) — routed to M{tail}", step_order=idx)
    return _task_row(c, task_id)


@router.get("/{task_id}/poll")
def poll_task(task_id: int, scoped_id: Optional[int] = None,
              user=Depends(get_current_user)):
    """One round-trip for the open-modal liveness poll (board #148).

    Replaces the 3-4 separate calls the modal fired every tick (comments +
    runs(scoped) + runs(task) + a full board refetch). Returns the scoped
    thread and its run rows, the header task's run rows when a subtask is
    scoped, and a cheap board-change probe: the single most-recent
    dev_tasks.updated_at (kept fresh by the trg_dev_tasks_touch trigger). The
    caller compares that probe to its last-seen value and refetches the whole
    120-row board ONLY when it moved, instead of every tick.

    All reads use list semantics (`.data or []`) — never `.single()` — so a
    task with zero comments or run rows returns `[]`, not a PGRST116/406 that
    Supabase would log as an error.
    """
    c = _admin()
    sid = scoped_id if scoped_id is not None else task_id
    comments = c.table("dev_task_comments").select("*") \
        .eq("task_id", sid).order("created_at").execute().data or []
    runs = c.table("run_history").select("*").eq("task_id", sid) \
        .order("requested_at", desc=True).limit(50).execute().data or []
    # The header RunButton scopes to the modal task; only fetch its runs
    # separately when a subtask is selected (otherwise `runs` already covers it).
    header_runs = None
    if sid != task_id:
        header_runs = c.table("run_history").select("*") \
            .eq("task_id", task_id).order("requested_at", desc=True) \
            .limit(50).execute().data or []
    probe = c.table("dev_tasks").select("updated_at") \
        .order("updated_at", desc=True).limit(1).execute().data
    return {
        "comments": comments,
        "runs": runs,
        "header_runs": header_runs,
        "board_max_updated_at": (probe[0]["updated_at"] if probe else None),
    }


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
    c = _admin()
    # Tag the comment with the task's current step (board #182, decision 11).
    # Unrecoverable if not captured at write time; best-effort so a read hiccup
    # never blocks a comment. NULL for legacy tasks and finished chains.
    step_order = None
    try:
        crow = c.table("dev_tasks").select("checklist") \
            .eq("id", task_id).execute().data
        cl = (crow[0].get("checklist") if crow else None) or []
        if _is_process_chain(cl):
            step_order = _current_step_index(cl)
    except Exception as e:  # noqa: BLE001 — never let step tagging break a comment
        log.warning("step_order stamp failed for task %s: %s", task_id, e)
    res = c.table("dev_task_comments").insert(
        {"task_id": task_id, "author": author, "body": body,
         "step_order": step_order}).execute()
    row = res.data[0] if res.data else None
    # Fan @tokens out to task_mentions so nothing rots un-circled-back-to
    # (board #143). Best-effort — never breaks the comment.
    if row:
        _write_mentions(c, row["id"], task_id, author, body)
    return row or {"status": "added"}
