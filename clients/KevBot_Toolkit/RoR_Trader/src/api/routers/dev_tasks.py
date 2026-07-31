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

from api.deps import get_current_user, get_service_or_user

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
    "standing_approval", "ai_eligible", "task_type",
    # Board #262 — the goal type's parameters. Shipped as a column by #261's
    # migration and read by nothing until now. WHOLE-OBJECT WRITE: `goal_params`
    # is JSONB, and a partial dict REPLACES the value rather than merging it
    # (memory: feedback_jsonb_partial_updates), so the UI always sends the full
    # object. `autonomy` is deliberately NOT one of its keys — it lives on the
    # `standing_approval` column.
    "goal_params",
}

# ── THE TASK-TYPE MODEL (board #261) ────────────────────────────────────────
# The server half of the ONE type map. Its mirror is `TASK_TYPES` in
# frontend/src/views/taskBoardShared.tsx, and the two are asserted equal BY
# PARSED VALUE (the TS map is EXECUTED, not string-matched) in
# src/test_task_type_model_261.py — boards #219/#245: a constant shipped without
# its frontend mirror put dev red for hours, and a quote-naive string-compare
# "mirror" read two identical lists as different.
#
# `statuses` here is DECLARATIVE, NOT A GATE. Nothing in this module rejects an
# out-of-set status and nothing may start to in #261: 236 rows predate this map
# and some hold a status no type lists — which is exactly why the UI keeps
# `withLegacy`. Enforcement is a later task, to be taken deliberately.
#
# The vocabulary mirrors the DB CHECK constraint in
# migrations/dev_tasks_task_type.sql. `loop` is deliberately absent — Kevin
# raised it as a possible later type and excluded it from this plan.
TASK_TYPES = {
    "action": {
        "label": "Action",
        "icon": "",
        "def": "action — a subtask or loose task; where the work actually happens",
        "statuses": ["Backlog", "Scoping", "Approval", "Todo", "In Progress",
                     "Review", "Staged", "Blocked", "Stand By", "Done", "Closed"],
        "children": False,
        "session": False,
    },
    "vision": {
        "label": "Vision",
        "icon": "◈",
        "def": "vision — the default parent container; tracks via its subtasks",
        # The pre-#261 VISION_STATUSES, unchanged: no Approval/Review/Staged,
        # because a container has no branch to review or to stage.
        "statuses": ["Backlog", "Scoping", "Todo", "In Progress", "Blocked",
                     "Stand By", "Done", "Closed"],
        "children": True,
        "session": False,
    },
    "goal": {
        "label": "Goal",
        "icon": "⚑",
        "def": "goal — a parent container that carries its own session",
        # Approval, because a goal is authorised to START; still no
        # Review/Staged, because a goal never ships — its children do.
        "statuses": ["Backlog", "Scoping", "Approval", "In Progress", "Blocked",
                     "Stand By", "Done", "Closed"],
        "children": True,
        "session": True,
    },
}

# The column default, and what the backfill left on every non-container row.
# Declared, never silently substituted: an ABSENT `task_type` on a read is
# resolved by the UI's `taskTypeOf`, which falls back to the pre-#261 derivation
# rather than assuming 'action' (no-silent-defaults).
DEFAULT_TASK_TYPE = "action"

# Blast-radius chip values (board #136) — see migrations/dev_tasks_lifecycle.sql.
_IMPACTS = {"contained", "app", "engine", "live"}

# Board #232 — THE ONE SHARED "FINISHED" SET for the API side. Mirrors
# dispatcher.FINISHED_STATUSES: `Done` is M's close, `Closed` is Kevin's
# retrospective look after it, and BOTH mean the task is over. Every "is this
# task finished?" read in this module goes through `_is_finished`, never a bare
# `== "Done"` — the failure #232 exists to kill is a finished-check that knows
# only one of the two spellings and silently changes behaviour when a task moves
# from one to the other.
# NOT the same question as "not dispatchable": see _UNDISPATCHABLE_STAGES.
FINISHED_STATUSES = ("Done", "Closed")


def _is_finished(status) -> bool:
    """True when the task is OVER — `Done` or `Closed` (board #232)."""
    return status in FINISHED_STATUSES


# Kanban pipeline stages that are NOT dispatchable work (board #136):
# Approval = awaiting Kevin's stamp; Review/Staged = the work already ran;
# Stand By = Kevin SAW it, wants it, and chose not yet (board #232) — the Run
# button overrides queue ORDER, never a human's explicit "not yet".
# The organic dispatcher queue is status=Todo only; these guard the
# run-request button lane. Mirrors dispatcher.run_requested()'s refusals.
_UNDISPATCHABLE_STAGES = ("Approval", "Review", "Staged", "Stand By")

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
_NEW_SHAPE_KEYS = ("id", "owner", "title", "body", "mode", "origin", "stamp",
                   "deliverables")
# Completion state is server-owned (managed by the /steps/* action endpoints);
# a generic PATCH may edit structure but never these fields on an existing step.
_STEP_COMPLETION_FIELDS = ("done", "completed_at", "completed_by")

# ── Step deliverables (board #184) ───────────────────────────────────────────
# A step may declare DELIVERABLES — the evidence it asks for, captured where it
# is asked for rather than in a comment the surface cannot see. Purely additive
# and feature-detected exactly like the #182 step object: a step with no
# `deliverables` key behaves byte-identically to today. See
# migrations/dev_tasks_step_deliverables.sql and docs/_active/Spec_Step_Deliverables.md.
#
# THE CENTRAL RULE (spec §2, mirroring _STEP_COMPLETION_FIELDS): a deliverable's
# SPEC (kind/label/required/hint) is author-owned and editable through the
# generic checklist PATCH; its FILLED STATE is server-owned and reachable only
# through the /steps/deliverables/* endpoints. A PATCH that touches a fill field
# is a 409, not a silently-accepted write.
_DELIVERABLE_FILL_FIELDS = ("value", "filename", "size_bytes", "content_type",
                            "filled_at", "filled_by")
_DELIVERABLE_SPEC_FIELDS = ("id", "kind", "label", "required", "hint")
# Three kinds, no more (spec §1). The moment this grows selects/dates/numbers it
# stops being a step and starts being a form builder.
_DELIVERABLE_KINDS = ("text", "link", "file")
_MAX_DELIVERABLES_PER_STEP = 8
_DELIVERABLE_LABEL_MAX = 120
_DELIVERABLE_TEXT_MAX = 4000
_DELIVERABLE_HINT_MAX = 200
_DELIVERABLE_FILE_MAX_BYTES = 10 * 1024 * 1024
# The upload gate is the FILE EXTENSION, not the client-declared content-type:
# the extension is a durable property of the stored object, while a
# Content-Type header is whatever the uploader claims. The declared type is
# still recorded (the read path needs it) after a shape check — recorded, not
# trusted. 415 names this list, so a refusal is readable.
_DELIVERABLE_EXTS = ("png", "jpg", "jpeg", "gif", "webp", "pdf", "csv", "txt",
                     "md", "json", "log", "zip")
# Private bucket (spec §4). NOT created by this code — creating a prod bucket is
# the same class of action as DDL and is Kevin's to authorize; the migration
# file records exactly what is needed. Upload/sign go through the SAME storage
# client `src/bar_cache_store.py` already uses (`get_admin_client().storage`),
# deliberately: one client path, not two.
DELIVERABLES_BUCKET = "task-deliverables"
_DELIVERABLE_URL_TTL_S = 3600   # §3.4 — minted on click, never on board load


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
    _ensure_deliverable_ids(checklist)
    return checklist


# ── Deliverable helpers (board #184) ─────────────────────────────────────────

def _deliverables(step) -> list:
    """A step's deliverable list — [] when the step never declared one.

    ABSENT and EMPTY are the same thing to every reader here, which is what
    makes back-compat hold: a pre-#184 step simply has nothing to ask for."""
    d = (step or {}).get("deliverables")
    return d if isinstance(d, list) else []


def _ensure_deliverable_ids(checklist):
    """Assign a stable id to any deliverable that lacks one — the exact
    `_ensure_step_ids` pattern, and for the same reason: the fill endpoints
    address a deliverable BY ID, so a reorder or an inserted row must not
    silently re-point a stored value at a different ask."""
    for s in (checklist or []):
        if not isinstance(s, dict):
            continue
        for d in _deliverables(s):
            if isinstance(d, dict) and not d.get("id"):
                d["id"] = uuid.uuid4().hex[:12]
    return checklist


def _is_filled(d) -> bool:
    return bool((d or {}).get("value"))


def _unfilled(step, required: bool) -> list:
    """Labels of this step's unfilled deliverables, required ones or optional
    ones. The two halves of T10: required-unfilled BLOCKS (§3.5), and
    optional-unfilled is RECORDED (never blocked, decision D2)."""
    return [str(d.get("label") or "(unlabelled)")
            for d in _deliverables(step)
            if isinstance(d, dict) and bool(d.get("required")) is required
            and not _is_filled(d)]


def _find_deliverable(checklist, deliverable_id: str):
    """(step_index, step, deliverable) for an id, or (None, None, None)."""
    for i, s in enumerate(checklist or []):
        if not isinstance(s, dict):
            continue
        for d in _deliverables(s):
            if isinstance(d, dict) and d.get("id") == deliverable_id:
                return i, s, d
    return None, None, None


def _validate_deliverable_specs(step):
    """Fail loud on a malformed / over-limit deliverable SPEC (§1 limits).

    Runs on the generic PATCH path, where specs are authored. Fill values are
    validated at their own endpoints, where the kind is already known.
    """
    ds = _deliverables(step)
    if len(ds) > _MAX_DELIVERABLES_PER_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"a step may declare at most {_MAX_DELIVERABLES_PER_STEP} "
                   f"deliverables (got {len(ds)}) — if a step needs more than "
                   "that, it is really two steps")
    for d in ds:
        if not isinstance(d, dict):
            raise HTTPException(
                status_code=400, detail="each deliverable must be an object")
        if d.get("kind") not in _DELIVERABLE_KINDS:
            raise HTTPException(
                status_code=400,
                detail=f"deliverable kind must be one of "
                       f"{sorted(_DELIVERABLE_KINDS)} (got {d.get('kind')!r})")
        label = d.get("label")
        if not isinstance(label, str) or not label.strip():
            raise HTTPException(
                status_code=400,
                detail="a deliverable needs a label — the ask, in one line")
        if len(label) > _DELIVERABLE_LABEL_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"deliverable label is {len(label)} chars — the limit is "
                       f"{_DELIVERABLE_LABEL_MAX} (put the detail in the hint)")
        if "required" in d and not isinstance(d["required"], bool):
            raise HTTPException(
                status_code=400, detail="deliverable.required must be a boolean")
        hint = d.get("hint")
        if hint is not None and (not isinstance(hint, str)
                                 or len(hint) > _DELIVERABLE_HINT_MAX):
            raise HTTPException(
                status_code=400,
                detail=f"deliverable hint must be a string of at most "
                       f"{_DELIVERABLE_HINT_MAX} chars")


def _guard_deliverable_patch(c, task_id: int, incoming_step, prev_step):
    """The §2 rule as an API refusal: a generic PATCH may edit deliverable
    SPECS on a not-yet-done step, and may NEVER touch a fill.

    Four refusals, each one a thing that would otherwise corrupt the record:
      • changing a fill field           → 409 (fills are endpoint-owned)
      • a NEW deliverable arriving filled → 400 (same shape as "an inserted
        step cannot start completed")
      • ANY change to a COMPLETED step's deliverables → 409 (T4' immutability)
      • deleting a FILLED deliverable on an open step → ALLOWED, because the
        author may have asked for the wrong thing while the step is still
        open — but it posts a system comment naming what was dropped, so the
        audit trail survives the edit.
    """
    prev_by_id = {d["id"]: d for d in _deliverables(prev_step)
                  if isinstance(d, dict) and d.get("id")}
    incoming = _deliverables(incoming_step)
    incoming_ids = set()
    if prev_step.get("done"):
        # A completed step is part of the audit trail — its asks and its
        # evidence are frozen together (T4').
        same = ([_deliverable_identity(d) for d in incoming]
                == [_deliverable_identity(d) for d in _deliverables(prev_step)])
        if not same:
            raise HTTPException(
                status_code=409,
                detail='a completed step\'s deliverables are immutable — they '
                       "are part of the audit trail (T4'; course-correct by "
                       "inserting an origin='audible' step instead)")
        return
    for d in incoming:
        if not isinstance(d, dict):
            continue
        did = d.get("id")
        prev = prev_by_id.get(did) if did else None
        if prev is not None:
            incoming_ids.add(did)
            if any(d.get(f) != prev.get(f) for f in _DELIVERABLE_FILL_FIELDS):
                raise HTTPException(
                    status_code=409,
                    detail="a deliverable's value is managed by "
                           "/steps/deliverables — a PATCH cannot fill or clear "
                           "one")
        elif any(d.get(f) is not None for f in _DELIVERABLE_FILL_FIELDS):
            raise HTTPException(
                status_code=400,
                detail="a newly inserted deliverable cannot start filled "
                       "(value must be absent)")
    for did, prev in prev_by_id.items():
        if did not in incoming_ids and _is_filled(prev):
            _sys_comment(
                c, task_id,
                f'⚠️ filled deliverable "{prev.get("label")}" was removed from '
                f'step "{_step_title(prev_step)}" — its recorded value was '
                f'`{str(prev.get("filename") or prev.get("value"))[:120]}`')


def _deliverable_identity(d):
    """Everything about a deliverable, for the completed-step equality check."""
    if not isinstance(d, dict):
        return d
    return tuple(sorted((k, v) for k, v in d.items()
                        if k in _DELIVERABLE_SPEC_FIELDS
                        or k in _DELIVERABLE_FILL_FIELDS))


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
        # Board #184 — spec limits on the deliverable SPECS this PATCH authors.
        # Runs for every step (new or existing); the fill/immutability guards
        # below need a stored counterpart and so only run for existing ones.
        _validate_deliverable_specs(s)
        sid = s.get("id")
        prev = stored_by_id.get(sid) if sid else None
        if prev is not None:
            incoming_ids.add(sid)
            _guard_deliverable_patch(c, task_id, s, prev)
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
        else:
            # Board #184 — the same rule one level down: a step arriving for the
            # first time may declare asks, never answers.
            for d in _deliverables(s):
                if isinstance(d, dict) and any(
                        d.get(f) is not None for f in _DELIVERABLE_FILL_FIELDS):
                    raise HTTPException(
                        status_code=400,
                        detail="a newly inserted deliverable cannot start "
                               "filled (value must be absent)")
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
    # Board #184 — shape only (a list of objects). The SPEC limits and the
    # fill-ownership rules are enforced in _prepare_checklist_patch, where the
    # STORED counterpart is available to compare against; a shape check has no
    # way to tell an edit from an insert.
    if "deliverables" in s:
        if not isinstance(s["deliverables"], list):
            return False
        if not all(isinstance(d, dict) for d in s["deliverables"]):
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
    # Board #261 — VOCABULARY only. This refuses a type that is not one of the
    # three (the same set the DB CHECK constraint pins), so a typo fails here
    # with a readable 400 instead of as a constraint violation. It emphatically
    # does NOT check the row's STATUS against that type's status list: legacy
    # rows would be rejected, which is the failure `withLegacy` exists for.
    if "task_type" in row and row["task_type"] not in TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"task_type must be one of {sorted(TASK_TYPES)}")
    # Board #262 — SHAPE only, matching the #261 fence above: an object or null.
    # Deliberately NOT a required-key check — `done_when` is enforced at RENDER
    # time (the context block refuses to generate without it), not here, so that
    # a half-authored goal can still be SAVED. That was M's step-1 decision and
    # it keeps #261's "no API-level enforcement in this arc" fence intact;
    # #168 owns any status/required enforcement.
    if "goal_params" in row and row["goal_params"] is not None:
        if not isinstance(row["goal_params"], dict):
            raise HTTPException(
                status_code=400,
                detail="goal_params must be an object (or null) — send the "
                       "WHOLE object; JSONB is replaced, not merged")
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
    # ── Board #262 — A GOAL DOES NOT NEST ──────────────────────────────────
    # A goal is a TOP-LEVEL container: it holds action tasks, and it never sits
    # inside a vision or inside another goal. Both of those are one rule — a
    # `goal` row may not carry a `parent_id` — so this is one check, not two.
    #
    # This is NESTING enforcement, the same kind the block below already does
    # (one level, no self-parent); it is NOT the status enforcement #261 fenced
    # off. Mirrored in the UI by `goalNestError`, which states the rule where
    # the nesting would be done; neither is the other's only line of defence.
    #
    # BOTH DIRECTIONS, because a patch can arrive as either half: setting a
    # parent on a row that is already a goal, or typing a parented row `goal`.
    # So the EFFECTIVE type is resolved — the patch's `task_type` if it carries
    # one, else the stored row's — and likewise the effective parent.
    if task_id is not None and ("task_type" in row or "parent_id" in row):
        stored = c.table("dev_tasks").select("task_type,parent_id") \
            .eq("id", task_id).execute().data
        stored = stored[0] if stored else {}
    else:
        stored = {}
    eff_type = row.get("task_type", stored.get("task_type", DEFAULT_TASK_TYPE))
    eff_parent = row.get("parent_id", stored.get("parent_id"))
    if eff_type == "goal" and eff_parent is not None:
        raise HTTPException(
            status_code=400,
            detail=f"a goal is a top-level container — it cannot be filed "
                   f"under #{eff_parent} (not under a vision, and not under "
                   "another goal). Clear parent_id, or make it a vision.")

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
        # Board #232 — FINISHED, not the literal `Done`. A Closed task is just as
        # over; leaving it in would put Kevin's own closed work back in every
        # role session's "open queue" the moment he started using the status.
        for s in FINISHED_STATUSES:
            q = q.neq("status", s)
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

    Closing a task DISARMS it (`ai_eligible=false`, board #198) — see below.
    """
    row = {k: v for k, v in payload.items() if k in _EDITABLE}
    # Board #198 audible — DISARM ON CLOSE. `ai_eligible` is the arming switch
    # ("may an AI run this"); once a task is Done the answer is permanently no,
    # so closing it clears the arm rather than leaving `Done` + armed rows on the
    # board. A contradictory patch ({"status": "Done", "ai_eligible": true})
    # resolves to DISARMED: closing wins.
    # HYGIENE, NOT THE GUARANTEE. This covers one endpoint, so a direct Supabase
    # write or some future close path can skip it; the dispatcher's own
    # TERMINAL_STATUSES rail is the one that cannot be forgotten. Both exist
    # deliberately — this one means the refusal never has to fire, and it keeps
    # the DATA honest for the board UI's toggle column.
    # Board #232: FINISHED, not the literal `Done` — `Closed` disarms too.
    if _is_finished(row.get("status")):
        row["ai_eligible"] = False
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
        #
        # LITERAL `Done` ON PURPOSE (board #232 audit) — NOT the FINISHED set.
        # This is a TRANSITION rule about specific lifecycle columns, not the
        # question "is this task over". `Closed` is Kevin's own status, reached
        # from `Done` by Kevin himself, so the guard has nothing to protect
        # there. #232 stopped SETTING kevin_final (the stamp is one mode now)
        # but deliberately left the column, its rows and this guard standing:
        # existing two-touch tasks must keep behaving exactly as stamped.
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
    # Board #232: FINISHED, not the literal `Done` — a Closed task cannot be
    # started either. Mirrors dispatcher.run_requested()'s is_finished() refusal.
    if _is_finished(t.get("status")):
        raise HTTPException(status_code=400,
                            detail=f"task is {t['status']}")
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


# ── THE ONE COMPLETION PATH (board #182 T-gates + board #184 §3.6) ───────────
# Defined here, ABOVE its first caller: both the task-level `/stamp` and
# `/steps/complete` tick a step, and two paths that tick a step will always
# drift — they already had (see the docstring).
def _complete_current_step(c, task, actor: str, extra_update: Optional[dict] = None,
                           post_comment: bool = True,
                           now: Optional[str] = None) -> dict:
    """THE completion path — tick the current step, hand off, record it.

    Board #184 §3.6 extracted this out of `complete_step` so the task-level
    `/stamp` can run the SAME code instead of its own hand-rolled tick. That
    divergence is why a stamped chain step carried `done: True` with a null
    `completed_at`, no per-step `stamp{state,by,at}` and no T-gate: two paths
    that tick a step will always drift, so there is one path now.

    Enforces T8 (approved stamp) and T10 (required deliverables) in that order —
    the stamp gate first, because it is the older and more fundamental refusal.

    Returns a receipt describing what happened, so a caller that writes its own
    system comment (the stamp) can name the same facts this one would have.

    `now` lets a caller that is recording the SAME EVENT under two names — the
    stamp writes `stamp.at`, this writes `completed_at` — pass one timestamp so
    the two cannot describe different instants (asserted by #151's suite).
    """
    now = now or _now_iso()
    task_id = task["id"]
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
    # ── T10 (board #184) — required deliverables gate completion ──────────────
    # Required-unfilled BLOCKS; optional-unfilled never does (decision D2). The
    # refusal names the escalation path IN LINE (D3) so an agent that hits this
    # wall takes the one existing road instead of inventing a workaround.
    missing = _unfilled(step, required=True)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f'step "{_step_title(step)}" needs its required deliverable '
                   f'"{missing[0]}" before it can be completed (T10) — fill it, '
                   "or ⚠ Raise issue → M if you cannot")
    skipped = _unfilled(step, required=False)
    step["done"] = True
    step["completed_at"] = now
    step["completed_by"] = actor
    _ensure_step_ids(cl)
    old_assignee = task.get("assignee")
    new_assignee = _next_assignee(cl, old_assignee)
    update = {"checklist": cl, "assignee": new_assignee,
              "kevin_final": _derive_kevin_final(cl, task.get("kevin_final"))}
    update.update(extra_update or {})
    c.table("dev_tasks").update(update).eq("id", task_id).execute()
    if _current_step_index(cl) is None:
        tail = f"chain complete → assignee {new_assignee} (close)"
    elif (old_assignee or None) != (new_assignee or None):
        tail = f"handed off: {old_assignee or '—'} → {new_assignee}"
    else:
        tail = f"still on {new_assignee}"
    # The DURABLE half of "visible, not silent" (D2). A UI dialog nobody records
    # is not visibility — the thread has to be able to answer "what did we skip?"
    # months later, when the dialog is long gone.
    note = (f' · {len(skipped)} optional deliverable'
            f'{"s" if len(skipped) != 1 else ""} left unfilled: '
            + ", ".join(f'"{s}"' for s in skipped)) if skipped else ""
    if post_comment:
        _sys_comment(
            c, task_id,
            f'step {idx + 1} "{_step_title(step)}" complete (by {actor}) · '
            f'{tail}{note}', step_order=idx)
    return {"idx": idx, "title": _step_title(step), "tail": tail,
            "optional_note": note, "old_assignee": old_assignee,
            "new_assignee": new_assignee}


# THE Approval-stage stamp — ONE mode (board #232, Kevin 07-30).
#
# It was two (board #136): 'delegate' = "Approve — M closes" and 'final' =
# "Approve + I review before Done" (kevin_final, two-touch). Kevin's ruling:
# *"instead of 'approved and closes', we just have 'approved to start'… it's
# clear that the approval is to start. It's not necessarily 'approved that it
# looks great'."* The stamp is PERMISSION TO EXECUTE, not a verdict on the
# outcome, and the label now says so.
#
# WHY THE TWO-TOUCH MODE WENT AWAY RATHER THAN STAYING BESIDE IT: 'final' is a
# PRE-gate — Kevin commits up front to reviewing before Done, which is easy to
# forget by the time the work lands, days later. The new `Closed` status is a
# POST check on finished work, and it blocks nothing (the work has already
# shipped). Keeping both would make him review the same task twice.
#
# `kevin_final` is now VESTIGIAL and this endpoint never sets it. The COLUMN,
# its existing rows, its validation and the two-touch close guard all stay
# standing on purpose (#136/#151) — tasks stamped two-touch before today must
# keep behaving exactly as stamped. Removing the column is a follow-up.
#
# 'delegate' is accepted as a LEGACY ALIAS (same behaviour, one stamp) so a
# stored caller does not 400. 'final' is REFUSED LOUDLY rather than silently
# downgraded: a caller asking for a two-touch gate that no longer exists must
# find out, not be told yes and get something else.
_STAMP_MODE = "start"
_STAMP_MODE_ALIASES = ("start", "delegate")
_STAMP_LABEL = "approved to start"


@router.post("/{task_id}/stamp")
def stamp_approval(task_id: int, payload: dict = Body(...),
                   user=Depends(get_current_user)):
    """Record Kevin's approval stamp on a task sitting in Approval.

    ONE stamp, `approved to start` (board #232): arms `ai_eligible` (board
    #198), moves Approval → Todo, ticks the pending Kevin checklist step (board
    #151 — legacy `role`/`text` AND #182 chain `owner`/`title`, board #225,
    recording `completed_at`/`completed_by` and settling that step's own stamp
    gate to `approved`, board #244),
    hands the task off to the next incomplete step's owner (board #225, via the
    same `_next_assignee` /steps/complete uses — a stamped task must never be
    left sitting on `kevin`), and writes one system comment naming the stamp,
    the ticked step, the hand-off and who recorded it (M
    relaying Kevin's verbal OK passes author='kevin' and says so on the thread).
    It NO LONGER sets `kevin_final` — Kevin's look at the finished work is the
    `Closed` status now, after `Done`, where it blocks nothing.
    """
    mode = payload.get("mode") or _STAMP_MODE
    if mode == "final":
        # LOUD, not a silent downgrade — see _STAMP_MODE above.
        raise HTTPException(
            status_code=400,
            detail="the two-touch stamp ('final') was retired by board #232 — "
                   "there is one stamp now, 'approved to start' (mode='start'), "
                   "and Kevin's look at the finished work is the `Closed` "
                   "status after `Done`")
    if mode not in _STAMP_MODE_ALIASES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {sorted(_STAMP_MODE_ALIASES)}")
    author = (payload.get("author") or "kevin").strip() or "kevin"
    c = _admin()
    rows = c.table("dev_tasks").select("id,status,checklist,assignee") \
        .eq("id", task_id).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="task not found")
    if rows[0].get("status") != "Approval":
        raise HTTPException(
            status_code=409,
            detail=f"task is {rows[0].get('status')} — stamps only apply "
                   "to tasks in Approval")
    # The stamp ARMS the task for AI work (board #198). Kevin's review stays the
    # arming gate — nothing runs unapproved — but the guarantee now rides on
    # `ai_eligible` instead of on the task sitting in `Todo`, so the chain can
    # re-arm itself between steps without shoving the card back across the board.
    # `kevin_final` is DELIBERATELY ABSENT from this patch (board #232): the
    # stamp is permission to START, and it no longer decides who closes.
    update = {"status": "Todo", "ai_eligible": True}

    # The stamp IS the completion of the first pending Kevin checklist step
    # (board #151). Without ticking it, next_actor() keeps returning 'kevin'
    # after the task lands in Todo; the dispatcher's eligibility gate
    # (next_actor == assignee) then skips it, and an approved task sits in
    # Todo forever (dispatchable=0) — looking approved but going nowhere.
    # Tick ONLY the first un-done kevin step: a later Kevin review/close step
    # must stay open so its own sign-off is still asked for.
    #
    # Board #225: read the step through `_step_owner`/`_step_title`, NOT through
    # the legacy `role`/`text` keys. A #182 process chain carries `owner`/
    # `title`, so the raw-key version matched NOTHING on every chained task —
    # the stamp silently did half its job (3 stamps on 07-30, 2 hand-ticked).
    # #151's test stayed green because it builds a LEGACY checklist; the chain
    # cases below are the ones that fail without this.
    #
    # Board #244 — the tick must RECORD ITSELF, on both fields that describe it:
    #
    #  (a) `completed_at`/`completed_by`. This tick used to write `done` alone,
    #      so #243 — the first stamp to auto-tick — recorded WHO with a None.
    #      The whole reason agents tick through /steps/complete instead of
    #      PATCHing `done` is that the handler keeps that audit trail; a tick
    #      through the stamp is the same event and owes the same record. The
    #      actor is `author` — the person the stamp comment already attributes
    #      it to, so the two can never name different people (M relaying
    #      Kevin's verbal OK passes author='kevin' and both say kevin).
    #
    #  (b) `stamp.state`. A kevin step that gates on a stamp kept `pending`
    #      while the system comment said approved and the step read done — the
    #      modal drew a gold "🔏 needs stamp" chip beside its own ✓. This stamp
    #      IS that step's approval, so it settles the gate the same way
    #      /steps/stamp does (state/by/at), and the two paths agree. Only a
    #      step that ALREADY carries a stamp block is settled — a step with no
    #      gate has nothing to settle, and inventing one would rewrite the
    #      chain. A prior `rejected` is superseded rather than preserved: the
    #      stamp is the later act by the same authority (and it cannot arise in
    #      practice — a rejected step routes to M, leaving Approval behind).
    #
    # Board #184 §3.6 — THE STAMP GOES INTO THE STEP IT WAS MADE ON.
    #
    # (a) above records the stamp on the TASK. What it did not record is WHICH
    #     step Kevin signed off: a chain step's own `stamp` block stayed None, so
    #     the chain remembered that a sign-off happened but not what it was for.
    #     Kevin's ask ("move the stamp into the step") is the same principle as
    #     this task's original one — an artifact that is task-global but
    #     semantically step-local belongs where it was made.
    #
    # (b) The tick itself now runs through `_complete_current_step`, the SAME
    #     body /steps/complete uses, so a stamped step gets completion
    #     provenance, the hand-off, kevin_final and the T-gates instead of a
    #     hand-rolled half of them.
    #
    # FENCED to the case where the first un-done kevin step IS the current step,
    # because the shared path completes the CURRENT step (T3, strict order) and
    # nothing else. When they differ — a kevin step sitting behind an open F step
    # — routing through it would tick the WRONG step, which is worse than the gap
    # it fixes. Legacy (non-chain) checklists take the same fall-back, byte
    # for byte, so the 42 existing tasks are untouched. No retroactive backfill.
    checklist = rows[0].get("checklist") or []
    old_assignee = rows[0].get("assignee")
    new_assignee = old_assignee
    kevin_idx = next(
        (i for i, s in enumerate(checklist)
         if isinstance(s, dict) and not s.get("done")
         and str(_step_owner(s) or "").strip().lower() == "kevin"), None)
    chain_path = (_is_process_chain(checklist) and kevin_idx is not None
                  and kevin_idx == _current_step_index(checklist))

    if chain_path:
        step = checklist[kevin_idx]
        now = _now_iso()
        # The stamp IS this step's approval — record it ON the step, whether or
        # not the step declared a stamp gate. A gated step gets its `required`
        # preserved (that is the gate #244 settles); an ungated one gets
        # `required: False` and the same provenance, which is the #184 half: the
        # chain now carries WHICH step was signed off, by whom, when.
        stamp = step.get("stamp") if isinstance(step.get("stamp"), dict) else {}
        stamp.setdefault("required", False)
        stamp["state"] = "approved"
        stamp["by"] = author
        stamp["at"] = now
        step["stamp"] = stamp
        task_row = dict(rows[0])
        task_row["checklist"] = checklist
        # One write: the stamp's own columns ride along with the completion.
        receipt = _complete_current_step(
            c, task_row, author, extra_update=update, post_comment=False,
            now=now)   # ONE instant: stamp.at and completed_at are one event
        ticked, new_assignee = receipt["title"], receipt["new_assignee"]
        tick_note = f' · checklist: ticked "{ticked}"{receipt["optional_note"]}'
        hand_note = ("" if (new_assignee or None) == (old_assignee or None)
                     else f" · handed off: {old_assignee or '—'} → {new_assignee}")
    else:
        ticked = None
        for step in checklist:
            if (not step.get("done")
                    and str(_step_owner(step) or "").strip().lower() == "kevin"):
                now = _now_iso()
                step["done"] = True
                step["completed_at"] = now
                step["completed_by"] = author
                stamp = step.get("stamp")
                if isinstance(stamp, dict):
                    stamp["state"] = "approved"
                    stamp["by"] = author
                    stamp["at"] = now
                    step["stamp"] = stamp
                ticked = _step_title(step)
                update["checklist"] = checklist  # JSONB is replaced whole
                break

        # Second half of #225: ticking the step is not the hand-off. Without this
        # the task lands in Todo still assigned to `kevin` — it will not dispatch,
        # and it trips preflight invariant (7). Same helper /steps/complete uses, so
        # the two paths cannot drift on who the task waits on next.
        if ticked is not None:
            new_assignee = _next_assignee(checklist, old_assignee)
            if (new_assignee or None) != (old_assignee or None):
                update["assignee"] = new_assignee

        c.table("dev_tasks").update(update).eq("id", task_id).execute()
        tick_note = f' · checklist: ticked "{ticked}"' if ticked else ""
        hand_note = (f" · handed off: {old_assignee or '—'} → {new_assignee}"
                     if "assignee" in update else "")

    c.table("dev_task_comments").insert({
        "task_id": task_id, "author": "system",
        "body": f"stamp: {_STAMP_LABEL} (by {author}) · "
                f"status: Approval → Todo · ai_eligible: true"
                f"{tick_note}{hand_note}"}).execute()
    res = c.table("dev_tasks").select("*").eq("id", task_id).execute().data
    return res[0] if res else {"status": "stamped"}


# ── Process-chain step actions (board #182) ──────────────────────────────────
# The server side of the three step buttons (Step 5 UI). Completion state and
# hand-off live here — NOT in the generic checklist PATCH — so the transition
# rules (T1/T2/T3/T4'/T8/T9) are API refusals, not UI conventions.
#
# TWO of the three — /steps/complete and /steps/raise-issue, and ONLY those
# two — gate on `get_service_or_user` rather than `get_current_user` (board
# #217); /steps/stamp is held back, see the note above it. Both attribute
# their write to payload["actor"], never to the caller's identity: `user` is a
# pure gate here and is not read by either. A headless dispatched agent holds
# the service-role key but no Supabase user JWT, so under `get_current_user` the
# step-tick contract (#202) was unsatisfiable — every HTTP tick ever attempted
# was refused 401, and agents fell back to hand-rolling this handler's logic
# through PostgREST, losing completed_at/completed_by/_next_assignee/
# kevin_final and the hand-off comment. See api/deps.py for the blast-radius
# argument; the check lives THERE and is opted into HERE, never inside the
# shared `get_current_user`.

@router.post("/{task_id}/steps/complete")
def complete_step(task_id: int, payload: dict = Body(default={}),
                  user=Depends(get_service_or_user)):
    """Complete the current step and hand off (T1/T2/T3/T8/T10).

    Ticks the FIRST incomplete step only — you cannot complete a later step out
    of order (T3). A step that gates on a stamp cannot be completed until the
    stamp is approved (T8); a step with an unfilled REQUIRED deliverable cannot
    be completed at all (T10, board #184). Records completed_at/by and reassigns
    the task to the next incomplete step's owner — 'M' when the chain finishes,
    never kevin (T2). One action, one write.
    """
    actor = (payload.get("actor") or "").strip() or "system"
    c = _admin()
    _complete_current_step(c, _task_row(c, task_id), actor)
    return _task_row(c, task_id)


# ── Step deliverables — the capture endpoints (board #184, spec §3) ──────────
# `get_service_or_user`, on the same #217 argument /steps/complete uses and for
# the same reason: a deliverable is an AGENT'S EVIDENCE, and a headless agent
# holds the service key but no user JWT. Held to the same discipline too — the
# write is attributed to payload["actor"], never to the caller's identity.
# /steps/stamp stays user-only; a stamp is authority, evidence is not.

def _load_deliverable(c, task_id: int, deliverable_id: str, *, for_write: bool):
    """(task, checklist, step_index, step, deliverable) or the right refusal.

    `for_write` adds the T4' fence: a COMPLETED step's deliverables are frozen
    with the rest of it. Any NOT-yet-done step may be filled, not just the
    current one — evidence often arrives early, and refusing it would push
    people to park it in a comment where no surface can see it (spec §2).
    """
    task = _task_row(c, task_id)
    cl = task.get("checklist") or []
    idx, step, d = _find_deliverable(cl, deliverable_id)
    if d is None:
        raise HTTPException(
            status_code=404,
            detail=f"no deliverable {deliverable_id!r} on task #{task_id}")
    if for_write and step.get("done"):
        raise HTTPException(
            status_code=409,
            detail=f'step "{_step_title(step)}" is complete — its deliverables '
                   "are part of the audit trail and can no longer be changed "
                   "(T4')")
    return task, cl, idx, step, d


def _save_checklist(c, task_id: int, cl):
    """Whole-array write. NEVER a partial merge — a partial dict REPLACES a
    JSONB value rather than merging into it (memory: jsonb_partial_updates)."""
    _ensure_step_ids(cl)
    c.table("dev_tasks").update({"checklist": cl}).eq("id", task_id).execute()


@router.post("/{task_id}/steps/deliverables/{deliverable_id}")
def fill_deliverable(task_id: int, deliverable_id: str,
                     payload: dict = Body(...),
                     user=Depends(get_service_or_user)):
    """Fill a text or link deliverable (spec §3.1). A file goes to /upload."""
    actor = (payload.get("actor") or "").strip() or "system"
    c = _admin()
    task, cl, idx, step, d = _load_deliverable(
        c, task_id, deliverable_id, for_write=True)
    kind = d.get("kind")
    if kind not in ("text", "link"):
        raise HTTPException(
            status_code=409,
            detail=f'deliverable "{d.get("label")}" is a {kind} — POST the file '
                   "to .../upload instead")
    value = payload.get("value")
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=400, detail="value is required (a non-empty string)")
    value = value.strip()
    if kind == "text" and len(value) > _DELIVERABLE_TEXT_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"a text deliverable is at most {_DELIVERABLE_TEXT_MAX} "
                   f"chars (got {len(value)}) — attach a file instead")
    if kind == "link" and not re.match(r"^https?://\S+$", value):
        raise HTTPException(
            status_code=400,
            detail="a link deliverable must be an http(s):// URL")
    d["value"] = value
    d["filled_at"] = _now_iso()
    d["filled_by"] = actor
    _save_checklist(c, task_id, cl)
    _sys_comment(
        c, task_id,
        f'deliverable "{d.get("label")}" filled (by {actor}) on step {idx + 1} '
        f'"{_step_title(step)}"', step_order=idx)
    return _task_row(c, task_id)


@router.post("/{task_id}/steps/deliverables/{deliverable_id}/upload")
def upload_deliverable(task_id: int, deliverable_id: str,
                       payload: dict = Body(...),
                       user=Depends(get_service_or_user)):
    """Upload a file deliverable (spec §3.2): {filename, content_type, b64}.

    JSON+base64, NOT multipart, deliberately: `python-multipart` is not a
    dependency of this service (nothing in the API declares File()/UploadFile),
    and the frontend's `apiFetch` pins `Content-Type: application/json` on every
    request. Choosing multipart would mean a new runtime dependency plus a
    second client path — for a 10 MB ceiling whose base64 overhead is 33%. The
    spec left the choice open precisely so this one path serves the screenshots
    helper too (§4).

    Validates size + type BEFORE touching storage, so a refusal never leaves an
    orphan object behind.
    """
    import base64
    import binascii

    actor = (payload.get("actor") or "").strip() or "system"
    c = _admin()
    task, cl, idx, step, d = _load_deliverable(
        c, task_id, deliverable_id, for_write=True)
    if d.get("kind") != "file":
        raise HTTPException(
            status_code=409,
            detail=f'deliverable "{d.get("label")}" is a {d.get("kind")} — POST '
                   "its value to the fill endpoint instead")
    filename = (payload.get("filename") or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _DELIVERABLE_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"'{ext or filename}' is not an accepted file type — "
                   f"allowed: {', '.join(_DELIVERABLE_EXTS)}")
    content_type = payload.get("content_type")
    if content_type is not None and not isinstance(content_type, str):
        raise HTTPException(
            status_code=400, detail="content_type must be a string")
    try:
        blob = base64.b64decode(payload.get("b64") or "", validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=400, detail="b64 must be base64-encoded file bytes")
    if not blob:
        raise HTTPException(status_code=400, detail="the file is empty")
    if len(blob) > _DELIVERABLE_FILE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"the file is {len(blob) // 1024} KB — the limit is "
                   f"{_DELIVERABLE_FILE_MAX_BYTES // (1024 * 1024)} MB")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:96]
    path = (f"task-{task_id}/step-{step.get('id') or idx}/"
            f"{uuid.uuid4().hex}-{safe}")
    # The SAME storage client bar_cache_store.py:125 uses — one client path in
    # this repo, not two (spec §4). A storage failure must NOT half-write the
    # checklist, so the object lands first and the record only after.
    from db import get_admin_client
    try:
        get_admin_client().storage.from_(DELIVERABLES_BUCKET).upload(
            path, blob,
            {"content-type": content_type or "application/octet-stream",
             "upsert": "true"})
    except Exception as e:  # noqa: BLE001 — fail LOUD, never a silent no-op
        log.warning("deliverable upload failed (task %s, %s): %s",
                    task_id, path, e)
        raise HTTPException(
            status_code=502,
            detail=f"storage upload failed — is the '{DELIVERABLES_BUCKET}' "
                   f"bucket created? ({e})")
    d["value"] = path            # the OBJECT PATH, never a URL (spec §1)
    d["filename"] = filename
    d["size_bytes"] = len(blob)
    d["content_type"] = content_type
    d["filled_at"] = _now_iso()
    d["filled_by"] = actor
    _save_checklist(c, task_id, cl)
    _sys_comment(
        c, task_id,
        f'deliverable "{d.get("label")}" filled (by {actor}) on step {idx + 1} '
        f'"{_step_title(step)}" — 📎 {filename}', step_order=idx)
    return _task_row(c, task_id)


@router.delete("/{task_id}/steps/deliverables/{deliverable_id}/value")
def clear_deliverable(task_id: int, deliverable_id: str,
                      actor: Optional[str] = None,
                      user=Depends(get_service_or_user)):
    """Clear a fill on a not-yet-done step (spec §3.3).

    Deliberately does NOT delete the stored object: making an audit-trail edit
    destructive is how evidence disappears. The orphan sweep is a named
    follow-up (spec §9.1), not a silent omission.
    """
    who = (actor or "").strip() or "system"
    c = _admin()
    task, cl, idx, step, d = _load_deliverable(
        c, task_id, deliverable_id, for_write=True)
    if not _is_filled(d):
        return _task_row(c, task_id)          # already empty — idempotent
    was = d.get("filename") or d.get("value")
    for f in _DELIVERABLE_FILL_FIELDS:
        d[f] = None
    _save_checklist(c, task_id, cl)
    _sys_comment(
        c, task_id,
        f'deliverable "{d.get("label")}" cleared (by {who}) on step {idx + 1} '
        f"— was `{str(was)[:120]}`", step_order=idx)
    return _task_row(c, task_id)


@router.get("/{task_id}/steps/deliverables/{deliverable_id}/url")
def deliverable_url(task_id: int, deliverable_id: str,
                    user=Depends(get_service_or_user)):
    """Mint a short-lived signed read URL for a file deliverable (spec §3.4).

    ON CLICK, never on board load: the bucket is private (the board is
    admin-gated and a public-read URL walks straight past that gate), and the
    120-row board fetch must stay cheap.
    """
    c = _admin()
    _t, _cl, _i, _s, d = _load_deliverable(
        c, task_id, deliverable_id, for_write=False)
    if d.get("kind") != "file" or not _is_filled(d):
        raise HTTPException(
            status_code=404,
            detail="that deliverable has no stored file to read")
    from db import get_admin_client
    try:
        res = get_admin_client().storage.from_(DELIVERABLES_BUCKET) \
            .create_signed_url(d["value"], _DELIVERABLE_URL_TTL_S)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not sign URL: {e}")
    url = (res or {}).get("signedURL") or (res or {}).get("signedUrl")
    if not url:
        raise HTTPException(
            status_code=502, detail="storage returned no signed URL")
    return {"url": url, "filename": d.get("filename"),
            "content_type": d.get("content_type")}


@router.post("/{task_id}/steps/raise-issue")
def raise_issue(task_id: int, payload: dict = Body(...),
                user=Depends(get_service_or_user)):
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


# NOT opted into `get_service_or_user`, deliberately (board #217). The
# diagnosis proposed all three step-action routes; this one is held back. A
# stamp is Kevin's APPROVAL, not an agent's deliverable — it is only ever
# issued from the UI, where a real user JWT exists, and no headless run has
# ever needed it (the step-tick contract asks for /steps/complete, and the
# inbound check for /steps/raise-issue). Opening it would let a bearer of the
# service key post `{"actor": "kevin", "decision": "approve"}` and manufacture
# a sign-off through the very route whose output the T8 gate trusts. That the
# key ALREADY permits the equivalent PostgREST write is an argument for fixing
# that hole, not for widening a second one to match it.
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


# ── Cross-task comment feed (board #220 — the M-session dashboard) ───────────
# Every other comment read on this router is scoped to ONE task, because every
# other consumer opens one thread. The M-session activity log is the first
# reader that needs the board's comment stream as a whole, and fanning
# `/{task_id}/comments` over ~90 open tasks is exactly the poll-efficiency
# mistake board #148 measured. So: one list GET, newest first.
#
# Read-only, zero new capture (board #220 Phase 1 rail): the events already
# exist — system transitions, agent reports, Kevin's `@M` Ask-AI comments.
_FEED_LIMIT_CAP = 500
# Bodies are the whole cost here: 07-30 sampling put M's own comments at ~1,278
# chars each, and agent reports run longer. The feed renders an EXCERPT and
# deep-links to the thread for the rest (Kevin's pointer-not-repetition model),
# so the wire carries an excerpt too — with the true length alongside it, so the
# UI can say how much it is not showing instead of silently eliding.
_FEED_CHARS_CAP = 20000


@router.get("/comments/recent")
def recent_comments(limit: int = 300, max_chars: int = 1400,
                    user=Depends(get_current_user)):
    """Recent comments across ALL tasks, newest first (board #220).

    `max_chars` truncates each body for the wire; `body_chars` always carries
    the UNTRUNCATED length, so `len(body) < body_chars` is the truncation tell.
    `max_chars=0` returns full bodies (used by nothing on the page today —
    kept so a debugging read doesn't have to guess at a large number).

    Route order note: this path cannot collide with `/{task_id}/comments` —
    that template requires the SECOND segment to be the literal "comments",
    and here the second segment is "recent".
    """
    limit = min(max(limit, 1), _FEED_LIMIT_CAP)
    rows = _admin().table("dev_task_comments") \
        .select("id,task_id,author,body,created_at,step_order") \
        .order("created_at", desc=True).limit(limit).execute().data or []
    cap = min(max_chars, _FEED_CHARS_CAP) if max_chars and max_chars > 0 else 0
    for r in rows:
        body = r.get("body") or ""
        r["body_chars"] = len(body)
        if cap and len(body) > cap:
            r["body"] = body[:cap]
    return rows


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
