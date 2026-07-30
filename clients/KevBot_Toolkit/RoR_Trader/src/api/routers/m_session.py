"""M-Session Dashboard API — sections 3 and 4 (board #220 step 4).

Two things live here, and they are opposites on purpose:

  SECTION 3 · THE RULES, RENDERED FROM SOURCE (read-only, no DB at all).
    Kevin's design principle A: *"RENDER SOURCES, DO NOT RESTATE THEM. Section 3
    must render the charter's actual text (and the skills' actual rails) from
    their files — never a retyped summary. A hand-maintained copy of the rules
    on a page is a second source of truth that will silently diverge, which is
    the exact failure mode of the untracked-copy problem that bit us three times
    on 07-30."*

    So `/rules` reads `Session_Charters.md` and the `/task-chain` skill off
    disk and returns their VERBATIM text. It never paraphrases, never caches a
    copy, and — the rail with a test behind it — **FAILS LOUD when a source file
    or a section anchor is missing**. A renderer that quietly returns nothing
    when a heading is renamed is worse than no page: it reads as "there are no
    rules" instead of "the rules moved".

  SECTION 4 · M'S OWN STATE (the one genuinely new store).
    Kevin's design principle B: *"IT IS A VIEW, NOT A SECOND BOARD."* The ONLY
    state permitted here is M-specific and NOT derivable from the board — M's
    intended ordering, M's seen/actioned marks, M's canvas text
    (migrations/m_session_state.sql). Every write path below rejects a payload
    carrying a board-owned field, with a test that fails without the guard,
    because the drift this prevents is silent: two answers to one question, and
    no error anywhere to say which is right.

Admin-gated via the service-role client, same posture as dev_tasks.

MIGRATION ORDER: `m_session_state.sql` is AUTHORED AND UNAPPLIED at the time
this ships. Reads and writes therefore degrade to a structured
`available: false` + `reason` rather than a 500, so the page can say "the store
is not provisioned" in red instead of showing a stack trace — but that is a
diagnostic, NOT a licence to merge ahead of the migration. Hold the PR.
"""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

router = APIRouter(prefix="/api/m-session", tags=["m-session"])

log = logging.getLogger(__name__)


def _admin():
    from db import get_admin_client
    return get_admin_client()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Where the source files live ─────────────────────────────────────────────
# src/api/routers/m_session.py → parents: [0] routers, [1] api, [2] src,
# [3] RoR_Trader. The skills directory sits at the GIT REPO ROOT, three further
# levels up (clients/KevBot_Toolkit/RoR_Trader), which is why the two roots are
# named separately instead of being guessed at from one base.
_ROR_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _ROR_ROOT.parents[2]

# `RORT_RULES_ROOT` / `RORT_REPO_ROOT` exist for the deployed API, whose working
# copy may not carry the whole monorepo. Not a silent default: if the resulting
# path is absent, `/rules` reports the ABSOLUTE path it tried, so a deploy that
# cannot see the charter says exactly that instead of rendering an empty page.
_ROR_ROOT = Path(os.getenv("RORT_RULES_ROOT") or _ROR_ROOT)
_REPO_ROOT = Path(os.getenv("RORT_REPO_ROOT") or _REPO_ROOT)


class RuleSource:
    """One rendered block: a file, a heading anchor, and why M is shown it."""

    def __init__(self, sid: str, label: str, root: str, path: str,
                 anchor: str, why: str):
        self.id, self.label, self.root = sid, label, root
        self.path, self.anchor, self.why = path, anchor, why

    def resolve(self) -> Path:
        base = _ROR_ROOT if self.root == "ror" else _REPO_ROOT
        return base / self.path


# The watch-list of RULES the M session is held to. Adding one is adding a row
# here — never pasting text into the page component.
RULE_SOURCES = [
    RuleSource(
        "charter-4", "Charter §4 — shared guardrails (the assignee rule)",
        "ror", "docs/_active/Session_Charters.md", "## 4.",
        "assignee = whoever the task is WAITING ON. The rule whose violation "
        "made the entire Todo queue undispatchable for ~41 hours on 07-26/27.",
    ),
    RuleSource(
        "charter-7", "Charter §7 — the board, chains and retrofit",
        "ror", "docs/_active/Session_Charters.md", "## 7.",
        "How work is represented: the board is the coordination surface, and a "
        "chain scopes the work M dispatches.",
    ),
    RuleSource(
        "charter-8", "Charter §8 — release protocol (ephemeral R sessions)",
        "ror", "docs/_active/Session_Charters.md", "## 8.",
        "M writes the brief; R merges. M never self-merges — the default "
        "cadence for getting finished work out.",
    ),
    RuleSource(
        "chain-sizing", "/task-chain — the sizing law",
        "repo", ".claude/skills/task-chain/SKILL.md", "## Sizing law",
        "A step exists to carry a real hand-off. Argue the hand-off, never the "
        "number (Kevin, 07-30 on #184).",
    ),
    RuleSource(
        "chain-rails", "/task-chain — the REQUIRED RAILS",
        "repo", ".claude/skills/task-chain/SKILL.md", "## REQUIRED RAILS",
        "The three rails every chain M authors must emit verbatim: migration "
        "authorization, a failing test per rail, and delivered ≠ fires.",
    ),
    RuleSource(
        "priority-order", "M's priority order (Kevin, 07-30)",
        "ror", "docs/_active/M_Session_Priority_Order.md", "## Priority order",
        "What M picks up first. Ask AI outranks everything — he is waiting, and "
        "expects to be waiting briefly.",
    ),
]


def _heading_level(line: str) -> int:
    """Markdown ATX heading depth, or 0 if the line is not a heading."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return 0
    n = len(stripped) - len(stripped.lstrip("#"))
    return n if n <= 6 and stripped[n:n + 1] in (" ", "") else 0


def extract_section(text: str, anchor: str):
    """VERBATIM slice from the heading matching `anchor` to the next heading of
    the same or shallower depth. Returns `(section_text, error)`.

    `anchor` is a heading PREFIX (`"## 4."`), not a full title, so renaming
    "Shared guardrails" does not break the render while renumbering — a real
    structural change — does. Matching is on the raw line so the returned text
    is byte-identical to the file; nothing is normalised, reflowed or summarised.

    On a miss the caller gets an ERROR, never an empty string. That distinction
    is the rail: a silently-empty section reads on the page as "M has no rules".
    """
    lines = (text or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if _heading_level(line) and line.strip().startswith(anchor):
            start = i
            break
    if start is None:
        return None, f"anchor {anchor!r} not found — the heading was renamed, " \
                     f"renumbered or removed"
    level = _heading_level(lines[start])
    end = len(lines)
    for j in range(start + 1, len(lines)):
        lvl = _heading_level(lines[j])
        if lvl and lvl <= level:
            end = j
            break
    # Emptiness is judged on the lines UNDER the heading — the heading itself
    # is always present, so `body.strip()` would never be empty and a section
    # gutted to a bare title would render as if it still said something.
    if not "\n".join(lines[start + 1:end]).strip():
        return None, f"anchor {anchor!r} matched an EMPTY section — the " \
                     f"heading is there but nothing is under it"
    return "\n".join(lines[start:end]).rstrip(), None


def render_rule_source(src: RuleSource) -> dict:
    """One source → its rendered block, or a loud, actionable failure."""
    p = src.resolve()
    out = {
        "id": src.id, "label": src.label, "why": src.why,
        "path": src.path, "anchor": src.anchor, "resolved_path": str(p),
        "found": False, "text": "", "error": None,
    }
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        out["error"] = f"source file not found at {p}"
        return out
    except OSError as e:  # unreadable / permissions / a directory
        out["error"] = f"could not read {p}: {e}"
        return out
    body, err = extract_section(raw, src.anchor)
    if err:
        out["error"] = err
        return out
    out["found"], out["text"] = True, body
    return out


@router.get("/rules")
def rules(user=Depends(get_current_user)):
    """Section 3 — the rules that govern M, rendered from their source files.

    Always 200, even when every source is missing: the page must be able to
    SHOW the failure. `missing > 0` is the fail-loud signal the UI renders in
    red, listing the absolute path it tried.
    """
    blocks = [render_rule_source(s) for s in RULE_SOURCES]
    return {
        "sources": blocks,
        "missing": sum(1 for b in blocks if not b["found"]),
        "ror_root": str(_ROR_ROOT),
        "repo_root": str(_REPO_ROOT),
        "generated_at": _now(),
    }


# ── Section 4 + the ordering store ──────────────────────────────────────────

# Fields `dev_tasks` owns. Design principle B in one constant: if a write here
# carried any of these, the dashboard would hold a second, divergent answer to a
# question the board already answers — and nothing would surface the conflict.
# Rejected loudly (400) rather than stripped, because a caller that sent
# `status` believed it was setting one.
BOARD_OWNED_FIELDS = {
    "status", "assignee", "title", "description", "priority_phase",
    "priority_seq", "checklist", "tags", "area", "impact", "blocked_by",
    "is_urgent", "kevin_final", "parent_id", "origin", "notes",
}

_MARK_STATES = {"seen", "actioned", "no-action"}
_CANVAS_MAX = 200_000


def _reject_board_fields(payload: dict, where: str):
    bad = sorted(BOARD_OWNED_FIELDS & set(payload or {}))
    if bad:
        raise HTTPException(400, (
            f"{where}: {', '.join(bad)} is owned by the board (dev_tasks). "
            "The M-session store holds ONLY what the board cannot answer — M's "
            "ordering, marks and canvas. Storing task state here would give "
            "Kevin two answers to one question (board #220 design principle B)."
        ))


def _unprovisioned(e: Exception) -> bool:
    """Is this the store simply not existing yet? (migration unapplied)"""
    s = str(e).lower()
    return ("does not exist" in s or "not find the table" in s
            or "pgrst205" in s or "42p01" in s
            or "could not find" in s and "schema cache" in s)


@router.get("/state")
def get_state(user=Depends(get_current_user)):
    """M's canvas + intended ordering + event marks, in one read.

    One GET rather than three: the page renders them together, and three round
    trips for one screen is the poll-efficiency mistake board #148 measured.
    """
    c = _admin()
    try:
        canvas = (c.table("m_session_canvas").select("*")
                  .eq("id", 1).limit(1).execute().data or [])
        order = (c.table("m_session_queue_order")
                 .select("task_id,rank,why,updated_at,updated_by")
                 .order("rank").execute().data or [])
        marks = (c.table("m_session_event_marks")
                 .select("event_key,state,note,task_id,marked_at,marked_by")
                 .order("marked_at", desc=True).limit(1000).execute().data or [])
    except Exception as e:  # noqa: BLE001
        if _unprovisioned(e):
            return {
                "available": False,
                "reason": ("m_session_state.sql is authored but NOT APPLIED. "
                           "Sections 3 and 4 need it; Kevin authorizes by name "
                           "before it runs (board #220 step 4 rail)."),
                "canvas": None, "order": [], "marks": [],
            }
        raise
    return {
        "available": True, "reason": None,
        "canvas": canvas[0] if canvas else {"id": 1, "body": "",
                                            "updated_at": None,
                                            "updated_by": None},
        "order": order,
        "marks": marks,
    }


@router.put("/canvas")
def put_canvas(payload: dict = Body(...), user=Depends(get_current_user)):
    """Replace M's canvas. Whole-document PUT, deliberately.

    The canvas is prose M rewrites, not a record accumulated by appends — a
    partial update would invite the JSONB-partial trap (memory
    `feedback_jsonb_partial_updates`) on a field whose whole value IS the state.
    """
    _reject_board_fields(payload, "canvas")
    body = payload.get("body")
    if body is None or not isinstance(body, str):
        raise HTTPException(400, "canvas: 'body' (string) is required")
    if len(body) > _CANVAS_MAX:
        raise HTTPException(400, f"canvas: body exceeds {_CANVAS_MAX} chars")
    row = {"id": 1, "body": body, "updated_at": _now(),
           "updated_by": str(payload.get("actor") or "M")[:64]}
    c = _admin()
    try:
        c.table("m_session_canvas").upsert(row).execute()
    except Exception as e:  # noqa: BLE001
        if _unprovisioned(e):
            raise HTTPException(503, "canvas store not provisioned — "
                                     "m_session_state.sql is unapplied")
        raise
    return row


@router.put("/order")
def put_order(payload: dict = Body(...), user=Depends(get_current_user)):
    """Replace M's intended ordering with `items`, in the order given.

    `rank` comes from ARRAY POSITION, not from the caller: the client says
    "this is my plan, in order" and cannot hand in colliding or stale ranks.
    Each item is `{task_id, why?}` and nothing else — an item carrying a
    board-owned field is a 400, not a silent strip.

    Replace-whole rather than patch-one: M's ordering is a PLAN, and half a
    plan applied is a plan nobody stated.
    """
    _reject_board_fields(payload, "order")
    items = payload.get("items")
    if not isinstance(items, list):
        raise HTTPException(400, "order: 'items' (list) is required")
    actor = str(payload.get("actor") or "M")[:64]
    rows = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise HTTPException(400, f"order: item {i} is not an object")
        _reject_board_fields(it, f"order item {i}")
        extra = sorted(set(it) - {"task_id", "why"})
        if extra:
            raise HTTPException(400, (
                f"order item {i}: unexpected field(s) {', '.join(extra)}. "
                "An ordering row is (task_id, rank, why) — rank is the array "
                "position, and everything else about a task belongs to the "
                "board."))
        try:
            tid = int(it.get("task_id"))
        except (TypeError, ValueError):
            raise HTTPException(400, f"order: item {i} has no integer task_id")
        why = it.get("why")
        rows.append({"task_id": tid, "rank": i + 1,
                     "why": (str(why)[:500] if why else None),
                     "updated_at": _now(), "updated_by": actor})
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "order: duplicate task_id in items")
    c = _admin()
    try:
        # Delete-then-insert: a task dropped from the plan must LOSE its rank,
        # which an upsert alone would leave behind as a stale intention.
        c.table("m_session_queue_order").delete().neq("task_id", -1).execute()
        if rows:
            c.table("m_session_queue_order").insert(rows).execute()
    except Exception as e:  # noqa: BLE001
        if _unprovisioned(e):
            raise HTTPException(503, "ordering store not provisioned — "
                                     "m_session_state.sql is unapplied")
        raise
    return {"count": len(rows), "items": rows}


@router.post("/marks")
def post_mark(payload: dict = Body(...), user=Depends(get_current_user)):
    """Record that M saw an event, and what M did about it.

    Keyed by the FEED EVENT KEY (`comment:41`, `run:7`, and from Phase 2
    `env:*`) rather than a comment id — the coverage record must survive the
    arrival of environment events that have no row in `dev_task_comments` at
    all. Idempotent per key: re-marking updates the verdict.
    """
    _reject_board_fields(payload, "marks")
    key = str(payload.get("event_key") or "").strip()
    if not key:
        raise HTTPException(400, "marks: 'event_key' is required")
    state = str(payload.get("state") or "").strip()
    if state not in _MARK_STATES:
        raise HTTPException(400, f"marks: 'state' must be one of "
                                 f"{sorted(_MARK_STATES)}")
    task_id = payload.get("task_id")
    row = {
        "event_key": key[:200], "state": state,
        "note": (str(payload.get("note"))[:2000] if payload.get("note") else None),
        "task_id": int(task_id) if task_id not in (None, "") else None,
        "marked_at": _now(),
        "marked_by": str(payload.get("actor") or "M")[:64],
    }
    c = _admin()
    try:
        c.table("m_session_event_marks").upsert(row).execute()
    except Exception as e:  # noqa: BLE001
        if _unprovisioned(e):
            raise HTTPException(503, "marks store not provisioned — "
                                     "m_session_state.sql is unapplied")
        raise
    return row


@router.delete("/marks/{event_key:path}")
def delete_mark(event_key: str, user=Depends(get_current_user)):
    """Un-mark — because a wrong "I looked at this" is false coverage, and the
    task's own ruling is that visible gaps beat false coverage."""
    c = _admin()
    try:
        c.table("m_session_event_marks").delete() \
            .eq("event_key", event_key).execute()
    except Exception as e:  # noqa: BLE001
        if _unprovisioned(e):
            raise HTTPException(503, "marks store not provisioned — "
                                     "m_session_state.sql is unapplied")
        raise
    return {"deleted": event_key}
