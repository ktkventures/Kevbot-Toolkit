#!/usr/bin/env python3
"""Agent-side @-mention delivery (board #143 / V2.12, step 3).

F's half (PR #102) writes `task_mentions` rows at comment-POST time and gives
Kevin a Messages tab. That tab is a HUMAN inbox: today `@M` in a comment lands
somewhere only Kevin reads, so the agent it names never learns about it. This
module is the other direction — it reads the SAME table F built (no parallel
schema, no second definition of "unseen") and renders a block that gets injected
into an agent's context, then marks those rows seen.

Two consumers, one implementation:
  • tools/team_dispatcher/dispatcher.py — injects into a dispatched agent's prompt
  • .claude/hooks/team_board_context.py — injects at SessionStart for a human-
    driven role session (canonical copy: tools/team_dispatcher/hooks/)

WHY PostgREST and not F's /api/mentions router: the dispatcher and the hook are
Kevin-machine tools that already talk to Supabase directly with the service-role
key (dispatcher.api(), the hook's fetch_open_tasks()). F's router is auth-gated
behind get_current_user for the browser. Same table, same columns, same "unseen
= seen_at IS NULL" semantics — just the transport these two already use.

RAIL — FAIL OPEN, ALWAYS. Every public function swallows its own errors and
returns the empty/no-op value. A mentions bug must never be able to stall or
corrupt a dispatch: the dispatcher has already gone silently dead for 41 hours
once (board #171) and does not get a second mechanism for it. Delivery is
at-least-once by construction — marking seen is the LAST thing that happens, so
a failure anywhere re-delivers next time instead of losing the mention.
"""
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_REPO = "/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader"
# Prefer this checkout's .env (a worktree may run its own copy), fall back to the
# main checkout — .env is gitignored, so a fresh worktree has none.
ENV_CANDIDATES = (os.path.join(HERE, "..", "..", "src", ".env"),
                  f"{MAIN_REPO}/src/.env")

API_TIMEOUT_S = 10     # shorter than the dispatcher's 15: this is optional context
MENTION_LIMIT = 20     # newest-N guard; a flood can't blow up a prompt
EXCERPT_CHARS = 220    # chars of the comment body quoted per mention
TASKS_URL = "/admin/tasks"


# ── transport ────────────────────────────────────────────────────────────────
def _creds():
    for cand in ENV_CANDIDATES:
        path = os.path.abspath(cand)
        if not os.path.isfile(path):
            continue
        url = key = None
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("SUPABASE_URL="):
                url = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
        if url and key:
            return url.rstrip("/"), key
    raise RuntimeError("no SUPABASE creds found in " + str(ENV_CANDIDATES))


def _api(method, path, body=None):
    url, key = _creds()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{url}/rest/v1/{path}", data=data, method=method,
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", "Prefer": "return=representation"})
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as r:
        txt = r.read().decode()
        return json.loads(txt) if txt.strip() else None


# ── formatting helpers ───────────────────────────────────────────────────────
def _rel(ts, now=None):
    """'21h ago' from a timestamptz string; the raw value if it won't parse."""
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        secs = ((now or datetime.now(timezone.utc)) - t).total_seconds()
        if secs < 90:
            return "just now"
        if secs < 5400:
            return f"{int(secs // 60)}m ago"
        if secs < 172800:
            return f"{int(secs // 3600)}h ago"
        return f"{int(secs // 86400)}d ago"
    except Exception:
        return str(ts)


def _author_root(author):
    """'M·auto' → 'M'; 'kevin' → 'kevin'. The registry token behind a byline."""
    return str(author or "").split("·")[0].strip()


def _excerpt(body):
    one = " ".join(str(body or "").split())
    return one[:EXCERPT_CHARS] + ("…" if len(one) > EXCERPT_CHARS else "")


# ── read ─────────────────────────────────────────────────────────────────────
def fetch_unseen(role, api=None, limit=MENTION_LIMIT):
    """Unseen mentions addressed to `role`, oldest first. FAIL-OPEN → []."""
    try:
        role = (role or "").strip()
        if not role:
            return []
        rows = (api or _api)(
            "GET", "task_mentions"
                   f"?mentioned=eq.{urllib.parse.quote(role, safe='')}"
                   "&seen_at=is.null"
                   "&select=id,task_id,comment_id,mentioned,author,created_at"
                   f"&order=created_at&limit={int(limit)}")
        return [r for r in (rows or []) if isinstance(r, dict) and r.get("id")]
    except Exception:
        return []


def enrich(rows, api=None):
    """Attach task title/status + comment excerpt. FAIL-OPEN: on error the rows
    come back un-enriched rather than not at all — an id-only mention still
    tells the agent something happened."""
    rows = list(rows or [])
    if not rows:
        return rows
    call = api or _api
    tasks, comments = {}, {}
    try:
        ids = sorted({r["task_id"] for r in rows if r.get("task_id")})
        if ids:
            got = call("GET", f"dev_tasks?id=in.({','.join(str(i) for i in ids)})"
                              "&select=id,title,status") or []
            tasks = {t["id"]: t for t in got}
    except Exception:
        pass
    try:
        ids = sorted({r["comment_id"] for r in rows if r.get("comment_id")})
        if ids:
            got = call("GET", f"dev_task_comments?id=in.({','.join(str(i) for i in ids)})"
                              "&select=id,body") or []
            comments = {c["id"]: c for c in got}
    except Exception:
        pass
    for r in rows:
        t = tasks.get(r.get("task_id")) or {}
        r["task_title"] = t.get("title") or ""
        r["task_status"] = t.get("status") or ""
        r["excerpt"] = _excerpt((comments.get(r.get("comment_id")) or {}).get("body"))
    return rows


def unseen_for(role, api=None, limit=MENTION_LIMIT):
    """fetch + enrich in one call. FAIL-OPEN → []."""
    try:
        return enrich(fetch_unseen(role, api=api, limit=limit), api=api)
    except Exception:
        return []


# ── render ───────────────────────────────────────────────────────────────────
def _displayable(rows, role):
    """Mentions worth SHOWING `role`: everything except its own bylines. An
    agent's result report quoting '@M' would otherwise be re-served to M next
    dispatch forever. Suppressed rows are still marked seen (nothing is lost —
    you wrote them), so they can't accumulate as permanent phantom unread."""
    return [r for r in rows if _author_root(r.get("author")) != role]


def render_block(rows, role, now=None):
    """The injected text, or '' when there is nothing to say."""
    shown = _displayable(rows or [], role)
    if not shown:
        return ""
    lines = [f"=== UNSEEN @MENTIONS FOR {role} ({len(shown)}) ===",
             "Someone tagged you in a comment. This is your ONLY delivery — they are "
             "marked seen once shown, and nothing else will remind you.",
             "These may be on tasks OTHER than the one you are working. Read them; if "
             "one asks for work beyond your current step, do NOT silently do it — "
             "answer it, or say so in your final report / reassign per the assignee "
             "contract."]
    for r in shown:
        title = r.get("task_title") or ""
        status = f" [{r['task_status']}]" if r.get("task_status") else ""
        head = (f"- {r.get('author') or '?'} on #{r.get('task_id')}"
                f"{' ' + chr(34) + title[:70] + chr(34) if title else ''}{status}"
                f" · {_rel(r.get('created_at'), now=now)}")
        lines.append(head)
        if r.get("excerpt"):
            lines.append(f"    {r['excerpt']}")
        lines.append(f"    open: {TASKS_URL}?task={r.get('task_id')}")
    lines.append("=== END @MENTIONS ===")
    return "\n".join(lines)


# ── write (LAST, never before the block is delivered) ────────────────────────
def mark_seen(ids, api=None, now=None):
    """Stamp seen_at on the given mention ids. Returns how many rows came back.

    FAIL-OPEN → 0. Call this ONLY after render_block()'s output has actually
    reached the agent (prompt handed to spawn(), or printed by the hook). A
    mention marked seen but never delivered is a lost message; a mention
    delivered but not marked is merely shown twice."""
    try:
        ids = [int(i) for i in (ids or [])]
        if not ids:
            return 0
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        res = (api or _api)("PATCH",
                            f"task_mentions?id=in.({','.join(str(i) for i in ids)})"
                            "&seen_at=is.null",
                            body={"seen_at": stamp})
        return len(res or [])
    except Exception:
        return 0


# ── the two consumers' one-call entry points ─────────────────────────────────
def for_dispatch(role, api=None):
    """Dispatcher side: (block_text, ids_to_mark_after_spawn). FAIL-OPEN →
    ('', []). Marking is deliberately NOT done here — one_pass() marks after
    spawn() succeeds, so a dispatch that dies before launch re-delivers."""
    try:
        rows = unseen_for(role, api=api)
        return render_block(rows, role), [r["id"] for r in rows]
    except Exception:
        return "", []


def for_session(roles, api=None):
    """SessionStart side: (block_text, ids_to_mark_after_printing).

    `roles` is what the hook resolved (['M'], ['E2'], …). With NO role resolved
    (the main checkout, where the opener defines the role) we cannot address an
    inbox, so we show a read-only digest of everyone's unseen mentions and mark
    NOTHING — a digest can't lose a message the way a mis-addressed delivery
    can. FAIL-OPEN → ('', [])."""
    try:
        roles = [r for r in (roles or []) if str(r).strip()]
        if not roles:
            return _digest(api=api), []
        blocks, ids = [], []
        for role in roles:
            rows = unseen_for(role, api=api)
            text = render_block(rows, role)
            if text:
                blocks.append(text)
            ids.extend(r["id"] for r in rows)
        return "\n".join(blocks), ids
    except Exception:
        return "", []


def _digest(api=None):
    """Whole-board unread digest for a role-less session: counts only, no
    marking. Enough to make someone look; not a delivery."""
    try:
        rows = (api or _api)("GET", "task_mentions?seen_at=is.null"
                                    "&select=mentioned&limit=500") or []
        counts = {}
        for r in rows:
            m = r.get("mentioned")
            if m:
                counts[m] = counts.get(m, 0) + 1
        if not counts:
            return ""
        parts = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        return (f"[@Mentions] unread by role — {parts}. (No role resolved for this "
                f"checkout, so nothing was delivered or marked seen; your opener "
                f"defines your role — read your own with the Messages tab on "
                f"{TASKS_URL}.)")
    except Exception:
        return ""
