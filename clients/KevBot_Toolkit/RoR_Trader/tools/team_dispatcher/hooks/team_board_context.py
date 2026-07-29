#!/usr/bin/env python3
"""SessionStart hook: inject this checkout's role queue from the team board.

Primary source: the dev_tasks table (the app's /admin/tasks team board) via
Supabase REST with creds from src/.env. Fallback: the frozen Team_Board.md
snapshot if the API is unreachable. Single source of truth — referenced by
absolute path from the main checkout and all worktrees' settings.local.json.
Role resolution: .claude/role marker file, else path heuristics, else the
main-checkout default (whole-board digest; the opener paste defines the role).

Board #143 (V2.12): also injects this role's UNSEEN @-mentions and marks them
seen once printed — the human-session twin of the dispatcher's prompt injection,
so a comment tagging @M/@E/@F reaches the lane instead of sitting in a tab only
Kevin reads. Both sides share tools/team_dispatcher/mentions.py.

CANONICAL COPY — this file is the tracked source for the hook that
settings.local.json runs by absolute path from
/home/kevin/projects/Kevbot-Toolkit/.claude/hooks/team_board_context.py (that
path is untracked). Arm a change with:
    cp tools/team_dispatcher/hooks/team_board_context.py \\
       /home/kevin/projects/Kevbot-Toolkit/.claude/hooks/team_board_context.py
"""
import json
import os
import re
import sys
import urllib.request

REPO = "/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader"
ENV = f"{REPO}/src/.env"
BOARD_MD = f"{REPO}/docs/_active/Team_Board.md"
CHARTER = f"{REPO}/docs/_active/Session_Charters.md"
TASKS_URL = "/admin/tasks"


def resolve_roles():
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    marker = os.path.join(root, ".claude", "role")
    if os.path.isfile(marker):
        try:
            txt = open(marker, encoding="utf-8").read().strip()
            if txt:
                return [r.strip() for r in txt.replace(",", " ").split() if r.strip()], root
        except OSError:
            pass
    low = root.lower()
    if "kevbot-frontend" in low:
        return ["F"], root
    if "kevbot-wt-submin" in low:
        return ["E2"], root
    if "kevbot-engine" in low:
        return ["E"], root
    return [], root  # main checkout / unknown -> whole-board digest


def fetch_open_tasks():
    url = key = None
    for line in open(ENV, encoding="utf-8"):
        line = line.strip()
        if line.startswith("SUPABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
            key = line.split("=", 1)[1].strip().strip('"')
    q = ("dev_tasks?select=id,title,status,assignee,parent_id,tags,notes"
         "&status=neq.Done&order=priority_phase,priority_seq")
    r = urllib.request.Request(
        f"{url.rstrip('/')}/rest/v1/{q}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(r, timeout=6) as resp:
        return json.loads(resp.read().decode())


def print_api_queue(roles):
    tasks = fetch_open_tasks()
    # Team subtree only: V-numbered vision parents + their children. The legacy
    # flat backlog (pre-team dev_tasks rows) stays on /admin/tasks, not in queues.
    vision_ids = {t["id"] for t in tasks
                  if t.get("tags") and "vision" in t["tags"] and re.match(r"V\d", t["title"] or "")}
    lines = []
    for t in tasks:
        vision = t["id"] in vision_ids
        child = t.get("parent_id") in vision_ids
        if not (vision or child):
            continue
        owner = t.get("assignee") or "?"
        if roles and not vision and owner not in roles:
            continue
        if roles and vision and owner not in roles:
            continue  # role view: only your own lane's header + rows
        mark = "▸" if vision else "-"
        note = (t.get("notes") or "").split(" · Imported")[0]
        note = f" — {note[:100]}" if note and not vision else ""
        lines.append(f"{mark} ({owner}) [{t['status']}] {t['title']}{note}")
    scope = "+".join(roles) if roles else "ALL (your role comes from your opener/roster — act only on your own rows)"
    print(f"[Team Board] Open items, role(s): {scope}")
    print(f"Live board: {TASKS_URL} in the app UI · roster: {CHARTER} section 2")
    print("\n".join(lines) if lines else "(no open items)")


def print_md_fallback(roles):
    print("[Team Board] (API unreachable — frozen markdown snapshot, may be stale)")
    print(f"Live board: {TASKS_URL} · roster: {CHARTER} section 2")
    for line in open(BOARD_MD, encoding="utf-8"):
        if not line.startswith("| V"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or re.search(r"DONE|MOOT|RETIRED", cells[3], re.I):
            continue
        if roles and not (set(re.split(r"[/,\s]+", cells[2])) & set(roles)):
            continue
        print(f"- ({cells[2]}) {cells[0]} [{cells[3]}] {cells[1]}")


def print_mentions(roles):
    """Board #143 — this role's unseen @-mentions, then mark them seen.

    Marking happens AFTER the print, and only for a RESOLVED role: with no role
    (main checkout) for_session() returns a counts-only digest and an empty id
    list, so a role-less session can never consume someone else's inbox."""
    sys.path.insert(0, f"{REPO}/tools/team_dispatcher")
    import mentions  # noqa: PLC0415 — deliberately lazy; must not cost a session

    text, ids = mentions.for_session(roles)
    if not text:
        return
    print()
    print(text)
    if ids:                    # only ever AFTER the block has been printed
        mentions.mark_seen(ids)


def main():
    roles, _root = resolve_roles()
    try:
        print_api_queue(roles)
    except Exception:
        try:
            print_md_fallback(roles)
        except Exception:
            pass  # never block session start
    try:
        print_mentions(roles)
    except Exception:
        pass  # mentions are optional context — never block session start


if __name__ == "__main__":
    main()
