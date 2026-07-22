---
name: role-handoff
description: Role-aware session handoff for the multi-session workflow. Use when a role session (M/E/F/P or a numbered subordinate) hits a milestone or context decay and needs to hand its lane to a fresh successor session. Wraps /session-handoff with role identity, parent/subordinate naming, roster update in Session_Charters.md, and a paste-ready successor prompt. For plain end-of-session summaries without role succession, use /session-handoff instead.
---

# Role Handoff

Retire the current role session and produce everything a successor session needs to
take over the lane. Builds ON TOP of `/session-handoff` — that skill produces the core
state summary; this one adds role identity, succession, and the roster update.

Charter (read if not already in context):
`/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/_active/Session_Charters.md`

## Naming model (charter §3)

- Lane **parent** = bare letter: `E — Engine/Divergence`. Successor takes the **SAME
  name**; the outgoing session gets `(retired MM-DD)` appended by Kevin. Lineage lives
  in the roster (session id-prefix + dates), not in the name.
- **Subordinate** = numbered: `E1 — <task>`. Belongs to the lane, not a parent
  generation. A finished subordinate retires with NO successor. An unfinished
  subordinate hands off to the **next unused number** in the lane (numbers never reuse).

## When to invoke

- A milestone just closed (PR merged, flip validated, feature shipped) and the session
  is large; or the session is visibly re-asking things it once knew (context decay).
- NEVER mid-surgery (mid-deploy, mid-canary-watch, half-landed change). Finish or
  stabilize first.

## Steps

1. **Identify yourself** in the charter's Live Roster (§2): parent or subordinate, and
   your exact name. Determine the successor name per the naming model above. If you
   have no roster row, tell Kevin and propose one before proceeding.
2. **Generate the core handoff** by following the `/session-handoff` skill's template
   (full-conversation review, absolute paths, running state with shell IDs, etc.).
   EXCEPTION to that skill's chat-only rule: step 3 below edits the roster — that
   edit is required here.
3. **Update the roster** in `Session_Charters.md` §2: mark your row
   `RETIRED <date> → <successor-name>` (or just `RETIRED <date>` for a finished
   subordinate), and add the successor's row as `PLANNED — awaiting spawn`. Also update
   your lane's rows in `Team_Board.md` (reassign open tasks to the successor). Touch
   ONLY your lane's rows in both files.
4. **Output the successor paste-block** (chat, after the roster edit):

```
=== PASTE INTO NEW SESSION ===
You are <successor-name>, taking over from <your-name> (<id-prefix>) in the
multi-session workflow. In your FIRST reply, state your name so Kevin can set the
sidebar title, then:

1. Read the charter: /home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/_active/Session_Charters.md
   — you own your lane's "Owns" column and must not touch its "Must NOT touch" column.
   Note §5 (autonomy): default to getting work done in-session with loops/subagents.
2. Update your roster row from PLANNED to LIVE.
3. Check your lane's tasks in docs/_active/Team_Board.md.
4. Working directory / worktree: <absolute path>  ·  branch: <branch>
5. Read the key files listed below before doing anything else.

<full /session-handoff output pasted here>
=== END PASTE ===
```

5. **Tell Kevin** the two manual actions only he can do: append `(retired <MM-DD>)` to
   THIS session's sidebar title, and name the new session `<successor-name>`.

## Hard rules

- Roster/board edits are surgical: your lane's rows only, table structure preserved.
- The paste-block must be fully self-contained — assume the successor has NOTHING but
  the charter and what you paste.
- Absolute paths everywhere; the successor may run in a different worktree.
- No retro, no praise, no next-step speculation beyond `/session-handoff`'s
  "Pick up here" line.
