# Session Charters — multi-session working agreement

**Authoritative doc for running multiple Claude sessions on this repo in parallel.**
Every new role session reads this FIRST, adopts its name (see Naming), and obeys its
lane's ownership rules. Absolute path (readable from any worktree, regardless of branch):
`/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/_active/Session_Charters.md`

Companion skills: `/role-handoff` (retire a session → spawn successor) · `/release-brief`
(prepare the paste-block for an ephemeral Release session).
Team assignment board: `docs/_active/Team_Board.md` (see §7).

---

## 1. Role registry

| Role | Scope | Owns (exclusive) | Must NOT touch |
|------|-------|------------------|----------------|
| **M — Manager/Coordinator** | Cross-session coordination, doc/roadmap hygiene, big-picture tracking | This charter + roster; `Team_Board.md`; roadmap/organization docs; specs for org tooling (tasks-page upgrade, admin roadmap page) | Engine/data code; flags; deploys; heavy prod-DB; E's operational logs (`STATUS.md`, `Deploy_Log.md`, hunt logs) — M proposes restructures via spec, never live-edits E's logs |
| **E — Engine/Divergence** | Divergence debugging, bar integrity, fidelity, recomputes, bug-hunt loop, M-RS roadmap | Engine/data-path code; ALL flag flips; Railway deploys (`railway variables`, `railway up`); heavy prod-DB analysis; operational logs: `STATUS.md`, `Deploy_Log.md`, `Divergence_Hunt_Log.md`; project memory | — |
| **F — Frontend** | Portfolio pages, Strategy Health / health-overview UI, reporting UI, tasks-page implementation (per M's spec) | Next.js app UI code; its own plan docs | Engine/data files (e.g. `bar_cache.py`, `strategy_data.py`, `forward_test_service.py`, `fidelity_parity_suite.py`, worker/recompute lanes); flags; deploys; E's operational logs |
| **P — Packs** | User-pack accuracy (S/R pack first), new packs, pack-builder AI | Pack definitions, pack-builder code; its own plan docs | Same exclusions as F. **Plus:** pack edits change backtests → recomputes → shift paired-% baselines. Coordinate TIMING with E before any pack change lands. |
| **R — Release** | Ephemeral gatekeeper: validate → merge → deploy-watch → log → die | Nothing persistent | Never starts feature work; never flips flags beyond what the brief specifies |

Expand the registry by adding a row + a roster entry. Keep letters short and boring.

## 2. Live roster (M curates; each session may update its own row)

| Name | Session (sidebar title / id-prefix) | Status | Notes |
|------|-------------------------------------|--------|-------|
| M — Manager/Coordinator | "Evaluate Obsidian and Graphify…" (fff1fc59) | LIVE | Charter/roster/board owner; next: tasks-page + admin-roadmap specs |
| E — Engine/Divergence | "RoR Trader divergence overnight…" (bf8b3056) | LIVE — lane parent | 339 live-vs-settled watch + nightly hunt (board V1.6); ⚠ context decay observed 07-22 (stale-read incidents ×2) — run /role-handoff at tomorrow's morning brief |
| E1 — M-RS5a flip watch | "Build M-RS5a resident data window…" (1c7f99fb) | RETIRED 07-22 eve | Canary watch complete (loop self-stopped 21:18Z); leftovers → board V1.4/V1.5 |
| E2 — M-RS5a rollout & shadow lane | "E2: build sub minute canonical sta…" (579ed274) | LIVE — subordinate of E, **delegated authority**: shadow-worker/M-RS5a lane flags + railway-up deploys (fingerprint SOP) | Un-retired by Kevin 07-22; worktree `/home/kevin/projects/kevbot-wt-submin`; blocked on Kevin's container-fix decision (board V1.1); retires after fix lands + POLL_S=5 restored |
| — | "Continue MRS2 P2 rollout…" (54f3a789) | RETIRED 07-22 | MRS2 P2 armed+serving |
| — | Harness secondary-TF gap session (67106cca) | RETIRED 07-22 | Work committed 07-21 (b4d9a76); Kevin to add "(retired)" to title |
| F — Frontend | (to spawn — worktree `../Kevbot-frontend`) | PLANNED | Start with health-overview UI, then portfolio pages; tasks-page build once M specs it |
| P — Packs | (reserved) | HOLD | Spawn after M-RS5a flip settles + divergence board green |
| R | (ephemeral, per release) | as-needed | Name: `R — release <branch> <MM-DD>`; killed after merge+log |

## 3. Naming protocol

- **Lane parent = bare letter:** `<LETTER> — <scope>` (e.g. `E — Engine/Divergence`).
  Exactly ONE live parent per lane; the parent holds the lane's authority.
- **Subordinates = numbered:** `<LETTER><n> — <task>` (e.g. `E1 — M-RS5a flip watch`).
  Task-scoped, belong to the lane (not to a specific parent generation), retire when
  their task ships, never inherit lane authority. The parent's authority may be
  DELEGATED in a scoped slice (e.g. one service's flags/deploys) — recorded explicitly
  in the subordinate's roster row. Next unused number in the lane; numbers are not
  reused.
- **Succession keeps the name:** when a parent hands off, the successor takes the SAME
  name (`E — Engine/Divergence`); Kevin appends `(retired MM-DD)` to the outgoing
  session's title. The roster's session id-prefix + dates carry the lineage.
- **R is ephemeral:** `R — release <branch> <MM-DD>`, never numbered, never reused.
- Sessions cannot rename their own sidebar entry. A new session's FIRST reply must
  state its adopted name ("I am F — Frontend") so Kevin can set the sidebar title.

## 4. Shared guardrails (all roles)

**Git** (learned the hard way — see memory `feedback_multiagent_git_workflow`):
- Each role works in its own worktree/branch. Branch from **latest** `origin/dev`:
  `git fetch origin && git worktree add ../Kevbot-<role> -b <branch> origin/dev`.
- One PR = one branch's life. After merge, cut a NEW branch.
- Before marking any PR ready: `git log origin/dev..HEAD --oneline` must show ONLY your
  commits, and `python src/fidelity_parity_suite.py` must be ALL PASS (30/30 as of
  2026-07-22; the count grows — all green is the bar).
- `--force-with-lease` on your own branch only. Never force-push or reset `dev`.
- Backup branch before anything risky lands on dev (`backup/dev-pre-<thing>`).

**Flags & deploys:** engine-adjacent changes ship behind a flag, default OFF, read at
runtime. Only E (or Kevin) flips flags or touches Railway. Shadow-worker deploys are
`railway up` from REPO ROOT only, never var-set — E lane exclusively.

**Prod DB:** exactly ONE session (E) runs heavy prod-DB analysis during market hours —
local load starves the live worker and skews alert-lag. Others keep prod reads light or
work off dev data.

**Shared docs:** operational logs (`STATUS.md`, `Deploy_Log.md`, hunt logs) and project
memory have ONE writer: E. Organization docs (this charter, `Team_Board.md`, roadmap
docs) have ONE writer: M. Other roles communicate via their own plan docs
(`docs/_active/Plan_*.md`), PR descriptions, and their `Team_Board.md` task rows.

## 5. Autonomy & delegation (Kevin's standing preference)

- **Default: get it done in-session.** Use background loops, subagents (Agent tool),
  and long iterations freely. Do NOT punt work because it "will take a long time" —
  Kevin's experience is that looping through it just works. Do not ask Kevin to create
  a new session as a substitute for doing the work.
- **Ask Kevin to spawn a subordinate session ONLY when:** (a) the work needs its own
  long-lived worktree/branch running in parallel beyond this session's lifetime, or
  (b) it will genuinely need heavy Kevin input/decisions throughout, or (c) context is
  at risk mid-large-task (then use `/role-handoff` instead). When asking, give Kevin
  the exact name: next unused `<LETTER><n> — <task>`.
- Escalate to Kevin only for: destructive/irreversible actions, scope changes,
  cross-lane conflicts, and anything the charter reserves to him.

## 6. Session lifecycle & handoff

- Hand off at **milestone boundaries** (PR merged, flip validated), never mid-surgery.
  Also hand off when a session starts re-asking things it knew earlier (context decay).
- Mechanism: run **`/role-handoff`** in the outgoing session. It produces a paste-block
  for the successor (built on `/session-handoff`) and updates the roster above.
- The successor's first acts: state its name, read this charter, read the key files
  from the handoff, check its lane's tasks in `Team_Board.md`, confirm its lane's
  "must not touch" list.

## 7. Team coordination — the board

**Interim (live now):** `docs/_active/Team_Board.md` is the team's home base — vision
items (big picture) with subtasks, each assigned a role letter. Rules: fixes and
follow-ups discovered mid-work get parented UNDER the vision item that spawned them
(this is how the big picture stays visible through rabbit holes). Every role session
checks its lane's rows at session start and updates status on completion — status
updates on your own rows are the one edit non-M roles make to this file. M restructures.

**Target (to build):** the app's Tasks page becomes the durable version of the board —
role assignee, parent-task/subtask hierarchy, status, origin — visible in the admin UI
alongside an M-owned roadmap page. M writes the spec (`Spec_Tasks_Team_Board.md`),
F implements, M validates and migrates the markdown board into it.

**Sessions do not talk to each other directly** (no live channel exists between VS Code
sessions). Coordination is async through this file, the board, plan docs, and Kevin.
Design work items accordingly: self-contained, with the context written down.

## 8. Release protocol (ephemeral R sessions)

1. Working session finishes a branch → runs **`/release-brief`** → outputs a paste-block.
2. Kevin opens a FRESH session (fresh eyes are the point), names it
   `R — release <branch> <MM-DD>`, pastes the brief.
3. R executes the brief top-to-bottom: verify branch state → run gates → backup branch →
   merge PR → watch deploy → write `Deploy_Log.md` entry → report → **die**. R never
   starts new work, never fixes non-trivial failures (it reports back to the working
   session instead), and aborts loudly on any gate failure.
