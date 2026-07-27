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

**RELEASES GO THROUGH A DEDICATED R SESSION — ALWAYS (Kevin, 07-26, restating the rule).**
M authors the brief; Kevin spawns R; R executes. M does NOT merge, even when authorized
and even when it would be faster. The 07-25/07-26 M-as-R runs were a ONE-TIME exception
granted while Kevin was away from the house and the permission layer was blocking the
normal path — not a precedent. If M believes an exception is warranted again, M ASKS
explicitly and names it as an exception; ambiguous go-aheads ("you're good to do the PR")
mean "proceed via R", never "do it yourself".

Expand the registry by adding a row + a roster entry. Keep letters short and boring.

## 2. Live roster (M curates; each session may update its own row)

| Name | Session (sidebar title / id-prefix) | Status | Notes |
|------|-------------------------------------|--------|-------|
| M — Manager/Coordinator | "Evaluate Obsidian and Graphify…" (fff1fc59) | RETIRED 07-27 → M — Manager/Coordinator | Founding M. Built the charter/board/dispatcher/registry arc 07-22→07-27; handed off at a clean boundary (all releases shipped, nothing Staged, checkout+charter synced) |
| M — Manager/Coordinator | (successor — awaiting spawn) | PLANNED | Board = SSOT at /admin/tasks. FIRST ACT: restart the dispatcher loop (it died with the previous session) |
| E — Engine/Divergence | "RoR Trader divergence overnight…" (bf8b3056) | RETIRED 07-22 → E — Engine/Divergence | Handed off at milestone boundary (V4.5 done, sub-min canonical closed) — moved up from the planned morning-brief handoff so the nightly runs in fresh context |
| E — Engine/Divergence | (abe9f5c3) | RETIRED 07-23 → E — Engine/Divergence | Milestone boundary: nightly hunt (PR #73 BAR_DUP_GUARD) + task-#100 EOD-churn triage (armed SUPPRESS_EOD_REENTRY) + #103 revision-drift heal + #101 append-supersede (PR #75) all Done. Inherited E2's shadow-worker steady-state watch (board #57 comment 78) |
| E — Engine/Divergence | (8b062a2b) | RETIRED 07-25 → E — Engine/Divergence | Monster session: nightly validation (3/3 arms passed) → #112 PREBAR evidence pack → #113 PREBAR armed post-close (VERIFIED 07-25 rebuild) → #114 carry-policy audit → #116 SPY revision-drift heal → #108 nightly settle-retrue shipped (PR #77) + armed (VERIFIED first run 00:20Z 07-25) → PR #78 health semantics sign-off + #118 340 diagnosis. Shadow-worker watch → successor |
| E — Engine/Divergence | (da15073c) | LIVE 07-25 | Successor to 8b062a2b. First act: /bug-hunt nightly (07-25 nightly already ran; PREBAR + settle-retrue first runs both CONFIRMED context). Holds V1.6 watch + shadow-worker steady-state watch |
| E1 — M-RS5a flip watch | "Build M-RS5a resident data window…" (1c7f99fb) | RETIRED 07-22 eve | Canary watch complete (loop self-stopped 21:18Z); leftovers → board V1.4/V1.5 |
| E2 — M-RS5a rollout & shadow lane | (579ed274) | RETIRED 07-23 19:49Z — clean | M-RS5a COMPLETE (fleet 23/23 @POLL_S=5). Handoff = comment #78 on board #57 (steady-state config + fingerprint SOP); shadow-worker steady-state watch → E lane; delegated authority dissolved |
| — | "Continue MRS2 P2 rollout…" (54f3a789) | RETIRED 07-22 | MRS2 P2 armed+serving |
| — | Harness secondary-TF gap session (67106cca) | RETIRED 07-22 | Work committed 07-21 (b4d9a76); Kevin to add "(retired)" to title |
| F — Frontend | (97dd74e4, anchored main window, works in `../Kevbot-frontend`) | LIVE 07-22 | Shipped V2.4 (PR #71); V2.5 in final tweaks on `feat/task-detail-panels` — merge AFTER tonight's nightly completes; then V2.2/V2.3, V2.6 agents page queued |
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

- **Spawning (Kevin's workflow: ONE window for the whole project):** all sessions are
  started with New Session in the main Kevbot-Toolkit window; the opener paste defines
  each session's ROLE. Branches are a property of checkouts, not sessions — a role
  session does ALL its file work inside its lane's worktree via absolute paths
  (`../Kevbot-frontend` = F on its branch; `../kevbot-wt-submin` = E2), and never edits
  the main checkout's tree unless the main checkout IS its lane (M docs, E engine).
  The SessionStart hook injects the whole open board grouped by role — act only on
  your own rows. Worktree-anchored windows also work (role auto-detected via
  `.claude/role` marker) but are not required.
- Hand off at **milestone boundaries** (PR merged, flip validated), never mid-surgery.
  Also hand off when a session starts re-asking things it knew earlier (context decay).
- Mechanism: run **`/role-handoff`** in the outgoing session. It produces a paste-block
  for the successor (built on `/session-handoff`) and updates the roster above.
- The successor's first acts: state its name, read this charter, read the key files
  from the handoff, check its lane's tasks in `Team_Board.md`, confirm its lane's
  "must not touch" list.

## 7. Team coordination — the board (LIVE at /admin/tasks since 07-22, PR #71)

The team board is the app's **Tasks page** (`/admin/tasks`, `dev_tasks` table).
Structure: vision items = top-level tasks titled `V# · …` tagged `vision`; work items =
subtasks (`parent_id`). Fixes/follow-ups discovered mid-work are created as subtasks
under the vision item that spawned them with `origin='discovered'` — never free-floating
(this is how the big picture survives rabbit holes). Assignees = role letters (+ kevin).

Each role session: the SessionStart hook injects your open queue automatically; update
YOUR tasks' status as you work (UI, or API `PATCH /api/dev-tasks/{id}`); leave **task
comments** for cross-session messages — the assignee sees them at their next check.
M curates structure and priorities; `Team_Board.md` is a frozen fallback snapshot only.
**Priority convention (Kevin, 07-23):** `priority_phase.seq` is M-MANAGED roadmap order
— sorted board = what's next. Kevin reads it, M adjusts it; phase buckets related work,
seq sequences within. Other roles don't edit priority except on their own new subtasks.
**Status definitions — the kanban lifecycle (Kevin+M, 07-25; supersedes 07-23 set):**
Backlog = captured, not next · Scoping = M fleshes out purpose/plan/impact so Kevin can
judge it · Approval = awaiting Kevin's stamp; NOTHING runs from here (dispatcher is
Todo-only by construction); Kevin stamps one of two ways: "Approve — M closes" or
"Approve + I review before Done" (kevin_final) · Todo = approved and queued; dispatch-
eligible · In Progress = actively worked or dispatch-claimed · Review = output done,
awaiting sign-off (M always; Kevin closes iff kevin_final) · Staged = reviewed, brief
held, waiting for a release train (trains ship everything Staged) · Done = shipped/
closed, never self-set by the agent that did the work · Blocked = anywhere-exception,
blocker named. EVERY task passes through Approval (universal, per Kevin — the impact
chip [contained·app·engine·live] makes small stamps fast); standing_approval covers
future recurring work (approved once for the series). Vision containers are exempt.
**Process-chain convention (universal, Kevin 07-25):** EVERY task carries a process
checklist — multi-role tasks spell their real handoff chain; simple tasks use the
default 4-step (do → M review → Kevin approval if flagged → close). Next-actor chip
derives from it.
**Approval tags (Kevin 07-25, batch-review protocol):** `needs-approval` = M requests
Kevin's eyes BEFORE execution (Kevin's inbox; renders amber). Kevin pre-approves by
flipping to `kevin-ok` (tag edit or a "good to go" comment — M's board-watcher converts
it) = run when bandwidth allows, no further check-ins absent surprises. `needs-review`
stays what it was: OUTPUT awaiting sign-off after a run.
**Assignee & sign-off (Kevin, 07-23):** assignee = whose hands the task is in RIGHT NOW
(reassign at each handoff; system-logged). No "primary executor" column — the chain's
role chips carry that. Sign-off tiers: assigned/next=M → M has full authority to mark
Done; assigned/next=kevin → Kevin's explicit approval, used for anything he flags plus
M's escalation criteria (user-facing UX, money-touching, scope decisions, releases).
M surfaces every "Next: kevin" item in conversation — Kevin never hunts his own queue.

**Sessions do not talk to each other directly** (no live channel exists between VS Code
sessions). Coordination is async through the board, plan docs, and Kevin. Design work
items accordingly: self-contained, with the context written down.

## 8. Release protocol (ephemeral R sessions)

1. Working session finishes a branch → runs **`/release-brief`** → outputs a paste-block.
2. Kevin opens a FRESH session (fresh eyes are the point), names it
   `R — release <branch> <MM-DD>`, pastes the brief.
3. R executes the brief top-to-bottom: verify branch state → run gates → backup branch →
   merge PR → watch deploy → write `Deploy_Log.md` entry → report → **die**. R never
   starts new work, never fixes non-trivial failures (it reports back to the working
   session instead), and aborts loudly on any gate failure.
