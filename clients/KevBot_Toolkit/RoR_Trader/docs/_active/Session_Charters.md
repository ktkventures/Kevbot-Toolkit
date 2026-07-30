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
| **M — Project Manager** (was "Manager/Coordinator", retitled 07-28) · **LIVE SESSION — never auto-dispatched** (registry `status=live-session`, board #222) | Cross-session coordination, doc/roadmap hygiene, big-picture tracking, ROUTING. Asks *"is this actually right?"* — the thorough look, where one is warranted | This charter + roster; `Team_Board.md`; roadmap/organization docs; specs for org tooling (tasks-page upgrade, admin roadmap page) | Engine/data code; flags; deploys; heavy prod-DB; E's operational logs (`STATUS.md`, `Deploy_Log.md`, hunt logs) — M proposes restructures via spec, never live-edits E's logs |
| **M-A — Manager Assistant** (headless half of the M lane, added 07-30, board #222) | The team's own INFRASTRUCTURE ENGINEER: builds M-lane tooling (dispatcher, preflight, skills, the board), writes specs and design docs, runs sweeps/audits/inventories, diagnoses bounded defects, verifies a release train's claims | Same as M — it inherits M's scope and boundaries verbatim from the registry | Same as M. **Plus, reserved to the M session:** reviewing another agent's work · merge reconciles needing judgement · restarting the loop · applying a migration · anything touching prod · setting priority · talking to Kevin · authoring a chain |
| **E — Engine/Divergence** | Divergence debugging, bar integrity, fidelity, recomputes, bug-hunt loop, M-RS roadmap | Engine/data-path code; ALL flag flips; Railway deploys (`railway variables`, `railway up`); heavy prod-DB analysis; operational logs: `STATUS.md`, `Deploy_Log.md`, `Divergence_Hunt_Log.md`; project memory | — |
| **F — Frontend** | Portfolio pages, Strategy Health / health-overview UI, reporting UI, tasks-page implementation (per M's spec) | Next.js app UI code; its own plan docs | Engine/data files (e.g. `bar_cache.py`, `strategy_data.py`, `forward_test_service.py`, `fidelity_parity_suite.py`, worker/recompute lanes); flags; deploys; E's operational logs |
| **P — Packs** | User-pack accuracy (S/R pack first), new packs, pack-builder AI | Pack definitions, pack-builder code; its own plan docs | Same exclusions as F. **Plus:** pack edits change backtests → recomputes → shift paired-% baselines. Coordinate TIMING with E before any pack change lands. |
| **TM — Task Manager** (planned, board #182) | Gates every process-chain hand-off. Asks *"was what was asked for actually delivered?"* — narrow, time-boxed, three outcomes only: **PASS** / **BOUNCE** (back to submitter with a reason) / **ESCALATE** (insert an audible PM step). Never performs the deep investigation; it ROUTES to one. **Gates M's steps too — that is the point.** | Board writes: tick steps, reassign, insert audible steps | Code; flags; deploys; prod-DB; never investigates — escalating is its answer to "this needs more" |
| **R — Release** (the CONDUCTOR since 07-30, board #229) · **LIVE SESSION — never auto-dispatched** (registry `status=live-session`) | Runs the release LANE, not one train: reads the `Staged` queue and decides what ships together and in what order; authors and ADAPTS briefs mid-flight; reconciles red gates and conflicts; calls incidents; holds the cross-train picture (wave numbering, what supersedes what). Retains long-term memory, and is the FALLBACK when automation fails | The release queue and the wave numbering; every brief R-A executes | Never starts feature work; never flips flags beyond what the brief specifies |
| **R-A — Release Executor** (headless half of the R lane, added 07-30, board #229) | ONE train, bounded context: merge → gate → deploy-watch → log, per a brief. Runs gates twice and verifies before merging; reports deploy state honestly and never stops at `pending` | Same as R — it inherits R's scope and boundaries verbatim from the registry (which already ARE the headless release scope Kevin granted 07-26) | Same as R. **Plus, reserved to the R session:** deciding what ships together · authoring or adapting a brief · reconciling a red gate or a conflict needing judgement · calling an incident · wave numbering. **ABORTS AND HANDS BACK rather than improvising** — that is the designed behaviour, not a failure |

**PM vs TM — the split (Kevin, 07-28).** `M` was doing both jobs and could not do the second
honestly: an agent that gates its own hand-offs is self-review wearing a badge. On 07-28 M made
four errors in one day — an invalid `created_at` inference, a lane-count comparison mixing one
lane against two, a query that silently truncated at 20,000 of 74,210 rows, and a backwards claim
about which lane flattens daily. M caught three; **Kevin caught the fourth.** Nothing structural
caught any of them. TM exists to close that gap. Second reason: context hygiene — M carries the
day's design conversations, release trains and investigations, so asking it a narrow operational
question filters the answer through all of it. TM opens fresh with only the step SOP and the
submission. **Letter unchanged** (`M`, not `PM`): renaming would touch the agent registry, every
prompt template, 16 board assignees and a pile of memory files for no functional payoff.

**TM rails:** time-boxed · bounce limit of two, then escalate to PM · **FAILS OPEN** (if TM cannot
run, the hand-off proceeds flagged "unreviewed" — the dispatcher died twice in two days, once
silently for 41 hours, and a hard gate would stall every task mid-chain) · silent on pass, loud on
problem.

**M vs M-A — the dividing line (Kevin, 07-30, board #222).** `M` named two different actors and
the board could not tell them apart: twice on 07-30 (#195, #219) a task assigned `M` and left ARMED
dispatched `M·auto` into work reserved for the session — one of those steps read *"M restarts the
loop"*. Measured across 35 `M·auto` runs, what the headless half ACTUALLY does is build the team's
own machinery; the session runs the operation. So they split, and **`M` now means the session**:

| Give it to **M-A** (headless, dispatchable) | Keep it on **M** (the session) |
|---|---|
| building M-lane infrastructure | reviewing another agent's work — the reviewer must hold context the builder lacked |
| writing a spec or design doc | merge reconciles needing judgement |
| sweeps, audits, inventories | restarting the loop · applying a migration · anything touching prod |
| diagnosing a bounded defect | deciding priority, and talking to Kevin |
| verifying a release train's claims | authoring a chain |

**Why M-A exists rather than doing it all in session:** it runs in PARALLEL (the session is strictly
serial), on FRESH BOUNDED CONTEXT that does not consume the session's, and every run leaves an
auditable `run_history` row. Real advantages, not a consolation prize.

**The mechanism is the registry, not a new flag.** `M` = `status=live-session`, `M-A` =
`status=headless`. `dispatcher.headless_agents()` selects `status=eq.headless`, and `triage_todo()`
already reports a non-headless assignee as *waiting, not stuck* — proven by `kevin` since 07-23. A
step-level `headless:false` flag was designed and deliberately DROPPED: the owner field already
says it, by construction. **Never infer session-only from step prose** — a keyword scan of #218
flagged *"restart the loop"* when the text said *"do NOT restart the loop"*. It must be declared.
The board renders the distinction by registry `status`/`kind` (round avatar = headless agent;
squared + ringed, "◧ session" = live session/human), never by a hard-coded letter, so the next
live-session lane inherits it for free.

**R vs R-A — the dividing line (Kevin, 07-30, board #229).** The same split, applied to the
release lane — and applied for the OPPOSITE reason to #222's. **`R·auto` is not the problem;
the evidence says it is good at its job.** Across 27 runs, 24 `ok`, 17 trains shipped, the
judgement is genuinely disciplined: it aborts on red gates rather than improvising (#217, #194
held, #191 Worker-failed), self-reports its own misses (*"flagging the deviation rather than
papering over it"* on Wave 19's pending `build`), watches deploys to settlement rather than
stopping at `pending`, resolved #210's conflict only after verifying it matched the narrow shape
the brief authorised, and independently confirmed a RED on #226 before trusting the fix.

**The problem is that every correct hand-back lands on M.** On 07-30 alone R correctly refused or
handed back, and *M* then reconciled #217's stale rail-5 assertion · overrode #222's false-negative
refusal with documentation · rebased Car 3 through a same-line conflict · **rewrote Wave 21's brief
three times** as reality moved · resolved a wave-number collision · triaged a Railway incident.
None of that is orchestration; all of it is release-lane technical work sitting in M's context.

The pair is therefore **not a demotion of the headless agent** — it is the same competence at two
context horizons (Kevin: *"the session-based agent is the big-picture thinker; their auto version
is trained on the same things and automates the work that falls in line with it… the session
retains long-term memory and can act as a fallback route in case we have automation issues"*):

| `R` — session, the CONDUCTOR | `R-A` — headless, the EXECUTOR |
|---|---|
| reads the `Staged` queue; decides **what ships together, and in what order** | merge → gate → deploy-watch → log, per a brief |
| authors and **adapts briefs mid-flight** (Wave 21 needed three rewrites) | runs gates twice, verifies before merging |
| reconciles red gates and conflicts | **aborts and hands back** rather than improvising |
| calls incidents (07-30's Railway queue) | reports deploy state honestly; never stops at `pending` |
| holds the cross-train picture: what supersedes what, wave numbering | one train, bounded context |

**M's remaining role in releases: mark a task `Staged`.** Everything after that is R's. That is the
load the split removes. Mechanism is identical to #222 — registry rows, no new machinery — and the
board rendering cost nothing, which is the proof #222's "the next live-session lane inherits it for
free" was true rather than merely intended.

**Two things the R lane inherits from M's first week of this (board #229):** (1) **a session with
nothing to wake it goes quiet** — M stalled twice on 07-30 until a heartbeat was armed, so the R
session needs its own monitor (Staged queue depth, oldest-Staged age, deploy state, open PRs, red
gates); (2) **disarm before restatusing** — two races on 07-30 came from changing an ARMED task's
status, and R touches statuses constantly.

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
| M — Project Manager | (c85c7f28) | RETIRED 07-29 → M — Project Manager | Successor to fff1fc59. 07-28/29: dispatcher V4.15→V4.17 (#171 assignee-authoritative + loud skips; #182 step-level dispatch), preflight #153 shipped, `local_update.py` env-parity gates, #117 Layer 1 executed against prod (2,555 rows annotated, 0 exits synthesized), #182 task-structure standard specified + built to 7/9. **21 PRs merged across 5 R trains, zero merge conflicts.** Retitled M → Project Manager and added the TM role (PR #113). Retired at a clean boundary: nothing half-shipped, dispatcher restarted on V4.17 and preflight-verified. |
| M — Project Manager | (9f1e8099) | RETIRED 07-29 → M — Project Manager | Killed by the 07-29 crash burst (board #188) after first acts only; no lane work landed |
| M — Project Manager | (c4764159) | RETIRED 07-29 → M — Project Manager | Verified the unverified TM gate and **rejected** it — 2 blockers: `--disallowedTools` is allow-by-default so it cannot make an agent tool-less (`Skill` stayed live, and `Skill` reaches `/local-update`, which writes the prod DB), and rail 2 was unreachable. Kevin approved the pivot → shipped the **INBOUND CHECK** instead (PR #123, dispatcher V4.18, 18 tests) — **#182 Step 7 DONE**. Also: release train Wave 6 (#190, 4 PRs, headless R), crash-burst diagnosis (#188), TM design parked (#189). Retired at a clean boundary: train merged, dispatcher restarted on V4.18 and verified against dev. |
| M — Project Manager | (1805c263) | LIVE 07-29 | Successor to c4764159. **Kevin's 07-29 direction: run HIGHER-LEVEL — coordinate with Kevin and DISPATCH agents; do not self-investigate.** Handoff doc: `docs/_active/handoffs/M_handoff_2026-07-29_v2.md`. Board = SSOT at /admin/tasks (`dev_tasks`); `Team_Board.md` is a FROZEN snapshot, do not edit it. Lane pickup: #182 **Step 8** (Kevin's end-to-end review) and **Step 9** (retrofit decision); watch the first real hand-offs for the inbound check's calibration — false raises vs rubber-stamping. |
| M-A — Manager Assistant | (headless; dispatched per task, no persistent session) | LIVE 07-30 | The M lane's headless half (board #222). Not a session anyone spawns — the dispatcher runs it per task from the registry row, which inherits M's scope/boundaries/worktree. Assign it here; assign the M **session** anything on the right-hand column of the §1 dividing line. |
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
| R — Release (conductor) | (live session, persistent across trains) | LIVE 07-30 | Board #229 promoted R from per-train ephemeral to the release LANE's session. Owns the `Staged` queue, wave numbering and every brief; dispatches the execution to `R-A`. Needs its own heartbeat monitor — a session with nothing to wake it goes quiet (M stalled twice on 07-30 before one was armed) |
| R-A — Release Executor | (headless; dispatched per train, no persistent session) | LIVE 07-30 | The R lane's headless half (board #229). Not a session anyone spawns — the dispatcher runs it per task from the registry row, which inherits R's scope/boundaries/worktree. One train per run; assign it here, and assign the R **session** anything on the right-hand column of the §1 R/R-A dividing line |

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
- **Split lanes use `<LETTER>-A`, never a number:** `M-A` (board #222), `R-A` (board #229). The
  suffix means "the headless half of that lane", which is a different thing from a numbered
  subordinate: `E1` is one task inside E's lane, `M-A` is M's lane running headless. A lane has
  at most one `-A`. `E2` predates the convention, is plan-specific and does not fit it — **retire
  it rather than repurpose it** (Kevin, 07-30). Kevin has HELD `E`/`E-A` and `F`/`F-A` until M
  and R prove the pattern out.
- **R is no longer ephemeral (board #229, 07-30).** The R *session* is persistent and conducts the
  lane; the per-train ephemeral role is now `R-A`, dispatched per release. The old
  `R — release <branch> <MM-DD>` naming applies to a manual fallback R-A run only.
- Sessions cannot rename their own sidebar entry. A new session's FIRST reply must
  state its adopted name ("I am F — Frontend") so Kevin can set the sidebar title.

## 4. Shared guardrails (all roles)

**THE ASSIGNEE IS WHOEVER THE TASK IS WAITING ON (Kevin, 2026-07-27 — binding).**
The `assignee` field is not a label for who *owns the topic*; it names the single person or
role the task is **blocked on right now**. Whoever is assigned owns the next action. If a
task is NOT assigned to you, acting on it is optional.
- **Need someone's input before you can progress? REASSIGN THE TASK TO THEM, then comment
  saying what you need.** A comment alone is not a handoff — the assignee is. Leaving a
  task on yourself while you are blocked is the failure mode this rule exists to kill.
- Kevin does the same in reverse. If he does not know who is next, **he assigns to M**, and
  M routes it — so M should read an unrouted task as a routing request, not a work request.
- **Before ending any turn on a task, set `assignee` to whoever it now waits on.** This
  applies to every role and every headless `*·auto` agent.
- **The checklist / progress bar is a GUIDELINE, never a gate.** It is shorthand for how
  the process usually goes; surprises happen and it will not always be followed. Where a
  checklist role and the assignee disagree, **the assignee wins.**
- Payoff: Kevin sorts the board by `assignee = kevin` and sees exactly what is blocked on
  him. Anything not assigned to him is not yet his problem.
- Cost of getting this wrong, measured: the dispatcher used to treat the checklist as
  authoritative (`next_actor()` overriding `assignee`), which silently made the **entire
  Todo queue undispatchable for ~41 hours** on 07-26/27 with no error, log, or comment.
  See memory `feedback_assignee_is_the_waiting_on`.

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
derives from it. A step carries `id · owner · title · body · mode · origin · stamp`;
a checklist becomes a process chain the moment ANY step carries one of those keys,
so conversion is opt-in per task with no migration and no flag day. `mode=discuss`
marks a step that wants a human REPLY — the dispatcher refuses to hand one to a
headless agent. The step's `body` IS the SOP the dispatcher sends that agent, which
is the whole payoff: context lives in the task instead of being hand-written at
dispatch time.
**Retrofit rule (Kevin, 07-29 — #182 Step 9's decision):** existing tasks are **NOT
batch-converted**. A task gets a real chain **when someone touches it** — written
before the work starts, if the absence of a chain would create a gap. Kevin's
reasoning, verbatim: *"if we do it one by one, it'll probably get better quality.
Sometimes I think when we do things in batches, we get a little lazy."* Legacy
`{role,text,done}` checklists keep rendering and dispatching untouched, so there is
no deadline and nothing rots. **Judgement applies:** a small contained task does not
need nine steps — a chain exists to carry a real hand-off, not to be ceremony. Write
the SOP bodies as *what the step is for, what "done" means, and what it must not do*;
retroactive steps also record what actually came out.
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

## 8. Release protocol (R conducts; R-A executes)

**DEFAULT CADENCE — SHIP EACH REVIEWED BRANCH IMMEDIATELY (Kevin, binding, 2026-07-29).**
A branch that has passed M review goes to a train **at once**, as its own single car if
nothing else is ready. Do **NOT** accumulate reviewed branches waiting for a fuller train.
- Kevin's words: *"don't worry about pushing the dev... my short-term goal is to be able to
  see the changes go live. When we're waiting on trains and gates and stuff, it could be
  hours before I see a UI change."*
- **The cost model that justified batching does not currently apply.** Each dev push
  restarts the live Worker (1–3 min bar-recording gap), so batching once saved real money.
  Kevin is **not trading live** and will not until the dev/prod split (board #204). Do not
  spend his time protecting a cost he has told you he is not paying.
- **Batching is the EXCEPTION and needs Kevin asking for it in the moment.** "You can group
  them" is permission for that instance, never a standing default.
- **Merging sooner is the conflict FIX, not the conflict risk.** Conflicts come from branch
  AGE. Measured 07-29: four branches (#195/#197/#198/#202) queued behind one train had all
  bumped `dispatcher.py`'s docstring version banner, guaranteeing a conflict R must abort
  on — a conflict produced purely by waiting.
- **The failure mode is drift back to caution.** On 07-29 M reverted to batching three
  times after being told twice not to, because caution feels responsible. It is not
  responsible when it costs the thing Kevin actually asked for.

**`R-A` is the `headless` registry row (board #229; the release scope itself is Kevin-granted
07-26) — releases dispatch from the board, not from a paste-block.** The brief is the task's
`description`, `assignee=R-A`, status `Todo`, and the dispatcher picks it up. `/release-brief`
is still the right tool for AUTHORING the brief; only its delivery changed.

**Board #229 changed WHO authors it.** M's job in a release now ends at marking a task `Staged`;
**R** reads that queue and owns everything after. Until #229's held `UPDATE R → live-session` is
applied and step 3's classification is done, `R` still dispatches — read `assignee=R-A` below as
the target state the train lands.

1. Working session finishes a branch → M reviews → **M marks it `Staged` and stops there**.
2. **R** (the session) reads the `Staged` queue, decides what ships together and in what order,
   and authors the brief (**`/release-brief`**) as a board task assigned to `R-A` — adapting it
   mid-flight when reality moves, which is normal, not an exception (Wave 21 took three passes).
3. The dispatcher dispatches `R-A·auto`. (A fresh human `R — release <branch> <MM-DD>` session
   pasting the brief still works and remains valid when Kevin prefers fresh human eyes.)
4. R-A executes the brief top-to-bottom: verify branch state → run gates → backup branch →
   merge PR → watch deploy → write `Deploy_Log.md` entry → report → **die**. R-A never
   starts new work, never fixes non-trivial failures, and aborts loudly on any gate failure —
   **it hands back to R, not to M.** That hand-back is the designed behaviour and the whole
   point of the split: R reconciles the red gate, the conflict or the incident, and M's context
   never sees it.
5. **R-A never restarts the dispatcher loop** — a dispatched run must not cycle the process
   that dispatched it. When a train carries a `dispatcher.py` change, the restart and its
   preflight (5) verification are **M's**, immediately after the merge.
