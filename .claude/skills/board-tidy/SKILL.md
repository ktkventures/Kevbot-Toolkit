---
name: board-tidy
description: Dev Task Tracker tidy-up — sweep /admin/tasks (dev_tasks) into compliance with the kanban lifecycle protocol (statuses honest, impact set, chains present, fragments merged), refill Kevin's Approval column, and report. Use when Kevin asks to tidy/clean up the board, after any big work arc, or weekly. M-lane skill.
---

# Board Tidy (M-lane)

Sweep the live board into compliance with the lifecycle protocol (charter §7). The
board API pattern: PostgREST via service-role creds from `src/.env` (see the
`board_api.py` helper pattern; tables `dev_tasks`, `dev_task_comments`, `run_history`).

## The status law (charter §7 is authoritative; this is the decision tree)

For every non-Done task, in order:
1. **Actually executing?** (run_history `running`, a session mid-work, or tagged
   `standing` watch) → In Progress. Otherwise it must NOT be In Progress — stale
   In Progress blocks Run buttons and lies to Kevin.
2. **Output on thread awaiting sign-off?** → Review (Kevin closes iff `kevin_final`,
   else M reviews then closes or stages).
3. **Reviewed, ship-pending?** → Staged (trains ship everything Staged).
4. **Scoped** (description = purpose + plan readable by Kevin; `impact` set; chain
   present; parent vision; honest priority) **and awaiting Kevin's stamp?** → Approval.
5. **Needs M scoping work or a Kevin conversation?** → Scoping (Kevin-conversation
   items stay here — never push them to Approval).
6. **Captured/parked?** → Backlog (deliberately-parked items keep the `parked` tag).
7. Todo only ever means: stamped + dispatch-eligible.

## Hygiene rules

- Every task: `impact` (contained | app | engine | live), a parent vision, honest
  `priority_phase.seq`.

### ⛔ NEVER write a process chain. Detect and REPORT. (Kevin, 07-30)
This skill previously said *"every task: a process chain (default: execute → M review →
close/ship)"* and *"batch the PATCHes (checklist = whole-array JSONB)"*. **Both are now
forbidden**, for two reasons Kevin named:

> *"I just don't want it to override the task chain... I'm just kind of worried about doing
> task chain in bulk because I just feel like it would produce lazy instructions rather
> than well-researched ones."*

1. **A default template chain is the lazy chain.** The whole payoff of a chain is that each
   step's `body` IS the SOP the dispatcher sends that agent. A generic three-step template
   ships an empty prompt with the shape of a real one — worse than no chain, because it
   *looks* scoped. `/task-chain`'s own hard rule is **"No batch mode. Ever."**, which came
   from this same instruction; a sweep that authors chains in bulk violates it wholesale.
2. **A whole-array `checklist` PATCH can DESTROY an existing chain.** JSONB is whole-array —
   a partial list deletes the rest (memory `feedback_jsonb_partial_updates`). Combined with
   batching, one bad sweep silently wipes hand-authored SOPs across the board.

**So: this skill MUST NOT PATCH `checklist` at all — not to add, not to fix, not to
reorder.** Chains belong to `/task-chain`, one task per invocation, on touch (charter §7's
retrofit rule).

**What to do instead:** report a **CHAIN GAPS** section — the tasks with no chain or an
obviously broken one (current-step owner ≠ assignee, a `kevin` step ahead of unfinished
agent work), **ordered by which are closest to being worked on**. Kevin or M then runs
`/task-chain <id>` per task, deliberately. A Backlog idea does not need a chain at all; it
earns one when it enters Scoping.
- **Merge** fragments too small to be their own run (one branch of related trivia =
  one task); close absorbed ids with a "MERGED into #N" note.
- **Split** anything with >1 independent deliverable.
- Visions are pipeline-exempt (In Progress = lane active is fine).
- Never change: Kevin-stamped decisions, Staged manifests, engine-cargo policy notes.

## Procedure

1. Fetch all non-Done tasks; print a compact table (id/status/assignee/impact/tags).
2. Walk the decision tree per task. PATCH `status`, `assignee`, `impact`, `priority_*`,
   `parent_id`, `tags`. **NEVER PATCH `checklist`** — see the hygiene rule above.
3. Merges: create the bundle task first, then close the absorbed ids with pointers.
4. **Look for SUPERSEDED tasks** — the failure mode Kevin named 07-30: *"sometimes we start
   working on these, and then we implement a different solution, and then these kinds of get
   left open."* For each open task, ask whether its stated ROOT CAUSE still exists. **#166 is
   the worked example:** its cause was *"integration requires Kevin"*, fixed sideways when R
   became headless a day before the task was even filed; it sat open for three days because
   nothing re-checked the premise. Report these as **PREMISE CHECK** with the evidence, and
   recommend — never close a task on this basis without Kevin.
5. Report to Kevin: demotions (Run buttons freed), merges/splits, **CHAIN GAPS** (for
   one-at-a-time `/task-chain`), **PREMISE CHECK** candidates, the new Approval count
   ("your inbox"), and what sits in Scoping awaiting his conversations.
5. If statuses/semantics changed since last run, update charter §7 FIRST, then sweep.

## Cadence

After every major arc (release train, multi-run harvest) and at least weekly. The
sweep is idempotent — running it twice changes nothing the second time.
