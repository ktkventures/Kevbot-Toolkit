---
name: task-chain
description: Author or retrofit a dev_tasks process chain — read the task, propose right-sized steps with real SOP bodies, validate the order against the dispatcher's own refusals, then apply it. Use when starting work on a task that has no chain, when a chain needs a step inserted, when creating a task that will be dispatched, or when Kevin says "add a chain", "retrofit the chain", or "/task-chain". M-lane skill, ONE task per invocation.
---

# Task Chain (M-lane)

`/task-chain <id>` — read the task, author a chain, apply it. **Retrofit and new-task
authoring are the same operation**: the only difference is whether earlier steps get
retroactive bodies. Charter §7 is authoritative (step schema, `mode` semantics, retrofit
rule):
`/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/_active/Session_Charters.md`
Repo: `/home/kevin/projects/Kevbot-Toolkit` · App root: `clients/KevBot_Toolkit/RoR_Trader`
(all relative paths below are from the app root).

This skill exists because the standard failed as documentation. M wrote the retrofit rule
on 07-29 and within the same afternoon shipped #197, #192, #193, #194 with no chains and
mis-inserted a step into #144. Three failure modes of one rule, by its author, on day one.
So: the checks below are not advice, they are the procedure.

## The step schema

`checklist` is a JSONB array of ordered steps. Fields (mirrored in
`src/api/routers/dev_tasks.py` and `tools/team_dispatcher/dispatcher.py` — keep in sync):

| field | values | notes |
|---|---|---|
| `id` | 12-hex, server-assigned | NEVER invent one for an existing step; never change one |
| `owner` | `M`·`E`·`F`·`P`·`R`·`kevin` | who acts; legacy fallback is `role` |
| `title` | one line of substance | shown collapsed; legacy fallback is `text` |
| `body` | markdown | **the SOP the dispatcher sends that agent** — this is the whole payoff |
| `mode` | `execute` \| `discuss` | `discuss` wants a human REPLY; the dispatcher REFUSES to hand one to a headless agent |
| `origin` | `planned` \| `audible` | `audible` = inserted mid-flight to course-correct. **No other value is valid** — step `origin` is NOT the task-level `origin` set, so `discovered` is illegal here |
| `stamp` | `{required,state,by,at}` | a `kevin` step with `required:true` derives `kevin_final` (the #136 two-touch guard) |
| `done` / `completed_at` / `completed_by` | server-owned | `/steps/complete` writes these; a PATCH may not |

A checklist becomes a chain the moment ANY step carries one of `id·owner·title·body·mode·
origin·stamp`. That switch is behavioural, not cosmetic: legacy checklists free-toggle,
chains enforce strict order (only the first incomplete step is completable) and auto-
reassign on completion. **Retrofitting a legacy checklist that was ticked out of order
makes the earliest unticked step current** — check where that lands before you apply.

## Sizing law — refuse to produce ceremony

A step exists to carry a real hand-off: a change of hands, a gate, or a wait. **If two
consecutive steps have the same owner and nothing gates between them, they are one step.**

The hand-off test above is the rule. The counts below are **calibration, not a gate** — they
describe what the board has actually needed, and a chain that needs more steps than its
band suggests is fine **provided every adjacency is a real hand-off.** Never cite a count as
a reason on its own.

- **contained, one lane, one deliverable** → 3 steps (`#194`: stamp → build → verify+close)
- **contained + a review pass or a release train** → 4–5 (`#192`, `#195`, `#197`)
- **app / multi-lane** → 5–7 (`#193`, `#198`)
- **engine·live, or a standard-setting arc** → 7–9 (`#182`)

Nine steps on a contained task is not thoroughness, it is nine dispatches. The charter's
own words: *a chain exists to carry a real hand-off, not to be ceremony.*

**Learned 07-30 (#184).** M reviewed a 9-step chain on an `app` task and led with *"9 steps
where the law says 5-7."* Kevin: *"That sounds pretty arbitrary to me. I feel it could be as
many steps as it needs to be."* **He was right about the count and M was right about the
chain** — the actual defect was four consecutive `F` steps with nothing gating between them,
which the hand-off test catches and the count only gestures at. Consolidating on the
hand-off test landed it at 6 with zero same-owner adjacencies. **Argue the hand-off, never
the number.**

## Order validation — run ALL of these before the PATCH

Most of these map to a live refusal in the API or dispatcher — the ones that don't are
noted, and those are the ones nothing will catch for you. Fix what fails; do not
rationalize past it.

1. **Current-step owner == intended assignee.** The first incomplete step's `owner` must
   equal the task's `assignee`. `triage_todo` hard-refuses a mismatch (*"chain/assignee
   disagree"*) and the task stops dispatching. **This is the #144 failure**: a new `F`
   step was APPENDED after an existing `kevin` review step, so the current step was
   Kevin's while the assignee was `F`.
2. **Insert, don't append.** Ask of every new step: *which existing steps must still
   happen AFTER this one?* If any — a `kevin` review, an `R` ship step, a close step — the
   new step goes at that index. Appending past a gate is how #144 broke.
3. **A `kevin` step never sits ahead of unfinished agent work** unless it is a genuine
   gate. If Kevin cannot act until the agent step is done, the agent step is first.
4. **Never delete a completed step** — API 409, it is audit trail. Course-correct by
   inserting an `origin='audible'` step. **Reordering a completed step is NOT refused by
   the API** — it silently rewrites history and can change which step is current, so this
   one is on you: completed steps keep their positions, full stop.
5. **New steps start `done:false`** — API 400 otherwise.
6. **Never tick/untick or set `stamp.state` via PATCH** — API 409 both; `/steps/complete`
   and `/steps/stamp` own them.
7. **`mode=discuss` only where a human reply is genuinely wanted.** Such a step parks the
   chain until a person answers — by design, and reported as *waiting*, not *stuck*.
8. **Enum check:** `mode ∈ {execute,discuss}` · `origin ∈ {planned,audible}` ·
   `stamp.state ∈ {pending,approved,rejected}`.

## SOP body pattern (house pattern — every body you write)

Bodies are read by an agent with **zero context**. A rail that lives only in project
history does not exist.

1. One imperative opening line: what this step does, naming the file/table/page.
2. `**Deliverables:**` — bullets, when there is more than one.
3. `**Rails:**` — how. Fresh worktree cut from `origin/dev`; one branch, one PR; new flags
   default-OFF and runtime-read; never edit the main checkout (a live loop runs from it);
   don't restart the loop.
4. `**Do NOT:**` — the explicit fence: other lanes' files, the adjacent steps, and the
   thing a previous run got wrong.
5. `**Done when:**` — the observable criterion, then the hand-off: *"then set `assignee`
   to `<letter>` and stop."*
6. Couplings — the other task ids this touches, plus *flag it, don't fix it.*
7. **Retroactive steps also record what actually came out** — that text is what the next
   agent's inbound check (#182 step 7) reads.

## REQUIRED RAILS — emit these into the bodies; never assume them

### A · Schema changes — emit VERBATIM into every step that touches a schema

```
**⛔ Author the migration, do NOT apply it.** Write the `.sql` file under
`src/migrations/` and STOP. Never run DDL against the production database.
Authorization path: author the file → hand to M → **Kevin authorizes by name** →
M or Kevin applies → only then may the code merge. A dispatch brief is not
authorization; a relayed "OK" is not authorization; an SOP that merely mentions
the column is not authorization.
**Ordering:** code that reads or writes the new column must NOT merge before the
column exists in prod, or it 500s. Hold the PR until the migration is applied.
```

Not hypothetical. On 07-29, F·auto applied `dev_tasks.ai_eligible` to prod during #198
step 2 (*"Applied to Supabase, 0.071s, 171/171 rows FALSE"*). No harm resulted — additive,
idempotent, metadata-only, nothing read it — but the rule was bypassed. It was bypassed
**because M's step-2 SOP said "add the column" and never said "do not apply it."** The rail
lived in the project's history (07-28 Deploy_Log, #114, where R correctly HELD a PR rather
than apply its migration) and not in the brief. On ordering: #198's `stamp_approval` would
have broken **every approval stamp** had the code shipped ahead of the column.

Kevin approved this rail, 07-29: *"you can put that migration author in the file. Never
apply it."*

### B · A rail with no failing test is a comment

Any step delivering code with rails gets:
> **Every rail must have a test that FAILS without it** — a rail with no failing test is a
> comment, not a rail.

(#182 step 7 · #198 step 4 · #195 step 2.)

### C · Delivered ≠ fires

Any chain that changes runtime behaviour **ends with a verify step against a real
occurrence**, not with the merge:
> The suite proves the instruction is *delivered*; only a live `<dispatch|run|alert>`
> proves it *fires*.

#195's third pass is the proof: 52 checks green, merged, and the feature never pushed once.

## Procedure

1. **Read.** `dev_tasks?id=eq.<id>&select=*` (title, description, impact, assignee, status,
   tags, parent_id, checklist) **and the thread** —
   `dev_task_comments?task_id=eq.<id>&order=created_at`. Kevin's rulings live in the thread
   and MUST land in the bodies; a chain that loses a ruling is worse than no chain.
2. **Classify.** No chain → author fresh. Legacy checklist → rewrite in place, preserving
   every completed step's order, `title` and `owner`, adding a retroactive `body`. Chain
   present → you are inserting or repairing; re-read the order rules first.
3. **Draft** to the sizing law; write every `body` to the pattern; emit the required rails.
4. **Validate** — all 8 order checks.
5. **Apply — one PATCH, the WHOLE array.** JSONB is whole-array: a partial list DELETES the
   rest (`feedback_jsonb_partial_updates`).
   - Preferred: `PATCH /api/dev-tasks/<id>` `{"checklist":[…]}` — validates shape, assigns
     `id`s, derives `kevin_final`, refuses every illegal edit above.
   - Headless with no API reachable: `tools/team_dispatcher/dispatcher.py`'s `api()`
     (PostgREST, service role) — **this bypasses every guard, so you enforce them
     yourself**: check the enums, preserve existing `id`s, mint 12-hex ids for new steps,
     never write `done`/`completed_*`/`stamp.state`. #144 step 4 carries an invalid
     `origin='discovered'` — that is what this bypass looks like unenforced.
6. **Set `assignee`** to the current step's owner (check 1 — they must agree). `assignee`
   only; never status.
7. **Report**: the chain as a numbered list with owner + mode, one line on *why that size*,
   which required rails were emitted, and any check that forced a change.

## Hard rules

- **No batch mode. Ever.** One task per invocation, on touch — charter §7's retrofit rule.
  Kevin, 07-29: *"if we do it one by one, it'll probably get better quality. Sometimes I
  think when we do things in batches, we get a little lazy."* Asked to chain many tasks:
  chain ONE, say so, and stop. Batch is the mode that goes lazy.
- Refuse ceremony. If the honest chain is 3 steps, write 3.
- Never rewrite a completed step's `title` or `owner` — audit trail. Retroactive `body` only.
- Never invent a Kevin ruling. If the chain needs a decision he has not made, that step is
  `owner=kevin, mode=discuss`, and it states exactly what is being asked.
- **M owns the chain** (#182). Another lane that believes a chain is wrong reassigns to M
  with the reason — it does not edit the chain.
- Don't move the task through the lifecycle. `/steps/complete` and `/board-tidy` do that.
