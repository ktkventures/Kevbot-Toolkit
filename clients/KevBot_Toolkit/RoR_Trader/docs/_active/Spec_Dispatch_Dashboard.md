# Spec — Dispatch dashboard: live lane view + run log (board #193, Half 1)

**Author:** M·auto (2026-07-29, board #193 Step 2) · **Implements:** F (Step 3)
**Scope:** Half 1 ONLY — the live view + the run log. The train builder (Half 2) is
out of scope and named in §9.
**Status:** ready to build. Zero schema change, zero new API endpoint, zero new
capture. Everything below is a READ over tables that already record it.

---

## 1. Problem

Kevin has to infer the state of the agent fleet from chat. His words (#193):

> what's actively being worked on, and what task + **step** each agent is on · when it
> was triggered · a historical log of when things happened · how many lanes are in use
> vs available

All of that is already in the database and none of it is on a screen. `run_history`
has recorded every dispatch since 07-25 (84 rows at spec time, ~21/day):
`task_id · agent_letter · run_id · requested_by · requested_at · started_at ·
finished_at · outcome · log_tail`. The lane math is two dispatcher constants against
those rows. The step an agent is on is the first un-done step of the task's chain —
the same computation the dispatcher itself does to build the prompt.

Two failures this week make the page load-bearing rather than decorative:

- **#197** — the reaper marked FAILED runs `Review` ("output awaiting sign-off"). The
  truth was in `run_history.outcome` and invisible everywhere. After #197 ships, a
  failed run quietly restores the task's pre-dispatch status, which makes the run log
  **the only place a failure is visible at all**. Failures must be prominent here.
- **Branch age** — board #166 ("kills the 16-28h branch-age problem"): work that is
  built but unshipped for a weekend is what produced Monday's conflicts. Age, not PR
  numbers, is the number that predicts that pain.

## 2. Placement — its own page, `/admin/dispatch`

Kevin stated no preference, so M rules: **its own page**, sidebar entry immediately
below `Agents` (`frontend/src/components/Sidebar.tsx`, admin children list).

Reasons: the Agents page is the per-agent **roster** (a card grid, "who owns what");
this is the fleet's **time axis** ("what is happening now, and what happened"). The
run log needs vertical room and filters, which a card grid cannot give it. And Kevin
wants one URL he can leave open. Reciprocal links, both one-liners:

- Agents page card → `→ dispatch` next to its RECENT RUNS heading.
- Dispatch page rows → `/admin/tasks?task=<id>` (the existing deep link) and the
  agent chip → `/admin/agents`.

## 3. Hard architectural rail — Supabase or nothing

The app runs on Railway. It **cannot** read the dispatcher's `state.json`, the git
worktrees, or GitHub. Every number on this page therefore comes from
`run_history` / `dev_tasks` / `agents` via the existing admin API, or is derived from
them in the browser.

Consequences the build must respect, not work around:

| Wanted | Available? | Rule |
|---|---|---|
| active runs | yes — `outcome='running'` rows | §4 lease rule applies |
| retry counter (#197) | **no** — lives in local `state.json` | derive "N consecutive failed runs since the last `ok`" from `run_history` instead |
| dispatcher liveness | **no** — there is no heartbeat row | never assert "dispatcher alive". §8 |
| PR / merge state | **no** — no GitHub token | show as a known gap, never as a negative. §7 |
| pushed branch | after #195 merges — `pushed_branch`, `pushed_at` | absent column = *unknown*, not *not pushed*. §7 |

## 4. Panel 1 — Lanes (the "in use vs available" answer)

Three slot tiles from `CONCURRENCY`, each either **occupied** (agent letter + task
+ elapsed) or **free**. Beside them: `runs started today / DAILY_CAP`.

**Occupancy is defined, not guessed:**

- **occupies a lane** = `outcome='running'` AND `started_at` within
  `RUN_TIMEOUT_S + 120s` grace (45m lease + 2m).
- **does NOT occupy a lane** = `outcome='running'` but older than that. The
  dispatcher would have reaped it; it did not. Render these in a **loud stale-runs
  banner** ("N runs stuck in `running` past their lease — the dispatcher may be down
  or a reap was missed"), with run_id and age, and **exclude them from the lane
  count**. Showing 3/3 saturation when nothing is running would be this page
  committing the #197 sin in a new place.
- **queued, not occupying** = `outcome='requested'` (button pressed, not yet claimed).

**One run per lane must also be visible** (`dispatcher.py` serialises per agent
letter — all of a letter's runs share one worktree). Under the slot tiles, one row
per headless lane from `/api/agents` (`status='headless'`): letter · busy/free ·
its next queued task. This is what answers *"why isn't my task running?"* — no free
slot, **or** its lane is busy, **or** it is not eligible (§6).

Dispatcher constants live in Python and cannot be read from the app. Declare them
**once**, in `taskBoardShared.tsx`, with a comment naming
`tools/team_dispatcher/dispatcher.py` (~L75) as SSOT, and label them "configured" in
the UI so a drift is legible rather than invisible:

```ts
// SSOT: tools/team_dispatcher/dispatcher.py (CONCURRENCY / DAILY_CAP /
// RUN_TIMEOUT_S). Mirrored here because the app cannot read it. POLL_S is the
// OPERATIONAL value the loop is started with (`--loop --poll 20`), not the
// module default (900) — it is used only to word the "can lag one poll" note.
export const DISPATCHER = { CONCURRENCY: 3, DAILY_CAP: 40, RUN_TIMEOUT_S: 2700, POLL_S: 20 };
```

`runs started today` = `run_history` rows with non-null `started_at` whose UTC date is
today. `requested` and `ignored` rows are requests, not runs — excluded. Label it
"started today (from run_history)": the dispatcher's own cap counts `state.json`, so
the two can differ by any run that failed to record, and the label keeps that honest.

**The cap's window is the UTC day** (`dispatcher.today_run_count()`), which rolls at
`00:00Z` = **18:00 MT** — an evening's throughput is charged to the *next* MT morning.
The lane panel must print that reset time beside the count: board #219 was filed
because a cap can otherwise be spent unknowingly overnight. `DAILY_CAP` itself is a
circuit breaker, raised 24 → 40 by Kevin on 07-30 (#219); a rolling-24h window
(#219 option 3) was NOT adopted.

## 5. Panel 2 — In flight (one card per live run)

Everything Kevin asked for, per running row:

- **task** — `#id` + title, linked to `/admin/tasks?task=<id>`
- **agent** — `RoleChip` (existing)
- **step** — `STEP n of N — "<title>" · owner · mode`, computed from
  `dev_tasks.checklist` with the **shared** helpers `isProcessChain` / `stepOwner` /
  `stepTitle` (first un-done step). Do **not** re-implement: if this page and the
  dispatcher ever disagree about the current step, the dispatcher is right by
  definition. Legacy `{role,text,done}` checklist → render `no chain — whole-task
  dispatch`. No chain at all → `no chain`.
- **triggered** — `requested_at` (absolute UTC + `relTime`), plus
  `requested_by ? 'Run button · by <who>' : 'organic queue'`
- **running for** — `elapsedShort(started_at)` and the lease remainder
  (`45m lease · 12m left`); past the lease, the stale treatment from §4
- **run_id** — monospace and copyable. It is the log filename on Kevin's machine
  (`tools/team_dispatcher/logs/<run_id>.log`), which is how he reads a live run.

If nothing is running, say so plainly (`no active runs`) — an empty panel reads as a
broken page.

## 6. Panel 3 — Queue: the two controls, deliberately separate

The conflation is the bug (#198). Two visually distinct controls, labelled as what
they are:

- **AI-eligible — a TOGGLE (input).** `dev_tasks.ai_eligible` (#198). Writes via the
  existing `PATCH /api/dev-tasks/{id}`.
- **Run state — a PILL (output).** Derived from the task's most recent `run_history`
  row: `idle` (no runs) · `queued` (`requested`) · `running` · `ok` · `failed`
  (`error` / `lease-expired`, with `×N` when N consecutive runs have failed since the
  last `ok`) · `ignored`.
- **The Run button stays exactly what it is** — a one-off queue jump. Unchanged
  behaviour, unchanged component.

**Reuse, do not fork.** #198 Step 3 (F, in flight at the time of writing) builds this
toggle and this pill for `/admin/tasks`. This page **imports the same components**.
If they land page-local inside `AdminTasksPage.tsx`, lift them into
`taskBoardShared.tsx` as part of this build. Two implementations of the input/output
distinction would recreate the exact defect the distinction exists to fix.

**Degrade, never lie.** If #198 is unmerged, `ai_eligible` is absent from the API
payload: render the eligibility column from `status === 'Todo'` alone and label it
`legacy Todo gate`. `undefined` must never render as a confident `off`.

The queue list itself: tasks whose assignee is a headless lane, split into
**dispatch-eligible now** and **waiting — with the reason**, using the existing
`runIneligibleReason()` (it already mirrors the dispatcher's gates). Note in the UI
that the mirror is a courtesy — the dispatcher re-gates at claim.

## 7. Panel 4 — Shipping lane: built → pushed → PR'd → merged

The Monday-conflict panel. One row per task carrying work that has not shipped,
**oldest first**, with **age as the headline number** — age is the pain, not the PR
number.

| Stage | Source | Honest today? |
|---|---|---|
| **built** | latest run `outcome='ok'` → `finished_at` | yes |
| **pushed** | `run_history.pushed_branch` / `pushed_at` (#195) | yes once #195 merges; NULL/absent = **unknown** |
| **PR'd** | — | **NOT TRACKED.** Needs `pr_number` on `dev_tasks` + a GitHub token (Half 2) |
| **merged / shipped** | board state: `Staged` = reviewed, awaiting a train · `Done` = shipped | proxy, labelled as board state |

So the pipeline this page can actually draw is **built → pushed → staged → done**,
with `PR'd` rendered as a greyed `—` whose tooltip names precisely what unlocks it.
State the substitution on the page in one line; do not let a board proxy pass as git
truth.

Two rails:

1. **`pushed` unknown ≠ not pushed.** Pre-#195 runs can never be known. An absent
   column or NULL value renders as a distinct grey `unknown`, never a red `not
   pushed`. A false red on this panel would send Kevin looking for a branch that
   shipped days ago.
2. **Never scrape comment prose for `PR #NN`.** Branch names and PR numbers live in
   prose precisely because there is no field; a scraped number goes stale silently,
   which is the failure mode this whole page exists to remove.

Age thresholds, calibrated on board #166's measured 16-28h: `<12h` neutral ·
`≥12h` amber · `≥24h` red.

## 8. Known gaps — printed ON the page

A short footer section, three lines, styled as a note. A dashboard whose blind spots
are invisible manufactures false confidence (the #157 "0 healthy" dead-alarm lesson):

1. **Dispatcher liveness is not observable here.** There is no heartbeat row. "Last
   run event 40m ago" is *not* proof the loop is alive — a quiet queue emits nothing.
   Word it as *last run event*, never as *dispatcher alive*. (Follow-up: §9.)
2. **PR / merge state is not tracked** (§7).
3. **Retry counters live in the dispatcher's local `state.json`** — the consecutive-
   failure count shown here is derived from `run_history` and can differ.

## 9. Non-goals (explicitly out of scope for this build)

- The **train builder** (#193 Half 2): PR readiness, conflict/CI state, a
  "push this set to dev" button. Needs structured `branch`/`pr_number` fields **and**
  a GitHub credential the API does not have — a credential decision, not a UI one.
- **Dispatcher heartbeat.** Worth a task of its own (a `dispatcher_heartbeat` row the
  loop touches every pass would make §8.1 answerable for ~10 lines). Recommended
  follow-up; not built here.
- Killing / cancelling a run from the UI — the app cannot reach Kevin's machine.
- Full run logs. Only `log_tail` (4000 chars) is in the database.
- Any new write path. This page mutates exactly two things, both already existing:
  `ai_eligible` (PATCH) and the run request (POST).

## 10. Data & fetch plan (no API change)

Three existing GETs per tick, joined client-side:

| Call | Why |
|---|---|
| `GET /api/run-history?limit=400` | ~19 days at the current ~21 runs/day (API max 500) |
| `GET /api/dev-tasks?include_done=true` | **`include_done=true` is required** — the run log references Done tasks; filtering them renders `#NNN (unknown)` |
| `GET /api/agents` | headless lane set + letters for §4 |

- **Auto-refresh 15s**, paused while `document.hidden`, with an `as of HH:MM:SSZ`
  stamp and a manual Refresh. Note on the page that the dispatcher polls every
  ~`POLL_S`, so a state change can be up to one poll behind — otherwise a 20s lag
  reads as a bug.
- **Three list fetches per tick, joined in memory — never a per-row fetch.** That is
  the #148 poll-efficiency lesson; a fan-out over 84+ runs would be worse than the
  problem it replaced. `log_tail` already rides in the row, so expanding a tail costs
  no request.
- All timestamps render **UTC with a `Z`**, paired with `relTime()` for the "ago".
  The deploy log, the dispatcher logs and Kevin's own chat are all UTC; a local-time
  render on this page would desync him from every other surface.

## 11. Files

| File | Change |
|---|---|
| `frontend/src/views/AdminDispatchPage.tsx` | new — the page |
| `frontend/src/app/admin/dispatch/page.tsx` | new — `dynamic(..., { ssr: false })`, mirroring `admin/agents/page.tsx` |
| `frontend/src/components/Sidebar.tsx` | one line: `{ href: '/admin/dispatch', label: 'Dispatch' }` after Agents |
| `frontend/src/views/taskBoardShared.tsx` | add `DISPATCHER` constants; lift #198's toggle/pill here if they landed page-local |
| `frontend/src/views/AdminAgentsPage.tsx` | one reciprocal `→ dispatch` link |

Reuse from `taskBoardShared.tsx` — do not re-create: `RunRow`, `Task`,
`ChecklistStep`, `isProcessChain`, `stepOwner`, `stepTitle`, `runIneligibleReason`,
`OutcomeChip`, `OUTCOME_STYLE`, `RoleChip`, `roleColor`, `relTime`, `elapsedShort`,
`badge`, `tagChip`, `STATUS_COLOR`.

**Production-bundle rails (CLAUDE.md, non-negotiable — these have bitten this app
before):** every `useMemo`/`useState`/`useEffect` before any early return; a variable
must be declared before the `useMemo` that references it; no IIFEs for
initialisation. The prototype's queries are correct and reusable
(`scratchpad/mksvg.py`: `run_history` order `id.desc`, the `active`/`requested`
split, the lane count against `CONCURRENCY`); its rendering is throwaway SVG.

## 12. Acceptance

Checkable on dev, in order:

1. With a run in flight, the page shows it in a lane slot, `lanes 1/3`, and a step
   line **matching what the dispatcher's own prompt said** for that run (`STEP n of
   N`). Cross-check against the run's log file.
2. A `running` row older than 47m appears in the stale-runs banner and is **excluded**
   from the lane count.
3. A failed run (`outcome='error'`) is findable in the run log without filtering —
   tinted row, counted in its day header — and the page does not describe it as
   having reviewable output.
4. The toggle and the pill are visually distinct, separately labelled input/output,
   and the toggle round-trips through `PATCH /api/dev-tasks/{id}`.
5. Run button behaviour is byte-identical to today.
6. **With #195 and #198 unmerged** (columns absent), the page renders with no console
   errors: `pushed` shows `unknown`, eligibility falls back to the labelled legacy
   Todo gate. Verify before either merges — this is the graceful-degradation test.
7. The shipping panel lists the oldest unshipped built work first, with the age
   colour thresholds, and `PR'd` as a greyed `—` with its tooltip.
8. Known-gaps footer present and worded per §8 (no "dispatcher alive" claim).

## 13. Lane & shipping

Frontend-only; org tooling, F-implemented per M's spec (PR #71 precedent). No engine
or data files, no migration, no API change, no flag — a new page is inert until
visited. Ships on a normal release train via R. Smoke: load `/admin/dispatch` on dev
with a run in flight and confirm acceptance 1, 3 and 6.

**Dependencies — soft, all of them.** #195 (`pushed_branch`) enriches §7, #198
(`ai_eligible`, toggle + pill) supplies §6's components, #197 makes §5/§6 truthful
about failures. The page must build and ship without any of them (acceptance 6);
none is a blocker.
