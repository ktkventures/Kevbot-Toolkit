# Spec — /admin/roadmap (V4.3, read-only big-picture view)

**Author:** M·auto (dispatched, board #78) · 2026-07-25
**Implements:** Phase 2 of `Spec_Tasks_Team_Board.md` · **Builds on:** V2.4/V2.5 board UI,
V4.10/V4.12 agents registry (all merged on `origin/dev` as of 07-25)
**Purpose:** the WHY leg of the hub (Tasks = what · Roadmap = why · Agents = who).
Kevin's glanceable big picture: what phase the roadmap is in, how far each vision item
has progressed, who's on it, who acts next — and above all, **what is waiting on him**.
It is a projection of the board, never a second place to edit it.

## 0. What this page does NOT do (read first)

- **No editing of any kind.** The page performs GET requests only — no PATCH/POST/DELETE
  code paths exist in the view. Status, priority, assignee, tags, checklists, comments:
  all of it is edited on `/admin/tasks`, reached via deep link.
- **No run buttons.** `RunButton`/run-history stay on `/admin/tasks` and `/admin/agents`.
- No task creation, no delete, no comment composer, no drag-and-drop reordering.
- No new API endpoints, no migrations, no new dependencies.
- No realtime/polling — manual ↻ refresh, same as the other admin pages.

Every interaction routes to `/admin/tasks?task=<id>`; the roadmap page itself is inert.

## 1. Current state (verified 07-25 against `origin/dev`, exact paths)

Everything the page needs already ships:

- **Data + API:** `dev_tasks` via `GET /api/dev-tasks`
  (`src/api/routers/dev_tasks.py:96` — returns ALL tasks ordered by
  `priority_phase, priority_seq, created_at`; `include_done` param).
  One fetch feeds the whole page.
- **Shared atoms:** `frontend/src/views/taskBoardShared.tsx` — `Task` type, `RoleChip`,
  `NextChip`, `nextActor()` (:124), `ProgressBar` (:329), `tagChip`/`badge`,
  `STATUS_COLOR`, `NEEDS_REVIEW_TAG`, `roleColor`. Import; do not reimplement.
- **Grouping convention:** `AdminTasksPage.tsx:176-193` — `byParent` map + the vision
  predicate (top-level AND (tagged `vision` OR has subtasks)) + `looseTasks`.
- **Deep links:** `/admin/tasks?task=<id>` already opens the detail modal
  (`AdminTasksPage.tsx:102-111`); the agents page already links this way
  (`AdminAgentsPage.tsx:240`).
- **Card pattern:** `AdminAgentsPage.tsx` — section headers (tiny uppercase) +
  `repeat(auto-fill, minmax(340px, 1fr))` card grid + top-3 queue one-liners.
- **Sidebar:** Admin children, Tasks at `Sidebar.tsx:116`, Agents at `:117`.

## 2. Files (the whole PR — 3 files + 1 shared-helper lift)

- `frontend/src/app/admin/roadmap/page.tsx` — thin route wrapper (mirror
  `admin/tasks/page.tsx`).
- `frontend/src/views/AdminRoadmapPage.tsx` — the view. Single file, target ≤ ~300
  lines (per `feedback_keep_modules_visible`; split only if it stops being readable).
- `frontend/src/components/Sidebar.tsx` — insert
  `{ href: '/admin/roadmap', label: 'Roadmap' }` **between Tasks and Agents** (hub
  reads what · why · who top-to-bottom).
- **Helper lift (refactor, not duplication):** the vision predicate + `byParent`
  grouping currently live inline in `AdminTasksPage.tsx:176-193`. Lift them into
  `taskBoardShared.tsx` (e.g. `groupBoard(tasks) → { byParent, visionItems,
  looseTasks }`) and have BOTH pages consume the one implementation. The two pages
  must never disagree about what a vision item is.

## 3. Data & derivations

Single `GET /api/dev-tasks` (default `include_done=true` — Done subtasks are needed
for rollups). All derivations client-side:

- `byParent`, `visionItems`, `looseTasks` — from the lifted `groupBoard()`.
- **Rollup per vision** = subtasks `done/total`. A vision with NO subtasks but a
  non-empty `checklist` falls back to checklist rollup — same rule as the modal
  header (Phase-3 amendment #4). Neither → no bar (`ProgressBar` returns null at
  total 0).
- **Next actor** = shared `nextActor()`, verbatim.
- **Live rollup per vision** = `impacts_live` on the vision itself, plus the count of
  open subtasks with `impacts_live`.

## 4. Layout (top → bottom)

### 4.1 Header + controls
`h1` "Roadmap" + one-line subtitle ("read-only big picture — click anything to open it
on Tasks; edits happen there") + counts line: `N phases · M visions open · K awaiting
Kevin`. Slim controls Card: `↻ refresh` + `show completed` toggle (default OFF; plain
state, no persistence needed).

### 4.2 "Awaiting Kevin" strip (the headline feature)
A visually distinct strip pinned above all phases — border/accent in
`roleColor('kevin')` gold, so it reads as *his* inbox.

**Membership** — every open (`status !== 'Done'`) **leaf-level** task where:
- (a) `nextActor(t, []).next === 'kevin'` (first un-done checklist step owned by
  kevin, or assignee kevin), **or**
- (b) `tags` include `needs-review` (the tag's definition IS "finished — waiting on a
  human"; render with the 👀 chip so the reason is visible).

*Leaf-level* = subtasks + visions without subtasks. Visions WITH subtasks are
excluded on purpose: their next-actor derives from their first open subtask, so
listing both would double-count, and when a kevin-assigned subtask's own checklist
says another role acts first, the ball genuinely isn't with Kevin yet —
`nextActor(subtask)` gets that right, the parent's derived value doesn't.

**Row contents:** reason chip (`→ K` handoff style via `NextChip` styling, or 👀) ·
`#id` + title · when a subtask, parent context "under #vid <vision title>" (truncated)
· status in `STATUS_COLOR` · 🔴 if `impacts_live` · whole row deep links to
`/admin/tasks?task=<id>`. Order: API priority order, preserved.

**Empty state:** the strip stays visible as one quiet line — "Nothing waiting on
Kevin" — a visible all-clear beats silent absence (`feedback_visual_observability`).

### 4.3 Phase sections
Vision cards grouped by `priority_phase`, ascending, **0 first** — the phase number
is M-managed roadmap order (charter §7): sorted page = what's next. Within a phase,
API order (= `priority_seq`) — must match the grouped order on `/admin/tasks`.

Section header styled like the agents page department header (tiny uppercase):
`PHASE 0 — INFRA`. Labels come from a `PHASE_LABELS: Record<number, string>` const in
the view, seeded `{ 0: 'infra' }`, fallback `PHASE ${n}`. Cosmetic only, M-maintained,
one line to extend — no schema.

Done visions are hidden by default; the `show completed` toggle reveals them in place
at `opacity: 0.55`.

### 4.4 Vision cards
Grid `repeat(auto-fill, minmax(340px, 1fr))` (agents-page pattern). Per card:

- **Header row:** `#id` + title (V-titles render as-is) · status chip
  (`STATUS_COLOR`) · ⚡ if `is_urgent` · 👀 if tagged needs-review.
- **Progress:** full-size `ProgressBar` + `n/m` from the §3 rollup.
- **People:** owner `RoleChip` (the vision's assignee) first, then deduped "crew"
  chips — distinct assignees of its open subtasks, owner excluded. Then the shared
  `NextChip` (quiet `→ X` when next == assignee, alert "handoff due → X" otherwise —
  this is the Next-actor surfacing on every card, not just the Kevin strip).
- **Live surfacing:** 🔴 `live` badge when the vision itself has `impacts_live`; when
  any open subtask does, a rolled-up count chip: `🔴 n subtasks touch live`.
- **Open-subtask preview:** top 3 open subtasks as one-liners (agents-page queue
  style): `#id` title · mini status color · assignee `RoleChip` · 🔍 if
  `origin === 'discovered'` · 🔴 if `impacts_live`. Each line deep links to its own
  `?task=`; `+n more →` links to `/admin/tasks`.
- **Whole card click** → `/admin/tasks?task=<vision id>` (subtask lines
  `stopPropagation`).

### 4.5 Ungrouped footer
`looseTasks` (top-level, untagged, childless) render as a compact single Card at the
bottom — one-liner rows with deep links, header "ungrouped — tasks without a vision
parent" (mirrors the tasks page). Nothing on the board is invisible here; a silent
gap would read as "covered" when it isn't.

## 5. Implementation notes for F

- **Branch:** new branch off latest `origin/dev` (e.g. `feat/admin-roadmap`) — one PR,
  UI-only diff: the 3 files + the `taskBoardShared` helper lift.
- **Hooks rules (CLAUDE.md):** all hooks before any early return; no IIFEs; variables
  referenced in a `useMemo` declared before it in source order (Terser).
- **Read-only invariant is testable:** the view contains no `apiFetch` call with a
  `method` — GETs only.
- Error/loading/empty states mirror the tasks/agents pages (`err` Card, `Loading…`).

## 6. Acceptance criteria (gate before release-brief)

1. `/admin/roadmap` renders from a single `GET /api/dev-tasks`; grep of the view shows
   no non-GET `apiFetch` and no mutation handlers.
2. Phase sections ascend from 0; card order within a phase matches the grouped order
   on `/admin/tasks` for the same data.
3. Progress bars match the `n/m` rollups shown on `/admin/tasks` (spot-check two
   visions, one with Done subtasks). A subtask-less vision with a checklist shows the
   checklist rollup; one with neither shows no bar.
4. Awaiting-Kevin strip: assign a test subtask to kevin → it appears; mark it Done →
   it disappears; tag a task `needs-review` → it appears with 👀; a vision whose first
   open subtask is kevin-assigned does NOT appear itself (only the subtask does).
   Empty case renders the "Nothing waiting on Kevin" line.
5. `impacts_live` on a vision → 🔴 badge on its card; on only a subtask → the rolled-up
   count chip; the flagged subtask's preview line also shows 🔴.
6. Card click opens `/admin/tasks?task=<vision id>` with the detail modal up; a
   subtask preview line opens its own id; browser Back returns to `/admin/roadmap`.
7. Sidebar shows Roadmap between Tasks and Agents with correct active-state highlight.
8. `show completed` toggle: Done visions hidden by default, revealed dimmed in place
   when on.
9. Both pages consume the lifted `groupBoard()` helper — the inline predicate is gone
   from `AdminTasksPage.tsx`; tasks-page grouped view is unregressed.
10. `python src/fidelity_parity_suite.py` all green; diff touches no engine/data files.

## Non-goals (this phase)

Timeline/gantt or burndown rendering, editing of any field, run buttons/run history,
drag re-prioritization, notifications, realtime updates, persisted per-phase collapse
state, auth-role hardening (stays `get_current_user`-gated like every admin page).
