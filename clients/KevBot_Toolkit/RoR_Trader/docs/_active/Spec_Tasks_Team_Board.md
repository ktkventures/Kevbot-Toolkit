# Spec — Tasks page as the Team Board (V4.2)

**Author:** M — Manager/Coordinator · 2026-07-22
**Implements:** board V2.4 (F builds) · **Pairs with:** V4.3 admin roadmap page (Phase 2 below)
**Goal:** make `/admin/tasks` the durable home base for the multi-session team — replacing
`docs/_active/Team_Board.md` — so vision items, their subtasks, and role assignments live in
the UI where Kevin can actually see them, and rabbit-hole fixes stay visibly parented under
the big-picture item that spawned them.

## Current state (verified 07-22, exact paths)

Good news: ~70% exists. `dev_tasks` Supabase table + trigger
(`src/migrations/dev_tasks_table.sql`), CRUD API (`src/api/routers/dev_tasks.py`, prefix
`/api/dev-tasks`, field whitelist at `:14-18`), and a working UI
(`frontend/src/views/AdminTasksPage.tsx`, route `frontend/src/app/admin/tasks/page.tsx`)
with status/priority/area/assignee editing, filters, and per-task comments.

Missing for team-board use: **(a)** no parent/subtask hierarchy (only a `blocked_by BIGINT[]`
dependency array, read-only in UI); **(b)** assignee is `'kevin' | 'claude'`
(`AdminTasksPage.tsx:36`) — predates roles; **(c)** `tags TEXT[]` exists in schema/API but is
never rendered; **(d)** no notion of a "vision" tier; **(e)** comment author hardcoded
`'kevin'` in UI (`AdminTasksPage.tsx:109`).

## Phase 1 — team board (one PR)

### 1. Migration (additive only — no breaking changes)
New file `src/migrations/dev_tasks_team_board.sql`:
- `ALTER TABLE dev_tasks ADD COLUMN parent_id BIGINT NULL REFERENCES dev_tasks(id) ON DELETE CASCADE;`
- `ALTER TABLE dev_tasks ADD COLUMN origin TEXT NOT NULL DEFAULT 'planned';`  -- 'planned' | 'discovered' | 'kevin'
- `CREATE INDEX idx_dev_tasks_parent ON dev_tasks(parent_id);`
- Convention (no schema needed): a **vision item** is a top-level task (`parent_id IS NULL`)
  tagged `vision`. One nesting level only in v1 (vision → subtask); enforce in API, not DB.

### 2. API (`src/api/routers/dev_tasks.py`)
- Add `parent_id`, `origin` to the editable-field whitelist.
- `POST`/`PATCH` validation: reject `parent_id` pointing at a task that itself has a
  `parent_id` (one level only, fail loud per project convention).
- `GET` unchanged (flat list; client groups). Add optional `?assignee=` filter param —
  this is what role sessions poll at session start.
- `POST .../comments`: accept `author` from the request body (already defaults `'claude'`);
  UI passes the role name.

### 3. UI (`frontend/src/views/AdminTasksPage.tsx`)
- **Assignee options** → `['', 'M', 'E', 'E2', 'F', 'P', 'R', 'kevin']` — but read the list
  from a small const so adding roles is one line. Keep legacy values rendering as-is.
- **Grouped view (default):** vision items as group headers (title + rollup: n/m subtasks
  done), subtasks indented beneath, ordered by existing `priority_phase.seq` within groups.
  Keep the current flat table as a toggle ("Flat" / "By vision").
- **"+ subtask" action** on vision rows (pre-fills `parent_id`); "+ New task" unchanged but
  gains a parent picker (default: none = vision item, auto-tag `vision`).
- **Render `tags`** as chips in the row + editable in the detail modal (comma input is fine).
- **`origin` badge** on subtasks: `discovered` renders a small 🔍 chip — this is the
  rabbit-hole marker; `planned` renders nothing.
- **`blocked_by` editing** in the detail modal (comma-separated ids); keep the ⛔ indicator.
- Comment author: dropdown of the same role list (persist last choice in localStorage)
  instead of hardcoded `'kevin'`.

### 4. Data migration (one-time, manual — M does this after ship)
M imports `Team_Board.md` (V1–V4 → vision items; rows → subtasks with assignee/status/notes),
then `Team_Board.md` gets a tombstone header pointing at `/admin/tasks` and moves out of
`_active/`. Charter §7 flips from "interim markdown" to "the Tasks page"; role sessions then
check `GET /api/dev-tasks?assignee=<role>&include_done=false` at session start.

### Acceptance (gate before release-brief)
1. Migration applies cleanly on dev DB; existing rows unaffected (`parent_id NULL`,
   `origin 'planned'`).
2. Create vision → add 2 subtasks (one `discovered`) → grouped view shows rollup 0/2;
   complete one → 1/2. Flat toggle still works.
3. Two-level nesting attempt returns a 4xx with a clear message.
4. Assignee filter + `?assignee=E` API param return matching sets.
5. No engine/data files touched; `python src/fidelity_parity_suite.py` all green
   (UI/API/migration only, but the gate is the gate).

## Phase 2 — `/admin/roadmap` (V4.3, separate PR, after Phase 1)
Read-only page at `frontend/src/app/admin/roadmap/page.tsx` + Sidebar entry
(`frontend/src/components/Sidebar.tsx` `navItems` children, next to Tasks at `:116`):
vision items as cards (progress bar from subtask rollup, assignee chips, `impacts_live`
flags surfaced), ordered by priority. No editing — it's Kevin's big-picture view; edits
happen on /admin/tasks. Detail spec follows Phase 1 ship.

## Non-goals (v1)
Real-time updates, drag-and-drop ordering, auth-role hardening (endpoint stays
`get_current_user`-gated as today), deeper nesting, notifications.

## Notes for F
- Branch: `feat/tasks-team-board` (the F worktree is already on it) — Kevin reordered
  07-22: this ships FIRST; health-overview (V2.2) follows on a new branch after this PR
  closes (one PR = one branch's life).
- The seed file `src/migrations/dev_tasks_seed.sql` shows the existing content style —
  don't clobber live rows; the migration is additive.
- `AdminTasksPage.tsx` is a single 309-line file — keep it one file if it stays readable,
  split view components only if the grouped view pushes it past ~500 lines
  (per `feedback_keep_modules_visible`).
