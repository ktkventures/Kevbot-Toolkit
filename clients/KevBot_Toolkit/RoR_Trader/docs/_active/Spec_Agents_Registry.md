# Spec — Agents Registry + /admin/agents (V4.10, Phase 1)

**Author:** M · 2026-07-23 · **Board:** #98 (spec, M) → #99 (build, F, V2.6)
**Purpose:** the WHO leg of the hub (Tasks = what · Roadmap = why · Agents = who).
Phase 1 is a read-only roster + the data model the V4.9 dispatcher will consume.
**SSOT rule:** until V4.9 lands, `Session_Charters.md` §1 remains authoritative and this
registry MIRRORS it; when the dispatcher ships, authority inverts (registry becomes SSOT,
charter §1 derives from it). Never edit scope in one place without the other.

## 1. Migration (additive — new table only): `src/migrations/agents_registry.sql`

```sql
CREATE TABLE agents (
  id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  letter       TEXT NOT NULL UNIQUE,          -- 'M','E','E2','F','P','R','kevin'
  name         TEXT NOT NULL,                 -- 'Engine/Divergence'
  kind         TEXT NOT NULL DEFAULT 'agent', -- 'agent' | 'human'
  department   TEXT NOT NULL DEFAULT 'dev',   -- 'dev' | 'builder' | 'marketing' | 'ops'
  status       TEXT NOT NULL DEFAULT 'dormant', -- 'live-session' | 'headless' | 'dormant' | 'retired' | 'ephemeral'
  scope        TEXT NOT NULL DEFAULT '',      -- owns (mirrors charter Owns column)
  boundaries   TEXT NOT NULL DEFAULT '',      -- must-not-touch (mirrors charter)
  worktree     TEXT NOT NULL DEFAULT '',      -- absolute path or '' (main checkout)
  context_docs TEXT[] NOT NULL DEFAULT '{}',  -- standing-context doc paths
  prompt_template TEXT NULL,                  -- Phase 2 (dispatcher identity prompt); nullable now to avoid a second migration
  notes        TEXT NOT NULL DEFAULT '',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- reuse the dev_tasks touch-trigger pattern for updated_at
```

RLS same as dev_tasks (service-role only). Seed rows: M, E, E2, F, P (dormant),
R (ephemeral), kevin (kind=human) — scope/boundaries copied from charter §1 verbatim.

## 2. API: `src/api/routers/agents.py` (mirror dev_tasks router pattern)

- `GET  /api/agents?active=true` → rows; `active` filters out retired.
- `POST /api/agents`, `PATCH /api/agents/{id}` — whitelist all columns except id/created.
  (UI stays read-only in Phase 1; the CRUD API exists so M manages rows
  programmatically, same as the board.)
- Register in `src/api/main.py` next to dev_tasks.

## 3. UI (read-only this phase)

- `frontend/src/app/admin/agents/page.tsx` + view component; Sidebar entry next to
  Tasks (`frontend/src/components/Sidebar.tsx` navItems children).
- Card per agent: letter chip (same colors as task-modal role chips), name, department
  chip, status chip, scope/boundaries (collapsed, expandable), worktree, context-doc
  links, and **current queue** — open tasks joined client-side from
  `GET /api/dev-tasks?assignee=<letter>` (count + top 3 titles, click → /admin/tasks).
- Group cards by department; 'human' kind renders with a distinct accent.

## 4. Assignee wiring (the immediate payoff)

- Task UI's assignee options (`ROLE_OPTIONS` const from V2.4) now load from
  `GET /api/agents?active=true` (letters, ordered), with the const as fallback if the
  fetch fails. Comment-author dropdown: same source.
- Session-start hook: NO change this phase (letters still match). Phase 2: hook reads
  the registry for role resolution instead of `.claude/role` files.

## 5. Acceptance (gate before release-brief)

1. Migration + seed applies; 7 rows; dev_tasks untouched.
2. /admin/agents renders all rows grouped by department with live queue counts that
   match /admin/tasks filtered by that assignee.
3. Assignee + comment-author dropdowns show registry letters; kill the API → fallback
   const still works (no broken selects).
4. PATCH an agent's status → card chip updates on refresh; updated_at moves.
5. Parity suite all green; no engine/data files touched.

## Non-goals (Phase 1)
Dispatch buttons, run history, prompt-template editing UI, permissions profiles,
builder/marketing functionality (departments exist as VALUES only), auth hardening.

## Phase 2 pointer (gated on V4.9 dispatcher design)
prompt_template + permissions profile consumed by the dispatcher; "Run task" button;
run history table; registry becomes SSOT with charter §1 generated from it.
