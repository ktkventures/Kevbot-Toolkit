-- Agents Registry (2026-07-23) — Spec_Agents_Registry.md Phase 1 (board #98/#99).
-- The WHO leg of the hub: a durable roster of the multi-session team, and the
-- data model the V4.9 dispatcher will consume. Additive: NEW table only.
--
-- SSOT rule (spec header): until the dispatcher ships, Session_Charters.md §1
-- stays authoritative and this table MIRRORS it — scope/boundaries below are
-- copied verbatim. Never edit one without the other.
--
-- Admin-only via service-role API (same posture as dev_tasks): RLS on, no
-- public policies.

CREATE TABLE IF NOT EXISTS agents (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    letter       TEXT NOT NULL UNIQUE,            -- 'M','E','E2','F','P','R','kevin'
    name         TEXT NOT NULL,                   -- 'Engine/Divergence'
    kind         TEXT NOT NULL DEFAULT 'agent',   -- 'agent' | 'human'
    department   TEXT NOT NULL DEFAULT 'dev',     -- 'dev' | 'builder' | 'marketing' | 'ops'
    status       TEXT NOT NULL DEFAULT 'dormant', -- 'live-session' | 'headless' | 'dormant' | 'retired' | 'ephemeral'
    scope        TEXT NOT NULL DEFAULT '',        -- owns (mirrors charter Owns column)
    boundaries   TEXT NOT NULL DEFAULT '',        -- must-not-touch (mirrors charter)
    worktree     TEXT NOT NULL DEFAULT '',        -- absolute path or '' (main checkout)
    context_docs TEXT[] NOT NULL DEFAULT '{}',    -- standing-context doc paths
    prompt_template TEXT NULL,                    -- Phase 2 (dispatcher identity prompt)
    notes        TEXT NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Same touch-trigger pattern as dev_tasks.
CREATE OR REPLACE FUNCTION agents_touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_agents_touch ON agents;
CREATE TRIGGER trg_agents_touch
    BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION agents_touch_updated_at();

ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
-- No public policies: only the service-role (admin) client reaches this table.

-- ── Seed (idempotent) — scope/boundaries VERBATIM from Session_Charters.md §1;
--    E2 is a roster §2 subordinate (not in §1) — see its notes.
INSERT INTO agents (letter, name, kind, department, status, scope, boundaries, worktree, context_docs, notes) VALUES
(
  'M', 'Manager/Coordinator', 'agent', 'dev', 'live-session',
  'This charter + roster; `Team_Board.md`; roadmap/organization docs; specs for org tooling (tasks-page upgrade, admin roadmap page)',
  'Engine/data code; flags; deploys; heavy prod-DB; E''s operational logs (`STATUS.md`, `Deploy_Log.md`, hunt logs) — M proposes restructures via spec, never live-edits E''s logs',
  '',
  ARRAY['docs/_active/Session_Charters.md','docs/_active/Spec_Tasks_Team_Board.md','docs/_active/Spec_Agents_Registry.md'],
  ''
),
(
  'E', 'Engine/Divergence', 'agent', 'dev', 'live-session',
  'Engine/data-path code; ALL flag flips; Railway deploys (`railway variables`, `railway up`); heavy prod-DB analysis; operational logs: `STATUS.md`, `Deploy_Log.md`, `Divergence_Hunt_Log.md`; project memory',
  '—',
  '',
  ARRAY['docs/_active/Session_Charters.md','docs/_active/STATUS.md','docs/Roadmap_Divergence_Hunting.md','docs/_active/Divergence_Hunt_Log.md'],
  ''
),
(
  'E2', 'M-RS5a rollout & shadow lane', 'agent', 'dev', 'live-session',
  'Subordinate of E — M-RS5a rollout & shadow lane. Delegated authority: shadow-worker/M-RS5a lane flags + railway-up deploys (fingerprint SOP)',
  'E-lane rules; delegated slice only — nothing beyond the shadow-worker/M-RS5a lane',
  '/home/kevin/projects/kevbot-wt-submin',
  ARRAY['docs/_active/Session_Charters.md','docs/_active/Impl_MRS5a_Resident_Window.md'],
  'Roster §2 subordinate (charter §1 covers lane parents only); retires after M-RS5a fix lands + POLL_S=5 restored'
),
(
  'F', 'Frontend', 'agent', 'dev', 'live-session',
  'Portfolio pages, Strategy Health / health-overview UI, reporting UI, tasks-page implementation (per M''s spec) — Next.js app UI code; its own plan docs',
  'Engine/data files (e.g. `bar_cache.py`, `strategy_data.py`, `forward_test_service.py`, `fidelity_parity_suite.py`, worker/recompute lanes); flags; deploys; E''s operational logs',
  '/home/kevin/projects/Kevbot-frontend',
  ARRAY['docs/_active/Session_Charters.md','docs/_active/Spec_Tasks_Team_Board.md','docs/_active/Spec_Agents_Registry.md'],
  ''
),
(
  'P', 'Packs', 'agent', 'dev', 'dormant',
  'User-pack accuracy (S/R pack first), new packs, pack-builder AI — pack definitions, pack-builder code; its own plan docs',
  'Same exclusions as F. Plus: pack edits change backtests → recomputes → shift paired-% baselines. Coordinate TIMING with E before any pack change lands.',
  '',
  ARRAY['docs/_active/Session_Charters.md'],
  'Spawn after M-RS5a flip settles + divergence board green'
),
(
  'R', 'Release', 'agent', 'dev', 'ephemeral',
  'Ephemeral gatekeeper: validate → merge → deploy-watch → log → die. Nothing persistent',
  'Never starts feature work; never flips flags beyond what the brief specifies',
  '',
  ARRAY['docs/_active/Session_Charters.md'],
  'Named `R — release <branch> <MM-DD>`; killed after merge+log; never reused'
),
(
  'kevin', 'Kevin', 'human', 'dev', 'live-session',
  'Product owner — final decision authority, sign-offs, flag-authority grants, session spawning',
  '',
  '',
  ARRAY['docs/_active/Session_Charters.md'],
  ''
)
ON CONFLICT (letter) DO NOTHING;
