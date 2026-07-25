-- Run History (2026-07-25) — dispatcher run lifecycle rows (board #109,
-- Registry Phase 2 / V4.12). One row per dispatch request or run of a board
-- task by a headless agent (tools/team_dispatcher/dispatcher.py).
--
-- Lifecycle (spec: requested → started → finished): the run-request API
-- endpoint inserts the row with outcome='requested' (records requested_by);
-- the dispatcher updates it on claim (run_id, started_at, outcome='running')
-- and on completion (finished_at, terminal outcome, log_tail). Organic
-- dispatches (queue-claimed, no button) insert their row at claim time with
-- requested_by NULL. outcome values: 'requested' | 'running' |
-- 'ok' | 'error' | 'lease-expired' | 'ignored' (request found ineligible).
--
-- Additive: NEW table only. Admin-only via service-role API (same posture as
-- dev_tasks): RLS on, no public policies.

CREATE TABLE IF NOT EXISTS run_history (
    id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    task_id       BIGINT      NOT NULL REFERENCES dev_tasks(id) ON DELETE CASCADE,
    agent_letter  TEXT        NOT NULL,           -- task assignee at request/claim
    run_id        TEXT,                           -- dispatcher run id, set at claim
    requested_by  TEXT,                           -- button author; NULL = organic dispatch
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at    TIMESTAMPTZ,                    -- claim/spawn time
    finished_at   TIMESTAMPTZ,                    -- reap time
    outcome       TEXT        NOT NULL DEFAULT 'requested',
    log_tail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_history_task ON run_history (task_id, requested_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_history_agent ON run_history (agent_letter, requested_at DESC);

ALTER TABLE run_history ENABLE ROW LEVEL SECURITY;
-- No public policies: only the service-role (admin) client reaches this table,
-- via admin-gated API endpoints + the local dispatcher. Service role bypasses RLS.
