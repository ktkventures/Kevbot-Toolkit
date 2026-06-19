-- Dev Task Tracker (2026-06-19) — a ClickUp-lite shared board for Kevin + Claude
-- to track work without losing things in a sea of docs. Admin-only, accessed
-- exclusively through admin-gated FastAPI endpoints using the service-role
-- client, so RLS is enabled with no public policy (service role bypasses it).
--
-- Priority model (Kevin's scheme): phase.seq, sorted ASCENDING so 1.1 = "do
-- next" sits at top. phase = management tier (1 = MVP scope); seq = execution
-- order within the phase (decimal allowed, so 1.15 slips between 1.1 and 1.2
-- without renumbering). "Urgent" is a TAG, not a priority level. Dependencies
-- are a separate blocked_by link, not encoded in the number.

CREATE TABLE IF NOT EXISTS dev_tasks (
    id              BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    title           TEXT        NOT NULL,
    description     TEXT        DEFAULT '',
    -- Backlog | Todo | In Progress | Blocked | Done  (free text; UI orders them)
    status          TEXT        NOT NULL DEFAULT 'Backlog',
    priority_phase  INTEGER     NOT NULL DEFAULT 1,
    priority_seq    NUMERIC     NOT NULL DEFAULT 99,
    is_urgent       BOOLEAN     NOT NULL DEFAULT FALSE,
    -- 🔴 touches the live engine / trading vs 🟢 safe/offline
    impacts_live    BOOLEAN     NOT NULL DEFAULT FALSE,
    -- flagged when a shipped change can't be trusted until live-market validation
    needs_live_validation BOOLEAN NOT NULL DEFAULT FALSE,
    -- engine | backtest | frontend | infra | docs | data | other
    area            TEXT        DEFAULT 'other',
    -- 'kevin' | 'claude' | NULL
    assignee        TEXT,
    -- task ids this is blocked by (dependency links)
    blocked_by      BIGINT[]    DEFAULT '{}',
    tags            TEXT[]      DEFAULT '{}',
    -- freeform notes / commit refs / validation steps
    notes           TEXT        DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

-- Primary sort: phase then seq, ascending → 1.1 surfaces first.
CREATE INDEX IF NOT EXISTS idx_dev_tasks_priority
    ON dev_tasks (priority_phase, priority_seq);
CREATE INDEX IF NOT EXISTS idx_dev_tasks_status ON dev_tasks (status);

CREATE TABLE IF NOT EXISTS dev_task_comments (
    id          BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    task_id     BIGINT      NOT NULL REFERENCES dev_tasks(id) ON DELETE CASCADE,
    author      TEXT        NOT NULL DEFAULT 'claude',   -- 'kevin' | 'claude'
    body        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dev_task_comments_task ON dev_task_comments (task_id);

-- Keep updated_at fresh on any task change.
CREATE OR REPLACE FUNCTION dev_tasks_touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    IF NEW.status = 'Done' AND (OLD.status IS DISTINCT FROM 'Done') THEN
        NEW.completed_at = now();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_dev_tasks_touch ON dev_tasks;
CREATE TRIGGER trg_dev_tasks_touch
    BEFORE UPDATE ON dev_tasks
    FOR EACH ROW EXECUTE FUNCTION dev_tasks_touch_updated_at();

ALTER TABLE dev_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE dev_task_comments ENABLE ROW LEVEL SECURITY;
-- No public policies: only the service-role (admin) client reaches these,
-- via admin-gated API endpoints. Service role bypasses RLS.
