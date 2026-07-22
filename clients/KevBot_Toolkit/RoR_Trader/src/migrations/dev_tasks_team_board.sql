-- Team Board upgrade (2026-07-22) — additive-only columns so /admin/tasks can
-- replace docs/_active/Team_Board.md as the multi-session team's home base
-- (Spec_Tasks_Team_Board.md, Phase 1). Existing rows are untouched: parent_id
-- stays NULL, origin defaults to 'planned'.
--
-- Hierarchy convention: a "vision item" is a top-level task (parent_id IS NULL)
-- tagged 'vision'. One nesting level only in v1 (vision -> subtask) — enforced
-- in the API, not the DB, per spec.

-- Parent link for the vision -> subtask hierarchy. ON DELETE CASCADE: deleting
-- a vision item removes its subtasks (they only make sense under their parent).
ALTER TABLE dev_tasks
    ADD COLUMN IF NOT EXISTS parent_id BIGINT NULL
        REFERENCES dev_tasks(id) ON DELETE CASCADE;

-- 'planned' | 'discovered' | 'kevin' — 'discovered' is the rabbit-hole marker:
-- work found mid-task, parented under the vision item that spawned it.
ALTER TABLE dev_tasks
    ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'planned';

CREATE INDEX IF NOT EXISTS idx_dev_tasks_parent ON dev_tasks(parent_id);
