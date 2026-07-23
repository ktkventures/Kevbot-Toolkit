-- Task-detail checklist (2026-07-22) — Spec_Tasks_Team_Board.md Phase 3B.
-- Additive only. Leaf tasks get a lightweight process checklist rendered in the
-- three-panel modal: array of {"text": str, "done": bool, "role": str|null}.
-- Steps deliberately have NO per-step comment thread — a step that needs one
-- gets promoted to a real subtask (the one-level nesting rule holds).
--
-- ⚠ JSONB semantics: the API/UI always write the WHOLE array on PATCH
-- (partial JSONB merges wipe — see memory feedback_jsonb_partial_updates).

ALTER TABLE dev_tasks
    ADD COLUMN IF NOT EXISTS checklist JSONB NOT NULL DEFAULT '[]'::jsonb;
