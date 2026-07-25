-- Kanban lifecycle columns (2026-07-25, board #136 — Kevin+M agreed pipeline):
--   Backlog → Scoping → Approval → Todo → In Progress → Review → Staged → Done
--   (Blocked = anywhere-exception; vision rows exempt from the pipeline).
-- Statuses stay free text in `status` (UI orders them); this migration adds the
-- three lifecycle columns. ADDITIVE + IDEMPOTENT — safe to run repeatedly and
-- safe against the pre-#136 app (whitelisted PATCH ignores unknown fields).

-- Blast-radius chip, shown next to status and prominent in the Approval view.
-- contained = this lane only, no shared surfaces · app = app-wide UI/API
-- surface · engine = engine code paths (correctness risk) · live = touches
-- live trading (deploy carefully). Values enforced by the API, not a CHECK
-- (matches `status` — free text + API whitelist).
ALTER TABLE dev_tasks
    ADD COLUMN IF NOT EXISTS impact TEXT NOT NULL DEFAULT 'contained';

-- Two-touch stamp (Approval stage): TRUE = Kevin picked "Approve + I review
-- before Done" — only Kevin signs the task off Review → Staged/Done
-- (Staged → Done release-train ships are exempt). FALSE = "Approve — M
-- closes". Set by POST /api/dev-tasks/{id}/stamp.
ALTER TABLE dev_tasks
    ADD COLUMN IF NOT EXISTS kevin_final BOOLEAN NOT NULL DEFAULT FALSE;

-- Standing approval (reserved by the 07-25 agreement for the pre-approved
-- class of work): M may route matching tasks straight through Approval
-- without a per-task stamp. No pipeline logic reads it yet — flag + Config
-- checkbox only until Kevin/M define the matching rules.
ALTER TABLE dev_tasks
    ADD COLUMN IF NOT EXISTS standing_approval BOOLEAN NOT NULL DEFAULT FALSE;
