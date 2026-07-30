-- Run History → pushed branch (2026-07-29, board #195) — push at run end.
--
-- A headless agent commits but cannot push, so the dispatcher pushes the run's
-- own branch at reap time. These two columns record WHAT it pushed, so the
-- dispatcher dashboard (board #193) can show built → pushed → PR'd → merged
-- without re-deriving any of it from git.
--
-- Additive + nullable: NULL means "this run pushed nothing" — no push attempted,
-- declined by a rail (dirty tree, protected branch, not this run's branch), or
-- the push failed. The dispatcher's rh_record_push() is written to survive this
-- migration NOT being applied (the PATCH fails, it logs, the reap completes).

ALTER TABLE run_history ADD COLUMN IF NOT EXISTS pushed_branch TEXT;
ALTER TABLE run_history ADD COLUMN IF NOT EXISTS pushed_at TIMESTAMPTZ;
