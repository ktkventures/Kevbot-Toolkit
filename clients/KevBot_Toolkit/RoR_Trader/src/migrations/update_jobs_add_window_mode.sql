-- Migration: extend update_jobs.mode CHECK to allow 'window' backfill mode.
-- Apply via Supabase SQL editor.
--
-- Background: the Manual Window Backfill feature (cb4939c) adds mode='window'
-- for targeted BT-lane backfill over an explicit [start, end) range. The
-- existing CHECK constraint only allowed ('all', 'new'), so /api/update-jobs/run
-- with mode=window was silently failing the insert (save_update_job
-- swallowed the constraint violation and returned None → API 500
-- "failed to persist update job").

ALTER TABLE update_jobs DROP CONSTRAINT IF EXISTS update_jobs_mode_check;
ALTER TABLE update_jobs
    ADD CONSTRAINT update_jobs_mode_check
    CHECK (mode IN ('all', 'new', 'window'));
