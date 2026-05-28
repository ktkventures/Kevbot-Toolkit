-- ============================================================================
-- Alerts REST-verification columns — schema migration
-- 2026-05-28 — M2 of /home/kevin/.claude/plans/breezy-dreaming-umbrella.md
--
-- Adds three top-level columns to the alerts table to support the new
-- `ws_rest_spliced` live model:
--
--   verification_status        — NULL | 'pending' | 'verified' |
--                                'corrected' | 'rest_unavailable'
--   verification_close_delta   — WS close minus REST close (NUMERIC)
--   verification_completed_at  — when the REST verifier finished
--
-- Workflow:
--   1. Ralph fires the alert (existing path), INSERT with status='pending'
--      if the strategy is on ws_rest_spliced and REST_VERIFY_ENABLED.
--   2. After grace_seconds, the verifier polls Polygon REST for the bar.
--   3. On comparison: UPDATE the alert row with status + close_delta +
--      completed_at.
--
-- No backfill on legacy alerts — pre-migration alerts have NULL across
-- all three columns and render as "unverified" in the UI.
--
-- Run in Supabase SQL Editor. Idempotent.
-- ============================================================================

ALTER TABLE alerts
  ADD COLUMN IF NOT EXISTS verification_status TEXT;

ALTER TABLE alerts
  ADD COLUMN IF NOT EXISTS verification_close_delta NUMERIC;

ALTER TABLE alerts
  ADD COLUMN IF NOT EXISTS verification_completed_at TIMESTAMPTZ;

-- Index for the "pending verifications" lookup the verifier may use to
-- recover after a worker restart (cheap; small column).
CREATE INDEX IF NOT EXISTS alerts_verification_status_idx
  ON alerts (verification_status)
  WHERE verification_status = 'pending';

-- Verify after running:
-- SELECT
--   verification_status,
--   count(*) AS n
-- FROM alerts
-- GROUP BY verification_status
-- ORDER BY n DESC;
