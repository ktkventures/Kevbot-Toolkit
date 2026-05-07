-- ============================================================================
-- Alerts live_model column — schema migration
-- 2026-05-07
--
-- Adds top-level `live_model` text column to the alerts table. Worker
-- dispatchers stamp the strategy's live_model at alert fire time so the
-- Divergence tab can render which live_model produced each alert. Future
-- changes to live_model affect alerts going forward only — historical
-- alerts stay null and render as "unknown" in the UI.
--
-- No backfill on legacy alerts (Kevin 2026-05-07: "honesty over speed").
--
-- Run in Supabase SQL Editor. Idempotent.
-- ============================================================================

ALTER TABLE alerts
  ADD COLUMN IF NOT EXISTS live_model TEXT;

-- Verify after running:
-- SELECT
--   live_model,
--   count(*) AS n
-- FROM alerts
-- GROUP BY live_model
-- ORDER BY n DESC;
