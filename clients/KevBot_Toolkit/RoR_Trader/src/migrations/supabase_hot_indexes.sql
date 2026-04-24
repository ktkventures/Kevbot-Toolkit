-- ============================================================================
-- Supabase hot-path indexes — 2026-04-24
--
-- Goal: eliminate sequential scans on the two highest-traffic query patterns.
-- These indexes are cheap to add (minutes), low disk cost (< 100 MB each even
-- at millions of rows), and should cut query latency 10–100× on the affected
-- paths. Idempotent — safe to re-run.
--
-- Run in Supabase SQL Editor: New query → paste → Run.
-- ============================================================================

-- 1. Alerts: recent alerts for a specific strategy
-- Covers: Details drawer alert lookup, algo-history reconciliation,
-- raw-alerts fallback pairing, Worker's alert save path.
-- Without this: sequential scan of the full alerts table (~31k+ rows today,
-- growing every minute markets are open).
CREATE INDEX IF NOT EXISTS alerts_strategy_time_idx
  ON alerts (strategy_id, "timestamp" DESC);

-- 2. Alerts: recent alerts for a user (used by the live dashboard /
-- notification panel and by admin queries). RLS filters by user_id so having
-- an index on (user_id, timestamp) makes RLS-scoped queries efficient too.
CREATE INDEX IF NOT EXISTS alerts_user_time_idx
  ON alerts (user_id, "timestamp" DESC);

-- 3. Alerts: filter by type within a strategy (entry vs exit vs
-- compliance_breach). Composite with strategy_id so "most recent exit for
-- strategy X" lands on an index scan rather than a filter-after-scan.
CREATE INDEX IF NOT EXISTS alerts_strategy_type_idx
  ON alerts (strategy_id, type, "timestamp" DESC);

-- 4. Monitor status poll — Worker's polling cycle reads monitor_status
-- every 15s. Primary key already covers it, but adding a partial index on
-- desired_state='running' rows (small subset) makes "which users want
-- engine on" a single-index lookup.
CREATE INDEX IF NOT EXISTS monitor_status_running_idx
  ON monitor_status (user_id)
  WHERE desired_state = 'running';

-- ============================================================================
-- Verification
-- Run this after the CREATE INDEX statements above to confirm all four
-- indexes are present. Expected: 4 rows in result.
-- ============================================================================
SELECT
  indexname,
  tablename,
  indexdef
FROM pg_indexes
WHERE indexname IN (
  'alerts_strategy_time_idx',
  'alerts_user_time_idx',
  'alerts_strategy_type_idx',
  'monitor_status_running_idx'
)
ORDER BY indexname;
