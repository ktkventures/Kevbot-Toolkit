-- ============================================================================
-- Strategy Health Badge — schema migration
-- 2026-04-27
--
-- Adds three columns to the `strategies` table to power the Strategy Health
-- Badge feature documented in docs/Strategy_Health_Badge_Design.md:
--
--   data_source      — 'rapid' | 'full' | 'hifi'  (which backtest path
--                      produced the current KPIs)
--   kpis_stale_since — TIMESTAMPTZ, set when a backtest-affecting field
--                      is edited; cleared on KPI recompute
--   kpis_computed_at — TIMESTAMPTZ, set when KPIs are written; drives the
--                      time-based-stale check (default 60 days)
--
-- Run in Supabase SQL Editor: New query → paste → Run. Idempotent.
-- ============================================================================

-- 1. Columns
ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS data_source TEXT
    CHECK (data_source IN ('rapid', 'full', 'hifi'))
    DEFAULT 'full';

ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS kpis_stale_since TIMESTAMPTZ;

ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS kpis_computed_at TIMESTAMPTZ;

-- 2. Backfill kpis_computed_at from updated_at where possible
-- (best available timestamp for legacy strategies' last KPI run)
UPDATE strategies
   SET kpis_computed_at = updated_at
 WHERE kpis_computed_at IS NULL;

-- 3. Backfill data_source via heuristic: Mass Builder strategies follow
-- the naming convention `<symbol> <direction> <tf> Mass #<n>` because the
-- builder auto-names them. Use that as the proxy for 'rapid' data
-- since there's no direct mass_search_id FK on strategies. Everything
-- else stays at the 'full' default.
--
-- Note: future Mass Builder saves should explicitly set data_source='rapid'
-- via the application path (see services.py); this is the one-time backfill
-- for legacy rows.
UPDATE strategies
   SET data_source = 'rapid'
 WHERE data_source = 'full'
   AND name LIKE '%Mass #%';

-- 4. Optional index (speeds Health Badge list queries that filter on stale)
CREATE INDEX IF NOT EXISTS strategies_kpis_stale_idx
  ON strategies (kpis_stale_since)
  WHERE kpis_stale_since IS NOT NULL;

-- 5. Verify
-- SELECT
--   data_source, COUNT(*) AS n,
--   COUNT(*) FILTER (WHERE kpis_stale_since IS NOT NULL) AS stale,
--   MIN(kpis_computed_at) AS oldest_kpi,
--   MAX(kpis_computed_at) AS newest_kpi
-- FROM strategies
-- GROUP BY data_source
-- ORDER BY n DESC;
