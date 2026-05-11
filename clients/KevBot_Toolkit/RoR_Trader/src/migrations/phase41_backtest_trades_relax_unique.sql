-- ============================================================================
-- Phase 41 — Backtest trades migration: relax unique constraint
-- 2026-05-11
--
-- The trades table's unique constraint
-- `(strategy_id, entry_fill_ts, exit_fill_ts)` was sized for Phase 40
-- when only algo trades lived in the table. For Phase 41 we add
-- backtest trades to the SAME table with `data_source='backtest_<model>'`.
--
-- Problem: REST and CACHE trades on a C-type entry both fill on the
-- same bar boundary timestamp. With the current constraint, only one
-- can be inserted. We need to include `data_source` in the dedupe key
-- so REST and CACHE trades coexist for the same trade event.
--
-- Migration:
--   1. UPDATE existing NULL data_source rows to 'cache_locked'
--      (cron has been writing them under algo_model=cache_locked)
--   2. DROP existing trades_dedupe_idx
--   3. CREATE new unique index including data_source
--
-- Run in Supabase SQL Editor. Idempotent.
-- ============================================================================

-- Step 1: Tag existing NULL data_source rows as 'cache_locked'.
-- These came from algo cron writes that didn't explicitly set
-- data_source. With the new convention, every row gets a label.
UPDATE trades
SET data_source = 'cache_locked'
WHERE data_source IS NULL;

-- Step 2: Drop the old unique index (does NOT block reads/writes
-- temporarily during the recreation — both old and new can coexist
-- briefly via CONCURRENTLY but we're not using that here for simplicity)
DROP INDEX IF EXISTS trades_dedupe_idx;

-- Step 3: Recreate with data_source included. Use COALESCE to handle
-- any future NULL inserts (treats NULL as empty string for dedupe).
CREATE UNIQUE INDEX IF NOT EXISTS trades_dedupe_idx
  ON trades (
    strategy_id,
    entry_fill_ts,
    COALESCE(exit_fill_ts, '1970-01-01'::timestamptz),
    COALESCE(data_source, '')
  );

-- Step 4: Add index on (strategy_id, data_source) for filtered reads
-- on the divergence endpoint + KPI fast paths.
CREATE INDEX IF NOT EXISTS trades_strategy_data_source_idx
  ON trades (strategy_id, data_source);

-- Verification queries (run after migration):
-- SELECT data_source, count(*) FROM trades GROUP BY data_source;
--   → Should show 'cache_locked' count = previous NULL count
-- SELECT indexdef FROM pg_indexes WHERE tablename = 'trades';
--   → Should show trades_dedupe_idx with data_source in COALESCE
