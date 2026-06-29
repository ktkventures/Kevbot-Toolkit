-- compute_jobs.payload — generic JSONB params for non-strategy job types.
--
-- The compute_jobs queue was strategy-centric (strategy_ids). Adding a 'backfill'
-- job type (symbol, timeframe, start/end) needs a place to carry those params, so
-- the batch-worker can run bar_cache backfills off the api service. Nullable +
-- additive: existing recompute rows ignore it. Adding a nullable column is an
-- instant catalog-only change in Postgres (no table rewrite, no lock contention).
--
-- Idempotent. Applied to prod 2026-06-29.

ALTER TABLE compute_jobs ADD COLUMN IF NOT EXISTS payload jsonb;
