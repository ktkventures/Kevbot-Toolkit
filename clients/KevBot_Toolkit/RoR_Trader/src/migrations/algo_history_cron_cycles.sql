-- ============================================================================
-- Algo-History Cron Cycle Stats — schema migration
-- 2026-05-07
--
-- Records every cron cycle the algo-history append daemon completes. Lets
-- the Jobs page surface throughput, budget exhaustion, oldest-stamp
-- starvation, and per-strategy elapsed times without scraping Railway logs.
--
-- Worker writes one row per user per cycle (after `append_recent_trades_for_user`
-- returns). API reads via /api/jobs/cron-stats. Auto-prunes to last 200 rows
-- per user via Worker-side cleanup so the table stays bounded.
--
-- Shape of `per_strategy` JSONB:
--   [
--     {"sid": 152, "name": "yearp1", "status": "appended",
--      "inserted": 7, "elapsed_s": 28.3, "engine_path": "windowed",
--      "bars_processed": 1024, "window_days": 1, "reason": null},
--     ...
--   ]
--
-- Run in Supabase SQL Editor. Idempotent.
-- ============================================================================

CREATE TABLE IF NOT EXISTS algo_history_cron_cycles (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NOT NULL,
  started_at TIMESTAMPTZ NOT NULL,
  ended_at TIMESTAMPTZ NOT NULL,
  processed INT NOT NULL DEFAULT 0,
  inserted_total INT NOT NULL DEFAULT 0,
  skipped INT NOT NULL DEFAULT 0,
  errors INT NOT NULL DEFAULT 0,
  elapsed_s NUMERIC NOT NULL DEFAULT 0,
  budget_exhausted BOOLEAN NOT NULL DEFAULT FALSE,
  per_strategy JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS algo_history_cron_cycles_user_started_idx
  ON algo_history_cron_cycles (user_id, started_at DESC);

-- Verify after running:
-- SELECT user_id, count(*) AS cycles,
--        max(started_at) AS most_recent,
--        sum(inserted_total) AS trades_appended,
--        sum(case when budget_exhausted then 1 else 0 end) AS budget_exhausted_count
-- FROM algo_history_cron_cycles
-- GROUP BY user_id;
