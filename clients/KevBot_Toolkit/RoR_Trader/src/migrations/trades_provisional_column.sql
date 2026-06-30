-- ============================================================================
-- trades.provisional column — M-RS4 Phase 3 (continuous-resident backtest engine)
-- 2026-06-30
--
-- Adds a top-level `provisional` boolean to the trades table. The shadow-worker's
-- resident engine emits trades for the UNSETTLED tail (the <~16-min window past the
-- trade-commit lag, before bars are final) marked `provisional=true`; the flag flips
-- to false once the underlying bars settle and the trade is re-trued. Decision §6
-- (Plan_M-RS4_Phase3_ContinuousBacktestEngine.md): ONE lane, ONE flag — provisional
-- trades still show in history (UI marks them not-finalized, e.g. an asterisk); there
-- is no separate provisional lane.
--
-- Default false: every existing trade is settled/committed, so they are correctly
-- non-provisional. NOT NULL + constant default = metadata-only in Postgres 11+ (no
-- table rewrite) — fast even on the large trades table.
--
-- Idempotent. Run in Supabase SQL Editor or via psycopg (SUPABASE_CONNECTION_STRING).
-- Roll back:  ALTER TABLE trades DROP COLUMN IF EXISTS provisional;
-- ============================================================================

ALTER TABLE trades
  ADD COLUMN IF NOT EXISTS provisional BOOLEAN NOT NULL DEFAULT false;

-- Verify after running:
-- SELECT provisional, count(*) AS n FROM trades GROUP BY provisional ORDER BY n DESC;
