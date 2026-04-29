-- ============================================================================
-- Strategy Parity Status — schema migration
-- 2026-04-28
--
-- Adds parity_status JSONB to strategies. Populated by
-- recompute_and_persist_stored_trades after every refresh: runs the parity
-- service against the freshly-written stored_trades and persists a compact
-- summary so the strategy card / list views can show a "live-executable"
-- badge without hitting the slow parity endpoint on render.
--
-- Shape:
--   {
--     "score": 0.96,                           -- matched_count / stored_count
--     "verdict": "PASS",                       -- PASS|PARTIAL|FAIL_LIVE_BLOCKED
--                                              --  |FAIL_OVER_FIRES|NO_TRADES
--                                              --  |NO_DATA|ERROR
--     "stored_count": 50,
--     "matched_count": 48,
--     "stored_only_count": 2,
--     "replay_only_count": 0,
--     "most_common_failing_gate": "1Min-MACD_LINE-S<M",  -- nullable
--     "computed_at": "2026-04-28T12:34:56+00:00",
--     "duration_s": 45,
--     "params": { "last_n": 200, "forward_test_only": false }
--   }
--
-- Run in Supabase SQL Editor: New query → paste → Run. Idempotent.
-- ============================================================================

ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS parity_status JSONB;

-- Index for filtering the strategy list by verdict (e.g. show only PASS).
-- Partial — only strategies that have been parity-checked.
CREATE INDEX IF NOT EXISTS strategies_parity_verdict_idx
  ON strategies ((parity_status->>'verdict'))
  WHERE parity_status IS NOT NULL;

-- Verify (paste into Supabase SQL editor after running the migration):
-- SELECT
--   parity_status->>'verdict' AS verdict,
--   COUNT(*) AS n,
--   ROUND(AVG((parity_status->>'score')::numeric), 3) AS avg_score
-- FROM strategies
-- WHERE parity_status IS NOT NULL
-- GROUP BY verdict
-- ORDER BY n DESC;
