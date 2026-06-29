-- Write-through unification: settled / in-flux marker on bar_cache
-- (Design_BarCache_WriteThrough_Unification.md §4)
-- ================================================================
-- Adds the settled/provisional primitive so the continuous data-worker
-- write-through can mark which bars are permanent truth (settled, past
-- RORT_SHADOW_SETTLE_MIN) vs still-revising tail (provisional).
--
-- DEFAULT true is deliberate + churn-safe:
--   - In PG11+ a NOT NULL column with a constant default is METADATA-ONLY
--     (no table rewrite, no dead tuples — respects the 2026-06-28 bloat
--     lessons #6/#7). Existing bar_cache rows are all historical/settled,
--     so true is the correct value for them with zero rewrite.
--   - The write-through path sets `settled` EXPLICITLY per-bar (false for
--     the unsettled tail, true once aged past the threshold), so new rows
--     are always accurate regardless of the default.
ALTER TABLE bar_cache
  ADD COLUMN IF NOT EXISTS settled    BOOLEAN     NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS revised_at TIMESTAMPTZ;

-- revised_at: last overwrite time (observability — how the tail moved).
-- Null for pre-existing rows; stamped by write-through on every upsert.

-- Verification:
--   \d bar_cache
--   SELECT settled, count(*) FROM bar_cache GROUP BY settled;
