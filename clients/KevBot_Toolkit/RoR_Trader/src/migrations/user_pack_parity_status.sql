-- ============================================================================
-- User Pack Parity Status — schema migration
-- 2026-04-27
--
-- Stores the last 4-quadrant parity-test result per (user, pack_slug) so the
-- user_packs detail page can show the verdict + per-quadrant chips without
-- re-running the test on every page load. Re-running the test overwrites
-- the prior row.
--
-- One row per (user_id, pack_slug). User packs are globally registered on
-- the worker but each user's parity-test result is preserved separately
-- (different test configs, different "last run" timestamps).
--
-- Run in Supabase SQL Editor: New query → paste → Run. Idempotent.
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_pack_parity_status (
  id                BIGSERIAL PRIMARY KEY,
  user_id           UUID    NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  pack_slug         TEXT    NOT NULL,

  -- Top-level verdict from run_pack_parity_test_4q
  overall_verdict   TEXT    NOT NULL
                            CHECK (overall_verdict IN ('PASS', 'WARN', 'FAIL')),
  summary           TEXT,

  -- Full quadrants payload (Q1/Q2/Q3/Q4 each with parity_score, verdict,
  -- explanation, divergent_examples). Matches the simulator's response
  -- shape so the frontend can render without reshaping.
  quadrants         JSONB   NOT NULL DEFAULT '{}'::jsonb,

  -- Test inputs that produced this result. Surface "tested on SPY 1Min/15Min/7d"
  -- in the UI so users know the scope of the verdict.
  test_config       JSONB   NOT NULL DEFAULT '{}'::jsonb,

  tested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (user_id, pack_slug)
);

-- Speed reads scoped to the current user (the common case)
CREATE INDEX IF NOT EXISTS user_pack_parity_status_user_idx
  ON user_pack_parity_status (user_id);

-- Auto-update the updated_at column on every UPDATE
CREATE OR REPLACE FUNCTION user_pack_parity_status_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS user_pack_parity_status_updated_at_trigger
  ON user_pack_parity_status;
CREATE TRIGGER user_pack_parity_status_updated_at_trigger
  BEFORE UPDATE ON user_pack_parity_status
  FOR EACH ROW
  EXECUTE FUNCTION user_pack_parity_status_set_updated_at();

-- Row-level security: users can read/write their own rows only
ALTER TABLE user_pack_parity_status ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS user_pack_parity_status_select_own
  ON user_pack_parity_status;
CREATE POLICY user_pack_parity_status_select_own
  ON user_pack_parity_status FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS user_pack_parity_status_insert_own
  ON user_pack_parity_status;
CREATE POLICY user_pack_parity_status_insert_own
  ON user_pack_parity_status FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS user_pack_parity_status_update_own
  ON user_pack_parity_status;
CREATE POLICY user_pack_parity_status_update_own
  ON user_pack_parity_status FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS user_pack_parity_status_delete_own
  ON user_pack_parity_status;
CREATE POLICY user_pack_parity_status_delete_own
  ON user_pack_parity_status FOR DELETE
  USING (auth.uid() = user_id);

-- Verify
-- SELECT pack_slug, overall_verdict, tested_at FROM user_pack_parity_status
-- ORDER BY tested_at DESC;
