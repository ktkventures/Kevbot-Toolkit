-- Migration: update_jobs table for async on-demand UAD
-- Apply via Supabase SQL editor before deploying the update_jobs router.
-- Mirrors the mass_searches table pattern.

CREATE TABLE IF NOT EXISTS update_jobs (
    id BIGINT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT 'Untitled Update',
    mode TEXT NOT NULL CHECK (mode IN ('all', 'new', 'window')),
    scope TEXT NOT NULL CHECK (scope IN ('single', 'bulk')) DEFAULT 'single',
    strategy_ids INTEGER[] NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'orphaned')),
    config_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- config_data contains:
    --   progress: {current_step, total_steps, current_sid, current_phase, per_strategy: {sid: {...}}}
    --   summary:  {succeeded, failed, total_trades_inserted, total_elapsed_s}
    --   results:  per-strategy result objects (mirrors mass_searches.config_data.results)
    --   error:    last-error string when status='failed'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS update_jobs_user_id_idx ON update_jobs (user_id);
CREATE INDEX IF NOT EXISTS update_jobs_status_idx ON update_jobs (status);
CREATE INDEX IF NOT EXISTS update_jobs_created_at_idx ON update_jobs (created_at DESC);

-- RLS: same pattern as mass_searches — user can read/write only their own rows.
ALTER TABLE update_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own update_jobs"
    ON update_jobs FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own update_jobs"
    ON update_jobs FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own update_jobs"
    ON update_jobs FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own update_jobs"
    ON update_jobs FOR DELETE
    USING (auth.uid() = user_id);

COMMENT ON TABLE update_jobs IS
    'Async UAD job state — mirrors mass_searches pattern. Background daemon thread updates progress_data + summary as the job runs. Survives worker restarts via orphan cleanup on api boot.';
