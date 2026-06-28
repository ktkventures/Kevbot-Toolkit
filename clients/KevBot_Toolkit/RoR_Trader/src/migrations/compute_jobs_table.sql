-- compute_jobs — durable, cross-service batch-job queue (M-RS3 Phase 2).
--
-- Today the recompute job (Update-All / Append-New) runs as a daemon THREAD
-- inside the API process with in-memory state (recompute_jobs._active_jobs).
-- A dedicated batch-worker service can't see that. This table is the handoff:
-- the API enqueues a row (when RORT_COMPUTE_REMOTE=1); the batch-worker claims
-- it (optimistic claim — UPDATE ... WHERE status='queued'), runs the pool, and
-- mirrors progress back; the API's GET /api/jobs/{id} reads the row. Columns
-- mirror the in-memory job dict so the API response shape is unchanged.
--
-- Generic on purpose (job_type + strategy_ids): full_recompute / append_recent
-- today, room for future tenants (e.g. Mass Builder enumeration) on one tier.

CREATE TABLE IF NOT EXISTS compute_jobs (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 text        NOT NULL,
    job_type                text        NOT NULL,           -- full_recompute | append_recent
    strategy_ids            jsonb       NOT NULL,           -- int[]
    force                   boolean     NOT NULL DEFAULT false,
    status                  text        NOT NULL DEFAULT 'queued',  -- queued|running|completed|failed|cancelled
    claimed_by              text,                            -- batch-worker replica id (claim safety)
    claimed_at              timestamptz,
    created_at              timestamptz NOT NULL DEFAULT now(),
    started_at              timestamptz,
    completed_at            timestamptz,
    current_strategy_idx    integer     NOT NULL DEFAULT 0,
    current_strategy_id     integer,
    current_strategy_name   text,
    progress_label          text        NOT NULL DEFAULT 'queued',
    per_strategy_results    jsonb       NOT NULL DEFAULT '[]'::jsonb,
    summary                 jsonb       NOT NULL DEFAULT '{}'::jsonb,
    error                   text,
    cancelled               boolean     NOT NULL DEFAULT false
);

-- Claim query: oldest queued first. Partial index keeps it tight as completed
-- rows accumulate.
CREATE INDEX IF NOT EXISTS idx_compute_jobs_queued
    ON compute_jobs (created_at)
    WHERE status = 'queued';

-- User job-list query (GET /api/jobs), newest first.
CREATE INDEX IF NOT EXISTS idx_compute_jobs_user_created
    ON compute_jobs (user_id, created_at DESC);

-- RLS: all server access is via the service-role admin client (RLS bypassed),
-- so this is defense-in-depth — if the frontend ever reads directly, a user
-- sees only their own jobs. Writes stay service-role only (no write policy).
ALTER TABLE compute_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS compute_jobs_select_own ON compute_jobs;
CREATE POLICY compute_jobs_select_own ON compute_jobs
    FOR SELECT USING (auth.uid()::text = user_id);
