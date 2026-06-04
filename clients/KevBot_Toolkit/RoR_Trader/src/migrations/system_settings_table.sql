-- Migration: system_settings — single key/value table for runtime-toggleable
-- platform-wide flags. Apply via Supabase SQL editor.
--
-- First user: data_worker_streaming_enabled (replaces the env-var-only
-- DATA_WORKER_STREAMING_ENABLED toggle). Data Worker caches the value
-- with ~30s TTL so changes take effect within a minute without redeploy.

CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by UUID REFERENCES auth.users(id)
);

CREATE INDEX IF NOT EXISTS system_settings_updated_at_idx
    ON system_settings (updated_at DESC);

-- RLS: read by any authenticated user (the toggle state is non-sensitive
-- and read-heavy from the worker); writes are admin-gated at the API
-- layer (the endpoint uses the admin client), so RLS on UPDATE/INSERT
-- doesn't need to mirror it.
ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Any authenticated user can read system_settings"
    ON system_settings FOR SELECT
    USING (auth.role() = 'authenticated');

-- INSERT/UPDATE/DELETE — service role only (admin endpoints). No user
-- policy granted; service role bypasses RLS.

COMMENT ON TABLE system_settings IS
    'Platform-wide runtime flags. Keys are well-known strings (see docs); values are JSONB so we can store booleans, numbers, or small objects without schema churn.';
