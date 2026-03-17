-- Migration: Create mass_searches table for Mass Strategy Builder
-- Run this in the Supabase SQL Editor

CREATE TABLE IF NOT EXISTS mass_searches (
    id TEXT PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT 'Untitled',
    status TEXT NOT NULL DEFAULT 'pending',
    config_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mass_searches_user ON mass_searches(user_id);
CREATE INDEX IF NOT EXISTS idx_mass_searches_status ON mass_searches(user_id, status);

-- RLS: users can only see their own searches
ALTER TABLE mass_searches ENABLE ROW LEVEL SECURITY;

CREATE POLICY mass_searches_select ON mass_searches
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY mass_searches_insert ON mass_searches
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY mass_searches_update ON mass_searches
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY mass_searches_delete ON mass_searches
    FOR DELETE USING (auth.uid() = user_id);
