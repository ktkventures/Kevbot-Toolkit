-- ============================================================================
-- Phase 39 schema migration
-- Adds: portfolios.webhook_group_id column, webhook_groups table, RLS policies
-- Seeds: Kevin's existing TTP group + subscribes Portfolios 19 & 27 to it
--
-- Run in Supabase SQL Editor: New query → paste this entire file → Run.
-- Safe to re-run (idempotent guards on every step).
-- ============================================================================

-- 1. Add the missing column on portfolios
ALTER TABLE portfolios
  ADD COLUMN IF NOT EXISTS webhook_group_id TEXT;

-- 2. Create the webhook_groups table
CREATE TABLE IF NOT EXISTS webhook_groups (
  id         TEXT PRIMARY KEY,
  user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  service    TEXT,
  url_mode   TEXT DEFAULT 'per_event',
  shared_url TEXT DEFAULT '',
  events     JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS webhook_groups_user_id_idx
  ON webhook_groups (user_id);

-- 3. Enable Row-Level Security
ALTER TABLE webhook_groups ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "webhook_groups_select_own" ON webhook_groups;
DROP POLICY IF EXISTS "webhook_groups_insert_own" ON webhook_groups;
DROP POLICY IF EXISTS "webhook_groups_update_own" ON webhook_groups;
DROP POLICY IF EXISTS "webhook_groups_delete_own" ON webhook_groups;

CREATE POLICY "webhook_groups_select_own"
  ON webhook_groups FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "webhook_groups_insert_own"
  ON webhook_groups FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "webhook_groups_update_own"
  ON webhook_groups FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "webhook_groups_delete_own"
  ON webhook_groups FOR DELETE USING (auth.uid() = user_id);

-- 4. Seed Kevin's existing TTP group (imported from webhook_groups.json)
INSERT INTO webhook_groups (id, user_id, name, service, url_mode, shared_url, events, created_at, updated_at)
VALUES (
  'grp_1e36f4',
  '19d47e46-f718-49a6-af32-5f5407f5b170',
  'TTP test',
  'tradethepool',
  'shared',
  'https://app.signalstack.com/hook/TESTfykgSVuGRBzELjSw5Jb3Mw',
  $json${
    "entry_long_market":   {"url": "https://app.signalstack.com/hook/TESTfykgSVuGRBzELjSw5Jb3Mw", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"buy\",\n  \"quantity\": {{quantity}}\n}"},
    "entry_long_limit":    {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"buy\",\n  \"quantity\": {{quantity}},\n  \"limit_price\": {{order_price}}\n}"},
    "entry_short_market":  {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"sell\",\n  \"quantity\": {{quantity}}\n}"},
    "entry_short_limit":   {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"sell\",\n  \"quantity\": {{quantity}},\n  \"limit_price\": {{order_price}}\n}"},
    "exit_long_market":    {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"sell\",\n  \"quantity\": {{quantity}}\n}"},
    "exit_long_limit":     {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"sell\",\n  \"quantity\": {{quantity}},\n  \"limit_price\": {{order_price}}\n}"},
    "exit_short_market":   {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"buy\",\n  \"quantity\": {{quantity}}\n}"},
    "exit_short_limit":    {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"buy\",\n  \"quantity\": {{quantity}},\n  \"limit_price\": {{order_price}}\n}"},
    "cancel_long":         {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"cancel\"\n}"},
    "cancel_short":        {"url": "", "payload": "{\n  \"symbol\": \"{{symbol}}\",\n  \"action\": \"cancel\"\n}"},
    "compliance_breach":   {"url": "", "payload": "{\n  \"symbol\": \"AAPL\",\n  \"action\": \"close\"\n}"}
  }$json$::jsonb,
  '2026-04-13T23:03:34.101696+00:00',
  '2026-04-14T17:11:27.255285+00:00'
)
ON CONFLICT (id) DO NOTHING;

-- 5. Subscribe test portfolios
--    (19 has currently-firing strategies; 27 is the 10s test)
UPDATE portfolios SET webhook_group_id = 'grp_1e36f4' WHERE id IN (19, 27);

-- 6. Verify — this SELECT should return 2 rows showing both portfolios
--    linked to the TTP group
SELECT p.id, p.name, p.webhook_group_id, g.name AS group_name
FROM portfolios p
LEFT JOIN webhook_groups g ON g.id = p.webhook_group_id
WHERE p.id IN (19, 27);
