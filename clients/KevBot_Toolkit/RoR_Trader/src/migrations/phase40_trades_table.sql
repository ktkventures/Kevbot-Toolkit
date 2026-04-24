-- ============================================================================
-- Phase 40 — trades table migration
-- 2026-04-24
--
-- Normalizes strategies.stored_trades (JSONB, unbounded growth) into a real
-- trades table. Fixes _persist_algo_trade statement-timeout cascades,
-- exit-webhook latency, and baseline Supabase read pressure.
--
-- Schema = Option B: 14 hot columns for the heavily-queried fields + a
-- single `data` JSONB for the long tail (matching the alerts-table pattern
-- established during the Trade_Timestamps_Spec work).
--
-- Run in Supabase SQL Editor: New query → paste → Run. Idempotent.
-- ============================================================================

-- 1. Create trades table (if not already there)
CREATE TABLE IF NOT EXISTS trades (
  id                BIGSERIAL PRIMARY KEY,
  strategy_id       INTEGER NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
  user_id           UUID    NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  entry_alert_id    BIGINT,
  exit_alert_id     BIGINT,

  -- 4-timestamp model (Trade_Timestamps_Spec)
  entry_trigger_ts  TIMESTAMPTZ,
  entry_fill_ts     TIMESTAMPTZ,
  exit_trigger_ts   TIMESTAMPTZ,
  exit_fill_ts      TIMESTAMPTZ,

  -- Core trade economics (heavily queried + aggregated)
  entry_price       NUMERIC,
  exit_price        NUMERIC,
  r_multiple        NUMERIC,
  dollar_pnl        NUMERIC,
  executed_quantity INTEGER,

  -- Routing + classification (heavily filtered)
  direction         TEXT,
  exec_type         TEXT,
  exit_reason       TEXT,
  data_source       TEXT,

  -- Long-tail fields: stop/target prices, risk/pnl raw, win, hold durations,
  -- bars_held, entry/exit trigger IDs, stop_exec_type, target_exec_type,
  -- behavior, confluence_records (list), planned/rpt/bp quantities,
  -- buying_power_used, risk_per_trade, slippage_r pair, theoretical
  -- entry/exit, entry_stop_price
  data              JSONB DEFAULT '{}'::jsonb,

  created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Indexes
CREATE INDEX IF NOT EXISTS trades_strategy_exit_idx
  ON trades (strategy_id, exit_fill_ts DESC);

CREATE INDEX IF NOT EXISTS trades_strategy_entry_idx
  ON trades (strategy_id, entry_fill_ts DESC);

CREATE INDEX IF NOT EXISTS trades_user_id_idx
  ON trades (user_id);

CREATE INDEX IF NOT EXISTS trades_entry_alert_idx
  ON trades (entry_alert_id) WHERE entry_alert_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS trades_exit_alert_idx
  ON trades (exit_alert_id) WHERE exit_alert_id IS NOT NULL;

-- Dedupe index makes the backfill idempotent AND acts as the natural
-- "this trade already exists" check for future re-runs. Multiple trades
-- CAN share identical fill timestamps if backtests pyramid positions at
-- the same bar, so allow NULLs through without collision (Postgres treats
-- NULLs as distinct in unique indexes).
CREATE UNIQUE INDEX IF NOT EXISTS trades_dedupe_idx
  ON trades (strategy_id, entry_fill_ts, exit_fill_ts);

-- 3. RLS policies (mirror webhook_groups + strategies pattern)
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "trades_select_own" ON trades;
DROP POLICY IF EXISTS "trades_insert_own" ON trades;
DROP POLICY IF EXISTS "trades_update_own" ON trades;
DROP POLICY IF EXISTS "trades_delete_own" ON trades;

CREATE POLICY "trades_select_own"
  ON trades FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "trades_insert_own"
  ON trades FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "trades_update_own"
  ON trades FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "trades_delete_own"
  ON trades FOR DELETE USING (auth.uid() = user_id);

-- 4. Backfill from strategies.stored_trades
-- Strips the hot-column fields out of each trade object and puts everything
-- else into `data`. Accepts both the canonical *_fill_ts fields and the
-- legacy entry_time / exit_time aliases (older strategies pre-dating the
-- Trade_Timestamps_Spec migration).
INSERT INTO trades (
  strategy_id, user_id,
  entry_alert_id, exit_alert_id,
  entry_trigger_ts, entry_fill_ts, exit_trigger_ts, exit_fill_ts,
  entry_price, exit_price,
  r_multiple, dollar_pnl, executed_quantity,
  direction, exec_type, exit_reason, data_source,
  data
)
SELECT
  s.id,
  s.user_id,
  NULLIF(t->>'entry_alert_id', '')::BIGINT,
  NULLIF(t->>'exit_alert_id', '')::BIGINT,
  NULLIF(t->>'entry_trigger_ts', '')::TIMESTAMPTZ,
  COALESCE(
    NULLIF(t->>'entry_fill_ts', '')::TIMESTAMPTZ,
    NULLIF(t->>'entry_time', '')::TIMESTAMPTZ
  ),
  NULLIF(t->>'exit_trigger_ts', '')::TIMESTAMPTZ,
  COALESCE(
    NULLIF(t->>'exit_fill_ts', '')::TIMESTAMPTZ,
    NULLIF(t->>'exit_time', '')::TIMESTAMPTZ
  ),
  NULLIF(t->>'entry_price', '')::NUMERIC,
  NULLIF(t->>'exit_price', '')::NUMERIC,
  NULLIF(t->>'r_multiple', '')::NUMERIC,
  NULLIF(t->>'dollar_pnl', '')::NUMERIC,
  NULLIF(t->>'executed_quantity', '')::INTEGER,
  t->>'direction',
  t->>'exec_type',
  t->>'exit_reason',
  COALESCE(t->>'data_source', 'backtest'),
  -- Strip hot-column keys from the record; remainder goes in data JSONB
  t - ARRAY[
    'entry_alert_id','exit_alert_id',
    'entry_trigger_ts','entry_fill_ts','exit_trigger_ts','exit_fill_ts',
    'entry_time','exit_time',
    'entry_price','exit_price',
    'r_multiple','dollar_pnl','executed_quantity',
    'direction','exec_type','exit_reason','data_source'
  ]
FROM strategies s,
     LATERAL jsonb_array_elements(COALESCE(s.stored_trades, '[]'::jsonb)) AS t
-- Only backfill rows that have at least one usable fill timestamp, to
-- satisfy the (non-null) dedupe index on typical trades. Records missing
-- both entry_fill_ts AND entry_time are malformed and will need manual
-- review if they exist.
WHERE COALESCE(
        NULLIF(t->>'entry_fill_ts','')::TIMESTAMPTZ,
        NULLIF(t->>'entry_time','')::TIMESTAMPTZ
      ) IS NOT NULL
ON CONFLICT (strategy_id, entry_fill_ts, exit_fill_ts) DO NOTHING;

-- 5. Verification
-- jsonb_trade_count and table_trade_count should be within ~1% of each
-- other. Any large discrepancy = backfill missed something; inspect a
-- few offenders manually.
SELECT
  (
    SELECT COALESCE(SUM(jsonb_array_length(COALESCE(stored_trades,'[]'::jsonb))), 0)
    FROM strategies
  ) AS jsonb_trade_count,
  (SELECT COUNT(*) FROM trades) AS table_trade_count,
  (
    SELECT COUNT(*) FROM strategies
    WHERE jsonb_array_length(COALESCE(stored_trades,'[]'::jsonb)) > 0
  ) AS strategies_with_trades;

-- Optional: per-strategy sanity check — the 5 fattest strategies with
-- their JSONB trade count vs backfilled row count.
SELECT
  s.id,
  s.name,
  jsonb_array_length(COALESCE(s.stored_trades,'[]'::jsonb)) AS jsonb_count,
  (SELECT COUNT(*) FROM trades t WHERE t.strategy_id = s.id) AS table_count
FROM strategies s
ORDER BY jsonb_array_length(COALESCE(s.stored_trades,'[]'::jsonb)) DESC
LIMIT 5;
