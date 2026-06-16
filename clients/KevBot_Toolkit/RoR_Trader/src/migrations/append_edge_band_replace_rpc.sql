-- ============================================================================
-- B2 fix — atomic edge-band replace RPC
-- 2026-06-15
--
-- "Update New Data" (append) under-generates trades near the data edge and
-- INSERT-only dedup never rewrites them, so the trailing band fossilizes and
-- live↔backtest pairing degrades the closer you measure to "now" (the
-- "clean-then-messier" artifact). The fix: each append RECOMPUTES its trailing
-- ~120-min band with full UAD-parity warmup and REPLACES it via this function.
--
-- Why an RPC: the API and the live worker append the same strategy from
-- SEPARATE processes, and the Supabase client runs DELETE and INSERT as
-- separate HTTP requests (no shared transaction). A naive delete-then-insert
-- can briefly expose an empty band that a concurrent dashboard read mistakes
-- for a phantom spike. This function wraps pg_advisory_xact_lock + DELETE +
-- INSERT in ONE transaction, so concurrent appends serialize and the band is
-- never observed empty.
--
-- The Python caller (db.replace_trades_in_window_admin) prefers this RPC and
-- falls back to client-side delete-then-insert if it isn't installed yet.
--
-- Run in Supabase SQL Editor: New query → paste → Run. Idempotent.
-- ============================================================================

CREATE OR REPLACE FUNCTION replace_trades_in_window(
  p_strategy_id INTEGER,
  p_user_id     UUID,
  p_ds_filter   TEXT,          -- e.g. 'backtest_%' or 'cache_%'
  p_entry_floor TIMESTAMPTZ,   -- inclusive lower bound on entry_fill_ts
  p_entry_ceil  TIMESTAMPTZ,   -- exclusive upper bound on entry_fill_ts
  p_rows        JSONB          -- array of row objects (hot cols + data jsonb)
) RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_inserted INTEGER;
BEGIN
  -- Serialize concurrent band replaces for this strategy (txn-scoped).
  PERFORM pg_advisory_xact_lock(hashtext('append_band:' || p_strategy_id::text));

  DELETE FROM trades
   WHERE strategy_id   = p_strategy_id
     AND data_source   LIKE p_ds_filter
     AND entry_fill_ts >= p_entry_floor
     AND entry_fill_ts <  p_entry_ceil;

  INSERT INTO trades (
    strategy_id, user_id, entry_alert_id, exit_alert_id,
    entry_trigger_ts, entry_fill_ts, exit_trigger_ts, exit_fill_ts,
    entry_price, exit_price, r_multiple, dollar_pnl, executed_quantity,
    direction, exec_type, exit_reason, data_source, data
  )
  SELECT
    p_strategy_id, p_user_id, r.entry_alert_id, r.exit_alert_id,
    r.entry_trigger_ts, r.entry_fill_ts, r.exit_trigger_ts, r.exit_fill_ts,
    r.entry_price, r.exit_price, r.r_multiple, r.dollar_pnl, r.executed_quantity,
    r.direction, r.exec_type, r.exit_reason, r.data_source, COALESCE(r.data, '{}'::jsonb)
  FROM jsonb_to_recordset(p_rows) AS r(
    entry_alert_id    BIGINT,
    exit_alert_id     BIGINT,
    entry_trigger_ts  TIMESTAMPTZ,
    entry_fill_ts     TIMESTAMPTZ,
    exit_trigger_ts   TIMESTAMPTZ,
    exit_fill_ts      TIMESTAMPTZ,
    entry_price       NUMERIC,
    exit_price        NUMERIC,
    r_multiple        NUMERIC,
    dollar_pnl        NUMERIC,
    executed_quantity INTEGER,
    direction         TEXT,
    exec_type         TEXT,
    exit_reason       TEXT,
    data_source       TEXT,
    data              JSONB
  )
  -- ON CONFLICT with no named target: skips any row violating ANY unique
  -- constraint without requiring an exact constraint match. (Named target
  -- (strategy_id, entry_fill_ts, exit_fill_ts) raised 42P10 "no unique or
  -- exclusion constraint matching the ON CONFLICT specification" — the
  -- trades_dedupe_idx isn't inferable as a conflict arbiter here, likely
  -- due to its nullable cols. The band is DELETEd first in this same txn,
  -- so conflicts are only a stray-duplicate safety net anyway.)
  ON CONFLICT DO NOTHING;

  GET DIAGNOSTICS v_inserted = ROW_COUNT;
  RETURN v_inserted;
END
$$;

-- Allow the service role (admin client) to call it.
GRANT EXECUTE ON FUNCTION replace_trades_in_window(
  INTEGER, UUID, TEXT, TIMESTAMPTZ, TIMESTAMPTZ, JSONB
) TO service_role;
