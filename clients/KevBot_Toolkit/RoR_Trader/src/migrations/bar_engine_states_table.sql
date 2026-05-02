-- Engine indicator state per bar (Milestone 8.7 M6 — engine-truth capture)
-- =================================================================
-- Captures the live engine's complete indicator state at every bar
-- close, per strategy. Lets us reconstruct EXACTLY what the engine
-- was looking at at any historical bar (engine-truth precision), as
-- opposed to recomputing from cached OHLCV (cache-reconstruction
-- precision in `live_bars`).
--
-- Today is data-capture only — no read path, no UI integration. The
-- point is to start collecting now so we have the data when/if
-- engine-truth replay becomes a real feature. Skip the writes by
-- not setting BAR_ENGINE_STATE_WRITE_ENABLED on the worker.
--
-- Companion writer: src/bar_engine_state_writer.py (mirror of
-- live_bars_writer.py pattern — env-flag gated, fire-and-forget,
-- thread-pool-backed).
--
-- Schema rationale (mirrors live_bars patterns):
--   - Composite PK (strategy_id, bar_start) — one row per strategy
--     per bar close. Idempotent via on-conflict upsert.
--   - JSONB indicator_state — Dict[str, float|bool] from
--     IncrementalIndicatorEngine.state.current. Indicator keys vary
--     per strategy (depend on required indicators), so JSONB is the
--     right shape rather than fixed columns.
--   - source column distinguishes 'ws' (live worker) from
--     'rest_backfill' (future — replay from historical data).
--
-- Storage estimate:
--   ~14k rows/day across 27 strategies (mixed TFs, RTH only)
--   ~900 bytes/row JSONB content + overhead = ~1KB/row
--   ~13 MB/day, ~4.7 GB/year. Manageable.
--
-- Retention: indefinite for now. Revisit if Postgres footprint
-- becomes a concern.

CREATE TABLE IF NOT EXISTS bar_engine_states (
  strategy_id        INT              NOT NULL,
  bar_start          TIMESTAMPTZ      NOT NULL,
  indicator_state    JSONB            NOT NULL,
  source             TEXT             NOT NULL DEFAULT 'ws',
  written_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
  PRIMARY KEY (strategy_id, bar_start)
);

-- Composite PK already supports (strategy_id, bar_start) range
-- queries. Explicit covering index for documentation.
CREATE INDEX IF NOT EXISTS bar_engine_states_lookup_idx
  ON bar_engine_states (strategy_id, bar_start);

-- RLS: disabled. bar_engine_states is engine telemetry, not
-- user-scoped data. Worker writes via service-role key (bypasses
-- RLS anyway).
ALTER TABLE bar_engine_states DISABLE ROW LEVEL SECURITY;

-- =================================================================
-- Verification queries:
--
-- 1. Confirm table exists + structure:
--    \d bar_engine_states
--
-- 2. After 1 hour of trading post-deploy:
--    SELECT strategy_id, count(*), min(bar_start), max(bar_start)
--    FROM bar_engine_states
--    WHERE bar_start > now() - interval '1 hour'
--    GROUP BY strategy_id
--    ORDER BY 2 DESC;
--
-- 3. Sample a row:
--    SELECT strategy_id, bar_start, indicator_state
--    FROM bar_engine_states
--    ORDER BY written_at DESC LIMIT 5;
--
-- 4. Indicator-key distribution (sanity check on JSONB content):
--    SELECT jsonb_object_keys(indicator_state), count(*)
--    FROM bar_engine_states
--    WHERE bar_start > now() - interval '1 hour'
--    GROUP BY 1 ORDER BY 2 DESC;
-- =================================================================
