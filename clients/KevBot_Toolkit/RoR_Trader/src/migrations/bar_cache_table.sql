-- Persistent bar cache (Layer 1 of project_persistent_bar_cache.md)
-- =================================================================
-- Stores raw OHLCV bars from Polygon REST. Used by
-- data_loader.cached_load_market_data() to avoid re-fetching the same
-- historical bars on every chart-data, confluence-chart, refresh,
-- backtest, mass-builder, or worker-warmup call.
--
-- Schema rationale:
--   - Composite primary key (symbol, timeframe, ts) — same bar on
--     repeated fetch is idempotent via ON CONFLICT DO NOTHING
--   - Session NOT in the key — we cache RAW (extended hours included)
--     and apply the session filter on read. Avoids cache duplication
--     across (RTH vs Pre-Market) filter views of the same bar
--   - Timestamps stored as TIMESTAMPTZ to preserve UTC consistency
--
-- Storage estimate: ~80 bytes/row. SPY 1Min × 1 year RTH+ext = ~150k
-- rows = ~12 MB per symbol per year. Active 10-symbol set × 2 years =
-- ~250 MB. Well within Supabase free-tier limits.

CREATE TABLE IF NOT EXISTS bar_cache (
  symbol      TEXT             NOT NULL,
  timeframe   TEXT             NOT NULL,
  ts          TIMESTAMPTZ      NOT NULL,
  open        DOUBLE PRECISION,
  high        DOUBLE PRECISION,
  low         DOUBLE PRECISION,
  close       DOUBLE PRECISION,
  volume      DOUBLE PRECISION,
  -- Cache provenance for debugging — drop later if not useful
  cached_at   TIMESTAMPTZ      NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, timeframe, ts)
);

-- Range-query index. Covers the common access pattern:
--   SELECT * FROM bar_cache
--   WHERE symbol=? AND timeframe=? AND ts BETWEEN ? AND ?
--   ORDER BY ts
-- The composite PK already supports this exact pattern, so this index
-- is technically redundant. Kept commented as a reminder; uncomment if
-- query plans show seq scans on large tables.
--
-- CREATE INDEX IF NOT EXISTS bar_cache_range_idx
--   ON bar_cache (symbol, timeframe, ts);

-- RLS notes: Disabled for now. The bar cache is global market data, not
-- user-scoped. Anyone reading the table sees the same rows. If we ever
-- want per-user cache (unlikely for OHLCV which is the same for all),
-- add user_id column + RLS policies later.
ALTER TABLE bar_cache DISABLE ROW LEVEL SECURITY;

-- Worker / API services use the service-role key which bypasses RLS
-- anyway, but explicit DISABLE makes the intent clear.

-- =================================================================
-- Verification queries (paste into Supabase SQL editor):
--
-- 1. Confirm table exists + structure:
--    \d bar_cache
--
-- 2. Confirm primary key:
--    SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'bar_cache';
--
-- 3. After first cached run, sanity check:
--    SELECT symbol, timeframe, count(*) AS rows,
--           min(ts) AS earliest, max(ts) AS latest
--    FROM bar_cache GROUP BY symbol, timeframe;
-- =================================================================
