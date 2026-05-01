-- Live bar cache (Milestone 8.7 — Live Bar Cache)
-- =================================================================
-- Stores OHLCV bars produced by the live worker's BarBuilder as
-- they close in real-time. Distinct from `bar_cache` (which caches
-- Polygon REST aggregates for backtest reads):
--
--   bar_cache  — REST-derived, read-driven, opt-in caching layer
--   live_bars  — WS-derived, write-driven recording from the worker
--
-- Purpose: unify the data layer so backtest, chart, and heatmap
-- can eventually read the same bytes the live worker observed.
-- Tonight's scope is recording only — read path (data_loader
-- integration) ships in a later phase. See
-- docs/Plan_Live_Bar_Cache_2026-04-30.md.
--
-- Schema rationale:
--   - Composite PK (symbol, timeframe_seconds, bar_start) — same
--     bar on duplicate write is idempotent via INSERT ON CONFLICT
--     DO NOTHING (worker may rebroadcast a corrected bar).
--   - timeframe_seconds (INT) instead of timeframe text — matches
--     ralph_engine's TIMEFRAME_SECONDS convention; avoids "1Min"
--     vs "1min" string drift.
--   - source column distinguishes 'ws' (live worker, authoritative)
--     from 'rest_backfill' (REST-derived, used to fill gaps when the
--     worker was down). Lets the empirical forward-test experiment
--     compare WS-only vs REST-equivalent backtest results.
--
-- Storage estimate: ~150 bytes/row. ~50K rows/day during RTH across
-- active symbols × TFs. <1 GB/year. Negligible. No retention policy
-- initially — see plan doc for revisit criteria.

CREATE TABLE IF NOT EXISTS live_bars (
  symbol             TEXT             NOT NULL,
  timeframe_seconds  INT              NOT NULL,
  bar_start          TIMESTAMPTZ      NOT NULL,
  open               DOUBLE PRECISION NOT NULL,
  high               DOUBLE PRECISION NOT NULL,
  low                DOUBLE PRECISION NOT NULL,
  close              DOUBLE PRECISION NOT NULL,
  volume             DOUBLE PRECISION NOT NULL,
  trade_count        INT,
  vwap               DOUBLE PRECISION,
  source             TEXT             NOT NULL DEFAULT 'ws',
  written_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, timeframe_seconds, bar_start)
);

-- Range-query index for the read path that ships later. Composite
-- PK already supports (symbol, tf, bar_start) range queries, but
-- having the explicit covering index documented makes intent clear
-- and keeps the option open if access patterns shift.
CREATE INDEX IF NOT EXISTS live_bars_lookup_idx
  ON live_bars (symbol, timeframe_seconds, bar_start);

-- Fast filter on source for the dual-source forward-test experiment
-- (compare ws vs rest_backfill rows for overlapping windows).
CREATE INDEX IF NOT EXISTS live_bars_source_idx
  ON live_bars (source, symbol, timeframe_seconds);

-- RLS: disabled. live_bars is global market data, not user-scoped.
-- Worker writes via service-role key (bypasses RLS anyway).
ALTER TABLE live_bars DISABLE ROW LEVEL SECURITY;

-- =================================================================
-- Verification queries (paste into Supabase SQL editor):
--
-- 1. Confirm table exists + structure:
--    \d live_bars
--
-- 2. After worker has written for ~10 min, sanity check:
--    SELECT symbol, timeframe_seconds, source, count(*) AS rows,
--           min(bar_start) AS earliest,
--           max(bar_start) AS latest,
--           max(written_at) AS last_write
--    FROM live_bars GROUP BY symbol, timeframe_seconds, source;
--
-- 3. Fresh-write latency (written_at vs bar_start gap should be
--    seconds, not minutes — confirms fire-and-forget is working):
--    SELECT symbol, timeframe_seconds,
--           extract(epoch FROM (written_at - bar_start)) AS lag_s
--    FROM live_bars
--    ORDER BY written_at DESC LIMIT 20;
-- =================================================================
