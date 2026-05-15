-- Phase H.1 (2026-05-15) — trade-channel shadow bar storage
-- =================================================================
-- Stores OHLCV bars built from Polygon's T (trade) WebSocket channel
-- rather than the pre-aggregated AM/A channels. Distinct from
-- `live_bars` because:
--
--   1. Multiple WAIT-TIME VARIANTS coexist for the same (symbol, tf,
--      bar_start) — we shadow 0ms / 200ms / 500ms simultaneously to
--      empirically find the latency-vs-accuracy sweet spot. PK
--      includes `source` to disambiguate.
--   2. `live_bars` is the production cache that drives algo replay +
--      backtests; this table is observation-only during the shadow
--      phase. Decoupling avoids any risk of contaminating production
--      bars if the shadow builder has bugs.
--   3. After Phase H.3 (gated promotion), the winning variant might
--      be wired back into `live_bars` directly.
--
-- Source labels in use:
--   't_wait0'   — bar closes immediately when next-bucket trade arrives
--   't_wait200' — bar closes 200ms after period end
--   't_wait500' — bar closes 500ms after period end
--
-- The trade_count column counts OHLC-eligible trades that
-- contributed to the bar. filtered_trades counts trades that arrived
-- in the bucket window but were excluded by Polygon's canonical
-- OHLC eligibility rules (Form T, Odd Lot, Cash Sale, etc.).
-- See src/polygon_conditions.py for the filter logic shared with
-- flat_file_ingestion.

CREATE TABLE IF NOT EXISTS live_bars_trades (
  symbol             TEXT             NOT NULL,
  timeframe_seconds  INT              NOT NULL,
  bar_start          TIMESTAMPTZ      NOT NULL,
  source             TEXT             NOT NULL,
  open               DOUBLE PRECISION NOT NULL,
  high               DOUBLE PRECISION NOT NULL,
  low                DOUBLE PRECISION NOT NULL,
  close              DOUBLE PRECISION NOT NULL,
  volume             BIGINT           NOT NULL,
  trade_count        INT              NOT NULL DEFAULT 0,
  filtered_trades    INT              NOT NULL DEFAULT 0,
  written_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

  CONSTRAINT live_bars_trades_pk PRIMARY KEY (symbol, timeframe_seconds, bar_start, source)
);

CREATE INDEX IF NOT EXISTS live_bars_trades_lookup_idx
  ON live_bars_trades (symbol, timeframe_seconds, bar_start DESC);

CREATE INDEX IF NOT EXISTS live_bars_trades_source_idx
  ON live_bars_trades (source, written_at DESC);

ALTER TABLE live_bars_trades DISABLE ROW LEVEL SECURITY;

COMMENT ON TABLE live_bars_trades IS
  'Phase H (2026-05-15): trade-channel shadow OHLCV. Multiple wait-time variants per (symbol, tf, bar_start) for latency-vs-accuracy comparison. Production alerts unaffected.';
