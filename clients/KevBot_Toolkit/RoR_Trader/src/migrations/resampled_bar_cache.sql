-- Canonical Resampled Bar Store (M-RS2 Phase 2)
-- =================================================================
-- ⚠️ NOT YET APPLIED. This is the M1b schema, authored during the M1a
--    offline byte-identity milestone. Do NOT run it until the offline
--    matrix (src/_resampled_store_byteident.py) is green AND the main
--    session has reviewed the store against the byte-identity gate.
--    Creating this table writes to prod; the M1a milestone is read-only.
-- =================================================================
--
-- Session-keyed store of RESAMPLED COARSE bars (2Min..1Day) that BOTH the
-- live and backtest lanes read, so they can no longer construct the same
-- confluence-gate bar via different code paths (the 313 / 325 / 338 class).
--
-- The store must be BYTE-IDENTICAL to the canonical definition:
--   resample_to_timeframe( _filter_session( load_1min(window), session ), tf )
-- (see resampled_bar_store.py). Resample ALWAYS from 1Min, never native coarse
-- (Polygon native daily has split bugs — CLAUDE.md).
--
-- Schema rationale (extends the bar_cache model; contrast noted):
--   - SESSION is a FIRST-CLASS KEY here (unlike bar_cache, which caches RAW
--     all-hours 1Sec/1Min and filters on read). A coarse bar is DEFINED by its
--     session: the RTH 4H bar and the Extended-Hours 4H bar aggregate different
--     intraday members, so they are different rows. Skipping the session key is
--     exactly what caused 313.
--   - timeframe_seconds INT (not the TF string): the store's TF identity is the
--     resolution in seconds (2Min=120 .. 1Day=86400). resampled_bar_store.tf_seconds()
--     bridges the canonical TF strings to this.
--   - source_1min_span_hash: hash of the 1Min bars that produced the coarse bar.
--     Detects when the underlying 1Min changed (T+2 revision / settle-sweeper
--     correction) so the affected bucket can be re-resampled.
--   - settled / revised_at mirror bar_cache write-through semantics. NOTE the
--     write path must WINDOW-REPLACE (delete+insert the (symbol,tf,session,day)
--     scope), NOT blind-upsert — an upsert can't remove a bar Polygon retracted.
--   - Timestamps stored as TIMESTAMPTZ (UTC), like bar_cache.
--
-- Storage estimate: coarse bars are tiny vs 1Min. SPY 4H RTH ~2 bars/day, 1Day
--   ~1/day; the whole coarse fleet across all sessions is a few thousand rows per
--   symbol-year. Negligible next to bar_cache's 14M 1Min/1Sec rows.

CREATE TABLE IF NOT EXISTS resampled_bar_cache (
  symbol                TEXT             NOT NULL,
  timeframe_seconds     INTEGER          NOT NULL,
  session               TEXT             NOT NULL,
  ts                    TIMESTAMPTZ      NOT NULL,
  open                  DOUBLE PRECISION,
  high                  DOUBLE PRECISION,
  low                   DOUBLE PRECISION,
  close                 DOUBLE PRECISION,
  volume                DOUBLE PRECISION,
  -- Fidelity / maintenance provenance
  source_1min_span_hash TEXT,
  settled               BOOLEAN          NOT NULL DEFAULT true,
  revised_at            TIMESTAMPTZ,
  PRIMARY KEY (symbol, timeframe_seconds, session, ts)
);

-- The composite PK (symbol, timeframe_seconds, session, ts) already covers the
-- primary access pattern (range read for one symbol/tf/session), same as
-- bar_cache. Add an explicit range index only if query plans show seq scans.

-- Unsettled-tail index — mirrors bar_cache_unsettled_index.sql. Keeps the
-- settle-sweep UPDATE (flip aged-out unsettled rows) and the admin freshness
-- view sub-millisecond even as the table grows.
CREATE INDEX IF NOT EXISTS idx_resampled_bar_cache_unsettled
  ON resampled_bar_cache (symbol, timeframe_seconds, session)
  WHERE settled = false;

-- Global market data, not user-scoped (same as bar_cache). Service-role key
-- bypasses RLS anyway; explicit DISABLE makes intent clear.
ALTER TABLE resampled_bar_cache DISABLE ROW LEVEL SECURITY;

-- =================================================================
-- Verification queries (paste into Supabase SQL editor after M1b apply):
--
-- 1. Structure + PK:
--    \d resampled_bar_cache
--    SELECT indexname, indexdef FROM pg_indexes
--     WHERE tablename = 'resampled_bar_cache';
--
-- 2. Coverage per (symbol, tf, session) after first build:
--    SELECT symbol, timeframe_seconds, session, count(*) AS bars,
--           min(ts) AS earliest, max(ts) AS latest
--    FROM resampled_bar_cache
--    GROUP BY symbol, timeframe_seconds, session
--    ORDER BY symbol, timeframe_seconds, session;
--
-- 3. Freshness / unsettled tail:
--    SELECT symbol, timeframe_seconds, session, count(*) AS unsettled
--    FROM resampled_bar_cache WHERE settled = false
--    GROUP BY symbol, timeframe_seconds, session;
-- =================================================================
