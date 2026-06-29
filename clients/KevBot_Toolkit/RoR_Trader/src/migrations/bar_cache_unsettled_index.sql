-- bar_cache: partial index to support settle_sweep() on high-row symbols.
--
-- Context (2026-06-29 burn-in): once RORT_BARCACHE_WRITETHROUGH was enabled, the
-- data-worker's settle_sweep() — UPDATE ... WHERE symbol=$1 AND settled=false AND
-- ts < cutoff — began timing out (57014 statement timeout) for million-row symbols
-- (TSLA, SPY, TSLL). The only index was the PK (symbol, timeframe, ts); with no
-- index on `settled`, the sweep had to scan all rows for a symbol to find the small
-- unsettled tail. Aged-out 1Sec bars could not be flipped to settled (metadata-flag
-- lag; not data corruption) and each cycle wasted DB CPU scanning to timeout.
--
-- Fix: a PARTIAL index over only the unsettled rows. Tiny (indexes ~the live tail,
-- a few thousand rows), self-maintaining (rows leave the index as they settle), and
-- turns the sweep into an index scan over just the unsettled rows for a symbol.
--
-- Applied to prod 2026-06-29 via CREATE INDEX CONCURRENTLY (non-blocking, 64s).
-- After it landed: TSLA sweep plan = Index Scan; drain of 1532 aged rows = 2.4s.
--
-- Idempotent. CONCURRENTLY cannot run inside a transaction block; run outside one
-- (psycopg autocommit / psql default). Drop to roll back:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_bar_cache_unsettled;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_bar_cache_unsettled
    ON bar_cache (symbol, ts)
    WHERE settled = false;
