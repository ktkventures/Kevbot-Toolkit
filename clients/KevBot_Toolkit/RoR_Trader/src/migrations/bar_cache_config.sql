-- M-RS2 Phase 1 — bar-cache capture config (the admin control surface)
-- =====================================================================
-- One row per (symbol, timeframe) the user wants captured + kept updated.
-- The admin page CRUDs this table; a maintenance loop reads the enabled
-- rows and calls bar_cache.maintain_symbol() for each.
--
--   - enabled       : toggle capture on/off without deleting the row
--   - capture_days  : how far back to backfill (NULL → resolution default)
--   - last_*_at     : provenance for the admin page (last backfill / maintain)
--
-- Native-only resolutions to capture (Kevin's plan): 1Sec + 1Min are the
-- stored source layers; sub-minute materializes from 1Sec, coarse from 1Min
-- in Phase 2 (serving). Storing 1Sec + 1Min gives the "ample supply".

CREATE TABLE IF NOT EXISTS bar_cache_config (
  symbol              TEXT        NOT NULL,
  timeframe           TEXT        NOT NULL,
  enabled             BOOLEAN     NOT NULL DEFAULT true,
  capture_days        INTEGER,                 -- NULL = resolution default
  last_backfill_at    TIMESTAMPTZ,
  last_maintained_at  TIMESTAMPTZ,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, timeframe)
);

-- Global market-data config (not user-scoped) — same posture as bar_cache.
ALTER TABLE bar_cache_config DISABLE ROW LEVEL SECURITY;

-- Verification:
--   SELECT * FROM bar_cache_config ORDER BY symbol, timeframe;
