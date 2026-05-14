-- Polygon flat-file observable bars (Phase E, 2026-05-14)
-- =================================================================
-- 1-second OHLCV bars reconstructed from Polygon's daily trades
-- flat-file (us_stocks_sip/trades_v1/YYYY/MM/YYYY-MM-DD.csv.gz)
-- using `sip_timestamp` as the bucket key — i.e., a trade contributes
-- to the bar at second N if SIP emitted it during second N. This is
-- the "observable bars" methodology: it's what a real-time subscriber
-- WOULD have seen at the moment of decision, as opposed to settled
-- bars (REST aggregates) which include late-print corrections.
--
-- Purpose: diagnostic dataset for the Admin > Parity > Ticks tab.
-- Compare live_bars cache (what our engine wrote in real time) vs
-- observable (what was actually emitted) vs settled REST (the
-- post-correction view). If `live_bars` ≈ observable → live engine
-- is faithful, divergence is downstream. If `live_bars` ≠ observable
-- → WS handling has gaps.
--
-- Scoping decisions (2026-05-14 with Kevin):
--   - Symbols: SPY + TSLA initially. Expand via env var.
--   - Window: 7-day rolling. Daily ingestion + daily purge.
--   - Storage: ~40MB/week for SPY alone. ~$0.50/mo above Supabase
--     Pro's 8GB included tier. Negligible at this scope.
--
-- Schema rationale:
--   - Composite PK (ticker, sip_second_ts) — same second on
--     re-ingest is idempotent via INSERT ON CONFLICT DO NOTHING.
--   - sip_second_ts is TIMESTAMPTZ (second-aligned). The flat file
--     gives nanoseconds; we floor to seconds for bucketing.
--   - trade_count is informational — helps spot bars with abnormal
--     trade density (e.g., a single late-print block trade).
--   - ingested_at lets us see when each row landed and purge old
--     ingestion runs cleanly.

CREATE TABLE IF NOT EXISTS polygon_observable_bars (
  ticker         TEXT             NOT NULL,
  sip_second_ts  TIMESTAMPTZ      NOT NULL,
  open           DOUBLE PRECISION NOT NULL,
  high           DOUBLE PRECISION NOT NULL,
  low            DOUBLE PRECISION NOT NULL,
  close          DOUBLE PRECISION NOT NULL,
  volume         BIGINT           NOT NULL,
  trade_count    INT              NOT NULL DEFAULT 0,
  ingested_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

  CONSTRAINT polygon_observable_bars_pk PRIMARY KEY (ticker, sip_second_ts)
);

CREATE INDEX IF NOT EXISTS idx_polygon_observable_bars_ts
  ON polygon_observable_bars (sip_second_ts DESC);

-- Per-ticker lookup index (read path: API endpoint filters by ticker + window).
CREATE INDEX IF NOT EXISTS idx_polygon_observable_bars_ticker_ts
  ON polygon_observable_bars (ticker, sip_second_ts DESC);

-- Retention helper view — surfaces rows eligible for purge. Daily
-- cron deletes WHERE sip_second_ts < NOW() - INTERVAL '7 days'.
COMMENT ON TABLE polygon_observable_bars IS
  'Phase E (2026-05-14): 1-sec observable bars rebuilt from Polygon flat-file trades. 7-day rolling retention.';
