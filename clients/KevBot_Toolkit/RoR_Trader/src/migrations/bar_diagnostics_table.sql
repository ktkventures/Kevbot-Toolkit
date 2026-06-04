-- Migration: bar_diagnostics — per-bar indicator state for live/backtest comparison.
-- Apply via Supabase SQL editor before deploying the engine instrumentation.
--
-- Purpose: capture the few indicator values per pack that we care about
-- (declared as `diagnostic_columns` in each pack manifest) so we can
-- compare live vs backtest state at the same bar and pinpoint where
-- trailing-stop / hysteresis-style indicators diverge — the root cause
-- behind today's phantom-entry investigation (task #53).
--
-- Storage estimate: 50 strategies × 8640 bars/day × ~30 bytes ≈ 13 MB/day.
-- Manageable on Supabase. Periodic GC (drop rows older than 60d) handled
-- elsewhere.

CREATE TABLE IF NOT EXISTS bar_diagnostics (
    strategy_id INT NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    bar_ts TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('live', 'backtest', 'algo')),
    values JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (strategy_id, bar_ts, source)
);

CREATE INDEX IF NOT EXISTS bar_diagnostics_strategy_ts_idx
    ON bar_diagnostics (strategy_id, bar_ts DESC);

CREATE INDEX IF NOT EXISTS bar_diagnostics_created_at_idx
    ON bar_diagnostics (created_at);

-- RLS: read by any authenticated user (admin diagnostics, non-sensitive).
-- Writes are via service-role only (engine-side); no user policy needed.
ALTER TABLE bar_diagnostics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Any authenticated user can read bar_diagnostics"
    ON bar_diagnostics FOR SELECT
    USING (auth.role() = 'authenticated');

COMMENT ON TABLE bar_diagnostics IS
    'Per-bar indicator state captured opportunistically by both live (ralph_engine) and backtest (unified_engine) engines. Used to diagnose state-drift divergences. Columns logged are declared by the pack manifest field `diagnostic_columns`.';
