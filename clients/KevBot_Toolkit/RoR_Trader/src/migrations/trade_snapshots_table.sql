-- trade_snapshots — admin trade-level fidelity snapshots (M-RS3 follow-up).
--
-- The per-strategy count/date/KPI view tells you a strategy CHANGED but not
-- WHICH trades. This stores a point-in-time capture of every backtest-lane
-- trade (keyed by entry|exit fill ts, with r/prices/reason/trigger ts), so two
-- snapshots diff to the exact added / removed / changed trades — pinpointing
-- data revision vs an engine change, and giving bit-perfect-vs-before proof.
--
-- The payload is multi-MB for the full fleet, so it is gzip+base64 in a TEXT
-- column (`snapshot_gz`) rather than raw JSONB — raw multi-MB JSONB writes are
-- the exact failure mode Phase 41 moved trades OUT of. Admin-only, service-role
-- access (RLS on, no public policy).

CREATE TABLE IF NOT EXISTS trade_snapshots (
    id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    label         TEXT        NOT NULL,
    user_id       TEXT        NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'running',  -- running|ready|failed
    n_strategies  INTEGER     NOT NULL DEFAULT 0,
    n_trades      INTEGER     NOT NULL DEFAULT 0,
    strategy_ids  JSONB       NOT NULL DEFAULT '[]'::jsonb,
    snapshot_gz   TEXT,                                    -- gzip+base64 payload
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_trade_snapshots_user_created
    ON trade_snapshots (user_id, created_at DESC);

ALTER TABLE trade_snapshots ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS trade_snapshots_select_own ON trade_snapshots;
CREATE POLICY trade_snapshots_select_own ON trade_snapshots
    FOR SELECT USING (auth.uid()::text = user_id);
