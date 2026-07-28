-- t0_health_snapshots — daily T+0 fidelity snapshot (board #160).
-- Run this in the Supabase SQL Editor.
--
-- WHY (the trap this closes): every intraday fidelity check pits TODAY's
-- unsettled T+0 data against fully-settled prior days (their bars are final and
-- last night's settle rebuilt their lanes). That comparison is structurally
-- guaranteed to look bad, EVERY day, forever — so mid-session we cannot answer
-- the only question that matters: *is today abnormal?* We have never recorded
-- what a NORMAL T+0 looks like. Worse, the evidence is destroyed nightly: the
-- settle pass delete->reinserts each day's trade rows, so a day's T+0 state
-- cannot be reconstructed after the fact (07-27: 338's rows all rewritten
-- 00:42-00:48Z).
--
-- WHAT: one row per strategy per session, captured at a FIXED clock mark
-- (~15:00Z, ~90 min after the open — what Kevin actually looks at) and RETAINED
-- forever. Stores paired/phantom/missed/tbd/combined at BOTH 10s and 60s
-- tolerance (alert-history lane = live alerts vs backtest edges, the
-- get_strategy_health default), plus the coverage/lane-asymmetry context and
-- alert-lane freshness at capture time. Written ONLY at the T+0 mark, NEVER by
-- the nightly settle — so the retained value is the genuine unsettled state.
--
-- PAYOFF: tomorrow, "is today weaker than usual?" becomes a T+0-vs-T+0 lookup
-- answered in seconds, and the divergence loop finally gets a trustworthy
-- intraday paired-% signal (memory divergence_loop_cadence).
--
-- Additive: NEW table only. Admin-only via service-role writes (data-worker) +
-- admin-gated reads (same posture as run_history / trade_snapshots): RLS on, no
-- public policies. Service role bypasses RLS.

CREATE TABLE IF NOT EXISTS t0_health_snapshots (
    id                       BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

    -- Identity / dedup key: one row per (trading day, strategy).
    snapshot_date            DATE        NOT NULL,        -- UTC trading day
    strategy_id              INTEGER     NOT NULL,

    -- Capture metadata.
    captured_at              TIMESTAMPTZ NOT NULL,         -- exact wall-clock of the capture
    target_utc               TEXT,                         -- intended mark, e.g. '15:00'
    health_now               TIMESTAMPTZ,                  -- the `now` the health call used
    window_hours             INTEGER,                      -- divergence window used

    -- Strategy descriptors, denormalized so a snapshot stays interpretable even
    -- if the strategy is later edited or deleted.
    user_id                  TEXT,
    symbol                   TEXT,
    timeframe                TEXT,
    direction                TEXT,
    data_source              TEXT,
    backtest_model           TEXT,
    forward_testing          BOOLEAN,
    trade_count_backtest      INTEGER,

    -- Alert-history lane counts @ 10s tolerance (Kevin's tradable bar).
    paired_10                INTEGER,
    phantom_10               INTEGER,
    missed_10                INTEGER,
    tbd_10                   INTEGER,
    combined_pct_10          NUMERIC(6,2),

    -- Alert-history lane counts @ 60s tolerance (the dashboard default).
    paired_60                INTEGER,
    phantom_60               INTEGER,
    missed_60                INTEGER,
    tbd_60                   INTEGER,
    combined_pct_60          NUMERIC(6,2),

    -- Lane-asymmetry / coverage context. This is what lets a reader tell a
    -- genuinely weak day from a merely-unsettled one: how far the backtest lane
    -- had actually caught up at capture, and whether the nightly-only ALGO lane
    -- (cache_% trades) even existed yet at T+0 (comparing to it is the exact
    -- asymmetry M hit on 07-27).
    last_recompute_until_ts  TIMESTAMPTZ,
    bt_current_through       TIMESTAMPTZ,                  -- shadow-engine heartbeat edge
    bt_heartbeat_at          TIMESTAMPTZ,
    snapshot_age_sec         INTEGER,                      -- engine snapshot staleness
    last_alert_at            TIMESTAMPTZ,                  -- alert-lane freshness = alert-lag proxy
    last_alert_age_sec       INTEGER,
    algo_lane_present        BOOLEAN,                      -- cache_% algo-lane trades present today at capture?

    -- Full fidelity: the complete get_strategy_health row (tol=60) so future
    -- fields are recoverable without a schema change.
    health_row               JSONB,

    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT t0_health_snapshots_day_sid_uniq UNIQUE (snapshot_date, strategy_id)
);

-- Primary read: "how has THIS strategy's T+0 looked over the last N days?"
CREATE INDEX IF NOT EXISTS idx_t0_health_sid_date
    ON t0_health_snapshots (strategy_id, snapshot_date DESC);

-- Fleet read for a given day.
CREATE INDEX IF NOT EXISTS idx_t0_health_date
    ON t0_health_snapshots (snapshot_date DESC);

ALTER TABLE t0_health_snapshots ENABLE ROW LEVEL SECURITY;
-- No public policies: only the service-role (admin) client reaches this table,
-- via the data-worker capture thread + admin-gated API/analysis. Service role
-- bypasses RLS.
