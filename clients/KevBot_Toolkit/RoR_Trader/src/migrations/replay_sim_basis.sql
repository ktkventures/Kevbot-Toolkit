-- Replay-simulated last-10 basis — results cache (board #120, V1.16).
-- Design: docs/_active/Design_Replay_Sim_Basis.md §4.1.
--
-- One row per completed replay of a strategy. The health page READS the
-- newest `ok`/`partial` row per strategy and pairs its cached would-have-fired
-- edges against the CURRENT last-10 backtest edges cheaply at read time — the
-- expensive replay (the real engine over recorded decision-time bars) is NEVER
-- run on page load. Producer = replay_sim_job.py, off-hours, prod-load-light
-- (feedback_local_analysis_starves_live_worker).
--
-- History is kept (no upsert-in-place) so a moved score is auditable.
-- Additive, service-role only (same RLS posture as run_history / compute_jobs /
-- dev_tasks). Run in Supabase SQL editor. Idempotent.

CREATE TABLE IF NOT EXISTS replay_sim_basis (
    id              BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    strategy_id     BIGINT      NOT NULL,          -- soft ref (sid may be deleted)
    symbol          TEXT        NOT NULL,          -- provenance
    timeframe_secs  INTEGER     NOT NULL,          -- provenance

    -- the window actually replayed (§6). window_since already EXCLUDES warmup;
    -- warmup depth is implicit (harness 7d fine / 60d coarse) and NOT reducible.
    window_since    TIMESTAMPTZ NOT NULL,
    window_until    TIMESTAMPTZ NOT NULL,
    warmup_days     INTEGER     NOT NULL,          -- the preamble the harness used (audit)

    -- the reference bt edge set this replay was sized to cover (§4.3 staleness).
    bt_span_start   TIMESTAMPTZ,                   -- first of the covered bt entries
    bt_span_end     TIMESTAMPTZ,                   -- last of the covered bt exits
    bt_trade_count  INTEGER,                       -- covered completed bt trades (denom basis)

    -- the replay lane's would-have-fired edges (the whole point of the cache).
    -- v1.0: ["2026-07-24T13:41:07+00:00", ...]. v1.1 enrichment (§2) may store
    -- [{"ts":..,"side":..,"price":..,"reason":..}, ...]; the reader pairs on ts.
    replay_entries  JSONB       NOT NULL DEFAULT '[]',
    replay_exits    JSONB       NOT NULL DEFAULT '[]',

    -- TRUST provenance (Kevin req 1) — everything the UI surfaces next to the score.
    corrected       BOOLEAN     NOT NULL DEFAULT false,  -- decision-time (false) vs healed (true)
    coverage        REAL,                          -- fraction of primary closes in-window that had
                                                   --   live_bars decision-time rows (§8 D4). <1 ⇒ partial.
    divergences     JSONB       NOT NULL DEFAULT '[]',   -- applicable ledger entries + run footnotes (§8)
    flags_fp        TEXT,                          -- fingerprint of ARMED_FLAGS at compute time (§8 D9)
    engine_sha      TEXT,                          -- git sha of the engine code that produced this

    status          TEXT        NOT NULL DEFAULT 'ok',   -- ok | partial | unres | error
    compute_secs    REAL,                          -- budget telemetry (§7)
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    requested_by    TEXT                           -- button author; NULL = nightly/organic
);

CREATE INDEX IF NOT EXISTS idx_replay_sim_basis_sid
    ON replay_sim_basis (strategy_id, computed_at DESC);

ALTER TABLE replay_sim_basis ENABLE ROW LEVEL SECURITY;
-- No public policies: only the service-role (admin) client reaches this table,
-- via admin-gated API endpoints + the local producer. Service role bypasses RLS.
