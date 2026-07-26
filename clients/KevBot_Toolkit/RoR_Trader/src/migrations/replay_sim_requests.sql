-- Replay-simulated basis — request queue (board #120, V1.16).
-- Design: docs/_active/Design_Replay_Sim_Basis.md §4.2 (runner-host = local,
-- option A). Mirrors run_history's declarative lifecycle: the "run SIM" button
-- (POST /api/strategy-health-sim/{sid}/run) INSERTs a row with
-- outcome='requested'; the local poller (tools/team_sim/replay_sim_poller.py)
-- claims it (outcome='running'), runs replay_sim_job, and writes the terminal
-- outcome + a link to the replay_sim_basis result row. The nightly opted-in
-- sweep (Phase 2) enqueues rows here too, with requested_by NULL.
--
-- Railway can't reach Kevin's machine, so button presses are DECLARATIVE — the
-- API only writes the row; the local poller is what EXECUTES it. Same shape as
-- the Run-button dispatcher (board #109).
--
-- Additive, service-role only. Run in Supabase SQL editor. Idempotent.

CREATE TABLE IF NOT EXISTS replay_sim_requests (
    id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    strategy_id   BIGINT      NOT NULL,
    requested_by  TEXT,                            -- button author; NULL = nightly sweep
    source        TEXT        NOT NULL DEFAULT 'button',  -- button | nightly
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    outcome       TEXT        NOT NULL DEFAULT 'requested', -- requested|running|ok|partial|unres|error|ignored
    result_id     BIGINT,                          -- → replay_sim_basis.id on success
    claimed_by    TEXT,                            -- poller worker id (replica-safe claim)
    log_tail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_replay_sim_requests_pending
    ON replay_sim_requests (outcome, requested_at);

ALTER TABLE replay_sim_requests ENABLE ROW LEVEL SECURITY;
-- No public policies: service-role only, via admin-gated API + the local poller.
