-- Replay-simulated basis — per-strategy nightly opt-in (board #120, V1.16,
-- Kevin 07-26 scaling refinement). Design: Design_Replay_Sim_Basis.md §3/§7.
--
-- Kevin: "put a setting on the strategy health page, turn it off or on for each
-- different strategy ID ... that way we can be a bit more selective." The
-- nightly SIM sweep (Phase 2) is most valuable feeding the nightly bug-hunt
-- (surfacing problems BEFORE live data exists), but with many more strategies
-- planned, fleet-wide cost must NOT scale with fleet size. So the nightly sweep
-- processes ONLY opted-in strategies: cost = (# opted-in) × ~2min, a dial Kevin
-- holds directly. DEFAULT OFF (absence of a row = opted out).
--
-- A small dedicated table (not a `strategies` column) keeps this admin-only /
-- service-role, matching the rest of the SIM surface, and avoids touching the
-- user-facing `strategies` RLS. Run in Supabase SQL editor. Idempotent.

CREATE TABLE IF NOT EXISTS replay_sim_optin (
    strategy_id  BIGINT      PRIMARY KEY,          -- soft ref (sid may be deleted)
    enabled      BOOLEAN     NOT NULL DEFAULT false,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by   TEXT
);

CREATE INDEX IF NOT EXISTS idx_replay_sim_optin_enabled
    ON replay_sim_optin (enabled) WHERE enabled = true;

ALTER TABLE replay_sim_optin ENABLE ROW LEVEL SECURITY;
-- No public policies: service-role only, via the admin-gated toggle endpoint.
