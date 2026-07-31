-- ============================================================================
-- pilot_alerts — the V5.1 pilot fleet's ENTIRE write surface (board #163).
-- 2026-07-31 · authored by E·auto (step 3) · AUTHORED, NOT APPLIED.
--
-- ⛔ DO NOT RUN THIS UNTIL KEVIN HAS AUTHORIZED IT BY NAME (board #163 step 4).
--    Authorization path: author (this file) -> M -> Kevin authorizes by name ->
--    M or Kevin applies in the Supabase SQL Editor -> only THEN may anything
--    write to it. A dispatch brief is not authorization; a relayed "OK" is not
--    authorization.
--
-- ----------------------------------------------------------------------------
-- WHAT THIS MIGRATION DOES, EXACTLY
--
--   CREATE TABLE pilot_alerts  + 2 indexes on it + RLS on it + comments on it.
--
-- WHAT IT DOES NOT TOUCH — this is the whole safety argument, so it is stated
-- as a promise and not left to be inferred:
--
--   * It does NOT alter, index, constrain, rename or backfill `alerts`.
--   * It does NOT alter `trades`, `strategies`, `monitor_status`, `live_bars`,
--     `bar_cache`, `resampled_bar_cache`, `bar_diagnostics`,
--     `bar_engine_states`, `engine_audit_log`, or any other existing table.
--   * It creates NO foreign key. In particular `user_id` is a bare UUID with no
--     REFERENCES auth.users(id) — deliberately, even though `alerts` has one.
--     An FK would install referential-integrity triggers ON auth.users, i.e. it
--     would modify an existing object, which is exactly the property this
--     migration is claiming it does not have. The pilot is a throwaway
--     experiment; it does not get to touch the auth table to get it.
--   * It takes NO lock on any hot table. Every index here is built on a table
--     created empty in the same script, so the ACCESS EXCLUSIVE lock is on a
--     brand-new, zero-row, zero-reader relation. This is why CREATE INDEX
--     CONCURRENTLY is NOT used and is NOT needed here (cf. the open
--     "CREATE INDEX CONCURRENTLY on hot tables" hygiene item, which is about
--     `alerts` at ~645k rows — a table this file never names in DDL).
--
-- Net effect on the money path if this is applied and the pilot never runs:
-- one empty table nobody reads. Retiring it is `DROP TABLE pilot_alerts;`
-- (see the bottom of this file) and nothing else has to be unwound.
--
-- ----------------------------------------------------------------------------
-- WHY A NEW TABLE INSTEAD OF A LANE COLUMN ON `alerts`
-- (Design_V51_Pilot_Fleet_Tagging.md §1.2; M verified all three premises)
--
-- The existing precedent for lane-scoping is `trades.data_source`
-- (`backtest_rest_hifi`, `cache_cache_locked`), and the instinct was to follow
-- it. It does not transfer, because the two tables have OPPOSITE defaults:
--
--   * On `trades`, every reader already scopes by lane, so a NEW `pilot_%`
--     prefix is INVISIBLE to all of them. Safe by default.
--   * On `alerts` there has never been a second lane — `source` is uniformly
--     'ralph' (it names the producer, not a lane) and ~32 read sites read the
--     table unscoped. A discriminator column there would make pilot rows
--     VISIBLE to every reader on day one, with correctness contingent on ~17
--     edits landing AND staying landed. That is the
--     feedback_lane_filter_symmetry failure with every read on the wrong side.
--
-- The failure mode being engineered out is SILENCE, not error: a pilot row that
-- throws is fine; a pilot row that quietly joins the money lane's statistics is
-- the one that costs a week chasing a divergence that was never real. The two
-- reads that make that concrete: get_strategy_health (strategy_health.py:377)
-- would pair pilot alerts against backtest edges and INFLATE paired-% — the
-- headline metric AND the #264 pre-split baseline ruler this entire V5 split is
-- measured against — and pack_canary_service.py:243-250, which IS the parity
-- gate. Neither needs a single line changed under this design, because neither
-- has ever heard of this table.
--
-- So: a pilot row is distinguishable from a live row because it is in a table
-- the live fleet never writes. That is a structural property, not a convention
-- that 32 readers have to keep honouring.
--
-- ----------------------------------------------------------------------------
-- THE LIMITATION THIS TABLE EXISTS TO MEASURE RATHER THAN FOOTNOTE
--
-- The pilot reads bars from `live_bars` (source='ws_agg') instead of opening a
-- second Polygon socket — that is what makes V5.1 cheap. It also means the
-- pilot runs ~3.5 s median (p90 ~4-6 s, measured 07-27) BEHIND a fleet reading
-- the socket. Fine for "do the two fleets take the same TRADES?", fatal for "do
-- they take them at the same TIME?" — V5.1 answers the first and must never be
-- quoted on the second (that is 5.2).
--
-- `bar_written_at` / `bar_read_at` are on every row so the asymmetry is
-- measured PER DECISION rather than assumed at a session average. The step-6
-- verdict can then say "the decision sets differ by N, and the observed feed lag
-- on those N was X±Y" instead of leaving the reader to guess whether N is logic
-- or lag.
--
-- Run in the Supabase SQL Editor. Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS pilot_alerts (
    id                BIGSERIAL PRIMARY KEY,

    -- ---- fleet provenance: the tag ----------------------------------------
    -- fleet_tag names the fleet AND its feed, in the existing `<lane>_<model>`
    -- shape (cf. backtest_rest_hifi, cache_cache_locked) so it reads as a
    -- sibling rather than a new vocabulary. V5.1 value: 'pilot_ws_agg_db'.
    fleet_tag         TEXT        NOT NULL,

    -- Deploy provenance (shadow_worker.py:52-70, the 2026-07-07 lesson):
    -- "which code produced this row" must be answerable FROM THE ROW, not from
    -- Railway metadata that has since rolled forward. sha256(engine srcs)[:12].
    code_fingerprint  TEXT,

    -- The RORT_* delta vs the live fleet. Without it a divergence is
    -- unattributable to the flag that caused it and the pilot answers nothing —
    -- which is the entire point of running a second fleet on a different flag
    -- set. Empty object = "no delta declared", NOT "no flags" (the writer is
    -- required to stamp the delta explicitly; see feedback_no_silent_defaults).
    flag_set          JSONB       NOT NULL DEFAULT '{}'::jsonb,

    -- ---- the feed asymmetry, measured per decision -------------------------
    bar_source        TEXT        NOT NULL DEFAULT 'live_bars_ws_agg',
    bar_written_at    TIMESTAMPTZ,   -- live_bars.written_at of the deciding bar
    bar_read_at       TIMESTAMPTZ,   -- when the pilot actually pulled that bar

    -- ---- alert shape, mirroring ALERT_COLUMN_FIELDS (db.py:331) ------------
    -- Mirrored so the Strategy-Health pairing algorithm and the existing
    -- alert row-shaping can be reused verbatim against this table.
    -- user_id: bare UUID, no FK — see the header. NOT the live user's row in
    -- any sense; it only records which user's strategy the pilot evaluated.
    user_id           UUID        NOT NULL,
    type              TEXT        NOT NULL,   -- 'entry_signal' | 'exit_signal'
    strategy_id       INTEGER     NOT NULL,
    strategy_name     TEXT,
    symbol            TEXT        NOT NULL,
    direction         TEXT,
    timeframe         TEXT,

    -- Trade_Timestamps_Spec 4-timestamp model + metadata.
    event_type        TEXT,
    side              TEXT,
    trigger_ts        TIMESTAMPTZ,
    fill_ts           TIMESTAMPTZ,
    exec_type         TEXT,
    trigger_id        TEXT,
    price             DOUBLE PRECISION,
    actual_price      DOUBLE PRECISION,
    bar_time          TIMESTAMPTZ,
    hold_duration_s   DOUBLE PRECISION,
    behavior          TEXT,
    live_model        TEXT,

    data              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- ---- the tag cannot lie ------------------------------------------------
    -- Enforced in the DB, not in the writer: no row in this table can ever
    -- carry a tag that reads as a live/backtest lane. A writer bug that stamps
    -- 'ralph', '' or 'backtest_rest_hifi' fails loudly on INSERT instead of
    -- producing a row that a later analyst mistakes for a live one.
    -- (left()/length() rather than LIKE 'pilot\_%' so the constraint does not
    -- depend on backslash-escape semantics to mean what it reads as.)
    CONSTRAINT pilot_alerts_fleet_tag_is_pilot
        CHECK (left(fleet_tag, 6) = 'pilot_' AND length(fleet_tag) > 6)
);

-- DELIBERATELY ABSENT — this is a rail, not an oversight.
--   acknowledged · webhook_sent · webhook_deliveries ·
--   verification_status · verification_close_delta · verification_completed_at ·
--   would_fire_post_correction · source · orphaned_at · orphaned_reason
--
-- Those columns only mean something for a row on the money path. Omitting them
-- means a pilot row cannot even be SHAPED like a delivered live alert — and
-- because PostgREST rejects an unknown column with PGRST204, a pilot writer
-- that ever tries to record a webhook delivery or a verification result FAILS
-- LOUDLY at the insert rather than silently succeeding. That is the intended
-- behaviour (feedback_no_silent_defaults). Do not "fix" such an error by adding
-- the column; it is the rail firing.
--
-- `source` is omitted for the same reason it could not be used as the lane axis:
-- it names the producer ('ralph'), and fleet_tag now carries that meaning here.

-- Primary read: pairing / comparison for one strategy over a session window.
CREATE INDEX IF NOT EXISTS pilot_alerts_sid_ts_idx
    ON pilot_alerts (strategy_id, timestamp DESC);

-- Fleet-wide read: "everything fleet X decided this session", and the scoping
-- key for the retirement DELETE in the cleanup note below.
CREATE INDEX IF NOT EXISTS pilot_alerts_fleet_idx
    ON pilot_alerts (fleet_tag, timestamp DESC);

-- ---- RLS: ON, with NO policies ---------------------------------------------
-- Fail-closed. The service-role (admin) client bypasses RLS and is the only
-- writer and the only intended reader; every other client — anon, a logged-in
-- user JWT, the frontend — sees ZERO rows rather than pilot decisions rendered
-- somewhere they would be read as real.
--
-- NOTE, deviation from Design_V51_Pilot_Fleet_Tagging.md §2.1, which said
-- "RLS: disabled, same posture as live_bars": live_bars is global market data
-- with no user column, so disabling RLS there costs nothing. pilot_alerts
-- carries user_id and strategy_id and looks exactly like a trading decision, so
-- it takes the stricter of the two house postures — the one t0_health_snapshots
-- and run_history use. Strictly more isolating than the accepted design, and
-- reversible with one ALTER if a reader ever legitimately needs it.
ALTER TABLE pilot_alerts ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE pilot_alerts IS
    'V5.1 pilot fleet decisions (board #163). NOT the money path, NOT live '
    'alerts, NEVER delivered to a webhook. Written only by a pilot engine '
    'reading bars from live_bars (ws_agg) and therefore running ~3.5s behind '
    'the live fleet: valid for comparing WHICH trades the fleets take, invalid '
    'for comparing WHEN. The live fleet writes public.alerts and never this '
    'table. Nothing in the product reads this table.';

COMMENT ON COLUMN pilot_alerts.fleet_tag IS
    'Fleet + feed, e.g. pilot_ws_agg_db. CHECK-constrained to pilot_% so a row '
    'here can never claim a live or backtest lane name.';
COMMENT ON COLUMN pilot_alerts.flag_set IS
    'The RORT_* flag delta vs the live fleet at decision time — what makes a '
    'divergence attributable.';
COMMENT ON COLUMN pilot_alerts.bar_written_at IS
    'live_bars.written_at of the deciding bar; with bar_read_at this measures '
    'the DB-feed lag per decision instead of assuming a session average.';

-- ============================================================================
-- RETIREMENT (not part of this migration — do not run now)
--
--   Scoped:  DELETE FROM pilot_alerts WHERE fleet_tag = 'pilot_ws_agg_db';
--   Total:   DROP TABLE pilot_alerts;
--
-- Either is optional and neither has an ordering dependency, because no live
-- table was touched and nothing reads this one. That is the payoff of the
-- one-new-table design.
-- ============================================================================
