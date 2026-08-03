-- ============================================================================
-- dev_alerts — the dev-environment fleet's ENTIRE write surface (board #165,
-- Phase B). 2026-08-03 · authored by E·auto (step 28) · AUTHORED, NOT APPLIED.
--
-- ⛔ DO NOT RUN THIS UNTIL KEVIN HAS AUTHORIZED IT BY NAME.
--    Authorization path: author (this file) -> M -> Kevin authorizes by name ->
--    M or Kevin applies in the Supabase SQL Editor -> only THEN may any code
--    that reads or writes this table merge. A dispatch brief is not
--    authorization; a relayed "OK" is not authorization.
--
--    ORDERING: code referencing this table must NOT merge before the table
--    exists in production, or it 500s.
--
-- Design: docs/_active/Design_Live_Bars_Consumer.md
--
-- ----------------------------------------------------------------------------
-- NAMING — read this before assuming
--
--   `dev_alerts` is THE DEV ENVIRONMENT'S ALERT LANE.
--   It has NOTHING to do with `dev_tasks`, which is the team board. The two
--   names are adjacent and mean unrelated things; the prefix here is the
--   environment (#165 dev vs production), not the board.
--
-- ----------------------------------------------------------------------------
-- WHAT THIS MIGRATION DOES, EXACTLY
--
--   CREATE TABLE dev_alerts + 3 indexes on it + RLS on it + comments on it.
--
-- WHAT IT DOES NOT TOUCH — this is the whole safety argument, so it is a
-- promise stated outright rather than left to be inferred:
--
--   * It does NOT alter, index, constrain, rename or backfill `alerts`.
--   * It does NOT alter `trades`, `strategies`, `live_bars`, `bar_cache`,
--     `resampled_bar_cache`, `pilot_alerts`, `monitor_status`, or any other
--     existing table.
--   * It creates NO foreign key. `user_id` is a bare UUID with no
--     REFERENCES auth.users(id) — deliberately, even though `alerts` has one.
--     An FK would install referential-integrity triggers ON auth.users, i.e.
--     it would modify an existing object, which is exactly the property this
--     migration claims not to have.
--   * It takes NO lock on any hot table. Every index below is built on a table
--     created empty in the same script, so the ACCESS EXCLUSIVE lock is on a
--     brand-new, zero-row, zero-reader relation. This is why CREATE INDEX
--     CONCURRENTLY is not used and is not needed here (cf. the open
--     "CONCURRENTLY on hot tables" hygiene item, which is about `alerts`).
--
-- Net effect if applied and the dev fleet never runs: one empty table nobody
-- reads. Retirement is `DROP TABLE dev_alerts;` and nothing else unwinds.
--
-- ----------------------------------------------------------------------------
-- WHY A NEW TABLE AND NOT A LANE COLUMN ON `alerts`
--
-- The precedent for lane-scoping is `trades.data_source`, and it does not
-- transfer, because the two tables have OPPOSITE defaults:
--
--   * On `trades`, every reader already scopes by lane, so a new value is
--     INVISIBLE to all of them. Safe by default.
--   * On `alerts` there has never been a second lane — `source` is uniformly
--     'ralph' (it names the producer, not a lane) and NOT ONE call site is
--     lane-scoped (verified 2026-08-03: 33 `table('alerts')` sites in src/,
--     excluding one-off scripts and tests). A discriminator column there would
--     make dev rows VISIBLE to every reader on day one, with correctness
--     contingent on every one of those edits landing AND staying landed.
--
-- The two reads that make it concrete: get_strategy_health
-- (strategy_health.py:377) would pair dev alerts against backtest edges and
-- INFLATE paired-% — the headline metric AND the #264 pre-split baseline ruler
-- this entire split is measured against — and pack_canary_service.py:243-250,
-- which IS the parity gate. Neither needs a line changed under this design,
-- because neither has ever heard of this table.
--
-- The failure mode being engineered out is SILENCE, not error.
--
-- ----------------------------------------------------------------------------
-- WHY NOT REUSE `pilot_alerts` WITH A NEW TAG
--
-- `pilot_alerts` (migrations/pilot_alerts_table.sql, board #163) is the right
-- PATTERN and the wrong HOME. Two reasons:
--
--   1. Its documented retirement is `DROP TABLE pilot_alerts;` and its own
--      header calls it a throwaway experiment. A STANDING environment lane must
--      not live inside a table someone is already authorized to drop.
--   2. Its CHECK constrains fleet_tag to 'pilot_%'. The dev fleet would have to
--      call itself a pilot to satisfy a rail whose entire purpose is that the
--      tag cannot lie.
--
-- So this mirrors that table column-for-column and adds the two columns Phase B
-- needs: `outcome` and `consumer_lag_s`.
--
-- ----------------------------------------------------------------------------
-- THE ONE STRUCTURAL ADDITION: `outcome`
--
-- `pilot_alerts` records a decision TAKEN. It has no way to record a decision
-- that was IMPOSSIBLE to take — a bar that never arrived produces no row, which
-- is byte-identical, to any later analyst, to "evaluated and chose not to fire."
-- Measured: the deciding bar is absent from `live_bars` for 2.8% of 1Min live
-- decisions. Without `outcome`, that 2.8% presents as logic divergence.
--
-- So every bar-close the consumer EVALUATES gets a row, and absence is recorded
-- positively (feedback_no_silent_defaults). This was carried forward as design
-- gap #1 of V51_Pilot_Session_Verdict_2026-07-31.md §6; this column closes it.
--
-- Run in the Supabase SQL Editor. Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS dev_alerts (
    id                BIGSERIAL PRIMARY KEY,

    -- ---- fleet provenance: the tag ----------------------------------------
    -- fleet_tag names the fleet AND its feed, in the existing `<lane>_<model>`
    -- shape (cf. backtest_rest_hifi, pilot_ws_agg_db) so it reads as a sibling
    -- rather than a new vocabulary. Phase B value: 'dev_live_bars_1min'.
    fleet_tag         TEXT        NOT NULL,

    -- Deploy provenance (shadow_worker.py:52-70, the 2026-07-07 lesson):
    -- "which code produced this row" must be answerable FROM THE ROW, not from
    -- Railway metadata that has since rolled forward. sha256(engine srcs)[:12].
    code_fingerprint  TEXT,

    -- The RORT_* delta vs the live fleet. Phase B runs a NULL DELTA (identical
    -- armed stack) so any divergence is attributable to the feed path and live
    -- process state rather than to flags. Empty object = "no delta declared",
    -- NOT "no flags" — the writer must stamp the delta explicitly.
    flag_set          JSONB       NOT NULL DEFAULT '{}'::jsonb,

    -- ---- what happened at this bar close ----------------------------------
    -- THE column `pilot_alerts` lacks. See the header.
    --   decision          the consumer evaluated the bar and produced a signal
    --   bar_absent        the bar never landed in live_bars (2.8% at 1Min)
    --   bar_late          the bar arrived after the engine had moved past it;
    --                     strictly forward-only, so it was NOT replayed
    --   bar_skipped       consumer lag exceeded its ceiling; deciding stopped
    --   consumer_restart  process restarted mid-session; state re-warmed, and
    --                     the wall-clock hole is named in `data`
    outcome           TEXT        NOT NULL DEFAULT 'decision',

    -- ---- the feed asymmetry, measured per decision -------------------------
    -- A DB-fed fleet is structurally behind a socket-fed one: median +3.2s,
    -- p90 +4.1s, and the deciding bar landed AFTER live had already decided in
    -- 1,317 of 1,317 measured cases (100%, zero overlap). Recording it per row
    -- means the verdict can say "the decision sets differ by N, and the
    -- observed lag on those N was X±Y" instead of leaving a reader to guess
    -- whether N is logic or lag.
    bar_source        TEXT        NOT NULL DEFAULT 'live_bars',
    bar_written_at    TIMESTAMPTZ,        -- live_bars.written_at of the bar
    bar_read_at       TIMESTAMPTZ,        -- when the consumer pulled that bar
    consumer_lag_s    DOUBLE PRECISION,   -- bar_read_at - (bar_start + tf)

    -- ---- alert shape, mirroring ALERT_COLUMN_FIELDS (db.py:331) ------------
    -- Mirrored so the Strategy-Health pairing algorithm and the existing alert
    -- row-shaping can be reused verbatim against this table.
    -- user_id: bare UUID, no FK — see the header. It records only which user's
    -- strategy the dev fleet evaluated.
    user_id           UUID        NOT NULL,
    type              TEXT,                   -- 'entry_signal' | 'exit_signal'
    strategy_id       INTEGER     NOT NULL,
    strategy_name     TEXT,
    symbol            TEXT        NOT NULL,
    direction         TEXT,
    timeframe         TEXT,
    timeframe_seconds INT         NOT NULL,   -- the dedupe key's tf component
    bar_start         TIMESTAMPTZ NOT NULL,   -- the evaluated bar; also the
                                              -- restart cursor (design §3.3)

    -- Trade_Timestamps_Spec 4-timestamp model + metadata. NULL on every
    -- non-`decision` outcome — there was no decision to timestamp.
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
    -- Enforced in the DB, not in the writer: no row here can ever carry a tag
    -- that reads as a live or backtest lane. A writer bug stamping 'ralph', ''
    -- or 'backtest_rest_hifi' fails loudly on INSERT instead of producing a row
    -- a later analyst mistakes for a live one.
    -- (left()/length() rather than LIKE 'dev\_%' so the constraint does not
    -- depend on backslash-escape semantics to mean what it reads as.)
    CONSTRAINT dev_alerts_fleet_tag_is_dev
        CHECK (left(fleet_tag, 4) = 'dev_' AND length(fleet_tag) > 4),

    -- The outcome vocabulary is closed. A new outcome is a schema change and a
    -- design conversation, not a string a writer invents at 3am.
    CONSTRAINT dev_alerts_outcome_known
        CHECK (outcome IN ('decision', 'bar_absent', 'bar_late',
                           'bar_skipped', 'consumer_restart')),

    -- A 'decision' row must actually carry a decision. Anything else must not.
    -- This is the rail that stops a bookkeeping row from being counted as a
    -- trade by a reader that forgot to filter on `outcome`.
    CONSTRAINT dev_alerts_decision_has_side
        CHECK ((outcome = 'decision' AND side IS NOT NULL)
               OR (outcome <> 'decision' AND side IS NULL))
);

-- DELIBERATELY ABSENT — this is a rail, not an oversight.
--   acknowledged · webhook_sent · webhook_deliveries ·
--   verification_status · verification_close_delta · verification_completed_at ·
--   would_fire_post_correction · source · orphaned_at · orphaned_reason
--
-- Those columns only mean something for a row on the money path. Omitting them
-- means a dev row cannot even be SHAPED like a delivered live alert — and
-- because PostgREST rejects an unknown column with PGRST204, a consumer that
-- ever tries to record a webhook delivery or a verification result FAILS LOUDLY
-- at the insert rather than silently succeeding. That is the intended behaviour
-- (feedback_no_silent_defaults). Do not "fix" such an error by adding the
-- column; it is the rail firing.
--
-- `source` is omitted for the same reason it could not be the lane axis on
-- `alerts`: it names the producer ('ralph'), and fleet_tag carries that meaning
-- here.

-- ---- indexes ---------------------------------------------------------------

-- Primary read: pairing / comparison for one strategy over a session window.
CREATE INDEX IF NOT EXISTS dev_alerts_sid_ts_idx
    ON dev_alerts (strategy_id, timestamp DESC);

-- Fleet-wide read: "everything fleet X decided this session", and the scoping
-- key for the retirement DELETE in the cleanup note below.
CREATE INDEX IF NOT EXISTS dev_alerts_fleet_idx
    ON dev_alerts (fleet_tag, timestamp DESC);

-- THE DEDUPE RAIL (design §3.2). The GRACE-overlap poll re-reads the same rows
-- every tick by construction, and apply_rest_correction upserts already-consumed
-- live_bars rows. In-memory dedupe is the fast path; this is what makes a
-- dedupe BUG fail loudly instead of producing two decisions on one bar.
-- It is also the restart cursor's uniqueness guarantee (§3.3).
CREATE UNIQUE INDEX IF NOT EXISTS dev_alerts_dedupe_idx
    ON dev_alerts (fleet_tag, strategy_id, symbol, timeframe_seconds,
                   bar_start, outcome);

-- ---- RLS: ON, with NO policies ---------------------------------------------
-- Fail-closed. The service-role (admin) client bypasses RLS and is the only
-- writer and the only intended reader; every other client — anon, a logged-in
-- user JWT, the frontend — sees ZERO rows rather than dev decisions rendered
-- somewhere they would be read as real.
--
-- Same posture as pilot_alerts, t0_health_snapshots and run_history. NOT
-- live_bars' posture (RLS disabled), because live_bars is global market data
-- with no user column, whereas this table carries user_id and strategy_id and
-- looks exactly like a trading decision.
ALTER TABLE dev_alerts ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE dev_alerts IS
    'Dev-environment fleet decisions (board #165 Phase B). NOT the money path, '
    'NOT live alerts, NEVER delivered to a webhook. Unrelated to dev_tasks (the '
    'team board) — the prefix is the ENVIRONMENT. Written only by a consumer '
    'reading bars from live_bars and therefore running ~3.2s median behind the '
    'live fleet: valid for comparing WHICH trades the fleets take at 1Min+, '
    'invalid for comparing WHEN, and out of scope entirely below 1Min. The live '
    'fleet writes public.alerts and never this table. Nothing in the product '
    'reads this table.';

COMMENT ON COLUMN dev_alerts.fleet_tag IS
    'Fleet + feed, e.g. dev_live_bars_1min. CHECK-constrained to dev_% so a row '
    'here can never claim a live or backtest lane name.';
COMMENT ON COLUMN dev_alerts.outcome IS
    'What happened at this bar close. Absence is recorded POSITIVELY: a bar '
    'that never arrived writes bar_absent rather than no row, so a feed gap '
    'cannot be misread as "evaluated and declined to fire" (2.8% of 1Min live '
    'decisions have no bar in live_bars).';
COMMENT ON COLUMN dev_alerts.consumer_lag_s IS
    'bar_read_at - (bar_start + timeframe_seconds). The DB-feed cost per '
    'decision, measured rather than assumed at a session average.';
COMMENT ON COLUMN dev_alerts.flag_set IS
    'The RORT_* flag delta vs the live fleet at decision time — what makes a '
    'divergence attributable. Phase B runs a null delta.';

-- ============================================================================
-- RETIREMENT (not part of this migration — do not run now)
--
--   Scoped:  DELETE FROM dev_alerts WHERE fleet_tag = 'dev_live_bars_1min';
--   Total:   DROP TABLE dev_alerts;
--
-- Either is optional and neither has an ordering dependency, because no live
-- table was touched and nothing reads this one. That is the payoff of the
-- one-new-table design.
-- ============================================================================
