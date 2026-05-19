# Backtest "Update Data" Speed — Implementation Plan (2026-05-19)

Companion to `Backtest_Speed_Analysis_2026-05-19.md` (the diagnosis).
This is the build plan for the three tiers. Written while context is
fresh — treat each tier's "Open questions" as gates to resolve before
coding that tier.

## Connective tissue

All three tiers are the same lesson the Worker lag fix taught:
**persist incremental state and resume; never replay history you have
already processed.** The indicator engine (`IncrementalIndicatorEngine`)
is already a pure bar-by-bar machine. `snapshot_state()` /
`restore_state()` (built for the 2026-05-19 lag fix) already deep-copy
exactly the state a resume needs. The tiers progressively exploit that.

Key code map:
- `app.py::bulk_refresh_all_strategies` — the "Update All Data" loop.
- `app.py::refresh_strategy_data` — per-strategy refresh (incremental
  path + cold-start path).
- `app.py::_generate_incremental_trades` — windowed trade generation.
- `app.py::get_strategy_trades` / `_unified_trades` — full backtest.
- `unified_engine.py::run_unified_backtest` — bar-by-bar engine.
- `unified_engine.py::IncrementalIndicatorEngine` —
  `update_bar`, `snapshot_state`, `restore_state`, `_pre_bar_snapshot`.

---

## TIER 1 — Parallelize + fix O(N²) I/O in bulk refresh

**Goal:** make "Update All Data" tolerable and stop the cron crashes.
No engine-semantics change.

### Problems being fixed
1. `bulk_refresh_all_strategies` is a sequential `for` loop.
2. `refresh_strategy_data` calls `load_strategies()` (whole collection)
   on entry and saves the whole collection on exit — **per strategy**.
   N strategies ⇒ N full loads + N full saves.

### Steps
1. **Split `refresh_strategy_data`** into:
   - `refresh_strategy_data_inplace(strat, *, persist=False)` — operates
     on an already-loaded `strat` dict, mutates it, does NOT load/save
     the collection. Returns success bool.
   - `refresh_strategy_data(strategy_id)` — thin wrapper that keeps the
     current behavior (load → call inplace → save) for single-strategy
     callers (e.g. `_do_chart_refresh`).
2. **Rewrite `bulk_refresh_all_strategies`:**
   - `load_strategies()` **once**.
   - Refresh all processable strategies (in memory) via the inplace fn.
   - `save_strategies()` **once** at the end (or batched in chunks if a
     single write is too large for the DB path).
3. **Parallelize** the refresh loop with a `ThreadPoolExecutor`
   (4–8 workers). Each task is independent (distinct strat dict).

### Open questions / risks (resolve before coding)
- **Streamlit cache from worker threads.** `_generate_incremental_trades`
  → `prepare_data_with_indicators`, which is `@st.cache_data`. Calling
  it from non-main threads emits "missing ScriptRunContext" warnings and
  may bypass caching. Options: (a) add a plain (non-`st.cache_data`)
  data-load entry point used by the bulk path; (b) use a process pool;
  (c) pre-warm the cache on the main thread, then parallelize only the
  CPU/trade-gen step. **Decision needed.** (a) is cleanest.
- **DB write size.** One `save_strategies()` of the whole collection —
  confirm the DB path (`USE_DB`) handles a large single upsert; chunk if
  not. Never pass partial dicts through `_strategy_to_row` (see
  `feedback_jsonb_partial_updates` — wipes JSONB).
- **Failure isolation.** One strategy raising must not abort the batch
  or corrupt the single save — collect per-strategy success/failure,
  save the successes.
- Thread-safety of `load_strategies`/shared module state during the
  parallel section — the parallel work should be pure compute on
  independent dicts; only the final save touches shared state.

### Validation
- Bulk-refresh a copy of production strategies; diff `stored_trades` /
  `kpis` against a sequential run — must be identical.
- Time it: target ≥4× wall-clock improvement.
- Confirm the cron job completes without timeout/OOM.

### Estimate
Small–moderate. ~1 focused session. Low risk (no engine change) once the
Streamlit-cache question is decided.

---

## TIER 2 — Persist engine state for true incremental updates

**Goal:** "Update New Data" = restore snapshot + apply only new bars.
O(new bars), no warmup reload. Also fixes the open-trade-exit bug.

### Current bug this also fixes
`refresh_strategy_data`'s incremental path filters `entry_time > last`,
so a trade that was **open** at the last refresh never gets its exit
updated. Resuming from a persisted engine state (which includes the
position state machine) closes that gap.

### Steps
1. **Serialize the resume state.** After a backtest/refresh, persist:
   - the `IncrementalIndicatorEngine` snapshot (`snapshot_state()`),
   - the `PositionStateMachine` state,
   - the last processed bar timestamp,
   - any cross-TF (`_mtf_confluence` / secondary builder) state,
   - a config fingerprint (hash of the strategy's indicator/trigger/
     stop/confluence config).
   Store as a blob keyed by strategy_id (DB `bytea` or a side table).
2. **Resume path** in `_generate_incremental_trades`: if a valid
   persisted state exists (fingerprint matches), restore it and feed
   only bars after `last_processed_ts` through `update_bar` +
   position/trigger evaluation. Else fall back to the current windowed
   path (and persist fresh state at the end).
3. **Invalidate** the persisted state whenever the strategy config
   changes (fingerprint mismatch) — forces one clean full rebuild.
4. Persist updated state at the end of every refresh.

### Open questions / risks
- **Serialization format.** Deep-copy works in-memory; persistence needs
  `pickle` (user-pack engine objects are arbitrary classes — JSON is not
  viable). Pickle version/security: it is our own trusted data, but pin
  it and guard unpickle failures → fall back to full rebuild.
- **Schema drift.** A pickled engine from an old code version may not
  load after a deploy. Store a code/schema version with the blob;
  mismatch ⇒ discard + full rebuild. (Cheap insurance.)
- **VWAP / session-cumulative state** — already inside `IndicatorState`,
  so it travels with the snapshot. Verify after a session-gap reset.
- **Cross-TF** — secondary builders + `_mtf_confluence` must be part of
  the persisted state or cross-TF strategies resume wrong.
- **Storage cost** — one blob per strategy; size-check on the largest
  (many user packs). Compress if needed.
- Backtest-engine path currently leaves `_snapshot_enabled = False`
  (set True only by live `StrategyMonitor`/`_ShadowIndicatorEngine`).
  The backtest/refresh path would need to opt in.

### Validation
- Resume-vs-rebuild parity: an incrementally-updated strategy must
  produce byte-identical `stored_trades` + `kpis` to a from-scratch
  full backtest over the same total data.
- Specifically test a strategy that is **open** across the update
  boundary — its exit must now update correctly.
- Config-edit → fingerprint mismatch → clean rebuild.

### Estimate
Moderate–large. Its own focused project. Reuses `snapshot_state` from
the lag fix; the new surface is serialization + invalidation + the
resume wiring.

### Depends on
Tier 1 (so the bulk path is already structured around an in-memory
refresh it can hang resume logic onto). Not a hard dependency, but
cleaner after.

---

## TIER 3 — Unify full/rapid around "always start flat"

**Goal:** remove the full-vs-rapid divergence entirely. Every update is
a bounded-warmup windowed run; full-history replay is needed only for a
strategy's very first backtest.

### The contract
A strategy **starts flat at its forward-test boundary** — it never
inherits an open position across a window/reload boundary. Position-
state continuity then stops being unbounded, so an update needs only
the bounded indicator-warmup window.

### Steps (each needs spec-level detail before build)
1. **Backtest side.** A windowed backtest starting at the forward-test
   boundary begins `FLAT`; it does not synthesize an inherited position
   from pre-boundary history. (`run_unified_backtest` already has
   `include_open_position` — relate the new contract to it.)
2. **Live side.** The live engine does not fire/count an alert until it
   observes a fresh `FLAT → entry` transition after (re)start — it never
   assumes an inherited open position. This is the half that makes live
   and backtest agree. Ties into the existing position-health / monitor
   logic in `ralph_engine.py`.
3. **KPI continuity.** Decide how trades that straddle the boundary are
   counted (likely: only count trades opened at/after the boundary).
   `forward_test_start` stays the anchor (see
   `feedback_forward_test_start_immutable` — never reset it to mask a
   bug).
4. **Converge full & rapid.** Once "start flat" holds, "full" and
   "rapid" differ only in window length + stop fidelity, not semantics.
   Potentially collapse to one path with a fidelity parameter.

### Open questions / risks
- **Behavioral change to alerts.** Gating on a fresh FLAT→entry changes
  when a strategy first fires after a restart — must be intentional and
  documented; verify it does not silently drop legitimate signals.
- **Backtest/live parity is the #1 project priority** (memory:
  `feedback_indicator_warmup`). The "start flat" contract must be
  proven to keep them in agreement, not drift them apart.
- **Existing strategies** mid-open-position at rollout — define the
  migration (probably: next flat resets them onto the new contract).
- Interaction with Hi-Fi Pass 2 and webhook-origin strategies (their
  position lifecycle differs).

### Validation
- Backtest vs live parity on a basket of strategies across a restart /
  window boundary — must agree trade-for-trade.
- Confirm no legitimate first-signal-after-restart is dropped.

### Estimate
Large. Needs its own design spec before any code. This is the
principled end state, not an incidental change.

### Depends on
Tier 2 (persisted state) conceptually, and a written spec.

---

## Suggested sequencing

1. **Tier 1 now-ish** — small, safe, independent, kills the active pain
   (slow "Update All Data" + crashing cron). Resolve the Streamlit-cache
   question, then build.
2. **Tier 2 next** — own project; reuses the lag fix's `snapshot_state`;
   fixes the open-trade-exit correctness bug as a bonus.
3. **Tier 3** — write a dedicated spec first; build last. It is the
   end state that makes updates structurally cheap and erases the
   full/rapid split.

## Cross-cutting notes / gotchas to remember
- Never pass partial dicts through `_strategy_to_row` /
  `_prepare_portfolio_for_db` — emits empty JSONB, wipes the column
  (`feedback_jsonb_partial_updates`).
- `count_trading_days` on a trades DF returns 1 — always pass
  source-bar trading days to KPI calc (`feedback_trading_days_kpi`).
- `forward_test_start` is immutable — anchors to strategy creation,
  never "when it started working" (`feedback_forward_test_start_immutable`).
- No silent defaults on trading-critical fields
  (`feedback_no_silent_defaults`).
- Parity (backtest == live) is the #1 priority — every tier must be
  validated against it, not just for speed.
