# Spec — LEF Phase 2 + Tier 2 Implementation

Status: DRAFT (2026-05-20). Implementation-level companion to
`Spec_Live_Execution_Fidelity.md` (motivation + model taxonomy) and
`Backtest_Speed_Implementation_Plan.md` (Tier 2 — persist snapshot).
This doc nails the architectural choices for Phase 2 so the code-paths
are unambiguous before we start.

## 1. Goal

Wire `ws_agg_reconciled` (the timeframe-aware live model) into the live
engine, so a sub-minute strategy can fire on a *decision-grade* bar
sized by its own `grace_seconds`, with indicator state reconciled
via the O(1) `apply_last_bar_correction` path on late data, and resume
backtest refreshes from a persisted engine snapshot instead of
re-warming.

## 2. Non-goals (deferred)

- **Confidence-gated firing** (LEF spec §5.3) — defer to a Phase 2.5
  enhancement. Initial cut fires unconditionally at strategy-grace.
  With default grace 5 the phantom-alert risk on a partial bucket is
  low; if we observe phantoms in production, add per-trigger margin
  detection then. Rationale: the trigger-margin work is mechanical but
  touches every trigger evaluator — best done as its own focused
  change once we have evidence it's needed.
- **Per-strategy grace UI** — Phase 5.
- **Sub-minute REST backfill** — already shipped as Phase 3a
  (`live_bars_rest_backfill_subminute.py`).
- **Bug-fix the pre-existing `strat`-undefined in
  `_generate_webhook_backtest_trades`** — orthogonal.

## 3. Architectural choices

### 3.1 Builder grace = max(strategy grace) on that (symbol, TF)

The BarBuilder stays per-(symbol, TF), one instance shared by every
strategy on it — *not* one builder per strategy. Its `force_close_
stale_bar` grace becomes the **max** of `resolve_grace_seconds(s)`
across all monitors on that builder; default falls back to the existing
`SUBMINUTE_FORCE_CLOSE_GRACE_SEC = 5` when no monitors are registered.

- Set once at engine init from the monitor set; recomputed if monitors
  are added/removed (`RalphEngine._refresh_builder_grace(tf_seconds)`).
- BarBuilder gains a `grace_sec_override: Optional[int]` slot (default
  None → constant fallback). Per-strategy values never reach the
  builder directly.

### 3.2 Per-strategy fire timing via `compute_tentative_state`

Today the monitor pipeline fires triggers on builder bar-close
(`_run_monitor_pipeline_for_completed_bar`). Phase 2 splits that into
**two events** per bucket:

| Event | When | What |
|---|---|---|
| `monitor.fire_at_strategy_grace(bucket)` | `bar_start + tf + monitor.grace_seconds` | Take a tentative snapshot via `compute_tentative_state(forming_bar)` on the still-forming bucket; evaluate triggers; fire alerts. Does NOT commit indicator state. |
| `monitor.commit_bar_close(bucket)` | builder force-closes the bucket | Commit indicator state via `engine.update_bar(closed_bar)`. NO new alert evaluation — already done at fire-time. |

The scheduler is the existing per-second event loop. On each per-second
event from `on_second_bar`, for each monitor on the (symbol, TF):
- Compute `due_at = current_bucket_start + tf_seconds + monitor.grace_seconds`.
- If wall-clock ≥ `due_at` AND monitor has not yet fired for this
  bucket → fire (steps above). Mark `monitor._fired_bucket = bucket_start`.

A `_fired_bucket` flag per monitor per (symbol, TF) prevents double-fire
within the bucket. Reset when the builder advances to the next bucket.

**Why this is safe:** the existing `compute_tentative_state` already
deep-copies and restores engine state per forming-bar broadcast — we're
reusing the same primitive, just at a strategy-driven cadence instead
of a SymbolHub one. The position state machine's `entry_time` is
preserved on the partial bucket via the existing snapshot/restore.

**Late-data reconciliation:** between strategy-grace and builder-close
the bucket may absorb more per-second data. That data lands via
`accept_second_bar` like normal; on the eventual builder close,
`commit_bar_close` runs `engine.update_bar` once with the final
aggregate — same path as today. If a Polygon REBROADCAST corrects the
just-closed bar, `apply_last_bar_correction` runs unchanged. A fired
alert stands either way (the hard line from the master spec: *never
re-fire or un-fire*).

### 3.3 `ws_agg_locked` strategies are untouched

Only strategies with `live_model == 'ws_agg_reconciled'` enter the new
fire-at-strategy-grace path. `ws_agg_locked` (and the rest) keep
firing at builder bar-close exactly as today. The dispatch gate lives
in `_run_monitor_pipeline_for_completed_bar` (already source-filter-
aware).

### 3.4 Tier 2 — snapshot persistence

After a backtest/refresh, persist the engine snapshot for resume:

- **Format:** `pickle.dumps(snapshot)` — `snapshot_state()` returns
  arbitrary user-pack objects, so JSON is not viable. Pin a
  `SNAPSHOT_SCHEMA_VERSION = 1` constant; bump it whenever the snapshot
  shape changes.
- **Storage:** new `engine_snapshot` column on the `strategies` table
  (bytea) + `engine_snapshot_meta` (jsonb) with:
  - `last_bar_ts` (ISO-UTC, the timestamp of the most recent committed
    bar in the snapshot)
  - `schema_version` (int, matches SNAPSHOT_SCHEMA_VERSION)
  - `fingerprint` (sha256 hex of canonical strategy config —
    required_indicators, params, confluence_group_ids, pack manifest
    versions)
  - `created_at` (ISO-UTC)
  - JSON-mode (`USE_DB=false`): co-located inside the strategy dict.
- **Resume path** in `_generate_incremental_trades`: if persisted meta
  exists and `fingerprint` + `schema_version` match → unpickle
  snapshot, restore via `restore_state`, fetch only bars after
  `last_bar_ts`, feed each through `engine.update_bar` + position
  evaluation. Else fall back to the current windowed warmup.
- **Invalidation:** any of `fingerprint mismatch`, `schema_version
  mismatch`, `unpickle exception` → discard, full rebuild + re-persist.
  Logged as a warning so we notice churn.
- **Update on success:** at the end of every successful refresh,
  persist a fresh snapshot keyed to the new `last_bar_ts`.

## 4. File map (what each sub-phase touches)

| Sub-phase | Files | Risk |
|---|---|---|
| 2a — wire resolver into builder grace | `ralph_engine.py` (BarBuilder, RalphEngine), `strategy_models.py` (already has resolver) | Low — global behavior unchanged when all strategies use the default 5 |
| 2b — monitor fires at strategy-grace via tentative state | `ralph_engine.py` (StrategyMonitor + the per-second event loop in `on_second_bar`), `_run_monitor_pipeline_for_completed_bar` | High — touches the alert-firing path |
| 2c (Tier 2) — persist engine snapshot | `unified_engine.py` (serialize/restore + fingerprint), `app.py` (refresh path resume logic), `db.py` + a migration | Medium — backtest-side; failure modes fall back to current behavior |

Each sub-phase ships as its own commit. 2a is a refactor-only no-op (default grace stays 5 across all strategies until per-strategy override is set). 2b is the real behavior change — only active for strategies that pick `ws_agg_reconciled`, which is still `available: False` in `LIVE_MODELS`. 2c is independent of 2a/2b and can ship anytime.

## 5. Failure modes & fallback

- `compute_tentative_state` raises → log warning, monitor falls back to
  firing at builder-close like a `ws_agg_locked` strategy. No silent
  drop.
- Strategy declares `ws_agg_reconciled` but the model is
  `available: False` → `RalphEngine.__init__` substitutes the default
  (`ws_agg_locked`) with a warning. Same as today.
- Snapshot unpickle fails → full rebuild + log. No data loss.
- Builder grace set to less than max(strategy grace) (shouldn't
  happen; defensive check) → builder closes a bucket before a strategy
  fires → that strategy's `_fired_bucket` advance catches the next
  per-second tick and fires on the closed bucket instead. Soft
  degradation to `ws_agg_locked` behavior.

## 6. Verification (per sub-phase)

- **2a** (foundational wiring):
  - Unit: `BarBuilder.force_close_stale_bar` grace = `grace_sec_override
    or SUBMINUTE_FORCE_CLOSE_GRACE_SEC` across a few configurations.
  - In-app smoke: existing strategies (all on `ws_agg_locked`) behave
    exactly as today — `_test_tier1_determinism.py`-style diff confirms
    no change.
- **2b** (fire timing):
  - Unit: synthetic monitor with `grace_seconds=3` on a 10Sec bucket
    fires at `bar_start + 13`; doesn't double-fire on subsequent
    per-second ticks for the same bucket; commits indicator state on
    builder-close.
  - Live: flip ONE non-critical 10Sec strategy to `ws_agg_reconciled`
    + `grace_seconds = 3`, watch an RTH window — confirm alert
    timestamps shift ~2s earlier vs prior bar-close, indicator state
    on next bar matches a `ws_agg_locked` peer, no phantoms.
- **2c** (Tier 2 snapshot):
  - Unit: snapshot → fingerprint match → restore → `update_bar(new_bar)`
    produces byte-identical state to a from-scratch warmup of
    `df ∪ [new_bar]`.
  - Bug-fix verification: a strategy that's OPEN at the last refresh
    boundary gets its exit correctly applied on resume (the bug the
    current incremental path can't fix because of the
    `entry_time > last_entry` filter).
  - In-app: time "Update New Data" before/after — expect "after"
    to be O(new bars) regardless of history length.

## 7. Open decisions

- **Per-trigger marginality (confidence gating)** — when phantom-alert
  evidence emerges, add a `Trigger.margin(current, threshold)` API and
  a global `MARGINAL_EPSILON` (or per-pack overrides). Defer.
- **Snapshot storage table vs column** — column on `strategies` is
  simpler; a sibling table is cleaner. Lean column for v1; migrate if
  the blob gets unwieldy.
- **Snapshot for ws_agg_locked strategies** — Tier 2 (2c) is
  independent of the reconciled live_model, so it benefits *every*
  strategy regardless of `live_model`. Confirmed: snapshot persistence
  applies universally.
- **`compute_tentative_state` cost at strategy-grace** — already runs
  per-second for forming-bar broadcasts. Adding a per-monitor
  per-second check on `due_at` is O(N_monitors) per second per
  symbol — same order as the existing `on_tick` loop. No new
  performance concern at typical monitor counts.

## 8. Sequencing

1. **2a** (foundational wiring) — small, safe, ships first.
2. **2c (Tier 2)** in parallel with 2b — independent backtest-side
   work; can land at any time without affecting live alerts.
3. **2b** (the real behavior change) — last; ships when ready to
   watch an RTH window. Flip a single non-critical strategy first as
   a canary; widen later.
4. After 2b verified stable, optionally flip `ws_agg_reconciled` to
   `default: True` (LEF spec open decision §7 — Kevin: yes
   eventually).
5. Phase 2.5 (confidence gating) only if/when phantom alerts surface.
