# Design Doc — Real REST-vs-CACHE Divergence Comparison (v2)

**Status:** DESIGN. Not yet scoped for execution.
**Authored:** 2026-05-08
**Background:** v1 Divergence tab shipped 2026-05-07. Open question
captured in `project_strategy_builder_design_open.md` and yesterday's
plan files. This doc resolves the dual-storage question so we can
execute when ready.

## Why this exists (problem statement)

The v1 Divergence tab compares three lanes per strategy:

| Lane | Today's source |
|---|---|
| Backtest | `stored_trades` JSONB (whatever `backtest_model` was last run) |
| Algo | trades table (cron-appended under `algo_model`) |
| Live | alerts table |

This **doesn't directly compare two backtest models against each other**.
Each strategy only has ONE active `backtest_model` at a time —
running `recompute_and_persist_stored_trades` overwrites stored_trades
with whichever model's output. So the "Backtest" lane reflects whatever
was MOST RECENTLY run, not a stable REST baseline you can compare a
cache result against.

Kevin's goal (verbatim from 2026-05-07 PM):

> "I would be kind of curious to see how closely the cache model lines
> up with the backtest model or the rest-only. Just wondering how much
> those diverge and then how they diverge from live as well."

To support that comparison, we need **dual backtest storage**: two
sets of "the same strategy run under different backtest_models" coexisting
on disk, so the user can ask "how does REST vs CACHE compare for sid X
at the same point in time?"

## Goals

1. Run the same strategy against TWO distinct `backtest_model` values
   in the same flow.
2. Store both outputs side-by-side without one wiping the other.
3. Surface the comparison on the Divergence tab as a 4th lane (or a
   toggle) without breaking the existing 3-lane flow.
4. Keep the v1 trust model intact: backtest_model still drives KPIs;
   the second-stored model is "view-only" for divergence analysis.

## Non-goals

- Running 3+ backtest models simultaneously — only A vs B for now.
- Auto-running both models on every refresh — explicit user opt-in.
- New model registry — same `BACKTEST_MODELS` dict, two consumers.

## Three storage approaches considered

### Option A — New JSONB column `stored_trades_alt`

Add a second column to `strategies` table mirroring `stored_trades`,
plus metadata column `stored_trades_alt_meta` with `{model, generated_at}`.

**Pros:**
- Clean separation; no ambiguity about which is canonical
- KPI computation stays single-source (always `stored_trades`)
- Migration is one ALTER TABLE

**Cons:**
- Schema rigid: hardcodes 2-storage rather than N-storage
- Only one "alternate" — if user wants to try a 3rd model, no slot

### Option B — `stored_trades_by_model` JSONB map

One JSONB column keyed by model name:
```json
{
  "rest_hifi": [...trades...],
  "cache_locked": [...trades...]
}
```
Plus per-model metadata `{generated_at, kpi_summary}` as a sibling map.

**Pros:**
- N-storage by design; supports comparing 3+ models if ever needed
- Single column to reason about
- KPI source = `stored_trades_by_model[strategy.backtest_model]`

**Cons:**
- Bigger schema-change footprint (data migration of existing
  `stored_trades` into the new shape)
- Slight risk of bloat: if all 4 models accumulate trades for a
  strategy, JSONB row gets large

### Option C — trades table with `data_source` filter

Use the existing trades table (already has `data_source` column).
Each backtest run inserts rows with `data_source='rest_hifi'` or
`'cache_locked'`. Query filter at read time.

**Pros:**
- Schema already supports this (no migration)
- N-storage natively
- Aligns with the trades-table being our canonical "lane" store
  (which it's becoming via algo_model lane work)

**Cons:**
- Unique index `(strategy_id, entry_fill_ts, exit_fill_ts)` BLOCKS
  two models from inserting trades at the same timestamp — would
  need to relax the index to include `data_source`
- Mixes per-model backtest output with cron's algo-history append
  (currently `data_source=null` for cron) — needs careful query
  separation
- Backtest "snapshot" semantics get blurry when same store holds
  both stable backtests + continuous cron writes

### Recommendation: **Option B (`stored_trades_by_model` JSONB)**

Cleanest mental model. KPIs continue to come from
`backtest_model → stored_trades_by_model[backtest_model]`. Divergence
view picks any 2 models for comparison. Future-proof if we add more
models. Bigger upfront migration cost is paid once.

Alternative if migration risk is too high: Option A is acceptable for
v2 — captures the immediate "REST vs CACHE" use case without the
multi-model overhead.

## Schema proposal (Option B)

```sql
-- Migration: backtest_dual_storage.sql
ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS stored_trades_by_model JSONB DEFAULT '{}'::jsonb;

-- Backfill: copy current stored_trades into the by_model map under
-- whichever backtest_model the strategy currently uses.
UPDATE strategies
SET stored_trades_by_model = jsonb_build_object(
  COALESCE(config->>'backtest_model', 'rest_hifi'),
  COALESCE(stored_trades, '[]'::jsonb)
)
WHERE stored_trades IS NOT NULL
  AND stored_trades_by_model = '{}'::jsonb;

-- DON'T drop stored_trades yet — cutover happens after read path is
-- verified. Eventual deprecation: stored_trades becomes a computed
-- view of stored_trades_by_model[backtest_model].
```

Per-model metadata sibling JSONB:
```sql
ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS stored_trades_by_model_meta JSONB DEFAULT '{}'::jsonb;
-- Shape:
-- { "rest_hifi": {generated_at: ..., kpi_summary: {...}, hifi_resolved: true},
--   "cache_locked": {...} }
```

## Backend changes

### 1. New helper: `recompute_and_persist_stored_trades_under_model`

Accepts an explicit `model_override` parameter and writes the result
to `stored_trades_by_model[model]` rather than top-level
`stored_trades`. Reuses existing engine logic.

```python
def recompute_and_persist_stored_trades_under_model(
    strategy_id: int, user_id: str, model: str
) -> dict:
    """Run engine under `model` (overrides config.backtest_model for
    this run only) and write to stored_trades_by_model[model]. Updates
    stored_trades (canonical) only if model == config.backtest_model
    so KPI baseline stays stable."""
```

### 2. Update `recompute_and_persist_stored_trades` (existing)

Refactor the existing helper to:
1. Run engine under `config.backtest_model` (no change)
2. Write to BOTH `stored_trades` (canonical) AND
   `stored_trades_by_model[backtest_model]` (mirror)

So the existing flow doesn't lose dual-storage when the canonical
model runs.

### 3. New endpoint: `POST /api/strategies/{id}/backtest-under-model`

```
Body: {"model": "cache_locked"}
```
Triggers `recompute_and_persist_stored_trades_under_model`. Frontend
calls this for the "Run Comparison Backtest" button.

### 4. Extend `/divergence-data` to optionally include a 4th lane

```python
# Query param: comparison_model (optional)
# When set, include stored_trades_by_model[comparison_model] as a
# fourth lane "Backtest (alt)".
```

Or simpler: surface ALL stored model outputs as available lanes on
Divergence tab, let user toggle which two backtest models to compare.

## Frontend changes

### 1. Strategy Detail → Configuration tab

ModelsCard gains a "Compare against" section:
- Dropdown: "Compare backtest_model against [X]"
- Button: "Run [X] backtest"
- Status: "[X] last run: 2h ago"

### 2. Strategy Detail → Divergence tab

Lane status row gains a 4th lane (collapsed by default) for the
comparison model. Comparison table gains a column for the alt
backtest. Drift KPIs include backtest↔alt-backtest delta.

OR simpler v2 UI: keep the existing 3-lane Divergence tab; add a
"Compare backtest_model" toggle that swaps the Backtest lane source
between current backtest_model and a chosen alt model.

### 3. Strategy Builder + Mass Builder

No change. Users still pick one backtest_model at creation. Comparison
is opt-in post-creation.

## Phasing

### Phase 1 — Schema + backfill (~30 min)
- Migration adding `stored_trades_by_model` + meta
- Backfill existing strategies' stored_trades into the map
- Verify no data loss; KPIs unchanged

### Phase 2 — Dual write on existing recompute (~45 min)
- `recompute_and_persist_stored_trades` writes to BOTH stored_trades
  and stored_trades_by_model[backtest_model]
- Verify on a couple of strategies that both stores have identical
  contents post-refresh

### Phase 3 — New under-model endpoint + button (~1.5h)
- `recompute_and_persist_stored_trades_under_model` helper
- `POST /api/strategies/{id}/backtest-under-model` endpoint
- "Run alt backtest" button on Configuration tab
- Verify alt-model run populates `stored_trades_by_model[alt]`
  WITHOUT touching stored_trades

### Phase 4 — Divergence tab 4th lane (~2h)
- Endpoint `/divergence-data` accepts `comparison_model` param
- Joiner extended to handle 4 lanes
- Frontend lane status row + comparison table support 4th lane
- Drift KPIs computed for backtest↔alt pairing

### Phase 5 — Cutover (~15 min, deferred)
- After 1+ week of stable dual-write, deprecate top-level
  `stored_trades` column. Reads come from
  `stored_trades_by_model[backtest_model]`. Migration drops the
  column.

**Total v2 effort: ~5h core + 1h cutover.**

## Risks

- **Bloat**: a strategy run under 3 models accumulates 3x trade rows
  in JSONB. If a strategy has 5,000 trades and runs under 3 models,
  that's 15,000 rows in one JSONB cell. Acceptable; monitor.
- **KPI source ambiguity**: docs/help text need to be crystal-clear
  that KPIs ALWAYS reflect `config.backtest_model`. Switching the
  Divergence comparison view doesn't change KPIs.
- **Migration risk**: existing strategies' data needs to backfill
  cleanly. Mitigation: dry-run on staging, full backup of
  `stored_trades` snapshot before migrating.
- **Frontend complexity**: 4 lanes on a comparison table is busier
  UX than 3. Mitigation: collapsible/toggle UX so default view stays
  3-lane.

## Open questions

1. **Alt-model auto-run policy?** When user changes `backtest_model`
   on a strategy, do we auto-run the new model in background? Or
   require explicit "Run [X] backtest" click? Recommendation: explicit
   click — engine cost is real and shouldn't surprise users.
2. **Deprecation timing?** Phase 5 cutover safe after how long?
   Recommendation: 1 week of clean dual-write + manual spot-checks
   confirming `stored_trades_by_model[backtest_model]` ==
   `stored_trades` before dropping the latter.
3. **UI polish for 4 lanes?** Collapsible 4th column vs separate
   toggle vs side-by-side? Defer to user feedback after Phase 3
   ships and people have used the alt-model run capability.

## Files to modify (when execution begins)

**Backend:**
- `src/migrations/backtest_dual_storage.sql` (NEW)
- `src/api/services/forward_test_service.py` — extend
  `recompute_and_persist_stored_trades` with dual-write; add
  `recompute_and_persist_stored_trades_under_model`
- `src/api/routers/strategies.py` — new endpoint; extend
  `/divergence-data` query
- `src/api/services/divergence_service.py` — extend joiner to 4 lanes

**Frontend:**
- `frontend/src/views/StrategyDetailPage.tsx` — ModelsCard alt-run
  control; Divergence tab 4th lane
- `frontend/src/hooks/queries/useStrategies.ts` — new types + hook
- `frontend/src/hooks/mutations/useStrategyMutations.ts` — new
  mutation for under-model run

## Decision log

- **Storage approach: Option B** — JSONB map keyed by model. Future-proof;
  single column. Backfill from existing stored_trades is one UPDATE.
- **KPI canonical source: still `stored_trades`** — no change in KPI
  semantics during v2 transition. Top-level column eventually deprecated
  but keep through Phase 4.
- **Auto-run policy: explicit only** — no surprise engine spend on
  model changes.
- **UI default: still 3-lane** — 4th comparison lane is opt-in toggle.

## Tracking

Memory file: `project_algo_model_followup.md` v2 section.
Plan file: this doc supersedes the brief mention in
`~/.claude/plans/synchronous-tickling-yeti.md`.
