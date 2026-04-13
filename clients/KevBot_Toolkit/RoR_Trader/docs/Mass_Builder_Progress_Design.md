# Mass Builder Progress Bar — Design Proposal

**Status:** Design discussion. No code changes until user sign-off.
**Deliverable:** Decide phase model, bar sources, and update cadence before
implementation begins.

## Why redesign?

Today's two-bar display feels stalled during long backtests because:

- **Top bar** advances only when a full trigger backtest completes. On a
  10Sec × 180-day run, one backtest takes ~7 minutes of processing ~842k bars
  — the bar sits at the same percentage for 7 minutes straight.
- **Bottom bar** is wired for confluence-search progress only. It's blank
  during the trigger backtest phase (which is most of the wall time).
- **Label** shows ticker/TF/direction but doesn't reveal *what the engine is
  currently doing* (loading data? warming indicators? iterating bars?
  searching confluences?).

**Goal:** Replace "stalled for 7 minutes" with continuously-updating feedback
that demonstrates throughput (which is impressive — hundreds of thousands of
bars per second, thousands of confluence subsets per second).

## What exists today

**Backend** (`src/mass_builder.py`):
- Single `progress_callback(step, total_steps, label, conf_step=None, conf_total=None)` function
- Fires after every backtest completes, after data loads, and during confluence search
- Step counter increments per backtest (`step++` in the inner loop)
- Confluence callback fires per-combo inside `find_best_combinations()`
- Storage: in-memory `_active_searches` (all fields) + DB flush every 10s (step/total/label only)

**Frontend** (`MassBuilderPage.tsx:1467-1523`):
- Top bar: `progress / total_steps` → smooth visual, same source as step counter
- Bottom bar: confluence combo progress, only visible when confluence search active
- Polling: React Query at 2s interval
- Label rendering: single line showing `current_label` from backend

**What's missing:**
- No bar-by-bar progress feedback from inside `run_unified_backtest()`
- No phase label (just the symbol/TF/direction string)
- No visibility during data-load and indicator-warmup overhead (the ~23s per group)
- Confluence granularity is capped by 2s polling cadence

## Proposed design

### Two-bar model (refined)

**Top bar — "Overall Search Progress":**
- Source: step counter (unchanged)
- What it represents: fraction of total backtests completed across the whole search
- Update cadence: advances each time a backtest completes, each time a data load
  finishes, or each time a confluence search for a BT wraps. For a 12-BT run,
  this is ~12-25 discrete advances across the run.
- Label format: `Overall: N of M backtests complete` (no change, but framed as
  the "zoomed out" view)

**Bottom bar — "Current Activity":**
- Source: depends on current phase (see phase model below)
- What it represents: fine-grained progress inside the *currently executing*
  unit of work
- Update cadence: 500ms polling (reduced from 2s)
- Label format: phase name + specific details

### Phase model

The search executes in a loop of (ticker, TF) groups. Within each group, the
engine runs through distinct phases. Each phase drives the bottom bar from 0%
to 100%:

| # | Phase | Duration (typical) | Bottom bar source | Label |
|---|---|---|---|---|
| A | Loading data | 2-10s per group | fraction of expected bars loaded | "Loading TSLA 1Min (180 days)" |
| B | Indicator warmup | 1-5s per group | fraction of bars with indicators computed | "Preparing indicators for TSLA 1Min" |
| C | Running backtest | 1-400s per BT | `bar_count / total_bars` inside `run_unified_backtest` | "TSLA 1Min LONG · Backtest 3 of 4 · Bar 42,180 of 70,200" |
| D | Searching confluences | 0.1-100s per BT | `conf_step / conf_total` (existing) | "TSLA 1Min LONG · Confluence 1,205 of 2,625" |
| E | Persisting results | <1s per search | n/a — brief flash | "Saving 428 results" |

Phases A/B only happen once per (ticker, TF) group. Phases C and D repeat for
each backtest within the group. E happens once at end of search.

### Key insight: the top bar doesn't need to be "phase-counted"

The user's concern was that 4-5 phases would make the top bar feel slow.
Solution: **top bar isn't phase-based at all**. It stays tied to the
incrementing step counter (which advances frequently). The bottom bar's label
answers "what's happening right now" via the phase name, and its bar fills up
fractionally within each phase.

This gives:
- **Top bar:** a reliable measure of total search progress, advancing
  ~12-1600 times per run
- **Bottom bar:** always moving (sub-second polling, fractional fill), with a
  phase label that changes every few seconds to minutes

### Bar-count emission (new backend hook)

To feed phase C, `run_unified_backtest()` needs to emit per-bar progress.
Proposed:

```python
def run_unified_backtest(df, config, *, general_packs=None,
                         secondary_tf_map=None,
                         progress_cb=None,       # NEW
                         progress_interval=500   # NEW — ms between callbacks
                        ):
    ...
    for i in range(len(df)):
        # ... existing work ...
        if progress_cb and (time.monotonic() - last_cb_time > progress_interval / 1000):
            progress_cb(i, total_bars)
            last_cb_time = time.monotonic()
```

This is a tight change — ~10 lines inside `unified_engine.py`, fully opt-in via
the new `progress_cb` parameter, no impact when not used.

`run_mass_search()` threads the callback through to update `_active_searches`
with `bars_done` / `bars_total` / `phase` fields.

### Data-load phase granularity (phase A)

Currently data load is opaque (single `_progress` call before and after). For
bigger granularity, we'd need the data loader (Polygon or cache reader) to
emit progress. Options:

1. **Cheap approximation:** start a simulated progress bar using a time-based
   estimate (data load typically ~100ms per 10k bars). If actual load finishes
   early, snap to 100%.
2. **Real progress:** add a callback in `prepare_data_with_indicators()`.
   More accurate but touches more code.

Recommendation: **start with approach #1** for this phase only. It's
instantly satisfying visually and accurate enough (data load is rarely the
bottleneck).

### Polling cadence

- Keep 2s DB flush (DB is for crash recovery, not live UI)
- New 500ms in-memory polling for frontend — all live progress data is already
  in `_active_searches`, which the progress endpoint reads without hitting DB.
- Frontend React Query: `refetchInterval: 500` when status === 'running',
  back to 2s or disabled when completed

## Implementation outline (post-approval)

**Backend changes:**
1. `src/unified_engine.py` — add optional `progress_cb` param to `run_unified_backtest`
2. `src/mass_builder.py` — pass wrapper callback that updates `_active_searches` with `phase`, `bars_done`, `bars_total`, `phase_pct`
3. `src/api/routers/mass_builder.py` — include new fields in `/progress` response

**Frontend changes:**
1. `MassBuilderPage.tsx` progress panel — rewrite to use phase-driven bottom bar
2. `useMassProgress` hook — bump `refetchInterval` to 500ms while running
3. Phase label component: small colored pill above the bottom bar

**Effort estimate:** ~4-6 hours end to end (2h backend, 3h frontend, 1h testing).

## Open questions for user

1. **Per-bar polling rate inside `run_unified_backtest`:** 500ms keeps the bar
   visibly moving but emits thousands of callbacks on long runs. Acceptable?
   (Alternative: emit every 1000 bars regardless of time.)
2. **Phase A (data load) — real vs simulated progress?** Recommend simulated
   for now; add real progress later if data load becomes a noticeable wait.
3. **Show phase as a text label above bottom bar, a colored pill tag, or
   inside the bottom bar itself?** Mockup options welcome.
4. **Confluence search (phase D) combo count — still rely on backend
   sub-progress callback, or rewire to compute progress from elapsed time?**
   Backend callback is more accurate; recommend keeping it.

## Next step

User reviews this document, answers the open questions, and we lock in the
final spec. Implementation happens as a focused session after sign-off.
