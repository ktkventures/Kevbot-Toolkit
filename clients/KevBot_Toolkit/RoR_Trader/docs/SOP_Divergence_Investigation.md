# SOP — Divergence Investigation

> **Purpose:** Repeatable methodology for working through the `/admin/strategy-health/backlog` queue as a task list. Each `needs_investigation` event gets reviewed, classified (existing bucket or a new one), and the reasoning logged. The eventual goal: zero `needs_investigation` rows (every divergence either fixed or labeled with a known-cause bucket).

## When to run this

- After any live-model change (e.g., the 2026-05-28 `ws_rest_spliced` rollout)
- After a deploy that touches `unified_engine`, `ralph_engine`, dispatch, or stops/targets
- Periodically (weekly?) during normal operation, to catch drift
- When `phantom_count` or `missed_count` on `/admin/strategy-health` looks unusual

## The two complementary analyses

Both should be looked at together — they answer different questions.

### A. Divergence backlog (trade-level)
- **What it asks:** Did the right trades happen?
- **Source:** `/api/admin/strategy-health/backlog` endpoint, pairs alerts against the trades table within ±60s
- **Surface:** unpaired events labeled `phantom` (alert-only) or `missed` (backtest-trade-only)
- **Run with:** `apples_to_apples=true`, `only_needs_investigation=true`, `window_hours=1` (or wider for catch-up)

### B. Bar verification / live bar fidelity (bar-level)
- **What it asks:** Was each individual bar correctly aligned to REST?
- **Source:** `alerts.verification_status` column (populated by `rest_verifier`)
- **Statuses:**
  - `verified` — WS close matched REST within $0.01 (no drift)
  - `corrected` — WS-REST drift detected, engine spliced REST values into indicator history
  - `drift_uncorrected` — drift detected but bar was no longer latest; correction rejected (structurally common on sub-minute TFs)
  - `rest_unavailable` — REST didn't return data within max_wait (concerning if >0%)
  - `NULL` — verification not yet run / strategy wasn't on `ws_rest_spliced` at fire time
- **Run with:** see `_divergence_walkthrough.py` "bar verification mode" or direct DB query

## Existing classification buckets (auto-classifier)

Defined in `src/api/routers/strategy_health.py` near the `_CLASS_*` constants. Add new buckets here in code as we identify new bug classes; the UI's "needs investigation" filter hides anything other than the default.

| Bucket | Meaning |
|---|---|
| `needs_investigation` | Default — unexplained, the mysteries we chase here |
| `timestamps_out_of_sync` | Known cause: bar timestamps misaligned by spec drift |
| `phase2_signal_exit` | Known cause: Phase 2 signal-based exits don't pair with alerts |
| `non_fill_event` | Known cause: event isn't a fill (e.g., cancel) |
| `legacy_strategy` | Known cause: pre-spec strategy without canonical timestamps |

## Proposed new buckets (uncommitted to code — pending walkthrough findings)

Buckets we add to the codebase as patterns confirm:

| Bucket | Meaning |
|---|---|
| `ws_drift_phantom` | Phantom alert fired on WS bar where REST showed slightly different close; alert had `drift_uncorrected` status. Not a bug — it's the structural sub-minute correction limit. |
| `live_reentry_missed` | Backtest re-entered after an intra-bar stop within the same bar; live engine fired one entry but didn't fire the intra-bar stop + re-entry. Investigation target. |
| `cluster_duplicate` | Same divergence event affects N strategies sharing one trigger pack. Deduplicate by trigger signature so the "real" issue count is N/k not N. |

## Walkthrough procedure

For each `needs_investigation` event in the working log:

1. **Identify the cluster** — if multiple strategies share a trigger pack, treat them as one cluster
2. **Gather context** — pull surrounding alerts (±2 min, same strategy) and their `verification_status` + `verification_close_delta`
3. **Decide the cause:**
   - Match against an existing bucket?
   - Match against a proposed new bucket?
   - New bug class? Propose a new bucket name + describe
4. **Log it** in `docs/Divergence_Investigation_Log_YYYY-MM-DD.md` with reasoning
5. **If new bucket is proposed:** track it in this SOP under "Proposed new buckets" until promoted to code

## Tool

`src/_divergence_walkthrough.py` — wraps the backlog endpoint + context fetch. Default behavior:

```
python _divergence_walkthrough.py --window-hours 1 --max-events 20
# Pulls needs_investigation events, dedupes by (trigger, ts) cluster,
# fetches ±2min surrounding alert + trade context per cluster, prints a
# walkthrough-ready listing.
```

Output is shaped to copy into the day's investigation log file.

## Persistence

For now, classifications live in the markdown log only. If the rate of new findings stays manageable, that's enough. If we end up with hundreds of classified events, upgrade to a DB column (`alerts.user_classification` or a dedicated table).

## Linked artifacts

- `src/api/routers/strategy_health.py` — backlog endpoint + auto-classifier
- `src/_divergence_walkthrough.py` — the tool this SOP drives
- `memory: project_ws_rest_spliced_canary` — the rollout context
- `feedback_phantom_missed_trade_defs` — locked terminology
