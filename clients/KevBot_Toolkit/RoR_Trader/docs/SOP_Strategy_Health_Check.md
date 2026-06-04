# SOP — Strategy Health Check

> **Purpose:** A repeatable 4-layer methodology for verifying that the live alert engine, the backtest engine, and the trade pipeline are in sync. Run on cadence (post-deploy, weekly, when something looks off). Each layer answers a distinct question — the **cross-layer correlation** is where the real signal lives.

> **Was:** `SOP — Divergence Investigation` (Layers 1+2 only). Broadened 2026-05-29 to cover the full health check and add Layer 3 (fill delta) and Layer 4 (artisanal investigation).

## When to run

- After any live-model change (e.g., the 2026-05-28 `ws_rest_spliced` rollout)
- After a deploy touching `unified_engine`, `ralph_engine`, dispatch, stops/targets, or rest_verifier
- Periodically during normal operation (weekly suggested)
- When a strategy in `/admin/strategy-health` looks off
- When introducing a new symbol, timeframe, or trigger pack to the fleet

## The four layers

### Layer 1 — Bar fidelity
**Question:** *Are bars being updated correctly, in a time-efficient manner, early enough?*

This is the foundation. If bars are wrong or arrive late, every downstream layer's signal is corrupted.

- **Source:** `alerts.verification_status` column (populated by `rest_verifier`)
- **Tool:** `python _divergence_walkthrough.py --window-hours 1 --denom` → "bar verification breakdown" section
- **Statuses:**
  - `verified` — WS close matched REST within $0.01 (no drift)
  - `corrected` — WS-REST drift detected, engine spliced REST values into indicator history
  - `drift_uncorrected` — drift detected but bar was no longer latest; correction rejected (structurally common on sub-minute TFs)
  - `drift_corrected_cascade` — drift fired but newer bars had already been processed against the bad value
  - `rest_unavailable` — REST didn't return data within max_wait (concerning if >0%)
  - `NULL` — verification not yet run / strategy wasn't on `ws_rest_spliced` at fire time
- **Targets:**
  - `drift_uncorrected` ≤ 2% on 10Sec; ≤ 0.5% on 1Min+
  - `rest_unavailable` should be ~0%
  - `corrected` % rising over time = Phase B replay catching what used to leak through

### Layer 2 — Trade-pairing divergence (phantom / missed)
**Question:** *Are we missing trades? Are we producing phantom trades?*

Layer 2 has two altitudes. Always start with the overview as a screening step — if nothing's red, you don't need to dig into the backlog.

#### 2A. Strategy Health Overview (macro screening)
- **Source:** `/api/admin/strategy-health`
- **What it shows:** Fleet-wide rollup — per-strategy phantom/missed/paired counts in the window (now with `_global_fair` variants honoring the cross-strategy fair cutoff)
- **Use:** "Which strategies need attention?" If nothing crosses your phantom-rate threshold, stop here.

#### 2B. Divergence Backlog (per-event drill-down)
- **Source:** `/api/admin/strategy-health/backlog`, pairs alerts against trades within ±60s
- **Tool:** `python _divergence_walkthrough.py --window-hours 1 --max-events 20`
- **Surface:** Unpaired events labeled `phantom` (alert-only) or `missed` (trade-only), each tagged with a classifier bucket
- **Use:** For each `needs_investigation` event: identify the cluster, gather context, decide the cause, log it.

##### Existing classification buckets (auto-classifier)
Defined in `src/api/routers/strategy_health.py` near the `_CLASS_*` constants.

| Bucket | Meaning |
|---|---|
| `needs_investigation` | Default — unexplained, the mysteries we chase here |
| `timestamps_out_of_sync` | Known cause: bar timestamps misaligned by spec drift |
| `phase2_signal_exit` | Known cause: Phase 2 signal-based exits don't pair with alerts |
| `non_fill_event` | Known cause: event isn't a fill (e.g., cancel) |
| `legacy_strategy` | Known cause: pre-spec strategy without canonical timestamps |
| `cross_exec_type_mismatch` | **NEW 2026-06-04** — paired event whose live exec_type disagrees with backtest's implied exec_type. Surfaced as `event_type=mismatch` with both alert_id and trade_id. Mostly exit pairs where live fired C-type signal at bar close but backtest fired L-type stop intra-bar (or vice versa). Pair time delta > 2s. Don't count toward "phantom" or "missed" — they're real pairs but represent different events. |

##### Proposed new buckets (uncommitted to code — pending walkthrough findings)
| Bucket | Meaning |
|---|---|
| `ws_drift_phantom` | Phantom alert fired on WS bar where REST showed slightly different close; alert had `drift_uncorrected` status. Not a bug — structural sub-minute correction limit. |
| `live_reentry_missed` | Backtest re-entered after an intra-bar stop within the same bar; live engine fired one entry but didn't fire the intra-bar stop + re-entry. Investigation target. |
| `cluster_duplicate` | Same divergence event affects N strategies sharing one trigger pack. Deduplicate by trigger signature so the "real" issue count is N/k not N. |

##### Walkthrough procedure (for each `needs_investigation` event)
1. **Identify the cluster** — if multiple strategies share a trigger pack, treat as one
2. **Gather context** — pull surrounding alerts (±2 min, same strategy), their `verification_status` + `verification_close_delta`
3. **Decide the cause:** existing bucket / proposed bucket / new bug class
4. **Log it** in `docs/Divergence_Investigation_Log_YYYY-MM-DD.md` with reasoning
5. **If new bucket is proposed:** add to the "Proposed new buckets" table here until promoted to code

### Layer 3 — Fill-time delta on paired trades
**Question:** *For the trades that DO pair, how tight is the timing match between live alert and backtest trade?*

This is the user-experience metric. Only meaningful if Layers 1+2 are healthy — phantom-heavy strategies pollute the pairing window with cross-event matches.

- **Source:** Paired (alert, trade) tuples within ±60s. Computes per-strategy delta between `alert.fill_ts` and `trade.fill_ts`.
- **Tool:** `python _fill_delta_analysis.py --strategies <ids|cohort> --window-hours <N>`
- **Metric:** Per strategy, entry deltas and exit deltas — avg, median, max, % within ±5s
- **Targets:**
  - **Median 0s** on both entry and exit (most trades same-second)
  - **≥ 95% within ±5s** on entry; ≥ 90% on exit (allows for L-type slippage)
  - Max < 30s expected on fresh strategies; older strategies may show outliers from cross-event mispairings (see `cross_exec_type_mismatch` bucket)

### Layer 4 — Artisanal investigation
**Question:** *What are Layers 1-3 missing? What looks suspicious if I just LOOK at the data?*

**This layer is deliberately unstructured.** It's the explicit permission slip to go off-script. The standardized layers exist to make routine checks fast and consistent. Layer 4 exists because **standardization narrows what you see**, and the most valuable signals often live in the gap between what the framework asks and what the data is actually telling you.

Examples of Layer 4 work that has previously surfaced real bugs:
- Eyeballing alert clusters around news events for asymmetric drift
- Comparing two strategies on the same trigger pack to spot one outlier
- Reading the raw alert log on a specific symbol for an hour to see what "normal" looks like
- Checking the actual_price vs fill_ts vs alert.price triangle on a single suspicious trade
- Spinning up a fresh canary that mirrors a degraded production strategy to isolate one variable

**Don't skip Layer 4 just because Layers 1-3 came back green.** Spend at least 5 minutes looking at something unstructured every cycle. If you notice the same Layer 4 finding twice, it becomes a candidate for promotion to a structured layer.

**Question the layers themselves.** If a layer keeps returning "all green" while you keep finding real issues at Layer 4, the layer's metric or threshold is probably wrong. Layers are tools, not truth. Update this SOP when a layer needs revision.

## Cross-layer correlation

Running the layers in isolation is half the value. Build a per-strategy (or per-cohort) table with columns:

| Strategy | Layer 1 drift_uncorrected % | Layer 2 phantom rate % | Layer 3 entry ≤5s % | Layer 3 exit ≤5s % |

Reading the table:

- **Layer 1 high + Layer 3 wide** → bar drift is the cause. Fix at the engine layer (rest_verifier, snapshot buffer, splice depth).
- **Layer 1 clean + Layer 3 wide** → drift isn't the cause. Look at trigger pack determinism, snapshot restore correctness, alert dispatch overhead.
- **Layer 2 phantom heavy + Layer 3 looks tight** → pairing is artificially clean because the unpaired phantoms got filtered out. Look at what events fell out.
- **All three clean** → strategy is healthy. Move on.

## Suggested run order

1. **Layer 2A overview** — 30s screening. Anything red?
2. **Layer 1** — bar fidelity on flagged strategies (or whole fleet if introducing a model change)
3. **Layer 2B backlog** — drill into per-event for anything Layer 1 didn't explain
4. **Layer 3 delta** — per-strategy or per-cohort, for trade-pipeline accuracy
5. **Cross-layer table** — build the comparison view
6. **Layer 4** — at least 5 minutes off-script every time

## Persistence

Investigation findings live in `docs/Divergence_Investigation_Log_YYYY-MM-DD.md` (one file per session). If a Layer 4 finding repeats, promote it: add to the proposed bucket table or propose a new structured layer.

## Tools

| Tool | Layer | Purpose |
|---|---|---|
| `src/_divergence_walkthrough.py` | 1, 2B | Bar verification breakdown + per-event backlog walkthrough |
| `/api/admin/strategy-health` | 2A | Fleet-wide phantom/missed rollup |
| `/api/admin/strategy-health/backlog` | 2B | Per-event unpaired classification |
| `src/_fill_delta_analysis.py` | 3 | Per-strategy fill_ts delta on paired trades |
| Eyeballs + curiosity | 4 | Unstructured |

## Linked artifacts

- `src/api/routers/strategy_health.py` — backlog endpoint + auto-classifier
- `src/_divergence_walkthrough.py` — Layer 1+2B tool
- `src/_fill_delta_analysis.py` — Layer 3 tool
- `docs/Baseline_Metrics_2026-05-28_post-M8.md` — comparison baseline for post-fix evaluations
- `memory: project_ws_rest_spliced_canary` — the rollout context
- `memory: feedback_phantom_missed_trade_defs` — locked terminology

## Revision log

- **2026-05-29** — Initial 4-layer version. Promoted from `SOP_Divergence_Investigation.md`. Added Layer 1 phrasing (Kevin), Layer 3 standardization, Layer 4 (artisanal). Will be revised after the first +1h check at 13:13 MDT if the structure proves wrong.
