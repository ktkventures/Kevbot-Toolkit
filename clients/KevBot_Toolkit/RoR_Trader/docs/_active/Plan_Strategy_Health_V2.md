# Plan — Strategy Health v2 (Turnkey Validation)

**Status: DRAFT for Kevin's review — 2026-07-20.** Consolidates the Phase C/D destination of
`Plan_Measurement_Trust.md` (the measurement SSOT, agreed 07-15) into a concrete, sequenced
build. Phases A + B of that plan are DONE (the replay harness is dashboard-scored and runs over
any settled window); this doc is the productization that was designed but not built.

## The goal (Kevin, 07-20)
Make **adding + validating strategies turnkey**. Add a strategy → get a fast, honest read on
whether it's firing correctly and is wired/plumbed right — **even before live alerts accumulate**
— and surface any bug **loudly**. The Strategy Health page becomes the single source of truth,
with multiple levels of validation shown as (horizontally-scrollable) columns.

## The validation model — 4 levels (from Plan_Measurement_Trust Phase C)
Per strategy, at 10s **and** 60s tolerance:

| # | Column | Answers | What a bad value means |
|---|---|---|---|
| 1 | **Live paired %** (dashboard `get_strategy_health`) | Did live fire and match backtest? | the number you see today |
| 2 | **Replay ceiling — decision-time** | Would the engine fire it, realistically? (validates a strategy with **zero live alerts**) | live < ceiling ⇒ **operational loss** (stalls / lag / lost alerts — recoverable) |
| 3 | **Replay ceiling — corrected** | Is the *logic* clean, or just a data-timing floor? | corrected low ⇒ **real logic bug**; corrected≈100 but decision-time lower ⇒ **WS/REST floor** (not a bug) |
| 4 | **Gate-parity %** | Are gates/bars constructed the same live vs backtest? | gate-parity low ⇒ **bar/gate construction** divergence |

**Loud classification** (derived, one status chip per strategy):
`CLEAN` (live≈ceiling, high gate-parity) · `OPERATIONAL` (live<ceiling) · `LOGIC-BUG`
(corrected ceiling low) · `GATE-DIVERGENCE` (gate-parity low) · `FLOOR` (decision-time<corrected
only) · `UNRESOLVED` (harness couldn't reconstruct — see blocker below).

This directly answers "is it supposed to fire correctly, or did a bug surface?" — and the
"validate before live data" case is Level 2 (a high ceiling = plumbed correctly even with no
live alerts yet).

## ⛔ Prerequisite / BLOCKER — fix the offline-harness secondary-TF gap
The replay + gate-parity harnesses rebuild each strategy's **secondary-TF gate ribbons** from
history. For certain TF classes they can't fully resolve them, so they **under-count or produce
0** for strategies gated on those:
- **Fine-TF over a short window** — 3m ribbons go NaN-shallow (documented for 333/340).
- **Coarse gates (4h / 1h / 30m)** and **sub-primary gates (10s)** — need long / very-fine
  warmup the offline run doesn't reconstruct. This is why 07-20's replay produced **0 entries
  for 314 (4h/1h) and 339 (30m/10s)** — the gate never materialized, so nothing fired. That 0
  is a harness limit, **not** a strategy result.

**Until this is fixed, Levels 2–4 are untrustworthy for coarse/sub-primary/short-window-fine-
gated strategies.** Fix options (pick in build):
- (a) Warm secondary ribbons over a **longer lookback** so coarse/fine indicators resolve.
- (b) **Reuse the settled stored secondary-TF snapshot** instead of reconstructing from scratch.
- (c) **Fail LOUD** — when a ribbon is `resolved=False`, mark the row `UNRESOLVED` rather than
  silently emit a 0 ceiling. (Non-negotiable regardless of a/b — a 0 must never masquerade as a
  real result. Matches Kevin's "fail loud, no silent backstops" rule.)

## The compute-heavy reality → architecture
Levels 2–4 are **multi-minute offline jobs** (full replay + gate-ribbon reconstruction). They
**cannot** run inline on page load. Design:
- A **"Validate" action** (per-strategy and/or batch) kicks a **background job** (same pattern as
  the existing recompute jobs) that runs the replay + gate-parity over a **settled window**,
  then **stores** the results (new `validation_results` table/cache), timestamped.
- The Strategy Health page **reads the stored results** (fast) + the live dashboard number
  (real-time-ish). It **never** computes Levels 2–4 inline.
- Optional later: a **nightly auto-validate** of the fleet (like the 00:20Z recompute) so the
  columns stay fresh without a manual click.

## Phase D — the UI
- **D1. Refresh-only loading (QoL — the quick win).** Filter / date / tolerance changes update
  local state but do **not** fetch; only the **Refresh** button fetches. Today the custom-date
  selector triggers a full reload per adjustment (slow + DB-heavy), which blocks self-serve.
  *→ implementing 2026-07-20.*
- **D2. 10s tolerance toggle** (currently 60s only; the harness already computes 5/10/60).
- **D3. Validation columns** (Levels 2–4) read from stored validation results, with a per-
  strategy **"last validated at"** timestamp and a **"Validate"** button that enqueues the job.
- **D4. Loud status chips** (the classification above) so bugs surface at a glance.

## Sequencing (recommended)
1. **D1 refresh-QoL** — today. Small frontend change; immediate self-serve relief; lighter DB.
2. **Harness secondary-TF gap fix** — this week. BLOCKER for trustworthy Levels 2–4 (esp.
   fail-loud on unresolved).
3. **Background validation job + `validation_results` storage** — the on-demand "Validate"
   pipeline.
4. **D2/D3/D4 UI** — surface stored results + tolerance toggle + status chips.
5. **Bulk-delete server-side fix** — fold in (small, same cycle; see below).

## Open decisions (for Kevin)
1. **Trigger:** on-demand "Validate" button vs nightly-auto vs both? (Recommend: button first,
   nightly-auto later.)
2. **New page vs columns on the existing tab?** (Recommend: **columns on the existing Strategy
   Health tab** — you said you'll horizontally scroll; far less UI to build, one source of truth.)
3. **Reference window:** validate against the **nightly-settled prior day** (Plan_Measurement_
   Trust Phase B — trustworthy, immune to live contamination). Confirm.
4. **Tolerances:** ship 10s + 60s; include 5s? (harness computes it.)

## Related — bulk-delete server-side fix (from 07-20)
Same root cause surfaced today: the UI bulk-delete 500s because (a) the strategy delete cascade
of a large trades table exceeds the statement timeout, and (b) `bar_diagnostics.values` /
`trades.r_multiple` hold NaN/Inf that PostgREST can't serialize into the delete-return, and the
client won't honor `return=minimal`. **The clean fix is a server-side Postgres function**
(`delete_strategy_cascade(sid)` returning `void` — no serialization; batches dependents) invoked
by a `bulk_delete` that **also adds an ownership check** (it currently deletes any sid passed).
Small, well-scoped, and rides this cycle. (8 junk strategies remain undeletable via the client
until this ships: 277/279/281/283/293/295/297/299.)
