# Spec — Live Execution Fidelity

Status: DRAFT (2026-05-19). Supersedes the discussion notes in
`Live_Execution_Fidelity_Notes_2026-05-19.md`. Grounded in the
2026-05-19 grace-window sweep and the worker lag-spiral fix.

---

## 1. Motivation

The 2026-05-19 grace sweep established, under real RTH load:
- Sub-minute (10Sec) bars built from Polygon's `A` per-second channel
  arrive ~3.7s avg late (min 3.06s, tail ~8–12s).
- A "grace window" (holding a bucket open N seconds past `bar_end`
  before closing it) is required for a correct sub-minute bar. Any
  grace ≥ 2s eliminates the flat empty-bar bug; close-match to settled
  REST rises with grace — TSLA 100% at grace 5, SPY 81% at grace 5 and
  still climbing (6s/7s now being measured).

Two structural facts drive this spec:

1. **One bar, three jobs.** A freshly-closed bar feeds three consumers
   with *different* latency/fidelity needs:

   | Job | Needs | Tolerance for a slightly-wrong bar |
   |---|---|---|
   | Fire / no-fire decision | fast **and** correct | **Low** — wrong bar ⇒ phantom/missed alert; a sent webhook is irreversible |
   | Indicator state (forward) | correct eventually | **High** — `apply_last_bar_correction` (O(1)) heals it |
   | Chart / display | looks right | **High** — cosmetic; REST refresh fixes it |

   Today a single hardcoded grace in `force_close_stale_bar` serves all
   three. This spec decouples them.

2. **Fidelity varies by ticker.** SPY (late-print-heavy) needs more
   grace than TSLA. A single global grace is wrong for someone.

## 2. Goals

- Decouple the three jobs of a closed bar.
- Make the latency↔fidelity tradeoff a deliberate, per-strategy,
  *informed* choice.
- **Preserve every existing strategy's behavior** — no retroactive
  change.
- Never break backtest/live parity (the #1 project priority): any new
  live mode must be backtest-simulatable.
- **No phantom alerts.**

## 3. Non-goals

- Changing data provider (Polygon stays — Phase H shelved the trade
  channel).
- **Reworking 1Min+ behavior — confirmed unnecessary.** Measured
  2026-05-19 (`src/_cache_vs_rest1s.py`, 2.5h RTH): the `live_bars`
  cache — *both* decision-time (`first_close`) AND cache+backfill
  (`close`, all sources) — matches REST 1Sec-rolled at **100.0%** for
  SPY/AAPL/TSLA at 1Min and 5Min. The grace/reconciliation paradigm is
  a **sub-minute-only** problem; `ws_agg_locked` stays correct and
  untouched for 1Min+.

## 4. Answer to "do we need a new live model?" — YES

`strategy_models.py` already has the exact modular abstraction. Each
strategy carries three model fields in its `config` JSONB:
- `backtest_model` — KPI-baseline data source (`BACKTEST_MODELS`).
- `algo_model` — "live-accountability" replay lane (`BACKTEST_MODELS`).
- `live_model` — how the live engine sources bars + handles
  rebroadcasts (`LIVE_MODELS`, default `ws_agg_locked`).

The module states the contract explicitly: *"strategies authored under
model X stay valid even when defaults shift — every strategy carries
its declared models in config, no behavior changes retroactively."*

So the answer is not just "yes" — adding a new `live_model` is the
**designed-in** way to evolve execution behavior:
- Current behavior is preserved automatically — existing strategies
  keep their declared `live_model`; a new entry touches nothing.
- Established rollout pattern: add the model `available: False`
  ("coming soon", already rendered disabled in the frontend
  ModelsCard) → wire engine dispatch in a phase → flip
  `available: True` once proven.

### 4.1 Proposed model — one timeframe-aware model

`ws_agg_locked` is the **existing** current default
(`strategy_models.py` LIVE_MODELS, `available: True, default: True`) —
*not* something new. It locks the bar at close, no corrections.

Add **one** new live model:

- **`ws_agg_reconciled`** — "A-aggregated (reconciled)".
  **Timeframe-aware — one model, both behaviors:**
  - **1Min+** — locks at close, exactly like `ws_agg_locked`.
    Reconciliation is a no-op here (the cache is already 100% vs REST —
    see §3), so a 1Min strategy on this model behaves identically to
    `ws_agg_locked`.
  - **Sub-minute** — fires on a *decision-grade* bar (calibrated
    grace); reconciles indicator state via the O(1)
    `apply_last_bar_correction` path when the corrected bar lands.

  A strategy carries a single `live_model`; its **timeframe** decides
  which behavior runs — no need for the user to match a model to a TF.
  Ships `available: True, default: False`, then becomes the default
  once Phase 3 parity is proven.

`ws_agg_locked` stays in the registry as the legacy / explicit-lock
option. `ws_agg_reconciled` is the single forward model.

### 4.2 Grace is a parameter, not a model

The discrete `live_model` selects the **mode** (locked vs reconciled).
The **grace value** is a separate per-strategy parameter, because
Idea A (per-ticker / per-strategy tuning) needs continuous-ish control
that a handful of discrete models can't express cleanly.

- New strategy `config` field — `grace_seconds` (or a named tier that
  resolves to seconds).
- **Per-ticker calibrated default** from the shadow data (SPY ~5–7,
  TSLA ~4 — pending the 6s/7s read).
- If user-exposed (5.2), expose as **named tiers**, never a raw seconds
  field.

### 4.3 Model taxonomy after this spec (decided 2026-05-19)

The three model fields converge to a clear division of labour:

- **`backtest_model` = REST 1Sec aggs, rolled to any TF** — the
  "source of truth" / KPI baseline. Kevin's canonical reference.
  A new `rest_1s` entry is warranted (see 4.4) — not for a behavior
  change, but for *uniformity* (one construction for every TF) and
  *alignment robustness*.
- **`live_model` = `ws_agg_reconciled`** (the single timeframe-aware
  forward model — §4.1; `ws_agg_locked` kept as legacy) — what the
  engine fires on. `ws_agg_reconciled` is *designed to converge toward
  the backtest_model* — that convergence is the whole point, so it does
  not need a bespoke backtest simulator.
- **`algo_model` = a parity counterpart of `ws_agg_reconciled`** — the
  accountability lane: "what the reconciled live engine *should* have
  produced," for the Divergence tab. This is the one genuinely new
  `BACKTEST_MODELS`-side entry the spec needs.

**Decision:** `ws_agg_reconciled` should *eventually become the live
default* (Kevin, 2026-05-19) — once Phase 3 parity is proven. Until
then it ships `available: True, default: False`.

### 4.4 The `rest_1s` backtest model — why, and the caveat

Today the backtest data path picks Polygon-native aggregates per TF via
`_polygon_timespan` (`"10Sec" → (10,"second")`, `"1Min" → (1,"minute")`,
…) — *two different fetch shapes* (native sub-minute vs native minute).

Polygon **native N-second aggregates align to the query `from`
parameter**: a date-string `from` (the backtest path) → midnight →
absolute 10s boundaries (fine, matches the live `BarBuilder._align_to_
period`); but an arbitrary ms-timestamp `from` → phase-shifted buckets.
That `from`-dependence is a latent fragility.

A `rest_1s` model — fetch 1Sec aggs, roll with `(sec // tf) * tf` — is
**unconditionally absolute-aligned for every TF**, and is *literally
the ground truth the parity Bars Comparison already uses*. Making the
backtest model identical to the validator removes a class of "is the
truth even the truth" questions.

It is **not** expected to change results materially (native-N-sec with
a date `from` ≈ 1Sec-rolled — both Polygon settled, both absolute-
aligned). **Verify that equivalence first**, then adopt `rest_1s` as a
clarity + robustness unification, not a behavior change.

## 5. Detailed design

### 5.1 The three jobs, decoupled
- **Fire decision** → a *decision-grade* bar: enough grace that the
  trigger *outcome* is stable (not necessarily a perfect bar).
- **Indicator state** → fast bar + O(1) `apply_last_bar_correction`
  reconciliation on every correction.
- **Display** → fast bar + REST refresh (role demoted — 5.4).

### 5.2 Per-strategy grace (Idea A)
- `grace_seconds` config field; per-ticker calibrated default.
- Builder architecture — **do not build N builders per strategy.**
  Build once at the most-generous grace (the correct final bar). A
  strategy that wants to act earlier takes a **provisional read of the
  still-forming bucket** at its own offset — reuse the existing
  `compute_tentative_state` machinery — then reconciles at final close.
  Per-strategy "earliness" = when the provisional read is taken, not
  N parallel builders.
- UI: named tiers ("Fastest / Balanced / Highest-fidelity"), each
  showing its parity consequence inline. Never a raw seconds field.

### 5.3 Provisional fire + reconciliation (Idea B)
- **Confidence-gated firing.** When a trigger is *unambiguous* (bar
  close comfortably clear of the threshold), fire at the provisional /
  decision-grade bar. When *marginal* (close within ε of the
  threshold), wait for the reconciled bar before committing. Fast when
  safe, careful when close.
- **Indicator state** always reconciles via the O(1) path.
- **A fired alert is final** — never re-fire or un-fire. Reconciliation
  affects forward state only. (This is the hard line that keeps "no
  phantom alerts" true.)

### 5.4 REST refresh role demotion
With good cache fidelity, periodic REST refresh is no longer
load-bearing for fire decisions. Keep it as: (1) the **permanent**
parity validator in Bars Comparison, (2) a display backstop, (3) the
slow-path state reconciliation source. Lower frequency; never blocks.

### 5.5 Strategy-detail "Data Fidelity" tab (Idea C)
- Per-strategy, for that strategy's symbol+TF: close-match % at each
  grace, expected post-close alert latency, a **recommended grace**,
  and a live-vs-backtest parity indicator.
- **Recommendation-first**, not raw tables. e.g. *"SPY 10Sec — grace 5
  recommended: ~81% exact, ~97% within $0.10, ~5s latency. Faster
  (grace 3 ~50%) trades fidelity for ~2s."*
- Pairs with the Strategy Health Badge and the read-only ModelsCard;
  natural home for the 5.2 grace tier selector.

### 5.6 Outage fallback — and the sub-minute backfill gap

Reconciliation (fix a bar you *have*) and backfill (fill a bar you do
*not* have) are different mechanisms. A monitor outage is a backfill
problem, not a reconciliation problem.

Current fallback after an outage:
1. **Data** — `live_bars_rest_backfill.py` (`LIVE_BARS_BACKFILL_ENABLED`)
   fills missing **1Min** rows from Polygon REST, then rolls them up to
   the **derived ≥60s** timeframes. cache+backfill becomes whole for
   1Min+.
2. **Live engine indicator state** — on restart the engine warms up
   from the (now-backfilled) history; state is rebuilt across the gap.
3. **Missed signals** — flagged as `missed_alert` (alert recovery,
   shipped 2026-04-17).

Two honest gaps to record:
- The REST backfill is **cosmetic** — for charts/backtests; it does not
  feed the live engine in real time. The real-time gap is covered by
  warmup-on-restart, not by backfill.
- **There is no sub-minute backfill.** `live_bars_rest_backfill.py`
  covers 1Min + derived ≥60s only. A 10Sec strategy that suffers a
  monitor outage has a **permanent hole** in its 10Sec `live_bars`
  cache. This spec should add a **sub-minute REST backfill** (Polygon
  native 10-sec aggs, or 1Sec-rolled per §4.4) so cache+backfill is
  whole at sub-minute too — folded into Phase 1/3.

## 6. Phased delivery

- **Phase 1 — Calibration & scaffolding.** Collect grace 2–7 shadow
  data (in progress). Add `ws_agg_reconciled` to `LIVE_MODELS`
  (`available: False`). Define the `grace_seconds` config field +
  per-ticker defaults. No engine behavior change yet.
- **Phase 2 — Engine.** Provisional decision-grade bar + O(1)
  reconciliation in the live engine; confidence-gated firing. The
  actual `force_close_stale_bar` grace becomes parameter-driven.
- **Phase 3 — Backtest/algo parity counterpart.** Backtest can
  simulate the reconciled model; Divergence tab covers it. Add the
  **sub-minute REST backfill** (§5.6) so cache+backfill is whole at
  10Sec, closing the outage hole.
- **Phase 4 — Fidelity tab + ModelsCard wiring.** Flip
  `ws_agg_reconciled` to `available: True`.
- **Phase 5 — Per-strategy grace UI** (named tiers).

## 7. Open decisions

**Resolved 2026-05-19:**
- Grace = a per-strategy *parameter*, not a discrete model. ✓
- `ws_agg_reconciled` *will* become the live default once Phase 3
  parity is proven. ✓
- `ws_agg_reconciled` needs no bespoke backtest simulator — it is
  designed to converge toward `backtest_model`. The new model-side
  entry needed is an **`algo_model` parity counterpart** (4.3). ✓
- 1Min+ needs no grace — cache = 100% vs REST 1Sec (3, measured). ✓

**Still open (resolve before the relevant phase):**
- `ε` for "marginal trigger" confidence gating — global, or
  per-indicator / per-trigger-type?
- Final grace defaults — pending the 6s/7s sweep (esp. whether SPY
  wants 6–7).
- Adopt an explicit `rest_1s` `backtest_model`? (4.4 — recommended;
  verify native-N-sec ≈ 1Sec-rolled first.)
- Naming: `ws_agg_reconciled`, the `algo_model` counterpart.

## 8. Risks

- **Phantom alerts** — low grace without confidence gating fires on
  bars that the correction would have un-triggered. Irreversible;
  erodes trust faster than latency does. The #1 risk; 5.3 confidence
  gating is the mitigation.
- **Parity drift** — if the backtest can't simulate `ws_agg_reconciled`,
  live and backtest diverge. Phase 3 is non-optional.
- **Builder CPU** — only if N-builders is chosen over the
  provisional-read design (5.2).
- **Sequencing** — do not ship per-strategy grace (Phase 5) before the
  reconciliation engine (Phase 2).

## 9. Relationships

- **Shares the snapshot/restore primitive** with backtest-speed Tier 2
  (`Backtest_Speed_Implementation_Plan.md`) — `snapshot_state` /
  `apply_last_bar_correction`, built in the 2026-05-19 lag fix. Build
  once, use both places.
- **Cross-references backtest Tier 3** ("always start flat") — both
  serve live/backtest parity from different angles; both likely need to
  land before the system is "execution-trustworthy."
- Depends on the grace 6s/7s calibration data (collection started
  2026-05-19).
