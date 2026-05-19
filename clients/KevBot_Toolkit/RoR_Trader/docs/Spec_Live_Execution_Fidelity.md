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
- Reworking 1Min behavior (`ws_agg_locked` is bit-identical to REST on
  validated samples). The framework is general, but the work is
  sub-minute-focused.

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

### 4.1 Proposed model

Keep `ws_agg_locked` as-is (current default — lock at close, no
corrections). Add one new live model:

- **`ws_agg_reconciled`** — "A-aggregated (reconciled)". Fires on a
  *decision-grade* sub-minute bar; reconciles indicator state via the
  O(1) `apply_last_bar_correction` path when the corrected bar lands.
  `available: False` until the engine phase ships.

Start with the binary (locked vs reconciled). Resist a proliferation of
fixed-grace models — see 4.2.

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

### 4.3 Parity counterpart

For `ws_agg_reconciled` to be trustworthy it must be checkable on the
Divergence tab. The `algo_model` lane ("what the live engine *should*
have produced") must be able to replay the provisional+reconcile logic.
Open decision (7): whether existing `cache_locked` / `cache_corrected`
cover this or a new `BACKTEST_MODELS` entry is needed.

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

## 6. Phased delivery

- **Phase 1 — Calibration & scaffolding.** Collect grace 2–7 shadow
  data (in progress). Add `ws_agg_reconciled` to `LIVE_MODELS`
  (`available: False`). Define the `grace_seconds` config field +
  per-ticker defaults. No engine behavior change yet.
- **Phase 2 — Engine.** Provisional decision-grade bar + O(1)
  reconciliation in the live engine; confidence-gated firing. The
  actual `force_close_stale_bar` grace becomes parameter-driven.
- **Phase 3 — Backtest/algo parity counterpart.** Backtest can
  simulate the reconciled model; Divergence tab covers it.
- **Phase 4 — Fidelity tab + ModelsCard wiring.** Flip
  `ws_agg_reconciled` to `available: True`.
- **Phase 5 — Per-strategy grace UI** (named tiers).

## 7. Open decisions (resolve before the relevant phase)

- Grace as a discrete model attribute vs a continuous parameter.
  *Lean: continuous parameter + per-ticker default.*
- `ε` for "marginal trigger" confidence gating — global, or
  per-indicator/per-trigger-type?
- Does `ws_agg_reconciled` eventually become the default, or stay
  opt-in?
- Does the parity counterpart need a new `BACKTEST_MODELS` entry, or do
  `cache_locked` / `cache_corrected` suffice?
- Final grace defaults — pending the 6s/7s sweep (esp. whether SPY
  wants 6–7).
- Naming: `ws_agg_reconciled` vs alternatives.

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
