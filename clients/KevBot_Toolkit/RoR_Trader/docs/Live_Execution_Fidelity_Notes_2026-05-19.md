# Live Execution Fidelity — Design Notes (2026-05-19)

Discussion record, not a committed plan. Captures three intertwined
ideas raised after the grace-window sweep, and how they relate to the
backtest-speed Tier 1/2/3 plan. Written while context is fresh.

## The core reframe: one bar, three jobs

A freshly-closed sub-minute bar is used for three different things, and
they have *different* latency/fidelity needs. Today a single grace
window serves all three — that coupling is the thing to break.

| Job | Needs | Tolerance for a slightly-wrong bar |
|---|---|---|
| **Fire / no-fire decision** | low latency **and** correctness | **LOW** — a wrong bar → phantom or missed alert; a sent webhook is irreversible |
| **Indicator state (going forward)** | correctness, eventually | **HIGH** — `apply_last_bar_correction` (O(1)) heals it; small errors decay |
| **Chart / display** | looks right | **HIGH** — cosmetic; REST refresh fixes it |

Everything below follows from this table.

## Idea A — per-strategy modular grace

**The instinct is right; the framing is a footgun.** Grace genuinely
varies by ticker (SPY is late-print-heavy → needs ~5–7s; TSLA → ~4s).
But grace is **not a "speed vs lateness" knob — it is a fidelity knob.**
A user who sets grace=2 on a SPY strategy to "get in earlier" is not
getting a faster version of the same strategy; they are getting one
whose **live bars differ from its backtest bars** → live/backtest
parity breaks (the #1 project priority). The real cost of low grace is
"your tested edge does not transfer," not "you're a few seconds late."

**Architecture cost.** The sub-minute builder is one per (symbol, tf),
shared by all strategies on it. Naive per-strategy grace ⇒ N builders
(the grace-shadow's multi-variant machinery as production — more CPU;
the O(1) fix left headroom, but still a real cost).

**Better design — don't multiply builders.** Build at the *most
generous* grace once (the correct final bar), and let a strategy that
wants to act earlier take a **provisional read of the still-forming
bucket** at its own chosen offset, then reconcile at final close. That
is exactly the existing `compute_tentative_state` machinery. So
per-strategy "earliness" = per-strategy choice of *when to take the
tentative read* + reconciliation — one builder, no N×.

**Recommendation:** calibrate a per-ticker default grace from the
shadow data (what we're collecting). If user control is wanted, expose
it as a *named, bounded* choice ("Fastest / Balanced / Highest-
fidelity"), never a raw seconds field, with the parity consequence
shown inline — and only make the fast options safe via Idea B.

## Idea B — fast provisional bar + reconciliation; demote REST refresh

**What works:** indicator STATE and DISPLAY can run on a fast (small-
grace) bar. The O(1) `apply_last_bar_correction` reconciles state when
the corrected bar lands; display gets a REST refresh. Errors heal and
do not compound — "won't get too out of whack" is true *here*.

**The hard boundary:** the **fire decision is irreversible.** If the
engine fires an entry on a too-fast bar and the corrected bar would
have *not* triggered, that webhook is already gone — a phantom alert.
"It'll be updated in a few seconds anyway" does **not** rescue an
already-sent alert. So the fire decision needs a **decision-grade
bar** — enough grace that the trigger *outcome* is stable. Note that is
not the same as a 100%-perfect bar: it only has to be right enough that
the trigger does not flip.

**Refinement — confidence-gated firing.** The engine can tell when a
trigger is *marginal* (close within ε of the threshold) vs
*unambiguous*. Fire immediately when unambiguous (most bars); for a
marginal trigger, wait for more grace / the reconciled bar before
committing. Fast when safe, careful when close. This gets most of the
latency win without the phantom-alert risk.

**Is periodic REST refresh still worth it?** Yes — keep it, but its
**role demotes**. With good cache fidelity it is no longer load-bearing
for the fire decision. It becomes: (1) the **parity validator** — never
remove it from the Bars Comparison; (2) a correctness backstop for
display; (3) the slow-path reconciliation source for indicator state.
Lower frequency, lower priority, never blocks anything.

## Idea C — strategy-detail "Data Fidelity" tab

**Good — and it aligns with the project's stated values** (surface
things loudly, no silent defaults, parity is #1). It makes the abstract
notion of "parity" concrete and *per strategy*.

Design notes:
- Per-strategy, for that strategy's symbol+TF: close-match % at each
  grace, expected post-close alert latency, and a **recommended grace**.
- **Recommendation-first, not raw tables.** e.g. "SPY 10Sec — grace 5
  recommended: ≈81% exact, ≈97% within $0.10, ~5s post-close latency.
  Faster (grace 3 ≈50% exact) trades fidelity for ~2s." Let the user
  see the tradeoff and pick a *named* option.
- Natural home for Idea A's setting, and for a live-vs-backtest parity
  indicator. Dovetails with the existing Strategy Health Badge.

## How this flows into the Tier 1/2/3 plan

- **Tier 1/2/3 = backtest *update speed*.** Ideas A/B/C = **live model
  data-source / fidelity strategy.** Related (both serve parity, both
  lean on snapshot/restore) but distinct axes — keep them as separate
  initiatives.
- Call A/B/C the **"Live Execution Fidelity"** initiative, parallel to
  the backtest-speed tiers.
- **Shared primitive:** snapshot/restore. Tier 2 (persist engine
  snapshot for incremental backtest updates) and Idea B (O(1)
  reconciliation of a provisional live bar) are the *same machinery* —
  build it once, use it both places.
- **Spec A+B+C together.** They are interlinked: A (per-strategy grace)
  is a footgun without B (reconciliation) making low grace safe; C is
  the UI for both. A single spec, depending on the calibrated grace
  data (now being collected with the 6s/7s variants).
- **Cross-reference Tier 3.** Idea B ("fire on a decision-grade bar,
  reconcile state after") and Tier 3 ("always start flat") are both
  about live/backtest parity from different angles. Both likely need to
  land before the system is "execution-trustworthy." Cross-reference
  their specs.

## Sequencing thought
1. Tier 1 — still the immediate quick win (independent; unblocks the
   "Update All Data" / cron pain).
2. Live Execution Fidelity (A+B+C) — spec as a unit once the 6s/7s
   grace data is in; it reuses the snapshot/restore primitive.
3. Tier 2/3 — backtest-speed deep work; Tier 2 shares the primitive
   with Idea B.

## Concerns to keep on the record
- **Do not ship per-strategy grace before the reconciliation model.**
  Raw low grace without reconciliation is a direct parity footgun.
- **Phantom alerts are irreversible** — the low-grace fire-decision
  risk is the single most important thing to get right; it erodes
  trust faster than latency does.
- **Keep REST in the parity comparison permanently** — it is the
  ground-truth validator. (Agreed with Kevin.)
