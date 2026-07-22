# Plan — Replay-Harness Secondary-TF Gap (make the ceiling honest for coarse / sub-primary / short-fine gates)

**Status: ✅ EXECUTED 2026-07-21 (session ran Steps 0+1+2b+sub-minute re-true; all gates passed).
See "Step 0 verdicts + results" at the bottom.** Original scoping below, kept for the record.

The replay harness
(`src/replay_harness.py`) produces **0 entries** for strategies gated on certain secondary-TF
classes (observed: **314** 4h/1h coarse, **339** 30m + 10s, **333/340** short-window 3m). The V2
reporting columns and confident nightly auto-arm both depend on the harness telling the truth about
these strategies. This doc scopes the fix. It is deliberately **diagnose-first** — part of the 0
may not even be a bug.

## What we actually know (grounded 2026-07-20 — a refuted hypothesis)
The first theory ("no decision-time bars at coarse/10s → the gate can't advance") is **WRONG**. A
`live_bars` census for TSLA (314/339's symbol) shows rows at **every** TF they need — 10s (11.8k),
30m (96), 1h (48), 4h (12), 1d (2), all current. **The events exist.** So a 0 can only come from:
1. **Empty / short warmup** — `warmup_asof` returns too few (or zero) bars, so `if len(sdf):
   shadow.warmup(sdf)` is skipped, the shadow is never seeded, and the gate key never enters the
   confluence dict. Most suspect for **10s** (sub-minute historical warmup over 7d may not be
   servable) and deep-lookback coarse indicators.
2. **Indicator under-warmed** — bars present but the indicator (e.g. 30m Bollinger squeeze, a 4h
   indicator) is still NaN across the window → confluence never flips True → gate stays closed.
3. **The gate was genuinely closed** — 0 entries is the **correct** answer and "harness limit" was a
   wrong label. **We must not assume it away.** (Question-the-framework discipline.)

## ⛔ Rule #1 — HARNESS-ONLY. Do NOT touch shared engine/backtest code.
The harness is an **offline measurement tool that does not deploy.** All changes stay inside
`replay_harness.py` (and its own helpers). **No edits to the shared live/backtest engine, the
shadow engines, warmup builders, or `services.py`/`ralph_engine.py`/`data_*`.** This is what makes
the work **zero-risk to live strategies** (342/136/314 fire through the live engine, which this work
never touches). **If a fix seems to require a shared-engine change → STOP.** That is out of scope
here; it becomes a separate, flag-gated, byte-identical-validated engine change — never a quiet edit
riding this session. Confirm at the end: `git diff --stat` shows **only** `replay_harness.py`
(+ this doc / a test).

## The steps

### Step 0 — DIAGNOSE (do this first; it decides everything else)
Run 314 and 339 (and 333/340) under `RH_DEBUG=1` over a known window and, **per gated secondary
class**, classify the 0:
- Instrument the warmup: log `len(sdf)` per shadow after `warmup_asof`, and whether
  `seed_mtf_from_shadow` ran. **Empty warmup?** → Step 2a (data/depth).
- Instrument the ribbon: log the confluence state at each primary close (the `RH_DEBUG` gate dump
  already exists). **NaN / never-True?** → Step 2b (indicator warmup depth). **Resolved but closed
  the whole window?** → it's a **real 0**, mark it and move on (NOT a bug).
- Cross-check against the **settled backtest** for the same window: if backtest also took ~0 entries
  there, the harness 0 is correct. If backtest took N and the harness took 0, it's a genuine gap.

**Output of Step 0:** a per-class verdict (empty-warmup / under-warmed / real-closed) that selects
the fix — or proves no fix is needed.

### Step 1 — FAIL-LOUD (small, mandatory regardless of Step 0)
Make the harness **distinguish "couldn't resolve the ribbon" from "resolved and legitimately
closed."** When a gated shadow's ribbon is unresolved (empty warmup or all-NaN through the window),
the strategy's ceiling must be reported as **`UNRESOLVED`**, never a silent `0`. A `0` must mean "the
engine genuinely took no trades," full stop. This is the actual **reporting prerequisite** (it stops
the harness from lying) and the thing that makes nightly auto-arm safe on these classes (they
**hold**, not mis-arm). Matches the "fail loud, no silent backstops" rule.

### Step 2 — FIX per class, ONLY where Step 0 proved it's real (medium each)
- **2a — warmup availability/depth (10s sub-primary, short-fine 3m).** If the warmup is empty/thin,
  source it correctly: 10s from the **1Sec series** (the harness already loads `sec1` — resample to
  10s for warmup + advance) rather than native historical 10Sec; deepen `days` for short-fine so the
  indicator resolves. **Heaviest / most fragile — the 1Sec path is the deepest rabbit hole; leave it
  last.**
- **2b — coarse resolution (4h/1h/30m).** If under-warmed, extend the canonical resample-from-1Min
  advance (the `canonical_edge` path) to coarse TFs and/or lengthen coarse warmup, so the coarse
  ribbon advances from the engine's own 1Min series rather than depending on sparse coarse bars.
- Each fix carries a **hard validation gate** (below). No gate pass → mark `UNRESOLVED`, don't ship a
  number.

## 🎯 The hard validation gate (per class, non-negotiable)
A class is "fixed" ONLY when: **the canary goes from 0 → a real ceiling that matches its settled
backtest at the fire times** (same discipline as the harness's own trust gate — `REPLAY vs
settled ≥ ~90%` on a clean window). If you can't reach that, the honest outcome is **`UNRESOLVED`**,
not a fabricated ceiling. Never claim a fix without the settled-backtest match.

## Per-class canaries
| Class | Canary sid | Primary | Gated secondary | Note |
|---|---|---|---|---|
| Coarse | **314** | 15Sec | 4h + 1h | coarse-only; good first fix (2b) |
| Coarse + sub-primary | **339** | 30Sec | 30m **+ 10s** | the 10s is SMALLER than the primary (see heatmap note) |
| Short-fine | **333 / 340** | (fine) | 3m short-window | NaN-shallow warmup (documented) |

## Regression guard / standard precautions
- **Live strategies are untouched by construction** (Rule #1 — harness doesn't deploy). Turning on
  **342 / 136 / 314** is independent of this work and safe. 314 *fires correctly live today*; the
  harness merely can't SCORE it yet — its live paired-% is still measurable on the dashboard (live
  vs backtest), which is the primary validation. The harness ceiling is a secondary check that this
  work unblocks.
- If Step 0 ever tempts a shared-engine edit → out of scope (see Rule #1); if one is genuinely
  needed, spin it out as a flag-gated change and run `fidelity_parity_suite.py` (canary 267, 18/18)
  before anything.
- End-of-session check: `git diff --stat` = harness only.

## Related design decision — Confluence heatmap: BLUE = mixed (Kevin, 2026-07-20)
**Separate from the measurement fix (this is frontend viz), same sub-primary root theme — recorded
here so it isn't lost.** Today the confluence heatmap paints each primary bar **green** (gate valid)
or **red** (gate invalid). When a secondary TF is **smaller than the primary** (e.g. 339's 10s under
a 30Sec primary), the sub-primary gate can flip **within a single primary bar**, so neither green nor
red is honest. **Decision: paint those bars BLUE = "mixed"** — but ONLY for secondary TFs smaller
than that strategy's primary. Tri-state: green = all sub-intervals open, red = all closed, blue =
flipped within the bar.
- **Doable:** the backtest / strategy-mask backtester computes the sub-primary gate at its own
  resolution, so "did it flip within this primary bar" is derivable.
- **Clarify the semantics on the panel:** the *entry decision* still keys off the **last closed
  sub-primary bar before the primary close** (the existing ordering invariant) — blue is an honest
  *display* of intra-bar instability, not a change to decision logic. A blue bar can still have fired
  an entry (on its close-state); the color just tells the human the gate wasn't stable across the bar.
- **Scope:** a small frontend change on the Strategy Detail confluence heatmap / gate-parity panel —
  a sibling task, foldable into the V2 / gate-parity work, NOT part of the harness session's core.
  Kevin's bar: "ultimately if it lines up with the backtest, that's what I care about."

## Launch + done
- **Launch:** fresh session — "work `Plan_Harness_Secondary_TF_Gap.md`: Step 0 diagnose 314/339
  (+333/340), then Step 1 fail-loud, then Step 2b coarse; hard-gate every fix to the settled
  backtest or mark UNRESOLVED; harness-only per Rule #1." Can run as a `/loop` (diagnose → fix one
  class → validate → next) with a time-box.
- **Done (this session's target):** Step 0 verdicts recorded + Step 1 fail-loud shipped + the coarse
  class (314) either fixed-and-matched or honestly UNRESOLVED. 10s/1Sec + short-fine = follow-on.
- **Non-goals:** the 1Sec sub-primary path if it turns gnarly (defer); any shared-engine change; the
  heatmap-blue frontend (sibling).

---

# ✅ Step 0 verdicts + results (session 2026-07-21, window 2026-07-16 13:30–20:00Z)

Diagnostic window: **2026-07-16** — the one recent day where the settled backtest actually fired for
the canaries (314=62, 339=3, 340=2 entries). NOTE: **333 (and 327/328/329) no longer exist in the
DB** (removed in the bulk-delete work) — short-fine was diagnosed via 340; the fine-TF regression
canary is now **267** (2m gate, 91 bt entries on 07-16).

## Verdicts (per class)

| Canary | Class | Verdict | Root cause |
|---|---|---|---|
| **314** | Coarse 4h/1h | **HARNESS BUG — refresher-sim under-warmed (2b), FIXED** | 4h shadow seeded fine from a 60-day warmup (80 bars, stoch resolved), then the FIRST 120s refresher tick re-trued it from `canonical_upto` = the **7-day** 1Min resample = **11 bars** → stoch(14) NaN → published `[]` → coarse gate shut for the rest of the day. Live's refresher reloads per-TF rightsized depth (`_load_warmup_df`), so live never had this — it fired 271 entries while the harness said 0. |
| **339** | Sub-primary 10s (+30m, +GEN) | **Two causes: harness GEN-gap (FIXED) + real live-engine LABEL DRIFT (spin-out)** | (1) The harness built monitors with `general_packs=[]` → 339's required `GEN-TIME_OF_DAY` record could never appear → now mirrors the worker's admin pack load (GEN active 180/779 closes). (2) The 10s ribbon **fully resolves** (11,701-row warmup — the "sub-minute warmup not servable" theory is REFUTED; 2,533 publishes, all four UT_BOT states, no NaN) but emits records prefixed **`10Sec-`** while the normalized required token is **`10S-`** → `issubset` can never match. This is the REAL engine class path (live = same code) and explains 339's **0 live alerts ever** while its backtest fires. Replay 0 = the true live ceiling; flagged **LABEL-DRIFT**, not silent. |
| **340** | Short-fine 3m | **NO GAP — real ceiling** | Both ribbons resolve (no NaN); ceiling **100% @10s** with 2 entries matching the settled backtest's 2. The 07-17 `REFRESH_S=120` re-arm already cured the class the plan doc had observed. |
| **325** | Coarse, gate genuinely closed | **REAL 0 — correctly NOT flagged** | 4h SWING ribbon resolves (NEUTRAL all day, 81-bar deep rebuild); required `BULL_C2` never occurred; settled bt also 0 → reported n/a, no UNRES. The tri-state honesty check works. |

## What shipped (ALL inside `replay_harness.py` — Rule #1 held, no engine/shared code touched)

1. **Worker-mirror GEN pack load** — `get_enabled_general_packs(_parse_pack_list(load_general_packs_admin(ADMIN)))`,
   same loader/uid as the worker's 2026-06-22 startup fix.
2. **Step 1 fail-loud** — per required token the replay now classifies:
   `UNRESOLVED` (`PRIMARY-WARMUP-EMPTY` / `GEN-PACK-MISSING` / `NO-SHADOW-ENGINE` /
   `EMPTY-WARMUP-NO-EVENTS` / `RIBBON-NEVER-RESOLVED` / `UNPARSEABLE-TF-LABEL`) → ceiling prints
   **`UNRES`**, never a number; `LABEL-DRIFT` → number stands (it IS the live ceiling) + loud ⚠
   footnote naming the emitted vs required prefix. Plus `RH_DEBUG=1` now prints warmup rows/span
   per shadow, publish-change traces, per-token active-close counts, gate OPEN/CLOSED transitions,
   and a NaN census per shadow.
3. **Step 2b coarse fix** — `canonical_upto` serves coarse TFs (≥3600s) from a **60-day** 1Min
   series (`m1_deep`, as-of-safe, matching `warmup_asof`'s coarse depth). The fine path (60s–<3600s)
   keeps the original 7-day `m1` **byte-identical** (validated-behavior preservation).
4. **Sub-minute re-true fix** — the refresher sim used to re-true 10s shadows from a 1Min→10Sec
   resample (an UPSAMPLE: minute OHLC relabeled as 10s bars — garbage live never sees; live re-trues
   from native sub-minute REST). Now re-trues from the shadow's own seed-warmup + decision-time
   series.
5. Strategy-not-found now raises a clean `strategy N not found (deleted?)` instead of a NoneType
   subscript.

## 🎯 Hard validation gate — PASSED

- **314: ceiling 0% → 92.3% @10s / 95.3% @60s** (dashboard-scored vs settled backtest; 64 replay
  entries vs 62 bt), self-check replay≈live 96%, dashboard-live 85.5% sits below the ceiling as
  operational loss. ≥~90% on a clean window ⇒ gate met.
- Regressions: **267 = 100% @10s** (fine path, 91 bt entries), **340 = 100%** (identical pre/post),
  **325** honest n/a, **339** honest 0 + loud LABEL-DRIFT.

## 🔥 SPIN-OUT (engine bug, NOT fixed here — Rule #1): sub-minute confluence label drift

`_ShadowIndicatorEngine._tf_short_label` shortens `Min/Hour/Day/Week` but **not `Sec`** → a 10s
shadow emits `10Sec-UT_BOT_V4-…` while `_normalize_confluence_label` normalizes the strategy's
required `10s-…` to `10S-…` → **any sub-minute secondary confluence gate can NEVER pass live**
(fail-closed ⇒ silent dead strategy; backtest evaluates its own namespace and fires normally —
exactly 339's live-0-vs-bt-fires signature). Fix belongs in a flag-gated engine session with
byte-parity validation (`fidelity_parity_suite.py`, canary 267 18/18), NOT a quiet edit. Until
then the harness prints the ⚠ LABEL-DRIFT footnote whenever the class appears.

## Follow-ons
- The engine label-drift fix above (owns 339's tradability).
- Confluence-heatmap BLUE=mixed (frontend sibling, unchanged).
- 314's residual live gap (85.5% live vs 92.3% ceiling ⇒ ~7pp operational) — ordinary
  divergence-loop material, no harness work needed.
