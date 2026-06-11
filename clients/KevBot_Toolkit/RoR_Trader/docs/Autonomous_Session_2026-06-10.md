# Autonomous Session — 2026-06-10 (evening, market closed)

Branch: `feat/parity-tonight-2026-06-10` (nothing pushed to dev; nothing deleted).

## 1. Shipped (local, on branch): Per-Bar Parity Drift custom-window picker
Added a **start/end datetime range** control to the Per-Bar Parity Drift card (drives BOTH v1
and v2 ribbons; v1's compare logic untouched — additive `startUtc`/`endUtc` props). When set, the
ribbons restrict to `[start,end]` and raise the bar cap (`CUSTOM_MAX_BARS=2600`) so a full RTH 10s
session renders. Inputs are in the user's display tz (matches the ribbon tooltips). Verified live on
303: applying today's RTH (13:30–20:00 UTC) flips the header to "custom window" and renders the RTH
window. Files: `ParityDriftRibbon.tsx`, `ParityDriftRibbonV2.tsx`, `StrategyDetailPage.tsx`.

## 2. MAJOR diagnosis (Goal 1) — the bar-exists gap is a REAL live-engine decision gap, and bigger than thought
Using the new window on 303's **RTH** (13:30–20:00 UTC, 2026-06-10), confirmed at the **raw** source
level (not a pipeline artifact):

| source | RTH 10s bars | vs REST |
|---|---|---|
| REST (`load_market_data 10Sec`) | 2,341 | — |
| live cache `ws`/`ws_agg` (alert lens) | 2,019 | **−322 (−14%)** |
| raw `live_bars` (all sources) | ~2,303 | −38 (−1.6%) |

- The **322 missing RTH bars are real, high-volume** (median **16,036** shares, 0 zero-vol), scattered
  across the whole session — i.e. the live WS feed genuinely missed ~14% of RTH 10s windows.
- The gap is **NOT after-hours-only** (my earlier framing was wrong) — it's ~14% even in liquid RTH.
- A **`rest_backfill`** process already writes ~283 of those into `live_bars` (different `source`), so
  raw `live_bars` is ~98% complete — but the alert lens **excludes `rest_backfill`** (`sources=['ws','ws_agg']`)
  on purpose, to show what the engine saw live. **So the 86% is faithful, not cosmetic.**

### Root cause (confirmed in code)
`apply_rest_correction` (`ralph_engine.py:3008`) only **corrects values of bars already in the engine's
history**; if the WS missed a bar entirely it logs *"bar not in history — skipping"* and returns False
(`:3057`). So `rest_verifier` fixes values but **cannot add missing bars**. The engine therefore made its
real-time decisions over the 14%-sparser WS series → genuine live↔backtest divergence on the 10Sec cohort
(this likely dwarfs the gap-fill effect we fixed earlier).

### Open question (needs a live look tomorrow)
Is the WS feed (Polygon `A.*` per-second) genuinely not delivering those windows, or is the worker's
aggregation / fire-and-forget `live_bars` writer dropping them? If the engine's in-memory history *did*
have them (write dropped), decisions are fine and only the lens understates; if not, decisions diverge.
The `rest_backfill` rows existing suggests the system already detects the gap.

### Fix direction (Goal 1, real — for tomorrow, with you on)
Extend the engine to **INSERT** WS-missed bars into `builder.history` near-real-time (not skip them),
so the incremental indicators incorporate them and live decisions converge to backtest. `apply_rest_correction`
already does multi-bar-back *value* replay (path 2); this adds an *insertion* path. Caveats: (a) insertion
mid-history requires re-replaying indicators forward from the insert point; (b) the verifier settles
seconds-to-minutes late, so there's an inherent lag for an intra-bar 10Sec strategy — it converges going
forward but can't retroactively un-make a decision. Worth weighing vs. simply accepting that 10Sec WS
fidelity has a floor and/or coarsening the cohort's primary TF.

## 3. Deferred (notes for next)
- **Goal 2 (gate green):** the 2m gate bars were already flat-free; the residual is 2m WS-vs-REST values /
  incremental-vs-batch — not yet characterized. Do on a clean RTH window tomorrow.
- **Dashboard 500:** `/api/admin/strategy-health/by-hour` & `/by-deploy` work locally but 500 on dev
  (the logic is fine — it's an auth/Railway-env issue). Needs the *deployed* traceback to fix; can't
  diagnose blind. Also `deploy_history.json` is stale (last 06-06) so by-deploy can't split pre/post-`0d3affb`.

## Pre-market plan (tomorrow, you on early)
1. Quick `mode=new` UAD on the cohort so the BT lane covers the new window (fast append).
2. Decide Goal 1 approach (engine-insert vs accept floor vs TF change) — it's a real live change; deploy
   pre-market and watch the gate parity + alerts as volume builds; pull back if the microscope disagrees.
3. Characterize the Goal 2 gate residual on a clean RTH window.
4. Re-run `_cohort_health_baseline.py` after a full post-fix day to read the deploy's impact.
