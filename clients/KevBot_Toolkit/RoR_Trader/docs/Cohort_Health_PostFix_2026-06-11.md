# Cohort Health — Post-Gap-Healer Measurement (2026-06-11)

**Window:** 16:58–18:06Z (post-16:28 deploy + 30 min warmup; ends 15 min before the
earliest BT-lane append so backtest coverage is solid). **Tolerance:** ±5s.
**Method:** alerts ↔ BT entry/exit edges, greedy two-pointer (`_remeasure_pair_rates_5s`
pairing), BT lanes freshly rebuilt same-day (full UAD for 303; mode=new appends for rest).
**Comparator:** identical time-of-day window YESTERDAY (2026-06-10), paired against the
SAME current backtest code — so the only variable is which live engine produced the alerts.

## Results (gate cohort, 10Sec primary + 2m gate, SPY)

| sid | today P/Ph/M | today combined% | yesterday P/Ph/M | yesterday combined% |
|-----|--------------|-----------------|------------------|---------------------|
| 303 | 17/38/3      | 29.3%           | 7/48/2           | 12.3% |
| 277 | 9/37/3       | 18.4%           | 7/50/7           | 10.9% |
| 301 | 30/16/4      | **60.0%**       | 17/33/13         | 27.0% |
| 293 | 16/30/6      | 30.8%           | 17/33/5          | 30.9% |
| 299 | 16/30/6      | 30.8%           | 6/46/6           | 10.3% |
| **ALL** | **88/151/22** | **33.7%**   | **54/210/33**    | **18.2%** |

**Verdict (SOP thresholds): IMPROVED** — +15.5 pts combined window-over-window
(threshold for "improved" is +5), phantoms −28%, paired +63%. 4 of 5 strategies improved
substantially (301 +33 pts, 299 +20, 303 +17, 277 +8); 293 flat (its divergence
evidently was never bar-driven).

## Interpretation

1. **The bar layer is FIXED** — independently verified: Per-Bar Parity Drift ribbon at
   100% on every row (bar-exists, OHLC, indicators, 2m gate state) on post-fix windows;
   REST-vs-cache coverage ≈100% on SPY/TSLA/DIA/KO.
2. **The remaining divergence is phantom-heavy and is NOT bars.** With bars + gate
   states at parity, the residual matches the standing decision-timing hypothesis
   ([[project-gate-parity-diagnosis]]): live evaluates triggers intra-bar on the forming
   bar (fires at strategy grace, `fire_on_partial_bucket`); backtest evaluates at bar
   close. A level-cross can fire intra-bar live and revert by close. This is a DIFFERENT
   bug class from today's fix and is the next campaign target.
3. Caveats: ~1h window (SOP prefers ≥2h), one RTH session, simplified local pairing
   (directionally consistent with the by-hour endpoint's method). Re-measure tomorrow
   over a full uninterrupted RTH day — no deploys mid-session.

## Context: what shipped today (iter 0611a)

Gap healer (WS-missed bar insertion + forward recompute), lens faithfulness
(rest_correction/rest_insert/warmup_seed in ENGINE_CONSUMED_SOURCES), deploy-hole cache
reconcile (warmup_seed), flush_stale_bars detection hook, canaries 306 (DIA)/307 (KO).
Full detail: `docs/iterations/2026-06-11_iter_0611a.md`.

Yesterday's full-day baseline (pre-fix, ±5s): 35.0% combined — but that number spans
all hours incl. quiet periods; the apples-to-apples comparison is the same-window
column above (18.2% → 33.7%).
