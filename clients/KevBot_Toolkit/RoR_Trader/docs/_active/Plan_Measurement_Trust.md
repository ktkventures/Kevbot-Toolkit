# Plan — Measurement Trust & Replay Triangulation

**SSOT for the measurement-hardening workstream** (agreed Kevin ↔ Claude, 2026-07-15).
Goal: make the numbers we compare TRUSTWORTHY and IDENTICAL in definition, so bug-hunting
is fast and honest. Do this BEFORE hard bug-hunting. Don't-code-yet items are marked.

## Why this is first
The replay harness (`src/replay_harness.py`, memory `project_replay_harness`) was validated
07-15: REPLAY-vs-LIVE agreement 271=98%, 267=91%, 308=96%, 333=94%, and its LIVE reproduction
matches the Strategy Health dashboard within 1-2% where numbers were given (136=50%, 308≈92%,
310=0%, 266=12.5% all confirmed via a direct `get_strategy_health` call). BUT the harness's
own paired-% is computed by a DIFFERENT formula than the dashboard, so cross-metric comparison
is not yet apples-to-apples. Fix the numbers, then triangulate.

## The root discrepancy (confirmed in code 07-15)
- Dashboard `combined_pct` (`api/routers/strategy_health.py:578`):
  `paired_cov / (paired_cov + phantom_cov + missed_cov)`, with a `coverage_unix` cutoff that
  AUTO-EXCLUDES the unsettled tail as **TBD**. So the ~15-min settle window never drags the
  score down — recent unsettled trades just wait.
- Harness `combined_pct`: `2·paired / (live + bt)` (symmetric F1-style), NO coverage cutoff →
  it penalizes recent-unsettled trades the dashboard excludes. **Different thing entirely.**
- **Kevin's ruling (correct):** the backtest "staleness" is NOT a blocker. TBD-exclusion
  handles the settle window; snapshot-age ≠ lane-staleness (310 had 8 recent missed / 0 TBD →
  lane current despite a 19-day cold-start snapshot). We do NOT wait for a settled reference;
  we need IDENTICAL calculation building blocks. The only genuinely-stale case (forward-test
  hook actually stalled) is already visible as a large TBD / old coverage edge.

## ⭐ The decision-time vs REST paradigm (Kevin, 07-15) — governs the ceiling
Live is a HYBRID: it DECIDES on decision-time WS bars (`live_bars.first_*`), then the bars
self-heal toward REST (`live_bars` main OHLC = latest/corrected). The Per-Bar Parity panel's
toggle is exactly this: **"first (decision-time)"** vs **"latest (corrected)"**. Fully relying
on REST looks very different from this hybrid.
- The backtest lane is REST hi-fi (settled).
- **The honest ceiling replays on DECISION-TIME (`first_*`) bars** — reproduces what live
  actually decided on, WS-vs-REST tip included. (The harness already does this.) A pure-REST
  ceiling would look artificially clean and represent something live can never hit real-time.
- **Bonus splitter:** run the ceiling BOTH ways. Decision-time ceiling = realistic achievable
  (includes the irreducible WS/REST floor). REST-corrected ceiling = pure logic (floor
  removed). Corrected≈100% but decision-time<100% ⇒ the gap is the data-timing FLOOR, not a
  bug. Corrected also <100% ⇒ a REAL logic bug.

## Phase A — Unify the pairing (the core fix; small)
- LIVE lane = call `get_strategy_health(user=None, window_hours|start/end, tolerance_seconds)`
  directly. DONE — it IS the dashboard number, same code the UI runs → no session-to-session
  drift (Kevin's UI-consistency concern). Use as source of truth.
- REPLAY lane = score the replay's trades with the SAME `_pair_phantom_missed(edge_isos,
  replay_unix_list, coverage_unix=<same cutoff>)` + `paired/(paired+phantom+missed)`. TBD
  excluded on BOTH sides by the same cutoff → directly comparable to the dashboard.
- Report at BOTH 10s and 60s (10s is the honest bar: 308 drops 92%→76% at 10s; 60s hides real
  timing divergence). **Acceptance:** healthy high-N strategy, no ops loss → replay-ceiling
  (decision-time) ≈ dashboard-live within ~2%.

## Phase B — Historical replay (leverage data we already have)
- The ceiling is immune to LIVE contamination (stalls, lost alerts, dispatch lag) — it
  reconstructs from recorded `first_*`. So it runs over ANY recorded window, not just a clean
  live day ⇒ DECOUPLED from waiting for pristine sessions; bug-hunt retroactively on collected
  days. Only dependency: backtest lane covers the window (coverage cutoff already reports how
  far; the rest is TBD-excluded).
- **⭐ PREFER A NIGHTLY-RECOMPUTED SETTLED WINDOW as the reference (Kevin, 07-15).** TODAY's
  stored lane is built incrementally (forward-test hook + settle sweeper) and mid-day it can
  disagree with a full recompute. YESTERDAY (or any day already through the 00:20Z nightly
  Update-All) has a settled, current-logic, full-flags stored lane = a KNOWN-GOOD reference.
  Since the replay ceiling is immune to that day's live contamination, **replay-ceiling vs the
  settled stored lane = a trustworthy fidelity read even on a stall-contaminated day** (the
  self-check will read low for such a day — expected, not alarming). This sidesteps the
  stored-lane-staleness question entirely instead of trying to adjudicate it.

## ⚠️ Gate-parity caveats (learned 07-15 while using build_gate_parity_view)
- Its THEORETICAL backtest (`gate_parity_harness`) has OFFLINE RESOLUTION LIMITS: it cannot
  build some fine-TF secondary ribbons (e.g. 3m-STRAT_ASSISTANT, 3m-VWAP) over a short window
  (NaN-shallow) → `resolved=False` and its `theoretical_bt` UNDER-counts for those strategies.
  Do NOT treat its theoretical-bt as ground truth for 3m-gated sids (333, 340). Use per-gate
  `cb_open_pct` + `divergent_gate` + `phantom_cb_fail` as the signal; get the true entry count
  from the settled STORED lane.
- MIRROR PROD OFFLINE FLAGS when running it (esp. `RORT_ENFORCE_1MIN_GATE=1`) or 1M-gated sids
  are mis-scored (07-15: an un-mirrored run made 329 look like a 5-trade divergence; mirrored
  it's 0-vs-1). Same flag-mirror trap as local-update hard rule 0.

## Phase C — Triangulation signals (the bug-finding lens)
Per strategy, at 10s + 60s:
- **Dashboard-live** (what's happening) × **Replay-ceiling, decision-time** (realistic
  achievable) × **Replay-ceiling, corrected** (pure logic) × **Gate-parity %** (per-gate open%
  agreement live vs backtest — isolates bar/gate fidelity; different bars → different
  indicators → different triggers).
- Classification: live < ceiling ⇒ operational; corrected-ceiling low ⇒ real logic bug;
  gate-parity low ⇒ bar/gate construction; decision<corrected only ⇒ WS/REST floor.

## Phase D — Productize into the UI (destination; deterministic, no drift)
1. **QoL (Kevin, 07-15): Strategy Health should fetch only on an explicit REFRESH click, not
   on every filter change.** Today the custom date selector triggers a full reload per
   adjustment (per-day change = full reload), which is slow and DB-heavy and is why Kevin
   defaults to "last few hours." Debounce to a single fetch on Refresh. Easier on the DB too.
2. Add a **10s-tolerance toggle** (currently 60s only).
3. Add **replay-ceiling** (decision-time + corrected) and **gate-parity** columns, run as a
   BACKGROUND job (the harness is a multi-minute offline job — not inline). Real build.
4. Graduate the harness to its own skill (`/replay-check`) once Phase A validates.

## ✅ Non-circularity + robustness VALIDATED (07-15, answers Kevin's "too good to be true")
Ran 327/328 decision-time + corrected across 3 settled days (07-08/09/10). The ceiling is
NOT a trivial ~100%:
- **327 ceiling = ~100% every day** (dtime==corr: 100/100/100 @60s) → logic-clean; its low
  dashboard-live (43-71%) is 100% operational (recoverable by the fixes + resync-delete).
- **328 ceiling = STABLE ~91% every day** (94.1 / 90.9 / 91.3 @60s), and dtime==corr EXACTLY
  → the ~9% is NOT the WS/REST floor (correcting bars doesn't move it) and NOT operational →
  a genuine live-engine-vs-backtest-engine residual. The harness DISCRIMINATES 327 from 328
  under identical engine/refresher/day → it is not echoing the backtest.
- Self-check (replay≈live) moves independently and CRATERS on contaminated days (07-09:
  327=60% 328=38%) because the FIXED replay engine correctly refuses to reproduce that day's
  broken pre-fix live. A circular harness can't do that.

### 328's residual, characterized trade-by-trade (the honest ~9%)
Every day, ALL entries are second-identical to the backtest EXCEPT exactly ONE, and it's one
of two near-irreducible micro-causes (NOT a systemic bug):
- **07-10:** first entry of the session 3.5 min late (bt 13:45:30 vs replay 13:49:00) —
  session-open fine-TF gate convergence (replay seeds gates as-of 13:30; bt has continuous
  full-history state). Same family as the coarse/fine session-boundary gate work.
- **07-08 & 07-09:** one MISSED entry inside a RAPID RE-ENTRY cluster (bt enters twice 60-90s
  apart — 18:06:00+18:07:30; 19:35:30+19:36:30 — replay takes only the second). Almost
  certainly an exit-timing→re-entry-eligibility interaction (stop fires a bar apart, re-entry
  still position-blocked when the trigger re-fires). NEXT DEEP-DIVE if 328 must clear 90%@10s.
- Consequence: 328's ceiling straddles 90%@10s (88.6/90.9/91.3) — its best achievable is
  borderline against the target; the rapid-re-entry class is the lever to lift it.

## Status / next
- Phase A DONE (harness is dashboard-scored). Non-circularity + multi-day robustness DONE ✅.
- IN FLIGHT: same corrected-splitter across 329/333/340 (07-09/10) → do the rest of the Five
  carry a real logic residual or are they clean/floor like 327/328?
- QUEUED (post-20:00Z close, RTH canary races before then): forward-check — recompute
  328/308/327 fresh + measure TODAY's live (resync-deleted, post-fix) vs the fresh lane. If
  today's live tracks the ceiling, the operational-fix is proven end-to-end on clean data.
- Open question for Kevin (write here if blocked): none — validation is holding.
