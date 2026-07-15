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

## Status / next
- Phase A = next build (small). Validate acceptance on 308 + 271 (high-N healthy) first.
- Then Phase C triangulation on the Five (327/328/329/333/340) + the day's real divergers
  (269, 310 flagged 07-15).
- Open question for Kevin (write here if blocked): none currently — plan agreed.
