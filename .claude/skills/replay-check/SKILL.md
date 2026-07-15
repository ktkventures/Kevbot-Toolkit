---
name: replay-check
description: Measure a strategy's ACHIEVABLE fidelity ceiling by replaying its recorded decision-time bars through the REAL live engine and scoring the result with the dashboard's own get_strategy_health. Use to answer "would live have taken that trade with no stalls/lost alerts?" and to split a divergence into operational-loss vs real-logic-bug vs WS/REST-floor — in minutes, without staking out a market day. Reads prod DB (light load); does NOT write. Runs any recorded window, not just clean days.
---

# RoR Trader — Replay Check (achievable fidelity ceiling)

## What this is
`src/replay_harness.py` replays the RECORDED decision-time bars (`live_bars.first_*`
— the exact WS values the engine saw at decision time) back through the REAL live
engine classes (`StrategyMonitor` / `SymbolHub` / `_ShadowIndicatorEngine` — never a
re-implementation), under the ARMED prod flag stack, then scores the resulting trades
with the dashboard's own `get_strategy_health` (identical pairing / coverage /
TBD-exclusion). The gaps between three numbers are the whole point.

## Invocation (run from `src/`)
```
cd clients/KevBot_Toolkit/RoR_Trader/src
../.venv/bin/python replay_harness.py --sids 327,328 --since 2026-07-10T13:30:00Z --until 2026-07-10T20:00:00Z
../.venv/bin/python replay_harness.py --sids 328 --since ...Z --until ...Z --corrected   # floor-vs-logic splitter
```
Reports each sid at **@10s AND @60s** (10s is the honest bar; 60s hides real timing
divergence). Multi-minute offline job — fine to background it.

## The numbers, and how to read them
- **DASHBOARD-live** = what actually happened (the Strategy Health number).
- **CEILING(dtime)** = replay on decision-time (`first_*`) bars = realistic achievable
  (includes the irreducible WS/REST tip floor).
- **CEILING(corr)** = replay on REST-healed bars = pure logic (`--corrected`).
- **self (r≈l)** = replay-vs-live agreement = whether the replay reproduces THIS day's
  live. On a clean day it should be ≥90%; it CRATERS on ops-contaminated days (expected).

Classification:
- `live < ceiling` ⇒ **operational loss** (stalls, lost alerts, dispatch lag — recoverable).
- `ceiling(corr) ≈ 100 but ceiling(dtime) lower` ⇒ **WS/REST data-timing floor** (not a bug).
- `ceiling(corr) ALSO low (== dtime)` ⇒ a **real live-vs-backtest logic residual**.
- `self < 90%` ⇒ the replay didn't reproduce live ⇒ that day was ops-contaminated.

## ⚠️ SCOPE LIMIT — read before trusting a clean ceiling (Brandon audit 2026-07-15)
This is a **BAR/ENGINE replay, NOT a provider-event replay.** It consumes the
ALREADY-AGGREGATED `live_bars.first_*` rows and calls the monitor/shadow classes
directly, so it **bypasses** WS parsing, the WsAggMinute/BarBuilder construction,
reconnect/dup/correction ordering, the flush-vs-fan-out race, and grace firing.
It therefore certifies **"the decision LOGIC is faithful GIVEN the recorded bars"** —
it does NOT certify the bars were CONSTRUCTED / ROUTED / grace-gated correctly. A clean
ceiling does NOT clear the live pipeline. The known live-path construction defects live
BELOW this harness and are locked by `src/test_current_dev_candle_parity_audit.py`
(P0 `>=60s` partial force-close, P1 default-model routing gap, P1 grace suppression).
Cross-check those tests; don't sell the ceiling as a full-pipeline oracle. The durable
fix / harness upgrade is a provider-event replay (route raw WS events through the real
builders) — until then, ceiling = logic layer only.

## Hard rules (each cost hours — do not "simplify" away)
0. **Mirror the ARMED prod flags** (same trap as local-update hard rule 0), or you
   answer a question nobody asked. The harness sets them; verify against
   `railway variables --service Worker`.
1. **Prefer a NIGHTLY-RECOMPUTED SETTLED day as the reference.** Today's stored lane is
   built incrementally and mid-day can disagree with a full recompute. Any day already
   through the 00:20Z nightly Update-All is a known-good reference. The ceiling is immune
   to that day's live contamination, so replay-ceiling vs the settled lane is trustworthy
   even on a stall-hit day (self-check reads low then — expected, not alarming).
2. **Run light on the prod DB during a live session** ([[feedback_local_analysis_starves_live_worker]]).
   The harness loads bars once, not per-tick, for this reason.
3. **Fidelity invariants (in the module header, do not touch):** `bar_count` must
   INCREMENT like the live builder; feed 1Sec through `on_tick` before each close;
   refresher uses strict `>`; no look-ahead (warmup ENDS at `since`); primary runs before
   the secondary fan-out at coinciding closes.

## Validation status (2026-07-15)
- **Non-circular + robust:** 327/328 across 07-08/09/10 — 327 ceiling ~100% every day
  (logic-clean), 328 STABLE ~91% every day (a real, reproducible residual). The harness
  DISCRIMINATES → not echoing the backtest. `--corrected` == decision-time for 328 ⇒ its
  ~9% is a logic residual, not a WS/REST floor; characterized to ONE trade/day
  (session-open convergence + rapid re-entry).
- **Ceilings route through the dashboard's correct optimal pairing**, not the harness's
  local helpers (those were fixed 07-15 per Brandon's audit; his 2 replay-scoring tests
  are green).

## Trade-level diff (which trade diverged, phantom vs missed)
`scratchpad`-style: call `replay(sid, since, until, sb)['entries']` vs `lanes(sb, sid,
since, until)` bt-entries, greedy-unpair at 10s/60s. Pattern in the session's
`char_328.py` / `char_340.py`.

Related: memory `[[project_replay_harness]]`, `docs/_active/Plan_Measurement_Trust.md`
(SSOT), the bug-hunt skill (uses this as the primary DIAGNOSE tool),
`docs/audits/Current_Dev_Candle_Parity_Audit_2026-07-15.md` (the below-the-harness bugs).
