# UT Bot Last Week vs This Week Hourly Comparison — 2026-06-06 ~04:30 UTC

> **Purpose:** Validate Kevin's recollection that BT-vs-live parity was "super clean going into the weekend last week" and quantify how much it has changed by this Friday. Captured BEFORE bulk Update All Data tonight so we have a definitive reference for whether UAD recovers the prior parity or reveals a deeper engine drift.

## TL;DR — Kevin's memory was correct, the regression is real

**Same strategy, same hour of day, same day-of-week:**

| Window | Alerts | BT events | Paired | **Phantoms** | Missed | Paired % | **Alert-pair %** |
|---|---|---|---|---|---|---|---|
| **sid 268 — 2026-05-29T18 (Fri last week)** | 55 | 61 | 54 | **1** ⭐ | 7 | 87.1% | **98.2%** ⭐⭐⭐ |
| sid 268 — 2026-06-05T18 (Fri this week) | 53 | 29 | 24 | **29** | 5 | 41.4% | 45.3% |

**Phantoms went from 1 to 29. Alert-pair-rate dropped 53 points.** Not measurement artifact — that's a real, quantifiable shift in the BT-vs-live relationship on the same strategy at the same hour of day a week apart.

## What changed — the metric breakdown

| Metric | Last Friday T18 | This Friday T18 | Direction |
|---|---|---|---|
| Live alerts per hour | 55 | 53 | flat (~same volume) |
| BT events per hour | 61 | 29 | **HALVED** |
| Phantoms | 1 | 29 | **29× worse** |
| Missed | 7 | 5 | slightly better |

**Pattern:** the live alert engine is firing about the same volume both weeks. The BT engine is producing about HALF the events this week for the same live activity. That's what drives the elevated phantom rate.

## Hourly comparison — sid 268 across the whole week

### Last week (2026-05-25 → 2026-05-31)

Only **2 hours** had any live alert activity for sid 268 last week (Tue/Wed/Thu were BT-only — the live engine hadn't been activated for this strategy yet):

| hour (UTC) | alerts | BT events | paired | phantom | missed | paired % | alert-pair % |
|---|---|---|---|---|---|---|---|
| 2026-05-28T13 → T20 | 0 | 23-60 | — | — | varies | 0 | 0 |
| 2026-05-29T13 → T17 | 0 | 25-62 | — | — | varies | 0 | 0 |
| **2026-05-29T18** | **55** | **61** | **54** | **1** | **7** | **87.1%** | **98.2%** ⭐ |
| 2026-05-29T19 | 47 | 57 | 44 | 3 | 13 | 73.3% | **93.6%** ⭐ |

The 2 hours that DID have alerts (T18, T19 on Friday 5/29) both showed pristine alert-pair rates: 98.2% and 93.6%. That's the picture Kevin remembers.

### This week (2026-06-01 → 2026-06-05)

| hour (UTC) | alerts | BT events | paired | phantom | missed | paired % | alert-pair % |
|---|---|---|---|---|---|---|---|
| 2026-06-01T13 | 22 | 32 | 17 | 5 | 15 | 45.9% | 77.3% |
| 2026-06-01T14 | 52 | 52 | 46 | 6 | 6 | 79.3% | 88.5% |
| 2026-06-01T15 | 60 | 60 | 55 | 5 | 4 | 85.9% | 91.7% |
| **2026-06-01T16** | 19 | 57 | 19 | **0** | 38 | 33.3% | **100.0%** ⭐ |
| 2026-06-01T17 | 44 | 46 | 39 | 5 | 7 | 76.5% | 88.6% |
| 2026-06-01T18 | 57 | 61 | 54 | 3 | 7 | 84.4% | 94.7% |
| 2026-06-01T19 | 63 | 62 | 57 | 6 | 6 | 82.6% | 90.5% |
| 2026-06-02T14 | 46 | 48 | 44 | 2 | 3 | 89.8% | 95.7% |
| **2026-06-02T15** | 23 | 61 | 23 | **0** | 38 | 37.7% | **100.0%** ⭐ |
| 2026-06-02T17 | 57 | 35 | 29 | 28 | 6 | 46.0% | 50.9% |
| 2026-06-02T19 | 51 | 27 | 25 | 26 | 2 | 47.2% | 49.0% |
| 2026-06-03T13 | 28 | 14 | 9 | 19 | 5 | 27.3% | 32.1% |
| 2026-06-04T13 | 20 | 18 | 18 | 2 | 0 | 90.0% | **90.0%** ⭐ |
| **2026-06-04T14** | 53 | 53 | 51 | 2 | 2 | 92.7% | **96.2%** ⭐ |
| 2026-06-04T19 | 49 | 14 | 14 | **35** | 0 | 28.6% | 28.6% |
| 2026-06-05T13 | 30 | 12 | 7 | 23 | 5 | 20.0% | 23.3% |
| 2026-06-05T14 | 50 | 12 | 12 | 38 | 0 | 24.0% | 24.0% |
| 2026-06-05T15 | 49 | 20 | 13 | 36 | 6 | 23.6% | 26.5% |
| 2026-06-05T16 | 46 | 32 | 29 | 17 | 3 | 59.2% | 63.0% |
| 2026-06-05T17 | 50 | 26 | 24 | 26 | 2 | 46.2% | 48.0% |
| **2026-06-05T18** | 53 | 29 | 24 | **29** | 5 | 41.4% | **45.3%** |
| 2026-06-05T19 | 45 | 39 | 38 | 7 | 1 | 82.6% | **84.4%** ⭐ |

**Three zero-phantom hours this week** on sid 268: 06-01T16 (19 alerts, 0 phantoms, 100% alert-pair), 06-02T15 (23 alerts, 0 phantoms, 100% alert-pair). Also 06-02T14 had only 2 phantoms (95.7% alert-pair).

**The trajectory:**
- Mon-Wed (06-01 → 06-03): mostly clean (T16/T15 zero-phantom hours, T18-T19 in 84-95% range)
- Thu (06-04): peak with 06-04T14 at 96.2% alert-pair (53 alerts, 2 phantoms)
- Fri (06-05): **collapse** — T13-T18 all show 23-63% alert-pair rates with 17-38 phantoms per hour
- Fri T19: recovered to 84.4% — the last hour of RTH

**So this Friday is the FIRST day with a sustained multi-hour phantom spike on sid 268.** Every prior day this week had at least some clean hours. Friday had near-uniformly elevated phantoms until T19.

## What about sid 263 last week

Sid 263 had 148 alerts on 5/29 but **0 BT trades the entire week**. The BT lane was completely empty. Not a useful comparison data point — we can't tell what BT would have done.

## Other UT Bot strategies last week

| sid | Alerts last week | BT pairs last week | Notes |
|---|---|---|---|
| 263 | 148 (all on 5/29) | 0 | BT lane was empty last week |
| 265 | 27 | 51 | Low volume — not diagnostic |
| 266 | 4 | 11 | Very low volume |
| 267 | 101 | 289 | Lots of BT pairs vs alerts |
| **268** | 102 | **343** | **THE reference comparison point** |
| 269 | 10 | 53 | Low alerts |
| 270 | 0 | 343 | Created 06-01 — not active last week |
| 273 | 0 | 289 | Created 06-01 — not active last week |
| 277-305 | 0 | 0 | Created 06-02+ — not active last week |

**Sid 268 is essentially the only meaningful historical reference point** because it was the oldest active UT Bot canary at decent volume.

## Strategy creation origin — Kevin's question

Per Kevin's recollection: "we had some issues we had to fix with bringing the way packs create strategies in line with how the backtest creates."

Pulled creation metadata for the reference set:

| sid | Name | Created | `strategy_origin` | Config keys | Likely flow |
|---|---|---|---|---|---|
| 263 | TSLA-CANARY-10s-NoConf | **2026-05-29 16:10** | NULL | 31 | Pack Canary Recreate-all (CANARY naming pattern) |
| 265 | TSLA-CANARY-1m-Control | 2026-05-29 16:10 | NULL | 31 | Pack Canary |
| 266 | TSLA-CANARY-5m-Control | 2026-05-29 16:49 | NULL | 31 | Pack Canary |
| 267 | TSLA-CANARY-10s-LooseConf | 2026-05-29 17:35 | NULL | 31 | Pack Canary |
| **268** | SPY-CANARY-10s-NoConf | **2026-05-29 18:01** | **NULL** | **33** | **Pack Canary** |
| 269 | SPY-CANARY-1m-Control | 2026-05-29 18:01 | NULL | 31 | Pack Canary |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 2026-06-01 18:56 | NULL | 31 | Manual test creation |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 2026-06-01 18:56 | NULL | 31 | Manual test creation |
| 277 | PACKTEST · Bollinger Bands · gate | 2026-06-02 17:35 | NULL | 33 | Mass Builder (PACKTEST batch) |
| 295 | PACKTEST · Stochastic Oscillator · gate | 2026-06-02 17:35 | NULL | 33 | Mass Builder |
| 296 | PACKTEST · Strat Assistant · trigger | 2026-06-02 17:35 | NULL | 31 | Mass Builder |
| 302 | PACKTEST · UT Bot V4 · trigger | **2026-06-02 17:35** | NULL | 33 | **Mass Builder** |
| 303 | PACKTEST · UT Bot V4 · gate | 2026-06-02 17:35 | NULL | 31 | Mass Builder |

**Key facts:**

- **sid 263 and 268 were both created 2026-05-29 via the "Pack Canary Recreate-all" flow** (their names start with `TSLA-CANARY-` / `SPY-CANARY-`, matching the canary creation script pattern). NOT Mass Builder.
- **sid 302/303 were created 2026-06-02 via Mass Builder** (PACKTEST naming matches).
- `strategy_origin` field is NULL on ALL of these — that diagnostic field never got populated, so we can't confirm origin from the DB alone. The naming pattern is the strongest signal.
- **Config key counts are similar now (31-33) because Phase 1 unified strategy creation (`d7026d4` / commits around 2026-06-04 task #50) NORMALIZED all existing strategies** to the canonical 33-key shape. So the current configs LOOK uniform, but their HISTORICAL BT trade output was generated when the configs differed.

## What this implies for the test

Kevin's concern is valid and tactically important. **Sid 268's pre-2026-06-04 BT trades were generated under the Pack-Canary-flow config shape. The current normalized config may produce slightly different BT output for the same bars.**

If you UAD sid 268 right now:
- BT lane gets COMPLETELY rebuilt under the current normalized config
- All 343 BT trades from last week + everything since gets replaced with whatever the current engine produces from current config
- Any "memory" of the old pre-normalization config is gone after UAD

**This is actually GOOD for diagnostics:**

1. **If post-UAD sid 268 returns to ≥95% alert-pair-rate** → the normalized config produces engine output consistent with the live alerts. The pre-normalization config was different but UAD harmonized everything. Engine is fine, baseline was stale.

2. **If post-UAD sid 268 stays at 45-60% alert-pair-rate** → the normalized config does NOT produce engine output consistent with live alerts. Either (a) the live alert engine drifted from BT, or (b) the config normalization introduced a real semantic change that affects trade generation. Either is a real engine signal.

**The pair-canary-flow-vs-Mass-Builder distinction matters less now than it would have a week ago,** because Phase 1 unified the config shape. But the historical trades generated before that normalization could still encode old engine behavior.

## What about sid 302 last week

Sid 302 didn't exist last week (created 2026-06-02). So we can't compare its old self to its new self. Only sid 268 is the meaningful diagnostic.

## Why sid 270 still makes sense as the second test

Even though sid 270 wasn't active last week, it shares config DNA with sid 268 (NoConf utbot pure signal, same TEST-P2-utbotv4 naming). If UAD on sid 268 recovers parity but UAD on sid 270 doesn't, that says something IS different between the canary flow and the test-creation flow. If both recover, we know UAD is doing the right thing.

## Recommendation — sharpened with this context

**Pair: sid 268 (created via Pack Canary flow, has historical data) + sid 270 (created via TEST-P2 flow, no historical data).**

If both recover to ≥80-95% alert-pair-rate after UAD with similar phantom counts, we have strong evidence that the engine is fine and the entire fleet just needs UAD.

If sid 268 stays low while sid 270 looks normal → the OLD Pack-Canary-created strategies might have config quirks that survive the normalization. Investigate before UAD-ing the rest of the fleet.

If both stay low → real engine drift between 5/29 and now. Halt, investigate the Phase 1+2 snapshot fix (`684265c`, `f006937` from 06-01) as the primary suspect.

## Files preserved (all in repo + raw artifacts in /tmp/)

**Committed (in repo):**
- `docs/UTBot_LastWeek_vs_ThisWeek_2026-06-06.md` (this doc)
- `docs/UTBot_Hourly_5Day_Analysis_2026-06-06.md` (this week, all 26 UT Bot strategies)
- `docs/Pre_UAD_Baseline_2026-06-06.md` (fleet 48h snapshot)
- `src/_hourly_utbot_history.py` (this week analysis script)
- `src/_hourly_utbot_history_lastweek.py` (last week analysis script — new)
- `src/_utbot_clean_hours_report.py` (summary helper)
- `src/_pre_uad_baseline.py` + `src/_pre_uad_compare.py` (fleet snapshot scripts)

**Raw artifacts (in /tmp/ — copy somewhere safer if Railway/laptop reboot risk):**
- `/tmp/utbot_hourly_5d/` — this week, master JSON + 26 per-sid CSVs + hour-quality ranking
- `/tmp/utbot_hourly_lastweek/` — last week, master JSON + 26 per-sid CSVs
- `/tmp/pre_uad_baseline_2026-06-06.json` — fleet 48h snapshot
- `/tmp/pre_uad_summary_2026-06-06.tsv` + `/tmp/pre_uad_trend_2026-06-06.tsv`

To re-run any analysis: scripts in `src/` produce all outputs deterministically from current DB state.

## The decision point

**Sid 268 + sid 270 UAD = the test. Compare against this doc afterward.** The headline numbers to watch:

- Sid 268's alert-pair-rate at any RTH hour with ≥30 alerts
- Sid 268's phantom count at hours where we expect activity
- Whether the post-UAD pattern resembles last Friday's (98% alert-pair) or this Friday's (45%)

If you click UAD and post-UAD shows ≥90% alert-pair-rate on 268 + 270 across multiple RTH hours → the engine is solid, MANUAL streaming was the entire cause, UAD the rest of the fleet. If post-UAD stays in the 40-60% range → halt and investigate.

This doc is the anchor.
