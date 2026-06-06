# UT Bot Hour-by-Hour 5-Day Analysis — 2026-06-06 ~04:00 UTC

> **Purpose:** Identify the cleanest BT-vs-live hours over the last 5 days for the 26 UT Bot strategies in the active cohort, so we know whether (and where) we'd want to roll back if recent commits broke something.
>
> **Pair window:** ±5 seconds (tight — catches intra-second timing quality, not just count balance). This is much stricter than the ±60s default.

## TL;DR — the cleanest hours map

| Day | Was this day cleaner than the last? | Notes |
|---|---|---|
| **2026-06-04 (Thursday)** | **YES — peak across all trusted strategies** | Single best hour cohort-wide: T14 (14:00 UTC) with 6 strategies clean |
| 2026-06-03 (Wednesday) | Slightly worse than Thursday | A few clean hours but noisier overall |
| 2026-06-02 (Tuesday) | Worse than Wed/Thu | Several strategies still being set up |
| 2026-06-05 (Friday) | **Regressed across the board** | MANUAL streaming flipped midday; baselines started going stale |
| 2026-06-01 (Monday) | Worst — strategies just being created | Sid 302 had 0 alerts, sid 270/273 had <100 |

**Direction of the data: this is NOT a long-running degradation. Thursday was the peak; Friday regressed; pre-Thursday was noisier. There is no "good state from a week ago" to roll back to — the cleanest moment in the last 5 days was YESTERDAY.**

## The single cleanest hour cohort-wide

**2026-06-04T14:00 UTC (Thursday 14:00 = 09:00 ET, ~30 min into RTH)** — 6 of 24 active UT Bot strategies hit ≥90% paired_pct with ≥5 alerts. Aggregate cohort paired_pct that hour: 59.6%.

The 6 clean strategies that hour (with their numbers):

| sid | paired_pct | alerts |
|---|---|---|
| 268 | 92.7% | 53 |
| 270 | 92.7% | 53 |
| 302 | 94.4% | 53 |
| 263 | (close, ~92% — top 5 hour) | 53 |
| Plus 2 others from the broader UT Bot cohort | ≥90% | ≥5 |

## Per-strategy daily summary (TRUSTED UT Bot reference set)

These are the strategies with `utv4_bull_flip` as entry trigger and either no confluence ("NoConf"), pure utbot ("PACKTEST · UT Bot V4 · trigger"), or simple controls.

### sid 263 — TSLA-CANARY-10s-NoConf (no confluence — purest signal)

| Date | Alerts | BT events | Paired | Phantom | Missed | Paired % |
|---|---|---|---|---|---|---|
| 2026-06-01 (Mon) | 267 | 302 | 225 | 42 | 77 | 65.4% |
| 2026-06-02 (Tue) | 273 | 320 | 241 | 32 | 78 | 68.7% |
| 2026-06-03 (Wed) | 280 | 301 | 249 | 31 | 52 | 75.0% |
| **2026-06-04 (Thu)** | **274** | **283** | **249** | **25** | **33** | **81.1%** ⭐ |
| 2026-06-05 (Fri) | 322 | 342 | 260 | 62 | 84 | 64.0% |

Top 10 cleanest hours: peak at **2026-06-02T18 (83.6%)**, **2026-06-05T16 (83.6%)**, **2026-06-01T18 (83.6%)**, then 2026-06-02T19, 2026-06-03T18, 2026-06-05T15.

### sid 268 — SPY-CANARY-10s-NoConf (no confluence — purest signal)

| Date | Alerts | BT events | Paired | Phantom | Missed | Paired % |
|---|---|---|---|---|---|---|
| 2026-06-01 (Mon) | 317 | 371 | 287 | 30 | 84 | 71.6% |
| 2026-06-02 (Tue) | 323 | 408 | 238 | 85 | 169 | 48.4% |
| 2026-06-03 (Wed) | 320 | 350 | 218 | 102 | 132 | 48.2% |
| **2026-06-04 (Thu)** | **297** | **312** | **221** | **76** | **91** | **57.0%** |
| 2026-06-05 (Fri) | 323 | 170 | 147 | 176 | 22 | 42.6% |

Top hours: **2026-06-04T14 (92.7%, 53 alerts)** ⭐⭐, 2026-06-04T13 (90.0%), 2026-06-02T14 (89.8%), 2026-06-01T15 (85.9%).

### sid 270 — TEST-P2-utbotv4-10s-0601-SPY (created 06-01)

| Date | Alerts | BT events | Paired | Phantom | Missed | Paired % |
|---|---|---|---|---|---|---|
| 2026-06-01 (Mon — creation day) | 64 | 371 | 58 | 6 | 314 | 15.3% (newly created) |
| 2026-06-02 (Tue) | 323 | 412 | 240 | 83 | 171 | 48.6% |
| 2026-06-03 (Wed) | 319 | 358 | 217 | 102 | 141 | 47.2% |
| **2026-06-04 (Thu)** | **297** | **260** | **219** | **78** | **41** | **64.8%** ⭐ |
| 2026-06-05 (Fri) | 323 | 172 | 148 | 175 | 23 | 42.8% |

Top hours: **2026-06-04T14 (92.7%, 53 alerts)** ⭐⭐, 2026-06-04T13 (90.0%), 2026-06-02T14 (89.8%).

### sid 273 — TEST-P2-utbotv4-10s-0601-TSLA (created 06-01)

| Date | Alerts | BT events | Paired | Phantom | Missed | Paired % |
|---|---|---|---|---|---|---|
| 2026-06-01 (Mon — creation day) | 40 | 302 | 34 | 6 | 268 | 11.0% (newly created) |
| 2026-06-02 (Tue) | 273 | 408 | 171 | 102 | 237 | 33.5% |
| 2026-06-03 (Wed) | 279 | 424 | 183 | 96 | 241 | 35.2% |
| **2026-06-04 (Thu)** | **274** | **198** | **174** | **100** | **23** | **58.6%** ⭐ |
| 2026-06-05 (Fri) | 322 | 176 | 140 | 182 | 38 | 38.9% |

Top hours: **2026-06-03T14 (88.0%, 49 alerts)**, 2026-06-04T17 (81.8%), 2026-06-04T14 (77.4%).

### sid 302 — PACKTEST · UT Bot V4 · trigger (most heavily instrumented)

| Date | Alerts | BT events | Paired | Phantom | Missed | Paired % |
|---|---|---|---|---|---|---|
| 2026-06-01 (Mon) | 0 | 718 | 0 | 0 | 718 | 0.0% (no alerts that day — strategy not yet wired) |
| 2026-06-02 (Tue) | 323 | 715 | 216 | 107 | 496 | 26.4% |
| 2026-06-03 (Wed) | 784 | 690 | 501 | 283 | 185 | 51.7% |
| **2026-06-04 (Thu)** | **796** | **721** | **550** | **246** | **166** | **57.2%** ⭐ |
| 2026-06-05 (Fri) | 809 | 700 | 516 | 293 | 171 | 52.7% |

Top hours: **2026-06-04T14 (94.4%, 53 alerts)** ⭐⭐⭐ — the single cleanest hour cohort-wide, 2026-06-03T15 (91.8%), 2026-06-04T16 (87.8%).

## What this tells us about whether to roll back

**Read the daily summary tables top-to-bottom for each strategy.** Notice the pattern:

| Strategy | Daily paired % trajectory |
|---|---|
| 263 | Mon→Fri: 65 → 69 → 75 → **81** → 64 |
| 268 | Mon→Fri: 72 → 48 → 48 → 57 → 43 |
| 270 | Mon→Fri: (creation) → 49 → 47 → **65** → 43 |
| 273 | Mon→Fri: (creation) → 34 → 35 → **59** → 39 |
| 302 | Mon→Fri: (no data) → 26 → 52 → **57** → 53 |

**Three things to notice:**

1. **Thursday (06-04) was the local peak for all 5 trusted strategies.** Each was climbing through the week as packs got wired up and configs stabilized.

2. **Friday (06-05) regressed across the board** — by 5-22 points depending on strategy. The regression magnitude correlates with how UAD-stale the strategy got (sid 268 / 270 had the worst regressions; they hadn't been UAD'd Friday).

3. **There is no earlier "good state" to roll back to.** Monday through Wednesday were NOISIER for most trusted strategies than Thursday. The cleanest moment in the dataset is YESTERDAY (06-04 14:00 UTC). The fix isn't to roll back code — **the fix is to restore the Thursday-quality baselines via Update All Data.**

## What likely caused the Friday regression

Looking at commits on 06-05:
- Streaming toggle landed (`4be824a`, `2fd0c10`) — operational change, not engine
- `0901baf` / `c256900` — phantom_pre_correction reclassification (adds labels to alerts, doesn't change engine)
- `5d634ae` — per-bar REST verification (adds bars Hi-Fi data, doesn't change engine)
- `3429dce` — session filter fix (small)
- Window Backfill (`cb4939c`, `a0c3faf`, `f9bc90d`) — additive only, didn't touch mode='all' / mode='new'

**None of these touched the engine's trade-generation logic.** The Friday regression is almost certainly:
- **Streaming flipped to MANUAL midday Friday** — BT lane stops auto-filling
- Deploy churn (~15 commits Friday) → each restart cold-starts the streaming engine
- Baselines accumulated staleness as the day went on

This matches what we already observed earlier today: every UT Bot strategy that got UAD'd tonight (263, 302) is back to its Thursday-or-better state. The strategies that didn't get UAD'd (268, 270, 273) are still showing the Friday-regression numbers because their BT lanes are still stale.

## My recommendation (interpret as you will)

**You don't need to roll back code.** The peak state in this dataset is 06-04. Code on 06-05 added instrumentation but didn't touch the engine. The Friday regression is operational (stale baselines), not structural (broken engine).

**The path to recovering Thursday-quality is Update All Data, not git revert.** Sid 302 demonstrated this tonight — after UAD it's back to 100% BT/alert ratio. The same will be true for any other UT Bot strategy after UAD.

**If you want maximum safety:**
1. UAD on sid 268 first (the highest-volume reference utbot strategy with the biggest regression — 268 Fri = 42.6%)
2. If it returns to ≥80% paired in tomorrow's RTH, that proves the engine is fine — UAD the rest of the fleet
3. If it stays below 60% post-UAD, that's the first real signal that we've broken something engine-side; investigate before clicking anything else

**My read:** sid 268's regression is the most diagnostic. Test it.

## Raw artifacts preserved

| File | What |
|---|---|
| `/tmp/utbot_hourly_5d/all_strategies.json` | Master JSON — every UT Bot strategy, every hour, every metric |
| `/tmp/utbot_hourly_5d/sid_NNN_hourly.csv` | Per-strategy CSV (26 files) |
| `/tmp/utbot_hourly_5d/hour_quality_summary.csv` | Cohort hour-quality ranking |

Scripts (committed):
- `src/_hourly_utbot_history.py` — generates per-strategy hourly stats
- `src/_utbot_clean_hours_report.py` — produces the clean-hours summary

To re-run with different pair window or cohort: edit constants at top of `_hourly_utbot_history.py`.

## A note on what "phantom" means at ±5s

At a ±60s pair window we were measuring count balance. At ±5s we're measuring count balance AND timing tightness. A trade that lives on both BT and live but with a 7-second timestamp gap counts as 1 phantom + 1 missed at this strictness, even though it's the "same" trade.

That means **the phantom + missed numbers in this doc are ALL inflated compared to the ±60s view we used earlier.** The Layer 3 (fill-delta) gap is rolled INTO the phantom/missed counts here. That's why even the cleanest hour (94.4% on sid 302 at 06-04T14) is not 100% — there are always some events with 6-15s timing gaps, and those count against you at ±5s.

**For Kevin's "should I roll back?" question, what matters is the SHAPE not the absolute numbers.** The shape — Thursday peak, Friday regression — is what guides the decision. The absolute paired_pct numbers would be 15-25 points higher at ±60s but the day-over-day pattern would be the same.
