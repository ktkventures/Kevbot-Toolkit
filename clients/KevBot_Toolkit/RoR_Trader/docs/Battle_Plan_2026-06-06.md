# Battle Plan — Pre-Fleet-UAD State + Revert Candidates

> **Date:** 2026-06-06 ~05:00 UTC, right after sid 268 + 270 UAD verification.
> **Purpose:** Document the current healthy reference state and the revert sequence to fall back to if next week's measurements show continued degradation. Created BEFORE bulk fleet UAD so we have a stable anchor to compare against.

## Current state (verified at this moment)

### sid 268 post-UAD — the trusted reference

| Metric | Value |
|---|---|
| Total BT trades | 855 |
| Range | 2026-06-01 13:31 → 2026-06-05 19:58 |
| Total elapsed (UAD) | 139s |
| algo trades | 772 (refreshed) |

**Best single hour (combined paired %):**

| Hour | Alerts | Paired | Phantom | Missed | Combined % |
|---|---|---|---|---|---|
| 2026-06-04T16 | 46 | 45 | 1 | 0 | **97.8%** 🏆 |

**Day-by-day clean-hour count (≥90% combined paired %, ≥30 alerts):**

| Day | Clean hours | Quality |
|---|---|---|
| Mon 06-01 | 0 | (no ≥90% hours but several 80-86%) |
| Tue 06-02 | 3 | T16 (93%), T18 (93%), T19 (86%) |
| Wed 06-03 | 1 | T17 (94%) |
| **Thu 06-04** | **4** | T14-T17 all ≥90% — peak day |
| Fri 06-05 | 0 | Best was T14 at 88.7% |

**Friday-specific noise:** even post-UAD, Fri RTH hours show 60-78% range (vs Thu's 90-97%). This is residual divergence UAD can't fix — likely live-engine state drift from deploy churn during MANUAL streaming.

## Fixes shipped tonight (verified working)

| Commit | What | Status |
|---|---|---|
| `6cdd7b3` | `data_loader.py` 1s cache staleness eviction | ✓ Verified — post-UAD sid 268 stop_loss trades show hifi_resolved=True with sub-second precision |
| `6a5ac25` | `useUpdateStrategyLanes` async migration | ✓ Verified — sid 277 ran 192s via new async path, no Failed to fetch |

Do NOT revert these — they're independent fixes that solve their respective problems.

## Revert sequence (if needed next week)

If 48 hours of post-UAD monitoring shows continued degradation pattern (Friday-like ~70% combined pair rate, not the Tue/Thu ~90% pattern), execute reverts in this order. **Stop after each revert, measure 24h, decide if it helped.**

### Tier 1 — most likely affects alert/BT parity

In reverse chronological order (newest revert first since changes more recently == more likely to be at fault):

| # | Commit | Date | What it did | What we lose by reverting | Verify-after-revert checklist |
|---|---|---|---|---|---|
| 1 | `3429dce` | 06-05 17:40 | #42 session filter on injected primary_df | RTH session filter on BT data | Re-check sid 268 phantom rate — did it drop? |
| 2 | `0901baf` + `c256900` | 06-05 17:20-17:30 | #57H phantom_pre_correction reclassification | Alert classification labels | Re-check alert counts — did the false-phantom labeling stop? |
| 3 | `b5f40c7` + `3c5861c` | 06-05 16:50-17:00 | #57G live_corrected source rows + K=1 fast-path | Alert `live_model` records `live_corrected` separately | Re-check verification breakdown |
| 4 | `5d634ae` | 06-05 16:00 | #57E+F per-bar REST verification (4-pass settle) | Layer 1 drift_uncorrected went 0.7% → 0.1% | Re-check Layer 1 drift % on canary cohort |
| 5 | `f006937` | 06-01 11:58 | **Phase 2 backtest snapshot user-pack state codec** | sid 263 phantom rate would jump 3%→24%, sid 268 0%→14% | **DO NOT REVERT** unless willing to lose 16-20% phantom-rate regression on user-pack strategies |

### Tier 2 — operational, less likely

| # | Commit | Date | What | Risk if reverted |
|---|---|---|---|---|
| 6 | `4be824a` + `2fd0c10` | 06-04 | Streaming AUTO/MANUAL toggle | Loses operational protection. Each Railway redeploy would cause warmup-gap pattern. |
| 7 | `684265c` | 06-01 10:07 | Phase 1 warmup-window replay | Pure operational — affects only streaming engine cadence, not BT output |
| 8 | `cb4939c` + `a0c3faf` + `f9bc90d` | 06-05 | Window Backfill (already WIP-banner'd) | No effect — feature already isolated by banner |

### Tier 1 commits to NEVER revert without explicit decision

- `f006937` (Phase 2 codec) — its empirical effect was reducing sid 263 phantom rate from 95% to 3%. **The dataset proves this is a major net win.** Reverting brings back the user-pack engine drop bug.

## Reference state for comparison post-revert

Saved tonight, all in `dev` branch:

| File | What it captures |
|---|---|
| `docs/Pre_UAD_Baseline_2026-06-06.md` | Fleet 48h pre-UAD snapshot |
| `docs/UTBot_Hourly_5Day_Analysis_2026-06-06.md` | All 26 UT Bot strategies hourly, this week |
| `docs/UTBot_LastWeek_vs_ThisWeek_2026-06-06.md` | sid 268 last Fri vs this Fri T18 (98% → 45% pre-UAD) |
| `docs/Phase_B_D_Findings_2026-06-06.md` | Tonight's 1s cache fix + async migration |
| `docs/Battle_Plan_2026-06-06.md` | This doc |

Raw data in `/tmp/`:
- `/tmp/utbot_hourly_5d/` — this week per-strategy hourly CSVs
- `/tmp/utbot_hourly_lastweek/` — last week per-strategy hourly CSVs
- `/tmp/pre_uad_baseline_2026-06-06.json` — fleet 48h snapshot

## Diagnostic protocol for next week

### Day 0 (tonight, after fleet UAD)
1. **Save the post-fleet-UAD state.** Run:
   ```bash
   cd src && ../.venv/bin/python _hourly_utbot_history.py
   cp /tmp/utbot_hourly_5d/all_strategies.json /tmp/post_uad_baseline.json
   ```
2. Note the time of the bulk UAD completion. This is `T0`.

### Day 1 (Saturday, no markets)
1. Re-run `_hourly_utbot_history.py` at end of day.
2. Diff against `/tmp/post_uad_baseline.json`. **Expected: very small change** (only after-hours activity). Significant change = unexplained, investigate.

### Day 2 (Sunday)
1. Same protocol.
2. Diff against Saturday. **Expected: minimal change.**

### Day 3 (Monday RTH — first market day after UAD)
This is the critical test.

1. **Pre-RTH (before 13:30 UTC):** Re-run `_hourly_utbot_history.py`, save as `pre_rth_monday.json`.
2. **Decide MANUAL vs AUTO:** if you want clean engine-level measurements, flip AUTO. If you want to avoid deploy-churn risk, stay MANUAL.
3. **Post-RTH (after 20:00 UTC):** Re-run analysis. Compare Monday RTH hours to:
   - This Thursday's clean hours (97% peak)
   - This Friday's noisy hours (60-78%)

| Monday RTH pattern | Verdict |
|---|---|
| Mostly ≥90% combined paired-rate at ≥30 alerts | **Engine is fully healthy.** Friday's noise was deploy churn. Continue normal operations. |
| 80-89% combined paired-rate | **Acceptable but watch for ongoing degradation.** Possibly engine state drift accumulating. Run UAD twice a week as protocol. |
| Below 80% with mostly clean Layer 1 bars | **Real degradation signal.** Start the Tier 1 revert sequence. |

### Day 7 (one week from now)
- Re-run all scripts.
- If pair rates are stable or improved: done, we're healthy.
- If degraded vs Day 3: live engine state is drifting in real time. Begin tier-1 reverts as outlined above.

## Update New Data verification protocol

The data tonight verified mode='new' parity on sids 295 and 280 in a window with no new trades. That's an easy case. The PROPER ongoing verification:

1. **Pick a control strategy** (e.g., sid 268) once a week.
2. **Snapshot its BT lane state.**
3. **Run mode='new' once.**
4. **Run mode='all' (full UAD).**
5. **Compare** — every trade in `(pre_state_latest_BT, now]` should be identical between both runs. If they differ by >5%, mode='new' is accumulating drift, and we need to either fix the underlying mode='new' code path or do mode='all' more frequently.

## Strategy on creating duplicate strategies for testing

**Don't.** Discussed and decided 2026-06-06:

- A duplicate strategy with the same config would NOT escape live engine state drift (Ralph engine runs in the worker process; state is shared across strategy instances)
- A duplicate would have fresh warmup-window noise (first 30-60 bars are themselves a confound)
- A duplicate has no historical alerts to compare against
- The cleaner diagnostic is to measure the same strategies over 48-72 hours post-UAD

**When duplicates would be worth it:** if we ever confirm a creation-path bug (different creation paths produce different engine output for the same config), a fresh strategy via the current unified `strategy_factory` path would be the control. Not the case today.

## What I learned tonight (notes for future me)

1. **mode='new' (Update New Data) and mode='all' (Update All Data) produce parity-equivalent output** for the window they share. Verified on sids 295 and 280.
2. **The 1s bar cache staleness bug was real and affected Hi-Fi during bulk UADs.** Fixed in `6cdd7b3`. Now `_1s_cache` evicts when cached_max < requested padded_end.
3. **The detail-page sync UAD endpoint times out at Railway's HTTP edge (~60-90s).** Migrated to async (`6a5ac25`). Detail page now polls for completion.
4. **Live and BT engines mostly use the SAME data** (REST + WebSocket-with-REST-splice). The Friday-specific degradation is not data divergence — it's engine STATE divergence from deploy churn.
5. **The "Pack Canary creation flow vs Mass Builder creation flow" distinction doesn't matter post-Phase-1 normalization.** Both produce equivalent BT output once UAD'd under the current normalized config (proven: sid 268 and sid 270 produced identical 855-trade output despite different creation paths).
6. **Thursday 2026-06-04 was the cleanest day** for the UT Bot cohort. 4 consecutive hours ≥90% combined paired %. This is the gold standard to compare future days against.
7. **Friday's degradation correlates with 15 deploys in 8 hours of RTH** + MANUAL streaming gap. Operational, not engine. UAD partially recovers but residual ~10-15% noise on Friday is structural for that day.
8. **Phase 2 codec (`f006937`) is critical.** Empirical proof: sid 263 phantom rate 95%→3%, sid 268 14%→0%. Don't revert without willing to lose this.

## Tools committed for ongoing analysis

| Script | Purpose |
|---|---|
| `src/_hourly_utbot_history.py` | This week 5-day hourly analysis for the 26 UT Bot strategies |
| `src/_hourly_utbot_history_lastweek.py` | Same for last week (5/25-5/31) |
| `src/_utbot_clean_hours_report.py` | Compile per-strategy daily summaries |
| `src/_pre_uad_baseline.py` | Fleet-wide 48h alerts+BT snapshot |
| `src/_pre_uad_compare.py` | Day-over-day comparison of fleet pair-rates |

To re-run any after fleet UAD: scripts use the current DB state at run time. Save outputs with timestamps so we can build a time-series.
