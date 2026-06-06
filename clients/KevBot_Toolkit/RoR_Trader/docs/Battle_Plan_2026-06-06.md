# Battle Plan — Pre-Fleet-UAD State + Revert Candidates

> **Date:** 2026-06-06 ~05:00 UTC, right after sid 268 + 270 UAD verification.
> **Purpose:** Document the current healthy reference state and the revert sequence to fall back to if next week's measurements show continued degradation. Created BEFORE bulk fleet UAD so we have a stable anchor to compare against.

---

## 🚨 ACTIVE DIAGNOSTIC TEST — DO NOT VIOLATE 🚨

### T0 RECORDED — Battle Plan window is OFFICIALLY ACTIVE

| Event | Timestamp |
|---|---|
| **T0** — bulk fleet UAD completion | **2026-06-06T10:10:50 UTC** |
| Test window opens | T0 |
| Test window closes | **2026-06-09 Monday ~20:00 UTC** (RTH close + analysis), or when user explicitly says "exit Battle Plan window" |
| Job that established T0 | `#462509980` — Update all (45 strategies), completed in 14195s, 45 succeeded, 0 failed |

**Baseline snapshots saved at T0:**
- `/tmp/post_uad_baseline_T0_2026-06-06T10-10-50Z.json` — canonical post-fleet-UAD hourly per-strategy state
- `/tmp/utbot_hourly_5d/all_strategies.json` — same content (working copy that gets overwritten on each script run; the timestamped file is the immutable T0 anchor)

**Test window dates:** 2026-06-06 (Saturday) through **2026-06-09 Monday post-RTH (~20:00 UTC, 16:00 ET)**.

**The diagnostic question:** does the live engine accumulate state drift in real-time during a market session, or was Friday's noise a one-time deploy-churn event?

**To get a clean answer, certain actions are blocked. AI assistant instruction: at the START of every turn during this window, check the current date. If it's between 2026-06-06 and 2026-06-09, treat the rules below as ACTIVE. If the user requests something on the DON'T list, FLAG IT before executing — say "this would violate the active diagnostic test (Battle Plan); confirm you want to proceed anyway?" and wait for explicit confirmation.**

### ✅ DO — these are safe during the test window

| Action | Why it's safe |
|---|---|
| **Commit locally** to `dev` branch but DON'T push | No Railway redeploy until pushed |
| **Push to feature branches** (e.g., `dev-feat-X`) | Railway only auto-deploys from `dev`, not other branches |
| **Update docs, memory files, analysis scripts** | If you only need to commit locally or to a feature branch, no deploy impact |
| **Read code, plan, discuss, design** | Pure analysis activity |
| **Create new strategies via the UI** | Strategy creation goes through API but doesn't restart worker engine state for existing strategies |
| **UAD non-canary strategies** | Running UAD on, e.g., sid 305 doesn't affect sid 268's state |
| **Click Update New Data on non-canary strategies** | Same |
| **Work on tools / scripts in `src/_*` files** | Scripts run locally; they don't deploy |

### ❌ DON'T — these would violate the test

| Action | What it breaks |
|---|---|
| **Push to `dev` branch** | Triggers Railway redeploy of worker service → Ralph engine restart → erases accumulated state we're trying to measure |
| **UAD any of the 11 canary strategies** | Wipes their BT lane reference baseline. Canary list: **263, 268, 270, 277, 280, 284, 286, 290, 295, 296, 302** |
| **Click "Update New Data" on canary strategies** | Same — replaces the reference BT data |
| **Restart worker service manually** via Railway dashboard | Resets Ralph engine state |
| **Flip streaming AUTO ↔ MANUAL toggle** | Changes worker behavior during the observation window |
| **Modify engine code:** `unified_engine.py`, `ralph_engine.py`, `data_worker.py`, `data_worker_engine.py`, `worker.py`, anything in `src/user_packs/`, `pack_registry.py` | Would change engine output mid-test — even unpushed, if pushed later during window it invalidates the test |
| **Register new user packs or modify existing ones** | Worker reads pack registry; changes affect output |
| **Recreate/duplicate any canary strategy** | New strategies for the canary IDs would distort their state history |
| **Run `gap_detector` followed by window-backfill on canaries** | Window-backfill changes the BT lane (and is buggy anyway) |

### Weekend deploy window — UPDATED 2026-06-06

**Saturday/Sunday daytime deploys to `dev` are safe** (markets closed → engine is idle → no active drift accumulation to protect). The test only requires no deploys during Monday RTH.

**Specific timing:**
- ✅ **Saturday all day UTC** — full green light
- ✅ **Sunday before ~22:00 UTC** — safe (gives 15+ hours of stable runtime before Monday RTH)
- 🟡 **Sunday after 22:00 UTC / Monday before 13:30 UTC** — borderline; avoid unless urgent
- ❌ **Monday 13:30-20:00 UTC (during RTH)** — hard no; violates the test
- ❌ **Canary strategy UAD any time** — still violates the test (wipes their BT reference baseline)

This means dashboard/UI work on a feature branch OR direct to `dev` is fine this weekend as long as we respect the Monday window.

### 🟡 ASK FIRST — these are gray area; AI should flag and confirm

| Action | Why it's gray |
|---|---|
| **Bug fix in `frontend/`** that needs pushing to `dev` | Frontend service redeploys but not worker. Probably safe but confirm worker isn't auto-redeployed on every push. |
| **Bug fix in `src/api/`** (NOT in worker code) that needs pushing | Same — API service redeploy may not trigger worker redeploy. Worth confirming once via a low-risk push if needed. |
| **Schema migration on Supabase** | DB changes can affect worker if it reads from changed tables. Confirm the change is read-only or doesn't affect alerts/trades/strategies tables. |
| **Hi-Fi Pass 2 or pure-script changes** (no engine impact) | If they only affect post-processing, fine. Confirm the change doesn't touch engine state. |
| **Adding new strategies via Mass Builder** | Mass Builder uses worker via the unified path. Confirm this doesn't restart Ralph for existing strategies. |

### 🚨 The AI assistant's contract during this window

When the user asks the assistant to do something that lands in the **DON'T** column above:

1. **Pause before executing.**
2. **State explicitly:** "this would push to `dev` / UAD canary sid X / modify worker code — that violates the active Battle Plan diagnostic test ending 2026-06-09 ~20:00 UTC."
3. **Ask for explicit confirmation** before proceeding.
4. **If confirmed**, proceed AND note in the response that the test was deliberately broken (so we know during analysis that Monday's RTH numbers may not reflect a clean accumulation period).

The user has full authority to break the test if they decide it's worth it. The assistant's job is only to make the trade-off visible, not to enforce it unilaterally.

### When the test window ends

Either:
- **Monday 2026-06-09 after 20:00 UTC** when the post-RTH analysis is done (normal completion), OR
- **Whenever the user explicitly says "exit Battle Plan window"** or commits a worker-restart change

After that, all restrictions lift. Update this doc to mark the test as COMPLETE and record the verdict.

---

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

## Diagnostic protocol — practical cadence (UPDATED 2026-06-06 ~05:30 UTC)

**The 48-hour total-stillness recommendation was overly conservative.** Updated to be more practical for getting actual work done while still preserving the diagnostic signal.

### T0 — Tonight, after fleet UAD finishes
1. **Save the post-fleet-UAD state.** Run:
   ```bash
   cd src && ../.venv/bin/python _hourly_utbot_history.py
   cp /tmp/utbot_hourly_5d/all_strategies.json /tmp/post_uad_baseline.json
   ```
2. Note the timestamp of the bulk UAD completion. This is `T0`.

### Saturday — work freely, optional 5-min check-in
- **Markets closed; Ralph is idle.** Even if you push docs/scripts, low risk of affecting Monday's RTH numbers.
- **5-min spot-check (OPTIONAL):** re-run `_hourly_utbot_history.py`, glance at sid 268's row. Expected: tiny change from after-hours activity. If you see a big shift, investigate.
- **All dev work is fine** within the Battle Plan rules above (no worker code changes, no canary UADs).

### Sunday — work freely, optional 5-min check-in
- Same as Saturday.
- **One thing worth doing:** by Sunday evening, decide whether to flip streaming AUTO for Monday or stay MANUAL.
  - **MANUAL Monday** = clean test of Friday's pattern (does deploy-churn-free RTH look like Thursday or like Friday?)
  - **AUTO Monday** = test whether streaming pipeline now works correctly without deploy churn
  - **Recommend MANUAL** for the first test — fewer variables.

### Monday — THE TEST WINDOW

**Pre-RTH (before 13:30 UTC):**
1. Re-run `_hourly_utbot_history.py`. Save as `/tmp/pre_rth_monday.json`.
2. **Do NOT push anything worker-affecting after this point until post-RTH.**

**During RTH (13:30 → 20:00 UTC) — 6.5 hours, the actual restricted window:**
- Don't redeploy worker code.
- Don't UAD canary strategies.
- Don't change pack registrations or engine code.
- Don't toggle streaming mode.
- **EVERYTHING ELSE IS FINE.** Frontend work, API work, docs, new strategies, UAD non-canaries, analysis, planning.

**Post-RTH (after 20:00 UTC):**
1. Re-run `_hourly_utbot_history.py`.
2. Compare Monday's RTH hours (13:00-19:00 UTC) to:
   - **This Thursday's clean hours** (97% peak — the gold standard)
   - **This Friday's noisy hours** (60-78% — the degraded state)

| Monday RTH pattern | Verdict | Next step |
|---|---|---|
| Mostly ≥90% combined paired-rate at ≥30 alerts | **Engine is fully healthy.** Friday's noise was deploy churn. | Continue normal operations. Exit Battle Plan window. |
| 80-89% combined paired-rate | **Acceptable but watch for ongoing degradation.** Possibly engine state drift accumulating. | Run UAD twice a week as ongoing protocol. Exit Battle Plan window. |
| Below 80% with mostly clean Layer 1 bars | **Real degradation signal.** | Start the Tier 1 revert sequence (above). |
| Big asymmetric missed-trade pattern (BT >> live) | **Live engine state has drifted.** Investigate which live alert-recording change is at fault. | Start with #57H phantom_pre_correction revert. |

### Day 7 (one week from now)
- Re-run all scripts.
- Stable or improved → healthy state confirmed.
- Degraded vs Monday post-RTH → live engine accumulating drift in real time → begin Tier 1 reverts.

## Summary — the only "do nothing" window is Monday RTH (6.5 hours)

| Time | Status | What's restricted |
|---|---|---|
| Sat 06-07 | Open for work | No worker deploys / canary UADs (other dev work fine) |
| Sun 06-08 | Open for work | Same |
| Mon 06-09 before 13:30 UTC | Final pre-RTH measurement, then quiet | Same |
| **Mon 06-09 13:30 → 20:00 UTC** | **TEST WINDOW** | Same restrictions; this is the active measurement |
| Mon 06-09 after 20:00 UTC | Battle Plan window ENDS after measurement + verdict | All restrictions lift after the verdict is recorded |

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
