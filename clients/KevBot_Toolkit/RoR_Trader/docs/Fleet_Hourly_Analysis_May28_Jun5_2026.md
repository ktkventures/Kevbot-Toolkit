# Fleet Hourly Divergence Analysis — 2026-05-28 through 2026-06-05

> **Generated:** 2026-06-06 ~15:00 UTC (post-fleet-UAD)
> **Author:** Comprehensive analysis at Kevin's request before deciding on rollback
> **Methodology:** Local pairing at ±5s window, computed independently from strategy_health endpoint
> **Cohort:** All 43 active strategies; 9-day window
> **Pair-rate definition:** combined_pct = paired / (paired + phantom + missed) — the truer single number

## EXECUTIVE SUMMARY (read this first)

After analyzing **9 days × 43 strategies × hour-by-hour** with tight ±5s pairing, the data tells a clear story:

### 1. There IS a real day-over-day pattern — but it's NOT a uniform engine regression

| Day | Trigger-style avg | Gate-style avg | Difference |
|---|---|---|---|
| 2026-06-02 | 19.6% | 9.3% | +10.3 |
| 2026-06-03 | 38.0% | 20.9% | +17.1 |
| 2026-06-04 | **49.3%** | **37.8%** | +11.5 |
| 2026-06-05 | **47.4%** | **15.2%** | **+32.2** ⚠ |

**Trigger-style strategies (use pack's own trigger as entry) stayed roughly flat between Thursday and Friday: -1.9 pts.**

**Gate-style strategies (use utv4_bull_flip entry + pack as confluence gate) collapsed: -22.6 pts.**

### 2. The Friday collapse is concentrated in user-pack-dependent gate logic, NOT the core engine

The 15 trigger-style PACKTEST strategies stayed in the 47-65% range Friday. The 14 gate-style strategies dropped to 5-25% range Friday. **Whatever broke Friday affects how gates evaluate user pack confluence — not the underlying trigger generation.**

### 3. Layer 1 bar fidelity (drift_uncorrected) jumped on 6/04, persisted on 6/05

| Day | Avg drift_uncorrected | Max drift_uncorrected |
|---|---|---|
| 2026-06-02 | 0.10% | 2.17% |
| 2026-06-03 | 0.05% | 0.71% |
| **2026-06-04** | **2.28%** ⚠ | **9.73%** ⚠ |
| 2026-06-05 | 1.64% | 4.78% |

**A 22x jump in average bar drift on 6/04**, persisting elevated on 6/05. This correlates with the data_worker watchdog phase 2 deploy (06-03 10:42 `1dae0d1`) which added auto-restart on persistent stale. Each restart can leave a brief window of unverified bars.

### 4. UAD recovered the fleet today — BUT some strategies still show degraded recent-day performance

Post-fleet-UAD spot-check at T0+4.5h shows zero drift accumulation since UAD. However the 6/05 historical data still shows the gate degradation because those live alerts were recorded as-fired during Friday's deploy churn — UAD can't change historical alert records.

### Bottom line: NO rollback recommended, but targeted investigation needed

The data does NOT support reverting any single commit because:
- Trigger-style engine output remains stable Thursday → Friday
- The Friday collapse is specific to gate evaluation, which involves user-pack interaction known to be fragile
- The Layer 1 drift jump correlates with watchdog auto-restart, which is operational (more restarts = more brief bar gaps)

What IS needed:
- **Investigation into the gate-specific Friday regression** (root cause: which #57 commit + how it interacts with gates)
- **Watchdog auto-restart frequency tuning** (reduce restart cadence to lower drift)
- **Continue Monday RTH measurement** to see if gate regression persists post-UAD with no deploy churn

Details follow.

---

## METHODOLOGY

### Pair window: ±5 seconds (tight)

Strict pairing catches both count balance AND timing tightness. An alert at 14:35:10 pairs only with a BT trade event at 14:35:05-14:35:15. Looser windows (±60s) would mask real timing-quality issues.

### Combined pair-rate formula

```
combined_pct = paired / (paired + phantom + missed)
```

This is the truest single number — penalizes BOTH phantoms (live alert without paired BT trade) AND missed (BT trade event without paired live alert). 100% means every event paired with a counterpart.

### What was pulled

- **Alerts:** all alerts on strategy_id in COHORT, fill_ts in [2026-05-28, 2026-06-06)
- **BT trades:** all trades with strategy_id in COHORT and data_source LIKE 'backtest_%' with entry_fill_ts in same range
- **Each BT trade contributes 2 events:** entry + exit (both timestamps pair-tested independently)
- **Layer 1 verification:** alerts.verification_status counted per hour

### Cohort: 43 active strategies (any alert in last 48h)

```
136, 174, 194, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275,
276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
303, 304, 305
```

### Output artifacts (in `/tmp/fleet_hourly_may28_jun5/`)

- `all_strategies.json` — master JSON with everything
- `sid_<N>_hourly.csv` (43 files) — per-strategy hourly drill-down
- `per_day_per_strategy.csv` — daily summary
- `cohort_by_hour.csv` — cohort aggregate per hour
- `best_worst_hours.csv` — hours ranked by clean-strategy count

Scripts (committed locally, will push after Battle Plan window closes):
- `src/_fleet_hourly_may28_jun5.py` — main analysis
- Reusable for re-running with different date ranges

---

## DATA AVAILABILITY MAP

### Pre-5/28 data — VERY LIMITED

Only 3 strategies have alert data before 2026-05-28:

| sid | Pre-5/28 alerts | Pre-5/28 BT pairs | Earliest alert |
|---|---|---|---|
| 136 | 36 | 92 | 2026-05-22 |
| 174 | 94 | **0** | 2026-05-22 (chronic — never paired) |
| 194 | 4 | 68 | 2026-05-26 |

**Sample sizes are tiny.** Not enough volume for meaningful day-over-day comparison. The pre-5/28 picture is essentially:
- sid 174 was broken before our analysis window even started (still broken)
- sid 136 has BT > alerts ratio similar to current state
- sid 194 has essentially no live alerts

**I'm not lying about pre-5/28 data — it exists but isn't statistically useful.**

### 5/28-5/29 data — partial

- 5/28: only sids 136, 174 had alerts (32 total alerts cohort-wide, 11.8% paired)
- 5/29: 7 strategies had alerts (427 total), 0% paired — but **this is artifact, not real**

The 5/29 0% is misleading: the canary cohort (sids 263-269) was CREATED on 5/29 and started firing alerts, but their BT lanes don't begin until 2026-06-01 13:31 UTC (per current strategy config's data_days/backtest_start_date setting). So those 5/29 alerts have no possible BT pair because BT data doesn't extend that far back.

**This is a data availability artifact, NOT a divergence signal.**

### 6/01-6/05 — full cohort data progressively comes online

- 6/01: 12 strategies active (canaries fully online)
- 6/02: 42 strategies active (PACKTEST cohort created)
- 6/03-6/05: full 42-strategy cohort active

---

## COHORT DAILY AGGREGATES

Only counting strategies that had ≥5 alerts on that day:

| Date | Strategies | Alerts | Paired | Phantom | Missed | Agg paired % |
|---|---|---|---|---|---|---|
| 2026-05-28 | 2 | 32 | 4 | 28 | 2 | 11.8% |
| 2026-05-29 | 7 | 427 | 0 | 427 | 2 | 0.0% (artifact) |
| 2026-06-01 | 12 | 1,249 | 984 | 265 | 1,072 | 42.4% |
| 2026-06-02 | 42 | 6,272 | 3,790 | 2,482 | 9,537 | 24.0% |
| 2026-06-03 | 41 | 14,728 | 7,706 | 7,022 | 4,668 | 39.7% |
| **2026-06-04** | **42** | **16,437** | **10,634** | **5,803** | **3,761** | **52.6%** 🏆 |
| 2026-06-05 | 38 | 14,353 | 8,653 | 5,700 | 3,927 | 47.3% |

**Thursday 6/04 was peak fleet-wide.** Friday 6/05 regressed -5.3 pts cohort-wide.

But the cohort-wide number masks the real story. Looking at trigger-style vs gate-style separately reveals what actually broke.

---

## THE SMOKING GUN: TRIGGER vs GATE PATTERN

### Strategy classification

**Trigger-style** (15 strategies): use a pack's own trigger as the entry signal. No utv4 dependency.
- 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304

**Gate-style** (14 strategies): use `utv4_bull_flip` as entry + pack confluence as gate. Heavy user-pack dependency.
- 277, 279, 281, 283, 285, 287, 291, 293, 295, 297, 299, 301, 303, 305

### Daily comparison

| Date | Trigger avg | Trigger N | Gate avg | Gate N | Difference |
|---|---|---|---|---|---|
| 2026-06-02 | 19.6% | 14 | 9.3% | 4 | +10.3 |
| 2026-06-03 | 38.0% | 15 | 20.9% | 13 | +17.1 |
| **2026-06-04** | **49.3%** | 15 | **37.8%** | 14 | **+11.5** |
| **2026-06-05** | **47.4%** | 15 | **15.2%** | 14 | **+32.2** ⚠⚠⚠ |

### Key observations

1. **Trigger-style stayed flat Thursday → Friday: -1.9 pts.** This is operational noise, not a meaningful regression.

2. **Gate-style collapsed Thursday → Friday: -22.6 pts (a 60% relative drop).**

3. **The gap between trigger and gate strategies WIDENED dramatically on Friday: +11.5 → +32.2.**

4. **Both types were IMPROVING together Mon-Thu (gate +28.5 pts, trigger +29.7 pts).** Whatever was happening earlier in the week affected both equally, and was net-positive (config normalization + EMA PP v3 fix + opposite_signal fix all helped).

5. **Only on Friday do they diverge.** Friday's changes affected gate evaluation specifically.

### Verdict on rollback at this level

A core-engine regression would affect trigger-style AND gate-style equally. The Friday data shows trigger-style is fine. So the cause isn't a core engine break — it's something specific to how gates evaluate user-pack confluence.

---

## PER-STRATEGY DAILY DETAIL

### Trigger-style strategies (combined_pct % with ≥30 alerts)

| sid | 6/01 | 6/02 | 6/03 | 6/04 | 6/05 | Thu→Fri | Verdict |
|---|---|---|---|---|---|---|---|
| 276 | — | 21.7 | 42.2 | 57.5 | 45.1 | -12.4 | Modest dip |
| 278 | — | 2.4 | 17.2 | 60.9 | 59.8 | -1.1 | Flat |
| 280 | — | 2.4 | 17.1 | 60.9 | 59.8 | -1.1 | Flat |
| 282 | — | 19.6 | 34.4 | 50.3 | 43.4 | -6.9 | Modest dip |
| 284 | — | 22.8 | 48.0 | 54.0 | 52.1 | -1.9 | Flat |
| 286 | — | 22.4 | 33.8 | 41.8 | 41.6 | -0.2 | Flat |
| 288 | — | 25.7 | 51.5 | 55.4 | 55.3 | -0.1 | Flat |
| 290 | — | 23.9 | 44.6 | 50.7 | 46.2 | -4.5 | Modest dip |
| 292 | — | 13.8 | 22.2 | 21.9 | 19.3 | -2.6 | Flat (low) |
| 294 | — | 25.0 | 46.3 | 43.2 | 50.0 | +6.8 | **Improved Friday** |
| 296 | — | 27.8 | 61.9 | 64.8 | 64.6 | -0.2 | **Flat-and-healthy** ✓ |
| 298 | — | 16.6 | 28.4 | 36.0 | 35.9 | -0.1 | Flat |
| 300 | — | 24.0 | 66.8 | 74.4 | 69.6 | -4.8 | Modest dip |
| 302 | — | 26.4 | 51.7 | 57.2 | 55.0 | -2.2 | Flat |
| 304 | — | — | 4.0 | 10.5 | 13.2 | +2.7 | **Improving** |

**Trigger-style average Thursday → Friday: -1.9 pts.** Of 15 strategies, 11 were within ±5 pts, 2 improved, 2 dropped 5-15 pts.

### Gate-style strategies (combined_pct % with ≥30 alerts)

| sid | 6/01 | 6/02 | 6/03 | 6/04 | 6/05 | **Thu→Fri** | Verdict |
|---|---|---|---|---|---|---|---|
| 277 | — | 5.7 | 13.8 | 44.1 | 12.1 | **-32.0** ⚠ | Collapsed |
| 279 | — | — | 20.2 | 51.5 | 12.1 | **-39.4** ⚠ | Collapsed |
| 281 | — | — | 20.2 | 51.5 | 12.1 | **-39.4** ⚠ | Collapsed |
| 283 | — | — | 33.3 | 66.1 | 8.6 | **-57.5** ⚠⚠ | Catastrophic |
| 285 | — | — | 16.1 | 22.3 | 22.6 | +0.3 | Flat (already low) |
| 287 | — | 4.2 | 13.4 | 43.8 | 19.4 | -24.4 ⚠ | Collapsed |
| 291 | — | — | 7.1 | 5.7 | 8.7 | +3.0 | Stuck low |
| 293 | — | — | 20.3 | 48.9 | 13.4 | **-35.5** ⚠ | Collapsed |
| 295 | — | — | 5.5 | 22.2 | 11.1 | -11.1 | Drop |
| 297 | — | — | 15.7 | 15.9 | 9.1 | -6.8 | Stuck low |
| 299 | — | 3.1 | 29.7 | 59.4 | 21.4 | **-38.0** ⚠ | Collapsed |
| 301 | — | 24.2 | 46.7 | 47.9 | 46.0 | -1.9 | **Healthy exception** ✓ |
| 303 | — | — | 29.4 | 46.9 | 16.3 | **-30.6** ⚠ | Collapsed (known eppv3 bug) |
| 305 | — | — | — | 3.6 | 0.0 | -3.6 | Already broken |

**Gate-style average Thursday → Friday: -22.6 pts.** 8 of 14 strategies dropped 24+ pts.

**Notable exception: sid 301 (PACKTEST · Swing 1-2-3 · gate) stayed FLAT Thursday → Friday at 46-47%.** What's different about sid 301? Worth investigating — it's the one gate strategy that didn't regress.

### TSLA/SPY canaries (older, mostly Trigger-style with NoConf/Control variants)

| sid | Name | 6/01 | 6/02 | 6/03 | 6/04 | 6/05 | Thu→Fri |
|---|---|---|---|---|---|---|---|
| 263 | TSLA-CANARY-10s-NoConf | 65.4 | 68.7 | 75.0 | 81.1 | 63.6 | -17.5 |
| 267 | TSLA-CANARY-10s-LooseConf | 65.6 | 68.7 | 72.9 | 80.3 | 61.0 | -19.3 |
| 268 | SPY-CANARY-10s-NoConf | 71.6 | 77.7 | 73.7 | 85.9 | 70.3 | -15.6 |
| 270 | TEST-P2-utbotv4-10s-SPY | 15.3 | 77.7 | 73.4 | 85.9 | 70.3 | -15.6 |
| 273 | TEST-P2-utbotv4-10s-TSLA | 11.0 | 68.7 | 74.7 | 81.1 | 63.6 | -17.5 |

These NoConf/Control variants regressed 15-19 pts Thursday → Friday. **Less than gate-style's 22-57 pt drops but more than trigger-style's ~2 pt change.** These do have user-pack dependency on utv4 itself even though they're "NoConf."

---

## HOUR-BY-HOUR PATTERNS

### Top 25 hours cohort-wide (by # of strategies hitting "excellent" = ≥95% with ≥30 alerts)

| Hour (UTC) | Active strategies | Clean (≥90%) | Excellent (≥95%, ≥30 alerts) | Agg paired % | Total alerts |
|---|---|---|---|---|---|
| **2026-06-03T14** | 33 | 6 | **3** | 49.0% | 1,452 |
| **2026-06-04T16** | 36 | 6 | **2** | 71.8% | 1,510 |
| **2026-06-05T16** | 27 | 1 | **1** | 67.0% | 1,036 |
| 2026-06-04T14 | 42 | 9 | 0 | 64.9% | 1,784 |
| 2026-06-04T17 | 40 | 9 | 0 | 64.6% | 1,578 |
| 2026-06-02T18 | 29 | 5 | 0 | 59.1% | 1,179 |
| 2026-06-03T15 | 34 | 4 | 0 | 48.9% | 1,343 |
| 2026-06-05T14 | 31 | 4 | 0 | 61.2% | 1,343 |
| 2026-06-04T15 | 40 | 3 | 0 | 68.0% | 1,411 |
| 2026-06-03T17 | 37 | 2 | 0 | 64.6% | 1,090 |

**Observations:**

1. **Thursday 6/04 dominates the top 10 hours.** T14, T15, T16, T17 all in top 10 — 4 consecutive clean hours.
2. **Friday 6/05 has only 2 hours in top 10** (T14 at #8 and T16 at #3 — the latter only because some strategies hit excellent there).
3. **No hour cohort-wide hit >72% agg paired_pct.** The fleet average runs much lower than individual strategy peaks (sid 268 hit 97.8% on 6/04T16 in our earlier analysis).
4. **2026-06-05T16 stands out** as an unusual Friday hour — agg 67% with 1 excellent strategy. Worth a separate drill.

### Friday-specific hourly patterns for gates (the broken cohort)

Looking at 2026-06-05 hour-by-hour for gate strategies sid 277, 283, 299:

The gate-collapse hours align with: **the highest deploy density on Friday was 11:00-17:30 UTC (10 commits in 6.5 hours).** Gate strategies show worst phantom rates during this window, recovering somewhat in T19-T20 post-deploy-storm.

---

## LAYER 1 BAR FIDELITY — drift_uncorrected by day

This measures how often bars from the live WebSocket disagreed with REST verification (and weren't corrected).

| Date | Avg drift% | Max drift% | N strategies | Verdict |
|---|---|---|---|---|
| 2026-05-29 | 0.00% | 0.00% | 4 | Perfect |
| 2026-06-01 | 0.52% | 2.50% | 11 | Mostly clean |
| 2026-06-02 | 0.10% | 2.17% | 29 | Clean |
| 2026-06-03 | 0.05% | 0.71% | 39 | **Extremely clean** |
| **2026-06-04** | **2.28%** ⚠ | **9.73%** ⚠ | 40 | **22x jump** |
| 2026-06-05 | 1.64% | 4.78% | 38 | Elevated |

**The 6/04 drift jump is significant and persistent.** It's NOT explained by the #57E+F per-bar REST verification commit (`5d634ae` 2026-06-05 10:09) — that came AFTER the 6/04 jump.

The 6/04 commits most likely to have caused the drift jump:
- `e52aba2` 2026-06-04 09:29 — data_worker watchdog: guard STALE detection on is_market_window()
- `2fd0c10` 2026-06-04 14:08 — Streaming mode toggle DB-backed
- `4be824a` 2026-06-04 13:09 — Phase 2 backend: streaming toggle + async on-demand UAD jobs

**Most likely cause: `1dae0d1` (06-03 10:42) — watchdog phase 2 with HTTP timeouts + auto-restart on persistent stale.** This shipped 6/03 but its effect on bar fidelity would show as the streaming engine got auto-restarted more frequently throughout 6/04. Each restart creates a brief window of unverified bars before REST splice catches up.

---

## PRE-5/28 BASELINE (limited)

For honesty: we have data going back further for 3 strategies, but volumes are too low to draw conclusions.

**Sid 136 — full historical context:**
- 2025-09 to 2026-05 BT trades exist (low-volume strategy)
- 2026-05-22 first alert
- Current state: 12% (pre-UAD), 20% (post-UAD) — minimal alert volume

**Sid 174:**
- 0% paired throughout entire history
- Chronic config issue — appears never to have worked
- Volume: 253 alerts over 9 days, 0 BT trades

**Sid 194:**
- 4 alerts pre-5/28, 68 BT trades
- BT lane has data back to 2026-02-17
- Volume too low for hourly analysis

**Conclusion on pre-5/28:** the strategies we have older data for are either chronic-broken or too low-volume to be diagnostic. **The meaningful comparison window starts 6/01 when the canary cohort came fully online.**

---

## COMMIT TIMELINE CORRELATION

### Pre-window commits (effect lingers)

- `684265c` 2026-06-01 10:07 Phase 1 warmup-window replay
- `f006937` 2026-06-01 11:58 Phase 2 backtest snapshot user-pack codec — **major fix, big improvement**

### 2026-06-02 — codification + cleanup

- 15:54 `734e03e` GATE-DIAG logging (added then removed same day)
- 16:04 `ac26fd4` GATE-DIAG removed
- 16:06 `e3c3034` User_Pack_Roadmap doc

### 2026-06-03 — fixes that helped

- 10:35 `ee7f246` watchdog phase 1 (log only — safe)
- **10:42 `1dae0d1` watchdog phase 2 — HTTP timeouts + auto-restart ⚠** likely cause of drift jump
- 11:16 `9db405e` ema_pp_v3/v4 manifests: trigger_levels → trigger_levels_phase2 — **helps gates**
- 12:05 `0a1d158` user-pack trigger inference skip — **helps gates**
- 14:24 `e313e5d` opposite_signal sentinel resolution — **helps gates**
- 15:51 `69c3f7f` SwingStop fallback removal — neutral
- 18:50 `b2387ff` ralph_engine stamp resolved default live_model — operational

**6/03 was net positive (lots of small fixes), confirmed by the +17 pt improvement Tue → Wed.**

### 2026-06-04 — major refactor day

- 09:29 `e52aba2` watchdog STALE guard — operational
- 12:25 `abb6745` strategy_factory unified config — **major refactor**
- 12:28 `8c20122` pack_canary route through factory
- 12:33 `f061076` _backfill_strategy_configs — normalization migration
- 13:09 `4be824a` Phase 2 backend streaming toggle
- 13:38 `f7eec9b` Phase 2 Streamlit
- 14:00 `768e353` Phase 2 frontend Next.js admin
- 14:08 `2fd0c10` Streaming mode DB-backed toggle
- 15:27 `3322298` ALGO-APPEND Hi-Fi incremental
- 17:00 `7dba9ea` cross_exec_type_mismatch auto-classifier
- 17:33 `b87f059` #57 A/B/D bar_diagnostics schema

**6/04 was Thursday peak day. Despite many changes, the fleet hit its best performance of the week.** This is important — many of these changes were SHIPPED 6/04 and did NOT cause regression. The drift jump on 6/04 was the only signal.

### 2026-06-05 — the regression day

- 08:58 `c1cfb4b` #57C engine write hooks for bar_diagnostics
- 10:09 `5d634ae` **#57E+F per-bar REST verification (4-pass settle)** ⚠ candidate
- 11:31 `b5f40c7` **#57G live_corrected source rows** ⚠ candidate
- 11:38 `3c5861c` #57G K=1 fast-path
- 12:01 `0901baf` **#57H phantom_pre_correction reclassification** ⚠ candidate
- 12:10 `c256900` #57H bugfix
- 12:23 `3429dce` **#42 session filter** ⚠ candidate
- 15:25-17:00 Window Backfill (additive, WIP-banner protected)

**The gate collapse mostly happened DURING and AFTER the #57E/F/G/H series + #42 deploys.** These changes specifically alter:
- How REST verification interacts with live alerts (#57E/F/G — could affect what alerts get recorded and how)
- How phantoms get reclassified (#57H — could affect what we MEASURE as paired)
- BT session filtering (#42 — could affect BT trade generation)

**Most suspicious: #57H phantom_pre_correction.** If this misclassifies legitimate alerts as "would not have fired post-correction," they become phantoms in our measurement. Gate alerts (which depend on confluence values subject to REST correction) would be MORE affected than trigger alerts.

---

## UPDATED ROLLBACK RECOMMENDATION

### What the data supports

1. **Don't revert core-engine commits (`684265c`, `f006937`, `abb6745`, etc.).** Trigger-style stayed stable. Reverting these would erase major fixes.

2. **Don't revert 6/03 commits.** They were net-positive across the cohort.

3. **6/04 watchdog auto-restart MAY need tuning.** Drift jumped on 6/04 and stayed elevated. But this isn't a "revert" — it's a "tune restart frequency" task.

4. **6/05 #57 series + #42 are the primary suspects for gate-specific regression.** These need investigation, but reverting blindly could erase real fidelity improvements.

### Specific investigation steps before any revert

**Step 1: Check whether the gate regression PERSISTS post-UAD into Monday.**

The data showing gate collapse is from 6/05 alerts. Today the BT lane was rebuilt via fleet UAD. Monday's RTH will tell us whether:
- (a) Gate alerts still fire excessively on Monday → live engine has a permanent gate-evaluation regression → investigate #57H specifically
- (b) Gate alerts return to Thursday-quality on Monday → the Friday regression was deploy-churn specific, not commit-induced

**This is the answer to the rollback question.** Without Monday's data, any revert is premature.

**Step 2: If gate regression persists Monday, the investigation order is:**

1. **`0901baf` + `c256900` #57H phantom_pre_correction** — does removing this fix gate pair rates?
2. **`b5f40c7` + `3c5861c` #57G live_corrected** — does removing this fix gate pair rates?
3. **`5d634ae` #57E+F per-bar REST verification** — major change to verification flow
4. **`3429dce` #42 session filter** — affects BT side, could affect gates if BT under-produces
5. **`1dae0d1` watchdog phase 2 auto-restart** — operational tuning (reduce restart frequency, not full revert)

**Step 3: For each tier-1 revert, measure before/after on the gate cohort specifically.**

- Pick 3 representative gates: sid 277 (Bollinger), sid 283 (EMA Stack), sid 299 (SuperTrend)
- Measure pair rate over a 4-hour window
- Revert the candidate commit
- Wait 4 hours, measure again
- Compare specifically the gate cohort numbers

### What was IN MY EARLIER battle plan vs WHAT THIS NEW DATA SUGGESTS

Earlier I recommended Tier 1 reverts in chronological order: `3429dce`, `#57H`, `#57G`, `#57E+F`, then Phase 2 codec. **This new data SHARPENS that — focus the diagnostic specifically on gate strategies, not the overall fleet.** The fleet-wide regression is muted by trigger-style stability.

The earlier "do not revert Phase 2 codec" advice STILL HOLDS. Trigger-style strategies are healthy, which proves Phase 2 codec works for the engine's core. Reverting it would re-introduce the 95→3% phantom rate fix.

---

## SPECIFIC NEXT STEPS (in order)

### Monday 2026-06-09 (active Battle Plan window)

1. **Pre-RTH (~13:00 UTC):** Re-run `_fleet_hourly_may28_jun5.py` with date range updated to include 06-08 → 06-09. Save as `monday_pre_rth_baseline.json`.
2. **Monday RTH (13:30 → 20:00 UTC):** No worker deploys, no canary UAD. Allow full clean accumulation.
3. **Monday post-RTH:** Re-run analysis. Specifically compare:
   - **Trigger-style Monday RTH average vs Thursday 6/04 (49.3%)**
   - **Gate-style Monday RTH average vs Thursday 6/04 (37.8%)**
   - **Layer 1 drift_uncorrected vs 6/03 (0.05%) and 6/04 (2.28%)**

### Decision tree based on Monday results

| Monday outcome | Verdict | Action |
|---|---|---|
| Trigger ≥45%, Gate ≥35%, drift <0.5% | Engine fully recovered. No rollback needed. | Continue normal ops. |
| Trigger ≥45%, Gate <25%, drift <1% | Gate-specific bug remains. | Investigate #57H + #57G impact on gate engine first. |
| Trigger <40% across the board | New regression has appeared. | Investigate what changed after the Battle Plan window. |
| Drift >2% across the board | Watchdog is over-aggressive. | Tune watchdog restart frequency. |

### If Battle Plan window forces a code change pre-Monday

If anything urgent comes up that requires a worker-affecting deploy before Monday:
- **Acknowledge it breaks the test** — the diagnostic answer becomes "we don't know if it would have recovered."
- **Re-establish T0 = the new deploy timestamp** — restart the 48-hour clean-window timer.
- **Document the trade-off in this file.**

---

## REFERENCE TABLES — ARTIFACTS PRESERVED

All saved locally in `/tmp/fleet_hourly_may28_jun5/`:

| File | What it contains |
|---|---|
| `all_strategies.json` | Master JSON: every strategy × every hour with paired/phantom/missed/combined_pct/drift |
| `sid_<N>_hourly.csv` (43 files) | Per-strategy hourly drill — open any in Excel/VS Code |
| `per_day_per_strategy.csv` | Daily roll-up for cross-strategy comparison |
| `cohort_by_hour.csv` | Cohort aggregate per hour |
| `best_worst_hours.csv` | Top 100 hours ranked by clean-strategy count |

Reproducible: `cd src && ../.venv/bin/python _fleet_hourly_may28_jun5.py`
Date range and pair window configurable at top of script.

---

## APPENDIX: TOP 5 EXAMPLES OF EACH PATTERN

### Top 5 "trigger-style stayed healthy" examples (Thursday → Friday flat)

| sid | 6/04 | 6/05 | Pattern |
|---|---|---|---|
| 296 (Strat Assistant trigger) | 64.8 | 64.6 | Almost identical |
| 288 (RSI Zones 2 trigger) | 55.4 | 55.3 | Identical |
| 286 (MACD Line v2 trigger) | 41.8 | 41.6 | Identical |
| 280 (EMA PP v4 trigger) | 60.9 | 59.8 | -1.1 |
| 302 (UT Bot V4 trigger) | 57.2 | 55.0 | -2.2 |

### Top 5 "gate-style collapsed" examples

| sid | 6/04 | 6/05 | Drop |
|---|---|---|---|
| 283 (EMA Stack v2 gate) | 66.1 | 8.6 | **-57.5** |
| 279 (EMA PP v3 gate) | 51.5 | 12.1 | -39.4 |
| 281 (EMA PP v4 gate) | 51.5 | 12.1 | -39.4 |
| 299 (SuperTrend gate) | 59.4 | 21.4 | -38.0 |
| 293 (S/R Channels gate) | 48.9 | 13.4 | -35.5 |

### The exception: sid 301 (Swing 1-2-3 gate)

| sid | 6/04 | 6/05 | Pattern |
|---|---|---|---|
| 301 (Swing 1-2-3 gate) | 47.9 | 46.0 | -1.9 (flat — only gate that didn't regress) |

**Worth investigating: why did 301 escape the gate collapse?** If we can identify what makes 301 different from the other gates, we have a clue about the root cause. Possible factors:
- Swing 1-2-3 pack might not use REST-correction-sensitive confluence values
- Swing 1-2-3 might have simpler gate logic less affected by phantom_pre_correction
- Could be a coincidence (Friday had specific market conditions that interacted with each pack differently)

This is the single highest-value follow-up investigation. **If we understand why 301 stayed healthy, we understand why the other 13 gates broke.**
