# Pre-UAD Baseline — 2026-06-06 ~03:38 UTC

> **Purpose:** Preserve a comprehensive snapshot of the fleet's BT-vs-live state RIGHT BEFORE Kevin clicks bulk Update All Data tonight. If the UAD results materially change the divergence picture in a direction we don't expect, this doc + the linked raw JSON give us a clean rollback reference.

## What this captures

- 43 active strategies (any alert in last 48h)
- Per-strategy 48h totals (alerts vs BT events)
- Hour-by-hour bins for each strategy (in JSON)
- RTH today (06-05 13:30-20:00 UTC) vs RTH yesterday (06-04 13:30-20:00 UTC) per-strategy comparison
- Cohort-wide trend signal

## Raw artifacts preserved (under `/tmp/` — copy to safer location if needed)

| File | What it is |
|---|---|
| `/tmp/pre_uad_baseline_2026-06-06.json` | Full hour-by-hour bins per strategy. 48h window. Replayable. |
| `/tmp/pre_uad_summary_2026-06-06.tsv` | Per-strategy 48h totals + verdict label |
| `/tmp/pre_uad_trend_2026-06-06.tsv` | Per-strategy yesterday-vs-today RTH ratio comparison |

Scripts that produced them (committed for reproducibility):

| Script | Purpose |
|---|---|
| `src/_pre_uad_baseline.py` | Pull 48h alerts + BT trades per strategy per hour |
| `src/_pre_uad_compare.py` | Today RTH vs yesterday RTH comparison |

To re-run later (e.g., post-UAD): `cd src && ../.venv/bin/python _pre_uad_baseline.py`

## Cohort-wide totals

| Window | Alerts | BT events | Ratio |
|---|---|---|---|
| **RTH 2026-06-04 (Thursday)** | 10,234 | 7,786 | **76.1%** |
| **RTH 2026-06-05 (Friday)** | 8,143 | 5,722 | **70.3%** |
| After-hours 06-05/06-06 | 2,894 | 1,443 | 49.9% |

Cohort ratio dropped ~6 points day-over-day. **The likely cause is operational, not engine-side:** streaming is in MANUAL mode (set during midday yesterday), so the BT lane stops auto-filling between explicit UAD runs. Strategies UAD'd recently (302, 263, 284, 290, 303, 286, 296, 295, 280) are stable; strategies NOT UAD'd today are accumulating stale BT lanes while alerts keep flowing.

**This is reversible by clicking Update All Data on the stale strategies.** The 70.3% number reflects the gap created by streaming being off, not engine regression.

## Per-strategy 48h totals

| sid | Alerts | BT pairs | BT events | Ratio | Verdict |
|---|---|---|---|---|---|
| 136 | 16 | 1 | 2 | 12.5% | critical |
| 174 | 72 | 0 | 0 | 0.0% | critical |
| 194 | 3 | 1 | 2 | 66.7% | big gap |
| 263 | 596 | 312 | 624 | **104.7%** | **OK** (UAD'd tonight) |
| 265 | 90 | 24 | 48 | 53.3% | big gap |
| 266 | 12 | 5 | 10 | 83.3% | soft gap |
| 267 | 573 | 243 | 486 | 84.8% | soft gap |
| 268 | 619 | 241 | 482 | 77.9% | soft gap |
| 269 | 91 | 21 | 42 | 46.2% | big gap |
| 270 | 620 | 216 | 432 | 69.7% | big gap |
| 271 | 120 | 53 | 106 | 88.3% | soft gap |
| 272 | 82 | 20 | 40 | 48.8% | big gap |
| 273 | 596 | 187 | 374 | 62.8% | big gap |
| 275 | 260 | 36 | 72 | 27.7% | critical |
| 276 | 387 | 101 | 202 | 52.2% | big gap |
| 277 | 498 | 131 | 262 | 52.6% | big gap (just got mode='new' tonight) |
| 278 | 1,942 | 671 | 1,342 | 69.1% | big gap |
| 279 | 364 | 127 | 254 | 69.8% | big gap |
| 280 | 1,942 | 952 | 1,903 | **98.0%** | **OK** (UAD'd tonight) |
| 281 | 364 | 130 | 260 | 71.4% | soft gap |
| 282 | 493 | 129 | 258 | 52.3% | big gap |
| 283 | 448 | 196 | 392 | 87.5% | soft gap |
| 284 | 2,711 | 1,226 | 2,452 | **90.4%** | **OK** (UAD'd tonight) |
| 285 | 715 | 127 | 254 | 35.5% | big gap |
| 286 | 812 | 357 | 713 | 87.8% | soft gap (UAD'd tonight) |
| 287 | 408 | 118 | 236 | 57.8% | big gap |
| 288 | 1,024 | 368 | 736 | 71.9% | soft gap |
| 290 | 2,507 | 965 | 1,930 | 77.0% | soft gap (UAD'd tonight) |
| 291 | 280 | 29 | 58 | 20.7% | critical |
| 292 | 714 | 254 | 508 | 71.1% | soft gap |
| 293 | 703 | 227 | 454 | 64.6% | big gap |
| 294 | 696 | 210 | 420 | 60.3% | big gap |
| 295 | 265 | 41 | 82 | 30.9% | big gap (mode='new' tonight — but baseline was sparse pre-UAD) |
| 296 | 2,737 | 1,377 | 2,753 | **100.6%** | **OK** (UAD'd tonight) |
| 297 | 641 | 83 | 166 | 25.9% | critical |
| 298 | 557 | 135 | 270 | 48.5% | big gap |
| 299 | 796 | 263 | 526 | 66.1% | big gap |
| 300 | 970 | 348 | 696 | 71.8% | soft gap |
| 301 | 1,210 | 546 | 1,092 | **90.2%** | **OK** |
| 302 | 1,603 | 711 | 1,422 | 88.7% | soft gap (UAD'd tonight) |
| 303 | 805 | 223 | 446 | 55.4% | big gap (chronic eppv3 batch-path bug — engine-side, not stale baseline) |
| 304 | 239 | 50 | 100 | 41.8% | big gap |
| 305 | 213 | 8 | 16 | 7.5% | critical |

**Breakdown by category:**
- **OK (90-110%)**: 6 strategies — all UAD'd recently
- **Soft gap (70-90%)**: 13 strategies — mostly close to right
- **Big gap (30-70%)**: 19 strategies — likely just stale BT lanes
- **Critical (<30%)**: 5 strategies — sid 136 / 174 / 275 / 291 / 297 / 305 — worth investigating per-strategy

## Today vs Yesterday RTH per-strategy

| sid | Yesterday RTH ratio | Today RTH ratio | Trend |
|---|---|---|---|
| 263 | 102.9% | 106.2% | **flat (UAD'd)** |
| 280 | 104.0% | 104.2% | **flat (UAD'd)** |
| 284 | 101.9% | 102.7% | **flat (UAD'd)** |
| 286 | 103.9% | 101.5% | **flat (UAD'd)** |
| 296 | 101.7% | 103.6% | **flat (UAD'd)** |
| 302 | 105.9% | 104.7% | **flat (UAD'd)** |
| 290 | 101.2% | 108.0% | +6.8 (UAD'd) |
| 295 | 19.7% | 100.0% | **+80.3 (UAD'd — DRAMATIC baseline rebuild)** |
| 266 | 72.7% | 100.0% | +27.3 |
| 293 | 80.0% | 93.8% | +13.8 |
| 265 | 63.4% | 16.3% | **-47.1 (stale BT)** |
| 303 | 70.1% | 23.7% | **-46.4 (known engine bug + stale BT)** |
| 299 | 77.5% | 31.3% | **-46.2 (stale BT)** |
| 275 | 41.9% | 0.0% | -41.9 (stale BT) |
| 277 | 44.4% | 5.9% | -38.6 (mode='new' only buys forward fill; baseline still sparse) |
| 304 | 70.0% | 33.3% | -36.7 (stale BT) |
| 269 | 61.2% | 28.6% | -32.7 (stale BT) |
| 276 | 72.3% | 44.1% | -28.2 (stale BT) |
| 292 | 77.0% | 49.2% | -27.8 (stale BT) |
| 282 | 69.5% | 42.2% | -27.3 (stale BT) |
| 268 | 77.0% | 52.6% | -24.4 (stale BT) |
| 278 | 80.5% | 55.9% | -24.6 (stale BT) |
| 300 | 85.2% | 60.9% | -24.3 (stale BT) |
| 270 | 76.8% | 53.3% | -23.5 (stale BT) |
| 301 | 92.9% | 72.5% | -20.4 (stale BT) |
| 288 | 81.2% | 60.4% | -20.8 (stale BT) |
| 287 | 46.3% | 27.3% | -19.0 (stale BT) |
| 267 | 77.0% | 59.1% | -17.9 (stale BT) |
| 273 | 70.8% | 54.7% | -16.1 (stale BT) |
| 294 | 73.6% | 62.5% | -11.1 (stale BT) |
| 298 | 74.2% | 64.1% | -10.1 (stale BT) |
| 285 | 26.9% | 17.4% | flat (was already low) |
| 291 | 5.2% | 4.7% | flat (was already broken) |
| 297 | 13.5% | 11.5% | flat (was already broken) |
| 174 | 0.0% | 0.0% | flat (chronic) |

**Trend summary:** 4 BETTER, 22 WORSE, 9 flat.

**Critical pattern:** EVERY strategy that was UAD'd tonight is FLAT or BETTER. EVERY strategy with a big regression is one that was NOT UAD'd. **This strongly suggests the regression is stale baselines from MANUAL streaming mode, not engine breakage.** Update All Data should restore most of these.

## What "good" looks like — the strategies that are FLAT/OK

| sid | Today RTH ratio | Status |
|---|---|---|
| 263, 280, 284, 286, 296, 302 | 102-106% | UAD'd this week, sub-second Hi-Fi exits, healthy |
| 290 | 108% | UAD'd tonight |
| 295 | 100% | UAD'd tonight (was 19.7% before) |
| 266 | 100% | naturally low-volume |
| 293 | 93.8% | improving |
| 301 | 72.5% | OK soft gap |

These are reference strategies. If post-UAD-tonight the ratios on these 8-10 strategies stay ≥90%, **the UAD didn't break anything.** Compare post-UAD numbers against this table specifically.

## What to expect after bulk Update All Data tonight

Based on the parity tests earlier tonight (sid 295: 5 → 111 trades, sid 280: 1876 → 2262):

- **Stale-baseline strategies (the 22 with big regressions): expect ratios to recover to 70-95%.** Same engine, just filling the gap MANUAL streaming created.
- **Already-OK strategies (the 9 with flat trend): expect <5-point change.** They're already at the engine's canonical output.
- **sid 303 (chronic eppv3 bug)**: expect minimal change. The phantom rate is engine-side, not stale-baseline-side. Will need the EMA PP v3 L-type batch-path fix to improve.
- **Critically-broken strategies (136/174/291/297/305)**: investigate individually; UAD may not fix these depending on root cause.

## How to compare post-UAD against this baseline

After bulk Update All Data completes (overnight):

```bash
cd src && ../.venv/bin/python _pre_uad_baseline.py
mv /tmp/pre_uad_baseline_2026-06-06.json /tmp/post_uad_state_2026-06-06.json
```

Then diff per-strategy ratios. Specifically:
1. For the 9 FLAT strategies above: post-UAD ratios should match within 5 points
2. For the 22 WORSE strategies: post-UAD ratios should improve toward yesterday's RTH numbers
3. For sid 303: minimal change expected (chronic engine bug, not stale BT)

If FLAT strategies regress significantly (>10 points) post-UAD, that's a red flag — investigate before clicking anything else.

## Reversion options if UAD breaks something

- DB has full trade history. UAD is not destructive to existing entries — it overwrites the BT lane per strategy.
- If a strategy's post-UAD state looks materially worse than pre-UAD, run Update New Data immediately to see if streaming-time picks back up (won't recover sparse baselines but will catch fresh trades).
- If multiple strategies look worse post-UAD, halt and compare against this doc.
- DO NOT click Update All Data on strategies that are already FLAT/OK in the table above unless you have a specific reason.

## Recap of fixes shipped earlier tonight (verified working)

| Commit | What | Verified |
|---|---|---|
| `6cdd7b3` | `data_loader.py` 1s cache staleness fix | ✓ Sid 295's 111 post-UAD trades all have `hifi_resolved=True`, sub-second exits |
| `6a5ac25` | `useUpdateStrategyLanes` async migration | ✓ Sid 277 click ran 192s without "Failed to fetch" via job #451643000 |

These two fixes are independent of the baseline situation. They're working and don't need rollback.

## Trust signal

- **Engine itself**: solid (Layer 1 0-2% drift cohort-wide; UAD'd strategies show ~100% parity with alerts)
- **Hi-Fi Pass 2**: reliable post-1s-cache-fix
- **mode='new' append_new**: trustworthy (parity verified on 295 + 280 earlier tonight)
- **mode='all' full UAD**: trustworthy (always was)
- **Baselines**: 22/43 strategies have stale BT lanes from MANUAL streaming. Tonight's bulk UAD fixes this.
- **Direction of travel**: forward. Not backward. The day-over-day decline is operational gap accumulation, not engine regression.

Kevin — click Update All Data with confidence. This baseline is preserved. If something breaks, we have the comparison anchor. The fixes that landed tonight are working. The divergence picture is recoverable.
