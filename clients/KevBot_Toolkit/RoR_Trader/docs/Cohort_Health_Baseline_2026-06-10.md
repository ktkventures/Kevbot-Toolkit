# Cohort Health Baseline — 2026-06-10 (pre cache-cleanup)

_Generated 2026-06-10T21:37:56Z · pairing ±5s · alerts↔REST-backtest · last ~15min clipped (BT-lane lag)._

**Context:** baseline BEFORE deleting the SPY 10s flat gap-fill cache rows. These metrics pair recorded alerts vs REST backtest trades — the cache cleanup will NOT move them (display-only). The engine gap-fill fix `0d3affb` deployed ~2026-06-10T19:05Z; this window is almost entirely PRE-fix, so treat it as the buggy-period baseline to compare post-fix days against.

## Per-strategy (7-day window, ±5s)

| sid | strategy | alerts | bt | paired | phantom | missed | combined% | alert-pair% |
|----:|----------|------:|---:|------:|------:|-----:|--------:|-----------:|
| 303 | UT_BOT_V4 | 2227 | 1196 | 829 | 1398 | 353 | 32.1 | 37.2 |
| 277 | BOLLINGER | 1830 | 746 | 474 | 1356 | 272 | 22.5 | 25.9 |
| 293 | SR_CHANNELS | 2235 | 1432 | 815 | 1420 | 611 | 28.6 | 36.5 |
| 299 | SUPERTREND | 2616 | 1572 | 1058 | 1558 | 511 | 33.8 | 40.4 |
| 301 | SWING_123 | 3206 | 2374 | 1534 | 1672 | 827 | 38.0 | 47.8 |
| 267 | TSLA SWING_123 | 1455 | 1202 | 1002 | 453 | 200 | 60.5 | 68.9 |
| — | **COHORT** | 13569 | 8522 | 5712 | 7857 | 2774 | **35.0** | 42.1 |

## Cohort combined% by hour (most recent 24)

| hour (UTC) | paired | phantom | missed | combined% |
|-----------|------:|------:|-----:|--------:|
| 2026-06-09T14 | 84 | 190 | 8 | 29.8 |
| 2026-06-09T15 | 74 | 249 | 14 | 22.0 |
| 2026-06-09T16 | 75 | 236 | 26 | 22.3 |
| 2026-06-09T17 | 126 | 122 | 20 | 47.0 |
| 2026-06-09T18 | 184 | 122 | 14 | 57.5 |
| 2026-06-09T19 | 72 | 198 | 54 | 22.2 |
| 2026-06-09T20 | 83 | 98 | 56 | 35.0 |
| 2026-06-09T21 | 21 | 81 | 25 | 16.5 |
| 2026-06-09T22 | 28 | 80 | 33 | 19.9 |
| 2026-06-09T23 | 37 | 59 | 30 | 29.4 |
| 2026-06-10T08 | 21 | 37 | 43 | 20.8 |
| 2026-06-10T09 | 29 | 70 | 26 | 23.2 |
| 2026-06-10T10 | 38 | 94 | 40 | 22.1 |
| 2026-06-10T11 | 53 | 66 | 16 | 39.3 |
| 2026-06-10T12 | 138 | 71 | 64 | 50.5 |
| 2026-06-10T13 | 134 | 83 | 42 | 51.7 |
| 2026-06-10T14 | 141 | 135 | 26 | 46.7 |
| 2026-06-10T15 | 64 | 175 | 15 | 25.2 |
| 2026-06-10T16 | 89 | 240 | 17 | 25.7 |
| 2026-06-10T17 | 45 | 244 | 25 | 14.3 |
| 2026-06-10T18 | 36 | 144 | 26 | 17.5 |
| 2026-06-10T19 | 36 | 118 | 16 | 21.2 |
| 2026-06-10T20 | 27 | 188 | 2 | 12.4 |
| 2026-06-10T21 | 14 | 58 | 1 | 19.2 |

## By-deploy note

deploy history covers 78 commits; most-recent tracked = `031229f` (feat: Strategy Health 'By Hour' tab — cross-hour K) — window to now. **Today's deploys (c0116ea / a9c59fe / 0d3affb) are NOT in deploy_history.json yet**, so the gap-fill fix isn't its own bucket. Update deploy_history to split pre/post-fix.

## Caveats

- ±5s pairing; last ~15min clipped (BT-lane lag tail).
- Cache flat-cleanup will NOT change these (alerts/BT not cache-derived); the engine deploy `0d3affb` is what should improve post-fix days.
- Re-run this same script after a full post-fix day to compare.

