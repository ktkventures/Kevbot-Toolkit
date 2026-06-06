# Hourly Cohort Tables — 2026-06-06

> Same data as `Strategy_Hourly_Tables_2026-06-06.md` but pivoted: one table per HOUR timestamp, rows are strategies. Lets you see which strategies were cleanest in any given hour cohort-wide.
> Pair window: ±5s. Combined % = paired / (paired + phantom + missed). Alert-pair % = paired / (paired + phantom). Rank is per-hour cohort-wide: rank 1 = the cleanest strategy in this hour by Combined %. Only strategies with alerts ≥ 1 AND BT events ≥ 1 in that hour are ranked.

## Index of hours

- [2026-05-28T13 UTC](#hour-2026-05-28t13-utc) — 9 strategies with activity
- [2026-05-28T14 UTC](#hour-2026-05-28t14-utc) — 11 strategies with activity
- [2026-05-28T15 UTC](#hour-2026-05-28t15-utc) — 9 strategies with activity
- [2026-05-28T16 UTC](#hour-2026-05-28t16-utc) — 11 strategies with activity
- [2026-05-28T17 UTC](#hour-2026-05-28t17-utc) — 12 strategies with activity
- [2026-05-28T18 UTC](#hour-2026-05-28t18-utc) — 11 strategies with activity
- [2026-05-28T19 UTC](#hour-2026-05-28t19-utc) — 11 strategies with activity
- [2026-05-28T20 UTC](#hour-2026-05-28t20-utc) — 3 strategies with activity
- [2026-05-29T13 UTC](#hour-2026-05-29t13-utc) — 8 strategies with activity
- [2026-05-29T14 UTC](#hour-2026-05-29t14-utc) — 10 strategies with activity
- [2026-05-29T15 UTC](#hour-2026-05-29t15-utc) — 10 strategies with activity
- [2026-05-29T16 UTC](#hour-2026-05-29t16-utc) — 13 strategies with activity
- [2026-05-29T17 UTC](#hour-2026-05-29t17-utc) — 13 strategies with activity
- [2026-05-29T18 UTC](#hour-2026-05-29t18-utc) — 10 strategies with activity
- [2026-05-29T19 UTC](#hour-2026-05-29t19-utc) — 10 strategies with activity
- [2026-05-30T13 UTC](#hour-2026-05-30t13-utc) — 1 strategies with activity
- [2026-06-01T08 UTC](#hour-2026-06-01t08-utc) — 27 strategies with activity
- [2026-06-01T09 UTC](#hour-2026-06-01t09-utc) — 27 strategies with activity
- [2026-06-01T10 UTC](#hour-2026-06-01t10-utc) — 29 strategies with activity
- [2026-06-01T11 UTC](#hour-2026-06-01t11-utc) — 29 strategies with activity
- [2026-06-01T12 UTC](#hour-2026-06-01t12-utc) — 23 strategies with activity
- [2026-06-01T13 UTC](#hour-2026-06-01t13-utc) — 35 strategies with activity
- [2026-06-01T14 UTC](#hour-2026-06-01t14-utc) — 31 strategies with activity
- [2026-06-01T15 UTC](#hour-2026-06-01t15-utc) — 40 strategies with activity
- [2026-06-01T16 UTC](#hour-2026-06-01t16-utc) — 40 strategies with activity
- [2026-06-01T17 UTC](#hour-2026-06-01t17-utc) — 39 strategies with activity
- [2026-06-01T18 UTC](#hour-2026-06-01t18-utc) — 37 strategies with activity
- [2026-06-01T19 UTC](#hour-2026-06-01t19-utc) — 34 strategies with activity
- [2026-06-01T20 UTC](#hour-2026-06-01t20-utc) — 26 strategies with activity
- [2026-06-01T21 UTC](#hour-2026-06-01t21-utc) — 19 strategies with activity
- [2026-06-01T22 UTC](#hour-2026-06-01t22-utc) — 23 strategies with activity
- [2026-06-01T23 UTC](#hour-2026-06-01t23-utc) — 21 strategies with activity
- [2026-06-02T00 UTC](#hour-2026-06-02t00-utc) — 5 strategies with activity
- [2026-06-02T08 UTC](#hour-2026-06-02t08-utc) — 29 strategies with activity
- [2026-06-02T09 UTC](#hour-2026-06-02t09-utc) — 25 strategies with activity
- [2026-06-02T10 UTC](#hour-2026-06-02t10-utc) — 20 strategies with activity
- [2026-06-02T11 UTC](#hour-2026-06-02t11-utc) — 24 strategies with activity
- [2026-06-02T12 UTC](#hour-2026-06-02t12-utc) — 28 strategies with activity
- [2026-06-02T13 UTC](#hour-2026-06-02t13-utc) — 41 strategies with activity
- [2026-06-02T14 UTC](#hour-2026-06-02t14-utc) — 42 strategies with activity
- [2026-06-02T15 UTC](#hour-2026-06-02t15-utc) — 38 strategies with activity
- [2026-06-02T16 UTC](#hour-2026-06-02t16-utc) — 38 strategies with activity
- [2026-06-02T17 UTC](#hour-2026-06-02t17-utc) — 40 strategies with activity
- [2026-06-02T18 UTC](#hour-2026-06-02t18-utc) — 35 strategies with activity
- [2026-06-02T19 UTC](#hour-2026-06-02t19-utc) — 39 strategies with activity
- [2026-06-02T20 UTC](#hour-2026-06-02t20-utc) — 33 strategies with activity
- [2026-06-02T21 UTC](#hour-2026-06-02t21-utc) — 27 strategies with activity
- [2026-06-02T22 UTC](#hour-2026-06-02t22-utc) — 29 strategies with activity
- [2026-06-02T23 UTC](#hour-2026-06-02t23-utc) — 25 strategies with activity
- [2026-06-03T08 UTC](#hour-2026-06-03t08-utc) — 27 strategies with activity
- [2026-06-03T09 UTC](#hour-2026-06-03t09-utc) — 24 strategies with activity
- [2026-06-03T10 UTC](#hour-2026-06-03t10-utc) — 24 strategies with activity
- [2026-06-03T11 UTC](#hour-2026-06-03t11-utc) — 28 strategies with activity
- [2026-06-03T12 UTC](#hour-2026-06-03t12-utc) — 25 strategies with activity
- [2026-06-03T13 UTC](#hour-2026-06-03t13-utc) — 37 strategies with activity
- [2026-06-03T14 UTC](#hour-2026-06-03t14-utc) — 35 strategies with activity
- [2026-06-03T15 UTC](#hour-2026-06-03t15-utc) — 34 strategies with activity
- [2026-06-03T16 UTC](#hour-2026-06-03t16-utc) — 33 strategies with activity
- [2026-06-03T17 UTC](#hour-2026-06-03t17-utc) — 37 strategies with activity
- [2026-06-03T18 UTC](#hour-2026-06-03t18-utc) — 40 strategies with activity
- [2026-06-03T19 UTC](#hour-2026-06-03t19-utc) — 40 strategies with activity
- [2026-06-03T20 UTC](#hour-2026-06-03t20-utc) — 32 strategies with activity
- [2026-06-03T21 UTC](#hour-2026-06-03t21-utc) — 21 strategies with activity
- [2026-06-03T22 UTC](#hour-2026-06-03t22-utc) — 25 strategies with activity
- [2026-06-03T23 UTC](#hour-2026-06-03t23-utc) — 25 strategies with activity
- [2026-06-04T08 UTC](#hour-2026-06-04t08-utc) — 28 strategies with activity
- [2026-06-04T09 UTC](#hour-2026-06-04t09-utc) — 27 strategies with activity
- [2026-06-04T10 UTC](#hour-2026-06-04t10-utc) — 26 strategies with activity
- [2026-06-04T11 UTC](#hour-2026-06-04t11-utc) — 25 strategies with activity
- [2026-06-04T12 UTC](#hour-2026-06-04t12-utc) — 28 strategies with activity
- [2026-06-04T13 UTC](#hour-2026-06-04t13-utc) — 40 strategies with activity
- [2026-06-04T14 UTC](#hour-2026-06-04t14-utc) — 42 strategies with activity
- [2026-06-04T15 UTC](#hour-2026-06-04t15-utc) — 40 strategies with activity
- [2026-06-04T16 UTC](#hour-2026-06-04t16-utc) — 36 strategies with activity
- [2026-06-04T17 UTC](#hour-2026-06-04t17-utc) — 40 strategies with activity
- [2026-06-04T18 UTC](#hour-2026-06-04t18-utc) — 41 strategies with activity
- [2026-06-04T19 UTC](#hour-2026-06-04t19-utc) — 39 strategies with activity
- [2026-06-04T20 UTC](#hour-2026-06-04t20-utc) — 23 strategies with activity
- [2026-06-04T21 UTC](#hour-2026-06-04t21-utc) — 21 strategies with activity
- [2026-06-04T22 UTC](#hour-2026-06-04t22-utc) — 25 strategies with activity
- [2026-06-04T23 UTC](#hour-2026-06-04t23-utc) — 23 strategies with activity
- [2026-06-05T00 UTC](#hour-2026-06-05t00-utc) — 2 strategies with activity
- [2026-06-05T08 UTC](#hour-2026-06-05t08-utc) — 25 strategies with activity
- [2026-06-05T09 UTC](#hour-2026-06-05t09-utc) — 26 strategies with activity
- [2026-06-05T10 UTC](#hour-2026-06-05t10-utc) — 25 strategies with activity
- [2026-06-05T11 UTC](#hour-2026-06-05t11-utc) — 28 strategies with activity
- [2026-06-05T12 UTC](#hour-2026-06-05t12-utc) — 22 strategies with activity
- [2026-06-05T13 UTC](#hour-2026-06-05t13-utc) — 32 strategies with activity
- [2026-06-05T14 UTC](#hour-2026-06-05t14-utc) — 33 strategies with activity
- [2026-06-05T15 UTC](#hour-2026-06-05t15-utc) — 29 strategies with activity
- [2026-06-05T16 UTC](#hour-2026-06-05t16-utc) — 27 strategies with activity
- [2026-06-05T17 UTC](#hour-2026-06-05t17-utc) — 32 strategies with activity
- [2026-06-05T18 UTC](#hour-2026-06-05t18-utc) — 30 strategies with activity
- [2026-06-05T19 UTC](#hour-2026-06-05t19-utc) — 32 strategies with activity
- [2026-06-05T20 UTC](#hour-2026-06-05t20-utc) — 27 strategies with activity
- [2026-06-05T21 UTC](#hour-2026-06-05t21-utc) — 24 strategies with activity
- [2026-06-05T22 UTC](#hour-2026-06-05t22-utc) — 27 strategies with activity
- [2026-06-05T23 UTC](#hour-2026-06-05t23-utc) — 28 strategies with activity

---

## Hour 2026-05-28T13 UTC

_Strategies with activity this hour: 9; ranked (alerts≥1, BT≥1): 1_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 1/1 |
| 174 | TSLA LONG 1Min Mass #2 | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |

---

## Hour 2026-05-28T14 UTC

_Strategies with activity this hour: 11; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |

---

## Hour 2026-05-28T15 UTC

_Strategies with activity this hour: 9; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 51 | 0 | 0 | 51 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 51 | 0 | 0 | 51 | 0.0% | 0.0% | — |

---

## Hour 2026-05-28T16 UTC

_Strategies with activity this hour: 11; ranked (alerts≥1, BT≥1): 1_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 1 | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 1/1 |
| 265 | TSLA-CANARY-1m-Control | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 56 | 0 | 0 | 56 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |

---

## Hour 2026-05-28T17 UTC

_Strategies with activity this hour: 12; ranked (alerts≥1, BT≥1): 2_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 4 | 2 | 0 | 2 | 50.0% | 100.0% | 1/2 |
| 174 | TSLA LONG 1Min Mass #2 | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 1 | 3 | 1 | 0 | 2 | 33.3% | 100.0% | 2/2 |
| 265 | TSLA-CANARY-1m-Control | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |

---

## Hour 2026-05-28T18 UTC

_Strategies with activity this hour: 11; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |

---

## Hour 2026-05-28T19 UTC

_Strategies with activity this hour: 11; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |

---

## Hour 2026-05-28T20 UTC

_Strategies with activity this hour: 3; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 268 | SPY-CANARY-10s-NoConf | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |

---

## Hour 2026-05-29T13 UTC

_Strategies with activity this hour: 8; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 265 | TSLA-CANARY-1m-Control | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |

---

## Hour 2026-05-29T14 UTC

_Strategies with activity this hour: 10; ranked (alerts≥1, BT≥1): 1_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 4 | 4 | 2 | 2 | 2 | 33.3% | 50.0% | 1/1 |
| 174 | TSLA LONG 1Min Mass #2 | 7 | 0 | 0 | 7 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 47 | 0 | 0 | 47 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 62 | 0 | 0 | 62 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 62 | 0 | 0 | 62 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 47 | 0 | 0 | 47 | 0.0% | 0.0% | — |

---

## Hour 2026-05-29T15 UTC

_Strategies with activity this hour: 10; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 44 | 0 | 0 | 44 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 47 | 0 | 0 | 47 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 47 | 0 | 0 | 47 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 35 | 0 | 0 | 35 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 44 | 0 | 0 | 44 | 0.0% | 0.0% | — |

---

## Hour 2026-05-29T16 UTC

_Strategies with activity this hour: 13; ranked (alerts≥1, BT≥1): 2_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 4 | 2 | 0 | 4 | 2 | 0.0% | 0.0% | 2/2 |
| 263 | TSLA-CANARY-10s-NoConf | 24 | 0 | 0 | 24 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 5 | 8 | 3 | 2 | 5 | 30.0% | 60.0% | 1/2 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 34 | 0 | 0 | 34 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 34 | 0 | 0 | 34 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 34 | 0 | 0 | 34 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |

---

## Hour 2026-05-29T17 UTC

_Strategies with activity this hour: 13; ranked (alerts≥1, BT≥1): 3_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 36 | 0 | 0 | 36 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 4 | 7 | 4 | 0 | 3 | 57.1% | 100.0% | 1/3 |
| 266 | TSLA-CANARY-5m-Control | 2 | 1 | 1 | 1 | 0 | 50.0% | 50.0% | 2/3 |
| 267 | TSLA-CANARY-10s-LooseConf | 13 | 43 | 11 | 2 | 32 | 24.4% | 84.6% | 3/3 |
| 268 | SPY-CANARY-10s-NoConf | 0 | 55 | 0 | 0 | 55 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 55 | 0 | 0 | 55 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 37 | 0 | 0 | 37 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |

---

## Hour 2026-05-29T18 UTC

_Strategies with activity this hour: 10; ranked (alerts≥1, BT≥1): 5_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 40 | 0 | 0 | 40 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 8 | 8 | 8 | 0 | 0 | 100.0% | 100.0% | 1/5 |
| 266 | TSLA-CANARY-5m-Control | 2 | 3 | 1 | 1 | 2 | 25.0% | 50.0% | 5/5 |
| 267 | TSLA-CANARY-10s-LooseConf | 40 | 40 | 36 | 4 | 4 | 81.8% | 90.0% | 3/5 |
| 268 | SPY-CANARY-10s-NoConf | 55 | 61 | 54 | 1 | 7 | 87.1% | 98.2% | 2/5 |
| 269 | SPY-CANARY-1m-Control | 4 | 8 | 3 | 1 | 5 | 33.3% | 75.0% | 4/5 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |

---

## Hour 2026-05-29T19 UTC

_Strategies with activity this hour: 10; ranked (alerts≥1, BT≥1): 4_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 10 | 0 | 0 | 10 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 48 | 0 | 0 | 48 | 0 | 0.0% | 0.0% | — |
| 265 | TSLA-CANARY-1m-Control | 10 | 13 | 6 | 4 | 7 | 35.3% | 60.0% | 4/4 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 48 | 48 | 39 | 9 | 9 | 68.4% | 81.2% | 3/4 |
| 268 | SPY-CANARY-10s-NoConf | 47 | 57 | 44 | 3 | 13 | 73.3% | 93.6% | 2/4 |
| 269 | SPY-CANARY-1m-Control | 6 | 6 | 6 | 0 | 0 | 100.0% | 100.0% | 1/4 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |

---

## Hour 2026-05-30T13 UTC

_Strategies with activity this hour: 1; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T08 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 54 | 0 | 0 | 54 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 31 | 0 | 0 | 31 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 65 | 0 | 0 | 65 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T09 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 31 | 0 | 0 | 31 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 36 | 0 | 0 | 36 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 13 | 0 | 0 | 13 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 13 | 0 | 0 | 13 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T10 UTC

_Strategies with activity this hour: 29; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T11 UTC

_Strategies with activity this hour: 29; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 53 | 0 | 0 | 53 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 53 | 0 | 0 | 53 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 28 | 0 | 0 | 28 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 69 | 0 | 0 | 69 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 35 | 0 | 0 | 35 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 45 | 0 | 0 | 45 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 83 | 0 | 0 | 83 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T12 UTC

_Strategies with activity this hour: 23; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 54 | 0 | 0 | 54 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 54 | 0 | 0 | 54 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 56 | 0 | 0 | 56 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 58 | 0 | 0 | 58 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 78 | 0 | 0 | 78 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 50 | 0 | 0 | 50 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 50 | 0 | 0 | 50 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T13 UTC

_Strategies with activity this hour: 35; ranked (alerts≥1, BT≥1): 5_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 24 | 24 | 14 | 10 | 10 | 41.2% | 58.3% | 2/5 |
| 265 | TSLA-CANARY-1m-Control | 2 | 2 | 0 | 2 | 2 | 0.0% | 0.0% | 5/5 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 24 | 24 | 14 | 10 | 10 | 41.2% | 58.3% | 3/5 |
| 268 | SPY-CANARY-10s-NoConf | 22 | 32 | 17 | 5 | 15 | 45.9% | 77.3% | 1/5 |
| 269 | SPY-CANARY-1m-Control | 6 | 4 | 1 | 5 | 3 | 11.1% | 16.7% | 4/5 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 70 | 0 | 0 | 70 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 70 | 0 | 0 | 70 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 101 | 0 | 0 | 101 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 34 | 0 | 0 | 34 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 73 | 0 | 0 | 73 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 111 | 0 | 0 | 111 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 47 | 0 | 0 | 47 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T14 UTC

_Strategies with activity this hour: 31; ranked (alerts≥1, BT≥1): 5_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 45 | 45 | 38 | 7 | 7 | 73.1% | 84.4% | 3/5 |
| 265 | TSLA-CANARY-1m-Control | 8 | 8 | 7 | 1 | 1 | 77.8% | 87.5% | 2/5 |
| 267 | TSLA-CANARY-10s-LooseConf | 45 | 45 | 38 | 7 | 7 | 73.1% | 84.4% | 4/5 |
| 268 | SPY-CANARY-10s-NoConf | 52 | 52 | 46 | 6 | 6 | 79.3% | 88.5% | 1/5 |
| 269 | SPY-CANARY-1m-Control | 10 | 12 | 9 | 1 | 3 | 69.2% | 90.0% | 5/5 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 45 | 0 | 0 | 45 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 92 | 0 | 0 | 92 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 92 | 0 | 0 | 92 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 95 | 0 | 0 | 95 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 35 | 0 | 0 | 35 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 63 | 0 | 0 | 63 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 74 | 0 | 0 | 74 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 59 | 0 | 0 | 59 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 131 | 0 | 0 | 131 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 54 | 0 | 0 | 54 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T15 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 5_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 44 | 47 | 39 | 5 | 8 | 75.0% | 88.6% | 2/5 |
| 265 | TSLA-CANARY-1m-Control | 8 | 6 | 5 | 3 | 1 | 55.6% | 62.5% | 4/5 |
| 266 | TSLA-CANARY-5m-Control | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 44 | 47 | 39 | 5 | 8 | 75.0% | 88.6% | 3/5 |
| 268 | SPY-CANARY-10s-NoConf | 60 | 60 | 55 | 5 | 4 | 85.9% | 91.7% | 1/5 |
| 269 | SPY-CANARY-1m-Control | 6 | 10 | 4 | 2 | 6 | 33.3% | 66.7% | 5/5 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 47 | 0 | 0 | 47 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 34 | 0 | 0 | 34 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 84 | 0 | 0 | 84 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 84 | 0 | 0 | 84 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 127 | 0 | 0 | 127 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 45 | 0 | 0 | 45 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 70 | 0 | 0 | 70 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 116 | 0 | 0 | 116 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 49 | 0 | 0 | 49 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 60 | 0 | 0 | 60 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T16 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 5_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 21 | 48 | 18 | 3 | 30 | 35.3% | 85.7% | 2/5 |
| 265 | TSLA-CANARY-1m-Control | 4 | 7 | 1 | 3 | 6 | 10.0% | 25.0% | 5/5 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 20 | 48 | 18 | 2 | 30 | 36.0% | 90.0% | 1/5 |
| 268 | SPY-CANARY-10s-NoConf | 19 | 57 | 19 | 0 | 38 | 33.3% | 100.0% | 3/5 |
| 269 | SPY-CANARY-1m-Control | 2 | 7 | 2 | 0 | 5 | 28.6% | 100.0% | 4/5 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 79 | 0 | 0 | 79 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 79 | 0 | 0 | 79 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 106 | 0 | 0 | 106 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 88 | 0 | 0 | 88 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 121 | 0 | 0 | 121 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 37 | 0 | 0 | 37 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 45 | 0 | 0 | 45 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 31 | 0 | 0 | 31 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T17 UTC

_Strategies with activity this hour: 39; ranked (alerts≥1, BT≥1): 6_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 9 | 0 | 0 | 9 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 44 | 43 | 36 | 8 | 7 | 70.6% | 81.8% | 3/6 |
| 265 | TSLA-CANARY-1m-Control | 9 | 9 | 8 | 1 | 1 | 80.0% | 88.9% | 1/6 |
| 266 | TSLA-CANARY-5m-Control | 3 | 2 | 1 | 2 | 1 | 25.0% | 33.3% | 6/6 |
| 267 | TSLA-CANARY-10s-LooseConf | 44 | 43 | 36 | 8 | 7 | 70.6% | 81.8% | 4/6 |
| 268 | SPY-CANARY-10s-NoConf | 44 | 46 | 39 | 5 | 7 | 76.5% | 88.6% | 2/6 |
| 269 | SPY-CANARY-1m-Control | 7 | 7 | 5 | 2 | 2 | 55.6% | 71.4% | 5/6 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 62 | 0 | 0 | 62 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 62 | 0 | 0 | 62 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 92 | 0 | 0 | 92 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 37 | 0 | 0 | 37 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 76 | 0 | 0 | 76 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 116 | 0 | 0 | 116 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 44 | 0 | 0 | 44 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T18 UTC

_Strategies with activity this hour: 37; ranked (alerts≥1, BT≥1): 6_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 1 | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 1/6 |
| 174 | TSLA LONG 1Min Mass #2 | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 48 | 53 | 46 | 2 | 7 | 83.6% | 95.8% | 3/6 |
| 265 | TSLA-CANARY-1m-Control | 6 | 4 | 2 | 4 | 2 | 25.0% | 33.3% | 6/6 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 48 | 53 | 46 | 2 | 7 | 83.6% | 95.8% | 4/6 |
| 268 | SPY-CANARY-10s-NoConf | 57 | 61 | 54 | 3 | 7 | 84.4% | 94.7% | 2/6 |
| 269 | SPY-CANARY-1m-Control | 7 | 7 | 6 | 1 | 1 | 75.0% | 85.7% | 5/6 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 53 | 0 | 0 | 53 | 0.0% | 0.0% | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 81 | 0 | 0 | 81 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 81 | 0 | 0 | 81 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 97 | 0 | 0 | 97 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 90 | 0 | 0 | 90 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 36 | 0 | 0 | 36 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 99 | 0 | 0 | 99 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 35 | 0 | 0 | 35 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T19 UTC

_Strategies with activity this hour: 34; ranked (alerts≥1, BT≥1): 10_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 5 | 5 | 3 | 2 | 2 | 42.9% | 60.0% | 10/10 |
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 41 | 42 | 34 | 7 | 8 | 69.4% | 82.9% | 5/10 |
| 265 | TSLA-CANARY-1m-Control | 8 | 8 | 6 | 2 | 2 | 60.0% | 75.0% | 8/10 |
| 267 | TSLA-CANARY-10s-LooseConf | 41 | 42 | 34 | 7 | 8 | 69.4% | 82.9% | 6/10 |
| 268 | SPY-CANARY-10s-NoConf | 63 | 62 | 57 | 6 | 6 | 82.6% | 90.5% | 3/10 |
| 269 | SPY-CANARY-1m-Control | 7 | 7 | 5 | 2 | 2 | 55.6% | 71.4% | 9/10 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 64 | 62 | 58 | 6 | 5 | 84.1% | 90.6% | 1/10 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 64 | 62 | 58 | 6 | 5 | 84.1% | 90.6% | 2/10 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 38 | 46 | 32 | 6 | 15 | 60.4% | 84.2% | 7/10 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 40 | 42 | 34 | 6 | 8 | 70.8% | 85.0% | 4/10 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 40 | 0 | 0 | 40 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 84 | 0 | 0 | 84 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 84 | 0 | 0 | 84 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 118 | 0 | 0 | 118 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 31 | 0 | 0 | 31 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 55 | 0 | 0 | 55 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 91 | 0 | 0 | 91 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 28 | 0 | 0 | 28 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 112 | 0 | 0 | 112 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 62 | 0 | 0 | 62 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 62 | 0 | 0 | 62 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T20 UTC

_Strategies with activity this hour: 26; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 268 | SPY-CANARY-10s-NoConf | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 269 | SPY-CANARY-1m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 49 | 0 | 0 | 49 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 49 | 0 | 0 | 49 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 59 | 0 | 0 | 59 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 58 | 0 | 0 | 58 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 83 | 0 | 0 | 83 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T21 UTC

_Strategies with activity this hour: 19; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 44 | 0 | 0 | 44 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 44 | 0 | 0 | 44 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 45 | 0 | 0 | 45 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 56 | 0 | 0 | 56 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T22 UTC

_Strategies with activity this hour: 23; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 31 | 0 | 0 | 31 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 13 | 0 | 0 | 13 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 70 | 0 | 0 | 70 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 34 | 0 | 0 | 34 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 34 | 0 | 0 | 34 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |

---

## Hour 2026-06-01T23 UTC

_Strategies with activity this hour: 21; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 31 | 0 | 0 | 31 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 42 | 0 | 0 | 42 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T00 UTC

_Strategies with activity this hour: 5; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T08 UTC

_Strategies with activity this hour: 29; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 28 | 0 | 0 | 28 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 28 | 0 | 0 | 28 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 55 | 0 | 0 | 55 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T09 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T10 UTC

_Strategies with activity this hour: 20; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 37 | 0 | 0 | 37 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T11 UTC

_Strategies with activity this hour: 24; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 66 | 0 | 0 | 66 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 50 | 0 | 0 | 50 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 78 | 0 | 0 | 78 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T12 UTC

_Strategies with activity this hour: 28; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 56 | 0 | 0 | 56 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 56 | 0 | 0 | 56 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 84 | 0 | 0 | 84 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 46 | 0 | 0 | 46 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T13 UTC

_Strategies with activity this hour: 41; ranked (alerts≥1, BT≥1): 10_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 17 | 21 | 10 | 7 | 11 | 35.7% | 58.8% | 3/10 |
| 265 | TSLA-CANARY-1m-Control | 5 | 5 | 2 | 3 | 3 | 25.0% | 40.0% | 7/10 |
| 266 | TSLA-CANARY-5m-Control | 2 | 2 | 0 | 2 | 2 | 0.0% | 0.0% | 10/10 |
| 267 | TSLA-CANARY-10s-LooseConf | 17 | 21 | 10 | 7 | 11 | 35.7% | 58.8% | 4/10 |
| 268 | SPY-CANARY-10s-NoConf | 31 | 32 | 25 | 6 | 7 | 65.8% | 80.6% | 1/10 |
| 269 | SPY-CANARY-1m-Control | 4 | 5 | 2 | 2 | 3 | 28.6% | 50.0% | 6/10 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 31 | 32 | 25 | 6 | 7 | 65.8% | 80.6% | 2/10 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 30 | 1 | 0 | 30 | 1 | 0.0% | 0.0% | 9/10 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 31 | 1 | 0 | 31 | 1 | 0.0% | 0.0% | 8/10 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 17 | 21 | 10 | 7 | 11 | 35.7% | 58.8% | 5/10 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 17 | 0 | 0 | 17 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 83 | 0 | 0 | 83 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 83 | 0 | 0 | 83 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 111 | 0 | 0 | 111 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 31 | 0 | 0 | 31 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 35 | 0 | 0 | 35 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 23 | 0 | 0 | 23 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 115 | 0 | 0 | 115 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 39 | 0 | 0 | 39 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 59 | 0 | 0 | 59 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 59 | 0 | 0 | 59 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 28 | 0 | 0 | 28 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T14 UTC

_Strategies with activity this hour: 42; ranked (alerts≥1, BT≥1): 10_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 10 | 0 | 0 | 10 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 2 | 4 | 2 | 0 | 2 | 50.0% | 100.0% | 8/10 |
| 263 | TSLA-CANARY-10s-NoConf | 44 | 51 | 39 | 5 | 12 | 69.6% | 88.6% | 4/10 |
| 265 | TSLA-CANARY-1m-Control | 10 | 8 | 7 | 3 | 1 | 63.6% | 70.0% | 7/10 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 44 | 51 | 39 | 5 | 12 | 69.6% | 88.6% | 5/10 |
| 268 | SPY-CANARY-10s-NoConf | 46 | 48 | 44 | 2 | 3 | 89.8% | 95.7% | 2/10 |
| 269 | SPY-CANARY-1m-Control | 7 | 7 | 7 | 0 | 0 | 100.0% | 100.0% | 1/10 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 46 | 48 | 44 | 2 | 3 | 89.8% | 95.7% | 3/10 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 46 | 0 | 0 | 46 | 0 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 46 | 7 | 5 | 41 | 2 | 10.4% | 10.9% | 10/10 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 44 | 51 | 39 | 5 | 12 | 69.6% | 88.6% | 6/10 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 44 | 12 | 8 | 36 | 4 | 16.7% | 18.2% | 9/10 |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 33 | 0 | 0 | 33 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 76 | 0 | 0 | 76 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 76 | 0 | 0 | 76 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 112 | 0 | 0 | 112 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 85 | 0 | 0 | 85 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 13 | 0 | 0 | 13 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 98 | 0 | 0 | 98 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 49 | 0 | 0 | 49 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T15 UTC

_Strategies with activity this hour: 38; ranked (alerts≥1, BT≥1): 8_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 16 | 48 | 14 | 2 | 34 | 28.0% | 87.5% | 3/8 |
| 265 | TSLA-CANARY-1m-Control | 4 | 11 | 3 | 1 | 8 | 25.0% | 75.0% | 6/8 |
| 267 | TSLA-CANARY-10s-LooseConf | 16 | 48 | 14 | 2 | 34 | 28.0% | 87.5% | 4/8 |
| 268 | SPY-CANARY-10s-NoConf | 23 | 61 | 23 | 0 | 38 | 37.7% | 100.0% | 1/8 |
| 269 | SPY-CANARY-1m-Control | 3 | 9 | 1 | 2 | 8 | 9.1% | 33.3% | 7/8 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 23 | 61 | 23 | 0 | 38 | 37.7% | 100.0% | 2/8 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 23 | 0 | 0 | 23 | 0 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 23 | 11 | 0 | 23 | 11 | 0.0% | 0.0% | 8/8 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 16 | 48 | 14 | 2 | 34 | 28.0% | 87.5% | 5/8 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 16 | 0 | 0 | 16 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 76 | 0 | 0 | 76 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 76 | 0 | 0 | 76 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 89 | 0 | 0 | 89 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 44 | 0 | 0 | 44 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 78 | 0 | 0 | 78 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 28 | 0 | 0 | 28 | 0.0% | 0.0% | — |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 45 | 0 | 0 | 45 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 15 | 0 | 0 | 15 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 117 | 0 | 0 | 117 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 59 | 0 | 0 | 59 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 51 | 0 | 0 | 51 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 61 | 0 | 0 | 61 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 29 | 0 | 0 | 29 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T16 UTC

_Strategies with activity this hour: 38; ranked (alerts≥1, BT≥1): 12_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 6 | 2 | 1 | 5 | 1 | 14.3% | 16.7% | 12/12 |
| 174 | TSLA LONG 1Min Mass #2 | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 44 | 48 | 40 | 4 | 8 | 76.9% | 90.9% | 4/12 |
| 265 | TSLA-CANARY-1m-Control | 10 | 9 | 8 | 2 | 1 | 72.7% | 80.0% | 7/12 |
| 266 | TSLA-CANARY-5m-Control | 1 | 2 | 1 | 0 | 1 | 50.0% | 100.0% | 9/12 |
| 267 | TSLA-CANARY-10s-LooseConf | 44 | 48 | 40 | 4 | 8 | 76.9% | 90.9% | 5/12 |
| 268 | SPY-CANARY-10s-NoConf | 56 | 58 | 55 | 1 | 3 | 93.2% | 98.2% | 1/12 |
| 269 | SPY-CANARY-1m-Control | 8 | 9 | 5 | 3 | 4 | 41.7% | 62.5% | 10/12 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 56 | 58 | 55 | 1 | 3 | 93.2% | 98.2% | 2/12 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 56 | 58 | 55 | 1 | 3 | 93.2% | 98.2% | 3/12 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 56 | 38 | 36 | 20 | 2 | 62.1% | 64.3% | 8/12 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 44 | 48 | 40 | 4 | 8 | 76.9% | 90.9% | 6/12 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 44 | 20 | 18 | 26 | 2 | 39.1% | 40.9% | 11/12 |
| 276 | PACKTEST · Bollinger Bands · trigger | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 77 | 0 | 0 | 77 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 77 | 0 | 0 | 77 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 0 | 13 | 0 | 0 | 13 | 0.0% | 0.0% | — |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 102 | 0 | 0 | 102 | 0.0% | 0.0% | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 0 | 25 | 0 | 0 | 25 | 0.0% | 0.0% | — |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 71 | 0 | 0 | 71 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 0 | 137 | 0 | 0 | 137 | 0.0% | 0.0% | — |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 0 | 11 | 0 | 0 | 11 | 0.0% | 0.0% | — |
| 299 | PACKTEST · SuperTrend · gate | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 52 | 0 | 0 | 52 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 302 | PACKTEST · UT Bot V4 · trigger | 0 | 57 | 0 | 0 | 57 | 0.0% | 0.0% | — |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T17 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 26_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 49 | 49 | 45 | 4 | 3 | 86.5% | 91.8% | 2/26 |
| 265 | TSLA-CANARY-1m-Control | 8 | 6 | 3 | 5 | 3 | 27.3% | 37.5% | 15/26 |
| 266 | TSLA-CANARY-5m-Control | 2 | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 1/26 |
| 267 | TSLA-CANARY-10s-LooseConf | 49 | 49 | 45 | 4 | 3 | 86.5% | 91.8% | 3/26 |
| 268 | SPY-CANARY-10s-NoConf | 57 | 61 | 52 | 5 | 9 | 78.8% | 91.2% | 5/26 |
| 269 | SPY-CANARY-1m-Control | 8 | 9 | 7 | 1 | 2 | 70.0% | 87.5% | 8/26 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 57 | 61 | 52 | 5 | 9 | 78.8% | 91.2% | 6/26 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 57 | 34 | 30 | 27 | 4 | 49.2% | 52.6% | 9/26 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 57 | 20 | 19 | 38 | 1 | 32.8% | 33.3% | 11/26 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 49 | 49 | 45 | 4 | 3 | 86.5% | 91.8% | 4/26 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 49 | 43 | 39 | 10 | 3 | 75.0% | 79.6% | 7/26 |
| 276 | PACKTEST · Bollinger Bands · trigger | 4 | 14 | 3 | 1 | 11 | 20.0% | 75.0% | 20/26 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 1 | 78 | 1 | 0 | 77 | 1.3% | 100.0% | 25/26 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 1 | 78 | 1 | 0 | 77 | 1.3% | 100.0% | 26/26 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 2 | 13 | 1 | 1 | 12 | 7.1% | 50.0% | 24/26 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 30 | 103 | 24 | 6 | 78 | 22.2% | 80.0% | 18/26 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 14 | 30 | 12 | 2 | 17 | 38.7% | 85.7% | 10/26 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 10 | 47 | 6 | 4 | 41 | 11.8% | 60.0% | 22/26 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 28 | 81 | 23 | 5 | 57 | 27.1% | 82.1% | 16/26 |
| 292 | PACKTEST · Support Resistance Channels · trig | 9 | 35 | 5 | 4 | 30 | 12.8% | 55.6% | 21/26 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 11 | 29 | 9 | 2 | 20 | 29.0% | 81.8% | 13/26 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 38 | 110 | 31 | 7 | 78 | 26.7% | 81.6% | 17/26 |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 4 | 15 | 2 | 2 | 13 | 11.8% | 50.0% | 23/26 |
| 299 | PACKTEST · SuperTrend · gate | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 11 | 46 | 10 | 1 | 36 | 21.3% | 90.9% | 19/26 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 21 | 62 | 20 | 1 | 42 | 31.7% | 95.2% | 12/26 |
| 302 | PACKTEST · UT Bot V4 · trigger | 19 | 62 | 18 | 1 | 44 | 28.6% | 94.7% | 14/26 |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T18 UTC

_Strategies with activity this hour: 35; ranked (alerts≥1, BT≥1): 25_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 56 | 56 | 51 | 5 | 5 | 83.6% | 91.1% | 8/25 |
| 265 | TSLA-CANARY-1m-Control | 5 | 7 | 4 | 1 | 3 | 50.0% | 80.0% | 20/25 |
| 266 | TSLA-CANARY-5m-Control | 2 | 2 | 0 | 2 | 2 | 0.0% | 0.0% | 25/25 |
| 267 | TSLA-CANARY-10s-LooseConf | 56 | 56 | 51 | 5 | 5 | 83.6% | 91.1% | 9/25 |
| 268 | SPY-CANARY-10s-NoConf | 59 | 59 | 57 | 2 | 2 | 93.4% | 96.6% | 3/25 |
| 269 | SPY-CANARY-1m-Control | 9 | 9 | 8 | 1 | 1 | 80.0% | 88.9% | 15/25 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 59 | 59 | 57 | 2 | 2 | 93.4% | 96.6% | 4/25 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 59 | 0 | 0 | 59 | 0 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 59 | 0 | 0 | 59 | 0 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 56 | 56 | 51 | 5 | 5 | 83.6% | 91.1% | 10/25 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 56 | 52 | 47 | 9 | 5 | 77.0% | 83.9% | 17/25 |
| 276 | PACKTEST · Bollinger Bands · trigger | 18 | 18 | 17 | 1 | 1 | 89.5% | 94.4% | 6/25 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 4 | 92 | 3 | 1 | 88 | 3.3% | 75.0% | 22/25 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 4 | 92 | 3 | 1 | 88 | 3.3% | 75.0% | 23/25 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 23 | 23 | 22 | 1 | 1 | 91.7% | 95.7% | 5/25 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 106 | 107 | 94 | 12 | 11 | 80.3% | 88.7% | 14/25 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 30 | 30 | 29 | 1 | 1 | 93.5% | 96.7% | 2/25 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 55 | 55 | 47 | 8 | 8 | 74.6% | 85.5% | 18/25 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 82 | 80 | 72 | 10 | 7 | 80.9% | 87.8% | 13/25 |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 54 | 45 | 22 | 32 | 24 | 28.2% | 40.7% | 21/25 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 25 | 21 | 21 | 4 | 0 | 84.0% | 84.0% | 7/25 |
| 296 | PACKTEST · Strat Assistant · trigger | 120 | 121 | 106 | 14 | 13 | 79.7% | 88.3% | 16/25 |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 10 | 10 | 0 | 0 | 100.0% | 100.0% | 1/25 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 40 | 38 | 32 | 8 | 6 | 69.6% | 80.0% | 19/25 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 59 | 59 | 53 | 6 | 6 | 81.5% | 89.8% | 11/25 |
| 302 | PACKTEST · UT Bot V4 · trigger | 59 | 59 | 53 | 6 | 6 | 81.5% | 89.8% | 12/25 |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 3 | 8 | 0 | 3 | 8 | 0.0% | 0.0% | 24/25 |

---

## Hour 2026-06-02T19 UTC

_Strategies with activity this hour: 39; ranked (alerts≥1, BT≥1): 23_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 10 | 0 | 0 | 10 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 47 | 46 | 42 | 5 | 4 | 82.4% | 89.4% | 6/23 |
| 265 | TSLA-CANARY-1m-Control | 10 | 9 | 8 | 2 | 1 | 72.7% | 80.0% | 16/23 |
| 266 | TSLA-CANARY-5m-Control | 2 | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 1/23 |
| 267 | TSLA-CANARY-10s-LooseConf | 47 | 46 | 42 | 5 | 4 | 82.4% | 89.4% | 7/23 |
| 268 | SPY-CANARY-10s-NoConf | 51 | 51 | 47 | 4 | 4 | 85.5% | 92.2% | 4/23 |
| 269 | SPY-CANARY-1m-Control | 7 | 5 | 4 | 3 | 1 | 50.0% | 57.1% | 20/23 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 51 | 51 | 47 | 4 | 4 | 85.5% | 92.2% | 5/23 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 51 | 0 | 0 | 51 | 0 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 51 | 23 | 19 | 32 | 4 | 34.5% | 37.3% | 21/23 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 47 | 46 | 42 | 5 | 4 | 82.4% | 89.4% | 8/23 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 47 | 13 | 11 | 36 | 2 | 22.4% | 23.4% | 22/23 |
| 276 | PACKTEST · Bollinger Bands · trigger | 14 | 16 | 14 | 0 | 2 | 87.5% | 100.0% | 3/23 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 76 | 0 | 0 | 76 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 76 | 0 | 0 | 76 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 19 | 21 | 18 | 1 | 3 | 81.8% | 94.7% | 9/23 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 40 | 0 | 0 | 40 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 101 | 104 | 81 | 20 | 18 | 68.1% | 80.2% | 18/23 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 36 | 38 | 32 | 4 | 6 | 76.2% | 88.9% | 14/23 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 49 | 47 | 36 | 13 | 9 | 62.1% | 73.5% | 19/23 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 98 | 95 | 78 | 20 | 10 | 72.2% | 79.6% | 17/23 |
| 292 | PACKTEST · Support Resistance Channels · trig | 41 | 17 | 11 | 30 | 8 | 22.4% | 26.8% | 23/23 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 38 | 0 | 0 | 38 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 19 | 18 | 18 | 1 | 0 | 94.7% | 94.7% | 2/23 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 108 | 110 | 93 | 15 | 14 | 76.2% | 86.1% | 13/23 |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 8 | 10 | 8 | 0 | 2 | 80.0% | 100.0% | 10/23 |
| 299 | PACKTEST · SuperTrend · gate | 0 | 48 | 0 | 0 | 48 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 38 | 37 | 31 | 7 | 4 | 73.8% | 81.6% | 15/23 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 52 | 52 | 46 | 6 | 6 | 79.3% | 88.5% | 11/23 |
| 302 | PACKTEST · UT Bot V4 · trigger | 52 | 52 | 46 | 6 | 6 | 79.3% | 88.5% | 12/23 |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T20 UTC

_Strategies with activity this hour: 33; ranked (alerts≥1, BT≥1): 13_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 263 | TSLA-CANARY-10s-NoConf | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 8 | 3 | 9 | 5 | 17.6% | 25.0% | 11/13 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 58 | 0 | 0 | 58 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 58 | 0 | 0 | 58 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 30 | 0 | 0 | 30 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 8 | 12 | 2 | 6 | 10 | 11.1% | 25.0% | 12/13 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 92 | 81 | 44 | 48 | 37 | 34.1% | 47.8% | 7/13 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 28 | 22 | 4 | 24 | 18 | 8.7% | 14.3% | 13/13 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 27 | 34 | 14 | 13 | 19 | 30.4% | 51.9% | 8/13 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 83 | 42 | 28 | 55 | 14 | 28.9% | 33.7% | 9/13 |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 16 | 17 | 10 | 6 | 6 | 45.5% | 62.5% | 4/13 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 11 | 10 | 7 | 4 | 3 | 50.0% | 63.6% | 2/13 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 79 | 78 | 51 | 28 | 28 | 47.7% | 64.6% | 3/13 |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 16 | 0 | 0 | 16 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 18 | 10 | 5 | 13 | 5 | 21.7% | 27.8% | 10/13 |
| 299 | PACKTEST · SuperTrend · gate | 0 | 41 | 0 | 0 | 41 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 16 | 15 | 11 | 5 | 2 | 61.1% | 68.8% | 1/13 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 59 | 41 | 27 | 32 | 13 | 37.5% | 45.8% | 5/13 |
| 302 | PACKTEST · UT Bot V4 · trigger | 59 | 41 | 27 | 32 | 13 | 37.5% | 45.8% | 6/13 |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T21 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 15_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 2 | 1 | 9 | 1 | 9.1% | 10.0% | 13/15 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 1 | 45 | 0 | 1 | 45 | 0.0% | 0.0% | 14/15 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 1 | 45 | 0 | 1 | 45 | 0.0% | 0.0% | 15/15 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 29 | 16 | 5 | 24 | 11 | 12.5% | 17.2% | 10/15 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 59 | 50 | 28 | 31 | 22 | 34.6% | 47.5% | 6/15 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 32 | 15 | 4 | 28 | 11 | 9.3% | 12.5% | 12/15 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 39 | 37 | 31 | 8 | 6 | 68.9% | 79.5% | 1/15 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 95 | 32 | 21 | 74 | 11 | 19.8% | 22.1% | 8/15 |
| 292 | PACKTEST · Support Resistance Channels · trig | 11 | 22 | 5 | 6 | 17 | 17.9% | 45.5% | 9/15 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 8 | 2 | 1 | 7 | 1 | 11.1% | 12.5% | 11/15 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 57 | 60 | 46 | 11 | 13 | 65.7% | 80.7% | 2/15 |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 25 | 10 | 7 | 18 | 3 | 25.0% | 28.0% | 7/15 |
| 299 | PACKTEST · SuperTrend · gate | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 5 | 5 | 4 | 1 | 2 | 57.1% | 80.0% | 3/15 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 46 | 38 | 25 | 21 | 12 | 43.1% | 54.3% | 4/15 |
| 302 | PACKTEST · UT Bot V4 · trigger | 47 | 38 | 25 | 22 | 12 | 42.4% | 53.2% | 5/15 |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T22 UTC

_Strategies with activity this hour: 29; ranked (alerts≥1, BT≥1): 24_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 17 | 2 | 1 | 16 | 1 | 5.6% | 5.9% | 16/24 |
| 277 | PACKTEST · Bollinger Bands · gate | 18 | 1 | 0 | 18 | 1 | 0.0% | 0.0% | 21/24 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 19 | 23 | 5 | 14 | 18 | 13.5% | 26.3% | 11/24 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 18 | 3 | 1 | 17 | 2 | 5.0% | 5.6% | 17/24 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 19 | 23 | 5 | 14 | 18 | 13.5% | 26.3% | 12/24 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 18 | 3 | 1 | 17 | 2 | 5.0% | 5.6% | 18/24 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 16 | 4 | 0 | 16 | 4 | 0.0% | 0.0% | 22/24 |
| 283 | PACKTEST · EMA Stack v2 · gate | 29 | 3 | 1 | 28 | 2 | 3.2% | 3.4% | 20/24 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 48 | 37 | 12 | 36 | 25 | 16.4% | 25.0% | 8/24 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 12 | 7 | 3 | 9 | 4 | 18.8% | 25.0% | 7/24 |
| 286 | PACKTEST · MACD Line v2 · trigger | 15 | 7 | 0 | 15 | 7 | 0.0% | 0.0% | 23/24 |
| 287 | PACKTEST · MACD Line v2 · gate | 22 | 3 | 1 | 21 | 2 | 4.2% | 4.5% | 19/24 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 25 | 13 | 7 | 18 | 6 | 22.6% | 28.0% | 6/24 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 64 | 19 | 9 | 55 | 10 | 12.2% | 14.1% | 13/24 |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 14 | 11 | 3 | 11 | 8 | 13.6% | 21.4% | 10/24 |
| 293 | PACKTEST · Support Resistance Channels · gate | 11 | 8 | 2 | 9 | 6 | 11.8% | 18.2% | 14/24 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 13 | 6 | 4 | 9 | 4 | 23.5% | 30.8% | 5/24 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 47 | 42 | 34 | 13 | 8 | 61.8% | 72.3% | 1/24 |
| 297 | PACKTEST · Strat Assistant · gate | 4 | 4 | 0 | 4 | 4 | 0.0% | 0.0% | 24/24 |
| 298 | PACKTEST · SuperTrend · trigger | 20 | 6 | 2 | 18 | 4 | 8.3% | 10.0% | 15/24 |
| 299 | PACKTEST · SuperTrend · gate | 29 | 0 | 0 | 29 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 5 | 3 | 3 | 2 | 1 | 50.0% | 60.0% | 2/24 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 18 | 29 | 6 | 12 | 23 | 14.6% | 33.3% | 9/24 |
| 302 | PACKTEST · UT Bot V4 · trigger | 38 | 29 | 16 | 22 | 13 | 31.4% | 42.1% | 4/24 |
| 303 | PACKTEST · UT Bot V4 · gate | 6 | 5 | 3 | 3 | 2 | 37.5% | 50.0% | 3/24 |
| 304 | PACKTEST · VWAP v2 · trigger | 18 | 0 | 0 | 18 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-02T23 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 25_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 5 | 4 | 8 | 1 | 30.8% | 33.3% | 12/25 |
| 277 | PACKTEST · Bollinger Bands · gate | 15 | 15 | 11 | 4 | 3 | 61.1% | 73.3% | 1/25 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 17 | 33 | 13 | 4 | 20 | 35.1% | 76.5% | 8/25 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 17 | 33 | 13 | 4 | 20 | 35.1% | 76.5% | 9/25 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 8 | 3 | 0 | 8 | 3 | 0.0% | 0.0% | 23/25 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 76 | 59 | 26 | 50 | 32 | 24.1% | 34.2% | 17/25 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 8 | 7 | 4 | 4 | 3 | 36.4% | 50.0% | 6/25 |
| 286 | PACKTEST · MACD Line v2 · trigger | 38 | 17 | 10 | 28 | 7 | 22.2% | 26.3% | 18/25 |
| 287 | PACKTEST · MACD Line v2 · gate | 13 | 13 | 9 | 4 | 3 | 56.2% | 69.2% | 2/25 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 29 | 12 | 6 | 23 | 6 | 17.1% | 20.7% | 19/25 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 121 | 66 | 46 | 75 | 16 | 33.6% | 38.0% | 10/25 |
| 291 | PACKTEST · Relative Volume v2 · gate | 6 | 6 | 3 | 3 | 3 | 33.3% | 50.0% | 11/25 |
| 292 | PACKTEST · Support Resistance Channels · trig | 5 | 5 | 0 | 5 | 5 | 0.0% | 0.0% | 24/25 |
| 293 | PACKTEST · Support Resistance Channels · gate | 10 | 12 | 3 | 7 | 9 | 15.8% | 30.0% | 20/25 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 33 | 20 | 12 | 21 | 8 | 29.3% | 36.4% | 15/25 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 6 | 10 | 4 | 2 | 5 | 36.4% | 66.7% | 7/25 |
| 296 | PACKTEST · Strat Assistant · trigger | 59 | 65 | 42 | 17 | 19 | 53.8% | 71.2% | 3/25 |
| 297 | PACKTEST · Strat Assistant · gate | 6 | 10 | 1 | 5 | 9 | 6.7% | 16.7% | 22/25 |
| 298 | PACKTEST · SuperTrend · trigger | 26 | 14 | 5 | 21 | 9 | 14.3% | 19.2% | 21/25 |
| 299 | PACKTEST · SuperTrend · gate | 23 | 17 | 12 | 11 | 4 | 44.4% | 52.2% | 5/25 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 11 | 11 | 5 | 6 | 6 | 29.4% | 45.5% | 14/25 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 29 | 45 | 17 | 12 | 28 | 29.8% | 58.6% | 13/25 |
| 302 | PACKTEST · UT Bot V4 · trigger | 49 | 45 | 31 | 18 | 13 | 50.0% | 63.3% | 4/25 |
| 303 | PACKTEST · UT Bot V4 · gate | 15 | 9 | 5 | 10 | 4 | 26.3% | 33.3% | 16/25 |
| 304 | PACKTEST · VWAP v2 · trigger | 1 | 6 | 0 | 1 | 6 | 0.0% | 0.0% | 25/25 |

---

## Hour 2026-06-03T08 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 22_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 3 | 2 | 8 | 1 | 18.2% | 20.0% | 5/22 |
| 277 | PACKTEST · Bollinger Bands · gate | 3 | 7 | 1 | 2 | 6 | 11.1% | 33.3% | 9/22 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 10 | 5 | 1 | 9 | 4 | 7.1% | 10.0% | 14/22 |
| 283 | PACKTEST · EMA Stack v2 · gate | 2 | 6 | 0 | 2 | 6 | 0.0% | 0.0% | 22/22 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 55 | 28 | 7 | 48 | 21 | 9.2% | 12.7% | 13/22 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 6 | 1 | 0 | 6 | 1 | 0.0% | 0.0% | 20/22 |
| 286 | PACKTEST · MACD Line v2 · trigger | 24 | 7 | 0 | 24 | 7 | 0.0% | 0.0% | 16/22 |
| 287 | PACKTEST · MACD Line v2 · gate | 1 | 9 | 1 | 0 | 8 | 11.1% | 100.0% | 10/22 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 19 | 14 | 3 | 16 | 11 | 10.0% | 15.8% | 11/22 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 79 | 24 | 14 | 65 | 9 | 15.9% | 17.7% | 7/22 |
| 291 | PACKTEST · Relative Volume v2 · gate | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 12 | 12 | 6 | 6 | 7 | 31.6% | 50.0% | 2/22 |
| 293 | PACKTEST · Support Resistance Channels · gate | 9 | 2 | 0 | 9 | 2 | 0.0% | 0.0% | 19/22 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 16 | 7 | 2 | 14 | 5 | 9.5% | 12.5% | 12/22 |
| 296 | PACKTEST · Strat Assistant · trigger | 38 | 42 | 31 | 7 | 13 | 60.8% | 81.6% | 1/22 |
| 297 | PACKTEST · Strat Assistant · gate | 4 | 1 | 0 | 4 | 1 | 0.0% | 0.0% | 21/22 |
| 298 | PACKTEST · SuperTrend · trigger | 20 | 6 | 0 | 20 | 6 | 0.0% | 0.0% | 17/22 |
| 299 | PACKTEST · SuperTrend · gate | 9 | 14 | 3 | 6 | 11 | 15.0% | 33.3% | 8/22 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 2 | 3 | 1 | 1 | 2 | 25.0% | 50.0% | 3/22 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 26 | 16 | 7 | 19 | 9 | 20.0% | 26.9% | 4/22 |
| 302 | PACKTEST · UT Bot V4 · trigger | 32 | 16 | 7 | 25 | 9 | 17.1% | 21.9% | 6/22 |
| 303 | PACKTEST · UT Bot V4 · gate | 11 | 9 | 1 | 10 | 8 | 5.3% | 9.1% | 15/22 |
| 304 | PACKTEST · VWAP v2 · trigger | 15 | 11 | 0 | 15 | 11 | 0.0% | 0.0% | 18/22 |

---

## Hour 2026-06-03T09 UTC

_Strategies with activity this hour: 24; ranked (alerts≥1, BT≥1): 18_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 8 | 2 | 0 | 8 | 2 | 0.0% | 0.0% | 17/18 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 21 | 0 | 0 | 21 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 16 | 2 | 0 | 16 | 2 | 0.0% | 0.0% | 15/18 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 46 | 25 | 12 | 34 | 13 | 20.3% | 26.1% | 8/18 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 30 | 12 | 3 | 27 | 9 | 7.7% | 10.0% | 12/18 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 18 | 20 | 11 | 7 | 9 | 40.7% | 61.1% | 5/18 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 98 | 34 | 21 | 77 | 12 | 19.1% | 21.4% | 10/18 |
| 291 | PACKTEST · Relative Volume v2 · gate | 12 | 8 | 6 | 6 | 2 | 42.9% | 50.0% | 4/18 |
| 292 | PACKTEST · Support Resistance Channels · trig | 14 | 9 | 2 | 12 | 7 | 9.5% | 14.3% | 11/18 |
| 293 | PACKTEST · Support Resistance Channels · gate | 17 | 2 | 1 | 16 | 1 | 5.6% | 5.9% | 13/18 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 26 | 11 | 6 | 20 | 5 | 19.4% | 23.1% | 9/18 |
| 296 | PACKTEST · Strat Assistant · trigger | 37 | 37 | 32 | 5 | 4 | 78.0% | 86.5% | 1/18 |
| 297 | PACKTEST · Strat Assistant · gate | 0 | 7 | 0 | 0 | 7 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 28 | 6 | 0 | 28 | 6 | 0.0% | 0.0% | 14/18 |
| 299 | PACKTEST · SuperTrend · gate | 4 | 5 | 0 | 4 | 5 | 0.0% | 0.0% | 18/18 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 2 | 3 | 2 | 0 | 1 | 66.7% | 100.0% | 2/18 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 21 | 25 | 10 | 11 | 15 | 27.8% | 47.6% | 7/18 |
| 302 | PACKTEST · UT Bot V4 · trigger | 35 | 25 | 19 | 16 | 6 | 46.3% | 54.3% | 3/18 |
| 303 | PACKTEST · UT Bot V4 · gate | 2 | 6 | 2 | 0 | 4 | 33.3% | 100.0% | 6/18 |
| 304 | PACKTEST · VWAP v2 · trigger | 14 | 2 | 0 | 14 | 2 | 0.0% | 0.0% | 16/18 |

---

## Hour 2026-06-03T10 UTC

_Strategies with activity this hour: 24; ranked (alerts≥1, BT≥1): 20_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 5 | 2 | 10 | 3 | 13.3% | 16.7% | 10/20 |
| 277 | PACKTEST · Bollinger Bands · gate | 4 | 4 | 0 | 4 | 4 | 0.0% | 0.0% | 20/20 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 13 | 0 | 0 | 13 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 13 | 0 | 0 | 13 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 9 | 5 | 0 | 9 | 5 | 0.0% | 0.0% | 17/20 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 61 | 24 | 12 | 49 | 12 | 16.4% | 19.7% | 8/20 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 7 | 7 | 1 | 6 | 6 | 7.7% | 14.3% | 14/20 |
| 286 | PACKTEST · MACD Line v2 · trigger | 18 | 5 | 0 | 18 | 5 | 0.0% | 0.0% | 16/20 |
| 287 | PACKTEST · MACD Line v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 13 | 13 | 9 | 4 | 4 | 52.9% | 69.2% | 2/20 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 84 | 32 | 21 | 63 | 12 | 21.9% | 25.0% | 4/20 |
| 291 | PACKTEST · Relative Volume v2 · gate | 6 | 4 | 0 | 6 | 4 | 0.0% | 0.0% | 19/20 |
| 292 | PACKTEST · Support Resistance Channels · trig | 10 | 17 | 3 | 7 | 15 | 12.0% | 30.0% | 12/20 |
| 293 | PACKTEST · Support Resistance Channels · gate | 9 | 4 | 0 | 9 | 4 | 0.0% | 0.0% | 18/20 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 19 | 8 | 4 | 15 | 4 | 17.4% | 21.1% | 6/20 |
| 296 | PACKTEST · Strat Assistant · trigger | 35 | 37 | 26 | 9 | 11 | 56.5% | 74.3% | 1/20 |
| 297 | PACKTEST · Strat Assistant · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 15 | 5 | 1 | 14 | 4 | 5.3% | 6.7% | 15/20 |
| 299 | PACKTEST · SuperTrend · gate | 7 | 10 | 2 | 5 | 8 | 13.3% | 28.6% | 11/20 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 2 | 4 | 1 | 1 | 3 | 20.0% | 50.0% | 5/20 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 13 | 19 | 6 | 7 | 13 | 23.1% | 46.2% | 3/20 |
| 302 | PACKTEST · UT Bot V4 · trigger | 29 | 19 | 7 | 22 | 12 | 17.1% | 24.1% | 7/20 |
| 303 | PACKTEST · UT Bot V4 · gate | 8 | 3 | 1 | 7 | 2 | 10.0% | 12.5% | 13/20 |
| 304 | PACKTEST · VWAP v2 · trigger | 10 | 6 | 2 | 8 | 4 | 14.3% | 20.0% | 9/20 |

---

## Hour 2026-06-03T11 UTC

_Strategies with activity this hour: 28; ranked (alerts≥1, BT≥1): 26_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 5 | 1 | 11 | 4 | 6.2% | 8.3% | 23/26 |
| 277 | PACKTEST · Bollinger Bands · gate | 24 | 14 | 7 | 17 | 7 | 22.6% | 29.2% | 15/26 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 22 | 18 | 9 | 13 | 9 | 29.0% | 40.9% | 10/26 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 43 | 0 | 0 | 43 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 22 | 18 | 9 | 13 | 9 | 29.0% | 40.9% | 11/26 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 25 | 17 | 5 | 20 | 12 | 13.5% | 20.0% | 21/26 |
| 283 | PACKTEST · EMA Stack v2 · gate | 36 | 34 | 20 | 16 | 14 | 40.0% | 55.6% | 4/26 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 67 | 50 | 25 | 42 | 25 | 27.2% | 37.3% | 12/26 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 21 | 9 | 5 | 16 | 4 | 20.0% | 23.8% | 17/26 |
| 286 | PACKTEST · MACD Line v2 · trigger | 32 | 21 | 8 | 24 | 13 | 17.8% | 25.0% | 20/26 |
| 287 | PACKTEST · MACD Line v2 · gate | 38 | 30 | 17 | 21 | 13 | 33.3% | 44.7% | 8/26 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 47 | 37 | 25 | 22 | 13 | 41.7% | 53.2% | 3/26 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 92 | 38 | 23 | 69 | 17 | 21.1% | 25.0% | 16/26 |
| 291 | PACKTEST · Relative Volume v2 · gate | 4 | 6 | 2 | 2 | 4 | 25.0% | 50.0% | 14/26 |
| 292 | PACKTEST · Support Resistance Channels · trig | 19 | 19 | 9 | 10 | 10 | 31.0% | 47.4% | 9/26 |
| 293 | PACKTEST · Support Resistance Channels · gate | 21 | 16 | 2 | 19 | 14 | 5.7% | 9.5% | 24/26 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 21 | 12 | 5 | 16 | 7 | 17.9% | 23.8% | 19/26 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 2 | 2 | 0 | 2 | 2 | 0.0% | 0.0% | 26/26 |
| 296 | PACKTEST · Strat Assistant · trigger | 66 | 65 | 52 | 14 | 14 | 65.0% | 78.8% | 1/26 |
| 297 | PACKTEST · Strat Assistant · gate | 10 | 14 | 4 | 6 | 10 | 20.0% | 40.0% | 18/26 |
| 298 | PACKTEST · SuperTrend · trigger | 27 | 13 | 3 | 24 | 10 | 8.1% | 11.1% | 22/26 |
| 299 | PACKTEST · SuperTrend · gate | 55 | 44 | 27 | 28 | 17 | 37.5% | 49.1% | 6/26 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 10 | 8 | 7 | 3 | 2 | 58.3% | 70.0% | 2/26 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 45 | 47 | 24 | 21 | 23 | 35.3% | 53.3% | 7/26 |
| 302 | PACKTEST · UT Bot V4 · trigger | 55 | 47 | 29 | 26 | 18 | 39.7% | 52.7% | 5/26 |
| 303 | PACKTEST · UT Bot V4 · gate | 18 | 17 | 7 | 11 | 10 | 25.0% | 38.9% | 13/26 |
| 304 | PACKTEST · VWAP v2 · trigger | 7 | 16 | 1 | 6 | 15 | 4.5% | 14.3% | 25/26 |

---

## Hour 2026-06-03T12 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 23_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 6 | 3 | 7 | 3 | 23.1% | 30.0% | 20/23 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 63 | 0 | 0 | 63 | 0.0% | 0.0% | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | 8 | 2 | 2 | 6 | 0 | 25.0% | 25.0% | 17/23 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 63 | 0 | 0 | 63 | 0.0% | 0.0% | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | 8 | 2 | 2 | 6 | 0 | 25.0% | 25.0% | 18/23 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 20 | 15 | 8 | 12 | 6 | 30.8% | 40.0% | 10/23 |
| 283 | PACKTEST · EMA Stack v2 · gate | 8 | 4 | 3 | 5 | 1 | 33.3% | 37.5% | 9/23 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 101 | 88 | 62 | 39 | 25 | 49.2% | 61.4% | 7/23 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 2 | 3 | 1 | 1 | 2 | 25.0% | 50.0% | 19/23 |
| 286 | PACKTEST · MACD Line v2 · trigger | 29 | 27 | 13 | 16 | 14 | 30.2% | 44.8% | 11/23 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 30 | 31 | 26 | 4 | 6 | 72.2% | 86.7% | 3/23 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 124 | 93 | 71 | 53 | 16 | 50.7% | 57.3% | 5/23 |
| 291 | PACKTEST · Relative Volume v2 · gate | 12 | 6 | 6 | 6 | 0 | 50.0% | 50.0% | 6/23 |
| 292 | PACKTEST · Support Resistance Channels · trig | 32 | 22 | 11 | 21 | 11 | 25.6% | 34.4% | 15/23 |
| 293 | PACKTEST · Support Resistance Channels · gate | 20 | 12 | 7 | 13 | 5 | 28.0% | 35.0% | 13/23 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 30 | 30 | 27 | 3 | 3 | 81.8% | 90.0% | 1/23 |
| 296 | PACKTEST · Strat Assistant · trigger | 89 | 106 | 67 | 22 | 39 | 52.3% | 75.3% | 4/23 |
| 297 | PACKTEST · Strat Assistant · gate | 4 | 9 | 0 | 4 | 9 | 0.0% | 0.0% | 23/23 |
| 298 | PACKTEST · SuperTrend · trigger | 11 | 11 | 5 | 6 | 6 | 29.4% | 45.5% | 12/23 |
| 299 | PACKTEST · SuperTrend · gate | 6 | 8 | 0 | 6 | 8 | 0.0% | 0.0% | 22/23 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 25 | 25 | 22 | 3 | 3 | 78.6% | 88.0% | 2/23 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 40 | 53 | 20 | 20 | 33 | 27.4% | 50.0% | 14/23 |
| 302 | PACKTEST · UT Bot V4 · trigger | 70 | 53 | 39 | 31 | 14 | 46.4% | 55.7% | 8/23 |
| 303 | PACKTEST · UT Bot V4 · gate | 12 | 8 | 4 | 8 | 4 | 25.0% | 33.3% | 16/23 |
| 304 | PACKTEST · VWAP v2 · trigger | 9 | 6 | 2 | 7 | 4 | 15.4% | 22.2% | 21/23 |

---

## Hour 2026-06-03T13 UTC

_Strategies with activity this hour: 37; ranked (alerts≥1, BT≥1): 31_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 20 | 24 | 11 | 9 | 13 | 33.3% | 55.0% | 12/31 |
| 265 | TSLA-CANARY-1m-Control | 3 | 6 | 3 | 0 | 3 | 50.0% | 100.0% | 2/31 |
| 266 | TSLA-CANARY-5m-Control | 1 | 2 | 1 | 0 | 1 | 50.0% | 100.0% | 3/31 |
| 267 | TSLA-CANARY-10s-LooseConf | 20 | 24 | 11 | 9 | 13 | 33.3% | 55.0% | 13/31 |
| 268 | SPY-CANARY-10s-NoConf | 28 | 20 | 12 | 16 | 8 | 33.3% | 42.9% | 10/31 |
| 269 | SPY-CANARY-1m-Control | 4 | 5 | 0 | 4 | 5 | 0.0% | 0.0% | 30/31 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 28 | 20 | 12 | 16 | 8 | 33.3% | 42.9% | 11/31 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 20 | 24 | 11 | 9 | 13 | 33.3% | 55.0% | 14/31 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 0 | 24 | 0 | 0 | 24 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 8 | 6 | 1 | 7 | 5 | 7.7% | 12.5% | 25/31 |
| 277 | PACKTEST · Bollinger Bands · gate | 32 | 2 | 0 | 32 | 2 | 0.0% | 0.0% | 29/31 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 18 | 67 | 11 | 7 | 55 | 15.1% | 61.1% | 23/31 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 18 | 67 | 11 | 7 | 55 | 15.1% | 61.1% | 24/31 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 20 | 17 | 7 | 13 | 10 | 23.3% | 35.0% | 18/31 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 103 | 96 | 60 | 43 | 33 | 44.1% | 58.3% | 4/31 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 52 | 25 | 11 | 41 | 14 | 16.7% | 21.2% | 21/31 |
| 286 | PACKTEST · MACD Line v2 · trigger | 37 | 33 | 16 | 21 | 18 | 29.1% | 43.2% | 15/31 |
| 287 | PACKTEST · MACD Line v2 · gate | 30 | 0 | 0 | 30 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 42 | 49 | 33 | 9 | 15 | 57.9% | 78.6% | 1/31 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 98 | 67 | 40 | 58 | 22 | 33.3% | 40.8% | 9/31 |
| 291 | PACKTEST · Relative Volume v2 · gate | 32 | 10 | 6 | 26 | 4 | 16.7% | 18.8% | 22/31 |
| 292 | PACKTEST · Support Resistance Channels · trig | 29 | 24 | 8 | 21 | 17 | 17.4% | 27.6% | 20/31 |
| 293 | PACKTEST · Support Resistance Channels · gate | 48 | 2 | 0 | 48 | 2 | 0.0% | 0.0% | 28/31 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 33 | 25 | 15 | 18 | 11 | 34.1% | 45.5% | 8/31 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 32 | 0 | 0 | 32 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 99 | 94 | 58 | 41 | 35 | 43.3% | 58.6% | 5/31 |
| 297 | PACKTEST · Strat Assistant · gate | 14 | 11 | 4 | 10 | 7 | 19.0% | 28.6% | 19/31 |
| 298 | PACKTEST · SuperTrend · trigger | 19 | 13 | 2 | 17 | 11 | 6.7% | 10.5% | 26/31 |
| 299 | PACKTEST · SuperTrend · gate | 78 | 2 | 0 | 78 | 2 | 0.0% | 0.0% | 27/31 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 26 | 38 | 14 | 12 | 24 | 28.0% | 53.8% | 17/31 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 62 | 59 | 34 | 28 | 25 | 39.1% | 54.8% | 7/31 |
| 302 | PACKTEST · UT Bot V4 · trigger | 78 | 59 | 41 | 37 | 18 | 42.7% | 52.6% | 6/31 |
| 303 | PACKTEST · UT Bot V4 · gate | 48 | 32 | 18 | 30 | 14 | 29.0% | 37.5% | 16/31 |
| 304 | PACKTEST · VWAP v2 · trigger | 2 | 4 | 0 | 2 | 4 | 0.0% | 0.0% | 31/31 |

---

## Hour 2026-06-03T14 UTC

_Strategies with activity this hour: 35; ranked (alerts≥1, BT≥1): 28_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 7 | 0 | 0 | 7 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 2 | 2 | 0 | 2 | 2 | 0.0% | 0.0% | 28/28 |
| 263 | TSLA-CANARY-10s-NoConf | 49 | 49 | 48 | 1 | 1 | 96.0% | 98.0% | 4/28 |
| 265 | TSLA-CANARY-1m-Control | 7 | 7 | 5 | 2 | 2 | 55.6% | 71.4% | 19/28 |
| 267 | TSLA-CANARY-10s-LooseConf | 49 | 49 | 48 | 1 | 1 | 96.0% | 98.0% | 5/28 |
| 268 | SPY-CANARY-10s-NoConf | 53 | 51 | 46 | 7 | 5 | 79.3% | 86.8% | 8/28 |
| 269 | SPY-CANARY-1m-Control | 6 | 9 | 5 | 1 | 4 | 50.0% | 83.3% | 20/28 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 53 | 51 | 46 | 7 | 5 | 79.3% | 86.8% | 9/28 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 49 | 49 | 48 | 1 | 1 | 96.0% | 98.0% | 6/28 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 49 | 14 | 14 | 35 | 0 | 28.6% | 28.6% | 23/28 |
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 12 | 12 | 0 | 0 | 100.0% | 100.0% | 2/28 |
| 277 | PACKTEST · Bollinger Bands · gate | 52 | 4 | 3 | 49 | 1 | 5.7% | 5.8% | 26/28 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 0 | 85 | 0 | 0 | 85 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 0 | 85 | 0 | 0 | 85 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 16 | 16 | 14 | 2 | 2 | 77.8% | 87.5% | 11/28 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 111 | 109 | 89 | 22 | 14 | 71.2% | 80.2% | 14/28 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 52 | 18 | 15 | 37 | 3 | 27.3% | 28.8% | 24/28 |
| 286 | PACKTEST · MACD Line v2 · trigger | 17 | 17 | 17 | 0 | 0 | 100.0% | 100.0% | 1/28 |
| 287 | PACKTEST · MACD Line v2 · gate | 52 | 0 | 0 | 52 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 42 | 42 | 34 | 8 | 7 | 69.4% | 81.0% | 16/28 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 79 | 75 | 61 | 18 | 10 | 68.5% | 77.2% | 17/28 |
| 291 | PACKTEST · Relative Volume v2 · gate | 52 | 0 | 0 | 52 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 31 | 27 | 17 | 14 | 10 | 41.5% | 54.8% | 21/28 |
| 293 | PACKTEST · Support Resistance Channels · gate | 52 | 0 | 0 | 52 | 0 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 23 | 23 | 21 | 2 | 1 | 87.5% | 91.3% | 7/28 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 52 | 0 | 0 | 52 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 136 | 133 | 107 | 29 | 15 | 70.9% | 78.7% | 15/28 |
| 297 | PACKTEST · Strat Assistant · gate | 52 | 6 | 4 | 48 | 2 | 7.4% | 7.7% | 25/28 |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 10 | 10 | 0 | 0 | 100.0% | 100.0% | 3/28 |
| 299 | PACKTEST · SuperTrend · gate | 53 | 41 | 36 | 17 | 4 | 63.2% | 67.9% | 18/28 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 57 | 55 | 49 | 8 | 6 | 77.8% | 86.0% | 10/28 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 53 | 51 | 45 | 8 | 5 | 77.6% | 84.9% | 12/28 |
| 302 | PACKTEST · UT Bot V4 · trigger | 53 | 51 | 45 | 8 | 5 | 77.6% | 84.9% | 13/28 |
| 303 | PACKTEST · UT Bot V4 · gate | 55 | 30 | 23 | 32 | 6 | 37.7% | 41.8% | 22/28 |
| 304 | PACKTEST · VWAP v2 · trigger | 16 | 30 | 2 | 14 | 28 | 4.5% | 12.5% | 27/28 |

---

## Hour 2026-06-03T15 UTC

_Strategies with activity this hour: 34; ranked (alerts≥1, BT≥1): 28_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 44 | 47 | 40 | 4 | 7 | 78.4% | 90.9% | 12/28 |
| 265 | TSLA-CANARY-1m-Control | 4 | 6 | 4 | 0 | 2 | 66.7% | 100.0% | 20/28 |
| 266 | TSLA-CANARY-5m-Control | 1 | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 1/28 |
| 267 | TSLA-CANARY-10s-LooseConf | 44 | 47 | 40 | 4 | 7 | 78.4% | 90.9% | 13/28 |
| 268 | SPY-CANARY-10s-NoConf | 47 | 47 | 44 | 3 | 3 | 88.0% | 93.6% | 8/28 |
| 269 | SPY-CANARY-1m-Control | 12 | 10 | 9 | 3 | 1 | 69.2% | 75.0% | 17/28 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 47 | 47 | 44 | 3 | 3 | 88.0% | 93.6% | 9/28 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 44 | 47 | 40 | 4 | 7 | 78.4% | 90.9% | 14/28 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 44 | 16 | 15 | 29 | 1 | 33.3% | 34.1% | 22/28 |
| 276 | PACKTEST · Bollinger Bands · trigger | 18 | 18 | 16 | 2 | 2 | 80.0% | 88.9% | 11/28 |
| 277 | PACKTEST · Bollinger Bands · gate | 47 | 0 | 0 | 47 | 0 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 2 | 73 | 1 | 1 | 72 | 1.4% | 50.0% | 27/28 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 2 | 73 | 1 | 1 | 72 | 1.4% | 50.0% | 28/28 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 16 | 16 | 15 | 1 | 1 | 88.2% | 93.8% | 6/28 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 91 | 91 | 79 | 12 | 12 | 76.7% | 86.8% | 15/28 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 47 | 2 | 2 | 45 | 0 | 4.3% | 4.3% | 25/28 |
| 286 | PACKTEST · MACD Line v2 · trigger | 22 | 22 | 21 | 1 | 1 | 91.3% | 95.5% | 5/28 |
| 287 | PACKTEST · MACD Line v2 · gate | 47 | 0 | 0 | 47 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 43 | 43 | 38 | 5 | 4 | 80.9% | 88.4% | 10/28 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 81 | 79 | 63 | 18 | 13 | 67.0% | 77.8% | 19/28 |
| 291 | PACKTEST · Relative Volume v2 · gate | 47 | 0 | 0 | 47 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 44 | 24 | 13 | 31 | 12 | 23.2% | 29.5% | 23/28 |
| 293 | PACKTEST · Support Resistance Channels · gate | 47 | 0 | 0 | 47 | 0 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 26 | 26 | 25 | 1 | 1 | 92.6% | 96.2% | 2/28 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 47 | 0 | 0 | 47 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 101 | 103 | 81 | 20 | 18 | 68.1% | 80.2% | 18/28 |
| 297 | PACKTEST · Strat Assistant · gate | 47 | 2 | 2 | 45 | 0 | 4.3% | 4.3% | 26/28 |
| 298 | PACKTEST · SuperTrend · trigger | 16 | 16 | 15 | 1 | 1 | 88.2% | 93.8% | 7/28 |
| 299 | PACKTEST · SuperTrend · gate | 47 | 23 | 22 | 25 | 1 | 45.8% | 46.8% | 21/28 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 66 | 64 | 55 | 11 | 6 | 76.4% | 83.3% | 16/28 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 47 | 47 | 45 | 2 | 2 | 91.8% | 95.7% | 3/28 |
| 302 | PACKTEST · UT Bot V4 · trigger | 47 | 47 | 45 | 2 | 2 | 91.8% | 95.7% | 4/28 |
| 303 | PACKTEST · UT Bot V4 · gate | 54 | 7 | 5 | 49 | 2 | 8.9% | 9.3% | 24/28 |

---

## Hour 2026-06-03T16 UTC

_Strategies with activity this hour: 33; ranked (alerts≥1, BT≥1): 27_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 43 | 47 | 38 | 5 | 9 | 73.1% | 88.4% | 6/27 |
| 265 | TSLA-CANARY-1m-Control | 13 | 13 | 11 | 2 | 2 | 73.3% | 84.6% | 5/27 |
| 267 | TSLA-CANARY-10s-LooseConf | 39 | 47 | 34 | 5 | 13 | 65.4% | 87.2% | 16/27 |
| 268 | SPY-CANARY-10s-NoConf | 46 | 52 | 39 | 7 | 13 | 66.1% | 84.8% | 14/27 |
| 269 | SPY-CANARY-1m-Control | 11 | 11 | 9 | 2 | 2 | 69.2% | 81.8% | 11/27 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 46 | 52 | 39 | 7 | 13 | 66.1% | 84.8% | 15/27 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 43 | 47 | 38 | 5 | 9 | 73.1% | 88.4% | 7/27 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 26 | 0 | 0 | 26 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 11 | 11 | 9 | 2 | 2 | 69.2% | 81.8% | 12/27 |
| 277 | PACKTEST · Bollinger Bands · gate | 28 | 0 | 0 | 28 | 0 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 9 | 61 | 8 | 1 | 52 | 13.1% | 88.9% | 25/27 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 8 | 61 | 7 | 1 | 53 | 11.5% | 87.5% | 26/27 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 11 | 11 | 8 | 3 | 3 | 57.1% | 72.7% | 19/27 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 90 | 94 | 76 | 14 | 16 | 71.7% | 84.4% | 10/27 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 40 | 16 | 15 | 25 | 1 | 36.6% | 37.5% | 21/27 |
| 286 | PACKTEST · MACD Line v2 · trigger | 26 | 28 | 21 | 5 | 8 | 61.8% | 80.8% | 17/27 |
| 287 | PACKTEST · MACD Line v2 · gate | 28 | 0 | 0 | 28 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 36 | 38 | 27 | 9 | 11 | 57.4% | 75.0% | 18/27 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 74 | 72 | 61 | 13 | 10 | 72.6% | 82.4% | 8/27 |
| 291 | PACKTEST · Relative Volume v2 · gate | 28 | 2 | 0 | 28 | 2 | 0.0% | 0.0% | 27/27 |
| 292 | PACKTEST · Support Resistance Channels · trig | 47 | 37 | 16 | 31 | 23 | 22.9% | 34.0% | 24/27 |
| 293 | PACKTEST · Support Resistance Channels · gate | 28 | 12 | 11 | 17 | 1 | 37.9% | 39.3% | 20/27 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 22 | 24 | 21 | 1 | 3 | 84.0% | 95.5% | 1/27 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 28 | 0 | 0 | 28 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 100 | 106 | 90 | 10 | 17 | 76.9% | 90.0% | 4/27 |
| 297 | PACKTEST · Strat Assistant · gate | 30 | 10 | 9 | 21 | 1 | 29.0% | 30.0% | 22/27 |
| 298 | PACKTEST · SuperTrend · trigger | 11 | 11 | 10 | 1 | 1 | 83.3% | 90.9% | 2/27 |
| 299 | PACKTEST · SuperTrend · gate | 28 | 0 | 0 | 28 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 52 | 57 | 48 | 4 | 8 | 80.0% | 92.3% | 3/27 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 44 | 52 | 39 | 5 | 13 | 68.4% | 88.6% | 13/27 |
| 302 | PACKTEST · UT Bot V4 · trigger | 46 | 52 | 41 | 5 | 11 | 71.9% | 89.1% | 9/27 |
| 303 | PACKTEST · UT Bot V4 · gate | 40 | 14 | 12 | 28 | 2 | 28.6% | 30.0% | 23/27 |

---

## Hour 2026-06-03T17 UTC

_Strategies with activity this hour: 37; ranked (alerts≥1, BT≥1): 35_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 46 | 46 | 41 | 5 | 5 | 80.4% | 89.1% | 12/35 |
| 265 | TSLA-CANARY-1m-Control | 6 | 6 | 5 | 1 | 1 | 71.4% | 83.3% | 23/35 |
| 266 | TSLA-CANARY-5m-Control | 1 | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 5/35 |
| 267 | TSLA-CANARY-10s-LooseConf | 44 | 46 | 39 | 5 | 7 | 76.5% | 88.6% | 19/35 |
| 268 | SPY-CANARY-10s-NoConf | 45 | 46 | 44 | 1 | 2 | 93.6% | 97.8% | 6/35 |
| 269 | SPY-CANARY-1m-Control | 6 | 8 | 5 | 1 | 3 | 55.6% | 83.3% | 28/35 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 45 | 46 | 44 | 1 | 2 | 93.6% | 97.8% | 7/35 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 4 | 4 | 4 | 0 | 0 | 100.0% | 100.0% | 1/35 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 46 | 46 | 41 | 5 | 5 | 80.4% | 89.1% | 13/35 |
| 276 | PACKTEST · Bollinger Bands · trigger | 13 | 13 | 12 | 1 | 1 | 85.7% | 92.3% | 9/35 |
| 277 | PACKTEST · Bollinger Bands · gate | 33 | 24 | 21 | 12 | 3 | 58.3% | 63.6% | 27/35 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 5 | 59 | 3 | 2 | 56 | 4.9% | 60.0% | 34/35 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 4 | 4 | 4 | 0 | 0 | 100.0% | 100.0% | 2/35 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 5 | 59 | 3 | 2 | 56 | 4.9% | 60.0% | 35/35 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 4 | 4 | 4 | 0 | 0 | 100.0% | 100.0% | 3/35 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 17 | 17 | 15 | 2 | 2 | 78.9% | 88.2% | 15/35 |
| 283 | PACKTEST · EMA Stack v2 · gate | 4 | 4 | 4 | 0 | 0 | 100.0% | 100.0% | 4/35 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 85 | 87 | 69 | 16 | 16 | 68.3% | 81.2% | 25/35 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 39 | 20 | 18 | 21 | 2 | 43.9% | 46.2% | 30/35 |
| 286 | PACKTEST · MACD Line v2 · trigger | 26 | 29 | 24 | 2 | 5 | 77.4% | 92.3% | 18/35 |
| 287 | PACKTEST · MACD Line v2 · gate | 23 | 18 | 16 | 7 | 2 | 64.0% | 69.6% | 26/35 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 37 | 36 | 33 | 4 | 3 | 82.5% | 89.2% | 11/35 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 79 | 77 | 68 | 11 | 8 | 78.2% | 86.1% | 16/35 |
| 292 | PACKTEST · Support Resistance Channels · trig | 33 | 28 | 16 | 17 | 15 | 33.3% | 48.5% | 31/35 |
| 293 | PACKTEST · Support Resistance Channels · gate | 33 | 20 | 17 | 16 | 3 | 47.2% | 51.5% | 29/35 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 15 | 18 | 15 | 0 | 3 | 83.3% | 100.0% | 10/35 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 19 | 6 | 5 | 14 | 1 | 25.0% | 26.3% | 33/35 |
| 296 | PACKTEST · Strat Assistant · trigger | 91 | 95 | 77 | 14 | 17 | 71.3% | 84.6% | 24/35 |
| 297 | PACKTEST · Strat Assistant · gate | 39 | 12 | 12 | 27 | 0 | 30.8% | 30.8% | 32/35 |
| 298 | PACKTEST · SuperTrend · trigger | 15 | 15 | 14 | 1 | 1 | 87.5% | 93.3% | 8/35 |
| 299 | PACKTEST · SuperTrend · gate | 43 | 46 | 39 | 4 | 7 | 78.0% | 90.7% | 17/35 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 58 | 59 | 49 | 9 | 9 | 73.1% | 84.5% | 20/35 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 40 | 46 | 36 | 4 | 10 | 72.0% | 90.0% | 22/35 |
| 302 | PACKTEST · UT Bot V4 · trigger | 44 | 46 | 40 | 4 | 6 | 80.0% | 90.9% | 14/35 |
| 303 | PACKTEST · UT Bot V4 · gate | 38 | 31 | 29 | 9 | 2 | 72.5% | 76.3% | 21/35 |
| 304 | PACKTEST · VWAP v2 · trigger | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-03T18 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 36_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 44 | 47 | 41 | 3 | 6 | 82.0% | 93.2% | 4/36 |
| 265 | TSLA-CANARY-1m-Control | 5 | 8 | 4 | 1 | 4 | 44.4% | 80.0% | 23/36 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 43 | 47 | 40 | 3 | 7 | 80.0% | 93.0% | 6/36 |
| 268 | SPY-CANARY-10s-NoConf | 52 | 52 | 47 | 5 | 5 | 82.5% | 90.4% | 2/36 |
| 269 | SPY-CANARY-1m-Control | 5 | 7 | 4 | 1 | 3 | 50.0% | 80.0% | 22/36 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 52 | 52 | 47 | 5 | 5 | 82.5% | 90.4% | 3/36 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 50 | 16 | 13 | 37 | 3 | 24.5% | 26.0% | 29/36 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 43 | 47 | 40 | 3 | 7 | 80.0% | 93.0% | 7/36 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 31 | 12 | 12 | 19 | 0 | 38.7% | 38.7% | 27/36 |
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 12 | 12 | 0 | 0 | 100.0% | 100.0% | 1/36 |
| 277 | PACKTEST · Bollinger Bands · gate | 40 | 18 | 15 | 25 | 2 | 35.7% | 37.5% | 28/36 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 9 | 68 | 7 | 2 | 61 | 10.0% | 77.8% | 33/36 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 46 | 26 | 21 | 25 | 4 | 42.0% | 45.7% | 24/36 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 9 | 68 | 7 | 2 | 61 | 10.0% | 77.8% | 34/36 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 46 | 26 | 21 | 25 | 4 | 42.0% | 45.7% | 25/36 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 20 | 24 | 19 | 1 | 5 | 76.0% | 95.0% | 9/36 |
| 283 | PACKTEST · EMA Stack v2 · gate | 50 | 52 | 43 | 7 | 8 | 74.1% | 86.0% | 11/36 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 98 | 107 | 79 | 19 | 25 | 64.2% | 80.6% | 15/36 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 39 | 6 | 5 | 34 | 1 | 12.5% | 12.8% | 32/36 |
| 286 | PACKTEST · MACD Line v2 · trigger | 29 | 28 | 23 | 6 | 5 | 67.6% | 79.3% | 14/36 |
| 287 | PACKTEST · MACD Line v2 · gate | 37 | 20 | 16 | 21 | 3 | 40.0% | 43.2% | 26/36 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 49 | 51 | 36 | 13 | 15 | 56.2% | 73.5% | 20/36 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 79 | 81 | 70 | 9 | 8 | 80.5% | 88.6% | 5/36 |
| 291 | PACKTEST · Relative Volume v2 · gate | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 26 | 21 | 4 | 22 | 16 | 9.5% | 15.4% | 35/36 |
| 293 | PACKTEST · Support Resistance Channels · gate | 43 | 48 | 35 | 8 | 12 | 63.6% | 81.4% | 17/36 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 17 | 19 | 15 | 2 | 4 | 71.4% | 88.2% | 12/36 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 30 | 8 | 7 | 23 | 1 | 22.6% | 23.3% | 30/36 |
| 296 | PACKTEST · Strat Assistant · trigger | 120 | 124 | 98 | 22 | 19 | 70.5% | 81.7% | 13/36 |
| 297 | PACKTEST · Strat Assistant · gate | 43 | 12 | 10 | 33 | 2 | 22.2% | 23.3% | 31/36 |
| 298 | PACKTEST · SuperTrend · trigger | 8 | 10 | 8 | 0 | 2 | 80.0% | 100.0% | 8/36 |
| 299 | PACKTEST · SuperTrend · gate | 43 | 52 | 36 | 7 | 15 | 62.1% | 83.7% | 18/36 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 30 | 32 | 26 | 4 | 5 | 74.3% | 86.7% | 10/36 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 44 | 52 | 37 | 7 | 14 | 63.8% | 84.1% | 16/36 |
| 302 | PACKTEST · UT Bot V4 · trigger | 45 | 52 | 36 | 9 | 15 | 60.0% | 80.0% | 19/36 |
| 303 | PACKTEST · UT Bot V4 · gate | 43 | 30 | 25 | 18 | 4 | 53.2% | 58.1% | 21/36 |
| 304 | PACKTEST · VWAP v2 · trigger | 24 | 16 | 2 | 22 | 13 | 5.4% | 8.3% | 36/36 |

---

## Hour 2026-06-03T19 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 36_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 4 | 4 | 2 | 2 | 2 | 33.3% | 50.0% | 23/36 |
| 174 | TSLA LONG 1Min Mass #2 | 9 | 0 | 0 | 9 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 34 | 41 | 30 | 4 | 11 | 66.7% | 88.2% | 6/36 |
| 265 | TSLA-CANARY-1m-Control | 9 | 10 | 7 | 2 | 3 | 58.3% | 77.8% | 17/36 |
| 266 | TSLA-CANARY-5m-Control | 2 | 2 | 1 | 1 | 1 | 33.3% | 50.0% | 24/36 |
| 267 | TSLA-CANARY-10s-LooseConf | 34 | 41 | 30 | 4 | 11 | 66.7% | 88.2% | 7/36 |
| 268 | SPY-CANARY-10s-NoConf | 49 | 52 | 40 | 9 | 12 | 65.6% | 81.6% | 11/36 |
| 269 | SPY-CANARY-1m-Control | 10 | 8 | 7 | 3 | 1 | 63.6% | 70.0% | 15/36 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 48 | 52 | 39 | 9 | 13 | 63.9% | 81.2% | 14/36 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 45 | 14 | 10 | 35 | 4 | 20.4% | 22.2% | 26/36 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 49 | 44 | 33 | 16 | 11 | 55.0% | 67.3% | 21/36 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 34 | 41 | 30 | 4 | 11 | 66.7% | 88.2% | 8/36 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 34 | 41 | 30 | 4 | 11 | 66.7% | 88.2% | 9/36 |
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 10 | 8 | 4 | 2 | 57.1% | 66.7% | 18/36 |
| 277 | PACKTEST · Bollinger Bands · gate | 49 | 6 | 5 | 44 | 1 | 10.0% | 10.2% | 29/36 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 15 | 70 | 11 | 4 | 58 | 15.1% | 73.3% | 27/36 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 49 | 0 | 0 | 49 | 0 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 15 | 70 | 11 | 4 | 58 | 15.1% | 73.3% | 28/36 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 49 | 0 | 0 | 49 | 0 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 14 | 14 | 11 | 3 | 3 | 64.7% | 78.6% | 13/36 |
| 283 | PACKTEST · EMA Stack v2 · gate | 47 | 2 | 2 | 45 | 0 | 4.3% | 4.3% | 35/36 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 110 | 120 | 91 | 19 | 27 | 66.4% | 82.7% | 10/36 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 47 | 6 | 4 | 43 | 2 | 8.2% | 8.5% | 31/36 |
| 286 | PACKTEST · MACD Line v2 · trigger | 21 | 22 | 17 | 4 | 5 | 65.4% | 81.0% | 12/36 |
| 287 | PACKTEST · MACD Line v2 · gate | 44 | 0 | 0 | 44 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 25 | 28 | 19 | 6 | 9 | 55.9% | 76.0% | 20/36 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 72 | 79 | 53 | 19 | 21 | 57.0% | 73.6% | 19/36 |
| 291 | PACKTEST · Relative Volume v2 · gate | 45 | 2 | 0 | 45 | 2 | 0.0% | 0.0% | 36/36 |
| 292 | PACKTEST · Support Resistance Channels · trig | 44 | 32 | 16 | 28 | 18 | 25.8% | 36.4% | 25/36 |
| 293 | PACKTEST · Support Resistance Channels · gate | 47 | 26 | 20 | 27 | 6 | 37.7% | 42.6% | 22/36 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 26 | 23 | 20 | 6 | 3 | 69.0% | 76.9% | 2/36 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 48 | 4 | 4 | 44 | 0 | 8.3% | 8.3% | 30/36 |
| 296 | PACKTEST · Strat Assistant · trigger | 98 | 101 | 83 | 15 | 18 | 71.6% | 84.7% | 1/36 |
| 297 | PACKTEST · Strat Assistant · gate | 46 | 6 | 3 | 43 | 3 | 6.1% | 6.5% | 33/36 |
| 298 | PACKTEST · SuperTrend · trigger | 12 | 12 | 9 | 3 | 3 | 60.0% | 75.0% | 16/36 |
| 299 | PACKTEST · SuperTrend · gate | 46 | 2 | 2 | 44 | 0 | 4.3% | 4.3% | 34/36 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 47 | 48 | 38 | 9 | 9 | 67.9% | 80.9% | 3/36 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 48 | 52 | 40 | 8 | 12 | 66.7% | 83.3% | 4/36 |
| 302 | PACKTEST · UT Bot V4 · trigger | 48 | 52 | 40 | 8 | 12 | 66.7% | 83.3% | 5/36 |
| 303 | PACKTEST · UT Bot V4 · gate | 38 | 6 | 3 | 35 | 2 | 7.5% | 7.9% | 32/36 |

---

## Hour 2026-06-03T20 UTC

_Strategies with activity this hour: 32; ranked (alerts≥1, BT≥1): 18_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 268 | SPY-CANARY-10s-NoConf | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 7 | 8 | 4 | 3 | 4 | 36.4% | 57.1% | 7/18 |
| 277 | PACKTEST · Bollinger Bands · gate | 34 | 0 | 0 | 34 | 0 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 42 | 61 | 25 | 17 | 34 | 32.9% | 59.5% | 8/18 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 34 | 0 | 0 | 34 | 0 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 42 | 61 | 25 | 17 | 34 | 32.9% | 59.5% | 9/18 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 34 | 0 | 0 | 34 | 0 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 6 | 7 | 2 | 4 | 5 | 18.2% | 33.3% | 14/18 |
| 283 | PACKTEST · EMA Stack v2 · gate | 34 | 0 | 0 | 34 | 0 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 74 | 80 | 43 | 31 | 38 | 38.4% | 58.1% | 6/18 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 34 | 0 | 0 | 34 | 0 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 22 | 35 | 11 | 11 | 24 | 23.9% | 50.0% | 12/18 |
| 287 | PACKTEST · MACD Line v2 · gate | 34 | 0 | 0 | 34 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 34 | 23 | 12 | 22 | 11 | 26.7% | 35.3% | 11/18 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 66 | 69 | 45 | 21 | 23 | 50.6% | 68.2% | 2/18 |
| 291 | PACKTEST · Relative Volume v2 · gate | 31 | 4 | 2 | 29 | 2 | 6.1% | 6.5% | 17/18 |
| 292 | PACKTEST · Support Resistance Channels · trig | 6 | 18 | 3 | 3 | 14 | 15.0% | 50.0% | 15/18 |
| 293 | PACKTEST · Support Resistance Channels · gate | 30 | 0 | 0 | 30 | 0 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 25 | 26 | 11 | 14 | 15 | 27.5% | 44.0% | 10/18 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 28 | 0 | 0 | 28 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 87 | 92 | 55 | 32 | 37 | 44.4% | 63.2% | 4/18 |
| 297 | PACKTEST · Strat Assistant · gate | 29 | 16 | 5 | 24 | 11 | 12.5% | 17.2% | 16/18 |
| 298 | PACKTEST · SuperTrend · trigger | 17 | 11 | 5 | 12 | 6 | 21.7% | 29.4% | 13/18 |
| 299 | PACKTEST · SuperTrend · gate | 30 | 0 | 0 | 30 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 26 | 30 | 20 | 6 | 11 | 54.1% | 76.9% | 1/18 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 50 | 54 | 29 | 21 | 25 | 38.7% | 58.0% | 5/18 |
| 302 | PACKTEST · UT Bot V4 · trigger | 59 | 54 | 37 | 22 | 17 | 48.7% | 62.7% | 3/18 |
| 303 | PACKTEST · UT Bot V4 · gate | 32 | 6 | 2 | 30 | 4 | 5.6% | 6.2% | 18/18 |
| 304 | PACKTEST · VWAP v2 · trigger | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-03T21 UTC

_Strategies with activity this hour: 21; ranked (alerts≥1, BT≥1): 18_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 11 | 7 | 5 | 4 | 43.8% | 58.3% | 10/18 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 64 | 64 | 54 | 10 | 10 | 73.0% | 84.4% | 1/18 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 64 | 64 | 54 | 10 | 10 | 73.0% | 84.4% | 2/18 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 9 | 8 | 2 | 7 | 6 | 13.3% | 22.2% | 17/18 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 114 | 93 | 69 | 45 | 23 | 50.4% | 60.5% | 6/18 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 16 | 17 | 2 | 14 | 15 | 6.5% | 12.5% | 18/18 |
| 286 | PACKTEST · MACD Line v2 · trigger | 29 | 16 | 8 | 21 | 8 | 21.6% | 27.6% | 16/18 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 45 | 40 | 30 | 15 | 10 | 54.5% | 66.7% | 4/18 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 94 | 80 | 53 | 41 | 23 | 45.3% | 56.4% | 9/18 |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 14 | 20 | 7 | 7 | 14 | 25.0% | 50.0% | 15/18 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 23 | 16 | 13 | 10 | 4 | 48.1% | 56.5% | 7/18 |
| 296 | PACKTEST · Strat Assistant · trigger | 88 | 97 | 64 | 24 | 33 | 52.9% | 72.7% | 5/18 |
| 297 | PACKTEST · Strat Assistant · gate | 16 | 12 | 8 | 8 | 4 | 40.0% | 50.0% | 12/18 |
| 298 | PACKTEST · SuperTrend · trigger | 20 | 13 | 9 | 11 | 4 | 37.5% | 45.0% | 13/18 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 31 | 27 | 22 | 9 | 5 | 61.1% | 71.0% | 3/18 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 31 | 45 | 19 | 12 | 26 | 33.3% | 61.3% | 14/18 |
| 302 | PACKTEST · UT Bot V4 · trigger | 53 | 45 | 31 | 22 | 13 | 47.0% | 58.5% | 8/18 |
| 303 | PACKTEST · UT Bot V4 · gate | 12 | 10 | 7 | 5 | 4 | 43.8% | 58.3% | 11/18 |
| 304 | PACKTEST · VWAP v2 · trigger | 16 | 0 | 0 | 16 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-03T22 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 20_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 6 | 4 | 1 | 5 | 3 | 11.1% | 16.7% | 14/20 |
| 277 | PACKTEST · Bollinger Bands · gate | 1 | 11 | 1 | 0 | 10 | 9.1% | 100.0% | 15/20 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 37 | 38 | 20 | 17 | 18 | 36.4% | 54.1% | 6/20 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 37 | 38 | 20 | 17 | 18 | 36.4% | 54.1% | 7/20 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 14 | 6 | 0 | 14 | 6 | 0.0% | 0.0% | 19/20 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 49 | 49 | 20 | 29 | 28 | 26.0% | 40.8% | 9/20 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 17 | 10 | 1 | 16 | 9 | 3.8% | 5.9% | 17/20 |
| 286 | PACKTEST · MACD Line v2 · trigger | 27 | 15 | 3 | 24 | 12 | 7.7% | 11.1% | 16/20 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 18 | 26 | 10 | 8 | 15 | 30.3% | 55.6% | 8/20 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 80 | 68 | 41 | 39 | 28 | 38.0% | 51.2% | 5/20 |
| 291 | PACKTEST · Relative Volume v2 · gate | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 8 | 19 | 0 | 8 | 19 | 0.0% | 0.0% | 20/20 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 20 | 14 | 5 | 15 | 9 | 17.2% | 25.0% | 13/20 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 71 | 72 | 49 | 22 | 21 | 53.3% | 69.0% | 2/20 |
| 297 | PACKTEST · Strat Assistant · gate | 8 | 8 | 3 | 5 | 5 | 23.1% | 37.5% | 11/20 |
| 298 | PACKTEST · SuperTrend · trigger | 27 | 16 | 7 | 20 | 9 | 19.4% | 25.9% | 12/20 |
| 299 | PACKTEST · SuperTrend · gate | 1 | 31 | 1 | 0 | 30 | 3.2% | 100.0% | 18/20 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 10 | 12 | 9 | 1 | 3 | 69.2% | 90.0% | 1/20 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 28 | 34 | 12 | 16 | 21 | 24.5% | 42.9% | 10/20 |
| 302 | PACKTEST · UT Bot V4 · trigger | 45 | 34 | 22 | 23 | 11 | 39.3% | 48.9% | 4/20 |
| 303 | PACKTEST · UT Bot V4 · gate | 8 | 13 | 7 | 1 | 6 | 50.0% | 87.5% | 3/20 |
| 304 | PACKTEST · VWAP v2 · trigger | 17 | 0 | 0 | 17 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-03T23 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 21_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 9 | 2 | 10 | 7 | 10.5% | 16.7% | 18/21 |
| 277 | PACKTEST · Bollinger Bands · gate | 5 | 5 | 1 | 4 | 4 | 11.1% | 20.0% | 17/21 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 39 | 37 | 19 | 20 | 18 | 33.3% | 48.7% | 4/21 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 39 | 37 | 19 | 20 | 18 | 33.3% | 48.7% | 5/21 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 12 | 8 | 1 | 11 | 7 | 5.3% | 8.3% | 20/21 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 83 | 52 | 21 | 62 | 31 | 18.4% | 25.3% | 10/21 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 17 | 5 | 1 | 16 | 4 | 4.8% | 5.9% | 21/21 |
| 286 | PACKTEST · MACD Line v2 · trigger | 22 | 10 | 2 | 20 | 8 | 6.7% | 9.1% | 19/21 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 25 | 19 | 5 | 20 | 15 | 12.5% | 20.0% | 16/21 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 126 | 50 | 33 | 93 | 18 | 22.9% | 26.2% | 7/21 |
| 291 | PACKTEST · Relative Volume v2 · gate | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 14 | 27 | 5 | 9 | 22 | 13.9% | 35.7% | 14/21 |
| 293 | PACKTEST · Support Resistance Channels · gate | 14 | 8 | 3 | 11 | 5 | 15.8% | 21.4% | 11/21 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 31 | 16 | 8 | 23 | 9 | 20.0% | 25.8% | 9/21 |
| 296 | PACKTEST · Strat Assistant · trigger | 59 | 65 | 47 | 12 | 18 | 61.0% | 79.7% | 1/21 |
| 297 | PACKTEST · Strat Assistant · gate | 4 | 4 | 1 | 3 | 3 | 14.3% | 25.0% | 13/21 |
| 298 | PACKTEST · SuperTrend · trigger | 28 | 14 | 5 | 23 | 9 | 13.5% | 17.9% | 15/21 |
| 299 | PACKTEST · SuperTrend · gate | 11 | 5 | 2 | 9 | 3 | 14.3% | 18.2% | 12/21 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 4 | 6 | 3 | 1 | 3 | 42.9% | 75.0% | 2/21 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 21 | 36 | 10 | 11 | 26 | 21.3% | 47.6% | 8/21 |
| 302 | PACKTEST · UT Bot V4 · trigger | 45 | 36 | 21 | 24 | 15 | 35.0% | 46.7% | 3/21 |
| 303 | PACKTEST · UT Bot V4 · gate | 13 | 3 | 3 | 10 | 0 | 23.1% | 23.1% | 6/21 |
| 304 | PACKTEST · VWAP v2 · trigger | 7 | 0 | 0 | 7 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-04T08 UTC

_Strategies with activity this hour: 28; ranked (alerts≥1, BT≥1): 21_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 9 | 9 | 5 | 4 | 4 | 38.5% | 55.6% | 5/21 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 47 | 47 | 33 | 14 | 13 | 55.0% | 70.2% | 1/21 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 47 | 47 | 33 | 14 | 13 | 55.0% | 70.2% | 2/21 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 11 | 8 | 0 | 11 | 8 | 0.0% | 0.0% | 19/21 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 28 | 0 | 0 | 28 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 78 | 51 | 24 | 54 | 26 | 23.1% | 30.8% | 11/21 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 4 | 2 | 0 | 4 | 2 | 0.0% | 0.0% | 20/21 |
| 286 | PACKTEST · MACD Line v2 · trigger | 24 | 16 | 5 | 19 | 11 | 14.3% | 20.8% | 15/21 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 23 | 21 | 5 | 18 | 16 | 12.8% | 21.7% | 16/21 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 94 | 64 | 37 | 57 | 25 | 31.1% | 39.4% | 7/21 |
| 291 | PACKTEST · Relative Volume v2 · gate | 2 | 4 | 0 | 2 | 4 | 0.0% | 0.0% | 21/21 |
| 292 | PACKTEST · Support Resistance Channels · trig | 12 | 16 | 1 | 11 | 15 | 3.7% | 8.3% | 18/21 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 36 | 24 | 12 | 24 | 13 | 24.5% | 33.3% | 10/21 |
| 296 | PACKTEST · Strat Assistant · trigger | 76 | 78 | 49 | 27 | 29 | 46.7% | 64.5% | 3/21 |
| 297 | PACKTEST · Strat Assistant · gate | 6 | 4 | 2 | 4 | 2 | 25.0% | 33.3% | 9/21 |
| 298 | PACKTEST · SuperTrend · trigger | 19 | 17 | 8 | 11 | 9 | 28.6% | 42.1% | 8/21 |
| 299 | PACKTEST · SuperTrend · gate | 10 | 25 | 5 | 5 | 20 | 16.7% | 50.0% | 14/21 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 14 | 10 | 7 | 7 | 4 | 38.9% | 50.0% | 4/21 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 28 | 36 | 12 | 16 | 24 | 23.1% | 42.9% | 12/21 |
| 302 | PACKTEST · UT Bot V4 · trigger | 52 | 36 | 21 | 31 | 15 | 31.3% | 40.4% | 6/21 |
| 303 | PACKTEST · UT Bot V4 · gate | 4 | 14 | 2 | 2 | 12 | 12.5% | 50.0% | 17/21 |
| 304 | PACKTEST · VWAP v2 · trigger | 16 | 12 | 5 | 11 | 7 | 21.7% | 31.2% | 13/21 |
| 305 | PACKTEST · VWAP v2 · gate | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-04T09 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 24_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 15 | 11 | 7 | 8 | 4 | 36.8% | 46.7% | 11/24 |
| 277 | PACKTEST · Bollinger Bands · gate | 20 | 16 | 9 | 11 | 7 | 33.3% | 45.0% | 13/24 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 51 | 45 | 29 | 22 | 15 | 43.9% | 56.9% | 4/24 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 51 | 45 | 29 | 22 | 15 | 43.9% | 56.9% | 5/24 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 21 | 7 | 3 | 18 | 4 | 12.0% | 14.3% | 22/24 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 18 | 0 | 0 | 18 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 69 | 53 | 27 | 42 | 26 | 28.4% | 39.1% | 16/24 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 24 | 14 | 9 | 15 | 5 | 31.0% | 37.5% | 15/24 |
| 286 | PACKTEST · MACD Line v2 · trigger | 28 | 16 | 6 | 22 | 10 | 15.8% | 21.4% | 21/24 |
| 287 | PACKTEST · MACD Line v2 · gate | 18 | 18 | 9 | 9 | 9 | 33.3% | 50.0% | 14/24 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 31 | 21 | 17 | 14 | 5 | 47.2% | 54.8% | 3/24 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 112 | 60 | 49 | 63 | 13 | 39.2% | 43.8% | 7/24 |
| 292 | PACKTEST · Support Resistance Channels · trig | 11 | 20 | 8 | 3 | 12 | 34.8% | 72.7% | 12/24 |
| 293 | PACKTEST · Support Resistance Channels · gate | 4 | 8 | 2 | 2 | 6 | 20.0% | 50.0% | 20/24 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 14 | 10 | 4 | 10 | 6 | 20.0% | 28.6% | 19/24 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 16 | 10 | 7 | 9 | 3 | 36.8% | 43.8% | 10/24 |
| 296 | PACKTEST · Strat Assistant · trigger | 64 | 72 | 49 | 15 | 23 | 56.3% | 76.6% | 2/24 |
| 297 | PACKTEST · Strat Assistant · gate | 8 | 8 | 1 | 7 | 7 | 6.7% | 12.5% | 24/24 |
| 298 | PACKTEST · SuperTrend · trigger | 31 | 15 | 8 | 23 | 7 | 21.1% | 25.8% | 18/24 |
| 299 | PACKTEST · SuperTrend · gate | 30 | 17 | 13 | 17 | 4 | 38.2% | 43.3% | 9/24 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 14 | 15 | 13 | 1 | 2 | 81.2% | 92.9% | 1/24 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 28 | 47 | 14 | 14 | 33 | 23.0% | 50.0% | 17/24 |
| 302 | PACKTEST · UT Bot V4 · trigger | 58 | 47 | 32 | 26 | 15 | 43.8% | 55.2% | 6/24 |
| 303 | PACKTEST · UT Bot V4 · gate | 20 | 12 | 9 | 11 | 3 | 39.1% | 45.0% | 8/24 |
| 304 | PACKTEST · VWAP v2 · trigger | 5 | 5 | 1 | 4 | 4 | 11.1% | 20.0% | 23/24 |

---

## Hour 2026-06-04T10 UTC

_Strategies with activity this hour: 26; ranked (alerts≥1, BT≥1): 20_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 37 | 35 | 23 | 14 | 12 | 46.9% | 62.2% | 3/20 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 37 | 35 | 23 | 14 | 12 | 46.9% | 62.2% | 4/20 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 17 | 5 | 1 | 16 | 4 | 4.8% | 5.9% | 18/20 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 70 | 47 | 22 | 48 | 25 | 23.2% | 31.4% | 11/20 |
| 286 | PACKTEST · MACD Line v2 · trigger | 27 | 23 | 5 | 22 | 18 | 11.1% | 18.5% | 14/20 |
| 287 | PACKTEST · MACD Line v2 · gate | 6 | 4 | 3 | 3 | 0 | 50.0% | 50.0% | 2/20 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 25 | 9 | 3 | 22 | 6 | 9.7% | 12.0% | 15/20 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 86 | 34 | 26 | 60 | 7 | 28.0% | 30.2% | 10/20 |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 5 | 20 | 2 | 3 | 18 | 8.7% | 40.0% | 16/20 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 34 | 22 | 15 | 19 | 8 | 35.7% | 44.1% | 8/20 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 2 | 4 | 1 | 1 | 3 | 20.0% | 50.0% | 12/20 |
| 296 | PACKTEST · Strat Assistant · trigger | 51 | 53 | 33 | 18 | 20 | 46.5% | 64.7% | 5/20 |
| 297 | PACKTEST · Strat Assistant · gate | 10 | 9 | 2 | 8 | 7 | 11.8% | 20.0% | 13/20 |
| 298 | PACKTEST · SuperTrend · trigger | 31 | 10 | 3 | 28 | 7 | 7.9% | 9.7% | 17/20 |
| 299 | PACKTEST · SuperTrend · gate | 12 | 8 | 7 | 5 | 0 | 58.3% | 58.3% | 1/20 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 8 | 8 | 5 | 3 | 4 | 41.7% | 62.5% | 6/20 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 35 | 35 | 15 | 20 | 18 | 28.3% | 42.9% | 9/20 |
| 302 | PACKTEST · UT Bot V4 · trigger | 49 | 35 | 23 | 26 | 10 | 39.0% | 46.9% | 7/20 |
| 303 | PACKTEST · UT Bot V4 · gate | 6 | 2 | 0 | 6 | 2 | 0.0% | 0.0% | 19/20 |
| 304 | PACKTEST · VWAP v2 · trigger | 1 | 3 | 0 | 1 | 3 | 0.0% | 0.0% | 20/20 |

---

## Hour 2026-06-04T11 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 25_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 13 | 14 | 10 | 3 | 4 | 58.8% | 76.9% | 12/25 |
| 277 | PACKTEST · Bollinger Bands · gate | 16 | 24 | 14 | 2 | 10 | 53.8% | 87.5% | 15/25 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 63 | 63 | 52 | 11 | 8 | 73.2% | 82.5% | 4/25 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 63 | 63 | 52 | 11 | 8 | 73.2% | 82.5% | 5/25 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 17 | 17 | 15 | 2 | 2 | 78.9% | 88.2% | 2/25 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 88 | 79 | 66 | 22 | 12 | 66.0% | 75.0% | 9/25 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 24 | 22 | 15 | 9 | 7 | 48.4% | 62.5% | 16/25 |
| 286 | PACKTEST · MACD Line v2 · trigger | 23 | 21 | 14 | 9 | 7 | 46.7% | 60.9% | 19/25 |
| 287 | PACKTEST · MACD Line v2 · gate | 24 | 20 | 17 | 7 | 3 | 63.0% | 70.8% | 10/25 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 34 | 32 | 28 | 6 | 4 | 73.7% | 82.4% | 3/25 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 90 | 74 | 60 | 30 | 14 | 57.7% | 66.7% | 13/25 |
| 291 | PACKTEST · Relative Volume v2 · gate | 8 | 10 | 4 | 4 | 6 | 28.6% | 50.0% | 21/25 |
| 292 | PACKTEST · Support Resistance Channels · trig | 35 | 37 | 13 | 22 | 24 | 22.0% | 37.1% | 22/25 |
| 293 | PACKTEST · Support Resistance Channels · gate | 16 | 24 | 5 | 11 | 19 | 14.3% | 31.2% | 24/25 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 26 | 16 | 11 | 15 | 5 | 35.5% | 42.3% | 20/25 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 2 | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 1/25 |
| 296 | PACKTEST · Strat Assistant · trigger | 100 | 110 | 84 | 16 | 25 | 67.2% | 84.0% | 7/25 |
| 297 | PACKTEST · Strat Assistant · gate | 6 | 11 | 3 | 3 | 8 | 21.4% | 50.0% | 23/25 |
| 298 | PACKTEST · SuperTrend · trigger | 21 | 16 | 12 | 9 | 4 | 48.0% | 57.1% | 17/25 |
| 299 | PACKTEST · SuperTrend · gate | 42 | 36 | 28 | 14 | 8 | 56.0% | 66.7% | 14/25 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 25 | 31 | 23 | 2 | 7 | 71.9% | 92.0% | 6/25 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 39 | 49 | 28 | 11 | 21 | 46.7% | 71.8% | 18/25 |
| 302 | PACKTEST · UT Bot V4 · trigger | 55 | 49 | 39 | 16 | 10 | 60.0% | 70.9% | 11/25 |
| 303 | PACKTEST · UT Bot V4 · gate | 25 | 25 | 20 | 5 | 5 | 66.7% | 80.0% | 8/25 |
| 304 | PACKTEST · VWAP v2 · trigger | 16 | 7 | 2 | 14 | 5 | 9.5% | 12.5% | 25/25 |

---

## Hour 2026-06-04T12 UTC

_Strategies with activity this hour: 28; ranked (alerts≥1, BT≥1): 28_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 16 | 16 | 12 | 4 | 4 | 60.0% | 75.0% | 12/28 |
| 277 | PACKTEST · Bollinger Bands · gate | 36 | 30 | 22 | 14 | 8 | 50.0% | 61.1% | 15/28 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 77 | 73 | 58 | 19 | 14 | 63.7% | 75.3% | 7/28 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 57 | 50 | 29 | 28 | 21 | 37.2% | 50.9% | 20/28 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 77 | 73 | 58 | 19 | 14 | 63.7% | 75.3% | 8/28 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 57 | 50 | 29 | 28 | 21 | 37.2% | 50.9% | 21/28 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 25 | 19 | 15 | 10 | 4 | 51.7% | 60.0% | 13/28 |
| 283 | PACKTEST · EMA Stack v2 · gate | 73 | 69 | 56 | 17 | 13 | 65.1% | 76.7% | 5/28 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 92 | 91 | 73 | 19 | 17 | 67.0% | 79.3% | 3/28 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 22 | 28 | 11 | 11 | 17 | 28.2% | 50.0% | 23/28 |
| 286 | PACKTEST · MACD Line v2 · trigger | 31 | 31 | 17 | 14 | 13 | 38.6% | 54.8% | 19/28 |
| 287 | PACKTEST · MACD Line v2 · gate | 49 | 43 | 35 | 14 | 8 | 61.4% | 71.4% | 11/28 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 46 | 44 | 30 | 16 | 13 | 50.8% | 65.2% | 14/28 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 98 | 90 | 72 | 26 | 15 | 63.7% | 73.5% | 9/28 |
| 291 | PACKTEST · Relative Volume v2 · gate | 6 | 2 | 1 | 5 | 1 | 14.3% | 16.7% | 27/28 |
| 292 | PACKTEST · Support Resistance Channels · trig | 34 | 35 | 13 | 21 | 22 | 23.2% | 38.2% | 24/28 |
| 293 | PACKTEST · Support Resistance Channels · gate | 15 | 20 | 5 | 10 | 15 | 16.7% | 33.3% | 26/28 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 17 | 15 | 9 | 8 | 6 | 39.1% | 52.9% | 18/28 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 18 | 12 | 12 | 6 | 0 | 66.7% | 66.7% | 4/28 |
| 296 | PACKTEST · Strat Assistant · trigger | 92 | 101 | 74 | 18 | 25 | 63.2% | 80.4% | 10/28 |
| 297 | PACKTEST · Strat Assistant · gate | 16 | 16 | 5 | 11 | 11 | 18.5% | 31.2% | 25/28 |
| 298 | PACKTEST · SuperTrend · trigger | 25 | 19 | 13 | 12 | 6 | 41.9% | 52.0% | 17/28 |
| 299 | PACKTEST · SuperTrend · gate | 75 | 69 | 58 | 17 | 11 | 67.4% | 77.3% | 1/28 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 39 | 39 | 30 | 9 | 8 | 63.8% | 76.9% | 6/28 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 40 | 69 | 27 | 13 | 42 | 32.9% | 67.5% | 22/28 |
| 302 | PACKTEST · UT Bot V4 · trigger | 75 | 69 | 58 | 17 | 11 | 67.4% | 77.3% | 2/28 |
| 303 | PACKTEST · UT Bot V4 · gate | 38 | 31 | 22 | 16 | 8 | 47.8% | 57.9% | 16/28 |
| 304 | PACKTEST · VWAP v2 · trigger | 11 | 29 | 4 | 7 | 25 | 11.1% | 36.4% | 28/28 |

---

## Hour 2026-06-04T13 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 37_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 1 | 2 | 0 | 1 | 2 | 0.0% | 0.0% | 37/37 |
| 263 | TSLA-CANARY-10s-NoConf | 14 | 14 | 11 | 3 | 3 | 64.7% | 78.6% | 16/37 |
| 265 | TSLA-CANARY-1m-Control | 3 | 4 | 1 | 2 | 3 | 16.7% | 33.3% | 33/37 |
| 266 | TSLA-CANARY-5m-Control | 2 | 1 | 0 | 2 | 1 | 0.0% | 0.0% | 36/37 |
| 267 | TSLA-CANARY-10s-LooseConf | 10 | 14 | 9 | 1 | 5 | 60.0% | 90.0% | 21/37 |
| 268 | SPY-CANARY-10s-NoConf | 20 | 19 | 16 | 4 | 3 | 69.6% | 80.0% | 12/37 |
| 269 | SPY-CANARY-1m-Control | 3 | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 1/37 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 20 | 19 | 16 | 4 | 3 | 69.6% | 80.0% | 13/37 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 14 | 14 | 11 | 3 | 3 | 64.7% | 78.6% | 17/37 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 14 | 14 | 11 | 3 | 3 | 64.7% | 78.6% | 18/37 |
| 276 | PACKTEST · Bollinger Bands · trigger | 16 | 14 | 13 | 3 | 1 | 76.5% | 81.2% | 7/37 |
| 277 | PACKTEST · Bollinger Bands · gate | 24 | 12 | 4 | 20 | 8 | 12.5% | 16.7% | 34/37 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 63 | 61 | 56 | 7 | 5 | 82.4% | 88.9% | 3/37 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 33 | 26 | 19 | 14 | 7 | 47.5% | 57.6% | 26/37 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 63 | 61 | 56 | 7 | 5 | 82.4% | 88.9% | 4/37 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 33 | 26 | 19 | 14 | 7 | 47.5% | 57.6% | 27/37 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 20 | 22 | 17 | 3 | 5 | 68.0% | 85.0% | 14/37 |
| 283 | PACKTEST · EMA Stack v2 · gate | 49 | 45 | 42 | 7 | 3 | 80.8% | 85.7% | 5/37 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 81 | 83 | 69 | 12 | 11 | 75.0% | 85.2% | 9/37 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 22 | 8 | 6 | 16 | 2 | 25.0% | 27.3% | 30/37 |
| 286 | PACKTEST · MACD Line v2 · trigger | 25 | 23 | 19 | 6 | 4 | 65.5% | 76.0% | 15/37 |
| 287 | PACKTEST · MACD Line v2 · gate | 27 | 15 | 11 | 16 | 4 | 35.5% | 40.7% | 28/37 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 41 | 41 | 34 | 7 | 7 | 70.8% | 82.9% | 11/37 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 78 | 70 | 55 | 23 | 15 | 59.1% | 70.5% | 22/37 |
| 291 | PACKTEST · Relative Volume v2 · gate | 24 | 4 | 4 | 20 | 0 | 16.7% | 16.7% | 32/37 |
| 292 | PACKTEST · Support Resistance Channels · trig | 52 | 39 | 20 | 32 | 24 | 26.3% | 38.5% | 29/37 |
| 293 | PACKTEST · Support Resistance Channels · gate | 25 | 28 | 19 | 6 | 9 | 55.9% | 76.0% | 24/37 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 23 | 21 | 19 | 4 | 2 | 76.0% | 82.6% | 8/37 |
| 296 | PACKTEST · Strat Assistant · trigger | 105 | 106 | 88 | 17 | 15 | 73.3% | 83.8% | 10/37 |
| 297 | PACKTEST · Strat Assistant · gate | 22 | 2 | 0 | 22 | 2 | 0.0% | 0.0% | 35/37 |
| 298 | PACKTEST · SuperTrend · trigger | 22 | 16 | 13 | 9 | 3 | 52.0% | 59.1% | 25/37 |
| 299 | PACKTEST · SuperTrend · gate | 40 | 31 | 26 | 14 | 5 | 57.8% | 65.0% | 23/37 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 48 | 46 | 41 | 7 | 5 | 77.4% | 85.4% | 6/37 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 37 | 47 | 32 | 5 | 15 | 61.5% | 86.5% | 20/37 |
| 302 | PACKTEST · UT Bot V4 · trigger | 48 | 47 | 43 | 5 | 4 | 82.7% | 89.6% | 2/37 |
| 303 | PACKTEST · UT Bot V4 · gate | 29 | 18 | 18 | 11 | 0 | 62.1% | 62.1% | 19/37 |
| 304 | PACKTEST · VWAP v2 · trigger | 2 | 3 | 1 | 1 | 2 | 25.0% | 50.0% | 31/37 |

---

## Hour 2026-06-04T14 UTC

_Strategies with activity this hour: 42; ranked (alerts≥1, BT≥1): 38_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 48 | 47 | 41 | 7 | 5 | 77.4% | 85.4% | 16/38 |
| 265 | TSLA-CANARY-1m-Control | 6 | 6 | 3 | 3 | 3 | 33.3% | 50.0% | 31/38 |
| 266 | TSLA-CANARY-5m-Control | 2 | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 3/38 |
| 267 | TSLA-CANARY-10s-LooseConf | 48 | 47 | 41 | 7 | 5 | 77.4% | 85.4% | 17/38 |
| 268 | SPY-CANARY-10s-NoConf | 53 | 53 | 51 | 2 | 2 | 92.7% | 96.2% | 9/38 |
| 269 | SPY-CANARY-1m-Control | 16 | 14 | 13 | 3 | 1 | 76.5% | 81.2% | 19/38 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 53 | 53 | 51 | 2 | 2 | 92.7% | 96.2% | 10/38 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 43 | 13 | 13 | 30 | 0 | 30.2% | 30.2% | 32/38 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 48 | 47 | 41 | 7 | 5 | 77.4% | 85.4% | 18/38 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 48 | 25 | 22 | 26 | 2 | 44.0% | 45.8% | 30/38 |
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 10 | 10 | 0 | 0 | 100.0% | 100.0% | 2/38 |
| 277 | PACKTEST · Bollinger Bands · gate | 53 | 17 | 16 | 37 | 0 | 30.2% | 30.2% | 33/38 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 75 | 75 | 60 | 15 | 12 | 69.0% | 80.0% | 22/38 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 53 | 29 | 28 | 25 | 0 | 52.8% | 52.8% | 27/38 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 75 | 75 | 60 | 15 | 12 | 69.0% | 80.0% | 23/38 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 53 | 29 | 28 | 25 | 0 | 52.8% | 52.8% | 28/38 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 22 | 22 | 21 | 1 | 0 | 95.5% | 95.5% | 4/38 |
| 283 | PACKTEST · EMA Stack v2 · gate | 53 | 53 | 51 | 2 | 1 | 94.4% | 96.2% | 5/38 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 115 | 115 | 100 | 15 | 13 | 78.1% | 87.0% | 15/38 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 53 | 14 | 13 | 40 | 0 | 24.5% | 24.5% | 36/38 |
| 286 | PACKTEST · MACD Line v2 · trigger | 20 | 20 | 18 | 2 | 2 | 81.8% | 90.0% | 13/38 |
| 287 | PACKTEST · MACD Line v2 · gate | 53 | 30 | 29 | 24 | 0 | 54.7% | 54.7% | 26/38 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 47 | 47 | 39 | 8 | 5 | 75.0% | 83.0% | 20/38 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 52 | 48 | 39 | 13 | 6 | 67.2% | 75.0% | 24/38 |
| 291 | PACKTEST · Relative Volume v2 · gate | 53 | 0 | 0 | 53 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 28 | 33 | 13 | 15 | 20 | 27.1% | 46.4% | 34/38 |
| 293 | PACKTEST · Support Resistance Channels · gate | 53 | 53 | 51 | 2 | 1 | 94.4% | 96.2% | 6/38 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 20 | 20 | 20 | 0 | 0 | 100.0% | 100.0% | 1/38 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 29 | 6 | 4 | 25 | 2 | 12.9% | 13.8% | 37/38 |
| 296 | PACKTEST · Strat Assistant · trigger | 106 | 106 | 95 | 11 | 8 | 83.3% | 89.6% | 12/38 |
| 297 | PACKTEST · Strat Assistant · gate | 53 | 14 | 14 | 39 | 0 | 26.4% | 26.4% | 35/38 |
| 298 | PACKTEST · SuperTrend · trigger | 16 | 16 | 15 | 1 | 1 | 88.2% | 93.8% | 11/38 |
| 299 | PACKTEST · SuperTrend · gate | 53 | 41 | 39 | 14 | 1 | 72.2% | 73.6% | 21/38 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 59 | 57 | 51 | 8 | 6 | 78.5% | 86.4% | 14/38 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 53 | 53 | 51 | 2 | 1 | 94.4% | 96.2% | 7/38 |
| 302 | PACKTEST · UT Bot V4 · trigger | 53 | 53 | 51 | 2 | 1 | 94.4% | 96.2% | 8/38 |
| 303 | PACKTEST · UT Bot V4 · gate | 61 | 48 | 38 | 23 | 8 | 55.1% | 62.3% | 25/38 |
| 304 | PACKTEST · VWAP v2 · trigger | 8 | 4 | 4 | 4 | 0 | 50.0% | 50.0% | 29/38 |
| 305 | PACKTEST · VWAP v2 · gate | 31 | 2 | 2 | 29 | 0 | 6.5% | 6.5% | 38/38 |

---

## Hour 2026-06-04T15 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 36_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 44 | 48 | 41 | 3 | 7 | 80.4% | 93.2% | 14/36 |
| 265 | TSLA-CANARY-1m-Control | 6 | 6 | 4 | 2 | 2 | 50.0% | 66.7% | 28/36 |
| 266 | TSLA-CANARY-5m-Control | 1 | 2 | 1 | 0 | 1 | 50.0% | 100.0% | 29/36 |
| 267 | TSLA-CANARY-10s-LooseConf | 44 | 48 | 41 | 3 | 7 | 80.4% | 93.2% | 15/36 |
| 268 | SPY-CANARY-10s-NoConf | 41 | 43 | 40 | 1 | 3 | 90.9% | 97.6% | 2/36 |
| 269 | SPY-CANARY-1m-Control | 3 | 6 | 2 | 1 | 4 | 28.6% | 66.7% | 32/36 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 41 | 43 | 40 | 1 | 3 | 90.9% | 97.6% | 3/36 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 21 | 13 | 13 | 8 | 0 | 61.9% | 61.9% | 25/36 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 44 | 48 | 41 | 3 | 7 | 80.4% | 93.2% | 16/36 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 24 | 0 | 0 | 24 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 18 | 17 | 16 | 2 | 1 | 84.2% | 88.9% | 6/36 |
| 277 | PACKTEST · Bollinger Bands · gate | 21 | 21 | 21 | 0 | 0 | 100.0% | 100.0% | 1/36 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 71 | 73 | 63 | 8 | 7 | 80.8% | 88.7% | 11/36 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 39 | 25 | 22 | 17 | 3 | 52.4% | 56.4% | 26/36 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 71 | 73 | 63 | 8 | 7 | 80.8% | 88.7% | 12/36 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 39 | 25 | 22 | 17 | 3 | 52.4% | 56.4% | 27/36 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 14 | 16 | 13 | 1 | 3 | 76.5% | 92.9% | 19/36 |
| 283 | PACKTEST · EMA Stack v2 · gate | 41 | 43 | 38 | 3 | 5 | 82.6% | 92.7% | 7/36 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 105 | 107 | 92 | 13 | 14 | 77.3% | 87.6% | 18/36 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 21 | 6 | 6 | 15 | 0 | 28.6% | 28.6% | 31/36 |
| 286 | PACKTEST · MACD Line v2 · trigger | 30 | 30 | 28 | 2 | 2 | 87.5% | 93.3% | 4/36 |
| 287 | PACKTEST · MACD Line v2 · gate | 21 | 20 | 18 | 3 | 2 | 78.3% | 85.7% | 17/36 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 41 | 43 | 39 | 2 | 4 | 86.7% | 95.1% | 5/36 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 72 | 74 | 63 | 9 | 11 | 75.9% | 87.5% | 20/36 |
| 291 | PACKTEST · Relative Volume v2 · gate | 21 | 0 | 0 | 21 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 20 | 11 | 3 | 17 | 8 | 10.7% | 15.0% | 36/36 |
| 293 | PACKTEST · Support Resistance Channels · gate | 31 | 43 | 29 | 2 | 14 | 64.4% | 93.5% | 23/36 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 22 | 24 | 18 | 4 | 6 | 64.3% | 81.8% | 24/36 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 41 | 8 | 8 | 33 | 0 | 19.5% | 19.5% | 33/36 |
| 296 | PACKTEST · Strat Assistant · trigger | 122 | 124 | 108 | 14 | 12 | 80.6% | 88.5% | 13/36 |
| 297 | PACKTEST · Strat Assistant · gate | 35 | 4 | 4 | 31 | 0 | 11.4% | 11.4% | 35/36 |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 10 | 9 | 1 | 1 | 81.8% | 90.0% | 9/36 |
| 299 | PACKTEST · SuperTrend · gate | 41 | 39 | 34 | 7 | 5 | 73.9% | 82.9% | 21/36 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 44 | 45 | 40 | 4 | 5 | 81.6% | 90.9% | 10/36 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 37 | 43 | 34 | 3 | 9 | 73.9% | 91.9% | 22/36 |
| 302 | PACKTEST · UT Bot V4 · trigger | 41 | 43 | 38 | 3 | 5 | 82.6% | 92.7% | 8/36 |
| 303 | PACKTEST · UT Bot V4 · gate | 45 | 18 | 15 | 30 | 3 | 31.2% | 33.3% | 30/36 |
| 305 | PACKTEST · VWAP v2 · gate | 21 | 4 | 4 | 17 | 0 | 19.0% | 19.0% | 34/36 |

---

## Hour 2026-06-04T16 UTC

_Strategies with activity this hour: 36; ranked (alerts≥1, BT≥1): 34_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 10 | 0 | 0 | 10 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 33 | 33 | 31 | 2 | 2 | 88.6% | 93.9% | 10/34 |
| 265 | TSLA-CANARY-1m-Control | 10 | 10 | 9 | 1 | 1 | 81.8% | 90.0% | 19/34 |
| 266 | TSLA-CANARY-5m-Control | 2 | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 3/34 |
| 267 | TSLA-CANARY-10s-LooseConf | 33 | 33 | 31 | 2 | 2 | 88.6% | 93.9% | 11/34 |
| 268 | SPY-CANARY-10s-NoConf | 46 | 45 | 45 | 1 | 0 | 97.8% | 97.8% | 4/34 |
| 269 | SPY-CANARY-1m-Control | 3 | 4 | 1 | 2 | 3 | 16.7% | 33.3% | 34/34 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 46 | 45 | 45 | 1 | 0 | 97.8% | 97.8% | 5/34 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 33 | 33 | 31 | 2 | 2 | 88.6% | 93.9% | 12/34 |
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 13 | 12 | 0 | 1 | 92.3% | 100.0% | 6/34 |
| 277 | PACKTEST · Bollinger Bands · gate | 40 | 30 | 27 | 13 | 3 | 62.8% | 67.5% | 27/34 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 76 | 76 | 66 | 10 | 10 | 76.7% | 86.8% | 22/34 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 46 | 40 | 37 | 9 | 3 | 75.5% | 80.4% | 24/34 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 76 | 76 | 66 | 10 | 10 | 76.7% | 86.8% | 23/34 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 46 | 40 | 37 | 9 | 3 | 75.5% | 80.4% | 25/34 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 17 | 17 | 17 | 0 | 0 | 100.0% | 100.0% | 1/34 |
| 283 | PACKTEST · EMA Stack v2 · gate | 46 | 46 | 43 | 3 | 3 | 87.8% | 93.5% | 13/34 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 113 | 113 | 99 | 14 | 10 | 80.5% | 87.6% | 20/34 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 40 | 8 | 8 | 32 | 0 | 20.0% | 20.0% | 31/34 |
| 286 | PACKTEST · MACD Line v2 · trigger | 22 | 22 | 18 | 4 | 4 | 69.2% | 81.8% | 26/34 |
| 287 | PACKTEST · MACD Line v2 · gate | 40 | 16 | 15 | 25 | 1 | 36.6% | 37.5% | 29/34 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 46 | 46 | 39 | 7 | 4 | 78.0% | 84.8% | 21/34 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 74 | 74 | 68 | 6 | 2 | 89.5% | 91.9% | 8/34 |
| 292 | PACKTEST · Support Resistance Channels · trig | 32 | 27 | 12 | 20 | 16 | 25.0% | 37.5% | 30/34 |
| 293 | PACKTEST · Support Resistance Channels · gate | 46 | 46 | 43 | 3 | 3 | 87.8% | 93.5% | 14/34 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 10 | 10 | 10 | 0 | 0 | 100.0% | 100.0% | 2/34 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 46 | 10 | 9 | 37 | 1 | 19.1% | 19.6% | 32/34 |
| 296 | PACKTEST · Strat Assistant · trigger | 123 | 123 | 110 | 13 | 11 | 82.1% | 89.4% | 18/34 |
| 297 | PACKTEST · Strat Assistant · gate | 46 | 10 | 9 | 37 | 1 | 19.1% | 19.6% | 33/34 |
| 298 | PACKTEST · SuperTrend · trigger | 9 | 8 | 8 | 1 | 0 | 88.9% | 88.9% | 9/34 |
| 299 | PACKTEST · SuperTrend · gate | 46 | 46 | 43 | 3 | 3 | 87.8% | 93.5% | 15/34 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 50 | 48 | 47 | 3 | 1 | 92.2% | 94.0% | 7/34 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 46 | 46 | 43 | 3 | 3 | 87.8% | 93.5% | 16/34 |
| 302 | PACKTEST · UT Bot V4 · trigger | 46 | 46 | 43 | 3 | 3 | 87.8% | 93.5% | 17/34 |
| 303 | PACKTEST · UT Bot V4 · gate | 60 | 50 | 42 | 18 | 8 | 61.8% | 70.0% | 28/34 |
| 305 | PACKTEST · VWAP v2 · gate | 40 | 0 | 0 | 40 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-04T17 UTC

_Strategies with activity this hour: 40; ranked (alerts≥1, BT≥1): 37_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 52 | 52 | 50 | 2 | 2 | 92.6% | 96.2% | 7/37 |
| 265 | TSLA-CANARY-1m-Control | 6 | 8 | 5 | 1 | 3 | 55.6% | 83.3% | 27/37 |
| 266 | TSLA-CANARY-5m-Control | 3 | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 3/37 |
| 267 | TSLA-CANARY-10s-LooseConf | 52 | 52 | 50 | 2 | 2 | 92.6% | 96.2% | 8/37 |
| 268 | SPY-CANARY-10s-NoConf | 41 | 42 | 40 | 1 | 2 | 93.0% | 97.6% | 5/37 |
| 269 | SPY-CANARY-1m-Control | 7 | 8 | 5 | 2 | 3 | 50.0% | 71.4% | 29/37 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 41 | 42 | 40 | 1 | 2 | 93.0% | 97.6% | 6/37 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 31 | 31 | 30 | 1 | 1 | 93.8% | 96.8% | 4/37 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 52 | 52 | 50 | 2 | 2 | 92.6% | 96.2% | 9/37 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 21 | 12 | 11 | 10 | 1 | 50.0% | 52.4% | 28/37 |
| 276 | PACKTEST · Bollinger Bands · trigger | 17 | 17 | 17 | 0 | 0 | 100.0% | 100.0% | 2/37 |
| 277 | PACKTEST · Bollinger Bands · gate | 41 | 13 | 9 | 32 | 4 | 20.0% | 22.0% | 32/37 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 71 | 71 | 60 | 11 | 11 | 73.2% | 84.5% | 23/37 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 41 | 29 | 25 | 16 | 4 | 55.6% | 61.0% | 25/37 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 71 | 71 | 60 | 11 | 11 | 73.2% | 84.5% | 24/37 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 41 | 29 | 25 | 16 | 4 | 55.6% | 61.0% | 26/37 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 21 | 21 | 21 | 0 | 0 | 100.0% | 100.0% | 1/37 |
| 283 | PACKTEST · EMA Stack v2 · gate | 41 | 41 | 36 | 5 | 5 | 78.3% | 87.8% | 17/37 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 109 | 109 | 93 | 16 | 14 | 75.6% | 85.3% | 18/37 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 41 | 7 | 4 | 37 | 3 | 9.1% | 9.8% | 35/37 |
| 286 | PACKTEST · MACD Line v2 · trigger | 37 | 37 | 33 | 4 | 4 | 80.5% | 89.2% | 14/37 |
| 287 | PACKTEST · MACD Line v2 · gate | 39 | 13 | 10 | 29 | 3 | 23.8% | 25.6% | 31/37 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 49 | 49 | 43 | 6 | 4 | 81.1% | 87.8% | 12/37 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 86 | 86 | 76 | 10 | 8 | 80.9% | 88.4% | 13/37 |
| 291 | PACKTEST · Relative Volume v2 · gate | 31 | 4 | 2 | 29 | 2 | 6.1% | 6.5% | 36/37 |
| 292 | PACKTEST · Support Resistance Channels · trig | 25 | 38 | 8 | 17 | 30 | 14.5% | 32.0% | 34/37 |
| 293 | PACKTEST · Support Resistance Channels · gate | 39 | 41 | 34 | 5 | 7 | 73.9% | 87.2% | 19/37 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 18 | 18 | 16 | 2 | 2 | 80.0% | 88.9% | 15/37 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 39 | 2 | 2 | 37 | 0 | 5.1% | 5.1% | 37/37 |
| 296 | PACKTEST · Strat Assistant · trigger | 107 | 108 | 94 | 13 | 11 | 79.7% | 87.9% | 16/37 |
| 297 | PACKTEST · Strat Assistant · gate | 39 | 6 | 6 | 33 | 0 | 15.4% | 15.4% | 33/37 |
| 298 | PACKTEST · SuperTrend · trigger | 15 | 14 | 13 | 2 | 1 | 81.2% | 86.7% | 11/37 |
| 299 | PACKTEST · SuperTrend · gate | 39 | 41 | 34 | 5 | 7 | 73.9% | 87.2% | 20/37 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 32 | 32 | 30 | 2 | 1 | 90.9% | 93.8% | 10/37 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 39 | 41 | 34 | 5 | 7 | 73.9% | 87.2% | 21/37 |
| 302 | PACKTEST · UT Bot V4 · trigger | 39 | 41 | 34 | 5 | 7 | 73.9% | 87.2% | 22/37 |
| 303 | PACKTEST · UT Bot V4 · gate | 55 | 29 | 26 | 29 | 2 | 45.6% | 47.3% | 30/37 |
| 305 | PACKTEST · VWAP v2 · gate | 39 | 0 | 0 | 39 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-04T18 UTC

_Strategies with activity this hour: 41; ranked (alerts≥1, BT≥1): 36_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 174 | TSLA LONG 1Min Mass #2 | 3 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | — |
| 194 | TSLL LONG 1Min Mass #30 | 1 | 8 | 1 | 0 | 7 | 12.5% | 100.0% | 36/36 |
| 263 | TSLA-CANARY-10s-NoConf | 46 | 48 | 41 | 5 | 7 | 77.4% | 89.1% | 6/36 |
| 265 | TSLA-CANARY-1m-Control | 3 | 8 | 3 | 0 | 5 | 37.5% | 100.0% | 32/36 |
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 42 | 48 | 38 | 4 | 10 | 73.1% | 90.5% | 9/36 |
| 268 | SPY-CANARY-10s-NoConf | 47 | 52 | 44 | 3 | 8 | 80.0% | 93.6% | 4/36 |
| 269 | SPY-CANARY-1m-Control | 8 | 11 | 6 | 2 | 5 | 46.2% | 75.0% | 28/36 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 47 | 52 | 44 | 3 | 8 | 80.0% | 93.6% | 5/36 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 45 | 32 | 24 | 21 | 8 | 45.3% | 53.3% | 29/36 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 46 | 48 | 41 | 5 | 7 | 77.4% | 89.1% | 7/36 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 36 | 34 | 19 | 17 | 15 | 37.3% | 52.8% | 33/36 |
| 276 | PACKTEST · Bollinger Bands · trigger | 11 | 11 | 10 | 1 | 1 | 83.3% | 90.9% | 1/36 |
| 277 | PACKTEST · Bollinger Bands · gate | 46 | 31 | 28 | 18 | 3 | 57.1% | 60.9% | 21/36 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 67 | 78 | 51 | 16 | 27 | 54.3% | 76.1% | 23/36 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 47 | 41 | 36 | 11 | 5 | 69.2% | 76.6% | 16/36 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 67 | 78 | 51 | 16 | 27 | 54.3% | 76.1% | 24/36 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 47 | 41 | 36 | 11 | 5 | 69.2% | 76.6% | 17/36 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 18 | 23 | 17 | 1 | 6 | 70.8% | 94.4% | 10/36 |
| 283 | PACKTEST · EMA Stack v2 · gate | 47 | 52 | 41 | 6 | 11 | 70.7% | 87.2% | 11/36 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 90 | 95 | 71 | 19 | 22 | 63.4% | 78.9% | 19/36 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 23 | 9 | 9 | 14 | 0 | 39.1% | 39.1% | 31/36 |
| 286 | PACKTEST · MACD Line v2 · trigger | 20 | 24 | 15 | 5 | 9 | 51.7% | 75.0% | 27/36 |
| 287 | PACKTEST · MACD Line v2 · gate | 23 | 13 | 13 | 10 | 0 | 56.5% | 56.5% | 22/36 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 37 | 47 | 29 | 8 | 18 | 52.7% | 78.4% | 25/36 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 59 | 65 | 48 | 11 | 16 | 64.0% | 81.4% | 18/36 |
| 291 | PACKTEST · Relative Volume v2 · gate | 23 | 0 | 0 | 23 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 43 | 44 | 28 | 15 | 19 | 45.2% | 65.1% | 30/36 |
| 293 | PACKTEST · Support Resistance Channels · gate | 47 | 52 | 41 | 6 | 11 | 70.7% | 87.2% | 12/36 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 11 | 11 | 10 | 1 | 1 | 83.3% | 90.9% | 2/36 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 23 | 4 | 4 | 19 | 0 | 17.4% | 17.4% | 34/36 |
| 296 | PACKTEST · Strat Assistant · trigger | 102 | 108 | 79 | 23 | 24 | 62.7% | 77.5% | 20/36 |
| 297 | PACKTEST · Strat Assistant · gate | 39 | 10 | 7 | 32 | 3 | 16.7% | 17.9% | 35/36 |
| 298 | PACKTEST · SuperTrend · trigger | 13 | 15 | 12 | 1 | 3 | 75.0% | 92.3% | 8/36 |
| 299 | PACKTEST · SuperTrend · gate | 47 | 52 | 41 | 6 | 11 | 70.7% | 87.2% | 13/36 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 52 | 54 | 48 | 4 | 6 | 82.8% | 92.3% | 3/36 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 47 | 52 | 41 | 6 | 11 | 70.7% | 87.2% | 14/36 |
| 302 | PACKTEST · UT Bot V4 · trigger | 48 | 52 | 41 | 7 | 11 | 69.5% | 85.4% | 15/36 |
| 303 | PACKTEST · UT Bot V4 · gate | 43 | 41 | 29 | 14 | 13 | 51.8% | 67.4% | 26/36 |
| 305 | PACKTEST · VWAP v2 · gate | 23 | 0 | 0 | 23 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-04T19 UTC

_Strategies with activity this hour: 39; ranked (alerts≥1, BT≥1): 37_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | 2 | 2 | 0 | 2 | 2 | 0.0% | 0.0% | 37/37 |
| 174 | TSLA LONG 1Min Mass #2 | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 37 | 41 | 34 | 3 | 7 | 77.3% | 91.9% | 5/37 |
| 265 | TSLA-CANARY-1m-Control | 7 | 8 | 2 | 5 | 6 | 15.4% | 28.6% | 36/37 |
| 266 | TSLA-CANARY-5m-Control | 1 | 2 | 1 | 0 | 1 | 50.0% | 100.0% | 28/37 |
| 267 | TSLA-CANARY-10s-LooseConf | 37 | 41 | 34 | 3 | 7 | 77.3% | 91.9% | 6/37 |
| 268 | SPY-CANARY-10s-NoConf | 49 | 57 | 45 | 4 | 12 | 73.8% | 91.8% | 11/37 |
| 269 | SPY-CANARY-1m-Control | 9 | 10 | 8 | 1 | 2 | 72.7% | 88.9% | 14/37 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 49 | 57 | 45 | 4 | 12 | 73.8% | 91.8% | 12/37 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 43 | 57 | 39 | 4 | 18 | 63.9% | 90.7% | 17/37 |
| 272 | TEST-P2-stoch-10s-0601-SPY | 18 | 10 | 10 | 8 | 0 | 55.6% | 55.6% | 25/37 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 37 | 41 | 34 | 3 | 7 | 77.3% | 91.9% | 7/37 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 29 | 24 | 18 | 11 | 6 | 51.4% | 62.1% | 27/37 |
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 8 | 8 | 2 | 0 | 80.0% | 80.0% | 4/37 |
| 277 | PACKTEST · Bollinger Bands · gate | 17 | 16 | 12 | 5 | 4 | 57.1% | 70.6% | 22/37 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 75 | 79 | 66 | 9 | 11 | 76.7% | 88.0% | 8/37 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 17 | 22 | 14 | 3 | 8 | 56.0% | 82.4% | 23/37 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 75 | 79 | 66 | 9 | 11 | 76.7% | 88.0% | 9/37 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 17 | 22 | 14 | 3 | 8 | 56.0% | 82.4% | 24/37 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 16 | 14 | 14 | 2 | 0 | 87.5% | 87.5% | 2/37 |
| 283 | PACKTEST · EMA Stack v2 · gate | 49 | 37 | 28 | 21 | 9 | 48.3% | 57.1% | 29/37 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 89 | 95 | 72 | 17 | 20 | 66.1% | 80.9% | 16/37 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 10 | 12 | 6 | 4 | 6 | 37.5% | 60.0% | 33/37 |
| 286 | PACKTEST · MACD Line v2 · trigger | 27 | 31 | 26 | 1 | 5 | 81.2% | 96.3% | 3/37 |
| 287 | PACKTEST · MACD Line v2 · gate | 28 | 18 | 8 | 20 | 10 | 21.1% | 28.6% | 34/37 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 26 | 32 | 21 | 5 | 10 | 58.3% | 80.8% | 21/37 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 73 | 83 | 65 | 8 | 15 | 73.9% | 89.0% | 10/37 |
| 292 | PACKTEST · Support Resistance Channels · trig | 39 | 42 | 11 | 28 | 29 | 16.2% | 28.2% | 35/37 |
| 293 | PACKTEST · Support Resistance Channels · gate | 49 | 35 | 26 | 23 | 9 | 44.8% | 53.1% | 31/37 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 17 | 21 | 15 | 2 | 7 | 62.5% | 88.2% | 20/37 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 4 | 6 | 3 | 1 | 3 | 42.9% | 75.0% | 32/37 |
| 296 | PACKTEST · Strat Assistant · trigger | 102 | 105 | 87 | 15 | 17 | 73.1% | 85.3% | 13/37 |
| 297 | PACKTEST · Strat Assistant · gate | 17 | 0 | 0 | 17 | 0 | 0.0% | 0.0% | — |
| 298 | PACKTEST · SuperTrend · trigger | 8 | 8 | 8 | 0 | 0 | 100.0% | 100.0% | 1/37 |
| 299 | PACKTEST · SuperTrend · gate | 49 | 41 | 31 | 18 | 10 | 52.5% | 63.3% | 26/37 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 53 | 52 | 43 | 10 | 9 | 69.4% | 81.1% | 15/37 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 49 | 57 | 41 | 8 | 16 | 63.1% | 83.7% | 18/37 |
| 302 | PACKTEST · UT Bot V4 · trigger | 49 | 57 | 41 | 8 | 16 | 63.1% | 83.7% | 19/37 |
| 303 | PACKTEST · UT Bot V4 · gate | 14 | 14 | 9 | 5 | 5 | 47.4% | 64.3% | 30/37 |

---

## Hour 2026-06-04T20 UTC

_Strategies with activity this hour: 23; ranked (alerts≥1, BT≥1): 18_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 9 | 8 | 3 | 6 | 5 | 21.4% | 33.3% | 12/18 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 69 | 75 | 44 | 25 | 28 | 45.4% | 63.8% | 2/18 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 69 | 75 | 44 | 25 | 28 | 45.4% | 63.8% | 3/18 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 14 | 19 | 6 | 8 | 13 | 22.2% | 42.9% | 11/18 |
| 283 | PACKTEST · EMA Stack v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 77 | 91 | 49 | 28 | 38 | 42.6% | 63.6% | 4/18 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 0 | 19 | 0 | 0 | 19 | 0.0% | 0.0% | — |
| 286 | PACKTEST · MACD Line v2 · trigger | 23 | 22 | 6 | 17 | 16 | 15.4% | 26.1% | 15/18 |
| 287 | PACKTEST · MACD Line v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 26 | 43 | 18 | 8 | 24 | 36.0% | 69.2% | 9/18 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 73 | 74 | 41 | 32 | 32 | 39.0% | 56.2% | 8/18 |
| 292 | PACKTEST · Support Resistance Channels · trig | 12 | 17 | 5 | 7 | 12 | 20.8% | 41.7% | 14/18 |
| 293 | PACKTEST · Support Resistance Channels · gate | 49 | 43 | 20 | 29 | 22 | 28.2% | 40.8% | 10/18 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 33 | 26 | 17 | 16 | 9 | 40.5% | 51.5% | 6/18 |
| 296 | PACKTEST · Strat Assistant · trigger | 84 | 97 | 53 | 31 | 42 | 42.1% | 63.1% | 5/18 |
| 297 | PACKTEST · Strat Assistant · gate | 6 | 6 | 0 | 6 | 6 | 0.0% | 0.0% | 18/18 |
| 298 | PACKTEST · SuperTrend · trigger | 17 | 12 | 2 | 15 | 10 | 7.4% | 11.8% | 16/18 |
| 299 | PACKTEST · SuperTrend · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 11 | 16 | 9 | 2 | 7 | 50.0% | 81.8% | 1/18 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 35 | 69 | 18 | 17 | 50 | 21.2% | 51.4% | 13/18 |
| 302 | PACKTEST · UT Bot V4 · trigger | 63 | 69 | 37 | 26 | 30 | 39.8% | 58.7% | 7/18 |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 5 | 0 | 0 | 5 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 19 | 2 | 0 | 19 | 2 | 0.0% | 0.0% | 17/18 |

---

## Hour 2026-06-04T21 UTC

_Strategies with activity this hour: 21; ranked (alerts≥1, BT≥1): 19_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 2 | 2 | 10 | 0 | 16.7% | 16.7% | 13/19 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 32 | 23 | 15 | 17 | 8 | 37.5% | 46.9% | 4/19 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 32 | 23 | 15 | 17 | 8 | 37.5% | 46.9% | 5/19 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 18 | 13 | 5 | 13 | 8 | 19.2% | 27.8% | 11/19 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 43 | 28 | 8 | 35 | 21 | 12.5% | 18.6% | 16/19 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 13 | 3 | 0 | 13 | 3 | 0.0% | 0.0% | 18/19 |
| 286 | PACKTEST · MACD Line v2 · trigger | 20 | 12 | 5 | 15 | 7 | 18.5% | 25.0% | 12/19 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 24 | 25 | 16 | 8 | 9 | 48.5% | 66.7% | 2/19 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 78 | 28 | 23 | 55 | 8 | 26.7% | 29.5% | 8/19 |
| 292 | PACKTEST · Support Resistance Channels · trig | 4 | 3 | 2 | 2 | 1 | 40.0% | 50.0% | 3/19 |
| 293 | PACKTEST · Support Resistance Channels · gate | 22 | 7 | 4 | 18 | 3 | 16.0% | 18.2% | 14/19 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 12 | 1 | 0 | 12 | 1 | 0.0% | 0.0% | 19/19 |
| 296 | PACKTEST · Strat Assistant · trigger | 38 | 40 | 31 | 7 | 11 | 63.3% | 81.6% | 1/19 |
| 297 | PACKTEST · Strat Assistant · gate | 4 | 6 | 2 | 2 | 4 | 25.0% | 50.0% | 9/19 |
| 298 | PACKTEST · SuperTrend · trigger | 24 | 8 | 3 | 21 | 5 | 10.3% | 12.5% | 17/19 |
| 299 | PACKTEST · SuperTrend · gate | 11 | 20 | 4 | 7 | 16 | 14.8% | 36.4% | 15/19 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 4 | 1 | 1 | 3 | 0 | 25.0% | 25.0% | 10/19 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 26 | 23 | 11 | 15 | 12 | 28.9% | 42.3% | 7/19 |
| 302 | PACKTEST · UT Bot V4 · trigger | 27 | 23 | 12 | 15 | 11 | 31.6% | 44.4% | 6/19 |
| 303 | PACKTEST · UT Bot V4 · gate | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 304 | PACKTEST · VWAP v2 · trigger | 14 | 0 | 0 | 14 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-04T22 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 23_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 1 | 1 | 11 | 0 | 8.3% | 8.3% | 17/23 |
| 277 | PACKTEST · Bollinger Bands · gate | 4 | 2 | 1 | 3 | 1 | 20.0% | 25.0% | 11/23 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 38 | 19 | 10 | 28 | 9 | 21.3% | 26.3% | 8/23 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 38 | 19 | 10 | 28 | 9 | 21.3% | 26.3% | 9/23 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 13 | 7 | 3 | 10 | 4 | 17.6% | 23.1% | 12/23 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 66 | 39 | 18 | 48 | 22 | 20.5% | 27.3% | 10/23 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 40 | 6 | 3 | 37 | 3 | 7.0% | 7.5% | 18/23 |
| 286 | PACKTEST · MACD Line v2 · trigger | 28 | 15 | 6 | 22 | 9 | 16.2% | 21.4% | 13/23 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 19 | 13 | 8 | 11 | 5 | 33.3% | 42.1% | 4/23 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 90 | 47 | 28 | 62 | 19 | 25.7% | 31.1% | 6/23 |
| 291 | PACKTEST · Relative Volume v2 · gate | 8 | 2 | 0 | 8 | 2 | 0.0% | 0.0% | 21/23 |
| 292 | PACKTEST · Support Resistance Channels · trig | 13 | 15 | 1 | 12 | 14 | 3.7% | 7.7% | 19/23 |
| 293 | PACKTEST · Support Resistance Channels · gate | 24 | 2 | 0 | 24 | 2 | 0.0% | 0.0% | 20/23 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 16 | 11 | 3 | 13 | 8 | 12.5% | 18.8% | 14/23 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 48 | 52 | 33 | 15 | 19 | 49.3% | 68.8% | 1/23 |
| 297 | PACKTEST · Strat Assistant · gate | 6 | 10 | 4 | 2 | 6 | 33.3% | 66.7% | 5/23 |
| 298 | PACKTEST · SuperTrend · trigger | 26 | 11 | 4 | 22 | 7 | 12.1% | 15.4% | 15/23 |
| 299 | PACKTEST · SuperTrend · gate | 6 | 6 | 0 | 6 | 6 | 0.0% | 0.0% | 22/23 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 3 | 2 | 1 | 2 | 1 | 25.0% | 33.3% | 7/23 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 34 | 30 | 17 | 17 | 13 | 36.2% | 50.0% | 3/23 |
| 302 | PACKTEST · UT Bot V4 · trigger | 54 | 30 | 23 | 31 | 7 | 37.7% | 42.6% | 2/23 |
| 303 | PACKTEST · UT Bot V4 · gate | 8 | 3 | 1 | 7 | 2 | 10.0% | 12.5% | 16/23 |
| 304 | PACKTEST · VWAP v2 · trigger | 6 | 1 | 0 | 6 | 1 | 0.0% | 0.0% | 23/23 |
| 305 | PACKTEST · VWAP v2 · gate | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-04T23 UTC

_Strategies with activity this hour: 23; ranked (alerts≥1, BT≥1): 19_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 9 | 3 | 1 | 8 | 2 | 9.1% | 11.1% | 13/19 |
| 277 | PACKTEST · Bollinger Bands · gate | 1 | 2 | 1 | 0 | 1 | 50.0% | 100.0% | 1/19 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 30 | 19 | 9 | 21 | 10 | 22.5% | 30.0% | 5/19 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 30 | 19 | 9 | 21 | 10 | 22.5% | 30.0% | 6/19 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 4 | 5 | 0 | 4 | 5 | 0.0% | 0.0% | 18/19 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 48 | 37 | 9 | 39 | 28 | 11.8% | 18.8% | 12/19 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 3 | 2 | 1 | 2 | 1 | 25.0% | 33.3% | 4/19 |
| 286 | PACKTEST · MACD Line v2 · trigger | 14 | 9 | 0 | 14 | 9 | 0.0% | 0.0% | 16/19 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 6 | 11 | 0 | 6 | 11 | 0.0% | 0.0% | 17/19 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 83 | 41 | 22 | 61 | 18 | 21.8% | 26.5% | 7/19 |
| 292 | PACKTEST · Support Resistance Channels · trig | 4 | 11 | 1 | 3 | 10 | 7.1% | 25.0% | 15/19 |
| 293 | PACKTEST · Support Resistance Channels · gate | 8 | 6 | 2 | 6 | 4 | 16.7% | 25.0% | 8/19 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 23 | 18 | 3 | 20 | 15 | 7.9% | 13.0% | 14/19 |
| 296 | PACKTEST · Strat Assistant · trigger | 38 | 51 | 21 | 17 | 30 | 30.9% | 55.3% | 2/19 |
| 297 | PACKTEST · Strat Assistant · gate | 4 | 4 | 1 | 3 | 3 | 14.3% | 25.0% | 11/19 |
| 298 | PACKTEST · SuperTrend · trigger | 19 | 5 | 3 | 16 | 2 | 14.3% | 15.8% | 10/19 |
| 299 | PACKTEST · SuperTrend · gate | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | 20 | 24 | 6 | 14 | 18 | 15.8% | 30.0% | 9/19 |
| 302 | PACKTEST · UT Bot V4 · trigger | 39 | 24 | 14 | 25 | 10 | 28.6% | 35.9% | 3/19 |
| 303 | PACKTEST · UT Bot V4 · gate | 1 | 5 | 0 | 1 | 5 | 0.0% | 0.0% | 19/19 |
| 304 | PACKTEST · VWAP v2 · trigger | 15 | 0 | 0 | 15 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-05T00 UTC

_Strategies with activity this hour: 2; ranked (alerts≥1, BT≥1): 0_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 284 | PACKTEST · MACD Histogram v2 · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 290 | PACKTEST · Relative Volume v2 · trigger | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |

---

## Hour 2026-06-05T08 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 23_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 2 | 2 | 10 | 0 | 16.7% | 16.7% | 13/23 |
| 277 | PACKTEST · Bollinger Bands · gate | 8 | 2 | 2 | 6 | 0 | 25.0% | 25.0% | 7/23 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 39 | 39 | 20 | 19 | 18 | 35.1% | 51.3% | 3/23 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 39 | 39 | 20 | 19 | 18 | 35.1% | 51.3% | 4/23 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 14 | 2 | 0 | 14 | 2 | 0.0% | 0.0% | 22/23 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 51 | 45 | 18 | 33 | 26 | 23.4% | 35.3% | 9/23 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 10 | 6 | 3 | 7 | 3 | 23.1% | 30.0% | 10/23 |
| 286 | PACKTEST · MACD Line v2 · trigger | 17 | 13 | 2 | 15 | 11 | 7.1% | 11.8% | 17/23 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 15 | 14 | 4 | 11 | 10 | 16.0% | 26.7% | 14/23 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 115 | 48 | 36 | 79 | 12 | 28.3% | 31.3% | 5/23 |
| 291 | PACKTEST · Relative Volume v2 · gate | 3 | 2 | 0 | 3 | 2 | 0.0% | 0.0% | 23/23 |
| 292 | PACKTEST · Support Resistance Channels · trig | 20 | 2 | 0 | 20 | 2 | 0.0% | 0.0% | 21/23 |
| 293 | PACKTEST · Support Resistance Channels · gate | 8 | 17 | 1 | 7 | 16 | 4.2% | 12.5% | 19/23 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 17 | 10 | 2 | 15 | 8 | 8.0% | 11.8% | 16/23 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 8 | 2 | 2 | 6 | 0 | 25.0% | 25.0% | 8/23 |
| 296 | PACKTEST · Strat Assistant · trigger | 59 | 55 | 40 | 19 | 16 | 53.3% | 67.8% | 1/23 |
| 297 | PACKTEST · Strat Assistant · gate | 6 | 6 | 1 | 5 | 5 | 9.1% | 16.7% | 15/23 |
| 298 | PACKTEST · SuperTrend · trigger | 23 | 6 | 1 | 22 | 5 | 3.6% | 4.3% | 20/23 |
| 299 | PACKTEST · SuperTrend · gate | 24 | 0 | 0 | 24 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 4 | 10 | 4 | 0 | 6 | 40.0% | 100.0% | 2/23 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 26 | 31 | 9 | 17 | 22 | 18.8% | 34.6% | 12/23 |
| 302 | PACKTEST · UT Bot V4 · trigger | 45 | 31 | 13 | 32 | 18 | 20.6% | 28.9% | 11/23 |
| 303 | PACKTEST · UT Bot V4 · gate | 16 | 2 | 1 | 15 | 1 | 5.9% | 6.2% | 18/23 |
| 304 | PACKTEST · VWAP v2 · trigger | 10 | 15 | 5 | 5 | 10 | 25.0% | 50.0% | 6/23 |
| 305 | PACKTEST · VWAP v2 · gate | 0 | 14 | 0 | 0 | 14 | 0.0% | 0.0% | — |

---

## Hour 2026-06-05T09 UTC

_Strategies with activity this hour: 26; ranked (alerts≥1, BT≥1): 21_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 5 | 1 | 9 | 4 | 7.1% | 10.0% | 18/21 |
| 277 | PACKTEST · Bollinger Bands · gate | 8 | 4 | 2 | 6 | 2 | 20.0% | 25.0% | 6/21 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 27 | 20 | 13 | 14 | 6 | 39.4% | 48.1% | 2/21 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 27 | 20 | 13 | 14 | 6 | 39.4% | 48.1% | 3/21 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 10 | 5 | 2 | 8 | 4 | 14.3% | 20.0% | 11/21 |
| 283 | PACKTEST · EMA Stack v2 · gate | 18 | 0 | 0 | 18 | 0 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 58 | 27 | 8 | 50 | 19 | 10.4% | 13.8% | 14/21 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 10 | 2 | 1 | 9 | 1 | 9.1% | 10.0% | 16/21 |
| 286 | PACKTEST · MACD Line v2 · trigger | 21 | 4 | 0 | 21 | 4 | 0.0% | 0.0% | 20/21 |
| 287 | PACKTEST · MACD Line v2 · gate | 10 | 4 | 2 | 8 | 2 | 16.7% | 20.0% | 7/21 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 17 | 11 | 4 | 13 | 8 | 16.0% | 23.5% | 8/21 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 73 | 17 | 7 | 66 | 10 | 8.4% | 9.6% | 17/21 |
| 291 | PACKTEST · Relative Volume v2 · gate | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 11 | 5 | 1 | 10 | 4 | 6.7% | 9.1% | 19/21 |
| 293 | PACKTEST · Support Resistance Channels · gate | 18 | 12 | 3 | 15 | 9 | 11.1% | 16.7% | 13/21 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 16 | 8 | 2 | 14 | 6 | 9.1% | 12.5% | 15/21 |
| 296 | PACKTEST · Strat Assistant · trigger | 39 | 41 | 30 | 9 | 11 | 60.0% | 76.9% | 1/21 |
| 297 | PACKTEST · Strat Assistant · gate | 6 | 6 | 0 | 6 | 6 | 0.0% | 0.0% | 21/21 |
| 298 | PACKTEST · SuperTrend · trigger | 21 | 6 | 3 | 18 | 2 | 13.0% | 14.3% | 12/21 |
| 299 | PACKTEST · SuperTrend · gate | 24 | 7 | 4 | 20 | 2 | 15.4% | 16.7% | 10/21 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 2 | 2 | 1 | 1 | 1 | 33.3% | 50.0% | 4/21 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 28 | 16 | 6 | 22 | 10 | 15.8% | 21.4% | 9/21 |
| 302 | PACKTEST · UT Bot V4 · trigger | 37 | 16 | 10 | 27 | 5 | 23.8% | 27.0% | 5/21 |
| 303 | PACKTEST · UT Bot V4 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-05T10 UTC

_Strategies with activity this hour: 25; ranked (alerts≥1, BT≥1): 25_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 16 | 1 | 0 | 16 | 1 | 0.0% | 0.0% | 23/25 |
| 277 | PACKTEST · Bollinger Bands · gate | 20 | 10 | 7 | 13 | 3 | 30.4% | 35.0% | 6/25 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 22 | 15 | 11 | 11 | 4 | 42.3% | 50.0% | 2/25 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 24 | 12 | 8 | 16 | 4 | 28.6% | 33.3% | 8/25 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 22 | 15 | 11 | 11 | 4 | 42.3% | 50.0% | 3/25 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 24 | 12 | 8 | 16 | 4 | 28.6% | 33.3% | 9/25 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 14 | 3 | 0 | 14 | 3 | 0.0% | 0.0% | 24/25 |
| 283 | PACKTEST · EMA Stack v2 · gate | 26 | 16 | 9 | 17 | 7 | 27.3% | 34.6% | 11/25 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 58 | 18 | 4 | 54 | 14 | 5.6% | 6.9% | 19/25 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 6 | 6 | 3 | 3 | 3 | 33.3% | 50.0% | 5/25 |
| 286 | PACKTEST · MACD Line v2 · trigger | 22 | 5 | 0 | 22 | 5 | 0.0% | 0.0% | 22/25 |
| 287 | PACKTEST · MACD Line v2 · gate | 22 | 14 | 9 | 13 | 5 | 33.3% | 40.9% | 4/25 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 16 | 3 | 1 | 15 | 2 | 5.6% | 6.2% | 20/25 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 72 | 18 | 10 | 62 | 8 | 12.5% | 13.9% | 15/25 |
| 292 | PACKTEST · Support Resistance Channels · trig | 12 | 3 | 0 | 12 | 3 | 0.0% | 0.0% | 25/25 |
| 293 | PACKTEST · Support Resistance Channels · gate | 22 | 3 | 1 | 21 | 2 | 4.2% | 4.5% | 21/25 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 13 | 6 | 2 | 11 | 4 | 11.8% | 15.4% | 16/25 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 8 | 8 | 2 | 6 | 6 | 14.3% | 25.0% | 13/25 |
| 296 | PACKTEST · Strat Assistant · trigger | 32 | 30 | 25 | 7 | 6 | 65.8% | 78.1% | 1/25 |
| 297 | PACKTEST · Strat Assistant · gate | 16 | 8 | 2 | 14 | 6 | 9.1% | 12.5% | 18/25 |
| 298 | PACKTEST · SuperTrend · trigger | 22 | 4 | 3 | 19 | 1 | 13.0% | 13.6% | 14/25 |
| 299 | PACKTEST · SuperTrend · gate | 24 | 17 | 9 | 15 | 8 | 28.1% | 37.5% | 10/25 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 28 | 19 | 9 | 19 | 10 | 23.7% | 32.1% | 12/25 |
| 302 | PACKTEST · UT Bot V4 · trigger | 30 | 19 | 11 | 19 | 8 | 28.9% | 36.7% | 7/25 |
| 303 | PACKTEST · UT Bot V4 · gate | 8 | 2 | 1 | 7 | 1 | 11.1% | 12.5% | 17/25 |

---

## Hour 2026-06-05T11 UTC

_Strategies with activity this hour: 28; ranked (alerts≥1, BT≥1): 23_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 4 | 1 | 9 | 3 | 7.7% | 10.0% | 20/23 |
| 277 | PACKTEST · Bollinger Bands · gate | 6 | 2 | 1 | 5 | 1 | 14.3% | 16.7% | 17/23 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 48 | 42 | 25 | 23 | 14 | 40.3% | 52.1% | 4/23 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 48 | 42 | 25 | 23 | 14 | 40.3% | 52.1% | 5/23 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 16 | 6 | 1 | 15 | 5 | 4.8% | 6.2% | 21/23 |
| 283 | PACKTEST · EMA Stack v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 83 | 62 | 33 | 50 | 29 | 29.5% | 39.8% | 9/23 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 8 | 4 | 2 | 6 | 2 | 20.0% | 25.0% | 14/23 |
| 286 | PACKTEST · MACD Line v2 · trigger | 26 | 18 | 7 | 19 | 11 | 18.9% | 26.9% | 15/23 |
| 287 | PACKTEST · MACD Line v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 26 | 26 | 18 | 8 | 8 | 52.9% | 69.2% | 2/23 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 86 | 45 | 32 | 54 | 14 | 32.0% | 37.2% | 7/23 |
| 291 | PACKTEST · Relative Volume v2 · gate | 6 | 6 | 3 | 3 | 3 | 33.3% | 50.0% | 6/23 |
| 292 | PACKTEST · Support Resistance Channels · trig | 18 | 20 | 4 | 14 | 17 | 11.4% | 22.2% | 18/23 |
| 293 | PACKTEST · Support Resistance Channels · gate | 52 | 16 | 9 | 43 | 7 | 15.3% | 17.3% | 16/23 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 26 | 22 | 8 | 18 | 14 | 20.0% | 30.8% | 13/23 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 74 | 81 | 53 | 21 | 29 | 51.5% | 71.6% | 3/23 |
| 297 | PACKTEST · Strat Assistant · gate | 4 | 4 | 0 | 4 | 4 | 0.0% | 0.0% | 23/23 |
| 298 | PACKTEST · SuperTrend · trigger | 18 | 8 | 2 | 16 | 6 | 8.3% | 11.1% | 19/23 |
| 299 | PACKTEST · SuperTrend · gate | 24 | 18 | 9 | 15 | 8 | 28.1% | 37.5% | 10/23 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 8 | 10 | 7 | 1 | 4 | 58.3% | 87.5% | 1/23 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 40 | 44 | 15 | 25 | 28 | 22.1% | 37.5% | 12/23 |
| 302 | PACKTEST · UT Bot V4 · trigger | 62 | 44 | 25 | 37 | 17 | 31.6% | 40.3% | 8/23 |
| 303 | PACKTEST · UT Bot V4 · gate | 12 | 4 | 3 | 9 | 1 | 23.1% | 25.0% | 11/23 |
| 304 | PACKTEST · VWAP v2 · trigger | 12 | 5 | 0 | 12 | 5 | 0.0% | 0.0% | 22/23 |

---

## Hour 2026-06-05T12 UTC

_Strategies with activity this hour: 22; ranked (alerts≥1, BT≥1): 20_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 10 | 6 | 4 | 3 | 46.2% | 60.0% | 11/20 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 86 | 75 | 62 | 24 | 13 | 62.6% | 72.1% | 2/20 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 86 | 75 | 62 | 24 | 13 | 62.6% | 72.1% | 3/20 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 12 | 8 | 6 | 6 | 3 | 40.0% | 50.0% | 15/20 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 106 | 106 | 76 | 30 | 30 | 55.9% | 71.7% | 6/20 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 16 | 16 | 10 | 6 | 4 | 50.0% | 62.5% | 8/20 |
| 286 | PACKTEST · MACD Line v2 · trigger | 30 | 22 | 13 | 17 | 9 | 33.3% | 43.3% | 17/20 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 32 | 32 | 24 | 8 | 7 | 61.5% | 75.0% | 4/20 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 94 | 76 | 55 | 39 | 17 | 49.5% | 58.5% | 9/20 |
| 291 | PACKTEST · Relative Volume v2 · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 30 | 23 | 17 | 13 | 6 | 47.2% | 56.7% | 10/20 |
| 293 | PACKTEST · Support Resistance Channels · gate | 40 | 14 | 7 | 33 | 5 | 15.6% | 17.5% | 19/20 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 31 | 25 | 16 | 15 | 9 | 40.0% | 51.6% | 13/20 |
| 296 | PACKTEST · Strat Assistant · trigger | 116 | 114 | 82 | 34 | 31 | 55.8% | 70.7% | 7/20 |
| 297 | PACKTEST · Strat Assistant · gate | 8 | 10 | 1 | 7 | 9 | 5.9% | 12.5% | 20/20 |
| 298 | PACKTEST · SuperTrend · trigger | 20 | 14 | 10 | 10 | 3 | 43.5% | 50.0% | 12/20 |
| 299 | PACKTEST · SuperTrend · gate | 26 | 0 | 0 | 26 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 29 | 29 | 24 | 5 | 5 | 70.6% | 82.8% | 1/20 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 36 | 68 | 25 | 11 | 41 | 32.5% | 69.4% | 18/20 |
| 302 | PACKTEST · UT Bot V4 · trigger | 76 | 68 | 54 | 22 | 12 | 61.4% | 71.1% | 5/20 |
| 303 | PACKTEST · UT Bot V4 · gate | 24 | 18 | 12 | 12 | 6 | 40.0% | 50.0% | 14/20 |
| 304 | PACKTEST · VWAP v2 · trigger | 14 | 12 | 7 | 7 | 5 | 36.8% | 50.0% | 16/20 |

---

## Hour 2026-06-05T13 UTC

_Strategies with activity this hour: 32; ranked (alerts≥1, BT≥1): 26_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 263 | TSLA-CANARY-10s-NoConf | 41 | 26 | 16 | 25 | 12 | 30.2% | 39.0% | 19/26 |
| 265 | TSLA-CANARY-1m-Control | 0 | 3 | 0 | 0 | 3 | 0.0% | 0.0% | — |
| 266 | TSLA-CANARY-5m-Control | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 41 | 26 | 16 | 25 | 12 | 30.2% | 39.0% | 20/26 |
| 268 | SPY-CANARY-10s-NoConf | 30 | 26 | 17 | 13 | 9 | 43.6% | 56.7% | 14/26 |
| 269 | SPY-CANARY-1m-Control | 4 | 2 | 2 | 2 | 0 | 50.0% | 50.0% | 13/26 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 30 | 26 | 17 | 13 | 9 | 43.6% | 56.7% | 15/26 |
| 271 | TEST-P2-multipack-10s-0601-SPY | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | 0 | 6 | 0 | 0 | 6 | 0.0% | 0.0% | — |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 41 | 26 | 16 | 25 | 12 | 30.2% | 39.0% | 21/26 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 41 | 0 | 0 | 41 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 12 | 10 | 9 | 3 | 1 | 69.2% | 75.0% | 4/26 |
| 277 | PACKTEST · Bollinger Bands · gate | 26 | 4 | 3 | 23 | 1 | 11.1% | 11.5% | 24/26 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 86 | 91 | 65 | 21 | 20 | 61.3% | 75.6% | 6/26 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 86 | 91 | 65 | 21 | 20 | 61.3% | 75.6% | 7/26 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 12 | 12 | 11 | 1 | 1 | 84.6% | 91.7% | 1/26 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 106 | 104 | 80 | 26 | 18 | 64.5% | 75.5% | 5/26 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 34 | 10 | 5 | 29 | 5 | 12.8% | 14.7% | 23/26 |
| 286 | PACKTEST · MACD Line v2 · trigger | 38 | 36 | 30 | 8 | 5 | 69.8% | 78.9% | 3/26 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 52 | 56 | 40 | 12 | 14 | 60.6% | 76.9% | 8/26 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 84 | 91 | 60 | 24 | 25 | 55.0% | 71.4% | 11/26 |
| 291 | PACKTEST · Relative Volume v2 · gate | 26 | 8 | 5 | 21 | 3 | 17.2% | 19.2% | 22/26 |
| 292 | PACKTEST · Support Resistance Channels · trig | 46 | 45 | 25 | 21 | 20 | 37.9% | 54.3% | 16/26 |
| 293 | PACKTEST · Support Resistance Channels · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 27 | 25 | 23 | 4 | 2 | 79.3% | 85.2% | 2/26 |
| 296 | PACKTEST · Strat Assistant · trigger | 120 | 121 | 87 | 33 | 28 | 58.8% | 72.5% | 9/26 |
| 297 | PACKTEST · Strat Assistant · gate | 32 | 6 | 1 | 31 | 5 | 2.7% | 3.1% | 26/26 |
| 298 | PACKTEST · SuperTrend · trigger | 16 | 12 | 7 | 9 | 5 | 33.3% | 43.8% | 18/26 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 35 | 37 | 25 | 10 | 12 | 53.2% | 71.4% | 12/26 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 46 | 66 | 29 | 17 | 37 | 34.9% | 63.0% | 17/26 |
| 302 | PACKTEST · UT Bot V4 · trigger | 66 | 66 | 47 | 19 | 18 | 56.0% | 71.2% | 10/26 |
| 303 | PACKTEST · UT Bot V4 · gate | 30 | 2 | 1 | 29 | 1 | 3.2% | 3.3% | 25/26 |

---

## Hour 2026-06-05T14 UTC

_Strategies with activity this hour: 33; ranked (alerts≥1, BT≥1): 28_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 9 | 0 | 0 | 9 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 46 | 53 | 35 | 11 | 18 | 54.7% | 76.1% | 19/28 |
| 265 | TSLA-CANARY-1m-Control | 9 | 9 | 6 | 3 | 3 | 50.0% | 66.7% | 22/28 |
| 267 | TSLA-CANARY-10s-LooseConf | 46 | 53 | 35 | 11 | 18 | 54.7% | 76.1% | 20/28 |
| 268 | SPY-CANARY-10s-NoConf | 50 | 50 | 47 | 3 | 3 | 88.7% | 94.0% | 5/28 |
| 269 | SPY-CANARY-1m-Control | 7 | 9 | 7 | 0 | 2 | 77.8% | 100.0% | 15/28 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 50 | 50 | 47 | 3 | 3 | 88.7% | 94.0% | 6/28 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 46 | 53 | 35 | 11 | 18 | 54.7% | 76.1% | 21/28 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 46 | 0 | 0 | 46 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 9 | 9 | 9 | 0 | 0 | 100.0% | 100.0% | 4/28 |
| 277 | PACKTEST · Bollinger Bands · gate | 50 | 6 | 4 | 46 | 2 | 7.7% | 8.0% | 28/28 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 75 | 75 | 62 | 13 | 11 | 72.1% | 82.7% | 16/28 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 75 | 75 | 62 | 13 | 11 | 72.1% | 82.7% | 17/28 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 11 | 11 | 11 | 0 | 0 | 100.0% | 100.0% | 2/28 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 117 | 117 | 104 | 13 | 10 | 81.9% | 88.9% | 9/28 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 50 | 22 | 20 | 30 | 2 | 38.5% | 40.0% | 24/28 |
| 286 | PACKTEST · MACD Line v2 · trigger | 24 | 24 | 21 | 3 | 3 | 77.8% | 87.5% | 14/28 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 29 | 29 | 27 | 2 | 2 | 87.1% | 93.1% | 7/28 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 64 | 64 | 56 | 8 | 7 | 78.9% | 87.5% | 11/28 |
| 291 | PACKTEST · Relative Volume v2 · gate | 50 | 0 | 0 | 50 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 29 | 24 | 9 | 20 | 15 | 20.5% | 31.0% | 26/28 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 22 | 22 | 22 | 0 | 0 | 100.0% | 100.0% | 1/28 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 119 | 119 | 106 | 13 | 12 | 80.9% | 89.1% | 10/28 |
| 297 | PACKTEST · Strat Assistant · gate | 50 | 8 | 8 | 42 | 0 | 16.0% | 16.0% | 27/28 |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 10 | 10 | 0 | 0 | 100.0% | 100.0% | 3/28 |
| 299 | PACKTEST · SuperTrend · gate | 44 | 38 | 33 | 11 | 5 | 67.3% | 75.0% | 18/28 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 53 | 53 | 48 | 5 | 5 | 82.8% | 90.6% | 8/28 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 50 | 50 | 44 | 6 | 6 | 78.6% | 88.0% | 12/28 |
| 302 | PACKTEST · UT Bot V4 · trigger | 50 | 50 | 44 | 6 | 6 | 78.6% | 88.0% | 13/28 |
| 303 | PACKTEST · UT Bot V4 · gate | 41 | 12 | 9 | 32 | 3 | 20.5% | 22.0% | 25/28 |
| 304 | PACKTEST · VWAP v2 · trigger | 12 | 14 | 8 | 4 | 5 | 47.1% | 66.7% | 23/28 |

---

## Hour 2026-06-05T15 UTC

_Strategies with activity this hour: 29; ranked (alerts≥1, BT≥1): 26_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 7 | 0 | 0 | 7 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 43 | 48 | 41 | 2 | 7 | 82.0% | 95.3% | 3/26 |
| 265 | TSLA-CANARY-1m-Control | 7 | 7 | 6 | 1 | 1 | 75.0% | 85.7% | 6/26 |
| 267 | TSLA-CANARY-10s-LooseConf | 35 | 48 | 33 | 2 | 15 | 66.0% | 94.3% | 10/26 |
| 268 | SPY-CANARY-10s-NoConf | 49 | 55 | 39 | 10 | 16 | 60.0% | 79.6% | 16/26 |
| 269 | SPY-CANARY-1m-Control | 3 | 8 | 3 | 0 | 5 | 37.5% | 100.0% | 21/26 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 49 | 55 | 39 | 10 | 16 | 60.0% | 79.6% | 17/26 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 43 | 48 | 41 | 2 | 7 | 82.0% | 95.3% | 4/26 |
| 275 | TEST-P2-stoch-10s-0601-TSLA | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 13 | 15 | 13 | 0 | 2 | 86.7% | 100.0% | 2/26 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 86 | 92 | 68 | 18 | 21 | 63.6% | 79.1% | 12/26 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 86 | 92 | 68 | 18 | 21 | 63.6% | 79.1% | 13/26 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 12 | 17 | 10 | 2 | 7 | 52.6% | 83.3% | 19/26 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 105 | 108 | 84 | 21 | 20 | 67.2% | 80.0% | 9/26 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 23 | 3 | 1 | 22 | 2 | 4.0% | 4.3% | 25/26 |
| 286 | PACKTEST · MACD Line v2 · trigger | 23 | 26 | 16 | 7 | 10 | 48.5% | 69.6% | 20/26 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 47 | 52 | 34 | 13 | 15 | 54.8% | 72.3% | 18/26 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 56 | 64 | 45 | 11 | 17 | 61.6% | 80.4% | 15/26 |
| 292 | PACKTEST · Support Resistance Channels · trig | 19 | 21 | 7 | 12 | 14 | 21.2% | 36.8% | 22/26 |
| 293 | PACKTEST · Support Resistance Channels · gate | 0 | 8 | 0 | 0 | 8 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 32 | 29 | 27 | 5 | 3 | 77.1% | 84.4% | 5/26 |
| 296 | PACKTEST · Strat Assistant · trigger | 109 | 118 | 89 | 20 | 23 | 67.4% | 81.7% | 8/26 |
| 297 | PACKTEST · Strat Assistant · gate | 19 | 8 | 2 | 17 | 6 | 8.0% | 10.5% | 23/26 |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 10 | 9 | 1 | 0 | 90.0% | 90.0% | 1/26 |
| 299 | PACKTEST · SuperTrend · gate | 22 | 8 | 2 | 20 | 6 | 7.1% | 9.1% | 24/26 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 46 | 53 | 40 | 6 | 13 | 67.8% | 87.0% | 7/26 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 47 | 55 | 39 | 8 | 13 | 65.0% | 83.0% | 11/26 |
| 302 | PACKTEST · UT Bot V4 · trigger | 49 | 55 | 39 | 10 | 13 | 62.9% | 79.6% | 14/26 |
| 303 | PACKTEST · UT Bot V4 · gate | 46 | 4 | 0 | 46 | 4 | 0.0% | 0.0% | 26/26 |

---

## Hour 2026-06-05T16 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 25_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 49 | 52 | 46 | 3 | 6 | 83.6% | 93.9% | 3/25 |
| 265 | TSLA-CANARY-1m-Control | 8 | 5 | 5 | 3 | 0 | 62.5% | 62.5% | 19/25 |
| 267 | TSLA-CANARY-10s-LooseConf | 45 | 52 | 43 | 2 | 9 | 79.6% | 95.6% | 7/25 |
| 268 | SPY-CANARY-10s-NoConf | 46 | 52 | 40 | 6 | 12 | 69.0% | 87.0% | 14/25 |
| 269 | SPY-CANARY-1m-Control | 7 | 9 | 5 | 2 | 4 | 45.5% | 71.4% | 21/25 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 46 | 52 | 40 | 6 | 12 | 69.0% | 87.0% | 15/25 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 49 | 52 | 46 | 3 | 6 | 83.6% | 93.9% | 4/25 |
| 276 | PACKTEST · Bollinger Bands · trigger | 8 | 8 | 6 | 2 | 2 | 60.0% | 75.0% | 20/25 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 57 | 66 | 48 | 9 | 17 | 64.9% | 84.2% | 16/25 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 57 | 66 | 48 | 9 | 17 | 64.9% | 84.2% | 17/25 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 11 | 11 | 9 | 2 | 2 | 69.2% | 81.8% | 12/25 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 95 | 100 | 78 | 17 | 18 | 69.0% | 82.1% | 13/25 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 46 | 12 | 11 | 35 | 0 | 23.9% | 23.9% | 22/25 |
| 286 | PACKTEST · MACD Line v2 · trigger | 23 | 23 | 21 | 2 | 2 | 84.0% | 91.3% | 2/25 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 38 | 42 | 34 | 4 | 6 | 77.3% | 89.5% | 8/25 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 68 | 72 | 59 | 9 | 12 | 73.8% | 86.8% | 10/25 |
| 292 | PACKTEST · Support Resistance Channels · trig | 20 | 19 | 7 | 13 | 13 | 21.2% | 35.0% | 24/25 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 36 | 36 | 36 | 0 | 0 | 100.0% | 100.0% | 1/25 |
| 296 | PACKTEST · Strat Assistant · trigger | 96 | 98 | 82 | 14 | 15 | 73.9% | 85.4% | 9/25 |
| 297 | PACKTEST · Strat Assistant · gate | 20 | 8 | 2 | 18 | 6 | 7.7% | 10.0% | 25/25 |
| 298 | PACKTEST · SuperTrend · trigger | 8 | 10 | 7 | 1 | 3 | 63.6% | 87.5% | 18/25 |
| 299 | PACKTEST · SuperTrend · gate | 5 | 0 | 0 | 5 | 0 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 60 | 64 | 52 | 8 | 11 | 73.2% | 86.7% | 11/25 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 46 | 52 | 43 | 3 | 8 | 79.6% | 93.5% | 5/25 |
| 302 | PACKTEST · UT Bot V4 · trigger | 46 | 52 | 43 | 3 | 8 | 79.6% | 93.5% | 6/25 |
| 303 | PACKTEST · UT Bot V4 · gate | 44 | 13 | 11 | 33 | 2 | 23.9% | 25.0% | 23/25 |

---

## Hour 2026-06-05T17 UTC

_Strategies with activity this hour: 32; ranked (alerts≥1, BT≥1): 29_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 1 | 0 | 0 | 1 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 46 | 52 | 42 | 4 | 10 | 75.0% | 91.3% | 5/29 |
| 265 | TSLA-CANARY-1m-Control | 9 | 12 | 7 | 2 | 5 | 50.0% | 77.8% | 21/29 |
| 266 | TSLA-CANARY-5m-Control | 0 | 2 | 0 | 0 | 2 | 0.0% | 0.0% | — |
| 267 | TSLA-CANARY-10s-LooseConf | 46 | 52 | 42 | 4 | 10 | 75.0% | 91.3% | 6/29 |
| 268 | SPY-CANARY-10s-NoConf | 50 | 52 | 45 | 5 | 7 | 78.9% | 90.0% | 2/29 |
| 269 | SPY-CANARY-1m-Control | 7 | 15 | 6 | 1 | 9 | 37.5% | 85.7% | 22/29 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 50 | 52 | 45 | 5 | 7 | 78.9% | 90.0% | 3/29 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 46 | 52 | 42 | 4 | 10 | 75.0% | 91.3% | 7/29 |
| 276 | PACKTEST · Bollinger Bands · trigger | 13 | 14 | 11 | 2 | 3 | 68.8% | 84.6% | 16/29 |
| 277 | PACKTEST · Bollinger Bands · gate | 15 | 2 | 0 | 15 | 2 | 0.0% | 0.0% | 29/29 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 80 | 80 | 62 | 18 | 17 | 63.9% | 77.5% | 19/29 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 80 | 80 | 62 | 18 | 17 | 63.9% | 77.5% | 20/29 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 17 | 19 | 15 | 2 | 3 | 75.0% | 88.2% | 8/29 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 95 | 97 | 78 | 17 | 18 | 69.0% | 82.1% | 15/29 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 49 | 21 | 14 | 35 | 7 | 25.0% | 28.6% | 24/29 |
| 286 | PACKTEST · MACD Line v2 · trigger | 31 | 33 | 26 | 5 | 6 | 70.3% | 83.9% | 12/29 |
| 287 | PACKTEST · MACD Line v2 · gate | 15 | 13 | 12 | 3 | 1 | 75.0% | 80.0% | 9/29 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 44 | 46 | 35 | 9 | 9 | 66.0% | 79.5% | 18/29 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 59 | 64 | 50 | 9 | 14 | 68.5% | 84.7% | 17/29 |
| 291 | PACKTEST · Relative Volume v2 · gate | 9 | 0 | 0 | 9 | 0 | 0.0% | 0.0% | — |
| 292 | PACKTEST · Support Resistance Channels · trig | 37 | 36 | 11 | 26 | 25 | 17.7% | 29.7% | 26/29 |
| 293 | PACKTEST · Support Resistance Channels · gate | 13 | 43 | 12 | 1 | 31 | 27.3% | 92.3% | 23/29 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 13 | 13 | 12 | 1 | 1 | 85.7% | 92.3% | 1/29 |
| 296 | PACKTEST · Strat Assistant · trigger | 115 | 119 | 97 | 18 | 18 | 72.9% | 84.3% | 10/29 |
| 297 | PACKTEST · Strat Assistant · gate | 39 | 4 | 2 | 37 | 2 | 4.9% | 5.1% | 28/29 |
| 298 | PACKTEST · SuperTrend · trigger | 14 | 16 | 13 | 1 | 3 | 76.5% | 92.9% | 4/29 |
| 299 | PACKTEST · SuperTrend · gate | 49 | 15 | 12 | 37 | 3 | 23.1% | 24.5% | 25/29 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 58 | 62 | 50 | 8 | 12 | 71.4% | 86.2% | 11/29 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 50 | 52 | 42 | 8 | 10 | 70.0% | 84.0% | 13/29 |
| 302 | PACKTEST · UT Bot V4 · trigger | 50 | 52 | 42 | 8 | 10 | 70.0% | 84.0% | 14/29 |
| 303 | PACKTEST · UT Bot V4 · gate | 54 | 13 | 10 | 44 | 3 | 17.5% | 18.5% | 27/29 |

---

## Hour 2026-06-05T18 UTC

_Strategies with activity this hour: 30; ranked (alerts≥1, BT≥1): 29_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 58 | 60 | 51 | 7 | 9 | 76.1% | 87.9% | 3/29 |
| 265 | TSLA-CANARY-1m-Control | 9 | 9 | 6 | 3 | 3 | 50.0% | 66.7% | 21/29 |
| 267 | TSLA-CANARY-10s-LooseConf | 56 | 60 | 50 | 6 | 10 | 75.8% | 89.3% | 5/29 |
| 268 | SPY-CANARY-10s-NoConf | 53 | 56 | 46 | 7 | 10 | 73.0% | 86.8% | 6/29 |
| 269 | SPY-CANARY-1m-Control | 6 | 5 | 4 | 2 | 1 | 57.1% | 66.7% | 20/29 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 53 | 56 | 46 | 7 | 10 | 73.0% | 86.8% | 7/29 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 58 | 60 | 51 | 7 | 9 | 76.1% | 87.9% | 4/29 |
| 276 | PACKTEST · Bollinger Bands · trigger | 3 | 4 | 2 | 1 | 2 | 40.0% | 66.7% | 22/29 |
| 277 | PACKTEST · Bollinger Bands · gate | 3 | 4 | 1 | 2 | 3 | 16.7% | 33.3% | 26/29 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 83 | 86 | 67 | 16 | 16 | 67.7% | 80.7% | 10/29 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 83 | 86 | 67 | 16 | 16 | 67.7% | 80.7% | 11/29 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 12 | 13 | 11 | 1 | 2 | 78.6% | 91.7% | 2/29 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 84 | 97 | 65 | 19 | 29 | 57.5% | 77.4% | 19/29 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 6 | 4 | 1 | 5 | 3 | 11.1% | 16.7% | 27/29 |
| 286 | PACKTEST · MACD Line v2 · trigger | 24 | 26 | 20 | 4 | 6 | 66.7% | 83.3% | 13/29 |
| 287 | PACKTEST · MACD Line v2 · gate | 3 | 5 | 2 | 1 | 3 | 33.3% | 66.7% | 23/29 |
| 288 | PACKTEST · RSI Zones 2 · trigger | 35 | 36 | 29 | 6 | 6 | 70.7% | 82.9% | 9/29 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 71 | 75 | 54 | 17 | 17 | 61.4% | 76.1% | 16/29 |
| 292 | PACKTEST · Support Resistance Channels · trig | 24 | 22 | 1 | 23 | 21 | 2.2% | 4.2% | 28/29 |
| 293 | PACKTEST · Support Resistance Channels · gate | 7 | 9 | 6 | 1 | 3 | 60.0% | 85.7% | 17/29 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 25 | 43 | 25 | 0 | 18 | 58.1% | 100.0% | 18/29 |
| 296 | PACKTEST · Strat Assistant · trigger | 90 | 97 | 73 | 17 | 20 | 66.4% | 81.1% | 14/29 |
| 297 | PACKTEST · Strat Assistant · gate | 22 | 6 | 0 | 22 | 6 | 0.0% | 0.0% | 29/29 |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 12 | 10 | 0 | 2 | 83.3% | 100.0% | 1/29 |
| 299 | PACKTEST · SuperTrend · gate | 3 | 5 | 2 | 1 | 3 | 33.3% | 66.7% | 24/29 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 62 | 65 | 51 | 11 | 14 | 67.1% | 82.3% | 12/29 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 47 | 56 | 40 | 7 | 16 | 63.5% | 85.1% | 15/29 |
| 302 | PACKTEST · UT Bot V4 · trigger | 53 | 56 | 46 | 7 | 10 | 73.0% | 86.8% | 8/29 |
| 303 | PACKTEST · UT Bot V4 · gate | 14 | 8 | 5 | 9 | 3 | 29.4% | 35.7% | 25/29 |

---

## Hour 2026-06-05T19 UTC

_Strategies with activity this hour: 32; ranked (alerts≥1, BT≥1): 29_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 174 | TSLA LONG 1Min Mass #2 | 8 | 0 | 0 | 8 | 0 | 0.0% | 0.0% | — |
| 263 | TSLA-CANARY-10s-NoConf | 39 | 51 | 28 | 11 | 23 | 45.2% | 71.8% | 22/29 |
| 265 | TSLA-CANARY-1m-Control | 7 | 8 | 5 | 2 | 3 | 50.0% | 71.4% | 20/29 |
| 266 | TSLA-CANARY-5m-Control | 1 | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 2/29 |
| 267 | TSLA-CANARY-10s-LooseConf | 39 | 51 | 28 | 11 | 23 | 45.2% | 71.8% | 23/29 |
| 268 | SPY-CANARY-10s-NoConf | 45 | 45 | 38 | 7 | 7 | 73.1% | 84.4% | 7/29 |
| 269 | SPY-CANARY-1m-Control | 8 | 10 | 7 | 1 | 3 | 63.6% | 87.5% | 17/29 |
| 270 | TEST-P2-utbotv4-10s-0601-SPY | 45 | 45 | 38 | 7 | 7 | 73.1% | 84.4% | 8/29 |
| 273 | TEST-P2-utbotv4-10s-0601-TSLA | 39 | 51 | 28 | 11 | 23 | 45.2% | 71.8% | 24/29 |
| 276 | PACKTEST · Bollinger Bands · trigger | 10 | 8 | 8 | 2 | 0 | 80.0% | 80.0% | 5/29 |
| 277 | PACKTEST · Bollinger Bands · gate | 8 | 2 | 2 | 6 | 0 | 25.0% | 25.0% | 26/29 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 75 | 75 | 59 | 16 | 15 | 65.6% | 78.7% | 14/29 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 75 | 75 | 59 | 16 | 15 | 65.6% | 78.7% | 15/29 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 15 | 15 | 14 | 1 | 1 | 87.5% | 93.3% | 3/29 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 101 | 99 | 81 | 20 | 15 | 69.8% | 80.2% | 10/29 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 45 | 18 | 15 | 30 | 3 | 31.2% | 33.3% | 25/29 |
| 286 | PACKTEST · MACD Line v2 · trigger | 40 | 38 | 31 | 9 | 7 | 66.0% | 77.5% | 13/29 |
| 287 | PACKTEST · MACD Line v2 · gate | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 33 | 33 | 23 | 10 | 10 | 53.5% | 69.7% | 19/29 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 62 | 71 | 49 | 13 | 21 | 59.0% | 79.0% | 18/29 |
| 292 | PACKTEST · Support Resistance Channels · trig | 18 | 16 | 5 | 13 | 11 | 17.2% | 27.8% | 28/29 |
| 293 | PACKTEST · Support Resistance Channels · gate | 10 | 0 | 0 | 10 | 0 | 0.0% | 0.0% | — |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 29 | 29 | 26 | 3 | 3 | 81.2% | 89.7% | 4/29 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 4 | 2 | 2 | 2 | 0 | 50.0% | 50.0% | 21/29 |
| 296 | PACKTEST · Strat Assistant · trigger | 107 | 113 | 93 | 14 | 16 | 75.6% | 86.9% | 6/29 |
| 297 | PACKTEST · Strat Assistant · gate | 45 | 8 | 6 | 39 | 2 | 12.8% | 13.3% | 29/29 |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 10 | 10 | 0 | 0 | 100.0% | 100.0% | 1/29 |
| 299 | PACKTEST · SuperTrend · gate | 24 | 24 | 19 | 5 | 5 | 65.5% | 79.2% | 16/29 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 65 | 65 | 54 | 11 | 11 | 71.1% | 83.1% | 9/29 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 45 | 45 | 37 | 8 | 8 | 69.8% | 82.2% | 11/29 |
| 302 | PACKTEST · UT Bot V4 · trigger | 45 | 45 | 37 | 8 | 8 | 69.8% | 82.2% | 12/29 |
| 303 | PACKTEST · UT Bot V4 · gate | 49 | 14 | 11 | 38 | 3 | 21.2% | 22.4% | 27/29 |

---

## Hour 2026-06-05T20 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 22_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 266 | TSLA-CANARY-5m-Control | 0 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | — |
| 276 | PACKTEST · Bollinger Bands · trigger | 16 | 17 | 15 | 1 | 2 | 83.3% | 93.8% | 2/22 |
| 277 | PACKTEST · Bollinger Bands · gate | 22 | 2 | 0 | 22 | 2 | 0.0% | 0.0% | 19/22 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 72 | 82 | 59 | 13 | 21 | 63.4% | 81.9% | 5/22 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 72 | 82 | 59 | 13 | 21 | 63.4% | 81.9% | 6/22 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 14 | 13 | 9 | 5 | 4 | 50.0% | 64.3% | 10/22 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 99 | 104 | 74 | 25 | 29 | 57.8% | 74.7% | 8/22 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 26 | 26 | 1 | 25 | 25 | 2.0% | 3.8% | 17/22 |
| 286 | PACKTEST · MACD Line v2 · trigger | 22 | 31 | 15 | 7 | 16 | 39.5% | 68.2% | 13/22 |
| 287 | PACKTEST · MACD Line v2 · gate | 22 | 0 | 0 | 22 | 0 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 36 | 43 | 29 | 7 | 12 | 60.4% | 80.6% | 7/22 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 71 | 77 | 61 | 10 | 15 | 70.9% | 85.9% | 3/22 |
| 291 | PACKTEST · Relative Volume v2 · gate | 4 | 6 | 2 | 2 | 4 | 25.0% | 50.0% | 15/22 |
| 292 | PACKTEST · Support Resistance Channels · trig | 20 | 25 | 4 | 16 | 21 | 9.8% | 20.0% | 16/22 |
| 293 | PACKTEST · Support Resistance Channels · gate | 22 | 2 | 0 | 22 | 2 | 0.0% | 0.0% | 20/22 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 24 | 24 | 17 | 7 | 7 | 54.8% | 70.8% | 9/22 |
| 295 | PACKTEST · Stochastic Oscillator · gate | 22 | 0 | 0 | 22 | 0 | 0.0% | 0.0% | — |
| 296 | PACKTEST · Strat Assistant · trigger | 109 | 118 | 89 | 20 | 24 | 66.9% | 81.7% | 4/22 |
| 297 | PACKTEST · Strat Assistant · gate | 22 | 8 | 0 | 22 | 8 | 0.0% | 0.0% | 21/22 |
| 298 | PACKTEST · SuperTrend · trigger | 10 | 13 | 6 | 4 | 7 | 35.3% | 60.0% | 14/22 |
| 299 | PACKTEST · SuperTrend · gate | 22 | 2 | 0 | 22 | 2 | 0.0% | 0.0% | 22/22 |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 41 | 43 | 39 | 2 | 2 | 90.7% | 95.1% | 1/22 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 40 | 50 | 27 | 13 | 23 | 42.9% | 67.5% | 12/22 |
| 302 | PACKTEST · UT Bot V4 · trigger | 42 | 50 | 29 | 13 | 21 | 46.0% | 69.0% | 11/22 |
| 303 | PACKTEST · UT Bot V4 · gate | 23 | 14 | 0 | 23 | 14 | 0.0% | 0.0% | 18/22 |
| 304 | PACKTEST · VWAP v2 · trigger | 19 | 0 | 0 | 19 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 4 | 0 | 0 | 4 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-05T21 UTC

_Strategies with activity this hour: 24; ranked (alerts≥1, BT≥1): 20_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 17 | 19 | 9 | 8 | 10 | 33.3% | 52.9% | 12/20 |
| 277 | PACKTEST · Bollinger Bands · gate | 12 | 18 | 6 | 6 | 12 | 25.0% | 50.0% | 13/20 |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 53 | 59 | 41 | 12 | 16 | 59.4% | 77.4% | 5/20 |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 53 | 59 | 41 | 12 | 16 | 59.4% | 77.4% | 6/20 |
| 282 | PACKTEST · EMA Stack v2 · trigger | 21 | 15 | 12 | 9 | 3 | 50.0% | 57.1% | 7/20 |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 71 | 73 | 46 | 25 | 26 | 47.4% | 64.8% | 8/20 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 26 | 14 | 7 | 19 | 7 | 21.2% | 26.9% | 15/20 |
| 286 | PACKTEST · MACD Line v2 · trigger | 22 | 28 | 14 | 8 | 14 | 38.9% | 63.6% | 11/20 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 34 | 31 | 27 | 7 | 4 | 71.1% | 79.4% | 2/20 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 58 | 57 | 35 | 23 | 22 | 43.8% | 60.3% | 9/20 |
| 292 | PACKTEST · Support Resistance Channels · trig | 18 | 22 | 7 | 11 | 15 | 21.2% | 38.9% | 16/20 |
| 293 | PACKTEST · Support Resistance Channels · gate | 7 | 47 | 3 | 4 | 44 | 5.9% | 42.9% | 19/20 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 12 | 7 | 3 | 9 | 4 | 18.8% | 25.0% | 17/20 |
| 296 | PACKTEST · Strat Assistant · trigger | 55 | 71 | 47 | 8 | 24 | 59.5% | 85.5% | 4/20 |
| 297 | PACKTEST · Strat Assistant · gate | 5 | 11 | 1 | 4 | 10 | 6.7% | 20.0% | 18/20 |
| 298 | PACKTEST · SuperTrend · trigger | 12 | 9 | 4 | 8 | 5 | 23.5% | 33.3% | 14/20 |
| 299 | PACKTEST · SuperTrend · gate | 0 | 17 | 0 | 0 | 17 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 25 | 25 | 23 | 2 | 2 | 85.2% | 92.0% | 1/20 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 27 | 47 | 22 | 5 | 24 | 43.1% | 81.5% | 10/20 |
| 302 | PACKTEST · UT Bot V4 · trigger | 48 | 47 | 37 | 11 | 9 | 64.9% | 77.1% | 3/20 |
| 303 | PACKTEST · UT Bot V4 · gate | 4 | 8 | 0 | 4 | 8 | 0.0% | 0.0% | 20/20 |
| 304 | PACKTEST · VWAP v2 · trigger | 25 | 0 | 0 | 25 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 14 | 0 | 0 | 14 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-05T22 UTC

_Strategies with activity this hour: 27; ranked (alerts≥1, BT≥1): 19_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 18 | 14 | 6 | 12 | 7 | 24.0% | 33.3% | 12/19 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 56 | 49 | 37 | 19 | 11 | 55.2% | 66.1% | 3/19 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 56 | 49 | 37 | 19 | 11 | 55.2% | 66.1% | 4/19 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 26 | 0 | 0 | 26 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 12 | 6 | 1 | 11 | 5 | 5.9% | 8.3% | 19/19 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 32 | 0 | 0 | 32 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 70 | 63 | 34 | 36 | 26 | 35.4% | 48.6% | 8/19 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 16 | 18 | 9 | 7 | 8 | 37.5% | 56.2% | 7/19 |
| 286 | PACKTEST · MACD Line v2 · trigger | 23 | 19 | 5 | 18 | 14 | 13.5% | 21.7% | 15/19 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 21 | 21 | 8 | 13 | 14 | 22.9% | 38.1% | 13/19 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 94 | 58 | 30 | 64 | 27 | 24.8% | 31.9% | 11/19 |
| 292 | PACKTEST · Support Resistance Channels · trig | 15 | 14 | 3 | 12 | 11 | 11.5% | 20.0% | 17/19 |
| 293 | PACKTEST · Support Resistance Channels · gate | 36 | 17 | 11 | 25 | 5 | 26.8% | 30.6% | 10/19 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 14 | 11 | 4 | 10 | 7 | 19.0% | 28.6% | 14/19 |
| 296 | PACKTEST · Strat Assistant · trigger | 75 | 83 | 58 | 17 | 24 | 58.6% | 77.3% | 2/19 |
| 297 | PACKTEST · Strat Assistant · gate | 13 | 5 | 2 | 11 | 4 | 11.8% | 15.4% | 16/19 |
| 298 | PACKTEST · SuperTrend · trigger | 21 | 11 | 3 | 18 | 7 | 10.7% | 14.3% | 18/19 |
| 299 | PACKTEST · SuperTrend · gate | 0 | 49 | 0 | 0 | 49 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 17 | 11 | 8 | 9 | 3 | 40.0% | 47.1% | 6/19 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 31 | 49 | 20 | 11 | 29 | 33.3% | 64.5% | 9/19 |
| 302 | PACKTEST · UT Bot V4 · trigger | 51 | 49 | 32 | 19 | 16 | 47.8% | 62.7% | 5/19 |
| 303 | PACKTEST · UT Bot V4 · gate | 14 | 11 | 9 | 5 | 1 | 60.0% | 64.3% | 1/19 |
| 304 | PACKTEST · VWAP v2 · trigger | 23 | 0 | 0 | 23 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 24 | 0 | 0 | 24 | 0 | 0.0% | 0.0% | — |

---

## Hour 2026-06-05T23 UTC

_Strategies with activity this hour: 28; ranked (alerts≥1, BT≥1): 20_

| sid | Strategy name | Alerts | BT events | Paired | Phantom | Missed | Combined % | Alert-pair % | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 276 | PACKTEST · Bollinger Bands · trigger | 16 | 12 | 9 | 7 | 4 | 45.0% | 56.2% | 5/20 |
| 277 | PACKTEST · Bollinger Bands · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 278 | PACKTEST · EMA Price Position v3 · trigger | 55 | 51 | 38 | 17 | 12 | 56.7% | 69.1% | 2/20 |
| 279 | PACKTEST · EMA Price Position v3 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 280 | PACKTEST · EMA Price Position v4 · trigger | 55 | 51 | 38 | 17 | 12 | 56.7% | 69.1% | 3/20 |
| 281 | PACKTEST · EMA Price Position v4 · gate | 0 | 4 | 0 | 0 | 4 | 0.0% | 0.0% | — |
| 282 | PACKTEST · EMA Stack v2 · trigger | 21 | 12 | 7 | 14 | 5 | 26.9% | 33.3% | 14/20 |
| 283 | PACKTEST · EMA Stack v2 · gate | 0 | 20 | 0 | 0 | 20 | 0.0% | 0.0% | — |
| 284 | PACKTEST · MACD Histogram v2 · trigger | 79 | 55 | 36 | 43 | 19 | 36.7% | 45.6% | 10/20 |
| 285 | PACKTEST · MACD Histogram v2 · gate | 6 | 4 | 0 | 6 | 4 | 0.0% | 0.0% | 20/20 |
| 286 | PACKTEST · MACD Line v2 · trigger | 28 | 18 | 7 | 21 | 11 | 17.9% | 25.0% | 15/20 |
| 287 | PACKTEST · MACD Line v2 · gate | 0 | 10 | 0 | 0 | 10 | 0.0% | 0.0% | — |
| 288 | PACKTEST · RSI Zones 2 · trigger | 28 | 27 | 17 | 11 | 10 | 44.7% | 60.7% | 7/20 |
| 290 | PACKTEST · Relative Volume v2 · trigger | 82 | 44 | 34 | 48 | 9 | 37.4% | 41.5% | 8/20 |
| 291 | PACKTEST · Relative Volume v2 · gate | 4 | 4 | 1 | 3 | 3 | 14.3% | 25.0% | 16/20 |
| 292 | PACKTEST · Support Resistance Channels · trig | 9 | 17 | 6 | 3 | 11 | 30.0% | 66.7% | 13/20 |
| 293 | PACKTEST · Support Resistance Channels · gate | 38 | 4 | 2 | 36 | 1 | 5.1% | 5.3% | 18/20 |
| 294 | PACKTEST · Stochastic Oscillator · trigger | 27 | 15 | 5 | 22 | 10 | 13.5% | 18.5% | 17/20 |
| 296 | PACKTEST · Strat Assistant · trigger | 65 | 67 | 46 | 19 | 21 | 53.5% | 70.8% | 4/20 |
| 297 | PACKTEST · Strat Assistant · gate | 16 | 14 | 9 | 7 | 4 | 45.0% | 56.2% | 6/20 |
| 298 | PACKTEST · SuperTrend · trigger | 26 | 21 | 13 | 13 | 9 | 37.1% | 50.0% | 9/20 |
| 299 | PACKTEST · SuperTrend · gate | 0 | 22 | 0 | 0 | 22 | 0.0% | 0.0% | — |
| 300 | PACKTEST · Swing 1-2-3 · trigger | 9 | 7 | 4 | 5 | 3 | 33.3% | 44.4% | 12/20 |
| 301 | PACKTEST · Swing 1-2-3 · gate | 29 | 45 | 19 | 10 | 25 | 35.2% | 65.5% | 11/20 |
| 302 | PACKTEST · UT Bot V4 · trigger | 59 | 45 | 38 | 21 | 7 | 57.6% | 64.4% | 1/20 |
| 303 | PACKTEST · UT Bot V4 · gate | 14 | 1 | 0 | 14 | 1 | 0.0% | 0.0% | 19/20 |
| 304 | PACKTEST · VWAP v2 · trigger | 11 | 0 | 0 | 11 | 0 | 0.0% | 0.0% | — |
| 305 | PACKTEST · VWAP v2 · gate | 2 | 0 | 0 | 2 | 0 | 0.0% | 0.0% | — |

---
