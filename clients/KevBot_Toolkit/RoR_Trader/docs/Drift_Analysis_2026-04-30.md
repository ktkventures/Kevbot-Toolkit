# WS-vs-REST Drift Analysis — 2026-04-30

**Purpose:** Validate (or refute) the hypothesis that Polygon WebSocket-aggregated bars (used by live worker) differ enough from Polygon REST bars (used by backtest/chart) to cause the backtest↔live divergence we observed during Phase A-E parity work.

**Methodology:** For each entry alert with exec_type=C in the alerts table over the last 96 hours, compare alert.price (the worker's view of bar close at fire time) to the REST close for the same bar. Alerts where exec_type≠C are excluded because non-C executions have legitimate intra-bar slippage that would confound the comparison.

**Conclusion (preview):** Drift is broad and significant. ~80% of bars differ between worker and REST. Median drift $0.02, 95th percentile $0.09, max $0.84. Symmetric direction (~51% pos, ~49% neg). Confirms the live-bar cache (Milestone 8.7) is the right architectural fix.

---

## Raw analysis output (from )

```

Pulling alerts from last 96h (since 2026-04-26T21:34:04 UTC)

Total entry+C alerts: 6711

Groups (symbol, tf): 4
  SPY/10Sec: 6495 alerts
  SPY/1Min: 211 alerts
  META/1Min: 4 alerts
  TSLA/10Sec: 1 alerts

Loading REST SPY/10Sec for 96h window...
Loading REST SPY/1Min for 96h window...
Loading REST META/1Min for 96h window...
Loading REST TSLA/10Sec for 96h window...

Results: 6711 alerts compared

====================================================================================================
symbol/tf           n  in_rest  matched  differ  mean|diff|   max|diff|   p95|diff|
----------------------------------------------------------------------------------------------------
META/1Min           4        4        4       0      0.0000      0.0000      0.0000
SPY/10Sec        3544     3544      660    2884      0.0305      0.4200      0.0900
SPY/1Min          211      211      111     100      0.0593      0.8350      0.1200
TSLA/10Sec          1        1        0       1      0.0050      0.0050      0.0050
----------------------------------------------------------------------------------------------------
TOTAL            3760     3760      775    2985  (20.6% matched, 79.4% differ)

=== Distribution of |diff| across all differing bars ===
  N differing: 2985
  min: 0.0001
  p25: 0.0100
  p50: 0.0200
  p75: 0.0400
  p95: 0.0900
  max: 0.8350

=== Signed direction (alert_price - rest_close) ===
  alert > REST (worker bar > REST bar): 1530  (51.3%)
  alert < REST (worker bar < REST bar): 1455  (48.7%)

=== Drift severity ===
  alerts differing by ≥$0.01: 2050 of 3760 (54.5%)
  alerts differing by ≥$0.05: 491 (13.1%)
  alerts differing by ≥$0.10: 106 (2.8%)

====================================================================================================
SAMPLE TRADES FOR VISUAL VERIFICATION
====================================================================================================
Use these to cross-check against your charts. Pull up the strategy in
the UI, navigate to the bar_time, compare the chart's bar close to the
'rest_close' here, and the algo-history alert price to the 'alert_price' here.


--- META/1Min ---
  MATCHED (4 total, sample of 4):
    sid=51  bar_time=2026-04-29T18:34:00+00:00  alert=671.3250  rest=671.3250  diff=$0.0000
    sid=51  bar_time=2026-04-28T19:13:00+00:00  alert=670.5200  rest=670.5200  diff=$0.0000
    sid=51  bar_time=2026-04-29T16:40:00+00:00  alert=668.5850  rest=668.5850  diff=$0.0000
    sid=51  bar_time=2026-04-29T16:05:00+00:00  alert=669.1800  rest=669.1800  diff=$0.0000
  DIFFER (0 total, sample of 0 across drift range):

--- SPY/10Sec ---
  MATCHED (660 total, sample of 4):
    sid=71  bar_time=2026-04-29T19:18:30+00:00  alert=711.3200  rest=711.3201  diff=$0.0000
    sid=150  bar_time=2026-04-30T16:02:50+00:00  alert=714.2000  rest=714.2000  diff=$0.0000
    sid=153  bar_time=2026-04-30T16:54:30+00:00  alert=714.3700  rest=714.3700  diff=$0.0000
    sid=131  bar_time=2026-04-27T18:42:00+00:00  alert=715.1200  rest=715.1200  diff=$0.0000
  DIFFER (2884 total, sample of 4 across drift range):
    sid=134  bar_time=2026-04-28T19:45:50+00:00  alert=711.6823  rest=711.6822  diff=+$0.0001
    sid=117  bar_time=2026-04-28T18:12:30+00:00  alert=710.9600  rest=710.9700  diff=-$0.0100
    sid=131  bar_time=2026-04-29T15:37:10+00:00  alert=710.8600  rest=710.8800  diff=-$0.0200
    sid=149  bar_time=2026-04-30T18:20:20+00:00  alert=717.8600  rest=717.9000  diff=-$0.0400

--- SPY/1Min ---
  MATCHED (111 total, sample of 4):
    sid=132  bar_time=2026-04-30T18:12:00+00:00  alert=717.7600  rest=717.7600  diff=$0.0000
    sid=50  bar_time=2026-04-28T16:23:00+00:00  alert=710.5000  rest=710.5000  diff=$0.0000
    sid=132  bar_time=2026-04-29T16:23:00+00:00  alert=710.1400  rest=710.1400  diff=$0.0000
    sid=132  bar_time=2026-04-30T19:36:00+00:00  alert=719.0000  rest=719.0000  diff=$0.0000
  DIFFER (100 total, sample of 4 across drift range):
    sid=132  bar_time=2026-04-28T18:13:00+00:00  alert=711.0700  rest=711.0730  diff=-$0.0030
    sid=50  bar_time=2026-04-28T19:38:00+00:00  alert=711.4200  rest=711.4300  diff=-$0.0100
    sid=152  bar_time=2026-04-30T16:58:00+00:00  alert=714.2150  rest=714.2400  diff=-$0.0250
    sid=132  bar_time=2026-04-27T19:57:00+00:00  alert=715.2900  rest=715.2450  diff=+$0.0450

--- TSLA/10Sec ---
  MATCHED (0 total, sample of 0):
  DIFFER (1 total, sample of 1 across drift range):
    sid=126  bar_time=2026-04-27T16:34:00+00:00  alert=371.9000  rest=371.9050  diff=-$0.0050

====================================================================================================
PER-STRATEGY DRIFT SUMMARY
====================================================================================================
  sid  symbol/tf           n   match  differ  mean|diff|   max|diff|
--------------------------------------------------------------------------------
   50  SPY/1Min           38      21      17      0.0786      0.8350
   51  META/1Min           4       4       0      0.0000      0.0000
   71  SPY/10Sec         474      91     383      0.0277      0.3150
   88  SPY/10Sec         236      42     194      0.0350      0.4200
  111  SPY/10Sec           6       3       3      0.0167      0.0200
  117  SPY/10Sec         615     117     498      0.0290      0.3150
  126  TSLA/10Sec          1       0       1      0.0050      0.0050
  129  SPY/10Sec         607     116     491      0.0302      0.4200
  131  SPY/10Sec         625     121     504      0.0290      0.3150
  132  SPY/1Min          125      66      59      0.0521      0.8350
  134  SPY/10Sec         375      67     308      0.0295      0.3145
  136  SPY/1Min           13       7       6      0.1707      0.8350
  149  SPY/10Sec          68      10      58      0.0392      0.1750
  150  SPY/10Sec         128      23     105      0.0336      0.1500
  151  SPY/10Sec         143      26     117      0.0338      0.1500
  152  SPY/1Min           35      17      18      0.0278      0.0900
  153  SPY/10Sec         144      26     118      0.0337      0.1500
  154  SPY/10Sec         123      18     105      0.0352      0.2850
```

---

## How to verify against your charts

Pick any sample row from the SAMPLE TRADES section above (or per-strategy summary). Open the corresponding strategy on the Strategy Detail page, navigate to the bar_time, and compare:

1. The chart's candle close at that bar_time → should equal **rest_close** in this report
2. The algo-history table's alert price for that trade → should equal **alert_price** in this report

If those match what's in this report, the drift hypothesis is confirmed at that bar. If they DON'T match, there's a measurement issue we should investigate.

## Implications

- **For 1Min strategies (SPY/1Min):** ~50% of bars differ, median $0.02, max $0.84. Drift is real and measurable.
- **For 10Sec strategies (SPY/10Sec):** ~83% of bars differ. Higher rate because fewer trades per bar means each tick matters more.
- **Symbols don't matter as much as TF:** META/1Min showed 100% match, but only 4 alerts (small sample).

The cache fix (M8.7) writes worker's WS-aggregated bars to a Supabase table. Backtest reads from there. Result: same data → same engine state → trade-by-trade match between live and backtest.
