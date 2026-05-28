# Findings — REST vs WS Latency on SPY, 2026-05-27 Extended Hours

**Goal:** Empirically measure Polygon REST latency to inform the choice
between B1 (pure REST poll) and Hybrid (REST + WS tip) for the new
live model.

**Method:** 10-minute high-frequency poll of Polygon REST aggregates
(`_polygon_rest_latency_probe.py`) every 250ms during extended hours.
Tracked each bar from first-appearance through value-settle.

## Headline numbers

### SPY per-second REST aggregates (n=122 bars)

| Stat | First-appearance latency | Value-settle latency |
|---|---|---|
| min | 1.92s | 1.92s |
| **median** | **2.61s** | **2.61s** |
| p75 | 2.94s | 2.94s |
| p90 | 3.13s | 3.13s |
| p95 | 3.24s | 3.24s |
| max | 4.16s | 4.16s |

**Key:** value-settle == first-appearance. **No late corrections.**
The bar's values are FINAL on first sight.

### SPY 1Min REST aggregates (n=11 bars, smaller sample)

| Stat | First-appearance latency | Value-settle latency |
|---|---|---|
| min | 1.10s | 1.10s |
| **median** | **2.03s** | **2.03s** |
| p75 | 2.44s | 2.44s |
| p90 | 3.71s | 3.71s |
| p95 | 58.06s | 58.06s ← outlier |
| max | 58.06s | 58.06s |

**Key:** median 2 sec, p90 under 4 sec. Single outlier at 58s — likely
a low-liquidity minute that Polygon didn't aggregate immediately.
Investigate during RTH where activity is higher.

**Confirmation from separate 3-bar sample:** consecutive 1Min closes
at 22:07, 22:08, 22:09 all first-appeared at 1.5-2.0 sec, no changes
after. So the 2.0s median holds in normal conditions; 58s outlier was
an anomaly.

## What this tells us

### For B1 (pure REST poll) grace defaults

| TF | Suggested grace | Coverage |
|---|---|---|
| 1Min+ | **3 sec** | Catches ~90% of normal closes |
| 10Sec | **4 sec** | Catches Polygon per-sec p95 + margin |
| 30Sec | **3 sec** | Lower volatility than 10Sec |
| 5Sec or below | **5 sec** | Tight TF, more margin |

### For Hybrid (REST + WS tip) — what would it actually buy?

The "WS tip" only helps if we want sub-second intra-bar visibility.
But the user's strategies fire on:
- **C-type triggers (bar-close):** `utv4_bull_flip`, `eppv3_cross_short_up`
  etc. fire ONLY on bar close. WS doesn't help here — they fire at
  exactly the same moment as B1 (bar_close + grace).
- **L-type stops (intra-bar level crosses):** fire when price crosses
  a stop level mid-bar. WS gives ms-resolution; REST gives 1-sec.
  **But backtest's Hi-Fi pass uses REST 1-sec bars too.** So matching
  REST gives backtest parity by construction. Matching WS would put
  live AHEAD of backtest, recreating the same divergence we're
  solving.

**Conclusion:** the hybrid would HURT parity. B1 alone matches backtest's
data source exactly.

## Comparison with current WS pipeline (ws_agg_reconciled)

From earlier measurements:

| Metric | Current WS path | B1 REST path |
|---|---|---|
| Median bar arrival | 7-8 sec (live_bars `written_at`) | 2-3 sec |
| p90 bar arrival | 8.7s (10Sec) / 45s (1Min) | 3.1s / 3.7s |
| Bar coverage on SPY 10Sec ext hours | 73% | ~100% |
| Close match vs REST (when both have data) | 100% | 100% by construction |
| Late corrections | Yes (Polygon WS rebroadcast handling exists) | None observed |

**B1 is materially faster AND has better coverage.**

## Honest caveat on the WS measurement

The 7-8 sec "WS path" latency I measured isn't actually WS-WS latency.
It's the time from bar close to when Ralph WRITES to `live_bars`. That
write happens after BarBuilder finalizes the bar, which depends on
when the next bar's first tick arrives PLUS Supabase upsert time. So
the bar might be IN RAM at Ralph in 1-2 sec but the persisted snapshot
to `live_bars` is delayed.

That said, the persisted snapshot is also what downstream tools (chart,
divergence dashboards) read from. So 7-8 sec is the right number for
"data visible to other systems."

For the new `rest_polled` model, Ralph reads bars directly from Polygon
REST → no Supabase round-trip → 2-3 sec actual latency to in-RAM bar.

## What I'd recommend

1. **Pure B1, no hybrid.** Cleaner architecture, better parity with
   backtest, no downside for current trigger types.
2. **Per-TF grace defaults: 3s for 1Min, 4s for 10Sec.** Tighter than
   my first-draft spec; can loosen if MVP shows misses.
3. **Investigate the 58-sec outlier on 1Min** before locking in 3s
   grace. Probably extended-hours quiet bar; RTH measurement should
   confirm or refute.
4. **Add an empirical-monitoring hook** to the MVP: record actual
   bar-receive latency on every alert so we can see if 3s grace is
   too tight on real bars.

## Hybrid simulation results (added 2026-05-27 evening)

Question: would a hybrid "REST for past, WS for last 4-5 sec" model
work as well as pure REST? Simulated by comparing Ralph's historical
`live_bars` (WS captures) against REST resamples in two windows.

### Active period (18:00-19:30 UTC, post-RTH but volume present)

| Metric | Value |
|---|---|
| WS bars captured | 401 |
| REST 10s bars (resampled from 1s) | 541 |
| Bars in BOTH | 401 |
| **WS-only (REST resample empty)** | **0** |
| **REST-only (WS DROPPED them)** | **140 = 25.9%** ← gap |
| Close match where both exist (`first_close` vs REST) | 399/401 = **99.5%** |
| Close match (settled close vs REST) | 399/401 = **99.5%** |

The 140 REST-only bars had real volume (7K-40K each). These aren't
quiet-period artifacts — they're active bars Ralph's WS aggregator
dropped.

### Quiet period (recent ~4h, extended-hours wind-down)

| Metric | Value |
|---|---|
| WS bars captured | 1,438 |
| REST 10s bars (resampled) | 227 |
| WS-only (REST resample empty) | 1,211 |
| REST-only (WS missed) | 0 |
| Close match where both exist | 227/227 = **100%** |

In quiet hours, WS emits gap-fill bars (zero volume, last close
repeated). REST resampling produces no row when no trades occurred
in the window. So WS-only ≠ "WS lies" here — it's WS's gap-fill
behavior.

### What this tells us about hybrid feasibility

- **WS value accuracy when WS captures the bar:** 99.5%-100%. Excellent.
- **WS coverage:** 74% during active periods, 100%+ during quiet.
- **The hybrid (WS-tip + REST-verify) would inherit WS's 26% coverage
  gap during active hours.** REST already has those bars; the
  hybrid would still need REST to fill them. So the hybrid's
  latency advantage at +0s only applies to 74% of active bars; for
  the other 26%, the hybrid still has to wait for REST anyway.

- **Pure REST gets us 100% coverage at +2-3s latency.** Trade-off:
  ~3 sec slower on the bars where WS would've worked (74% of active),
  but no coverage gap.

### Recommendation reinforced

Pure B1 wins on COVERAGE + parity. Hybrid trades a 26% miss-rate for
2-3 sec speed gain. Not worth it for current strategies.

### Coverage gap is structural, not deploy-related (added late 2026-05-27)

Kevin asked: are the missing bars due to our many deploys today, or
real WS pipeline issues?

Investigated 18:00-19:30 UTC SPY 10Sec gap runs:

```
  18:06:00 → 18:22:10  gap=970s (96 bars missing)  ← 16 min!
  18:22:10 → 18:23:00  gap=50s (4 bars)
  19:10:30 → 19:11:30  gap=60s (5 bars)
  19:11:30 → 19:12:10  gap=40s (3 bars)
  19:13:00 → 19:15:00  gap=120s (11 bars)
  19:16:00 → 19:17:10  gap=70s (6 bars)
  19:17:10 → 19:18:10  gap=60s (5 bars)
  19:20:00 → 19:21:10  gap=70s (6 bars)
  19:21:10 → 19:22:00  gap=50s (4 bars)
```

**Cross-check: during the 16-min 18:06 gap, Ralph wrote bars for 22
OTHER symbol/TF combos** (TSLA 5s/10s/15s/30s/60s, TSLL, SPY 120s/180s,
etc). So Ralph was alive. The gap is specific to SPY 10Sec.

Conclusion: **the 26% bar-drop is a structural bug in Ralph's WS
pipeline for SPY sub-minute, not a deploy/restart artifact.** It will
persist even when we stop deploying. This is more evidence for pure
B1.

The bug pattern is interesting: SPY sub-minute drops bars for minutes
at a time while TSLA/other tickers continue normally. Possible
causes: Polygon's per-ticker WS subscription stuck after a reconnect;
race condition in BarBuilder for SPY's high tick rate; Polygon
A-channel throttling on SPY specifically. **File as a known bug; not
attempting to fix it since we're migrating away.**

## Open data point still to collect

- **RTH measurement.** Tomorrow during 13:30-20:00 UTC, repeat the
  probe. Higher trade volume should mean less Polygon aggregation
  delay and fewer outliers. Confirms (or refutes) the 1Min p95
  outlier.
- **Active-hour WS coverage during RTH.** Does Ralph still drop 26%
  of active SPY 10Sec bars during real market hours, or was the
  18:00-19:30 window an after-close anomaly? Critical to know
  whether to ship B1 with confidence.

---

2026-05-27 — measurement run during 21:58:57 - 22:08:41 UTC (post-RTH
extended hours). Data sample limited but trend is clear.
