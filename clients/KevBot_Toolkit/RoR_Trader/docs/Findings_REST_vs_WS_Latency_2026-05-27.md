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

## Open data point still to collect

- **RTH measurement.** Tomorrow during 13:30-20:00 UTC, repeat the
  probe. Higher trade volume should mean less Polygon aggregation
  delay and fewer outliers. Confirms (or refutes) the 1Min p95
  outlier.

---

2026-05-27 — measurement run during 21:58:57 - 22:08:41 UTC (post-RTH
extended hours). Data sample limited but trend is clear.
