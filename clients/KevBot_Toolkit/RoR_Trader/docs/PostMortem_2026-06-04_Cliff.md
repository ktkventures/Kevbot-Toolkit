# Post-Mortem — the 2026-06-04 combined% "cliff"

**Question:** around 6/4 the fleet showed good live↔backtest pairing (by-hour avg ~74.7%, max
100%); later that day it "fell off a cliff." What changed, and was it the bar work?

**Headline answer: it was NOT the bars.** The live bars match REST in value 98–100% every day, and
coverage only declined modestly. The cliff is in the **decision/alert-pairing layer**, is
**fleet-wide** (worker-level, not per-strategy), and is partly an artifact of **backtest-reference
drift** (we compare historical alerts to *today's* much-changed backtest). Leading suspects: the
**Phase 2 streaming-toggle / async-UAD worker churn** and the **`#42` session-filter** change — not
the bar pipeline.

## Evidence

### 1. The cliff (by-deploy, ±5s pairing, current BT trades vs historical alerts)
| when (UTC) | sha | combined% | note |
|---|---|---:|---|
| 06-04 09:29 | e52aba2 | 66.2 | watchdog guard |
| 06-04 12:33 | f061076 | 68.1 | config backfill migration |
| 06-04 13:09 | 4be824a | **70.2** | Phase 2 backend: streaming toggle + async UAD |
| 06-04 13:38 | f7eec9b | 47.7 | Phase 2 frontend wired to async update_jobs |
| 06-04 14:08 | 2fd0c10 | 39.4 | **Streaming mode toggle (Auto/Manual) DB switch** |
| 06-04 15:27 | 3322298 | **22.8** | ALGO-APPEND `incremental=True` (phantom-heavy) |
| 06-04 17:00 | 7dba9ea | **15.4** | (low point) |
| 06-05 08:58–12:10 | #57… | 59→**72.6** | **REST-verification work — RECOVERED** |
| 06-05 12:23 | 3429dce | 56.6 | **`#42` session filter on injected primary_df** |
| 06-05 15:25–16:45 | window-backfill | 35→33 | manual window-backfill (marked "WIP / do not trust") |

The drop is fleet-wide (~40 strategies), recovers with REST-verification work, and re-drops with
session-filter / window-backfill work — i.e. it tracks **code changes**, not the live feed.

### 2. Bars are fine (SPY 10s, RTH 13:30–20:00 UTC)
| date | REST | WS-cache | coverage% | close value-match% (≤1¢) |
|---|---:|---:|---:|---:|
| 06-03 | 2340 | 2289 | 97.8 | 98.0 |
| 06-04 | 2340 | 2121 | 90.6 | 99.6 |
| 06-05 | 2340 | 2012 | 86.0 | 99.2 |
| 06-08 | 2340 | 2071 | 88.5 | 100.0 |
| 06-10 | 2340 | 2018 | 86.2 | 99.9 |

- **Value match 98–100% throughout** → REST verification is healthy; bars don't drift in value.
- **Coverage 86–98%, gradual** → a steady ~10–14% drag (the WS-misses-real-10s-bars gap from the
  evening session), but NOT a cliff and not intraday on 6/4.

### Why the bars can't be the cliff
The combined% went 70→15 *within the afternoon of 6/4*, while bar coverage on 6/4 was a flat ~90%
all day and values matched 99.6%. A bar problem would show as a coverage/value change; it didn't.

## Leading hypothesis (calibrated — multi-factor)
1. **Phase 2 streaming-toggle / async-UAD churn (`2fd0c10`/`4be824a`/`f7eec9b` + watchdog
   auto-restart).** ~7 deploys that afternoon = ~7 worker restarts; the new Auto/Manual streaming
   switch changed how/when the engine streams. The phantom surge is the signature of the live
   engine's *decisions* diverging while the worker was churned.
2. **Backtest-reference drift (confound, possibly large).** by-deploy compares historical alerts to
   *today's* backtest; the backtest code changed a lot since 6/4 (session filter, exec types, stop
   methods). 6/4's alerts may have matched 6/4's backtest fine and only look bad now. Part of the
   "cliff" is the reference moving, not live regressing.
3. **`#42` session-filter (`3429dce`).** "apply session filter to injected primary_df" — a
   backtest-side change that dropped 6/5 PM and could create a live↔backtest **session mismatch**
   (different session windows → pairing divergence). Strong suspect for a *persistent* regression.

## Next (tomorrow)
- Dig into **`#42` session-filter** (`services.py:264` injected primary_df) — confirm live and
  backtest use the SAME session window; a mismatch would produce exactly this pairing divergence.
- Audit the **streaming toggle / worker lifecycle** — confirm the worker isn't in a degraded
  streaming mode now; check restart frequency.
- To remove the reference-drift confound, compare a fixed recent window's alerts against the
  current backtest only (apples-to-apples), rather than reaching back to 6/4.
