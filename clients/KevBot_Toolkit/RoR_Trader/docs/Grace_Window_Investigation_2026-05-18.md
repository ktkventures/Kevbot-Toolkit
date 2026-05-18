# 10Sec Builder — Grace-Window Investigation (2026-05-18)

## Problem

The sub-minute (10Sec) `live_bars` cache was producing **flat, zero-volume
bars** — SPY 10Sec was ~46% empty bars, TSLA ~20%. The 12 active 10Sec
strategies (11 SPY, 1 TSLA) were computing indicators on a half-fake series.

Two root causes were found in `ralph_engine.py`'s `BarBuilder`:

- **Bug A — late-bar misattribution (FIXED, commit `ba560f1`).**
  `accept_second_bar` had no branch for a per-second bar older than the
  current partial; late bars were aggregated into the *current* bucket,
  dumping a full minute of volume into one 10s bar. Fixed: late bars are
  dropped.
- **Bug B — wall-clock force-close (THIS investigation).**
  `force_close_stale_bar` closes a 10s bucket on wall-clock at `bar_end`.
  If the per-second data hasn't arrived yet, the bucket emits a flat
  carry-forward bar.

## Why Bug B happens — the A-channel ingestion lag

Instrumentation (`RalphEngine._record_a_lag`, commit `f501126`) measured
the gap between a Polygon `A` (per-second) bar's start and when our worker
receives it:

```
A-channel lag: avg ~3.05s, min 3.04s, max 3.08s (one 6.76s outlier window)
```

It is a **structural, near-constant ~3-second delay** — Polygon's
per-second aggregate pipeline (≈1s bar duration + ~2s aggregation/emission
+ network). The worker itself is healthy; this is not worker lag.

**Consequence:** a 10s bucket `[B, B+10)`'s last second (`B+9`) arrives at
≈ `B+12`. The builder force-closes the bucket at `B+10` — ~2 seconds
*before* its data can physically arrive. Hence the flat empty bars.

**Caveat:** that measurement was taken **after-hours** (thin, 3–11 events
per 30s). Re-confirm during RTH at full `A`-event volume — see test plan.

The 10Sec data is *structurally* ~3s late. A grace window does not *add*
latency; it accepts latency that already exists and produces a correct bar
instead of a fake flat one.

## The grace-window shadow (commit — this change)

`src/grace_shadow.py` builds 10Sec bars at **three grace windows — 2s /
3s / 4s past `bar_end`** — in parallel from the same `A` per-second stream,
for SPY and TSLA. Shadow-only: writes to `live_bars_trades` with
`source = grace_2s / grace_3s / grace_4s` (PK includes `source`, so the
variants coexist). Production alerts are unaffected.

- Module: `src/grace_shadow.py` — `GraceShadowManager` + per-(symbol,grace)
  `GraceBuilder`. Reuses `trade_bar_builder`'s writer pool.
- Wiring: `ralph_engine.py` — manager init, fed from the `A`/`XAS` router,
  flushed on the engine heartbeat.
- Endpoint: `/api/admin/parity/trade-bars` now accepts `grace_2s/3s/4s`.
- UI: Bars Comparison has three new sources — "Grace 2s/3s/4s — 10Sec
  shadow".
- Env: `GRACE_SHADOW_ENABLED` (default ON), `GRACE_SHADOW_SYMBOLS`
  (SPY,TSLA), `GRACE_SHADOW_VARIANTS` (2,3,4).

## Test plan — tomorrow (2026-05-19) RTH

1. **Re-measure the A-channel lag** for ~15 min after the open — confirm
   the ~3s holds under full RTH `A`-event volume (today's sample was thin
   after-hours). Worker logs: `A-channel lag (30s window): …`.
2. **Compare the grace variants** in Admin → Parity → Bars Comparison:
   strategy on SPY (or TSLA), **Timeframe = 10Sec**, left pane =
   `Grace 2s/3s/4s` in turn, right pane = `REST 1Sec aggs`. Eyeball
   close-match % and candle shape for each.

### Predicted outcome (from the ~3s lag)
- **Grace 2s** — bucket closes at `B+12`; last second arrives ≈`B+12.05`
  → just misses it. Expect slightly worse close-match.
- **Grace 3s** — closes at `B+13`; catches the structural lag. Expect the
  best balance.
- **Grace 4s** — closes at `B+14`; same as 3s plus margin for outliers,
  at +1s more latency.

### Decision
Pick the smallest grace that reaches the ~99% close-match ceiling (the
Claude-app analysis put correctly-built 10Sec bars at ~99% exact vs
settled). Likely **3s**. Then promote that grace into the real
sub-minute builder (`force_close_stale_bar`) — that is the Bug B fix.

Bucket the misses by time-of-day when comparing: clustered at the
open/close auctions → an auction bug; uniform → physics ceiling.

## Reference

- Lag instrumentation: `ralph_engine.py` `_record_a_lag`, commit `f501126`.
- Bug A fix + derived-TF backfill: commit `ba560f1`.
- Backup branch: `backup-pre-grace-shadow-2026-05-18`.
- Related: `Claude_App_REST_vs_Cache_Analysis.md` (settlement-window
  analysis — note its "1s" figure is settlement-only; our total pipeline
  lag is ~3s).
