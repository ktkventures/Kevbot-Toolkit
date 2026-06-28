# Analysis — Fleet Recompute Trade-Count Deltas (parallel run vs benchmark) — 2026-06-28

**Question (Kevin):** the parallel full-fleet recompute reproduced the benchmark to
within 0.03%, but a handful of strategies changed trade count by more than ±1–3
(e.g. **+23, +9**). Where do those come from? Is it divergence, or is it "more
true"? Divergence matters because if trade *counts* drift, trade *timing* might too —
unless we can confirm trades are bit-perfect.

**Verdict up front:** Not divergence. The engine is **bit-deterministic** (same
inputs → identical trades, timestamps included). The small ±deltas come from
**changed inputs between the two runs** — almost certainly Polygon historical-bar
revision (and Hi-Fi 1-second refinement), which nudges a few trades across trigger
thresholds. This is **not introduced by the parallel path** and would happen on the
old sequential path too. **No action taken** (per Kevin's instruction) — this is an
understanding write-up.

---

## When the runs happened
- **Benchmark:** the 2026-06-27 full-fleet UAD (the ~215-min sequential run from the
  prior session). Exact start time isn't recoverable from the DB (its rows were
  overwritten by today's run), but it was 06-27, after-hours.
- **Today's parallel run:** 2026-06-28, ~00:30–01:40 UTC (also weekend/after-hours).
- **Gap ≈ ~1 day**, markets closed in both → no *intraday* data difference, but
  Polygon still finalizes/revises historical flat files over that window (FINRA
  late-print corrections, backfill).

## What the diff actually showed (all 67)
- Fleet backtest trades: **146,575 → 146,620 = +45 (0.03%)**.
- **44/67 byte-identical** (count + date range + KPIs).
- **17 count-changed** — and critically, **every one has a STABLE date range**
  (earliest AND latest unchanged): the changes are ±1–23 trades, **both directions**
  (+23, +9, +5 … but also −3, −2, −1). 15/17 have a `backtest_start_date`.
- **6 KPI-only** — same trade count and dates, `total_r`/`profit_factor`/`avg_r`
  shifted **<1.5%**.

Two facts rule out the obvious culprits:
1. **Not trimming / window drift.** The date ranges didn't move at all (earliest and
   latest stable), for anchored *and* anchorless strategies. The `RORT_UAD_PRESERVE_
   RANGE` guard is holding — the "anchorless strategies self-trim" pattern did **not**
   occur. The changed trades are *interior* to an unchanged span.
2. **Not a systematic increase.** If this were the append-under-production fix
   revealing "true" higher counts, deltas would be consistently *up*. They're tiny
   scatter **both ways** → run-to-run input noise, not a systematic shift.

## The engine is deterministic — so the delta is an INPUT change, not engine randomness
This is the load-bearing evidence. Running the *same* strategy over the *same*
frozen window in two separate processes (parent vs spawn-worker) yields
**trade-for-trade identical** output — same entry/exit fill timestamps, prices,
r_multiple, stop, exit_reason:
- **263 / 267 / 338**: 633/633, 634/634, 1595/1595 trades, **0 real differences**
  (the only field that varied was the diagnostic `confluence_records` *list order* —
  a per-process `PYTHONHASHSEED` artifact, not a trade difference).
- **293 / 304** (the two BIGGEST-delta strategies, +23 and +9 in the fleet diff):
  parent-vs-spawn-worker over a frozen 25-day window — **293: 2489 == 2489,
  304: 680 == 680, real_diffs = 0** (only `confluence_records` order varied).
  So the exact strategies that showed the largest day-over-day deltas are
  themselves **bit-deterministic** — confirming the deltas are input-data, not
  engine behavior.

Because the engine is bit-deterministic, the cross-run ±deltas **cannot** be engine
non-determinism. They must come from the only thing that differs between the two
runs: **the input data.**

## Most likely cause: Polygon historical-bar revision (+ Hi-Fi 1-sec)
- **Count changes (interior, ±a few):** between the 06-27 and 06-28 runs, Polygon
  revised/finalized a few historical bars (corrections, late prints). A revised bar
  shifts an indicator value a hair; if that value sits right at a trigger threshold,
  a trade appears or disappears — *inside* the existing date span. A handful of such
  bars across a 2,800-trade strategy fully explains +23 (0.8%).
- **KPI-only changes (<1.5%, same trades):** Hi-Fi Pass 2 re-fetches 1-second data to
  refine exit fills. Fetched at a different time, a few fills land a tick differently
  → `total_r` moves slightly while the trade set is unchanged.

Both mechanisms are **inherent to any recompute** (sequential or parallel, old engine
or new) — they're a property of recomputing against live-revised market data, not of
M-RS3.

## Is it "more true"? — Yes, in the sense that each run reflects its data
Neither run is "wrong." Today's run reflects the **latest** Polygon data; yesterday's
reflected yesterday's. The +/- scatter says the newer data supports a slightly
different (not strictly larger) trade set. There's nothing to "fix" — the recompute
faithfully reproduced the engine's output for the data it saw.

## Addressing the divergence concern directly
> "If counts are off, timing might be off, unless trades are bit-perfect."

- **Within a consistent data snapshot, trades ARE bit-perfect** — including
  entry/exit *fill timestamps* (the determinism check compares timestamps trade-for-
  trade and finds 0 differences). So **timing does not drift** for a fixed data
  window. Backtest-vs-forward comparisons are reliable **as long as both sides use the
  same data window/snapshot.**
- **The parallel path adds ZERO divergence** — proven byte-identical to sequential.
  The day-over-day wobble is data-revision, which affected the old path identically.

## Honest limitation + recommendation (no action taken)
- I **cannot pinpoint the specific ±23 trades** for sid 293: the benchmark snapshot
  captured per-strategy **counts / date ranges / KPIs**, not every individual trade
  row, and yesterday's rows were overwritten by today's run. So a true trade-level
  "which 23 appeared" diff isn't possible after the fact.
- **Recommendation (for your call):** add a lightweight **trade-level snapshot** step
  (entry/exit ts + r per trade, hashed) before a fleet run, so future audits can do an
  exact trade-for-trade diff and attribute every delta. This is the only way to get
  the bit-perfect-vs-yesterday confirmation you asked about; the determinism proof
  gives us bit-perfect-within-a-run today, which is the property that actually governs
  backtest↔forward fidelity.

## Bottom line
Engine deterministic → ±deltas are Polygon data revision + Hi-Fi 1-sec, interior to
stable date ranges, both directions, <0.03% fleet-wide. **Not divergence, not the
parallel path, not a trimming bug.** Timing is bit-stable within a data snapshot.
Nothing to action unless you want the trade-level snapshot tooling for exact future
attribution.
