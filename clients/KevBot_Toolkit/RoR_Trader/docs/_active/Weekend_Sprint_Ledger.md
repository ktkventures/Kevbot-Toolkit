# Weekend Sprint Ledger — Resampled Store Fleet Verification

**Sprint item 1 of `Plan_MRS2_P2_Rollout.md` Phase 2** (weekend sprint — GREEN/DRIFT ledger per target).
Generated 2026-07-11 ~05:30Z (Saturday, markets closed — no forming-bar noise). Store re-trued fresh as of ~05:20Z 2026-07-11.
READ-ONLY pass: `rbs.shadow_check(sym, tf, session, now − warmup, now, settled_only=True)` per unique fleet target, where warmup = `_tf_warmup_days(tf, 250)` (the engines' real `SECONDARY_WARMUP_BARS` depth) **+ 5-day probe buffer**. Store coverage from `min(ts)/max(ts)` per `(symbol, timeframe_seconds, session)` series.

**Bottom line: ZERO drift.** 18/18 unique targets byte-identical to the canonical resample on the store's covered range (0 cell diffs, 0 store-only bars, fleet-wide). 15/18 GREEN over the full buffered window; 3/18 (†) have a head gap that exists **only inside the +5d probe buffer** — the engines' real 250-bar warmup windows are fully covered (proof in §C.2).

Fleet scope: 70 strategies total → **41 with non-empty `config.confluence`** (sessions: 23 RTH, 18 Extended Hours) → **18 unique (symbol, tf, session) verify targets** across 55 store-TF gate records.

## A — Per-target verdicts (deduped fleet targets)

| Symbol | TF | Session | Verdict | n_store/n_canon | Probe window | Store depth (min → max ts, UTC) | Head gap vs probe | Dependent sids |
|---|---|---|---|---|---|---|---|---|
| DIA | 2Min | Extended Hours | GREEN | 1542/1542 | 7d | 2026-04-13 → 2026-07-10 | 0 | 306 |
| KO | 2Min | Extended Hours | GREEN | 1429/1429 | 7d | 2026-04-13 → 2026-07-10 | 0 | 307 |
| SPY | 2Min | Extended Hours | GREEN | 2365/2365 | 7d | 2026-04-13 → 2026-07-10 | 0 | 277, 279, 281, 283, 285, 287, 289, 291, 293, 295, 297, 299, 301, 303, 305 |
| SPY | 5Min | RTH | GREEN | 546/546 | 10d | 2026-04-13 → 2026-07-10 | 0 | 271 |
| SPY | 10Min | RTH | GREEN | 390/390 | 15d | 2026-04-13 → 2026-07-10 | 0 | 272 |
| SPY | 15Min | RTH | GREEN | 364/364 | 19d | 2026-04-13 → 2026-07-10 | 0 | 136 |
| TSLA | 2Min | RTH | GREEN | 975/975 | 7d | 2026-04-13 → 2026-07-10 | 0 | 174, 267, 309, 327 |
| TSLA | 3Min | RTH | GREEN | 650/650 | 8d | 2026-04-13 → 2026-07-10 | 0 | 309, 333, 340 |
| TSLA | 5Min | RTH | GREEN | 546/546 | 10d | 2026-04-13 → 2026-07-10 | 0 | 274, 310, 327, 328, 329, 340 |
| TSLA | 10Min | RTH | GREEN | 390/390 | 15d | 2026-04-13 → 2026-07-10 | 0 | 275, 311, 331 |
| TSLA | 15Min | RTH | GREEN | 364/364 | 19d | 2026-04-13 → 2026-07-10 | 0 | 333 |
| TSLA | 30Min | RTH | GREEN | 299/299 | 33d | 2026-04-13 → 2026-07-10 | 0 | 325, 330, 339 |
| TSLA | 1Hour | RTH | GREEN | 266/266 | 57d | 2026-04-13 → 2026-07-10 | 0 | 311, 314 |
| TSLA | 4Hour | RTH | GREEN † | 289/301 | 187d | 2026-01-09 → 2026-07-10 | 4.27d | 313, 314, 325, 330, 331 |
| TSLA | 1Day | Extended Hours | GREEN † | 269/273 | 368d | 2025-07-14 → 2026-07-10 | 5.77d | 338 |
| TSLA | 1Day | RTH | GREEN † | 250/254 | 368d | 2025-07-14 → 2026-07-10 | 5.77d | 310, 312, 313 |
| TSLL | 2Min | RTH | GREEN | 975/975 | 7d | 2026-04-13 → 2026-07-10 | 0 | 194 |
| TSLL | 10Min | RTH | GREEN | 390/390 | 15d | 2026-04-13 → 2026-07-10 | 0 | 194 |

† = GREEN on covered range; head gap is within the +5d probe buffer only (see §C.2). Verdict basis: `settled_only=True` byte-compare (OHLCV exact per bar); n_store/n_canon are settled bars in the probe window.

## B — Per-strategy aggregate verdicts

| sid | Name | Symbol | Primary TF | Session | Store gate TFs | Verdict | Notes |
|---|---|---|---|---|---|---|---|
| 136 | SPY LONG - Mass #11 [mirror 50] | SPY | 1Min | RTH | 15Min | ALL-GREEN | 1M sentinel skipped: 1M-MACD_LINE_V2-M>S- |
| 174 | TSLA LONG 1Min Mass #2 | TSLA | 1Min | RTH | 2Min | ALL-GREEN | — |
| 194 | TSLL LONG 1Min Mass #30 | TSLL | 1Min | RTH | 2Min, 10Min | ALL-GREEN | — |
| 267 | TSLA-CANARY-10s-LooseConf | TSLA | 10Sec | RTH | 2Min | ALL-GREEN | — |
| 271 | TEST-P2-multipack-10s-0601-SPY | SPY | 10Sec | RTH | 5Min | ALL-GREEN | — |
| 272 | TEST-P2-stoch-10s-0601-SPY | SPY | 10Sec | RTH | 10Min | ALL-GREEN | — |
| 274 | TEST-P2-multipack-10s-0601-TSLA | TSLA | 10Sec | RTH | 5Min | ALL-GREEN | — |
| 275 | TEST-P2-stoch-10s-0601-TSLA | TSLA | 10Sec | RTH | 10Min | ALL-GREEN | — |
| 277 | PACKTEST · Bollinger Bands · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 279 | PACKTEST · EMA Price Position v3 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 281 | PACKTEST · EMA Price Position v4 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 283 | PACKTEST · EMA Stack v2 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 285 | PACKTEST · MACD Histogram v2 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 287 | PACKTEST · MACD Line v2 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 289 | PACKTEST · RSI Zones 2 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 291 | PACKTEST · Relative Volume v2 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 293 | PACKTEST · Support Resistance Channels · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 295 | PACKTEST · Stochastic Oscillator · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 297 | PACKTEST · Strat Assistant · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 299 | PACKTEST · SuperTrend · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 301 | PACKTEST · Swing 1-2-3 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 303 | PACKTEST · UT Bot V4 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 305 | PACKTEST · VWAP v2 · gate | SPY | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 306 | TEST-GAPHEAL-DIA-10s-2m-0611 | DIA | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 307 | TEST-GAPHEAL-KO-10s-2m-0611 | KO | 10Sec | Extended Hours | 2Min | ALL-GREEN | — |
| 309 | TSLA LONG 15Sec Mass #2 | TSLA | 15Sec | RTH | 2Min, 3Min | ALL-GREEN | — |
| 310 | TSLA LONG 15Sec Mass #3 | TSLA | 15Sec | RTH | 5Min, 1Day | GREEN (head-buffer note) | head-buffer: 1Day |
| 311 | TSLA LONG 15Sec Mass #18 | TSLA | 15Sec | RTH | 10Min, 1Hour | ALL-GREEN | — |
| 312 | TSLA LONG 15Sec Mass #21 | TSLA | 15Sec | RTH | 1Day | GREEN (head-buffer note) | head-buffer: 1Day |
| 313 | TSLA LONG 15Sec Mass #9 | TSLA | 15Sec | RTH | 4Hour, 1Day | GREEN (head-buffer note) | head-buffer: 4Hour, 1Day |
| 314 | TSLA LONG 15Sec Mass #22 | TSLA | 15Sec | RTH | 1Hour, 4Hour | GREEN (head-buffer note) | head-buffer: 4Hour |
| 325 | TSLA LONG 30Sec Mass #4 | TSLA | 30Sec | RTH | 30Min, 4Hour | GREEN (head-buffer note) | head-buffer: 4Hour |
| 327 | TSLA LONG 30Sec Mass #33 | TSLA | 30Sec | RTH | 2Min, 5Min | ALL-GREEN | — |
| 328 | TSLA LONG 30Sec Mass #13 | TSLA | 30Sec | RTH | 5Min | ALL-GREEN | — |
| 329 | TSLA LONG 30Sec Mass #14 | TSLA | 30Sec | RTH | 5Min | ALL-GREEN | 1M sentinel skipped: 1M-SWING_123-BULL_C2 |
| 330 | TSLA LONG 30Sec Mass #9 | TSLA | 30Sec | RTH | 30Min, 4Hour | GREEN (head-buffer note) | head-buffer: 4Hour |
| 331 | TSLA LONG 30Sec Mass #19 | TSLA | 30Sec | RTH | 10Min, 4Hour | GREEN (head-buffer note) | head-buffer: 4Hour |
| 333 | TSLA LONG 30Sec Mass #2 | TSLA | 30Sec | RTH | 3Min, 15Min | ALL-GREEN | — |
| 338 | TEST-COARSEGATE-10s-1d-TSLA-0626 | TSLA | 10Sec | Extended Hours | 1Day | GREEN (head-buffer note) | head-buffer: 1Day |
| 339 | TSLA LONG 30Sec Mass #1 | TSLA | 30Sec | RTH | 30Min | ALL-GREEN + non-store TF | non-store: 10s-UT_BOT_V4-BULL_TREND |
| 340 | TSLA LONG 1Min Mass #31 | TSLA | 1Min | RTH | 3Min, 5Min | ALL-GREEN | — |

Aggregate counts: **32 ALL-GREEN** · **8 GREEN (head-buffer note)** · **1 ALL-GREEN + non-store TF** · 0 has-DRIFT · 0 has-engine-coverage-gap.

## C — Exceptions & notes

### C.1 Non-store gate TFs (per sid)
- **sid 339** (`TSLA LONG 30Sec Mass #1`): gate `10s-UT_BOT_V4-BULL_TREND` — sub-minute, resample-from-1Sec = deferred Phase 2b (seam reserved in `base_layer_for_tf`). Its 30Min store gate is GREEN.
- **sid 136** (`SPY LONG - Mass #11 [mirror 50]`): record `1M-MACD_LINE_V2-M>S-` — uppercase `1M` = primary-TF sentinel, not a secondary; skipped by design. Its 15Min store gate is GREEN.
- **sid 329** (`TSLA LONG 30Sec Mass #14`): record `1M-SWING_123-BULL_C2` — same primary-TF sentinel skip. Its 5Min store gate is GREEN.
- No lowercase `1m` (1Min base-layer) gates and no GEN- records inside `confluence` lists fleet-wide; no malformed/unknown TF prefixes.

### C.2 Head-gap detail (the 3 † targets) — buffer artifact, NOT an engine coverage gap
Store head depth varies by TF: fine TFs (2Min…1Hour) share a ~90-day seed floor (series min ts 2026-04-13, far deeper than their 250-bar warmup), while the deep TFs (4Hour, 1Day) were seeded to the **exact** 250-bar engine warmup depth (no buffer), anchored at the seed date. The probe window here is engine warmup **+5d**, so for the deep TFs the probe head predates the seed head. Confirmation re-check (read-only, `start = store min(ts)`): **all 3 targets match=True with 0 cell diffs** — every mismatched bar was head-side, older than store depth.

| Target | Store min(ts) | Engine warmup | Engine window start (2026-07-11 run) | Engine covered? |
|---|---|---|---|---|
| TSLA 4Hour RTH | 2026-01-09 12:00Z | 182d | 2026-01-10 | YES (store starts 0.7d earlier) |
| TSLA 1Day RTH | 2025-07-14 00:00Z | 363d | 2025-07-13 (a **Sunday**) | YES — first tradable bar of the engine window is Mon 2025-07-14 = store min, exact |
| TSLA 1Day Extended Hours | 2025-07-14 00:00Z | 363d | 2025-07-13 (Sunday) | YES — same as above |

Margin note: the 1Day series has **zero spare head margin** today (store min == the engine window's first bar). This is self-healing — as `now` advances the needed warmup start moves forward while the store head is fixed, so margin only grows. No backfill required for the current fleet; if a deeper-warmup consumer appears (e.g. warmup right-size multiplier changes), reseed the head first.

### C.3 DRIFT cells
None. 0 cell diffs across all 18 targets (including the 3 † targets on their covered range).

### C.4 Method footnotes
- Warmup source: `strategy_data.SECONDARY_WARMUP_BARS = 250`, `_tf_warmup_days(tf, 250)` (trading→calendar ×365/252, ceil). Probe adds +5 calendar days.
- Comparator: `resampled_bar_store.shadow_check` → `compare_store_vs_canonical` (OHLCV exact per bar; binary-format float8 read — the same acceptance criterion as M1a/M1b).
- `settled_only=True` trims both sides to the settled cutoff; store max(ts) is Fri 2026-07-10 across targets (fresh through the last session).
- Store TF catalog verified = 10 coarse TFs (2Min…1Day), all confluence-enabled; sub-minute and 1Min excluded by design.
- Sweep was sequential, ~2-4s per target; no transient errors, no retries needed.
