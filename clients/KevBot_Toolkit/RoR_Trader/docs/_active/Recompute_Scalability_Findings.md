# Recompute Scalability — Bottleneck Findings & Long-Term Fix Plan — 2026-06-24

**Goal:** make full Update-All recompute fast + scalable enough to validate thousands of
strategies against backtest logic. NOT band-aids — a solid architecture. Supersedes the
narrower `Design_Recompute_Bar_Cache.md` (the bar-cache turned out NOT to be the lever).

## The problem
Full recompute is slow: **325 (TSLA 30Sec, 4h+30m SWING_123 gates) = 24 min; 309/311 = 70/76
min.** Full 66-strategy catalog ≈ days. Blocks scale + slows every debug/validate cycle.

## MEASURED breakdown (2026-06-24, sid 325, local, BAR_CACHE off)
| Component | Time | Notes |
|---|---|---|
| 1-min bar load, **95-day** window (isolated) | 3 s | ⚠️ NON-representative — real recompute loads ~1yr (below) |
| 1-min bar load, **in-pipeline** (~1yr, 555k bars, 111 calls) | **353 s** | the real coarse-warmup load (cProfile) |
| 1-sec fetch, one trading day (22.8k bars) | 2.3 s | day-level cached in-proc |
| `get_strategy_trades` (no Hi-Fi), full config | **953 s (15.9 min)** | **the bulk** |
| — of which: CPU (user time) | 385 s (6.4 min) | 41% CPU utilization |
| — of which: non-CPU WAIT | ~570 s (~9.5 min) | **NOT the 3s load — hidden wait (cProfile pending)** |
| `get_strategy_trades` stripped (30d, NO secondaries) | 41 s | vs 953s full → **secondaries (4h/30m) ≈ ~900s of the cost** |
| Hi-Fi Pass 2 (the remaining recompute time) | ~8 min | 1-sec fetches day-cached (~2.3min) + per-trade refinement |

### Key conclusions
1. **Bar-cache is STILL not the lever — but for a sharper reason than "3s".** The in-pipeline
   load is 353s (not 3s — my 95-day isolated test was the wrong window). BUT the right fix is to
   **AVOID the ~1yr load entirely** (use the cached/snapshotted 4h series), not to *cache* a
   year of 1-min. The existing `bar_cache.py` was also 0.5× (slower) for 30-day loads. The
   secondary-TF snapshot avoids the load AND shrinks the engine window — strictly better.
2. **The dominant cost is the SECONDARY-TF processing** for coarse-gate strategies (4h/30m
   resample + SWING_123 interpreter + cross-TF shadow over the full window): 41s without
   secondaries vs 953s with. The secondary-TF SNAPSHOT (already built + byte-identical-proven)
   currently only accelerates the APPEND path, NOT `get_strategy_trades` (full recompute).
3. **The hidden wait was the bar load after all — but a ~1-YEAR load, not 95 days.** cProfile
   (full config) breakdown of get_strategy_trades(325):
   - `load_market_data` = **353s (29%)** — 111 Polygon paginated calls @3.1s fetching ~555k
     1-min bars (~1 year) to warm the 4h secondary. (My earlier "3s" was a mismeasurement —
     a 95-day window, not the real warmup.)
   - `run_unified_backtest` engine = **378s (31%)** — `process_bar` ×**153,301 bars** (the
     year-long warmup window; the visible backtest is only ~47k bars).
   - `run_indicators_for_group` (user-pack SWING_123) = **299s (24%)** CPU.
   - `run_all_interpreters` = 97s (8%).
   **ROOT CAUSE: coarse-gate recompute LOADS + PROCESSES ~1 year of 1-min just to warm the 4h
   secondary** — inflating both I/O (353s) and CPU (engine+indicators over 153k bars).
4. Hi-Fi (~8 min) is a secondary cost, not the headline (earlier guess was wrong).
5. The recompute has real CPU (6.4 min/strategy) and the catalog is **embarrassingly parallel**
   (independent strategies) → parallelism is a guaranteed throughput multiplier.

## LONG-TERM SCALABLE FIX — milestone sequence (the plan, not band-aids)
cProfile shows ~29% I/O (the ~1yr load) + ~70% CPU (engine + user-pack indicators +
interpreters), but BOTH are inflated by the same root cause: **the coarse-gate warmup
loads + processes ~1 year of 1-min** when the 4h secondary only needs ~250 bars (~125 days).
Three complementary levers — the data store speeds I/O + kills cross-strategy redundancy;
warmup-sizing + parallelism cut the CPU. Sequenced cheapest-first:

### M-RS1 — Right-size the warmup (cheapest, biggest bang, do FIRST)
The full path (`get_strategy_trades` → `strategy_data.compute_warmup_days`) uses a flat
`visible_days × WARMUP_MULTIPLIER(=2)` with **no secondary-TF/indicator awareness** → total
load = 3× visible (e.g. 540-day visible → ~1,620 days ≈ 555k 1-min bars), and the engine
chews all of it (~2/3 is warmup that's trimmed away).
- **Fix:** size warmup to the indicator/binding-TF requirement — `max(visible×MULT,
  longest_indicator_lookback)` — tying into the standard that ALREADY exists:
  `ralph_engine._secondary_warmup_days` (`_SHADOW_WARMUP_TARGET_BARS=250`) and the append
  path's binding-bpd logic (`get_strategy_trades_for_window`). Makes the full path consistent
  instead of the odd one out (there's even a TODO at `strategy_data.py:134`).
- **Effect:** cuts the ~1yr load AND ~2/3 of engine+indicator bars for long-visible strategies.
- **Gate:** byte-identical (trades + indicator cols) vs current, on an ungated 15Sec, a
  sub-minute+secondary (309), and a 1d/4h coarse-gate (313/325). Kill-switch.

### M-RS2 — Shared symbol-level 1-second bar store ("super dough" — the durable foundation)
Cache **raw 1-second OHLCV per symbol** in Supabase, fetched ONCE and reused by every
strategy/backtest/append/live on that ticker (most of the catalog is one ticker → fetch
once, the whole cohort reuses). 1-second is the right base **because sub-minute strategies
(5/10/15/30Sec) can't be built from 1-min** — you build everything up from 1s.
- **Store 1s as source of truth; materialize a 1-min layer on top** (resample higher TFs
  from the 1-min cache — don't re-resample 5.9M 1s rows every recompute).
- **Fidelity guard (the whole risk):** settled bars (older than a revision horizon)
  immutable → served from store; the recent window is ALWAYS re-fetched from Polygon and
  overwrites the tail (Polygon revises recent bars). Live still hits Polygon for the tip.
- **Storage:** ~0.5–0.7 GB/ticker/yr at 1s (98k rows/yr at 1-min ≈ 10 MB). 5 tickers×2yr ≈
  ~6 GB; 50×2yr ≈ ~60 GB ≈ ~$6/mo over Supabase Pro's 8 GB. $ is a non-issue; the real costs
  are Postgres perf at tens of M rows (→ partition by ticker+month, or parquet-in-Storage for
  bulk history + Postgres for the hot window), the one-time Polygon backfill, and the guard.
- **Control surface:** an agent page listing all Polygon-supported tickers; checkbox which to
  backfill + the capture range (1yr/2yr). Flip on → start tracking 1s for that ticker.
- **NOTE:** this speeds I/O + kills redundant re-fetching across the thousands-per-ticker
  cohort; it does NOT by itself speed the CPU buckets (that's M-RS1 + M-RS3).
- **Gate:** byte-identical (store-fed vs Polygon-fed) on the same 3 strategy shapes. Kill-switch.

### M-RS3 — Parallel recompute + dedicated Railway update service (throughput multiplier)
The catalog is embarrassingly parallel (independent strategies). Multiprocessing / worker
pool over strategies → N cores ≈ N× throughput, operating on the now-cheap (M-RS1 right-sized,
M-RS2 cached) per-strategy unit. Pair with a **dedicated Railway update service** — vertical
(cores/RAM) + horizontal (replicas) — so catalog-wide validation is routine, not an all-nighter.
- (Maybe) **decouple Hi-Fi** (~8 min) from the bulk recompute (recompute fast; Hi-Fi
  incrementally) if it proves to be on the critical path after M-RS1.

**Order:** M-RS1 (this week — makes a single recompute fast, fraction of the effort) →
M-RS2 (the scalable foundation) → M-RS3 (scale out). Each validated byte-identical +
kill-switched before shipping.

## What we will NOT do
- Cache a year of 1-min as a band-aid. The base store is 1-second (sub-minute strategies need
  it) with a derived 1-min layer; and warmup right-sizing (M-RS1) avoids over-loading at all.
- Trust the store for recent bars (Polygon revises them) — always re-fetch the tip.
- Skip Hi-Fi / blindly trim windows — fix the structural warmup cost (M-RS1) first.
