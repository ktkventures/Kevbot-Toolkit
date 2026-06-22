# Design — Long-window (incl. sub-minute) backtests without OOM

**Status:** design / approved direction (2026-06-21). Not yet built.
**Trigger:** pressure test (search `1801762576`, NVDA+TSLA × 30Sec+1Min × 365d, 4
cross-pack entries, 61 confluences, HiFi+OOS) **OOM-crashed the API** in Stage-2 prep
of the *first* group (NVDA/30Sec), 0 results, orphaned, no checkpoint. 365-day
sub-minute prep does not fit in the API pod's memory, and the data LOAD alone took
~42 min. Goal: make long windows (and eventually 20-hour multi-trigger sweeps) viable
**without breaking backtest↔live↔Polygon fidelity.**

## 1. Vocabulary — the three stages (shared SSOT)

| Stage | Plain name | What it produces | Code | Cost |
|---|---|---|---|---|
| **1 — Candles** | the bars | Load + resample OHLCV for (symbol, TF, window) | `load_market_data` → `resample_to_timeframe` | The **load** (year of sub-minute data = the bottleneck) |
| **2 — Dough** | where triggers fire | On the candles: indicators + interpreter states + **detect every trigger's fire-points** (per-bar booleans), frozen per bar | `prepare_data_with_indicators` → `precompute_bar_cache` (`CachedBarState[]`) | The **prep** — O(bars × indicators); this is what **OOM'd** |
| **3 — Gates → trades** | toppings | Replay the dough applying a specific confluence/gate set + entry/exit/stop → trades for that combo | `run_trades_from_cache` | Cheap replay (10–50×) |

**"The dough" = Stage 1 + Stage 2** — candles **plus** trigger fire-points. It does
**NOT** include the gates (Stage 3 is applied fresh + cheap per combo). So one dough,
keyed on **(symbol, timeframe, window)**, serves *every* confluence search on that
symbol/TF/window. (Pizza: bake the dough once = expensive; toppings = gates = cheap.)

## 2. FIDELITY — the non-negotiable constraint (Kevin's requirement)

Any candle we serve from the local store, and any dough we persist + reuse, **MUST
equal what a fresh Polygon REST fetch would return.** Otherwise the store/dough silently
diverges from live and from a normal backtest — the exact class of bug we fight hardest.

Rules:
1. **Finalization window.** Polygon aggregates can still move shortly after a bar closes
   (late-reported trades, corrections); they're stable after a settling lag (~15 min per
   Kevin; treat as a tunable `FINALIZATION_LAG`, default conservative). **Only finalized
   bars (older than `FINALIZATION_LAG`) may be served from the store or baked into a
   persisted dough.** The trailing edge (last ~15 min) is always fetched fresh from REST,
   never cached — same boundary the live append-edge logic already honors
   (see `Append_Edge_Fossilization.md`).
2. **Store == REST validation.** A reconciliation check: for a sampled window, assert the
   store's bars are byte-identical to a fresh Polygon REST fetch (OHLCV + timestamps).
   Run it as part of the Fidelity Gate before trusting the store. If Polygon issues a
   late correction to an already-finalized bar (rare; can be EOD/T+1 for some symbols),
   the reconciliation must detect the drift and **invalidate** the affected store rows +
   any dough baked from them.
3. **Dough is only as trustworthy as its bars.** A persisted dough carries the
   finalization boundary + a content hash of its source bars. On reuse, if the underlying
   bars changed (correction) the dough is invalidated and re-baked. Dough freshness/keying
   already exists in `bar_cache_store.py` (Tier-2: key on symbol/tf/session/data_days,
   36h freshness, content via the cache) — extend it with the finalization boundary +
   source-bar hash rather than inventing a new store.
4. **No fidelity regression** vs today: the Fidelity Gate goldens (308/309/313) must stay
   byte-identical after any of this lands.

## 3. The four features (priority: 3 & 4 = the crash fix; 1 & 2 = speed)

**Feature 3 — Chunk / stream Stages 1+2 (REQUIRED — directly stops the OOM).**
Process the window in chunks (e.g. monthly) so a full year of sub-minute bars never sits
in RAM at once: load chunk → run indicators/triggers → fold into the dough → free the
chunk. Indicators carry state across chunk boundaries (warmup continuity), so chunking
must preserve the exact same indicator trajectory as a single-pass run (validate
byte-identical). Lets the window grow without a memory ceiling.

**Feature 4 — Dedicated backtest worker, off the API pod (REQUIRED — survives crashes).**
Today the run is a daemon thread inside the API process → competes with API memory, no
heartbeat during the load phase, and an API restart kills it (we just saw this). Move the
pipeline to a dedicated worker (extend `data-worker`, or a new `backtest-worker`) with:
more memory; a real job queue; **heartbeat-during-load** (board task 2.31 — today the DB
isn't updated during Stage 1, so a live run *looks* queued); and **resume** that also
recovers `queued`/mid-load jobs (board task 2.32). This turns "OOM → lose everything"
into "survives + resumes."

**Feature 1 — Serve candles from the 1-sec store, not a year-long Polygon fetch (SPEED).**
The 42-min load was Polygon REST paginating a year of sub-minute aggregates. The
data-worker already stores 1-sec bars; read Stage-1 candles from it (resampling up to the
target TF), backfilling from REST only what's missing — subject to §2 fidelity. Turns a
42-min fetch into a fast local read.

**Feature 2 — Persist + reuse the dough across runs (SPEED — the "pre-bake" vision).**
Save the Stage-1+2 output keyed on (symbol, TF, window) (reuse/extend `bar_cache_store.py`
Tier-2); later runs on the same window skip Stages 1+2 entirely and go straight to Stage-3
gate replays. First bake is slow; every subsequent sweep on that window is near-instant.
Subject to §2 (finalization boundary + invalidation).

## 4. Sequencing
1. **Feature 3 (chunk/stream)** + **Feature 4 (dedicated worker + heartbeat + resume)** —
   these make long sub-minute runs *possible* and *crash-safe*. Do first.
2. **Feature 1 (store-served candles)** + **Feature 2 (persisted dough)** — speed
   multipliers on top, both gated by §2 fidelity.
3. Each step validated against the Fidelity Gate (byte-identical) + a new store-vs-REST
   reconciliation check before being trusted.

## 5. Interim workflow (no engineering)
- **1Min for broad discovery** (≈5× lighter than 30Sec, fits memory now), then confirm
  the few winners at sub-minute via targeted runs.
- Avoid 365-day **sub-minute** sweeps until Features 3+4 land.

## 6. Open questions
- Exact `FINALIZATION_LAG` (15 min vs longer for EOD/T+1 corrections) — measure per asset.
- Does the data-worker store hold enough history (365d) or do we backfill on demand?
- Chunk size (monthly?) vs warmup-continuity cost.
- Worker home: extend `data-worker` vs a new `backtest-worker` service.

## Related
Board tasks **2.30** (store + chunk + dough), **2.31** (load-phase heartbeat),
**2.32** (queued/mid-load recovery). Docs: `Append_Edge_Fossilization.md` (finalization
boundary), `Fidelity_Gate_Guide.md`, `bar_cache_store.py` (existing Tier-2 dough cache).

## Status note (2026-06-21 EOD) — paused, not yet built
- **OOM confirmed** via Railway logs: the `api` service logged `Killed` (SIGKILL = OOM-killer) mid-mass-search, then restarted. Search `1801762576` orphaned, 0 results.
- **Key finding:** it's the WINDOW LENGTH, not just sub-minute. Prep is vectorized over the whole window + the dough holds ~1 object/bar (~80 KB/bar). So **365-day 1Min (~142k bars ≈ ~7–11 GB) would ALSO OOM**, not just 30Sec. Earlier working runs were all short (3–7d) windows.
- **Decision:** truly enabling 365-day sub-hour = multi-part effort (chunk/stream prep + slim the O(bars) dough + maybe stream the confluence search), fidelity-sensitive → NOT same-day. Avoid bumping API pod memory (a 7 GB search inside the API process risks the whole API — Feature 4's point).
- **Next safe step (not yet built):** a fail-loud **memory guard** in `run_mass_search` — estimate largest group's bar count up front, reject oversized runs with a "reduce to ~N days / coarser TF" message so the API never OOMs. Interim workflow: ~90–120 day windows at 1Min.
- Railway CLI access confirmed (ktkventures@gmail.com) for logs + Feature 4 work.
