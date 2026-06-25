# Design — M-RS2: Shared symbol-level 1-second bar store ("super dough") — 2026-06-24

**Status:** DESIGN (react before building). Part of the Recompute Scalability track — see
`Recompute_Scalability_Findings.md` (M-RS1 warmup right-size → **M-RS2 this doc** → M-RS3
parallel + Railway).

## ⚠️ TERMINOLOGY — two different caches, do NOT conflate (read this first)
This system has TWO cache systems with confusingly similar names. They serve OPPOSITE
purposes and must never be mixed. Always qualify which one you mean.

| | **`live_bars` cache** (M8.7) | **REST Canonical Bar Store** (this doc) |
|---|---|---|
| Module / table | `live_bars_writer.py` → `live_bars` | `bar_cache.py` → `bar_cache` |
| **Source** | **WebSocket-built bars** (`source='ws_agg'`, BarBuilder from Polygon AM/A/T) = *what the engine saw at decision time*. REST only patches genuine WS gaps (`source='rest_backfill'`). | **Polygon REST aggregates ONLY** (fed via `load_from_polygon`). **No WS writer ever touches it.** |
| Purpose | Record the engine's decision-time view, for backtest↔live divergence analysis | Reusable REST-canonical history for backtest / recompute / append / Hi-Fi |
| Truth model | "what we saw live" (may differ from REST by ~16-20% at bar close) | "what a fresh Polygon REST pull would return" |

**FIDELITY INVARIANT (non-negotiable):** the REST Canonical Bar Store must be **byte-identical
to a fresh Polygon REST pull**, and **WS data must NEVER write to it**. It is categorically NOT
`live_bars`. The live engine's decision tip stays WS (`live_bars`); this store serves only the
REST-canonical paths — backtest, recompute, append, Hi-Fi, and live *warmup/backfill* (never
the live decision tip). Conflating them silently corrupts the backtest lane's REST-canonical
assumption — the exact bug class this whole project exists to kill.

(The existing `bar_cache.py` is ALREADY REST-fed, so it is the correct V0 of this store — the
name collision with `live_bars` is the trap, not the source.)

## Goal
One canonical, reusable, fidelity-guarded **REST-canonical bar asset per ticker**, fetched from
Polygon REST **once** and reused by every backtest / append / full-recompute / Hi-Fi /
(eventually) live REST-warmup. Kill the redundant per-strategy re-fetching: today, every strategy on a ticker
re-pulls its own months/years of bars from Polygon. With thousands of strategies per ticker
that's thousands of identical fetches + thousands× the Polygon API cost. Fetch once, reuse
everywhere — the way TradingView/real platforms work (central normalized bar store, indicators
computed per-request on top).

## Scope boundary (be honest about what this does and does NOT do)
- **DOES:** speed the I/O load (the 353s in the cProfile) and eliminate cross-strategy
  redundant fetching; give sub-minute strategies a native base; finally persist 1-second across
  sessions; serve as the foundation a dedicated Railway recompute service reads from.
- **DOES NOT:** speed the CPU buckets — engine `process_bar`, user-pack indicators,
  interpreters (~70% of the recompute). Those chew bars regardless of source. CPU is cut by
  **M-RS1** (right-size warmup → fewer bars) and **M-RS3** (parallelize). Don't expect the
  store alone to make a single recompute fast. It makes the *fleet* cheap and the *fetch* free.

## Current state — V0 already exists (this is an EVOLUTION, not greenfield)
`src/bar_cache.py` + Supabase `bar_cache` table already implement most of the plumbing:
- Stores **1Min** bars (raw, all-hours), keyed `(symbol, timeframe, ts)`.
- Delta-fetch on read: backfills historical tail + forward-fills trailing edge (gap > 60s).
- `on_conflict='symbol,timeframe,ts'` upsert → re-fetched bars overwrite.
- Session filter + resample applied **on read** (matches CLAUDE.md "everything from 1Min").
- Gated `BAR_CACHE_ENABLED` (default OFF). Already wired into `load_market_data`
  (`data_loader.py:684-699`) — when enabled it routes through `cached_load_market_data`.

### V0's three gaps (what M-RS2 must fix)
1. **1Min base can't serve sub-minute strategies.** You can't build a 5Sec/30Sec bar from
   1Min. Kevin runs 5/10/15/30Sec strategies → the base must be **1-second** for those tickers.
2. **No revision-horizon guard → interior-revision staleness.** V0 only fetches *after*
   `last_cached_ts` (trailing edge) + *before* `first_cached` (historical tail). A bar cached
   at T and **revised by Polygon at T+3min is never re-fetched** → silently stale → divergence.
   Polygon revises recent bars (documented: "TV revises closed 1Min within ~5 min";
   `project_tv_stability_2026-05-04`). This is THE correctness fix.
3. **Pagination cap = slow.** V0 smoke test was 0.5× (slower) for 30-day loads because
   PostgREST caps at 1000 rows/request unless "Max Rows" is bumped (Supabase → Settings → API
   → Max Rows). At 1-second volumes this is fatal without the bump + proper indexing.

## V1 (M-RS2) design

### ⚠️ CRITICAL FINDING (2026-06-24) — volume is NOT additive across resolutions
Empirical de-risk (TSLA, 2 settled RTH days, true 1s via `_polygon_fetch_bars(...,'second')`):
- **OHLC: 1s→1Min resample == native Polygon 1Min, BYTE-IDENTICAL** (max|Δ|=0.00000).
- **VOLUME: diverges** — 245-252 of 390 bars differ, max|Δ| up to 511,556 (worst at the 09:30
  opening-cross minute). Native 1Min volume ≠ sum of 1s volumes (auction + condition-trade
  aggregation differs between Polygon's 1s and 1min aggregates). Same class as
  [[project_forming_bar_fidelity_nonohlc_packs]]. Also true for native 30Sec vs 1s-summed.

**Implication:** "store 1s, build every TF from it" is faithful for PRICE but corrupts VOLUME.
Volume packs (RVOL, VWAP) served 1s-derived bars would silently diverge from current backtests.

**DESIGN PIVOT (recommended) — cache NATIVE bars per (symbol, timeframe), don't derive.** The
reuse win ("don't re-pull the same data 1000×") comes from CACHING, not from deriving TFs from
1s. So cache each native Polygon resolution a strategy actually uses (1s for Hi-Fi + sub-minute
primaries; native 1Min; native coarse TFs) — fetch-once-per-(symbol,TF), reuse across all
strategies. This preserves byte-identical fidelity (incl. volume) AND keeps the cross-strategy
reuse benefit. It's a refinement of Kevin's idea: same goal, fidelity-safe mechanism. Storage is
basically unchanged (1s dominates the row count either way; native 1Min/coarse are tiny). The
existing `bar_cache` already keys by (symbol, timeframe), so this is the natural extension.
**AWAITING KEVIN'S CALL** before building step 2 — supersedes the "materialized 1Min from 1s"
plan below if accepted.

### 1. Base resolution = 1-second, ALWAYS (DECIDED 2026-06-24) + materialized 1-minute layer
> ⚠️ The "materialized 1Min FROM 1s" part is SUPERSEDED by the volume finding above — materialize
> would corrupt volume. Keep 1s as the sub-minute base + Hi-Fi source, but cache native 1Min/
> coarse TFs rather than deriving them. (Section kept for history; see pivot above.)
- **Store 1Sec as source of truth for EVERY tracked ticker** — uniform, no per-ticker
  resolution policy. Kevin's call: any ticker we'd ever trade also needs 1Sec for Hi-Fi, and
  **consistency = stability**, which is what this store needs most. One rule, no branches.
- Build everything up from 1Sec (1Sec → 5Sec/10Sec/30Sec, and 1Sec → 1Min → 4h). Can't go the
  other way — which is exactly why 1Sec is the right base.
- **Materialize a 1Min layer** (derived from 1Sec, stored as `timeframe='1Min'` rows) so the
  1Min-and-coarser cohort doesn't *resample* ~5.9M 1Sec rows on every read. (Materialize =
  compute the 1Min bars from 1Sec ONCE and store them as their own rows; resample = recombine
  on every read. We materialize 1Min; coarser TFs — 5Min/30m/4h — resample cheaply from the
  ready-made 1Min layer, ~98k rows/yr.) DECIDED: materialize 1Min.

### 2. Fidelity guard — the revision horizon (the core correctness change)
The "revision horizon" = the **age cutoff at which a bar is declared final**. Polygon's recent
bars aren't final the instant they form — late trades trickle in (Kevin: **~15 min typical**,
occasionally longer via FINRA late prints).
- **Settled bars** (`ts < now - horizon`) → immutable → served from store, never re-fetched.
- **Unsettled window** (`ts >= now - horizon`) → **always re-fetched from Polygon REST and
  overwritten** (on_conflict upsert). Never trust the store for the unsettled tip.
- **Horizon value:** anchor on Kevin's ~15-min settling observation, **padded** for safety
  (start ~30-60 min), **adjustable**. Plus a **once-a-day re-pull of the prior trading day** to
  catch rare late prints. Cheap in practice — backtests read mostly OLD data, so the re-pull
  only ever touches the very tip.
- This closes V0's interior-revision gap and makes the store safe for **both** backtest and
  live REST-warmup.
- Live's *current forming bar* always comes from Polygon/WS (`live_bars`) — this REST store
  serves warmup/history, not the live decision tip.
- **Note — same mechanism as the live model's multi-pass refresh:** Kevin's eventual goal is to
  refresh the unsettled tail fast (like the live model prints then refreshes the last few
  candles over ~4 passes as they settle), rather than a flat delay. The revision-horizon
  re-fetch IS that mechanism — start simple (short delay / on-read re-pull), evolve toward the
  fast multi-pass refresh later. Same idea, different speeds.

### 3. Persistence + scale at 1-second volumes
- Table is already persistent (Postgres). Add for 1Sec scale:
  - **Index/partition:** partition by `(symbol, month)` or add a BRIN index on `ts` per
    `(symbol, timeframe)` for fast range scans at tens of millions of rows.
  - **Bump Supabase Max Rows** to 50000+ (removes the V0 pagination penalty).
- **DECIDED 2026-06-24: Postgres-only first.** (Postgres = the standard SQL database Supabase
  already gives us; data lives as rows in tables. Parquet = compressed data *files* in file
  storage — leaner for huge frozen history, but more moving parts.) Start Postgres-only:
  simplest, already built, one system to reason about (fits the stability priority). Measure at
  real 1Sec volumes; add the parquet hybrid ONLY if Postgres proves slow/pricey.

### 4. Ticker-selection control surface (agent page)
- List **all Polygon-supported tickers** (so you can see what's available), checkbox the ones
  to track. Per selected ticker, configure the **capture range** — **default 1yr, adjustable**
  in the admin UI (DECIDED 2026-06-24). Base resolution is always 1Sec (no per-ticker choice).
- Persist selections in a small `bar_cache_tickers` config table. Flipping a ticker ON triggers
  a backfill; flipping OFF stops maintenance (optionally retains or purges data).

### 5. Backfill + maintenance jobs
- **One-time backfill** per newly-enabled ticker (the big initial Polygon fetch — done ONCE,
  then reused forever). This is the heaviest single operation; run it off the critical path.
- **Trailing-edge extend** (cron / on-demand): pull new settled bars as they appear.
- **Unsettled-window refresh:** re-fetch + overwrite the within-horizon window to capture
  Polygon revisions (the fidelity guard, applied as maintenance + on read).

### 6. Read-path integration (single source for backtest/append/recompute/Hi-Fi)
- `load_market_data` already routes to `cached_load_market_data` when enabled. Extend that path
  to (a) serve sub-minute timeframes from the 1Sec base, (b) apply the revision-horizon guard,
  (c) materialize/serve the 1Min layer.
- **Hi-Fi:** point `fetch_1s_bars_for_window` (`data_loader.py:462`, currently in-memory
  `_1s_cache` only, lost per process) at the persistent 1Sec store → cross-session reuse +
  no repeated 1s day-fetches.
- **Relationship to the secondary-TF snapshot:** complementary, not redundant. The snapshot
  caches *warmed indicator state* for coarse secondaries; this caches *raw bars*. With M-RS1
  right-sizing the warmup, the snapshot may become less necessary — but they don't conflict.

### 7. Fidelity validation gate (MUST pass before enabling — fidelity is paramount)
Byte-identical **store-fed == Polygon-fed** for `unified_trades` output (entry/exit ts, price,
exec_type) AND prepared indicator columns, on:
- an ungated **sub-minute** strategy (30Sec) — exercises the 1Sec base + materialized 1Min,
- a **sub-minute + secondary** (sid 309),
- a **coarse 1d/4h** gate (sid 313 / 325).
Plus a **revision-horizon test:** seed the store with a stale unsettled bar, confirm a read
within the horizon re-fetches + overwrites it (served fresh, not stale). Reuse the discipline
from the secondary-TF snapshot + `_compare_cache_replay.py`. **Kill-switch `BAR_CACHE_ENABLED`
(already exists), default OFF until byte-identical-proven.**

## Cost recap
- **Storage $:** ~0.5–0.7 GB/ticker/yr at 1Sec (~10 MB/yr at 1Min). 5 tickers×2yr ≈ ~6 GB;
  50×2yr ≈ ~60 GB ≈ ~$6/mo over Supabase Pro's included 8 GB. **$ is a non-issue.**
- **Real costs:** Postgres perf at tens of M rows (→ partition/index, maybe parquet hybrid);
  the one-time Polygon backfill per ticker; the operational fidelity guard.

## Phase 2 PROGRESS (2026-06-24 autonomous, branch feat/m-rs2-phase2-readpath)
- **Built `bar_cache.read_bars()`** — direct-PG read primitive (psycopg, raw cursor),
  byte-identical to the REST read. Committed.
- **Wiring surface is tiny (confirmed in code):** `prepare_data_with_indicators` loads ONLY the
  PRIMARY via `load_market_data` (services.py:314); SECONDARIES are resampled from the primary
  df (services.py:360), not separately loaded. So for sub-minute strategies the cache only needs
  to serve the **one primary load** — everything downstream (secondaries, indicators, engine)
  derives deterministically. Byte-identical primary ⇒ byte-identical full backtest.
- **CONCLUSIVE A/B on sid 325's primary (TSLA 30Sec, 1 month):** Polygon 34,772 rows/16.0s vs
  cache (read_bars) 34,772 rows/**1.82s = 8.8× faster, BYTE-IDENTICAL (0 value diffs).**
  Sub-minute is the cache's sweet spot (Polygon's sub-minute fetch is slow).
- **Remaining (reviewed next step — NOT done autonomously, touches shared load path):**
  (1) extend `cached_load_market_data` to serve any natively-cached TF via `read_bars` +
  revision-fetch the tip (so `load_market_data(BAR_CACHE_ENABLED)` serves sub-minute from cache);
  (2) end-to-end `get_strategy_trades(325)` cache-vs-Polygon byte-identical + timing; (3) point
  Hi-Fi 1s at the cache; (4) deploy needs psycopg in requirements + SUPABASE_CONNECTION_STRING
  on Railway. Backfill TSLA 30Sec was in progress (need its full window before end-to-end test).

## Phase 2 read-path + speed/reliability findings (2026-06-24, measured)
Backfilled TSLA 1Min (235k, full year) + 1Sec (in progress, ~5.3M) and A/B'd loads:
- **1-Min load (full year, 235k rows):** Polygon 9.3s · Supabase PostgREST 7.6s · **direct
  Postgres 5.7s**. → **Phase 2 reads bulk bars via a DIRECT POSTGRES connection** (psycopg +
  `SUPABASE_CONNECTION_STRING` in src/.env, session pooler), NOT the supabase-py REST client.
  Direct PG is the fastest + is what the read path should use. (1-min is the *floor* of the
  benefit — Polygon's 1-min API is already fast.)
- **1-SECOND speed (CORRECTED 2026-06-24):** cache speedup scales with window size —
  **~1.3× for 1 day** (Polygon's 1s day-fetch is already fast ~4s) and **~4.2× for 1 week**
  (cache ~2.2s vs Polygon ~9s), growing from there. Direct-PG read stays ~2s while Polygon's
  fetch climbs with the window. Plus the fleet-reuse win (fetch once, not per-strategy).
  - **CORRECTION — earlier "Polygon 1s unreliable/throttled/ages-out" claims were WRONG**, an
    artifact of a positional-arg bug in the test harness (`load_from_polygon('TSLA','1Sec',…)`
    bound `'1Sec'` to the `days` param → `timeframe` defaulted to 1Min → all "Polygon 1s"
    numbers were actually 1-minute). Verified: Polygon serves full 1-second (raw endpoint
    25,769 rows/day) and the **cache matches Polygon exactly** (no fidelity/retention issue).
    Always call `load_from_polygon` with `timeframe=` as a keyword.
  → M-RS2's justification is the (window-scaling) speed + fleet-reuse + offline/deterministic
  reads, NOT any Polygon unreliability.
- **Deps added:** `psycopg[binary]` (installed in .venv); `SUPABASE_CONNECTION_STRING` env.
  Supabase "Max Rows" already 50000 (lets PostgREST page in ~50k chunks); bulk reads still
  prefer direct PG.

## Decisions (LOCKED 2026-06-24)
1. **Base resolution:** 1-second, ALWAYS, every tracked ticker (uniform = stable; trading needs
   1Sec for Hi-Fi anyway). Materialize a 1Min layer from it.
2. **Storage engine:** Postgres-only first; add parquet hybrid only if it proves slow/pricey.
3. **Revision horizon:** anchor on ~15-min settling, padded (~30-60 min start), adjustable,
   + once-a-day prior-day re-pull for rare late prints.
4. **1Min layer:** materialize (compute once, store), not resample-on-read.
5. **Live integration:** deferred to a later phase. When added, this store serves ONLY
   REST-canonical warmup/backfill; the live decision tip stays WS (`live_bars`). Evolve the
   unsettled-tail refresh toward the live-model-style multi-pass (fast) over time; OK to start
   with a simple delay.
6. **Backfill range:** default 1yr, adjustable in the admin UI.
7. **Framing confirmed:** REST-canonical — the store must match 1:1 what a fresh Polygon REST
   pull would return. "Speed up what isn't broken without changing what it looks like."

## Sequencing within M-RS2
1. Schema/index changes + Max Rows bump + revision-horizon guard in `bar_cache.py` (kill-switch).
2. 1Sec base + materialized 1Min layer; extend the read path + Hi-Fi to use it.
3. Backfill + maintenance jobs.
4. Ticker-selection agent page + `bar_cache_tickers` config.
5. Byte-identical + revision-horizon validation → enable.
(Build AFTER M-RS1 — right-sized warmup reduces how much the store even needs to load, and
M-RS1 is the cheaper win that makes a single recompute fast first.)
