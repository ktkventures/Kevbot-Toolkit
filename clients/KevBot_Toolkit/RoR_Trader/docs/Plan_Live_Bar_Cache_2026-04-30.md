# Plan — Live Bar Cache

**Created:** 2026-04-30
**Status:** Plan / awaiting approval before implementation
**Estimated effort:** 2-4 days

## Goal

Unify the bar OHLCV data source so that **live** (Polygon WebSocket
aggregated via worker BarBuilder) and **backtest/chart**
(`prepare_data_with_indicators` via `load_market_data`) operate on
**identical bar OHLCV**. Once unified, indicator state (EMAs etc.)
will agree between live and backtest by construction, eliminating the
trade-set divergence we measured at ~5-15% (see
`docs/Parity_Trust_Roadmap_2026-04-29.md`).

## Background — what we're solving

**Today:**
- Live worker uses Polygon WebSocket → BarBuilder → 1Min/10Sec/etc bars
- Backtest uses Polygon REST → `load_market_data` → 1Min/10Sec/etc bars
- For ~58% of bars these agree exactly. For ~42% they differ by
  $0.01-$0.07 (direction varies).
- Engine math is correct on both paths — but the inputs differ, so
  cumulative EMA state diverges, and sensitive triggers (line-vs-signal
  crosses) flip differently between paths.

**Why "two data sources" happened:**
- Polygon REST and Polygon WebSocket are separate APIs that don't share
  aggregation results
- WebSocket aggregates trades into bars in real-time as ticks arrive
- REST returns bars after Polygon's server-side aggregation completes
- They mostly agree but small per-bar differences exist

**Gold standard (TradingView et al.):** one data source, one bar set.
Live, chart, backtest all read the same bars.

## Findings & open questions added 2026-04-30 (post-discussion)

### Polygon explicitly does NOT claim WS = REST parity

Researched Polygon's (now rebranded **Massive** — same vendor) official
documentation. Three direct quotes that contradict our prior assumption:

> *"Users sometimes notice that their own reconstructed OHLCV values
> — whether built from tick data or from WebSocket trades — don't
> always match the aggregates returned by Massive's Aggregates
> endpoint... This is normal."*
> — KB: How does Massive create aggregate bars?

> *"FINRA... has up to 15 minutes to report a trade that happened on
> any of the dark pools. ... if a trade comes in after that 15-minute
> window, it will not be included in the candle until the end of the
> day."*
> — Blog: Understanding Aggregate Bar Delays

> *"There is no universal standard for OHLC data across all vendors."*

Community confirmation: GitHub `polygon-io/issues#88` ("Historical
data never matches streaming data") documents real WS-vs-REST volume
divergence in the wild.

**Implication:** the comment in `_reconcile_with_rest` claiming
"Polygon WS bars = REST bars by construction — no reconciliation
needed" was an unfounded user assumption. Polygon's own docs say
they will differ. The drift we measured (79.4% of bars differ,
median $0.02, p95 $0.09, max $0.835) is consistent with their
documented late-print mechanism.

### The REST reconciliation safety net is currently disabled

`ralph_engine._reconcile_with_rest` (line 2755) was built for the
Alpaca era to correct WS-built bars against REST after settlement.
When we migrated to Polygon (Phase 31), it was made a no-op based
on the faulty assumption above. So we've been running without that
safety net since Phase 31.

**Re-enabling it is NOT the right fix** — it would retroactively
rewrite bars the engine had already evaluated and fired alerts on,
introducing a different kind of inconsistency. M8.7's cache is the
correct architectural answer.

### REST-as-consensus hypothesis (Kevin, 2026-04-30)

Important counterweight to the "WS is more honest" framing:

REST/settled data is what TradingView and most discretionary traders
look at. If a strategy uses an indicator like "price crosses above
9 EMA," **the 9 EMA those other traders see is the REST 9 EMA**, not
our WS 9 EMA. Their trading decisions — and therefore the price
action our strategy is reacting to — are anchored to REST-based
indicators.

Two coherent worldviews emerge:

- **(A) Settled bars are truth** — what TV / REST / consensus shows.
  Backtest-on-REST aligns with the indicators the broader market is
  watching. Mild lookahead bias (data wasn't fully available in
  real-time) but matches "what other traders saw."
- **(B) Real-time tape is truth** — what our WS / live worker sees.
  Honest about information available at decision time. May diverge
  from consensus indicators by pennies-to-dimes.

Neither is universally correct. The choice depends on whether your
edge comes from following consensus (favors A) or from acting on
information faster than consensus (favors B).

### The dual-source forward-test experiment

Plan: instead of pre-committing to one data source, build M8.7 such
that **both REST and WS bars exist in the cache** (distinguished by
`source` column: `'ws'` for live worker writes, `'rest_backfill'`
for REST-derived rows). Then after 4-6 weeks of cache accumulation:

1. Pick ~10 strategies that have been live throughout the cache window
2. Run each backtest twice: once with REST data, once with cache (WS) data
3. Compare each backtest's predictions to that strategy's actual
   forward-test trades over the same window
4. Whichever backtest source correlates better with forward results
   wins for that strategy class

This decides empirically — not by argument — which source is the
better predictor. Possible outcomes:
- WS-backtest wins → default everyone to cache, REST becomes fallback
- REST-backtest wins → keep REST as default, WS-cache only for live-
  alert-explanation
- Mixed by indicator class (e.g., trend strategies prefer REST,
  scalp strategies prefer WS) → expose per-strategy/per-pack toggle

**No user-selectable toggle in v1.** Adds UI surface for a setting
data may say isn't needed. Build the foundation, run the experiment,
then decide. The two sources COEXIST naturally (REST is the existing
backtest path; cache is what M8.7 adds), so we get the experiment
for free without dual-rendering.

### Heatmap and price chart are aligned for completed bars

Confirmed by code trace. Both `chart-data` endpoint and the heatmap
read from `prepare_data_with_indicators` (REST path). The forming
bar at the right edge of the chart is the only WS-derived element,
and the heatmap doesn't render a cell for the forming bar at all.
So heatmap and chart are apples-to-apples for every completed bar
shown in the UI today.

After M8.7 cutover, both will read from cache for completed bars —
still aligned, but now also aligned with what the live worker saw.

### Concrete data flows for each surface (current state)

| Surface | Bar source today |
|---|---|
| Live worker (alerts) | REST for 7-day warmup at startup, WS for everything since |
| Backtest / algo history | 100% REST via `prepare_data_with_indicators` |
| Price chart in UI | REST for completed bars; WS for forming bar (Supabase Realtime broadcast) |
| Heatmap | 100% REST (same as backtest) |

After M8.7 cutover, all four shift to cache-first with REST fallback.
Alert history doesn't change (it's already authoritative WS). The
others all align toward agreement.

## Architectural approach

### Layer 1 — Worker writes bars to a cache table

When the worker's BarBuilder closes a bar, it writes the bar's OHLCV
to a new Supabase table `live_bars`:

```sql
CREATE TABLE live_bars (
    symbol TEXT NOT NULL,
    timeframe_seconds INT NOT NULL,
    bar_start TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    trade_count INT,
    vwap DOUBLE PRECISION,
    written_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT NOT NULL DEFAULT 'ws',  -- 'ws' (worker) or 'rest_backfill'
    PRIMARY KEY (symbol, timeframe_seconds, bar_start)
);
CREATE INDEX live_bars_lookup ON live_bars (symbol, timeframe_seconds, bar_start);
```

Per-bar write volume estimate (rough):
- Active strategies: ~17 currently, mostly SPY 10Sec / 1Min
- Bars/min/symbol/tf: 6 for 10Sec, 1 for 1Min
- Symbols: ~3-5 (SPY, QQQ, AAPL, etc.)
- Timeframes per strategy: primary + 1-2 secondaries
- Net: ~50-200 writes/min during market hours
- Daily: ~30K writes during 6.5h RTH
- Per row: ~150 bytes
- Daily storage: ~5 MB

This is well within Supabase free tier limits.

### Layer 2 — Backtest reads from cache first, REST fallback

Modify `data_loader.load_market_data` to:

1. Query `live_bars` for the requested (symbol, tf, time range)
2. Identify any gaps in the cache coverage
3. Fill gaps from REST (`load_market_data` legacy path)
4. Optionally backfill those REST-fetched bars into `live_bars` with
   `source='rest_backfill'` so subsequent calls don't repeat the REST
   fetch
5. Return the merged frame

Cache hit rate over time:
- Day 1 of cache: 0% hit (no cached bars yet)
- Day 7: ~100% hit for the last 7 days, REST for older
- Day 30+: ~100% hit for everything since cache start

### Layer 3 — Migration / coexistence

Phased rollout:
1. **Phase 1 (read-write):** Worker writes to `live_bars`. Backtest
   still reads from REST as primary (don't break anything yet). Cache
   accumulates silently. Validate writes are correct.
2. **Phase 2 (read-cache):** Backtest checks `live_bars` first, falls
   back to REST. Verify backtest agrees with live for cached-period
   strategies.
3. **Phase 3 (parity sweep):** Re-run the parity/fidelity sweep on
   strategies whose data is fully cached. Expect fidelity to climb to
   ~100%.

Each phase is independently revertable.

## Tasks (proposed)

### Required (in order)

- [ ] **8.7a — Schema + migration**
  Create `live_bars` table in Supabase (manual SQL or migration file).
  Confirm RLS rules: admin client writes, admin client reads, no user-
  level access (it's worker telemetry, not user data).

- [ ] **8.7b — Worker write path**
  In `ralph_engine.SymbolHub` BarBuilder bar-close hook, add a write
  to `live_bars` after each successful bar emission. Use admin client.
  Best-effort write (log + continue on failure — don't crash the
  worker on Supabase hiccups).

- [ ] **8.7c — Validation script**
  `_validate_live_bars_cache.py` — query `live_bars` for the last
  hour, compare to Polygon REST for the same range. Confirm match
  rates. Surface any rows where cache and REST disagree.

- [ ] **8.7d — Backtest read path**
  Add `_load_from_cache_with_rest_fallback()` helper in `data_loader.py`.
  `load_market_data` opt-in via env var or kwarg initially. Coexists
  with REST-only path.

- [ ] **8.7e — Gap detection + backfill**
  When cache is incomplete for a requested range, fetch the gap from
  REST and insert into `live_bars` with `source='rest_backfill'`. So
  cache becomes self-healing.

- [ ] **8.7f — Cutover**
  Flip `load_market_data` default to use the cache-first path. Keep
  REST-only as escape hatch.

- [ ] **8.7g — Re-run parity sweep + Q3 fidelity drill**
  Run `_rerun_all_parities.py` and `_fidelity_check_overnight.py` after
  cutover. Confirm fidelity at ~100% for cached-period strategies.
  Document the delta.

### Polish (after Required is done)

- [ ] **8.7h — Cache cleanup job**
  Periodic deletion of bars older than N days (configurable). Default
  90 days. Older history → REST.

- [ ] **8.7i — Cache hit-rate metric**
  Log cache hit-rate to a metrics endpoint. Surface in admin UI later.

- [ ] **8.7j — Worker downtime gap handling**
  When worker restarts after downtime, REST-backfill the missed window
  on startup. So the cache stays continuous.

### Deferred

- [ ] **8.7k — Cache strategies' historical backfill**
  For each currently-active strategy, REST-backfill the last 90 days
  of bars at cache-start so existing backtests benefit immediately.
  ~1 hour batch run.

- [ ] **8.7l — Cross-process cache invalidation**
  If we ever have multiple workers on the same symbol, deduplicate
  writes via primary-key UPSERT instead of INSERT. Currently 1 worker
  per symbol so no conflict.

## Risk + rollback

**Risks:**

1. **Worker write latency** — Supabase write per bar adds latency.
   Mitigation: batch writes (every 1-5 bars), or use async/fire-and-
   forget. Even with 50ms per write, that's <2% of a 10Sec bar window.

2. **Cache vs REST disagreement on backfilled bars** — when we REST-
   backfill into the cache, those bars came from REST, not WS. So they
   won't actually match what live worker would have produced. This is
   acceptable: we mark them `source='rest_backfill'` and accept the
   small drift on historical bars — only NEW bars (post-cache-start)
   are guaranteed to match.

3. **Schema migrations** — adding a new table is low-risk, but worth
   doing when no critical writes are in flight. Maintenance window.

**Rollback:**

Each phase is revertable:
- Phase 1: drop the worker write code → cache stops accumulating but
  doesn't break anything
- Phase 2: revert `load_market_data` to REST-only via env flag → no
  impact on writes
- Phase 3: stop relying on parity sweep results, return to drift-aware
  trading

Backup branch saved before any code changes:
`dev-backup-pre-live-bar-cache-2026-04-30`.

## Success criteria

After all Required tasks ship:

1. Worker writes ~30K bars/day to `live_bars` table during RTH.
2. `load_market_data(symbol='SPY', days=7, ...)` reads ~100% of bars
   from cache, 0% from REST.
3. Re-running the parity sweep on previously-PARTIAL strategies shows
   fidelity → ~100% on bars sourced from the cache.
4. New live alerts continue to match a fresh local backtest run on
   the same bars (cached) within $0.0001 of close prices.

## Decision points — RESOLVED 2026-04-30

These have been agreed on with Kevin:

1. **Bar timeframes to cache: ALL TFs the worker emits** (1Min, 10Sec,
   5Min, 15Min, 1Hour, 1Day, etc. — whatever's in the BarBuilder set
   at any moment). NOT 1Sec for now — that's deferred to M8.6 (Hi-Fi)
   so we don't conflate scopes. Resampling at read-time would add
   ~50-100ms latency per read which compounds in mass builder, and
   storage of all TFs is only ~20-40 MB/day.

2. **Symbols to cache: dynamic.** Whatever the worker is monitoring at
   any given moment. As strategies are added/removed, the set adapts.

3. **Retention: indefinite (revised 2026-04-30).** Original plan said
   90 days; reconsidered after recognizing the dual-source experiment
   and the compounding-trust value of long cache history. Math: ~30K
   bars/day × 365 = ~11M rows/year ≈ <1 GB/year. Negligible. Keep
   indefinitely; revisit only if Postgres footprint becomes a real
   concern. Partition by month later if needed.

4. **Write frequency: per-bar.** Simpler. Revisit if Supabase write
   rate becomes a bottleneck (unlikely at <200 writes/min).

5. **Historical REST backfill at cache-start: forward-only.** Older
   periods use REST as today; only bars from cache-start onwards are
   guaranteed to come from cache. After 90 days, the active backtest
   window is fully cache-sourced.

## Worker outage handling (decision noted)

When the worker comes up after downtime, REST-backfills the gap window
for (symbol, TF) pairs it's now monitoring. Backfilled rows have
`source='rest_backfill'` so they're distinguishable from `source='ws'`
rows. **Important caveat:** these gap-bars have the same WS-vs-REST
drift property as historical REST data — they're not "true live."
Cache continuity is preserved but trades evaluated on these specific
gap-bars retain a small data-source asymmetry. Acceptable trade-off
for not having a hole in the cache.

## Future: Hi-Fi extension (M8.6 enabling)

Caching the BarBuilder-aggregated bars for M8.7 is a stepping stone.
When Hi-Fi work resumes (M8.6), the natural next layer is to cache
1Sec bars (the BarBuilder's source data) to a related table
`live_bars_1s`. Then Hi-Fi 1Sec recomputation uses the same source as
live → guaranteed parity for Hi-Fi triggers AND CB-fidelity confluence.
NOT in scope for M8.7, but architecturally the schema and write path
generalize cleanly.

## Verification checklist (post-implementation)

Run these BEFORE marking M8.7 complete. Mix of automated checks
(Claude can run) and visual checks (Kevin verifies in browser).

### Automated checks (Claude)

- [ ] **Cache write integrity** — Worker writes ~30K bars/day during
  RTH; rows include all expected TFs and symbols; written_at is fresh
  (within seconds of bar_start)
- [ ] **Cache read parity** — `_validate_live_bars_cache.py` confirms
  cached bars match live alert prices on the bars where they fired
- [ ] **Phase D regressions** — `python test_parity_regression.py`
  passes (7/7 — no engine changes, but sanity check)
- [ ] **Synthetic probe** — `_phase_e_retrofit.py` re-run on all 10 v2
  packs; expect verdicts at or above previous results (5 PASS / 5 PARTIAL)
- [ ] **Parity sweep on cached-period strategies** —
  `_rerun_all_parities.py` on sids that have been live since cache
  start; expect aggregate parity score climb above 0.79
- [ ] **Q3 fidelity drill** — `_fidelity_check_overnight.py` on the
  same strategies; expect fidelity rate ~100% on cached-period bars
- [ ] **Mass builder dry run** — load any saved mass-search config and
  run a small N=10 sweep; confirm trades produced match pre-cache
  baseline (caching is data-source-only; trade results should be
  IDENTICAL on cached-period bars)

### Visual checks (Kevin)

- [ ] **Heatmap rendering** — Strategy Detail page heatmaps render
  exactly as before; gates that were green before are still green;
  red bars still red. (No expected changes, just sanity.)
- [ ] **Algo history vs alert history alignment** — On a strategy
  monitored over the last few days, the algo-history table and
  alert-history table on the Strategy Detail page should now line up
  much more closely than before. Mismatches that were "live fired
  but algo didn't" should be substantially reduced.
- [ ] **Strategy Builder functional** — Try saving a new strategy from
  Strategy Builder; backtest renders; KPIs populate. (Currently
  stalling per follow-up; this is also a chance to fix or confirm.)
- [ ] **Mass Strategy Builder functional** — Try a small mass search;
  results render; trades match.
- [ ] **Live chart real-time** — Live chart on Strategy Detail still
  updates with each tick; no degradation from the worker's added
  write step.
- [ ] **Pack 4Q test in UI** — Run "Run Parity Test" on one user pack
  via the UI; confirm verdict + score still display correctly.

### Acceptance gate

M8.7 is COMPLETE when:
- All 7 automated checks pass
- All 6 visual checks pass
- Aggregate parity sweep score has climbed (target: >0.90 from
  current 0.79)
- Q3 fidelity check shows >95% on cached-period strategies (from
  current ~50-85%)
- Worker has been running with cache writes for ≥7 days with no
  noticeable performance degradation

## Deferred for later

- Per-tick cache (one row per trade) for true tick-level Hi-Fi
  backtests. Massive volume increase. Defer until Hi-Fi parity work
  resumes.
- Multi-data-source cache (Alpaca, IBKR) when we eventually expand
  beyond Polygon.
- Historical reconstruction from live ticks (would let us claim
  tick-level fidelity). Not worth the effort for current trading
  style.
- **Polygon Snapshot endpoint as cache health probe.** Polygon
  publishes a `/snapshot/locale/us/markets/stocks/tickers/{ticker}`
  endpoint that returns the canonical "current state" view (last
  trade, last quote, current/prev-day aggregates). Polygon docs
  position it as the authoritative current-state query. We don't
  use it today — our cache is fed by WS, never reconciled against
  Snapshot. Potential future use: periodic health-check that
  compares the most recent live_bars row against Snapshot's
  current-day aggregate as a sanity probe. NOT in M8.7 scope. Worth
  knowing it exists for later debugging if we ever doubt the cache.

## Backtest-model-as-strategy-variable (idea recorded 2026-05-01)

Kevin proposed treating the data source as an explicit, per-strategy
configurable optimizable variable (alongside trigger packs, stop
methods, exit packs, time-exit packs). Instead of one global default
that changes when we ship cache cutover, each strategy declares
which "model" it was authored against — backtest model + live model.

Likely model values once we have data:
- `rest_settled` (current behavior — Polygon REST aggregates)
- `ws_settled`   (cache `close` — WS with rebroadcast corrections)
- `ws_first`     (cache `first_close` — what the live engine actually saw)
- `hifi_*`       (per-second / tick refinement, future)

Benefits: existing strategies never break when defaults shift; we
can backtest the same strategy on multiple models and pick what
forward-tests best; Hi-Fi gates collapse from a separate UX axis
into a model variant. Cost: cognitive surface for users + UI work
to expose the choice. NOT scoped — record for the dual-source
experiment outcome to inform whether this is worth building.

---

## Approved 2026-04-30 — implementation starting tonight

Tonight's scope (minimal, reversible):
- 8.7a (schema + migration)
- 8.7b (worker write path, fire-and-forget, behind
  `LIVE_BAR_CACHE_WRITE_ENABLED` env flag)
- 8.7c (validation script for tomorrow morning sanity check)

Read path (8.7d–8.7f), gap detection, parity sweep, and frontend
changes deferred to subsequent sessions. Tonight's deploy is pure
recording — zero behavior change for anyone, full kill-switch via
env flag.

Tomorrow's three-way comparison (our WS cache vs Polygon REST vs
TradingView SPY 1Min export) feeds back into the dual-source
experiment plan documented above.
