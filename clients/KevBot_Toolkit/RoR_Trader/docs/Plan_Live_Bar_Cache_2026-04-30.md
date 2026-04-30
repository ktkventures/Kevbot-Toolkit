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

## Decision points before starting

These are open and worth aligning on before code:

1. **Bar timeframes to cache.** All TFs the worker uses, or just
   primary 1Min and let other TFs resample? Recommendation: cache
   all TFs the worker emits, since BarBuilder already produces them.

2. **Symbols to cache.** All active strategies' symbols, or a fixed
   set? Recommendation: dynamic — cache whatever symbols the worker
   is monitoring at any given moment.

3. **Retention period.** 30/60/90 days? Recommendation: 90 days
   default, configurable. Older → REST as fallback.

4. **Write frequency.** Per-bar (current proposal) or batched
   (e.g., every 5 bars)? Per-bar is simpler. Batched reduces write
   pressure but adds complexity. Recommendation: per-bar; revisit
   if Supabase write rate becomes a bottleneck.

5. **REST backfill of historical data.** Backfill 90 days at cache
   start, or let it grow forward only? Recommendation: forward-only
   for simplicity. Historical period uses REST as today; only
   new-from-here-on bars come from cache. After 90 days, the cache
   covers the full active backtest window.

## Deferred for later

- Per-tick cache (one row per trade) for true tick-level Hi-Fi
  backtests. Massive volume increase. Defer until Hi-Fi parity work
  resumes.
- Multi-data-source cache (Alpaca, IBKR) when we eventually expand
  beyond Polygon.
- Historical reconstruction from live ticks (would let us claim
  tick-level fidelity). Not worth the effort for current trading
  style.

---

## Awaiting approval

This plan is the proposed approach. Before implementing:
- Review the schema, write path, and read path for any concerns
- Confirm the phased rollout (read-write → read-cache → cutover) is
  acceptable
- Confirm the decision points above
- Approve the ~2-4 day timeline

Once approved, implementation starts with task 8.7a (schema +
migration) and proceeds in order.
