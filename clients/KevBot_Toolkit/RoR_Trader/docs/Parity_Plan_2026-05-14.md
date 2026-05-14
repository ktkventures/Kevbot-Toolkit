# Parity Plan — RoR Trader Model Alignment (2026-05-14)

## Status (as of EOD 2026-05-14)

| Phase | Status | Notes |
|---|---|---|
| Step 0 | ✅ DONE | Plan persisted to repo |
| Phase A — Round-robin live-bar throttle | ✅ DONE | Live badges working again post-deploy |
| Phase B — Bars Comparison tab | ✅ DONE | Live; user reports browser-crash from too many rows → addressed in Phase F |
| Phase C — Entry Overlay + Divergence Heatmap | ✅ DONE | Live; user reports marker convention should match existing chart-and-trades tab → addressed in Phase F |
| Phase D — Hi-Fi cross-pollination ruled out | ✅ DONE | Investigation report shipped; defensive `data_source_filter` added; **architectural decision deferred to Phase E findings** |
| **Phase F — UI Polish (NEW, next)** | ⏳ STARTING | Pagination, synced charts, marker convention, indicator overlays. Promoted ahead of Phase E because user reports B's table is unusable as-is. |
| **Phase E — Flat-file observable bars (REVISED, was gated)** | ⏳ NEXT | DE-GATED. Diagnostic-first approach: reconstruct what was observable from Polygon flat-file ticks, compare to live_bars cache. Tells us WHERE divergence comes from. |
| **Phase G — Live engine fix (PLACEHOLDER)** | 🔒 GATED | Only runs if Phase E reveals our live engine has gaps vs flat-file observable. Specifics TBD by what Phase E surfaces. |

## Roles framing (corrected 2026-05-14 EOD)

Kevin clarified the role of each model — this changes the parity target:

- **Backtest model** (default `rest_hifi`): "what to backtest against" → REST settled data is fine. Not expected to match live's data perfectly; it's the post-correction view.
- **Algo model** (default `cache_locked`): "keep live accountable" → MUST mirror what live saw faithfully, including any cache gaps. If algo and live diverge, that isolates an execution issue.
- **Live model** (default `ws_agg_locked`): the actual production execution path.

**Real parity target: live ↔ algo.** algo ↔ bt divergence is expected (different data sources by design). Don't optimize for it.

This means option γ (changing algo default to `rest_hifi`) is REJECTED — it would defeat algo's accountability role.

## Context

We have three execution models in RoR Trader:
- **Live** (Ralph engine, `ralph_engine.py`, `live_model='ws_agg_locked'`)
- **Algo** (replays through `unified_engine.py` with `algo_model='cache_locked'`, then Hi-Fi Pass 2 refines)
- **Backtest** (replays through `unified_engine.py` with `backtest_model='rest_hifi'`, then Hi-Fi Pass 2 refines)

Today's canary divergence numbers (sid 171/172, 25-min RTH window post-task-#11+#12 deploy):
- live ↔ algo: 8/10 = 80%
- live ↔ backtest: 7-9/9 = ~85%
- algo ↔ backtest: 8-9/10 = 85-90%

Kevin's external tick-data analysis (his Claude-app session, captured in `docs/Claude_App_REST_vs_Cache_Analysis.md`) proved that **Polygon's observable-vs-settled divergence on SPY 10Sec is 99.1% match, max 6¢ on a high**. That's the ceiling. We're at 80%. The gap is in OUR pipeline, not Polygon's.

Target: **≥95% match, stretch 99%+.** Approach: build visual validation tooling first, then make targeted fixes based on what the tooling surfaces. No more "trust the theory" — every claim demonstrably visible in the app.

## Step 0 — Persist this plan into the repo

Before Phase A starts, copy this plan into the repo at:

  `clients/KevBot_Toolkit/RoR_Trader/docs/Parity_Plan_2026-05-14.md`

so Kevin can reference it from inside the project and so we stay aligned across sessions. Same content, no edits. Commit on dev (no backup branch needed — pure doc add).

## Standing Practices

- **Autonomous boundary**: Continuous execution through Phases A → D. Pause and confirm before Phase E (tick ingestion is real infra).
- **Backup branch per phase**: `dev-backup-pre-phase{A|B|C|D|E}-2026-05-14` before any code changes.
- **Commit + push at phase boundary**, then start next phase.
- **No worker restarts during active data-collection tests** Kevin has running. The throttle fix (Phase A) requires a Railway redeploy, which IS a worker restart. Confirm Kevin isn't mid-test before pushing Phase A.
- **Canary log update** (`docs/Canary_Test_Log.md`, new file) when any test strategy is created or retired.

## Phase A — Live-chart throttle round-robin

**Goal**: fix "Not Live" badge so visual validation in subsequent phases is possible.

**Root cause**: commit `738ab38` changed throttle key from `(symbol, tf_seconds)` → `user_id` to fix Supabase outages. The per-user 1Hz budget means whichever `(symbol, tf)` consistently fires first each second wins; others deterministically starve. Kevin sees "Not Live" on most charts.

**Files**:
- `src/live_bar_publisher.py` (sole edit)

**Change**:
- Augment `_last_published: dict[str, float]` (user_id → epoch_ms) with `_pending_by_user: dict[str, collections.deque[(symbol, tf)]]` per-user FIFO queue
- Replace `_should_throttle(user_id)` deterministic-first-wins logic with: track which `(symbol, tf)` was published last for this user in the current 1-sec window; on next publish attempt, prefer a tuple NOT recently published. Effectively: 1Hz budget, rotated fairly across all active tuples.
- Preserve existing 1Hz cap (Supabase load unchanged)
- Preserve existing API: `publish_async(user_id, symbol, tf_seconds, bar, is_forming, extras)` unchanged

**Callers (no changes needed)**: ralph_engine.py:1428 (sole call site, via `_publish_completed_bar`)

**Verification**:
- Local: unit test verifies 5 distinct `(symbol, tf)` tuples publishing concurrently each get 1-of-5 broadcasts over 5 seconds (vs current behavior: 1 of them gets all 5, others get 0)
- Production: Kevin opens 3 Strategy Detail pages on different (symbol, tf) tuples; all three show "● Live" badge within 50s

**Rollback**: if Supabase load spikes, env var `LIVE_BAR_THROTTLE_DISABLE_ROUNDROBIN=true` reverts to current behavior without redeploy.

## Phase B — Admin > Parity: scaffold + Bars Comparison tab

**Goal**: prove or disprove the "cache coverage" theory visually. Side-by-side `live_bars` cache rows vs Polygon REST aggregates for the same window.

**Files (new)**:
- `frontend/src/app/admin/parity/page.tsx` — Next.js route wrapper, dynamic import + ssr:false
- `frontend/src/views/AdminParityPage.tsx` — main page logic with TabBar
- `frontend/src/charts/ParityBarComparison.tsx` — synced two-pane chart (live_bars cache | REST aggregates) using `SyncedChartPane` primitive
- `frontend/src/hooks/queries/useAdminParity.ts` — orchestrates existing hooks

**Files (edited)**:
- `frontend/src/components/Sidebar.tsx` — add `{ href: '/admin/parity', label: 'Parity' }` to admin children (line ~109)

**Reuse**:
- `useStrategyCacheBars` from `useStrategies.ts` for live_bars rows
- `useBars` from `useMarketData.ts` for Polygon REST aggregates
- `SyncedChartPane` for the side-by-side rendering
- `PageHeader`, `TabBar`, `Card` layout primitives

**Tabs (Phase B implements first one only; others stubbed)**:
- **Bars Comparison** (✅ Phase B): live_bars cache vs REST aggregates, OHLCV diff column
- **Entry Overlay** (stub → Phase C): "Coming in Phase C"
- **Divergence Heatmap** (stub → Phase C): "Coming in Phase C"
- **Ticks** (stub → Phase E): "Coming in Phase E (flat-file ingestion required)"

**Verification**: navigate to `/admin/parity`, select sid 171, choose window 19:35-20:00 UTC today. See side-by-side panes: left shows live_bars cache OHLCV, right shows Polygon REST OHLCV. Diff table below highlights any minute where they disagree by >1¢ or >5% volume. **If they fully agree → "cache coverage" theory was wrong, divergence comes from somewhere else (Phase C will find it).**

## Phase C — Admin > Parity: Entry Overlay + Divergence Heatmap

**Goal**: visualize WHERE live/algo/BT disagree on entries/exits at the trade level. Click into any disagreement to see why.

**Files (new)**:
- `src/api/routers/admin_parity.py` — endpoint `/api/admin/parity/snapshot?strategy_id=&start=&end=` returning unified `{bars, live_alerts, algo_trades, bt_trades}` payload
- `frontend/src/charts/ParityEntryOverlay.tsx` — TradingChart with three marker colors (green=live, orange=algo, blue=bt) on same price chart
- `frontend/src/charts/ParityDivergenceHeatmap.tsx` — grid of (minute) × (lane) showing match/mismatch state

**Files (edited)**:
- `frontend/src/views/AdminParityPage.tsx` — wire up Entry Overlay + Heatmap tabs

**Reuse**:
- `SyncedChartPane` + `TradingChart` for chart + markers
- `useStrategyTrades`, `useStrategyAlgoTrades`, `useStrategyAlerts` for the three trade streams
- AdminDivergencePage's color scheme (green ≤2s drift, yellow ≤30s, red >30s)

**Interaction model**:
- Select strategy + window from page-level filter
- Entry Overlay tab: price chart with overlaid markers; click any marker → side panel shows that entry's full record from each lane
- Heatmap tab: row per strategy, column per minute, colored by 3-way match state; click any cell → drill into that minute's lane states

**Verification**: select sid 171, window 19:35-20:00 UTC. See ~9 live markers, ~10 algo markers, ~13 BT markers. The 19:40:10 algo-only marker shows visually. Click it → drill-down shows: live position state = "IN_TRADE since 19:39:20"; algo position state = "FLAT (exited at 19:39:50 via L-type stop @ 712.45)". **This visually surfaces the actual divergence cause for Phase D.**

## Phase D — Root-cause investigation + targeted fix

**Goal**: address the largest source of residual divergence Phase C surfaces.

**Phase 1 hypotheses (updated post-exploration)**:

1. **Hi-Fi Pass 2 cross-pollination**: `run_hifi_pass2` (strategies.py:654) reads ALL trades for a strategy via `load_trades_admin(strategy_id, user_id)` with NO `data_source_filter`. Algo + BT trades get refined together. If trade A is algo's 19:39:50 exit and trade B is BT's 19:39:55 exit, Hi-Fi might cross-refine them in unintended ways.

2. **Engine-implementation divergence**: live runs `ralph_engine.py`; algo + bt both run `unified_engine.py` with model_override. Even though both should be "bar-by-bar incremental," they're different code paths. L-type stop evaluation might differ between them.

3. **L-type stop reference price**: live evaluates stops on per-second WebSocket prices; algo replays from cached bar high/low. Same logical stop level, different observation resolution → different fire timestamps.

**Investigation method**: use Phase C's drill-down to identify which hypothesis is dominant for the largest divergence cluster. Don't pre-commit to a fix until the data shows.

**Likely fix targets** (placeholder, will refine post-investigation):
- `src/api/services/forward_test_service.py:1607-1618` (Hi-Fi Pass 2 call site) — add `data_source_filter` parameter
- `src/api/routers/strategies.py:654` (Hi-Fi Pass 2 entry point) — propagate filter
- `src/ralph_engine.py` L-type stop evaluation
- `src/unified_engine.py` matching code path

**Verification**: re-pull canary divergence query post-fix. Target match% improvement: 80% → 90%+ (lower bound), 95%+ (target), 99%+ (stretch).

## Phase F — UI Polish (added 2026-05-14 EOD, next up)

**Goal**: make Phases B/C visually usable before any architectural work. User feedback after first use surfaced specific issues.

**Files (edited)**:
- `frontend/src/charts/ParityBarComparison.tsx` — pagination, replace dual TradingChart with SyncedChartPane (shared x-axis), add disagreement-state histogram pane, indicator overlays
- `frontend/src/charts/ParityEntryOverlay.tsx` — replace lightweight-charts built-in markers with price-aware glyphs at actual entry price
- `frontend/src/views/AdminParityPage.tsx` — minor wiring for indicator-toggle controls

**Changes**:
1. **Bars Comparison pagination** — table renders 50 rows per page, sortable descending by default. Don't fetch differently; just slice. Browser-crash regression eliminated.
2. **Shared chart x-axis** — `SyncedChartPane` pattern instead of two independent `TradingChart` panes. Synced time-axis state, equal zoom/scroll behavior.
3. **Disagreement histogram** — strip below the price charts showing per-minute match state (green = cache≈REST within $0.01 + ±5% vol, red = diverge). Same visual primitive as the existing confluence heatmap.
4. **Indicator overlays** — toggle to render the strategy's indicators (EMA stack, MACD, VWAP, etc.) on the cache + REST charts using the same code path as the Lab tab. Confirms or refutes "the indicators look right on each."
5. **Entry Overlay marker convention** — replace `arrowUp` / `circle` aboveBar/belowBar markers with price-aware glyphs:
   - **Live** = X cross (matches existing chart-and-trades convention)
   - **Algo** = + cross (matches existing chart-and-trades convention)
   - **Backtest** = hollow circle (new convention agreed with Kevin)
   - All three overlap at the same point when 3-way matched → visually obvious "perfect entry"
   - Color encodes lane (green / orange / blue); shape encodes source. Together they're more readable than the current dual-encoding-on-shape attempt.

**Verification**: navigate to `/admin/parity` on sid 171 today's window. Browser doesn't crash. Two charts scroll together. Disagreement strip shows specific minutes as red. Entry Overlay renders 3-symbol stacks at matched entries.

## Phase E — Flat-file observable bars (DE-GATED, REVISED 2026-05-14 EOD)

**Status**: NOT gated anymore. Diagnostic-first principle: until we know whether our live engine matches flat-file observable, all architectural decisions are guessing.

**Goal**: build the dataset that proves or disproves whether our `live_bars` cache faithfully reflects what was emitted to subscribers in real time. Tells us if live-engine divergence is upstream (data) or downstream (engine).

**Methodology choice**: build **observable bars** — 1-second OHLCV reconstructed from the trades flat file with `sip_timestamp` as the bucket key (the bar gets a trade if SIP emitted it during that second). This is the Claude-app methodology. NOT tick-replay (which is HFT-style ticker animation).

**Scoping decisions (recorded 2026-05-14 EOD with Kevin)**:
- **Symbols (initial)**: SPY + TSLA only. Expand later if useful.
- **Window**: 7-day rolling. Ingest daily; drop entries older than 7 days. Bounds storage.
- **Storage approach**: do NOT persist raw ticks. Process trades file in-stream, filter to target symbols, build 1-second observable bars, store only those.
- **Cron timing**: weekdays at ~12:30 PM ET (after Polygon publishes day-N flat file at ~11 AM ET on day N+1).
- **Cost estimate**: storage ≤ $0.50/month above Supabase Pro's included 8GB. Egress/processing absorbed by existing Railway worker. Total marginal cost ≈ $1/month.
- **Expansion path**: if Phase E delivers value, scope decision is "watch symbols" (~5-10) vs "all tickers" (~180GB/year = ~$20/month). That decision deferred.

**Files (new)**:
- `src/services/flat_file_ingestion.py` — boto3 client for Polygon S3 (us_stocks_sip/trades_v1 bucket). Streams + filters per-symbol → builds 1-second observable bars → writes to Supabase. Watchlist driven by env or config.
- Supabase migration: new `polygon_observable_bars` table — `(ticker, sip_second_ts, open, high, low, close, volume, trade_count)` indexed on `(ticker, sip_second_ts)`. 7-day TTL via daily purge.
- `src/cron/flat_file_daily.py` — wraps the ingestion in a Railway cron entry point.
- `railway.json` or equivalent — add daily cron schedule (weekdays, 12:30 PM ET).
- `src/api/routers/admin_parity.py` — extend with `/api/admin/parity/observable?symbol=&start=&end=&tf=` endpoint that returns observable bars (aggregated to requested TF on read).
- `frontend/src/charts/ParityObservableComparison.tsx` — three-pane: cache (live_bars) | observable (flat-file rebuilt) | settled (REST). Same SyncedChartPane primitive as Phase F.
- `frontend/src/views/AdminParityPage.tsx` — wire the Ticks tab to the new comparison.

**Verification**:
- Ingest SPY for 2026-05-07 (the day Claude-app covered) → observable bars for that day land in Supabase
- Compare to settled REST bars for same day → max 6¢ divergence on highs (matches Claude-app finding)
- Navigate to `/admin/parity` > Ticks tab on sid 171 → three-pane comparison renders. If `live_bars` cache deviates from observable, the chart shows it; if they match, divergence is downstream of the WS layer.

## Phase G — Live engine fix (placeholder, GATED on Phase E)

**Status**: only runs if Phase E reveals our `live_bars` cache differs meaningfully from flat-file observable. If they agree, Phase G is unnecessary and parity work is done.

**Possible scopes** (will refine after Phase E):
- WebSocket subscription health checks (are we silently dropping messages?)
- Reconnect timing tuning
- Per-second bar aggregation correctness (are we bucketing correctly?)
- Storage-write ordering / rebroadcast handling

Specifics depend entirely on what Phase E surfaces. Don't pre-design.

## Critical Files to Read Before Each Phase

| Phase | Read first |
|---|---|
| A | `src/live_bar_publisher.py` (177 lines), `src/ralph_engine.py:1397-1437` (single call site) |
| B | `frontend/src/views/AdminDivergencePage.tsx`, `frontend/src/components/Sidebar.tsx:101-109`, `frontend/src/charts/SyncedChartPane.tsx:1-80` |
| C | `frontend/src/views/StrategyDetailPage.tsx:611-709` (DivergenceTabContent), `frontend/src/charts/TradingChart.tsx:26-32` (TradeMarker interface) |
| D | `src/api/services/forward_test_service.py:1607-1618`, `src/api/routers/strategies.py:654-700`, `src/recompute_engine*.py` |
| E | Kevin's Claude-app analysis recap in `docs/Claude_App_REST_vs_Cache_Analysis.md` |

## Reusable Utilities (DO NOT reimplement)

- **Auth**: `frontend/src/providers/AuthProvider.tsx` — Supabase session, localStorage tokens
- **Sidebar nav pattern**: `frontend/src/components/Sidebar.tsx:101-109` (admin children array)
- **Page layout**: `PageHeader` + `Card` + `TabBar` primitives in `frontend/src/components/`
- **Chart primitives**: `SyncedChartPane.tsx` (synced multi-pane) + `TradingChart.tsx` (single-pane + markers)
- **Existing data hooks**: `useBars`, `useStrategyCacheBars`, `useStrategyTrades`, `useStrategyAlgoTrades`, `useStrategyAlerts`, `useAdminDivergenceSummary`
- **Existing models module**: `src/strategy_models.py` (BACKTEST_MODELS, LIVE_MODELS dicts with availability flags)
- **Hi-Fi Pass 2 entry point**: `src/api/routers/strategies.py:654` (`run_hifi_pass2`) — already applies to algo + bt; Phase D may add `data_source_filter` param

## Out of Scope (this plan)

- Task #8 (chart timestamp gaps — frontend only, deferred per Kevin)
- Strategy-creation UI changes (algo_model default is `cache_locked` per `strategy_models.py:180-190`; changing it is a separate UX question)
- Webhook event redesign progress (separate phase)
- Phase 38 frontend migration (separate phase)
- sid 154 clean post-deploy comparison (5-min DB query, do it tomorrow as a one-off, not part of this plan)
