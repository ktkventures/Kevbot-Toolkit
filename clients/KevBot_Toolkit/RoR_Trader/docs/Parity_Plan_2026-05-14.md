# Parity Plan — RoR Trader Model Alignment (2026-05-14)

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

## Phase E — Flat-file tick replay tab (GATED)

**Status**: Pause and confirm with Kevin before starting. Skip entirely if Phase D lands ≥95% match.

**Goal**: recreate Kevin's Claude-app tick analysis within RoR Trader so the "Polygon ceiling = 99.1%" claim is demonstrable in the app, not in an external chat.

**Files (new)**:
- `src/services/flat_file_ingestion.py` — boto3 client for Polygon S3 (us_stocks_sip bucket), daily Railway cron
- Supabase migration: new `polygon_ticks` table partitioned by `trade_date`, indexed on `(ticker, sip_timestamp)`
- Railway cron entry: ~12:30 PM ET daily, ingests previous-session trades for symbols in active strategies
- `src/api/routers/admin_parity.py` — extend with `/api/admin/parity/ticks/{symbol}?date=&tf=` endpoint that rebuilds observable + settled bars from ticks
- `frontend/src/charts/ParityTickReplay.tsx` — four-pane comparison (observable / settled / cache / REST)

**Pre-requisites**:
- Polygon S3 credentials confirmed (Stocks Advanced tier includes Flat Files at no extra cost per the Claude-app analysis)
- Storage estimate: ~750GB compressed/year all-tickers; smaller if filtered to active universe
- Decide: ingest all tickers (simple, large) or filter to active strategy symbols (small, requires watchlist sync)

**Verification**: select sample day 2026-05-07 (the day Kevin's Claude-app analysis covered), symbol SPY, TF 10Sec. See observable vs settled bars with max 6¢ divergence on the highs, matching the Claude-app numbers exactly. Validates our ingestion + bar-construction logic.

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
