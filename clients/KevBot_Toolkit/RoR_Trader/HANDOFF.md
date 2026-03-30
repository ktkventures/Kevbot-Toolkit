# Session Handoff: Strategy Builder Price Chart + Analysis Tabs

## Status
The Strategy Builder analysis tabs are working (TF Conditions, General, Stop, Target, Entry, Exit). The price chart upgrade is IN PROGRESS — it renders the multi-pane SyncedChartPane but has two remaining issues.

## Price Chart Issues to Fix Next

### 1. Too many indicators loaded as overlays
The backtest service dumps ALL float columns as overlay indicators. This includes vol_sma, trade_count, vwap bands, every EMA, etc. — many at wildly different scales (volume in millions vs price at ~635), which crushes the chart.

**Fix needed**: Port the EXACT indicator classification logic from the Strategy Detail chart-data endpoint (`GET /api/strategies/{id}/chart-data` in strategies.py lines 313-473). This endpoint:
- Determines which confluence groups the strategy uses
- Only returns indicator columns relevant to those groups
- Classifies as overlay vs oscillator based on template definitions
- The key function is around lines 370-470 where it uses `TEMPLATES` to classify

**DO NOT** just dump all float columns. Match the Strategy Detail approach.

### 2. Heatmap not showing
The heatmap conditions need `_state_` columns in the chart data AND the confluence condition matching. The current code tries to match but the `_state_` column naming may not align with what `buildStrategyChartPanes` expects.

**Check**: The Strategy Detail page builds heatmap from `heatmap_conditions` array where each entry has `column`, `label`, `needed_state`, `has_data`. The chart builder looks for `b[_state_{cond.column}]` in the bar data. Verify the column names match.

### 3. TF Conditions only showing same timeframe
The analyze endpoint returns conditions from `confluence_records` in trades. If no secondary timeframe data is loaded (the backtest doesn't include `secondary_tfs`), only 1M conditions appear.

**Fix**: The frontend needs to pass `secondary_tfs` in the backtest request, derived from the enabled TF Confluence Packs' timeframe settings. Port from Streamlit: `get_required_tfs_from_confluence()` in data_loader.py.

## What Works Well
- Entry/Exit/TF Conditions/General/Stop/Target analyze tabs with real KPIs
- Fast condition analysis using ported analyze_confluences() (single backtest + filtering)
- Equity curve (per-trade + per-day modes)
- Trade markers on price chart (arrows + R-labels)
- Filter/sort on analysis results
- All Replace/Add buttons update Optimizable Variables

## Key Files
- `/src/api/services/backtest_service.py` — indicator classification code to fix
- `/src/api/routers/strategies.py` lines 313-473 — the REFERENCE implementation to port
- `/frontend/src/charts/buildStrategyChartPanes.ts` — shared chart builder
- `/frontend/src/views/StrategyBuilderPage.tsx` — main page
- `/src/services.py` — analyze_confluences(), find_best_combinations()

## Status
The Strategy Builder analysis module (Entry/Exit/TF Conditions/General/Stop/Target tabs)
has been partially wired but has multiple issues. The next session should do a PROPER PORT
of the Streamlit analysis logic rather than rebuilding from scratch.

## What Works
- Entry tab analyze: shows per-trigger results (tested, works)
- Exit tab analyze: shows per-exit-trigger results (partially works)
- Stop Loss tab: shows results but may have issues with identical KPIs
- Take Profit tab: shows results but KPIs appear identical across packs
- Filter/sort modal UI exists but filters only applied to Entry tab

## What's Broken
1. TF Conditions tab: mock cards still show before/after analyze. Analyze button
   runs but results don't populate into the existing condition cards. The condition
   cards are hardcoded from API_CONFLUENCE_CONDITIONS, not replaced by analysis results.
2. General tab: analyze button fires but no results appear
3. Take Profit results all show identical KPIs (likely all using same stop/target config)
4. Replace button on Stop/Take Profit doesn't update Optimizable Variables display
5. Filters (min PF, min WR, sort) not applied to Exit/TF/General/Stop/Target tabs
6. Confluence Depth selector not wired (should control combination search depth)

## Key Streamlit Functions to Port
All in `/clients/KevBot_Toolkit/RoR_Trader/src/app.py`:

### analyze_confluences() — lines 1482-1544
- Takes trades_df, required set, min_trades threshold
- Runs ONE backtest with no confluence filtering
- Filters trades by confluence_records set membership per condition
- Returns DataFrame with per-condition KPIs
- THIS IS THE CORE FUNCTION that makes TF Conditions and General tabs fast

### apply_confluence_filters() — lines 2037-2074
- Filters results by text search, min_win_rate, min_profit_factor, min_daily_r, etc.
- Sorts by selected column
- THIS is what powers the filter/sort modal

### find_best_combinations() — lines referenced in auto-search
- Tests all 1/2/3/4-depth condition combinations
- Uses pre-computed numpy boolean masks for fast AND operations
- THIS is what Confluence Depth selector (1/2/3/4) should trigger

### TF Conditions tab UI flow — lines 5367-5498
- Has two modes: drill-down (individual conditions) and auto-search (combinations)
- Drill-down: analyze_confluences() → filter → display top 20
- Auto-search: find_best_combinations() → display top 50 combos

## Key Architecture Differences (Streamlit vs Next.js)
- Streamlit: all in Python, trades_df is a pandas DataFrame in memory
- Next.js: need API endpoint that returns JSON, frontend renders cards
- The backtest result already includes trades with confluence_records field
- The API should accept the current backtest trades + filter params, return per-condition KPIs
- OR: the analyze endpoint runs the base backtest internally (current approach)

## Files That Need Changes
### Backend
- `/src/api/routers/backtest.py` — rewrite _analyze_conditions_impl to properly
  handle confluence_records serialization (set→list in JSON), fix target analysis
  to actually vary the target config
- Consider adding analyze_confluences() as a proper service function in services.py

### Frontend
- `/frontend/src/views/StrategyBuilderPage.tsx` —
  - TF Conditions: replace hardcoded condition cards with analysis results
  - General: same treatment, filter for GEN- prefix conditions
  - Wire filter/sort to all tabs (currently only Entry)
  - Wire Replace buttons to update state + Optimizable Variables
  - Wire Confluence Depth selector to control combination search

## Important Notes
- DO NOT REBUILD — port the Streamlit logic directly
- The confluence_records field in trade records is a set in Python but serialized
  as a list in JSON. The backend filtering must handle both: isinstance(r, (set, list))
- The Optimizable Variables component receives stopPacks/targetPacks as props —
  the selected pack state needs to trigger a re-render
- The equity curve per-day mode is now working correctly (HWM data merge fix)
- All exec type badges (C/L/LC/CC) are working in triggers and trade history
