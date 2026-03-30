# Session Handoff: Strategy Builder Analysis Module Port

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
