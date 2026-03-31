# Session Handoff: Strategy Builder — Phase C Feature Wiring

## Status: Analysis Tabs Fully Wired + Price Chart Upgraded
All 6 analysis tabs are working with real backtest data. Confluence Depth combinations (depth 2+) work for both TF Conditions and General tabs. Price chart uses SyncedChartPane with indicators, oscillators, heatmap, and trade markers.

*Last Updated: March 31, 2026*

---

## What's Working

### Analysis Tabs (All 6)
- **Entry**: Per-trigger analysis with exec type badges (C/LC/CC/L), Replace/Add buttons
- **Exit**: Per-exit-trigger analysis, Add button wires selected exit
- **TF Conditions**: Fast single-backtest + confluence_records filtering via `analyze_confluences()`. Depth 1 = individual conditions, Depth 2+ = `find_best_combinations()` combination search
- **General**: Same as TF Conditions but filtered to `GEN-` prefix conditions. Depth 2+ uses `include_prefix` passed directly to `find_best_combinations()`
- **Stop Loss**: Tests each enabled RM pack's stop config
- **Take Profit**: Tests each enabled RM pack's target config
- **Filter/Sort modal**: Applied to all tabs (min PF, min WR, sort column)
- **Confluence Depth selector**: Pinned at bottom of analysis card, depth 1-4 for TF/General

### Price Chart (SyncedChartPane)
- Multi-pane layout: heatmap + price/overlays + oscillator panes
- Indicator classification ported from Strategy Detail (OVERLAY_TEMPLATES / OSCILLATOR_TEMPLATES)
- Entry arrows + exit cross markers with R-value labels
- C-type timestamp shift (candle-close entries plotted on next bar open)
- Heatmap conditions built from confluence_records

### Equity Curve
- Per-trade and per-day modes
- HWM line merged into chartData (fixes Recharts domain doubling bug)

### Other
- Backtest progress bar (indeterminate during analysis)
- Replace/Add buttons update Optimizable Variables for all tabs
- Secondary TFs always loaded (5M, 15M, 1H, 1D) for cross-TF analysis
- inf/nan sanitization via `_safe_float()` / `_sf()` in all KPI serialization

---

## Bugs Fixed This Session

1. **SyntaxError crash loop** (API down): `_analyze_combinations_impl` had `try:` block with `except` placed after `return` — fixed indentation
2. **General depth 2+ empty results**: `find_best_combinations()` processed ALL conditions, returned top 100 (all non-GEN), then post-filter eliminated everything. Fixed by adding `include_prefix` param to filter conditions before building masks
3. **Stale depth results**: Switching depth 2→1 still showed old combinations. Fixed by clearing sibling result key (`combinations`↔`condition`, `general_combinations`↔`general`) on new analysis
4. **Stale closure for depth**: `tfDepth`/`generalDepth` missing from `handleAnalyze` useCallback deps — sent old depth value to API
5. **Spinner not showing for combinations mode**: Loading check only matched individual mode names, not combinations variants
6. **Dead code after return**: Unreachable `_load_analyze_data` body copy after `_analyze_targets_impl`

---

## Remaining Work

### Priority 1: Polish & UX
- [ ] Stop/Take Profit pack parameter separation (UI shows mixed params from RM packs)
- [ ] Price chart show/hide toggles for individual conditions and triggers
- [ ] TF Conditions fidelity badge (show PB/CB instead of C/L exec type)
- [ ] Strategy Builder defaults/settings page

### Priority 2: Performance
- [ ] Progress tracking with actual scenario counts (needs SSE — currently indeterminate)
- [ ] Consider caching base backtest trades for rapid re-analysis at different depths

### Priority 3: Cleanup
- [ ] Legacy pack cleanup (old/duplicate RM packs)
- [ ] Remove debug logging from analyze endpoints (once stable)
- [ ] Audit unused imports in backtest.py (`traceback` import)

---

## Key Files

### Backend
| File | Purpose |
|------|---------|
| `src/api/routers/backtest.py` | `/analyze` endpoint — routes to mode-specific impl functions |
| `src/services.py` | `analyze_confluences()`, `find_best_combinations()` — ported from Streamlit |
| `src/api/services/backtest_service.py` | Backtest runner + indicator classification for chart data |
| `src/confluence_groups.py` | `get_enabled_groups()`, `get_entry_triggers()`, `get_exit_triggers()` |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/views/StrategyBuilderPage.tsx` | Main page — all 6 tabs, depth selector, handleAnalyze |
| `frontend/src/charts/buildStrategyChartPanes.ts` | Shared chart builder (heatmap, overlays, oscillators) |
| `frontend/src/charts/EquityCurve.tsx` | Per-trade + per-day equity curve with HWM |
| `frontend/src/hooks/queries/useBacktest.ts` | `useRunBacktest`, `useAnalyzeTriggers` mutations |

---

## Architecture Notes

### Analysis Flow (TF Conditions / General)
```
1. Run ONE backtest with NO confluence gating → get all possible trades
2. analyze_confluences(): filter trades by confluence_records set membership → per-condition KPIs
3. find_best_combinations(): pre-computed numpy boolean masks → AND combinations at depth 2+
```

### Confluence Depth Modes
- Depth 1 → `condition` or `general` mode (individual conditions via `analyze_confluences`)
- Depth 2+ → `combinations` or `general_combinations` mode (via `find_best_combinations`)
- TF Conditions: `exclude_prefix='GEN-'` (skip general conditions)
- General: `include_prefix='GEN-'` (only general conditions)

### Key Conventions
- `response_model=BacktestResponse` was removed from `/run` endpoint to allow extra fields (chart indicators)
- Results stored under mode-specific keys in `analysisResults` state — sibling keys cleared on new analysis
- `_safe_float()` / `_sf()` sanitizes inf/nan before JSON serialization (profit_factor=inf when 0 losses)
