# RoR Trader — Charting & Data Conventions

## Critical Rule: Always Use Resampled 1-Minute Bars

When creating ANY price chart that shows a non-1-minute timeframe (daily, hourly, 15-min, etc.):

**NEVER load native bars from Polygon for coarser timeframes.**

Instead:
1. Load 1-minute bars via `load_market_data(symbol, days=N, timeframe='1Min')`
2. Resample to the target timeframe via `resample_to_timeframe(df, target_tf)`
3. Run indicators on the resampled bars

**Why:** Polygon's native daily/hourly bars have stock split adjustment issues (`adjusted=true` doesn't reliably work). Loading native daily bars for NVDA returned $4,700 prices (pre-10:1 split). Resampling from 1-minute bars avoids this because 1-min bars are always recent and post-split.

This matches how the Streamlit app, Chart & Trades heatmap, and unified trades engine all handle cross-timeframe data.

## Chart Data Pipeline (Proven Pattern)

```
1-min bars from Polygon → resample_to_timeframe() → run_all_indicators() → run_indicators_for_group() → run_all_interpreters()
```

For multi-timeframe confluence:
```
prepare_data_with_indicators(symbol, days, timeframe='1Min', secondary_tfs=('1Day', '15Min'))
```
This internally resamples 1-min bars to each secondary TF and creates suffixed columns like `MACD_LINE__1d`.

## Legacy Template Aliases

Old confluence groups reference `utbot` and `ema_price_position` (not the v2 versions). These legacy templates are registered in `confluence_groups.py` TEMPLATES dict. If you see "Warning: skipping group — template not found", a template alias is missing.

## React Hooks Rules (Next.js Production)

- ALL `useMemo`, `useState`, `useEffect` calls MUST come BEFORE any early returns (`if (loading) return`)
- Variables referenced in `useMemo` must be DECLARED BEFORE that `useMemo` in source order (Terser minifier reorders declarations)
- Never use IIFEs `(() => { ... })()` for variable initialization — use `useMemo` instead
- Use `swcMinify: false` in next.config.mjs (Terser instead of SWC)

## API Data Conventions

- API returns snake_case, frontend uses camelCase — always map
- Use plain `fetch()` with localStorage token for settings/auth to avoid circular dependency through `apiFetch`
- The `apiFetch` import chain can cause "Cannot access X before initialization" in production bundles
