# Phase 38 Execution Plan — Source of Truth

**Status:** ACTIVE
**Date:** 2026-03-25
**Branch:** `dev`
**Backup:** `dev-backup-pre-38-impl`

---

## Approach

Copy each locked V5/V6/V7/V1 design file exactly into `views/`, then surgically replace mock data:

| Data State | Display | Meaning |
|-----------|---------|---------|
| Real API data | Actual value | Wired and working |
| `--` | Double dash | Wired to API but data not available yet |
| `{{field_name}}` | Curly brackets | Not wired yet — needs backend or feature work |

**Rule:** No fake numbers that look like real data. Ever.

### Lesson Learned (Batch 1 QA)

V5 files have TWO layers of mock data:
1. **Top-level constants** (e.g., `const strategy = {...}`, `const mockTrades = [...]`) — easy to find and replace
2. **Inline mock data in JSX** (e.g., hardcoded prices `142.35`, dates `2026-03-18 09:42:15`, table rows with fake trades) — hidden throughout tab content, much harder to find

The copy-and-wire procedure MUST include:
1. `grep` audit for inline mock data AFTER replacing top-level constants. Patterns: specific prices (`14[0-9]\.`), specific dates (`2026-03-1[0-9]`), `mock` references, `Math.random`.
2. **React hooks order check**: verify ALL `useState`, `useMemo`, `useEffect`, `usePopover` calls come BEFORE any `if (isLoading) return` or `if (error) return` early exits. This has caused "Rendered more hooks" errors in Batch 1 AND Batch 2.

### Known Issues per Tab (Strategy Detail V5)

| Tab | Mock Data Status | What Needs Wiring |
|-----|-----------------|-------------------|
| Equity & KPIs | Top-level replaced, extended KPIs show 0 | Wire `useStrategyKPIs()` secondary_kpis to extendedKPIs |
| Chart & Trades | Trade tables wired but may show 0 FWD trades | Wire `useStrategyForwardTest()` properly; bar_count_exit = valid exit |
| Confluence Analysis | Entire tab is inline mock data (stateTimeline, triggerEvents) | Needs new API endpoint for per-group state/trigger history |
| Configuration | Mostly wired | bar_count_exit should display as exit trigger |
| Alerts | mockAlerts array + inline trade-to-alert mapping table | Wire `useStrategyAlerts()` to replace mockAlerts |
| Alert Analysis | mockAlertAnalysis with phantom trades, timing deltas, slippage | Needs new API endpoint for discrepancy/timing analysis |

---

## Phase A: Frontend Faithful Copy

### Batch 1: Core Flow
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A1.1 | `strategies/versions/V5.tsx` (885 lines) | `views/StrategiesPage.tsx` | **DONE** |
| A1.2 | `strategies/[id]/versions/V5.tsx` (2,177 lines) | `views/StrategyDetailPage.tsx` | **DONE** |
| | | **Kevin QA checkpoint** | |

### Batch 2: Dashboard + Alerts
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A2.1 | `dashboard/versions/V7.tsx` (2,492 lines) | `views/DashboardPage.tsx` | **DONE** |
| A2.2 | `alerts/versions/V5.tsx` (450 lines) | `views/AlertsPage.tsx` | **DONE** |
| A2.3 | `alerts/webhook-templates/versions/V5.tsx` (323 lines) | `views/WebhookTemplatesPage.tsx` | **DONE** |
| A2.4 | `alerts/webhook-templates/[id]/versions/V5.tsx` (518 lines) | `views/WebhookTemplateDetailPage.tsx` | **DONE** |
| | | **Kevin QA checkpoint** | |

### Batch 3: Confluence Packs
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A3.1 | `tf-confluence/versions/V5.tsx` (1,855 lines) | `views/TfConfluencePage.tsx` | **DONE** |
| A3.2 | `general/versions/V5.tsx` (813 lines) | `views/GeneralPacksPage.tsx` | **DONE** |
| A3.3 | `user-packs/versions/V5.tsx` (1,743 lines) | `views/UserPacksPage.tsx` | **DONE** |
| A3.4 | `pack-builder/versions/V7.tsx` (1,047 lines) | `views/PackBuilderPage.tsx` **NEW** | **DONE** |
| A3.5 | `stop-loss/versions/V1.tsx` (541 lines) | `views/StopLossPage.tsx` | **DONE** |
| A3.6 | `take-profit/versions/V1.tsx` (515 lines) | `views/TakeProfitPage.tsx` | **DONE** |
| A3.7 | `timeframes/versions/V5.tsx` (192 lines) | `views/TimeframesPage.tsx` | **DONE** |
| | | **Kevin QA checkpoint** | |

### Batch 4: Portfolios
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A4.1 | `portfolios/versions/V5.tsx` (688 lines) | `views/PortfoliosPage.tsx` | **DONE** |
| A4.2 | `portfolios/[id]/versions/V5.tsx` (1,820 lines) | `views/PortfolioDetailPage.tsx` | **DONE** |
| A4.3 | `portfolios/new/versions/V5.tsx` (1,158 lines) | `views/PortfolioNewPage.tsx` **NEW** | **DONE** |
| A4.4 | `portfolio-requirements/versions/V5.tsx` (685 lines) | `views/RequirementsPage.tsx` | **DONE** |
| | | **Kevin QA checkpoint** | |

### Batch 5: Mass Builder + Settings
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A5.1 | `mass-builder/versions/V6.tsx` (1,386 lines) | `views/MassBuilderPage.tsx` | **DONE** |
| A5.2 | `mass-results/versions/V5.tsx` (241 lines) | `views/MassResultsPage.tsx` | **DONE** |
| A5.3 | `settings/display/versions/V5.tsx` (1,353 lines) | `views/SettingsDisplayPage.tsx` | **DONE** |
| A5.4 | `settings/connections/versions/V5.tsx` (426 lines) | `views/SettingsConnectionsPage.tsx` | **DONE** |
| A5.5 | `settings/account/versions/V4.tsx` (276 lines) | `views/SettingsAccountPage.tsx` **NEW** | **DONE** |
| A5.6 | `settings/profile/versions/V4.tsx` (215 lines) | `views/SettingsProfilePage.tsx` **NEW** | **DONE** |
| | | **Kevin QA checkpoint** | |

### Strategy Builder
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A0.1 | `strategy-builder/versions/V5.tsx` (2,494 lines) | `views/StrategyBuilderPage.tsx` | **DONE** |

---

## Phase B: Backend Alignment

### B1: Stop Loss / Take Profit Separation
| Task | Description | Status |
|------|------------|--------|
| B1.1 | `GET /api/packs/stop-loss` — filtered RM packs | **DONE** |
| B1.2 | `GET /api/packs/stop-loss/templates` — stop-only templates | **DONE** |
| B1.3 | `PUT /api/packs/stop-loss` — save via RM pipeline | **DONE** |
| B1.4 | Repeat for take-profit | **DONE** |
| B1.5 | `useStopLossPacks()` + `useTakeProfitPacks()` hooks | **DONE** |

### B2: Execution Type Parameters
| Task | Description | Status |
|------|------------|--------|
| B2.1 | Extend TEMPLATES with exec variant schemas | **DONE** |
| B2.2 | Update templates API serialization | **DONE** |
| B2.3 | Update `ConfluenceTemplateDTO` TypeScript type | **DONE** |

### B3: Fidelity Type [PB]/[CB]
| Task | Description | Status |
|------|------------|--------|
| B3.1 | Add fidelity_type to confluence condition records | **DONE** |
| B3.2 | Default existing conditions to [PB] | **DONE** |

### B4: Dashboard Endpoints
| Task | Description | Status |
|------|------------|--------|
| B4.1 | `GET /api/dashboard/equity-curve` | **DONE** |
| B4.2 | `GET /api/dashboard/daily-pnl` | **DONE** |
| B4.3 | `GET /api/dashboard/positions` | **DONE** |
| B4.4 | `GET /api/dashboard/health` | **DONE** |
| B4.5 | `GET /api/dashboard/activity` | **DONE** |

### B5: Missing Strategy Fields
| Task | Description | Status |
|------|------------|--------|
| B5.1 | Forward test + alert KPIs on strategy endpoint | **DONE** |
| B5.2 | Sigma deviation computation | **DONE** |
| B5.3 | Strategy status derivation | **DONE** |
| B5.4 | Equity curve data in strategy list | **DONE** |

---

## Phase A QA Notes (2026-03-26) — RESOLVED

- [x] Stop Loss detail Preview tab: replaced with empty state
- [x] Take Profit detail Preview tab: replaced with empty state
- [x] Portfolio Detail Anomaly Detection: replaced with empty state
- [x] Portfolio Detail Correlation Heatmap: replaced with empty state
- [x] Exit triggers: fixed `get_exit_triggers()` to include entry-type triggers
- [x] User Packs: "Create from scratch" → links to Pack Builder
- [ ] X-axis toggle (Date vs Trade #) — deferred to Phase C

---

## Phase C: Feature Wiring

Replace `{{field_name}}` with real computed data. Working through these on Railway dev.

### C1: Strategy Analytics
| Task | Description | Status |
|------|------------|--------|
| C1.1 | Forward KPIs on strategy list + detail | **DONE** — enrich_strategy + full_compute |
| C1.2 | Sigma deviation badges | **DONE** — fixed to compare avg_r vs backtest distribution |
| C1.3 | Strategy status (On Track/Outperforming/etc.) | **DONE** — derived from sigma |
| C1.4 | Alerts always-on for forward testing strategies | **DONE** — enrich_strategy sets flag |
| C1.5 | Update Data button (list page — bulk refresh) | **DONE** — sequential refresh with progress |
| C1.6 | Update Data button (detail page — single refresh) | **DONE** — POST /api/strategies/{id}/refresh |
| C1.7 | Loading indicators for long computations | **DONE** — spinner banners + table loaders |
| C1.8 | Forward test trades in Chart & Trades tab | **DONE** — wired to useStrategyForwardTest |
| C1.9 | Extended KPIs (Sharpe, Sortino, etc.) in detail | Not Started — useStrategyKPIs hook exists, needs frontend wiring |
| C1.10 | Analysis tab computations (per-trigger comparison) | Not Started — needs new API endpoint |
| C1.11 | Rolling metrics chart | Not Started — needs analytics.py integration |
| C1.12 | Return distribution (histogram + stats) | Not Started — R values available, needs charting |
| C1.13 | Markov Motor (transition matrix) | Not Started — needs analytics.py integration |
| C1.14 | Equity curve x-axis toggle (Date vs Trade #) | Not Started |
| C1.15 | **Real equity sparklines on My Strategies cards** | Not Started — equity_curve_data in API, need to replace V5 fake MiniEquityCurve with SparkLine reading real data |
| C1.16 | **Real equity curve on Strategy Detail (3-segment)** | Not Started — EquityCurve component exists, need to wire to equity_curve_data with boundary_index |
| C1.17 | **Live price chart on Strategy Detail (Chart & Trades tab)** | Not Started — TradingChart component exists, need data endpoint (OHLCV + indicators + trade markers) |
| C1.18 | **Price chart on Strategy Builder (results panel)** | **DONE** — SyncedChartPane with overlays, oscillators, heatmap, entry/exit markers, C-type timestamp shift |
| C1.19 | **Settings Display → chart preferences applied globally** | Not Started — candle style, colors, grid, visible candles from useDisplayStore |
| C1.20 | **Strategy Builder: All 6 analysis tabs wired** | **DONE** — Entry, Exit, TF Conditions, General, Stop Loss, Take Profit with real backtest KPIs |
| C1.21 | **Strategy Builder: Confluence depth combinations** | **DONE** — Depth 2+ via find_best_combinations() for TF Conditions (exclude GEN-) and General (include GEN-) |
| C1.22 | **Strategy Builder: Equity curve (per-trade + per-day)** | **DONE** — HWM line merged into chartData, per-day grouping by YYYY-MM-DD |
| C1.23 | **Strategy Builder: Filter/sort on analysis results** | **DONE** — min PF, min WR, sort column applied to all tabs |
| C1.24 | **Strategy Builder: Replace/Add buttons update Optimizable Variables** | **DONE** — all tabs wire selected component back to strategy config |
| C1.25 | Strategy Builder: Stop/Target pack parameter separation | Not Started — UI shows mixed params from RM packs |
| C1.26 | Strategy Builder: Price chart show/hide toggles | Not Started — toggle individual conditions/triggers on chart |
| C1.27 | Strategy Builder: TF Conditions fidelity badge (PB/CB) | Not Started — currently shows C/L exec type |
| C1.28 | Strategy Builder: Progress tracking with scenario counts | Not Started — needs SSE for real progress (currently indeterminate bar) |

### C2: Dashboard Widgets
| Task | Description | Status |
|------|------------|--------|
| C2.1 | KPI strip from API | **DONE** — useDashboardSummary wired |
| C2.2 | Strategy health widget | **DONE** — /api/dashboard/health endpoint |
| C2.3 | Active positions from engine state | **DONE** — /api/dashboard/positions endpoint |
| C2.4 | Activity feed from alerts | **DONE** — /api/dashboard/activity endpoint |
| C2.5 | Portfolio equity curve chart | Not Started — needs aggregated equity data |
| C2.6 | Daily P&L bar chart | Not Started — needs daily P&L computation |
| C2.7 | P&L Calendar heatmap | Not Started — needs daily P&L data |
| C2.8 | Market regime indicator | Not Started — needs VIX/breadth data source |
| C2.9 | Monthly goal progress | Not Started — needs settings-driven goal config |

### C3: Portfolio Analytics
| Task | Description | Status |
|------|------------|--------|
| C3.1 | Portfolio equity curve | Not Started — compute endpoint exists |
| C3.2 | Strategy correlation heatmap | Not Started — compute endpoint exists |
| C3.3 | Monte Carlo simulation display | Not Started — compute endpoint exists |
| C3.4 | Buying power tracker | Not Started — account endpoint exists |
| C3.5 | Anomaly detection display | Not Started — anomalies endpoint exists |
| C3.6 | Portfolio "Re-analyze" button | Not Started — needs recommendation endpoint |
| C3.7 | Worst-case analysis | Not Started — needs computation |

### C4: Pack Configuration
| Task | Description | Status |
|------|------------|--------|
| C4.1 | TF Confluence pack parameter editing + save | Not Started — save endpoint exists |
| C4.2 | Trigger exec type parameter tabs (Trigger Parameters tab) | Not Started — exec_variants in TEMPLATES |
| C4.3 | General pack parameter editing + save | Not Started — save endpoint exists |
| C4.4 | Stop Loss / Take Profit pack CRUD | Not Started — dedicated endpoints exist |
| C4.5 | Create Variation flow | Not Started |
| C4.6 | Pack Builder AI wizard steps | Not Started — future feature (needs AI API) |

### C5: Confluence Pack Modernization
| Task | Description | Status |
|------|------------|--------|
| C5.1 | Duplicate current packs, tag originals as "legacy (Default)" | Not Started |
| C5.2 | Create new default packs with [PB]/[CB] fidelity options | Not Started |
| C5.3 | Create new default packs with [C]/[L]/[LC]/[CC] exec variants | Not Started |
| C5.4 | Create Take Profit pack templates (currently only Stop Loss exist) | Not Started |
| C5.5 | Entry/Exit trigger parity — ensure full trigger sets on both | **DONE** — get_exit_triggers fixed |

---

## Phase D: Future Pages + QA + Polish

### D1: Secondary Pages (no backend yet — all `{{field}}`)
| Task | Description | Status |
|------|------------|--------|
| D1.1 | Admin Dashboard, Users, Curation (V4 copies) | Not Started |
| D1.2 | Creator Dashboard, Earnings, Publish (V4 copies) | Not Started |
| D1.3 | Marketplace, Prop Firms, Subscriptions (V4 copies) | Not Started |
| D1.4 | Pricing page (V4 copy) | Not Started |
| D1.5 | Onboarding flow | Not Started |

### D2: Railway Deployment + Production Readiness
| Task | Description | Status |
|------|------------|--------|
| D2.1 | API service on Railway dev | **DONE** |
| D2.2 | Frontend service on Railway dev | **DONE** |
| D2.3 | CORS + env vars configured | **DONE** |
| D2.4 | Deploy to Railway main (production) | Not Started |
| D2.5 | DNS setup (rortrader.com → frontend) | Not Started |

### D3: Visual QA
| Task | Description | Status |
|------|------------|--------|
| D3.1 | Kevin full pass: toggle PageSwitch on every page | Not Started |
| D3.2 | Log discrepancies per page | Not Started |
| D3.3 | Fix all flagged discrepancies | Not Started |

### D4: Polish
| Task | Description | Status |
|------|------------|--------|
| D4.1 | Responsive breakpoints (sidebar collapse, card stacking) | Not Started |
| D4.2 | Loading skeleton consistency | In Progress — added to Strategy Detail |
| D4.3 | Error state consistency (retry buttons, helpful messages) | Not Started |
| D4.4 | Accessibility (aria labels, keyboard navigation) | Not Started |

---

## Known Issues (2026-03-26)

| Issue | Page | Description | Likely Cause |
|-------|------|-------------|-------------|
| Client-side exception on My Strategies | `/strategies` | "Application error: a client-side exception has occurred" when logged in as `kevin-migrate@rortrader.dev`. Does NOT occur for `VandosJohnson@gmail.com` account. | Likely a data shape difference — the migrate account may have strategies with fields that cause the V5 component to crash (null value where .toFixed() is called, or array field that's actually a string). The migrate account has 26 strategies; some may have edge-case data shapes from bulk creation. Debug: check browser console for the specific error, identify which strategy/field causes it. |
| Production bundle circular dependency | `/strategies` | Importing `apiFetch` from `@/lib/api/client` at module scope causes "Cannot access S before initialization" in production builds. | Workaround: use plain `fetch()` with localStorage token instead. All page.tsx files use `next/dynamic` with `ssr: false`. |
| Forward test computation slow | `/strategies/[id]` | Loading forward test data takes 10-30 seconds for 1-minute timeframe strategies | Expected — Polygon REST API + full backtest computation. Add caching/stored_trades refresh to mitigate. |

---

## Key Architectural Decisions

1. **Stop/Target:** Adapter endpoints over existing RM module — no DB migration
2. **Exec types:** Schema in TEMPLATES registry, stored in parameters JSONB
3. **Fidelity:** Derived from cross-TF shift logic — display only
4. **Dashboard:** New aggregation endpoints reusing existing modules
5. **views/ = live code, versions/ = frozen design reference**
