# QA Agent Briefing — RoR Trader Frontend

**Purpose:** This document trains a QA/dispatch agent to verify the RoR Trader Next.js frontend during Phase 38 migration.

---

## 1. Application Overview

RoR Trader is a trading strategy platform migrating from Streamlit to Next.js + FastAPI. The user builds trading strategies, backtests them against market data, monitors them with live alerts, and manages portfolios of strategies.

**Stack:**
- Frontend: Next.js 14 (React 18, TypeScript, Tailwind CSS)
- Backend API: FastAPI (Python) — wraps existing computation engine
- Database: Supabase (PostgreSQL with Row Level Security)
- Market Data: Polygon.io (REST + WebSocket)
- Alert Engine: Ralph (Python background worker on Railway)
- Charts: lightweight-charts (candlestick) + recharts (equity curves)

**Deployment:**
- Railway dev environment with 4 services: frontend, api, worker, streamlit (legacy)
- Frontend URL: `https://frontend-dev-e01a.up.railway.app`
- API URL: `https://api-dev-2c9d.up.railway.app`
- API health check: `https://api-dev-2c9d.up.railway.app/health`
- API docs: `https://api-dev-2c9d.up.railway.app/api/docs`

---

## 2. Data State Convention

Every piece of data on every page falls into one of three states:

| Display | Meaning | Example |
|---------|---------|---------|
| **Real value** | Live data from the API | `54.2%`, `SPY`, `2.14` |
| **`--`** | Wired to API but data not available yet | Waiting for backtest, worker not running |
| **`{{field_name}}`** | Not wired yet — needs backend or feature work | `{{daily_roi}}`, `{{market_regime}}` |

**Rule:** If you see a specific number (like `58.3%` or `142.35`) that looks like it could be real data but seems suspicious (same value across multiple strategies, or a value that doesn't match the strategy context), it's likely leftover mock data that wasn't cleaned. Flag it.

---

## 3. Pages and What to Check

### Core Trading (highest priority)

**My Strategies** (`/strategies`)
- Strategy cards should show real data from Supabase (strategy names, symbols, KPIs)
- Forward KPIs (fwdWinRate, fwdPF) should show values for strategies with forward_test_start
- Sigma badges should show reasonable numbers (typically -3 to +3 range, not -24.5)
- Status should be "On Track", "Outperforming", "Underperforming", or "Insufficient Data"
- Equity sparklines: currently fake SVG — will be real after C1.15
- Update Data button should show progress ("Refreshing 1/4...")
- Filters (ticker, direction, tag, sort) should work
- Bulk actions (checkbox, delete selected) should work

**Strategy Detail** (`/strategies/[id]`)
- 6 tabs: Equity & KPIs, Chart & Trades, Confluence Analysis, Configuration, Alerts, Alert Analysis
- Entry/exit/stop/target/confluence variables shown above tabs with exec badges
- Forward test trades table should populate (may take 10-30 seconds — loading spinner shown)
- Backtest trades table should show stored_trades
- Extended KPIs: currently showing 0 (needs C1.9 wiring)
- Price chart: currently placeholder (needs C1.17)
- Update Data button in header should trigger refresh with loading indicator
- Sigma badges in header area

**Strategy Builder** (`/strategy-builder`)
- Entry/exit triggers populated from API (confluence groups)
- Stop loss/take profit pack selectors populated from API
- Run Backtest button calls API, shows KPIs and trade count
- KPIs show 0 before backtest, real values after
- Analysis tabs show empty until wired (C1.10)
- Price chart: placeholder (needs C1.18)

**Dashboard** (`/dashboard`)
- KPI strip shows real counts (strategies, portfolios, monitored, trades, total R)
- Health widget shows real strategy names with deviation bars
- Positions widget: empty unless worker has open positions
- Activity feed: empty unless worker has generated alerts
- Equity curve, P&L, calendar: placeholder (needs C2.5-C2.7)

### Confluence Packs

**TF Confluence** (`/confluence-packs/tf-confluence`)
- Real groups from Supabase with enable/disable toggles
- Nested variations under defaults
- Exec type badges [C]/[L]/[LC]/[CC]
- Detail view with 7 tabs

**General Packs** (`/confluence-packs/general`)
- Real packs with parameter summaries
- 4 templates: Time of Day, Trading Session, Day of Week, Calendar

**Stop Loss** (`/confluence-packs/stop-loss`)
- Filtered from risk management packs
- Cards show effective formula

**Take Profit** (`/confluence-packs/take-profit`)
- May be empty if no target packs configured

**Timeframes** (`/confluence-packs/timeframes`)
- Grid with checkboxes per use case
- Save button persists to settings

### Portfolios

**My Portfolios** (`/portfolios`)
- Real portfolios from Supabase
- Strategy pills on cards

**Portfolio Detail** (`/portfolios/[id]`)
- 6 tabs: Live Dashboard, Performance, Strategies, Prop Firm Check, Account, Webhooks
- Most data depends on worker running (positions, anomalies, P&L)
- Strategy list should show real strategies in the portfolio

**Requirements** (`/portfolio-requirements`)
- Requirement sets from API
- Expandable rule cards

### Alerts

**Alerts & Signals** (`/alerts`)
- Engine status from monitor API
- Start/stop buttons for engine control
- Alert feed from worker (empty if worker not monitoring these strategies)

**Webhook Templates** (`/alerts/webhook-templates`)
- Template cards from API
- Click through to detail page

### Mass Operations

**Mass Builder** (`/mass-builder`)
- Config form with triggers/packs from API
- Run button calls async search

**Mass Results** (`/mass-results`)
- Saved searches from API

### Settings

**Display** (`/settings/display`) — 5 tabs for chart/formatting preferences
**Connections** (`/settings/connections`) — Engine status, service health
**Themes** (`/settings/themes`) — Theme switcher (fully functional)
**Account** (`/settings/account`) — User info from auth
**Profile** (`/settings/profile`) — Role system placeholders

---

## 4. Common Issues to Watch For

### Production Build Errors
- `Cannot access 'X' before initialization` — circular dependency in production bundle. Usually caused by importing `apiFetch` from `@/lib/api/client` at module scope. Fix: use plain `fetch()` or dynamic import.
- White screen after deploy — stale `.next` cache. Fix: Railway redeployment clears it automatically.

### React Hooks Ordering
- `Rendered more hooks than during the previous render` — a `useState`/`useMemo`/`useEffect` is placed after an `if (isLoading) return` early exit. All hooks MUST come before any early returns.

### Data Shape Mismatches
- API returns snake_case (`win_rate`, `entry_price`), V5 components expect camelCase (`winRate`, `entryPrice`). Each view has a mapper function (e.g., `apiToStrategy()`).
- If a field shows `undefined` or causes `.toFixed is not a function`, the mapper is missing that field.

### Empty States
- "No alerts available — enable monitoring to populate" — expected when worker doesn't monitor this strategy
- "No forward test trades" — expected if stored_trades hasn't been refreshed (click Update Data)
- Empty equity curves — expected until Update Data refreshes equity_curve_data

---

## 5. Worker Dependency

The Ralph engine (worker) generates:
- **Alerts** — entry/exit signals detected from live market data
- **Engine state** — open positions, symbols tracked
- **Monitor status** — running/stopped, heartbeat, ticks

Without the worker running and monitoring specific strategies, these features show empty states. The worker runs on Railway and monitors strategies where `alert_tracking_enabled=true`.

Currently monitored: strategies 56, 57 (older strategies). Strategies 68-72 (newer) have `alert_tracking_enabled=true` in the enriched response but the worker may not have picked them up yet.

---

## 6. Execution Plan Reference

Full task breakdown: `docs/Phase_38_Execution_Plan.md`

Current status:
- **Phase A** (Frontend Faithful Copy): DONE — 28 pages
- **Phase B** (Backend Alignment): DONE — stop/target, exec types, fidelity, dashboard, strategy enrichment
- **Phase C** (Feature Wiring): IN PROGRESS — charts, analytics, portfolio, pack editing
- **Phase D** (Future Pages + QA + Polish): NOT STARTED

---

## 7. How to Report Issues

For each issue found, report:
1. **Page URL** where the issue occurs
2. **What you see** (screenshot or text description)
3. **What you expected** (based on Design Reference or data convention)
4. **Severity:** Error (page broken), Data (wrong/mock data showing), Visual (styling mismatch), Missing (feature not present)
