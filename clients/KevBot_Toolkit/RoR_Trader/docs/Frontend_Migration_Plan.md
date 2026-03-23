# Frontend Migration Plan — Design Iteration & Implementation Order

## Context
We have 33 pages with 138 version files (V1-V4 each). This document defines the order to iterate on page designs, lock in final versions, and wire up backend logic — organized by business priority.

## Design Iteration Process (Per Page)
1. Review all 4 versions in browser
2. Identify elements to keep/discard from each version
3. Create final "locked" version combining the best elements
4. Wire up FastAPI backend endpoints
5. Replace mock data with real API calls
6. Test with real data

---

## Design Review Log

### Settings/Themes — LOCKED: V1
- **Decision:** V1 as-is, no changes needed.
- **Notes:** ThemeSwitcher component already works well. Simple and functional.

### Settings/Display — LOCKED: V5 (Tabbed)
- **Decision:** V5 with 5 tabs: Price Charts, Equity Curves, Formatting, Components, Tables.
- **Key decisions made during iteration:**
  - **Price Charts tab:** Chart library selector (Lightweight vs ECharts), candle style as visual grid (not dropdown), 6 candle color themes including "Theme" (inherits from app theme), visible candles slider, right offset slider, grid lines toggle, chart pane ordering (drag-to-reorder: Confluence Heatmap / Price Chart / Oscillators), position markers (arrow shapes, price-level marks, exit reason colors, labels toggle). Price chart preview includes EMA 9 overlay and position markers.
  - **Equity Curves tab:** 3-segment model — Backtest (blue) → Forward Test (orange) → Live Alerts (green, no gradient fill). Live line overlays forward test to show slippage. Line style (solid/smooth/stepped), x-axis (time vs trade #), gradient fill (BT/FW only), zero line, high water mark, edge check (21-MA + Bollinger), confidence bands (portfolio context).
  - **Formatting tab:** Timezone, date format, currency symbol, number format with live preview.
  - **Components tab:** Simplified tag system — badge shape (rounded/square), brackets (on/off), two colors only (one for execution types, one for fidelity types). Execution types use 31F naming: [C] Close, [L] Level, [LC] Level-Close, [CC] Close-Close. Fidelity types: [PB] Previous Bar, [CB] Current Bar. Pack variation names: parenthetical/badge/separator styles.
  - **Tables tab:** Density (compact/comfortable/spacious) with trade history preview.
  - **All left-side cards are collapsible** to reduce scrolling while editing.
  - **Right side always shows contextual live preview** matching the active tab.

### Settings/Connections — LOCKED: V4
- **Decision:** V4 (Creative) — network topology diagram with clickable nodes, latency indicators, connection history timeline, auto-diagnosis terminal.
- **Notes:** Locked as-is, no changes needed.

### Dashboard — LOCKED: V6 (Refined Cockpit)
- **Decision:** V6 refined layout, iterated from V5. V7 preserved as reference for detail modal concepts.
- **Layout:** 7/5 grid split. Left column: Equity Curve → Daily P&L → P&L Calendar. Right column: Active Positions → Performance Health → Market Regime → Monthly Goal.
- **Header row:** Dashboard title, portfolio multi-select filter, date range dropdown (7d/14d/30d/90d/MTD/YTD/All), system status indicator, quick action icon buttons (compact 32x32 with tooltips), Customize button.
- **Bottom row (full width):** Issues & Warnings + Recent Activity (50/50 split).
- **Customize modal (3 tabs):** KPIs (checkbox list), Widgets (toggle switches), Quick Actions (checkbox list with icons).
- **Performance Health widget:** Portfolios/Strategies tab toggle, per-item SD deviation bar visualization (green <1 SD, orange 1-2 SD, red >2 SD), trade count, exact SD value. Based on `classify_strategy_health()` logic.
- **Active Positions:** Match status badges (Matched/Anomaly), Close Early button, glowing border animation.
- **Not included:** Strategy leaderboard, KPI pencil icon (removed — KPI customization handled in Customize modal).
- **V7 reference:** Detail modals on every widget (equity breakdown, P&L analysis, position context, health deep-dive, market analysis, goal tracking, activity log). Preserved for implementation reference when wiring up backends.

### Settings/Account — LOCKED: V2
- **Decision:** V2 (Full Parity) — profile info, subscription tier, usage stats, API usage, password change, 2FA toggle, data export, sign out.
- **Notes:** Locked as-is, no changes needed.

---

## Phase 1: Personal Trading (I Can Use It Myself)
**Goal:** Replace Streamlit with Next.js for daily personal use. All existing features work.
**Prerequisite:** FastAPI backend with Supabase auth + core CRUD endpoints.

### Phase 1A: Foundation + Core Loop
*The minimum to build a strategy and view it. This is the inner loop you use every day.*

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 1 | **Settings (all 4)** | Lock in early — themes, display, connections, account affect everything | Auth endpoints, settings CRUD |
| 2 | **Dashboard** | Landing page, sets the tone for the whole app | Strategy/portfolio summary endpoints |
| 3 | **TF Confluence** | Must work before Strategy Builder — defines available indicators | Confluence groups CRUD, TEMPLATES registry |
| 4 | **General Packs** | Time/day/session conditions needed for confluence | General packs CRUD |
| 5 | **Risk Management** | Stop/target packs needed for strategy config | Risk packs CRUD |
| 6 | **Strategy Builder** | Most complex, most used page. Heart of the app. | Backtest engine endpoint, data loading, indicator computation |
| 7 | **My Strategies (list)** | View saved strategies, manage forward testing | Strategy CRUD, stored trades |
| 8 | **Strategy Detail** | Deep-dive into strategy performance | Trade history, KPI computation, chart data |

**Milestone:** Can build, save, and review strategies in Next.js.

### Phase 1B: Portfolio & Alerts
*Extend to full trading workflow — portfolios, compliance, and live alerts.*

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 9 | **Portfolio List** | View and manage portfolios | Portfolio CRUD |
| 10 | **Portfolio New/Edit** | Build portfolios from strategies | Portfolio builder, equity computation, recommendations |
| 11 | **Portfolio Detail** | All 7 tabs — most complex detail page | Combined analytics, prop firm check, account ledger, deploy |
| 12 | **Portfolio Requirements** | Prop firm rules engine | Requirement set CRUD, compliance check |
| 13 | **Alerts & Signals** | Monitor control, alert feed, position tracking | Ralph engine status, alert history, monitor control |
| 14 | **Webhook Templates** | Webhook payload management | Template CRUD, test delivery |

**Milestone:** Full trading workflow works — build strategies, assemble portfolios, receive alerts, trade via webhooks.

### Phase 1C: Power User Tools
*Mass operations and remaining pack management.*

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 15 | **Mass Builder** | Bulk strategy discovery | Mass backtest endpoint (compute-heavy) |
| 16 | **Mass Results** | Browse and save mass results | Mass results CRUD |
| 17 | **Timeframes** | TF configuration | Timeframe settings CRUD |
| 18 | **User Packs** | Custom pack management | User pack CRUD |
| 19 | **Pack Builder** | Build custom indicators | Pack definition CRUD, indicator validation |

**Milestone:** Feature parity with Streamlit. Can fully replace it for personal use.

---

## Phase 2: AI Agent Integration
**Goal:** AI agents (Claude) can create strategies, build portfolios, and curate the marketplace — operating as autonomous creators.
**Depends on:** Phase 1 complete (all trading logic wired up).

### Phase 2A: API-First for Agents

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 20 | **Admin Dashboard** | Platform metrics, overview of what agents are doing | Platform stats endpoints, agent activity logs |
| 21 | **Admin Curation** | Curate free-rotation portfolios (agents can recommend, admin approves) | Content curation CRUD, quality scoring |
| 22 | **Profile & Roles** | Role system (admin, creator roles for agents) | User roles CRUD, permission system |

### Phase 2B: Agent-Driven Content Creation
*No new UI pages — this is backend work. Agents use the same API endpoints as the Strategy Builder, Portfolio Builder, etc. but via API calls instead of the UI.*

| Priority | Work | Notes |
|----------|------|-------|
| 23 | **Agent API access layer** | API key auth for agents, rate limiting, audit logging |
| 24 | **Strategy generation pipeline** | Agents run mass backtests, save winners, start forward tests |
| 25 | **Portfolio assembly pipeline** | Agents combine top strategies into diversified portfolios |
| 26 | **Quality scoring system** | Automated scoring based on alert count, forward test duration, performance consistency |
| 27 | **Auto-curation** | Agents recommend portfolios for free rotation based on quality scores |

**Milestone:** AI agents autonomously create and curate trading portfolios. Admin reviews and approves.

---

## Phase 3: Marketplace MVP (First External Users)
**Goal:** Other people can sign up, browse portfolios, and subscribe. Revenue starts flowing.
**Depends on:** Phase 2 complete (portfolios exist and are curated).

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 28 | **Onboarding** | First thing new users see. Must be dead simple. | User creation, role assignment |
| 29 | **Pricing & Plans** | Tier selection during/after onboarding | Stripe/payment integration, subscription management |
| 30 | **Marketplace Browse** | Core marketplace — browse portfolios with performance data | Portfolio listing API, search/filter, verification badges |
| 31 | **Portfolio Subscription** | Subscribe → configure webhook → set risk → done | Subscription CRUD, webhook validation, billing |
| 32 | **My Subscriptions** | Manage active subscriptions, webhook health | Subscription management, alert delivery status |
| 33 | **Prop Firm Hub** | Browse compatible firms, affiliate links | Prop firm data, affiliate link tracking |

**Milestone:** External users can sign up, subscribe to portfolios, and receive trading alerts. Affiliate revenue from prop firms starts.

---

## Phase 4: Creator Economy
**Goal:** Open the platform to human creators. Contributors can publish and earn revenue.
**Depends on:** Phase 3 complete (marketplace works, payments flow).

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 34 | **Creator Dashboard** | Earnings, subscriber growth, content performance | Creator analytics, revenue tracking |
| 35 | **Earnings & Payouts** | Revenue details, payout history, tax docs | Payout system (Stripe Connect / PayPal), revenue splitting |
| 36 | **Publish** | List portfolios/strategies/packs on marketplace | Listing CRUD, verification system, pricing |
| 37 | **Admin Users** | Manage creator applications, moderate content | User management, content moderation tools |

**Milestone:** Full creator economy — people create, publish, and earn. Platform takes 15-20% fee.

---

## Phase 5: Scale & Ecosystem
**Goal:** Network effects, mobile, international, advanced features.
**Depends on:** Phase 4 stable and generating revenue.

| Priority | Work | Notes |
|----------|------|-------|
| 38 | Mobile app (React Native or PWA) | Subscribers primarily need mobile for alerts |
| 39 | Strategy/Pack marketplace expansion | Beyond portfolios — license individual strategies and packs |
| 40 | Advanced analytics | Monte Carlo, regime detection, portfolio optimization |
| 41 | Social features | Creator profiles, reviews, comments, following |
| 42 | International | Multi-currency, multi-language, timezone-aware |
| 43 | Scanner strategy method (Phase 23) | Real-time market scanning for entry conditions |
| 44 | Nonprofit/community fund | Portion of revenue to displaced worker programs |

---

## FastAPI Backend — Build Alongside Phase 1

The backend must be built incrementally as pages are wired up. Key endpoint groups:

### Core (Phase 1A)
```
POST   /api/auth/login
POST   /api/auth/signup
POST   /api/auth/refresh
GET    /api/settings
PUT    /api/settings
GET    /api/confluence-groups
POST   /api/confluence-groups
PUT    /api/confluence-groups/:id
DELETE /api/confluence-groups/:id
POST   /api/backtest/run
GET    /api/data/bars/:symbol
GET    /api/strategies
POST   /api/strategies
GET    /api/strategies/:id
PUT    /api/strategies/:id
DELETE /api/strategies/:id
```

### Extended (Phase 1B)
```
GET    /api/portfolios
POST   /api/portfolios
GET    /api/portfolios/:id
PUT    /api/portfolios/:id
POST   /api/portfolios/:id/compute
GET    /api/requirements
POST   /api/requirements
GET    /api/alerts
GET    /api/alerts/monitor/status
POST   /api/alerts/monitor/start
POST   /api/alerts/monitor/stop
GET    /api/webhooks/templates
POST   /api/webhooks/templates
POST   /api/webhooks/test
```

### Power User (Phase 1C)
```
POST   /api/mass-builder/run
GET    /api/mass-builder/results
GET    /api/timeframes
PUT    /api/timeframes
GET    /api/packs/user
POST   /api/packs/user
```

### Platform (Phase 2+)
```
GET    /api/admin/stats
GET    /api/admin/users
PUT    /api/admin/users/:id/role
GET    /api/marketplace/portfolios
POST   /api/marketplace/subscribe
GET    /api/marketplace/subscriptions
GET    /api/creator/dashboard
GET    /api/creator/earnings
POST   /api/creator/publish
```

---

## Design Iteration Schedule

### Week 1: Foundation
- Days 1-2: Lock in Settings pages (small, sets patterns)
- Days 3-4: Lock in Dashboard
- Days 5-7: Lock in TF Confluence + General Packs + Risk Management

### Week 2: Strategy Workflow
- Days 1-3: Lock in Strategy Builder (most complex, allow extra time)
- Days 4-5: Lock in My Strategies list + detail
- Days 6-7: Start FastAPI backend scaffolding

### Week 3: Portfolios
- Days 1-2: Lock in Portfolio list + builder
- Days 3-4: Lock in Portfolio detail (7 tabs)
- Days 5-7: Lock in Requirements + Alerts + Webhooks

### Week 4: Power Tools + Wiring
- Days 1-2: Lock in Mass Builder + Results
- Days 3-4: Lock in remaining pack pages
- Days 5-7: Begin backend wiring for Phase 1A pages

### Ongoing: Backend Wiring
- Each locked page gets API endpoints built and wired
- Mock data replaced with real Supabase queries
- Testing with real market data

---

## Summary

| Phase | Pages | Business Goal | Revenue |
|-------|-------|--------------|---------|
| **1A** | Settings, Dashboard, Confluence (3), Strategy Builder, My Strategies | Core trading loop works | Personal profits |
| **1B** | Portfolios (4), Requirements, Alerts (2) | Full workflow with alerts | Personal profits |
| **1C** | Mass Builder/Results, Timeframes, User Packs, Pack Builder | Feature parity | Personal profits |
| **2** | Admin Dashboard/Curation, Profile & Roles + agent API | AI creates content | Automated portfolio creation |
| **3** | Onboarding, Pricing, Marketplace, Subscriptions, Prop Firms | First paying users | Subscriptions + affiliates |
| **4** | Creator Dashboard, Earnings, Publish, Admin Users | Creator economy | Platform fees (15-20%) |
| **5** | Mobile, international, scanner, social | Scale | All revenue streams |
