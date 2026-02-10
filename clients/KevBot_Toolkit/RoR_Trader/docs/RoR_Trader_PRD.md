# RoR Trader - Product Requirements Document (PRD)

**Version:** 0.5
**Date:** February 9, 2026
**Author:** Kevin Johnson
**Status:** Phase 7 Complete — Phases 8-11 Roadmap Defined

---

## 1. Executive Summary

**RoR Trader** (Return on Risk Trader) is a web application designed to democratize profitable trading by making strategy creation, backtesting, and execution accessible to everyone—regardless of programming experience or trading background.

### Mission Statement
> Make jobs optional for our users by providing a data-backed, accessible path to trading profitability.

### Core Problem Statement
- Many people will be displaced from traditional work and need alternative income sources
- Current trading tools require programming knowledge for effective backtesting
- YouTube "gurus" promise results without providing data to back claims
- Indicators require subjective interpretation, leaving users guessing
- Building and validating strategies is painful and time-consuming

### Solution
A comprehensive platform that:
1. Provides a library of pre-built indicators, interpreters, and triggers
2. Eliminates subjective interpretation with clear, data-backed conditions
3. Enables no-code strategy creation through a guided workflow
4. Offers seamless backtesting → forward testing → live trading pipeline
5. Supports portfolio construction with prop firm compliance checking

---

## 2. Core Concepts & Terminology

### 2.1 Building Block Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                        PORTFOLIO                            │
│    (Collection of strategies traded in the same account)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        STRATEGY                             │
│     (Entry trigger + Exit trigger + Confluence conditions)  │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐      ┌───────────┐      ┌─────────────────┐
    │  TRIGGER  │      │  TRIGGER  │      │  INTERPRETATIONS │
    │  (Entry)  │      │  (Exit)   │      │  (Confluence)    │
    └───────────┘      └───────────┘      └─────────────────┘
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │ INTERPRETERS │
                                          └─────────────┘
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │ INDICATORS  │
                                          └─────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │ CHARTS / CANDLES / PA   │
                                    └─────────────────────────┘
```

### 2.2 Definitions

| Term | Definition |
|------|------------|
| **Chart/Candles** | Base price action data (OHLCV) for a given ticker and timeframe |
| **Indicator** | Mathematical calculations plotted on charts (e.g., EMA, VWAP, RSI) |
| **Interpreter** | Logic that examines indicators and/or price action and outputs mutually exclusive condition states |
| **Interpretation** | The output of an interpreter—a classified condition state that can be used as a confluence variable |
| **Trigger** | A specific condition that initiates entry or exit from a position |
| **Confluence** | A combination of interpretations that must be present to validate a trade |
| **Strategy** | A complete trading system: ticker + direction + entry trigger + exit trigger + confluence conditions |
| **Portfolio** | A collection of strategies traded together within the same account |
| **Prop Firm** | Proprietary trading firms (e.g., Trade The Pool) that provide funded accounts with specific rules |

### 2.3 The Interpreter Concept (Key Differentiator)

**The Problem:** Indicators show data, but traders are left to subjectively interpret what they mean. Two traders looking at the same chart may reach different conclusions.

**The Solution:** Interpreters bridge the gap between indicators and actionable conditions by:
- Taking indicator values, price action, and other inputs
- Applying defined logic to classify the current state
- Outputting a clear, mutually exclusive interpretation

**Example:**
```
Interpreter: EMA Stack Interpreter
Inputs: EMA 8, EMA 21, EMA 50, Price
Possible Interpretations:
  - "Full Bull Stack" (Price > 8 > 21 > 50)
  - "Bull Stack Below 8" (8 > Price > 21 > 50)
  - "Compression" (EMAs within X% of each other)
  - "Full Bear Stack" (Price < 8 < 21 < 50)
  - etc.
```

This allows us to:
1. **Quantify impact** - See how each interpretation affects win rate, profit factor, RoR
2. **Reverse optimize** - Find which confluence combinations yield best results
3. **Eliminate subjectivity** - Clear definitions, not vibes

---

## 3. User Workflow

### 3.1 Strategy Creation Flow

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Define Core Parameters                               │
├──────────────────────────────────────────────────────────────┤
│  • Select Ticker (e.g., SPY, AAPL, ES)                       │
│  • Select Direction (Long OR Short)                          │
│  • Select Entry Trigger                                      │
│  • Select Exit Trigger                                       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Refine with Confluence (Drill-Down)                  │
├──────────────────────────────────────────────────────────────┤
│  • View base strategy KPIs (win rate, profit factor, etc.)   │
│  • Browse available interpretations from enabled interpreters │
│  • Layer in confluence conditions one at a time              │
│  • See real-time impact on equity curve and KPIs             │
│  • Use "Find Optimal" to auto-discover best combinations     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Save & Validate                                      │
├──────────────────────────────────────────────────────────────┤
│  • Save as named Strategy                                    │
│  • Enable Forward Testing to validate edge persistence       │
│  • View forward test results over time                       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Deploy                                               │
├──────────────────────────────────────────────────────────────┤
│  • Add to Portfolio(s)                                       │
│  • Enable Alerts                                             │
│  • Connect to Trading Bots                                   │
│  • Export to TradingView (optional)                          │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Portfolio Management Flow

1. **Create Portfolio** - Name and define account parameters
2. **Add Strategies** - Select from saved strategies
3. **Analyze Combined Performance**
   - Combined equity curve
   - Drawdown analysis (how strategies interact)
   - Daily P&L distribution
4. **Prop Firm Compliance Check**
   - Select target prop firm (e.g., Trade The Pool)
   - Verify portfolio stays within rules (max drawdown, daily loss limits, etc.)
5. **Deploy Portfolio**
   - Enable alerts for all strategies
   - Connect to trading bots

---

## 4. Key Features

### 4.1 Core Platform Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Strategy Builder** | No-code interface to create strategies with triggers and confluence | P0 |
| **Backtesting Engine** | Programmatic backtesting with accurate historical data | P0 |
| **Interpreter Library** | Pre-built interpreters for common indicators/patterns | P0 |
| **Drill-Down Analysis** | Layer confluence conditions and see KPI impact | P0 |
| **Reverse Optimization** | Auto-discover optimal confluence combinations | P1 |
| **Forward Testing** | Track strategy performance on live data after creation | P1 |
| **Portfolio Builder** | Combine strategies and analyze collective performance | P1 |
| **Prop Firm Rules Engine** | Check portfolio compliance with prop firm requirements | P2 |
| **Alerts System** | Notifications when strategy conditions are met | P1 |
| **Trading Bot Integration** | Connect to third-party execution platforms | P2 |
| **Built-in Charting** | Native charting with strategy visualization | P2 |
| **TradingView Export** | Export strategies as TradingView PineScript | P2 |

### 4.2 KPIs & Metrics

Strategies and portfolios should display:
- **Return on Risk (RoR)** - Primary metric (namesake of the tool)
- **Profit Factor**
- **Win Rate**
- **Number of Trades**
- **Average Win / Average Loss**
- **Max Drawdown**
- **Daily P&L Distribution**
- **Equity Curve**
- **Sharpe Ratio**
- **Risk-Adjusted Return**

### 4.3 User Settings

- **Enabled Interpreters** - Select which interpreters to use for confluence options
- **Default Ticker/Timeframe**
- **Risk Parameters** - Default position sizing, max risk per trade
- **Alert Preferences**
- **Connected Accounts** - Trading bots, brokers

---

## 5. Marketplace & Community

### 5.1 User-Contributed Content

Users can contribute and monetize:

| Content Type | Description | Monetization |
|--------------|-------------|--------------|
| **Indicators** | Custom technical indicators | Free or Paid |
| **Interpreters** | Custom interpretation logic | Free or Paid |
| **Triggers** | Custom entry/exit conditions | Free or Paid |
| **Strategies** | Complete trading strategies | Subscription |
| **Signals** | Alerts from forward-tested strategies | Subscription |

### 5.2 Strategy Subscriptions

- Users can publish strategies with forward-test track records
- Subscribers receive:
  - Alerts when strategy triggers
  - Access to strategy performance data
  - Optional auto-execution via connected bots
- Contributors earn recurring revenue from subscribers

### 5.3 Trust & Validation

- Forward-test history is immutable and public
- Backtest-only strategies marked differently than forward-tested
- Verified track records build contributor reputation

---

## 6. Technical Architecture (Initial Thoughts)

### 6.1 Phase 1: Python Script (Personal Tool)
- Local data storage (SQLite or similar)
- Command-line or simple web interface (Flask/Streamlit)
- Prove core concepts with real usage

### 6.2 Phase 2: Web Application
- Modern web stack (React/Next.js frontend, Python backend)
- Cloud database for user data, strategies, portfolios
- Real-time data feeds for forward testing
- API for trading bot integrations

### 6.3 Data Requirements
- Historical OHLCV data (multiple timeframes)
- Real-time or delayed quotes for forward testing
- Indicator calculations
- Interpreter state storage

### 6.4 Data Provider: Alpaca
- **Selection:** Alpaca Markets API
- **Capabilities Needed:**
  - Historical bars (daily, intraday down to 1-minute)
  - Real-time/delayed quotes for forward testing
  - Equity market data (US stocks)
- **Next Step:** Set up Alpaca account and explore data structure

---

## 7. Application Sitemap

### 7.1 Information Architecture Overview

```
RoR Trader
│
├── 🏠 DASHBOARD
│   ├── Overview Cards (strategies, portfolios, alerts)
│   ├── Active Forward Tests Summary
│   ├── Recent Alerts
│   └── Quick Actions (New Strategy, New Portfolio)
│
├── 📊 STRATEGY BUILDER (Core Workflow)
│   │
│   ├── Step 1: Setup
│   │   ├── Ticker Selection (search/browse)
│   │   ├── Timeframe Selection
│   │   ├── Direction Toggle (Long / Short)
│   │   ├── Entry Trigger Dropdown
│   │   └── Exit Trigger Dropdown
│   │
│   ├── Step 2: Confluence (Drill-Down)
│   │   ├── Base Strategy KPIs Panel
│   │   ├── Equity Curve Chart
│   │   ├── Available Interpretations List
│   │   │   └── (grouped by interpreter)
│   │   ├── Active Confluence Conditions
│   │   ├── "Add Condition" → See Impact
│   │   ├── "Find Optimal" Button
│   │   └── Trade List / Details Table
│   │
│   └── Step 3: Save & Configure
│       ├── Strategy Name
│       ├── Strategy Summary (triggers, conditions)
│       ├── Final KPIs Display
│       ├── Enable Forward Testing Toggle
│       └── Save Strategy Button
│
├── 📁 MY STRATEGIES
│   ├── Strategy List View
│   │   ├── Filter: All / Backtest Only / Forward Testing / Deployed
│   │   ├── Sort: Name / Created / Performance
│   │   └── Strategy Cards showing:
│   │       ├── Name, Ticker, Direction
│   │       ├── Key KPIs (RoR, Win Rate, Profit Factor)
│   │       ├── Forward Test Status & Duration
│   │       └── Actions (Edit, Deploy, Delete)
│   │
│   └── Strategy Detail View
│       ├── Configuration Summary
│       ├── Backtest Results Tab
│       │   ├── Equity Curve
│       │   ├── KPI Dashboard
│       │   └── Trade History Table
│       ├── Forward Test Results Tab
│       │   ├── Live Equity Curve
│       │   ├── Backtest vs Forward Comparison
│       │   └── Recent Signals
│       ├── Alerts Tab
│       │   └── Alert Configuration
│       └── Actions
│           ├── Edit Strategy
│           ├── Add to Portfolio
│           ├── Enable/Disable Forward Test
│           ├── Configure Alerts
│           └── Export to TradingView
│
├── 💼 PORTFOLIOS
│   ├── Portfolio List View
│   │   └── Portfolio Cards showing:
│   │       ├── Name, # Strategies
│   │       ├── Combined KPIs
│   │       └── Prop Firm Compliance Status
│   │
│   ├── Portfolio Builder
│   │   ├── Portfolio Name
│   │   ├── Account Parameters (starting capital, etc.)
│   │   ├── Strategy Selector (from My Strategies)
│   │   ├── Position Sizing per Strategy
│   │   └── Save Portfolio
│   │
│   └── Portfolio Detail View
│       ├── Strategies Included List
│       ├── Combined Analysis Tab
│       │   ├── Combined Equity Curve
│       │   ├── Correlation Matrix
│       │   ├── Drawdown Analysis
│       │   └── Daily P&L Distribution
│       ├── Prop Firm Compliance Tab
│       │   ├── Rule Set Selector (Trade The Pool, etc.)
│       │   ├── Compliance Checklist
│       │   │   ├── ✓/✗ Max Daily Loss
│       │   │   ├── ✓/✗ Max Drawdown
│       │   │   ├── ✓/✗ Profit Target
│       │   │   └── etc.
│       │   ├── Recommendations (if non-compliant)
│       │   └── Prop Firm Suggestions
│       └── Deploy Tab
│           ├── Enable Portfolio Alerts
│           └── Connect Trading Bot
│
├── 🔔 ALERTS & SIGNALS
│   ├── Active Alerts List
│   ├── Alert History
│   ├── Webhook Configuration
│   └── Trading Bot Connections
│
├── 🏪 MARKETPLACE (Future)
│   ├── Browse
│   │   ├── Indicators
│   │   ├── Interpreters
│   │   ├── Triggers
│   │   └── Strategies (with forward-test records)
│   ├── My Contributions
│   └── My Subscriptions
│
├── 📈 CHARTS (Future)
│   ├── Chart View with Strategy Overlay
│   ├── Indicator Configuration
│   └── Interpreter Visualization
│
└── ⚙️ SETTINGS
    ├── Account Settings
    │   ├── Profile
    │   └── Subscription/Billing
    ├── Interpreter Library
    │   ├── Available Interpreters List
    │   ├── Enable/Disable Toggles
    │   └── Interpreter Details & Parameters
    ├── Trigger Library
    │   ├── Available Triggers List
    │   └── Enable/Disable Toggles
    ├── Default Preferences
    │   ├── Default Ticker
    │   ├── Default Timeframe
    │   └── Default Risk Parameters
    ├── Connections
    │   ├── Alpaca API Keys
    │   ├── Webhook URLs
    │   └── Trading Bot Integrations
    └── Prop Firm Rule Sets
        ├── Trade The Pool
        ├── [Other Firms]
        └── Custom Rules
```

### 7.2 Core User Journeys

**Journey 1: New User Creates First Strategy**
```
Dashboard → "New Strategy" → Strategy Builder Step 1 (Setup)
→ Step 2 (Drill-Down, add confluence) → Step 3 (Save)
→ My Strategies (view saved strategy) → Enable Forward Test
```

**Journey 2: Build Portfolio for Prop Firm**
```
My Strategies (select multiple) → Portfolios → "New Portfolio"
→ Add Strategies → Analyze Combined Performance
→ Prop Firm Compliance Tab → Check Trade The Pool rules
→ Adjust if needed → Deploy
```

**Journey 3: Set Up Live Trading**
```
My Strategies → Strategy Detail → Alerts Tab
→ Configure Webhook → Test Alert
→ (or) Portfolios → Deploy Tab → Connect Bot
```

**Journey 4: Optimize Existing Strategy**
```
My Strategies → Strategy Detail → Edit Strategy
→ Strategy Builder Step 2 → "Find Optimal"
→ Review suggestions → Apply changes → Save
```

### 7.3 Page Priority for MVP

| Page | Priority | Notes |
|------|----------|-------|
| Strategy Builder (all steps) | P0 | Core value proposition |
| My Strategies (list + detail) | P0 | Must see saved work |
| Dashboard | P1 | Nice landing page, not critical for MVP |
| Settings > Interpreter Library | P0 | Users need to enable interpreters |
| Settings > Connections (Alpaca) | P0 | Need data source |
| Forward Test Results | P1 | Key differentiator |
| Portfolios | P1 | Important but can come after single-strategy flow |
| Alerts & Webhooks | P1 | Needed for practical use |
| Prop Firm Compliance | P2 | Value-add, not core |
| Marketplace | P3 | Future phase |
| Built-in Charts | P2 | Can use TradingView initially |

---

## 8. Long-Term Vision

### 7.1 AI Integration
- AI agents to assist users in strategy creation
- Pattern recognition for new interpreter development
- Automated strategy optimization

### 7.2 Institutional/Non-Profit Applications
- **Trading Firms as Non-Profits** - Organizations that trade on behalf of members
- **Employee Benefits** - Companies offer RoR-managed trading as a benefit
- **Community Welfare** - Churches, state agencies use trading returns for welfare programs
- **Cooperative Trading** - Groups pool resources and share returns

### 7.3 Education Platform
- Guided learning paths for new traders
- Paper trading with real feedback
- Community mentorship

---

## 9. Success Metrics

| Metric | Target | Timeframe |
|--------|--------|-----------|
| Users creating strategies | 1,000 | 6 months post-launch |
| Strategies with positive forward-test | 30% | Ongoing |
| Users achieving consistent profitability | 20% | 12 months |
| Marketplace transactions | $10K/month | 12 months |
| User retention (monthly active) | 40% | Ongoing |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting in backtests | Emphasize forward testing; education on curve fitting |
| Regulatory compliance | Clear disclaimers; no financial advice; user responsibility |
| Data quality issues | Multiple data sources; validation checks |
| Users losing money | Risk management tools; paper trading first; education |
| Competition from established platforms | Focus on simplicity and interpreter concept |

---

## 11. Key Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| **Data Provider** | Alpaca | Need to set up subscription; will explore data structure |
| **Supported Markets** | Equities (initial) | Start focused, expand later to futures/crypto |
| **Pricing Model** | TBD | Focus on functional tool first, then determine pricing |
| **Trading Bot Integration** | Webhooks (MVP) | Start with webhook-based alerts; explore aggregator platforms later |
| **Prop Firm Approach** | Rule compliance checking | Start with Trade The Pool rules; add more rule sets over time |

### Future Prop Firm Features
- Check portfolio compliance against multiple prop firm rule sets
- Recommend prop firms based on portfolio behavior
- Suggest strategy adjustments to fit within rules (e.g., reduce position size to stay under max loss %)

---

## 12. Open Questions

1. **Legal Structure** - For non-profit trading firm concept
2. **Webhook Aggregator** - Which platform for multi-broker bot integration?
3. **Alpaca Tier** - Which Alpaca subscription level needed for historical + real-time?

---

## 13. Completed Milestones

1. [x] Define application sitemap and information architecture
2. [x] Set up Alpaca account and explore data structure
3. [x] Design wireframes/mockups for Strategy Builder flow (see `/docs/wireframes/`)
4. [x] Select frontend framework — **Streamlit** for MVP
5. [x] Develop first set of interpreters: EMA Stack, MACD (Line + Histogram), VWAP, RVOL, UT Bot
6. [x] Build Strategy Builder MVP — 3-step workflow (Setup, Confluence, Save)
7. [x] Implement Confluence Groups management system (template/version model)
8. [x] Integrate Alpaca API for real market data with mock data fallback
9. [x] Build backtesting engine with trade generation and KPI calculations
10. [x] Build My Strategies page (basic list view)
11. [x] Add Code/Preview tabs to Confluence Groups, fix EMA overlay bug, add strategy detail charts and confluence analysis
12. [x] Split MACD into separate templates (macd_line, macd_histogram), upgrade VWAP to 7-zone system
13. [x] Replace Plotly oscillator charts with synchronized lightweight-charts multi-pane rendering

---

## 14. Development Roadmap

### Phase 1: Harden the Core Engine ✓
*Fix bugs and remove fragility in the foundation all future features build on.*
*Completed: February 5, 2026*

- [x] Handle infinity gracefully in confluence analysis calculations (profit factor deltas, sorting) — display of infinity when no losses is correct and intentional
- [x] Add null guard on confluence record filtering (crash risk when trades have no confluence data)
- [x] Prevent same entry/exit trigger selection (validation + warning in Strategy Builder Step 1)
- [x] Replace hardcoded mappings (INTERPRETER_TO_TEMPLATE, base_trigger_map) with runtime-built maps from confluence groups
- [x] Add stop loss (ATR multiplier) configuration to Strategy Builder Step 1
- [x] Fix chart timestamp handling (fragile assumption about DataFrame column order)
- [x] Save complete strategy parameters on save (stop_atr_mult, data_days, data_seed)

### Phase 2: Complete My Strategies ✓
*The weakest existing page — P0 per this PRD but currently a display-only stub.*
*Completed: February 5, 2026*

- [x] Strategy detail view (full R-based KPIs, equity curve, R-distribution histogram, price chart, trade history table)
- [x] Edit strategy (reopen in Strategy Builder with saved configuration; warns if forward testing enabled)
- [x] Delete strategy (with inline confirmation dialog)
- [x] Clone/duplicate strategy (preserves original, disables forward testing on copy)
- [x] Re-backtest with current data (detail view re-runs backtest live using saved config)
- [x] Sorting and filtering (by ticker, direction, status; sort by name, date, win rate, profit factor, total R)
- [x] Fix strategies.json path to resolve relative to script location (not working directory)

### Design Decisions
- **R-based metrics at strategy level** — Avg R, Total R, Daily R keep comparisons apples-to-apples across strategies. Dollar sizing (risk_per_trade, starting_balance) deferred to portfolio level.
- **Stop loss as strategy parameter** — ATR multiplier affects trade outcomes (win rate, R distribution), so it belongs at strategy level. Future: add more stop loss types (fixed dollar, trailing, percentage).
- **Legacy strategy handling** — Strategies created before Phase 1 (IDs 1, 2) cannot be re-backtested or edited; they display saved KPIs only.

### Phase 3: Forward Testing — COMPLETED (Feb 5, 2026)
*Key differentiator — what separates RoR Trader from backtest-only tools.*

- [x] Track strategy performance on new data after save date (on-the-fly computation from forward_test_start to now)
- [x] Backtest vs. forward test comparison visualization (side-by-side KPIs with deltas, combined equity curve with split line)
- [x] Forward test data pipeline (date range support in data_loader.py, trade splitting at boundary)
- [x] Combined equity curve (blue backtest / green forward segments, orange vertical split line)
- [x] R-distribution comparison (side-by-side histograms)
- [x] Split trade history (forward trades expanded, backtest collapsed)
- [x] Status indicators on strategy cards (duration badge, e.g. "Forward Testing (14d)")
- [x] Mini equity curves on strategy list cards (sparkline per card, forward-test-aware coloring)
- [x] Timezone-aware datetime handling for Alpaca UTC timestamps

### Design Decisions (Phase 3)
- **On-the-fly computation** — No stored forward test results; always recompute from fresh data. Simpler architecture, always reflects latest data.
- **Mock data simulation** — Same seed produces identical backtest portion; new bars generated beyond save date for forward testing with mock data.
- **Cache-friendly end dates** — Forward test end_date rounded to market close (4 PM) so cached pipeline results are reused within the same day.

### Phase 4: Portfolios & Prop Firm Compliance — COMPLETED (Feb 5, 2026)
*Combine strategies and validate against real trading account rules.*

- [x] Portfolio builder with dollar-based risk sizing per strategy
- [x] Combined equity curve (multi-line: per-strategy dashed + combined bold)
- [x] Drawdown analysis with requirement set limit lines
- [x] Correlation matrix heatmap between strategies
- [x] Daily P&L distribution histogram
- [x] Compounding support (compound_rate 0-100% scales risk with account growth)
- [x] Prop firm rule sets (Trade The Pool, FTMO) as built-in templates
- [x] Compliance checker with pass/fail indicators, margin of safety, cross-set compatibility
- [x] Portfolio CRUD (create, edit, clone, delete with inline confirmation)
- [x] Cached KPIs on portfolio list cards with mini equity curves

### Phase 4B: Interactive Builder & Requirements System — COMPLETED (Feb 5, 2026)
*UX improvements to portfolio creation and prop firm rule management.*

- [x] Interactive portfolio builder (add/remove strategies one at a time with live metric updates)
- [x] Strategy recommendation engine (composite scoring: P&L, drawdown, PF, correlation, win rate)
- [x] Portfolio Requirements page (new nav page with CRUD for requirement sets)
- [x] Built-in TTP + FTMO requirement sets (non-deletable, duplicatable)
- [x] Custom requirement set creation with full rule editor
- [x] Migrated portfolios from legacy prop_firm/custom_rules to requirement_set_id
- [x] Prop Firm Check tab refactored to use requirement set selectbox

### Design Decisions (Phase 4)
- **Dollar risk per strategy** — Each strategy in a portfolio has its own risk_per_trade (not percentage allocations). Strategies work in R-multiples; portfolios convert to dollars.
- **Sequential compounding** — Trade-by-trade computation: `scaled_risk = base_risk * (1 + account_growth_pct * compound_rate)`. At 0% = fixed risk; at 100% = risk scales 1:1 with growth.
- **Requirement sets over inline rules** — Decoupled rule management from portfolio views. Users create/manage requirement sets on a dedicated page, then select from them in portfolio compliance checks.
- **Strategy recommendation scoring** — Weighted composite: P&L improvement (30%), drawdown reduction (25%), profit factor (20%), low correlation (15%), win rate (10%).

### Phase 5: Alerts & Deployment *(COMPLETED 2026-02-05)*
*Make strategies actionable in real time.*

- [x] Alert engine (`alerts.py`) — config CRUD, signal detection, position tracking, webhook delivery
- [x] Background monitor (`alert_monitor.py`) — standalone polling script with market hours awareness
- [x] Alerts & Signals navigation page — monitor start/stop, global settings, per-strategy/portfolio config, recent alerts list
- [x] Strategy detail Alerts tab — toggle entry/exit alerts, webhook override, recent alerts
- [x] Portfolio detail Deploy tab — toggle compliance breach alerts, webhook override, recent alerts
- [x] Webhook configuration UI — global + per-strategy + per-portfolio override URLs
- [x] Real-time signal detection — reuses existing indicator/interpreter/trigger pipeline on latest bars
- [x] Alert history and management — acknowledge/clear, color-coded by type, capped at 500
- [x] Portfolio-level enrichment — strategy signals include portfolio allocation context
- [x] Discord/Slack-compatible webhook payload format with embeds

### Phase 5B: Alert System Redesign *(COMPLETED 2026-02-06)*
*Overhaul alerts from single-webhook to a production-ready multi-webhook system.*

- [x] Replace global/strategy webhook URLs with per-portfolio multi-webhook system
- [x] LuxAlgo-style webhook builder — name, URL, event checkboxes (Entry Long/Short, Exit Long/Short, Compliance Breach), custom JSON payloads with {{placeholder}} insertion
- [x] Placeholder system — 22 dynamic placeholders including derived quantity, order_action, market_position
- [x] Webhook templates — default TradeThePool templates (Market/Limit Order Buy/Sell/Close) + user-created template CRUD
- [x] "Insert Template" and "Insert Placeholder" dropdowns in webhook editor
- [x] Alerts page redesign — 4 tabs (Strategy Alerts, Portfolio Alerts, Outbound Webhooks, Inbound Webhooks placeholder)
- [x] Date range filtering on all alert tabs (replaces Ack/Clear buttons)
- [x] Active alerts/webhooks management expander for quick deactivation at scale
- [x] Decouple strategy alert toggles from portfolio webhook delivery — webhooks always fire regardless of strategy-level toggle
- [x] Per-webhook delivery tracking with payload inspection on Outbound tab
- [x] Auto-migration of old alert_config.json schema on load
- [x] Webhook Templates navigation page with category grouping, default duplication, and custom CRUD

### Design Decisions (Phase 5/5B)
- **Strategy-level detection, portfolio-level enrichment** — Signals are detected per-strategy (unique symbol+trigger combinations). Portfolio context (position sizing, compliance) is added after detection. This avoids duplicate data fetches for the same symbol across portfolios.
- **Stateless position tracking** — Instead of maintaining a persistent position state machine, the monitor runs `generate_trades()` on recent bars and checks if the last trade is still open. Leverages existing engine without complex state management.
- **JSON file communication** — Monitor and Streamlit app communicate via `alert_config.json`, `alerts.json`, `monitor_status.json`, and `webhook_templates.json`. Simple, no database needed, human-readable.
- **Process management** — Monitor writes PID to status file, Streamlit sends SIGTERM to stop. Status file is verified against actual process liveness on each UI render.
- **Webhooks at portfolio level only** — Webhooks live on portfolios, not strategies or globally. A strategy signal enriched with portfolio context fires all matching portfolio webhooks. Strategy-level toggles only control in-app visibility, not webhook delivery.
- **Placeholder-driven payloads** — Custom JSON templates use `{{placeholder}}` tokens resolved at delivery time. Derived values (quantity from risk/stop distance, order_action from signal type) are computed dynamically.

### Phase 6: Dashboard — COMPLETED (Feb 6, 2026)
*Landing page that ties the application together.*

- [x] Overview cards (strategy count, forward testing count, portfolio count, recent alerts)
- [x] Top strategy highlight with mini equity curve (best by Total R)
- [x] Top portfolio highlight with key KPIs (best by P&L)
- [x] Recent alerts feed (last 5, reusing alert row component)
- [x] System status panel (data source, alert monitor state, forward test count)
- [x] Quick actions (New Strategy, View Strategies, View Portfolios) with programmatic navigation
- [x] Empty state for new users with onboarding message and pipeline explanation
- [x] `nav_target` session state pattern for cross-page button navigation

### Phase 7: Confluence Group Enhancements — COMPLETED (Feb 9, 2026)
*Verify and expand confluence group tooling — ensure indicators/interpreters are behaving correctly before going live.*

- [x] Code tab on Confluence Group detail page — displays underlying indicator/interpreter/trigger source code with active parameter values for transparency and debugging
- [x] Preview tab on Confluence Group detail page — generates sample data and shows price chart with indicator overlays (or synced oscillator pane), interpreter state timeline, and trigger event table
- [x] Indicator overlay on all strategy charts — fixed EMA overlay bug (template names → actual column names) so overlays now render correctly on strategy detail backtest and forward test views
- [x] Confluence Analysis tab on strategy detail pages — sub-tabs per enabled confluence group showing relevant chart, interpreter state changes, and trigger events for visual verification
- [x] Trade History price charts — candlestick chart with entry/exit markers on strategy detail trade history tabs (backtest and forward test)
- [x] Synchronized multi-pane charts — MACD and RVOL oscillator charts render as lightweight-charts panes below the price chart with shared zoom/scroll (TradingView-style), replacing standalone Plotly charts
- [x] Split MACD into separate templates — macd_line and macd_histogram are now independent confluence groups following the one-interpreter-per-group principle
- [x] Upgraded VWAP to 7-zone system — dual standard deviation bands (±1σ, ±2σ) with 7 mutually exclusive zones (ABOVE_SD2_UPPER through BELOW_SD2_LOWER), matching KevBot Toolkit reference
- [x] Backward-compatible config migrations — old MACD template name auto-converts to macd_line; old VWAP std_dev parameter auto-converts to sd1_mult/sd2_mult
- [x] Confluence Analysis filtered to relevant groups — only shows confluence groups actually used by the strategy (as entry/exit trigger or confluence condition), not all enabled groups
- [x] Trade entry/exit markers on Confluence Analysis charts — same entry/exit arrows as Trade History for cross-referencing indicator behavior with trade outcomes
- [x] Chart Presets sidebar control — "Visible Candles" selectbox (Tight 50, Close 100, Default 200, Wide 400, Full) controls initial zoom level on all price charts by trimming rendered data to last N candles; upstream calculations unaffected

### Design Decisions (Phase 7)
- **One interpreter per confluence group** — Each group maps to exactly one interpreter for clean state tracking and confluence analysis. MACD Line (bullish/bearish crossover states) and MACD Histogram (positive/negative momentum states) are separate groups because they produce different interpretations.
- **7-zone VWAP** — Matches KevBot Toolkit's proven zone system: >+2σ, >+1σ, >VWAP, @VWAP, <VWAP, <-1σ, <-2σ. More granular than the original 3-zone (above/at/below) system, enabling better confluence precision.
- **Synced chart panes via lightweight-charts** — `renderLightweightCharts` accepts a list of pane configs; multiple panes share a synchronized time axis for zoom/scroll. This replaces Plotly charts that couldn't sync with the price chart above. Used for MACD oscillator and RVOL histogram panes.
- **Code tab transparency** — Uses `inspect.getsource()` to show actual running Python code. Active parameter values are displayed alongside the source so users can see exactly what periods/multipliers are in effect.
- **Template → column name resolution** — The EMA overlay bug was caused by returning template abstract names (ema_short) instead of actual DataFrame column names (ema_9). Fixed by resolving group parameters to concrete column names in the overlay helper functions.
- **Chart presets via data trimming** — The `streamlit_lightweight_charts` component unconditionally calls `fitContent()` on render, overriding `barSpacing`. To control initial zoom, we trim rendered data to the last N candles so `fitContent()` fits only those. All upstream indicator/interpreter/backtest calculations use the full dataset.
- **Relevant groups only in Confluence Analysis** — Strategies may use only 2-3 of many enabled confluence groups. Showing all groups creates noise; filtering to groups referenced by the strategy's triggers or confluence conditions keeps the UI focused. Scales well as the group library grows.

### Phase 8: QA, Polish & UX — "Get Live-Tradeable"
*Comprehensive review pass and UX improvements — the gate to live trading with real money.*

**QA & Verification:**
- [ ] Indicator verification — confirm all indicators calculate correctly against known values
- [ ] Interpreter verification — validate all interpreter states produce expected outputs
- [ ] KPI accuracy audit — ensure metrics are correct and displayed in the right places
- [ ] Alert monitor end-to-end test — verify signals detect, webhooks fire, payloads resolve
- [ ] Forward testing validation — confirm live data pipeline produces accurate results
- [ ] Edge cases — empty states, single-trade strategies, zero-trade portfolios, missing data
- [ ] Performance — identify and address any slow-loading pages or redundant data fetches
- [ ] Bug fixes — address all known issues surfaced during development

**Backtest Settings Overhaul:**
- [ ] Replace "Data Settings" sidebar section with "Backtest Settings" — expanded controls for backtest data range
- [ ] Three look-back modes via selectbox:
  - **Days** (default) — slider from 7 to 1,825 (5 years); recommended for apples-to-apples comparison across strategies on different timeframes
  - **Bars/Candles** — number input (e.g., 500, 1000, 2000 candles); app calculates equivalent days based on selected timeframe
  - **Date Range** — two date pickers (start/end) for precise control
- [ ] Estimated bar count display — show "~98,000 bars" next to the setting so users understand data volume before running
- [ ] Performance warning — yellow banner when estimated bars exceed ~50K: "Large dataset — backtest may take longer"
- [ ] Result caching — cache trades/KPIs keyed on (symbol, timeframe, date range, strategy config) so repeated views load instantly after first computation
- [ ] Expand supported Alpaca timeframes — currently 7 presets; Alpaca supports any minute increment (1–59Min), 1–23Hour, and Day/Week/Month
- [ ] Timeframe-aware max range guidance — show recommended max alongside the slider (e.g., "1Min: up to 1 year recommended, Daily: up to 5 years")
- [ ] Fix mock data timeframe — mock data generator currently always produces 1Min bars regardless of selected timeframe
- [ ] Date range validation — prevent requests before 2016 (Alpaca data floor); warn on very large ranges
- [ ] Alpaca data source note — inform free-plan users that historical data comes from IEX (single exchange) vs. SIP (all exchanges) on the paid plan

**UX Improvements:**
- [ ] "Create New Strategy" button on My Strategies page (consistent with Portfolios page pattern)
- [ ] Top navigation bar — Dashboard, Confluence Groups, Strategies, Portfolios, Alerts (reflects the user workflow order; supplements existing sidebar, does not replace it)
- [ ] Utility buttons on Portfolios page — "Portfolio Requirements" and "Webhook Templates" links next to "New Portfolio" button
- [ ] Multiple exit triggers — expand from single exit trigger to array of exit triggers per strategy; first-to-fire wins; backward-compatible schema migration (string → array with auto-wrap); new exit triggers can be layered from confluence groups (e.g., end-of-day exit, time-based exit)
- [ ] 2-column card layout for strategy and portfolio list views (cards with embedded mini chart instead of full-width rows)

**After this phase: start live trading. All stored schemas (strategies.json, portfolios.json, alert_config.json) are stable. All subsequent phases are additive — no restructuring or data loss risk.**

### Phase 9: Analytics & Edge Detection
*Advanced performance metrics and strategy health monitoring — inspired by Davidd Tech.*
*Reference images: `/docs/reference_images/DaviddTech *.png`*

- [ ] Edge Check overlay on equity curves — toggleable 21-period MA + Bollinger Bands on equity curve chart (visual indicator of strategy health; equity below lower BB = statistically unusual underperformance)
- [ ] Expanded KPI panel — add: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Kelly Criterion, Daily Value-at-Risk, Expected Shortfall (CVaR), Max Consecutive Wins/Losses, Gain/Pain Ratio, Payoff Ratio, Common Sense Ratio, Tail Ratio, Outlier Win/Loss Ratio, Recovery Factor, Ulcer Index, Serenity Index, Skewness, Kurtosis, Expected Daily/Monthly/Yearly returns
- [ ] Rolling performance metrics chart — interactive chart with toggle buttons for rolling Win Rate, Profit Factor, and Sharpe over a configurable trade window
- [ ] Return distribution analysis — histogram, box plot, and violin plot views with skewness/kurtosis/tail risk callouts
- [ ] Cumulative vs. Simple P&L views — compounded equity curve (reinvested gains) alongside simple/sum-based P&L
- [ ] Markov Motor Analysis (advanced tab) — win/loss transition probabilities, win/loss streak distribution chart, consistency score, stability index, trend strength, market regime detection (favorable/unfavorable/neutral clustering), edge decay chart (rolling PF with threshold line), and Markov Intelligence Insights summary
- [ ] KPI placement audit — map out primary vs. secondary KPIs for strategy cards, strategy detail, portfolio cards, portfolio detail; ensure consistent and useful placement across all views

### Phase 10: Strategy Origins
*Expand strategy creation beyond the standard trigger-based approach — support webhook-driven and scanner-based strategies.*

- [ ] Step 0: Strategy Origin selection — new first step in Strategy Builder where user selects origin type: Standard (current), Webhook Inbound, or Scanner
- [ ] Origin-specific fields in Step 1 — after origin selection, show relevant configuration fields (additional fields per origin type; existing strategies default to "standard" with no migration needed)
- [ ] Webhook Inbound origin — entries/exits driven by inbound webhooks (e.g., TradingView alerts, LuxAlgo signals); user can still layer confluence conditions from market data on top of webhook triggers; CSV upload for backtest data from TradingView or spreadsheets
- [ ] Scanner origin — strategy not tied to a single ticker; runs against a universe of stocks matching screener criteria (Alpaca screener APIs); targets active day trading / scalping use cases (S&B Capital, Warrior Trading style); requires separate planning session for architecture given 1:many ticker relationship
- [ ] Backward-compatible schema — `strategy_origin: "standard"` defaulted for all existing strategies; origin-specific fields only present when relevant

### Phase 11: Live Portfolio Management
*Active trading account management — bridge between backtesting and real-world trading.*

- [ ] Account Management tab on portfolio detail page — separate from backtest/analysis tabs
- [ ] Deposit/withdrawal ledger — track additions to and deductions from the trading account balance
- [ ] Webhook trigger audit log — plot actual fired webhook alerts on the price chart for visual verification that signals are firing correctly in production
- [ ] Trading notes — freeform notes area for the user to document observations, adjustments, and context per portfolio
- [ ] Live balance tracking — actual account balance based on webhook triggers + manual adjustments, independent of backtest projections

---

## Appendix A: Interpreter Examples

### A.1 EMA Stack Interpreter
```
Inputs: EMA Short, EMA Mid, EMA Long, Price
Outputs:
  - FULL_BULL_STACK
  - BULL_BELOW_SHORT
  - BULL_COMPRESSION
  - NEUTRAL
  - BEAR_COMPRESSION
  - BEAR_ABOVE_SHORT
  - FULL_BEAR_STACK
```

### A.2 VWAP Interpreter (7-Zone System)
```
Inputs: Price, VWAP, ±1σ Bands, ±2σ Bands
Outputs:
  - ABOVE_SD2_UPPER   (Price > VWAP + 2σ)
  - ABOVE_SD1_UPPER   (Price > VWAP + 1σ)
  - ABOVE_VWAP        (Price > VWAP)
  - AT_VWAP           (Price ≈ VWAP within 0.05%)
  - BELOW_VWAP        (Price < VWAP)
  - BELOW_SD1_LOWER   (Price < VWAP - 1σ)
  - BELOW_SD2_LOWER   (Price < VWAP - 2σ)
```

### A.3 Volume Interpreter
```
Inputs: Current Volume, Average Volume
Outputs:
  - EXTREME_VOLUME (>200% avg)
  - HIGH_VOLUME (>150% avg)
  - NORMAL_VOLUME
  - LOW_VOLUME (<50% avg)
  - MINIMAL_VOLUME (<25% avg)
```

---

*Document will be updated as vision crystallizes and development progresses.*
