# RoR Trader - Product Requirements Document (PRD)

**Version:** 0.12
**Date:** February 12, 2026
**Author:** Kevin Johnson
**Status:** Phase 10 In Progress — Settings page ✓, sidebar-to-inline refactor ✓, backtest settings ✓, result caching ✓, timeframe expansion ✓; QA remaining; Phases 1–9 complete

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
│ STRATEGY BUILDER — Single-Page with Inline Config Bar        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ROW 1 (inline data bar):                                    │
│  [Method] [Ticker] [TF] [Dir] [Lookback] [Params]            │
│  [Strategy Name] [FT][AL] [Load Data]                        │
│                                                              │
│  ROW 2 (collapsible Strategy Config expander):               │
│  [Entry Trigger] [Exit Trigger] [Stop Loss] [Target]         │
│                                                              │
│  STATUS LINE: ~7,800 bars · 390 bars/day · :red[errors]      │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  MAIN AREA (after Load Data):                                │
│  • Strategy Name as title, caption with ticker/direction     │
│  • KPI Dashboard (Win Rate, PF, Avg R, Total R, etc.)        │
│  • Price Chart with entry/exit markers + oscillator panes    │
│  • Equity Curve                                              │
│  • R-Distribution Histogram                                  │
│  • Extended KPIs (secondary metrics expander)                │
│  • Optimizable Variables (collapsible, all 6 categories)     │
│  • Confluence Drill-Down (6-tab: Entry, Exit, TF, Gen,       │
│    Stop Loss, Take Profit)                                   │
│  • Trade History Table                                       │
│  • [ Save Strategy ] button (centered, bottom of page)       │
│                                                              │
│  SIDEBAR: App title + data source only (no config widgets)   │
│                                                              │
│  Parameter changes re-run backtest automatically on cached   │
│  data. "Load Data" only needed for symbol/timeframe changes. │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ DEPLOY                                                       │
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

- **Chart Defaults** - Default visible candles (chart zoom level) ✓
- **Default Triggers** - Default entry/exit triggers for new strategies ✓
- **Default Risk Management** - Default stop loss and target for new strategies ✓
- **Development** - Data seed for mock data mode ✓
- **Enabled Interpreters** - Select which interpreters to use for confluence options
- **Default Ticker/Timeframe** - Future
- **Risk Parameters** - Default position sizing, max risk per trade - Future
- **Alert Preferences** - Future
- **Connected Accounts** - Trading bots, brokers - Future

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
RoR Trader — Top Navigation Bar
──────────────────────────────────────────────────────────────────────
  Dashboard | Confluence Packs | Strategies | Portfolios | Alerts | Settings
──────────────────────────────────────────────────────────────────────

Sidebar: App title, data source indicator only.
         No config widgets — all inputs are inline.

│
├── 🏠 DASHBOARD
│   ├── Overview Cards (strategies, portfolios, alerts)
│   ├── Active Forward Tests Summary
│   ├── Recent Alerts
│   └── Quick Actions (New Strategy, View Strategies, View Portfolios)
│
├── 🔗 CONFLUENCE PACKS (sub-nav: TF Confluence | General | Risk Management)
│   │
│   ├── TF Confluence (existing — indicator-based, tied to chart timeframe)
│   │   ├── Pack List (template-based, versioned)
│   │   └── Pack Detail (Parameters, Outputs, Preview, Code, Danger Zone tabs)
│   │
│   ├── General (non-timeframe conditions)
│   │   ├── Pack List by category (Time, Calendar)
│   │   │   ├── Time of Day, Trading Session
│   │   │   └── Day of Week, Calendar Filter
│   │   └── Pack Detail (Parameters, Outputs, Preview, Code, Danger Zone tabs)
│   │       ├── Preview: price chart with condition state markers, state transition table, distribution metrics
│   │       └── Extended hours toggle for session-based packs
│   │
│   └── Risk Management (combined stop + target from shared parameters)
│       ├── Pack List by category (Volatility, Fixed, Structure, Composite)
│       │   ├── ATR-Based, Fixed Dollar, Percentage
│       │   └── Swing, Risk:Reward
│       └── Pack Detail (Parameters, Outputs, Preview, Code, Danger Zone tabs)
│           ├── Preview: configurable entry/exit triggers, trade chart with stop/target levels, KPI summary
│           └── Code: active config display, builder function source
│
├── 📊 STRATEGIES (sub-nav: Strategy Builder | My Strategies)
│   │
│   ├── Strategy Builder (single-page with inline config bar)
│   │   ├── Inline Config (main content area):
│   │   │   ├── Row 1: Method, Ticker, Timeframe, Direction, Lookback Mode,
│   │   │   │         Lookback Params, Strategy Name, FT/AL toggles, Load Data
│   │   │   ├── Row 2 (expander): Entry Trigger, Exit Trigger, Stop Loss, Target
│   │   │   └── Status Line: bar estimate + timeframe guidance + validation errors
│   │   └── Main Area (after Load Data):
│   │       ├── Strategy Name as header with ticker/direction caption
│   │       ├── KPI Dashboard
│   │       ├── Price Chart + Oscillator Panes
│   │       ├── Equity Curve
│   │       ├── R-Distribution Histogram
│   │       ├── Extended KPIs (secondary metrics expander)
│   │       ├── Optimizable Variables (collapsible box showing active variables by category with ✕ remove)
│   │       ├── Active Tags (removable chips above drill-down for selected interpretations)
│   │       ├── Optimization Drill-Down (6 tabs):
│   │       │   ├── Entry Trigger Tab (per-trigger KPI cards with "Replace" button)
│   │       │   ├── Exit Triggers Tab (Drill-Down + Auto-Search modes)
│   │       │   ├── Timeframe Conditions Tab (existing confluence drill-down)
│   │       │   ├── General Conditions Tab (enabled general packs with outputs)
│   │       │   ├── Stop Loss Tab (multi-backtest KPI cards across RM pack stop configs)
│   │       │   └── Take Profit Tab (multi-backtest KPI cards across RM pack target configs)
│   │       ├── Trade History Table
│   │       └── Save Strategy button (centered, bottom of page)
│   │
│   └── My Strategies
│       ├── Strategy List View
│       │   ├── Filter: All / Backtest Only / Forward Testing / Deployed
│       │   ├── Sort: Name / Created / Performance
│       │   └── Strategy Cards (2-column grid; Name, Status, Mini Equity, KPIs, Entry/Exit/Stop/Target Badges, Confluence, Actions)
│       └── Strategy Detail View
│           ├── Header: Ticker, Direction, TF, Entry, Exit, Stop·Target, TF+General Confluences
│           ├── Equity & KPIs Tab (primary + extended KPIs, equity curve, R-distribution)
│           ├── Equity & KPIs (Extended) Tab (lookback mode: Days/Bars/Date Range, up to 5 years)
│           ├── Price Chart Tab (full indicators + oscillator panes + trade table)
│           ├── Trade History Tab (clean chart + trade table)
│           ├── Confluence Analysis Tab
│           ├── Configuration Tab (TF Conditions + General Conditions)
│           ├── Alerts Tab
│           └── Actions (Edit, Clone, Delete, Add to Portfolio)
│
├── 💼 PORTFOLIOS (sub-nav: My Portfolios | Portfolio Requirements)
│   │
│   ├── My Portfolios
│   │   ├── Portfolio List View (2-column grid; Cards with KPIs, Metadata, Mini Equity, Compliance Status)
│   │   ├── Portfolio Builder (Name, Strategies, Position Sizing)
│   │   └── Portfolio Detail View
│   │       ├── Combined Analysis Tab (Equity, Correlation, Drawdown, P&L)
│   │       ├── Prop Firm Compliance Tab (Rule Sets, Checklist)
│   │       └── Deploy Tab (Alerts, Webhooks)
│   │
│   └── Portfolio Requirements
│       ├── Requirement Set List (TTP, FTMO built-in + custom)
│       └── Requirement Set Editor (Rules, Thresholds)
│
├── 🔔 ALERTS (sub-nav: Alerts & Signals | Webhook Templates)
│   │
│   ├── Alerts & Signals
│   │   ├── Strategy Alerts Tab
│   │   ├── Portfolio Alerts Tab
│   │   ├── Outbound Webhooks Tab
│   │   └── Inbound Webhooks Tab (placeholder)
│   │
│   └── Webhook Templates
│       ├── Template List (by category)
│       ├── Default Templates (TTP Buy/Sell/Close)
│       └── Custom Template CRUD
│
├── 🏪 MARKETPLACE (Future)
│
└── ⚙️ SETTINGS
    ├── Chart Defaults
    │   └── Default Visible Candles (Tight 50, Close 100, Default 200, Wide 400, Full)
    ├── Default Triggers
    │   ├── Default Entry Trigger (user's preferred starting entry for new strategies)
    │   └── Default Exit Trigger (user's preferred starting exit for new strategies)
    ├── Default Risk Management
    │   ├── Default Stop Loss (method + parameters; applied to new strategies)
    │   └── Default Target (method + parameters; applied to new strategies)
    ├── Development (mock data mode only)
    │   └── Data Seed (random seed for mock data generation)
    └── Connections (Alpaca, webhooks, etc.) — future
```

### 7.2 Core User Journeys

**Journey 1: New User Creates First Strategy**
```
Dashboard → "New Strategy" → Strategy Builder (inline bar: configure, Load Data)
→ Main area: review KPIs, drill-down to optimize → Save Strategy (bottom)
→ My Strategies (auto-navigates to saved strategy detail)
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
→ Strategy Builder (inline bar pre-populated, data auto-loaded)
→ Adjust triggers/confluence → Save Strategy (bottom)
```

**Journey 5: Build Strategy from Scratch via Optimization Workflow** *(Phase 9)*
```
Strategy Builder → Load Data → Entry Trigger tab
→ Drill down on entry triggers (default exit: N-bar close)
→ Select best entry → Exit Triggers tab
→ Drill down on exit triggers paired with selected entry
→ Timeframe Conditions tab → layer in timeframe confluences
→ General Conditions tab → layer in session/calendar/news filters
→ Stop Loss tab → compare stop configurations across pack variations
→ Take Profit tab → compare target configurations across pack variations
→ Review Optimizable Variables box → Save
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
14. [x] Execution model expansion — 4 stop loss methods, 5 take profit methods, up to 3 exit triggers, execution type metadata, `[C]`/`[I]` labels, full backward compatibility
15. [x] Navigation refactor — top horizontal nav bar with 5 sections and sub-nav radios; sidebar becomes context-aware config panel
16. [x] Strategy Builder single-page — collapsed 3-step wizard into single page with sidebar config panel; Strategy Origin placeholder for Phase 12
17. [x] KPI audit and enhancement — Max R Drawdown primary KPI, secondary KPIs expander (11 extended metrics), card-style drill-down/auto-search, unified infinity/format display, strategy cards (5 KPIs), portfolio cards (4 KPIs), sort options (Daily R, Max R DD)
18. [x] Strategy detail tab restructuring — split "Equity & Charts" / "Backtest Results" into 7-tab layout: Equity & KPIs, Equity & KPIs (Extended), Price Chart, Trade History, Confluence Analysis, Configuration, Alerts; KPIs moved into tabs; Extended tab loads configurable longer lookback (90–1825 days, default 365) with adjustable slider; Price Chart tab has full indicators + trade table; Trade History tab has clean chart + trade table; applies to both backtest-only and forward test views
19. [x] Per-chart visible candles selector — compact selectbox above every price chart (7 call sites); `@st.fragment` wrapper prevents full-page rerun on selection change (preserves active tab); options: Default, 50, 100, 200, 400, All
20. [x] 2-column card grid and trigger badges — strategy and portfolio lists in 2-column grid with stacked cards; strategy cards show Entry/Exit, Stop/Target, and Confluence badges below KPIs; strategy detail header adds Stop and Target metadata row; default strategy name shortened to `"{symbol} {direction} - {id}"`
21. [x] Confluence drill-down enhancements — unified search bar + filter dialog (`@st.dialog`) across Drill-Down and Auto-Search modes; text search filters by indicator/combination name; filter lightbox with sort (6 KPIs + direction), min thresholds (Trades, Win Rate, Profit Factor, Daily R, R²), and Auto-Search max depth; all settings persisted in `confluence_filters` session state; replaces hardcoded `min_trades=3` and inline sort dropdown
22. [x] "Exit After N Candles" bar count exit trigger — new `bar_count` EXIT-only template in TEMPLATES (no indicators/outputs); hybrid approach with `bar_count_exit` parameter in `generate_trades()` trade loop (can't pre-compute as DataFrame column); default 4 candles; priority 3 in exit chain (stop > target > bar_count > signal); migration auto-appends `bar_count_default` group for existing users; validation prevents multiple bar count exits per strategy
23. [x] 6-tab optimization drill-down with actionable cards — replaced single "Confluence Drill-Down" panel with 6-tab layout (Entry, Exit, TF Conditions, General, Stop Loss, Take Profit); Entry tab: per-trigger KPI cards with "Replace" button (swaps sidebar entry trigger via pending state pattern); Exit tab: Drill-Down mode with per-trigger KPI cards and "Add" button (appends to exits, up to 3) + Auto-Search mode with `find_best_exit_combinations()` testing combos of 1-3 exits and "Replace" button (swaps all exits); TF Conditions tab: existing drill-down with checkbox→"Add" button conversion + Auto-Search "Apply"→"Replace" rename; tabs 4-6 placeholder; `analyze_entry_triggers()` and `analyze_exit_triggers()` helpers use full current strategy config (not isolated baselines); compact toolbar with `[Search][Action][⚙]` layout; Streamlit widget key conflict resolved via pending session state pattern (`pending_entry_trigger`, `pending_add_exit`, `pending_replace_exits` consumed before sidebar selectbox instantiation)
24. [x] Optimizable Variables box and per-tab active tags — collapsible `st.expander("Optimizable Variables")` positioned below strategy title showing all 6 variable categories (Entry, Exits, TF Conditions, General placeholder, Stop Loss, Take Profit) with ✕ remove buttons; replaces old "Active Confluence Filters" tag bar; exit removal via `pending_remove_exit_idx` with shift-down logic; target removal via `pending_remove_target`; per-tab active tags: Entry tab shows current trigger caption, Exit tab shows removable exit trigger chips, TF Conditions tab shows removable confluence chips with "Clear All"; all tag removals sync with Optimizable Variables box via shared `selected_confluences` set and pending state patterns

25. [x] Confluence Packs rename, General Packs, and Risk Management Packs — renamed "Confluence Groups" to "Confluence Packs" across all user-facing labels for marketability; added sub-navigation (TF Confluence, General, Risk Management); new `general_packs.py` module with 4 templates (Time of Day, Trading Session, Day of Week, Calendar Filter), condition evaluation functions (`evaluate_condition()` dispatcher), and full CRUD with `config/general_packs.json`; new `risk_management_packs.py` module with 5 templates (ATR-Based, Fixed Dollar, Percentage, Swing, Risk:Reward), dual-output architecture (`get_stop_config()` + `get_target_config()` from shared parameters), builder functions, and full CRUD with `config/risk_management_packs.json`; both management pages have 5-tab detail panels (Parameters, Outputs, Preview, Code, Danger Zone); General Pack previews: extended hours mock data toggle, condition state change markers on price chart, state transition table, distribution metrics; Risk Management Pack previews: configurable entry/exit trigger selectors from TF Confluence Packs, trade chart with stop/target levels, KPI summary, trade details; Code tabs show `inspect.getsource()` for evaluation/builder functions; wired drill-down tabs 4-6 (General shows enabled packs with outputs, Stop Loss and Take Profit run `analyze_risk_management()` multi-backtest with KPI comparison cards); extended hours support in `mock_data.py` (`extended_hours` parameter for 4:00 AM - 8:00 PM bar generation); `extra_markers` parameter on `render_chart_with_candle_selector()` and `render_price_chart()` for condition state annotations

26. [x] Phase 9 completion — trade tagging, general drill-down, SL/TP replace buttons, strategy schema. General pack conditions tagged on trades as `GEN-{PACK_ID}-{STATE}` records via `general_columns` param threaded through `get_confluence_records()` → `generate_trades()` → all 11 call sites. General Conditions tab transformed into full drill-down with KPI cards and "Add" buttons; GEN- records filtered out of TF Conditions tab. SL/TP drill-down cards gain "Replace" buttons using `pending_stop_config`/`pending_target_config` with widget key deletion pattern; `(current)` label on active config. Strategy save splits `confluence` and `general_confluences`; load merges both (backward compatible). Optimizable Variables box partitions TF and General columns by GEN- prefix. 4 remaining polish items (trigger params, variation tags, caching, lazy loading) deferred to Phase 10.
27. [x] Settings page and sidebar-to-inline refactor — Settings nav page with Chart Defaults, Default Triggers, Default Risk Management, and Development (data seed) sections; all Strategy Builder inputs moved from sidebar to inline config bar (Row 1: Method/Ticker/TF/Dir/Lookback/Name/FT/AL/Load, Row 2 expander: Entry/Exit/Stop/Target); status line with bar estimate + validation errors; Save button moved to bottom of page; sidebar stripped to app title + data source only; strategy detail header expanded with general confluences; Extended KPIs tabs gain full lookback mode selector (Days/Bars/Date Range)

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

### Design Decisions (Phase 8 — Execution Model)
- **Five exit mechanisms, three categories** — Built-in exits (stop loss required, take profit optional) are price-level-based and checked every bar. Signal-based exits (up to 3 exit triggers from confluence groups) fire at bar close when any one triggers. Priority: stop > target > exit triggers. First to fire wins.
- **Nested config dicts over flat fields** — `stop_config: {"method": "atr", "atr_mult": 1.5}` is extensible (add new methods without schema changes) vs. flat fields like `stop_atr_mult`, `stop_dollar_amount`, etc. Backward compat: if `stop_config` is absent, engine builds it from legacy `stop_atr_mult`.
- **Up to 3 exit triggers** — Balances flexibility with UI simplicity. Any-of-3 semantics (first to fire wins) covers common multi-signal exit patterns without complex logic operators.
- **Execution type as metadata only (for now)** — All current triggers are `bar_close`. Adding `execution` field to TriggerDefinition and `[C]`/`[I]` labels builds the infrastructure without implementing intra-bar pricing, which requires UT Bot or other price-level triggers to be meaningful.
- **Same-bar conflict = worst outcome** — When stop and target are both breachable within a bar (high > target, low < stop), assume stop hit first. Keeps backtests pessimistic, giving strategies a built-in margin of safety.

### Phase 8: QA, Polish & UX — "Get Live-Tradeable"
*Comprehensive review pass and UX improvements — the gate to live trading with real money.*

**Bug Fixes (from Feb 10 testing) — COMPLETED (Feb 10, 2026):**
- [x] Edit strategy navigation broken — `load_strategy_into_builder()` was missing `nav_target = "Strategy Builder"`; added before `st.rerun()` so Edit now navigates to the builder
- [x] Fixed dollar stop exit reason incorrect — was a display bug, not logic: Step 2 Trade List showed `exit_trigger` column (trigger ID) instead of `exit_reason` column ("stop_loss", "target", etc.); swapped column and added proper labels
- [x] Step 1 state lost on "Back" from Step 2 — `data_days` and `data_seed` widgets had hardcoded defaults; now read from `st.session_state.strategy_config` so values persist when navigating back

**KPI Accuracy Audit — COMPLETED (Feb 10, 2026):**
- [x] Fix Daily R calculation — `calculate_kpis()` now accepts `total_trading_days` param; all call sites pass `count_trading_days(df)` which counts unique trading days in the full data period (not just days with exits); makes Daily R a true capital efficiency metric
- [x] Add equity curve smoothness metric — **R² of equity curve** added to `calculate_kpis()` return dict; displayed on Strategy Builder Step 2, Step 3 summary, live backtest, saved KPIs, forward test comparison (with delta), confluence drill-down (with sort option), and auto-search results
- [x] General KPI accuracy audit — comprehensive audit of all KPI display locations; added **Max R Drawdown** (peak-to-trough in cumulative R space) as new primary KPI to `calculate_kpis()`; added `calculate_secondary_kpis()` for extended metrics (win/loss counts, best/worst trade, avg win/loss, max consecutive wins/losses, payoff ratio, recovery factor, longest DD trades); added "Extended KPIs" expander to Strategy Builder, strategy detail backtest, and forward test comparison views
- [x] Validate KPI consistency — standardized all strategy views to 8 primary KPIs (Trades, WR, PF, Avg R, Total R, Daily R, R², Max R DD); strategy cards show 5 KPIs (WR, PF, Daily R, Trades, Max R DD); dashboard mirrors card KPIs; all infinity displays unified to "∞"; all win rate formats unified to `:.1f%`; portfolio cards and dashboard add Avg Daily P&L; confluence drill-down and auto-search restructured as cards with 6 KPIs (Trades, PF, WR, Avg R, Daily R, R²); added sort options for Daily R and Max R DD on My Strategies page

**QA Sandbox Page** — *moved to Phase 10*

**QA & Verification** — *moved to Phase 10*

**Confluence Drill-Down Enhancements — COMPLETED (Feb 11, 2026):**
- [x] Card-style result layout — replaced single-row display with `st.container(border=True)` cards: confluence name on top row (with checkbox for drill-down / depth badge for auto-search), 6 KPIs on bottom row (Trades, PF, WR, Avg R, Daily R, R²); applies to both Drill-Down and Auto-Search modes
- [x] Sort by any KPI — `@st.dialog` filter lightbox with 6 sort options (Profit Factor, Win Rate, Daily R, R² Smoothness, Trades, Avg R) plus ascending/descending direction toggle; replaces inline sort selectbox
- [x] Advanced filtering — min threshold inputs for key KPIs (Min Trades, Min Win Rate, Min Profit Factor, Min Daily R, Min R²) in filter dialog; replaces hardcoded `min_trades=3` with user-configurable value; all filter settings persisted in `confluence_filters` session state across mode switches and reruns
- [x] Text search — search bar above results filters by indicator/combination display name (case-insensitive); shared across Drill-Down and Auto-Search modes
- [x] Unified toolbar — both modes share identical search bar + filter button layout; Auto-Search filter dialog additionally exposes Max Factors depth slider
- [x] Auto-Search parity — Auto-Search results now display the same 6-KPI card format as Drill-Down, with depth badge and Apply button; `top_n` increased to 50 for broader initial search with UI-side filtering to 20

**Backtest Settings Overhaul** — *moved to Phase 10*

**Execution Model & Stop/Target Expansion — COMPLETED (Feb 10, 2026):**
- [x] Expand stop loss methods — Strategy Builder Step 1 "Risk Management" section with selectbox:
  - **ATR** (default) — `entry ± ATR × multiplier`
  - **Fixed Dollar** — `entry ± $X`
  - **Percentage** — `entry ± (entry × X%)`
  - **Swing Low/High** — `min(low[lookback]) - padding` / `max(high[lookback]) + padding`
- [x] Expand take profit / exit target methods (optional, default None):
  - **Risk:Reward** — `entry ± (risk × R:R ratio)`
  - **ATR** — `entry ± ATR × multiplier`
  - **Fixed Dollar** — `entry ± $X`
  - **Percentage** — `entry ± (entry × X%)`
  - **Swing High/Low** — `max(high[lookback]) + padding` / `min(low[lookback]) - padding`
- [x] Multiple exit triggers — up to 3 signal-based exit triggers from confluence groups per strategy; any-of-3 fires → exit at bar close; add/remove UI with duplicate and entry-conflict validation
- [x] Nested config dicts — `stop_config` and `target_config` dicts in strategy schema (e.g., `{"method": "atr", "atr_mult": 1.5}`); backward-compatible with legacy `stop_atr_mult` field
- [x] Execution type metadata — `"execution": "bar_close"` added to TriggerDefinition dataclass and all TEMPLATES trigger dicts; infrastructure ready for future `"intra_bar"` triggers
- [x] Execution type labels — `[C]` (bar close) / `[I]` (intra-bar) suffix on trigger names in Strategy Builder entry/exit dropdowns
- [x] Same-bar conflict resolution — stop checked before target before signal triggers; worst-outcome assumption documented in engine
- [x] Display helpers — `format_stop_display()`, `format_target_display()`, `format_exit_triggers_display()` used across strategy detail pages (backtest tab, forward test tab, saved KPIs), Step 2 header, and Step 3 summary
- [x] Alert engine updated — `alerts.py` uses `calculate_stop_price()` for all stop methods; supports multi-exit trigger detection
- [x] Full backward compatibility — no migration needed; existing strategies load and backtest correctly via fallback logic
- [ ] Intra-bar entry pricing — deferred until UT Bot or other `[I]` triggers are implemented:
  - **Price-level triggers** `[I]` — fill at the trigger price (e.g., UT Bot trail cross fills at the trail price using bar high/low)
  - **Indicator-state triggers** `[C]` — fill at bar close (e.g., EMA crossover, MACD cross, RVOL threshold)
- [ ] *(Optional)* Strategy-level execution mode — deferred:
  - **Conservative `[C]`** (default) — all entries/exits at bar close in backtests
  - **Intra-bar `[I]`** — entries/exits at estimated trigger price using bar high/low in backtests

**UX Improvements — Quick Wins COMPLETED (Feb 10, 2026):**
- [x] Oscillator panes on Strategy Builder Step 2 chart — new `build_secondary_panes()` helper detects MACD/RVOL groups from enabled groups, deduplicates (one pane per type), and passes `secondary_panes` to `render_price_chart()`; also refactored Confluence Analysis and Preview tabs to use the same helper
- [x] Oscillator panes on Strategy Detail main chart — Live Backtest and Forward Test price charts now auto-include oscillator panes via `build_secondary_panes()`
- [x] Save navigates to strategy detail — after save/update, sets `viewing_strategy_id` + `nav_target = "My Strategies"` and calls `st.rerun()` to land on the saved strategy's detail page
- [x] Step 1 state preservation — all widgets now read from `edit_config` (session state); `risk_per_trade` and `starting_balance` were the last two hardcoded defaults, now fixed
- [x] "Create New Strategy" button on My Strategies page — follows Portfolios page pattern with `st.columns([4, 1])` header layout

**Navigation & Strategy Builder Refactor — COMPLETED (Feb 10, 2026):**
- [x] Top navigation bar — `st.radio(horizontal=True)` with 5 sections: Dashboard, Confluence Groups, Strategies, Portfolios, Alerts; sub-nav radios for multi-page sections (Strategies: Builder/My Strategies; Portfolios: My Portfolios/Requirements; Alerts: Signals/Templates)
- [x] Sidebar refactored to context-aware config panel — app title + data source + chart presets as base; Strategy Builder adds its own sidebar config panel
- [x] Strategy Builder single-page — collapsed 3-step wizard into single page; all configuration in sidebar config panel (origin, data, triggers, risk, save); "Load Data" as only gate; parameter changes re-run backtest on cached data; save form in sidebar
- [x] Strategy Origin field — selectbox at top of sidebar config panel (`["Standard"]` only for now, Phase 12 placeholder); saved as `strategy_origin: "standard"` in strategy dict; backward-compatible via `.get('strategy_origin', 'standard')`
- [x] `NAV_TARGET_MAP` — translates old 8-page nav targets to new section + sub-page pairs; preserves all existing programmatic navigation call sites
- [x] Removed step indicator CSS and `step` session state — replaced with `builder_data_loaded` boolean
- [x] Fix programmatic navigation (Edit Strategy, New Strategy buttons) — `st.radio` `index` parameter is ignored after first user interaction; switched to explicit `key` params (`main_nav`, `sub_nav_*`) with direct `st.session_state[key]` writes for reliable programmatic nav

**UX Improvements — COMPLETED (Feb 11, 2026):**
- [x] Per-chart visible candles adjustment — compact selectbox ("Default", 50, 100, 200, 400, "All") above every price chart; `render_chart_with_candle_selector()` wrapper uses `@st.fragment` so changing the selector only reruns the chart, not the full page (preserves active tab); 7 call sites (Strategy Builder, backtest Price Chart, backtest Trade History, Confluence Analysis, forward test Price Chart, forward test Trade History, Confluence Group Preview)
- [x] Strategy name and trigger display improvements — default name shortened to `"{symbol} {direction} - {id}"`; strategy cards and detail header display Entry, Exit, Stop, and Target as caption-style badges; detail header adds second metadata row with Stop and Target
- [x] 2-column card layout — both strategy and portfolio lists render in 2-column grid with stacked card layout; strategy cards: Name, Status, Mini Equity Curve, 5 KPIs (WR, PF, Daily R, Trades, Max DD), Entry/Exit badges, Stop/Target badges, Confluence tags ("None" placeholder for uniform height), Action buttons; portfolio cards: Name, Metadata (strategies, balance, scaling, avg risk/trade, trades/day), Strategy names, Mini Equity Curve, 4 KPIs (P&L, Max DD, WR, Avg Daily), Requirement summary badge, Action buttons

**UX Improvements — Remaining** — *moved to Phase 10*

### Design Decisions (Phase 8 — QA & UX)
- **Daily R as capital efficiency metric** — `total_r / all_trading_days` (not just days with exits) answers "where should I park my capital for the best risk-adjusted return?" A strategy that trades once per week but earns 5R should show lower Daily R than one earning 3R every day, because capital is idle in the first scenario.
- **R-squared for equity curve smoothness** — Linear regression R² of the cumulative equity curve. R² ≈ 1.0 means steady, predictable growth. R² < 0.7 means choppy or dependent on outlier trades. Chosen over Ulcer Index/Serenity Index for Phase 8 because it's intuitive (0–1 scale), fast to compute, and directly answers "is this strategy consistently profitable or just lucky?" The full suite (Ulcer, Serenity, etc.) deferred to Phase 11.
- **QA Sandbox as dev-only page** — Not exposed to end users; exists purely for developer QA. Validates that stop/target calculations, trade generation, and signal detection behave as intended. Charts plot stop/target price levels as horizontal lines per trade for visual verification. This replaces ad-hoc testing with a systematic, repeatable QA workflow.
- **Card-style drill-down over row tables** — Showing multiple KPIs per confluence combination requires more vertical space than a 5-column table row allows. Cards give room for 6+ KPIs while keeping the combination name prominent. The same card format is reused for both Drill-Down (single-factor) and Auto-Search (multi-factor combinations).
- **`st.radio(horizontal=True)` over `st.tabs()`** — `st.tabs()` renders ALL tab contents on every re-run (even hidden tabs), which would run expensive backtests and data loads when viewing other pages. `st.radio(horizontal=True)` only renders the selected page and supports programmatic selection via `index` for the existing `nav_target` pattern.
- **Single-page Strategy Builder** — Eliminates the back-navigation state loss problem entirely. All parameters are visible and editable at all times via inline config bar. "Load Data" is the only gate (needed for symbol/timeframe changes). Trigger/risk changes re-run backtest automatically on cached data via Streamlit's natural re-run behavior. Save button at bottom of page after analysis.
- **`builder_data_loaded` boolean over `step` integer** — The 3-step flow is gone; the only meaningful state is "has data been loaded?" This boolean gates the main area content (KPIs, charts) while allowing the config bar to always be visible.
- **Strategy Origin as Phase 12 placeholder** — Adding the selectbox now (with only "Standard" option) establishes the UI pattern and schema field without implementing the full feature. Existing strategies default to `"standard"` via `.get('strategy_origin', 'standard')` — no migration needed.

- **Max R Drawdown as strategy-level risk metric** — Peak-to-trough drawdown in cumulative R space, analogous to portfolio's dollar-based Max Drawdown but expressed in R-multiples. Named "Max R DD" to distinguish from portfolio's "Max DD". Computed from `np.maximum.accumulate(cumulative_r) - cumulative_r`. A strategy with Max R DD of -3.2R had a worst losing streak that erased 3.2 risk units from peak equity. Added to `calculate_kpis()` and saved to strategies.json for card display.
- **Secondary KPIs as live-computed expander** — Extended metrics (win/loss counts, best/worst trade, avg win/loss, streaks, payoff ratio, recovery factor, longest DD) are always computed live from `trades_df`, never saved to JSON. Displayed in a collapsed `st.expander("Extended KPIs")` below primary KPI rows. This avoids bloating strategies.json with 11+ additional fields while keeping the metrics available in all detail views. Phase 11's advanced statistical metrics (Sharpe, Sortino, etc.) will extend this pattern.
- **Strategy cards: Daily R over Total R** — Strategy cards prioritize Daily R because it enables apples-to-apples comparison across strategies with different data periods. A 30-day strategy with 10R total and a 90-day strategy with 20R total aren't directly comparable; Daily R normalizes for time.
- **R-based vs dollar-based drawdown naming** — Strategy "Max R DD" uses R-multiples (risk-normalized). Portfolio "Max DD" uses dollar/percentage (account-level). The naming distinction prevents confusion between the two scopes.
- **7-tab strategy detail layout** — Separated "Equity & Charts" into distinct tabs for three reasons: (1) KPIs belong with their equity curves, not floating above tabs; (2) the extended backtest needs its own data load and KPI computation at a different date range; (3) price charts with indicators and clean trade history charts serve different purposes (indicator analysis vs. clean entry/exit review) and deserve their own space. The extended tab has an adjustable slider (90–1825 days) so users can explore different historical depths on the fly.
- **Extended lookback as per-strategy default + per-view override** — The Strategy Builder saves a default `extended_data_days` (used as the slider's initial value on the detail page). The slider on the Extended tab lets users adjust without editing the strategy. This balances convenience (sensible default) with flexibility (situational exploration).
- **`@st.fragment` for per-chart candle selector** — Without `@st.fragment`, changing a selectbox inside `st.tabs()` triggers a full page rerun which resets the active tab to the first one. Wrapping the candle selector + chart in `@st.fragment` isolates the rerun to just the chart fragment, preserving tab state. Each `render_chart_with_candle_selector()` call creates its own fragment instance.
- **2-column stacked cards over side-by-side split** — At full width, strategy cards used a `[3, 2]` info/chart split. At half width in a 2-column grid, that split wastes horizontal space. Stacking vertically (name → status → equity curve → KPIs → badges → buttons) uses the narrower column more efficiently. Entry/Exit, Stop/Target, and Confluence badges placed below KPIs so the most scannable info (name, status, equity curve, KPIs) is at top.
- **Confluence "None" placeholder** — Cards without confluence conditions show "Confluence: None" to maintain uniform card height across the grid, preventing visual misalignment between adjacent cards.
- **`@st.dialog` filter lightbox over inline controls** — Confluence drill-down previously used inline sort selectbox (Drill-Down) and inline sliders (Auto-Search), creating inconsistent UIs. Moving all filter/sort controls into a shared `@st.dialog` lightbox keeps the main view clean (just search bar + filter button), unifies the two modes, and provides room for KPI threshold inputs without cluttering the card results area. Filter state persists in `confluence_filters` session state so settings survive mode switches and page reruns.

**Phase 8 core work complete.** Remaining items (QA Sandbox, Backtest Settings, UX utility buttons) deferred to Phase 10 — they depend on Phase 9's schema changes (general confluence groups, stop/target packs, trade tagging). QA should validate the final data model, and Backtest Settings caching should account for multi-backtest patterns introduced by stop/target pack drill-downs.

### Phase 9: Optimization Workflow — "Systematic Strategy Construction"
*Transform strategy building from manual configuration into a guided, data-driven optimization sequence. Users isolate and evaluate each variable category independently, layering decisions in a logical order.*

**Core Concept:** Every strategy is composed of 6 optimizable variable categories, evaluated in sequence:
1. **Entry Trigger** — the signal that opens a position
2. **Exit Triggers** — one or more signals; whichever fires first closes the position
3. **Timeframe Confluence Conditions** — interpretation states from timeframe-based groups (existing)
4. **General Confluence Conditions** — interpretation states from non-timeframe groups (time of day, session, calendar, news, etc.)
5. **Stop Loss** — parameterized stop configurations packaged as optimizable packs
6. **Take Profit** — parameterized target configurations packaged as optimizable packs

**The fundamental model:** Enter when the entry trigger fires AND all active confluence conditions (timeframe + general) are aligned. Exit when any exit trigger fires, stop is hit, or target is hit — whichever comes first. This model is unchanged from today; Phase 9 adds the tooling to systematically find the best values for each variable.

---

**General Confluence Packs (sub-page under Confluence Packs):** ✓
- [x] New "General" sub-page — same template/version/pack structure as TF Confluence but for non-timeframe variables
- [x] General pack template framework — 4 templates that produce categorical condition states from non-chart data:
  - **Time of Day** — configurable time window (start/end hour:minute)
  - **Trading Session** — pre-market, regular, after-hours, extended session filter
  - **Day of Week** — per-day allow/block toggles (Mon–Fri)
  - **Calendar Filter** — block FOMC/NFP/OpEx days with configurable buffer
  - **News/Event** — extensible framework for external data feeds (architecture ready, future implementation)
  - **Market Regime** — broad market conditions from index data (future)
- [x] Condition evaluation system — `evaluate_condition()` dispatcher with per-template evaluators (`_eval_time_of_day`, `_eval_trading_session`, `_eval_day_of_week`, `_eval_calendar_filter`) that return condition state Series
- [x] Management page — pack list by category, + New Pack dialog, detail panel with 5 tabs (Parameters, Outputs, Preview, Code, Danger Zone)
- [x] Preview tab — extended hours toggle, price chart with condition state change markers (colored circles), state transition table, distribution metrics
- [x] Code tab — `inspect.getsource()` for evaluation functions
- [x] Extended hours mock data — `generate_mock_bars(extended_hours=True)` generates 4:00 AM – 8:00 PM bars for session-based preview validation
- [x] Template structure — same `TEMPLATES` dict pattern with `parameters_schema`, `outputs`, `output_descriptions`, `condition_logic`, `triggers`
- [x] General confluence record format — `"GEN-{PACK_ID}-{STATE}"` prefix distinguishes general records from timeframe records (e.g., `"GEN-TOD_NY_OPEN-IN_WINDOW"`); `format_confluence_record()` resolves GEN- records to pack display names
- [x] Trade tagging — `get_confluence_records()` extended with `general_columns` param; trades tagged with general confluence records at entry time alongside timeframe records in same `confluence_records` set

**Risk Management Packs (sub-page under Confluence Packs — replaces separate Stop Loss / Take Profit Packs):** ✓
- [x] New "Risk Management" sub-page — each pack bundles both stop-loss AND take-profit configurations from shared parameters
- [x] Dual-output architecture — `get_stop_config()` and `get_target_config()` methods generate both configs from one parameter set, analogous to how TF Confluence Packs output both triggers AND conditions
- [x] 5 templates: ATR-Based (volatility), Fixed Dollar (fixed), Percentage (fixed), Swing (structure), Risk:Reward (composite — any stop method paired with R:R target)
- [x] Builder function pattern — `build_stop` and `build_target` function references stored in TEMPLATES dict, called by dataclass methods
- [x] Management page — pack list by category, + New Pack dialog, detail panel with 5 tabs (Parameters, Outputs, Preview, Code, Danger Zone)
- [x] Preview tab — configurable entry/exit trigger selectors from TF Confluence Packs, generates trades with pack's stop/target config, chart with trade markers, KPI summary, trade details table
- [x] Code tab — active config display, builder function source via `inspect.getsource()`, dataclass method source
- [x] Multi-backtest computation — `analyze_risk_management()` helper varies either stop or target config across enabled RM packs while holding the other fixed; Stop Loss and Take Profit drill-down tabs display KPI comparison cards
- [x] Built-in packs — ATR Default (1.5x/3x), ATR Tight (1x/2x), Fixed $1/$2, Percentage 0.5%/1%, Swing 2R
- [x] Custom packs — users can create custom packs with arbitrary parameter combinations
- [x] Conditional parameter visibility — `rr_ratio` composite template only shows params relevant to selected stop method
- [x] Format helpers — `format_stop_summary()`, `format_target_summary()`, `format_parameters()` for display across UI

**"Exit After N Candles" Default Exit:** ✓
- [x] New interpreter/trigger — "Bar Count Exit" that fires after N candles since entry (configurable N)
- [x] Default exit trigger for new strategies — when creating a new strategy, start with "Exit after N candles" as the default exit trigger
- [x] Purpose — isolates entry trigger quality by removing exit signal noise; if an entry consistently produces positive movement within N bars, the entry has genuine edge
- [x] Configurable N — parameter on the trigger (e.g., 1, 2, 3, 4, 5, 10, 20 bars)
- [x] Works as a confluence group template — follows existing template/version structure so it appears in exit trigger dropdowns

**Optimizable Variables Box (Strategy Builder):** ✓
- [x] Collapsible `st.expander("Optimizable Variables")` positioned below strategy title, above KPI dashboard
- [x] Displays all active variables organized by 6 columns: Entry, Exit(s), TF Conditions, General, Stop Loss, Take Profit
- [x] Exit triggers have "✕" remove buttons (hidden when only 1 exit); removal uses `pending_remove_exit_idx` with shift-down logic for specific index removal
- [x] TF Conditions have "✕" remove buttons per confluence, synced with `selected_confluences` set
- [x] Take Profit has "✕" remove button (sets target to None via `pending_remove_target`)
- [x] Entry and Stop Loss display-only (always required)
- [x] Replaces old "Active Confluence Filters" tag bar
- [x] ~~Trigger parameters visible and expandable~~ — deferred to Phase 10

**Active Tags (per-tab):** ✓
- [x] Per-tab tag chips positioned between toolbar and drill-down/auto-search content in each tab
- [x] Entry tab: shows current entry trigger name as caption
- [x] Exit tab: shows current exit triggers as removable chips (✕ with pending removal pattern); non-removable caption for single exit or bar_count
- [x] TF Conditions tab: shows selected TF conditions (non-GEN-) as removable chips with "Clear TF" button (preserves GEN- selections)
- [x] General Conditions tab: shows selected GEN- conditions as removable chips with "Clear Gen" button (preserves TF selections)
- [x] Tags sync with Optimizable Variables box — both operate on shared `selected_confluences` set and pending state patterns; UI partitions by GEN- prefix

**6-Tab Optimization Drill-Down:**
- [x] Replace current single drill-down panel with 6 tabs matching the optimization sequence
- [x] Each tab uses the same search bar + filter dialog pattern (from Phase 8 drill-down enhancements)
- [x] **Entry Trigger tab** — shows KPI cards for each available entry trigger using current strategy config; "Replace" button swaps sidebar entry trigger; compact `[Search][Analyze][⚙]` toolbar
- [x] **Exit Triggers tab** — Drill-Down mode with per-trigger KPI cards and "Add" button (appends up to 3); Auto-Search mode with `find_best_exit_combinations()` and "Replace" button; compact toolbar with mode-aware action button
- [x] **Timeframe Conditions tab** — existing confluence drill-down with "Add" button (replaces checkbox) + Auto-Search with "Replace" button (replaces "Apply"); Auto-Search gets compact toolbar with "Search" action button
- [x] **General Conditions tab** — full drill-down with KPI cards for each GEN- condition state; "Add" button adds condition to `selected_confluences`; active tags with "Clear Gen" button; search + filter toolbar; GEN- records filtered out of TF Conditions tab
- [x] **Stop Loss tab** — search/analyze/filter toolbar; `analyze_risk_management()` multi-backtest across enabled RM pack stop configs (holding current target fixed); KPI comparison cards with pack name; "Replace" button swaps sidebar stop config via `pending_stop_config` pattern; `(current)` label on matching config
- [x] **Take Profit tab** — same pattern as Stop Loss; varies target config across enabled RM packs (holding current stop fixed); KPI comparison cards; "Replace" button swaps sidebar target config via `pending_target_config` pattern; `(current)` label on matching config
- [x] Per-tab drill-down with full-config KPIs — each drill-down card shows KPIs based on the full current strategy config where the only change is the one thing that card represents; all 6 tabs follow this pattern
- [x] Auto-Search available on Entry (N/A — single trigger), Exit, and TF Conditions tabs

**Data Model Changes:**
- [x] Extend `confluence_records` set on trades — GEN- prefixed general confluence records included alongside timeframe records via `general_columns` param threading
- [x] ~~Stop/target variation tags on trades~~ — deferred to Phase 10
- [x] Strategy schema additions:
  - `general_confluences: List[str]` — selected general confluence records (GEN- prefixed); saved separately from TF confluences, merged on load
  - `stop_pack_id: Optional[str]` — reference to the stop loss pack used for optimization (if any) — deferred
  - `target_pack_id: Optional[str]` — reference to the take profit pack used for optimization (if any) — deferred
- [x] New config files:
  - `config/general_packs.json` — general pack definitions (template/version/parameters structure)
  - `config/risk_management_packs.json` — risk management pack definitions (dual stop+target configs)
- [x] Backward compatibility — existing strategies without `general_confluences` load correctly (defaults to empty set); `confluence_set` construction merges both fields

**Performance Considerations:**
- [x] General confluence records are cheap to compute (clock/calendar lookups) — no performance concern; evaluated in `prepare_data_with_indicators()`
- [x] ~~Multi-backtest progress indicator + caching~~ — deferred to Phase 10
- [x] ~~Lazy tab loading~~ — deferred to Phase 10

### Design Decisions (Phase 9 — Optimization Workflow)
- **Sequential optimization over simultaneous** — Evaluating all 6 variable categories at once creates a combinatorial explosion. Sequential evaluation (entry → exit → conditions → stop → target) is tractable, intuitive, and mirrors how experienced traders build strategies: find an edge first, then refine execution.
- **"Exit After N Candles" as entry quality isolator** — By fixing the exit to a simple time-based close, entry trigger quality is measured in isolation. If entry + 4-bar-exit is profitable, the entry has genuine predictive power. More sophisticated exits can only improve on that baseline. This prevents the common mistake of attributing edge to the entry when it actually comes from a clever exit.
- **General Confluence Groups as separate sub-page** — Time-of-day, session windows, and calendar conditions don't derive from chart indicators. They need different interpreter logic (clock/calendar lookups vs. indicator math), different parameter schemas, and a different mental model. A separate sub-page under Confluence Groups preserves the pack/template structure while acknowledging the conceptual difference.
- **Stop/target packs as confluence-group-like entities** — Packaging stop/target variations into "packs" with the same template/version structure means the drill-down UI can treat them identically to confluence interpretations. A stop variation is just another interpretation state that can be evaluated for KPI impact. This keeps the architecture consistent across all 6 variable categories.
- **Multi-backtest for stop/target drill-down (Option A)** — Pre-computing trades across all pack variations gives true apples-to-apples KPI comparison. The alternative (showing one config at a time) doesn't let users see "ATR 1.5x: PF 2.3 vs ATR 2.0x: PF 1.8" side by side. The computational cost is bounded by pack size (typically 5-10 variations) and can be cached aggressively.
- **Active tags above mode toggle** — Tags represent selections that apply to both Drill-Down and Auto-Search. Placing them above the mode radio makes this visually clear and prevents the tags from being associated with only one mode.
- **Interpretation as the universal unit** — Entry triggers, exit triggers, timeframe conditions, general conditions, stop configs, and target configs are all treated as "interpretations" in the drill-down. This unifying abstraction means one drill-down UI pattern works across all 6 tabs, and the `apply_confluence_filters()` helper extends naturally.
- **Phase 9 before Phase 8 remainders (now Phase 10)** — QA Sandbox validates data schemas, and Backtest Settings caches results keyed on strategy config. Both would need reworking if built on the pre-Phase-9 schema. Building the optimization workflow first means QA and caching are designed for the final data model.
- **"Confluence Packs" over "Confluence Groups"** — "Packs" is more marketable and conveys a bundled, configurable product. Internal code retains `groups` naming where appropriate to avoid unnecessary refactoring, but all user-facing labels use "Packs."
- **"General" over "Miscellaneous"** — "Miscellaneous" has a junk-drawer connotation, while "General" conveys broadly applicable conditions. General Packs are strategy-wide filters that aren't tied to chart indicators — they operate on time, calendar, and external event data.
- **Risk Management Packs as dual-output entities** — Each pack produces both a `stop_config` and a `target_config` from shared parameters, analogous to how TF Confluence Packs output both triggers AND conditions from the same indicator. This keeps stop and target conceptually linked (e.g., "ATR-Based" applies ATR to both) while allowing independent drill-down in separate tabs. Replaces the original design of separate Stop Loss Packs and Take Profit Packs.
- **Condition evaluation dispatcher pattern** — `evaluate_condition(df, pack)` dispatches to per-template evaluators based on `condition_logic` field in TEMPLATES. This is extensible (add a new template = add one evaluator function and one TEMPLATES entry) and keeps evaluation logic co-located with template definitions.
- **Extended hours for preview validation** — General Pack previews need bars outside regular hours to demonstrate session/time conditions meaningfully. `mock_data.py` gains an `extended_hours` parameter (4:00 AM – 8:00 PM) with realistic lower volume in pre/after-market periods. The preview defaults to extended hours for `trading_session` template so IN/OUT states are both visible.
- **Chart condition markers via `extra_markers`** — Rather than building a separate chart component for condition state annotations, the existing `render_price_chart()` gains an `extra_markers` parameter. Markers are colored circles with state labels at each condition transition point, overlaid on the candlestick chart. This reuses the proven chart infrastructure without modification.

### Phase 10: QA, Polish & Backtest Settings — "Get Live-Tradeable"
*Deferred Phase 8 items — completing QA validation and backtest configuration after Phase 9 schemas are stable.*

**QA Sandbox Page:** ✓ *(covered by existing pack preview tabs)*
- [x] ~~Stop/Target Validation~~ — covered by Risk Management Pack Preview tab (trades on price chart with stop/target levels, entry/exit markers, KPI summary, trade details with Stop $, Target $, R multiple)
- [x] ~~Signal Detection~~ — covered by TF Confluence Pack Preview tab (interpreter state timeline with last 25 changes, trigger events table with time/name/direction/type/price)
- [x] ~~General Pack Verification~~ — covered by General Pack Preview tab (condition state markers on chart, state transition table, distribution metrics)
- [x] ~~Risk Management Pack Verification~~ — covered by RM Pack Preview (individual) + Strategy Builder SL/TP drill-down tabs (multi-backtest comparison across packs)
- [ ] Backtesting Verification — controlled synthetic scenarios with known expected outputs; developer-only concern, deferred indefinitely

**QA & Verification:**
- [ ] Alert monitor end-to-end test — verify signals detect, webhooks fire, payloads resolve
- [ ] Forward testing validation — confirm live data pipeline produces accurate results
- [ ] Edge cases — empty states, single-trade strategies, zero-trade portfolios, missing data
- [ ] Performance — identify and address any slow-loading pages or redundant data fetches

**Backtest Settings Overhaul — COMPLETED (Feb 11, 2026):**
- [x] Replace sidebar data settings with inline config bar — all data-loading inputs moved from sidebar to compact inline rows at top of Strategy Builder main area
- [x] Three look-back modes via selectbox:
  - **Days** (default) — slider from 7 to 1,825 (5 years); recommended for apples-to-apples comparison across strategies on different timeframes
  - **Bars/Candles** — number input (e.g., 500, 1000, 2000 candles); app calculates equivalent days based on selected timeframe via `days_from_bar_count()`
  - **Date Range** — two date pickers (start/end) for precise control
- [x] Estimated bar count display — status line shows "~7,800 bars · 390 bars/day" below Row 1; computed via `estimate_bar_count()`
- [x] Performance warning — `:orange[Large dataset]` inline when estimated bars exceed 50K
- [x] Timeframe-aware max range guidance — status line shows recommended max (e.g., "1Min: ≤1yr recommended")
- [x] Lookback modes also available on strategy detail Extended KPIs tab — Days/Bars/Date Range selector replaces simple days slider for both backtest and forward test views
- [x] Result caching — three-tier caching system (see "Result Caching" section below)
- [x] Expand supported Alpaca timeframes — 13 presets (see "Timeframe Expansion" section below)
- [x] Fix mock data timeframe — volatility/drift now scale by `sqrt(tf_minutes)` for realistic higher-timeframe bars
- [x] Date range validation — warns when Days/Bars lookback extends before 2016 (Alpaca data floor); strengthened large-dataset warnings
- [x] Alpaca data source note — sidebar caption: "Free plan: IEX data · Paid plan: SIP (all exchanges)"

**Settings Page — COMPLETED (Feb 11, 2026):**
- [x] Settings navigation page — new top-level nav item (6th section in top bar)
- [x] Chart Defaults — Visible Candles selectbox (Tight 50, Close 100, Default 200, Wide 400, Full); writes to `chart_visible_candles` session state; replaces sidebar chart preset selector
- [x] Default Triggers — Default Entry Trigger and Default Exit Trigger selectboxes; applied to new strategies when no saved config exists
- [x] Default Risk Management — Default Stop Loss (method + parameters) and Default Target (method + parameters) with full config UI; applied to new strategies via `default_stop_config` / `default_target_config` session state
- [x] Development section (mock data mode only) — Data Seed number input; writes to `global_data_seed` session state; replaces per-strategy sidebar Data Seed widget

**Sidebar-to-Inline Refactor — COMPLETED (Feb 11, 2026):**
- [x] Move data-loading inputs from sidebar to inline bar — Strategy Origin, Ticker, Timeframe, Direction, Lookback Mode, Lookback params, Load Data button moved to Row 1 columns at top of Strategy Builder main area
- [x] Move Strategy Name, Forward Testing, Alerts to Row 1 — Name as text_input, FT/AL as compact checkboxes with tooltip help text
- [x] Move Entry/Exit/Stop/Target to collapsible expander — Row 2 as `st.expander("Strategy Config")` with 4 equal columns
- [x] Add inline status line — bar estimate, bars/day, timeframe guidance, and validation errors (`:red[text]` colored) displayed via `st.empty()` placeholder pattern
- [x] Move Save button to bottom of page — centered with `st.columns([3,1,3])`, disabled when validation fails
- [x] Remove Add/Remove Exit buttons from sidebar — users manage exits via drill-down "Add" button on Exit tab instead; `sb_additional_exits` session state list tracks additional exit CIDs
- [x] Strip global sidebar — reduced from ~8 widgets to just app title + data source status; eliminates ghost sidebar widgets when navigating between pages
- [x] Move editing banner inline — `st.columns([5,1])` with info message + Cancel Edit button above Row 1

**Strategy Detail Page Enhancements — COMPLETED (Feb 11, 2026):**
- [x] Updated header to 6 columns — Ticker, Direction, Timeframe, Entry, Exit, Stop·Target; plus general confluences display below
- [x] Strategy name as Strategy Builder title — `### {strategy_name}` with `{symbol} | {direction} | {entry → exit}` as caption
- [x] Extended KPIs lookback mode selector — Days/Bars/Date Range options replace simple days slider on both backtest and forward test Extended tabs
- [x] Configuration tabs show general confluences — TF Conditions and General Conditions displayed separately
- [x] Optimizable Variables moved below Extended KPIs — better visual flow in Strategy Builder

**Result Caching — COMPLETED (Feb 11, 2026):**
- [x] Three-tier caching architecture:
  - **Tier 1 — Persistent (JSON):** `equity_curve_data` (exit_times + cumulative_r/pnl + boundary_index) saved to strategies.json and portfolios.json on save; list pages render mini equity curves from stored data with zero computation
  - **Tier 2 — Session State:** trade DataFrames cached per strategy/portfolio ID (`bt_trades_{id}`, `ft_data_{id}`, `port_data_{id}`); detail pages compute once per session, then instant on subsequent views
  - **Tier 3 — Existing `@st.cache_data`:** 1hr TTL on `prepare_data_with_indicators()` unchanged
- [x] Strategy list page — reads persisted `equity_curve_data` and saved KPIs; zero backtests needed for rendering cards and mini equity curves
- [x] Dashboard — best strategy mini equity curve from persisted data; no backtest
- [x] Portfolio list page — reads persisted `equity_curve_data` and `cached_kpis`; `get_portfolio_trades()` only called lazily for portfolios with `requirement_set_id` needing compliance evaluation
- [x] Strategy detail page — backtest trades cached in session state (`bt_trades_{id}`); first visit computes, subsequent visits instant
- [x] Forward test view — forward test data cached in session state (`ft_data_{id}`); first visit computes, subsequent visits instant
- [x] Portfolio detail page — portfolio data cached in session state (`port_data_{id}`) using `get_cached_strategy_trades` for constituent strategies; first visit computes, subsequent visits instant
- [x] Helper functions — `extract_equity_curve_data()`, `extract_portfolio_equity_curve_data()`, `render_mini_equity_curve_from_data()` for persistent data extraction and rendering
- [x] Cache invalidation — session caches cleared on strategy save/update/delete and portfolio save/update/delete; portfolio caches invalidated when constituent strategies change
- [x] Lazy migration — existing strategies/portfolios without `equity_curve_data` auto-backfilled on first list load (one-time cost), then persisted for future instant loads; also backfills missing `max_r_drawdown` and `r_squared` KPIs

**Timeframe Expansion & Data Validation — COMPLETED (Feb 11, 2026):**
- [x] Expanded from 7 to 13 supported timeframes: 1Min, 2Min, 3Min, 5Min, 10Min, 15Min, 30Min, 1Hour, 2Hour, 4Hour, 1Day, 1Week, 1Month
- [x] All timeframe maps updated across 3 files: `TIMEFRAMES`/`TIMEFRAME_GUIDANCE` (app.py), `tf_map`/`BARS_PER_DAY` (data_loader.py), `_parse_timeframe`/`resample_bars` (mock_data.py)
- [x] Weekly/monthly mock data: generates daily bars then resamples via `resample_bars()` for realistic OHLCV aggregation
- [x] Mock data volatility scaling: `volatility = base × sqrt(tf_minutes)`, `drift = base × tf_minutes`, `intrabar_range` scaled by `sqrt(tf_minutes)` — higher timeframes now show proportionally larger price movements
- [x] Date range validation: Days and Bars/Candles modes compute `implied_start = today - data_days` and warn (`:orange[]`) if it falls before 2016-01-01 (`ALPACA_DATA_FLOOR`)
- [x] Strengthened large-dataset warnings: 200K+ bars = `:red[**Very large dataset — may be slow**]`, 50K+ = `:orange[Large dataset]`
- [x] IEX/SIP data source note: `st.caption()` in sidebar below Alpaca status indicator — "Free plan: IEX data · Paid plan: SIP (all exchanges)"
- [x] `estimate_bar_count()` returns `int` via `max(1, int(...))` to handle fractional `BARS_PER_DAY` values for 1Week (0.2) and 1Month (1/21)

**Deferred from Phase 9:**
- [ ] Trigger parameters visible and expandable in Optimizable Variables — show EMA periods, ATR multiplier, etc. (not just trigger name)
- [ ] Stop/target variation tags on trades — tag individual trades with pack ID when running multi-backtest comparisons
- [ ] Multi-backtest progress indicator + caching — progress bar for SL/TP drill-down; cache keyed on (symbol, timeframe, date range, strategy config, pack ID)
- [ ] Lazy tab loading — only compute drill-down results when a tab is first opened, not all 6 on page load

**UX Polish:**
- [ ] Utility buttons on Portfolios page — "Portfolio Requirements" and "Webhook Templates" links next to "New Portfolio" button

### Design Decisions (Phase 10 — Settings & Inline Refactor)
- **Inline config bar over sidebar** — Sidebar config panels create "ghost widgets" in Streamlit when navigating away — stale sidebar DOM elements from previous pages cause visual artifacts and key conflicts. Moving all data-loading inputs to inline columns in the main content area eliminates this class of bugs entirely. The global sidebar shrinks to just app title + data source status.
- **`st.empty()` placeholder for status line** — The bar estimate and validation errors depend on widget values defined earlier in the render flow. Using `st.empty()` reserves visual space at the right position, then fills it after all widget values are resolved. This avoids the Streamlit issue of text appearing before its dependent widgets.
- **FT/AL as compact checkboxes with tooltips** — Forward Testing and Alerts are boolean toggles that don't need full labels consuming column width. Single-character labels ("FT", "AL") with `help=` tooltip text explain the feature on hover while keeping Row 1 compact.
- **`sb_additional_exits` session state list over sidebar selectboxes** — The old approach used N sidebar selectboxes with Add/Remove buttons, requiring complex index management. The new approach stores exit CIDs in a flat list managed entirely via the drill-down "Add" button and Optimizable Variables "✕" removal. Simpler state, fewer widgets, no sidebar needed.
- **Settings page defaults via session state** — Settings page writes to session state keys (`default_stop_config`, `default_target_config`, `global_data_seed`, `chart_visible_candles`). Strategy Builder reads from these when no saved config exists (new strategy). When editing a saved strategy, the saved config takes precedence via `'key' in edit_config` existence check. This avoids a separate `settings.json` file while still providing app-wide defaults.
- **`'key' in edit_config` over `.get()` for None-valued fields** — `edit_config.get('target_config')` returns `None` both when the key is missing (new strategy) and when the value is explicitly `None` (strategy saved with no target). Using `'key' in edit_config` distinguishes these cases: missing means "use Settings default", present-but-None means "user chose no target."
- **Lookback mode on Extended KPIs tab** — Strategy detail's Extended tab previously had only a days slider. Adding the full Days/Bars/Date Range selector gives users the same flexibility as the Strategy Builder, enabling precise historical analysis on saved strategies without re-editing them.

**After this phase: start live trading. All stored schemas (strategies.json, portfolios.json, alert_config.json, general_packs.json, risk_management_packs.json) are stable. All subsequent phases are additive — no restructuring or data loss risk.**

### Phase 11: Analytics & Edge Detection
*Advanced performance metrics and strategy health monitoring — inspired by Davidd Tech.*
*Reference images: `/docs/reference_images/DaviddTech *.png`*

- [ ] Edge Check overlay on equity curves — toggleable 21-period MA + Bollinger Bands on equity curve chart (visual indicator of strategy health; equity below lower BB = statistically unusual underperformance)
- [ ] Expanded KPI panel — add: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Kelly Criterion, Daily Value-at-Risk, Expected Shortfall (CVaR), Max Consecutive Wins/Losses, Gain/Pain Ratio, Payoff Ratio, Common Sense Ratio, Tail Ratio, Outlier Win/Loss Ratio, Recovery Factor, Ulcer Index, Serenity Index (builds on R² from Phase 8), Skewness, Kurtosis, Expected Daily/Monthly/Yearly returns
- [ ] Rolling performance metrics chart — interactive chart with toggle buttons for rolling Win Rate, Profit Factor, and Sharpe over a configurable trade window
- [ ] Return distribution analysis — histogram, box plot, and violin plot views with skewness/kurtosis/tail risk callouts
- [ ] Cumulative vs. Simple P&L views — compounded equity curve (reinvested gains) alongside simple/sum-based P&L
- [ ] Markov Motor Analysis (advanced tab) — win/loss transition probabilities, win/loss streak distribution chart, consistency score, stability index, trend strength, market regime detection (favorable/unfavorable/neutral clustering), edge decay chart (rolling PF with threshold line), and Markov Intelligence Insights summary
- [ ] KPI placement audit — map out primary vs. secondary KPIs for strategy cards, strategy detail, portfolio cards, portfolio detail; ensure consistent and useful placement across all views

### Phase 12: Strategy Origins
*Expand strategy creation beyond the standard trigger-based approach — support webhook-driven and scanner-based strategies.*

- [ ] Expand Strategy Origin selectbox — add "Webhook Inbound" and "Scanner" options to existing sidebar selectbox (currently shows "Standard" only, added in Phase 8 as placeholder)
- [ ] Origin-specific sidebar fields — after origin selection, show relevant configuration fields below (additional fields per origin type; existing strategies already have `strategy_origin: "standard"` with no migration needed)
- [ ] Webhook Inbound origin — entries/exits driven by inbound webhooks (e.g., TradingView alerts, LuxAlgo signals); user can still layer confluence conditions from market data on top of webhook triggers; CSV upload for backtest data from TradingView or spreadsheets
- [ ] Scanner origin — strategy not tied to a single ticker; runs against a universe of stocks matching screener criteria (Alpaca screener APIs); targets active day trading / scalping use cases (S&B Capital, Warrior Trading style); requires separate planning session for architecture given 1:many ticker relationship
- [ ] Backward-compatible schema — `strategy_origin: "standard"` defaulted for all existing strategies; origin-specific fields only present when relevant

### Phase 13: Live Portfolio Management
*Active trading account management — bridge between backtesting and real-world trading.*

- [ ] Account Management tab on portfolio detail page — separate from backtest/analysis tabs
- [ ] Deposit/withdrawal ledger — track additions to and deductions from the trading account balance
- [ ] Webhook trigger audit log — plot actual fired webhook alerts on the price chart for visual verification that signals are firing correctly in production
- [ ] Trading notes — freeform notes area for the user to document observations, adjustments, and context per portfolio
- [ ] Live balance tracking — actual account balance based on webhook triggers + manual adjustments, independent of backtest projections
- [ ] Intra-bar real-time alert engine — WebSocket streaming via Alpaca for `[I]` triggers:
  - Subscribe to real-time trades/quotes for active strategy symbols
  - Build partial OHLCV bars from tick data
  - Check price-level trigger conditions against live ticks (e.g., price crossing UT Bot trail)
  - Fire alerts the moment condition is met, not at bar close
  - `[C]` triggers continue using the existing bar-close polling model
  - Mirrors TradingView's "once per bar" (intra-bar) vs. "once per bar close" alert modes
  - Requires Alpaca paid plan ($99/mo) for SIP real-time feed with no symbol limit (free plan limited to IEX, 30 symbols)
  - The `[C]` / `[I]` execution type property added in Phase 8 determines which alert engine each trigger uses

### Phase 14: Settings Page — COMPLETED (merged into Phase 10, Feb 11, 2026)
*Implemented as part of the sidebar-to-inline refactor in Phase 10. See Phase 10 "Settings Page" section for details.*

- [x] Settings navigation page — 6th top-level nav item
- [x] Chart Defaults — Visible Candles selectbox (replaces sidebar chart preset selector)
- [x] Default Triggers — Default Entry and Exit Trigger selectboxes for new strategies
- [x] Default Risk Management — Default Stop Loss and Target with full method+parameter config
- [x] Development — Data Seed (mock mode only)
- [ ] **Connections** subpage — Alpaca API configuration, data source status (future)
- [ ] Persist settings to `settings.json` — loaded on app startup, available via `get_setting(key, default)` helper (future; currently uses session state)

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
