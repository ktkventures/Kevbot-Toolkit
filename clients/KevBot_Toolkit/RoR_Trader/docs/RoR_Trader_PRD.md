# RoR Trader - Product Requirements Document (PRD)

**Version:** 0.38
**Date:** February 19, 2026
**Author:** Kevin Johnson
**Status:** Phase 18A In Progress — Multi-Timeframe Confluence: Timeframe management page, TF label utilities, enabled_timeframes settings. Phase 17D complete; Phases 17A–C, 11–16 complete

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

**QA & Verification — COMPLETED (Feb 13, 2026):**
- [x] Alert monitor end-to-end test — "Send Test Alert" button on Alerts page fires synthetic alert through full pipeline (save → deliver to all active webhooks → report results); placed in collapsible "E2E Test" expander
- [x] Forward testing validation — diagnostic caption on forward test view showing `BT: N trades (Xd) · FW: M trades (Yd) · Boundary: YYYY-MM-DD` for boundary split verification
- [x] Edge cases — JSON corruption guards on `load_strategies()`, `load_portfolios()`, `load_alerts()`, `load_alert_config()` with graceful fallback to empty defaults
- [x] Performance — eliminated duplicate `load_alerts()` on Dashboard (single load, slice for display); cached extended data views in session state (`bt_ext_`, `ft_ext_` keys) with invalidation on strategy save/delete; batched portfolio strategy lookups via `strat_by_id` dict instead of per-card file reads
- [x] Webhook payload template fix — `render_payload()` now auto-quotes string placeholder values and leaves numbers bare, producing valid JSON regardless of whether user quotes `{{placeholders}}` in templates; two-pass substitution handles both `"{{key}}"` (explicit) and `{{key}}` (auto) patterns
- [x] Days slider → number input — replaced `st.slider` with `st.number_input` (step=7) for all Days inputs (Strategy Builder, Extended backtest/forward test tabs, Settings page) for precise value entry

**Backtest Settings Overhaul — COMPLETED (Feb 11, 2026):**
- [x] Replace sidebar data settings with inline config bar — all data-loading inputs moved from sidebar to compact inline rows at top of Strategy Builder main area
- [x] Three look-back modes via selectbox:
  - **Days** (default) — number input from 7 to 1,825 (5 years) with step=7; recommended for apples-to-apples comparison across strategies on different timeframes
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

*Deferred items moved to Phase 16 (low-priority cleanup).*

### Design Decisions (Phase 10 — Settings & Inline Refactor)
- **Inline config bar over sidebar** — Sidebar config panels create "ghost widgets" in Streamlit when navigating away — stale sidebar DOM elements from previous pages cause visual artifacts and key conflicts. Moving all data-loading inputs to inline columns in the main content area eliminates this class of bugs entirely. The global sidebar shrinks to just app title + data source status.
- **`st.empty()` placeholder for status line** — The bar estimate and validation errors depend on widget values defined earlier in the render flow. Using `st.empty()` reserves visual space at the right position, then fills it after all widget values are resolved. This avoids the Streamlit issue of text appearing before its dependent widgets.
- **FT/AL as compact checkboxes with tooltips** — Forward Testing and Alerts are boolean toggles that don't need full labels consuming column width. Single-character labels ("FT", "AL") with `help=` tooltip text explain the feature on hover while keeping Row 1 compact.
- **`sb_additional_exits` session state list over sidebar selectboxes** — The old approach used N sidebar selectboxes with Add/Remove buttons, requiring complex index management. The new approach stores exit CIDs in a flat list managed entirely via the drill-down "Add" button and Optimizable Variables "✕" removal. Simpler state, fewer widgets, no sidebar needed.
- **Settings page defaults via session state + disk persistence** — Settings page writes to session state keys (`default_stop_config`, `default_target_config`, `global_data_seed`, `chart_visible_candles`, etc.). Strategy Builder reads from these when no saved config exists (new strategy). When editing a saved strategy, the saved config takes precedence via `'key' in edit_config` existence check. A "Save Settings" button persists all settings to `config/settings.json`; on startup, `load_settings()` merges saved values over `SETTINGS_DEFAULTS` so new keys auto-get defaults without migration.
- **`'key' in edit_config` over `.get()` for None-valued fields** — `edit_config.get('target_config')` returns `None` both when the key is missing (new strategy) and when the value is explicitly `None` (strategy saved with no target). Using `'key' in edit_config` distinguishes these cases: missing means "use Settings default", present-but-None means "user chose no target."
- **Lookback mode on Extended KPIs tab** — Strategy detail's Extended tab previously had only a days slider. Adding the full Days/Bars/Date Range selector gives users the same flexibility as the Strategy Builder, enabling precise historical analysis on saved strategies without re-editing them.

**After this phase: start live trading. All stored schemas (strategies.json, portfolios.json, alert_config.json, general_packs.json, risk_management_packs.json) are stable. All subsequent phases are additive — no restructuring or data loss risk.**

### Phase 10B: Alert & Forward Testing UX Overhaul ✓
*Smart polling, forward testing always-on, alert simplification, webhook editor UX fixes.*

**Webhook Editor UX:**
- [x] Fix template insert dropdown — write directly to `st.session_state[widget_key]` + `st.rerun()` to update text area (Streamlit only reads `value=` on first render)
- [x] Placeholder auto-append — selecting a placeholder from dropdown appends `{{key}}` to the text area and reruns
- [x] Show resolved payload on webhook test — displays rendered JSON below success/error message for verification
- [x] Tooltips on portfolio alert toggles — `help=` parameter on compliance breach toggle

**Forward Testing Always-On:**
- [x] Remove FT checkbox from Strategy Builder — `enable_forward = True` as constant
- [x] `save_strategy()` and `update_strategy()` always set `forward_testing: True` and `forward_test_start`
- [x] Remove "Forward Testing" / "Backtest Only" filter from strategy list (all strategies are forward-testing)
- [x] Remove forward test guard from strategy alerts tab

**Alert Simplification:**
- [x] Monitoring scope: strategy is monitored if it belongs to ANY portfolio with at least one enabled webhook (replaces per-strategy `alerts_enabled` + per-portfolio `alerts_enabled` toggles)
- [x] Remove `alerts_enabled` toggle from portfolio webhooks page — webhooks are the control point
- [x] Simplify strategy alerts tab — show which portfolios link this strategy and their webhook status
- [x] Simplify "Manage Active Alerts" section — show monitored strategies and webhook-enabled portfolios (read-only overview)
- [x] `deliver_alert()` checks for active webhooks instead of `alerts_enabled` flag

**Smart Candle-Close-Aligned Polling (`alert_monitor.py`):**
- [x] Group monitored strategies by timeframe
- [x] `seconds_until_next_close()` — compute time until next candle close + 3s buffer
- [x] Main loop sleeps until next candle close instead of fixed-interval polling
- [x] Double-poll prevention — track last poll epoch per timeframe to avoid re-polling same candle

**In-Memory Data Cache (`alert_monitor.py`):**
- [x] Module-level `_data_cache` dict keyed by `(symbol, timeframe)`
- [x] `load_cached_bars()` — incremental fetch (only new bars since last cached bar)
- [x] Symbol deduplication — pre-load data once per unique symbol before polling timeframe group
- [x] `detect_signals()` accepts optional `df` parameter for pre-loaded data

**Dynamic Bar Count (`alerts.py`):**
- [x] `compute_signal_detection_bars(timeframe)` — `max(full_trading_day, indicator_warmup=50, floor=200)`
- [x] 1Min = 390 bars (full day for VWAP accuracy), 5Min = 200, 1Day = 200
- [x] Replaces hardcoded `SIGNAL_DETECTION_BARS = 200`

**Card Indicators:**
- [x] Strategy cards show `:orange[Monitored]` badge when strategy is in a webhook-enabled portfolio
- [x] Portfolio cards show webhook count in metadata caption line

### Design Decisions (Phase 10B — Alert UX Overhaul)
- **Forward testing always-on** — Forward testing is just a timestamp boundary; computation only happens on view. Making it always-on removes a checkbox that confused users ("should I enable this?") while ensuring every strategy accumulates forward test data from day one. Cost: zero (no background computation).
- **Webhooks as the control point** — Instead of three separate toggles (strategy alerts_enabled, portfolio alerts_enabled, webhook enabled), the webhook's `enabled` flag is the single control point. If a portfolio has active webhooks and contains a strategy, that strategy is monitored. This eliminates the "I configured everything but alerts don't fire" failure mode.
- **Candle-close-aligned polling** — Fixed-interval polling (every 60s) wastes cycles checking for signals mid-candle and misses the critical moment right after candle close. Aligning polls to candle boundaries + 3s buffer ensures we check for signals when new data is available. For 1-min strategies this means checking every 63s aligned to clock minutes. For 5-min strategies, every 303s aligned to :00/:05/:10 etc.
- **In-memory data cache** — The monitor process is long-running, so caching DataFrames in memory between polls avoids redundant full data loads. Incremental fetching (only new bars since last cached bar) reduces API calls to ~1 bar per poll per symbol. Symbol deduplication means 3 strategies on SPY 1Min share one data load.
- **Dynamic bar count** — VWAP requires a full trading day of data (390 bars on 1-min). The old hardcoded 200 bars produced inaccurate VWAP values for intraday strategies. `compute_signal_detection_bars()` ensures enough data for both indicator warmup and VWAP accuracy.

**Future Phases (from this discussion):**
- Phase 14B: Unified Streaming Alert Engine — WebSocket streaming replaces polling for both `[I]` and `[C]` triggers; sub-millisecond alert latency; sub-minute candle support (10s, 30s); polling retained as fallback only

### Phase 10C: Incremental Data Refresh ✓
*"Update Data" button with incremental refresh — only processes new forward test data, not historical trades.*

- [x] `stored_trades` — minimal trade records (entry_time, exit_time, r_multiple, win) persisted per strategy for incremental refresh without full pipeline re-runs
- [x] `_extract_minimal_trades()` / `_trades_df_from_stored()` — helpers to convert between full trades DataFrame and minimal JSON-serializable records
- [x] `_generate_incremental_trades()` — loads only a small data window (indicator warmup + new bars since last trade) and generates trades for that window only
- [x] `refresh_strategy_data()` — incremental path (appends new trades to stored, recomputes KPIs from stored) with cold-start migration fallback (full pipeline for strategies without stored_trades)
- [x] `bulk_refresh_all_strategies()` — iterates all non-legacy strategies with progress callback
- [x] "Update Data" button on My Strategies page — triggers bulk refresh with progress bar, session cache clearing, and status messages
- [x] Save flow persists `stored_trades` alongside `equity_curve_data` and `kpis`
- [x] `data_refreshed_at` timestamp persisted per strategy to track last refresh
- [x] Lazy-load Extended tabs — "Equity & KPIs (Extended)" tab on strategy detail pages deferred behind a "Load Extended Data" button instead of auto-loading on every page render; eliminates 10-20s blocking that previously stalled all tabs
- [x] Data View filter dropdown on My Strategies page — filters `stored_trades` by time window (Last 7/30/90 Days, Backtest Only, Forward Test Only) and recomputes KPIs + equity curves from filtered subset; instant, no pipeline re-runs; KPI-based sorting reflects filtered values

### Design Decisions (Phase 10C — Incremental Data Refresh)
- **Incremental over full rebuild** — Trade history is append-only. Backtest trades never change; forward test trades accumulate. Re-running the full pipeline from strategy creation is wasteful. Instead, we store minimal trade records and only load/process the recent data window (warmup + new bars). For 1-min strategies, the incremental window is ~1 day vs. 30-60+ days for a full rebuild.
- **Stored trades as source of truth** — `stored_trades` contains the 4 fields needed to recompute any KPI or equity curve: entry_time, exit_time, r_multiple, win. On each refresh, new trades are appended and KPIs/equity curves are recomputed from the full stored list using existing `calculate_kpis()` and `extract_equity_curve_data()` functions (no duplicated math).
- **Cold-start migration** — Strategies created before this feature (without `stored_trades`) get a one-time full pipeline run that populates the field. All subsequent refreshes are incremental.
- **Warmup buffer** — The incremental data window starts `warmup_bars / bars_per_day` calendar days before the last known trade. 100 bars covers 2× the longest indicator (EMA-50) for safety. For 1-min timeframes this is ~1 calendar day; for 1-day timeframes it's ~145 days (necessary for daily indicator accuracy).
- **Lazy-load Extended tabs** — Streamlit's `st.tabs()` executes ALL tab content on every render, even invisible tabs. The Extended tab's 365-day data load blocked all 7 tabs from rendering (~10-20s). Gating it behind a button means the page renders in ~2-5s and users only pay the cost when they actually want extended data.
- **Data View filter vs. expanded backtest** — The Data View filter (completed) filters existing `stored_trades` by date — instant, no data loading. Expanded backtest (Phase 16, low priority) extends the backtest beyond original settings by running the pipeline for an expanded window — slower but additive. These are distinct use cases: "show me only recent performance" vs. "what if the backtest started earlier?"

### Phase 10D: Non-Optimizable Edits ✓
*Allow minor strategy edits without resetting the forward test — preserve accumulated forward test data for non-trade-affecting changes.*

- [x] `OPTIMIZABLE_PARAMS` frozenset — classifies 20 strategy parameters that affect trade generation (triggers, confluence, stops, targets, timeframe, symbol, data window, data_seed); all others are non-optimizable
- [x] `_has_optimizable_changes()` — compares old vs. new strategy on optimizable params with normalization (sorted string lists, None vs. missing, dict deep compare)
- [x] `update_strategy()` conditional preservation — if only non-optimizable fields changed, preserves `stored_trades`, `forward_test_start`, `equity_curve_data`, `kpis`, and `data_refreshed_at` from the old strategy
- [x] KPI-only recomputation — when `risk_per_trade` or `starting_balance` changes without optimizable changes, recomputes KPIs from `stored_trades` with new dollar values (equity curve is R-based, unaffected)
- [x] Save feedback toasts — `update_strategy()` returns `'preserved'` or `'reset'`; strategy detail page shows appropriate toast message after save
- [x] Edit confirmation dialogs updated — both strategy list card and detail page dialogs changed from `st.warning` to `st.info` explaining which edits are safe vs. reset-triggering; "Edit Anyway" renamed to "Edit"
- [x] Editing banner enhancement — shows forward test age (e.g., "forward test: 14d") to remind user of accumulated data

### Design Decisions (Phase 10D — Non-Optimizable Edits)
- **Parameter classification via constant** — `OPTIMIZABLE_PARAMS` is a frozenset of the 20 parameters that feed into `generate_trades()` or control the data window. This is checked by `_has_optimizable_changes()` at save time. `risk_per_trade` is excluded despite appearing in `generate_trades()` signature because it's never referenced in the function body (confirmed via code analysis) — it only affects `calculate_kpis()` dollar values.
- **Preservation in update_strategy, not save flow** — The save flow always rebuilds `stored_trades` from the current backtest (which only contains backtest trades, not forward test trades). For non-optimizable edits, `update_strategy()` overwrites the incoming `stored_trades`, `kpis`, and `equity_curve_data` with the old strategy's preserved versions. This keeps the save flow unchanged and centralizes preservation logic.
- **Return type as feedback channel** — `update_strategy()` returns `'preserved'`, `'reset'`, or `False` instead of just `bool`. Both string values are truthy, so existing `if update_strategy(...)` checks still work. The save flow captures the specific return value to display the appropriate toast.

> **Detailed implementation spec for Phases 11–14:** See [`docs/Implementation_Spec_Phases_11-14.md`](Implementation_Spec_Phases_11-14.md) — contains file locations, data structures, function signatures, UI layouts, and implementation order for autonomous execution.

### Phase 11: Analytics & Edge Detection — COMPLETE (Feb 14, 2026)
*Advanced performance metrics and strategy health monitoring — inspired by Davidd Tech.*
*Reference images: `/docs/reference_images/DaviddTech *.png`*

- [x] Edge Check overlay on equity curves — toggleable 21-period MA + Bollinger Bands on equity curve chart (visual indicator of strategy health; equity below lower BB = statistically unusual underperformance)
- [x] Expanded KPI panel — add: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Kelly Criterion, Daily Value-at-Risk, Expected Shortfall (CVaR), Max Consecutive Wins/Losses, Gain/Pain Ratio, Payoff Ratio, Common Sense Ratio, Tail Ratio, Outlier Win/Loss Ratio, Recovery Factor, Ulcer Index, Serenity Index (builds on R² from Phase 8), Skewness, Kurtosis, Expected Daily/Monthly/Yearly returns
- [x] Rolling performance metrics chart — interactive chart with toggle buttons for rolling Win Rate, Profit Factor, and Sharpe over a configurable trade window
- [x] Return distribution analysis — histogram, box plot, and violin plot views with skewness/kurtosis/tail risk callouts
- [x] Cumulative vs. Simple P&L views — compounded equity curve (reinvested gains) alongside simple/sum-based P&L
- [x] Markov Motor Analysis (advanced tab) — win/loss transition probabilities, win/loss streak distribution chart, consistency score, stability index, trend strength, market regime detection (favorable/unfavorable/neutral clustering), edge decay chart (rolling PF with threshold line), and Markov Intelligence Insights summary
- [x] KPI placement audit — map out primary vs. secondary KPIs for strategy cards, strategy detail, portfolio cards, portfolio detail; ensure consistent and useful placement across all views

### Phase 12: Webhook Inbound Strategy Origin — COMPLETE (Feb 14, 2026)
*Allow strategies driven by inbound webhooks — entries/exits from external sources (TradingView, LuxAlgo, custom scripts) with RoR Trader confluence, stops, and backtesting layered on top.*

- [x] Expand Strategy Origin selectbox — add "Webhook Inbound" option to existing sidebar selectbox (currently shows "Standard" only, added in Phase 8 as placeholder)
- [x] Origin-specific sidebar fields — after origin selection, show webhook configuration fields (secret, endpoint URL, signal JSON path, direction mapping); hide standard entry/exit trigger sections
- [x] Webhook Inbound origin — entries/exits driven by inbound webhooks; user can still layer confluence conditions from market data on top of webhook triggers
- [x] Inbound webhook receiver — lightweight HTTP server (Flask/FastAPI background thread) to receive POST requests from external alert sources; validates webhook secret; stores signals for processing
- [x] CSV upload for backtest data — import historical signals from TradingView strategy tester exports or spreadsheets; apply stop/target logic to signal pairs; generate R-multiples and KPIs
- [x] Forward test for webhook origin — process real-time inbound signals same as backtest signals; append trades to `stored_trades`; standard forward test boundary mechanics apply
- [x] Backward-compatible schema — `strategy_origin: "standard"` defaulted for all existing strategies; `webhook_config` dict only present when origin is `"webhook_inbound"`

### Phase 13: Live Alerts Validation — COMPLETE (Feb 14, 2026)
*Three-tier confidence visualization — validate that alert executions match theoretical forward test trades before trusting them with real money.*

The strategy lifecycle has three confidence tiers, each progressively closer to reality:
1. **Backtest** — theoretical, computed from historical data (lowest confidence)
2. **Forward test** — theoretical but real-time, parameters locked before data appeared (medium confidence)
3. **Live/triggered** — actual alert executions, reflects real-world timing, slippage, and missed signals (highest confidence)

- [x] Alert execution correlation — match fired alerts (from `alerts.json`) to forward test trades by symbol, signal type, and timestamp proximity; store matched execution data (alert trigger price, alert fire time) alongside theoretical trade data
- [x] Three-color equity curve — backtest segment (blue), forward test segment (orange), and live/triggered segment (green) each rendered in distinct colors on both strategy detail full-size equity curves and strategy card mini equity curves; transition points visible at a glance
- [x] Strategy card caption enhancement — pipe-delimited status line shows backtest duration, forward test duration, and monitored duration with color-coded text matching equity curve segment colors (e.g., `SPY LONG | BT 45d | Fwd 14d | Live 5d`)
- [x] Discrepancy detection — identify cases where forward test shows a trade but no corresponding alert fired (potential alert/webhook issue), and cases where an alert fired but no forward test trade exists; surface discrepancies as annotations on equity curve and as a count on strategy cards
- [x] Alert tracking mode — enable alert execution tracking per strategy independent of portfolio webhook allocation; allows users to validate alert behavior and build confidence before committing to a live portfolio
- [x] Live segment uses actual alert trigger prices rather than theoretical bar prices — captures real-world slippage between theoretical entry/exit and actual alert fire time
- [x] Strategy detail page — dedicated "Live vs. Forward" comparison tab showing side-by-side metrics: theoretical forward test KPIs vs. actual live execution KPIs, with delta highlighting
- [x] Alert Analysis tab — dedicated tab on strategy detail page (appears when alert tracking enabled with live executions): summary metrics table (FT vs Live KPIs with delta indicators, avg slippage, webhook success rate, missed/phantom counts), trade-by-trade comparison table (per-trade entry/exit slippage, adjusted R, webhook status, coverage percentage), and discrepancy detail (missed alerts + phantom alerts breakdown)
- [x] Alert tracking toggle on strategy cards — compact toggle on each forward-testing strategy card for enabling/disabling alert tracking directly from the list view, with persistence and live data cleanup on disable
- [x] Trade history graceful degradation — `exit_reason` column made optional in trade tables; stored_trades (minimal 4-field format) render without crash on both backtest-only and forward test detail views
- [x] Timezone-safe trade boundary splitting — `split_trades_at_boundary()` handles all timezone mismatch combinations (aware/naive, different timezones) via normalize-before-compare pattern

### Design Decisions (Phase 13 — Live Alerts Validation)
- **Three-tier confidence model** — Backtest is retrospective curve-fitting. Forward test proves the strategy works on unseen data but still uses theoretical bar prices. Live alert data is the closest proxy to actual trading — it captures webhook delivery timing, missed signals, and price differences between bar close and alert fire time. Visualizing all three tiers on one equity curve lets users see exactly where theory diverges from reality.
- **Discrepancy as signal, not noise** — A forward test trade with no matching alert is not just a data gap — it's actionable information. It could indicate webhook misconfiguration, alert monitor downtime, or a timing edge case. Surfacing these discrepancies helps users debug their alert pipeline before going live with real money.
- **Alert tracking separate from portfolio webhooks** — Users should be able to track alert executions for confidence-building without routing them to a broker. This is essentially paper trading validation — "would my alerts have fired correctly?" — without the commitment of a live portfolio.
- **Override vs. additive** — When live data exists for a period, it replaces the forward test data for that period on the equity curve (since live is higher fidelity). Periods without live data fall back to forward test. Periods before forward test start show backtest data.

### Phase 14: Live Portfolio Management — 14A COMPLETE (Feb 14, 2026); 14B COMPLETE (Feb 18, 2026); 14C COMPLETE (Feb 19, 2026)
*Active trading account management — bridge between backtesting and real-world trading.*

- [x] Account Management tab on portfolio detail page — separate from backtest/analysis tabs
- [x] Deposit/withdrawal ledger — track additions to and deductions from the trading account balance
- [x] Trading notes — freeform notes area for the user to document observations, adjustments, and context per portfolio
- [x] Live balance tracking — actual account balance based on webhook triggers + manual adjustments, independent of backtest projections
- [x] **Alpaca SIP data feed upgrade** — User upgraded Alpaca plan to paid SIP ($99/mo); all data paths now use consolidated SIP feed (all exchanges) instead of IEX (single exchange). Candlestick prices now match TradingView exactly
- [x] **Data feed wiring** — `feed` parameter threaded through entire data pipeline: `data_loader.py` → `app.py` → `alert_monitor.py` → `alerts.py`. Feed setting (`sip`/`iex`) stored in `config/settings.json` and selectable on Connections settings page
- [x] **UTC timezone fix** — `datetime.now()` replaced with `datetime.now(timezone.utc)` in Alpaca API calls to prevent timezone-dependent data truncation
- [x] **RTH (Regular Trading Hours) filter** — `_filter_rth()` strips pre-market (4:00–9:29 AM ET) and after-hours (4:01–8:00 PM ET) bars from Alpaca data to match TradingView RTH mode
- [x] **Actual data source tracking** — `get_data_source()` now reports what was *actually* used (e.g., "Alpaca SIP" or "Mock Data") rather than what was configured, preventing silent mock-data fallback from going unnoticed
- [x] **EMA warmup fix** — All preview charts load 30 days of data (~11,700 RTH bars) for indicator warmup, then trim to last 3 days for display. EMA 200 now converges properly, matching TradingView
- [x] **Connections settings subpage** — Alpaca API key status display, data feed selector (IEX/SIP), real-time engine toggle
- [x] **Unified Streaming Alert Engine** — Replace polling-based alert monitor with WebSocket-first architecture. Single Alpaca SIP stream handles both `[I]` and `[C]` triggers:
  - [x] `UnifiedStreamingEngine` — lifecycle manager with singleton API (`start_engine`, `stop_engine`, `engine_status`). Spawns daemon thread with asyncio WebSocket loop, `ThreadPoolExecutor(4)` for non-blocking webhook delivery
  - [x] `BarBuilder` — clock-aligned OHLCV bar aggregation from ticks. `_align_to_period()` snaps timestamps to bar boundaries (e.g., 09:31:23 → 09:31:00 for 60s bars). Maintains 500-bar rolling DataFrame per (symbol, timeframe) with warmup data from `load_latest_bars()`
  - [x] `SymbolHub` — per-symbol tick dispatcher managing multiple `BarBuilder` instances (one per timeframe). On bar close, runs full `detect_signals()` pipeline (indicators → interpreters → triggers) then enriches + saves + delivers alerts via callback
  - [x] `AlertCooldown` — deduplication keyed by `"strategy_id:signal_type"` with configurable cooldown window (default: bar duration)
  - [x] `TriggerLevelCache` — stub for future `[I]` intra-bar triggers. All current triggers are `bar_close`; cache will be populated when UT Bot / VWAP triggers gain `intra_bar` execution type
  - [x] **Polling fallback** — `alert_monitor.py` checks `streaming_connected` flag in `monitor_status.json`; when True, poller sleeps (5s loop) instead of polling. Engine sets flag on connect/disconnect
  - [x] **Exponential backoff reconnection** — 5s → 10s → 20s → 40s → 60s cap on WebSocket disconnect; resets to 5s on successful reconnect
  - [x] **Signal handler guard** — `alert_monitor.py` `signal.signal()` calls wrapped in `if __name__ == "__main__"` to allow safe import from non-main threads (Streamlit sessions, daemon threads)
  - [x] **Sub-minute timeframe support** — `10Sec` and `30Sec` added to `TIMEFRAME_SECONDS` (alert_monitor, realtime_engine) and `BARS_PER_DAY` (data_loader). UI selection deferred
  - [x] **App integration** — Start Monitor button routes to streaming engine when RT Engine toggle is enabled; Stop button stops both engine and poller subprocess; status bar shows "Engine: Streaming" / "Monitor: Polling" / "Monitor: Stopped"; Settings page shows live engine stats (connection status, symbol count, tick count)
  - The `[C]` / `[I]` execution type property on `TriggerDefinition` determines evaluation path (every tick vs. bar close)

### Design Decisions (Phase 14B — Unified Streaming Alert Engine)
- **Streaming-first over polling-first** — The original design used bar-close polling for `[C]` triggers and added WebSocket streaming only for `[I]` triggers. The unified approach uses WebSocket streaming as the primary path for *both* trigger types. Rationale: (1) if you're already receiving ticks for `[I]` evaluation, you have all the data needed to build bars locally — polling the REST API for the same data is redundant; (2) local bar completion detection is milliseconds vs. 4-6 seconds for API poll + compute; (3) one data source is simpler than two parallel systems.
- **Local bar builder over Alpaca bar API** — Building bars from ticks enables sub-minute timeframes (10s, 30s) that Alpaca's REST API doesn't serve. It also eliminates the 3-second "finalization buffer" the poller needed — the bar is complete the instant a tick arrives in the next period. For standard timeframes (1m, 5m), locally-built bars produce identical OHLCV to Alpaca's server-side aggregation since both use the same tick stream.
- **Pre-computed trigger levels for `[I]` evaluation** — `[I]` trigger conditions (e.g., "price crosses above UT Bot trailing stop") depend on indicator values computed from completed bars. The trigger *level* is computed once at bar close, then held constant until the next bar closes. Each incoming tick only needs a float comparison against this cached level — no indicator recomputation. This makes per-tick evaluation O(1) regardless of indicator complexity.
- **Incremental indicator pipeline** — On bar close, the new bar is appended to a rolling history DataFrame and indicators are recomputed for the latest bar only. For most indicators (EMA, ATR, MACD), the rolling nature means only the most recent value depends on the new bar. This is bounded by the number of unique (symbol, timeframe) pairs, not the number of strategies — 50 strategies on SPY 1Min share one indicator computation.
- **Polling as fallback, not co-primary** — `alert_monitor.py` is retained as a degraded-mode fallback when the WebSocket disconnects. The engine detects disconnection, switches to polling temporarily, and attempts reconnection with exponential backoff. This ensures alerts continue (with higher latency) during network issues rather than going silent.
- **Alert cooldown for `[I]` triggers** — Without deduplication, an `[I]` trigger could fire hundreds of times per second as price oscillates around the trigger level. A configurable cooldown (default: once per bar period) prevents duplicate alerts while still capturing the first crossing.
- **Sub-minute candles for HFT use cases** — 10-second and 30-second bars are built from the same tick stream as standard bars. The bar builder maintains multiple timeframe aggregators per symbol. This is prerequisite for high-frequency strategies where a 5-15 second alert delay on 1-minute candles is unacceptable.
- **Scalability by symbol deduplication** — The expensive operations (tick processing, bar building, indicator computation) are per-symbol, not per-strategy. 1,000 strategies across 50 symbols = 50 bar builders and 50 indicator pipelines. Strategy evaluation (trigger level comparison) is trivially cheap. This architecture scales linearly with symbol count, not strategy count — critical for future multi-tenant deployment.

### Phase 14C: Trading Session Input — COMPLETE (Feb 19, 2026)
*"Trading Session" added as a first-class strategy input controlling when a strategy is active, how data is loaded, and when the streaming engine processes ticks.*

**Problem Statement (resolved):**
- All strategies previously assumed RTH — no way to create pre-market, after-hours, or extended-hours strategies
- Indicators like VWAP reset at session open — the definition of "session open" changes depending on the trading session
- The streaming engine and data loader needed to know which sessions are active to subscribe to the correct data windows

**Trading Session Types:**
- **RTH** (Regular Trading Hours, 9:30 AM – 4:00 PM ET) — default for all existing and new strategies
- **Pre-Market** (4:00 AM – 9:30 AM ET) — strategies that trade the pre-market session only
- **After Hours** (4:00 PM – 8:00 PM ET) — strategies that trade the after-hours session only
- **Extended Hours** (4:00 AM – 8:00 PM ET) — strategies that span the full extended session (pre-market + RTH + after hours)

**Strategy Builder Integration:**
- [x] Trading Session selector — primary input in Strategy Builder alongside Symbol, Direction, Timeframe. Selectbox with four session types, defaulting to RTH (`app.py:3262`)
- [x] Strategy schema gains `trading_session` field. Existing strategies use `.get('trading_session', 'RTH')` fallback — no migration needed
- [x] Session type displayed on strategy cards and detail page headers (`app.py:5316`)

**Data Pipeline Impact:**
- [x] `data_loader.py` — `SESSION_HOURS` dict defines all 4 session windows; `_filter_session()` filters bars to the strategy's session; all `load_market_data()` paths accept `session` parameter
- [x] `estimate_bar_count()` and `days_from_bar_count()` are session-aware (`data_loader.py:295-312`)
- [x] Backtest engine — trade entry/exit windows constrained to the strategy's trading session via session-filtered data input

**Streaming Engine Impact:**
- [x] Session gate in `realtime_engine.py:316-318` — `_is_in_session()` prevents signal evaluation outside strategy's session window
- [x] Engine subscribes to ticks for the union of all active strategies' sessions (`alert_monitor.py:237-244`)
- [x] `BarBuilder` bar boundaries clock-aligned; bars only created when ticks arrive during active session
- [x] Ticks outside a strategy's session window ignored for that strategy but processed for others with overlapping sessions

**Confluence System:**
- [x] Session filter interpreter in `general_packs.py:403-419` — "Filter trades by market session" confluence condition evaluates `IN_SESSION` / `OUT_OF_SESSION` per bar

**Known Limitations:**
- ~~VWAP does not yet reset at session boundaries~~ — **Resolved in Phase 17D.** VWAP now computes cumulative session VWAP from scratch with session-aware reset (gap > 30 min detection). Alpaca's per-bar VWAP column is ignored.

### Design Decisions (Phase 14C — Trading Session Input)
- **First-class input over confluence condition** — Trading session fundamentally shapes the data pipeline: which bars exist, when indicators reset, when the engine listens. A confluence condition is evaluated *after* data is loaded and indicators are computed — by then it's too late. Session must be known before any data loading occurs, making it a peer of Symbol, Direction, and Timeframe.
- **Four discrete sessions over custom time ranges** — Custom time ranges (e.g., "10:00 AM – 2:00 PM") add complexity with minimal benefit. The four standard sessions cover 99% of retail trading use cases and align with how brokers, exchanges, and data providers categorize market hours. Custom ranges can be added later if needed.
- **RTH as default** — The vast majority of retail strategies operate during regular trading hours. Defaulting to RTH means existing strategies require no migration and new users get sensible behavior without configuration.
- **Union subscription for streaming** — The engine subscribes to the broadest window needed across all active strategies. This is simpler than managing per-strategy subscriptions and avoids reconnection overhead when strategies with different sessions are started/stopped. The per-strategy filtering happens at evaluation time, not subscription time.

### Phase 15: Settings Page — COMPLETED (merged into Phase 10, Feb 11, 2026; Connections added Feb 17, 2026)
*Implemented as part of the sidebar-to-inline refactor in Phase 10. See Phase 10 "Settings Page" section for details.*

- [x] Settings navigation page — 6th top-level nav item
- [x] Chart Defaults — Visible Candles selectbox (replaces sidebar chart preset selector)
- [x] Default Triggers — Default Entry and Exit Trigger selectboxes for new strategies
- [x] Default Risk Management — Default Stop Loss and Target with full method+parameter config
- [x] Development — Data Seed (mock mode only)
- [x] **Connections** subpage — Alpaca API key status (masked display), data feed selector (IEX/SIP with description captions), real-time engine enable/disable toggle. Data feed default changed from IEX to SIP after Alpaca plan upgrade
- [x] Persist settings to `config/settings.json` — `load_settings()` / `save_settings()` helpers with merge-on-load for forward compatibility; Settings page "Save Settings" button writes all defaults to disk; loaded into session state on app startup with fallback to `SETTINGS_DEFAULTS`

### QA Notes — Phases 11–14 (Feb 16, 2026)
*Issues identified and resolved during QA session.*

- [x] **Webhook strategy detail page — "No data available"** — Webhook strategies now render correctly on the detail page using `stored_trades`. Price Chart, Extended Lookback, and Confluence Analysis tabs show informational messages (no market data for webhook strategies). Fixed `NoneType.startswith` crash for null `exit_trigger_confluence_id`.
- [x] **Phase 13 test data** — SPY LONG strategy (id=1) populated with spoofed live execution data: 40 live_executions (20 matched trades) across 11 trading days, 5 discrepancies (3 missed alerts, 2 phantom alerts), 42 matching alerts in alerts.json. Forward test start moved to Jan 20 for 27 days of forward testing coverage.
- [x] **Ledger record deletion** — Each ledger row now has a trash-can delete button with two-step confirmation (Yes/No). Uses existing `remove_ledger_entry()` from portfolios.py.
- [x] **Confluence pack filtering scoped to builder** — Disabled confluence packs now only affect the Strategy Builder evaluation/drill-down tabs. Existing strategies, portfolios, strategy detail pages, alert monitor, and webhook processing use ALL interpreters and GP columns regardless of enabled state. Previously, disabling a pack could cause "No trades generated" on portfolio pages.
- [x] **Strategy card BT days caption** — BT days now shows on strategy cards using `data_days` as fallback when `lookback_start_date` is not set. Caption format: `SPY LONG | BT 30d | Fwd 27d | Live 11d`.

### QA Notes — Alert Analysis & Demo Strategies (Feb 18, 2026)
*Demo strategies created for UI validation; bugs found and fixed.*

- [x] **Demo strategies created** — Strategy 19 (SPY LONG - EMA Cross Demo, standard origin, 299 trades, 118 FT trades, 104 live executions, 11 discrepancies) and Strategy 20 (AAPL LONG - Webhook Inbound Demo, webhook origin, 121 trades, 48 FT trades, 44 live executions, 4 discrepancies). Portfolios 7 and 8 with webhook configurations. 153 alert records in alerts.json.
- [x] **KeyError `exit_reason` crash** — `format_trade_table()` and `render_backtest_trade_table()` expected `exit_reason` column, but `stored_trades` only store 4 fields (entry_time, exit_time, r_multiple, win). Fixed: column presence check before including in display. Affected both webhook and standard strategies using stored_trades path.
- [x] **Timezone mismatch in `split_trades_at_boundary()`** — Hardened to handle all 4 timezone combinations: trades tz-aware + boundary naive (localize), trades tz-aware + boundary aware (convert), trades naive + boundary aware (strip), both naive (no-op). Previously only handled the first case.
- [x] **Mini equity curve 3-color support** — `render_mini_equity_curve_from_data()` now accepts optional `strat` parameter. When alert tracking is enabled with live_executions, computes green overlay trace from slippage-adjusted matched trades. Previously only rendered 2 colors (blue BT, orange FT).
- [x] **Alert tracking toggle on strategy cards** — `st.toggle("Alert Tracking")` added to each forward-testing strategy card. Mirrors the detail page toggle behavior: persists via `update_strategy()`, clears live_executions and discrepancies on disable.
- [x] **Alert Analysis tab** — New `render_alert_analysis_tab()` function (~120 lines). Conditionally appears as 8th tab in both `render_live_backtest()` and `render_forward_test_view()` when strategy has alert tracking + live executions. Three sections: summary metrics (FT vs Live KPIs with deltas), trade-by-trade comparison, discrepancy detail.

### QA Notes — Phase 14B Streaming Engine (Feb 18, 2026)
*Implementation complete. Live verification performed during RTH; after-hours tick issue identified and under investigation.*

- [x] **Syntax check** — All 4 modified files (`realtime_engine.py`, `alert_monitor.py`, `data_loader.py`, `app.py`) pass `py_compile`
- [x] **Module import** — `realtime_engine.py` imports cleanly, singleton API returns correct default status, all 5 classes instantiable
- [x] **BarBuilder unit test** — Clock-aligned bar aggregation verified: 3 ticks in same period → 1 completed bar on next-period tick. OHLCV values correct (O=first, H=max, L=min, C=last, V=sum)
- [x] **AlertCooldown unit test** — Cooldown suppression within window, firing after window, independent keys all verified
- [x] **Signal handler guard** — `alert_monitor.py` importable from non-main threads after `if __name__ == "__main__"` guard on `signal.signal()` calls
- [x] **Live streaming (market hours)** — Verified during RTH: WebSocket connects, ticks flow, bars build, pipeline runs on bar close. Working correctly during regular trading hours
- [x] **Polling bypass** — With streaming engine connected, `alert_monitor.py` subprocess sleeps instead of polling (`streaming_connected` flag verified)
- [x] **Disconnect/reconnect** — Engine detects disconnect → `streaming_connected` flips to False → poller resumes → engine reconnects with backoff. Verified
- [~] **Stop lifecycle** — Basic stop flow works (engine stops, poller killed, UI reverts). Needs verification that `streaming_engine.log` records clean shutdown correctly
- [x] **After-hours tick data** — After RTH ended, tick count initially dropped to zero. Root cause: stale app instance from prior session; restarting RoR Trader resolved the issue. After-hours ticks confirmed working via Alpaca SIP

### Phase 16: AI-Assisted Confluence Pack Builder — COMPLETE (Feb 17, 2026)
*Standardize the process for adding new indicators, interpreters, and confluence packs to the platform. A guided UI collects user intent and generates a structured prompt that can be fed to any LLM (Claude, ChatGPT, Gemini, etc.). The LLM output is pasted back, validated, and hot-loaded into the system — no manual code wiring required.*

**Problem Statement:**
- Adding new indicators today requires manually writing Python indicator functions, interpreter logic, trigger definitions, and template registration — a developer-only task
- Users discover compelling indicators on TradingView or in trading education but have no way to bring them into the platform
- The existing clone-and-customize flow (rename, adjust parameters) only works with indicators already in the system

**Design Principles:**
- **LLM-agnostic** — The system generates prompts and consumes structured output; it never calls an AI API directly. Users choose whatever LLM they prefer or have access to
- **Standardized schema** — A strict Pack Spec format (JSON + Python functions) that any generated pack must conform to, enabling deterministic validation regardless of which LLM produced it
- **Future API-ready** — The prompt-generation and validation layers are cleanly separated so a direct API integration (Claude, OpenAI, etc.) can be dropped in later without rearchitecting

**Phase 16A — Pack Spec Standard & User Packs Infrastructure (COMPLETE):**
- [x] **Pack Spec schema definition** — `src/pack_spec.py` defines manifest schema, allowed imports, disallowed calls/modules. `validate_manifest()`, `validate_python_file()` (AST-based safety check), `validate_function_exists()` for function signature verification
- [x] **`user_packs/` directory structure** — Convention: `user_packs/<pack_slug>/` containing `indicator.py`, `interpreter.py`, and `manifest.json` (metadata + version + parameters + outputs). Kept separate from core pack files
- [x] **Hot-load registry** — `src/pack_registry.py` scans `user_packs/` on startup, validates manifests, AST-checks Python files, dynamically imports via `importlib.util`, and registers into `TEMPLATES`, `INTERPRETER_FUNCS`, `TRIGGER_FUNCS`, `GROUP_INDICATOR_FUNCS`. Auto-creates `ConfluenceGroup` for new packs
- [x] **Registry-based dispatch** — Refactored `interpreters.py` (`run_all_interpreters()`, `detect_all_triggers()`) and `indicators.py` (`run_indicators_for_group()`) from hard-coded dispatch to mutable registries. Built-in and user packs share the same execution path
- [x] **User pack CRUD** — User Packs tab on Confluence Packs page shows installed packs with expandable cards (metadata, parameters, outputs/triggers, source files). Delete button with confirmation removes pack from disk and all registries. Refresh button hot-reloads without restart

**Phase 16B — Prompt Generator & Paste-Back Workflow (COMPLETE):**
- [x] **Architecture Context Document** — `src/pack_builder_context.py` generates a comprehensive context document containing: Pack Spec schema, indicator/interpreter function signature patterns, column naming conventions, complete examples of all 3 pack types (TF Confluence, General, Risk Management), Pine Script → Python translation reference, and output format instructions
- [x] **Pack Builder UI** — "Pack Builder" tab on Confluence Packs page with guided form: pack type selector (TF Confluence / General / Risk Management) with mutual exclusivity, plain-language description, optional Pine Script/pseudocode input, dynamic parameter rows (name, type, default), "Generate Prompt" button
- [x] **Prompt assembly engine** — Combines architecture context + user form inputs into a structured prompt. Type-specific instructions and examples included based on selected pack type
- [x] **Copy-to-clipboard UX** — Generated prompt displayed in scrollable text area with copy button and clear instructions for pasting into preferred AI assistant
- [x] **Response paste-back area** — Text area for pasting LLM response with "Import Pack" button

**Phase 16C — Validation, Preview & Installation (COMPLETE):**
- [x] **Response parser** — Extracts JSON manifest and Python code blocks from pasted LLM output. Tolerant of markdown formatting, extra commentary, whitespace variations
- [x] **Schema validation** — Validates extracted JSON against Pack Spec schema with clear error messages for missing fields, invalid types, naming convention violations
- [x] **Code validation** — AST-based validation: verifies function signatures, checks for disallowed imports (only `pandas`, `numpy`, `math`, `collections` allowed), flags I/O or network calls, rejects `exec`/`eval`/`compile`/`__import__`
- [x] **Column collision check** — Verifies generated column names don't collide with existing built-in or user pack columns
- [x] **Dry-run preview** — Runs generated indicator + interpreter on live market data (30 days SPY via Alpaca SIP). Shows: dynamic candlestick chart with indicator overlays (using TradingView Lightweight Charts), interpreter state distribution bar chart, trigger fire counts, computed column samples. Preview uses real data, not mock
- [x] **Install to `user_packs/`** — On approval, writes pack files to `user_packs/<pack_slug>/`, registers into hot-load registry, and surfaces new pack in Confluence Pack settings with enable/disable checkbox

**Pine Script Porting (built into prompt context):**
- [x] Architecture context document includes Pine Script → Python translation reference (e.g., `ta.ema()` → `df['close'].ewm(span=N).mean()`, `ta.atr()` → ATR helper, `ta.crossover()` → trigger pattern, `ta.rsi()` → RSI formula)
- [x] When user provides Pine Script input, the prompt assembly engine wraps it with translation-specific instructions
- [x] Common Pine Script patterns covered: overlays, oscillators, band-based indicators, volume indicators

**Integration Points:**
- [x] User packs appear in Confluence Packs settings pages alongside built-in packs with enable/disable checkboxes
- [x] User packs available in Strategy Builder trigger/confluence selection dropdowns
- [x] User packs participate in the full pipeline: `prepare_data_with_indicators()` → `run_all_interpreters()` → `detect_all_triggers()` → `generate_trades()`
- [ ] Version tracking on user-created packs — edit history stored in manifest, rollback by restoring previous version files (deferred to Phase 22)

### Phase 17: Indicator & Confluence Maturity
*Validate, expand, and harden the indicator/confluence library to a production-ready standard. Upgrade charting infrastructure to support TradingView-quality visualizations. Goal: a trusted foundation of indicators, interpreters, and chart rendering that can support real trading strategies and live algorithmic execution with confidence.*

**Motivation:**
- Before building real strategies and deploying live webhooks, the indicator foundation must be validated against TradingView for data integrity
- The Pack Builder (Phase 16) enables adding new indicators, but known charting/plotting limitations need to be addressed (e.g., support/resistance channels, band fills, multi-line overlays)
- A broader indicator library gives more confluence options for strategy construction and optimization
- Users need to visually verify that indicators, interpreter states, and trigger events behave correctly on the chart before trusting them for live trading

**Phase 17A: Charting Infrastructure Upgrade** *(COMPLETE)*
*Fork the `streamlit-lightweight-charts` wrapper to unlock the TradingView Lightweight Charts v4.1+ Plugins/Primitives API. The core JS library already supports all needed features — the bottleneck is the Python→JS wrapper only exposing the 6 basic series types and never calling `attachPrimitive()`.*

Track A — Quick wins (no fork needed, passthrough to existing LWC JS):
- [x] `reference-indicators/` folder created with Pine Script references (MACD, Swing 123, SR Channel, UT Bot Alerts, VWAP, RVOL Status, UT Bot Conflu MAIN, Strat Assistant, SuperTrend)
- [x] **Per-candle dynamic coloring** — `color`, `wickColor`, `borderColor` per candle data dict, driven by `candle_color_column` in `plot_config`. Wired through `render_price_chart()` to all chart call sites
- [x] **Dashed/dotted line styles** — `lineStyle: 0-4` passthrough in Line series options, driven by `plot_config.line_styles` per indicator column. Wired through all overlay and oscillator pane renderers
- [x] **Horizontal reference lines** — constant-value Line series for MACD zero line (dashed gray) and generic oscillator panes via `plot_config.reference_lines`
- [x] **Band fills** — driven by `plot_config.band_fills` in manifest, wired through all 6 chart call sites. Initially implemented as overlapping Area series (Track A), then upgraded to proper BandIndicator primitive (Track B)
- [x] **`plot_config` manifest field** — new optional field in `pack_spec.py` with validation for `band_fills`, `reference_lines`, `line_styles`, `candle_color_column`. Bollinger Bands user pack updated as reference implementation
- [ ] **Extended marker shapes** — `circle` and `square` in addition to existing `arrowUp`/`arrowDown`. Use for SuperTrend/UT Bot signal markers (deferred — already supported by LWC, just needs pack config)

Track B — Fork work (vendored wrapper with LWC v4.2+):
- [x] **Vendored fork created** — `streamlit_lwc_fork/` directory with upstream source, custom `setup.py` (v0.8.0.dev0), installed via `pip install -e`
- [x] **LWC JS upgraded to v4.2.3** — `package.json` bumped from `^4.0.0` to `^4.2.0`, production build successful (142KB gzipped). Pinned `streamlit-component-lib-react-hooks@1.1.1` for Node 18 / webpack compatibility
- [x] **`createPriceLine()` support** — series-level price lines wired in `LightweightCharts.tsx`. Python side: add `"priceLines": [...]` to any series dict
- [x] **Primitives dispatcher skeleton** — `chartsData[i].primitives` array processed with switch dispatch by type. Plugin type cases to be added as implementations land
- [x] **BandIndicator plugin** — TypeScript primitive that draws filled polygon between upper/lower price curves. Replaces Area series interim. Wired through primitives dispatcher and `render_price_chart()`
- [x] **Rectangle plugin** — TypeScript primitive for box annotations at time/price coordinates (S/R zones, support/resistance boxes)
- [x] **AnchoredText plugin** — TypeScript primitive for text labels at time/price coordinates (state labels, pattern names)
- [x] **SessionHighlighting plugin** — TypeScript primitive for full-height background color zones using `drawBackground()` (interpreter state visualization)

### Design Decisions (Phase 17A — Charting Infrastructure)
- **Fork existing wrapper over migrating to ECharts or other library** — The current `streamlit-lightweight-charts` (v0.7.20) wrapper already renders TradingView-identical charts. The core LWC v4.1+ JS library supports all needed features via its Plugins/Primitives API (`attachPrimitive()`, `ISeriesPrimitive`, `CanvasRenderingContext2D`). Forking the wrapper to wire through primitives is a scoped JS change that preserves all existing chart code and the TradingView visual identity. ECharts was evaluated as the strongest alternative (checks every feature box, excellent performance) but would require rewriting `render_price_chart()` and produces a TradingView-*like* but not identical look. The fork approach is lower risk and lower migration cost.
- **Quick wins before fork** — Per-candle coloring, dashed lines, area series band fills, and reference lines all work through the existing wrapper today (options pass through to the JS library). These were implemented first for immediate value while the fork work proceeded.
- **Manifest-driven plot configuration** — Rather than hardcoding each indicator's chart rendering in `app.py`, the pack manifest `plot_config` field lets packs declaratively specify fill regions, line styles, reference lines, and candle coloring. The rendering engine reads the manifest and dispatches to the appropriate chart features. This keeps the Pack Builder workflow intact — LLM-generated packs can specify their own charting requirements.
- **Vendored fork over upstream PR** — Upstream repo (`freyastreamlit/streamlit-lightweight-charts`) appears unmaintained (last commit 2023). Vendoring gives full control over LWC version, TypeScript modifications, and plugin integration without waiting on upstream merges.

**Phase 17B: Interpreter & Trigger Chart Overlays**
*Add toggle controls on preview tabs that overlay interpreter states and trigger events directly on the price chart. Bridges the gap between tabular data and visual chart analysis — users can see exactly when and where conditions changed and triggers fired.*

- [x] **"Show Conditions" toggle** — next to interpreter state tables on preview tabs. When enabled:
  - Background color bands on the price chart for each interpreter state transition (e.g., green band during `FULL_BULL_STACK`, red during `FULL_BEAR_STACK`, gray during `NEUTRAL`)
  - Text label at each state transition point showing the new state name
  - Selector for which interpreter to overlay when multiple are active (one at a time to avoid visual clutter)
- [x] **"Show Triggers" toggle** — next to trigger event tables on preview tabs. When enabled:
  - Marker + text label at each bar where a trigger fired (e.g., "LONG ENTRY" arrow, "EXIT" marker)
  - Distinct from trade entry/exit markers — shows raw trigger fires regardless of whether confluence filtered them into an actual trade
  - Helps debug "why didn't a trade happen here?" — trigger fired but interpreter state was wrong, or vice versa
- [x] Both toggles off by default to keep charts clean; user enables as needed for analysis

**Phase 17C: Pine Script Export**
*Add a "Copy Pine Script" button to indicator preview/code tabs. Enables cross-referencing RoR Trader indicator behavior against TradingView by pasting the same indicator into both platforms.*

- [x] **Copy Pine Script button** — on preview tabs for indicators that have a Pine Script reference in `reference-indicators/`. One-click copy to clipboard
- [x] **Pack Builder Pine Script output** — when creating a new indicator via Pack Builder, optionally generate a Pine Script equivalent alongside the Python implementation. Lets users verify the indicator plots identically in TradingView
- [ ] **Future: Composite Pine Script generator** — given a strategy's full confluence setup (multiple indicators + interpreters + triggers), generate a single TradingView study that reproduces the complete signal chain. Useful for visual validation of the entire strategy logic against TradingView charts. (Deferred — scoping TBD)

**Phase 17D: Indicator Audit & Expansion**
*Validate existing indicators and add new ones from the reference library.*

- [x] Audit all existing built-in indicators against TradingView Pine Script references — verified EMA Stack, MACD (Line + Histogram), VWAP, RVOL, UT Bot, Bollinger Bands, SR Channels. Full audit documented in `docs/Implementation_Spec_Phase_17D.md`
- [x] Fix EMA Stack `LMS` dead code bug — condition `p < s < m < l` was identical to MLS; corrected to `l > m > s > p` (true bear stack)
- [x] Fix VWAP session-aware reset — fallback cumulative VWAP now resets at session boundaries (gaps > 30 min); SD bands use session-aware expanding deviation
- [x] Implement UT Bot indicator + interpreter + triggers — was template-only with empty function lists; now fully implemented with ATR trailing stop, BULL/BEAR states, buy/sell triggers
- [x] Add new packs from reference library:
  - SuperTrend (`user_packs/supertrend/`) — ATR-based trend following with 4 states (BULL_TRENDING, BULL_NEAR_STOP, BEAR_TRENDING, BEAR_NEAR_STOP) and 4 triggers
  - Swing 123 (`user_packs/swing_123/`) — C2/C3 pattern detection with candle coloring, 5 states and 4 triggers
  - Strat Assistant (`user_packs/strat_assistant/`) — Bar pattern classification (1/2/3), 16+ strategy combos, Shooter/Hammer signals, 4 states and 6 triggers. FTC deferred (requires multi-TF infrastructure)
- [x] Review interpreter output states for consistency — all packs produce mutually exclusive, exhaustive states
- [x] Update Pack Builder context document — added `plot_config` documentation, `candle_color_column` guidance, Wilder smoothing note, expanded reserved names for all packs
- [x] Update `pack_spec.py` reserved names — trigger prefixes, interpreter keys, and indicator columns updated for all installed packs
- [x] Add execution mode tags (`[C]`, `[I?]`, `[I]`) to all Outputs & Triggers displays — Confluence Packs, User Packs, General Packs, and Pack Builder review panel. Intra-bar candidate triggers identified and tagged `[I?]`
- [x] Split EMA Stack into pure EMA ordering (no price) + new EMA Price Position pack (4-char PSML codes with all 24 permutations). EMA Stack now only compares Short/Mid/Long relative to each other. EMA Price Position includes price in the ordering with triggers for price crossing Short/Mid EMAs.
- [x] Fix VWAP SD bands — replaced simple expanding std with volume-weighted standard deviation matching TradingView formula
- [x] Fix UT Bot preview — removed `utbot_direction` from chart overlay columns, added UTBOT to interpreter config so it actually runs
- [x] Add Show Conditions background painting to General Packs preview tab
- [x] Fix VWAP cumulative calculation — Alpaca's `vwap` column is a per-bar VWAP (volume-weighted average within each 1-min bar), not a cumulative session VWAP. Using it collapsed SD bands to ~0. Now always computes our own cumulative session VWAP from scratch, matching TradingView's `ta.vwap()` behavior.
- [x] Add Trading Session dropdown (RTH / Pre-Market / After Hours / Extended Hours) to all preview tabs — TF Confluence, General Packs, and Pack Builder. Previews load data with `session="Extended Hours"` so all session filters work.
- [x] Auto-migration for EMA Price Position default confluence group — existing configs automatically gain `ema_price_position_default` group on load

**End State:** A library of validated, production-quality indicators and interpreters with TradingView-quality chart rendering. Chart overlays for interpreter states and trigger events provide full visual transparency into the signal chain. Pine Script export enables cross-platform validation. New strategies built after this phase can be trusted for live trading without concern about data integrity or rendering issues.

### Phase 18: Multi-Timeframe Confluence
*Evaluate confluence conditions across multiple timeframes — a single strategy can check higher-timeframe context (e.g., 15-min EMA trend) before entering on a lower timeframe (e.g., 1-min candles). One of the most common edges in professional trading.*

**Problem Statement:**
- Currently, all TF confluence conditions evaluate on the strategy's primary timeframe only. A 1-min strategy with Bollinger Bands as confluence only sees BB values on 1-min bars
- Traders commonly use higher-timeframe context for confluence (e.g., "only take 1-min longs when 15-min EMA Stack is bullish")
- TF conditions currently display without a timeframe prefix (e.g., "Bollinger Bands Default: Squeeze Mid"), which is ambiguous once multiple timeframes are in play

**Phase 18A: Timeframe Management & Data Model — COMPLETE (Feb 19, 2026)**
- [x] New "Timeframes" sub-page under Confluence Packs — enable/disable individual timeframes (1m through 1d), grid layout with checkboxes
- [x] `enabled_timeframes` setting in `config/settings.json` — persisted list of enabled timeframes, defaults to `["1Min"]`
- [x] TF label mapping utilities in `data_loader.py` — `TF_LABELS`, `TF_FROM_LABEL`, `get_tf_label()`, `get_tf_from_label()`, `get_required_tfs_from_confluence()`
- [x] Matrix summary display — shows N timeframes x M packs = N*M condition groups available
- [x] Condition matrix preview table when multiple timeframes enabled

**Phase 18B: Multi-TF Backtest Data Pipeline**
- [ ] `prepare_data_with_indicators()` extended to accept + process secondary timeframes
- [ ] `resample_to_timeframe()` utility — resample primary TF OHLCV to coarser timeframes using pandas
- [ ] Run indicators + interpreters independently on each secondary TF's resampled DataFrame
- [ ] Forward-fill interpreter STATE columns only (not raw indicators) to primary TF index with `{INTERP}__{tf_label}` column naming
- [ ] `get_mtf_confluence_records()` helper — builds confluence records for primary + all secondary TFs
- [ ] TF confluence conditions display with timeframe prefix: `5m: EMA Stack Default: SML`, `15m: MACD Line: M>S+`
- [ ] Primary TF records keep `"1M"` prefix (backward-compatible). Secondary TF records use lowercase labels (`"5m"`, `"15m"`, etc.)
- [ ] Strategy's required secondary TFs inferred from selected confluence conditions (no explicit schema field needed)
- [ ] Multi-timeframe data alignment — higher-TF values forward-filled to primary TF index (standard TradingView `request.security()` pattern)

**Phase 18C: Streaming Engine MTF Integration**
- [ ] `SymbolHub.start()` registers required secondary TFs from strategy confluence conditions
- [ ] `SymbolHub._on_bar_close()` gathers secondary TF interpreter states from other BarBuilders
- [ ] `detect_signals()` accepts `secondary_tf_dfs` parameter, runs pipeline on secondary TFs, builds MTF confluence records
- [ ] Alert monitor polling fallback loads secondary TF data for strategies that need it

**Strategy Builder UI (18B):**
- [ ] Drill-down TF Conditions tab shows conditions grouped by timeframe with timeframe headers
- [ ] Optimizable Variables box shows timeframe prefix on selected conditions
- [ ] Auto-Search can search across timeframe × condition combinations

### Design Decisions (Phase 18 — Multi-Timeframe Confluence)
- **Timeframe management page over per-strategy timeframe config** — A centralized page where users enable timeframes mirrors the existing pattern for confluence packs (enable/disable globally, use in any strategy). This avoids per-strategy timeframe UI complexity and keeps the drill-down experience consistent.
- **Matrix approach (timeframes × packs)** — Rather than manually configuring "run BB on 5m" per strategy, the system generates all valid combinations from enabled timeframes and enabled packs. The drill-down then lets users pick the specific TF:condition pairs that improve their strategy. This is consistent with how entry/exit/stop/target drill-down already works.
- **Forward-fill for multi-TF alignment** — A 15-min EMA value computed at 9:45:00 should apply to all 1-min bars from 9:45:00 to 9:59:59. Forward-filling the higher-TF series onto the primary-TF index is the standard approach (same as TradingView's `request.security()` for MTF indicators). The value updates when the higher-TF bar closes.
- **After Phase 14B** — The streaming engine's `SymbolHub` with multiple `BarBuilder` instances per symbol is the natural foundation for real-time MTF evaluation. Building MTF confluence first would require the backtest-only pipeline now and streaming retrofit later — double integration work. Phase 14B's architecture was designed with this use case in mind (`SymbolHub` accommodates single-strategy-multiple-timeframes, not just multiple-strategy-different-timeframes).

### Phase 19: Intra-Bar Trigger Evaluation
*Evaluate select triggers tick-by-tick against pre-computed levels instead of waiting for bar close — enables faster entries/exits for price-vs-level crossover triggers while preserving bar-close semantics for pattern-based triggers.*

**Problem Statement:**
- All triggers currently evaluate once per bar close, even when the trigger condition is simply "price crossed above/below a known level" (e.g., VWAP cross, UT Bot stop, SuperTrend flip, Bollinger Band cross)
- For fast-moving markets, waiting for bar close means entries can be late by up to one full bar duration (1–5 minutes)
- Pattern-based triggers (EMA stack changes, MACD histogram shifts, Swing 123 patterns, Strat combos) inherently require a completed bar and must remain bar-close

**Trigger Classification:**
- `[C]` Bar-Close — trigger requires a completed candle to evaluate (EMA crossovers, MACD histogram, bar patterns, candle patterns). Remains bar-close only.
- `[I]` Intra-Bar — trigger compares current tick price against a pre-computed level and can fire mid-bar. Level is recalculated on each bar close.
- `[I?]` Intra-Bar Candidate — trigger *could* be intra-bar but is not yet wired. UI tags already show these.

**Intra-Bar Candidates:**
- [ ] VWAP crosses (price vs VWAP line, price enters/exits SD band extremes)
- [ ] UT Bot signals (price crosses trailing stop level)
- [ ] SuperTrend flips (price crosses ST line)
- [ ] Bollinger Band crosses (price vs upper/lower/basis)
- [ ] SR Channel breaks (price vs nearest support/resistance level)
- [ ] RVOL spikes (volume vs threshold — bar-volume based, may remain bar-close)

**Streaming Engine Integration (depends on Phase 14B):**
- [ ] `SymbolHub.on_trade()` path gains an intra-bar trigger evaluation hook
- [ ] Intra-bar triggers register their "level" (a float) after each bar close; tick handler checks `price >= level` or `price <= level`
- [ ] When intra-bar trigger fires, alert and confluence state update immediately (do not wait for bar close)
- [ ] Backtest approximation — use bar high/low to determine if level was crossed within the bar; entry price = the level (limit-fill assumption)

**UI:**
- [x] Execution tags (`[C]`, `[I]`, `[I?]`) already display on all Outputs & Triggers tabs (implemented in Phase 17D)
- [ ] Per-trigger toggle in strategy drill-down: "Evaluate intra-bar" checkbox (default off, only shown for `[I]`-capable triggers)
- [ ] Alert monitor respects intra-bar setting — fires alert on tick cross rather than bar close

### Design Decisions (Phase 19 — Intra-Bar Trigger Evaluation)
- **Level-based approach** — Rather than re-running the full indicator/interpreter pipeline on every tick (expensive), intra-bar triggers simply compare the current price against a pre-computed level that updates on each bar close. This keeps tick-path evaluation O(1) per trigger.
- **After Phase 18 (Multi-Timeframe)** — Intra-bar evaluation shares the same streaming engine infrastructure as multi-timeframe confluence. Building MTF first ensures the `SymbolHub` architecture is solid before adding tick-level evaluation on top.
- **Opt-in per trigger** — Not all traders want intra-bar evaluation (some prefer bar-close discipline). The per-trigger checkbox keeps it explicit.
- **Backtest uses high/low approximation** — Without true tick data in backtest, checking if bar high ≥ level (for long) or bar low ≤ level (for short) determines if the level was breached. Entry price uses the level itself (equivalent to a limit order fill). This is the standard approach in TradingView strategy backtests.

### Phase 20: General & Risk Management Pack Audit
*Audit, validate, and expand the General Confluence and Risk Management pack libraries. Ensure condition logic, parameter schemas, preview rendering, and pipeline integration match production expectations. Identify structural improvements now that the indicator audit (17D) and intra-bar infrastructure (19) are complete.*

**General Packs Audit:**
- [ ] Audit all general pack templates (Time of Day, Trading Session, Day of Week, Calendar Filter) — verify condition evaluation logic, parameter schemas, output states, and preview rendering
- [ ] Evaluate whether general packs need triggers (e.g., session open/close trigger, time-window entry trigger)
- [ ] Ensure preview tab Show Conditions overlay renders correctly for all general pack types
- [ ] Identify any new general pack templates to add (e.g., market regime, volatility filter, news blackout)

**Risk Management Packs Audit:**
- [ ] Audit all risk management templates (ATR-Based, Fixed Dollar, Percentage, Swing, Risk:Reward) — verify stop/target calculation logic, parameter schemas, and interaction with trade execution
- [ ] Evaluate intra-bar exit timing for risk management — stops and targets should ideally evaluate tick-by-tick once Phase 19 infrastructure is available (exit at stop price rather than waiting for bar close)
- [ ] Review risk management parameter ranges and defaults for realistic trading scenarios
- [ ] Ensure risk management packs integrate correctly with the execution model (stop_config, target_config in strategy schema)
- [ ] Identify structural improvements — e.g., trailing stops, breakeven stops, partial profit taking, time-based exits

**Preview & UI Improvements:**
- [ ] Consistent preview rendering across all pack types (TF Confluence, General, Risk Management) — condition overlay, state timeline, distribution metrics
- [ ] Risk management preview — visualize stop/target levels on sample trades

### Phase 21: Scanner Strategy Origin
*Strategy origin not tied to a single ticker — runs against a universe of stocks matching screener criteria. Targets active day trading / scalping use cases (S&B Capital, Warrior Trading style).*

- [ ] Add "Scanner" option to Strategy Origin selectbox
- [ ] Scanner configuration fields — screener criteria (price range, volume, gap %, sector, float), universe source (Alpaca screener APIs), scan frequency
- [ ] 1:many ticker architecture — a single scanner strategy evaluates triggers across all matching symbols; trades attributed to individual symbols but KPIs aggregated at strategy level
- [ ] Scanner backtest — run trigger/confluence evaluation across historical screener results; requires architecture planning for data volume and performance
- [ ] Scanner forward test — periodic scan + signal detection across matching symbols in real-time
- [ ] Requires separate planning session for architecture given fundamental 1:many ticker relationship vs. current 1:1 model

### Phase 22: Low-Priority Cleanup & Enhancements
*Deferred items and nice-to-haves — polish, performance, and convenience improvements.*

**Expanded Backtest Range:**
- [ ] Date range picker on My Strategies page or strategy detail — select a custom backtest start date earlier than the original
- [ ] Run full pipeline for the expanded window and merge new backtest trades into `stored_trades` (additive, does not affect existing forward test data)
- [ ] Distinct from Data View filter (Phase 10C) — filter shows subsets of existing data instantly; expanded backtest generates new data by running the pipeline

**Optimization Workflow Polish (deferred from Phase 9/10):**
- [ ] Trigger parameters visible and expandable in Optimizable Variables — show EMA periods, ATR multiplier, etc. (not just trigger name)
- [ ] Stop/target variation tags on trades — tag individual trades with pack ID when running multi-backtest comparisons
- [ ] Multi-backtest progress indicator + caching — progress bar for SL/TP drill-down; cache keyed on (symbol, timeframe, date range, strategy config, pack ID)
- [ ] Lazy tab loading — only compute drill-down results when a tab is first opened, not all 6 on page load

**UX Polish:**
- [ ] Utility buttons on Portfolios page — "Portfolio Requirements" and "Webhook Templates" links next to "New Portfolio" button

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
