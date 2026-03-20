# RoR Trader - Product Requirements Document (PRD)

**Version:** 0.53
**Date:** March 13, 2026
**Author:** Kevin Johnson
**Status:** Phase 32 (Crypto Support) COMPLETE. Phase 30A–L (Unified Chart Engine) COMPLETE. Phase 28 (Ralph Wiggum Alert Engine) COMPLETE. Phase 22A–E (Web Deployment) COMPLETE. Phase 31 (Polygon.io Migration) planned. Phases 11–21, 24 complete

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
- **Display Timezone** - All timestamps (text displays, price chart axes, trade markers, alert badges) rendered in a user-chosen timezone (US/Eastern, US/Central, US/Mountain, US/Pacific, UTC). Does not affect calculations or stored data. ✓
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
- **Capabilities Used:**
  - Historical bars (daily, intraday down to 1-minute) via `StockHistoricalDataClient` and `CryptoHistoricalDataClient`
  - Real-time trade WebSocket via `StockDataStream` (equities) and `CryptoDataStream` (crypto)
  - Equity market data (US stocks) — SIP and IEX feeds
  - Cryptocurrency market data — BTC/USD, ETH/USD, and all Alpaca-supported crypto pairs (24/7)
- **Asset Types:** Equities (initial, SIP/IEX feeds) and Crypto (added Phase 32, CryptoFeed.US)

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
| **Supported Markets** | Equities + Crypto | Equities via Alpaca SIP/IEX; Crypto via Alpaca CryptoFeed (Phase 32). Futures not yet supported. |
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
- [x] New "Timeframes" sub-page under Confluence Packs — enable/disable individual timeframes (5s through 1d), grid layout with checkboxes
- [x] `enabled_timeframes` setting in `config/settings.json` — persisted list of enabled timeframes, defaults to `["1Min"]`
- [x] TF label mapping utilities in `data_loader.py` — `TF_LABELS`, `TF_FROM_LABEL`, `get_tf_label()`, `get_tf_from_label()`, `get_required_tfs_from_confluence()`
- [x] Sub-minute timeframes (5Sec, 10Sec, 15Sec, 30Sec) — available for streaming engine only, marked "(streaming)" in UI, no REST API / backtest support
- [x] `SUB_MINUTE_TIMEFRAMES` set and `BARS_PER_DAY` entries for sub-minute timeframes
- [x] Matrix summary display — shows N timeframes x M packs = N*M condition groups available
- [x] Condition matrix preview table when multiple timeframes enabled

**Phase 18B: Multi-TF Backtest Data Pipeline — COMPLETE (Feb 19, 2026)**
- [x] `prepare_data_with_indicators()` extended with `secondary_tfs` parameter — resamples primary TF data to each secondary TF
- [x] `resample_to_timeframe()` utility in `data_loader.py` — resamples OHLCV to coarser timeframes using pandas
- [x] Indicators + interpreters run independently on each secondary TF's resampled DataFrame
- [x] Forward-fills interpreter STATE columns only (not raw indicators) to primary TF index with `{INTERP}__{tf_label}` column naming
- [x] `get_mtf_confluence_records()` helper in `interpreters.py` — builds confluence records for primary ("1M") + all secondary TFs (lowercase labels)
- [x] `format_confluence_record()` shows timeframe prefix for non-primary TFs: `5m: EMA Stack (Default): SML`
- [x] `generate_trades()` accepts `secondary_tf_map` parameter — uses MTF records for confluence filtering and trade tagging
- [x] Strategy's required secondary TFs inferred from confluence conditions via `get_required_tfs_from_confluence()`
- [x] All trade generation call sites wired: strategy builder, forward test, incremental trades, strategy detail, extended backtest, drill-down analysis (entry/exit triggers, exit combos, risk management)
- [x] Drill-down groups conditions by timeframe with section headers when MTF data present
- [x] `get_secondary_tf_map()` utility extracts TF map from `__`-suffixed column names
- [x] `_get_secondary_tfs()` helper computes enabled secondary TFs (excludes primary TF and sub-minute streaming-only TFs)

**Phase 18C: Streaming Engine MTF Integration — COMPLETE (Feb 19, 2026)**
- [x] `StreamingEngine.start()` registers required secondary TFs from strategy confluence conditions via `get_required_tfs_from_confluence()`
- [x] `SymbolHub._on_bar_close()` gathers secondary TF histories from other BarBuilders and passes as `secondary_tf_dfs` to `detect_signals()`
- [x] `detect_signals()` accepts `secondary_tf_dfs` parameter — runs pipeline on each secondary TF df, forward-fills interpreter states, builds MTF confluence records via `get_mtf_confluence_records()`
- [x] Alert monitor polling fallback loads secondary TF data per strategy and passes to `detect_signals()`

### Design Decisions (Phase 18 — Multi-Timeframe Confluence)
- **Timeframe management page over per-strategy timeframe config** — A centralized page where users enable timeframes mirrors the existing pattern for confluence packs (enable/disable globally, use in any strategy). This avoids per-strategy timeframe UI complexity and keeps the drill-down experience consistent.
- **Matrix approach (timeframes × packs)** — Rather than manually configuring "run BB on 5m" per strategy, the system generates all valid combinations from enabled timeframes and enabled packs. The drill-down then lets users pick the specific TF:condition pairs that improve their strategy. This is consistent with how entry/exit/stop/target drill-down already works.
- **Forward-fill for multi-TF alignment** — A 15-min EMA value computed at 9:45:00 should apply to all 1-min bars from 9:45:00 to 9:59:59. Forward-filling the higher-TF series onto the primary-TF index is the standard approach (same as TradingView's `request.security()` for MTF indicators). The value updates when the higher-TF bar closes.
- **After Phase 14B** — The streaming engine's `SymbolHub` with multiple `BarBuilder` instances per symbol is the natural foundation for real-time MTF evaluation. Building MTF confluence first would require the backtest-only pipeline now and streaming retrofit later — double integration work. Phase 14B's architecture was designed with this use case in mind (`SymbolHub` accommodates single-strategy-multiple-timeframes, not just multiple-strategy-different-timeframes).

### Phase 19: Intra-Bar Trigger Evaluation — COMPLETE (Feb 19, 2026)
*Evaluate select triggers tick-by-tick against pre-computed levels instead of waiting for bar close — enables faster entries/exits for price-vs-level crossover triggers while preserving bar-close semantics for pattern-based triggers.*

**Problem Statement:**
- All triggers currently evaluate once per bar close, even when the trigger condition is simply "price crossed above/below a known level" (e.g., VWAP cross, UT Bot stop, SuperTrend flip, Bollinger Band cross)
- For fast-moving markets, waiting for bar close means entries can be late by up to one full bar duration (1–5 minutes)
- Pattern-based triggers (EMA stack changes, MACD histogram shifts, Swing 123 patterns, Strat combos) inherently require a completed bar and must remain bar-close

**Trigger Classification:**
- `[C]` Bar-Close — trigger requires a completed candle to evaluate (EMA crossovers, MACD histogram, bar patterns, candle patterns). Remains bar-close only.
- `[I]` Intra-Bar — trigger compares current tick price against a pre-computed level and can fire mid-bar. Level is recalculated on each bar close.

**Phase 19A: Dual Trigger Definitions + TriggerLevelCache — COMPLETE**
- [x] VWAP crosses — `cross_above_ib`, `cross_below_ib`, `enter_upper_extreme_ib`, `enter_lower_extreme_ib`
- [x] UT Bot signals — `buy_ib`, `sell_ib`
- [x] EMA Price Position — `cross_short_up_ib`, `cross_short_down_ib`, `cross_mid_up_ib`, `cross_mid_down_ib`
- [x] SuperTrend flips — `bull_flip_ib`, `bear_flip_ib` (user pack manifest)
- [x] Bollinger Band crosses — `cross_upper_ib`, `cross_lower_ib`, `cross_basis_up_ib`, `cross_basis_down_ib` (user pack manifest)
- [x] SR Channel breaks — `resistance_broken_ib`, `support_broken_ib` (user pack manifest)
- [x] `INTRABAR_LEVEL_MAP` constant — maps 18 trigger bases to indicator columns and crossing directions
- [x] `TriggerLevelCache` rewrite — O(1) crossing detection with `prev_side` tracking per trigger key
- [x] RVOL/vwap_return_to_vwap remain bar-close only (not viable for level-cross approach)

**Phase 19B: Trigger Column Wiring — COMPLETE**
- [x] `column_base` field on `TriggerDefinition` — `_ib` triggers share boolean columns with `[C]` counterparts
- [x] `_ib` suffix stripping in `generate_trades()` for entry/exit column lookup
- [x] Both `[C]` and `[I]` variants appear in drill-down results and trigger dropdowns

**Phase 19C: Streaming Engine Tick-Level Evaluation — COMPLETE**
- [x] `on_tick()` calls `_check_intrabar_triggers()` on every tick
- [x] Session gate, confluence gate (bar-close confluence state), cooldown, dedup
- [x] Alerts fire mid-bar when price crosses cached indicator level
- [x] Bar-close dedup prevents duplicate alerts when intra-bar already fired

**Phase 19D: Backtest Level-Fill Pricing — COMPLETE**
- [x] `[I]` entry triggers fill at indicator level price (not bar close)
- [x] Bar high/low validation — level must be reached within bar for fill
- [x] `[I]` exit triggers use same level-fill logic
- [x] `[C]` triggers unchanged (fill at bar close)

**Phase 19E: Chart Price Arrows — COMPLETE**
- [x] Left ▸ arrow (target price) — always shown on entry/exit candles from trades DataFrame
- [x] Right ◂ arrow (alert price) — only shown when real alerts are recorded (from alerts.json)
- [x] Gap between arrows shows real slippage
- [x] Existing below/above bar arrows unchanged

**Phase 19F: Polish — COMPLETE**
- [x] `:violet[Intra-bar]` badge on alerts with `source == "intra_bar"` (4 display locations)
- [x] PRD and implementation spec updated

### Design Decisions (Phase 19 — Intra-Bar Trigger Evaluation)
- **Level-based approach** — Rather than re-running the full indicator/interpreter pipeline on every tick (expensive), intra-bar triggers simply compare the current price against a pre-computed level that updates on each bar close. This keeps tick-path evaluation O(1) per trigger.
- **Dual triggers, not checkboxes** — Instead of a per-trigger toggle, each eligible trigger gets a companion `[I]` version (e.g., "VWAP Cross Above `[C]`" and "VWAP Cross Above `[I]`"). Both appear in drill-down analysis with independent KPIs, so users can directly compare performance. The trigger CID encodes the execution mode — no separate config field needed.
- **After Phase 18 (Multi-Timeframe)** — Intra-bar evaluation shares the same streaming engine infrastructure as multi-timeframe confluence. Building MTF first ensures the `SymbolHub` architecture is solid before adding tick-level evaluation on top.
- **Backtest uses high/low approximation** — Without true tick data in backtest, checking if bar high ≥ level (for long) or bar low ≤ level (for short) determines if the level was breached. Entry price uses the level itself (equivalent to a limit order fill). This is the standard approach in TradingView strategy backtests.
- **Confluence evaluates at bar close only** — Intra-bar triggers fire mid-bar only if confluence was satisfied at the last bar close. This avoids re-running the full confluence pipeline per tick while still gating triggers on confluence state.

### Phase 20: General & Risk Management Pack Audit ✅
*Audit, validate, and expand the General Confluence and Risk Management pack libraries. Ensure condition logic, parameter schemas, preview rendering, and pipeline integration match production expectations. Identify structural improvements now that the indicator audit (17D) and intra-bar infrastructure (Phase 19) are complete.*

**General Packs Audit (20A):**
- [x] Audit all general pack templates — fixed calendar_filter (avoid_opex now implemented as 3rd-Friday proxy, buffer_minutes help text clarified), DRYed time_of_day / trading_session shared eval logic, fixed weekend BLOCKED_DAY handling, removed UNKNOWN fallback state, added validate_parameters()
- [x] Evaluate and add general pack triggers (20B) — window_open/close for time_of_day, session_open/close for trading_session, event_block_start/clear for calendar_filter, with detect_triggers() and preview markers
- [x] Preview Show Conditions overlay renders correctly; added empty-data warning after session filter
- [ ] New GP templates identified but deferred: market regime (VIX), volatility filter (ATR), news blackout

**Risk Management Packs Audit (20C):**
- [x] Audit all RM templates — fixed swing stop lookback bias (exclude current bar), added ATR NaN fallback logging, added parameter clamping to all builder functions, added validate_parameters()
- [ ] Intra-bar exit timing for RM — identified; requires position tracking in streaming engine (deferred)
- [x] Reviewed parameter ranges and defaults — all builders now clamp to schema min/max
- [x] RM packs integrate correctly with execution model; target_config=None documented as "stop/signal only"
- [x] Trailing stops and breakeven stops implemented (20D) — update_stop_price() in triggers.py, atr_trailing and breakeven_stop templates with default packs, strategy builder UI with trailing/breakeven checkboxes
- [x] Fixed trailing stop R-multiple inflation bug — R-multiple calculation now uses initial stop price (pre-trail) as risk denominator instead of final (trailed) stop price, which was artificially shrinking risk and inflating R-multiples

**Preview & UI Improvements (20E):**
- [x] GP preview: trigger arrow markers at state transitions, empty-data warning; RM preview: stop/target level markers, exit reason breakdown metrics
- [x] Risk management preview — stop/target markers on trades with initial vs final stop for trailing visualization

**Deferred for future phases:**
- Partial profit taking, pyramiding, dynamic position sizing, event calendar API, GP triggers as entry/exit triggers, new GP templates, RM intra-bar exits

### Phase 21: Alert & Execution Fidelity — "Alerts Match Backtest"
*Ensure streaming alerts (both bar-close and intra-bar) fire with the same timing, direction, and pricing that the backtest logic produces. Provide reliable, real-time validation tools so discrepancies between alert execution and backtest expectations are immediately visible and actionable. This is a critical quality gate before expanding to new strategies or live capital.*

**Why this phase exists:**
The alert pipeline has been built incrementally across Phases 5/5B, 13, 14B, and 19. Each phase added capability (webhooks, streaming engine, intra-bar triggers) but integration bugs have surfaced at each layer boundary — trigger name mismatches, missing indicator columns, duplicate signals, empty webhook fields. This phase consolidates all alert-related fixes and adds the validation infrastructure to confirm correctness with confidence.

**21A: Intra-Bar Alert Reliability — DONE**
- [x] Fix INTRABAR_LEVEL_MAP key mismatch — `_get_intrabar_trigger_bases` now returns full trigger names (e.g., `utbot_v2_buy_ib`) that map correctly after `removesuffix("_ib")`
- [x] Fix missing group-specific indicator columns — `_on_bar_close` trigger cache enrichment now runs `run_indicators_for_group()` for all enabled confluence groups after `_run_pipeline()`, producing columns like `utbot_stop_prev` and `utbot_stop`
- [x] Fix intra-bar/bar-close dedup ordering — `fired_ib` set is now read before clearing, so bar-close signals are properly suppressed when the same trigger already fired intra-bar
- [x] Position state tracking — `_position_state` dict tracks whether each strategy is in a position; entry signals suppressed when already in position, exit signals suppressed when flat
- [x] Position state initialization — `_init_position_state()` runs `generate_trades()` on warmup data at engine startup to determine current position state, preventing false entry on first tick
- [x] Trigger cache seeding — intra-bar trigger levels pre-cached from warmup data so crossing detection works from tick one
- [x] Stop price & quantity in intra-bar webhooks — `calculate_stop_price()` called for intra-bar entry signals so `{{quantity}}` placeholder resolves to a valid number (was empty string, causing SignalStack "invalid values quantity" errors)
- [x] Fix duplicate entry alerts — entry signals now suppressed when position state is already LONG/SHORT, preventing re-entry alerts on continued level crossing
- [x] Fix exit triggers not firing — exit signal evaluation was blocked by stale trigger cache; exits now properly evaluate on each bar close and level crossing
- [x] Fix empty quantity on exit webhooks — exit signal path now resolves `{{quantity}}` from position state instead of recalculating (was producing empty string for exits)
- [ ] Monitor for edge cases: engine restart mid-position, session boundary crossings, confluence state changes between bars

**21B: Real-Time Price Chart Refresh — DONE**
- [x] "Refresh" button added to every price chart via `render_candle_selector` (same row as candle count dropdown)
- [x] Verify chart refresh actually shows data up to the current minute — investigate if `prepare_data_with_indicators` cache, data loader staleness, or Alpaca API delay is causing stale charts even after refresh
- [x] Ensure trade history markers update on refresh to reflect the most recent entries/exits
- [x] Goal: after clicking Refresh, the price chart and trade markers should reflect data no more than ~1 minute old

**21C: Alert-vs-Backtest Validation — DONE**
- [x] Trigger Timing Analysis table — shows theoretical trigger time vs actual alert time for each matched execution, with Time Delta (seconds), theoretical vs alert price, and slippage in R-multiples
- [x] Timezone normalization — both Theo Time and Alert Time now display in user-configurable display timezone (was showing UTC vs local, causing visual 7-hour gap confusion); superseded by app-wide Display Timezone setting (see Phase 26)
- [x] Split time delta metrics — "Bar-Close Time Delta" computed only from bar-close actions (meaningful); intra-bar count shown separately since intra-bar timing delta vs bar boundary is not meaningful (the alert IS the trigger)
- [x] Summary metrics — 4-column layout: FT (All), FT (Alerts-Enabled), Alert Actual, Delta. Delta compares FT (Alerts-Enabled) vs Alert Actual for apples-to-apples execution fidelity measurement
- [x] FT (Alerts-Enabled) column — FT KPIs computed only from trades during the alert-enabled window (first alert onward), isolating execution quality from strategy quality
- [x] Dollar per-share slippage — Avg Entry Slip ($/sh) and Avg Exit Slip ($/sh) in summary metrics, plus dedicated "Trade-by-Trade: Dollar Per Share Slippage" expandable table showing Theo vs Alert prices and per-share deltas
- [x] R-multiple and dollar TBT tables — separated into two expandable sections to avoid column overload; R-based for risk-normalized view, $/share for intuitive per-share cost view
- [x] Enriched execution records — `bar_time` and `source` fields added to entry/exit execution records in `match_alerts_to_trades()` for timing analysis
- [x] Enriched phantom alert discrepancies — now include `alert_type`, `bar_time`, `source`, and `price` fields for better debugging
- [x] Alert analysis caching — heavy computation extracted into `_compute_alert_analysis()` with session state cache keyed by `data_refreshed_at` timestamp, eliminating recomputation on tab switches
- [x] Alert matching performance — pre-parsed timestamps split by entry/exit type before nested loops, sorted for efficient matching (eliminated O(n²) datetime parsing)
- [x] Discrepancy detection — missed alerts (FT trades without matching alerts) and phantom alerts (alerts without matching FT trades) surfaced with full context
- [x] Discrepancy management — Reset Alert Tracking, Dismiss Discrepancies, and Date Filter on Alert Analysis tab (see Phase 26)

**21D: Webhook Payload Reliability**
- [x] Fix empty `{{quantity}}` on exit webhooks — exit signal path now resolves quantity from position state
- [ ] Audit all webhook placeholder values for both bar-close and intra-bar signal paths — ensure `{{stop_price}}`, `{{order_price}}`, `{{market_position}}`, and `{{order_action}}` always resolve to valid values
- [ ] Template validation — warn user at webhook creation time if template contains placeholders that may produce invalid JSON
- [ ] Webhook delivery confirmation — log HTTP response status from SignalStack/other endpoints; surface delivery failures in the Alerts UI
- [ ] Retry logic for transient webhook failures (timeout, 5xx) with exponential backoff

**21E: Strategy Detail Rendering Performance — DONE**
- [x] Vectorized price chart timestamps — replaced per-row `pd.to_datetime().timestamp()` with single vectorized conversion on entire Series; reduced Price Chart tab from ~100s to ~7s for strategies with 3000+ trades
- [x] Merged chart marker loops — entry/exit markers and target price lines now built in a single pass over trades instead of three separate loops
- [x] Pre-parsed alert timestamps for chart matching — alert timestamps parsed once before trade matching loop with early-skip for out-of-window trades
- [x] Vectorized indicator overlay — replaced `iterrows()` with column mask + zip for indicator data point extraction
- [x] Total page render time reduced from 2+ minutes to ~13 seconds (14.6x improvement)
- [x] All changes purely in rendering/display code — trade generation pipeline and backtest logic completely untouched

**21F: Streaming Engine Reliability — DONE**
- [x] Fix premature `streaming_connected` flag — was set to `true` before `_run_forever()` authenticated; alpaca-py catches "connection limit exceeded" internally and retries forever without re-raising, so the flag never reset. Moved flag-set into the `on_trade` callback — only marks `streaming_connected=true` on first confirmed tick
- [x] Route alpaca-py WebSocket logger (`alpaca.data.live.websocket`) to `streaming_engine.log` — connection errors (e.g., "connection limit exceeded") were only visible in Streamlit stderr, invisible in the engine log
- [x] Strategy hot-reload — `_refresh_strategies()` auto-refreshes monitored strategies every 5 minutes from `alert_config.json` without disrupting the WebSocket connection; adds new strategies to existing SymbolHubs with warmup data and position state initialization; removes de-monitored strategies; logs warning when new symbols require a manual restart
- [x] Per-strategy position init — `_init_position_state_for(strat)` on SymbolHub initializes position state and trigger cache for a single hot-loaded strategy without re-running the full startup loop

**21G: Alert Display & Pricing Fidelity — DONE**
- [x] Alert timestamps show seconds — changed format from `'%m/%d %H:%M'` to `'%m/%d %H:%M:%S'` in strategy detail alerts, `_render_alert_row()`, and portfolio alerts tabs
- [x] Trigger reason displayed on alerts — `_TRIGGER_LABELS` dict maps trigger IDs (e.g., `stop_loss`, `bar_count_exit`, `opposite_signal`) to human-readable labels; shown alongside alert price in strategy detail and alert row rendering
- [x] Gap-aware stop-loss exit price — streaming engine `_check_managed_exits()` now computes `min(stop_price, open)` for LONG and `max(stop_price, open)` for SHORT stop exits, matching the backtest fill logic in `triggers.py`. Previously used `close_price`, causing vertical gap between chart entry (+) and exit (x) markers

**21H: Alert Analysis Matching Accuracy — DONE**
- [x] Dynamic match window — `match_alerts_to_trades()` window changed from fixed 300s to `2 * bar_period_seconds` (derived from strategy timeframe). Prevents cross-matching wrong trades on short timeframes while remaining generous on longer ones
- [x] Consistent timezone normalization in alert analysis — `_is_ft()` and FT alerts-enabled window computation now use `_to_utc()` consistently instead of `.replace(tzinfo=None)` (which stripped timezone without converting, causing comparison errors between UTC-aware and naive timestamps)

**Definition of Done:**
- Intra-bar and bar-close alerts fire with timing and pricing consistent with backtest logic
- Price charts refresh to within ~1 minute of current time on demand
- Webhook payloads always contain valid values for all placeholders
- No duplicate entry alerts for the same position
- A clear validation workflow exists to compare alert execution against backtest expectations
- Strategy detail pages load in under 15 seconds even for strategies with 3000+ trades
- Streaming engine reliably connects and self-reports connection status accurately
- New strategies are picked up by the running monitor within 5 minutes without restart
- Alert chart markers (+ and x) align vertically with backtest trade markers for stop-loss exits

### Phase 22: Web Deployment & Multi-User Infrastructure — COMPLETED 2026-03-12
*Deploy the application to production hosting so the alert monitor runs reliably in the cloud without requiring a local machine. Add user authentication and migrate from JSON file storage to a proper database for multi-user support. Full spec: `docs/Implementation_Spec_Phase_22.md`.*

**Why this phase exists:**
The application is currently a local Streamlit app with JSON file storage, no authentication, and the Ralph engine (ralph_engine.py) running as a local subprocess for live alert monitoring. To use it for actual trading, the engine must run reliably in the cloud, data must be safe in a database, and the app must be accessible from any device with proper auth. This is the critical infrastructure phase that transitions RoR Trader from a local development tool to a production-ready web application.

**Current engine architecture (post-Phase 30, updated Phase 32):**
- `unified_engine.py` — Single bar-by-bar engine for both backtest and live (replaces old batch pipeline). Asset-agnostic — works on any OHLCV DataFrame.
- `ralph_engine.py` — O(1) incremental streaming alert engine wrapping the unified engine's components (IncrementalIndicatorEngine, TriggerEvaluator, PositionStateMachine). Manages dual Alpaca WebSocket feeds (`StockDataStream` for equities, `CryptoDataStream` for crypto), bar building, cross-timeframe confluence via shadow engines, fidelity auditing, and alert dispatch. This is the process that becomes the Railway worker.
- `data_loader.py` — Dual-client data loading: `StockHistoricalDataClient`/`StockBarsRequest` for equities, `CryptoHistoricalDataClient`/`CryptoBarsRequest` for crypto. Auto-detects asset type via `is_crypto(symbol)` (`"/" in symbol`). Session filtering skipped for crypto (24/7 market).
- `app.py` — Streamlit UI with Asset Type selector in Strategy Builder (Equity/Crypto); session auto-set to "24/7" for crypto.

**Hosting Stack:**
- **Railway** — App hosting (web service + background worker). Usage-based pricing (~$8-13/month). Git push-to-deploy.
- **Supabase** — PostgreSQL database + authentication. Free tier for 5-10 users ($0-25/month). Row Level Security for multi-user data isolation.
- **Namecheap** — Custom domain DNS pointed to Railway. Railway auto-provisions SSL.

**22A: Supabase Setup & Database Schema** ✓
- [x] Create Supabase project, configure email/password auth provider
- [x] Design PostgreSQL schema — tables for strategies, portfolios, requirement_sets, alerts, alert_config, webhook_templates, monitor_status, user_settings, confluence_groups, general_packs, risk_management_packs
- [x] Design principle: proper columns for queryable fields (id, user_id, name, symbol), JSONB for nested blobs (KPIs, equity curves, rules, stored_trades)
- [x] Row Level Security policies on all tables: `auth.uid() = user_id`
- [x] Built-in data (TTP/FTMO requirement sets, default webhook templates) stored with `user_id IS NULL`

**22B: Data Access Layer** ✓
- [x] New `src/db.py` module wrapping all Supabase/PostgreSQL operations — `get_client()`, `get_admin_client()`, user context management
- [x] `USE_DB` environment flag for incremental development (toggle JSON ↔ database)
- [x] Transformation layer (`_row_to_strategy()` / `_strategy_to_row()`) so rest of app sees same dict shapes — minimizes downstream code changes
- [x] Rewire strategy CRUD in app.py: `load_strategies()`, `save_strategy()`, `get_strategy_by_id()`, `update_strategy()`, `delete_strategy()`, `duplicate_strategy()`
- [x] Rewire portfolio + requirements CRUD in portfolios.py
- [x] Rewire alert system CRUD in alerts.py: alerts, alert_config, monitor_status, webhook templates
- [x] Rewire config/pack storage: confluence_groups.py, settings in app.py
- [x] Rewire ralph_engine.py data access: config loading, alert dispatch, state persistence (replaces JSON file IPC: engine_status.json, engine_state.json, engine_audit.jsonl, engine_reload.flag)

**22C: Authentication** ✓
- [x] New `src/auth.py` module with Supabase Auth integration (sign up, sign in, sign out, session refresh)
- [x] Login/signup page in Streamlit — email + password (OAuth deferred)
- [x] Auth gate at top of app.py: unauthenticated users see login page, authenticated users see the app
- [x] JWT stored in `st.session_state`, auto-refresh on expiry, establishes RLS context for all database queries
- [x] New user onboarding: seed default confluence groups, packs, settings on first login
- [x] Pre-registration: simple email collection form (waitlist) on public landing page

**22D: Worker Service (Ralph Engine Integration)** ✓
- [x] New `src/worker.py` — standalone entry point that manages per-user `RalphEngine` instances
- [x] Each user's RalphEngine runs in its own asyncio event loop (Ralph already handles WebSocket feeds, bar building, indicator computation, trigger evaluation, position tracking, and alert dispatch)
- [x] Uses admin Supabase client (service role key) for cross-user monitoring
- [x] Monitor control via database: UI writes `desired_state: 'running'|'stopped'` to `monitor_status` table; worker polls and starts/stops user RalphEngine instances accordingly — no PID management
- [x] Ralph's hot-reload (5-min refresh, engine_reload.flag) rewired to read config from database instead of JSON files
- [x] Ralph's alert dispatch (`_on_alert`) rewired to write alerts to database instead of alerts.json
- [x] Ralph's IPC files (engine_status.json, engine_state.json, engine_audit.jsonl) replaced by database writes
- [x] Update app.py monitor controls: Enable/Disable Monitoring toggle writes `desired_state` to DB (replaces subprocess/PID/zombie-detection code)
- [x] Alpaca API keys are system-level env vars on Railway (not per-user) — Alpaca provides market data infrastructure, not a per-user service
- [x] `src/migrations/copy_user_config.py` — admin script to copy confluence groups, packs, strategies, portfolios between user accounts
- [x] Live chart in DB/cloud mode loads from Alpaca REST API (pickle files are local-only); `@st.fragment(run_every=5)` auto-refreshes with thread-local JWT re-establishment

**22E: Containerization & Deployment** ✓
- [x] `Dockerfile` for web service (Streamlit app + vendored LWC fork)
- [x] `Dockerfile.worker` for background worker service
- [x] `.streamlit/config.toml` for production configuration
- [x] Railway project setup: web service + worker service, shared environment variables
- [x] Environment variables: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- [x] Health checks: Streamlit `/_stcore/health` endpoint, worker `/tmp/worker_alive` touch file

**22F: Data Migration & DNS**
- [x] Migration script (`src/migrate_json_to_db.py`): reads all JSON files, transforms, inserts into Supabase
- [x] Execution order respects foreign keys (requirement_sets → strategies → portfolios → alerts)
- [x] Verification: row count comparison, spot-check specific records
- [x] `.gitignore` update: exclude runtime JSON files, .env, logs
- [ ] Namecheap DNS CNAME → Railway custom domain
- [ ] Railway auto-provisions SSL via Let's Encrypt

**Git Workflow & Environments:**
- `main` — Production (auto-deploys to Railway production environment: web + worker services)
- `dev` — Active development (auto-deploys to Railway dev environment: web + worker services)
- `main-backup-pre-22d` — Snapshot of main before Phase 22D worker changes (safety net)
- Feature branches off dev for each sub-phase
- **Railway environments:** Production (main branch, paper Alpaca keys) and Dev (dev branch, live Alpaca keys). Separate Alpaca API keys per environment avoids WebSocket connection limit conflicts. Both environments share the same Supabase database.
- **Caution:** Since dev and prod share Supabase, only enable monitoring in one environment at a time — both workers would try to start an engine for the same user via the shared `monitor_status` table.

**Design Decisions (Phase 22):**
- **Railway chosen over Render** (re-evaluated 2026-03-11): Render offers a clean multi-project dashboard and fixed-price tiers ($7/service), but Railway's usage-based billing saves money for a trading app — the worker only runs during market hours (~6.5h/day), not 24/7. Render's free tier spins down after 15 min of inactivity, which is incompatible with a persistent alert monitor. Both platforms handle WebSocket, Docker, custom domains, and auto-SSL equally well. Railway total: ~$8-13/month vs Render: ~$14/month (2× $7 for web + worker).
- **Supabase chosen** for combined database + auth in one service with excellent Python client and Row Level Security. Used regardless of hosting platform — Render's built-in PostgreSQL (90-day free trial) lacks auth and RLS.
- **Firebase rejected**: Cloud Run's scale-to-zero fights Streamlit's stateful WebSocket model; Firestore NoSQL is a worse fit for relational data; higher cost
- Database migration done upfront (not deferred) to avoid JSON file locking issues with multiple users and to enable RLS-based data isolation
- Design/aesthetics deferred to a separate phase — focus this phase on infrastructure, auth, and reliability
- Config tables (settings, confluence_groups, packs) use single JSONB column per user since they're always loaded/saved as whole documents — preserves current load/save pattern
- **Worker wraps RalphEngine** (not the legacy alert_monitor.py). Ralph already owns the full live pipeline: Alpaca WebSocket → BarBuilder → IncrementalIndicatorEngine → TriggerEvaluator → PositionStateMachine → AlertDispatcher. The worker just needs to manage per-user Ralph instances and rewire Ralph's I/O from JSON files to Supabase.
- Worker service always runs on Railway; monitor start/stop controlled via database flag (not PID/signal management) — simpler, more reliable, works across services
- `USE_DB` toggle flag enables incremental development — each CRUD module can be switched independently
- **Multi-account support**: Users will maintain separate accounts (e.g., production strategies vs. dev/testing), so RLS-based data isolation is critical from day one
- **Alpaca keys are system-level** (decided 2026-03-12): Alpaca provides market data infrastructure (SIP/IEX feeds), not a per-user service. Keys are env vars on Railway shared by all users. Removes unnecessary per-user key input from Settings UI.
- **Single WebSocket connection**: Alpaca allows only 1 WebSocket per API key. Worker owns the connection; app uses REST API for chart data. Rolling deploys can cause "connection limit exceeded" during the handoff — connection slots take several minutes to clear on Alpaca's side.
- **Live chart in cloud mode**: Worker writes data to DB (status, alerts, positions). App's live chart loads recent bars via Alpaca REST (completed bars only, ~60s behind). Near-real-time chart requires future work: worker writes forming bar to a DB table every few seconds (planned).
- **Streamlit fragment JWT fix**: `@st.fragment(run_every=5)` auto-reruns don't re-execute `require_auth()`, losing thread-local JWT. Fragment functions must re-establish user context from `st.session_state`. Config loaders (general_packs, confluence_groups, risk_management_packs) defensively return in-memory defaults if save fails due to missing JWT.

**Definition of Done:**
- App accessible via custom domain with HTTPS and user authentication
- All data persisted in Supabase PostgreSQL (no JSON file dependencies)
- Alert monitor runs reliably as a Railway background worker
- Multiple users can log in with isolated data (strategies, portfolios, alerts)
- Monitor start/stop controllable from the web UI
- Existing strategies, portfolios, and alerts successfully migrated from JSON
- Git push to main triggers automatic deployment

**22G: Admin Page & Role-Based Access**
*Dedicated Admin page in the sidebar for infrastructure controls. Separates admin-level operations from user-facing features. Only visible to admin users — prevents testers from accidentally disabling shared services.*

- [ ] `is_admin` flag on users (Supabase user metadata or `user_roles` table)
- [ ] New "Admin" page in sidebar navigation (only rendered for admin users)
- [ ] Move Enable/Disable Monitoring toggle from Alerts & Signals page to Admin page
- [ ] Move E2E test section from Alerts & Signals page to Admin page (or remove)
- [ ] Admin page contents: monitor controls, worker status/logs, system-level settings, data feed status
- [ ] Alerts & Signals page cleaned up: only Manage Alerts, Webhooks, and strategy alert tabs remain
- [ ] Regular users see read-only monitoring status indicator (e.g. sidebar badge or status bar) without controls

### Phase 23: Scanner Strategy Method — Stock Scanner & Dynamic Universe Trading
*Third strategy method type: scanner-based strategies that trade a dynamic universe of stocks matching configurable filter criteria. Targets active day trading / scalping use cases (Warrior Trading pre-market news plays, SMB Capital stocks-in-play scalping, gap-and-go setups).*

**Dependency:** Phase 31 (Polygon.io migration) — scanner requires Polygon's all-tickers snapshot endpoint, reference data API, and news API.

**Detailed reference:** `docs/Scanner_Reference.md` — complete filter metrics tree, UI specification, Scanz reference screenshots, backtesting architecture, and system integration vision.

**Scanner UI (three collapsible sections):**
- [ ] **Filter Rules** — rule builder with metric tree (8 categories: Price, Change, Liquidity, Technical, Capital Structure, Short Interest, Financials, News), operators (>, <, =, >=, <=), static values or metric-vs-metric comparisons with temporal offsets (N days/candles ago, average of N), session scoping (Pre-Market, Regular Hours, After Hours, Full Day), per-rule toggle on/off, drag reorder, "Add as column to results" checkbox
- [ ] **Results Table** — dynamic watchlist of qualifying tickers with real-time updates, sortable columns, Market/Security Type/Watchlist filters, customizable columns
- [ ] **Activity Log** — timestamped ENTERED/EXITED events tracking when tickers join or leave the qualifying set

**Scanner CRUD:**
- [ ] Save/load scanner presets with custom names (JSON-based, like strategies/portfolios)
- [ ] Pre-built "Stocks in Play" scanner templates (pre-market movers, after-hours gainers, etc.)
- [ ] Ticker detail pane — click a ticker to see price chart + news feed

**Strategy Builder integration:**
- [ ] Add "Scanner-Based" option to Strategy Method selector (alongside Ticker-Based and Webhook-Based)
- [ ] Scanner selection dropdown (select from saved scanners, similar to confluence pack selection)
- [ ] Entry/exit triggers, confluence, stops configured identically to ticker-based strategies
- [ ] Scanner acts as additional filter layer: trade fires only when ticker qualifies AND entry trigger fires AND confluence met

**Backtest engine (time-varying universe):**
- [ ] Two-loop architecture: outer loop evaluates scanner filters per bar across all tickers, inner loop runs strategy triggers on qualifying tickers
- [ ] Staged filtering for performance: cheap filters first (static reference data), then snapshot-level (price/volume/change), then expensive (technical indicators) only on survivors
- [ ] Position persistence: once in a position, exit governed by strategy triggers (stop/target/signal), NOT by ticker falling out of scanner
- [ ] Portfolio-level position limits enforced across all scanner-sourced trades

**Backtest results (multi-ticker):**
- [ ] Aggregate KPIs across all trades on all qualifying tickers (win rate, avg R, profit factor, etc.)
- [ ] Ticker column in trade table identifying which stock each trade was on
- [ ] Scanner Activity timeline showing historical qualification events
- [ ] Composite equity curve aggregating P&L across all ticker trades
- [ ] Per-ticker drill-down reusing existing chart infrastructure
- [ ] Scanner hit rate metric (qualifying events that produced actual entries)

**Live trading integration:**
- [ ] Scanner evaluation engine polls Polygon snapshot endpoint on configurable interval
- [ ] Qualifying tickers get Ralph Engine instances (WebSocket subscription + unified engine triggers)
- [ ] Alert dispatch via existing webhook pipeline (Discord/Slack) — no downstream changes needed
- [ ] Tickers unsubscribed when disqualifying (if FLAT; held positions continue until strategy exit)

**News evaluation pipeline (future enhancement):**
- [ ] Polygon News API integration for ticker-level news feed
- [ ] LLM-based news classification: novelty detection (is this actually new?), category (earnings/FDA/partnership/offering), sentiment, magnitude
- [ ] News as scanner filter or confluence condition (e.g., "news count > 0 AND sentiment = positive")
- [ ] Historical news data enables backtesting news-driven strategies

**Data sources (Polygon):**
- [ ] Snapshot endpoint (`/v2/snapshot/locale/us/markets/stocks/tickers`) — all-market scanner sweeps
- [ ] Reference data (`/v3/reference/tickers`) — float, market cap, shares outstanding, sector
- [ ] Historical aggregates — for RVOL, ATR, moving average lookbacks
- [ ] News API (`/v2/reference/news`) — timestamped news articles with ticker tags
- [ ] Short interest — NOT available from Polygon; evaluate FINRA/Ortex as supplementary source

### Phase 24: Sub-Minute Historical Data & HFT Backtesting
*Build our own sub-minute historical data by recording tick streams for priority tickers, and/or integrate alternative data providers (Databento, Polygon.io, etc.) to enable backtesting on sub-minute timeframes (5s, 10s, 15s, 30s).*

**Problem Statement:**
- Alpaca's REST API does not provide sub-minute historical bars — the finest granularity available is 1-minute
- Sub-minute timeframes (5s, 10s, 15s, 30s) currently work only via the streaming engine (live data), making them impossible to backtest
- HFT-style strategies and scalping setups often rely on sub-minute charts that need backtesting to validate edge before going live
- Without sub-minute historical data, traders cannot evaluate or optimize sub-minute confluence conditions

**Self-Recorded Tick Data:**
- [ ] Priority ticker list — user-configurable list of high-priority symbols to continuously record
- [ ] Tick stream recorder — background service that connects to Alpaca's streaming API and persists raw trades/quotes for priority tickers
- [ ] Local storage format — efficient on-disk format (Parquet, HDF5, or SQLite) for raw tick data, partitioned by date and symbol
- [ ] Sub-minute bar aggregation — utility to build 5s/10s/15s/30s bars from recorded tick data for backtesting
- [ ] Retention policy — configurable data retention window (e.g., 30/60/90 days) to manage disk usage
- [ ] Recording status UI — show which tickers are being recorded, data freshness, disk usage, and recording health

**Alternative Data Providers:**
- [ ] Research and evaluate sub-minute data providers: Databento, Polygon.io, FirstRate Data, Kibot, TickData
- [ ] Provider comparison — cost, granularity, coverage, API quality, data quality, latency
- [ ] Integration module — pluggable data source for sub-minute historical bars alongside existing Alpaca integration
- [ ] Provider connection settings — API keys, subscription tier, rate limits managed in Settings > Connections

**Backtest Integration:**
- [ ] `load_from_alpaca()` and `load_market_data()` extended to detect sub-minute timeframes and route to self-recorded data or alternative provider
- [ ] Backtest pipeline works identically for sub-minute TFs — same indicator/interpreter/trigger evaluation, just on finer-grained bars
- [ ] Sub-minute backtest performance — ensure pipeline handles the higher bar counts efficiently (a 30-day 5-second backtest = ~140k bars per session)

**Design Considerations:**
- Self-recording creates a "cold start" problem — data only exists from when recording began. Alternative providers solve this but add cost.
- Hybrid approach likely best: use a provider for historical backfill, self-record going forward for zero ongoing cost.
- Sub-minute bars have significantly more noise — strategies built on these timeframes need robust confluence filtering.

### Phase 25: Execution Realism — Spread, Slippage & Order Types
*Improving backtest accuracy and live execution to reflect real-world trading costs. Needs discussion before committing to a direction.*

**Gap-Aware Stop/Target Fills (DONE):**
- [x] When a bar opens past the stop level (overnight gap, flash crash), fill at the open instead of the stop — `min(stop, open)` for LONG, `max(stop, open)` for SHORT
- [x] Same for targets: gap through target fills at the open (windfall) — `max(target, open)` for LONG, `min(target, open)` for SHORT
- [x] No config changes — always active, universally more realistic

**Bid-Ask Spread Modeling (TO DISCUSS):**
- [ ] Configurable spread cost deducted from each trade's P&L (fixed amount or percentage)
- [ ] Potentially different defaults for regular hours vs extended hours (SPY: ~$0.01 RTH, ~$0.02–$0.05 extended)
- [ ] On tight-stop strategies, spread can eat a significant % of risk — users need visibility into this
- [ ] Open question: per-symbol defaults? Per-session? Or just a single global setting?

**Order Type Selection (TO DISCUSS):**
- [ ] Per-strategy setting: Market vs Limit for live alert execution
- [ ] **Market order**: fire alert → immediate execution at market price (current implicit behavior)
- [ ] **Limit order**: fire alert → place limit at target entry level → auto-cancel after configurable timeout if unfilled
- [ ] Primarily affects the streaming engine / alert system, not the backtester
- [ ] Trade-off: limit orders avoid overpaying but risk missing the trade entirely; market orders guarantee entry but with potential slippage

**General Slippage Estimation (TO DISCUSS):**
- [ ] Small configurable slippage amount added to entries and deducted from exits
- [ ] Models market impact beyond bid-ask spread (order queue, volatility, size)
- [ ] Could be a per-trade fixed cost or a percentage of trade value

> **Note:** The gap-fill fix is implemented. The remaining items need discussion to decide overall direction — whether to tackle spread/slippage as a single unified "execution cost" model or as separate independent features, and how order types interact with the existing alert pipeline.

### Phase 27: Portfolio Risk Intelligence & Balance-Aware Execution
*Enhanced portfolio analytics, balance-aware quantity sizing, and compliance-driven position management. Full spec: `docs/Implementation_Spec_Phase_26.md` (authored as Phase 26; renumbered).*

**27A: Enhanced Risk Analytics**
- [ ] Daily Drawdown Chart — bar chart of daily P&L % with `max_daily_loss_pct` and `daily_pause_pct` threshold overlay lines from requirement set; highlight breach days
- [ ] New rule type `daily_pause_pct` — soft limit (pause, resume next day) distinct from `max_daily_loss_pct` (hard limit); add to TTP template and evaluation engine
- [ ] Historical Worst-Case Analysis — worst single day, worst losing streak, top 5 worst days table with breach status per rule
- [ ] Monte Carlo Simulation — shuffle trade order (daily blocks/weekly blocks/individual trades), 500–5,000 runs; output bust probability, daily pause probability, max DD distribution histogram, equity curve confidence bands (5th/25th/50th/75th/95th percentiles)
- [ ] Profit Target Progress — progress bar on Prop Firm Check tab showing current P&L % vs `min_profit_pct` target with estimated days remaining

**27B: Balance-Aware Quantity Sizing**
- [ ] `get_available_balance()` — compute buying power from ledger (deposits - withdrawals + trading P&L - estimated open position capital)
- [ ] Optional `auto_adjust_sizing` toggle per portfolio — when enabled, cap webhook quantity at `min(risk_quantity, buying_power / entry_price)`
- [ ] Insufficient balance handling — quantity >= 1: fire with reduced quantity + `adjusted_quantity` flag; quantity < 1: skip webhook + `skipped_reason: "insufficient_balance"`
- [ ] Warning banners on Performance/Deploy tabs when risk-per-trade exceeds available balance
- [ ] Concurrent entry tie-break: first-come-first-served, profit factor as tie-breaker

**27C: Portfolio-Specific Compliance Actions**
- [ ] Portfolio-scoped webhook suppression — `compliance_paused` flag per portfolio; suppresses entry webhooks only (exits always fire); other portfolios with same strategies unaffected
- [ ] Real-time compliance check on each exit — evaluate `max_daily_loss_pct`, `daily_pause_pct`, `max_total_drawdown_pct` from ledger P&L after each exit delivery
- [ ] Auto-close on breach — fire per-strategy close alerts for all open positions in the breached portfolio, plus portfolio-level compliance breach alert
- [ ] Auto-resume for daily pause (next market open); manual resume required for max DD breach
- [ ] Compliance pause UI — banner on Deploy tab, "Resume Trading" button, "Paused" badge on portfolio cards, breach annotation on equity curve

**27D: Future — Advanced Prop Firm Rules (deferred)**
- [ ] Min trade duration (`min_trade_duration_sec`) — post-trade validation; TTP requires 10 seconds
- [ ] Min profit per share (`min_profit_per_share`) — TTP requires $0.10/share for profit to count; trades below don't count toward profit target
- [ ] Position sizing limits (`max_position_shares`, `max_position_dollars`, `max_position_pct`)
- [ ] Alpaca API integration — real-time balance, positions, order history; reconcile with ledger
- [ ] Multi-account portfolio architecture — portfolio maps to specific trading account with its own balance/rules/endpoints

**Design Decisions (Phase 27):**
- Account balance from existing ledger system, not broker API — users may be on different platforms; API integration deferred to 26D
- Compliance evaluation uses ledger P&L (webhook exits), not backtest KPIs — closer to actual account state including real slippage
- Monte Carlo default: shuffle daily blocks to preserve intraday correlation (time-of-day matters); user can switch to weekly blocks or individual trades
- On compliance breach: close ALL open positions in portfolio regardless of which strategy caused the loss — prop firms treat the account as one unit of risk
- Concurrent entry buying power: first-come-first-served to avoid adding latency; profit factor tie-break for truly simultaneous triggers

### Phase 26: Low-Priority Cleanup & Enhancements
*Deferred items and nice-to-haves — polish, performance, and convenience improvements.*

**Expanded Backtest Range:**
- [ ] Date range picker on My Strategies page or strategy detail — select a custom backtest start date earlier than the original
- [ ] Run full pipeline for the expanded window and merge new backtest trades into `stored_trades` (additive, does not affect existing forward test data)
- [ ] Distinct from Data View filter (Phase 10C) — filter shows subsets of existing data instantly; expanded backtest generates new data by running the pipeline

**Optimization Workflow Polish (deferred from Phase 9/10):**
- [ ] Trigger parameters visible and expandable in Optimizable Variables — show EMA periods, ATR multiplier, etc. (not just trigger name)
- [ ] Stop/target variation tags on trades — tag individual trades with pack ID when running multi-backtest comparisons
- [x] ~~Multi-backtest progress indicator + caching~~ — implemented in Phase 30K (bar cache fast path + progress bars on all 5 analyzer call sites)
- [ ] Lazy tab loading — only compute drill-down results when a tab is first opened, not all 6 on page load

**Data Refresh (Hard vs Soft):**
- [x] Hard refresh button on strategy detail page — calls `refresh_strategy_data()`, clears all session caches, recomputes KPIs
- [x] Hard refresh button on portfolio detail page — clears portfolio analytics cache and underlying strategy caches
- [ ] Two refresh modes: **Soft** (current behavior — appends new bars) and **Hard** (full recompile — re-fetches all historical bars from scratch). Current refresh button uses incremental (soft) mode.
- [ ] UI: dropdown or toggle for soft vs hard refresh mode

**Strategy Copy — Default Preservation:**
- [ ] Copying a strategy should carry over ALL config fields including `bar_count_exit` and other defaults
- [ ] Investigate why Copy drops bar_count_exit — likely the copy flow doesn't include fields that come from default settings vs explicit user selections
- [ ] Audit the copy/save flow to ensure default entry trigger, default exit trigger, and default stop/target all persist correctly

**Strategy Builder Default Behavior:**
- [ ] Review how default triggers and default exits populate on new/copied strategies — current behavior is occasionally inconsistent
- [ ] Ensure default_entry_trigger, default_exit_trigger, and default_stop_config from settings always apply cleanly when no saved value exists

**Alert Pipeline Fix (Prep to Go Live):**
- [x] Auto-sync `alert_config.json` when "Track Alerts" toggle is changed — enables/disables `alerts_enabled` for the strategy in the alert config, so the alert monitor picks up real strategies
- [x] One-time backfill on app startup for strategies with `alert_tracking_enabled` but missing from `alert_config.json`
- [x] Alert Analysis tab now shows when discrepancies exist (not just live_executions) — displays missed/phantom alert tables with explanation
- [x] Alerts tab shows discrepancy context when no alerts exist but discrepancies are detected

**Strategy Builder Improvements:**
- [x] Ticker input changed from hardcoded 8-symbol dropdown to free-text input — supports any Alpaca-tradeable stock ticker
- [x] Strategy list ticker filter dynamically populated from tickers in use (not hardcoded)
- [x] Entry price and exit price columns added to all trade list tables (Strategy Builder, Trade History, Split Trade History)
- [x] `_extract_minimal_trades()` now persists entry_price and exit_price in stored trades

**Execution Mode Tags — Universal [C]/[I] Display (DONE):**
- [x] `get_trigger_display_name()` and `format_exit_triggers_display()` updated with optional `trigger_defs` parameter — when provided, entry/exit names are prefixed with `[C]` (bar close) or `[I]` (intra-bar) execution tags
- [x] Strategy cards — entry/exit triggers on the card grid now show [C]/[I] tags
- [x] Strategy detail page header — meta row Entry/Exit fields now show tags
- [x] Config tabs (live backtest + forward test) — Entry/Exit display shows tags
- [x] Saved KPIs / legacy strategy view — exit triggers display shows tags
- [x] `bar_count_exit` display fix — strategies using only a candle count exit (e.g., 4-bar exit) now correctly show the exit name instead of "Unknown"; bar count exits always tagged `[C]`
- [x] Strategies with both signal exits and bar_count_exit display both (e.g., "`[I]` UTBot Below, `[C]` 4-bar exit")
- [x] Data confirmation caption after load: shows bar count, symbol, timeframe, session, and data source for verification

**Display Timezone — User-Configurable Timestamps (DONE):**
- [x] New `display_timezone` setting (default US/Eastern) with dropdown in Settings > Display section — options: US/Eastern, US/Central, US/Mountain, US/Pacific, UTC
- [x] `format_display_ts()` central helper — converts any timestamp (str, datetime, pd.Timestamp) to display timezone; naive timestamps assumed UTC; graceful fallback on parse errors
- [x] `_to_chart_unix()` helper — converts UTC timestamps to display-timezone Unix seconds for lightweight-charts (which has no native TZ support); handles both Series (vectorized) and scalar inputs
- [x] All text timestamp displays (~30 locations) route through `format_display_ts()` — alert badges, strategy metadata, trade tables, timing analysis, discrepancy tables, engine status
- [x] All price chart timestamps (~10 conversion points) route through `_to_chart_unix()` — candlestick data, trade entry/exit markers, alert-price markers, trigger overlays, condition markers, secondary panes (oscillator, MACD, RVOL)
- [x] All `datetime.now()` calls standardized to `datetime.now(timezone.utc)` across app.py, alerts.py, and realtime_engine.py — ensures stored timestamps are consistently UTC
- [x] Zero changes to trade generation, signal detection, or any calculation logic — timezone conversion is purely a display-layer concern

**Alert Data Management (DONE):**
- [x] "Reset Alert Tracking" button on strategy detail action bar — clears `live_executions`, `discrepancies`, and all alerts from `alerts.json` for that strategy; keeps `alert_tracking_enabled` on; confirmation dialog before destructive action
- [x] `delete_alerts_for_strategy()` function in alerts.py — removes all alerts for a given strategy_id
- [x] "Dismiss Discrepancies" button — stores `discrepancies_dismissed_at` timestamp on strategy; badge only counts discrepancies with `detected_at > dismissed_at`; button available in both discrepancies-only view and full analysis expander
- [x] `detected_at` preservation on refresh — `match_alerts_to_trades()` always sets `detected_at = now`, so `refresh_strategy_data()` now merges `detected_at` from previously-known discrepancies (keyed by missed/trade_index or phantom/alert_id) to prevent dismissed discrepancies from reappearing as "new"
- [x] Date filter on Alert Analysis tab — selectbox with "All Time", "Last 7 Days", "Last 14 Days", "Last 30 Days"; filters `live_executions` (by `alert_timestamp`) and `discrepancies` (by `detected_at`) before analysis; cache key includes filter selection; scoped to Analysis tab only
- [x] Dismissed state automatically cleared when alert tracking toggled off (both card and detail page) or when Reset is used

**UX Polish:**
- [ ] Utility buttons on Portfolios page — "Portfolio Requirements" and "Webhook Templates" links next to "New Portfolio" button

### Phase 27: Unified Pipeline Alert System + Live Chart
*Replaced dual alert paths (bar-close pipeline + TriggerLevelCache intra-bar crossing) with a single throttled pipeline evaluator. Alerts now fire from the same code that produces the chart — trigger-agnostic, eliminates phantom alerts. Full spec: `docs/Implementation_Spec_Phase_27.md`.*

**Problem:** The streaming engine had two parallel alert paths that diverged — bar-close ran `_run_pipeline()` (transition-based, matched chart), while intra-bar used `TriggerLevelCache` (price-vs-level crossing, fundamentally different computation). Result: 5:1 alert-to-trade ratio, phantom alerts, alerts firing before entry candles.

**Unified Pipeline Evaluator (DONE):**
- [x] `TriggerStateTracker` class — tracks `trig_*` boolean column values per strategy across evaluations; fires on `False→True` transition; trigger-agnostic (works for UT Bot, EMA, VWAP, MACD, any future pack)
- [x] `BarBuilder.get_df_with_partial()` — returns history + current partial bar as a single DataFrame for sub-bar-close evaluation
- [x] `_evaluate_pipeline_throttled()` on SymbolHub — runs full indicator→interpreter→trigger pipeline every 500ms; replaces both `_check_intrabar_triggers()` and bar-close `detect_signals()` signal processing
- [x] Pipeline runs once per timeframe, shared across all strategies on that timeframe (e.g., 5 SPY strategies = 1 pipeline run)
- [x] `_on_bar_close()` simplified to housekeeping — sets `_first_bar_closed` flag + evaluates managed exits (stop loss, bar count) only
- [x] TriggerLevelCache usage removed from SymbolHub (class + `INTRABAR_LEVEL_MAP` preserved for `triggers.py` backtest fill pricing)
- [x] Warmup seeding — `TriggerStateTracker.seed()` prevents false fires on first evaluation; warmup enriched data written to pickle for immediate Live Chart availability

**Live Chart Tab (DONE):**
- [x] `render_live_chart_tab()` — `@st.fragment(run_every=2)` auto-refreshing chart using existing TradingView LC `render_price_chart()` machinery
- [x] Reads `live_data_{symbol}_{tf}.pkl` written by streaming engine via atomic `os.replace()`
- [x] Indicator filtering — shows only strategy-relevant overlay indicators and oscillator panes (via `_get_strategy_relevant_groups(strat)`), not all enabled groups
- [x] Trade markers — receives pre-computed trades from parent backtest/forward-test pipeline; `render_price_chart()` filters to visible window and renders entry/exit arrows, +markers, R-multiple labels
- [x] Trade table below chart — `render_backtest_trade_table(trades)` matches price chart tab layout
- [x] Conditionally appears in both backtest and forward test views when streaming engine is active or pickle data exists
- [x] Status bar: last bar timestamp, close price, bar count, data freshness indicator

**Price Chart Indicator Filtering (DONE):**
- [x] Backtest and forward test Price Chart tabs now use `_get_strategy_relevant_groups(strat)` instead of `get_enabled_groups()` — only shows indicators used in entry, exit, or confluence conditions
- [x] `_get_strategy_relevant_groups()` updated to check `exit_trigger_confluence_ids` (plural list) and `general_confluences` in addition to singular exit ID and TF confluence

**Phase 27B: `generate_trades()` as Single Source of Truth (COMPLETE):**
- [ ] `TradeListTracker` class — replaces `TriggerStateTracker` with trade-list diffing; runs `generate_trades()` on each eval, diffs against previous result, fires entry/exit alerts on new events
- [ ] Single code path — chart entries/exits AND alert detection derive from the SAME `generate_trades()` call; eliminates parallel logic, drift, and phantom alerts from partial bar reversals
- [ ] Confluence fix — passes both `confluence` AND `general_confluences` (Time of Day, Day of Week) to `generate_trades()`; the TriggerStateTracker approach only checked `strat.get('confluence')`
- [ ] Live chart uses `generate_trades()` directly on live pickle data (replaces passing parent backtest trades)
- [ ] Managed exits dedup — `_fired_exits` set prevents duplicate exit alerts when `_check_managed_exits()` fires before `generate_trades()` catches up at bar close

**Data Quality Standard:**
- The Price Chart tab closely matches TradingView for the same symbol/timeframe — this is the quality benchmark
- The Live Chart tab should converge toward the same standard; known divergences (warmup depth, tick-vs-API bar aggregation) are documented below and are non-blocking for alert accuracy

**Known Divergence — Live Chart vs Price Chart (cosmetic, non-blocking):**
- Candle bodies may differ slightly: live chart builds candles from real-time tick aggregation (BarBuilder, 500ms updates including partial bars), price chart loads completed pre-aggregated bars from Alpaca API
- Indicator lines (especially longer EMAs) may diverge: streaming engine warms up with ~500 bars (~1 trading day), price chart uses ~11,700 bars (30 days) — different initialization windows cause EMA convergence differences
- **Trigger logic is identical** — both paths use the same `generate_trades()` pipeline with the same strategy config; entry/exit fidelity is preserved
- **Future fix (low priority):** increase streaming engine warmup window to match backtest data_days; would align indicators at the cost of more memory and slower startup

**Performance:** ~50-65ms per evaluation at 500ms cadence (10-13% CPU) — includes `generate_trades()` per strategy (~28ms) on top of pipeline cost (~25ms). For 5 strategies on same symbol: ~165ms (33% utilization).

**Phase 27C: Alpaca REST Bar Backfill (IMPLEMENTED — VERIFYING):**

*Motivation:* Tick-aggregated streaming bars differ from Alpaca REST bars in low-liquidity periods (after-hours OHLC diffs up to $0.23 for SPY). Zero-volume gap fills and OHLCV tick-aggregation differences cascade through cumulative indicators like UTBOT trailing stop ($0.65 divergence for SPY). Data integrity between all four pipelines (backtest, forward test, live chart, alert detection) is paramount.

*Scope expanded:* Originally planned as gap-fill replacement only. Now replaces ALL streaming bars (not just zero-volume) with canonical Alpaca REST data, matching the exact data source used by the price chart and backtest. Only the most recent 2 bars remain from streaming (too new for REST finalization).

- [x] `_backfill_from_rest()` method on SymbolHub — every 60s, fetches Alpaca REST bars from streaming start to now minus 2-bar grace period, replaces OHLCV in builder.history for matching timestamps
- [x] Non-blocking — runs on existing ThreadPoolExecutor, never blocks `process_tick()` or pipeline eval
- [x] Timeframe-aware grace period — 2 bars (not minutes), scaling with TF: 1Min=2min, 5Min=10min, 1Hour=2hr
- [x] Sub-minute TF support — gracefully skips timeframes not available in REST API (e.g., 10-second bars); streaming data remains source of truth for sub-minute
- [x] Alerts untouched — backfill only corrects underlying candle OHLCV; fired alerts and their chart markers (X symbols) are stored separately in alerts.json and are never overwritten
- [x] Graceful fallback — if API call fails, streaming data remains; retry on next cycle
- [x] No explicit dirty flag — next `_evaluate_pipeline_throttled()` cycle (within 500ms) automatically recalculates indicators with corrected data
- [x] Executor wired from `UnifiedStreamingEngine.start()` to each `SymbolHub` instance

*Verification (next session — requires market hours):*
1. Restart the app during market hours (RTH or extended)
2. Wait 60+ seconds for first backfill cycle
3. Check logs for `Backfill SPY/60: replaced N bars` messages confirming REST data is being applied
4. Compare live chart vs price chart — after-hours bars should now match
5. Confirm indicator values (especially UTBOT trailing stop) converge with backtest/forward test
6. Verify alert markers (X symbols) are unaffected by backfill cycles

### Phase 28: Ralph Wiggum Alert Engine (Branch: `ralph-wiggum`)
*Complete rewrite of the alert detection engine — replaces O(N×M) batch pipeline re-runs with O(1) incremental indicator updates per tick. Self-contained in `ralph_engine.py` (~2,900 lines).*

**Problem:** The existing `realtime_engine.py` (Phases 27/27B/27C) re-ran the full indicator → interpreter → trigger pipeline on every 500ms evaluation cycle, scaling as O(N×M) where N = number of bars and M = number of strategies. For multi-symbol/multi-timeframe monitoring, this became a CPU bottleneck and introduced latency between tick arrival and alert dispatch.

**Architecture — O(1) Incremental Engine (Phases 1–5):**
- [x] `BarBuilder` — aggregates ticks into OHLCV candles per timeframe; emits bar-close events; maintains rolling history window (1,000 bars)
- [x] `IncrementalIndicatorEngine` — updates EMA, MACD, VWAP, RVOL, ATR incrementally from single new bar (O(1) per indicator); no DataFrame recomputation
- [x] `TriggerEvaluator` — evaluates trigger conditions (UTBot, EMA Stack, MACD Cross, VWAP Zone, etc.) from latest indicator values; returns boolean per trigger
- [x] `PositionStateMachine` — tracks FLAT → IN_POSITION → FLAT lifecycle per strategy; handles entry/exit/opposite-signal transitions
- [x] `StrategyMonitor` — orchestrates per-strategy evaluation: builds confluence records, checks entry/exit triggers against required confluence, manages position state
- [x] `AlertDispatcher` — fires alerts via webhook (Discord/Slack-compatible embeds); enriches payload with portfolio context, position sizing, target price
- [x] `FidelityAuditor` — periodic batch pipeline comparison (every 120s); logs drift between incremental and batch indicator values to `engine_audit.jsonl`
- [x] `SymbolHub` — per-symbol coordinator: owns BarBuilders (one per timeframe), routes ticks, coordinates bar-close evaluation, manages REST backfill and pickle writes
- [x] `RalphEngine` — top-level orchestrator: Alpaca WebSocket connection, strategy loading, hot-reload, graceful shutdown, status IPC

**Phase 6: Gap Analysis Fixes (COMPLETE):**
- [x] General Pack scalar evaluators — time-of-day, day-of-week, trading session, calendar filter checks evaluated at bar-close; `GEN-{pack_id}-{state}` records added to confluence set
- [x] EMA period resolution from confluence group parameters — replaces hardcoded [8, 21, 50] with actual short_period/mid_period/long_period from strategy's confluence group
- [x] `opposite_signal` exit handling — resolves via `get_opposite_trigger()` from triggers.py using suffix-based pair matching (_buy↔_sell, _bull↔_bear, etc.)
- [x] `target_price` added to alert dispatch payload
- [x] Sub-minute timeframe guard in REST reconciliation (skip tf_seconds < 60)
- [x] Thread-safe reconciliation with `copy()` on shared state
- [x] Removed dead imports and cached pytz timezone at module level
- [x] Fixed `_start_time` ordering (set before first `_write_status()`)

**Runtime Bug Fixes (COMPLETE):**
- [x] Duplicate logging guard — `__main__` + `ralph_engine` double-import caused duplicate RotatingFileHandlers; guard checks for existing handler before adding
- [x] False-positive "Engine Streaming" status — `_ws_confirmed` flag replaces unreliable `_stream_ref is not None` check; only True when first trade actually arrives
- [x] Zombie process detection in app.py — `os.kill(pid, 0)` returns success for zombies; added `/proc/{pid}/status` check for zombie state
- [x] Retry storm fix — Alpaca SDK's `_run_forever()` catches ValueError/WebSocketException internally and retries with sleep(0); `_patched_run_forever` now propagates ALL errors to outer handler's exponential backoff (5s → 120s max)
- [x] `dotenv` override — `load_dotenv(override=True)` ensures env changes picked up on engine restart
- [x] Configurable data feed — `ALPACA_DATA_FEED` env var (default "sip"); supports SIP and IEX via Alpaca `DataFeed` enum

**Live Chart & Data Quality Fixes (COMPLETE):**
- [x] Pickle enrichment pipeline — `_write_chart_pickles()` now runs the full indicator pipeline matching backtest's `prepare_data_with_indicators()`: `run_all_indicators()` → `run_indicators_for_group()` per enabled group → `run_all_interpreters()` → `detect_all_triggers()` → general pack evaluation. Previously only ran `run_all_indicators()` and `detect_all_triggers()`, causing custom EMAs, utbot_stop, and interpreter overlays to be missing from the live chart.
- [x] Trade condition filtering — `EXCLUDED_TRADE_CONDITIONS` set (18 CTA/UTP codes) filters odd-lot, average-price, out-of-sequence, and other non-standard trades in the `on_trade` WebSocket handler. Matches Alpaca's server-side filtering on historical bar aggregation, eliminating false price spikes on live candles.
- [x] Alert History table — `render_alert_trade_table()` in app.py displays real-time alert-based trade history on the Live Chart tab (above the theoretical trade table). Pairs entry/exit alerts into trade rows with HH:MM:SS timestamps; shows "Open" for entries awaiting exits.
- [x] Alert entry/exit pairing — Pairing logic uses exit alert's `entry_price` field for matching ($0.01 tolerance) instead of sequential order, preventing mismatches when orphaned alerts from previous sessions exist.
- [x] Atomic alerts.json writes — `_save_all_alerts()` in alerts.py uses tmp file + `os.replace()` to prevent partial reads during concurrent Streamlit UI access.
- [x] Live chart right offset — 10-candle buffer on the right side of price charts for indicator label clearance (`rightOffset: 10` in timeScale config).
- [x] Graceful shutdown — `RalphEngine.stop()` uses SIGTERM → wait → SIGKILL escalation with force WebSocket close to prevent orphaned connections holding the Alpaca single-connection-per-key limit.

**Backtest–Live Trigger Parity (IN PROGRESS):**

The backtest pipeline and Ralph's live engine must evaluate triggers identically — any divergence means live alerts fire on conditions that backtests never tested, undermining strategy validity.

**Core discovery (2026-03-04):** All `_ib` (intra-bar) triggers in `check_intrabar()` check only whether the current tick price is above/below a cached indicator level. They do **not** verify the transition condition (e.g., "previous bar closed on the opposite side") that the backtest's `detect_*_triggers()` functions require. This means any `_ib` trigger fires as a **condition** ("are we above the level?") rather than a **crossover** ("did we just cross the level?"), producing phantom entries whenever price is trending above/below the level for multiple bars.

**Example (observed live 2026-03-04):** TSLA LONG V2 S-Cross strategy using `ema_pp_v2_cross_short_up_ib` re-entered immediately after every `bar_count_exit` because price was above the 9 EMA — the intra-bar trigger fired on the first tick of every bar since `price > ema_8_prev` was always true while trending up. The backtest correctly required a transition (`prev_close <= prev_ema AND current_close > ema`) and would not have re-entered.

**Fix (two-tier gating):** L-type and H-type `_ib` triggers require different gate logic to match the backtest while preserving intra-bar speed:

- **L-type triggers** (EMA Price Position v1/v2, VWAP crosses): Gate is "prev-bar-close-opposite-side" — the just-closed bar's close must be on the opposite side of the indicator level from the cross direction (e.g., `close <= ema` for an "above" cross). Computed in `_compute_ib_gates()` at bar close, consumed by `check_intrabar()` on the next bar's ticks via `_ib_gate_open`. This matches the backtest's crossover condition (`prev_close <= prev_ema`) and fires immediately when the tick crosses the level — no bar-close delay.
- **H-type triggers** (UT Bot v1/v2): Gate is "bar-close-trigger-fired" — the bar-close boolean (direction flip) must have evaluated True on the previous completed bar. Uses existing `_bar_close_triggers` dict.

Additionally, v2 `_prev` cached levels (EMA and UT Bot) were corrected to cache the just-closed bar's indicator value (matching the backtest's `.shift(1)` semantics) instead of the bar-before-that's value.

- [x] UT Bot (v1 & v2) — Bar-close direction flip gates intra-bar level crossing via `_bar_close_triggers`
- [x] UT Bot `_ib` trigger registration — `_process_trigger_id()` registers base trigger alongside `_ib` variant
- [x] **L-type prev-bar-close-opposite-side gate** — EMA PP (v1 & v2), VWAP crosses use `_ib_gate_open` dict populated by `_compute_ib_gates()` at bar close; `_IB_L_TYPE_TRIGGERS` frozenset dispatches gate type in `check_intrabar()`
- [x] **v2 `_prev` level correction** — `_update_cached_levels()` now caches the just-closed bar's indicator value for `_prev` keys, fixing off-by-one vs backtest `.shift(1)`
- [ ] **Document parity contract** — Formalize the rule that any new trigger added to Ralph must match the backtest evaluation path. New `_ib` triggers MUST declare their type (L or H) and have corresponding gate logic.

**Additional fixes (2026-03-04):**
- [x] Alert history table timing — switched from `bar_time` (candle open) to `timestamp` (actual dispatch time) so times match the Alerts tab
- [x] Bar-close signal timestamps — `on_bar_close()` now uses raw `bar_start` timestamp (matching backtest's `df.index = bar-start` convention and chart candle positioning). Originally set to `bar_start + timeframe - 1s` but reverted in Phase 30I QA to eliminate timestamp offset between alert markers and backtest markers.
- [x] Alerts.json race condition — added `threading.Lock` around all read-modify-write cycles in `alerts.py`; tmp files use unique names (`{pid}.{thread_id}`); `deliver_alert()` in `alert_monitor.py` uses new `update_alert()` helper instead of raw load/modify/save

**Cross-Timeframe Confluence in Live Engine (2026-03-13):**

Strategies with multi-timeframe (MTF) confluences (e.g., "15M VWAP > 2σ" + "5M MACD H+" + "1H UO Bullish" on a 1-min trigger) worked correctly in backtests but never fired entries in the live engine. The issue was a missing data-feed step in ralph_engine's orchestration layer — not a logic divergence between engines (`check_entry()` is literally the same code).

**Root cause:** `StrategyMonitor.on_bar_close()` only built confluence records from its own timeframe. `PositionStateMachine.check_entry()` checks `confluence_set.issubset(confluence_records)`, so cross-TF records were always missing → entry blocked forever.

**Fix components:**
- [x] **Label case normalization** — `_normalize_confluence_label()` uppercases TF prefix (backtest stores lowercase "15m-...", live produces uppercase "15M-..."). Called in `StrategyMonitor.__init__` to normalize `confluence_set`.
- [x] **`_required_secondary_tf: Set[int]`** — Parsed from strategy's confluence records in `StrategyMonitor.__init__`; identifies which other timeframes are needed.
- [x] **`SymbolHub._mtf_confluence: Dict[int, Set[str]]`** — Shared buffer mapping `tf_seconds → latest interpreter confluence records` from that TF. Updated on every bar close.
- [x] **BarBuilder creation for secondary TFs** — `add_monitor()` creates BarBuilders for secondary TFs so ticks build bars at those timeframes.
- [x] **`_ShadowIndicatorEngine`** — Lightweight class (IncrementalIndicatorEngine + TriggerEvaluator, no position management) that computes indicators/interpreters on secondary TF bars and returns confluence records. Created by `SymbolHub.finalize_shadow_engines()` only for TFs not covered by real monitors.
- [x] **Cross-TF merge in `on_bar_close()`** — After building own confluence, merges records from `_mtf_confluence` for other TFs. Existing `check_entry(confluence_records=...)` call now includes cross-TF data.
- [x] **Worker hot-reload** — `finalize_shadow_engines()` called after adding new monitors; shadow engines warmed up for secondary TFs.
- [x] **Confluence timing** — Cross-TF confluence updates at secondary bar close (not intra-bar), matching backtest's forward-fill behavior.

**Live chart fixes (2026-03-13):**
- [x] **Missing tables below live chart** — `render_alert_trade_table()` and `render_backtest_trade_table()` were after a `return None` in `_load_live_chart_pickle`. Moved into `render_live_chart_tab` after conditions render.
- [x] **Empty-state headers** — Always show "Current Conditions", "Alert History", "Trade History (Backtest)" headers with descriptive empty-state messages even when no data exists.
- [x] **Cross-TF conditions table** — `_render_live_conditions()` was filtering by strategy's own TF prefix only, so cross-TF confluences never matched. Fixed to parse TF from each confluence record directly.
- [x] **`_load_secondary_tf_states()`** — New function that loads REST data for secondary TFs, runs indicator/interpreter pipeline, returns real current states for the conditions table instead of placeholder text.
- [x] **Live chart trade markers for MTF strategies** — `render_live_chart_tab()` was calling `run_unified_backtest()` without `secondary_tf_map`, so strategies with cross-TF confluence requirements could never generate trades on the live chart. Fixed by resampling primary data to required secondary TFs, running the same indicator/interpreter pipeline as `prepare_data_with_indicators()`, and passing the resulting `secondary_tf_map` through to the unified engine.

**IPC Files (runtime, gitignored):**
- `engine_status.json` — running/connected/tick_count/PID for Streamlit UI polling
- `engine_state.json` — position states per strategy for persistence across restarts
- `engine_audit.jsonl` — fidelity audit log (incremental vs batch drift)
- `engine_reload.flag` — touched by UI to trigger hot-reload of strategies.json
- `live_data_{symbol}_{tf_seconds}.pkl` — enriched DataFrames for Live Chart tab (symbol "/" sanitized to "_" for crypto)

**CLI Interface:**
- `python ralph_engine.py` — start engine (foreground)
- `python ralph_engine.py --status` — print current engine status
- `python ralph_engine.py --stop` — send SIGTERM to running engine
- `python ralph_engine.py --dry-run` — validate config without connecting

**Verification (live market testing — started 2026-03-03, QA session 2026-03-06):**
1. [x] Start engine during RTH or extended hours with SIP feed
2. [x] Confirm WebSocket connects and ticks arrive (tick_count > 0 in status)
3. [x] Watch bar building and indicator updates in log
4. [x] Verify alerts fire on trigger transitions with correct confluence matching
5. [x] Confirm Live Chart tab renders from pickle data with full indicator overlay
6. [x] End-to-end alert lifecycle: entry alert → position tracking → exit alert → paired in Alert History table (verified 2026-03-06)
7. [x] L-type entry markers (× and +) align within acceptable tolerance — verified 2026-03-06
8. [ ] Check fidelity audit log for acceptable drift levels
9. [ ] Compare live chart candle ranges to backtest — confirm trade condition filtering eliminates false spikes
10. [x] Bar-count exit markers show expected slippage from WebSocket vs REST close price drift (see Phase 30I analysis)

---

### Phase 29: Trigger Classification & Backtest-Live Consistency

*Establish a formal trigger execution type system, ensure backtest and live behavior match for every trigger type, and surface reliability warnings to users.*

**Motivation:** Live market testing (Phase 28) revealed that intra-bar triggers fire on conditions rather than crossovers, producing phantom entries inconsistent with backtest results. Additionally, hybrid triggers (e.g., UT Bot) have inherent flicker risk where the indicator toggles buy/sell within a candle but the backtest only sees the final state. Users need transparency about which triggers are reliable for live execution and which carry caveats.

#### Trigger Execution Types

Every trigger is classified into one of three execution types:

| Type | Label | Description | Flicker Risk | Backtest Reliability |
|------|-------|-------------|-------------|---------------------|
| **C** | Bar Close | Fires only at bar close. No intra-bar variant. | None | High — backtest = live |
| **L** | Level Cross | Fires intra-bar when price crosses a level, once per bar. | None (once gated) | High after Phase B adjustment |
| **H** | Hybrid | Requires internal state flip at bar close + level cross intra-bar. | Yes — state can toggle within candle | Medium — backtest sees only final state |

**Trigger classification:**
- **C triggers:** EMA Stack crosses (`ema_cross_bull/bear`, `ema_mid_cross_bull/bear`), MACD crosses (`macd_cross_bull/bear`, `macd_zero_cross_up/down`), MACD Histogram flips/momentum shifts, RVOL triggers
- **L triggers:** VWAP crosses, EMA Price Position (v1 & v2)
- **H triggers:** UT Bot (v1 & v2)

#### Phase A: Trigger Classification Metadata & UI Warnings
- [ ] Add `execution_type: "C" | "L" | "H"` field to confluence group TEMPLATES dict
- [ ] Display execution type badge next to trigger name in Strategy Builder and Strategy Detail page
- [ ] For H triggers: show warning icon with tooltip explaining flicker risk and that backtest results may differ from live execution
- [ ] Add execution type to confluence pack builder spec so user-created packs self-declare their characteristics
- [ ] Ensure ALL trigger types (including C / bar-close only) produce alerts and appear in the Live Chart alert history table

#### Phase B: Backtest-Live Consistency for L Triggers
- [x] L-type triggers now use prev-bar-close-opposite-side gating in Ralph (Phase 28 fix), matching the backtest's crossover boolean. No backtest adjustment needed — the prev-bar-close gate IS the backtest's crossover condition evaluated one bar earlier, and the `_ib` fill is at the indicator level (not close), matching `INTRABAR_LEVEL_MAP`.
- [ ] Consider whether backtest `detect_*_triggers()` should also support a "level-fill-only" mode (fire when bar's price range touches the level, regardless of crossover) as an alternative to crossover-based triggering. This would be a new trigger variant, not a replacement.
- [ ] For H triggers (UT Bot): consider adding a third trigger variant (L-type) that fires purely on stop level cross with backtest adjusted to match, giving users the choice between C, L, and H behavior per trigger

#### Phase C: Optional Strategy Safeguards
- [ ] Add `re_entry_cooldown_bars` as an optional strategy parameter — prevents re-entry for N bars after an exit
- [ ] Not a substitute for correct trigger logic, but useful for strategy tuning and noise reduction
- [ ] Consider default of 0 (no cooldown) with explicit opt-in

#### Phase D: Advanced Execution Modes (Future)
- [ ] **Sub-timeframe candle confirmation** — Evaluate triggers on smaller timeframe bars (e.g., 10-second) while using higher timeframes for context via existing MTF confluence infrastructure. Requires data provider upgrade (Polygon/Massive for sub-minute historical data).
- [ ] **Flicker guard** — Probationary entry with tight intra-candle stop at the cross level; if price wicks back below before bar close, exit at small loss. After bar close confirms, move to real strategy stop. Complex — touches webhooks, position tracking, and alert lifecycle.
- [x] **Data provider upgrade decision (2026-03-06)** — Decided to migrate from Alpaca market data to Polygon.io/Massive for pre-aggregated WebSocket bars. See Phase 31 for implementation plan.

#### Long-Term Architecture: Unified Chart/Engine Convergence

The current architecture maintains two separate computation paths: the batch pipeline (indicators.py → interpreters.py → triggers.py) for backtesting/charting, and the incremental pipeline (ralph_engine.py) for live execution. Every trigger must be implemented identically in both paths, and any divergence produces incorrect live behavior (as demonstrated by the _ib gate and _prev level bugs).

The long-term direction is to converge on a **single unified engine** (TradingView model) where:
1. One computation path handles both historical and live data
2. Indicators, interpreters, and triggers are defined once and evaluated identically
3. The bar-by-bar state machine processes historical bars for backtesting and live bars for alerting
4. Charting reads from the same computed state regardless of whether data is historical or live

This eliminates the parity problem entirely: if there is only one engine, backtest and live behavior are identical by construction. The Polygon/Massiv migration is a natural inflection point for this convergence. The batch pipeline would still handle rapid backtesting over long history, but the forward/live portion would be unified under one engine.

---

### Phase 30: Unified Chart Engine (Implementation Spec)

**Goal:** Replace the dual-pipeline architecture with a single bar-by-bar engine that evaluates all trigger types identically for backtesting and live execution, eliminating parity bugs by construction.

#### Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  Unified Engine                       │
│                                                      │
│  Historical bars ──┐                                 │
│                    ├──► Bar-by-bar state machine      │
│  Live bars ────────┘    (indicators → triggers →     │
│                          position tracking)           │
│                                                      │
│  Outputs:                                            │
│    - Trade log (backtest results)                    │
│    - Chart data (indicator overlays, signals)        │
│    - Live alerts (webhook dispatch)                  │
│    - Position state (entry/exit tracking)            │
└──────────────────────────────────────────────────────┘
```

The engine processes bars sequentially regardless of source. Historical bars replay at full speed; live bars arrive from the WebSocket feed. The same code path evaluates indicators, checks triggers, and manages positions in both cases.

#### Trigger Execution Types

Every trigger is classified into one of four execution types. The type is intrinsic to the trigger definition and appears as a distinct trigger variant in the strategy builder. Because each execution type produces its own backtest KPIs, the builder naturally surfaces which execution mode is optimal for a given indicator cross — a user comparing triggers will see C-type, L-type, HM-type, and HL-type variants side by side with their respective profit factors, win rates, and drawdowns.

##### C-Type (Bar Close)
- **Evaluation:** At bar close only
- **Fill price:** Bar close price
- **Examples:** EMA stack state changes, MACD line cross, MACD histogram flip, bar count exit
- **Backtest:** `entry_price = close` on the bar where the boolean fires
- **Live:** Evaluated in `evaluate_bar_close()`, fill at close price

##### L-Type (Level Cross)
- **Evaluation:** Intra-bar (on every tick during live; simulated via high/low reachability in backtest)
- **Fill price:** The indicator level being crossed
- **Gate:** Previous bar's close must be on the **opposite side** of the level from the cross direction (prevents phantom entries on bars where price is already on the correct side)
- **Examples:** EMA Price Position v1/v2 crosses, VWAP crosses, VWAP extreme zone entries
- **Backtest:**
  - Boolean: `(prev_close <= level.shift(1))` for "above" crosses (no `close > level` requirement)
  - Reachability: `high >= level_prev` (confirms intra-bar price reached the level)
  - Fill: `entry_price = level_prev` (the indicator level, not close)
- **Live:**
  - Gate computed at bar close: `close <= level` opens the gate for "above" crosses on the next bar
  - First tick crossing the cached level fires entry immediately at the level price
  - One-shot per bar (no re-fire after first trigger)

##### HM-Type (Hybrid — Level Cross with Market Exit on Non-Confirmation)
- **Evaluation:** Intra-bar for entry (same as L-type), bar close for confirmation
- **Fill price:** The indicator level being crossed (same as L-type)
- **Gate:** Same as L-type — previous bar's close on the opposite side
- **Entry behavior:** Identical to L-type — enters immediately when price crosses the cached indicator level
- **Confirmation check:** After entry, the engine monitors the bar close. If the bar closes and the indicator state does NOT confirm the direction of the trade, the position exits immediately.
- **Unconfirmed exit:** Close position at the next bar's open (or next available tick in live). This is the conservative approach — limits exposure to one bar of unconfirmed price action.
- **Examples:** UT Bot ATR trailing stop crosses (conservative)
- **Backtest simulation:**
  - Entry: Same as L-type (level cross on the bar, fill at level)
  - Confirmation: Check if the entry bar's close confirms the indicator state
  - If unconfirmed: exit at next bar's open price
  - If confirmed: position continues under normal exit rules
- **Live simulation:**
  - Entry: Same as L-type (tick crosses cached level, enter at level)
  - At bar close: evaluate indicator state
  - If confirmed: position continues under normal exit rules
  - If unconfirmed: dispatch exit alert, close at market on next tick

##### HL-Type (Hybrid — Level Cross with Limit Exit on Non-Confirmation)
- **Evaluation:** Intra-bar for entry (same as L-type), bar close for confirmation
- **Fill price:** The indicator level being crossed (same as L-type)
- **Gate:** Same as L-type — previous bar's close on the opposite side
- **Entry behavior:** Identical to L-type — enters immediately when price crosses the cached indicator level
- **Confirmation check:** Same as HM-type — monitors bar close for indicator state confirmation
- **Unconfirmed exit:** Place a virtual limit order at the entry price (break-even exit). If price returns to the entry level, exit at no loss. If price never returns, fall through to normal exit triggers (stop loss, opposite signal, bar count). This is the optimistic approach — gives the trade room to recover while capping downside to the original entry level.
- **Examples:** UT Bot ATR trailing stop crosses (optimistic)
- **Backtest simulation:**
  - Entry: Same as L-type (level cross on the bar, fill at level)
  - Confirmation: Check if the entry bar's close confirms the indicator state
  - If unconfirmed: scan subsequent bars for `low <= entry_price` (longs) or `high >= entry_price` (shorts); if reached, exit at entry price; otherwise, fall through to normal exits
  - If confirmed: position continues under normal exit rules
- **Live simulation:**
  - Entry: Same as L-type (tick crosses cached level, enter at level)
  - At bar close: evaluate indicator state
  - If confirmed: position continues under normal exit rules
  - If unconfirmed: set virtual limit at entry price; exit if price returns to that level; otherwise, normal exit rules apply

> **Design rationale — HM vs HL as separate execution types:** Rather than making the unconfirmed exit mode a post-hoc configuration, HM and HL are first-class trigger variants that appear in the strategy builder alongside C-type and L-type. This leverages the existing builder infrastructure that surfaces optimal triggers by backtest KPIs. A user evaluating an EMA 9 short cross will see all four variants — C, L, HM, HL — with their respective profit factors, win rates, and max drawdowns. The builder naturally recommends the execution mode that performs best for that specific indicator and timeframe combination, without requiring the user to manually A/B test configurations.

#### Data Flow

1. **Bar arrives** (historical or live close)
2. **Indicators update** — EMA, MACD, VWAP, RVOL, ATR, UT Bot (O(1) incremental)
3. **Trigger evaluation:**
   - C-type: Evaluate boolean at close
   - L-type: Evaluate level cross (backtest: reachability check; live: already fired intra-bar or fall back to close)
   - HM/HL-type: Same as L-type for entry
4. **Position state machine** — Check entries, exits, stop losses
5. **Confirmation check** — For any HM/HL-type position entered this bar, check indicator state at close
6. **Output** — Append to trade log, update chart state, dispatch alerts (live only)
7. **Gate update** — Cache indicator levels and compute L-type/HM/HL-type gates for next bar
8. **Intra-bar loop** (live only): Between bar closes, check L-type and HM/HL-type level crosses on each tick

#### Implementation Phases

##### Phase 30A: Unified Engine Core — COMPLETED 2026-03-03
- [x] Create `unified_engine.py` with bar-by-bar state machine
- [x] Port incremental indicator calculations from `ralph_engine.py`
- [x] Implement C-type trigger evaluation (bar close only)
- [x] Implement L-type trigger evaluation with reachability simulation for backtest
- [x] Add L-type prev-bar-close gate (same logic as current `_compute_ib_gates`)
- [x] Position state machine with entry/exit tracking
- [x] Trade log output matching current `generate_trades()` DataFrame format
- [x] Verify backtest parity: unified engine produces identical trades to current batch pipeline on historical data

##### Phase 30B: HM/HL-Type Confirmation Protocol — COMPLETED 2026-03-03
- [x] Implement HM-type entry (same path as L-type) with bar-close confirmation check
- [x] Implement HM-type unconfirmed exit (market exit at next bar open)
- [x] Implement HL-type entry (same path as L-type) with bar-close confirmation check
- [x] Implement HL-type unconfirmed exit (limit exit at entry price, fallback to normal exits)
- [x] Register HM and HL trigger variants in confluence group templates
- [x] Backtest simulation of confirmation and unconfirmed exits for both modes
- [x] Verify backtest produces expected results for UT Bot strategies under HM and HL

##### Phase 30C: Live Integration — COMPLETED 2026-03-04
- [x] Refactor `ralph_engine.py` to import shared components from `unified_engine.py` (3079 → 1680 lines)
- [x] Add live-tick methods to unified PSM: `check_entry_intrabar()`, `check_exit_tick()`, `check_exit_intrabar()`, `check_exit_bar_close()`
- [x] Add `check_intrabar()` to TriggerEvaluator with HM/HL gate support
- [x] Add `PositionState.from_dict()` for state persistence, `_signal_exit()` for live signal dicts
- [x] Intra-bar tick processing for L-type and HM/HL-type level crosses
- [x] HM/HL live confirmation flow: entry on tick → confirmation at bar close → pending exit on next tick
- [x] Position state persistence (engine_state.json) via unified `PositionState.to_dict()`/`from_dict()`
- [x] Hot-reload, status reporting, fidelity auditing (retained from ralph_engine)
- [x] **Live market verification** — passed 2026-03-05 (6,986 audit entries, 0 errors, 898/898 entries/exits balanced)

##### Phase 30D: Chart Convergence — COMPLETED 2026-03-05
- [x] Replace `generate_trades()` with `run_unified_backtest()` at 6 strategy-level call sites via `_unified_trades()` helper
- [x] MTF fallback: strategies with multi-timeframe confluence records fall back to `generate_trades()` automatically
- [x] Error fallback: unified engine failures degrade gracefully to `generate_trades()` with log warning
- [x] Live chart tab (`render_live_chart_tab`) retained on `generate_trades()` — reads pre-enriched pickle data, no recomputation needed
- [x] Batch pipeline (`indicators.py → interpreters.py → triggers.py`) retained for chart indicator overlays only — not for trade generation

##### Phase 30E: MTF Support + Swing Stops — COMPLETED 2026-03-05
- [x] MTF support: `run_unified_backtest()` accepts `secondary_tf_map` parameter, reads pre-computed `{INTERP}__{tf_label}` columns and merges into confluence records
- [x] MTF fallback removed: `_has_mtf_confluence()` guard eliminated — unified engine handles all strategies including MTF
- [x] Swing stops: `PositionStateMachine` maintains rolling `_high_low_buffer` (deque, maxlen=50) for lookback-based min/max stop and target calculation
- [x] Swing targets: `_compute_target(method='swing')` uses same lookback buffer — max(highs) + padding for LONG, min(lows) - padding for SHORT
- [x] Fallback: insufficient history degrades to ATR * 1.5 for stops, returns None for targets
- [x] 6 new tests: 5 swing stop/target tests + 1 MTF confluence gating test (27 total unified tests)

##### Phase 30F: Live Chart Parity + Open Position Markers — COMPLETED 2026-03-05
- [x] Migrated `render_live_chart_tab()` from `generate_trades()` to `_unified_trades()` — all charts now use the unified engine, eliminating divergence risk
- [x] Open-position entry markers: `run_unified_backtest(include_open_position=True)` appends synthetic trade row with `exit_time=None` when position is open at end of data
- [x] Chart renders orange "Open" arrow at entry point immediately (before trade closes), distinct from blue "Entry" arrows on closed trades
- [x] `calculate_kpis()` filters out `exit_reason='open'` rows to prevent KPI contamination
- [x] Trade history table shows "Open" result for open-position rows

##### Phase 30G: Execution Type Naming Convention + L-type Bugfix — COMPLETED 2026-03-05
- [x] Fixed `evaluate_bar_for_backtest()` bug: `_update_cached_levels()` was overwriting `_prev` values before L-type reachability checks, causing V2 triggers to use current bar's level instead of previous bar's
- [x] Split single `L` execution type into `L0` (current-bar level) and `L1` (previous-bar level):
  - `L0`: VWAP triggers — cross level updates continuously during bar (fill = current bar's indicator)
  - `L1`: V2 EMA/UT Bot triggers — cross level fixed from previous bar's close (fill = previous bar's indicator)
- [x] `get_trigger_exec_type()` derives L0/L1 from `INTRABAR_LEVEL_MAP` column: `_prev` suffix → L1, else L0
- [x] UI tags updated: `[L0]`, `[L1]` replace `[L]` throughout Strategy Builder and detail views
- [x] Deleted legacy strategies (IDs 23, 24, 25) using deprecated non-V2 triggers
- [x] Removed outdated default templates (`ema_price_position`, `utbot`) from TEMPLATES dict

#### Execution Type Naming Convention

| Tag | Name | Level Source | Fill Price | Gate Logic |
|-----|------|-------------|-----------|------------|
| `C` | Bar Close | Close detection | Close price | N/A |
| `L0` | Level Cross (current) | Current bar's indicator | Indicator level | Prev close on opposite side |
| `L1` | Level Cross (previous) | Previous bar's indicator | Previous bar's level | Prev close on opposite side |
| `HM` | Hybrid Market | L1 cross + bar-close confirmation | Market at confirmation | Same as L1 |
| `HL` | Hybrid Limit | L1 cross + bar-close confirmation | Limit at level | Same as L1 |

> **Future: Order Type Extensions (CM/CL)** — Exploring market vs limit order semantics for bar-close triggers. CM (close + market order at next open) vs CL (close + limit order at close price). Requires unfilled-order logic and next-bar fill verification. Not yet prioritized.

##### Phase 30H: L-type Gate Timing Fix — COMPLETED 2026-03-05
- [x] Fixed gate/crossing level mismatch: `_compute_ib_gates()` gate_map for V2 triggers used `utbot_stop_prev` (bar N-2's stop) while crossing checked `_cached_levels['utbot_stop_prev']` (bar N-1's stop) — during direction changes these diverge, causing false entries in the wrong direction
- [x] Fix: V2 gate_map entries now reference `utbot_stop` (same bar's stop as the crossing level), matching TradingView's `ta.crossover(close, trail)` semantics where gate and crossing reference the same bar
- [x] Fixed backtest same-bar timing: `evaluate_bar_for_backtest()` was computing gate from bar N then running L-type reachability on bar N (same bar), causing entries on nearly every bar where `close <= stop AND high > stop`
- [x] Fix: Save `prev_gates` and `prev_cached` before `evaluate_bar_close()` overwrites them, use saved values for L-type reachability checks — matches live timing where gate is set at bar N-1's close and crossings happen during bar N
- [x] UT Bot V2 triggers (`utbot_v2_buy`, `utbot_v2_sell`) confirmed working as L-type with correct buy/sell direction discrimination
- [x] Backtest selectivity improved: test strategies went from 68 entries (24 level-filled) to 61 entries (all level-filled)

> **Future: Minimum Gate Bars Parameter** — Adding a `min_gate_bars` parameter to L-type triggers that requires price to remain on the opposite side of the level for N consecutive bars before the gate opens. This filters out "flash" crosses where price briefly dips below/above the level for a single bar before reverting. Default of 1 preserves current behavior. Requires counter tracking in TriggerEvaluator + Strategy Builder UI field.

##### Phase 30I: Live QA Hardening — COMPLETED 2026-03-09
Live QA session with SPY UT Bot Extended Test strategy (L1-type, 1-min bars, 4-bar count exit, 5-bar swing stop) uncovered and fixed multiple backtest–live parity issues:

- [x] **Alert timestamp alignment** — `on_bar_close()` in ralph_engine was using `bar_start + tf_seconds - 1` (bar-end) for alert timestamps; backtest uses `bar_start` (bar-open). Fixed: alerts now use raw bar_start timestamp, matching backtest convention and chart candle positioning.
- [x] **Swing stop buffer population** — `update_high_low()` was called in backtest `process_bar()` but never in ralph_engine's `on_bar_close()`, causing swing stops to fall back to ATR × 1.5 in live. Fixed: added `update_high_low(bar['high'], bar['low'])` call after `update_bar()`.
- [x] **Entry bar count rebase** — After engine restart, `_bar_count` resets from warmup length (~200) but restored `entry_bar_count` from old session (e.g., 4764) made bar_count_exit impossible. Fixed: after warmup, rebases any restored position's `entry_bar_count` to current `_bar_count` if the old value exceeds it.
- [x] **Open trade alert matching** — `_to_chart_unix(None)` for open trades (exit_time=None) caused them to be skipped in the alert marker matching loop. Fixed with `is_open` guard.
- [x] **Partial bar suppression (C-type only)** — Live chart runs backtest on pickle data including a still-forming candle. Originally suppressed ALL signals on partial bars; corrected to only suppress C-type triggers and bar_count_exit. L-type signals still fire on partial bars since they represent real intra-bar level crosses.
- [x] **Same-bar L-type stop/target guard** — Bar-close exit check uses full bar OHLC, but for L-type entries mid-bar, the bar's open/low from BEFORE entry creates false stop triggers with incorrect gap-aware fills. Fixed: `same_bar_ltype` guard skips OHLC-based stop/target checks on the entry bar for L-type entries (both backtest `check_exit()` and live `check_exit_bar_close()`).
- [x] **1-bar cooldown** — Prevents re-entry on the same bar as exit, reducing whipsaw trades from rapid exit→re-entry sequences. Added `last_exit_bar_count` field to `PositionState`; both `check_entry()` and `check_entry_intrabar()` skip if `last_exit_bar_count >= bar_count`. Preserved across `_reset_position()`.
- [x] **L-type entry ATR parity** — Backtest was using current bar's ATR for L-type entry stop computation; live only has previous bar's ATR since current bar hasn't closed. Fixed: backtest uses `prev_values` for L-type stop/target computation, matching live behaviour.
- [x] **`last_exit_bar_count` propagation in live path** — `_signal_exit()` accepts optional `bar_count` parameter; `check_exit_bar_close()` passes it for all exit types. Tick-level exits (`check_exit_tick()`, `check_exit_intrabar()`) record via caller (`on_tick()` in ralph_engine).
- [x] **L-type intra-bar `entry_bar_count` alignment** — `check_entry_intrabar()` was setting `entry_bar_count = bar_count` (the builder's count from the *previous* bar close), but `on_bar_close` receives `bar_count + 1` (incremented when the entry bar closes). This off-by-one caused `bars_held` to be 1 on the entry bar close (instead of 0), making bar_count_exit fire 1 bar early for L-type entries. Also broke `same_bar_ltype` guard and HM/HL confirmation check. Fixed: `entry_bar_count = bar_count + 1` in `check_entry_intrabar()`, aligning with the bar_count value the entry bar's `on_bar_close` will receive. C-type entries (via `check_entry()`) unchanged — already aligned.
- [x] **`last_exit_bar_count` rebase on restart** — After engine restart, `last_exit_bar_count` from the old session (e.g., 4900) persisted while the new session's `bar_count` restarted from warmup length (~300). The 1-bar cooldown check (`last_exit_bar_count >= bar_count`) permanently blocked ALL entries for every strategy. The existing rebase logic only handled `entry_bar_count` for IN_POSITION strategies. Fixed: reset `last_exit_bar_count` to 0 for any strategy where the old value exceeds the current `bar_count` after warmup.

**Live QA verification — 2026-03-09:**
- All trigger execution types (C-type and L-type) confirmed firing correctly with proper timing
- Entry and exit alert markers (×) align with backtest markers (+) on the live chart
- Alert dispatch and webhook delivery working as expected across all monitored strategies

**Live QA findings — alert vs backtest exit price drift:**
- L-type entry markers (× and +) align closely because both systems price off computed indicator levels
- C-type exit markers (bar_count_exit) show sporadic price drift (~$0.05–$0.20) because:
  - Alert × uses WebSocket-aggregated close (real-time, from BarBuilder)
  - Backtest + uses REST API-reconciled close (retrospective, from `_reconcile_bars()` every 60s)
  - These are fundamentally different data sources for the bar's close price
- This is inherent to tick-stream bar building and cannot be fully eliminated without switching to pre-aggregated bars
- **Decision: Migrate to Polygon.io/Massive** for pre-aggregated WebSocket bars (per-second and per-minute), eliminating the bar-building parity problem entirely. See Phase 31.

**Strategy Builder analyzer migration (2026-03-06):**
- [x] Migrated all 4 strategy builder analyzer functions from `generate_trades()` (batch pipeline) to `_unified_trades()` (unified engine): `analyze_entry_triggers()`, `analyze_exit_triggers()`, `find_best_exit_combinations()`, `analyze_risk_management()`
- [x] Each analyzer builds a synthetic strategy dict with the varied parameter and delegates to `_unified_trades()`, ensuring Entry/Exit/Stop Loss/Take Profit card KPIs match the top-level strategy KPIs exactly
- [x] Fixed `UnboundLocalError` for `general_cols` in non-webhook strategy builder path

##### Phase 30J: Stop-Validity Guard + Position Health Monitor — COMPLETED 2026-03-10

Live QA with Spy UTBot Extended Test revealed 4 phantom alerts (2 entry + 2 exit pairs) caused by L-type intra-bar entries where the swing stop was at or above the entry price, producing guaranteed-loss trades that the backtest never reproduces (different REST bar data means the signal doesn't fire at all). Also added position health tracking to catch stuck-position and sequence anomalies.

- [x] **Stop-validity guard** — Both `check_entry()` and `check_entry_intrabar()` in `PositionStateMachine` now reject entries where the computed stop is on the wrong side of the fill price (stop >= entry for LONG, stop <= entry for SHORT). Prevents the swing stop edge case where a gap-down L-type fill enters below the lookback window's lowest low, creating an instant stop. Guard applies to both backtest and live paths since Ralph imports `PositionStateMachine` from unified_engine. All 27 parity tests pass.

- [x] **`compute_position_health()` helper** — Extracted reusable function that walks a strategy's alert sequence chronologically, tracking position state transitions. Returns: current state (FLAT/IN_POSITION), entry details if open, hold time statistics, entry/exit balance check, and anomaly list. Used by both the Alert Analysis tab and the sidebar widget.

- [x] **Position Health expander (Alert Analysis tab)** — New "Position Health" section in `render_alert_analysis_tab()`, placed after Summary Metrics. Shows 4 metrics: Position Status (with OVERDUE flag if held > 2x expected max hold), Entries/Exits balance, Avg Hold Time, Expected Max Hold. Detects 3 anomaly types: double entry (entry while already in position), double exit (exit while flat), long hold (position held > 2x bar_count_exit * timeframe). Auto-expands when anomalies detected or position is open.

- [x] **Sidebar Position Monitor** — New "Position Monitor" section in the sidebar, visible on every page. Scans all alert-tracked strategies via `compute_position_health()`. Shows green "All clear" when healthy, blue info for open positions without anomalies, or red error badges per strategy with anomalies/mismatches/overdue positions. Each issue has a "View" button that navigates directly to the strategy detail view (sets `viewing_strategy_id`, `main_nav`, and `sub_nav_strategies`).

**Cross-strategy analysis findings (2026-03-10):**
- Spy UTBot Extended Test (id=32): 4 phantom alerts from 2 rapid stop→re-entry cycles caused by swing stop above entry price. Stop-validity guard eliminates this class.
- HL Campaign (id=34): 2 double-entry sequences in live_executions (reconciliation issue, not engine-level). 5 missed + 2 phantom alerts point to HM/HL type reconciliation as most drift-prone area.
- alerts.json across all 8 active strategies shows properly alternating entry/exit sequences — no stuck positions at the dispatch level.

**Alert Analysis & Pack Builder updates (2026-03-09):**
- [x] **Alert Analysis exec type fix** — Timing analysis was checking `source == 'intra_bar'` but ralph_engine sets `source: 'ralph'` on all alerts, causing ALL alerts to be treated as bar-close for timing delta calculations. Fixed: derives exec type from trigger ID via `get_trigger_exec_type()`, correctly classifying C vs L0/L1/HM/HL
- [x] **Alert Analysis new columns** — Added "Exec" (C/L0/L1/HM/HL) and "Trigger" columns to Trigger Timing table; added entry/exit trigger columns to Matched Trades table
- [x] **Alert matching trigger field** — `match_alerts_to_trades()` now includes `trigger` field in execution records for downstream analysis
- [x] **Pack Builder execution types** — Expanded `VALID_TRIGGER_EXECUTIONS` in pack_spec.py to accept `"hybrid_market"` and `"hybrid_limit"` alongside existing `"bar_close"` and `"intra_bar"`
- [x] **Pack Builder AI prompt** — Rewrote intra-bar section of `pack_builder_context.md` with full execution type documentation (C/L0/L1/HM/HL), manifest field requirements (`level_column`, `cross`), gate logic, and examples for all four execution types
- [x] **Pack Builder intra-bar registration** — New `_inject_intrabar_level_map()` in pack_registry.py auto-registers user pack triggers in `INTRABAR_LEVEL_MAP` and `_IB_L_TYPE_TRIGGERS` at install time, enabling tick-level detection in the live engine
- [x] **`_IB_L_TYPE_TRIGGERS` extensibility** — Changed from `frozenset` to `set` so user packs can register L-type triggers at runtime

> **Note:** Pack Builder and Alert Analysis updates should be re-validated after HM/HL live QA is complete.

**Execution type tags & chart enhancements (2026-03-09):**
- [x] **Execution type tags** — All 6 locations in app.py that displayed binary `[C]`/`[I]` badges now call `_execution_tag()`, correctly rendering `[C]`, `[L0]`, `[L1]`, `[HM]`, `[HL]` across strategy overview, "Current:" label, entry/exit analyzer results, and `find_best_exit_combinations()`
- [x] **`exec_type` in closed trade records** — `get_trade_record()` in unified_engine.py was missing `exec_type` field (only open-position records had it). Now all trade records include `exec_type` for downstream use by chart markers, Alert Analysis, and trade tables
- [x] **Exit price marker color coding** — Exit price markers (`+` and `x`) are now color-coded by exit reason: green (signal/target exit), red (stop loss), orange (HM/HL unconfirmed), teal (bar count exit). Exit arrows remain green/red for win/loss
- [x] **Chart legend tooltip** — Added `st.popover("Chart Legend")` above every price chart documenting all marker types: entry/exit arrows, exit price markers with color meanings, and alert price markers

##### Phase 30K: Strategy Builder Speed Optimization — COMPLETED 2026-03-11

The unified engine's bar-by-bar architecture caused a regression in Strategy Builder analysis speed — each analyzer call recomputed all indicators from scratch (O(N) per call vs old pipeline's O(1) lookup). With 5+ analyzer functions each running multiple backtests, total analysis time increased from seconds to minutes.

- [x] **Bar cache fast path** — `precompute_bar_cache()` runs the indicator/trigger pipeline once and stores per-bar state (indicators, triggers, confluence records) in a `CachedBarState` list. `run_trades_from_cache()` replays only `PositionStateMachine` logic against the cached state, skipping all indicator computation. ~70x speedup for analyzer functions.

- [x] **`_unified_trades()` cache integration** — Added `bar_cache` and `cache_metadata` optional params. When provided, takes the fast path via `run_trades_from_cache()` with fallback to full backtest on error.

- [x] **Analyzer cache threading** — All 4 analyzer functions (`analyze_entry_triggers`, `analyze_exit_triggers`, `find_best_exit_combinations`, `analyze_risk_management`) accept `bar_cache` and `cache_metadata` params, passing them through to `_unified_trades()`. Cache is built once after "Load Data" and stored in `builder_bar_cache` / `builder_cache_metadata` session state.

- [x] **Numpy boolean mask auto-search** — Rewrote `find_best_combinations()` to pre-compute per-record numpy boolean arrays, then AND masks for each combination instead of per-combination `pandas.apply(lambda)`. Also pre-filters records by `min_trades` frequency to prune the combination search space.

- [x] **Progress bars** — All 5 analyzer call sites (Entry, Exit Drill-Down, Exit Auto-Search, SL/TP, TF Conditions Auto-Search) replaced `st.spinner()` with `st.progress()` bars showing "Backtesting X of Y..." or "Evaluating combination X of Y..." with real iteration counts.

#### Migration Path

1. **Phase 30A** ✅ runs in parallel with existing system — backtest parity verified (8 tests)
2. **Phase 30B** ✅ adds HM/HL-type without affecting existing strategies — new strategies can opt in (7 tests)
3. **Phase 30C** ✅ refactored ralph_engine.py to import from unified_engine.py — 1400 lines deduped (6 live tests + 14 ralph fidelity tests)
4. **Phase 30D** ✅ trade generation converged to unified engine — batch pipeline retained for chart overlays only
5. **Phase 30E** ✅ MTF support + swing stops — all strategy types handled natively by unified engine (6 tests)
6. **Phase 30F** ✅ live chart migrated to unified engine + open-position entry markers
7. **Phase 30G** ✅ L0/L1 naming + _prev level bugfix — backtest fill prices now correct for V2 triggers
8. **Phase 30H** ✅ L-type gate timing fix — correct crossover semantics for UT Bot V2 and EMA V2 triggers
9. **Phase 30I** ✅ Live QA hardening — timestamp alignment, swing stops, same-bar guards, 1-bar cooldown, ATR parity, L-type bar_count alignment, restart rebase
10. **Phase 30J** ✅ Stop-validity guard + Position Health Monitor — prevents guaranteed-loss entries, adds position tracking to alert analysis and sidebar
11. **Phase 30K** ✅ Strategy Builder speed optimization — bar cache fast path + numpy auto-search + progress bars
12. **Phase 30L** ✅ Cross-TF confluence in live engine — shadow indicator engines, shared confluence buffer, label normalization, live chart table fixes
13. Existing C-type strategies migrate automatically (trigger classification is additive)
14. HM/HL-type is opt-in: existing UT Bot strategies remain L1-type unless user explicitly changes execution type

---

### Phase 31: Data Provider Migration — Polygon.io/Massive

**Goal:** Replace Alpaca's tick-level WebSocket feed with Polygon.io's pre-aggregated bar WebSocket, eliminating bar-building parity issues and unlocking sub-minute historical data for backtesting.

**Motivation (2026-03-06):** Live QA (Phase 30I) confirmed that building OHLCV bars from raw WebSocket ticks produces close prices that drift from official bar data. This causes visible slippage between alert markers (real-time WebSocket close) and backtest markers (REST API-reconciled close) on bar-close exits. The drift is inherent to tick-stream bar building and cannot be fully eliminated with filtering alone. Polygon.io streams pre-aggregated bars computed server-side with proper CTA/UTP rules, guaranteeing WebSocket/REST parity.

**Key Benefits:**
- **WebSocket/REST bar parity** — Pre-aggregated bars from the WebSocket match historical REST API bars exactly (same aggregation engine), eliminating the close-price drift observed with tick-built bars
- **Per-second bars** — WebSocket streams per-second and per-minute aggregated OHLCV, enabling sub-minute granularity for L-type trigger detection without raw tick processing
- **Sub-minute historical data** — REST API supports custom aggregate windows (e.g., 10-second, 30-second bars historically), enabling more precise backtesting of L-type strategies
- **Simplified BarBuilder** — No longer need to aggregate ticks into bars; receive complete OHLCV bars directly. Eliminates `EXCLUDED_TRADE_CONDITIONS` filtering, `_reconcile_bars()`, and associated drift
- **CTA/UTP compliance by construction** — Polygon applies the full 3-tier update rules (update all / update H/L+V only / update V only) server-side; current implementation uses binary include/exclude which misses partial-update conditions

**Architecture:**
- Alpaca remains the **broker** for order execution (no change to trading infrastructure)
- Polygon.io becomes the **market data provider** for both live WebSocket bars and historical REST API data
- `data_loader.py` — Replace Alpaca REST calls with Polygon REST calls for historical bars; add sub-minute timeframe support
- `ralph_engine.py` — Replace Alpaca WebSocket trade subscription + BarBuilder with Polygon WebSocket bar subscription; `on_bar_close()` receives completed bars directly
- `BarBuilder` — Simplified to buffer incoming bar events (no tick aggregation); gap-filling logic retained for missing bars
- Tick-level processing retained optionally for L-type intra-bar detection via Polygon's trade WebSocket (if per-second bars prove insufficient for trigger precision)

**Pricing:** Polygon.io Advanced plan ($199/mo) includes WebSocket streaming + unlimited REST API. Starter ($29/mo) may suffice for REST-only historical data during development.

**Implementation Phases (not yet started):**
- [ ] **31A:** Polygon REST integration in `data_loader.py` — historical bars with sub-minute support; parallel to existing Alpaca calls for validation
- [ ] **31B:** Polygon WebSocket integration in `ralph_engine.py` — replace Alpaca trade stream + BarBuilder with Polygon per-minute bar stream
- [ ] **31C:** Per-second bar support — subscribe to Polygon per-second bars for L-type intra-bar trigger detection; evaluate whether this replaces tick-level processing
- [ ] **31D:** Remove Alpaca market data dependencies — drop `alpaca-py` data subscriptions, `EXCLUDED_TRADE_CONDITIONS`, `_reconcile_bars()`, tick-based BarBuilder logic
- [ ] **31E:** Sub-minute backtesting — expose 10-second/30-second timeframes in Strategy Builder; update unified engine to handle sub-minute bar data
- [ ] **31F:** HiFi Backtest — sub-bar resolution via selective zoom (see below)
- [ ] **31G:** Confluence execution modes — `confirmed` vs `hifi` (see below)

#### Phase 31F–31G: HiFi Backtest (Sub-Bar Resolution)

**Goal:** Improve backtest fidelity by "zooming in" to 1-second candles on ambiguous bars — bars where the 1-minute OHLC doesn't tell the full story. This closes the three biggest gaps between backtest and live trading.

**Motivation:** The unified engine processes backtests bar-by-bar using only OHLC data. This creates fidelity gaps:

1. **Stop/Target Ambiguity** — When a bar hits both stop and target, the engine hardcodes "stop wins" (`check_exit()` checks stop before target). In reality, one hit first — we just can't tell from OHLC.
2. **L-Type Fill Imprecision** — L0/L1 entries fill at the indicator level (e.g., VWAP), but we don't know *when* in the bar it was crossed. The actual live fill could differ due to momentum.
3. **Same-Bar Suppression** — L-type entries suppress stop/target checks on the entry bar (`same_bar_ltype` guard) because we can't tell if the stop was hit before or after entry. Sub-bar data eliminates this guard.
4. **Cross-TF Confluence Timing** — The forward-shift fix delays secondary TF state by one full period. With 1-second data, we can know the *exact second* a higher-TF bar closes and use the state from that point.

**Approach: Selective Zoom (Two-Pass Backtest)**

Rather than backtesting everything at 1-second resolution (~2.1M bars for 90 days of 1-min data), use a targeted two-pass approach:

**Pass 1 (fast):** Run normal bar-by-bar backtest on the primary timeframe. Identify "ambiguous bars" where:
- An L-type entry fired
- A stop or target was hit
- Both stop AND target were reachable on the same bar (high >= target, low <= stop)
- An HM/HL entry was pending confirmation
- A cross-TF confluence condition changed state near a period boundary

**Pass 2 (targeted):** For ONLY the ambiguous bars (typically 60-100 bars across 30-50 trades):
- Fetch 1-second OHLCV data for that bar's time window from Polygon REST API
- Walk through 60 one-second bars sequentially
- For stop/target: first level hit wins (replaces hardcoded stop priority)
- For L-type entries: find exact second of level cross, record that price
- For same-bar stops: determine if stop was hit before or after entry second
- For cross-TF: determine exact second the higher-TF bar closed, use correct state

**Performance:** ~60-100 API calls for a typical 90-day backtest with 30-50 trades. Polygon rate limit (100 calls/min on Advanced plan) means this adds under a minute. 1-second data cached per symbol per day (~1MB/day/ticker) so subsequent backtests are instant.

**Confluence Execution Modes:**

Per-strategy setting controlling how cross-TF confluence is evaluated:
- **`confirmed`** (default, current behavior) — Uses previous closed bar of the secondary timeframe. Conservative, avoids look-ahead. Some traders prefer this because they only act on confirmed states.
- **`hifi`** — Uses the real-time state at the second of entry, determined by 1-second zoom data. More accurate simulation of what a live trader or alert engine would see at the moment of entry.

This is a strategy-level setting (not per-condition) to avoid confusion. Both modes remain available because they represent different trading philosophies, not just different precision levels.

**UI Changes:**

- **Strategy Builder:** "Confluence Mode" selector (Confirmed / HiFi) — only visible when Polygon integration is active
- **Strategy Builder:** "Enable HiFi Backtest" toggle — runs two-pass backtest with zoom resolution
- **Strategy Cards:** Small "HiFi" badge next to the BT duration when `hifi_mode: true`
- **Backtest Results:** "Zoom Events" count showing how many bars were resolved via zoom
- **Mass Builder:** HiFi as an iterable variable — test same strategy with and without to compare impact
- **Progress bar:** "Resolving 23 ambiguous bars..." during Pass 2

**Data Caching:**
- Cache 1-second data in local files: `zoom_{symbol}_{date}.pkl`
- One full trading day of 1-second data ≈ 1MB per ticker
- Shared across strategies for the same symbol — cache once, reuse everywhere
- Cloud (Railway): cache in Supabase storage or ephemeral disk; invalidation by date

**Strategy Config Fields:**
- `hifi_mode: bool` — enables two-pass backtest with zoom resolution
- `confluence_exec_mode: str` — `"confirmed"` (default) or `"hifi"`
- Neither field is in `OPTIMIZABLE_PARAMS` initially — changing them resets forward test since trade outcomes change

**What HiFi Does NOT Solve:**
- Slippage modeling (order book depth not captured in OHLCV)
- Market impact (backtest assumes infinite liquidity)
- Bid/ask spread (1-second bars are still mid-price OHLCV)
- These are acceptable — the goal is second-level resolution, not exchange microstructure simulation

**Safety Notes:**
- HiFi is backtest-only — the live Ralph engine already has tick-level fidelity
- Create a backup branch before implementation (touching unified_engine.py and data_loader.py)
- No changes to alert execution or position management — HiFi only affects backtest trade records
- Forward test and live tracking are unaffected

**Priority:** Medium-High. Depends on 31A (Polygon REST client). Should be designed alongside 31A-31E so the caching layer and API client account for 1-second bar requirements from the start.

---

### Phase 32: Crypto / Multi-Asset Support — COMPLETED 2026-03-13

*Add cryptocurrency as a first-class asset type alongside equities, enabling 24/7 strategy testing and monitoring via Alpaca's crypto market data APIs.*

**Motivation:** The user needed to battle-test the full trade execution pipeline (Strategy Builder → backtest → live monitoring → alert dispatch) over the weekend when equity markets are closed. Crypto trades 24/7 on Alpaca, providing a continuous testing environment. All Alpaca-supported crypto pairs are included (BTC/USD, ETH/USD, LTC/USD, AVAX/USD, SOL/USD, etc.), not just BTC.

**Detection heuristic:** Crypto vs equity is auto-detected throughout the codebase via `"/" in symbol` (e.g., "BTC/USD" → crypto, "SPY" → equity). This is consistent with Alpaca's symbol format.

**Changes by file:**

**`data_loader.py`:**
- [x] `is_crypto(symbol)` — detects crypto via `"/" in symbol`
- [x] `load_from_alpaca_crypto()` — uses `CryptoHistoricalDataClient` + `CryptoBarsRequest` (no `feed` param, no session filtering)
- [x] `load_market_data()` — routes crypto to `load_from_alpaca_crypto()`, sets source to "Alpaca Crypto"
- [x] `CRYPTO_BARS_PER_DAY` — 24h/day calculations (1440 for 1Min, 288 for 5Min, 24 for 1Hour, etc.)
- [x] `_bars_per_day()` / `estimate_bar_count()` / `days_from_bar_count()` — accept `asset_type` param; crypto uses 365/365 trading days (no weekday ratio)
- [x] `load_latest_bars()` — auto-detects crypto for bars-per-day calculation
- [x] `TRADING_SESSIONS` — added "24/7" entry

**`ralph_engine.py`:**
- [x] `StrategyMonitor.__init__` — forces `session = '24/7'` for crypto symbols (detected via `"/" in symbol`)
- [x] `_is_in_session()` — returns `True` unconditionally for `'24/7'` session
- [x] `_stream_data()` — splits symbols into stock vs crypto; runs dual WebSocket streams (`StockDataStream` + `CryptoDataStream`) concurrently via `asyncio.wait(return_when=FIRST_EXCEPTION)`. If either fails, all are cancelled and reconnected.
- [x] `_make_on_trade()` — async handler factory (CryptoDataStream requires coroutine handlers); skips `EXCLUDED_TRADE_CONDITIONS` filtering for crypto (no CTA/UTP condition codes)
- [x] `_crypto_stream_ref` — stored for clean shutdown; `stop()` closes both stock and crypto streams
- [x] Pickle file paths — sanitizes `"/"` to `"_"` for crypto symbols (`live_data_BTC_USD_60.pkl`)

**`worker.py`:**
- [x] Hot-reload stream reconnect — closes both stock and crypto streams via `loop.call_soon_threadsafe()` when new symbols are added

**`app.py`:**
- [x] Strategy Builder — "Asset Type" selectbox (Equity/Crypto) before symbol input; help text updates dynamically
- [x] Session selector — disabled "24/7" text input for crypto (replaces session dropdown)
- [x] Strategy config — saves `asset_type` field (lowercase: "equity" or "crypto"); defaults to "equity" for backwards compatibility
- [x] `AVAILABLE_CRYPTO_SYMBOLS` — reference list (BTC/USD, ETH/USD, LTC/USD, AVAX/USD, SOL/USD, DOGE/USD, LINK/USD, DOT/USD, UNI/USD, AAVE/USD)
- [x] Bar estimation — all `estimate_bar_count()` / `days_from_bar_count()` calls pass `asset_type` from strategy
- [x] Pickle read path — sanitizes `"/"` for crypto symbols

**Files NOT modified:**
- `unified_engine.py` — IncrementalIndicatorEngine, TriggerEvaluator, PositionStateMachine all work asset-agnostically on OHLCV DataFrames. Zero changes needed.
- `indicators.py`, `interpreters.py`, `triggers.py` — Pure math on price data, asset-agnostic.

**Runtime fix (2026-03-14):** `CryptoDataStream.subscribe_trades()` requires `async def` handler — sync handlers cause `"handler must be a coroutine function"` error and the crypto WebSocket never connects. Fixed by making `on_trade` handler async (works for both `StockDataStream` and `CryptoDataStream`).

---

### Post-Phase Fixes & Improvements (2026-03-13)

**Live chart MTF confluence (app.py):**
- [x] `render_live_chart_tab()` now resamples primary data to required secondary TFs, runs the same indicator/interpreter pipeline as `prepare_data_with_indicators()`, and passes `secondary_tf_map` to `run_unified_backtest()`. Previously, strategies with cross-TF confluence requirements could never generate trade markers on the live chart.

**KPI consistency (app.py):**
- [x] Strategies now save `backtest_start_date` / `backtest_end_date` from the actual DataFrame used in the Strategy Builder. The detail page (`render_live_backtest`) uses these pinned dates instead of a rolling `data_days` window, ensuring KPIs match the builder.
- [x] Date range caption added to both `render_live_backtest` ("Backtest range: ...") and `render_forward_test_view` ("Data range: ...") so users can see exactly what data window is being used.

**Trigger analyzer cross-pack fix (unified_engine.py, app.py):**
- [x] `precompute_bar_cache()` now stores `available_triggers` in metadata.
- [x] `run_trades_from_cache()` checks if the requested entry trigger exists in the cache. If not (cross-pack trigger), raises `ValueError` so `_unified_trades` falls back to full backtest. Previously, cross-pack triggers silently produced 0 trades because their trigger booleans weren't in the cache.
- [x] Analyzer results wrapped in scrollable container (`st.container(height=480)`) to prevent pushing trade list off-screen.

### Cross-TF Look-Ahead Bias Fix & Confluence Heatmap (2026-03-16)

**Cross-TF confluence look-ahead bias fix (app.py, prepare_dataframe):**
- [x] Secondary TF interpreter states are now **shifted forward by one period** before forward-filling to the primary TF index. Previously, a 15M bar's interpreter state was visible to 1M bars at the *start* of the 15M period (look-ahead). Now it only becomes visible after the 15M bar closes — matching Ralph engine live behaviour.
- [x] Speculative (unshifted) columns (`_spec_{INTERP}__{tf_label}`) stored alongside shifted columns for heatmap yellow detection.
- [x] `get_secondary_tf_map()` filters out `_spec_` columns so they don't affect trade gating.
- **Impact:** Backtest cross-TF confluence gating now matches live engine exactly. Strategies with cross-TF confluence conditions may show fewer trades in backtests (previously inflated by look-ahead).

**Confluence heatmap pane (app.py, build_confluence_heatmap_pane):**
- [x] New secondary chart pane showing per-bar confluence state as a color-coded heatmap below the price chart.
- [x] Each row = one confluence condition (same-TF, cross-TF, or general pack). Colors: green (met/confirmed), red (not met), yellow (speculative — met on forming bar but not on previous closed bar, cross-TF last bar only).
- [x] Stacked Histogram series per condition with thin separator lines between rows.
- [x] Floating condition labels in right margin — invisible Line series with `lastValueVisible=true` and `title` set to condition name + needed state (e.g., "EMA_STACK (15M): BULL STACK").
- [x] `_render_heatmap_legend()` — compact caption below chart mapping row numbers to condition names.
- [x] Wired into live chart (`last_bar_partial=True` for yellow), backtest detail, and forward test chart views.
- **Design note:** Heatmap colors reflect each bar's *own* closed interpreter state. Entries during a bar are gated by the *previous* bar's confirmed state, so an entry can legitimately appear on a red bar if the prior bar was green.

---

### Strategy Builder Performance Optimization — COMPLETED 2026-03-16

**Problem:** The Strategy Builder reran expensive operations (`prepare_data_with_indicators()` + `_unified_trades()`) on every UI interaction — filter changes, confluence add/remove, auto-search toggle, etc. — causing a multi-second "Loading market data" spinner on every click.

**Solution:**
- [x] `_builder_config_hash()` computes a hash from data-affecting params (symbol, TF, dates, session, triggers, stops, targets, direction). Display-only keys (name, risk_per_trade, starting_balance, selected_confluences) are excluded.
- [x] `df`, `trades`, `bar_cache` cached in `session_state` keyed by hash. On rerun, if hash matches cached data → skip expensive operations entirely.
- [x] Smart Load Data button: `Load Data` (first load) → `✓ Loaded` (disabled, cache current) → `Reload` (primary, params changed). Visual stale-data warning when params change but user hasn't reloaded.
- [x] Display-only actions (confluence add/remove, filter apply, auto-search toggle) now use cached data instantly — no spinner, no delay.
- [x] Cache cleared on strategy edit load and new strategy creation to ensure fresh data.

### Phase 33: Mass Strategy Builder — Bulk Strategy Discovery & Optimization

**Goal:** Allow users to test thousands of strategy combinations across multiple tickers, timeframes, triggers, confluences, and risk profiles in a single run. Surface the best-performing strategies as saveable cards, enabling rapid portfolio construction from the cream of the crop.

**Two new pages:**
1. **Mass Strategy Builder** — configure search parameters, run analysis, monitor progress
2. **Mass Strategy Results** — saved search history, revisit past runs, review/save/pass on results

**UI Layout (Mass Strategy Builder page):**

| Section | Description |
|---------|-------------|
| **1. Header Row** | Search name (text input) + Save button |
| **2. Config Row** | `Select Tickers` · `Select Variables` · `Select Required Performance` · **`Analyze`** (red action button). Each opens a modal/popover. |
| **3. Preview + Progress** | Combination count preview, estimated time, progress bar during analysis |
| **4. Post-Analysis Filters** | Sort by + KPI filter row (win rate, PF, daily R, R², trades, etc.) |
| **5. Result Cards** | Strategy cards with KPIs, equity curve sparkline, config summary. Buttons: `Save to My Strategies` (marks card as saved) · `Pass` (grays out, reversible) |

**Select Variables modal tabs:**
- **Date Range** — lookback days / date range picker
- **Timeframes** — checkboxes for each available TF (1Min, 5Min, 15Min, etc.)
- **Direction** — Long, Short, or both
- **Entry Triggers** — checkboxes from enabled confluence packs (with execution type tags `[C]`/`[L0]`/`[HM]` etc.)
- **Exit Triggers** — checkboxes (all triggers, same as Strategy Builder) + depth selector (best 1/2/3/4 combined). Bar count exits automatically separated from signal exits.
- **TF Confluences** — checkboxes from enabled confluence packs + depth selector (max factors 1–4)
- **General Confluences** — checkboxes from enabled general packs
- **Stop Loss** — Strategy Builder-style config: method dropdown (ATR/Fixed$/Pct%/Swing), params, trailing stop, breakeven stop
- **Take Profit** — Strategy Builder-style config: method dropdown (None/R:R/ATR/Fixed$/Pct%/Swing), params

**Required Performance popover:**
- **Prioritize by** — metric the engine sorts final results by (Daily R, Win Rate, PF, R², Avg R, Total R)
- **Max results** — cap on total results returned (default 500)
- Min thresholds: trades, win rate, profit factor, daily R, R²

**Combination engine:**
- Groups by (symbol, timeframe, session) to minimize `prepare_data_with_indicators()` calls (one per group, `@st.cache_data`)
- Within each group: `run_unified_backtest()` per (direction × entry × exit_combo) with single stop/target config
- Bar count exits properly separated from signal exits (mirrors Strategy Builder logic)
- For each base config that meets minimum trade count: auto-search TF + General confluences via `find_best_combinations()` up to selected depth
- Confluence results per base config scale with max_results (20–50 per base, capped)
- Pre-filter at each level by Required Performance thresholds
- Final sort by user-chosen priority metric, trim to max_results
- Diagnostic counters track: data loads/failures, backtests run/failed, zero-trade combos, below-min combos, direction skips

**Execution model:**
- Synchronous with live progress bar (runs in Streamlit process)
- Results auto-saved on completion to Supabase (`mass_searches` table) or local JSON
- Diagnostic summary displayed after completion

**Result cards include:**
- KPIs: trades, win rate, PF, avg R, total R, daily R, R², max R drawdown
- Plotly equity curve sparkline (cumulative R series, green/red fill)
- Config summary: ticker, TF, direction, entry trigger, exit triggers, confluences, stop/target
- `Save to My Strategies` button → creates a full strategy in My Strategies (identical to Strategy Builder output)
- `Pass` toggle → grays out card (reversible visual indicator)
- Post-analysis filters: sort by 6 KPI options, min thresholds, show/hide passed

**Mass Strategy Results page:**
- Flat card layout: search name, result count, date
- Load / Copy / Delete buttons per search
- Copy duplicates search config (without results) for iterative refinement

**Save to My Strategies:**
- Creates identical strategy format to Strategy Builder save (same config keys, same `save_strategy()` path)
- Includes `kpis`, `stored_kpis`, `backtest_start_date`, `backtest_end_date`, `forward_test_start`
- Saved strategies work in all existing flows: detail page, portfolios, alerts, live chart, forward testing

**Strategy Default data view (My Strategies page):**
- New default data view option: "Strategy Default"
- Filters trades to `entry_time >= backtest_start_date` (backtest window + forward test trades)
- Equity curves and KPIs reflect the strategy's intended backtest window, not all available data
- Legacy strategies without pinned dates gracefully fall back to "All Data"

**Date range consistency fixes:**
- Forward test view now trims warmup trades (before `backtest_start_date`) from backtest results
- Data range display shows pinned backtest start, not raw `df.index[0]` (which includes warmup)
- Trading days count excludes warmup period
- Mass Builder result cards show pinned date range
- Mass Builder preview shows configured date range before analysis runs

**Implementation status (2026-03-17):**

| Sub-Phase | Status |
|-----------|--------|
| **33A** | COMPLETE — UI skeleton, data model, Select Variables modals (9 tabs), Supabase migration |
| **33B** | COMPLETE — Combination engine, `run_unified_backtest()` per combo, confluence auto-search, bar_count exit separation, diagnostic counters |
| **33C** | COMPLETE (simplified) — Synchronous execution with live progress bar. Background threading deferred (JWT context issues). |
| **33D** | COMPLETE — Result cards with equity curves, Save to My Strategies, Pass toggle, post-analysis filters, search history |

**Post-implementation fixes (2026-03-17):**
- [x] Exit triggers: use `get_all_triggers()` (same as Strategy Builder), not `get_exit_triggers()` (EXIT-only subset)
- [x] Execution type tags (`[C]`/`[L0]`/`[HM]` etc.) on entry and exit trigger checkboxes
- [x] Bar count exits separated from signal exits (mirrors Strategy Builder logic)
- [x] JWT context propagated to background thread for Supabase user data access
- [x] KPIs saved as both `kpis` and `stored_kpis` (My Strategies cards read `kpis`)
- [x] Null-safe `(s.get('kpis') or {})` pattern across dashboard, My Strategies, Mass Builder
- [x] `backtest_start_date`/`backtest_end_date` pinned from `df.index[0]`/`df.index[-1]` at search time
- [x] Forward test view warmup trade trimming + correct trading days count
- [x] Mass search persistence via Supabase `mass_searches` table + JSON fallback
- [x] Session selector added to Mass Builder (RTH/Pre-Market/After Hours/Extended Hours, auto-24/7 for crypto)
- [x] Strategy hydration: Save to My Strategies now runs backtest to generate `stored_trades` + `equity_curve_data`, ensuring full compatibility with portfolios, compliance checks, and card rendering
- [x] Strategy Default data view: new default on My Strategies page, filters to backtest window + forward test trades
- [x] Trading days count excludes warmup period (was showing 124d instead of ~63d for 90-day backtest)

**Known limitations / future work:**
- Bar_cache optimization deferred — each combo runs full `run_unified_backtest()` (~1-2s each). Per-trigger-set cache could make this 10-50x faster.
- Background threading needs JWT propagation fix for Supabase user context. Currently runs synchronously (blocks page during analysis).
- Stop/Target not iterable yet — single config per search. Future: RM pack-style variations for bulk stop/target testing.
- **Indicator warmup in backtests** — see Known Issues below.

**Full implementation spec:** `docs/Mass_Strategy_Builder_Spec.md`

---

### Phase 34: Stop Loss & Take Profit Packs — PLANNED

**Goal:** Restructure stop loss and take profit configuration into pack-based templates (like TF Confluence packs), enabling variation creation and bulk iteration in the Mass Strategy Builder.

**Motivation:** Currently stop/target configs are set directly in the Strategy Builder with no way to save reusable variations. Users must manually adjust parameters each time. The Mass Builder can only use one stop/target config per search — it cannot iterate over variations. By treating stop/target as packs with templates and variations, users can create "Swing (Default)", "Swing (Tight)", "ATR 1.5x", "ATR 2.0x" etc. and iterate over them like any other variable.

**Two new Confluence Pack pages:**
1. **Stop Loss Packs** — templates: ATR, Fixed $, Percentage, Swing. Each template has default parameters + user-created variations. Optional trailing stop and breakeven stop settings per pack.
2. **Take Profit Packs** — templates: None, R:R, ATR, Fixed $, Percentage, Swing. Each template has default parameters + user-created variations.

**Migration plan (safe, non-breaking):**
1. Create new `stop_loss_packs` and `take_profit_packs` data structures (following ConfluenceGroup pattern)
2. Create default packs from current Risk Management pack templates
3. Add new pages to Confluence Packs navigation
4. Update Strategy Builder and Mass Builder to reference packs (with fallback to raw `stop_config`/`target_config` for backwards compatibility)
5. Mass Builder: add Stop Loss and Take Profit as iterable variables (checkboxes, like entry/exit triggers)
6. Migrate existing strategies: add `stop_pack_id` / `target_pack_id` fields alongside existing `stop_config` / `target_config` (both formats supported)
7. Once validated: deprecate Risk Management packs page

**Key constraint:** Existing strategies must continue to work unchanged. The unified engine reads `stop_config` / `target_config` dicts — packs are a UI/config layer that produces those same dicts. No engine changes needed.

**Priority:** Medium. Not blocking live trading. Current strategies work fine with direct configs.

---

### Phase 35: Capital Efficiency Metric (ROI) — PLANNED

**Goal:** Add a KPI that measures return relative to capital deployed, helping users compare strategies across different price points. Two strategies with identical R-multiple performance differ in capital efficiency if one trades a $500 stock vs a $20 stock.

**Motivation:** R-multiples measure edge (return per unit of risk) but don't account for how much buying power is consumed. A user with a $25K prop firm account gets more portfolio diversification from strategies on $20 stocks than $500 stocks, even with identical R performance.

**Proposed metrics:**
- **Daily ROI %** = `(daily_r × risk_per_trade) / avg_position_cost × 100` — daily return as percentage of capital tied up per trade
- **R per $1K deployed** = `daily_r / (avg_entry_price × shares_per_trade / 1000)` — normalizes edge by capital required
- **Capital efficiency ratio** = `total_r / (avg_entry_price × 100)` — R generated per $100 of stock price

**Display locations:**
- Strategy Builder KPI row (new metric)
- My Strategies cards (new column)
- Mass Builder result cards and sort options
- Portfolio analysis (aggregate across strategies)

**Data requirements:** `avg_entry_price` from trades DataFrame (already available). `risk_per_trade` from strategy config (already available).

**Key constraint:** Pure display/KPI addition. No changes to engine, triggers, or alerts. Zero risk to execution reliability.

**Priority:** Low. Nice-to-have for portfolio construction optimization. Not blocking live trading.

---

### Phase 36: Strategy Tags & Bulk Actions — COMPLETED 2026-03-18

**Goal:** Add tagging, multi-select, and bulk operations to My Strategies to support high-volume strategy management workflows (e.g., mass-testing dozens of strategies and organizing them into portfolios).

**Motivation:** When mass-testing many ticker/confluence combinations, users accumulate dozens of strategies quickly. Without tags or bulk actions, organizing them into portfolios requires adding strategies one by one, and cleaning up rejects requires deleting one at a time.

**Features implemented:**

**Strategy Tags:**
- Small inline pill-shaped tags rendered below the strategy name on each card
- Tags managed via a "Tags" popover button on each card (add/remove)
- Tags stored as `tags: list[str]` in strategy config JSONB (no DB migration needed)
- Tag filter added to the My Strategies filter bar (Ticker | Direction | Tag | Data View | Sort)
- Tags are lowercase, deduplicated, and persist across sessions

**Multi-Select Mode:**
- "Select" button in the top-right toggles selection mode on/off
- Checkboxes appear on each strategy card in select mode
- Selection state managed natively by Streamlit widget keys (no manual set tracking)
- "Cancel" button exits select mode and clears all checkbox states

**Bulk Actions (shown when strategies are selected):**
- **Delete Selected** — confirmation dialog listing strategy names, then bulk delete with cache cleanup
- **Create Portfolio** — pre-loads selected strategies into the portfolio builder with their `risk_per_trade` values, navigates to Portfolios page

**Performance fix (critical):**
- `get_cached_strategy_trades()` now checks `stored_trades` before falling back to full backtest
- Previously, every portfolio page navigation triggered `prepare_data_with_indicators()` for each strategy (minutes per strategy on 90-day 1-min data)
- All portfolio paths (builder, save, list, detail, compliance) go through this single function — fixing it once fixed everything
- Portfolio save now shows a progress bar: "Computing trades for [name]..." → "Computing portfolio metrics..." → "Saving..."
- Portfolio builder shows a loading indicator during initial strategy data load

**Post-implementation fixes:**
- [x] Checkbox selection: replaced manual set + `st.rerun()` with Streamlit-native widget state derivation
- [x] Portfolio save: switched from `get_strategy_trades` (full backtest) to `get_cached_strategy_trades` (stored_trades fast path)
- [x] Portfolio list: fixed migration fallback and compliance check to use cached trades
- [x] Root cause fix: `get_cached_strategy_trades` now checks `stored_trades` first (instant) before falling back to `get_strategy_trades` (full backtest)

---

### Phase 37: Portfolio Live Dashboard & Active Management — COMPLETED 2026-03-20

**Goal:** Transform the portfolio detail page from a backtest-only view into an active portfolio management dashboard. The backtest becomes the benchmark/plan; alert-based trades become reality. Answer: "Is my portfolio on track?"

**Motivation:** With strategies live and firing alerts via Ralph engine + webhooks, the portfolio page needs to reflect real performance — not just historical backtests. Users need to see actual trades, compare to expectations, track buying power, and catch anomalies early.

**Sub-phases:**

##### Phase 37A: Portfolio Alert Trade Aggregation (Data Foundation)
- [ ] `get_portfolio_alert_trades()` — aggregate `live_executions` across all portfolio strategies into unified trade dataset
- [ ] Each alert trade: strategy context, entry/exit prices (alert + theoretical), slippage, R-multiple, dollar P&L, quantity, buying power used, matched/phantom flag
- [ ] `compute_strategy_r_distribution()` — extract backtest R distribution (avg, std, variance) per strategy
- [ ] `get_portfolio_strategy_alerts_bulk()` — single-pass alert loading for portfolio strategies

##### Phase 37B: Benchmark & Confidence Bands Engine
- [ ] `compute_portfolio_benchmark()` — "Plan" line + confidence bands from backtest R distributions
- [ ] Trade-based X-axis (not calendar): cumulative expected R stepping up per trade
- [ ] Confidence bands: 1SD (68%) and 2SD (95%), widening with sqrt(N)
- [ ] Strategy filter support: "All Strategies" or individual strategy view
- [ ] `classify_strategy_health()` — on_track / outperforming / underperforming / insufficient_data / correlated

##### Phase 37C: Live Dashboard Tab — Core UI
- [ ] New "Live Dashboard" tab (first tab on portfolio detail page)
- [ ] Strategy filter dropdown (All Strategies + each individual)
- [ ] Alert-based equity curve with plan line + confidence bands (Plotly)
- [ ] KPI row: Total Alert Trades, Win Rate, Total P&L, vs Plan Delta
- [ ] Trade history table: strategy, symbol, direction, entry/exit prices, exit reason, quantity, BP used, R, P&L, matched/phantom
- [ ] "View Chart" button per trade → `@st.dialog` modal with price chart zoomed to trade window
- [ ] Open positions summary (strategies currently IN_POSITION)

##### Phase 37D: Buying Power Tracker & Anomaly Detection
- [ ] `compute_alert_buying_power()` — intra-trade buying power timeline from alert trades + account balance
- [ ] Buying power chart (area chart, reference line at 0)
- [ ] Insufficient capital event detection and warning banners
- [ ] `detect_portfolio_anomalies()` — overexposure (same symbol), phantom trades, buying power exceeded, long holds
- [ ] Anomaly cards with severity badges, "Cover" button placeholder
- [ ] `buying_power_mode` portfolio setting: scale_down / force_one / reject

##### Phase 37E: Account Tab Enhancements — Change Log & Journal
- [ ] `add_change_log_entry()` — auto-record strategy added/removed, risk adjusted, requirement set changed
- [ ] Change log instrumented in portfolio save/edit flow
- [ ] "Change History" expander in Account tab with type badges
- [ ] `compute_daily_journal()` — auto-generated daily P&L + trade count + changes
- [ ] "Daily Journal" expander with date navigator and editable notes per date
- [ ] Change log entries appear as informational ledger rows

##### Phase 37F: Strategies Tab Enhancements & Cover Webhook
- [ ] Three-line equity curves per strategy: BT (gray dashed) + FW (blue) + Alert (green)
- [ ] Health badge per strategy from `classify_strategy_health()` with recommendation text
- [ ] "View Strategy" button → navigate to strategy detail page
- [ ] "View Chart" button → modal with current price chart
- [ ] `send_cover_webhook()` — one-click close for excess positions
- [ ] Cover webhook settings in Webhooks tab (URL, template, enable toggle)

##### Phase 37G: Alert Tracking Liberalization (Optional)
- [ ] Strategies with `alert_tracking_enabled=True` record alerts regardless of portfolio membership
- [ ] Portfolio-linked strategies get processing priority in Ralph engine
- [ ] Non-portfolio alerts: skip webhook delivery but still record and update `live_executions`

**Schema changes (portfolio dict, backward-compatible):**
- `change_log: list` — audit trail of portfolio changes
- `buying_power_mode: str` — 'scale_down' | 'force_one' | 'reject'
- `journal_entries: dict` — date-keyed editable journal with auto-summaries

**Key design decisions:**
- **Data source = alerts** — no broker API; alert executions are the source of truth
- **Trade-based X-axis** — benchmark uses trade number, not calendar date; normalizes for trade frequency differences
- **Dynamic benchmark** — recomputes as portfolio composition changes
- **Confidence bands** — cumulative variance model, bands widen as sqrt(N trades)
- **Account balance = buying power authority** — position sizing uses Account tab's current balance

**Build order:** A → B → C → E → D → F → G

**Phase 37 Implementation Status (as of 2026-03-19):**

Phases 37A-37F deployed to production. Multiple QA rounds completed. Core features working:
- [x] Live Dashboard tab with Performance vs Plan chart (confidence bands, plan line, actual line)
- [x] Strategy filter dropdown (All + individual)
- [x] KPI row (Alert Trades, Win Rate, Total P&L, Expected P&L, vs Plan)
- [x] Trade history table with headers, scrollable container
- [x] Trade detail modal (simple Plotly chart — TradingView-style upgrade deferred)
- [x] Open Positions section (with empty state)
- [x] Buying Power Tracker expander
- [x] Anomaly Detection section (with empty state)
- [x] Strategy health badges + recommendations on Strategies tab
- [x] View Strategy / View Chart buttons on Strategies tab
- [x] Equity curves on Strategies tab using standard BT/FW/Alert three-segment style
- [x] Change History section in Account tab (auto-logged portfolio changes)
- [x] Daily ledger aggregation (daily rows with Details modal: Trades + Changes + Notes tabs)
- [x] Auto-enable alert_tracking_enabled on portfolio save
- [x] "Update Strategies" button on portfolio detail page
- [x] Raw alert fallback for strategies without live_executions
- [x] Auto-refresh fragment for open positions / anomalies (10s interval)

**Resolved bugs:**
- [x] Custom notes persisting — fixed via session-state handoff pattern (`@st.dialog` can't reliably write to DB)
- [x] Summary column "has notes" indicator — works after notes persistence fix
- [x] Renamed "Note" column to "Summary" in ledger
- [x] Full Rebacktest button added (clears stored_trades, forces cold-start rebuild)

**Known limitations (accepted):**
- Older alert trades (before Phase 37 deployment) show qty=1 and "signal" as exit reason because raw alert fallback doesn't have stop_price/exit_reason. New trades going forward have proper data via expanded `_extract_minimal_trades()`.

**Nice-to-haves for future:**
- [ ] Redirect to portfolio detail page after edit/save instead of portfolio list
- [ ] Equity curve x-axis selector (by day vs by trade number) — dropdown wherever equity curves exist
- [ ] Trade detail modal: use full TradingView-style chart (render_price_chart) with indicators, confluence heatmap, +/x markers, entry/exit trigger labels — reuse strategy live chart rendering
- [ ] Webhook system rework: template-based per-account webhooks with event-type-specific payloads (C, L0/L1, HM/HL), cover/close via existing exit webhooks with quantity-specific orders
- [ ] Cover erroneous positions: use strategy's existing long-exit/short-exit webhook with specific quantity rather than a separate cover webhook; auto-detect position direction and fire appropriate exit event
- [ ] Performance vs Plan P&L: default to planned quantity, add dropdown to switch between planned/executed quantity view
- [ ] QA open positions and anomaly sections with live data when positions are active during market hours

---

## Known Issues To Address

### Indicator Warmup in Backtests

**Problem:** The unified engine's `IncrementalIndicatorEngine` seeds EMAs with the first bar's close price and converges incrementally. This means the first ~200 bars of any backtest have inaccurate indicator values (EMAs haven't converged, MACD is unreliable, etc.). Triggers that fire during this warmup period may produce trades based on incorrect indicator states.

**Impact by configuration:**

| Config | Bars loaded | Warmup bars (~200) as % | Risk |
|--------|------------|-------------------------|------|
| 90 days, 1Min | ~23,000 | < 1% | Low — negligible impact |
| 90 days, 5Min | ~4,700 | ~4% | Low |
| 90 days, 15Min | ~1,560 | ~13% | Moderate |
| 90 days, 1Hour | ~195 | ~100% | **High — EMAs never converge** |
| 30 days, 15Min | ~520 | ~38% | **High** |
| 30 days, 1Hour | ~65 | **300% — insufficient data** | **Critical** |

**Current mitigations:**
- Forward test view loads `data_days * 2` for warmup before the backtest window
- Ralph (live engine) calls `warmup(df)` with historical data before processing live ticks
- Strategy Builder loads the exact requested range — no extra warmup

**Proposed fix:** Load extra warmup bars before the backtest window in all contexts (Strategy Builder, Mass Builder). The warmup bars are used only for indicator convergence — trades generated during warmup are excluded from backtest results. The number of warmup bars should be at least `max(EMA_long_period, MACD_slow_period + MACD_signal_period)` — typically ~200 bars for standard settings.

**Priority:** Medium. Low risk for typical 1Min/5Min configurations with 90+ days. Becomes significant for coarser timeframes (15Min+) or shorter lookback windows (30 days). Should be addressed before users build strategies on 1Hour or 4Hour timeframes.

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
