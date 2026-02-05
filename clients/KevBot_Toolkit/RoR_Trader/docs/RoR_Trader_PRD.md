# RoR Trader - Product Requirements Document (PRD)

**Version:** 0.2
**Date:** February 5, 2026
**Author:** Kevin Johnson
**Status:** MVP Built — Active Development

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

---

## 14. Development Roadmap

### Phase 1: Harden the Core Engine
*Fix bugs and remove fragility in the foundation all future features build on.*

- [ ] Handle infinity gracefully in confluence analysis calculations (profit factor deltas, sorting) — display of infinity when no losses is correct and intentional
- [ ] Add null guard on confluence record filtering (crash risk when trades have no confluence data)
- [ ] Complete opposite trigger mapping in triggers.py (incomplete mapping causes positions that never close)
- [ ] Replace hardcoded mappings (INTERPRETER_TO_TEMPLATE, base_trigger_map) with runtime-built maps from confluence groups
- [ ] Add risk/stop-loss parameter configuration to Strategy Builder Step 1 (currently hardcoded, not saved with strategy)
- [ ] Fix chart timestamp handling (fragile assumption about DataFrame column order)
- [ ] Save complete strategy parameters on save (risk_per_trade, stop_atr_mult, timeframe usage)

### Phase 2: Complete My Strategies
*The weakest existing page — P0 per this PRD but currently a display-only stub.*

- [ ] Strategy detail view (full KPIs, equity curve, trade history table)
- [ ] Edit strategy (reopen in Strategy Builder with saved configuration)
- [ ] Delete strategy (with confirmation dialog)
- [ ] Clone/duplicate strategy
- [ ] Re-backtest with fresh data
- [ ] Sorting and filtering (by ticker, direction, performance metrics)

### Phase 3: Dashboard
*Landing page that ties the application together.*

- [ ] Overview cards (strategy count, best performer, recent activity)
- [ ] Quick actions (New Strategy, View Strategies)
- [ ] Mini equity curves for saved strategies
- [ ] Data source status and connection health
- [ ] Empty state for new users

### Phase 4: Forward Testing
*Key differentiator — what separates RoR Trader from backtest-only tools.*

- [ ] Track strategy performance on new data after save date
- [ ] Backtest vs. forward test comparison visualization
- [ ] Immutable forward test history (builds trust and credibility)
- [ ] Status indicators on strategy cards (backtested, forward testing, validated)

### Phase 5: Portfolios & Prop Firm Compliance
*Combine strategies and validate against real trading account rules.*

- [ ] Portfolio builder (select strategies, set allocations)
- [ ] Combined equity curve and drawdown analysis
- [ ] Correlation matrix between strategies
- [ ] Prop firm rule sets (Trade The Pool, FTMO, etc.)
- [ ] Compliance checker with pass/fail indicators and recommendations

### Phase 6: Alerts & Deployment
*Make strategies actionable in real time.*

- [ ] Webhook configuration UI
- [ ] Real-time signal detection from forward-tested strategies
- [ ] Alert history and management
- [ ] Trading bot connection framework

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

### A.2 VWAP Interpreter
```
Inputs: Price, VWAP, VWAP StdDev Bands
Outputs:
  - ABOVE_UPPER_BAND
  - ABOVE_VWAP
  - AT_VWAP
  - BELOW_VWAP
  - BELOW_LOWER_BAND
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
