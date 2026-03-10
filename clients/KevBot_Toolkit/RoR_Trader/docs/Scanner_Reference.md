# Stock Scanner — Reference & Planning Document

**Date:** March 9, 2026
**Status:** Research & Requirements Gathering
**Dependency:** Phase 31 (Polygon.io migration) — scanner data feeds will leverage Polygon's broader market coverage

---

## 1. Overview

A stock scanner that identifies tradeable opportunities across the market in real-time, serving as the foundation for a third strategy method type:

| Strategy Method | Description |
|---|---|
| **Ticker-Based** | Fixed ticker, standard entry/exit triggers (current) |
| **Webhook-Based** | Inbound webhook signals drive entries (current) |
| **Scanner-Based** | Pre-built scanner filters dynamically select tickers that meet criteria, then evaluate entry/exit triggers on qualifying stocks |

The scanner method enables strategies that aren't tied to a single ticker — e.g., trading high-volume breakouts on low-float stocks, gap plays, momentum runners, etc.

---

## 2. Reference Software: Scanz

Current tool used for intraday stock scanning. Subscription expires 2026-03-10.

### 2.1 What Works Well

- **Flexible filter builder** — Rules are composed piece by piece with metric + operator + value/metric. Extremely granular: you can compare any metric against a static value OR against another metric with temporal offsets (N candles ago, avg of N candles, N days ago, avg of N days).
- **Real-time updates** — Results table updates by tick. Tickers enter/exit the watchlist dynamically as they meet or fail criteria.
- **Activity log** — Tracks when tickers ENTER or EXIT the qualifying list with timestamps, giving visibility into scanner dynamics.
- **Session awareness** — Every metric can be scoped to a session: Pre-Market, Regular Hours, After Hours, or Full Day. Column headers reflect the active session (e.g., "% Chg AH", "Vol AH").
- **"Add as column to scan results"** — When building a filter referencing a metric (e.g., yesterday's close), you can checkbox to add that computed value as a column in the results table. Excellent for at-a-glance context.
- **Contextual second-filter narrowing** — When building a metric-vs-metric comparison (e.g., "Last Price >= [second metric]"), only compatible categories are shown for the second metric (e.g., only Price and Technical, not Financials or News). Reduces invalid configurations.
- **Metric-vs-metric relative comparison** — Can compare two metrics with a "by" modifier: "Last Price >= Close by more than 5%". The "by" dropdown offers Value, Percent, or Times (multiplier).
- **Toggleable rules** — Each filter rule has an on/off toggle, so you can disable rules without deleting them. Useful for A/B testing scanner configurations.
- **Draggable rule ordering** — Rules can be reordered via drag handles.
- **Saved scanner presets** — Left sidebar shows saved scanners organized into categories. Pre-built "Stocks in Play" scanners for each session.
- **Ticker detail pane** — Clicking a ticker opens a side panel with price chart, core info, and tabbed sections for News, Prints (Time & Sales), and L2 (Level 2 depth).
- **News integration** — Ticker detail pane shows timestamped news headlines, which is actionable for catalyst-driven trading.

### 2.2 What's Missing / Could Be Better

- No backtesting integration — Scanz finds stocks but can't evaluate strategy performance on them
- No automated entry/exit — purely a discovery tool, requires manual trading or separate tools
- No portfolio-level awareness — doesn't know about existing positions, risk limits, or prop firm rules
- Limited indicator customization — fixed set of technical indicators, can't add custom packs
- No confluence concept — can't combine scanner results with our interpreter/trigger system

### 2.3 Reference Images

Place screenshots in `docs/reference_images/` with prefix `scanz_`.

| Image | Description |
|---|---|
| `scanz_default scanner view.png` | Full app layout: left sidebar with saved scanners, main area with results table, session tabs |
| `scanz_ scanner view with filters shown and activity log shown.png` | Three-section layout: filter rules (top), results table (middle), activity log (bottom) |
| `scanz_select filter button experience.png` | "Select Filter" dropdown showing 8 top-level metric categories |
| `scanz_filter price options.png` | Price category expanded: Last Price, Open, High, Low, Close, Bid, % On The Bid, Ask, % On The Ask, Spread, % Spread, Range, % Range |
| `scanz_filter change options.png` | Change category: Net Change, Percent Change |
| `scanz_filter liquidity options.png` | Liquidity category: Volume, Trades, Dollar Volume, Relative Volume, Relative Dollar Volume, Relative Trades |
| `scanz_filter technical options.png` | Technical category: SMA, EMA, Triple EMA, Bollinger Bands, ADX, ATR, CCI, MACD, MFI, ROC, RSI, Parabolic SAR, Stochastics, VWAP |
| `scanz_filter capital structure options.png` | Capital Structure: Market Cap, Shares Outstanding, Float, Percent Float, Float Rotation, Percent Insiders, Percent Institutions |
| `scanz_filter short interest options.png` | Short Interest: Short Shares, Short Ratio, Short Shares (Prior Month), Short % Change MoM, Days To Cover |
| `scanz_filters financials options.png` | Financials: Revenue, Gross Profit, EBITDA, Cash, Debt |
| `scanz_filter news options.png` | News: News Count |
| `scanz_second filter options.png` | Second filter context: only Price and Technical categories shown (contextual narrowing) |
| `scanz_ second filter options price cat.png` | Second filter Price submenu: same price metrics available for comparison |
| `scanz_filters filter vs filter calculation.png` | Metric-vs-metric with "by more than" modifier + Value/Percent/Times dropdown |
| `scanz_second filter close details form with daily data interval.png` | Close metric config: Data Interval=Daily, Session=Regular Hours, Offset=Current day, "Add as column" checkbox |
| `scanz_second filter close details with intraday interval.png` | Close metric config: Data Interval=1 minute, "Include Extended Hours" checkbox, Offset options: Current candle / N candles ago / Avg. of N candles |
| `scanz_second filter options close details with n candles ago offset selected.png` | Close metric with N candles ago offset, N=1, "Add as column" checked |
| `scanz_ticker details pane.png` | Ticker detail panel: price chart + News tab with timestamped headlines |
| `scanz_ticker detail pane with prints and l2 tab selected.png` | Ticker detail panel: Prints & L2 tab showing Level 2 depth and Time & Sales |
| `scanz_filter technical upper bollinger details form.png` | Upper Bollinger config: Interval, Time Period, Deviations Up/Down, Offset, "Add as column" |
| `scanz_security type dropdown.png` | Security Type filter: Regular, Preferred, ETFs, Class A-Z, Stock Split, Delinquent, Foreign, Convertible Bonds, etc. |
| `scanz_markets dropdown.png` | Markets filter: NASDAQ, NYSE, AMEX, OTC |
| `scanz_ sound dropdown.png` | Activity log sound notification options with play buttons |

---

## 3. Scanner UI Layout

Three collapsible sections, mirroring the Scanz layout:

### 3.1 Filter Rules (collapsible)
Each rule is a horizontal row:
```
[drag handle] [toggle] [Metric] [Session Badge] [Operator] [Value / Metric Reference] [delete]
```

- **"+ Add Filter"** button at bottom
- **"Select Filter"** button opens the metrics tree
- **"Hide/Show Filters"** toggle collapses the entire filter section

### 3.2 Results Table (collapsible)
Dynamic watchlist of qualifying tickers:
- Header row with global filters + result count badge:
  - **Markets:** NASDAQ, NYSE, AMEX, OTC (multi-select checkboxes, default All)
  - **Security Type:** Regular Securities, Preferred Securities (M,N,O,P), ETFs, Classes A-Z Shares (A,B), Stock Split (D), Delinquent (E), Foreign (F), First Convertible Bond (G), Second Convertible Bond, Third Convertible Bond, Voting Trust Shares, Non-Voting Shares, Bankrupt Shares of Beneficial Issues, Securities with Rights, Units, Warrants, ADRs (multi-select checkboxes with Select All / UnSelect All)
  - **Watchlist:** Filter to specific watchlist
- **"Columns"** button to customize visible columns
- Sortable columns (click header to sort)
- Session-scoped column headers (e.g., "Last Price AH", "Vol AH")
- Custom columns from "Add as column" checkbox
- Real-time updating values (green = positive, red = negative)
- Clickable ticker rows open the **Ticker Detail Pane**

### 3.3 Activity Log (collapsible)
Tracks ticker enter/exit events:

| Column | Description |
|---|---|
| # | Event sequence number |
| @Time | Timestamp of event |
| Ticker | Symbol |
| Last Price FD | Price at time of event (FD = Full Day) |
| Count | Number of times ticker has entered (cumulative) |
| COND CHG | Event type: ENTERED or EXITED |

- Event count badge (e.g., "23 Events")
- Sound notification option ("No sound" dropdown — selectable sound effects with preview play buttons)
- Event type filter ("Event Type: 1 Active" dropdown)

**Sound/notification note:** Scanz supports audible alerts on scanner events. For our use case, scanners are less about active day trading and more about feeding hundreds of scanner-based strategies automatically. Sound alerts would be overwhelming. A better fit for us:
- Toast notifications in the Streamlit UI (low priority, nice-to-have)
- Webhook alerts (existing system) for scanner-triggered entries/exits
- Sound alerts could be a general enhancement to our alert system down the line, not scanner-specific

### 3.4 Ticker Detail Pane (slide-in panel)
Opens when clicking a ticker in the results table:
- **Header:** Ticker symbol, company name, exchange
- **Price chart:** Intraday candlestick chart with volume
- **Tabs:**
  - **News** — Timestamped news headlines (most valuable for catalyst trading)
  - **Prints & L2** — Time & Sales data + Level 2 order book depth
- Lower priority for us: Prints & L2 (unless data is easy to get from Polygon)

---

## 4. Filter Builder — Detailed Specification

The filter builder is the most complex piece. Each rule follows this grammar:

### 4.1 Simple Rule (metric vs static value)
```
[Metric A] [Session] [Operator] [Static Value]
```
**Example:** `Percent Change [AH] greater than 1`

### 4.2 Metric-vs-Metric Rule (metric vs another metric)
```
[Metric A] [Operator] [Metric B] [Details Config]
```
**Example:** `Last Price >= Close (Daily, Regular Hours, 1 day ago)`

### 4.3 Metric-vs-Metric with Relative Comparison
```
[Metric A] [Operator] [Metric B] by [more/less than] [N] [Value/Percent/Times]
```
**Example:** `Last Price >= Last Price by more than 5 Percent`

### 4.4 Operators
- `>` (greater than)
- `<` (less than)
- `=` (equal to)
- `>=` (greater than or equal to)
- `<=` (less than or equal to)

### 4.5 Metric Details Configuration (for Metric B)
When Metric B is selected, a config popover appears:

**For Daily interval:**
| Field | Options |
|---|---|
| Data Interval | Daily |
| Session | Regular Hours, After Hours, Pre-Market, Full Day |
| Offset | Current day, N days ago, Average of N days |
| N = | Integer input (when N-based offset selected) |

**For Intraday interval:**
| Field | Options |
|---|---|
| Data Interval | 1 minute (and presumably other bar sizes) |
| Include Extended Hours | Checkbox |
| Offset | Current candle, N candles ago, Avg. of N candles |
| N = | Integer input (when N-based offset selected) |

Both show: **"Add this as a column to scan results"** checkbox.

### 4.6 Contextual Category Narrowing
When selecting Metric B (the comparison side), the metrics tree is filtered to only show compatible categories. Observed behavior:
- First filter (Metric A): shows all 8 categories (Price, Change, Liquidity, Technical, Capital Structure, Short Interest, Financials, News)
- Second filter (Metric B, when Metric A is a Price metric): shows only **Price** and **Technical** (numeric, comparable values)

This prevents invalid comparisons like "Last Price > News Count".

### 4.7 Session Scoping
Metrics that vary by session display a session badge (e.g., `[AH]`, `[PM]`, `[RH]`, `[FD]`):
- **AH** — After Hours
- **PM** — Pre-Market
- **RH** — Regular Hours
- **FD** — Full Day

The active session determines which data interval the metric uses.

---

## 5. Scanner Metrics Tree (Complete)

Exhaustive tree based on Scanz, organized by category. Each leaf is a selectable metric.

### 5.1 Price
| Metric | Description |
|---|---|
| Last Price | Current/most recent trade price |
| Open | Session open price |
| High | Session high |
| Low | Session low |
| Close | Session close (requires temporal config for lookback) |
| Bid | Current best bid |
| % On The Bid | Percentage of trades hitting the bid |
| Ask | Current best ask |
| % On The Ask | Percentage of trades hitting the ask |
| Spread | Ask - Bid |
| % Spread | Spread as % of midpoint |
| Range | High - Low |
| % Range | Range as % of open |

### 5.2 Change
| Metric | Description |
|---|---|
| Net Change | Dollar change from reference point (session-dependent) |
| Percent Change | Percentage change from reference point |

### 5.3 Liquidity
| Metric | Description |
|---|---|
| Volume | Total shares traded in session |
| Trades | Number of individual trades |
| Dollar Volume | Volume x VWAP (total dollar value traded) |
| Relative Volume | Current volume / N-day average volume |
| Relative Dollar Volume | Current dollar volume / N-day average |
| Relative Trades | Current trades / N-day average trades |

### 5.4 Technical
| Metric | Sub-metrics | Description |
|---|---|---|
| Simple Moving Average | — | SMA with configurable period |
| Exponential Moving Average | — | EMA with configurable period |
| Triple Exponential Average | — | TEMA with configurable period |
| Bollinger Bands | Upper, Middle, Lower | Bollinger Band levels |
| Accumulation/Distribution Oscillator | — | A/D line value |
| Average Directional Index | — | ADX trend strength |
| Average True Range | — | ATR volatility measure |
| Commodity Channel Index | — | CCI momentum oscillator |
| Moving Average Convergence Divergence | MACD Line, Signal, Histogram | MACD components |
| Money Flow Index | — | MFI volume-weighted RSI |
| Rate Of Change | — | ROC momentum |
| Relative Strength Index | — | RSI oscillator |
| Parabolic SAR | — | Parabolic stop-and-reverse level |
| Stochastics | %K, %D | Stochastic oscillator lines |
| Volume Weighted Average Price | — | VWAP level |

**Technical indicator configuration:** When selecting a technical sub-metric (e.g., Upper Bollinger), a details form appears with:
- **Interval** — Daily or intraday (1 min, etc.)
- **Time Period** — Lookback period (e.g., 20 for Bollinger)
- **Indicator-specific params** — e.g., Deviations Up / Deviations Down for Bollinger Bands
- **Include Extended Hours** checkbox (intraday intervals only)
- **Offset** — 0 for current, N for lookback
- **"Add this as a column to scan results"** checkbox

This means each technical metric is fully configurable — period, parameters, interval, and offset — not fixed. Our system could extend this with user-installed Confluence Pack indicators (e.g., UT Bot levels, Swing 123 levels).

### 5.5 Capital Structure
| Metric | Description |
|---|---|
| Market Capitalization | Share price x shares outstanding |
| Shares Outstanding | Total shares issued |
| Float | Shares available for public trading |
| Percent Float | Float / shares outstanding |
| Float Rotation | Volume / float (how many times float has traded) |
| Percent Insiders | Insider ownership % |
| Percent Institutions | Institutional ownership % |

### 5.6 Short Interest
| Metric | Description |
|---|---|
| Short Shares | Current shares sold short |
| Short Ratio | Short shares / avg daily volume |
| Short Shares (Prior Month) | Previous month's short shares |
| Short Percent Change Month Over Month | MoM change in short interest |
| Days To Cover | Short shares / avg daily volume |

### 5.7 Financials
| Metric | Description |
|---|---|
| Revenue | Total revenue |
| Gross Profit | Revenue - COGS |
| EBITDA | Earnings before interest, taxes, depreciation, amortization |
| Cash | Cash and equivalents |
| Debt | Total debt |

### 5.8 News
| Metric | Description |
|---|---|
| News Count | Number of news articles in lookback window |

---

## 6. Scanner → Strategy Integration

### 6.1 Lifecycle

```
Saved Scanner (named, reusable)
    ├── Filter rules (persisted as JSON)
    ├── Session scope
    ├── Update frequency
    └── Column configuration
            │
            ▼
    Qualifying Tickers (dynamic watchlist)
    ├── Continuously evaluated against filter rules
    ├── Activity log tracks ENTER/EXIT events
    └── Max tickers cap (optional)
            │
            ▼
    Strategy Method: Scanner-Based
    ├── User selects a saved scanner during strategy builder flow
    ├── Strategy engine runs entry/exit triggers on each qualifying ticker
    ├── Position management per ticker (stop, target, bar count exit)
    └── Portfolio-level risk limits apply across all scanner-sourced positions
```

### 6.2 Strategy Builder Integration
- Scanners are saved with custom names, like confluence groups
- During strategy builder, user selects strategy method:
  1. Ticker-Based (existing)
  2. Webhook-Based (existing)
  3. **Scanner-Based** (new) → select from saved scanners
- The selected scanner determines the universe of tickers
- Entry/exit triggers, confluence conditions, and risk management are configured as normal
- Backtesting would run the scanner criteria historically, then evaluate strategy on qualifying tickers

### 6.3 Confluence Pack Integration (Future Idea)
User-installed packs could expose indicator values as scanner metrics. For example, a UT Bot pack could add "UT Bot Buy Level" and "UT Bot Sell Level" to the Technical scanner metrics tree. This would tie the pack system directly into the scanner, so custom indicators are scannable.

---

## 7. Data Requirements & Polygon Considerations

### 7.1 Core Data Endpoints

| Need | Polygon Endpoint | Notes |
|---|---|---|
| Real-time price/volume | WebSocket trades + quotes | Per-tick updates for scanner results |
| Market-wide snapshot | `/v2/snapshot/locale/us/markets/stocks/tickers` | Returns price, volume, change, prev close for ALL tickers in one call — ideal for scanner sweeps |
| Historical bars | `/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}` | For computing moving averages, ATR, RVOL lookbacks |
| Reference data | `/v3/reference/tickers` | Float, market cap, shares outstanding, sector/industry |
| Financials | `/vX/reference/financials` | Revenue, EBITDA, cash, debt |
| News | `/v2/reference/news` | Timestamped news articles with ticker tags |
| Short interest | Not available in Polygon | May need separate data source (FINRA, Ortex, or similar) |

### 7.2 News Data
Polygon provides news via `/v2/reference/news?ticker={ticker}` with article title, author, published date, article URL, and ticker references. This covers the baseline Scanz-like news functionality. For more advanced news evaluation (sentiment scoring, novelty detection, categorization), we could layer on:
- **Benzinga** — Real-time news with sentiment tags (Polygon partners with them)
- **Custom NLP/LLM layer** — Classify news as actionable vs noise, detect "actually new" vs rehashed stories

### 7.3 Short Interest Data
Polygon does not provide short interest data. Options:
- **FINRA** — Official source, published bi-monthly (limited frequency)
- **Ortex** — Real-time short interest estimates (paid API)
- **S3 Partners** — Similar to Ortex
- **Defer** — Start without short interest; add later if needed

### 7.4 Refresh Strategy
- **Snapshot polling** — Use Polygon's all-tickers snapshot endpoint on a timer (e.g., every 5-10 seconds) for the main scanner evaluation loop
- **WebSocket streaming** — Subscribe to qualifying tickers for tick-level price updates in the results table
- **Hybrid** — Snapshot for scanner evaluation, WebSocket for real-time display of qualifying tickers

---

## 8. Open Questions

- How many scanner presets should a user be able to save? (Unlimited? Cap?)
- Should scanners run continuously or on-demand? (Scanz runs continuously)
- How to handle position limits when scanner finds many qualifying tickers? (Max simultaneous positions per scanner-strategy?)
- Should scanner results feed into existing portfolio/requirement framework? (Almost certainly yes)
- How does backtesting work for scanner-based strategies? (Need historical scanner evaluation — computationally expensive)
- Should we support "Stocks in Play" pre-built scanners like Scanz has? (Pre-Market movers, After Hours gainers, etc.)
- What's the update frequency target? (Tick-level like Scanz, or is 5-10 second polling sufficient?)
- How to handle the transition from qualifying to disqualifying? (Immediate position exit? Hold until strategy exit triggers?)
- Level 2 / Time & Sales data — worth including in ticker detail? (Polygon provides L2 on higher plans)

---

## 9. Complexity Assessment

The scanner feature has several distinct complexity layers:

| Component | Complexity | Notes |
|---|---|---|
| Filter rule CRUD & persistence | Low | JSON-based, similar to existing patterns |
| Filter builder UI | **High** | Metric tree, contextual narrowing, temporal offset config, metric-vs-metric with relative comparison |
| Scanner evaluation engine | **High** | Must evaluate all filter rules against all tickers efficiently, handle session scoping, temporal lookbacks |
| Real-time results table | Medium | WebSocket-driven updates, dynamic column management |
| Activity log | Low | Event logging with timestamps |
| Ticker detail pane | Medium | Price chart, news feed, optional L2/prints |
| Strategy builder integration | Medium | New strategy method type, scanner selection UI |
| Historical backtesting for scanner strategies | **High** | Requires replaying scanner criteria historically across the full market |

---

## 10. Scanner-Based Backtesting & System Integration Vision

This section captures how scanner-based strategies would integrate with our existing architecture, from the strategy builder UX through backtesting execution to live trading.

### 10.1 The Core Problem: Time-Varying Universe

Traditional backtests run one strategy on one ticker over a fixed time window. Scanner-based strategies operate on a **dynamic universe** — the set of eligible tickers changes bar by bar. A stock might qualify for 20 minutes during a pre-market volume spike, then drop out when volume fades. The backtest engine must replay this dynamic qualification historically.

### 10.2 Backtest Engine Architecture

The scanner backtest has two nested evaluation loops:

```
For each bar (time step):
    1. SCANNER EVALUATION — apply all filter rules against every ticker
       → produces: qualifying_tickers (set of symbols that pass filters this bar)

    2. STRATEGY EVALUATION — for each qualifying ticker:
       - If FLAT:  check entry triggers + confluence conditions
       - If IN_POSITION: check exit triggers, stops, targets
         (even if ticker no longer qualifies — see note below)

    3. PORTFOLIO CONSTRAINTS — enforce cross-ticker position limits
       (max simultaneous positions, risk allocation per position, etc.)
```

**Critical design decision:** Once a position is entered, exit logic is governed by the strategy's triggers (stop loss, take profit, signal exit, bar count exit) — NOT by the ticker falling out of the scanner. The scanner determines *when to start looking*; once you're in, your risk management takes over. A ticker dropping out of the scanner mid-trade should not force an exit.

### 10.3 Performance Optimization: Staged Filtering

Evaluating all filter rules against all tickers on every bar is computationally expensive. The solution is staged filtering — apply cheap filters first, then compute expensive metrics only on survivors:

```
Stage 1 (cheap): Static reference data filters
    Float < 20M, Market Cap > $50M, Security Type = Regular
    → These rarely change; cache and refresh daily
    → Eliminates ~80% of tickers immediately

Stage 2 (moderate): Snapshot-level filters
    Price > $1, % Change > 5%, Volume > 100K
    → Available from Polygon snapshot endpoint (one API call for all tickers)

Stage 3 (expensive): Technical indicator filters
    RSI < 30, Price > 20 EMA, RVOL > 3
    → Computed per-ticker, but only on the survivors from Stage 1+2
    → Likely reduces to <100 tickers on any given bar
```

For historical backtesting, Stage 1 filters pull from reference data (mostly static), Stage 2 from historical daily/minute bars, and Stage 3 requires computing indicators on each candidate ticker's bar history. Polygon's historical aggregates API supports this.

### 10.4 Strategy Builder UX

The strategy builder flow for scanner-based strategies mirrors the existing ticker-based flow, with the scanner replacing the ticker selection:

1. **Select strategy method:** Scanner-Based
2. **Select saved scanner** (dropdown, similar to selecting a confluence pack)
3. **Configure direction** (LONG/SHORT)
4. **Configure entry trigger + confluence conditions** — identical to current flow
5. **Configure exit triggers, stops, bar count exit** — identical to current flow
6. **Click Backtest** → engine replays scanner historically, runs strategy on qualifying tickers

The scanner acts as an additional filter layer on top of the existing trigger/confluence system. A trade only fires when:
- The ticker qualifies under the scanner filters, AND
- The entry trigger fires, AND
- All confluence conditions are met

### 10.5 Backtest Results View

Scanner-based backtest results differ from ticker-based results because trades span many tickers:

**Aggregate KPIs (top level):**
- Same KPIs as today (win rate, avg R, profit factor, max drawdown, etc.)
- Computed across ALL trades on ALL qualifying tickers
- Answers: "How does this strategy perform on stocks that meet these scanner criteria?"

**New elements:**
- **Ticker column** in trade table — identifies which stock each trade was on
- **Scanner Activity timeline** — shows which tickers qualified when (like the Scanz activity log, but historical). Helps visualize "how often does this scanner find candidates?"
- **Composite equity curve** — aggregated P&L across all ticker trades over time
- **Per-ticker drill-down** — click a ticker to see its individual trades and chart (reuses existing chart infrastructure)
- **Scanner hit rate** — how many qualifying events led to actual entries (not all qualifications produce entry triggers)

### 10.6 Live Trading Integration

For live execution, the scanner evaluation engine runs continuously (or on a polling interval):

```
Scanner Engine (live)
    │
    ├── Polls Polygon snapshot endpoint every N seconds
    ├── Evaluates filter rules → qualifying tickers
    ├── Compares to previous qualifying set → ENTER/EXIT events
    ├── Logs activity (activity log)
    │
    └── For each qualifying ticker:
        └── Ralph Engine instance (per ticker)
            ├── Subscribes to WebSocket bars for qualifying ticker
            ├── Runs unified engine entry/exit triggers
            ├── Dispatches alerts via existing webhook system
            └── Unsubscribes when ticker disqualifies (if FLAT)
```

This integrates with our existing alert infrastructure — webhook dispatch, Discord/Slack notifications, alert analysis — with no changes to the downstream pipeline. The scanner just determines which tickers get Ralph Engine instances.

### 10.7 Portfolio Integration

Scanner-based strategies naturally produce many simultaneous positions across different tickers. This maps well to our existing portfolio and requirements system:

- **Position limits** — max simultaneous positions per scanner-strategy (e.g., top 5 by RVOL)
- **Risk allocation** — per-position risk as a fraction of total portfolio risk budget
- **Prop firm compliance** — daily loss limits, max positions, etc. apply across all scanner-sourced trades
- **Portfolio builder** — scanner-based strategies are added to portfolios alongside ticker-based and webhook-based strategies

### 10.8 Example Use Cases

**Pre-market news play (Ross Cameron style):**
- Scanner: float < 20M, pre-market % change > 10%, pre-market volume > 500K, news count > 0
- Timeframe: 10-second or 1-minute bars
- Entry: momentum trigger (price breaks pre-market high, or UT Bot buy signal)
- Exit: quick scalp target (2R), tight stop below entry candle low
- Edge: systematic entry on scanner qualification + trigger, no emotional FOMO

**Intraday "Stocks in Play" (SMB Capital style):**
- Scanner: RVOL > 3, price > $5, ATR > $0.50, float < 50M
- Timeframe: 1-minute or 5-minute bars
- Entry: pullback to VWAP with EMA confluence
- Exit: trailing stop or target at prior high
- Edge: only trades stocks with genuine institutional interest (high RVOL)

**Gap and go:**
- Scanner: gap up > 5% from previous close, pre-market volume > 200K, price > $2
- Timeframe: 1-minute bars, regular hours only
- Entry: first 1-min candle break of pre-market high
- Exit: bar count exit (5 bars) or trailing stop

### 10.9 News Evaluation Pipeline (Future)

Breaking news is one of the highest-edge opportunities in trading. A pipeline for this could look like:

```
Polygon News API (polling every 30s)
    │
    ├── New article detected for ticker
    ├── LLM classification:
    │   ├── Is this actually NEW information? (vs rehash/update)
    │   ├── Category: earnings, FDA, partnership, offering, insider, macro
    │   ├── Sentiment: positive / negative / neutral
    │   └── Magnitude: high / medium / low impact
    │
    ├── If actionable (new + high magnitude):
    │   └── Ticker gets priority qualification in scanner
    │       (could bypass normal filter thresholds or get weighted higher)
    │
    └── Historical news data enables backtesting:
        "How do stocks perform after positive FDA news with float < 10M?"
```

This is a differentiator — Scanz shows you the news but doesn't evaluate it. We could objectively classify news and use that classification as a scanner filter or confluence condition. The LLM layer is what makes this possible at scale without a human reading every headline.

### 10.10 Why This Matters

Most retail traders face a choice:
1. **Systematic/algo trading** — objective, backtested, but limited to fixed tickers
2. **Discretionary/active trading** — flexible across the market, but no data-backed validation

Scanner-based strategies bridge this gap. You get the flexibility of scanning the entire market for opportunities (like an active trader) with the objectivity and discipline of systematic backtesting and automated execution. The backtest answers: "If I had traded this strategy on every stock that met these criteria over the past year, what would my results look like?" — a question that no retail tool currently answers well.

The infrastructure we've built (unified engine, confluence system, alert pipeline, portfolio management) is the foundation. The scanner adds the dynamic universe layer on top, and everything downstream works the same.

---

## 11. Raw Notes

- The filter builder's metric-vs-metric-with-temporal-offset pattern is the killer feature that sets Scanz apart from simpler scanners. It enables rules like "close of current 1-min candle > close of 20 candles ago" or "last price >= yesterday's close by more than 3%". This flexibility is what we need to replicate.
- The "Add this as a column to scan results" checkbox is a great UX pattern — lets users see the reference value they're filtering against right in the table.
- Consider whether our Confluence Pack indicators could be exposed as scanner metrics. E.g., if you install a UT Bot pack, "UT Bot Buy Level" becomes a scannable metric under Technical. This would be a differentiator vs Scanz.
- News is particularly valuable for catalyst-driven strategies. Being able to evaluate news quality/novelty programmatically (is this actually new information?) would be a significant advantage over Scanz. LLM-based news classification is a natural fit here.
- Breaking news trading is one of the best opportunities in stock trading. Building a pipeline that detects breaking news → evaluates significance → enters positions on qualifying tickers would be extremely powerful.
- The "Stocks in Play" concept (pre-built session-specific scanners) is good for onboarding — ship with sensible defaults.
- Example saved scanner names from Scanz sidebar: "1m HOD Pullbacks", "After Hours (copy)", "Close to HOD", "Fashionably Late Scalp", "Forming C1", "Gainers (copy)", "Goldilocks", "Poppin", "Strat Catcher", "WT at Hi", "Warrior Trading", "Trending". Shows the variety of use cases people build scanners for.
- For our use case of potentially hundreds of scanner-based strategies running simultaneously, the scanner evaluation engine needs to be efficient. Batch evaluation via Polygon's snapshot endpoint (one call returns all tickers) is key — we evaluate all scanner filter rules against the snapshot rather than polling per-ticker.
