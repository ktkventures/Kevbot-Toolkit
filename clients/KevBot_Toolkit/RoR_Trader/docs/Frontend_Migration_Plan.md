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

### TF Confluence — LOCKED: V5 (Production)
- **Decision:** V5 with all production changes.
- **Key decisions made during iteration:**
  - **All 8 templates** from backend (EMA Stack, MACD Line, MACD Histogram, VWAP, RVOL, UT Bot, EMA Price Position, Bar Count Exit)
  - **Parameters lock after save** to protect live strategies/portfolios. Unsaved drafts show warning banner. No save button on Parameters tab — only "Save as Variation" in header.
  - **Variations via Copy** — "Create Variation" button opens detail view in draft mode with all tabs accessible (including Preview). User sets name + params + trigger params, then saves.
  - **Nested under defaults** — Variations indent under parent template with expand/collapse chevron inside the card.
  - **Search bar** for filtering by name, version, or tag.
  - **Tags on cards** instead of category sections. Multiple tags per pack (e.g., "Moving Averages" + "Trend").
  - **"Create New Template"** button links to Pack Builder.
  - **Updated execution types** — `[C]`, `[L]`, `[LC]`, `[CC]` per display settings V5 spec. Single blue color for all exec badges, cyan for fidelity badges, pill-shaped, bracketed.
  - **Trigger Parameters tab** — source of truth for all execution type settings across the pack. [C] and [L] have standalone settings. [LC] inherits stage 1 from [L], only exposes confirmation/bail. [CC] inherits stage 1 from [C], only exposes confirmation/bail. Bail options: immediate market exit, immediate limit exit, limit exit at breakeven.
  - **Outputs & Triggers tab** — [PB]/[CB] fidelity badges on outputs header, exec type badges on triggers header. Sentiment badges (bullish/bearish/neutral) replace LONG/SHORT and ENTRY/EXIT labels. Triggers expandable to show variant IDs.
  - **Card layout** — Row 1: name, version, tags, state count (tooltip with names), trigger count (tooltip with names). Row 2: all locked params in monospace shorthand — indicator params + exec type settings separated by pipe.
  - **7 tabs in detail view:** Parameters, Trigger Parameters, Plot Settings, Outputs & Triggers, Preview (chart + state timeline + trigger events tables), Code (collapsible indicator/interpreter/Pine Script), Danger Zone.

### General Packs — LOCKED: V5 (Production)
- **Decision:** V5, adapted from TF Confluence V5 patterns for scalar time-based conditions.
- **Key decisions:**
  - **4 templates:** Time of Day, Trading Session, Day of Week, Calendar Filter
  - Same pack architecture: locked params, nested variations, draft mode, search, "Create New Template" → Pack Builder
  - **No Plot Settings tab** (scalar conditions, nothing to visualize)
  - **No Trigger Parameters tab** (bar-close only, no execution type variants)
  - **No fidelity badges** (PB/CB don't apply to time-based conditions)
  - **5 tabs:** Parameters (template-specific: time inputs, session dropdown, day toggles, event toggles), Outputs & Triggers, Preview (condition bands + state timeline), Code (evaluator function), Danger Zone
  - **Binary outputs** (IN_WINDOW/OUT_OF_WINDOW, IN_SESSION/OUT_OF_SESSION, etc.)
  - **0-2 triggers per template** (Day of Week has none — condition-only gate)
  - **4 default packs** with mock variations (Power Hour, Midweek Only)
  - Sentiment badges on triggers (bullish/bearish/neutral)

### Stop Loss Packs — LOCKED: V1 (Production) — NEW PAGE
- **Decision:** New page at `/confluence-packs/stop-loss`, replacing the stop-loss half of Risk Management.
- **Key decisions:**
  - **6 templates:** ATR Stop, Fixed Dollar, Percentage, Swing, ATR Trailing, Breakeven
  - Same pack architecture: locked params, nested variations, draft mode, search
  - **5 tabs:** Parameters, Behavior (explains method mechanics + exit priority), Preview (chart + sample trades table), Code, Danger Zone
  - Cards show effective formula in monospace (e.g., `1.5x ATR`, `5-bar swing, $0.05 pad`)
  - Trailing/breakeven templates include activation_r threshold + modifier params
  - Behavior tab explains stop is checked first (highest exit priority, pessimistic fill)

### Take Profit Packs — LOCKED: V1 (Production) — NEW PAGE
- **Decision:** New page at `/confluence-packs/take-profit`, replacing the take-profit half of Risk Management.
- **Key decisions:**
  - **6 templates:** Risk:Reward, ATR Target, Fixed Dollar, Percentage, Swing Target, No Target
  - Same pack architecture as Stop Loss
  - R:R template notes dependency on stop loss to calculate risk
  - "No Target" template for signal/bar-count-only exit strategies
  - Behavior tab explains take profit is checked second (after stop loss)
  - Sidebar updated: "Risk Management" replaced with "Stop Loss" and "Take Profit"

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

### Strategy Builder — LOCKED: V5 (Pack Integrated)
- **Decision:** V5 based on V2 Full Parity with pack architecture integration.
- **Key decisions made during iteration:**
  - **Strategy Method selector** — card above core config with Standard (active), Inbound Webhook (coming soon), Scanner (coming soon). Backend keys off `strategyMethod` field.
  - **Stop Loss & Take Profit via pack selectors** — replaces inline slider config. Scrollable list of saved packs (same style as entry/exit trigger lists). Single selection per slot. Links to pack management pages.
  - **Updated trigger exec types** — `[C]`, `[L]`, `[LC]`, `[CC]` naming. All 8 TF templates represented in mock triggers. Uniform blue badges per display settings V5.
  - **Symmetric layout** — Left: equity curve/price chart (620px fixed) + trade history. Right: analysis tabs (620px fixed) + advanced analysis.
  - **Trade history timestamps** include seconds (HH:MM:SS).
  - **Confluence Depth selector** — pinned to bottom of analysis card. Button group (1-N) per tab. Entry/Stop/Target: max 1. Exit: max 3. TF Conditions/General: max 4. Label: "Individual" for 1, "Combinations of up to N" for 2+.
  - **Depth-dependent button labels** — Depth 1: "Add" (building up items). Depth 2+: "Replace" (picking a combination). Entry/Stop/Target always "Replace".
  - **Filter & Sort modal** — accessible via filter icon on every analysis tab toolbar. Sort by (PF, WR, Avg R, Daily R, R², Trades), ascending/descending toggle, 6 minimum threshold fields (Min PF, Min WR, Min Trades, Min R², Min Daily R, Min Avg R).
  - **Consistent toolbar** across all 6 analysis tabs: Search + Analyze + Filter icon. No "+ Add" buttons on toolbar row.
  - **Analysis tabs:** Entry, Exit, TF Conditions, General, Stop Loss, Take Profit.

### Strategy Detail — LOCKED: V5 (Production)
- **Decision:** V5 approved with all feedback incorporated.
- **Key features in V5:**
  - **6 tabs** (down from 9): Equity & KPIs, Chart & Trades, Confluence Analysis, Configuration, Alerts, Alert Analysis
  - Forward test and alerts always on — no separate BT-only view
  - **KPI comparison modes** matching My Strategies V5: Overall | BT vs FWD | FWD vs Alerts | BT vs Alerts. 7-column table (WR, PF, Daily R, Daily ROI, TPD, Max DD) with delta row.
  - **Extended KPIs** collapsible (open by default) with 5 sub-tabs: Performance, Risk-Adjusted, Distribution, Drawdown, Streaks
  - **Date Range + KPI Mode** on same row inside Equity & KPIs tab (not above tabs)
  - **3-segment equity curve** per display settings V5 (BT blue, FWD orange, Live green overlay). HWM/Edge Check/Confidence Bands toggleable.
  - **R-distribution** side-by-side BT vs FWD histograms
  - **Advanced Analysis** collapsible: Rolling Metrics (window slider + metric selector), Return Distribution (histogram/box/violin + stats), Markov Motor (transition matrix + edge decay)
  - **Chart & Trades merged**: OHLC chart → Position Status module → Current Conditions table → Alert History + Trade History (Backtest) side-by-side → FWD trades (expanded) + BT trades (collapsed)
  - **Confluence Analysis**: per-group sub-tabs with full-width indicator chart, then Interpreter State Timeline + Trigger Events side-by-side
  - **Alerts tab — Webhook Execution Flow (Phase 39):**
    - Webhook execution flow intro (driven by trigger parameters, not execution type labels)
    - Entry → Exit lifecycle diagram with step-by-step green/red dots and timing descriptions
    - Webhook Events table: event type, order type, when it fires, conditions (stop loss breach, exit trigger, target, bar count)
    - Trigger Configuration Webhook Reference: 4-card grid ([C] Bar Close, [L] Level Cross, [LC] Level Cross + Bar Close Confirm, [CC] Bar Close + Next Bar Close Confirm) with exec type badges for educational reference
    - Available Payload Data: placeholder table showing **last alert values** (not examples) so users can verify system accuracy
    - Recent Alerts table with new event type names (entry_long_market, etc.) and order type column
    - Trade-to-Alert mapping with timing deltas
  - **Alert Analysis**: discrepancies (missed + phantom), 4-column summary metrics (FT All | FT Alerts-On | Alert Actual | Delta), position health with anomalies, trigger timing analysis, trade-by-trade R-multiple + dollar slippage comparison
  - **Pack-aware variable display** consolidated to 2 lines: entry + exit on line 1 (pipe separated), stop + target + confluence on line 2
  - **Dual sigma badges** (FWD σ orange | Alert σ green) + pulsing monitored dot

### My Strategies (List) — LOCKED: V5 (Refined)
- **Decision:** V5 based on V2 with major card and UX overhaul.
- **Key decisions made during iteration:**
  - **Inline checkbox** on each card's action row (no select mode toggle). Checking any card surfaces bulk action bar.
  - **Bulk actions:** Delete Selected, Create Portfolio, Update Portfolio, Add Tag, Select All, Clear.
  - **3-segment equity curve** per display settings V5: Backtest (blue) → Forward (orange) → Live Alerts (green overlaid on forward x-axis showing slippage). Gradient fills on BT/FWD. Zero line, HWM, forward boundary.
  - **Viewing preferences row:** KPI Mode, Chart Height (S/M/L/XL), X-axis (Time/Trade #), High Water Mark (On/Off), Edge Check (On/Off), Confidence Bands (On/Off).
  - **KPI Mode dropdown** with 4 views: Overall (single row), Backtest vs Forward (table with deltas), Forward vs Alerts (table), Backtest vs Alerts (table). Comparison tables show WR, PF, Daily R, Daily ROI, TPD, Max DD with color-coded deltas.
  - **Daily ROI %** added as 6th KPI — `(avg daily P&L) / (avg capital deployed per trade) × 100`. Capital efficiency metric for portfolio decisions.
  - **TPD** (Trades Per Day) replaces raw trade count — normalized for apples-to-apples comparison across different time periods.
  - **Dual sigma badges** in top-right: forward test σ (orange) | alert σ (green). Color-coded: green <1σ, orange 1-2σ, red >2σ.
  - **Pulsing green dot** replaces "Monitored" text badge. Alerts on by default with toggle on action row.
  - **Alert accuracy metric** on meta line (green) for strategies with alert tracking.
  - **Data View filter** — Strategy Default, All Data, Last 7/30/90 Days, Backtest Only, Forward Test Only. Affects KPIs displayed on cards.
  - **Pack-aware strategy variables** displayed in 4 rows: entry (exec badge + pack > trigger), exit (supports multiple), stop (red) + target (green), confluence ([PB] fidelity badges + condition names). Always shows confluence row ("none" if empty).
  - **Filter row:** 6-column grid — Ticker, Direction, Tag, Status, Data View, Sort.

### Webhook Templates (List) — LOCKED: V5 (Account-Based, Phase 39)
- **Decision:** V5 account-based templates replacing V1-V4 individual payload templates.
- **Key decisions:**
  - **Account-based templates** — each template is tied to an exchange/service and contains payloads for all 11 webhook event types
  - **Card grid** with template name, exchange badge, portfolio count, event count, last delivery status dot (Healthy/Error), and URL preview
  - **Search bar** for filtering by name or exchange
  - **+ New Template modal** — 3-step: pick exchange preset (SignalStack, TradeThePool, Discord, Slack, Custom) → name → URL. Pre-fills all 11 event payloads with exchange defaults. Navigates to detail page on create.
  - Cards link to detail page (same pattern as strategies)
- **Reference spec:** `docs/Webhook_Event_System.md`

### Webhook Template Detail — LOCKED: V5 (Account-Based, Phase 39)
- **Decision:** V5 detail page with 4 tabs, matching strategy detail page pattern.
- **Key decisions:**
  - **Event Payloads tab** — category filter pills (Entry/Exit/Cancel/Compliance with color coding and counts), event list sidebar, payload template editor with resolved preview below, all-events overview table at bottom (clickable rows scroll to editor)
  - **Placeholders tab** — grouped by category (Signal, Order, Strategy, Portfolio, Indicator, Meta) with copy buttons per placeholder
  - **Delivery History tab** — 4 summary metric cards (total, success rate, avg latency, last sent), timeline chart placeholder, recent deliveries table with event badges and status dots
  - **Settings tab** — template config (name, exchange, URL, created/updated dates), portfolio usage list, danger zone with delete
  - **Header** — back link to list, Edit/Duplicate/Delete buttons, exchange badge, health status dot, portfolio count, URL

### My Portfolios (List) — LOCKED: V5 (Refined, My Strategies Style)
- **Decision:** V5 matching My Strategies V5 style with portfolio-specific KPIs and combined metrics.
- **Key decisions:**
  - **Card layout** matching strategies: pulsing green dot (enabled), name + status badge, sigma badges (FWD σ / Alert σ) top-right, tags below status, meta line (strategy count, balance, scaling %, avg risk, trades/day, webhook template name), strategy pills (symbol + direction), 3-segment equity curve (BT/FWD/Alerts)
  - **Viewing preferences row** matching strategies: KPIs dropdown (Overall/BT vs FWD/FWD vs Alerts/BT vs Alerts), Chart Height (S/M/L/XL), X-axis (Time/Trade #), HWM, Edge Check, Confidence Bands
  - **Filter row:** Status, Tag, Sort, Date Range (All Data/Last 7/30/90 Days/Backtest Only/Forward Only/Custom), Custom shows start + end date pickers inline
  - **KPI modes:** Overall = 6 KPIs (P&L, WR, PF, Max DD %, Avg Daily P&L, Trades). Comparison modes = 6-column table with delta row (P&L, WR, PF, Max DD, Avg Daily)
  - **Combined Metrics card** between preferences and cards — aggregates all visible portfolios: Total P&L, Combined Balance, ROI, Avg WR, Avg PF, Worst Max DD, Combined Daily P&L, Total Trades, Strategies count. Combined equity curve placeholder. Updates dynamically with filters.
  - **Requirement set badge** with pass count (green all pass, orange if violations)
  - **Action row:** View, Edit, Clone, Delete, Tags + Enabled toggle (replaces alert toggle) + bulk select checkbox
  - **Bulk actions:** Select All, Delete Selected, Clear
  - **Portfolio-specific KPIs** use dollar amounts (P&L, balance, daily P&L) vs R-multiples on strategies

### Portfolio Detail — LOCKED: V5 (Production)
- **Decision:** V5 based on V2 Full Parity with Phase 39 webhook integration, risk analytics, and Streamlit parity.
- **Key decisions:**
  - **6 tabs** (down from 7): Live Dashboard, Performance, Strategies, Prop Firm Check, Account, Webhooks
  - **Live Dashboard:**
    - Visibility disclaimer banner at top: "Dashboard metrics reflect visible strategies from the Strategies tab"
    - Planned/Executed data toggle on same banner — Planned = unlimited buying power (strategy performance), Executed = actual transactions (buying power constrained)
    - 5 KPIs: Alert Trades, Win Rate, Total P&L, Expected P&L, vs Plan
    - Performance vs Plan chart with 1SD/2SD confidence bands
    - Open Positions table with Status (Matched/Phantom) and Close Early button
    - Buying Power Tracker: date picker with left/right arrows for day navigation, 24-hour chart, 4 KPIs (starting balance, available, allocated, peak)
    - Anomaly Detection: category tabs (All / Alert Issues / Performance). Alert Issues = phantom trades, unclosed positions, missed trades. Performance = consecutive losses, etc.
    - Trade History: 13-column table matching Streamlit (#, Strategy, Symbol, Dir, Entry $, Exit $, Reason, P Qty, E Qty, R, P&L, Status, Chart). Chart button opens modal with trade metrics, execution info, slippage, and focused candlestick chart
  - **Performance:**
    - 6 KPIs (Trades, WR, PF, Total P&L, Balance, Max DD)
    - Combined equity curve: bold white portfolio line + per-strategy dashed colored lines
    - Drawdown analysis: red area chart with requirement set limit dashed line (e.g., FTMO -10%). Metrics: Max DD ($+%), Profitable Days, Avg Daily P&L with std, Current DD with margin
    - Daily P&L Distribution + Strategy Correlation Heatmap side by side
    - Risk Analytics section: Daily Peak Capital Deployed (bar chart by day with account balance threshold line), Daily P&L vs Limits, Worst-Case Analysis (5 KPIs + Top 5 Worst Days table), Monte Carlo Simulation (shuffle mode, sim count, 6 result KPIs, DD distribution + equity confidence bands)
  - **Strategies:**
    - Per-strategy cards with: status badge, sigma badges (FWD σ / Alert σ), KPIs (WR, PF, Daily R, Trades, P&L Contribution), 3-segment equity curve
    - Visible/Hidden toggle (controls Live Dashboard inclusion), Active/Paused toggle (controls webhook firing — alerts still track when paused)
    - Data View dropdown (All Data / Last 7/30/90 Days / Backtest Only / Forward Only / Custom with date pickers), X-axis toggle (Time / Trade #)
    - View Strategy, View Chart, Delete buttons
  - **Prop Firm Check:** Requirement set selector, per-rule compliance cards with progress bars, profit target progress, margin of safety. Future: multi-firm compatibility view.
  - **Account:** Balance KPIs (Current, Starting, Net Deposits, Trading P&L), Balance History chart (future: auto-adjusting std dev bands on deposit/withdrawal), Deposit/Withdrawal forms side by side, Ledger with daily detail modal, Trading Journal with mood/confidence/notes, Change History with typed badges. Account balance is source of truth for buying power and quantity calculations.
  - **Webhooks:** Template selector dropdown (Phase 39 account-based templates) + View Template link, Compliance Breach Alerts toggle, Webhook Delivery History table. Deploy status/monitoring/logs removed (better suited for Settings > Connections).
  - **Header:** Status badge, dual sigma badges, pulsing enabled dot, tags, meta line (strategies, balance, scaling, risk, trades/day, webhook template, requirement set)

### Portfolio New/Edit — LOCKED: V5 (Production)
- **Decision:** V5 based on V2 Full Parity with Phase 39 webhook template, risk analytics, projection views, and filter modal.
- **Key decisions:**
  - **Settings row:** 5-column grid — Name, Starting Balance, Risk Scaling slider, Requirement Set, Webhook Template (defaults to paper account)
  - **Primary KPIs (rate-based, duration-independent):** Daily Avg P&L, 30-Day Est., Win Rate, PF, Max DD, Trades/Day
  - **Secondary KPIs (8 metrics):** Monthly Est., Quarterly Est., Annual Est., Annual ROI %, Total Trades, Avg R/Trade, Payoff Ratio, Max Concurrent Positions
  - **Equity curve dual view:** Backtest (actual historical curves at true dates) vs Projected (normalized day-0 start, daily avg performance, ±1SD/±2SD confidence bands). X-axis toggle (Date / Trade #).
  - **Projection math (for implementation):** Per-strategy avg R + R variance from backtest trades + trade frequency. Combined at portfolio level weighted by trade frequency. Plan line = N × expected dollar step. Confidence bands = ±σ × √N. Risk scaling compounds over projection horizon. Daily Avg P&L = current-day expectation at current balance/risk. Monthly/quarterly/annual apply compounding.
  - **View filter consideration (for implementation):** Data View dropdown should affect KPI display (All Data, Last 30/90 Days, Forward Only). Full backtest = baseline, sliced views = analysis. Not a portfolio setting — just a display filter.
  - **Recommendation filter modal:** Sort by (P&L Impact, DD Impact, Correlation, Win Rate, PF, RODC, Hold Time). Risk assumption input for RODC calc. Min thresholds: Win Rate, PF, Max Correlation, Min RODC. Reset + Apply.
  - **RODC metric** (Return on Deployed Capital): $/day per $1k deployed. Factors in trade frequency, hold time, risk per trade.
  - **Risk Summary:** Full-width horizontal card with 4 metrics (Total Risk/Day, Max Daily Loss, Worst Case, Capital Utilization with progress bar)
  - **Risk Analytics:** Daily Peak Capital Deployed chart, Worst Case Analysis (5 KPIs), Monte Carlo Simulation (shuffle mode, sim count, 6 KPIs, DD distribution + equity confidence bands)
  - **Existing V2 features kept:** Strategy search/add with risk per trade, combined equity curve with per-strategy dashed lines, drawdown analysis, strategy cards with position sizing, drag-to-reorder, recommendation engine with analyze button

### Portfolio Requirements — LOCKED: V5 (Trade Qualification)
- **Decision:** V5 based on V3 streamlined inline editing with backend-aligned rule types and Trade Qualification Rules.
- **Key decisions:**
  - **Backend-aligned rule types:** Percentage-based (max_daily_loss_pct, max_total_drawdown_pct, daily_pause_pct, min_profit_pct, min_profitable_days, min_trading_days, max_position_size)
  - **Trade Qualification Rules section** per set: min hold time, min price move, min profit threshold. Orange-themed, visually distinct from compliance rules. Addresses prop firm nuances like TTP's 30-second hold and $0.10 minimum move.
  - **"Applies To" field** on each TQ rule: Wins only (green), Losses only (red), All trades (gray). TTP rules default to "Wins only" — losses always count, but wins only count if qualified.
  - **Clone button** on all sets (including built-in). Creates editable copy.
  - **Built-in set protections:** Lock icon, read-only rules, no delete. "Clone to customize" guidance.
  - **Rule descriptions:** Info tooltip (ⓘ) on each rule badge
  - **TQ Filter dropdown** surfaced on 6 pages:
    - My Strategies: filter row (after Status)
    - My Portfolios: filter row (after Tag)
    - Strategy Builder: inside Filter & Sort modal with explainer
    - Strategy Detail: Date Range + KPI Mode row
    - Portfolio Detail: Live Dashboard banner (next to Planned/Executed toggle)
    - Portfolio New: toggle above KPIs (when requirement set selected)
  - User picks a requirement set → all TQ rules applied as trade filters. "None" = unfiltered.
  - **Inline editing UX** from V3 preserved: expandable cards, click-to-edit values, inline add/delete

### Alerts & Signals — LOCKED: V5 (Phase 39)
- **Decision:** V5 with 4 tabs, engine admin moved to Connections.
- **Key decisions:**
  - **Engine status strip** at top (read-only, non-admin): running dot, uptime, ticks, symbols, strategies, last update
  - **Strategy Alerts tab:** Strategy + Type filter dropdowns. Two tables: Chronological Alert Feed (Time, Type, Strategy, Symbol, Price, Exec, Event, P&L, Status) + Alert History Entry/Exit Pairs (paired trades with webhook delivery status codes)
  - **Portfolio Alerts tab:** Portfolio filter dropdown. Same two-table pattern: Chronological Feed (Time, Portfolio, Strategy, Event, Price, Qty, Matched/Phantom) + Entry/Exit Pairs (Portfolio, Qty, Entry/Exit webhook status, P&L in R and $)
  - **Outbound Webhooks tab:** 4 summary KPIs (Total Sent, Success Rate, Avg Latency, Failed). Delivery log table with inline-expanding payload view per row (View/Hide toggles a full-width payload row beneath)
  - **Inbound Webhooks tab:** Placeholder with endpoint URL format, header reference, example payload. To be designed further during implementation.
  - **Admin controls** (Restart, Disable Monitoring, logs terminal) moved to Settings > Connections V5
  - **Monitored strategies module** removed (outdated, controls live on Portfolio Detail > Strategies tab via Active/Paused toggle)

### Settings/Connections — LOCKED: V5 (Engine Admin)
- **Decision:** V5 based on V4 topology diagram + Alert Engine (Ralph) admin module.
- **Key decisions:**
  - V4 network topology SVG with clickable nodes, latency labels, connection history, auto-diagnosis kept as-is
  - Added Alert Engine module: Running status badge, Restart/Disable buttons, 5 KPIs (Uptime, Ticks, Symbols, Active Strategies, Last Update), monitoring logs terminal
  - Admin-only controls separated from user-facing Alerts page

### Mass Builder — LOCKED: V6 (Strategy-Style Cards)
- **Decision:** V6 with V5 config panel + My Strategies-style result cards.
- **Key decisions:**
  - **9-tab config panel:** Tickers (presets + text input), Timeframes, Direction, Entry, Exit, TF Confluence, General, Stop Loss, Take Profit
  - **Entry/Exit tabs:** Multi-column grid grouped by Pack > Variation (parenthetical display). Trigger name shown once with inline exec type checkboxes ([C], [L], [LC], [CC]). Scrollable containers.
  - **TF Confluence tab:** State names with inline fidelity badges ([PB]/[CB]) in cyan. Select All / Clear buttons. "Fast filter" badge.
  - **General tab:** Simple checkboxes grouped by pack > variation. Select All / Clear. "Fast filter" badge.
  - **Stop Loss / Take Profit tabs:** Multi-select checkboxes grouped by pack > variation. "Full backtest each" badge.
  - **Cost indicators:** Orange "Full backtest each" on expensive tabs (Entry, Exit, Stop, Target). Green "Fast filter" on cheap tabs (TF Confluence, General).
  - **Preview + Required Performance + Analyze:** 3-panel 12-column grid. Preview compact (3 cols). Required Performance 4-column grid with all 7 fields (Prioritize By, Min Trades, Min WR%, Min PF, Min Daily R, Min R², Max Results). Analyze button (2 cols).
  - **Post-analysis filters:** 8-column row (Sort, Min WR%, Min PF, Min Trades, Min R², TQ filter, Show Passed + count)
  - **Result cards (V6):** My Strategies-style cards with: rank badge, search name in title, equity curve (S/M/L/XL heights), 2 rows × 4 KPIs (larger text), pack-aware variable display (entry/exit/stop/target/confluence badges), Save/Pass action row
  - **Viewing preferences:** Columns (1/2/3), Chart Height (S/M/L/XL), X-axis (Trade # default / Date), HWM (On/Off)
  - **Search name → strategy name:** Search name appears in result card titles and flows into saved strategy name

### Mass Results — LOCKED: V5 (Simple + Worker Status)
- **Decision:** V5 simple non-expandable cards with background worker visibility.
- **Key decisions:**
  - Non-expandable cards — View navigates to mass builder with results loaded
  - Config summary line: ticker count, TF count, directions, entries, exits, confluences, stops, targets, total evaluations
  - **Completed:** status badge with result count + 4 best KPIs (Daily R, WR, PF, R²)
  - **Running:** progress bar with percentage, current step, elapsed time, ETA. Cancel button.
  - **Queued:** status message showing what it's waiting for
  - Actions: View, Load, Copy, Cancel (running), Delete (inline confirm)
  - Sort by: Newest First, Most Results, Best Daily R

### Pack Builder — LOCKED: V7 (API-Connected, Revised Wizard)
- **Decision:** V7 API-connected wizard. V5 (original wizard) and V6 (revised prompt-based) preserved as reference.
- **Key design evolution:**
  - V5: Original 5-step wizard (Define Structure before prompt — too intimidating for users)
  - V6: Revised flow (Describe → AI generates structure → user refines → paste code → validate). Added: States rename (from Outputs), exec type params modal (⚙), Signal Validation tab, Parity Simulator with timing table, Request Fix button with surgical context
  - V7 (locked): Full AI API integration — no copy/paste anywhere. Model selector (Claude Sonnet/Opus, GPT-4/4o). Auto-fix loop (3 attempts). Request Fix sends directly to AI.
- **5-step wizard flow:**
  1. **Pack Info** — Pack type (TF Confluence / General), name, category pills, display type (TF only), description, optional Pine Script. Right panel: "How it works" numbered steps + type-specific info (exec types, fidelity, files)
  2. **Generate Structure** — 3-column: AI conversation (left) + pack summary + generated structure preview (right). AI proposes params/states/triggers from description. Regenerate button.
  3. **Refine Structure** — Editable params (name/label/type/default/min/max), states (code + description, fidelity badge for TF), triggers (name/sentiment/exec type checkboxes [C][L][LC][CC] with ⚙ params modal/fidelity selector/state transitions). General packs: [C] only, no fidelity, bool/select param types.
  4. **Generate & Validate Code** — 3-column: AI conversation (left), generated code tabs (center), 16-point validation checklist (right). Auto-Fix button sends failures back to AI (up to 3 iterations). Model selector persists.
  5. **Review & Install** — 5 tabs: Overview (summary + triggers with all badges), Chart Preview (confluence state shading + trigger markers + heatmap), Signal Validation (signal count, frequency, state coverage, per-trigger breakdown table), Parity Simulator (bar-by-bar replay chart + timing detail table + 4 parity KPIs), Code (collapsible dark terminal sections)
- **Request Fix** (Step 5): Orange button on action row. Modal with auto-captured context (active tab + pack summary) + user description. V7 sends directly to AI via API. Fix iteration counter. Surgical approach — sends only the issue + current code, not full regeneration.
- **Exec type params modal:** ⚙ icon per trigger opens modal with parameter sections per checked exec type ([C] reference_bar/order_type, [L] +hold_seconds/limit_duration, [LC] +confirm_bar_offset/bail_action, [CC] locked defaults)
- **16-point validation:** Schema (5), Safety (3), Functions (3), Execution (3), Backtest (2). Organized in collapsible categories with pass/fail/warn/pending status icons.
- **Pack types supported:** TF Confluence (3 files: manifest + indicator + interpreter, all exec types, fidelity) and General (2 files: manifest + evaluator, [C] only, binary states, no fidelity)
- **Parity Simulator:** Ticker/TF/bars selectors → Run Parity Test → bar-by-bar replay showing backtest vs live trigger markers. Timing detail table: Bar #, Timestamp, Trigger, Backtest result, Live result, Match, Delta. 4 KPIs: Total/Matched/Mismatched/Parity Score.
- **Signal Validation:** Runs pack on 90 days sample data. Total Signals, Avg Bars Between, State Coverage, All States Reached. Per-trigger breakdown table. Signal timeline chart.

### User Packs — LOCKED: V5 (Validation + Parity)
- **Decision:** V5 based on V2 with validation/parity tabs and visibility toggle.
- **Key decisions:**
  - **8-tab detail view:** Parameters, Plot Settings, States & Triggers (renamed from Outputs), Chart Preview (confluence state shading + trigger markers), Signal Validation (signal count/frequency/coverage/per-trigger breakdown), Parity Simulator (bar-by-bar replay with timing table), Code, Danger Zone
  - **Visibility toggle:** Private / Public dropdown on both list cards and detail header. Public = available to all users (admin feature, temporary until marketplace). Private = account-only.
  - **Status badges:** Pack type (TF Confluence / General), category, validation status (passed/warnings/failed), parity score, visibility
  - **Chart Preview:** Same confluence visualization as Pack Builder — background state shading, trigger arrow markers, Show Confluence / Show Triggers toggles
  - **Signal Validation & Parity Simulator:** Same as Pack Builder V7 — ongoing health checks for installed packs

### Timeframes — LOCKED: V5 (Use Case Grid)
- **Decision:** V5 clean grid table with per-use-case toggles.
- **Key decisions:**
  - **17 timeframes** (5Sec through 1Month) × **4 use case columns**: Strategy Primary, TF Confluence, Mass Builder, Chart Display
  - Checkbox toggle per cell — granular control over where each TF is available
  - **Default TF selector:** Radio buttons in grid + dropdown at top (only primary-enabled TFs)
  - **Bars/Day column:** RTH bar count for context
  - **Provider support badges:** All (green), Polygon only (accent), Stream only (muted)
  - **Sub-minute timeframes** dimmed with orange "sub-min" badge — require Polygon data feed
  - **Summary cards:** 4 cards showing enabled count per use case
  - Info note explaining sub-minute and weekly/monthly limitations

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
| 1 | **Settings (all 4)** | LOCKED (V1/V5/V4/V2) | Auth endpoints, settings CRUD |
| 2 | **Dashboard** | LOCKED (V6) | Strategy/portfolio summary endpoints |
| 3 | **TF Confluence** | LOCKED (V5) | Confluence groups CRUD, TEMPLATES registry |
| 4 | **General Packs** | LOCKED (V5) | General packs CRUD |
| 5 | **Stop Loss Packs** | LOCKED (V1) — new page, replaces Risk Management | Stop loss packs CRUD |
| 6 | **Take Profit Packs** | LOCKED (V1) — new page, replaces Risk Management | Take profit packs CRUD |
| 7 | **Strategy Builder** | LOCKED (V5) — pack selectors, method card, depth/filter system | Backtest engine endpoint, data loading, indicator computation |
| 8 | **My Strategies (list)** | LOCKED (V5) — equity curves, KPI modes, sigma badges, pack-aware variables | Strategy CRUD, stored trades |
| 9 | **Strategy Detail** | V5 IN REVIEW — 6 tabs, full alert analysis, advanced analytics | Trade history, KPI computation, chart data |

**Milestone:** Can build, save, and review strategies in Next.js.

### Phase 1B: Portfolio & Alerts
*Extend to full trading workflow — portfolios, compliance, and live alerts.*

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 9 | **Portfolio List** | LOCKED (V5) — My Strategies style, combined metrics, tag filter | Portfolio CRUD |
| 10 | **Portfolio New/Edit** | LOCKED (V5) — dual-view equity, projection KPIs, RODC, webhook template | Portfolio builder, equity computation, recommendations |
| 11 | **Portfolio Detail** | LOCKED (V5) — 6 tabs, Phase 39 webhooks, risk analytics, Planned/Executed toggle | Combined analytics, prop firm check, account ledger, webhook template selector |
| 12 | **Portfolio Requirements** | LOCKED (V5) — TQ rules, applies-to field, clone, built-in protections | Requirement set CRUD, compliance check, TQ filtering |
| 13 | **Alerts & Signals** | LOCKED (V5) — 4 tabs, inline payload expand, admin to Connections | Alert feed, webhook delivery log, entry/exit pairs |
| 14 | **Webhook Templates** | Webhook payload management | Template CRUD, test delivery |

**Milestone:** Full trading workflow works — build strategies, assemble portfolios, receive alerts, trade via webhooks.

### Phase 1C: Power User Tools
*Mass operations and remaining pack management.*

| Priority | Page | Design Notes | Backend Needed |
|----------|------|-------------|----------------|
| 15 | **Mass Builder** | LOCKED (V6) — Strategy-style cards, pack selectors, exec types, TQ filter | Mass backtest endpoint (compute-heavy) |
| 16 | **Mass Results** | LOCKED (V5) — simple cards, worker progress, queue status | Mass results CRUD |
| 17 | **Pack Builder** | LOCKED (V7) — AI-connected wizard, auto-fix, parity simulator, signal validation | Pack definition CRUD, AI API, indicator validation |
| 18 | **User Packs** | LOCKED (V5) — validation + parity tabs, visibility toggle, 8-tab detail | User pack CRUD |
| 19 | **Timeframes** | LOCKED (V5) — use case grid, default TF, provider badges | Timeframe settings CRUD |

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

## Design Iteration Progress

### COMPLETED — Phase 1A Core Trading Loop (12 pages locked)
- Settings/Themes (V1), Settings/Display (V5), Settings/Connections (V4), Settings/Account (V2)
- Dashboard (V6)
- TF Confluence (V5), General Packs (V5)
- Stop Loss Packs (V1), Take Profit Packs (V1) — new pages replacing Risk Management
- Strategy Builder (V5)
- My Strategies list (V5)
- Strategy Detail (V5) — IN REVIEW, pending user feedback

### THEN — Portfolios & Alerts (6 pages)
- Portfolio List, Portfolio New/Edit, Portfolio Detail (7 tabs)
- Portfolio Requirements, Alerts & Signals, Webhook Templates

### FINALLY — Power Tools (5 pages)
- Mass Builder, Mass Results (paired — bulk strategy discovery)
- Pack Builder, User Packs (paired — custom indicator creation + management)
- Timeframes (simple config page, lowest priority)

### Ongoing: Backend Wiring
- Each locked page gets API endpoints built and wired
- Mock data replaced with real Supabase queries
- Testing with real market data

---

## Summary

| Phase | Pages | Business Goal | Revenue |
|-------|-------|--------------|---------|
| **1A** | Settings (4), Dashboard, TF Confluence, General, Stop Loss, Take Profit, Strategy Builder, My Strategies | Core trading loop works | Personal profits |
| **1B** | Portfolios (4), Requirements, Alerts (2) | Full workflow with alerts | Personal profits |
| **1C** | Mass Builder/Results, Timeframes, User Packs, Pack Builder | Feature parity | Personal profits |
| **2** | Admin Dashboard/Curation, Profile & Roles + agent API | AI creates content | Automated portfolio creation |
| **3** | Onboarding, Pricing, Marketplace, Subscriptions, Prop Firms | First paying users | Subscriptions + affiliates |
| **4** | Creator Dashboard, Earnings, Publish, Admin Users | Creator economy | Platform fees (15-20%) |
| **5** | Mobile, international, scanner, social | Scale | All revenue streams |
