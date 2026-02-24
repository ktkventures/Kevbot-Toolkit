# Phase 26: Portfolio Risk Intelligence & Balance-Aware Execution

## Overview

Phase 26 enhances the portfolio system with three capabilities:
1. **Risk analytics** that answer "what are my chances of busting?" with Monte Carlo simulation and historical worst-case analysis
2. **Balance-aware quantity sizing** that ensures webhooks only fire quantities the account can actually afford
3. **Compliance-driven position management** that can auto-close positions and suppress entries when a portfolio breaches its rule set

All changes are additive — existing portfolio functionality (equity curves, drawdown chart, prop firm check, account ledger, webhook delivery) is preserved.

---

## What Already Exists

| Component | Status | Location |
|-----------|--------|----------|
| Requirement sets (TTP, FTMO, custom) | Complete | `portfolios.py` lines 22-54, `requirements.json` |
| `evaluate_requirement_set()` | Complete | `portfolios.py` lines 570-593 |
| Rule types: profit target, max daily loss, max DD, min profitable days, min trading days | Complete | `portfolios.py` lines 506-567 |
| Drawdown chart with max DD threshold overlay | Complete | `app.py` lines 9116-9150 |
| Daily P&L histogram | Complete | `app.py` lines 9155-9166 |
| Strategy correlation heatmap | Complete | `app.py` lines 9168-9181 |
| Account ledger (deposits, withdrawals, auto trading P&L) | Complete | `portfolios.py` lines 707-806 |
| Balance history chart | Complete | `app.py` lines 10177-10194 |
| Compliance breach webhook event type | Complete | `alerts.py` line 74, `alert_monitor.py` line 312 |
| Webhook delivery routing per-portfolio | Complete | `alert_monitor.py` lines 295-366 |
| Quantity calculation: `risk / |entry - stop|` | Complete | `alerts.py` lines 109-156 |

---

## Sub-Phase A: Enhanced Risk Analytics — COMPLETE

**Goal**: Charts and simulations that answer "given my portfolio's historical performance, what's the probability of hitting my max loss / daily pause threshold?"

**Status**: All items complete and deployed. All features shared between Performance tab and Portfolio Builder via `_render_risk_analytics()` helper.

### A1: Daily Drawdown Chart — COMPLETE

The existing drawdown chart (app.py line 9116) shows **cumulative peak-to-trough drawdown %** — addresses `max_total_drawdown_pct`. We need a complementary chart for **daily P&L** to address `max_daily_loss_pct` and `daily_pause_pct`.

**What to build:**
- New chart on the Performance tab (below existing drawdown chart): "Daily P&L vs Limits"
- X-axis: trading days
- Y-axis: daily P&L as % of starting balance
- Plot: bar chart of each day's P&L %
- Overlay lines:
  - `max_daily_loss_pct` threshold (red dashed) — from portfolio's requirement set
  - `daily_pause_pct` threshold (orange dashed) — if the requirement set has one
- Highlight bars that breach either threshold in red/orange
- Caption: count of days that would have breached each threshold

**Data source**: `daily_pnl` DataFrame already computed in `calculate_portfolio_kpis()` — has `date` and `daily_pnl` columns. Convert to % via `daily_pnl / starting_balance * 100`.

**Requirement set enhancement**: Add `daily_pause_pct` as a new rule type. TTP uses this (different from max daily loss — it pauses but doesn't disqualify). Add to `PROP_FIRM_RULES["ttp"]` and the evaluation engine.

**New rule type to add to `portfolios.py`:**
```python
{"name": "Daily Pause", "type": "daily_pause_pct", "value": 1.5,
 "description": "Pause trading if daily loss exceeds 1.5%"}
```

**Evaluation logic** (add to `_evaluate_single_rule`):
```python
# daily_pause_pct — same logic as max_daily_loss_pct but semantic difference:
# daily_pause = soft limit (pause, resume next day)
# max_daily_loss = hard limit (potential disqualification)
worst_day_pct = abs(daily_pnl['daily_pnl'].min()) / starting_balance * 100
passed = worst_day_pct <= rule_value
```

**Files**: `app.py` (chart rendering), `portfolios.py` (new rule type + evaluation)

---

### A2: Historical Worst-Case Analysis — COMPLETE

**What to build:**
- Expandable section on Performance tab: "Worst-Case Analysis"
- Table/metrics showing:
  - Worst single-day loss ($ and %)
  - Worst consecutive losing streak (days and total $ loss)
  - Worst N-day rolling drawdown
  - Number of days that would have triggered daily pause
  - Number of days that would have triggered max daily loss
- "Top 5 Worst Days" table: date, daily P&L ($), daily P&L (%), cumulative DD at that point, breach status per rule

**Data source**: `daily_pnl` and `combined_trades` DataFrames already available.

**Files**: `app.py` (new section in portfolio Performance tab)

---

### A3: Monte Carlo Simulation — COMPLETE

**What to build:**
- Expandable section on Performance tab: "Risk Simulation"
- Takes the portfolio's historical trade results and randomizes the order N times
- **Shuffle mode** (user selectable):
  - **Daily blocks** (default) — preserves intraday trade correlation, shuffles the order of entire trading days
  - **Weekly blocks** — preserves weekly correlation
  - **Individual trades** — widest distribution, breaks all time correlation
- **Simulation parameters**:
  - Number of simulations: default 1,000 (user can adjust via slider, e.g., 500–5,000)
  - Shuffle mode selector
- **For each simulation run**:
  - Track cumulative P&L, peak, max drawdown, worst daily loss
  - Check if max DD breaches `max_total_drawdown_pct`
  - Check if any day breaches `max_daily_loss_pct` / `daily_pause_pct`
- **Output**:
  - **Bust probability**: "X% of simulations breached max total drawdown"
  - **Daily pause probability**: "X% of simulations had at least one day breaching daily pause"
  - **Distribution chart**: histogram of max drawdown values across all simulations, with threshold line overlay
  - **Confidence bands**: equity curve with 5th/25th/50th/75th/95th percentile bands
  - Key metrics: median max DD, 95th percentile max DD, expected worst day
- "Run Simulation" button (not auto-run — computationally heavy)
- Cache results in session state keyed by portfolio ID + data refresh timestamp + shuffle mode

**Implementation approach:**
- Pure Python/NumPy for performance
- Daily blocks: group trades by date, shuffle the date order, recompute running stats
- Vectorized where possible: pre-allocate result arrays
- For 500 trades x 1,000 sims, expect <2 seconds

**Files**: `app.py` (UI), `portfolios.py` (new `run_monte_carlo()` function)

---

### A4: Profit Target Progress — COMPLETE

**What to build:**
- Progress indicator on the Prop Firm Check tab
- Shows current P&L as % of starting balance vs the `min_profit_pct` target
- Visual: progress bar with green fill
- Caption: "X.X% of Y.Y% target (estimated Z days remaining at current avg daily P&L)"
- If no profit target rule in requirement set, section hidden

**Data source**: `kpis['total_pnl']` and `starting_balance`.

**Files**: `app.py` (Prop Firm Check tab enhancement)

---

### A5: Capital Utilization & Buying Power — COMPLETE

**What was built:**
- `compute_capital_utilization()` in `portfolios.py`: computes a timeline of capital deployed in open positions and available buying power over the portfolio's trade history
- For each trade: `quantity = floor(scaled_risk / risk)`, `capital = quantity × entry_price`
- Events at entry (capital locked) and exit (capital returned + P&L realized)
- Area chart in `_render_risk_analytics()` showing available buying power over time with starting balance and zero reference lines
- Orange dotted line showing capital deployed in open positions
- Metrics: Peak Capital Deployed, Min Buying Power, Max Concurrent Positions
- Warning banner when buying power goes negative (insufficient capital)
- Shared between Performance tab and Portfolio Builder

**Files**: `portfolios.py` (`compute_capital_utilization()`), `app.py` (chart in `_render_risk_analytics()`)

---

### Additional: Strategy Recommendations Performance Fix — COMPLETE

**Problem**: "Analyze Recommendations" in Portfolio Builder loaded indefinitely because each candidate strategy triggered `get_strategy_trades()` → `prepare_forward_test_data()` with API calls.

**Fix**:
- `_get_fast_strategy_trades(strat)`: uses `stored_trades` from strategy objects (instant, no API calls) via existing `_trades_df_from_stored()` pattern
- Falls back to session cache, returns `None` if neither available
- Two-pass approach before running analysis: pre-cache from stored_trades, then fetch remaining with visible progress bar
- Recommendations now complete near-instantly when strategies have been recently updated

**Files**: `app.py` (`_get_fast_strategy_trades()`, updated recommendations flow)

---

### Shared Helper: `_render_risk_analytics()` — COMPLETE

All Sub-Phase A features (A1, A2, A3, A5) are rendered through a shared helper function `_render_risk_analytics(port, data, key_prefix)` that is called from both:
- `render_portfolio_performance()` (Performance tab) with `key_prefix="perf"`
- `render_portfolio_builder()` (Builder page) with `key_prefix="builder"`

This avoids code duplication and ensures both views stay in sync. The `key_prefix` parameter prevents Streamlit widget key collisions.

---

## Alert Pipeline Reliability Fixes — COMPLETE

Discovered and fixed during live alert testing between Sub-Phase A and Sub-Phase B.

### Fix 1: Spurious Alerts on Monitoring Startup — COMPLETE

**Problem**: Alerts fired within the first few minutes of starting monitoring that didn't correspond to any trade in the forward test history. Intra-bar triggers (VWAP cross, EMA cross) would fire immediately on the first live tick.

**Root cause**: `TriggerLevelCache.update_from_indicators()` seeded `prev_side` from the last warmup bar's close price (potentially hours old). The first live tick at a different price was detected as a false "crossing" — a gap between stale warmup data and the current market price, not a real signal.

**Fix**: Added `_first_bar_closed` flag to `SymbolHub.__init__()`. `_check_intrabar_triggers()` returns immediately if the flag is `False`. The flag is set to `True` on the first real `_on_bar_close()` call (~60 seconds max suppression). This ensures intra-bar triggers only fire after the trigger cache has been properly initialized from a real completed bar.

**Files**: `realtime_engine.py` (`SymbolHub.__init__`, `_check_intrabar_triggers`, `_on_bar_close`)

---

### Fix 2: Bar-Count Managed Exits Never Firing — COMPLETE

**Problem**: Strategies with bar-count-based exits (e.g., "exit after N bars") never triggered their managed exit because `bars_held` was always computed as ~0.

**Root cause**: `entry_bar_count` was set using `len(builder.history)`, which is capped at `MAX_HISTORY = 500` bars. Once history reached 500, both entry and current bar counts hovered near 500, so `bars_held = current - entry ≈ 0`. Positions got stuck open indefinitely.

**Fix**: Changed all 4 locations from `len(builder.history)` to `builder._bar_count` (a monotonically increasing counter that is never capped):
1. `_init_position_state()` — warmup entry tracking
2. `_check_intrabar_triggers()` — intra-bar entry tracking
3. `_on_bar_close()` — bar-close entry tracking
4. `_check_managed_exits()` — bar-count exit comparison

**Files**: `realtime_engine.py` (4 locations)

---

### Fix 3: Trigger Timing Delta Accuracy — COMPLETE

**Problem**: The trigger timing analysis on the Alerts tab showed bar-close alerts with ~60-second deltas, making it appear alerts were firing a full minute late. In reality, latency was ~2 seconds.

**Root cause**: Trade timestamps use the bar period START time (e.g., 13:44:00 for the 13:44-13:45 bar), but bar-close alerts fire at the bar CLOSE (~13:45:01). The raw delta of ~61 seconds was misleading — 59 seconds is the bar period, and ~2 seconds is actual latency.

**Fix**: For bar-close triggers, subtract `_bar_period_s` (looked up from `TIMEFRAME_SECONDS` based on the strategy's timeframe) from the raw delta. Intra-bar trigger deltas are unaffected since they fire at the actual crossing time.

**Files**: `app.py` (trigger timing analysis section)

---

## Sub-Phase B: Balance-Aware Quantity Sizing

*To be reviewed and approved before implementation. Touches the webhook pipeline.*

**Goal**: Ensure webhook quantities don't exceed what the account can actually buy.

### B1: Compute Available Balance from Ledger

- New function: `get_available_balance(portfolio) -> float`
- Sums all ledger entries (deposits - withdrawals + trading P&L)
- Subtracts estimated capital in open positions: `sum(risk_per_trade for each strategy with an open position in this portfolio)` as a conservative reserve
- Returns estimated buying power
- Uses existing `compute_account_balance()` as foundation

**Files**: `portfolios.py`

### B2: Quantity Cap in Webhook Payload

- Modify `build_placeholder_context()` in `alerts.py` to accept optional `max_buying_power`
- Cap: `quantity = min(risk_based_quantity, floor(buying_power / entry_price))`
- If `max_affordable < 1`: set quantity to 0, flag as `skipped_reason: "insufficient_balance"`

**Files**: `alerts.py`

### B3: Portfolio Setting: Auto-Adjust Sizing

- New boolean field on portfolio: `auto_adjust_sizing` (default: false)
- Toggle in portfolio edit form or Deploy tab
- When enabled: `deliver_alert()` computes available balance and passes to `build_placeholder_context()`
- When disabled: current behavior (quantity = risk / stop distance)
- Warning banners on Performance and Deploy tabs when:
  - Any strategy's risk_per_trade exceeds available balance
  - Estimated total concurrent capital needs exceed balance

**Files**: `portfolios.py` (schema), `app.py` (toggle + warnings), `alert_monitor.py` (balance lookup)

### B4: Insufficient Balance Alert

- When `auto_adjust_sizing` is enabled and balance insufficient:
  - quantity >= 1: fire with reduced quantity, add `adjusted_quantity: true` to alert
  - quantity < 1: skip webhook, save alert with `skipped_reason: "insufficient_balance"`
- Display in Alerts tab: "Reduced" / "Skipped" badge

**Concurrent entry handling**: First-come, first-served. If two strategies trigger simultaneously, the first to reach `deliver_alert()` gets the full available balance. The second sees reduced balance. In the extremely rare case of truly simultaneous triggers, prefer the strategy with the better profit factor. This is a tie-breaker only — we do not add latency to the delivery path.

**Files**: `alert_monitor.py`, `alerts.py`, `app.py`

---

## Sub-Phase C: Portfolio-Specific Compliance Actions

*To be reviewed and approved before implementation. High impact — modifies alert delivery behavior.*

**Goal**: When a portfolio breaches a rule, auto-close positions and suppress entries — scoped to that portfolio only.

### C1: Portfolio-Scoped Webhook Suppression

**Key constraint**: A strategy can belong to multiple portfolios. If Portfolio A breaches max loss, only Portfolio A's webhooks stop. Portfolio B (same strategy) continues.

**Architecture**: `deliver_alert()` already routes per-portfolio. Add a gate:
- New fields on portfolio: `compliance_paused`, `compliance_paused_at`, `compliance_pause_reason`, `compliance_paused_until`
- In `deliver_alert()`: if `compliance_paused` is true for a portfolio, skip entry webhooks for that portfolio (exits always fire)
- Auto-resume logic:
  - **Daily pause breach**: `compliance_paused_until` = next market open (9:30 ET). Auto-clears when `now > paused_until`.
  - **Max DD breach**: `compliance_paused_until` = null → stays paused until manual resume.

**Files**: `portfolios.py` (new fields), `alert_monitor.py` (gate logic)

### C2: Compliance Check on Each Trade Exit

After each exit webhook delivery (auto P&L ledger entry generated):
1. Load portfolio, compute balance from ledger
2. Compute today's P&L from today's ledger entries
3. Evaluate `max_daily_loss_pct` and `daily_pause_pct` against today's P&L
4. Evaluate `max_total_drawdown_pct` against cumulative balance
5. On breach:
   - Set `compliance_paused = True`
   - Generate `compliance_breach` alert with rule details
   - Fire compliance breach webhooks (to close positions at broker)
   - For all strategies in this portfolio with open positions: fire per-strategy close alerts with correct symbol + quantity

**Runs in delivery thread** (background) — must be wrapped in try/except, must never block or crash the engine, must never suppress exit webhooks.

**Ledger as source of truth**: Use webhook-exit P&L (ledger) for compliance evaluation, not backtest KPIs. This is closer to the actual account state since it includes real slippage.

**Files**: `alert_monitor.py` (compliance check), `portfolios.py` (evaluation helpers)

### C3: Compliance Pause UI

- Deploy tab: orange/red banner showing pause status and reason
- "Resume Trading" button (clears `compliance_paused`)
- Portfolio card: "Paused" badge
- Performance tab: vertical line + annotation on equity curve at breach point

**Files**: `app.py`

### C4: Close-All Webhook Delivery

On compliance breach, fire individual per-strategy close alerts:
- For each strategy in the portfolio with an open position (from `_position_state` in streaming engine)
- Generate an exit signal alert with the strategy's symbol, direction, and current quantity
- Route through existing webhook delivery (uses exit webhook template)
- Also fire a portfolio-level compliance breach alert (uses compliance webhook template)

**Files**: `alert_monitor.py`

---

## Sub-Phase D: Future — Advanced Prop Firm Rules

Deferred to a later phase. Documented for planning.

| Feature | Rule Type | Notes |
|---------|-----------|-------|
| Min trade duration | `min_trade_duration_sec` | Post-trade validation only (can't enforce at entry). TTP requires 10 seconds. |
| Min profit per share | `min_profit_per_share` | TTP requires $0.10/share. Trades below this don't count toward profit target but still count as loss. |
| Position sizing limits | `max_position_shares`, `max_position_dollars`, `max_position_pct` | Cap quantity at webhook time. |
| Alpaca API integration | N/A | Pull real-time balance, positions, order history. Reconcile with ledger. Requires API key in Settings > Connections. |
| Multi-account architecture | N/A | Portfolio maps to a specific trading account. Each has own balance, rules, endpoints. |

---

## Implementation Priority & Dependencies

```
Sub-Phase A: Risk Analytics — COMPLETE
  A1: Daily Drawdown Chart ✓
  A2: Worst-Case Analysis ✓
  A3: Monte Carlo Simulation ✓
  A4: Profit Target Progress ✓
  A5: Capital Utilization & Buying Power ✓
  Shared helper: _render_risk_analytics() ✓
  Recommendations performance fix ✓

Alert Pipeline Reliability Fixes — COMPLETE
  Fix 1: Spurious startup alerts (_first_bar_closed gate) ✓
  Fix 2: Bar-count managed exits (builder._bar_count) ✓
  Fix 3: Trigger timing delta accuracy (bar period offset) ✓

Sub-Phase B: Balance-Aware Sizing
  B1: Available Balance from Ledger ← uses existing account system
  B3: Auto-Adjust Setting ← portfolio schema change
  B2: Quantity Cap ← modifies webhook pipeline (careful)
  B4: Insufficient Balance Alert ← depends on B2 + B3

Sub-Phase C: Compliance Actions (review after B is complete)
  C1: Portfolio-Scoped Suppression ← modifies deliver_alert
  C2: Compliance Check per Trade ← depends on C1 + rule evaluation
  C3: Pause UI ← depends on C1 + C2
  C4: Close-All Template ← depends on C2
```

**Recommended next**: B1 → B3 → B2 → B4 → (review) → C1 → C2 → C3 → C4

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| A1-A5 (charts/analytics) | Low | Read-only, no pipeline impact — DEPLOYED |
| B2 (quantity cap) | Medium | Only active when `auto_adjust_sizing` explicitly enabled by user |
| B4 (skip webhook) | Medium | Alert still saved with `skipped_reason`, visible in UI for audit |
| C1 (suppress entries) | High | Only when rule actually breached; exits always fire; manual resume available |
| C2 (auto-compliance check) | High | Runs in try/except in delivery thread; never suppresses exits; logs all errors |
| C4 (close-all) | High | Only fires if compliance breach webhook explicitly enabled for that portfolio |

---

## Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Daily pause auto-resume? | Auto-resume at next market open for daily pause; manual resume for max DD | Daily pause is a soft limit (TTP pauses, doesn't disqualify); max DD is potentially account-ending |
| Close alerts on breach? | Fire close alerts for ALL strategies with open positions in that portfolio; suppress next entries | Prop firms don't care which strategy caused the loss — the account is the unit of risk |
| Which P&L for compliance? | Ledger P&L (webhook exits + deposits/withdrawals) | Closer to actual account state; includes real slippage |
| Monte Carlo shuffle mode? | User-selectable: daily blocks (default), weekly blocks, individual trades | Daily blocks preserve intraday correlation (time-of-day matters); options give flexibility |
| Concurrent entry buying power? | First-come, first-served; tie-break by profit factor | Don't add latency to delivery path; simultaneous triggers are extremely rare |
| Account balance source? | Existing ledger system (not Alpaca API) | Users may be on different platforms; API integration deferred to Phase D |
