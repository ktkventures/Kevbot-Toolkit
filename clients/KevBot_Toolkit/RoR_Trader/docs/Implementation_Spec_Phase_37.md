# Phase 37: Portfolio Live Dashboard & Active Management — Implementation Spec

**Status:** Approved — Building on `main` branch
**Date:** 2026-03-19
**Backup branch:** `main-backup-pre-37`
**Plan file:** `/home/kevin/.claude/plans/sequential-fluttering-sutherland.md`

---

## 1. Overview

Transform the portfolio detail page from backtest-only analytics into an active portfolio management dashboard. Alert-based trades become the reality; backtests become the benchmark.

**Build order:** 37A → 37B → 37C → 37E → 37D → 37F → 37G (optional)

---

## 2. Phase 37A: Portfolio Alert Trade Aggregation

### 2.1 What to Build

| Function | File | Purpose |
|----------|------|---------|
| `get_portfolio_alert_trades(portfolio, get_strategy_fn)` | `portfolios.py` | Aggregate `live_executions` across all portfolio strategies |
| `compute_strategy_r_distribution(stored_trades, forward_test_start)` | `portfolios.py` | Extract backtest R stats (avg, std, var, n, median) |
| `get_portfolio_strategy_alerts_bulk(strategy_ids)` | `alerts.py` | Single-pass alert loading for a set of strategy IDs |

### 2.2 Alert Trade Record Schema

```python
{
    'trade_number': int,          # Sequential across portfolio
    'strategy_id': int,
    'strategy_name': str,
    'symbol': str,
    'direction': str,             # 'LONG' or 'SHORT'
    'entry_price': float,         # Alert price
    'exit_price': float,          # Alert price
    'theoretical_entry': float,   # FT trade entry
    'theoretical_exit': float,    # FT trade exit
    'entry_time': str,            # ISO timestamp
    'exit_time': str,             # ISO timestamp
    'exit_reason': str,           # From stored_trades
    'r_multiple': float,          # Slippage-adjusted
    'entry_slippage_r': float,
    'exit_slippage_r': float,
    'risk_per_trade': float,      # From portfolio allocation
    'dollar_pnl': float,          # r_multiple × risk_per_trade
    'quantity': int,              # risk_per_trade / per_share_risk
    'buying_power_used': float,   # quantity × entry_price
    'matched': bool,              # True if matched to FT trade
    'phantom': bool,              # True if no FT match
    'exec_type': str,             # 'C', 'L0', 'L1', 'HM', 'HL'
}
```

### 2.3 Implementation Details

**Data flow:** For each strategy in portfolio:
1. Load strategy via `get_strategy_fn(sid)`
2. Read `strategy['live_executions']` (already populated by `match_alerts_to_trades()` during strategy refresh)
3. Build entry/exit maps keyed by `matched_trade_index` (same pattern as `_compute_alert_analysis()` at app.py:8483-8494)
4. For each matched trade index with both entry AND exit:
   - Read `stored_trades[idx]` for theoretical prices, exit_reason, r_multiple
   - Compute slippage-adjusted R: `stored_r - entry_slip - exit_slip`
   - Compute dollar_pnl: `adjusted_r × allocation_risk_per_trade`
   - Compute quantity: `int(risk_per_trade / abs(entry_price - stop_price))` (guard div-by-zero)
   - Compute buying_power_used: `quantity × entry_price`
5. Detect open positions: entry exec without matching exit exec

**Return:** `{alert_trades: list, open_positions: list, strategies_with_data: set}`

### 2.4 Self-Test Checkpoints

- [ ] **A1:** Function exists, callable with a portfolio dict and `get_strategy_by_id`
- [ ] **A2:** Returns empty `{alert_trades: [], open_positions: [], strategies_with_data: set()}` for portfolio with no live_executions
- [ ] **A3:** For a strategy with live_executions, correctly pairs entry+exit into trade records
- [ ] **A4:** `dollar_pnl` = `r_multiple × risk_per_trade` for each trade
- [ ] **A5:** `quantity` and `buying_power_used` compute correctly (guard division by zero)
- [ ] **A6:** `compute_strategy_r_distribution()` returns correct avg/std for known R values
- [ ] **A7:** Open positions detected when entry exists without matching exit
- [ ] **A8:** Add temporary `st.write(result)` in portfolio detail to verify data structure visually

---

## 3. Phase 37B: Benchmark & Confidence Bands Engine

### 3.1 What to Build

| Function | File | Purpose |
|----------|------|---------|
| `compute_portfolio_benchmark(portfolio, get_strategy_fn, get_trades_fn, filter_strategy_id=None)` | `portfolios.py` | Plan line + confidence bands |
| `classify_strategy_health(actual_trades, benchmark_per_strategy, strategy_id)` | `portfolios.py` | Health classification per strategy |

### 3.2 Benchmark Algorithm

```
For each strategy in portfolio:
    bt_trades = stored_trades before forward_test_start
    avg_r = mean(bt_trades.r_multiple)
    var_r = var(bt_trades.r_multiple)
    risk = portfolio_allocation.risk_per_trade
    trade_freq = len(bt_trades) / trading_days  # trades per day

Combined:
    total_freq = sum(trade_freq for each strategy)
    For trade N in range(1, max_trades):
        expected_dollar_step = sum(strat.avg_r × strat.risk × (strat.trade_freq / total_freq))
        per_trade_var = sum((strat.risk² × strat.var_r) × (strat.trade_freq / total_freq))
        plan[N] = N × expected_dollar_step
        band_1sd[N] = 1.0 × sqrt(N × per_trade_var)
        band_2sd[N] = 2.0 × sqrt(N × per_trade_var)

When filter_strategy_id is set:
    Use only that strategy's avg_r, var_r, risk
    plan[N] = N × (strat.avg_r × strat.risk)
    band[N] = k × sqrt(N × strat.risk² × strat.var_r)
```

### 3.3 Health Classification

```python
def classify_strategy_health(actual_trades, benchmark, strategy_id):
    strat_trades = [t for t in actual_trades if t['strategy_id'] == strategy_id]
    n = len(strat_trades)
    if n < 10:
        return {'status': 'insufficient_data', 'message': 'Need 10+ alert trades', ...}

    actual_cumulative_r = sum(t['r_multiple'] for t in strat_trades)
    expected_r = benchmark['per_strategy'][strategy_id]['avg_r'] * n
    std_r = benchmark['per_strategy'][strategy_id]['std_r']
    expected_std = std_r * sqrt(n)
    deviation_sd = (actual_cumulative_r - expected_r) / expected_std if expected_std > 0 else 0

    if deviation_sd > 1.5:   return 'outperforming'
    elif deviation_sd < -1.5: return 'underperforming'
    else:                     return 'on_track'
```

### 3.4 Self-Test Checkpoints

- [ ] **B1:** `compute_portfolio_benchmark()` returns correct plan line for known avg_r values (e.g., avg_r=0.5, risk=100 → plan[10] = $500)
- [ ] **B2:** Confidence bands widen as sqrt(N) — band at trade 100 should be 10× band at trade 1
- [ ] **B3:** `filter_strategy_id` correctly isolates a single strategy's benchmark
- [ ] **B4:** `classify_strategy_health()` returns 'insufficient_data' for < 10 trades
- [ ] **B5:** Returns 'on_track' when cumulative R is within 1 SD
- [ ] **B6:** Returns 'outperforming' when > 1.5 SD above for 5+ trades
- [ ] **B7:** Returns 'underperforming' when > 1.5 SD below for 5+ trades
- [ ] **B8:** Edge case: strategy with 0 backtest trades doesn't crash (returns empty/default)

---

## 4. Phase 37C: Live Dashboard Tab — Core UI

### 4.1 What to Build

| Component | File | Purpose |
|-----------|------|---------|
| Add "Live Dashboard" tab to `render_portfolio_detail()` | `app.py:11689` | New first tab |
| `render_portfolio_live_dashboard(port, alert_data, data)` | `app.py` | Main rendering function |
| Trade detail modal (`@st.dialog`) | `app.py` | Price chart popup per trade |

### 4.2 Tab Structure

```
Live Dashboard
├── Strategy Filter selectbox
├── KPI Row (Total Trades, Win Rate, Total P&L, vs Plan Delta)
├── Alert-Based Equity Curve (Plotly)
│   ├── 2SD band (light fill)
│   ├── 1SD band (medium fill)
│   ├── Plan line (white dashed)
│   └── Actual line (color-coded by band position)
├── Trade History Table
│   ├── Columns: #, Strategy, Symbol, Dir, Entry, Exit, Reason, Qty, BP, R, P&L, Status
│   └── "View Chart" button per row → modal
└── Open Positions Summary (if any)
```

### 4.3 Key Implementation Notes

- **Cache keys:** `port_alert_data_{pid}`, `port_benchmark_{pid}_{filter_id}`
- **Refresh button:** Add these keys to the clear list at line 11625
- **Empty state:** If no alert trades, show info message with guidance
- **Modal:** Use `@st.dialog("Trade Detail", width="large")` — load bars via `load_latest_bars()` for ±50 bars around trade, render with `render_price_chart()`

### 4.4 Self-Test Checkpoints

- [ ] **C1:** "Live Dashboard" tab appears as first tab on portfolio detail page
- [ ] **C2:** Strategy filter dropdown shows "All Strategies" + each strategy name
- [ ] **C3:** Equity curve renders with plan line and confidence bands (even with 0 alert trades — shows plan only)
- [ ] **C4:** Actual line appears when alert trades exist
- [ ] **C5:** KPI row shows correct values (verified against manual calculation)
- [ ] **C6:** Trade history table populates with correct columns
- [ ] **C7:** "View Chart" button opens modal with price chart
- [ ] **C8:** Strategy filter correctly segments equity curve and table
- [ ] **C9:** Open positions section appears when strategies are IN_POSITION
- [ ] **C10:** Refresh button clears alert data cache and recomputes
- [ ] **C11:** Empty state message shown when no alert trades exist
- [ ] **C12:** Page loads without errors for portfolios with 0 strategies having live_executions
- [ ] **C13:** Page loads without errors for portfolios where all strategies have live_executions

---

## 5. Phase 37E: Account Tab Enhancements

### 5.1 What to Build

| Function | File | Purpose |
|----------|------|---------|
| `add_change_log_entry(portfolio, change_type, details, description)` | `portfolios.py` | Record portfolio changes |
| `compute_daily_journal(alert_trades, change_log)` | `portfolios.py` | Auto-generate daily summaries |
| Enhance `render_portfolio_account()` | `app.py:13233` | Change History + Daily Journal sections |
| Instrument `render_portfolio_builder()` save flow | `app.py:~11548` | Detect and log changes |

### 5.2 Change Log Entry Schema

```python
{
    'id': int,                    # Auto-assigned
    'timestamp': str,             # ISO
    'change_type': str,           # 'strategy_added', 'strategy_removed', 'risk_adjusted',
                                  # 'requirement_set_changed', 'portfolio_created'
    'details': {                  # Type-specific data
        'strategy_id': int,
        'strategy_name': str,
        'old_value': any,
        'new_value': any,
    },
    'description': str,           # Human-readable summary
}
```

### 5.3 Save Flow Instrumentation

In `render_portfolio_builder()` save logic:
```python
# Before save: capture old state
old_port = get_portfolio_by_id(portfolio_id)
old_strat_ids = {s['strategy_id'] for s in old_port.get('strategies', [])}
old_risks = {s['strategy_id']: s['risk_per_trade'] for s in old_port.get('strategies', [])}

# After computing new state:
new_strat_ids = {s['strategy_id'] for s in new_strategies}

# Detect changes
for sid in new_strat_ids - old_strat_ids:
    add_change_log_entry(port, 'strategy_added', {'strategy_id': sid, 'strategy_name': ...}, ...)
for sid in old_strat_ids - new_strat_ids:
    add_change_log_entry(port, 'strategy_removed', {'strategy_id': sid, 'strategy_name': ...}, ...)
for sid in new_strat_ids & old_strat_ids:
    if new_risks[sid] != old_risks.get(sid):
        add_change_log_entry(port, 'risk_adjusted', {'strategy_id': sid, 'old': ..., 'new': ...}, ...)
```

### 5.4 Self-Test Checkpoints

- [ ] **E1:** `add_change_log_entry()` appends to `portfolio['change_log']` with auto-ID and timestamp
- [ ] **E2:** Edit a portfolio (add strategy) → change_log entry created with type 'strategy_added'
- [ ] **E3:** Edit a portfolio (remove strategy) → 'strategy_removed' entry
- [ ] **E4:** Edit a portfolio (change risk) → 'risk_adjusted' entry with old/new values
- [ ] **E5:** Change log auto-trims to 500 entries
- [ ] **E6:** "Change History" expander renders in Account tab with correct badges
- [ ] **E7:** `compute_daily_journal()` groups alert trades by date with correct P&L
- [ ] **E8:** "Daily Journal" section renders with date navigator
- [ ] **E9:** Journal notes save and persist across sessions
- [ ] **E10:** Ledger shows portfolio_change entries inline

---

## 6. Phase 37D: Buying Power Tracker & Anomaly Detection

### 6.1 What to Build

| Function | File | Purpose |
|----------|------|---------|
| `compute_alert_buying_power(alert_trades, open_positions, account_balance)` | `portfolios.py` | Intra-trade buying power timeline |
| `detect_portfolio_anomalies(alert_trades, open_positions, portfolio, get_strategy_fn)` | `portfolios.py` | Flag problematic positions |
| Buying Power section in Live Dashboard | `app.py` | Chart + KPIs + warnings |
| Anomaly section in Live Dashboard | `app.py` | Cards with severity badges |

### 6.2 Anomaly Types

| Type | Detection | Severity |
|------|-----------|----------|
| `overexposure` | Multiple open positions on same symbol | HIGH |
| `phantom_trade` | Alert trade with `matched=False` | MEDIUM |
| `buying_power_exceeded` | quantity × entry > available BP at time | HIGH |
| `long_hold` | Open > 2× expected max hold time | MEDIUM |

### 6.3 Self-Test Checkpoints

- [ ] **D1:** `compute_alert_buying_power()` returns correct timeline for sequential trades
- [ ] **D2:** Open positions reduce current buying power
- [ ] **D3:** Insufficient capital events detected when BP goes negative
- [ ] **D4:** Buying power chart renders in Live Dashboard
- [ ] **D5:** `detect_portfolio_anomalies()` flags overexposure (2 strategies on same symbol with open positions)
- [ ] **D6:** Phantom trades flagged when `matched=False`
- [ ] **D7:** Anomaly cards render with correct severity badges
- [ ] **D8:** `buying_power_mode` selector persists to portfolio

---

## 7. Phase 37F: Strategies Tab Enhancements & Cover Webhook

### 7.1 What to Build

| Component | File | Purpose |
|-----------|------|---------|
| Three-line equity curves (BT+FW+Alert) | `app.py:12083` | Richer strategy cards |
| Health badge + recommendation | `app.py:12083` | Per-strategy health indicator |
| "View Strategy" button | `app.py:12083` | Navigate to strategy detail |
| "View Chart" button + modal | `app.py:12083` | Current price chart popup |
| `send_cover_webhook()` | `alerts.py` | Close excess positions |
| Cover webhook settings | `app.py` (Webhooks tab) | URL + template config |

### 7.2 Self-Test Checkpoints

- [ ] **F1:** Strategy cards show three-line equity curve (BT gray dashed, FW blue, Alert green)
- [ ] **F2:** Health badge renders with correct color and recommendation text
- [ ] **F3:** "View Strategy" navigates to strategy detail page
- [ ] **F4:** "View Chart" opens modal with current price chart
- [ ] **F5:** Cover webhook settings section renders in Webhooks tab
- [ ] **F6:** `send_cover_webhook()` sends payload and returns delivery result
- [ ] **F7:** "Cover" button in anomaly section calls send_cover_webhook with confirmation
- [ ] **F8:** Cover action recorded as alert with `type='manual_cover'`

---

## 8. Phase 37G: Alert Tracking Liberalization (Optional)

### 8.1 Self-Test Checkpoints

- [ ] **G1:** Strategy with `alert_tracking_enabled=True` but NOT in any portfolio still records alerts
- [ ] **G2:** Portfolio-linked strategies processed before standalone strategies
- [ ] **G3:** Non-portfolio alerts skip webhook delivery
- [ ] **G4:** `live_executions` populated for standalone strategies

---

## 9. End-to-End Verification

After all phases complete:

1. [ ] Create/use a portfolio with 2+ strategies that have `alert_tracking_enabled=True` and populated `live_executions`
2. [ ] Open Live Dashboard tab → equity curve shows plan line, bands, and actual line
3. [ ] Strategy filter correctly segments the view
4. [ ] Trade table shows all columns with correct data
5. [ ] "View Chart" on a trade opens modal with price chart showing entry/exit markers
6. [ ] Buying power chart shows intra-trade timeline
7. [ ] Anomalies detected and displayed (if applicable)
8. [ ] Account tab shows Change History with recent edits
9. [ ] Daily Journal auto-populates from alert trades
10. [ ] Strategies tab shows three-line equity curves and health badges
11. [ ] "View Strategy" navigates correctly
12. [ ] "View Chart" on strategy card opens modal
13. [ ] No errors on portfolios with 0 alert data (graceful empty states)
14. [ ] Refresh button clears all new caches and recomputes
15. [ ] Page performance acceptable (< 3s load time for typical portfolio)
