# RoR Trader — Implementation Spec: Phases 11–14

**Version:** 0.2
**Date:** February 13, 2026
**Purpose:** Detailed, autonomous-implementation-ready spec for Phases 11–14. Designed for a "Ralph Wiggum" loop — each phase can be implemented without user validation between steps.

**Reference Images:** `docs/reference_images/DaviddTech *.png` (5 screenshots)

---

## Table of Contents

1. [Phase 11: Analytics & Edge Detection](#phase-11-analytics--edge-detection)
2. [Phase 12: Webhook Inbound Strategy Origin](#phase-12-webhook-inbound-strategy-origin)
3. [Phase 13: Live Alerts Validation](#phase-13-live-alerts-validation)
4. [Phase 14: Live Portfolio Management](#phase-14-live-portfolio-management)
5. [Cross-Phase Dependencies](#cross-phase-dependencies)
6. [Shared Conventions](#shared-conventions)

---

## Shared Conventions

### File Map

| File | Purpose | Approx Lines |
|------|---------|-------------|
| `src/app.py` | Main Streamlit app — all pages, tabs, UI | ~9,300 |
| `src/triggers.py` | Trade generation engine (`generate_trades`) | ~700 |
| `src/indicators.py` | Technical indicator calculations | ~440 |
| `src/alerts.py` | Alert detection, webhook delivery, templates | ~960 |
| `src/confluence_groups.py` | Confluence pack system, templates, interpreters | ~1,200 |
| `src/mock_data.py` | Mock OHLCV data generator | ~200 |
| `src/data_loader.py` | Alpaca/mock data loading | ~270 |
| `config/strategies.json` | Persisted strategy data | |
| `config/alert_config.json` | Alert monitoring & webhook config | |
| `config/alerts.json` | Fired alert history (last 500) | |
| `config/settings.json` | User settings | |

### Existing Data Structures

**Strategy dict** (persisted in `strategies.json`):
```python
{
    "id": 1,
    "name": "EMA Cross SPY",
    "symbol": "SPY",
    "direction": "LONG",
    "timeframe": "1Min",
    "entry_trigger": "cross_bull",
    "entry_trigger_confluence_id": "ema_stack_default_cross_bull",
    "exit_triggers": ["cross_bear"],
    "exit_trigger_confluence_ids": ["ema_stack_default_cross_bear"],
    "confluence": ["macd_default"],
    "general_confluences": ["GEN-rvol_default"],
    "stop_config": {"method": "atr", "atr_mult": 1.5},
    "target_config": {"method": "risk_reward", "rr_ratio": 2.0},
    "risk_per_trade": 100.0,
    "starting_balance": 10000.0,
    "data_days": 30,
    "strategy_origin": "standard",
    "kpis": { ... },
    "stored_trades": [{"entry_time": "...", "exit_time": "...", "r_multiple": 1.5, "win": true}, ...],
    "equity_curve_data": {"exit_times": [...], "cumulative_r": [...], "boundary_index": 42},
    "forward_test_start": "2026-02-01T12:00:00",
    "forward_testing": true,
    "alerts": false,
    "created_at": "...",
    "updated_at": "..."
}
```

**stored_trades record** (minimal, 4 fields):
```python
{"entry_time": "ISO", "exit_time": "ISO", "r_multiple": float, "win": bool}
```

**Alert record** (in `alerts.json`):
```python
{
    "id": 1, "type": "entry_signal"|"exit_signal"|"compliance_breach",
    "symbol": "SPY", "strategy_id": 1, "strategy_name": "...",
    "direction": "LONG", "price": 450.25, "stop_price": 448.50,
    "trigger": "cross_bull", "confluence_met": [...],
    "timestamp": "ISO", "acknowledged": false,
    "webhook_deliveries": [{"sent_at": "ISO", "webhook_id": "...", "success": true, ...}]
}
```

**Current KPIs** (from `calculate_kpis`, app.py L860-922):
```python
{
    "total_trades": int, "win_rate": float, "profit_factor": float,
    "avg_r": float, "total_r": float, "daily_r": float,
    "r_squared": float, "max_r_drawdown": float,
    "final_balance": float, "total_pnl": float
}
```

### Style Rules
- **No new files** unless structurally necessary (new module > 200 lines).
- **Streamlit patterns:** Use `st.tabs()` for sub-views, `st.columns()` for layouts, `st.metric()` for KPI cards, `st.plotly_chart()` for charts.
- **Chart library:** Two libraries in use. **Plotly** (`go.Figure`, `go.Scatter`, `go.Bar`, `go.Histogram`) for equity curves, KPI charts, analytics, and distribution plots. **TradingView Lightweight Charts** (via `streamlit-lightweight-charts`) for candlestick price charts with indicator overlays and synced oscillator panes. For Phases 11–14, all new charts are Plotly (no new candlestick/price charts). Mini equity curves use `render_mini_equity_curve()`.
- **Persistence:** JSON files in `config/`. Use existing `load_strategies()`/`save_strategies()` pattern.
- **KPI display:** `_display_kpi_card(label, value, fmt)` helper for consistent formatting.
- **Testing:** Run `cd src && python -c "import app"` as syntax check after major changes.

---

## Phase 11: Analytics & Edge Detection

**Goal:** Expand the analytics capabilities of strategy detail pages with advanced KPIs, equity curve health overlays, rolling metrics, distribution analysis, and Markov-chain edge detection. Inspired by DaviddTech's "Strategy Analysis & Data" dashboard.

**Visual Reference:**
- `DaviddTech Performance Metrics.png` — KPI grid layout, Trades per Day chart
- `DaviddTech Equity Curve Analysis.png` — Edge Check toggle (21-MA + Bollinger Bands), backtest vs live segments
- `DaviddTech Markov Motor Analysis pt 1.png` — Rolling metrics, transition probabilities, streak chart
- `DaviddTech Markov Motor Analysis pt 2.png` — Edge decay, return distribution, market regime
- `DaviddTech Markov Motor Analysis pt 3.png` — Markov Intelligence Insights (text summaries)

### 11.1 Expanded KPI Panel

**Location:** `app.py` — `calculate_kpis()` (L860-922) and `render_strategy_detail()` KPI display section.

**New KPIs to Add to `calculate_kpis()`:**

Add these to the returned dict. All are computed from the `trades_df` (r_multiple series) and `starting_balance`/`risk_per_trade` inputs.

| KPI | Formula / Logic | Category |
|-----|----------------|----------|
| `sharpe_ratio` | mean(daily_r) / std(daily_r) × √252. Use R-multiples grouped by trading day. | Risk-Adjusted |
| `sortino_ratio` | mean(daily_r) / downside_std(daily_r) × √252. Downside deviation = std of negative daily returns only. | Risk-Adjusted |
| `calmar_ratio` | annualized_return / max_drawdown. Annualized = total_r / trading_years × risk_per_trade / starting_balance. | Risk-Adjusted |
| `kelly_criterion` | win_rate - (1 - win_rate) / payoff_ratio. Payoff ratio = avg_win / avg_loss. Cap at 0.0-1.0 range. | Sizing |
| `daily_var_95` | 5th percentile of daily R returns × risk_per_trade. (Parametric or historical — use historical percentile for simplicity.) | Risk |
| `cvar_95` | Mean of daily returns below VaR threshold × risk_per_trade. Also called Expected Shortfall. | Risk |
| `max_consec_wins` | Longest streak of consecutive winning trades. | Streaks |
| `max_consec_losses` | Longest streak of consecutive losing trades. | Streaks |
| `gain_pain_ratio` | total_gain / abs(total_loss). Same as profit_factor conceptually — DaviddTech shows monthly variant too. | Risk-Adjusted |
| `payoff_ratio` | avg_win_r / abs(avg_loss_r). Also called "Reward to Risk Ratio". | Risk-Adjusted |
| `common_sense_ratio` | profit_factor × (1 - 1/total_trades). Penalizes small sample sizes. | Risk-Adjusted |
| `tail_ratio` | abs(95th percentile return) / abs(5th percentile return). > 1 means fatter right tail (good). | Distribution |
| `outlier_win_ratio` | max_win_r / avg_win_r. How dependent on outlier wins. | Distribution |
| `outlier_loss_ratio` | abs(max_loss_r) / abs(avg_loss_r). How dependent on outlier losses. | Distribution |
| `recovery_factor` | total_r / abs(max_r_drawdown). How quickly equity recovers. | Drawdown |
| `ulcer_index` | sqrt(mean(drawdown²)). Measures depth and duration of drawdowns. | Drawdown |
| `serenity_index` | total_r / ulcer_index. Builds on R² from Phase 8. Higher = smoother growth. | Drawdown |
| `skewness` | scipy.stats.skew(r_multiples) or manual calc. | Distribution |
| `kurtosis` | scipy.stats.kurtosis(r_multiples). Excess kurtosis (normal = 0). | Distribution |
| `expected_daily` | mean(daily_r) × risk_per_trade. Dollar expectation per day. | Returns |
| `expected_monthly` | expected_daily × 21. | Returns |
| `expected_yearly` | expected_daily × 252. | Returns |
| `volatility` | std(daily_returns) × √252 × 100. As percentage. | Risk |
| `longest_dd_days` | Longest duration (in trading days) spent in drawdown. | Drawdown |

**Implementation Notes:**
- Group daily returns by trading day: `trades_df.groupby(trades_df['exit_time'].dt.date)['r_multiple'].sum()`.
- For `ulcer_index`: compute running drawdown from cumulative R, square each value, take mean, then sqrt.
- If fewer than 5 trades, set ratio-based KPIs to `None` (avoid division by zero / meaningless stats).
- `scipy` should NOT be a new dependency — implement skewness/kurtosis manually with numpy: `skew = n/(n-1)/(n-2) * sum(((x-mean)/std)³)`, `kurt = ...`.

**KPI Display — Strategy Detail Page:**

Currently KPIs are shown as metrics in `render_strategy_detail()`. Reorganize into a tabbed or expandable layout:

```
┌─────────────────────────────────────────────────┐
│  Performance   │  Risk-Adjusted  │  Distribution │
├─────────────────────────────────────────────────┤
│ Total Trades   │ Sharpe Ratio    │ Skewness      │
│ Win Rate       │ Sortino Ratio   │ Kurtosis      │
│ Profit Factor  │ Calmar Ratio    │ Tail Ratio    │
│ Avg R          │ Kelly Criterion │ Outlier Win   │
│ Total R        │ Gain/Pain       │ Outlier Loss  │
│ Daily R        │ Payoff Ratio    │               │
│ Total P&L      │ Common Sense    │               │
│ Final Balance  │                 │               │
├─────────────────────────────────────────────────┤
│  Risk          │  Drawdown       │  Streaks      │
├─────────────────────────────────────────────────┤
│ Daily VaR 95%  │ Max DD (R)      │ Max Consec W  │
│ CVaR 95%       │ Recovery Factor │ Max Consec L  │
│ Volatility     │ Ulcer Index     │               │
│ Exp. Daily     │ Serenity Index  │               │
│ Exp. Monthly   │ R²             │               │
│ Exp. Yearly    │ Longest DD Days │               │
└─────────────────────────────────────────────────┘
```

Use `st.tabs(["Performance", "Risk-Adjusted", "Distribution", "Risk", "Drawdown", "Streaks"])` inside the KPI display area. Each tab shows a 2-3 column grid of `st.metric()` cards.

**KPI Card Strategy Cards (unchanged):** The strategy list cards continue showing the current compact set (Win Rate, PF, Avg R, Total R, Max DD, R²). The expanded KPIs are strategy-detail-only.

### 11.2 Edge Check Overlay on Equity Curve

**Location:** `app.py` — equity curve chart in `render_strategy_detail()` (currently ~L3067-3101).

**Behavior:**
- Add a toggle checkbox: `st.checkbox("Edge Check", key=f"edge_check_{strategy_id}")` above the equity curve chart.
- When enabled, overlay:
  - **21-period MA** of the cumulative R series (trade-indexed, not time-indexed).
  - **Bollinger Bands** (21-period MA ± 2× rolling std of cumulative R).
- Use the trade-indexed x-axis (trade number, not timestamp) when Edge Check is active. This matches DaviddTech's approach.
- Color scheme: MA line = dark olive/green (#808000), BB fill = semi-transparent yellow.

**Implementation:**
```python
if edge_check:
    cum_r = equity_df["cumulative_r"]
    ma_21 = cum_r.rolling(window=21, min_periods=1).mean()
    std_21 = cum_r.rolling(window=21, min_periods=1).std().fillna(0)
    bb_upper = ma_21 + 2 * std_21
    bb_lower = ma_21 - 2 * std_21
    # Add MA line trace
    # Add BB upper/lower as filled area
```

**Interpretation text** (show below chart when Edge Check is on):
> "When the equity curve drops below the lower Bollinger Band, performance is statistically unusual — the strategy's edge may be degrading."

### 11.3 Rolling Performance Metrics

**Location:** New section in strategy detail page, below the equity curve. Add as a new tab or an expander.

**UI:**
- Toggle buttons (use `st.segmented_control` or `st.radio` horizontal): **Win Rate** | **Profit Factor** | **Sharpe**
- Slider for rolling window size: 10–100 trades (default 20).
- Plotly line chart showing the selected metric rolling over trade number.

**Computation:**
```python
def compute_rolling_metrics(trades_df, window=20):
    """Compute rolling win rate, profit factor, and Sharpe over a trade window."""
    r = trades_df['r_multiple']
    w = trades_df['win'].astype(float)

    rolling_wr = w.rolling(window, min_periods=window).mean() * 100
    # Rolling PF: sum of wins / abs(sum of losses) over window
    rolling_wins = r.where(r > 0, 0).rolling(window, min_periods=window).sum()
    rolling_losses = r.where(r < 0, 0).rolling(window, min_periods=window).sum().abs()
    rolling_pf = rolling_wins / rolling_losses.replace(0, float('nan'))
    # Rolling Sharpe (simplified): mean(R) / std(R) over window
    rolling_mean = r.rolling(window, min_periods=window).mean()
    rolling_std = r.rolling(window, min_periods=window).std()
    rolling_sharpe = rolling_mean / rolling_std.replace(0, float('nan'))

    return rolling_wr, rolling_pf, rolling_sharpe
```

### 11.4 Return Distribution Analysis

**Location:** New section in strategy detail, as a tab alongside rolling metrics.

**UI:**
- Toggle buttons: **Histogram** | **Box Plot** | **Violin**
- Three stat callouts below the chart: **Skewness**, **Kurtosis**, **Tail Risk** (5th percentile R).

**Charts:**
- **Histogram:** `go.Histogram(x=r_multiples, nbinsx=30)`. Color positive/negative bars differently.
- **Box Plot:** `go.Box(y=r_multiples, name="R-Multiple Distribution")`.
- **Violin:** `go.Violin(y=r_multiples, name="R-Multiple Distribution")`.

### 11.5 Markov Motor Analysis

**Location:** New tab in strategy detail page. This is an advanced analysis section — add as a 7th drill-down tab or as a separate page section after the main strategy detail tabs.

**Recommended approach:** Add a new top-level section below the existing strategy detail content, with its own header: "Advanced Analysis". This avoids crowding the existing 6 drill-down tabs (Entry, Exit, TF Conditions, General, Stop Loss, Take Profit).

**Sub-sections:**

#### 11.5a Analysis Controls
- **Rolling Window Size:** Slider 10–100 (default 20 trades)
- **Edge Decay Threshold:** Slider 0.8–2.0 (default 1.2 PF)
- **Confidence Level:** Selectbox: 90%, 95%, 99% (default 95%)
- **"Update Analysis"** button (recomputes all Markov metrics with current settings)

#### 11.5b Rolling Performance Metrics (reuse from 11.3)
Same chart but embedded in the Markov section with the configurable window from 11.5a.

#### 11.5c Markov State Transitions
Treat each trade as a state: **W** (win) or **L** (loss).

**Compute:**
```python
def compute_markov_transitions(trades_df):
    """Compute Win/Loss Markov transition matrix."""
    wins = trades_df['win'].values
    transitions = {'WW': 0, 'WL': 0, 'LW': 0, 'LL': 0}
    for i in range(1, len(wins)):
        prev = 'W' if wins[i-1] else 'L'
        curr = 'W' if wins[i] else 'L'
        transitions[prev + curr] += 1

    total_from_w = transitions['WW'] + transitions['WL']
    total_from_l = transitions['LW'] + transitions['LL']

    probs = {
        'W_to_W': transitions['WW'] / max(total_from_w, 1),
        'W_to_L': transitions['WL'] / max(total_from_w, 1),
        'L_to_W': transitions['LW'] / max(total_from_l, 1),
        'L_to_L': transitions['LL'] / max(total_from_l, 1),
    }
    return probs, transitions
```

**Display:**
- **Transition Probability Table** (2×2 grid): From/To matrix showing W→W, W→L, L→W, L→L percentages.
- **Win/Loss Streak Bar Chart:** Horizontal bar chart — green bars above zero for win streaks, red bars below zero for loss streaks (sequential, trade-indexed). Like DaviddTech's streak visualization.
- **Current Trend** callout: "Stable →", "Improving ↑", or "Declining ↓" based on recent rolling PF trend.
- **Edge Strength** callout: "Strong (PF X.XX)", "Moderate", "Critical (PF X.XX)", "Lost" — based on whether rolling PF is above/below the edge decay threshold.

#### 11.5d Edge Decay Analysis
- Line chart: rolling Profit Factor over trade number.
- Horizontal threshold line at the configured edge decay threshold (default 1.2).
- Badge in top-right: "Edge Holding" (green) or "Edge Weakening" (red/yellow) based on whether the last N-trade rolling PF is above or below threshold.
- **Consistency Score** (0-100): `100 × (trades where rolling PF > threshold) / total rolling windows`.
- **Stability Index** (0-100): `100 × (1 - std(rolling_pf) / mean(rolling_pf))`. Capped at 0-100.
- **Trend Strength** (0-100): Linear regression slope of rolling PF, normalized to 0-100 scale. Positive slope = high score.

#### 11.5e Return Distribution (reuse 11.4 charts)
Same histogram/box/violin but embedded in Markov section. Add Skewness, Kurtosis, Tail Risk callouts.

#### 11.5f Market Regime Detection
Cluster trades into regimes based on rolling performance:

```python
def detect_market_regimes(trades_df, window=20):
    """Classify trade windows into Favorable / Unfavorable / Neutral regimes."""
    r = trades_df['r_multiple']
    rolling_mean = r.rolling(window, min_periods=window).mean()
    rolling_std = r.rolling(window, min_periods=window).std()

    regimes = []
    for i in range(len(trades_df)):
        if pd.isna(rolling_mean.iloc[i]):
            regimes.append('neutral')
        elif rolling_mean.iloc[i] > 0.5 * rolling_std.iloc[i]:
            regimes.append('favorable')
        elif rolling_mean.iloc[i] < -0.5 * rolling_std.iloc[i]:
            regimes.append('unfavorable')
        else:
            regimes.append('neutral')
    return regimes
```

**Display:**
- Scatter plot: trade number (x) vs R-multiple (y), color-coded by regime (green/red/gray).
- Three stat callouts: **Favorable Regime %**, **Avg Regime Duration** (trades), **Current Regime Age** (trades).

#### 11.5g Markov Intelligence Insights
Generate 2-4 plain-text insight sentences based on computed metrics. **No LLM call** — use rule-based templates:

```python
def generate_markov_insights(probs, rolling_pf, edge_threshold, consistency, stability, trend):
    insights = []
    if probs['W_to_W'] > 0.65:
        insights.append(f"Strong momentum detected: {probs['W_to_W']*100:.1f}% probability of consecutive wins suggests positive autocorrelation.")
    if trend > 70:
        insights.append("Strong performance trend detected. Current trajectory shows sustained edge.")
    if rolling_pf[-1] > edge_threshold * 1.25:
        insights.append(f"Exceptional recent performance. Profit factor trending above {edge_threshold * 1.25:.1f} indicates robust edge.")
    if probs['L_to_L'] > 0.6:
        insights.append(f"Loss clustering detected: {probs['L_to_L']*100:.1f}% probability of consecutive losses. Consider position sizing adjustment during losing streaks.")
    if consistency < 50:
        insights.append(f"Edge inconsistency warning: strategy only maintains edge in {consistency:.0f}% of rolling windows.")
    # Return top 3
    return insights[:3]
```

Display as styled cards with emoji indicators (similar to DaviddTech screenshot).

### 11.6 KPI Placement Audit

**Goal:** Ensure KPIs are displayed consistently and usefully across all views.

| View | Current KPIs | Phase 11 Change |
|------|-------------|----------------|
| Strategy card (My Strategies) | Win Rate, PF, Avg R, Total R, Max DD, R² | **No change** — keep compact |
| Strategy detail — KPI section | Same 6 + Total P&L, Final Balance, Daily R | **Replace with tabbed expanded panel** (11.1) |
| Strategy detail — Equity Curve | Cumulative R chart | **Add Edge Check toggle** (11.2) |
| Portfolio card | Composite KPIs from strategies | **No change** |
| Portfolio detail | Strategy-level KPIs | **No change** |

### 11.7 Implementation Order

1. **Expand `calculate_kpis()`** — add all new KPI computations. Ensure backward compatibility (existing code reading `kpis['win_rate']` etc. still works).
2. **Create analytics helper module** — `src/analytics.py` (new file, ~300-400 lines). Contains: `compute_rolling_metrics()`, `compute_markov_transitions()`, `detect_market_regimes()`, `generate_markov_insights()`, `compute_edge_scores()`. This keeps `app.py` from growing unboundedly.
3. **Redesign KPI display** in strategy detail — tabbed panel with all KPI categories.
4. **Add Edge Check toggle** to equity curve chart.
5. **Add "Advanced Analysis" section** to strategy detail — Rolling Metrics, Return Distribution tabs.
6. **Add Markov Motor Analysis section** — Analysis Controls, Transitions, Edge Decay, Regime Detection, Insights.
7. **Syntax check.**

### 11.8 Minimum Trade Thresholds

Many advanced metrics are meaningless with few trades. Apply these guards:

| Trades Required | Metrics |
|----------------|---------|
| ≥ 2 | Profit factor, win rate, avg R |
| ≥ 5 | Sharpe, Sortino, skewness, kurtosis |
| ≥ 10 | Calmar, VaR, CVaR, tail ratio |
| ≥ 20 | Rolling metrics, Markov transitions, regime detection |
| ≥ 30 | Edge decay analysis, consistency/stability/trend scores |

Below threshold: show "—" or "Insufficient data (need N+ trades)" in the UI.

---

## Phase 12: Webhook Inbound Strategy Origin

**Goal:** Allow strategies to be driven by inbound webhooks instead of the standard trigger-based approach. Users can receive entry/exit signals from external sources (TradingView alerts, LuxAlgo, custom scripts) and still layer RoR Trader's confluence conditions, stop/target management, and backtesting on top.

**Scope:** Webhook Inbound ONLY. Scanner origin is deferred to Phase 17.

**Visual Reference:** `docs/reference_images/luxalgo webhook creator.png`

### 12.1 Strategy Origin Selection

**Location:** `app.py` — Strategy Builder sidebar (search for `strategy_origin` in the builder section).

**Current state:** A `strategy_origin` field exists on strategy dicts (defaulting to `"standard"`), added in Phase 8 as a placeholder. The UI selectbox may already exist.

**Changes:**
- Expand the origin selectbox options: `["Standard", "Webhook Inbound"]`.
- When "Webhook Inbound" is selected, hide the standard Entry Trigger / Exit Trigger sections and show webhook-specific fields instead.
- The rest of the builder (confluence, stop/target, risk management) remains available for both origins.

### 12.2 Webhook Inbound Configuration Fields

When `strategy_origin == "webhook_inbound"`, show these fields in the Strategy Builder:

```
┌─────────────────────────────────────────────┐
│  WEBHOOK INBOUND CONFIGURATION              │
├─────────────────────────────────────────────┤
│  Webhook Secret: [auto-generated, copyable] │
│  Endpoint URL:   [display, copyable]        │
│                                             │
│  Entry Signal JSON Path:  [text_input]      │
│  Exit Signal JSON Path:   [text_input]      │
│                                             │
│  Signal Direction Mapping:                  │
│    Long entry value:  [text_input "buy"]    │
│    Short entry value: [text_input "sell"]   │
│    Exit value:        [text_input "close"]  │
│                                             │
│  ☑ Layer confluence conditions on webhook   │
│    signals (require confluence to be true    │
│    at signal time for entry to count)       │
│                                             │
│  Backtest Data:                             │
│    [Upload CSV] or [Paste JSON]             │
│    Preview: showing N signals loaded        │
└─────────────────────────────────────────────┘
```

### 12.3 Inbound Webhook Endpoint

**New endpoint** — RoR Trader needs a lightweight HTTP server to receive inbound webhooks. Options:

**Option A (Recommended): Streamlit + background thread with Flask/FastAPI**
- Spin up a minimal Flask/FastAPI server on a configurable port (default 8501) in a background thread when alert monitoring starts.
- Endpoint: `POST /webhook/inbound/{strategy_id}`
- Validates webhook secret from header (`X-Webhook-Secret`) or query param.
- Parses the JSON body to extract signal using the configured JSON path.
- Stores the signal in an inbound signal queue (JSON file or in-memory deque).

**Option B: Polling a webhook relay service**
- User configures a relay (e.g., webhook.site, Pipedream) and RoR Trader polls it.
- Simpler but adds latency and external dependency.

**Recommended: Option A.** Use Flask for simplicity (already available in most Python envs).

**Inbound webhook server (`src/webhook_server.py` — new file, ~150 lines):**

```python
"""
Lightweight inbound webhook receiver for RoR Trader.
Runs in a background thread alongside the Streamlit app.
"""
from flask import Flask, request, jsonify
import threading
import json
from datetime import datetime
from pathlib import Path

INBOUND_SIGNALS_FILE = Path(__file__).parent.parent / "config" / "inbound_signals.json"

app = Flask(__name__)
_webhook_secrets = {}  # {strategy_id: secret}

@app.route("/webhook/inbound/<int:strategy_id>", methods=["POST"])
def receive_webhook(strategy_id):
    # Validate secret
    secret = request.headers.get("X-Webhook-Secret") or request.args.get("secret")
    expected = _webhook_secrets.get(strategy_id)
    if not expected or secret != expected:
        return jsonify({"error": "Invalid secret"}), 401

    body = request.get_json(force=True, silent=True) or {}
    signal = {
        "strategy_id": strategy_id,
        "received_at": datetime.now().isoformat(),
        "payload": body,
        "processed": False,
    }
    _append_inbound_signal(signal)
    return jsonify({"status": "received", "signal_id": signal.get("id")}), 200

def start_webhook_server(port=8501, secrets=None):
    """Start the webhook server in a daemon thread."""
    if secrets:
        _webhook_secrets.update(secrets)
    thread = threading.Thread(target=lambda: app.run(port=port, debug=False), daemon=True)
    thread.start()
    return thread
```

### 12.4 Strategy Schema Additions

New fields on strategy dict when `strategy_origin == "webhook_inbound"`:

```python
{
    "strategy_origin": "webhook_inbound",
    "webhook_config": {
        "secret": "whsec_abc123...",           # Auto-generated UUID
        "signal_json_path": "action",           # JSONPath to signal value in payload
        "entry_long_value": "buy",              # Value that means "enter long"
        "entry_short_value": "sell",            # Value that means "enter short"
        "exit_value": "close",                  # Value that means "exit"
        "layer_confluence": true,               # Check confluence at signal time
        "backtest_signals": [                   # Uploaded/pasted historical signals
            {"timestamp": "ISO", "signal": "buy", "price": 450.25},
            {"timestamp": "ISO", "signal": "close", "price": 452.10},
            ...
        ]
    },
    # Standard fields still present:
    "entry_trigger": null,                      # Not used for webhook origin
    "exit_trigger": null,                       # Not used for webhook origin
    "confluence": [...],                        # Still used if layer_confluence=true
    "stop_config": {...},                       # Still used for risk management
    "target_config": {...},                     # Still used
}
```

### 12.5 Backtest Data: CSV Upload

**Purpose:** Webhook-origin strategies can't run the standard trigger-based backtest. Instead, users upload historical signal data (exported from TradingView, LuxAlgo, or a spreadsheet).

**CSV format (flexible, auto-detect columns):**
```csv
timestamp,signal,price
2025-01-15 10:30:00,buy,450.25
2025-01-15 11:45:00,close,452.10
2025-01-16 09:35:00,buy,448.50
2025-01-16 10:20:00,close,449.80
```

**Required columns:** `timestamp`, `signal` (or `action`). **Optional:** `price` (if not provided, look up from market data at that timestamp).

**Processing:**
1. Parse CSV, detect column names (case-insensitive matching).
2. Map signal values to entry/exit using the configured mapping.
3. Pair entries with exits chronologically.
4. For each pair, apply stop/target logic: load market data between entry and exit timestamps, check if stop/target would have been hit first.
5. Compute R-multiples and store as `stored_trades`.
6. Run `calculate_kpis()` on the resulting trades.

**Implementation:** Add `process_webhook_backtest_signals()` function to `app.py` (or to a new `webhook_backtest.py` helper if > 200 lines).

### 12.6 Forward Testing for Webhook Origin

- When inbound webhook signals arrive (via the webhook server), they're processed the same way as backtest signals but in real-time.
- The signal is paired with market data at the signal timestamp.
- Confluence conditions are checked (if `layer_confluence` is enabled).
- Stops/targets are evaluated against subsequent bars.
- Resulting trades are appended to `stored_trades` and KPIs are recomputed.
- The forward test boundary works identically to standard strategies.

### 12.7 UI Branching in Strategy Builder

**Key decision points in the Strategy Builder flow:**

| Builder Section | Standard Origin | Webhook Inbound Origin |
|-----------------|----------------|----------------------|
| Entry Trigger | Trigger search/select | Hidden |
| Exit Trigger | Trigger search/select | Hidden |
| Confluence | Full access | Available if `layer_confluence` enabled |
| Stop Loss | Full access | Full access |
| Take Profit | Full access | Full access |
| Risk Management | Full access | Full access |
| Data Window | Full access | Backtest signals determine range |
| Webhook Config | Hidden | Show inbound config fields |

### 12.8 Implementation Order

1. **Add `webhook_config` schema** — update strategy dict handling in save/load.
2. **Branch Strategy Builder UI** — hide entry/exit triggers for webhook origin, show webhook config fields.
3. **Create `src/webhook_server.py`** — inbound webhook receiver.
4. **Implement CSV upload + parsing** — file uploader, column detection, signal pairing.
5. **Implement webhook backtest processing** — apply stop/target logic to uploaded signals, generate trades.
6. **Wire forward test path** — process real-time inbound signals same as backtest.
7. **Update `OPTIMIZABLE_PARAMS`** — add `webhook_config` fields that affect trade generation (signal_json_path, entry/exit values, layer_confluence).
8. **Syntax check.**

---

## Phase 13: Live Alerts Validation

**Goal:** Add a third confidence tier — live/triggered — that captures actual alert executions and compares them to forward test trades. Visualize all three tiers (backtest, forward test, live) on the equity curve with distinct colors.

**Critical design principle:** Enabling live alert tracking does **NOT** turn off or replace the forward test. Both run in parallel on the same trades and time period. The difference between backtest and forward test is the date range (historical vs. real-time). The difference between forward test and live is **apples-to-apples comparison on the exact same trades** — same days, same signals — but live captures actual alert execution prices, timing, and missed signals. Live data is still a form of testing with additional analysis layered on top (slippage measurement, delivery reliability, etc.). The forward test curve always shows theoretical bar-close prices; the live curve shows what actually happened when alerts fired.

**Prerequisite:** Phase 11 (expanded KPIs) is helpful but not blocking. Phase 12 is not required.

### 13.1 Alert Execution Records

**Current state:** Alerts are saved in `alerts.json` with `webhook_deliveries` arrays. Each alert has a `timestamp`, `price`, `type`, `strategy_id`, etc.

**New field on strategy dict:** `live_executions` — a list of matched alert→trade records.

```python
{
    "live_executions": [
        {
            "alert_id": 42,
            "type": "entry",                        # "entry" or "exit"
            "alert_timestamp": "ISO",               # When alert fired
            "alert_price": 450.25,                  # Price at alert fire time
            "theoretical_price": 450.10,            # Price from forward test trade
            "slippage_r": 0.05,                     # (alert_price - theoretical_price) / risk in R terms
            "matched_trade_index": 15,              # Index into stored_trades
            "webhook_delivered": true,               # Whether any webhook fired successfully
            "webhook_delivery_count": 2,             # How many webhooks fired
        },
        ...
    ],
    "alert_tracking_enabled": true,                 # Per-strategy toggle
    "discrepancies": [
        {
            "type": "missed_alert",                 # Forward test trade exists, no matching alert
            "trade_index": 18,
            "trade_entry_time": "ISO",
            "detected_at": "ISO",
        },
        {
            "type": "phantom_alert",                # Alert fired but no matching forward test trade
            "alert_id": 55,
            "alert_timestamp": "ISO",
            "detected_at": "ISO",
        },
    ]
}
```

### 13.2 Alert→Trade Matching Algorithm

**Location:** New function in `alerts.py` or new `src/live_validation.py` file.

```python
def match_alerts_to_trades(strategy: dict, alerts: list) -> dict:
    """
    Match fired alerts to forward test trades for a strategy.

    Args:
        strategy: Strategy dict with stored_trades and forward_test_start
        alerts: List of alert records for this strategy

    Returns:
        dict with 'live_executions' and 'discrepancies' lists
    """
    stored_trades = strategy.get('stored_trades', [])
    ft_start = strategy.get('forward_test_start')
    if not ft_start:
        return {'live_executions': [], 'discrepancies': []}

    ft_start_dt = datetime.fromisoformat(ft_start)

    # Get only forward test trades (after ft_start)
    ft_trades = [t for t in stored_trades
                 if datetime.fromisoformat(t['entry_time']) >= ft_start_dt]

    # Get alerts for this strategy
    strategy_alerts = [a for a in alerts
                       if a.get('strategy_id') == strategy['id']
                       and a.get('type') in ('entry_signal', 'exit_signal')]

    # Match by proximity: for each FT trade entry, find closest entry alert within ±5 minutes
    MATCH_WINDOW = timedelta(minutes=5)
    executions = []
    matched_alert_ids = set()
    matched_trade_indices = set()

    for i, trade in enumerate(ft_trades):
        trade_entry = datetime.fromisoformat(trade['entry_time'])
        trade_exit = datetime.fromisoformat(trade['exit_time'])

        # Find entry alert match
        entry_match = None
        for alert in strategy_alerts:
            if alert['id'] in matched_alert_ids:
                continue
            if alert['type'] != 'entry_signal':
                continue
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if abs((alert_time - trade_entry).total_seconds()) < MATCH_WINDOW.total_seconds():
                entry_match = alert
                break

        if entry_match:
            matched_alert_ids.add(entry_match['id'])
            matched_trade_indices.add(i)
            # ... build execution record

    # Discrepancies: unmatched trades (missed alerts) and unmatched alerts (phantom alerts)
    discrepancies = []
    for i, trade in enumerate(ft_trades):
        if i not in matched_trade_indices:
            discrepancies.append({
                "type": "missed_alert",
                "trade_index": i,
                "trade_entry_time": trade['entry_time'],
            })
    for alert in strategy_alerts:
        if alert['id'] not in matched_alert_ids:
            discrepancies.append({
                "type": "phantom_alert",
                "alert_id": alert['id'],
                "alert_timestamp": alert['timestamp'],
            })

    return {'live_executions': executions, 'discrepancies': discrepancies}
```

### 13.3 Three-Color Equity Curve

**Location:** Replace the current single-color equity curve in `render_strategy_detail()`.

**Color scheme:**
| Segment | Color | Meaning |
|---------|-------|---------|
| Backtest | `#2196F3` (blue, current) | Historical backtest trades |
| Forward Test | `#FF9800` (orange) | Real-time theoretical trades |
| Live/Triggered | `#4CAF50` (green) | Alert-matched executions |

**Implementation approach:**

The equity curve shows three segments sequentially (backtest → forward test), but in the forward test period where live data exists, **both** the forward test and live lines are rendered so the user can see divergence:

```python
# Split equity curve data into segments
boundary_idx = equity_data.get('boundary_index')  # BT→FT boundary (exists)
live_start_idx = None  # First trade with a matched live execution

# Backtest segment: blue, up to forward test boundary
fig.add_trace(go.Scatter(  # Backtest segment
    x=trade_numbers[:boundary_idx],
    y=cumulative_r[:boundary_idx],
    line=dict(color="#2196F3"), fill="tozeroy", name="Backtest"
))

# Forward test segment: orange, runs for the FULL forward test period
# (does NOT stop when live starts — both lines coexist)
fig.add_trace(go.Scatter(  # Forward test segment
    x=trade_numbers[boundary_idx:],
    y=cumulative_r[boundary_idx:],
    line=dict(color="#FF9800"), fill="tozeroy", name="Forward Test"
))

# Live segment: green, overlaid on top of forward test where live data exists
# Uses actual alert execution prices — may diverge from forward test
if live_start_idx is not None:
    fig.add_trace(go.Scatter(  # Live segment (overlaid)
        x=trade_numbers[live_start_idx:],
        y=live_cumulative_r[live_start_idx:],
        line=dict(color="#4CAF50", width=2), name="Live"
    ))
```

**Forward test + live coexistence:** Both lines render for the same trades in the live period. The forward test line shows theoretical bar-close R-multiples; the live line shows actual alert-execution R-multiples. The gap between them is the slippage/divergence the user wants to monitor. This is an apples-to-apples comparison on the exact same trades.

### 13.4 Strategy Card Caption Enhancement

**Location:** Strategy cards in My Strategies page (search for card rendering, ~L584+).

**Current caption:** Shows symbol, direction, timeframe.

**Enhanced caption format:**
```
SPY LONG 1Min | BT 45d | Fwd 14d | Live 5d
```

Each segment duration is color-coded to match the equity curve:
- "BT 45d" in blue
- "Fwd 14d" in orange
- "Live 5d" in green (only if alert tracking enabled and live data exists)

**Implementation:** Use `st.markdown()` with inline HTML for colored spans:
```python
caption = f"""
{symbol} {direction} {timeframe} |
<span style="color:#2196F3">BT {bt_days}d</span> |
<span style="color:#FF9800">Fwd {fwd_days}d</span>
"""
if live_days > 0:
    caption += f' | <span style="color:#4CAF50">Live {live_days}d</span>'
st.markdown(caption, unsafe_allow_html=True)
```

### 13.5 Mini Equity Curve Update

**Location:** `render_mini_equity_curve()` function.

Update to show three color segments (matching the full equity curve). The mini version doesn't need interactivity — just color the line segments appropriately.

### 13.6 Discrepancy Display

**On strategy cards:**
- If discrepancies exist, show a small warning badge: "⚠ 3 discrepancies"

**On strategy detail — "Live vs. Forward" tab:**
- Table listing each discrepancy with type, timestamp, and action suggestion.
- "Missed Alert" rows suggest checking alert monitor configuration.
- "Phantom Alert" rows suggest the theoretical model may have a timing difference.

### 13.7 Alert Tracking Mode

**Location:** Strategy detail page — add a toggle in the settings/configuration area.

**Behavior:**
- `st.toggle("Track Alert Executions", key=f"alert_tracking_{sid}")` persisted to strategy dict as `alert_tracking_enabled`.
- When enabled, the alert matching runs on each data refresh.
- Independent of portfolio webhook allocation — a strategy can track alerts without being in a portfolio.
- When disabled, `live_executions` and `discrepancies` are cleared.

### 13.8 "Live vs. Forward" Comparison Tab

**Location:** New tab in strategy detail, alongside or replacing the current equity curve tab area.

**Layout:**
```
┌───────────────────────────────────────────────┐
│  Live vs. Forward Comparison                  │
├───────────────┬───────────────────────────────┤
│               │ Forward Test │ Live Execution  │
├───────────────┼──────────────┼─────────────────┤
│ Total Trades  │      42      │       38        │
│ Win Rate      │    58.3%     │     55.3%       │
│ Profit Factor │     1.82     │      1.65       │
│ Avg R         │     0.34     │      0.29       │
│ Total R       │    14.3      │     11.0        │
│ Avg Slippage  │      —       │    -0.05 R      │
│ Missed Alerts │      —       │       4         │
│ Phantom Alerts│      —       │       2         │
└───────────────┴──────────────┴─────────────────┘
```

Delta column with ▲/▼ indicators showing where live outperforms or underperforms forward test.

### 13.9 Implementation Order

1. **Add `live_executions`, `discrepancies`, `alert_tracking_enabled` fields** to strategy schema.
2. **Implement `match_alerts_to_trades()`** in `alerts.py` or new `src/live_validation.py`.
3. **Update equity curve** to three-color rendering (detail page + mini).
4. **Update strategy card captions** with colored duration segments.
5. **Add "Live vs. Forward" comparison tab** to strategy detail.
6. **Add alert tracking toggle** to strategy detail.
7. **Add discrepancy display** (cards + detail tab).
8. **Wire matching into data refresh cycle** — run matching on each incremental refresh when alert_tracking_enabled.
9. **Syntax check.**

---

## Phase 14: Live Portfolio Management

**Goal:** Bridge the gap between backtesting and real-world trading with account management, balance tracking, and real-time intra-bar alert capabilities.

**Prerequisite:** Phase 13 (alert tracking) should be complete for full value, but Phase 14A can be implemented independently.

**Split rationale:** Phase 14 is divided into **14A** (account management — no external dependencies) and **14B** (real-time intra-bar engine — requires paid Alpaca SIP subscription at $99/mo). 14A can be completed immediately. 14B should pause at the "ready to connect" point and prompt the user to upgrade their Alpaca account before proceeding with WebSocket integration.

### Phase 14A: Account Management (no Alpaca upgrade needed)
Sections: 14.1–14.6

### Phase 14B: Real-Time Intra-Bar Engine (requires Alpaca SIP subscription)
Sections: 14.7–14.8. **Stop point:** After scaffolding the engine and Connections UI, prompt the user to confirm Alpaca SIP access before wiring live WebSocket streams.

### 14.1 Account Management Tab

**Location:** Portfolio detail page — add as a new tab alongside existing portfolio tabs.

**Tab name:** "Account" or "Live Account"

**Layout:**
```
┌─────────────────────────────────────────────┐
│  LIVE ACCOUNT MANAGEMENT                    │
├─────────────────────────────────────────────┤
│                                             │
│  Account Balance: $12,450.00                │
│  Starting Balance: $10,000.00               │
│  Net Deposits: +$1,500.00                   │
│  Trading P&L: +$950.00                      │
│                                             │
│  ┌─── Balance History Chart ───────────┐    │
│  │  [Plotly area chart]                │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌─── Ledger ──────────────────────────┐    │
│  │ Date       │ Type     │ Amount      │    │
│  │ 2026-02-01 │ Deposit  │ +$10,000    │    │
│  │ 2026-02-05 │ Trading  │ +$320       │    │
│  │ 2026-02-08 │ Deposit  │ +$1,500     │    │
│  │ 2026-02-10 │ Trading  │ -$150       │    │
│  │ 2026-02-12 │ Trading  │ +$780       │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  [+ Add Deposit] [+ Add Withdrawal]        │
│                                             │
│  ┌─── Trading Notes ──────────────────┐    │
│  │ [Text area — freeform notes]       │    │
│  │ Markdown supported                 │    │
│  └─────────────────────────────────────┘    │
│  [Save Notes]                               │
│                                             │
└─────────────────────────────────────────────┘
```

### 14.2 Portfolio Schema Additions

New fields on portfolio dict:

```python
{
    "account": {
        "starting_balance": 10000.0,
        "ledger": [
            {
                "id": 1,
                "date": "2026-02-01",
                "type": "deposit",              # "deposit", "withdrawal", "trading_pnl"
                "amount": 10000.0,
                "note": "Initial funding",
                "auto": false                    # true for auto-generated trading P&L entries
            },
            ...
        ],
        "notes": "## Week 1\nStarted live trading...",
        "notes_updated_at": "ISO"
    }
}
```

**Balance computation:** Sum all ledger entries. `current_balance = sum(entry['amount'] for entry in ledger)`. Deposits are positive, withdrawals are negative, trading P&L can be either.

### 14.3 Deposit/Withdrawal UI

**"Add Deposit" dialog:**
```python
with st.form("add_deposit"):
    amount = st.number_input("Amount ($)", min_value=0.01, value=1000.0)
    date = st.date_input("Date", value=datetime.now().date())
    note = st.text_input("Note (optional)")
    if st.form_submit_button("Add Deposit"):
        # Append to ledger with type="deposit", amount=+amount
```

**"Add Withdrawal" dialog:** Same but `type="withdrawal"` and `amount=-amount`.

### 14.4 Trading P&L Integration

When webhook-triggered trades resolve (from Phase 13's live executions):
- Auto-generate a `trading_pnl` ledger entry with the dollar P&L.
- Mark as `"auto": true` so it's visually distinct in the ledger.
- This creates a running P&L that reflects actual live trading results.

### 14.5 Trading Notes

- `st.text_area()` with current notes content.
- "Save Notes" button persists to portfolio dict.
- Supports markdown formatting.
- Display the rendered markdown below the editor with `st.markdown()`.

### 14.6 Balance History Chart

**Plotly area chart:**
- X-axis: dates from ledger entries.
- Y-axis: running balance (cumulative sum of ledger amounts).
- Color: deposits/withdrawals shown as step changes, trading P&L as gradual.

### 14.7 Intra-Bar Real-Time Alert Engine

**This is the most complex item in Phase 14.** It enables real-time alerting for `[I]` (intra-bar) triggers using WebSocket streaming.

**Current state:** Alert monitoring uses bar-close polling — it loads the latest completed bar and checks trigger conditions. This works for `[C]` (close-only) triggers but misses intra-bar price movements.

**Architecture:**

```
┌─────────────────────────────────────────────────┐
│              ALERT ENGINE (alert_monitor.py)     │
├─────────────────────────────────────────────────┤
│                                                  │
│  [C] Triggers ──→ Bar-Close Poller (existing)   │
│                    - Runs every bar close        │
│                    - Uses completed OHLCV bars   │
│                    - detect_signals() from       │
│                      alerts.py                   │
│                                                  │
│  [I] Triggers ──→ WebSocket Streamer (NEW)      │
│                    - Alpaca real-time trades/    │
│                      quotes feed                 │
│                    - Builds partial OHLCV bars   │
│                    - Checks trigger conditions   │
│                      against live ticks          │
│                    - Fires alert immediately     │
│                      when condition met          │
│                                                  │
│  Alpaca Data Feed Requirements:                  │
│  - SIP feed ($99/mo): All symbols, real-time    │
│  - IEX feed (free): 30 symbols, ~15min delay    │
│                                                  │
└─────────────────────────────────────────────────┘
```

**New file: `src/realtime_engine.py` (~300-400 lines)**

```python
"""
Real-time intra-bar alert engine using Alpaca WebSocket streaming.

Subscribes to real-time trades/quotes for active [I]-trigger strategy symbols.
Builds partial OHLCV bars from tick data and checks trigger conditions.
"""
import asyncio
import threading
from datetime import datetime
from typing import Dict, Set, Optional
from collections import defaultdict

class RealtimeAlertEngine:
    def __init__(self):
        self.active_symbols: Set[str] = set()
        self.partial_bars: Dict[str, dict] = {}  # symbol -> current partial bar
        self.strategies: list = []                # Active [I]-trigger strategies
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, strategies: list):
        """Start the real-time engine for strategies with [I] triggers."""
        self.strategies = [s for s in strategies if self._has_intrabar_triggers(s)]
        if not self.strategies:
            return
        self.active_symbols = {s['symbol'] for s in self.strategies}
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _has_intrabar_triggers(self, strategy: dict) -> bool:
        """Check if strategy uses any [I] execution type triggers."""
        # Check trigger execution_type property from confluence_groups
        # [I] triggers fire intra-bar, [C] triggers fire at bar close
        # Implementation depends on trigger metadata
        pass

    def _run_loop(self):
        """Main event loop — connect to Alpaca WebSocket and process ticks."""
        asyncio.run(self._stream_data())

    async def _stream_data(self):
        """Subscribe to Alpaca real-time data and process ticks."""
        try:
            from alpaca.data.live import StockDataStream
            stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

            async def on_trade(trade):
                self._process_tick(trade.symbol, float(trade.price), trade.timestamp)

            for symbol in self.active_symbols:
                stream.subscribe_trades(on_trade, symbol)

            await stream.run()
        except Exception as e:
            print(f"Realtime stream error: {e}")

    def _process_tick(self, symbol: str, price: float, timestamp):
        """Process a single trade tick — update partial bar and check triggers."""
        # Update partial OHLCV bar
        bar = self.partial_bars.get(symbol)
        if bar is None or self._is_new_bar_period(bar, timestamp):
            # Start new bar
            self.partial_bars[symbol] = {
                'open': price, 'high': price, 'low': price,
                'close': price, 'volume': 0, 'bar_start': timestamp
            }
        else:
            bar['high'] = max(bar['high'], price)
            bar['low'] = min(bar['low'], price)
            bar['close'] = price
            bar['volume'] += 1

        # Check trigger conditions for strategies using this symbol
        for strategy in self.strategies:
            if strategy['symbol'] != symbol:
                continue
            self._check_intrabar_triggers(strategy, symbol, price)

    def _check_intrabar_triggers(self, strategy, symbol, current_price):
        """Check if any [I] trigger conditions are met at current price."""
        # This is the core logic — evaluate trigger conditions against live price
        # For price-level triggers (e.g., UT Bot trail crossing):
        #   Compare current_price against the trigger level computed from the last complete bar
        # If condition met: fire alert via save_alert() and send_webhook()
        pass
```

**Key design decisions:**
- The WebSocket streamer runs in a separate daemon thread.
- Partial bars are constructed from tick data but the full pipeline (indicators, interpreters) still runs on completed bars.
- `[I]` triggers check price-level crossings against levels computed from the last completed bar (e.g., UT Bot trailing stop level, EMA value).
- `[C]` triggers continue using the existing bar-close polling model unchanged.

### 14.8 Data Feed Configuration

**Location:** Settings page — new "Connections" subsection.

```
┌─────────────────────────────────────────────┐
│  CONNECTIONS                                │
├─────────────────────────────────────────────┤
│  Alpaca API                                 │
│  Status: ● Connected (Paper)                │
│                                             │
│  API Key: [text_input, masked]              │
│  Secret Key: [text_input, masked]           │
│  [Save] [Test Connection]                   │
│                                             │
│  Data Feed:                                 │
│  ○ IEX (Free — 30 symbols, basic quotes)   │
│  ● SIP ($99/mo — all symbols, real-time)    │
│                                             │
│  Real-Time Engine: [toggle]                 │
│  Active Symbols: SPY, AAPL, TSLA (3)       │
└─────────────────────────────────────────────┘
```

### 14.9 Implementation Order

**Phase 14A (no Alpaca upgrade needed):**
1. **Add `account` schema to portfolio dict** — ledger, notes, balance helpers.
2. **Build Account Management tab** — balance display, ledger table, deposit/withdrawal forms.
3. **Add Trading Notes** — text area with save/render.
4. **Add Balance History Chart** — Plotly area chart from ledger.
5. **Wire trading P&L integration** — auto-generate ledger entries from live executions (depends on Phase 13).
6. **Syntax check + commit.**

**Phase 14B (requires Alpaca SIP subscription — $99/mo):**
7. **Add Connections subsection to Settings** — Alpaca config, data feed selection, real-time engine toggle. This can be scaffolded without a live SIP connection.
8. **Create `src/realtime_engine.py`** — WebSocket streamer, partial bar builder, intra-bar trigger checker.
9. **STOP POINT:** Prompt user to confirm Alpaca SIP subscription is active before testing live WebSocket streams.
10. **Integrate real-time engine with alert monitor** — start/stop engine based on active [I]-trigger strategies.
11. **Syntax check.**

---

## Cross-Phase Dependencies

```
Phase 11 (Analytics)
    │
    │ (expanded KPIs used by Phase 13 comparison tab)
    ▼
Phase 13 (Live Alerts Validation)
    │
    │ (live_executions used by Phase 14 trading P&L)
    ▼
Phase 14 (Live Portfolio Management)

Phase 12 (Webhook Inbound)
    │
    │ (independent, can run in parallel with 11)
    │ (webhook server partially useful for Phase 13 inbound tracking)
    ▼
    (no hard downstream dependency)
```

**Recommended execution order:** 11 → 12 → 13 → 14A → 14B

**Parallel opportunities:**
- Phase 11 and Phase 12 are independent — can be done in either order.
- Phase 14A (Account Management) is independent of Phase 13.
- Phase 14B (Real-Time Engine) depends on alert infrastructure and requires user to upgrade Alpaca subscription before live testing.

---

## Verification Checklist Per Phase

### Phase 11
- [ ] `calculate_kpis()` returns all new KPI fields
- [ ] Strategy detail shows tabbed KPI panel with all categories
- [ ] Edge Check toggle shows 21-MA + Bollinger Bands on equity curve
- [ ] Rolling metrics chart renders with Win Rate / PF / Sharpe toggles
- [ ] Return distribution shows Histogram / Box Plot / Violin views
- [ ] Markov section shows transition probabilities, streak chart, edge decay
- [ ] Market regime detection scatter plot renders
- [ ] Markov insights generate 2-4 text summaries
- [ ] Guards work: metrics show "—" when trade count is below threshold
- [ ] Existing KPIs on strategy cards unchanged

### Phase 12
- [ ] Strategy origin selectbox shows "Standard" and "Webhook Inbound"
- [ ] Webhook Inbound hides entry/exit trigger sections
- [ ] Webhook config fields render (secret, JSON path, signal mapping)
- [ ] CSV upload parses signals and pairs entries/exits
- [ ] Backtest runs with stop/target logic on uploaded signals
- [ ] KPIs and equity curve generate from webhook backtest
- [ ] Webhook server starts and receives POST requests
- [ ] Inbound signals are processed and added to stored_trades
- [ ] `OPTIMIZABLE_PARAMS` updated for webhook fields

### Phase 13
- [ ] Alert matching correlates fired alerts with forward test trades
- [ ] Three-color equity curve renders (blue/orange/green segments)
- [ ] Strategy card captions show colored BT/Fwd/Live durations
- [ ] Mini equity curves show three-color segments
- [ ] Discrepancies displayed on cards (badge) and detail (table)
- [ ] Alert tracking toggle persists to strategy dict
- [ ] "Live vs. Forward" comparison tab shows side-by-side KPIs
- [ ] Matching runs on each data refresh when tracking enabled

### Phase 14A
- [ ] Account Management tab appears on portfolio detail
- [ ] Deposit and withdrawal forms work, append to ledger
- [ ] Balance computed correctly from ledger sum
- [ ] Balance history chart renders
- [ ] Trading notes save and render markdown
- [ ] Trading P&L auto-generates ledger entries from live executions

### Phase 14B (requires Alpaca SIP subscription)
- [ ] Connections section in Settings shows Alpaca status and data feed selection
- [ ] Real-time engine scaffolded (`src/realtime_engine.py` created)
- [ ] User prompted to confirm Alpaca SIP access at stop point
- [ ] Real-time engine connects to Alpaca WebSocket (when configured)
- [ ] Intra-bar triggers fire on tick data
