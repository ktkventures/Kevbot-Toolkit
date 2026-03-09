# Ralph Wiggum Alert Engine — Implementation Spec

**Status:** Approved — Building on `ralph-wiggum` branch
**Date:** 2026-02-27
**Replaces:** realtime_engine.py (Phase 27 streaming engine), alert_monitor.py (polling engine)

---

## 1. Why We're Starting Fresh

### 1.1 The Fundamental Problem

The current alert system re-runs the **entire backtest pipeline** (indicators → interpreters → triggers → `generate_trades()`) on every evaluation cycle just to answer: "did a new signal happen?" This is O(N × M) work per cycle (N=bars, M=indicators) to answer a question that should be **two comparisons**.

With 5 symbols, this saturates the CPU and makes the Streamlit app unresponsive. TradingView handles thousands of symbols per user because their alert evaluation is O(1) per tick.

### 1.2 Problems Encountered Over 5 Days of Phase 27

| # | Problem | Root Cause |
|---|---------|------------|
| 1 | CPU pegged at 90%+ with 5 symbols | Full pipeline re-run every 500ms per symbol on thread pool; GIL serializes to >100% one core |
| 2 | Phantom entry+exit alerts at same timestamp | Sliding window (1000 bars) vs full eval (2000 bars) produces different trade lists; `_diff_trades()` sees count change as "new trade" |
| 3 | `bar_count_exit` firing immediately | Trade-list diffing interprets window-boundary artifacts as instant entry+exit |
| 4 | Streamlit app freezes when engine running | Engine threads + pickle I/O + Streamlit reruns competing for CPU |
| 5 | Data age indicator bouncing "Live" ↔ "19s" | Thread pool scheduling contention; pickle writes delayed |
| 6 | Slow engine startup (30-60s) | Loading 120 days of 1-min bars for warmup (115K bars for INTC) |
| 7 | After-hours candle mismatch | Streaming tick-built bars don't align with Alpaca REST historical bars |
| 8 | Backfill cycles disrupting indicators | REST bar replacement shifts indicator values mid-cycle |
| 9 | Chart position resets on fragment refresh | Streamlit limitation — no persistent chart scroll state |
| 10 | Intra-bar triggers unreliable | Full pipeline too slow for sub-second detection; TriggerLevelCache exists but unused |

### 1.3 What's NOT Broken (Keep As-Is)

- **Backtest engine** (`triggers.py: generate_trades()`) — works correctly, matches TradingView
- **Indicator calculations** (`indicators.py`) — correct implementations
- **Interpreter state classification** (`interpreters.py`) — correct
- **Trigger detection** (`interpreters.py: detect_all_triggers()`) — correct boolean columns
- **Portfolio system** (`portfolios.py`) — CRUD, compliance, requirement sets
- **Webhook delivery** (`alerts.py: send_webhook()`) — works, 2 retries, 5s timeout
- **Payload template rendering** (`alerts.py: render_payload()`) — works
- **Alert storage format** (`alerts.json`) — adequate schema
- **Alert config format** (`alert_config.json`) — adequate schema
- **UI for alert management** — strategy alerts tab, portfolio webhook config, alert analysis

---

## 2. Architecture Overview

### 2.1 Design Principles

1. **O(1) per tick** — Indicator updates are incremental, not full-history recomputation
2. **State machine for positions** — No trade-list diffing; explicit enter/exit state transitions
3. **Separate process** — Alert engine runs independently from Streamlit
4. **Event-driven** — Signals fire immediately when conditions are met, not on a poll cycle
5. **Backtest alignment** — Uses the same trigger definitions so backtest results predict live behavior
6. **Provider-agnostic** — Data source (Alpaca, Massive, etc.) is a pluggable adapter

### 2.2 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT APP                         │
│                                                         │
│  Strategy Builder ─── Backtest Engine (unchanged)       │
│  My Strategies    ─── Portfolio Engine (unchanged)      │
│  Live Chart Tab   ─── reads chart_data.pkl (optional)   │
│  Alert Analysis   ─── reads alerts.json                 │
│  Monitor Controls ─── sends start/stop via IPC          │
│                                                         │
└─────────────┬───────────────────────────────────────────┘
              │ IPC: Unix socket or file-based
              ▼
┌─────────────────────────────────────────────────────────┐
│              RALPH WIGGUM ALERT ENGINE                   │
│              (separate process)                          │
│                                                         │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ Data Adapter │──▶│ Bar Builder  │──▶│ Incremental │  │
│  │ (Alpaca WS)  │   │ (per symbol/ │   │ Indicator   │  │
│  │              │   │  timeframe)  │   │ Engine      │  │
│  └─────────────┘   └──────┬───────┘   └──────┬──────┘  │
│                           │                   │         │
│                    tick    │            bar    │         │
│                    update  │            close  │         │
│                           ▼                   ▼         │
│                    ┌──────────────────────────────┐      │
│                    │   Trigger Evaluator          │      │
│                    │                              │      │
│                    │  Per-tick:                   │      │
│                    │   • Price vs cached levels   │      │
│                    │   • O(1) cross detection     │      │
│                    │                              │      │
│                    │  Per-bar-close:              │      │
│                    │   • Update indicator values  │      │
│                    │   • Evaluate trigger booleans│      │
│                    │   • Update cached levels     │      │
│                    └──────────┬───────────────────┘      │
│                               │                         │
│                        signal │                         │
│                               ▼                         │
│                    ┌──────────────────────────────┐      │
│                    │   Position State Machine     │      │
│                    │                              │      │
│                    │  FLAT ──entry_trigger──▶ IN  │      │
│                    │  IN ───exit_trigger───▶ FLAT │      │
│                    │  IN ───stop_hit───────▶ FLAT │      │
│                    │  IN ───bar_count──────▶ FLAT │      │
│                    │  IN ───target_hit─────▶ FLAT │      │
│                    └──────────┬───────────────────┘      │
│                               │                         │
│                        alert  │                         │
│                               ▼                         │
│                    ┌──────────────────────────────┐      │
│                    │   Alert Dispatcher           │      │
│                    │                              │      │
│                    │  • Enrich with portfolio ctx │      │
│                    │  • Write to alerts.json      │      │
│                    │  • Fire webhooks (async)     │      │
│                    │  • Write chart data pickle   │      │
│                    │  • Log for fidelity audit    │      │
│                    └─────────────────────────────-┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Requirements

### 3.1 Core Alert Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| R1 | Entry alerts fire within 1 second of trigger condition being met | MUST |
| R2 | Exit alerts (signal-based) fire within 1 second of trigger condition | MUST |
| R3 | Stop-loss exits fire within 1 second of price crossing stop level | MUST |
| R4 | Bar-count exits fire on the correct bar close (not before, not same-bar as entry) | MUST |
| R5 | Target exits fire within 1 second of price crossing target level | MUST |
| R6 | No phantom alerts — every alert corresponds to a real trigger condition | MUST |
| R7 | No missed alerts — every trigger condition produces an alert | MUST |
| R8 | No duplicate alerts — each trigger event produces exactly one alert | MUST |
| R9 | Alerts include accurate price, timestamp, trigger name, and direction | MUST |
| R10 | Support 50+ symbols without CPU degradation | MUST |
| R11 | Support 10-second candles (future) | SHOULD |

### 3.2 Webhook Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| W1 | Webhooks fire within 2 seconds of alert generation | MUST |
| W2 | Per-portfolio webhook routing (each portfolio has its own webhook config) | MUST |
| W3 | Event filtering per webhook (entry_long, exit_long, entry_short, exit_short, compliance_breach) | MUST |
| W4 | Custom payload templates with {{placeholder}} substitution | MUST |
| W5 | Quantity calculation: `int(position_risk / abs(price - stop_price))` | MUST |
| W6 | Retry: 2 attempts, 5s timeout per attempt | MUST |
| W7 | Delivery status recorded on alert record (webhook_deliveries array) | MUST |
| W8 | order_action: "buy" for LONG entry, "sell" for SHORT entry, "close" for exits | MUST |

### 3.3 Position State Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| P1 | Position state is a persistent state machine, not derived from trade-list replay | MUST |
| P2 | On startup, position state is initialized from the last alert record (not re-running generate_trades) | MUST |
| P3 | Entry signal only fires when FLAT; exit signal only fires when IN_POSITION | MUST |
| P4 | Confluence gating: entry only fires when all required confluence conditions are met | MUST |
| P5 | Stop price is computed at entry time and tracked until exit | MUST |
| P6 | Bar count tracked from entry bar, not from arbitrary reference point | MUST |
| P7 | Target price computed at entry time (if target_config specified) | MUST |

### 3.4 Indicator & Trigger Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| I1 | Incremental indicator update on each bar close: O(1) per indicator | MUST |
| I2 | Intra-bar price crossing detection for level-based triggers (VWAP, UT Bot stop, EMA, Bollinger, SuperTrend) | MUST |
| I3 | Interpreter state classification matches backtest output | MUST |
| I4 | Trigger boolean evaluation matches backtest output | MUST |
| I5 | Multi-timeframe confluence: secondary TF interpreters forward-filled into primary TF | MUST |
| I6 | Warmup: load enough historical bars to converge all indicators (200 bars minimum) | MUST |

### 3.5 Data Fidelity Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| D1 | Bar data matches Alpaca REST historical bars (no tick-aggregation drift) | MUST |
| D2 | Periodic REST bar reconciliation to correct any streaming drift | MUST |
| D3 | After-hours bar alignment: streaming bars must match historical bar boundaries | MUST |
| D4 | Indicator values at bar close match what the backtest engine would compute | MUST |

### 3.6 Fidelity Verification Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| V1 | Fidelity audit log: every bar close records indicator values, trigger states, and position state | MUST |
| V2 | Backtest comparison: can run generate_trades() on same data and compare signal-by-signal | MUST |
| V3 | Alert Analysis tab: FT vs Live comparison with slippage, missed/phantom detection | KEEP |
| V4 | Live execution matching: alerts matched to stored_trades with time-window tolerance | KEEP |
| V5 | Latency tracking: timestamp of trigger condition vs timestamp of webhook delivery | SHOULD |

### 3.7 UI / Integration Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| U1 | Start/Stop engine from Streamlit UI (Alerts & Signals page) | MUST |
| U2 | Engine status visible in UI (running, connected, tick count, symbols) | MUST |
| U3 | Live chart data available via pickle for Streamlit fragment display | SHOULD |
| U4 | Hot-reload: adding/removing strategies without engine restart | SHOULD |
| U5 | Graceful shutdown: stop engine cleanly on SIGTERM | MUST |
| U6 | Process management: PID file, liveness check, auto-restart on crash | SHOULD |
| U7 | alert_config.json and alerts.json formats remain backward-compatible | MUST |
| U8 | Portfolio context enrichment: same logic as current (portfolio_context array) | MUST |
| U9 | Compliance breach detection: same logic as current | KEEP |

---

## 4. Incremental Indicator Engine

### 4.1 The Key Insight

Every indicator we use can be updated incrementally with O(1) work per bar close:

| Indicator | Incremental Formula | State to Cache |
|-----------|-------------------|----------------|
| EMA(n) | `ema = alpha * close + (1 - alpha) * prev_ema` | `prev_ema` |
| MACD | Two EMAs (12, 26) + signal EMA(9) | `ema_12, ema_26, signal_ema` |
| VWAP | `cum_pv += price * vol; cum_vol += vol; vwap = cum_pv / cum_vol` | `cum_pv, cum_vol` (reset daily) |
| ATR(n) | `atr = alpha * tr + (1 - alpha) * prev_atr` (Wilder smoothing) | `prev_atr, prev_close` |
| RVOL | `rolling_avg_vol` (can use EMA of volume) | `vol_ema` |
| Bollinger | SMA(20) + rolling stddev — maintain circular buffer of 20 closes | `close_buffer[20]` |
| SuperTrend | ATR-based bands + direction flip logic | `prev_supertrend, prev_direction, prev_atr` |
| UT Bot | ATR trailing stop with direction | `prev_stop, prev_direction, prev_atr` |

**Per-tick work (intra-bar):** Only update the partial bar's OHLCV and check cached levels. No indicator recalculation.

**Per-bar-close work:** Update each indicator with the new bar's data using cached state. ~10 microseconds per indicator × ~10 indicators = ~100 microseconds total. Compare to current approach: ~250 milliseconds (2,500× slower).

### 4.2 Indicator State Store

```python
@dataclass
class IndicatorState:
    """Cached state for incremental indicator computation per symbol/timeframe."""
    # EMA states: {period: last_ema_value}
    ema: Dict[int, float] = field(default_factory=dict)

    # MACD
    macd_ema_fast: float = 0.0
    macd_ema_slow: float = 0.0
    macd_signal_ema: float = 0.0

    # VWAP (resets daily)
    vwap_cum_pv: float = 0.0
    vwap_cum_vol: float = 0.0
    vwap_date: Optional[date] = None  # reset when date changes

    # ATR
    atr_value: float = 0.0
    prev_close: float = 0.0

    # Bollinger
    close_buffer: deque = field(default_factory=lambda: deque(maxlen=20))

    # UT Bot
    utbot_stop: float = 0.0
    utbot_direction: int = 1  # 1 = long, -1 = short

    # SuperTrend
    supertrend_value: float = 0.0
    supertrend_direction: int = 1

    # Bar count since session start (for bar-count exits)
    bar_count: int = 0
```

### 4.3 Strategy-Scoped Indicator Resolution

Each strategy declares its needs at build time. The engine resolves what to compute:

```
Strategy config:
  entry_trigger: "utbot_v2_buy_ib"
  exit_triggers: []
  bar_count_exit: 4
  confluence: ["1M-EMA_STACK-SML"]
  stop_config: {"method": "atr", "atr_mult": 1.5}

Engine resolves → needed indicators:
  - ATR (required by UT Bot + stop config)
  - UT Bot trailing stop (required by entry trigger)
  - EMA 8, 21, 50 (required by EMA_STACK confluence)
  - Close, High, Low (always needed for bar data)

NOT computed:
  - MACD, VWAP, RVOL, Bollinger, SuperTrend, SR Channels, etc.
```

When multiple strategies share a symbol/timeframe, the engine computes the **union** of their required indicators — no duplication, no waste.

A resolution function maps trigger IDs and confluence IDs back to their parent confluence group (via `TEMPLATES` in `confluence_groups.py`), then extracts which indicator functions that group requires.

### 4.4 Warmup Strategy

On engine start, per symbol/timeframe:
1. Load last 200 bars from Alpaca REST API (covers EMA-200 convergence).
2. Run only the **required** indicator functions (not the full pipeline) to establish state.
3. Extract the final indicator values into `IndicatorState`.
4. From this point forward, only incremental updates.

This warmup takes ~1-2 seconds total (200 bars × required indicators only), compared to the current 30-60 seconds.

---

## 5. Trigger Evaluator

### 5.1 Bar-Close Triggers

After each bar close and incremental indicator update:

1. Run interpreters on the **current bar only** (not all bars):
   - EMA_STACK: compare ema_8, ema_21, ema_50, ema_200
   - MACD_LINE: check macd > 0 or macd < 0
   - etc.

2. Evaluate trigger booleans on the **current bar only**:
   - `utbot_buy`: direction flipped from -1 to 1 this bar
   - `ema_cross_above`: ema_fast crossed above ema_slow this bar
   - etc.

3. Check confluence: current interpreter states → confluence record set → subset check.

4. If trigger is True and confluence is met → fire signal to Position State Machine.

### 5.2 Intra-Bar (Tick-Level) Triggers

For triggers that can fire mid-bar (suffixed `_ib` in strategy config):

**On each tick:**
1. Update partial bar OHLCV (O(1) — just compare/update high, low, close, add volume).
2. For each active intra-bar trigger on this symbol:
   - Look up the cached level (e.g., UT Bot trailing stop = 178.80).
   - Compare current price to level.
   - If crossed in the required direction AND not already fired this bar → fire signal.
3. "Once per bar" semantics: after firing, lock until next bar close resets.

**Cached levels updated on each bar close** from the incremental indicator values.

**Supported intra-bar triggers** (from existing `INTRABAR_LEVEL_MAP`):
- VWAP cross above/below
- UT Bot buy/sell (price crosses trailing stop)
- SuperTrend buy/sell
- Bollinger upper/lower/mid cross
- EMA price position cross
- SR Channel support/resistance cross
- All `_v2` (Confirmed) variants

### 5.3 Trigger Match Guarantee

To ensure backtest alignment:
- Bar-close triggers use the **exact same boolean logic** as `detect_all_triggers()` in `interpreters.py`, but evaluated on a single row instead of vectorized across the DataFrame.
- We extract the per-row trigger logic into standalone functions that both the backtest (vectorized) and the alert engine (single-row) can call.
- Fidelity audit (Section 8) validates this match continuously.

---

## 6. Position State Machine

### 6.1 States

```
                  ┌─────────┐
     entry_trigger│         │exit_trigger / stop / bar_count / target
          ┌───────┤  FLAT   │◄──────────────────────┐
          │       │         │                       │
          │       └─────────┘                       │
          │                                         │
          ▼                                         │
     ┌──────────┐                              ┌────┴─────┐
     │ ENTERING │─── signal saved ────────────▶│   IN     │
     │ (transient)                              │ POSITION │
     └──────────┘                              └──────────┘
```

### 6.2 Per-Strategy State

```python
@dataclass
class PositionState:
    """Persistent state for one strategy."""
    status: Literal['FLAT', 'IN_POSITION'] = 'FLAT'
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    entry_bar_index: int = 0          # bar count at entry (for bar_count_exit)
    direction: str = 'LONG'
    entry_trigger: str = ''
```

### 6.3 State Transitions

**FLAT → IN_POSITION** (on entry trigger):
1. Trigger evaluator fires entry signal.
2. Confluence check passes.
3. Compute stop price using strategy's `stop_config` (ATR, swing, fixed).
4. Compute target price if `target_config` exists.
5. Record entry details.
6. Transition to IN_POSITION.
7. Generate alert → enrich → save → webhook.

**IN_POSITION → FLAT** (on exit):
Multiple exit paths, checked in priority order on each tick/bar:
1. **Stop loss** (per-tick): price crosses `stop_price` → exit at stop (or open if gap).
2. **Target** (per-tick): price crosses `target_price` → exit at target (or open if gap).
3. **Signal exit** (bar-close or intra-bar): exit trigger column is True.
4. **Bar count exit** (bar-close only): `current_bar_count - entry_bar_index >= bar_count_exit`.
5. **Opposite signal** (bar-close): if configured, opposite entry trigger fires.

On exit:
1. Record exit details (price, time, reason).
2. Transition to FLAT.
3. Generate alert → enrich → save → webhook.

### 6.4 State Persistence

Position state is persisted to `engine_state.json` on every transition:
```json
{
    "positions": {
        "29": {
            "status": "IN_POSITION",
            "entry_price": 23.88,
            "entry_time": "2026-02-27T19:29:19+00:00",
            "stop_price": 23.45,
            "target_price": null,
            "entry_bar_index": 1847,
            "direction": "LONG",
            "entry_trigger": "utbot_v2_buy_ib"
        }
    },
    "indicator_states": { ... },
    "last_bar_times": { "SPY/60": "2026-02-27T19:29:00+00:00", ... }
}
```

On engine restart, state is loaded from this file → no need to re-run `generate_trades()` to determine position state. If the file is missing or corrupt, fall back to warmup-based initialization.

### 6.5 Startup Position Initialization (Warmup Fallback)

If no `engine_state.json` exists (first start or state file lost):
1. Load 200 bars from REST API.
2. Run `generate_trades()` once (the backtest function) to determine current position.
3. If last trade is open → IN_POSITION with entry details from that trade.
4. Save state to `engine_state.json`.

This is the **only** time `generate_trades()` runs in the alert engine — at startup, once.

---

## 7. Alert Dispatcher

### 7.1 Signal → Alert → Webhook Pipeline

When the Position State Machine transitions:

```python
def on_signal(signal_type, strategy, price, timestamp, trigger, **details):
    # 1. Build alert dict (same schema as current alerts.json)
    alert = {
        'type': signal_type,          # 'entry_signal' or 'exit_signal'
        'trigger': trigger,
        'price': price,
        'bar_time': str(current_bar_time),
        'stop_price': details.get('stop_price'),
        'atr': current_atr,
        'level': 'strategy',
        'strategy_id': strategy['id'],
        'strategy_name': strategy.get('name'),
        'symbol': strategy.get('symbol'),
        'direction': strategy.get('direction'),
        'risk_per_trade': strategy.get('risk_per_trade', 100.0),
        'timeframe': strategy.get('timeframe'),
        'source': 'ralph',
        # Exit-specific fields:
        'entry_price': details.get('entry_price'),      # only on exits
        'entry_stop_price': details.get('entry_stop_price'),  # only on exits
    }

    # 2. Enrich with portfolio context (reuse existing function)
    alert = enrich_signal_with_portfolio_context(alert, strategy['id'])

    # 3. Save to alerts.json (reuse existing function)
    alert = save_alert(alert)

    # 4. Fire webhooks asynchronously
    asyncio.create_task(deliver_webhooks(alert))

    # 5. Log to fidelity audit
    audit_log.record(alert)

    # 6. Update chart data pickle (for Streamlit live chart)
    write_chart_pickle(symbol, timeframe)
```

### 7.2 Webhook Delivery

Reuse the existing `deliver_alert()` logic from `alert_monitor.py` with one change: run it as an async task instead of on the thread pool, since the engine is already async.

The existing flow is correct:
1. Resolve portfolios from `alert.portfolio_context`.
2. For each portfolio → for each enabled webhook → check event filter.
3. `build_placeholder_context()` → `render_payload()` → `send_webhook()`.
4. Record delivery results on alert.

### 7.3 Backward Compatibility

- `alerts.json` format: identical schema, new alerts have `source: 'ralph'`.
- `alert_config.json` format: read as-is, no changes needed.
- `portfolios.json`: read as-is for enrichment.
- `strategies.json`: read for strategy configs (triggers, confluence, stop config, etc.).
- `live_executions` matching: works as-is since alert format is compatible.

---

## 8. Fidelity Verification System

### 8.1 The Core Question

> "Does the alert engine fire signals at the same points where the backtest engine would have generated trades?"

### 8.2 Continuous Fidelity Audit

The engine maintains a **fidelity audit log** (`engine_audit.jsonl`, append-only):

```jsonl
{"ts": "2026-02-27T19:29:00Z", "symbol": "GME", "tf": 60, "bar_close": 23.88, "indicators": {"ema_8": 23.91, "ema_21": 23.85, "atr": 0.42, "utbot_stop": 23.45}, "triggers": {"utbot_v2_buy": false, "utbot_v2_buy_ib": true}, "position": {"29": "IN_POSITION"}}
```

On each bar close, records:
- All indicator values (incremental engine)
- All trigger booleans (incremental engine)
- Current position state per strategy

### 8.3 Offline Fidelity Check (On-Demand)

A verification script that:
1. Loads the same bar data the engine processed (from Alpaca REST, same time range).
2. Runs the full backtest pipeline (`_run_pipeline()` + `generate_trades()`).
3. Compares indicator values at each bar close against audit log.
4. Compares trade entry/exit points against alerts.
5. Reports:
   - **Indicator drift**: any bar where incremental vs batch indicators differ by > 0.01%.
   - **Missed signals**: backtest generated a trade but no alert was fired.
   - **Phantom signals**: alert was fired but backtest didn't generate a trade.
   - **Timing accuracy**: bar close where signal fired vs bar close where backtest entered.

### 8.4 Live Fidelity Dashboard (UI)

Integrate into existing Alert Analysis tab:
- "Engine Fidelity" section showing:
  - Last N bar closes with indicator match status (green/red).
  - Signal match rate (alerts vs what backtest would have produced).
  - Drift alerts: if incremental indicators start diverging, warn and suggest engine restart.

### 8.5 Self-Healing

If the fidelity audit detects indicator drift > threshold:
1. Log a warning.
2. Re-run full batch indicator pipeline on cached bars (200-bar window).
3. Reset `IndicatorState` from batch results.
4. Resume incremental updates.

This should rarely happen (incremental formulas are mathematically identical), but provides a safety net.

---

## 9. Data Layer

### 9.1 Data Adapter Interface

```python
class DataAdapter(ABC):
    """Provider-agnostic data source interface."""

    @abstractmethod
    async def subscribe_trades(self, symbols: List[str],
                                callback: Callable[[str, float, int, datetime], None]):
        """Subscribe to real-time trade ticks."""
        pass

    @abstractmethod
    async def get_historical_bars(self, symbol: str, timeframe: str,
                                   start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch historical OHLCV bars for warmup/reconciliation."""
        pass

    @abstractmethod
    async def close(self):
        """Clean shutdown."""
        pass
```

### 9.2 Alpaca Adapter

```python
class AlpacaAdapter(DataAdapter):
    """Alpaca SIP WebSocket + REST adapter."""
    # Uses alpaca-py StockDataStream for ticks
    # Uses alpaca-py StockHistoricalDataClient for REST bars
```

### 9.3 Bar Builder

Reuse existing `BarBuilder` class (it works correctly). One instance per symbol/timeframe:
- Aggregates ticks into OHLCV bars.
- Fires callback on bar completion.
- Tracks partial bar for intra-bar price checks.

### 9.4 Bar Reconciliation

Every 60 seconds (configurable), fetch the last 5 completed bars from REST API and compare against locally-built bars. If any differ:
1. Replace the local bar data.
2. Re-compute indicators for those bars (incremental update from the corrected values).
3. Do NOT re-evaluate triggers for past bars — only update cached state for future evaluations.

This handles the after-hours candle mismatch issue without disrupting the current bar's evaluation.

---

## 10. Process Architecture

### 10.1 Engine Process

```
ralph_engine.py — standalone script

  Usage:
    python ralph_engine.py                   # Normal start
    python ralph_engine.py --status          # Print current engine status
    python ralph_engine.py --stop            # Send SIGTERM to running engine

  main()
    ├── Load config (alert_config.json, strategies.json, portfolios.json)
    ├── Resolve monitored strategies (same logic as get_monitored_strategies)
    ├── Load/restore engine_state.json (or warmup from scratch)
    ├── Create DataAdapter (Alpaca)
    ├── Create BarBuilders (per symbol/timeframe)
    ├── Create IndicatorStates (per symbol/timeframe)
    ├── Create PositionStates (per strategy)
    ├── Write PID to engine_status.json
    ├── Start async event loop:
    │     ├── Data subscription (ticks → bar builders → indicator updates)
    │     ├── Trigger evaluation (bar close + intra-bar)
    │     ├── Position state machine
    │     ├── Alert dispatch (save + webhook)
    │     ├── Bar reconciliation (every 60s)
    │     ├── Strategy hot-reload (every 5 min)
    │     ├── Chart pickle writer (every 2s)
    │     ├── Fidelity audit logger (every bar close)
    │     └── State persistence (on every position transition)
    └── Signal handlers (SIGTERM → graceful shutdown + state save)
```

The CLI interface means the engine can be started/stopped/queried without the Streamlit UI, which is essential for:
- Development iteration (Claude can start/stop/check the engine directly)
- Railway deployment (worker process launched via CLI)
- Automated testing (start engine, wait, check alerts, stop)

### 10.2 IPC with Streamlit

**Streamlit → Engine:**
- Start: launch subprocess, write PID to `engine_status.json`.
- Stop: send SIGTERM to PID.
- Config reload: write a flag file `engine_reload.flag` → engine watches for it.

**Engine → Streamlit:**
- `engine_status.json`: running state, tick count, symbols, last tick time, errors.
- `alerts.json`: alert records (existing format).
- `live_data_{symbol}_{tf}.pkl`: chart data for live chart tab (existing format).
- `engine_state.json`: position states (for UI display).

No in-process communication. No shared memory. No threading conflicts with Streamlit.

### 10.3 Resource Budget

With the incremental architecture:

| Operation | Frequency | CPU per op | Total CPU (5 symbols) |
|-----------|-----------|-----------|----------------------|
| Tick → bar OHLCV update | ~100/sec/symbol | 1μs | 0.5ms/sec |
| Intra-bar level check | ~100/sec/symbol | 5μs | 2.5ms/sec |
| Bar close indicator update | 1/min/symbol | 100μs | 0.5ms/min |
| Bar close trigger eval | 1/min/symbol | 50μs | 0.25ms/min |
| Pickle write | 1/2sec | 20ms | 10ms/sec |
| REST reconciliation | 1/min | 200ms | 200ms/min |

**Total: ~1-2% CPU for 5 symbols.** Headroom for 50+ symbols easily.

---

## 11. Chart Data for Live Chart Tab

### 11.1 Purpose

The live chart's primary purpose is **visual verification** — the user sees the chart with indicators overlaid and can visually confirm "did this trigger fire at the right point?" It is not a full-featured charting platform. It supplements the backtest price chart by showing real-time data.

### 11.2 What to Write

The engine writes `live_data_{symbol}_{tf}.pkl` every 2 seconds per active symbol/timeframe:

- Last N bars of OHLCV + indicator values (N configurable per timeframe, default 300).
- Partial bar included (last row, updated with latest tick).
- Trigger boolean columns included (for entry/exit markers on chart).
- Timeframe matches whatever the strategy uses (1Min, 5Min, 15Min, etc.).

### 11.3 How It's Different

Currently, the engine re-runs the full pipeline to generate this data. In Ralph Wiggum, the indicator values are already maintained incrementally in a circular buffer. The pickle write just serializes the buffer into a DataFrame. No computation — just formatting and I/O (~20ms).

### 11.4 Chart Bar Count Defaults

| Timeframe | Default Bars | Approx Coverage |
|-----------|-------------|-----------------|
| 1Min | 300 | ~5 hours |
| 5Min | 200 | ~17 hours |
| 15Min | 150 | ~38 hours |
| 30Min | 100 | ~50 hours |
| 1Hour | 100 | ~100 hours |

---

## 12. Migration Plan

### 12.1 Phase 1: Build Core Engine (No UI Changes)

1. Create `ralph_engine.py` with:
   - DataAdapter (Alpaca)
   - BarBuilder (reuse existing)
   - IncrementalIndicatorEngine
   - TriggerEvaluator (single-row evaluation)
   - PositionStateMachine
   - AlertDispatcher (reuse `save_alert()`, `enrich_signal_with_portfolio_context()`, `send_webhook()`)
   - State persistence (`engine_state.json`)
   - Fidelity audit logger

2. Unit tests:
   - Incremental indicators match batch indicators on 1000+ bar sequences.
   - Trigger evaluator matches `detect_all_triggers()` output on same data.
   - Position state machine produces same entry/exit points as `generate_trades()`.

### 12.2 Phase 2: Integration

1. Wire up Streamlit start/stop controls to launch `ralph_engine.py` as subprocess.
2. Point live chart tab at pickle files written by Ralph engine.
3. Alert Analysis tab reads `alerts.json` (no changes needed — same format).
4. Remove old `realtime_engine.py` import paths from `app.py`.
5. Keep `alert_monitor.py` as degraded-mode fallback (polling) — no changes needed.

### 12.3 Phase 3: Verification & Polish

1. Run both engines side-by-side during market hours:
   - Old engine writes to `alerts_old.json`.
   - Ralph engine writes to `alerts.json`.
   - Compare signal-by-signal.

2. Fidelity audit validation: run offline checker on a full trading day's data.

3. Remove old streaming engine code once Ralph is validated.

---

## 13. What We're Reusing vs Building New

### Reusing (from existing codebase)

| Component | Source File | Usage |
|-----------|------------|-------|
| `BarBuilder` | realtime_engine.py | Tick → OHLCV aggregation |
| `save_alert()` | alerts.py | Write alert to alerts.json |
| `enrich_signal_with_portfolio_context()` | alerts.py | Portfolio context enrichment |
| `send_webhook()` | alerts.py | HTTP POST with retry |
| `build_placeholder_context()` | alerts.py | Webhook template variables |
| `render_payload()` | alerts.py | Template substitution |
| `deliver_alert()` | alert_monitor.py | Full webhook routing pipeline |
| `load_alert_config()` | alerts.py | Config loading with migration |
| `get_monitored_strategies()` | alert_monitor.py | Strategy resolution |
| `INTRABAR_LEVEL_MAP` | realtime_engine.py | Level-based trigger definitions |
| `SESSION_HOURS` | data_loader.py | Market session time ranges |
| Indicator formulas | indicators.py | Reference for incremental implementations |
| Interpreter logic | interpreters.py | Reference for single-row evaluation |
| Trigger logic | interpreters.py | Reference for single-row evaluation |

### Building New

| Component | Description |
|-----------|-------------|
| `IncrementalIndicatorEngine` | O(1) indicator updates per bar close |
| `TriggerEvaluator` | Single-row trigger evaluation (not vectorized) |
| `PositionStateMachine` | Explicit state machine replacing trade-list diffing |
| `DataAdapter` | Provider-agnostic data interface |
| `AlpacaAdapter` | Alpaca-specific implementation |
| `FidelityAuditor` | Continuous indicator/trigger verification |
| `StatePersistence` | engine_state.json read/write |
| `ralph_engine.py` | Main entry point / process orchestration |

---

## 14. Design Decisions (Resolved)

1. **Indicator subset — Strategy-scoped only.** Each strategy declares its entry triggers, exit triggers, and confluences at build time. The engine only computes indicators required by those specific triggers/confluences for that strategy. If a strategy uses UT Bot entry + bar count exit, only UT Bot indicators (ATR, trailing stop) are computed — not MACD, not Bollinger, not SuperTrend. This is critical for scalability: eventually there may be thousands of confluence packs, and loading all of them per strategy would be wasteful.

2. **Confluence group indicators — Derived from strategy config, not global enable/disable.** The confluence pack enable/disable in the UI controls what appears in the Strategy Builder. Once a strategy is built, everything it needs is saved on the strategy record (triggers, confluences, interpreter keys). The engine reads the strategy config and extracts exactly which indicator groups are needed. On hot-reload (new strategy added), the engine initializes only the indicators that new strategy requires.

3. **10-second candles — Deferred.** If we go the 10s route, we'd likely switch to Massive (which offers 10s historical data). The DataAdapter abstraction makes this a config change. Don't build 10s-specific logic now — just ensure the BarBuilder and indicator engine work for any timeframe value, which they already do.

4. **Chart data granularity — Configurable per timeframe.** The live chart should display whatever timeframe the strategy uses (1Min, 5Min, 15Min, etc.). Bar count per chart should be configurable. The default 300 bars works well for 1-min (5 hours of data), but for 5-min bars that would be 25 hours — may want fewer bars for higher timeframes.

5. **Railway deployment — Plan for it in the architecture.** The standalone process design naturally supports Railway: run `ralph_engine.py` as a worker process alongside the Streamlit web process. The IPC via JSON files works across processes on the same filesystem. Environment variables (`ALPACA_API_KEY`, etc.) work identically.

6. **Multiple instances — No.** Portfolios already handle routing differentiation. The same strategy can be in multiple portfolios, each with different webhook targets (paper trading vs live broker). The engine runs one instance monitoring all active strategies. Different "behaviors" are handled at the portfolio webhook level, not at the engine level.

### 14.1 Additional Design Decisions

7. **Engine auto-start for development iteration.** The engine must be startable programmatically (not just via UI button click). During Ralph Wiggum development, the engine needs to auto-start on app launch or be startable via a CLI command (`python ralph_engine.py --start`) so that iterative testing doesn't require manual UI interaction. The Streamlit UI start/stop controls remain for end-user use.

8. **Iteration speed for fidelity validation.** Phantom alerts and missed alerts typically manifest within ~20 minutes of live data, regardless of market session (pre-market, RTH, extended hours). Fidelity checks should run continuously and report in near-real-time, not require a full trading day of data collection before assessing.

---

## 15. Success Criteria

The Ralph Wiggum Alert Engine is considered complete when:

1. **CPU usage < 5%** with 5 symbols actively streaming during market hours.
2. **Alert latency < 1 second** from trigger condition to webhook delivery.
3. **Zero phantom alerts** over 20+ minutes of live data (any market session).
4. **Zero missed alerts** over 20+ minutes of live data (verified against backtest on same bars).
5. **Indicator fidelity > 99.99%** — incremental values match batch computation.
6. **Startup time < 5 seconds** (warmup from REST + state restore).
7. **Streamlit remains responsive** (< 2 second page loads) while engine runs.
8. **All existing webhook integrations work unchanged** (same payload format, same routing).
9. **Engine startable via CLI** (`python ralph_engine.py`) without UI interaction.
10. **Fidelity audit log** continuously validates indicator/trigger accuracy during operation.

### 15.1 Validation Protocol

For each iteration cycle:
1. Start engine via CLI.
2. Let it run for 20+ minutes during any market session.
3. Check `alerts.json` for any phantom or duplicate alerts.
4. Run the offline fidelity checker against the same bar data to verify no missed signals.
5. Check `engine_audit.jsonl` for indicator drift.
6. Verify webhook deliveries succeeded (if webhooks configured).
7. Confirm Streamlit app remains responsive during engine operation.

Issues found → fix → restart engine → repeat. No need to wait for a full trading day between iterations.
