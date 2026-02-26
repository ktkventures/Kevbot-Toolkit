# Implementation Spec — Phase 27: Unified Pipeline Alert System + Live Chart

## Problem Statement

The streaming engine has two parallel alert paths that diverge:

1. **Bar-close path**: runs `_run_pipeline()` → checks `trig_*` columns (transition-based, matches backtest/chart)
2. **Intra-bar path**: uses `TriggerLevelCache` comparing tick price vs cached level (NOT transition-based, diverges from chart)

This produces a **5:1 alert-to-trade ratio**, phantom alerts, wrong timing deltas, and alerts firing before entry candles (logically impossible). The root cause is that `TriggerLevelCache` performs price-vs-level crossing detection on every tick, while the backtest uses indicator **state transitions** (`(direction == 1) & (direction.shift(1) != 1)`). These are fundamentally different computations.

## Solution

Replace both paths with **one throttled pipeline evaluator** that runs every 500ms on `history + partial bar`. When a `trig_*` column transitions `False→True`, fire the alert. This is identical to what the chart shows — trigger-agnostic, works for any current or future indicator pack.

Add a **Live Chart** tab so the user can visually confirm "trigger plots = alert fires."

---

## Files Modified

| File | Scope |
|------|-------|
| `realtime_engine.py` | Major — new evaluator, BarBuilder enhancement, remove TriggerLevelCache usage |
| `app.py` | Medium — add Live Chart tab to both backtest and forward test views |
| `alerts.py` | Minor — no changes to `_run_pipeline()`, small cleanup |
| `indicators.py`, `interpreters.py`, `triggers.py` | None |

---

## Implementation Steps

### Step 1: BarBuilder.get_df_with_partial()

**File:** `realtime_engine.py`

Add method to `BarBuilder` that returns `history` + current partial bar as a single DataFrame. Uses same `pd.concat` + `pd.DatetimeIndex` pattern as `_append_to_history()`. Returns `history.copy()` if no partial bar exists.

### Step 2: TriggerStateTracker class

**File:** `realtime_engine.py` (new class, after `AlertCooldown`)

Tracks `trig_*` column boolean values per strategy across pipeline evaluations.

- `evaluate(strategy_id, df, trigger_cols)` → returns list of columns that transitioned `False→True`
- `seed(strategy_id, df, trigger_cols)` → initializes state from warmup data (prevents false fires on first eval)

This is the key design insight: comparing boolean column states between evaluations is **trigger-agnostic**. Works for UT Bot, EMA crosses, VWAP, MACD, any future indicator pack — no per-indicator logic needed.

### Step 3: _evaluate_pipeline_throttled()

**File:** `realtime_engine.py` — new method on `SymbolHub`

Replaces both `_check_intrabar_triggers()` AND the signal detection portion of `_on_bar_close()`.

**Flow:**

1. **Throttle check:** `time.monotonic()` — skip if <500ms since last eval
2. **Gate:** skip if `_first_bar_closed` is `False`
3. **Pipeline (per timeframe):**
   - `builder.get_df_with_partial()` → combined DataFrame
   - `_run_pipeline(df)` + `run_indicators_for_group()` per enabled group
   - Cache result in `_last_enriched_df` (for live chart)
4. **Strategy evaluation (per strategy):**
   - Session gate (market hours check)
   - Resolve trigger columns (`trig_{entry_trigger}`, `trig_{exit_triggers}`)
   - `TriggerStateTracker.evaluate()` → list of fired columns
   - Position gate (entry only when flat, exit only when in position, first exit wins)
   - Confluence gate (for entries)
   - Cooldown (one bar period via `AlertCooldown`)
   - Build signal dict, update position state, save alert, deliver webhook
5. **Managed exits:** Check `_check_managed_exits()` for strategies in position with no trigger fires
6. **Live chart data:** Write enriched DataFrame to pickle (`live_data_{symbol}.pkl` via atomic `os.replace()`)

### Step 4: Modify on_tick()

**File:** `realtime_engine.py`

Replace call to `_check_intrabar_triggers(price, timestamp)` with `_evaluate_pipeline_throttled(price, timestamp)`.

### Step 5: Simplify _on_bar_close()

**File:** `realtime_engine.py`

Strip down to housekeeping only:

- Set `_first_bar_closed = True` on first close
- Run `_check_managed_exits()` for stop/bar-count exits (these need completed bar data)
- **Remove:** full pipeline run, `detect_signals()`, trigger cache updates, dedup clearing, signal processing

Managed exits (stop loss, bar count, target) still need bar-close evaluation because they check `bar['low'] <= stop_price` etc.

### Step 6: Update _init_position_state()

**File:** `realtime_engine.py`

After running the warmup pipeline:

- Seed `TriggerStateTracker` with warmup data (prevents false fires on first eval)
- Remove `TriggerLevelCache` seeding
- Keep position state detection via `generate_trades()` (still needed for warmup)

### Step 7: Remove TriggerLevelCache usage from SymbolHub

**File:** `realtime_engine.py`

- Remove `_trigger_cache` attribute from `SymbolHub.__init__`
- Remove `_check_intrabar_triggers()` method entirely
- Remove `_get_intrabar_trigger_bases()` and `_get_group_params_for_trigger()` helper methods
- Remove `_intrabar_fired`, `_ib_cooldown`, `_confluence_met` state dicts
- Remove from `UnifiedStreamingEngine`: `_trigger_cache` creation, passing to hubs, `.clear()` in stop
- **Keep** `TriggerLevelCache` class and `INTRABAR_LEVEL_MAP` dict in file — `triggers.py` imports `INTRABAR_LEVEL_MAP` for backtest fill pricing

### Step 8: Live Chart tab — app.py

**8a: Add tab to both views**

In `render_live_backtest()` and `render_forward_test_view()`:

- Check if streaming engine is active for this strategy's symbol (read `monitor_status.json`)
- If active, insert "Live Chart" into tab names after "Price Chart"
- Adjust tab destructuring accordingly

**8b: render_live_chart_tab() function**

New function decorated with `@st.fragment(run_every=2)`:

- Reads `live_data_{symbol}.pkl` written by the streaming engine
- Uses existing `render_price_chart()` to render — reuses all TradingView LC machinery
- Shows only the strategy's relevant indicators (resolved from its confluence groups)
- Adds trigger markers (entry/exit arrows) where `trig_*` columns are True
- Shows last-bar timestamp, bar count, refresh status
- Visible candles: 100 (last ~1.5 hours for 1-min bars)

**8c: Auto-refresh**

`@st.fragment(run_every=2)` reruns only the fragment every 2 seconds without resetting the full page. The pipeline runs every 500ms (producing fresh data), the chart polls every 2 seconds (rendering the latest data). Alert fires at 500ms cadence; visual chart updates at 2s cadence to keep Streamlit responsive.

### Step 9: Cleanup

- Add `live_data_*.pkl` to `.gitignore`
- Clear old alert data for clean testing

---

## Performance Budget

| Operation | Time | Frequency | CPU % |
|-----------|------|-----------|-------|
| `get_df_with_partial()` | <1ms | 500ms | <0.2% |
| `_run_pipeline()` (500 bars) | 15-25ms | 500ms | 3-5% |
| Group indicators (per group) | 1-3ms | 500ms | 0.6% |
| `TriggerStateTracker.evaluate()` | <0.1ms | 500ms | ~0% |
| Pickle write (atomic) | 1-3ms | 500ms | 0.4% |
| **Total** | **~20-35ms** | **500ms** | **4-7%** |

This is **less CPU** than the current approach which runs `TriggerLevelCache.check()` on every tick (hundreds-thousands/second during active trading).

---

## Key Design Decisions

1. **500ms throttle via `time.monotonic()`** — simple, no asyncio complexity. Called from `on_tick()`, returns immediately if interval hasn't elapsed.

2. **Pipeline runs once per timeframe, shared across strategies** — if 5 strategies use 1-min SPY, the pipeline runs once and all 5 evaluate against the same enriched DataFrame.

3. **Pickle for thread→UI data sharing** — atomic via `os.replace()`. Streaming thread writes, Streamlit fragment reads. Simple, reliable, ~3ms per write.

4. **Managed exits stay at bar close** — stop loss and bar-count exits check `bar['low'] <= stop` which requires completed bars.

5. **TriggerLevelCache kept but unused for alerting** — `INTRABAR_LEVEL_MAP` is still imported by `triggers.py` for backtest fill pricing. The class stays in the file, just isn't used by SymbolHub anymore.

6. **Live chart updates at 2s, not 500ms** — the alert fires at 500ms cadence (the important thing). Visual chart at 2s keeps Streamlit responsive.

---

## Verification Plan

1. Reset all alerts, restart streaming engine
2. Let it run for 10-15 minutes on a strategy with an `_ib` entry trigger (e.g., UT Bot)
3. Compare alert count vs forward test trade count — should be close to 1:1
4. Check timing deltas — should be 0-60 seconds (positive) for 1-min bars
5. Verify no phantom alerts (or very few — only if the partial bar trigger reverses before close)
6. Open Live Chart tab — confirm candles update, trigger markers appear near-real-time
7. Verify bar-close-only triggers (MACD, RVOL) still fire correctly
8. Verify managed exits (stop loss, bar count) still work
