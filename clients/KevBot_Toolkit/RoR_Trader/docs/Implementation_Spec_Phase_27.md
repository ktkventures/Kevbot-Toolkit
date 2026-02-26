# Implementation Spec — Phase 27: Unified Pipeline Alert System + Live Chart

## Problem Statement

The streaming engine has two parallel alert paths that diverge:

1. **Bar-close path**: runs `_run_pipeline()` → checks `trig_*` columns (transition-based, matches backtest/chart)
2. **Intra-bar path**: uses `TriggerLevelCache` comparing tick price vs cached level (NOT transition-based, diverges from chart)

This produces a **5:1 alert-to-trade ratio**, phantom alerts, wrong timing deltas, and alerts firing before entry candles (logically impossible). The root cause is that `TriggerLevelCache` performs price-vs-level crossing detection on every tick, while the backtest uses indicator **state transitions** (`(direction == 1) & (direction.shift(1) != 1)`). These are fundamentally different computations.

## Solution (Original — Steps 1-9, COMPLETED)

Replace both paths with **one throttled pipeline evaluator** that runs every 500ms on `history + partial bar`. When a `trig_*` column transitions `False→True`, fire the alert. This is identical to what the chart shows — trigger-agnostic, works for any current or future indicator pack.

Add a **Live Chart** tab so the user can visually confirm "trigger plots = alert fires."

## Solution (Phase 27B — `generate_trades()` as Single Source of Truth)

### Why the original solution is insufficient

The TriggerStateTracker approach from Steps 1-9, while a big improvement over the dual-path system, still diverges from the backtest in several ways:

1. **Partial bar phantom alerts** — TriggerStateTracker fires when `trig_*` transitions False→True on a partial bar (500ms eval). If the bar closes and the trigger reverts to False, the alert is a phantom — it will never appear as a trade in the backtest.

2. **Parallel position tracking** — The streaming engine maintains its own `_position_state` dict (stateful, per-tick), while `generate_trades()` uses a stateless batch approach. These can drift apart, especially after restarts or edge cases.

3. **Confluence bug** — The streaming engine only checks `strat.get('confluence')`, missing `general_confluences` (Time of Day, Day of Week, etc.). The backtest's `generate_trades()` checks both.

4. **Trigger name resolution** — The streaming engine uses `strat['entry_trigger']` directly, which for legacy strategies may be the raw name (e.g., `cross_bull`) instead of the base trigger ID with template prefix (e.g., `ema_cross_bull`). `generate_trades()` expects the base ID.

5. **Exit logic differences** — `generate_trades()` handles stop loss, target, bar-count exit, opposite signal, trailing stop, and time exit as an integrated priority system. The streaming engine implements these separately in `_check_managed_exits()`.

**The fundamental issue:** alerts fire from one code path (TriggerStateTracker + custom position/confluence/exit logic in `_evaluate_pipeline_throttled`) while the chart and backtest use a different code path (`generate_trades()`). Any bug fix or feature added to one path must be duplicated in the other, or they diverge.

### The fix: `generate_trades()` diffing

Replace TriggerStateTracker-based alert detection with a **trade list diff** approach:

1. On each throttled evaluation, run `generate_trades()` on the enriched live DataFrame
2. Compare the resulting trade list against the previous evaluation's trade list
3. Fire **entry alert** when a new trade appears that wasn't in the previous list
4. Fire **exit alert** when a trade that was previously open (no exit_time) now has an exit

This guarantees: **chart entries/exits = alert fires = one `generate_trades()` call**. No parallel logic, no drift, no phantom alerts from partial bar reversals (because `generate_trades()` produces the same trades the chart would show).

### Performance

Benchmarked: `generate_trades()` on 500 bars = **~28ms**. Running once per 500ms throttle cycle adds ~28ms to the existing ~20-35ms pipeline cost, totaling ~50-65ms per cycle — well within the 500ms budget (10-13% utilization).

---

## Files Modified

| File | Scope |
|------|-------|
| `realtime_engine.py` | Major — new evaluator, BarBuilder enhancement, remove TriggerLevelCache usage |
| `app.py` | Medium — add Live Chart tab to both backtest and forward test views |
| `alerts.py` | Minor — no changes to `_run_pipeline()`, small cleanup |
| `indicators.py`, `interpreters.py`, `triggers.py` | None |

---

## Implementation Steps (Original — COMPLETED)

### Step 1: BarBuilder.get_df_with_partial() ✓

**File:** `realtime_engine.py`

Add method to `BarBuilder` that returns `history` + current partial bar as a single DataFrame. Uses same `pd.concat` + `pd.DatetimeIndex` pattern as `_append_to_history()`. Returns `history.copy()` if no partial bar exists.

### Step 2: TriggerStateTracker class ✓

**File:** `realtime_engine.py` (new class, after `AlertCooldown`)

Tracks `trig_*` column boolean values per strategy across pipeline evaluations.

- `evaluate(strategy_id, df, trigger_cols)` → returns list of columns that transitioned `False→True`
- `seed(strategy_id, df, trigger_cols)` → initializes state from warmup data (prevents false fires on first eval)

This is the key design insight: comparing boolean column states between evaluations is **trigger-agnostic**. Works for UT Bot, EMA crosses, VWAP, MACD, any future indicator pack — no per-indicator logic needed.

### Step 3: _evaluate_pipeline_throttled() ✓

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

### Step 4: Modify on_tick() ✓

**File:** `realtime_engine.py`

Replace call to `_check_intrabar_triggers(price, timestamp)` with `_evaluate_pipeline_throttled(price, timestamp)`.

### Step 5: Simplify _on_bar_close() ✓

**File:** `realtime_engine.py`

Strip down to housekeeping only:

- Set `_first_bar_closed = True` on first close
- Run `_check_managed_exits()` for stop/bar-count exits (these need completed bar data)
- **Remove:** full pipeline run, `detect_signals()`, trigger cache updates, dedup clearing, signal processing

Managed exits (stop loss, bar count, target) still need bar-close evaluation because they check `bar['low'] <= stop_price` etc.

### Step 6: Update _init_position_state() ✓

**File:** `realtime_engine.py`

After running the warmup pipeline:

- Seed `TriggerStateTracker` with warmup data (prevents false fires on first eval)
- Remove `TriggerLevelCache` seeding
- Keep position state detection via `generate_trades()` (still needed for warmup)

### Step 7: Remove TriggerLevelCache usage from SymbolHub ✓

**File:** `realtime_engine.py`

- Remove `_trigger_cache` attribute from `SymbolHub.__init__`
- Remove `_check_intrabar_triggers()` method entirely
- Remove `_get_intrabar_trigger_bases()` and `_get_group_params_for_trigger()` helper methods
- Remove `_intrabar_fired`, `_ib_cooldown`, `_confluence_met` state dicts
- Remove from `UnifiedStreamingEngine`: `_trigger_cache` creation, passing to hubs, `.clear()` in stop
- **Keep** `TriggerLevelCache` class and `INTRABAR_LEVEL_MAP` dict in file — `triggers.py` imports `INTRABAR_LEVEL_MAP` for backtest fill pricing

### Step 8: Live Chart tab — app.py ✓

**8a: Add tab to both views**

In `render_live_backtest()` and `render_forward_test_view()`:

- Check if streaming engine is active for this strategy's symbol (read `monitor_status.json`)
- If active, insert "Live Chart" into tab names after "Price Chart"
- Adjust tab destructuring accordingly

**8b: render_live_chart_tab() function**

New function decorated with `@st.fragment(run_every=2)`:

- Reads `live_data_{symbol}.pkl` written by the streaming engine
- Uses existing `render_price_chart()` to render — reuses all TradingView LC machinery
- Shows only the strategy's relevant indicators (resolved via `_get_strategy_relevant_groups(strat)`)
- Entry/exit trade markers via `generate_trades()` on the live data
- Trade table below chart
- Shows last-bar timestamp, bar count, refresh status
- Visible candles: 100 (last ~1.5 hours for 1-min bars)

**8c: Auto-refresh**

`@st.fragment(run_every=2)` reruns only the fragment every 2 seconds without resetting the full page. The pipeline runs every 500ms (producing fresh data), the chart polls every 2 seconds (rendering the latest data). Alert fires at 500ms cadence; visual chart updates at 2s cadence to keep Streamlit responsive.

### Step 9: Cleanup ✓

- Add `live_data_*.pkl` to `.gitignore`
- Clear old alert data for clean testing

---

## Implementation Steps (Phase 27B — generate_trades() as Single Source of Truth)

### Step 10: Add TradeListTracker to streaming engine

**File:** `realtime_engine.py`

New class that replaces TriggerStateTracker for alert detection:

```
class TradeListTracker:
    """Detect new entries and exits by diffing generate_trades() output.

    On each evaluation, runs generate_trades() on the enriched DataFrame
    and compares against the previous result. Fires alerts when:
    - A new trade appears (entry alert)
    - A previously-open trade now has an exit (exit alert)

    This is the SAME function that produces chart entries/exits,
    guaranteeing chart = alerts = one source of truth.
    """

    def __init__(self):
        self._prev_trades = {}  # strategy_id → DataFrame of trades

    def evaluate(self, strategy_id, df_enriched, strat_config) → list[dict]:
        """Run generate_trades() and diff against previous result.

        Args:
            strategy_id: Strategy ID
            df_enriched: Enriched DataFrame from pipeline
            strat_config: Strategy config dict

        Returns:
            List of signal dicts (entry_signal or exit_signal) for new events
        """

    def seed(self, strategy_id, df_enriched, strat_config):
        """Initialize with warmup data to prevent false fires on first eval."""
```

Key implementation details:
- Resolves trigger names via `get_base_trigger_id()` on confluence IDs (handles legacy strategies)
- Passes both `confluence` AND `general_confluences` to `generate_trades()`
- Compares trades by entry_time (stable identifier across evaluations)
- New trade with no exit_time = open position (entry alert)
- Previously-open trade that now has exit_time = closed position (exit alert)

### Step 11: Replace TriggerStateTracker usage with TradeListTracker

**File:** `realtime_engine.py` — modify `_evaluate_pipeline_throttled()`

Replace the per-strategy evaluation block (Steps 3.4) with:

1. Call `TradeListTracker.evaluate(strat_id, df_enriched, strat)`
2. For each returned signal:
   - Build signal dict (same format as current: type, trigger, bar_time, price, stop_price, atr, etc.)
   - `enrich_signal_with_portfolio_context()`
   - `save_alert()`
   - `_alert_callback()` for webhook delivery
3. No separate position gate, confluence gate, or cooldown needed — `generate_trades()` handles ALL of this internally

**What gets removed from the per-strategy block:**
- Manual trigger column resolution (`trig_{entry_trigger}` building)
- `TriggerStateTracker.evaluate()` call
- Position gate (`if sig_type == 'entry_signal' and in_pos: continue`)
- Confluence gate (the `strat.get('confluence')` check that was missing `general_confluences`)
- Cooldown check
- Manual position state updates (`_position_state`, `_position_entry`)
- Stop price calculation (already done by `generate_trades()`)

**What stays:**
- Managed exits (`_check_managed_exits()`) — stop loss and bar-count exits that check intra-bar price levels still need their own check on bar close, because `generate_trades()` on partial bars can't reliably detect intra-bar stop hits. **However**, `generate_trades()` WILL detect these exits once the bar closes, so alerts will fire — they just fire at bar close rather than intra-bar.

### Step 12: Update _init_position_state() warmup

**File:** `realtime_engine.py`

Replace `TriggerStateTracker.seed()` calls with `TradeListTracker.seed()`:
- Run `generate_trades()` on the warmup DataFrame
- Store the result as the initial "previous trades" baseline
- If the last trade has no exit, set `_position_state[strat_id] = True`

### Step 13: Update live chart to use generate_trades() on live data

**File:** `app.py` — `render_live_chart_tab()`

Replace the current approach (passing parent's backtest trades) with:
- Run `generate_trades()` on the live pickle data (same as the streaming engine does)
- Use `get_base_trigger_id()` to resolve trigger names from confluence IDs
- Pass both `confluence` and `general_confluences`
- This produces the SAME trade list the streaming engine uses for alerting

The chart and alerts now derive from identical `generate_trades()` calls on the same data. What you see on the chart IS what fires as alerts.

### Step 14: Managed exits integration

**File:** `realtime_engine.py`

Managed exits (stop loss hit intra-bar, bar-count exit, target hit intra-bar) present a special case:
- `generate_trades()` can detect stop/target hits on completed bars (checks `bar['low'] <= stop`)
- But for intra-bar detection (the "we don't want to wait until candle close" use case), `_check_managed_exits()` still runs on each tick
- When `_check_managed_exits()` fires, it immediately fires the exit alert
- On the next `generate_trades()` call (at bar close or next 500ms eval), the trade will show as closed — TradeListTracker skips it since the alert was already sent

To prevent duplicate exit alerts:
- TradeListTracker maintains a `_fired_exits` set of `(strategy_id, entry_time)` tuples
- Managed exit handler adds to this set when it fires
- TradeListTracker checks this set before firing exit alerts

---

## Performance Budget (Updated)

| Operation | Time | Frequency | CPU % |
|-----------|------|-----------|-------|
| `get_df_with_partial()` | <1ms | 500ms | <0.2% |
| `_run_pipeline()` (500 bars) | 15-25ms | 500ms | 3-5% |
| Group indicators (per group) | 1-3ms | 500ms | 0.6% |
| `generate_trades()` per strategy (500 bars) | ~28ms | 500ms | 5.6% |
| Pickle write (atomic) | 1-3ms | 500ms | 0.4% |
| **Total (1 strategy)** | **~50-60ms** | **500ms** | **10-12%** |
| **Total (5 strategies, shared pipeline)** | **~160-180ms** | **500ms** | **32-36%** |

Note: If multiple strategies share a symbol/timeframe, the pipeline runs once but `generate_trades()` runs per-strategy. For 5 strategies on SPY 1-min: 1 pipeline run (~25ms) + 5 × generate_trades (~140ms) = ~165ms. Still well within the 500ms budget.

For high strategy counts (>10 on same symbol), we may need to optimize `generate_trades()` or increase the throttle interval. This is a future concern — current usage is 3-5 strategies per symbol.

---

## Key Design Decisions

1. **500ms throttle via `time.monotonic()`** — simple, no asyncio complexity. Called from `on_tick()`, returns immediately if interval hasn't elapsed.

2. **Pipeline runs once per timeframe, shared across strategies** — if 5 strategies use 1-min SPY, the pipeline runs once and all 5 evaluate against the same enriched DataFrame.

3. **Pickle for thread→UI data sharing** — atomic via `os.replace()`. Streaming thread writes, Streamlit fragment reads. Simple, reliable, ~3ms per write.

4. **Managed exits stay at bar close with dedup** — stop loss and bar-count exits need intra-bar price checking (`bar['low'] <= stop`). They fire immediately via `_check_managed_exits()`, then TradeListTracker's `_fired_exits` set prevents duplicate alerts when `generate_trades()` catches up at bar close.

5. **TriggerLevelCache kept but unused for alerting** — `INTRABAR_LEVEL_MAP` is still imported by `triggers.py` for backtest fill pricing. The class stays in the file, just isn't used by SymbolHub anymore.

6. **Live chart updates at 2s, not 500ms** — the alert fires at 500ms cadence (the important thing). Visual chart at 2s keeps Streamlit responsive.

7. **`generate_trades()` as the single source of truth** — the same function produces chart entries/exits AND alert detection. All entry/exit logic (confluence, position tracking, stop/target, bar-count exit) lives in ONE place. Bug fixes and feature additions to `generate_trades()` automatically apply to both chart and alerts.

8. **Trade diffing by entry_time** — entry_time is a stable identifier (determined by the bar where the trigger fires). Comparing entry_times between evaluations reliably detects new trades without hash complexity.

9. **No cooldown needed** — `generate_trades()` naturally prevents duplicate entries (can't enter while in a position) and duplicate exits (trade exits once). The diff approach inherently deduplicates.

---

## Verification Plan

1. Reset all alerts, restart streaming engine
2. Let it run for 20-30 minutes on a strategy with an `_ib` entry trigger (e.g., UT Bot)
3. Compare alert count vs live chart trade count — should be **exactly 1:1**
4. Open Live Chart tab — verify entries/exits on chart match alerts in Alert Analysis tab
5. Check timing deltas — should be 0-500ms for throttled detection
6. Verify **zero phantom alerts** — every alert should correspond to a trade on the chart
7. Verify stop loss exits fire promptly (via managed exits path) and don't double-fire
8. Verify bar-close-only triggers (MACD, RVOL) still fire correctly
9. Verify confluence gating works (including general confluences like Time of Day)
10. Test strategy hot-reload — add a new strategy while engine is running, confirm it picks up
