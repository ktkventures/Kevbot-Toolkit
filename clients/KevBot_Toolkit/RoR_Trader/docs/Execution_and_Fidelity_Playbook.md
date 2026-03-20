# Execution & Fidelity Playbook

**Status:** ACTIVE — Source of truth for all trigger execution, confluence fidelity, and naming conventions.
**Date:** 2026-03-20
**Golden Rule:** If a feature can't produce consistent results between backtest and live, it needs a disclaimer or shouldn't ship.

---

## 1. Naming Conventions

### Square Brackets `[TAG]` — Execution / Fidelity Type

Appears BEFORE or AFTER the trigger or condition name. Designates HOW the trigger fires or HOW the condition is evaluated.

Examples:
- `[C] EMA Price Cross Up` — fires on bar close
- `[L0] VWAP Cross` — fires intra-bar when price crosses the level on the current bar
- `[CB] EMA Stack` — evaluated on the current forming bar (HiFi)
- `[PB] MACD Line` — evaluated on the previous closed bar (standard)

### Parentheses `(Variation)` — Parameter Variant

Designates which parameter set is being used for a confluence pack. The underlying logic is identical — only the parameter values differ.

Examples:
- `EMA Stack (Default)` — short=8, mid=21, long=50
- `EMA Stack (Wide)` — short=13, mid=34, long=100
- `MACD Line (Fast)` — fast=8, slow=17, signal=9

### Star/Asterisk `*` — HiFi Resolution Required

Appended to any tag that requires sub-bar (1-second) data for accurate backtesting. If you see `*`, the backtest zoomed into 1-second bars on relevant trades.

Examples:
- `[L0*]` — L0 trigger with HiFi sub-bar resolution in backtest
- `[CB*]` — Current-bar confluence evaluated via 1-second sub-data in backtest

---

## 2. Trigger Execution Types

### Standard (Existing — DO NOT MODIFY)

| Tag | Name | When it Fires | Backtest Behavior | Live Behavior | Webhook Timing |
|-----|------|---------------|-------------------|---------------|----------------|
| `[C]` | Bar Close | After the bar closes, evaluates on closed bar's data | Evaluates at each bar's close in sequence | Ralph fires on completed AM. bar from Polygon WS | Webhook sent after bar close evaluation |
| `[L0]` | Level Cross (Current Bar) | When price crosses a computed level during the current bar | Checks if bar's H/L crossed the level; fills at the level price | Ralph checks per-second bars (A. channel) for level cross | Webhook sent immediately on cross detection |
| `[L1]` | Level Cross (Previous Bar Level) | When price crosses a level computed from the PREVIOUS bar | Same as L0 but level is from prior bar's indicator values | Same as L0 live behavior | Same as L0 |
| `[HM]` | Hybrid Market | Market entry on intra-bar condition, confirmed or reversed on bar close | Entry at condition price; if bar doesn't close in direction, exit at close | Same — entry on detection, close-bar confirmation | Two webhooks: entry on detection, then confirm/reverse on close |
| `[HL]` | Hybrid Limit | Limit entry at computed level, confirmed or reversed on bar close | Entry at limit level if H/L reaches it; close-bar confirmation | Same — limit fill on level touch, close-bar confirmation | Two webhooks: fill notification, then confirm/reverse |

### HiFi (New — Phase 31F)

| Tag | Name | When it Fires | Backtest Behavior | Live Behavior | Webhook Timing |
|-----|------|---------------|-------------------|---------------|----------------|
| `[L0*]` | Level Cross (HiFi) | Same as L0 but with precise sub-bar fill timing | Pass 1: flag bar as ambiguous. Pass 2: walk 1-second bars to find exact cross second and price | Identical to L0 live (per-second bars already used) | Same as L0 |
| `[HM*]` | Hybrid Market (HiFi) | Same as HM but backtestable with 1-second resolution | Pass 1: flag entry+confirmation bars. Pass 2: 1-second walk determines exact entry time and whether confirmation held | Identical to HM live | Same as HM |
| `[HL*]` | Hybrid Limit (HiFi) | Same as HL but with limit-fill verification via 1-second data | Pass 1: flag. Pass 2: verify exact second of limit fill, check if price came back to limit level | Identical to HL live | Same as HL |
| `[HC]` | Hold Confirm | Entry only after condition holds for N seconds | Pass 2: after condition triggers, verify it holds for N consecutive 1-second bars | Per-second bar evaluation: condition must be true for N consecutive A. bars | Webhook sent after hold period confirmed |

### Redundancy Notes

- `[C]` — Never needs HiFi. Fires on closed bar. Keep as-is.
- `[L1]` — Level from previous bar. The level itself doesn't need HiFi. But the FILL timing could benefit from `[L1*]` if we want exact fill-second data. Low priority.
- `[L0]` vs `[L0*]` — In LIVE, these are identical (both use per-second bars). In BACKTEST, `[L0*]` adds the second pass to resolve fill timing. After validation, `[L0*]` could replace `[L0]` as the default for L0 backtests, making the non-HiFi `[L0]` redundant.
- `[HM]` / `[HL]` — Currently not fully backtestable (bar-level only). `[HM*]` / `[HL*]` make them properly backtestable. After validation, the non-HiFi variants should be replaced.

---

## 3. Confluence Fidelity Types

### Condition Evaluation Timing

| Tag | Name | What it Evaluates | Backtest Behavior | Live Behavior |
|-----|------|-------------------|-------------------|---------------|
| `[PB]` | Previous Bar | The indicator/interpreter state of the LAST CLOSED bar | Standard: reads from the closed bar's computed values | Standard: reads from the last completed AM. bar |
| `[CB*]` | Current Bar (HiFi) | The indicator/interpreter state of the FORMING bar using 1-second sub-data | Pass 2: on flagged bars, recompute indicators using 1-second bars up to the trigger moment | Per-second bars update indicators incrementally; condition checked at trigger moment |

### How This Works in Practice

**Example: 5Min EMA Stack as confluence for a 1Min strategy**

With `[PB]`: The strategy checks if the PREVIOUS completed 5Min bar had EMA Stack = BULL. This is the current behavior (forward-shift by one period).

With `[CB*]`: The strategy checks what the CURRENT forming 5Min bar's EMA Stack shows RIGHT NOW. If the short EMA just crossed above the mid EMA 30 seconds ago on the 5Min timeframe, that "hot cross" is visible immediately — not delayed by up to 5 minutes.

**Backtest**: On bars where an entry trigger fires, Pass 2 fetches 1-second data for the current higher-TF period and recomputes the indicator state at the exact second of the trigger. This tells us: "at the moment this trigger fired, was the 5Min EMA Stack already bullish?"

**Live**: Per-second bars from the A. channel update the shadow indicator engine for the higher TF. The condition is checked in real-time. Consistent with backtest.

### Cross-TF Confluence Tags

These combine TF prefix with fidelity tag:

- `5M-EMA_STACK-SML [PB]` — Previous 5Min bar EMA Stack in SML state
- `5M-EMA_STACK-SML [CB*]` — Current forming 5Min bar EMA Stack in SML state
- `1D-MACD_LINE-BULL [PB]` — Previous daily MACD Line bullish (standard, always PB for daily)

---

## 4. Stop/Target Resolution

### Current Behavior

When a bar's High >= Target AND Low <= Stop:
- Engine hardcodes **stop wins** (checks stop before target in `check_exit()`)
- This may misrepresent win rate on ambiguous bars

### HiFi Resolution

On ambiguous bars (both stop and target reachable):
1. Fetch 60 one-second bars for that minute
2. Walk through sequentially
3. First level hit wins (stop OR target)
4. Record the actual exit price and second

This is a Pass 2 operation — only runs on flagged bars.

### Swing Stop on Current Bar

Currently: swing stop uses the lowest low of the last N bars (all closed).

With HiFi: could use the current bar's developing low as a potential stop level. The 1-second data shows where the actual low formed within the bar.

---

## 5. Two-Pass Backtest Architecture

### Pass 1 (Fast — Current Engine, Unchanged)

Run the standard bar-by-bar backtest. Identify "ambiguous bars" to flag for Pass 2:

**Flag conditions:**
- Stop AND target both reachable on same bar (H >= target, L <= stop)
- L-type entry fired (`[L0]`, `[L1]`, `[HM]`, `[HL]`)
- Any `[CB*]` confluence condition is active on the strategy
- `[HC]` (hold confirm) trigger used
- HM/HL entry pending confirmation on this bar

### Pass 2 (Targeted — New, HiFi)

For ONLY the flagged bars (typically 60-100 across a full backtest):

1. Fetch 1-second OHLCV from Polygon REST for that bar's time window
2. Walk through 60 one-second bars sequentially
3. For each second:
   - Update indicators incrementally (same IncrementalIndicatorEngine)
   - Check trigger conditions
   - Check stop/target levels
   - Record first event that fires
4. Replace Pass 1's result for that bar with Pass 2's result

### Performance Estimate

- Pass 1: same speed as current (unchanged)
- Pass 2: ~60-100 bars × 1 API call each = 60-100 REST calls
  - Polygon Advanced: unlimited REST, ~50ms per call = 3-5 seconds total
  - Can batch multiple bars into single API calls (date range covering several bars)

---

## 6. Implementation Phases

### Phase 31F-1: Stop/Target HiFi Resolution
- Flag bars where both stop and target are reachable
- Fetch 1-second data, walk through, first level hit wins
- Compare Pass 1 vs Pass 2 results: how many trades change outcome?
- Add `hifi_resolved: true` flag to trade records

### Phase 31F-2: L-type HiFi Fill Timing (`[L0*]`)
- Flag L0/L1 entry bars
- 1-second walk to find exact cross second and fill price
- Record precise fill time (affects slippage calculation)

### Phase 31F-3: Current Bar Confluence (`[CB*]`)
- Add `[CB*]` as a fidelity option for cross-TF confluence conditions
- In Pass 2: recompute higher-TF indicators using 1-second data up to trigger moment
- In live: shadow engines already update per-second (Phase 31C)
- Strategy Builder: show `[PB]` and `[CB*]` as distinct condition options

### Phase 31F-4: HM*/HL* Backtestability
- Full 1-second simulation of hybrid order flow
- Entry detection → hold → close-bar confirmation/reversal
- Makes HM/HL strategies properly backtestable

### Phase 31F-5: Hold Confirm (`[HC]`) Execution Type
- New trigger type: condition must hold for N seconds
- Backtest: 1-second walk verifies sustained condition
- Live: per-second bar counter tracks consecutive true evaluations

### Phase 31F-6: QA Harness
- Dedicated verification log/UI
- Logs every HiFi-resolved trade with: strategy, bar time, Pass 1 result, Pass 2 result, 1-second data summary
- UI page or modal: "HiFi Verification Queue" — list of trades needing visual confirmation
- Each entry has a "View 1-Second Chart" button showing the zoomed-in execution

---

## 7. Verification Protocol

For each new execution/fidelity type:

1. **Backtest verification**: Run strategy with known trades. Compare Pass 1 (standard) vs Pass 2 (HiFi) results. Document which trades changed and why.
2. **Live verification**: Run strategy live. When a HiFi-relevant trade fires, the QA harness logs it. User opens the 1-second chart modal to visually confirm execution matches expectation.
3. **Consistency check**: For completed trades, compare backtest Pass 2 result vs actual live execution. They should match (same 1-second data, same logic).

**If verification fails**: The HiFi type gets a disclaimer badge in the UI. It's not removed — just flagged as "under review."

---

## 8. Files Affected

| File | What Changes |
|------|-------------|
| `unified_engine.py` | Pass 2 logic, ambiguous bar flagging, 1-second indicator recomputation |
| `data_loader.py` | 1-second bar fetching for specific time windows (already works via Polygon REST) |
| `ralph_engine.py` | Shadow engine updates for `[CB*]` live evaluation (partially done in 31C) |
| `app.py` | Strategy Builder UI for fidelity tags, QA harness page/modal, trade history HiFi badges |
| `triggers.py` | `[HC]` hold-confirm trigger type |
| `confluence_groups.py` | `[PB]`/`[CB*]` fidelity option per condition |
