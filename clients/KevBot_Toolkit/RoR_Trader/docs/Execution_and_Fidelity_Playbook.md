# Execution & Fidelity Playbook

**Status:** ACTIVE — Source of truth for all trigger execution, confluence fidelity, and naming conventions.
**Date:** 2026-03-20 (Updated)
**Golden Rule:** If a feature can't produce consistent results between backtest and live, it needs a disclaimer or shouldn't ship.

---

## 1. Naming Conventions

### Square Brackets `[TAG]` — Execution Type

Appears before the trigger or condition name. Designates the PHILOSOPHY of how the entry works. Two characters max.

| Tag | Name | Philosophy |
|-----|------|-----------|
| `[C]` | Close | Enter when bar closes. One-stage. Most reliable. |
| `[L]` | Level | Enter when price crosses a level intra-bar. One-stage. Requires HiFi for backtest. |
| `[LC]` | Level-Close | Two-stage: enter on level cross, then confirm at bar close. Bail if not confirmed. |
| `[CC]` | Close-Close | Two-stage: enter on bar close, confirm on next bar close. Bail if not confirmed. |

The first letter = how you get in. The second letter (if present) = how you confirm/resolve.

### Asterisk `*` — HiFi Resolution

Appended outside the bracket. Indicates the backtest uses 1-second sub-bar data for accurate resolution on entry/exit bars.

- `[L]*` — Level cross with HiFi backtest resolution
- `[LC]*` — Level-Close with HiFi backtest resolution
- `[C]` — Never needs `*` (bar is closed, no intra-bar ambiguity for entry)

**Default behavior:** HiFi runs on all exit bars (stop/target resolution) and on all entry bars for `[L]`, `[LC]`, `[CC]` types. This is the engine default, not a toggle.

### Parentheses `(Variation)` — Parameter Variant

Designates which parameter set is used for a confluence pack. The underlying logic is identical — only the parameter values differ.

- `EMA Stack (Default)` — short=8, mid=21, long=50
- `EMA Stack (Wide)` — short=13, mid=34, long=100

### Confluence Fidelity Tags — `[PB]` and `[CB]`

For cross-TF confluence conditions:

| Tag | Name | What it Evaluates |
|-----|------|-------------------|
| `[PB]` | Previous Bar | The indicator/interpreter state of the last CLOSED bar on the secondary TF |
| `[CB]` | Current Bar | The indicator/interpreter state of the FORMING bar on the secondary TF (uses 1-second sub-data in backtest) |

Example: `5M-EMA_STACK-SML (Default) [PB]` vs `5M-EMA_STACK-SML (Default) [CB]`

These are treated as two distinct conditions in the Strategy Builder and mass search.

---

## 2. Execution Parameter Sections

Every trigger in a confluence pack has **four execution type sections** in its parameters. Each section is self-contained. The pack creator sets sensible defaults. The Strategy Builder shows the trigger with each available tag as a distinct option.

### Section Structure

```
Indicator Parameters:
  (indicator-specific: EMA periods, ATR multiplier, etc.)

Execution Parameters:

  [C] Close:
    reference_bar: 0 | -1        # Which bar's indicator values to evaluate
    order_type: market | limit    # How the order executes
    # Limit-only:
    limit_price_source: level | entry | custom_offset
    limit_duration_seconds: 10

  [L] Level:
    reference_bar: -1 | 0        # Which bar's indicator level to cross
    order_type: market | limit
    hold_seconds: 0              # Condition must hold N seconds before entry
    # Limit-only:
    limit_price_source: level
    limit_duration_seconds: 10

  [LC] Level-Close:
    entry_reference_bar: -1 | 0  # Which bar's level to cross for entry
    entry_order_type: market | limit
    confirm_reference_bar: 0     # Which bar's close to check for confirmation
    confirm_bar_offset: 0 | +1   # Same bar (0) or next bar (+1)
    bail_action: exit_market     # What happens if confirmation fails
    hold_seconds: 0

  [CC] Close-Close:
    entry_reference_bar: 0       # Which bar's close triggers entry
    entry_order_type: market
    confirm_reference_bar: 0
    confirm_bar_offset: +1       # Next bar must confirm
    bail_action: exit_market
```

### How This Maps to Current Execution Types

| Old Tag | New Tag | Section Parameters |
|---------|---------|-------------------|
| `[C]` | `[C]` | reference_bar=0, order_type=market |
| `[L1]` | `[L]` | reference_bar=-1, order_type=market |
| `[HM]` | `[LC]` | entry_reference_bar=-1, entry_order_type=market, confirm_bar_offset=0, bail_action=exit_market |
| `[HL]` | `[LC]` | entry_reference_bar=-1, entry_order_type=limit, confirm_bar_offset=0, bail_action=exit_market |

**Note:** L1 → L (renamed, same logic). HM/HL → LC variants (renamed, same logic). No underlying code changes to these paths.

---

## 3. HiFi Backtest — Two-Pass Architecture

### Default Behavior

- **All exit bars** get HiFi resolution (stop/target — which was hit first?)
- **All entry bars for `[L]`, `[LC]`, `[CC]`** get HiFi resolution (precise fill timing and price)
- **`[C]` entry bars** do NOT need HiFi (bar is closed, no ambiguity)

### Performance

- Typical 90-day backtest: 30-200 trades = 60-400 bars to resolve
- At ~50ms per Polygon REST call = 3-20 seconds overhead
- Acceptable for all use cases

### Pass 1 (Current Engine — Unchanged)

Run the standard bar-by-bar backtest. Produces trade records as today.

### Pass 2 (HiFi Resolution — New, Additive)

For each entry/exit bar that qualifies:
1. Fetch 60 one-second OHLCV bars from Polygon REST for that bar's time window
2. Walk through sequentially
3. For **stop/target exits**: first level hit wins (replaces hardcoded "stop wins")
4. For **L-type entries**: find exact second of level cross, record precise fill price
5. For **LC entries**: simulate entry on cross, then check bar close for confirmation
6. For **CC entries**: verify next bar's close confirms
7. Replace Pass 1's result for that bar with Pass 2's result if different
8. Add `hifi_resolved: true` flag to trade record

### What This Fixes

| Problem | Current Behavior | HiFi Behavior |
|---------|-----------------|---------------|
| Stop AND target hit same bar | Stop always wins | First hit wins (1-second walk) |
| L-type fill price unknown | Fills at indicator level | Fills at exact cross price/second |
| Same-bar stop suppression | Guard prevents stop check on entry bar | 1-second walk determines if stop was hit before or after entry |
| HM/HL not backtestable | Bar-level approximation only | Full 1-second simulation |
| Cross-TF `[CB]` confluence | Delayed by one full period | Recomputed from 1-second data at trigger moment |

---

## 4. Confluence Fidelity — [PB] vs [CB]

### Previous Bar `[PB]` (Standard)

The strategy checks the indicator/interpreter state of the LAST CLOSED bar on the secondary timeframe. This is the current behavior (forward-shift by one period).

**Backtest:** Reads from closed bar's computed values. Always accurate.
**Live:** Reads from last completed AM. bar. Always accurate.

### Current Bar `[CB]` (HiFi)

The strategy checks what the FORMING bar's indicator state shows at the moment of the trigger.

**Backtest:** On flagged bars, recompute secondary-TF indicators using 1-second data up to the trigger second.
**Live:** Shadow engines already update per-second via A. channel (Phase 31C). Condition checked at trigger moment.

### Cross-TF Confluence Display

Full format: `{TF}-{INTERPRETER}-{STATE} ({Variation}) [{PB|CB}]`

Examples:
- `5M-EMA_STACK-SML (Default) [PB]` — Previous 5Min bar, default EMA params
- `5M-EMA_STACK-SML (Wide) [CB]` — Current forming 5Min bar, wide EMA params
- `1D-MACD_LINE-BULL (Default) [PB]` — Daily MACD, always PB (daily bars don't need CB)

---

## 5. Stop/Target Resolution

### Current Behavior

When a bar's High >= Target AND Low <= Stop:
- Engine hardcodes **stop wins** (checks stop before target in `check_exit()`)

### HiFi Default (All Exit Bars)

1. Fetch 60 one-second bars for that minute
2. Walk through sequentially
3. First level hit wins (stop OR target)
4. Record actual exit price and exact second
5. Add `hifi_resolved: true` to trade record

---

## 6. Live Behavior Consistency

For triggers using `[L]`, `[LC]`, `[CC]`:
- **Live:** Per-second bars from Polygon A. channel provide real-time intra-bar data. Triggers evaluate against this data as it arrives.
- **Backtest HiFi:** 1-second historical bars from Polygon REST provide the same resolution.
- **Consistency:** Same data source (Polygon 1-second bars), same evaluation logic. Backtest matches live.

For `[CB]` confluence:
- **Live:** Shadow engines update per-second. Condition checked at trigger moment.
- **Backtest HiFi:** Indicators recomputed from 1-second data at trigger moment.
- **Consistency:** Same computation, same data.

---

## 7. Webhook Behavior by Execution Type

| Tag | Webhooks Sent | Timing |
|-----|--------------|--------|
| `[C]` | 1: entry (or exit) | After bar close evaluation |
| `[L]` | 1: entry (or exit) | Immediately on level cross detection |
| `[LC]` | 1-2: entry on cross, then exit if bail | Entry: on cross. Bail: at bar close if not confirmed. |
| `[CC]` | 1-2: entry on close, then exit if bail | Entry: on bar close. Bail: on next bar close if not confirmed. |

**Important:** Bail actions send EXIT webhooks (exit_long or exit_short), NOT close-all webhooks. Quantity-aware to avoid closing positions from other strategies in the same portfolio.

---

## 8. Files Affected

| File | What Changes | Risk |
|------|-------------|------|
| `unified_engine.py` | Pass 2 HiFi logic (additive, after Pass 1) | Medium — new code only, Pass 1 untouched |
| `data_loader.py` | 1-second bar fetching for specific windows | Low — already works (Phase 31A) |
| `ralph_engine.py` | Shadow engine CB live evaluation | Low — builds on Phase 31C |
| `confluence_groups.py` | Execution parameter sections on pack templates | Medium — new parameter structure |
| `app.py` | Strategy Builder execution type UI, trade history badges | Low — UI only |
| `triggers.py` | Tag rename (L1→L, HM→LC, HL→LC) | Low — display name only, logic unchanged |
