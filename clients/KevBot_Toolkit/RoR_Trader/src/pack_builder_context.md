# RoR Trader — Confluence Pack Builder Reference

You are generating a custom confluence pack for the RoR Trader platform. A confluence pack adds a new technical indicator with state classification and trade triggers to the system.

## What You Need to Generate

You must produce **exactly three outputs** in your response:

1. **manifest.json** — Pack metadata, parameters, outputs, and trigger definitions
2. **indicator.py** — Python function that calculates indicator values on a DataFrame
3. **interpreter.py** — Python functions that classify states and detect trigger events

Each output must be in a fenced code block with the filename as a comment on the first line.

---

## Output Format

Your response MUST contain exactly three fenced code blocks in this order:

~~~
```json
// manifest.json
{ ... }
```

```python
# indicator.py
...
```

```python
# interpreter.py
...
```
~~~

---

## 1. manifest.json — Pack Spec Schema

```json
{
  "$schema": "pack_spec_v1",
  "slug": "your_pack_slug",
  "name": "Human Readable Name",
  "category": "Momentum | Trend | Volume | Volatility | Mean Reversion",
  "description": "One-sentence description of what this indicator does",
  "author": "user",
  "version": "1.0.0",
  "created_at": "2026-01-01T00:00:00Z",
  "pack_type": "tf_confluence",
  "display_type": "overlay | oscillator | hidden",

  "interpreters": ["YOUR_PACK_KEY"],
  "trigger_prefix": "your_prefix",

  "parameters_schema": {
    "param_name": {
      "type": "int | float | str | bool",
      "default": 14,
      "min": 1,
      "max": 100,
      "label": "Human Label"
    }
  },

  "plot_schema": {
    "line_color": {
      "type": "color",
      "default": "#8b5cf6",
      "label": "Line Color"
    }
  },

  "outputs": ["STATE_A", "STATE_B", "STATE_C"],
  "output_descriptions": {
    "STATE_A": "Description of when this state occurs",
    "STATE_B": "Description of when this state occurs",
    "STATE_C": "Description of when this state occurs"
  },

  "triggers": [
    {
      "base": "trigger_name",
      "name": "Human Trigger Name",
      "direction": "BOTH",
      "type": "BOTH"
    }
  ],

  "trigger_levels": {
    "trigger_name": {"level_column": "col_name_1", "cross": "above"}
  },

  "indicator_columns": ["col_name_1", "col_name_2"],
  "column_color_map": {
    "col_name_1": "line_color"
  },

  "indicator_function": "calculate_your_pack",
  "interpreter_function": "interpret_your_pack",
  "trigger_function": "detect_your_pack_triggers",

  "requires_indicators": ["col_name_1"],

  "incremental_class": {
    "module": "indicator_incremental",
    "class_name": "YourPackIncremental"
  }
}
```

**Trigger note:** Triggers carry no `execution` field. The system materializes runtime variants (C, L, LC, CC) automatically — see "Triggers and Execution Types" below. `trigger_levels` is optional; include it only for triggers that represent a meaningful intra-bar level cross.

### Manifest Rules

- **slug**: lowercase, underscores only, starts with letter (e.g., `keltner_channels`)
- **interpreters**: list with ONE key in UPPERCASE (e.g., `["KELTNER_CHANNELS"]`)
- **trigger_prefix**: short lowercase prefix for trigger IDs (e.g., `kc`)
- **display_type**: how this indicator is rendered on charts:
  - `"overlay"` — lines drawn on the price chart (EMAs, Bollinger Bands, Keltner Channels)
  - `"oscillator"` — separate pane below the price chart (RSI, Stochastic, CCI)
  - `"hidden"` — no chart visualization (utility packs like bar count)
- **outputs**: mutually exclusive states — every bar is classified into exactly one
- **triggers**: events that fire on state transitions (crosses, flips, etc.)
- **indicator_columns**: column names your indicator.py adds to the DataFrame
- **column_color_map**: maps indicator column names to plot_schema color keys for charting. Only include columns that should be plotted (e.g., omit `bb_bandwidth` from an overlay pack since it's not a price-scale value). Example: `{"bb_upper": "upper_color", "bb_basis": "basis_color"}`
- **indicator_function / interpreter_function / trigger_function**: exact function names in your Python files

### plot_config (Optional)

Advanced charting configuration. Include this object in the manifest for indicators that need more than simple line plots:

- **band_fills**: Array of fill definitions between two indicator columns
  ```json
  "band_fills": [{"upper_column": "bb_upper", "lower_column": "bb_lower", "fill_color_key": "band_fill_color"}]
  ```
- **reference_lines**: Array of horizontal reference lines for oscillators
  ```json
  "reference_lines": [{"value": 70, "color": "#ef4444", "label": "Overbought"}, {"value": 30, "color": "#22c55e", "label": "Oversold"}]
  ```
- **line_styles**: Dict mapping indicator column to line style integer (0=Solid, 1=Dotted, 2=Dashed, 3=LargeDashed, 4=SparseDotted)
  ```json
  "line_styles": {"signal_line": 2}
  ```
- **candle_color_column**: Column name containing hex color strings for bar coloring. Use this for pattern-based indicators (e.g., Swing 1-2-3, Strat bars) where the primary output is candle coloring rather than overlay lines. Set `display_type` to `"hidden"` when using this. The column should contain hex color strings (e.g., `"#22c55e"`) or empty strings for default candle color.
  ```json
  "candle_color_column": "my_candle_color"
  ```

### Reserved Names (DO NOT USE)

These trigger prefixes are taken: `ema`, `ema_pp`, `ema_pp_v2`, `macd`, `macd_hist`, `vwap`, `rvol`, `utbot`, `utbot_v2`, `bar_count`, `bb`, `src`, `st`, `sw123`, `strat`, `rsi`

These interpreter keys are taken: `EMA_STACK`, `EMA_PRICE_POSITION`, `EMA_PRICE_POSITION_V2`, `MACD_LINE`, `MACD_HISTOGRAM`, `VWAP`, `RVOL`, `UTBOT`, `UTBOT_V2`, `BOLLINGER_BANDS`, `SR_CHANNELS`, `SUPERTREND`, `SWING_123`, `STRAT_ASSISTANT`, `RSI_ZONES`

These indicator columns are taken: `ema_8`, `ema_21`, `ema_50`, `macd_line`, `macd_signal`, `macd_hist`, `vwap`, `vwap_sd1_upper`, `vwap_sd1_lower`, `vwap_sd2_upper`, `vwap_sd2_lower`, `atr`, `vol_sma`, `rvol`, `utbot_stop`, `utbot_stop_prev`, `utbot_direction`, `bb_upper`, `bb_basis`, `bb_lower`, `bb_bandwidth`, `src_nearest_top`, `src_nearest_bot`, `src_num_channels`, `src_in_channel`, `st_line`, `st_direction`, `st_atr`, `sw123_pattern`, `sw123_candle_color`, `strat_bar_type`, `strat_combo`, `strat_actionable`, `strat_candle_color`, `rsi_value`

---

## 2. indicator.py — Indicator Calculation Function

```python
import pandas as pd
import numpy as np

def calculate_your_pack(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate indicator values and add columns to the DataFrame.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        **params: Parameters from the confluence group's parameters dict

    Returns:
        Copy of DataFrame with new indicator columns added.
    """
    period = params.get("period", 14)
    result = df.copy()

    # Your indicator calculation here
    result["your_column"] = ...

    return result
```

### Indicator Rules

- Only import `pandas`, `numpy`, and `math` — no other imports allowed
- Function receives `**params` with all parameters from `parameters_schema`
- Must return a **copy** of the DataFrame with new columns added (`df.copy()`)
- Column names must match `indicator_columns` in manifest
- Available DataFrame columns: `open`, `high`, `low`, `close`, `volume`
- Use vectorized pandas/numpy operations where possible (not row-by-row loops)

---

## 3. interpreter.py — State Classification + Trigger Detection

```python
import pandas as pd
import numpy as np

def interpret_your_pack(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar into a mutually exclusive state.

    Args:
        df: DataFrame with indicator columns already present
        **params: Parameters from the confluence group

    Returns:
        Series of state label strings (from manifest "outputs")
        Return None for bars with insufficient data
    """
    threshold = params.get("threshold", 70.0)

    def classify(row):
        value = row.get("your_column", np.nan)
        if pd.isna(value):
            return None
        if value > threshold:
            return "STATE_A"
        else:
            return "STATE_B"

    return df.apply(classify, axis=1)


def detect_your_pack_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect trigger events (state transitions, crosses, etc.).

    Args:
        df: DataFrame with indicator columns already present
        **params: Parameters from the confluence group

    Returns:
        Dict mapping trigger_id -> boolean Series
        Each key must be: "{trigger_prefix}_{trigger_base}"
        Each value: True on bars where the trigger fires
    """
    threshold = params.get("threshold", 70.0)
    prefix = "your_prefix"  # Must match trigger_prefix in manifest

    triggers = {}

    col = df["your_column"]
    col_prev = col.shift(1)

    # Cross above threshold
    triggers[f"{prefix}_cross_above"] = (col > threshold) & (col_prev <= threshold)

    # Cross below threshold
    triggers[f"{prefix}_cross_below"] = (col < threshold) & (col_prev >= threshold)

    return triggers
```

### Interpreter Rules

- Only import `pandas`, `numpy`, and `math`
- `interpret_*` returns a Series with values from manifest `outputs` list
- Return `None` for bars with NaN/insufficient data
- **Mutual exclusivity is critical**: Every bar must map to exactly ONE output state. The outputs list should be a set of non-overlapping zones, not a mix of zones + conditions. For example:
  - GOOD: `["OVERBOUGHT", "BULLISH", "NEUTRAL", "BEARISH", "OVERSOLD"]` — every bar falls into exactly one zone
  - BAD: `["UPPER_ZONE", "MID_ZONE", "LOWER_ZONE", "SQUEEZE"]` — "SQUEEZE" can overlap with any zone
  - If you need a condition like "squeeze" or "trending", encode it as a modifier within the zone states (e.g., `"SQUEEZE_UPPER"`, `"SQUEEZE_MID"`, `"SQUEEZE_LOWER"`) or make it a trigger event instead of an output state
- Avoid output states that are "always true" for typical market data. Test your classification logic mentally: on a typical 1-minute chart, would each state fire a reasonable portion of the time? If one state would dominate 90%+ of bars, the thresholds need adjustment or the state definitions need rethinking
- `detect_*_triggers` returns a dict of boolean Series
- Trigger keys MUST be `{trigger_prefix}_{base}` matching manifest triggers
- Use `.shift(1)` to compare current bar with previous bar for crosses
- Trigger detection should be vectorized (boolean operations on Series)
- **Triggers are direction-agnostic and type-agnostic.** Always use `"direction": "BOTH"` and `"type": "BOTH"`. Users decide how to use a trigger (entry vs exit, long vs short) in Strategy Builder. Do NOT create separate LONG/SHORT or ENTRY/EXIT versions of the same trigger — this creates redundancy. One trigger, one boolean, usable in any context.
- **Avoid redundant triggers.** If two triggers would fire on the exact same bars with the same boolean logic, they should be one trigger. For example, "RSI crosses above midline" is one trigger — don't create separate "entry long on midline cross up" and "exit short on midline cross up" triggers.
- **Always provide level columns for intra-bar (L-type) execution when possible.** For every bar-close trigger, consider whether a faster intra-bar entry is possible by crossing a specific price level. Common level sources: previous bar's close, moving average values, band edges, threshold levels. Add a level column and define an `_ib` trigger variant in the manifest with `level_column` and `cross` fields. If the pack is purely pattern-based with no meaningful price level to cross, L-type triggers can be omitted — the system will only auto-generate C and CC variants.
- **CRITICAL: Do NOT use `.shift()` on level columns. The engine adds the 1-bar lag automatically.** The engine caches level column values at bar N close and uses them on bar N+1. This means you should provide the CURRENT bar's value (no shift), and the engine handles the time alignment. Use the `_prev` suffix on the column name to signal L1 classification (previous bar's level reference), but DON'T actually shift the values:
  ```python
  # CORRECT — engine adds lag automatically
  result["my_entry_level_prev"] = result["close"]  # current close, _prev for L1 naming

  # WRONG — double shift, causes 2-bar lag
  result["my_entry_level_prev"] = result["close"].shift(1)
  ```
  The `_prev` suffix is a NAMING convention only — it tells the engine to classify the trigger as L1. The actual time shift comes from the engine's cache-then-use-next-bar pattern.
- **CRITICAL: Level columns must have a value on EVERY bar, not just bars where the trigger fires.** Do NOT filter the level column with `.where(trigger_boolean, ...)` or similar conditional logic that sets it to NaN on most bars. The engine needs a valid level on EVERY bar so it can detect intra-bar crosses on any subsequent bar:
  ```python
  # CORRECT — level exists on every bar
  result["my_entry_level_prev"] = result["close"]

  # WRONG — level only exists on trigger bars; engine can't detect crosses
  result["my_entry_level_prev"] = result["close"].where(result["my_trigger"], other=np.nan)
  ```
  The L-type entry is a price-level event (price crosses a value). The pattern detection (the C-type trigger boolean) is a separate concern that fires at bar close. The L-type fires earlier, on any bar where price actually crosses the level — even if the full pattern isn't yet confirmed. If you want pattern-strict entries, use C-type or CC-type instead of L-type.

### Engine internals: how user pack L-type triggers flow

For user packs with L-type triggers, the system threads pack-specific data through several layers:

1. **Pack indicator** writes column values to the DataFrame (e.g., `result["my_level_prev"] = result["close"]`)
2. **Pack registry** registers the trigger in `INTRABAR_LEVEL_MAP` with the level column name and cross direction
3. **Backtest loop** detects which user pack columns are referenced by `INTRABAR_LEVEL_MAP` for the strategy's required triggers, and reads those values from each row
4. **process_bar** merges those values into the `current` dict (which is built from built-in incremental indicators)
5. **`_update_cached_levels`** caches the user pack column value
6. **`_compute_ib_gates`** opens the gate based on the close vs the level
7. **`_get_ib_checks`** returns the trigger with cached level for crossing checks
8. **`evaluate_bar_for_backtest`** checks reachability with the strict cross requirement: `low < level <= high`

The key insight: user pack indicator values are NOT computed by the incremental engine — they come from the batch DataFrame pipeline. The engine merges them into `current` so the L-type machinery can use them like built-in indicators. This is necessary because user packs use a different computation path than built-ins.

### False cross prevention for user pack L-type triggers

User pack L-type triggers use a stricter reachability check than built-ins:
- **Built-in** (VWAP, EMA, etc.): `high >= level` (or `low <= level`)
- **User pack**: `low < level <= high` (must touch BOTH sides of the level)

This prevents phantom fills on gap opens — if a bar opens above the level and never goes below, it didn't actually cross intra-bar. The strict check filters those out, ensuring the L-type fill reflects an actual price cross during the bar.
- **Level columns are internal engine columns, not visual indicators.** Columns used as `level_column` in trigger definitions (e.g., `sw123t_bull_entry_level`) are consumed by the execution engine for fill price calculation. They should be included in `indicator_columns` so they're computed, but the system automatically excludes them from chart overlay rendering. Do NOT include them in `column_color_map` or `plot_schema`.
- **Candle color columns are not overlay lines.** If the pack uses `candle_color_column` in `plot_config`, that column contains hex color strings for bar coloring. The system automatically uses it for candle coloring and excludes it from line overlays.

### Optional: Live-Mode Incremental Class

A pack's `indicator.py` is the **batch** implementation — it operates on a full DataFrame at once (vectorized pandas). That powers backtests. But the live worker receives one bar at a time from the WebSocket, so it can't call the batch function on every tick — it would O(N²) the entire history.

To run in live mode, a pack must additionally provide an **incremental class** that tracks rolling state and updates O(1) per bar. Built-in indicators (EMA, MACD, VWAP, RVOL, UTBOT) have this; user packs can opt in by declaring `incremental_class` in the manifest and shipping `indicator_incremental.py` alongside `indicator.py`.

A pack **without** `incremental_class` is explicitly batch-only. Backtests work; the live worker logs that the pack was skipped; the Parity Simulator returns `FAIL_SILENT`. This is a valid state — many indicators don't have a clean incremental form yet, and the parity simulator surfaces that gap rather than hiding it.

**Contract:**

```python
# indicator_incremental.py
class YourPackIncremental:
    def __init__(self, **params):
        """Receive the same params dict that calculate_your_pack(**params) would.

        Initialize whatever rolling state the indicator needs. Mirror the
        same defaults as the batch function — they MUST match, or backtest
        and live diverge.
        """
        self.period = params.get("period", 14)
        self._buffer = []           # rolling window state
        self._last_value = None     # most recent indicator value

    def warmup(self, df) -> None:
        """Optional: bulk-seed state from a prior DataFrame.

        Called once at engine startup with the warmup window. Defaults to
        replaying each row through update_bar() if not overridden.
        """
        for _, row in df.iterrows():
            self.update_bar(dict(row))

    def update_bar(self, bar: dict) -> dict:
        """Process one bar; return the new indicator column values.

        Args:
            bar: dict with keys timestamp, open, high, low, close, volume

        Returns:
            dict mapping column_name -> value, where keys MUST match
            indicator_columns in the manifest. The engine merges this
            into its `current` indicator state for the trigger evaluator.
        """
        # ... compute next value from previous state + new bar ...
        return {"your_column": new_value}
```

**Key rules:**
- The class's `__init__(**params)` must accept the exact same params dict that `calculate_*()` does.
- `update_bar(bar)` must return a dict whose keys are a subset of `indicator_columns` from the manifest. Internal scratch state is kept on `self` and never returned.
- Numeric outputs must match the batch function's outputs to within float tolerance for the same inputs. The Parity Simulator is the verification — `PASS` means the two paths agree on every fire bar; `PARTIAL` or `FAIL_SILENT` flags drift.
- Don't read from the network, filesystem, or any module outside the allowed import set (`pandas`, `numpy`, `math`, `typing`).
- Level columns for L-type triggers (`*_prev` columns) must be populated on every `update_bar()` call, same rule as the batch function.

**When to omit:**
- Pack uses look-ahead patterns that genuinely can't be computed online (e.g., needs to know "this is a swing high" which requires future bars). Document this in the description; users see "batch-only" in the UI.
- Pack is pure exploration / not yet ready for live deployment.

### Triggers and Execution Types — The Modular Pattern

This is the single most important section of this document. Read it carefully and follow it exactly.

#### Core principle

A **trigger** answers "WHEN does the signal fire?" — it's a single boolean event in time (e.g. "price crossed above the trailing stop"). An **execution type** answers "HOW does the order fill once the signal fires?" — it's an orthogonal mechanism (fill at bar close, fill at the level cross, fill with confirmation, etc.).

These two concerns are independent. **The pack manifest describes only the trigger; the engine layers the execution types on top automatically.**

You will emit **one trigger entry per logical signal**. Do NOT emit multiple variants of the same logical signal differing only in execution type. The system materializes the execution variants at install time.

#### What you emit in the manifest

Each trigger entry is minimal:
```json
"triggers": [
  {"base": "bull_flip", "name": "Bullish Trailing Stop Flip",
   "direction": "BOTH", "type": "BOTH"},
  {"base": "bear_flip", "name": "Bearish Trailing Stop Flip",
   "direction": "BOTH", "type": "BOTH"}
]
```

Required fields per trigger:
- `base`: short snake_case logical name. **Must NOT include exec-type suffixes** (`_ib`, `_lc`, `_cc`, `_hm`, `_hl`). The system applies suffixes automatically. Example: `bull_flip`, NOT `bull_flip_ib`.
- `name`: human-readable label.
- `direction`: always `"BOTH"`.
- `type`: always `"BOTH"`.

There is **no** `execution` field. There is **no** `column_base` field. There is **no** `level_column` or `cross` field on individual trigger entries. **DO NOT EMIT THESE FIELDS.**

#### How you declare a level cross (for L-type and LC executions)

If a trigger represents a level cross — i.e. the engine could meaningfully detect intra-bar that price crossed a specific indicator value — declare that level **once** in a top-level `trigger_levels` block:

```json
"trigger_levels": {
  "bull_flip": {"level_column": "my_indicator_prev", "cross": "above"},
  "bear_flip": {"level_column": "my_indicator_prev", "cross": "below"}
}
```

The keys are trigger `base` names that exist in your `triggers` array. Each entry has:
- `level_column`: the indicator column the engine should treat as the cross level. Must exist in `indicator_columns`.
- `cross`: `"above"` or `"below"` — which direction crossing the level confirms the signal.

Triggers without an entry in `trigger_levels` are pure bar-close events (no L-type variant; the engine still emits C and CC variants for them).

Triggers with an entry get the L (intra-bar) and LC (level-cross + bar-close confirmation) variants automatically.

#### What the engine produces from this at runtime

For each base trigger, the engine materializes runtime IDs by suffix. Given `trigger_prefix: "utbm"` and the example above, the engine registers:

| Trigger emitted in manifest | Runtime IDs produced |
|---|---|
| `bull_flip` (with trigger_levels entry) | `utbm_bull_flip` (C), `utbm_bull_flip_ib` (L), `utbm_bull_flip_lc` (LC), `utbm_bull_flip_cc` (CC) |
| `bull_flip` (no trigger_levels entry) | `utbm_bull_flip` (C), `utbm_bull_flip_cc` (CC) |

Strategy authors pick one of these runtime IDs in Strategy Builder. Users toggle which exec variants are enabled per trigger. **You do not enumerate these in the manifest. They are generated.**

#### The four canonical execution types

For reference only — you do not declare these in the manifest:

| Type | Code | Meaning |
|---|---|---|
| Bar Close | C | Order fills at the close of the bar where the signal fires |
| Level Cross | L | Order fills intra-bar when price crosses `level_column` (only when `trigger_levels` entry exists) |
| Level + Close Confirm | LC | Same intra-bar fill as L, but exits at market if the bar's close doesn't confirm direction |
| Close + Close Confirm | CC | Order fills at bar close, exits at market if the next bar's close doesn't confirm direction |

Legacy codes you may see in older built-in packs (`HM`, `HL`) are sub-modes of the LC type. You do not emit them.

#### L0 vs L1: the `_prev` suffix on level columns

When you declare a `level_column` in `trigger_levels`, choose between two patterns based on indicator behavior:

**L1 (previous bar's level — most common for dynamic indicators):** name the column with a `_prev` suffix. The engine uses the previous bar's value as the cross threshold. Use this for ATR trailing stops, adaptive moving averages, dynamic support/resistance — any line that recalculates or jumps when the signal fires.

```python
# In indicator.py:
result["my_indicator_prev"] = result["my_indicator"]   # CURRENT bar's value
# DO NOT .shift() here. The engine adds the 1-bar lag automatically.
```

The `_prev` suffix is a **naming convention** that signals to the engine "treat this as L1." The actual time-shift happens inside the engine's cache. You provide the current bar's value with no shift; the engine uses it on bar N+1.

**L0 (current bar's level — for static indicators):** name the column without `_prev`. Use for VWAP, fixed thresholds, slow EMAs — any line where the level barely changes bar-to-bar.

#### Gate logic (handled by the engine)

For L-type fills, the engine requires that the previous bar's close was on the *opposite* side of the level from the cross direction (so a "cross above" only fires if price was previously below the level). This prevents phantom fills when price is already past the line. You don't implement this; the engine does.

#### What you write in the trigger detector function

Your `detect_*_triggers()` function returns one boolean Series per trigger base — keyed exactly as `{trigger_prefix}_{base}`. **Do not include suffixed variants in the dict.**

For the UT Bot example with `trigger_prefix: "utbm"`:
```python
def detect_ut_bot_modular_triggers(df, **params):
    prev_close = df["close"].shift(1)
    prev_stop  = df["utbm_trailing_stop"].shift(1)
    return {
        "utbm_bull_flip": (prev_close < prev_stop) & (df["close"] > df["utbm_trailing_stop"]),
        "utbm_bear_flip": (prev_close > prev_stop) & (df["close"] < df["utbm_trailing_stop"]),
    }
```

Two keys, matching the two manifest entries. The engine produces all the suffixed variants from these two booleans.

### PB/CB Fidelity (Strategy Builder Concern)

PB (Previous Bar) and CB (Current Bar) fidelity control how confluence conditions are evaluated:
- **PB**: Conditions use the previous bar's closed indicator values (standard, fast)
- **CB**: Conditions recompute indicators using the current forming bar's partial data (more accurate, requires Hi-Fi mode)

Fidelity is NOT a pack-level setting — it's configured in the Strategy Builder when selecting TF Conditions. Both PB and CB variants are automatically available for any pack when Hi-Fi is enabled. Pack authors do not need to consider fidelity in their code.

### Verification Workflow

After installing a pack, validate it through the Strategy Builder:

1. **Create a strategy** in the Strategy Builder using the pack's triggers as entry/exit
2. **Run a backtest** — inspect results in the Enhanced Trade History table at the bottom of the page
3. **Click a trade number (#)** to open the Trade Replay modal — shows main chart, two 1-second hi-fi charts, and replay controls for stepping through the trade second by second
4. **Click entry/exit timestamps** for 1-second candle drill-downs to verify exact fill prices
5. **Click the exec badge** to see the execution workflow steps

This is the single source of truth for validation. Module-level sandbox/simulation tabs have been removed — all validation flows through Strategy Builder.

---

## Complete Example: RSI Zones Pack

### manifest.json
```json
{
  "$schema": "pack_spec_v1",
  "slug": "rsi_zones",
  "name": "RSI Zones",
  "category": "Momentum",
  "description": "Relative Strength Index with overbought/oversold zone classification",
  "author": "user",
  "version": "1.0.0",
  "created_at": "2026-01-01T00:00:00Z",
  "pack_type": "tf_confluence",
  "display_type": "oscillator",
  "interpreters": ["RSI_ZONES"],
  "trigger_prefix": "rsi",
  "parameters_schema": {
    "rsi_period": {"type": "int", "default": 14, "min": 2, "max": 100, "label": "RSI Period"},
    "overbought": {"type": "float", "default": 70.0, "min": 50, "max": 100, "label": "Overbought Level"},
    "oversold": {"type": "float", "default": 30.0, "min": 0, "max": 50, "label": "Oversold Level"}
  },
  "plot_schema": {
    "rsi_color": {"type": "color", "default": "#8b5cf6", "label": "RSI Line Color"}
  },
  "outputs": ["OVERBOUGHT", "BULLISH", "NEUTRAL", "BEARISH", "OVERSOLD"],
  "output_descriptions": {
    "OVERBOUGHT": "RSI above overbought threshold (>70 default)",
    "BULLISH": "RSI between 50 and overbought",
    "NEUTRAL": "RSI near 50 (45-55 range)",
    "BEARISH": "RSI between oversold and 50",
    "OVERSOLD": "RSI below oversold threshold (<30 default)"
  },
  "triggers": [
    {"base": "cross_into_overbought",   "name": "RSI Crosses Into Overbought",  "direction": "BOTH", "type": "BOTH"},
    {"base": "cross_out_of_overbought", "name": "RSI Leaves Overbought",        "direction": "BOTH", "type": "BOTH"},
    {"base": "cross_into_oversold",     "name": "RSI Crosses Into Oversold",    "direction": "BOTH", "type": "BOTH"},
    {"base": "cross_out_of_oversold",   "name": "RSI Leaves Oversold",          "direction": "BOTH", "type": "BOTH"},
    {"base": "cross_above_midline",     "name": "RSI Crosses Above Midline",    "direction": "BOTH", "type": "BOTH"},
    {"base": "cross_below_midline",     "name": "RSI Crosses Below Midline",    "direction": "BOTH", "type": "BOTH"}
  ],
  "indicator_columns": ["rsi"],
  "column_color_map": {"rsi": "rsi_color"},
  "indicator_function": "calculate_rsi_zones",
  "interpreter_function": "interpret_rsi_zones",
  "trigger_function": "detect_rsi_zones_triggers",
  "requires_indicators": ["rsi"]
}
```

> **Why no `trigger_levels` block?** RSI is an oscillator value, not a price level. Price doesn't "cross" 70 — RSI does, and RSI is computed only at bar close. So no L-type variant is meaningful here. The engine still emits C and CC variants automatically.

> **For an overlay pack with price-level triggers** (UT Bot, Bollinger Bands, EMAs), add a `trigger_levels` block that maps each level-cross trigger to its column, e.g.:
> ```json
> "trigger_levels": {
>   "bull_flip": {"level_column": "utbm_trailing_stop_prev", "cross": "above"}
> }
> ```
> The engine will materialize C, L, LC, and CC variants automatically. **Do not enumerate `_ib`/`_lc`/`_cc` triggers in the manifest.**

### indicator.py
```python
import pandas as pd
import numpy as np

def calculate_rsi_zones(df: pd.DataFrame, **params) -> pd.DataFrame:
    period = params.get("rsi_period", 14)
    result = df.copy()
    delta = result["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    result["rsi"] = 100 - (100 / (1 + rs))
    return result
```

### interpreter.py
```python
import pandas as pd
import numpy as np

def interpret_rsi_zones(df: pd.DataFrame, **params) -> pd.Series:
    overbought = params.get("overbought", 70.0)
    oversold = params.get("oversold", 30.0)
    def classify(row):
        rsi = row.get("rsi", np.nan)
        if pd.isna(rsi):
            return None
        if rsi >= overbought:
            return "OVERBOUGHT"
        elif rsi > 55:
            return "BULLISH"
        elif rsi >= 45:
            return "NEUTRAL"
        elif rsi > oversold:
            return "BEARISH"
        else:
            return "OVERSOLD"
    return df.apply(classify, axis=1)

def detect_rsi_zones_triggers(df: pd.DataFrame, **params) -> dict:
    overbought = params.get("overbought", 70.0)
    oversold = params.get("oversold", 30.0)
    triggers = {}
    rsi = df["rsi"]
    rsi_prev = rsi.shift(1)
    triggers["rsi_enter_overbought"] = (rsi >= overbought) & (rsi_prev < overbought)
    triggers["rsi_exit_overbought"] = (rsi < overbought) & (rsi_prev >= overbought)
    triggers["rsi_enter_oversold"] = (rsi <= oversold) & (rsi_prev > oversold)
    triggers["rsi_exit_oversold"] = (rsi > oversold) & (rsi_prev <= oversold)
    triggers["rsi_cross_above_50"] = (rsi > 50) & (rsi_prev <= 50)
    triggers["rsi_cross_below_50"] = (rsi < 50) & (rsi_prev >= 50)
    return triggers
```

---

## Built-In Pack Example: EMA Stack (for reference)

The EMA Stack is a built-in pack showing how the three-layer architecture works:

**Indicator** (`indicators.py`): Calculates `ema_8`, `ema_21`, `ema_50` via `df['close'].ewm(span=period, adjust=False).mean()`

**Interpreter** (`interpreters.py`): Classifies EMA alignment into 6 states:
- `SML`: Price > Short > Mid > Long (Full Bull Stack)
- `SLM`: Short > Price > Mid > Long
- `MSL`: Short > Mid > Price > Long
- `MLS`: Short > Mid > Long > Price
- `LSM`: Transitional state
- `LMS`: Long > Mid > Short (Full Bear Stack)

**Triggers**: Cross events — `ema_cross_bull` (Short crosses above Mid), `ema_cross_bear` (Short crosses below Mid), `ema_mid_cross_bull` (Mid crosses above Long), `ema_mid_cross_bear` (Mid crosses below Long)

---

## Pine Script Translation Reference

If the user provides TradingView Pine Script, translate using these mappings:

| Pine Script | Python Equivalent |
|---|---|
| `ta.ema(close, N)` | `df['close'].ewm(span=N, adjust=False).mean()` |
| `ta.sma(close, N)` | `df['close'].rolling(window=N).mean()` |
| `ta.rsi(close, N)` | Wilder RSI: `ewm(span=N)` on gain/loss |
| `ta.atr(N)` | True Range with Wilder smoothing: `tr = max(H-L, |H-prevC|, |L-prevC|)`, then `atr[i] = atr[i-1] + (1/N) * (tr[i] - atr[i-1])` (alpha=1/N, NOT 2/(N+1)) |
| `ta.stdev(src, N)` | `df[src].rolling(window=N).std()` |
| `ta.crossover(a, b)` | `(a > b) & (a.shift(1) <= b.shift(1))` |
| `ta.crossunder(a, b)` | `(a < b) & (a.shift(1) >= b.shift(1))` |
| `ta.highest(src, N)` | `df[src].rolling(window=N).max()` |
| `ta.lowest(src, N)` | `df[src].rolling(window=N).min()` |
| `ta.bb(src, N, mult)` | `basis = sma; dev = mult * std; upper = basis + dev; lower = basis - dev` |
| `ta.macd(src, fast, slow, sig)` | `fast_ema - slow_ema`, then `signal = macd.ewm(span=sig).mean()` |
| `ta.kc(src, N, mult)` | Keltner: `basis = ema(src, N); atr_val = atr(N); upper = basis + mult*atr; lower = basis - mult*atr` |
| `math.abs(x)` | `np.abs(x)` or `abs(x)` |
| `nz(x, 0)` | `x.fillna(0)` |
| `close[1]` | `df['close'].shift(1)` |
| `barstate.isconfirmed` | Always true (we process on bar close) |

### Common Pine Script Patterns

**Overlays** (drawn on price chart): EMAs, Bollinger Bands, Keltner Channels, Ichimoku — these produce indicator columns that overlap with price

**Oscillators** (drawn in separate pane): RSI, MACD, Stochastic, CCI — these produce values in their own range (0-100, etc.)

**Multi-output indicators**: Bollinger has upper/middle/lower bands — use multiple `indicator_columns`

When translating Pine Script:
1. Identify the core calculation and convert to pandas operations
2. Map Pine `plot()` calls to `indicator_columns`
3. Map Pine `alertcondition()` calls to triggers
4. Map any visual zones/fills to interpreter outputs
