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
      "direction": "LONG | SHORT | BOTH",
      "type": "ENTRY | EXIT",
      "execution": "bar_close"
    }
  ],

  "indicator_columns": ["col_name_1", "col_name_2"],

  "indicator_function": "calculate_your_pack",
  "interpreter_function": "interpret_your_pack",
  "trigger_function": "detect_your_pack_triggers",

  "requires_indicators": ["col_name_1"]
}
```

### Manifest Rules

- **slug**: lowercase, underscores only, starts with letter (e.g., `keltner_channels`)
- **interpreters**: list with ONE key in UPPERCASE (e.g., `["KELTNER_CHANNELS"]`)
- **trigger_prefix**: short lowercase prefix for trigger IDs (e.g., `kc`)
- **outputs**: mutually exclusive states — every bar is classified into exactly one
- **triggers**: events that fire on state transitions (crosses, flips, etc.)
- **indicator_columns**: column names your indicator.py adds to the DataFrame
- **indicator_function / interpreter_function / trigger_function**: exact function names in your Python files

### Reserved Names (DO NOT USE)

These trigger prefixes are taken: `ema`, `macd`, `macd_hist`, `vwap`, `rvol`, `utbot`, `bar_count`

These interpreter keys are taken: `EMA_STACK`, `MACD_LINE`, `MACD_HISTOGRAM`, `VWAP`, `RVOL`, `UTBOT`

These indicator columns are taken: `ema_8`, `ema_21`, `ema_50`, `macd_line`, `macd_signal`, `macd_hist`, `vwap`, `vwap_sd1_upper`, `vwap_sd1_lower`, `vwap_sd2_upper`, `vwap_sd2_lower`, `atr`, `vol_sma`, `rvol`, `utbot_stop`, `utbot_direction`

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
- `detect_*_triggers` returns a dict of boolean Series
- Trigger keys MUST be `{trigger_prefix}_{base}` matching manifest triggers
- Use `.shift(1)` to compare current bar with previous bar for crosses
- Trigger detection should be vectorized (boolean operations on Series)

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
    {"base": "enter_overbought", "name": "Enter Overbought", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
    {"base": "exit_overbought", "name": "Exit Overbought", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
    {"base": "enter_oversold", "name": "Enter Oversold", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
    {"base": "exit_oversold", "name": "Exit Oversold", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
    {"base": "cross_above_50", "name": "Cross Above 50", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
    {"base": "cross_below_50", "name": "Cross Below 50", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"}
  ],
  "indicator_columns": ["rsi"],
  "indicator_function": "calculate_rsi_zones",
  "interpreter_function": "interpret_rsi_zones",
  "trigger_function": "detect_rsi_zones_triggers",
  "requires_indicators": ["rsi"]
}
```

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
| `ta.atr(N)` | True Range EWM: `max(H-L, |H-prevC|, |L-prevC|).ewm(span=N).mean()` |
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
