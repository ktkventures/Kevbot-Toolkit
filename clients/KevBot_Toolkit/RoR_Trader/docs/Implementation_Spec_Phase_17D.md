# Phase 17D: Indicator Audit & Expansion — Implementation Spec

**Version:** 1.0
**Date:** February 19, 2026
**Purpose:** Methodical audit of every indicator against its Pine Script reference, documentation of all states/triggers/settings, design of three new packs (SuperTrend, Swing 123, Strat Assistant), and Pack Builder pipeline improvements. Designed for Ralph-mode autonomous execution.

---

## Table of Contents

1. [Conventions & File Map](#1-conventions--file-map)
2. [Audit: EMA Stack](#2-audit-ema-stack)
3. [Audit: MACD (Line + Histogram)](#3-audit-macd-line--histogram)
4. [Audit: VWAP](#4-audit-vwap)
5. [Audit: RVOL](#5-audit-rvol)
6. [Audit: UT Bot](#6-audit-ut-bot)
7. [Audit: Bollinger Bands (User Pack)](#7-audit-bollinger-bands-user-pack)
8. [Audit: SR Channels (User Pack)](#8-audit-sr-channels-user-pack)
9. [New Pack: SuperTrend](#9-new-pack-supertrend)
10. [New Pack: Swing 123](#10-new-pack-swing-123)
11. [New Pack: Strat Assistant](#11-new-pack-strat-assistant)
12. [Pack Builder Improvements](#12-pack-builder-improvements)
13. [Execution Order & Verification](#13-execution-order--verification)

---

## 1. Conventions & File Map

### Key Files

| File | Purpose | Approx Lines |
|------|---------|-------------|
| `src/indicators.py` | Built-in indicator calculations (EMA, MACD, VWAP, ATR, RVOL) | ~485 |
| `src/interpreters.py` | Built-in interpreter/trigger funcs + registries | ~699 |
| `src/confluence_groups.py` | Template definitions, ConfluenceGroup, default groups | ~600 |
| `src/app.py` | Main Streamlit app — chart rendering, code tabs, overlays | ~12,500+ |
| `src/pack_builder.py` | Prompt assembly, LLM response parser, installer | ~583 |
| `src/pack_builder_context.md` | Architecture reference embedded in prompts | ~410 |
| `src/pack_spec.py` | Manifest schema + AST safety validation | ~424 |
| `user_packs/bollinger_bands/` | Bollinger Bands pack (3 files) | |
| `user_packs/sr_channels/` | SR Channels pack (3 files) | |
| `reference-indicators/` | 10 Pine Script reference files | |

### Audit Notation

- ✅ **Pass** — Python matches Pine Script reference, all states/triggers correct
- ⚠️ **Fix Needed** — Deviation found, fix action documented
- 🆕 **New** — No existing implementation, to be created
- ❌ **Missing** — Expected implementation not found

### Indicator Architecture Layers

```
Layer 1: indicators.py     → calculate_*() → adds numeric columns to DataFrame
Layer 2: interpreters.py   → interpret_*() → classifies each bar into one state
Layer 3: interpreters.py   → detect_*()    → boolean Series for trigger events
```

User packs follow the same three-layer pattern in `indicator.py` + `interpreter.py`.

---

## 2. Audit: EMA Stack

### 2.1 Python Implementation

**Indicator** (`indicators.py:129-131`):
```python
def calculate_ema(df, period, column="close"):
    return df[column].ewm(span=period, adjust=False).mean()
```
Called for periods 8, 21, 50 via `_run_ema_stack_indicators()`.

**Interpreter** (`interpreters.py:86-128`):
Row-by-row classification of `close` vs `ema_8` vs `ema_21` vs `ema_50` ordering.

**Triggers** (`interpreters.py:131-163`):
- `ema_cross_bull`: ema_8 crosses above ema_21
- `ema_cross_bear`: ema_8 crosses below ema_21
- `ema_mid_cross_bull`: ema_21 crosses above ema_50
- `ema_mid_cross_bear`: ema_21 crosses below ema_50

### 2.2 Reference Comparison

Pine Script EMA: `ta.ema(close, N)` → Uses standard EMA formula: `α = 2/(N+1)`, recursive.
Python EMA: `ewm(span=N, adjust=False).mean()` → Same formula: `α = 2/(span+1)`, no adjustment.

**Match:** ✅ Exact match. `adjust=False` is the correct Pine Script equivalent.

### 2.3 States & Triggers Inventory

| State | Meaning | Coverage |
|-------|---------|----------|
| `SML` | Price > Short(8) > Mid(21) > Long(50) — Full Bull Stack | ✅ |
| `SLM` | Short > Price > Mid > Long — Price dipped below short | ✅ |
| `MSL` | Short > Mid > Price > Long — Price in middle | ✅ |
| `MLS` | Short > Mid > Long > Price — Price below all, bull order | ✅ |
| `LMS` | Price < Short < Mid < Long — Full Bear Stack | ✅ |
| `LSM` | All other transitional orderings | ✅ (catch-all) |

**Note:** The `LMS` condition in code (`p < s < m < l`) is exactly the same as `MLS` condition — both check `s > m > l > p`. But `LMS` is reached from a different branch:
- Line 122: `elif p < s < m < l` → `LMS` — This branch is actually **dead code** because line 119 (`elif s > m > l > p: → MLS`) catches the same condition first.

**Issue Found:** `LMS` state can never be reached. When `p < s < m < l`, both `s > m > l > p` (line 119) and `p < s < m < l` (line 122) are true, but MLS matches first.

**However:** Looking more carefully: `s > m > l > p` means Short > Mid > Long > Price. And `p < s < m < l` means Price < Short < Mid < Long. These are **identical conditions** (`s > m > l > p` ≡ `p < s < m < l`). So `LMS` is indeed unreachable — it's always classified as `MLS`.

**Fix needed?** The intent of `LMS` is "Full Bear Stack" (Long > Mid > Short > Price), which should be `l > m > s > p`. The current code has a bug — it says `p < s < m < l` which is the wrong ordering for a bear stack.

### 2.4 Verdict: ⚠️ Fix Needed

| Item | Status |
|------|--------|
| EMA calculation | ✅ Correct |
| Interpreter SML/SLM/MSL | ✅ Correct |
| Interpreter MLS | ✅ Correct (Short > Mid > Long > Price) |
| Interpreter LMS | ⚠️ Dead code — condition `p < s < m < l` is identical to MLS. Should be `l > m > s > p` (Long > Mid > Short, full bear) |
| Interpreter LSM | ✅ Catch-all for transitional states |
| Triggers | ✅ All 4 cross triggers correct |

**Fix Action:** Change `LMS` condition from `p < s < m < l` to `l > m > s > p` (or equivalently `p < s and s < m and m < l` → `l > m > s > p`). The intent is "bear stack" where the EMAs are in reverse order (long on top).

Wait — re-reading the code more carefully:

```python
# Line 119-120: Price below all EMAs but EMAs still bullish order
elif s > m > l > p:
    return "MLS"
# Line 121-123: Full bear stack: Price < Short < Mid < Long
elif p < s < m < l:
    return "LMS"
```

`s > m > l > p` means `s > m AND m > l AND l > p` → Short > Mid > Long > Price.
`p < s < m < l` means `p < s AND s < m AND m < l` → Price < Short < Mid < Long.

These are **mathematically identical**. So LMS is truly unreachable via this branch.

The intended meaning of LMS is "Long > Mid > Short" (bear order). The correct condition should be:
```python
elif l > m > s > p:   # Price below all, EMAs in bear order
    return "LMS"
```

And the current MLS condition (`s > m > l > p`) correctly captures "EMAs in bull order, price below all."

**Corrected fix:** Line 122, change `elif p < s < m < l:` to `elif l > m > s > p:`.

---

## 3. Audit: MACD (Line + Histogram)

### 3.1 Python Implementation

**Indicator** (`indicators.py:155-180`):
```python
ema_fast = df['close'].ewm(span=fast, adjust=False).mean()  # default fast=12
ema_slow = df['close'].ewm(span=slow, adjust=False).mean()  # default slow=26
macd_line = ema_fast - ema_slow
macd_signal = macd_line.ewm(span=signal, adjust=False).mean()  # default signal=9
macd_hist = macd_line - macd_signal
```

**MACD Line Interpreter** (`interpreters.py:170-201`):
4 states based on MACD vs Signal and MACD vs Zero:
- `M>S+`: above signal AND above zero
- `M>S-`: above signal BUT below zero
- `M<S-`: below signal AND below zero
- `M<S+`: below signal BUT above zero

**MACD Histogram Interpreter** (`interpreters.py:239-276`):
4 states based on histogram sign and direction:
- `H+up`: positive and rising
- `H+dn`: positive but falling
- `H-dn`: negative and falling
- `H-up`: negative but rising

### 3.2 Reference Comparison

**Pine Script** (`macd_line.pine:11-15`):
```pine
fast_ma = sma_source == "SMA" ? ta.sma(src, fast_length) : ta.ema(src, fast_length)
slow_ma = sma_source == "SMA" ? ta.sma(src, slow_length) : ta.ema(src, slow_length)
macd = fast_ma - slow_ma
signal = sma_signal == "SMA" ? ta.sma(macd, signal_length) : ta.ema(macd, signal_length)
hist = macd - signal
```

Pine Script defaults: `fast_length=12`, `slow_length=26`, `signal_length=9`, `sma_source="EMA"`, `sma_signal="EMA"`.

**Comparison:**
- Our Python uses EMA for both oscillator and signal line → matches Pine defaults.
- Parameters 12/26/9 match exactly.
- `adjust=False` is the correct Pine Script EMA equivalent.
- Histogram = MACD - Signal in both.

**Histogram color logic** (`macd_line.pine:21`):
```pine
color = (hist >= 0 ? (hist[1] < hist ? #26A69A : #B2DFDB) : (hist[1] < hist ? #FFCDD2 : #FF5252))
```
This maps exactly to our 4 histogram states:
- `hist >= 0 AND hist > hist[1]` → green (strong bull) → our `H+up`
- `hist >= 0 AND hist <= hist[1]` → light green (fading bull) → our `H+dn`
- `hist < 0 AND hist > hist[1]` → light red (fading bear) → our `H-up`
- `hist < 0 AND hist <= hist[1]` → red (strong bear) → our `H-dn`

**Match:** ✅ Perfect alignment.

**Pine alertcondition triggers** (`macd_line.pine:17-18`):
```pine
alertcondition(hist[1] >= 0 and hist < 0, title='Rising to falling')
alertcondition(hist[1] <= 0 and hist > 0, title='Falling to rising')
```
These map to our `macd_hist_flip_neg` and `macd_hist_flip_pos` triggers.

### 3.3 States & Triggers Inventory

**MACD Line:**

| State | Meaning | Correct |
|-------|---------|---------|
| `M>S+` | MACD > Signal, MACD > 0 | ✅ |
| `M>S-` | MACD > Signal, MACD < 0 | ✅ |
| `M<S-` | MACD < Signal, MACD < 0 | ✅ |
| `M<S+` | MACD < Signal, MACD > 0 | ✅ |

| Trigger | Condition | Correct |
|---------|-----------|---------|
| `macd_cross_bull` | MACD crosses above Signal | ✅ |
| `macd_cross_bear` | MACD crosses below Signal | ✅ |
| `macd_zero_cross_up` | MACD crosses above 0 | ✅ |
| `macd_zero_cross_down` | MACD crosses below 0 | ✅ |

**MACD Histogram:**

| State | Meaning | Correct |
|-------|---------|---------|
| `H+up` | Positive & rising | ✅ |
| `H+dn` | Positive & falling | ✅ |
| `H-dn` | Negative & falling | ✅ |
| `H-up` | Negative & rising | ✅ |

| Trigger | Condition | Correct |
|---------|-----------|---------|
| `macd_hist_flip_pos` | Hist crosses above 0 | ✅ |
| `macd_hist_flip_neg` | Hist crosses below 0 | ✅ |
| `macd_hist_momentum_shift_up` | Was falling, now rising | ✅ |
| `macd_hist_momentum_shift_down` | Was rising, now falling | ✅ |

### 3.4 Verdict: ✅ Pass

All MACD calculations, states, and triggers match the Pine Script reference exactly.

---

## 4. Audit: VWAP

### 4.1 Python Implementation

**Indicator** (`indicators.py:183-216`):
```python
def calculate_vwap(df, sd1_mult=1.0, sd2_mult=2.0):
    if 'vwap' in df.columns and df['vwap'].notna().any():
        vwap = df['vwap']  # Use pre-existing VWAP (from Alpaca)
    else:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    # Bands use rolling std of typical_price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    rolling_std = typical_price.rolling(window=20, min_periods=5).std()
    return {
        "vwap": vwap,
        "vwap_sd1_upper": vwap + (sd1_mult * rolling_std),
        "vwap_sd1_lower": vwap - (sd1_mult * rolling_std),
        ...
    }
```

**Interpreter** (`interpreters.py:314-377`):
7-zone system: `>+2σ`, `>+1σ`, `>V`, `@V`, `<V`, `<-1σ`, `<-2σ`.

**Triggers** (`interpreters.py:380-426`):
5 triggers: cross_above, cross_below, enter_upper_extreme, enter_lower_extreme, return_to_vwap.

### 4.2 Reference Comparison

**Pine Script** (`vwap.pine:53-63`):
```pine
[_vwap, _stdevUpper, _] = ta.vwap(src, isNewPeriod, 1)
stdevAbs = _stdevUpper - _vwap
bandBasis = calcModeInput == "Standard Deviation" ? stdevAbs : _vwap * 0.01
upperBandValue1 := _vwap + bandBasis * bandMult_1
lowerBandValue1 := _vwap - bandBasis * bandMult_1
```

**Key differences:**

1. **Session Reset:** Pine's `ta.vwap(src, isNewPeriod, 1)` resets cumulative sums when `isNewPeriod` fires (default: `timeframe.change("D")` i.e., daily session boundary). Our Python uses `cumsum()` which **never resets** — it accumulates across the entire DataFrame.

   **Impact:** For multi-day data, our VWAP drifts because cumulative volume/price from prior days carries forward. On a single-day view, the data loader typically returns one session, so the bug is masked. But on multi-day backtest windows, VWAP values are incorrect after day 1.

   **Note:** When Alpaca provides a pre-computed `vwap` column, we use it directly (which is session-aware). The bug only affects the fallback calculation path.

2. **Standard Deviation Bands:** Pine uses `ta.vwap()` which returns the VWAP standard deviation (population-weighted). Our Python uses `typical_price.rolling(window=20).std()` — a simple rolling standard deviation of typical price, **not** VWAP-weighted deviation.

   **Impact:** Our bands track recent price volatility rather than true VWAP deviation. This is a meaningful difference — Pine's VWAP SD bands widen based on how far price deviates from VWAP over the session, while ours track 20-bar price volatility regardless of VWAP.

### 4.3 States & Triggers Inventory

| State | Meaning | Correct |
|-------|---------|---------|
| `>+2σ` | Price > VWAP + 2×SD | ✅ (zone logic correct given bands) |
| `>+1σ` | Price between +1σ and +2σ | ✅ |
| `>V` | Price between VWAP+0.5σ and +1σ | ✅ |
| `@V` | Price within ±0.5σ of VWAP | ✅ |
| `<V` | Price between -0.5σ and -1σ | ✅ |
| `<-1σ` | Price between -1σ and -2σ | ✅ |
| `<-2σ` | Price < VWAP - 2×SD | ✅ |

| Trigger | Condition | Correct |
|---------|-----------|---------|
| `vwap_cross_above` | Close crosses above VWAP | ✅ |
| `vwap_cross_below` | Close crosses below VWAP | ✅ |
| `vwap_enter_upper_extreme` | Close enters >+2σ zone | ✅ |
| `vwap_enter_lower_extreme` | Close enters <-2σ zone | ✅ |
| `vwap_return_to_vwap` | From outside ±1σ back to @V zone | ✅ |

### 4.4 Verdict: ⚠️ Fix Needed

| Item | Status |
|------|--------|
| VWAP core (with Alpaca data) | ✅ Uses pre-computed session VWAP |
| VWAP fallback calculation | ⚠️ No session reset — `cumsum()` spans entire DataFrame |
| SD band calculation | ⚠️ Uses rolling std of typical_price instead of VWAP-weighted deviation |
| 7-zone interpreter | ✅ Correct zone logic |
| 5 triggers | ✅ Correct cross/return logic |

**Fix Actions:**

1. **Session-Aware Reset:** Add session boundary detection to `calculate_vwap()`. Detect session boundaries by checking for gaps ≥ 30 minutes between bars (market close → next open), then reset cumulative sums at each boundary. Implementation:

```python
def calculate_vwap(df, sd1_mult=1.0, sd2_mult=2.0):
    if 'vwap' in df.columns and df['vwap'].notna().any():
        vwap = df['vwap']
    else:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_vol = typical_price * df['volume']

        # Detect session boundaries (gap > 30 min between bars)
        if isinstance(df.index, pd.DatetimeIndex):
            gaps = df.index.to_series().diff()
            session_start = gaps > pd.Timedelta(minutes=30)
            session_start.iloc[0] = True  # first bar is always a session start
        else:
            session_start = pd.Series(False, index=df.index)
            session_start.iloc[0] = True

        # Session-aware cumulative sums
        session_id = session_start.cumsum()
        cum_tp_vol = tp_vol.groupby(session_id).cumsum()
        cum_vol = df['volume'].groupby(session_id).cumsum()
        vwap = cum_tp_vol / cum_vol

    # SD bands: use session-aware VWAP deviation
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    deviation = typical_price - vwap
    if isinstance(df.index, pd.DatetimeIndex):
        gaps = df.index.to_series().diff()
        session_start = gaps > pd.Timedelta(minutes=30)
        session_start.iloc[0] = True
        session_id = session_start.cumsum()
        rolling_std = deviation.groupby(session_id).apply(
            lambda s: s.expanding(min_periods=5).std()
        ).droplevel(0)
    else:
        rolling_std = deviation.rolling(window=20, min_periods=5).std()

    return {
        "vwap": vwap,
        "vwap_sd1_upper": vwap + (sd1_mult * rolling_std),
        "vwap_sd1_lower": vwap - (sd1_mult * rolling_std),
        "vwap_sd2_upper": vwap + (sd2_mult * rolling_std),
        "vwap_sd2_lower": vwap - (sd2_mult * rolling_std),
    }
```

**Note:** This fix only affects the fallback path. When Alpaca provides VWAP, we use it directly and both band approaches are approximations either way.

---

## 5. Audit: RVOL

### 5.1 Python Implementation

**Indicator** (`indicators.py:219-233`):
```python
vol_sma = df['volume'].rolling(window=period, min_periods=5).mean()  # default period=20
rvol = df['volume'] / vol_sma
```

**Interpreter** (`interpreters.py:433-463`):
5 states based on RVOL ratio: EXTREME (>3.0), HIGH (>1.5), NORMAL (0.75-1.5), LOW (0.5-0.75), MINIMAL (<0.5).

**Triggers** (`interpreters.py:466-488`):
- `rvol_spike`: crosses above 1.5
- `rvol_extreme`: crosses above 3.0
- `rvol_fade`: crosses below 1.0

### 5.2 Reference Comparison

**Pine Script** (`rvol.pine`):
```pine
f_rvol(tf) =>
    v  = request.security(syminfo.tickerid, tf, volume, gaps=barmerge.gaps_off)
    ma = request.security(syminfo.tickerid, tf, ta.sma(volume, len), gaps=barmerge.gaps_off)
    nz(ma) == 0 ? na : v / ma
```

**Key differences:**

1. **Multi-timeframe approach:** Pine Script calculates RVOL across 3 timeframes (10s, 1m, 5m) using `request.security()`. Our Python calculates a single RVOL on the current timeframe's bars.

2. **SMA match:** Both use SMA of volume. Pine uses `ta.sma(volume, len)` with default `len=20`. Our Python uses `rolling(window=20).mean()`. These are equivalent.

3. **Ratio calculation:** Both compute `volume / sma(volume)`. Match.

**Assessment:** The core RVOL calculation (volume/SMA) is correct. The multi-TF display is a Pine-specific visualization feature; our interpreter approach (single-TF RVOL with threshold classification) is a reasonable adaptation for the confluence system.

### 5.3 States & Triggers Inventory

| State | Threshold | Correct |
|-------|-----------|---------|
| `EXTREME` | RVOL > 3.0 | ✅ |
| `HIGH` | RVOL > 1.5 | ✅ |
| `NORMAL` | RVOL 0.75-1.5 | ✅ |
| `LOW` | RVOL 0.5-0.75 | ✅ |
| `MINIMAL` | RVOL < 0.5 | ✅ |

| Trigger | Condition | Correct |
|---------|-----------|---------|
| `rvol_spike` | Crosses above 1.5 | ✅ |
| `rvol_extreme` | Crosses above 3.0 | ✅ |
| `rvol_fade` | Crosses below 1.0 | ✅ |

### 5.4 Verdict: ✅ Pass

Core calculation matches. Multi-TF is a display adaptation, not a bug.

---

## 6. Audit: UT Bot

### 6.1 Current State

**Template definition** (`confluence_groups.py:270-295`):
```python
"utbot": {
    "name": "UT Bot",
    "interpreters": ["UTBOT"],
    "trigger_prefix": "utbot",
    "parameters_schema": {
        "atr_period": {"type": "int", "default": 10, ...},
        "atr_multiplier": {"type": "float", "default": 1.0, ...},
    },
    "outputs": ["BULL", "BEAR"],
    "triggers": [
        {"base": "buy", ...},
        {"base": "sell", ...},
    ],
    "indicator_columns": ["utbot_stop", "utbot_direction"],
}
```

**Code tab mapping** (`app.py:10025-10028`):
```python
"utbot": {
    "Indicator": [],
    "Interpreter": [],
    "Triggers": [],
},
```

**Actual functions:** ❌ **None exist.** No `calculate_utbot`, `interpret_utbot`, or `detect_utbot_triggers` functions in `indicators.py` or `interpreters.py`. No registered functions in `GROUP_INDICATOR_FUNCS`, `INTERPRETER_FUNCS`, or `TRIGGER_FUNCS`.

### 6.2 Reference (Pine Script)

**`utbot.pine` core logic:**
```pine
xATR = atr(c)                    // ATR(10)
nLoss = a * xATR                 // 1.0 × ATR = trailing distance

// Trailing stop: ratchets up in uptrend, ratchets down in downtrend
xATRTrailingStop := iff(src > nz(xATRTrailingStop[1], 0) and src[1] > nz(xATRTrailingStop[1], 0),
    max(nz(xATRTrailingStop[1]), src - nLoss),          // In uptrend: raise stop
    iff(src < nz(xATRTrailingStop[1], 0) and src[1] < nz(xATRTrailingStop[1], 0),
        min(nz(xATRTrailingStop[1]), src + nLoss),      // In downtrend: lower stop
        iff(src > nz(xATRTrailingStop[1], 0),
            src - nLoss, src + nLoss)))                  // Flip: set new stop

// Position from trailing stop crosses
pos := iff(src[1] < nz(xATRTrailingStop[1], 0) and src > nz(xATRTrailingStop[1], 0), 1,
    iff(src[1] > nz(xATRTrailingStop[1], 0) and src < nz(xATRTrailingStop[1], 0), -1,
        nz(pos[1], 0)))

// Signal conditions
ema = ema(src, 1)  // EMA(1) = just close
above = crossover(ema, xATRTrailingStop)
below = crossover(xATRTrailingStop, ema)
buy = src > xATRTrailingStop and above
sell = src < xATRTrailingStop and below
```

### 6.3 Required Implementation

**indicator.py logic** (to add to `indicators.py` as built-in):
```python
def calculate_utbot(df, atr_period=10, atr_multiplier=1.0):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)

    # ATR calculation
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values
    n_loss = atr_multiplier * atr

    # Trailing stop
    trail_stop = np.zeros(n)
    trail_stop[0] = close[0] - n_loss[0]
    for i in range(1, n):
        if close[i] > trail_stop[i-1] and close[i-1] > trail_stop[i-1]:
            trail_stop[i] = max(trail_stop[i-1], close[i] - n_loss[i])
        elif close[i] < trail_stop[i-1] and close[i-1] < trail_stop[i-1]:
            trail_stop[i] = min(trail_stop[i-1], close[i] + n_loss[i])
        elif close[i] > trail_stop[i-1]:
            trail_stop[i] = close[i] - n_loss[i]
        else:
            trail_stop[i] = close[i] + n_loss[i]

    # Direction: 1=bull, -1=bear
    direction = np.zeros(n)
    for i in range(1, n):
        if close[i-1] < trail_stop[i-1] and close[i] > trail_stop[i-1]:
            direction[i] = 1
        elif close[i-1] > trail_stop[i-1] and close[i] < trail_stop[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    result = df.copy()
    result['utbot_stop'] = trail_stop
    result['utbot_direction'] = direction
    return result
```

**interpreter logic:**
```python
def interpret_utbot(df):
    def classify(row):
        d = row.get('utbot_direction', np.nan)
        if pd.isna(d) or d == 0:
            return None
        return "BULL" if d == 1 else "BEAR"
    return df.apply(classify, axis=1)

def detect_utbot_triggers(df):
    d = df['utbot_direction']
    d_prev = d.shift(1)
    return {
        'utbot_buy': (d == 1) & (d_prev != 1),
        'utbot_sell': (d == -1) & (d_prev != -1),
    }
```

### 6.4 Verdict: ❌ Missing Implementation

| Item | Status |
|------|--------|
| Template definition | ✅ Exists in confluence_groups.py |
| Default group config | ✅ `utbot_default` with atr_period=10, mult=1.0 |
| Color mapping | ✅ `utbot_stop` → trail_color |
| Indicator function | ❌ Not implemented |
| Interpreter function | ❌ Not implemented |
| Trigger function | ❌ Not implemented |
| GROUP_INDICATOR_FUNCS registration | ❌ Missing |
| INTERPRETER_FUNCS registration | ❌ Missing |
| TRIGGER_FUNCS registration | ❌ Missing |
| Code tab display | ⚠️ Empty lists |

**Fix Action:** Implement the full UT Bot indicator/interpreter/trigger in `indicators.py` and `interpreters.py`, register in both dispatch registries, and update the code tab mapping in `app.py`.

---

## 7. Audit: Bollinger Bands (User Pack)

### 7.1 Python Implementation

**Indicator** (`user_packs/bollinger_bands/indicator.py`):
```python
basis = result["close"].rolling(window=period).mean()           # SMA(20)
std = result["close"].rolling(window=period).std()              # Std(20)
result["bb_upper"] = basis + mult * std                          # +2σ
result["bb_lower"] = basis - mult * std                          # -2σ
result["bb_bandwidth"] = (bb_upper - bb_lower) / basis          # Bandwidth %
```

**Interpreter** (`user_packs/bollinger_bands/interpreter.py`):
6 states: SQUEEZE_UPPER, SQUEEZE_MID, SQUEEZE_LOWER, UPPER_ZONE, MID_ZONE, LOWER_ZONE.
Squeeze threshold: `bb_bandwidth < 0.04` (4%).

**Triggers:**
- `bb_cross_upper`, `bb_cross_lower`: Band crosses
- `bb_cross_basis_up`, `bb_cross_basis_down`: Basis crosses
- `bb_squeeze_on`, `bb_squeeze_off`: Squeeze transitions

### 7.2 Reference Comparison

Standard Bollinger Bands (Pine `ta.bb()`):
```pine
basis = ta.sma(close, length)      // SMA
dev = mult * ta.stdev(close, length) // Standard deviation
upper = basis + dev
lower = basis - dev
```

**Match:** ✅ Our calculation is standard BB. `rolling(window).mean()` = SMA, `rolling(window).std()` = sample std (Pine uses population std by default, but the difference is negligible at window=20).

### 7.3 States & Triggers Inventory

| State | Meaning | Correct |
|-------|---------|---------|
| `SQUEEZE_UPPER` | In squeeze + price in upper zone | ✅ |
| `SQUEEZE_MID` | In squeeze + price near basis | ✅ |
| `SQUEEZE_LOWER` | In squeeze + price in lower zone | ✅ |
| `UPPER_ZONE` | Normal volatility, above basis+25% | ✅ |
| `MID_ZONE` | Normal volatility, near basis ±25% | ✅ |
| `LOWER_ZONE` | Normal volatility, below basis-25% | ✅ |

| Trigger | Condition | Correct |
|---------|-----------|---------|
| `bb_cross_upper` | Close crosses above upper band | ✅ |
| `bb_cross_lower` | Close crosses below lower band | ✅ |
| `bb_cross_basis_up` | Close crosses above basis | ✅ |
| `bb_cross_basis_down` | Close crosses below basis | ✅ |
| `bb_squeeze_on` | Bandwidth drops below threshold | ✅ |
| `bb_squeeze_off` | Bandwidth rises above threshold | ✅ |

### 7.4 Verdict: ✅ Pass

Well-designed pack with proper squeeze/zone combination. Mutually exclusive states, meaningful triggers.

---

## 8. Audit: SR Channels (User Pack)

### 8.1 Python Implementation

**Indicator** (`user_packs/sr_channels/indicator.py`):
- Detects pivot highs/lows over configurable lookback
- Clusters pivots into channels based on proximity (channel width %)
- Ranks channels by pivot count + bar touch strength
- Outputs nearest channel top/bottom, in_channel flag, breakout flags

**Interpreter** (`user_packs/sr_channels/interpreter.py`):
5 states: ABOVE_RESISTANCE, IN_RESISTANCE, BETWEEN_LEVELS, IN_SUPPORT, BELOW_SUPPORT.

**Triggers:**
- `src_resistance_broken`, `src_support_broken`: Breakout events
- `src_enter_sr_zone`, `src_exit_sr_zone`: Channel entry/exit

### 8.2 Reference Comparison

**Pine Script** (`sr_channels.pine`) — LonesomeTheBlue's "Support Resistance Channels":
This is a 193-line Pine Script that implements a similar pivot-based approach:
- Finds pivot highs/lows
- Clusters them into channels
- Ranks by strength (pivot count + bar touches)
- Shows strongest non-overlapping channels

**Key differences:**
1. Pine Script draws channel boxes visually; our pack outputs numeric columns
2. Channel selection algorithm is adapted but follows the same logic:
   - Find pivots → cluster by proximity → rank by strength → select top-N non-overlapping
3. Parameters map: `pivot_period`, `channel_width_pct`, `min_strength`, `max_num_sr`, `loopback`

The core algorithm is faithful to the reference.

### 8.3 States & Triggers Inventory

| State | Meaning | Correct |
|-------|---------|---------|
| `ABOVE_RESISTANCE` | Price above all channels | ✅ |
| `IN_RESISTANCE` | Inside channel, price ≥ midpoint | ✅ |
| `BETWEEN_LEVELS` | Between channels, not inside any | ✅ |
| `IN_SUPPORT` | Inside channel, price < midpoint | ✅ |
| `BELOW_SUPPORT` | Price below all channels | ✅ |

| Trigger | Condition | Correct |
|---------|-----------|---------|
| `src_resistance_broken` | Close breaks above channel top | ✅ |
| `src_support_broken` | Close breaks below channel bottom | ✅ |
| `src_enter_sr_zone` | in_channel transitions 0→1 | ✅ |
| `src_exit_sr_zone` | in_channel transitions 1→0 | ✅ |

### 8.4 Verdict: ✅ Pass

Faithful adaptation of the Pine Script reference. The loop-based calculation is slow for large datasets but functionally correct.

---

## 9. New Pack: SuperTrend

### 9.1 Pine Script Analysis

**Source:** `reference-indicators/supertrend.pine` (Pine v4, 37 lines)

**Core logic:**
```pine
atr = changeATR ? atr(Periods) : sma(tr, Periods)  // ATR(10) or SMA(TR, 10)
up = src - (Multiplier * atr)                        // Lower band: hl2 - 3×ATR
dn = src + (Multiplier * atr)                        // Upper band: hl2 + 3×ATR

// Ratchet: up can only rise, dn can only fall (within same trend)
up := close[1] > up1 ? max(up, up1) : up
dn := close[1] < dn1 ? min(dn, dn1) : dn

// Trend direction: 1=bull, -1=bear
trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend

buySignal = trend == 1 and trend[1] == -1
sellSignal = trend == -1 and trend[1] == 1
```

**Inputs:** ATR Period (default 10), Source (hl2), ATR Multiplier (default 3.0).

**Visual output:** Two lines (up trend line in green when bull, down trend line in red when bear), buy/sell signal markers.

### 9.2 Designed Manifest

```json
{
  "$schema": "pack_spec_v1",
  "slug": "supertrend",
  "name": "SuperTrend",
  "category": "Trend",
  "description": "ATR-based trend-following indicator with dynamic support/resistance bands and trend direction",
  "author": "user",
  "version": "1.0.0",
  "created_at": "2026-02-19T00:00:00Z",
  "pack_type": "tf_confluence",
  "display_type": "overlay",
  "interpreters": ["SUPERTREND"],
  "trigger_prefix": "st",
  "parameters_schema": {
    "atr_period": {
      "type": "int", "default": 10, "min": 1, "max": 50, "label": "ATR Period"
    },
    "atr_multiplier": {
      "type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "label": "ATR Multiplier"
    }
  },
  "plot_schema": {
    "bull_color": {"type": "color", "default": "#22c55e", "label": "Bull Trend Color"},
    "bear_color": {"type": "color", "default": "#ef4444", "label": "Bear Trend Color"}
  },
  "outputs": ["BULL_TRENDING", "BULL_NEAR_STOP", "BEAR_TRENDING", "BEAR_NEAR_STOP"],
  "output_descriptions": {
    "BULL_TRENDING": "Price above SuperTrend line, trending higher",
    "BULL_NEAR_STOP": "Price above SuperTrend but within 0.5× ATR of stop — tight to support",
    "BEAR_TRENDING": "Price below SuperTrend line, trending lower",
    "BEAR_NEAR_STOP": "Price below SuperTrend but within 0.5× ATR of stop — tight to resistance"
  },
  "triggers": [
    {"base": "bull_flip", "name": "Trend Flips Bullish", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
    {"base": "bear_flip", "name": "Trend Flips Bearish", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
    {"base": "near_stop_bull", "name": "Approaching Bull Stop", "direction": "BOTH", "type": "EXIT", "execution": "bar_close"},
    {"base": "near_stop_bear", "name": "Approaching Bear Stop", "direction": "BOTH", "type": "EXIT", "execution": "bar_close"}
  ],
  "indicator_columns": ["st_line", "st_direction", "st_atr"],
  "column_color_map": {
    "st_line": "bull_color"
  },
  "indicator_function": "calculate_supertrend",
  "interpreter_function": "interpret_supertrend",
  "trigger_function": "detect_supertrend_triggers",
  "requires_indicators": ["st_line", "st_direction"]
}
```

### 9.3 Designed indicator.py

```python
import pandas as pd
import numpy as np


def calculate_supertrend(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate SuperTrend indicator.

    SuperTrend uses ATR to create a trailing stop that ratchets in the
    direction of the trend. When price crosses the stop, trend flips.

    Outputs:
        st_line: The active SuperTrend level (support in uptrend, resistance in downtrend)
        st_direction: 1 for bullish, -1 for bearish
        st_atr: Current ATR value (for proximity calculations)
    """
    atr_period = params.get("atr_period", 10)
    multiplier = params.get("atr_multiplier", 3.0)

    result = df.copy()
    high = result["high"].values
    low = result["low"].values
    close = result["close"].values
    n = len(close)

    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    # ATR via EMA (matches Pine v4 atr() which uses RMA/Wilder)
    atr = np.zeros(n)
    atr[0] = tr[0]
    alpha = 1.0 / atr_period  # Wilder smoothing
    for i in range(1, n):
        atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])

    # Source: hl2
    src = (high + low) / 2.0

    # Basic bands
    basic_up = src - multiplier * atr
    basic_dn = src + multiplier * atr

    # Ratcheted bands and trend
    up = np.zeros(n)
    dn = np.zeros(n)
    trend = np.ones(n, dtype=int)
    st_line = np.zeros(n)

    up[0] = basic_up[0]
    dn[0] = basic_dn[0]

    for i in range(1, n):
        # Ratchet up band (can only rise in uptrend)
        up[i] = max(basic_up[i], up[i - 1]) if close[i - 1] > up[i - 1] else basic_up[i]
        # Ratchet dn band (can only fall in downtrend)
        dn[i] = min(basic_dn[i], dn[i - 1]) if close[i - 1] < dn[i - 1] else basic_dn[i]

        # Trend direction
        if trend[i - 1] == -1 and close[i] > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    # SuperTrend line: up when bull, dn when bear
    for i in range(n):
        st_line[i] = up[i] if trend[i] == 1 else dn[i]

    result["st_line"] = st_line
    result["st_direction"] = trend
    result["st_atr"] = atr

    return result
```

### 9.4 Designed interpreter.py

```python
import pandas as pd
import numpy as np


def interpret_supertrend(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar's position relative to SuperTrend.

    States:
        BULL_TRENDING: Price above ST line, not near stop
        BULL_NEAR_STOP: Price above ST but within 0.5× ATR of stop
        BEAR_TRENDING: Price below ST line, not near stop
        BEAR_NEAR_STOP: Price below ST but within 0.5× ATR of stop
    """
    def classify(row):
        direction = row.get("st_direction", np.nan)
        st_line = row.get("st_line", np.nan)
        atr_val = row.get("st_atr", np.nan)
        close = row.get("close", np.nan)

        if pd.isna(direction) or pd.isna(st_line) or pd.isna(atr_val):
            return None

        near_threshold = 0.5 * atr_val

        if direction == 1:
            if close - st_line <= near_threshold:
                return "BULL_NEAR_STOP"
            return "BULL_TRENDING"
        else:
            if st_line - close <= near_threshold:
                return "BEAR_NEAR_STOP"
            return "BEAR_TRENDING"

    return df.apply(classify, axis=1)


def detect_supertrend_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect SuperTrend trigger events.

    Triggers:
        st_bull_flip: Trend changes from bear to bull
        st_bear_flip: Trend changes from bull to bear
        st_near_stop_bull: Price approaches bull stop (within 0.5× ATR)
        st_near_stop_bear: Price approaches bear stop (within 0.5× ATR)
    """
    prefix = "st"
    triggers = {}

    direction = df["st_direction"]
    direction_prev = direction.shift(1)

    # Trend flips
    triggers[f"{prefix}_bull_flip"] = (direction == 1) & (direction_prev == -1)
    triggers[f"{prefix}_bear_flip"] = (direction == -1) & (direction_prev == 1)

    # Near stop triggers
    close = df["close"]
    st_line = df["st_line"]
    atr_val = df["st_atr"]
    threshold = 0.5 * atr_val

    close_prev = close.shift(1)
    st_line_prev = st_line.shift(1)
    threshold_prev = 0.5 * atr_val.shift(1)

    # Was not near, now near (bull)
    was_far_bull = (close_prev - st_line_prev) > threshold_prev
    now_near_bull = ((close - st_line) <= threshold) & (direction == 1)
    triggers[f"{prefix}_near_stop_bull"] = was_far_bull & now_near_bull

    # Was not near, now near (bear)
    was_far_bear = (st_line_prev - close_prev) > threshold_prev
    now_near_bear = ((st_line - close) <= threshold) & (direction == -1)
    triggers[f"{prefix}_near_stop_bear"] = was_far_bear & now_near_bear

    return triggers
```

### 9.5 plot_config

SuperTrend has a single line that changes color based on trend direction. We'll handle this via `candle_color_column` style approach — the `st_direction` column can drive conditional coloring in the chart render. The primary `st_line` column maps to `bull_color` via `column_color_map`.

No band_fills or reference_lines needed.

### 9.6 Verification Checklist

- [ ] `calculate_supertrend()` produces `st_line`, `st_direction`, `st_atr` columns
- [ ] ATR uses Wilder smoothing (1/period alpha) matching Pine v4 `atr()`
- [ ] Trailing stop ratchets correctly (up only rises, dn only falls in their respective trends)
- [ ] Trend flip logic matches Pine: bull→bear when close < up[prev], bear→bull when close > dn[prev]
- [ ] `interpret_supertrend()` returns 4 mutually exclusive states
- [ ] `detect_supertrend_triggers()` returns 4 trigger Series with `st_` prefix
- [ ] Manifest passes `validate_manifest()` — no reserved name collisions
- [ ] Pine Script reference copied to `user_packs/supertrend/reference.pine`

---

## 10. New Pack: Swing 123

### 10.1 Pine Script Analysis

**Source:** `reference-indicators/swing_123.pine` (Pine v6, 32 lines)

**Core logic:**
```pine
// Candle 2 conditions
bullC2 = low < low[1] and close > close[1]     // Lower low + reclaimed prior close
bearC2 = high > high[1] and close < close[1]    // Higher high + dropped below prior close

// Candle 3 conditions (continuation of C2)
bullC3 = bullC2[1] and close > high[1]           // Prior was bullC2, now close > prior high
bearC3 = bearC2[1] and close < low[1]            // Prior was bearC2, now close < prior low
```

**Visual output:** Bar coloring — C3 overrides C2:
- Bull C2: Medium yellow (#FFD11A)
- Bull C3: Neon yellow (#FFFF00)
- Bear C2: Medium pink (#FF66B3)
- Bear C3: Neon pink (#FF33CC)
- Otherwise: default candle color

This is a candle-coloring indicator with no lines — it classifies bar patterns.

### 10.2 Designed Manifest

```json
{
  "$schema": "pack_spec_v1",
  "slug": "swing_123",
  "name": "Swing 1-2-3",
  "category": "Trend",
  "description": "Candle 2 and Candle 3 pattern detection for swing reversal and continuation signals",
  "author": "user",
  "version": "1.0.0",
  "created_at": "2026-02-19T00:00:00Z",
  "pack_type": "tf_confluence",
  "display_type": "hidden",
  "interpreters": ["SWING_123"],
  "trigger_prefix": "sw123",
  "parameters_schema": {},
  "plot_schema": {
    "bull_c2_color": {"type": "color", "default": "#FFD11A", "label": "Bullish C2 Color"},
    "bull_c3_color": {"type": "color", "default": "#FFFF00", "label": "Bullish C3 Color"},
    "bear_c2_color": {"type": "color", "default": "#FF66B3", "label": "Bearish C2 Color"},
    "bear_c3_color": {"type": "color", "default": "#FF33CC", "label": "Bearish C3 Color"}
  },
  "plot_config": {
    "candle_color_column": "sw123_candle_color"
  },
  "outputs": ["BULL_C3", "BULL_C2", "BEAR_C3", "BEAR_C2", "NEUTRAL"],
  "output_descriptions": {
    "BULL_C3": "Bullish Candle 3: Prior bar was C2 and current close > prior high (continuation confirmed)",
    "BULL_C2": "Bullish Candle 2: Made lower low but closed above prior close (reversal candidate)",
    "BEAR_C3": "Bearish Candle 3: Prior bar was C2 and current close < prior low (continuation confirmed)",
    "BEAR_C2": "Bearish Candle 2: Made higher high but closed below prior close (reversal candidate)",
    "NEUTRAL": "No swing pattern detected on this bar"
  },
  "triggers": [
    {"base": "bull_c2", "name": "Bullish Candle 2", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
    {"base": "bull_c3", "name": "Bullish Candle 3", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
    {"base": "bear_c2", "name": "Bearish Candle 2", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
    {"base": "bear_c3", "name": "Bearish Candle 3", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"}
  ],
  "indicator_columns": ["sw123_pattern", "sw123_candle_color"],
  "column_color_map": {},
  "indicator_function": "calculate_swing_123",
  "interpreter_function": "interpret_swing_123",
  "trigger_function": "detect_swing_123_triggers",
  "requires_indicators": ["sw123_pattern"]
}
```

### 10.3 Designed indicator.py

```python
import pandas as pd
import numpy as np


def calculate_swing_123(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Detect Swing 1-2-3 candle patterns.

    Candle 2 (C2): Reversal candidate
        Bull C2: Makes lower low than prior bar AND closes above prior close
        Bear C2: Makes higher high than prior bar AND closes below prior close

    Candle 3 (C3): Continuation confirmation
        Bull C3: Prior bar was Bull C2 AND current close > prior high
        Bear C3: Prior bar was Bear C2 AND current close < prior low

    Outputs:
        sw123_pattern: Integer code (0=neutral, 1=bull_c2, 2=bull_c3, -1=bear_c2, -2=bear_c3)
        sw123_candle_color: Hex color string for bar coloring (or empty for default)
    """
    result = df.copy()
    close = result["close"].values
    high = result["high"].values
    low = result["low"].values
    n = len(close)

    pattern = np.zeros(n, dtype=int)
    colors = [""] * n

    # Default colors (can be overridden by plot_schema via params)
    bull_c2_color = "#FFD11A"
    bull_c3_color = "#FFFF00"
    bear_c2_color = "#FF66B3"
    bear_c3_color = "#FF33CC"

    for i in range(1, n):
        # Candle 2 conditions
        bull_c2 = low[i] < low[i - 1] and close[i] > close[i - 1]
        bear_c2 = high[i] > high[i - 1] and close[i] < close[i - 1]

        # Candle 3 conditions (check if prior bar was C2)
        bull_c3 = (pattern[i - 1] == 1) and close[i] > high[i - 1]
        bear_c3 = (pattern[i - 1] == -1) and close[i] < low[i - 1]

        # Priority: C3 > C2 (matching Pine Script)
        if bull_c3:
            pattern[i] = 2
            colors[i] = bull_c3_color
        elif bear_c3:
            pattern[i] = -2
            colors[i] = bear_c3_color
        elif bull_c2:
            pattern[i] = 1
            colors[i] = bull_c2_color
        elif bear_c2:
            pattern[i] = -1
            colors[i] = bear_c2_color

    result["sw123_pattern"] = pattern
    result["sw123_candle_color"] = colors

    return result
```

### 10.4 Designed interpreter.py

```python
import pandas as pd
import numpy as np


def interpret_swing_123(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar into a swing 1-2-3 pattern state.

    States:
        BULL_C3: Bullish continuation confirmed (C3)
        BULL_C2: Bullish reversal candidate (C2)
        BEAR_C3: Bearish continuation confirmed (C3)
        BEAR_C2: Bearish reversal candidate (C2)
        NEUTRAL: No pattern
    """
    pattern_map = {
        2: "BULL_C3",
        1: "BULL_C2",
        -2: "BEAR_C3",
        -1: "BEAR_C2",
        0: "NEUTRAL",
    }

    def classify(row):
        p = row.get("sw123_pattern", 0)
        if pd.isna(p):
            return None
        return pattern_map.get(int(p), "NEUTRAL")

    return df.apply(classify, axis=1)


def detect_swing_123_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect swing 1-2-3 pattern triggers.

    Each pattern detection IS the trigger (C2 and C3 are discrete events).
    """
    prefix = "sw123"
    triggers = {}

    pattern = df["sw123_pattern"]

    triggers[f"{prefix}_bull_c2"] = pattern == 1
    triggers[f"{prefix}_bull_c3"] = pattern == 2
    triggers[f"{prefix}_bear_c2"] = pattern == -1
    triggers[f"{prefix}_bear_c3"] = pattern == -2

    return triggers
```

### 10.5 plot_config

Uses `candle_color_column: "sw123_candle_color"` — the app renders candles with custom colors when this column has non-empty values. No lines to plot (display_type="hidden").

### 10.6 Verification Checklist

- [ ] `calculate_swing_123()` produces `sw123_pattern` and `sw123_candle_color` columns
- [ ] C2 detection: Bull C2 = lower low + close > prior close; Bear C2 = higher high + close < prior close
- [ ] C3 detection: Requires prior bar was C2, then close breaks prior high/low
- [ ] C3 priority over C2 in the same bar (matching Pine Script logic)
- [ ] `interpret_swing_123()` returns 5 mutually exclusive states
- [ ] `detect_swing_123_triggers()` returns 4 trigger Series with `sw123_` prefix
- [ ] Manifest passes `validate_manifest()` — no reserved name collisions
- [ ] Pine Script reference copied to `user_packs/swing_123/reference.pine`

---

## 11. New Pack: Strat Assistant

### 11.1 Pine Script Analysis

**Source:** `reference-indicators/strat_assistant.pine` (Pine v4, 455 lines)

This is the most complex indicator. It provides:

1. **Bar Pattern Classification (1/2/3):**
   - 1 (Inside): `high <= high[1] AND low >= low[1]` — contained within prior bar
   - 2-Up: `high > high[1] AND NOT (low < low[1])` — higher high, low held
   - 2-Down: `low < low[1] AND NOT (high > high[1])` — lower low, high held
   - 3 (Outside): `high > high[1] AND low < low[1]` — engulfing

2. **Strategy Combos** (multi-bar patterns looking back 2-4 bars):
   - Continuations: 222 bull/bear, 212 measured move
   - Reversals: 22, 212, 322, 32, 312, 122 (RevStrat), 2222 (Randy Jackson), 3 RevStrat

3. **Actionable Signals:**
   - Shooter: Upper wick ≥ 75% of bar range (bearish)
   - Hammer: Lower wick ≥ 75% of bar range (bullish)
   - Inside Bar: 1-bar pattern

4. **Full Time Frame Continuity (FTC):**
   - 8 timeframes: 15m, 30m, 1H, 4H, D, W, M, Q
   - Each TF classified: green (close ≥ open), red (close < open), inside, outside

**Scope decision for pack:** The FTC component requires multi-timeframe data (`request.security`), which our single-TF pack architecture doesn't support directly. We'll implement:
- Bar pattern classification (1/2/3) ✅
- Strategy combos (the core Strat patterns) ✅
- Actionable signals (Shooter/Hammer) ✅
- Candle coloring based on 1/2/3 type ✅
- FTC: **Deferred** — requires multi-TF data infrastructure not yet available

### 11.2 Designed Manifest

```json
{
  "$schema": "pack_spec_v1",
  "slug": "strat_assistant",
  "name": "Strat Assistant",
  "category": "Trend",
  "description": "The Strat bar pattern classification (1-2-3), strategy combo detection, and actionable signals (Shooter/Hammer)",
  "author": "user",
  "version": "1.0.0",
  "created_at": "2026-02-19T00:00:00Z",
  "pack_type": "tf_confluence",
  "display_type": "hidden",
  "interpreters": ["STRAT_ASSISTANT"],
  "trigger_prefix": "strat",
  "parameters_schema": {
    "wick_pct": {
      "type": "float", "default": 0.75, "min": 0.5, "max": 0.95,
      "label": "Action Wick Percentage"
    }
  },
  "plot_schema": {
    "inside_color": {"type": "color", "default": "#F6BE00", "label": "Inside Bar Color"},
    "two_up_color": {"type": "color", "default": "#22c55e", "label": "2-Up Color"},
    "two_down_color": {"type": "color", "default": "#ef4444", "label": "2-Down Color"},
    "outside_color": {"type": "color", "default": "#d946ef", "label": "Outside Bar Color"}
  },
  "plot_config": {
    "candle_color_column": "strat_candle_color"
  },
  "outputs": [
    "INSIDE",
    "TWO_UP",
    "TWO_DOWN",
    "OUTSIDE"
  ],
  "output_descriptions": {
    "INSIDE": "1-Bar: High ≤ prior high AND low ≥ prior low (consolidation/coiling)",
    "TWO_UP": "2-Up Bar: Higher high, low held (bullish expansion)",
    "TWO_DOWN": "2-Down Bar: Lower low, high held (bearish expansion)",
    "OUTSIDE": "3-Bar: Higher high AND lower low (outside/engulfing bar)"
  },
  "triggers": [
    {"base": "bull_c2", "name": "Bullish 2-Up", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
    {"base": "bear_c2", "name": "Bearish 2-Down", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
    {"base": "outside_bar", "name": "Outside Bar (3)", "direction": "BOTH", "type": "ENTRY", "execution": "bar_close"},
    {"base": "inside_bar", "name": "Inside Bar (1)", "direction": "BOTH", "type": "ENTRY", "execution": "bar_close"},
    {"base": "shooter", "name": "Shooter Signal", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
    {"base": "hammer", "name": "Hammer Signal", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"}
  ],
  "indicator_columns": ["strat_bar_type", "strat_combo", "strat_actionable", "strat_candle_color"],
  "column_color_map": {},
  "indicator_function": "calculate_strat_assistant",
  "interpreter_function": "interpret_strat_assistant",
  "trigger_function": "detect_strat_assistant_triggers",
  "requires_indicators": ["strat_bar_type"]
}
```

### 11.3 Designed indicator.py

```python
import pandas as pd
import numpy as np


def calculate_strat_assistant(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate Strat bar patterns, strategy combos, and actionable signals.

    Bar Types (comparing current bar to prior):
        1 = Inside bar (high <= prior high AND low >= prior low)
        2 = Two-Up (high > prior high, low NOT lower)
       -2 = Two-Down (low < prior low, high NOT higher)
        3 = Outside bar (high > prior high AND low < prior low)

    Strategy Combos (multi-bar patterns):
        Continuations: 222_BULL, 222_BEAR, 212_MM_BULL, 212_MM_BEAR
        Reversals: 22_REV_BULL, 22_REV_BEAR, 212_REV_BULL, 212_REV_BEAR,
                   322_REV_BULL, 322_REV_BEAR, 32_REV_BULL, 32_REV_BEAR,
                   312_REV_BULL, 312_REV_BEAR, 122_REVSTRAT_BULL, 122_REVSTRAT_BEAR,
                   3_REVSTRAT_BULL, 3_REVSTRAT_BEAR

    Actionable Signals:
        SHOOTER = upper wick >= wick_pct of bar range (bearish)
        HAMMER = lower wick >= wick_pct of bar range (bullish)
        INSIDE = inside bar pattern
    """
    wick_pct = params.get("wick_pct", 0.75)

    result = df.copy()
    high = result["high"].values
    low = result["low"].values
    close = result["close"].values
    opn = result["open"].values
    n = len(close)

    # Colors
    inside_color = "#F6BE00"
    two_up_color = "#22c55e"
    two_down_color = "#ef4444"
    outside_color = "#d946ef"

    # Bar type classification
    bar_type = np.zeros(n, dtype=int)
    colors = [""] * n

    for i in range(1, n):
        is_inside = high[i] <= high[i - 1] and low[i] >= low[i - 1]
        is_outside = high[i] > high[i - 1] and low[i] < low[i - 1]
        is_two_up = high[i] > high[i - 1] and not (low[i] < low[i - 1])
        is_two_down = low[i] < low[i - 1] and not (high[i] > high[i - 1])

        if is_inside:
            bar_type[i] = 1
            colors[i] = inside_color
        elif is_outside:
            bar_type[i] = 3
            colors[i] = outside_color
        elif is_two_up:
            bar_type[i] = 2
            colors[i] = two_up_color
        elif is_two_down:
            bar_type[i] = -2
            colors[i] = two_down_color

    # Strategy combo detection (look back 2-4 bars)
    combos = [""] * n

    for i in range(3, n):
        b0 = bar_type[i]      # current bar
        b1 = bar_type[i - 1]  # 1 bar back
        b2 = bar_type[i - 2]  # 2 bars back
        b3 = bar_type[i - 3] if i >= 4 else 0  # 3 bars back

        # Continuations
        if b2 == -2 and b1 == -2 and b0 == -2:
            combos[i] = "222_BEAR"
        elif b2 == 2 and b1 == 2 and b0 == 2:
            combos[i] = "222_BULL"
        elif b2 == -2 and b1 == 1 and b0 == -2:
            combos[i] = "212_MM_BEAR"
        elif b2 == 2 and b1 == 1 and b0 == 2:
            combos[i] = "212_MM_BULL"

        # Reversals (check most specific first)
        elif i >= 4 and b3 == -2 and b2 == -2 and b1 == 2 and b0 == -2:
            combos[i] = "2222_BEAR"
        elif i >= 4 and b3 == 2 and b2 == 2 and b1 == -2 and b0 == 2:
            combos[i] = "2222_BULL"
        elif b2 == 1 and b1 == 2 and b0 == -2:
            combos[i] = "122_REVSTRAT_BEAR"
        elif b2 == 1 and b1 == -2 and b0 == 2:
            combos[i] = "122_REVSTRAT_BULL"
        elif b2 == 3 and b1 == 2 and b0 == -2:
            combos[i] = "322_REV_BEAR"
        elif b2 == 3 and b1 == -2 and b0 == 2:
            combos[i] = "322_REV_BULL"
        elif b2 == 3 and b1 == 1 and b0 == -2:
            combos[i] = "312_REV_BEAR"
        elif b2 == 3 and b1 == 1 and b0 == 2:
            combos[i] = "312_REV_BULL"
        elif b1 == 3 and b0 == -2:
            combos[i] = "32_REV_BEAR"
        elif b1 == 3 and b0 == 2:
            combos[i] = "32_REV_BULL"
        elif b2 == 2 and b1 == 1 and b0 == -2:
            combos[i] = "212_REV_BEAR"
        elif b2 == -2 and b1 == 1 and b0 == 2:
            combos[i] = "212_REV_BULL"
        elif b1 == 2 and b0 == -2 and b2 != 1 and b2 != -2 and b2 != 3:
            combos[i] = "22_REV_BEAR"
        elif b1 == -2 and b0 == 2 and b2 != 1 and b2 != 2 and b2 != 3:
            combos[i] = "22_REV_BULL"

        # 3-bar RevStrat (outside bar closing beyond prior range)
        if b0 == 3:
            if close[i] < low[i - 1]:
                combos[i] = "3_REVSTRAT_BEAR"
            elif close[i] > high[i - 1]:
                combos[i] = "3_REVSTRAT_BULL"

    # Actionable signals
    actionable = [""] * n
    for i in range(1, n):
        bar_range = high[i] - low[i]
        if bar_range <= 0:
            continue

        wick_height = bar_range * wick_pct
        shooter_top = high[i] - wick_height
        hammer_bottom = low[i] + wick_height

        is_inside = bar_type[i] == 1

        if is_inside:
            actionable[i] = "INSIDE"
        elif opn[i] < shooter_top and close[i] < shooter_top:
            actionable[i] = "SHOOTER"
        elif opn[i] > hammer_bottom and close[i] > hammer_bottom:
            actionable[i] = "HAMMER"

    result["strat_bar_type"] = bar_type
    result["strat_combo"] = combos
    result["strat_actionable"] = actionable
    result["strat_candle_color"] = colors

    return result
```

### 11.4 Designed interpreter.py

```python
import pandas as pd
import numpy as np


def interpret_strat_assistant(df: pd.DataFrame, **params) -> pd.Series:
    """
    Classify each bar's Strat pattern type.

    States (based on bar type):
        INSIDE: 1-bar (consolidation)
        TWO_UP: 2-up bar (bullish expansion)
        TWO_DOWN: 2-down bar (bearish expansion)
        OUTSIDE: 3-bar (outside/engulfing)
    """
    type_map = {
        1: "INSIDE",
        2: "TWO_UP",
        -2: "TWO_DOWN",
        3: "OUTSIDE",
    }

    def classify(row):
        bt = row.get("strat_bar_type", 0)
        if pd.isna(bt) or bt == 0:
            return None
        return type_map.get(int(bt), None)

    return df.apply(classify, axis=1)


def detect_strat_assistant_triggers(df: pd.DataFrame, **params) -> dict:
    """
    Detect Strat pattern triggers.

    Triggers:
        strat_bull_c2: 2-Up bar detected
        strat_bear_c2: 2-Down bar detected
        strat_outside_bar: 3-bar (outside) detected
        strat_inside_bar: 1-bar (inside) detected
        strat_shooter: Shooter signal
        strat_hammer: Hammer signal
    """
    prefix = "strat"
    triggers = {}

    bar_type = df["strat_bar_type"]
    actionable = df["strat_actionable"]

    triggers[f"{prefix}_bull_c2"] = bar_type == 2
    triggers[f"{prefix}_bear_c2"] = bar_type == -2
    triggers[f"{prefix}_outside_bar"] = bar_type == 3
    triggers[f"{prefix}_inside_bar"] = bar_type == 1
    triggers[f"{prefix}_shooter"] = actionable == "SHOOTER"
    triggers[f"{prefix}_hammer"] = actionable == "HAMMER"

    return triggers
```

### 11.5 plot_config

Uses `candle_color_column: "strat_candle_color"` for bar coloring by pattern type. No overlay lines (display_type="hidden").

### 11.6 Verification Checklist

- [ ] `calculate_strat_assistant()` produces 4 columns: `strat_bar_type`, `strat_combo`, `strat_actionable`, `strat_candle_color`
- [ ] Bar type detection matches Pine: 1=inside, 2=two-up, -2=two-down, 3=outside
- [ ] Strategy combos detect all 16+ patterns from reference
- [ ] Shooter/Hammer wick percentage threshold (default 75%) matches Pine `_actionWickPercentage`
- [ ] Inside bar overrides Shooter/Hammer (matching Pine priority)
- [ ] `interpret_strat_assistant()` returns 4 mutually exclusive states (bar always has a type once bars exist)
- [ ] `detect_strat_assistant_triggers()` returns 6 trigger Series with `strat_` prefix
- [ ] Manifest passes `validate_manifest()` — no reserved name collisions
- [ ] Pine Script reference copied to `user_packs/strat_assistant/reference.pine`

---

## 12. Pack Builder Improvements

### 12.1 Prompt Refinements

Based on the audit findings, the following improvements to the Pack Builder prompt are recommended:

1. **Add `display_type: "hidden"` guidance for candle-coloring packs:**
   Currently the prompt describes overlay/oscillator/hidden but doesn't mention candle_color_column. Add guidance:
   > If the indicator's primary output is candle coloring (bar pattern classification, signal highlighting), use `display_type: "hidden"` and add `"candle_color_column": "your_color_col"` to `plot_config`. The color column should contain hex color strings (e.g., "#22c55e") or empty strings for default candle color.

2. **Add Wilder/RMA smoothing note to Pine Script translation table:**
   Pine v4 `atr()` uses Wilder smoothing (alpha = 1/N), not standard EMA (alpha = 2/(N+1)). Add:
   | `atr(N)` (Pine v4) | Wilder RMA: `alpha = 1/N`, loop: `atr[i] = atr[i-1] + (1/N) * (tr[i] - atr[i-1])` |

3. **Add `plot_config` documentation to the context document:**
   The current context document mentions `plot_config` as an optional manifest field but doesn't document its sub-fields (`band_fills`, `reference_lines`, `line_styles`, `candle_color_column`). Add a section with examples.

### 12.2 Validation Gaps

1. **candle_color_column validation:** `pack_spec.py` validates `candle_color_column` if present, but the Pack Builder context document doesn't mention it. LLMs generating candle-coloring packs won't know to include it.

2. **display_type defaulting:** If an LLM omits `display_type`, it defaults to None which passes validation. Should default to `"overlay"` in the prompt or enforce in validation.

3. **Trigger prefix uniqueness:** The validation checks against `BUILTIN_TRIGGER_PREFIXES` but doesn't check against other installed user packs. Two user packs could have the same prefix.

### 12.3 Context Document Updates

Update `pack_builder_context.md` with:

1. Add `plot_config` schema documentation after the manifest rules section:
   ```
   ### plot_config (Optional)

   - `band_fills`: Array of {upper_column, lower_column, fill_color_key} for shaded bands
   - `reference_lines`: Array of {value, color, label} for horizontal lines (e.g., RSI 70/30)
   - `line_styles`: Dict mapping column -> style int (0=Solid, 1=Dotted, 2=Dashed, 3=LargeDashed, 4=SparseDotted)
   - `candle_color_column`: Column name containing hex color strings for bar coloring
   ```

2. Add to reserved names:
   - Trigger prefixes: add `bb`, `src`, `st`, `sw123`, `strat` (for new packs)
   - Interpreter keys: add `BOLLINGER_BANDS`, `SR_CHANNELS`, `SUPERTREND`, `SWING_123`, `STRAT_ASSISTANT`
   - Indicator columns: add `bb_*`, `src_*`, `st_*`, `sw123_*`, `strat_*`

3. Add Wilder smoothing note to Pine Script Translation Reference.

---

## 13. Execution Order & Verification

### 13.1 Step-by-Step Execution Order

| Step | Action | Files Modified |
|------|--------|---------------|
| 1 | Fix EMA Stack `LMS` dead code bug | `src/interpreters.py` |
| 2 | Fix VWAP session-aware reset | `src/indicators.py` |
| 3 | Implement UT Bot indicator + interpreter + triggers | `src/indicators.py`, `src/interpreters.py` |
| 4 | Register UT Bot in dispatch registries | `src/indicators.py`, `src/interpreters.py` |
| 5 | Update UT Bot code tab mapping | `src/app.py` |
| 6 | Create SuperTrend user pack | `user_packs/supertrend/` (3 files + reference.pine) |
| 7 | Create Swing 123 user pack | `user_packs/swing_123/` (3 files + reference.pine) |
| 8 | Create Strat Assistant user pack | `user_packs/strat_assistant/` (3 files + reference.pine) |
| 9 | Update Pack Builder context document | `src/pack_builder_context.md` |
| 10 | Update pack_spec.py reserved names | `src/pack_spec.py` |
| 11 | Syntax-check all modified/new Python files | All `.py` files |
| 12 | Update PRD to mark 17D complete | `docs/RoR_Trader_PRD.md` |
| 13 | Commit | — |

### 13.2 Final Verification Checklist

- [ ] EMA Stack: `LMS` state now reachable (l > m > s > p condition)
- [ ] VWAP: Session-aware reset in fallback calculation path
- [ ] UT Bot: `calculate_utbot()`, `interpret_utbot()`, `detect_utbot_triggers()` all registered and functional
- [ ] UT Bot: Code tab shows non-empty function lists
- [ ] SuperTrend: Pack loads without validation errors
- [ ] Swing 123: Pack loads without validation errors, candle coloring works
- [ ] Strat Assistant: Pack loads without validation errors, bar classification correct
- [ ] Pack Builder context: plot_config documented, reserved names updated
- [ ] All Python files pass `py_compile.compile()` check
- [ ] No circular imports or runtime errors on app startup
