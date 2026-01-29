# Side Module Roadmap

This document captures the planned structure for all side modules (3-10) to be built for the KevBot Toolkit. Each module follows the hybrid architecture pattern established in v1.1.

*Created: January 2026*

---

## Overview: Planned Modules

| Module # | Name | Status | Focus |
|----------|------|--------|-------|
| 1 | EMA Stack | ✅ Complete | EMA ordering (S/M/L) |
| 2 | Placeholder | ✅ Complete | Testing template |
| 3 | Simple MACD Line | ✅ Complete | MACD vs Signal (no zero line) |
| 4 | MACD Line | 📋 Planned | MACD vs Signal + zero line context |
| 5 | MACD Histogram | 📋 Planned | Histogram momentum |
| 6 | MACD Divergence | 📋 Planned | Price vs MACD divergence |
| 7 | UT Bot | 📋 Planned | ATR trailing stop signals |
| 8 | VWAP | 📋 Planned | VWAP + standard deviation bands |
| 9 | RVOL | 📋 Planned | Relative volume analysis |
| 10 | SR Channel | 📋 Planned | Support/Resistance zones |
| -- | Swing 123 | 📋 Planned | Candle pattern recognition |

---

## Module 4: MACD Line (with Zero Context)

**Library Name:** `KevBot_TF_MACD_Line`
**Focus:** MACD vs Signal relationship WITH zero line context

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A | 12 | Fast EMA |
| B | 26 | Slow EMA |
| C | 9 | Signal smoothing |

### Conditions (mutually exclusive)

| Cond | Label | Description |
|------|-------|-------------|
| A | M>S+ | MACD > Signal AND MACD > 0 (strong bull) |
| B | M>S- | MACD > Signal AND MACD < 0 (recovering from bear) |
| C | M<S- | MACD < Signal AND MACD < 0 (strong bear) |
| D | M<S+ | MACD < Signal AND MACD > 0 (weakening from bull) |

### Triggers

| Trig | Description |
|------|-------------|
| A | Bullish Cross (MACD > Signal) |
| B | Bearish Cross (MACD < Signal) |
| C | Zero Cross Up (MACD crosses above 0) |
| D | Zero Cross Down (MACD crosses below 0) |

### Use Case
For traders who want to see MACD crossovers in context of the zero line. M>S+ is the strongest bullish state (MACD above signal AND above zero), while M>S- shows recovery (crossed bullish but still in negative territory).

---

## Module 5: MACD Histogram

**Library Name:** `KevBot_TF_MACD_Histogram`
**Focus:** Histogram momentum direction

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A | 12 | Fast EMA |
| B | 26 | Slow EMA |
| C | 9 | Signal smoothing |

### Conditions (mutually exclusive)

| Cond | Label | Description |
|------|-------|-------------|
| A | H+↑ | Histogram positive AND rising (accelerating bull) |
| B | H+↓ | Histogram positive AND falling (decelerating bull) |
| C | H-↓ | Histogram negative AND falling (accelerating bear) |
| D | H-↑ | Histogram negative AND rising (decelerating bear) |

### Triggers

| Trig | Description |
|------|-------------|
| A | Flip Bullish (histogram crosses from negative to positive) |
| B | Flip Bearish (histogram crosses from positive to negative) |
| C | Momentum Shift Up (histogram starts rising after falling) |
| D | Momentum Shift Down (histogram starts falling after rising) |

### Use Case
Early momentum detection. H+↓ often signals a coming bearish cross before it happens. H-↑ often signals a coming bullish cross. Useful for timing entries/exits.

---

## Module 6: MACD Divergence

**Library Name:** `KevBot_TF_MACD_Divergence`
**Focus:** Detecting price vs MACD divergences

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A | 12 | Fast EMA |
| B | 26 | Slow EMA |
| C | 9 | Signal smoothing |
| D | 5 | Pivot lookback (for swing detection) |
| E | 30 | Max bars to look back for divergence |

### Conditions (mutually exclusive)

| Cond | Label | Description |
|------|-------|-------------|
| A | DIV+ | Regular Bullish Divergence (price lower low, MACD higher low) |
| B | DIV- | Regular Bearish Divergence (price higher high, MACD lower high) |
| C | hDIV+ | Hidden Bullish Divergence (price higher low, MACD lower low) |
| D | hDIV- | Hidden Bearish Divergence (price lower high, MACD higher high) |
| E | NONE | No divergence detected |

### Triggers

| Trig | Description |
|------|-------------|
| A | Bullish Divergence Confirmed (swing low completes the pattern) |
| B | Bearish Divergence Confirmed (swing high completes the pattern) |
| C | Hidden Bullish Confirmed |
| D | Hidden Bearish Confirmed |

### Divergence Types Explained

| Type | Price Action | MACD Action | Signal |
|------|--------------|-------------|--------|
| Regular Bullish | Lower Low | Higher Low | Reversal up likely |
| Regular Bearish | Higher High | Lower High | Reversal down likely |
| Hidden Bullish | Higher Low | Lower Low | Uptrend continuation |
| Hidden Bearish | Lower High | Higher High | Downtrend continuation |

### Use Case
Powerful reversal and continuation signals. Regular divergences warn of potential trend reversals. Hidden divergences confirm trend continuation despite temporary MACD weakness.

---

## Module 7: UT Bot

**Library Name:** `KevBot_TF_UTBot`
**Focus:** ATR-based trailing stop with direction signals

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A | 1 | Key Value (ATR multiplier - sensitivity) |
| B | 10 | ATR Period |
| C | 0 | Use Heikin Ashi (0=No, 1=Yes) |

### Conditions (mutually exclusive)

| Cond | Label | Description |
|------|-------|-------------|
| A | UT+ | Price above ATR trailing stop (bullish bias) |
| B | UT- | Price below ATR trailing stop (bearish bias) |

### Triggers

| Trig | Description |
|------|-------------|
| A | Buy Signal (price crosses above trailing stop) |
| B | Sell Signal (price crosses below trailing stop) |

### Implementation Notes

The UT Bot uses an ATR-based trailing stop that:
1. Follows price with a buffer of `Key Value × ATR`
2. Only moves in the direction of the trend (ratchets)
3. Signals when price crosses the trailing stop

```
Trailing Stop Logic:
- If bullish: stop = max(prev_stop, close - nLoss)
- If bearish: stop = min(prev_stop, close + nLoss)
- Where nLoss = Key Value × ATR(period)
```

### Use Case
Excellent for capturing trend reversals and riding trends. The trailing stop naturally adapts to volatility. Lower Key Value = more sensitive (more signals, more whipsaws). Higher Key Value = less sensitive (fewer signals, larger swings).

**Recommended as primary trigger module** - pairs well with other modules for confluence.

---

## Module 8: VWAP

**Library Name:** `KevBot_TF_VWAP`
**Focus:** VWAP with standard deviation bands

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A | 1.0 | Band 1 Multiplier (standard deviations) |
| B | 2.0 | Band 2 Multiplier |
| C | 3.0 | Band 3 Multiplier |

### Conditions (mutually exclusive)

| Cond | Label | Description |
|------|-------|-------------|
| A | >+2σ | Price above VWAP + 2 SD (extended high) |
| B | >+1σ | Price between +1σ and +2σ |
| C | >V | Price between VWAP and +1σ |
| D | @V | Price near VWAP (within ±0.5σ) |
| E | <V | Price between VWAP and -1σ |
| F | <-1σ | Price between -1σ and -2σ |
| G | <-2σ | Price below VWAP - 2 SD (extended low) |

### Triggers

| Trig | Description |
|------|-------------|
| A | Cross Above VWAP |
| B | Cross Below VWAP |
| C | Enter Upper Extreme (cross into +2σ zone) |
| D | Enter Lower Extreme (cross into -2σ zone) |
| E | Return to VWAP (price returns to @V zone from extremes) |

### VWAP Considerations for Multi-TF

VWAP is typically session-anchored. For multi-TF analysis:
- Each TF calculates VWAP based on session anchor
- Higher TFs may show different VWAP values due to bar aggregation
- Consider using daily session VWAP consistently across all TFs

### Use Case
Day trading staple. Price tends to revert to VWAP. Extended moves to ±2σ often signal exhaustion. VWAP crossovers can confirm trend direction.

---

## Module 9: RVOL (Relative Volume)

**Library Name:** `KevBot_TF_RVOL`
**Focus:** Volume relative to historical average

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A | 20 | Lookback period for average volume |
| B | 1.5 | High threshold (RVOL considered elevated) |
| C | 2.0 | Very high threshold |
| D | 3.0 | Extreme threshold |

### Conditions (mutually exclusive)

| Cond | Label | Description |
|------|-------|-------------|
| A | RV! | Extreme volume (RVOL > Extreme threshold) |
| B | RV++ | Very high volume (RVOL > Very High threshold) |
| C | RV+ | Elevated volume (RVOL > High threshold) |
| D | RV= | Normal volume (RVOL between 0.5 and High) |
| E | RV- | Low volume (RVOL < 0.5) |

### Triggers

| Trig | Description |
|------|-------------|
| A | Volume Spike (RVOL crosses above High threshold) |
| B | Volume Extreme (RVOL crosses above Extreme threshold) |
| C | Volume Fade (RVOL drops below 1.0) |

### Label Display
Shows numeric RVOL value: "1.8x", "2.5x", "0.7x", etc.

### Use Case
Volume confirms price moves. High RVOL on breakouts = more conviction. Low RVOL on moves = suspect, may reverse. Useful as a filter - only take trades when volume supports the move.

---

## Module 10: SR Channel

**Library Name:** `KevBot_TF_SRChannel`
**Focus:** Dynamic Support/Resistance zone detection

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A | 10 | Pivot period (left/right bars) |
| B | 5 | Max channel width % |
| C | 1 | Minimum strength (pivot touches) |
| D | 290 | Lookback period for pivots |

### Conditions (mutually exclusive)

| Cond | Label | Description |
|------|-------|-------------|
| A | SUP | Price inside a support zone |
| B | RES | Price inside a resistance zone |
| C | >SR | Price above all identified S/R zones |
| D | <SR | Price below all identified S/R zones |
| E | MID | Price between zones (not in any zone) |

### Triggers

| Trig | Description |
|------|-------------|
| A | Support Break (price breaks below support zone) |
| B | Resistance Break (price breaks above resistance zone) |
| C | Enter Support Zone |
| D | Enter Resistance Zone |

### Implementation Notes

The SR Channel algorithm:
1. Identifies pivot highs/lows over the lookback period
2. Clusters nearby pivots into zones
3. Ranks zones by strength (number of touches)
4. Tracks price position relative to zones

### Use Case
Trade bounces off support, breakouts through resistance. Avoid buying into resistance, selling into support. Zone detection helps identify key price levels automatically.

---

## Bonus Module: Swing 123

**Library Name:** `KevBot_TF_Swing123`
**Focus:** Candle pattern recognition (1-2-3 reversal pattern)

### Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| A-F | - | Pattern is fixed, no parameters needed |

### Conditions

| Cond | Label | Description |
|------|-------|-------------|
| A | BC2 | Bullish Candle 2 active (lower low, close > prev close) |
| B | BC3 | Bullish Candle 3 active (close > prev high) |
| C | XC2 | Bearish Candle 2 active (higher high, close < prev close) |
| D | XC3 | Bearish Candle 3 active (close < prev low) |
| E | B↑ | Recent bullish setup (C2 or C3 within last 3 bars) |
| F | X↓ | Recent bearish setup (C2 or C3 within last 3 bars) |

### Triggers

| Trig | Description |
|------|-------------|
| A | Bullish C2 formed |
| B | Bullish C3 confirmation |
| C | Bearish C2 formed |
| D | Bearish C3 confirmation |

### Pattern Logic

**Bullish 1-2-3:**
- Candle 2: Makes lower low than prior candle BUT closes above prior close
- Candle 3: Closes above Candle 2's high (confirmation)

**Bearish 1-2-3:**
- Candle 2: Makes higher high than prior candle BUT closes below prior close
- Candle 3: Closes below Candle 2's low (confirmation)

### Use Case
Identifies potential reversal points. The pattern shows when price is "rejected" - it tried to continue the trend (new high/low) but failed (closed opposite). Candle 3 confirms the reversal.

---

## Integration Priority

Recommended order for building and integrating:

1. ✅ **Simple MACD Line** - Done
2. **UT Bot** - Your preferred trigger, simple to implement
3. **MACD Line (with zero)** - Builds on Simple MACD
4. **MACD Histogram** - Completes MACD family
5. **VWAP** - Common day trading tool
6. **RVOL** - Volume confirmation
7. **Swing 123** - Pattern recognition
8. **SR Channel** - Most complex, save for last
9. **MACD Divergence** - Requires swing detection, complex

---

## Architecture Reminder

All modules follow the hybrid pattern:

```pinescript
// TOOLKIT handles:
// 1. Input declarations (optimizer compatible)
// 2. request.security() calls for HTF data
// 3. Calling library's buildOutput() with fetched data

// LIBRARY handles:
// 1. Export TFModuleOutput type
// 2. Export buildOutput() function
// 3. All condition/trigger logic
```

This separation ensures:
- Parameters are optimizable by third-party tools
- Processing logic is modular and reusable
- Easy to add new modules without modifying core toolkit structure

---

*Last Updated: January 2026*
