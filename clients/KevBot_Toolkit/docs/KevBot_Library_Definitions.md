# KevBot Toolkit - Library Definitions

This document provides user-facing definitions for all Side Module libraries available in the KevBot Toolkit. Each library defines its own set of parameters, conditions, and triggers.

---

## Table of Contents

1. [EMA Stack (S/M/L)](#ema-stack-sml)
2. [Simple MACD Line](#simple-macd-line)
3. [MACD Line (with Zero Context)](#macd-line-with-zero-context)
4. [MACD Histogram](#macd-histogram)
5. [MACD Divergence](#macd-divergence)
6. [KevBot_TF_Placeholder](#kevbot_tf_placeholder)
7. [Adding New Libraries](#adding-new-libraries)

---

## EMA Stack (S/M/L)

**Library Name:** `KevBot_TF_EMA_Stack`
**Version:** 7
**Architecture:** Hybrid (Toolkit fetches data, library processes)

### Overview

The EMA Stack library analyzes the relative positioning of three Exponential Moving Averages (Short, Medium, Long) across multiple timeframes. It identifies trend alignment patterns and crossover events.

### Parameters

| Parameter | Name | Default | Description |
|-----------|------|---------|-------------|
| **Param A** | Short EMA Length | 10 | Period for the Short (fastest) EMA |
| **Param B** | Medium EMA Length | 20 | Period for the Medium EMA |
| **Param C** | Long EMA Length | 50 | Period for the Long (slowest) EMA |
| Param D | (Reserved) | - | Not used |
| Param E | (Reserved) | - | Not used |
| Param F | (Reserved) | - | Not used |

### Conditions (A-J)

Conditions evaluate the EMA ordering on each timeframe. The label shown in the Side Table indicates the current EMA stack order.

| Condition | Name | Description | Label |
|-----------|------|-------------|-------|
| **Cond A** | Bull Stack | Short > Medium > Long (fully bullish alignment) | SML |
| **Cond B** | Bear Stack | Long > Medium > Short (fully bearish alignment) | LMS |
| **Cond C** | SLM | Short > Long > Medium (partial bullish) | SLM |
| **Cond D** | MSL | Medium > Short > Long (transition state) | MSL |
| **Cond E** | MLS | Medium > Long > Short (transition state) | MLS |
| **Cond F** | LSM | Long > Short > Medium (partial bearish) | LSM |
| Cond G-J | (Reserved) | Not used | - |

### Triggers (A-J)

Triggers fire on crossover events on the chart timeframe.

| Trigger | Name | Description | Use Case |
|---------|------|-------------|----------|
| **Trig A** | S > M | Short EMA crosses above Medium EMA | Long Entry signal |
| **Trig B** | S < M | Short EMA crosses below Medium EMA | Short Entry / Long Exit signal |
| **Trig C** | S > L | Short EMA crosses above Long EMA | Trend reversal confirmation |
| **Trig D** | S < L | Short EMA crosses below Long EMA | Trend reversal confirmation |
| **Trig E** | M > L | Medium EMA crosses above Long EMA | Strong bullish trend signal |
| **Trig F** | M < L | Medium EMA crosses below Long EMA | Strong bearish trend signal |
| Trig G-J | (Reserved) | Not used | - |

### Recommended Configurations

#### Scalping (Fast Signals)
- Param A: 5
- Param B: 13
- Param C: 34
- Long Entry Trigger: Trig A (S > M)
- Long Entry Confluence: Cond A (Bull Stack)

#### Swing Trading (Standard)
- Param A: 10
- Param B: 20
- Param C: 50
- Long Entry Trigger: Trig A (S > M)
- Long Entry Confluence: Cond A (Bull Stack)

#### Position Trading (Slow Signals)
- Param A: 20
- Param B: 50
- Param C: 200
- Long Entry Trigger: Trig E (M > L)
- Long Entry Confluence: Cond A (Bull Stack)

---

## Simple MACD Line

**Library Name:** `KevBot_TF_MACD_Simple`
**Version:** 1
**Architecture:** Hybrid (Toolkit fetches data, library processes)

### Overview

The Simple MACD Line library analyzes the pure relationship between the MACD line and Signal line, without considering the zero line. It shows whether MACD is above or below Signal, and whether the gap between them is widening (strengthening) or narrowing (weakening).

This is ideal for strategies that focus purely on MACD/Signal crossovers without caring about whether MACD is above or below zero.

### Parameters

| Parameter | Name | Default | Description |
|-----------|------|---------|-------------|
| **Param A** | Fast EMA Length | 12 | Period for the Fast EMA |
| **Param B** | Slow EMA Length | 26 | Period for the Slow EMA |
| **Param C** | Signal Smoothing | 9 | Period for the Signal line EMA |
| Param D | (Reserved) | - | Not used |
| Param E | (Reserved) | - | Not used |
| Param F | (Reserved) | - | Not used |

### Conditions (A-J)

Conditions evaluate the MACD vs Signal relationship on each timeframe. The label shown indicates both the relationship and momentum direction.

| Condition | Name | Description | Label |
|-----------|------|-------------|-------|
| **Cond A** | Bull Widening | MACD > Signal, gap widening (strengthening bullish) | M>S↑ |
| **Cond B** | Bull Narrowing | MACD > Signal, gap narrowing (weakening bullish) | M>S↓ |
| **Cond C** | Bear Widening | MACD < Signal, gap widening (strengthening bearish) | M<S↓ |
| **Cond D** | Bear Narrowing | MACD < Signal, gap narrowing (weakening bearish) | M<S↑ |
| Cond E-J | (Reserved) | Not used | - |

### Triggers (A-J)

Triggers fire on crossover events on the chart timeframe.

| Trigger | Name | Description | Use Case |
|---------|------|-------------|----------|
| **Trig A** | Bullish Cross | MACD crosses above Signal | Long Entry signal |
| **Trig B** | Bearish Cross | MACD crosses below Signal | Short Entry / Long Exit signal |
| Trig C-J | (Reserved) | Not used | - |

### Recommended Configurations

#### Standard MACD
- Param A: 12
- Param B: 26
- Param C: 9
- Long Entry Trigger: Trig A (Bullish Cross)
- Long Entry Confluence: Cond A (Bull Widening on higher TFs)

#### Fast MACD (More Signals)
- Param A: 8
- Param B: 17
- Param C: 9
- Long Entry Trigger: Trig A (Bullish Cross)
- Long Entry Confluence: Cond A or Cond B (any bullish state)

#### Slow MACD (Fewer False Signals)
- Param A: 19
- Param B: 39
- Param C: 9
- Long Entry Trigger: Trig A (Bullish Cross)
- Long Entry Confluence: Cond A (Bull Widening for confirmation)

### Interpretation Guide

| Side Table Shows | Meaning | Trading Implication |
|------------------|---------|---------------------|
| M>S↑ across TFs | Strong bullish momentum building | High confidence long entries |
| M>S↓ across TFs | Bullish but losing steam | Consider taking profits or tightening stops |
| M<S↓ across TFs | Strong bearish momentum building | High confidence short entries / avoid longs |
| M<S↑ across TFs | Bearish but recovering | Watch for potential reversal |
| Mixed signals | Choppy/transitional market | Reduce position size or wait for clarity |

---

## MACD Line (with Zero Context)

**Library Name:** `KevBot_TF_MACD_Line`
**Version:** 1
**Architecture:** Hybrid (Toolkit fetches data, library processes)

### Overview

The MACD Line library analyzes the MACD vs Signal relationship WITH zero line context. It shows whether MACD is above or below Signal, AND whether MACD itself is above or below zero. This provides additional context on trend strength - being above signal AND above zero is stronger than being above signal but below zero.

### Parameters

| Parameter | Name | Default | Description |
|-----------|------|---------|-------------|
| **Param A** | Fast EMA Length | 12 | Period for the Fast EMA |
| **Param B** | Slow EMA Length | 26 | Period for the Slow EMA |
| **Param C** | Signal Smoothing | 9 | Period for the Signal line EMA |
| Param D | (Reserved) | - | Not used |
| Param E | (Reserved) | - | Not used |
| Param F | (Reserved) | - | Not used |

### Conditions (A-J)

| Condition | Name | Description | Label |
|-----------|------|-------------|-------|
| **Cond A** | Strong Bull | MACD > Signal AND MACD > 0 | M>S+ |
| **Cond B** | Recovering | MACD > Signal AND MACD < 0 (crossed bullish but still negative) | M>S- |
| **Cond C** | Strong Bear | MACD < Signal AND MACD < 0 | M<S- |
| **Cond D** | Weakening | MACD < Signal AND MACD > 0 (crossed bearish but still positive) | M<S+ |
| Cond E-J | (Reserved) | Not used | - |

### Triggers (A-J)

| Trigger | Name | Description | Use Case |
|---------|------|-------------|----------|
| **Trig A** | Bullish Cross | MACD crosses above Signal | Long Entry signal |
| **Trig B** | Bearish Cross | MACD crosses below Signal | Short Entry / Long Exit signal |
| **Trig C** | Zero Cross Up | MACD crosses above 0 | Trend confirmation (bullish) |
| **Trig D** | Zero Cross Down | MACD crosses below 0 | Trend confirmation (bearish) |
| Trig E-J | (Reserved) | Not used | - |

### Interpretation Guide

| Side Table Shows | Meaning | Trading Implication |
|------------------|---------|---------------------|
| M>S+ across TFs | Strong bullish trend | Highest confidence long entries |
| M>S- across TFs | Bullish crossover but recovering from bearish | Good entry, but wait for zero cross for confirmation |
| M<S- across TFs | Strong bearish trend | Avoid longs, consider shorts |
| M<S+ across TFs | Bearish crossover but was in bullish territory | Possible profit taking, not ideal for new longs |

---

## MACD Histogram

**Library Name:** `KevBot_TF_MACD_Histogram`
**Version:** 1
**Architecture:** Hybrid (Toolkit fetches data, library processes)

### Overview

The MACD Histogram library analyzes the histogram (MACD - Signal) momentum direction. It shows whether the histogram is positive/negative AND whether it's rising/falling. This provides early warning of momentum shifts before actual crossovers occur.

### Parameters

| Parameter | Name | Default | Description |
|-----------|------|---------|-------------|
| **Param A** | Fast EMA Length | 12 | Period for the Fast EMA |
| **Param B** | Slow EMA Length | 26 | Period for the Slow EMA |
| **Param C** | Signal Smoothing | 9 | Period for the Signal line EMA |
| Param D | (Reserved) | - | Not used |
| Param E | (Reserved) | - | Not used |
| Param F | (Reserved) | - | Not used |

### Conditions (A-J)

| Condition | Name | Description | Label |
|-----------|------|-------------|-------|
| **Cond A** | Accel Bull | Histogram positive AND rising (accelerating bullish momentum) | H+↑ |
| **Cond B** | Decel Bull | Histogram positive AND falling (decelerating bullish momentum) | H+↓ |
| **Cond C** | Accel Bear | Histogram negative AND falling (accelerating bearish momentum) | H-↓ |
| **Cond D** | Decel Bear | Histogram negative AND rising (decelerating bearish momentum) | H-↑ |
| Cond E-J | (Reserved) | Not used | - |

### Triggers (A-J)

| Trigger | Name | Description | Use Case |
|---------|------|-------------|----------|
| **Trig A** | Flip Bullish | Histogram crosses from negative to positive | Bullish crossover confirmed |
| **Trig B** | Flip Bearish | Histogram crosses from positive to negative | Bearish crossover confirmed |
| **Trig C** | Momentum Shift Up | Histogram starts rising after falling | Early bullish warning |
| **Trig D** | Momentum Shift Down | Histogram starts falling after rising | Early bearish warning |
| Trig E-J | (Reserved) | Not used | - |

### Interpretation Guide

| Side Table Shows | Meaning | Trading Implication |
|------------------|---------|---------------------|
| H+↑ across TFs | Strong bullish momentum building | High confidence long entries |
| H+↓ across TFs | Still bullish but momentum fading | Warning: bearish cross may be coming |
| H-↓ across TFs | Strong bearish momentum building | Avoid longs, consider shorts |
| H-↑ across TFs | Still bearish but momentum fading | Warning: bullish cross may be coming |

### Early Warning Usage

The histogram is valuable for **anticipating** MACD crossovers:
- When you see H+↓ (positive but falling), expect a bearish cross soon
- When you see H-↑ (negative but rising), expect a bullish cross soon

This gives you time to prepare entries/exits before the actual signal fires.

---

## MACD Divergence

**Library Name:** `KevBot_TF_MACD_Divergence`
**Version:** 1
**Architecture:** Hybrid (Toolkit fetches data, library processes)

> ⚠️ **Important:** Due to PineScript limitations, MACD Divergence works on the **chart timeframe only**. The same divergence state is displayed across all TF columns in the side table.

### Overview

The MACD Divergence library detects divergences between price action and MACD indicator. A divergence occurs when price makes a new high/low but MACD doesn't confirm. This often signals a potential reversal or continuation.

### Parameters

| Parameter | Name | Default | Description |
|-----------|------|---------|-------------|
| **Param A** | Fast EMA Length | 12 | Period for the Fast EMA |
| **Param B** | Slow EMA Length | 26 | Period for the Slow EMA |
| **Param C** | Signal Smoothing | 9 | Period for the Signal line EMA |
| **Param D** | Pivot Lookback | 5 | Left/right bars for swing detection |
| **Param E** | Max Divergence Lookback | 30 | Maximum bars to look back for divergence |
| Param F | (Reserved) | - | Not used |

### Conditions (A-J)

| Condition | Name | Description | Label |
|-----------|------|-------------|-------|
| **Cond A** | Regular Bullish | Price lower low + MACD higher low (reversal UP likely) | DIV+ |
| **Cond B** | Regular Bearish | Price higher high + MACD lower high (reversal DOWN likely) | DIV- |
| **Cond C** | Hidden Bullish | Price higher low + MACD lower low (uptrend continuation) | hDV+ |
| **Cond D** | Hidden Bearish | Price lower high + MACD higher high (downtrend continuation) | hDV- |
| **Cond E** | No Divergence | No divergence pattern detected | NONE |
| Cond F-J | (Reserved) | Not used | - |

### Triggers (A-J)

| Trigger | Name | Description | Use Case |
|---------|------|-------------|----------|
| **Trig A** | Bullish Div Confirmed | Regular bullish divergence completed | Long Entry signal |
| **Trig B** | Bearish Div Confirmed | Regular bearish divergence completed | Short Entry / Long Exit |
| **Trig C** | Hidden Bull Confirmed | Hidden bullish divergence completed | Add to longs in uptrend |
| **Trig D** | Hidden Bear Confirmed | Hidden bearish divergence completed | Add to shorts in downtrend |
| Trig E-J | (Reserved) | Not used | - |

### Divergence Types Explained

| Type | Price Action | MACD Action | Signal |
|------|--------------|-------------|--------|
| **Regular Bullish** | Lower Low | Higher Low | Reversal up likely |
| **Regular Bearish** | Higher High | Lower High | Reversal down likely |
| **Hidden Bullish** | Higher Low | Lower Low | Uptrend continuation |
| **Hidden Bearish** | Lower High | Higher High | Downtrend continuation |

### Usage Tips

- **Regular divergences** warn of potential trend reversals - use for counter-trend entries
- **Hidden divergences** confirm trend continuation - use to add to existing positions
- Divergences work best when combined with other confluence factors (support/resistance, EMA stack, etc.)
- The pivot lookback (Param D) affects sensitivity: smaller = more signals, larger = fewer but stronger signals

---

## KevBot_TF_Placeholder

**Library Name:** `KevBot_TF_Placeholder`
**Version:** 5
**Architecture:** Legacy (Library handles all logic internally)

### Overview

The Placeholder library is a development template demonstrating the Side Module interface. It provides static/dummy outputs for testing purposes.

### Parameters

| Parameter | Name | Default | Description |
|-----------|------|---------|-------------|
| Param A-F | (Unused) | 0 | Placeholder parameters for testing |

### Conditions (A-J)

| Condition | Description |
|-----------|-------------|
| Cond A | Returns true (always) |
| Cond B | Returns false (always) |
| Cond C-J | Returns false (always) |

### Triggers (A-J)

| Trigger | Description |
|---------|-------------|
| Trig A-J | Returns false (no triggers fire) |

---

## Adding New Libraries

To add a new Side Module library to the toolkit:

### 1. Library Requirements

Your library must export a `TFModuleOutput` type with these fields:

```pinescript
export type TFModuleOutput
    // Per-TF labels (6 timeframes)
    string tf1_label, tf2_label, tf3_label, tf4_label, tf5_label, tf6_label

    // Conditions A-J per TF (60 boolean fields)
    bool condA_tf1, condA_tf2, ..., condJ_tf6

    // Triggers A-J (10 boolean fields)
    bool trigA, trigB, ..., trigJ

    // Trigger metadata
    float trigger_price
    string trigger_label
```

### 2. Hybrid vs Legacy Architecture

**Hybrid Architecture (Recommended)**
- Toolkit owns all `request.security()` calls
- Library exports a `buildOutput()` function that processes pre-fetched data
- Parameters are fully optimizable by third-party tools

**Legacy Architecture**
- Library handles all data fetching internally
- Library exports `getTFConfluence()` function
- Parameters may not be optimizable

### 3. Documentation

When adding a new library:
1. Add an entry to this document
2. Document all parameters with their meanings
3. Document all conditions (A-J) that the library uses
4. Document all triggers (A-J) that the library uses
5. Provide recommended configurations for common use cases

---

## Glossary

| Term | Definition |
|------|------------|
| **Side Module** | A timeframe-based analysis module that evaluates conditions across 6 configurable timeframes |
| **Condition** | A boolean state evaluated per-timeframe (e.g., "Is price above EMA?") |
| **Trigger** | A boolean event on the chart timeframe (e.g., "EMA crossover just occurred") |
| **Confluence** | The combination of multiple conditions across timeframes to determine trade quality |
| **TH Score** | Threshold Score - points assigned when a condition is met on a timeframe |
| **Grade** | Overall confluence quality (A, B, C, or None) based on total TH Score |
| **Required** | A condition that must be met for the grade to be valid (acts as a filter) |

---

*Last Updated: January 29, 2026*
*Toolkit Version: 1.1 (Hybrid Architecture)*
