# KevBot Toolkit v3.0 Vision Document

**Status:** ✅ Implementation Complete
**Created:** February 1, 2026
**Last Updated:** February 2, 2026

---

## Executive Summary

Version 3.0 replaces the threshold-based scoring system with an AND/OR confluence group system. This change makes strategy logic more intuitive, aligns with how traders naturally think about confluence, and enables seamless integration with the Trade Analyzer tool.

---

## Core Concept: AND/OR Confluence Groups

### Current System (v2.0) - Threshold-Based
```
Each condition contributes points → Total score → Grade thresholds (C/B/A)
"Required" checkbox provides basic AND logic
```

**Problems:**
- Point values are arbitrary and hard to reason about
- "What score should I set for this condition?" is confusing
- Doesn't map well to how traders think ("I need X AND Y to be true")
- Hard to export/replicate strategy logic in Trade Analyzer

### New System (v3.0) - AND/OR Groups
```
1 AND Group:  ALL assigned conditions must be true
3 OR Groups:  User specifies minimum # of conditions that must be true

Entry Valid = AND Group passes AND OR1 passes AND OR2 passes AND OR3 passes
```

**Benefits:**
- Intuitive: "I need EMA bullish AND VWAP above, plus any 2 of these 5 confirmations"
- No arbitrary point values
- Direct mapping to Trade Analyzer logic
- Import/export via config strings

---

## Group Definitions

### AND Group
- **Rule:** ALL assigned conditions must be true
- **Use case:** Non-negotiable conditions (e.g., "must be above daily VWAP")
- **If empty:** AND requirement automatically passes

### OR Group 1
- **Rule:** At least N of M assigned conditions must be true
- **User configures:** Minimum required (e.g., "2 of 8")
- **Use case:** Primary confirmations with flexibility

### OR Group 2
- **Rule:** At least N of M assigned conditions must be true
- **User configures:** Minimum required (e.g., "1 of 4")
- **Use case:** Secondary confirmations

### OR Group 3
- **Rule:** At least N of M assigned conditions must be true
- **User configures:** Minimum required (e.g., "1 of 3")
- **Use case:** Tertiary/optional confirmations
- **If no conditions assigned:** OR3 requirement automatically passes

### Entry Logic
```
Entry Valid = (AND Group ALL true)
              AND (OR1 meets minimum)
              AND (OR2 meets minimum)
              AND (OR3 meets minimum)
```

---

## Input Structure Overview

### Key Architecture Decisions (Finalized)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Position Direction | Direction-specific (Long OR Short) | Cuts inputs in half, simpler UX |
| Trigger System | Centralized (1 entry, 1 exit dropdown) | Simpler than per-library triggers |
| Condition Assignment | Per-condition per-TF group assignment | Maximum flexibility |
| Side Table Ordering | Auto-order by enable sequence | Removes unnecessary input |
| Grades | Removed entirely | Replaced by AND/OR logic |

### What's Being REMOVED (v2.0 → v3.0)

| Removed Input | Reason |
|---------------|--------|
| TH Score (per condition) | Replaced by group assignment |
| Grade C/B/A Thresholds | No longer needed - using AND/OR |
| Required checkbox | Replaced by AND group |
| Long/Short direction per condition | Toolkit is direction-specific |
| Complex evaluation modes | Conditions are boolean |
| Side Table Slot | Auto-ordered by enable sequence |
| Per-library trigger settings (LE/SE/LX/SX) | Centralized trigger selection |
| Grade-based risk multipliers | Grades removed |

### What's Being ADDED (v3.0)

| New Input | Purpose |
|-----------|---------|
| Position Direction (Long/Short) | User selects once at top |
| Entry Trigger (centralized dropdown) | Single entry trigger from any enabled library |
| Exit Trigger (centralized dropdown) | Single exit trigger from any enabled library |
| Group Assignment (per condition per TF) | Assign each condition to None/AND/OR1/OR2/OR3 |
| OR1/OR2/OR3 Minimum Required | How many conditions must be true per group |

---

## Early Design Examples (Superseded)

**NOTE:** The examples below were from early design discussions and are now superseded by the "Finalized Input Structure (v3.0)" section. They are kept for historical reference only.

Key changes from these early examples:
- Removed Side Table Slot input (auto-ordered now)
- Removed Long/Short per-condition (direction-specific toolkit now)
- Changed from "select one condition" to "assign group to each condition"
- Centralized trigger selection instead of per-library triggers

---

<details>
<summary>Click to expand early design examples (historical reference only)</summary>

### Early Example: Confluence Groups

```pinescript
// OR Group Minimums
int conf_or1_min = input.int(1, "OR1: Minimum Required", minval = 1, maxval = 20)
int conf_or2_min = input.int(1, "OR2: Minimum Required", minval = 1, maxval = 20)
int conf_or3_min = input.int(1, "OR3: Minimum Required", minval = 1, maxval = 20)
```

### Early Example: EMA Stack (Before Finalization)

This early design had Long/Short assignments per condition. The finalized design uses direction-specific toolkit instead.

### Early Example: VWAP (Before Finalization)

This early design had zone dropdowns. The finalized design assigns groups directly to each zone condition.

</details>

---

## Import String Format

### Purpose
Allow users to paste a configuration string from Trade Analyzer to instantly configure all group assignments without manually setting each dropdown.

### Format Specification
```
VERSION:group_definitions;condition_assignments

Where:
- VERSION = "v1" (for future compatibility)
- group_definitions = OR1:min,OR2:min,OR3:min
- condition_assignments = comma-separated list of: lib_cond_tf_dir:group

Example:
v1;OR1:2,OR2:1,OR3:1;ema_A_TF4_L:AND,ema_B_TF4_L:OR1,ema_B_TF1_L:OR1,vwap_D_L:AND,macd_A_TF4_L:OR2
```

### ID Naming Convention
```
{library}_{condition}_{timeframe}_{direction}:{group}

Libraries:     ema, vwap, utbot, macd, macdh, smacd, rvol, swing
Conditions:    A, B, C, D, E (maps to each library's conditions)
Timeframes:    TF1, TF2, TF3, TF4, TF5, TF6 (side) or D, W, M (top/VWAP)
Direction:     L (long), S (short)
Groups:        AND, OR1, OR2, OR3
```

### Parsing Behavior
When `Use Import String` is checked:
1. Parse the import string
2. Override all manual dropdown values with parsed assignments
3. Any condition not in the string defaults to "None"
4. Display a warning/status if parsing fails

---

## Input Count Estimate (Updated)

**See "Finalized Input Structure (v3.0)" section below for detailed breakdown.**

### Summary

| Category | Inputs |
|----------|--------|
| Global Settings + Timeframes | 16 |
| Entry/Exit Configuration | 4 |
| Position Sizing | 22 |
| Backtesting | 14 |
| VWAP (Top Table) | 25 |
| EMA Stack | 41 |
| RVOL | 36 |
| UT Bot | 17 |
| Swing 123 | 38 |
| MACD Line | 29 |
| MACD Histogram | 29 |
| Simple MACD | 29 |
| **TOTAL** | **~306** |

**TradingView limit is ~500. We have significant room for growth.**

### Inputs Removed from v2.0

| Removed | Savings |
|---------|---------|
| TH Scores (per condition per TF) | ~200+ |
| Required flags (per condition per TF) | ~200+ |
| Grade thresholds (C/B/A × Long/Short) | 6 |
| Grade-based risk multipliers | 6 |
| Per-library trigger sections (LE/SE/LX/SX × 7 libs) | ~100+ |
| Side Table Slot (per library) | 7 |
| Duplicate Long/Short condition assignments | 50% reduction |

Direction-specific design and centralized triggers dramatically reduced input count.

---

## Confluence Engine Logic (Pseudocode)

```pinescript
// Collect all condition states into group buckets
int and_total = 0
int and_true = 0

int or1_total = 0
int or1_true = 0

int or2_total = 0
int or2_true = 0

int or3_total = 0
int or3_true = 0

// For each condition (example: EMA Stack Condition A, TF4, Long)
if lib_ema_A_TF4_long != "None"
    bool condState = lib_ema_output.condA_tf4  // Get actual condition state

    if lib_ema_A_TF4_long == "AND"
        and_total += 1
        and_true += condState ? 1 : 0
    else if lib_ema_A_TF4_long == "OR1"
        or1_total += 1
        or1_true += condState ? 1 : 0
    else if lib_ema_A_TF4_long == "OR2"
        or2_total += 1
        or2_true += condState ? 1 : 0
    else if lib_ema_A_TF4_long == "OR3"
        or3_total += 1
        or3_true += condState ? 1 : 0

// ... repeat for all conditions ...

// Evaluate groups
bool and_pass = (and_total == 0) or (and_true == and_total)
bool or1_pass = (or1_total == 0) or (or1_true >= conf_or1_min)
bool or2_pass = (or2_total == 0) or (or2_true >= conf_or2_min)
bool or3_pass = (or3_total == 0) or (or3_true >= conf_or3_min)

// Final entry validity
bool longEntryValid = and_pass and or1_pass and or2_pass and or3_pass
```

---

## Trade Analyzer Integration

### Export Format
Each bar exports condition states with their group assignments:
```csv
timestamp, condition_id, group, direction, state
2026-02-01 09:30, ema_A_TF4, AND, long, true
2026-02-01 09:30, ema_B_TF4, OR1, long, true
2026-02-01 09:30, vwap_D, AND, long, false
...
```

### Trade Analyzer Features
1. **Strategy Builder:** Visual drag-drop interface to assign conditions to groups
2. **Config Export:** Generate import string for TradingView
3. **Backtest Analysis:** Evaluate historical trades against group logic
4. **What-If Analysis:** "What if I moved this condition from OR1 to AND?"

---

## Migration Path (v2.0 → v3.0)

### For Users
1. v3.0 will be a separate indicator (not an update to v2.0)
2. Users can run both side-by-side during transition
3. No automatic migration of settings (clean start)

### For Development
1. Create `src/main/KevBot Toolkit v3.0.txt` as new working file
2. Keep v2.0 functional in `src/main/KevBot Toolkit v2.0.txt`
3. Port interpreter loaders (minimal changes needed)
4. Replace confluence engine entirely
5. Update table renderers to show group assignments

---

## Open Questions (Resolved)

1. **Trigger System:** ✅ RESOLVED - Triggers are separate from confluence conditions. Triggers are events; confluence conditions determine whether to upgrade triggers to position entries. Centralized trigger selection with one Entry Trigger and one Exit Trigger dropdown.

2. **Display:** ✅ RESOLVED - Top table shows AND/OR group status with numerator/denominator (e.g., "AND: 3/3 ✓", "OR1: 2/3 ✓")

3. **Long vs Short Separation:** ✅ RESOLVED - Toolkit is direction-specific. User selects "Long" or "Short" at the top. This cuts inputs roughly in half and simplifies the user experience. Users can run two indicators if they want both directions.

4. **Empty Groups:** ✅ RESOLVED - Empty groups auto-pass. Only requirement is that Entry Trigger and Exit Trigger are defined.

5. **Backtest Module:** ✅ RESOLVED - Keep backtest module mostly unchanged. Remove grade-related outputs since grades are removed.

---

## Finalized Input Structure (v3.0)

**Key Design Decisions:**
- **Direction-specific:** User selects Long or Short once at the top
- **Centralized triggers:** Single Entry Trigger and Exit Trigger dropdown (not per-library)
- **Per-condition group assignment:** Each condition from each library on each TF can be assigned to a group (None/AND/OR1/OR2/OR3)
- **No grades:** Removed all grade thresholds, grade-based risk multipliers, and grade displays
- **Auto-ordered side table:** Libraries appear in order they're enabled (no manual slot assignment)
- **Flexible conditions:** User can assign multiple conditions from the same library to different groups (e.g., SML on TF1 → AND, MSL on TF3 → OR1)

---

### Section 1: Global Toolkit Settings

```
1. Global Toolkit Settings
├── Position Direction: [Long | Short]
├── Enable Top Table: [bool]
├── Enable Side Table: [bool]
└── Use Dark Theme: [bool]

1.1 Timeframe Configuration
├── TF1 Enabled: [bool]   TF1: [timeframe]
├── TF2 Enabled: [bool]   TF2: [timeframe]
├── TF3 Enabled: [bool]   TF3: [timeframe]
├── TF4 Enabled: [bool]   TF4: [timeframe]
├── TF5 Enabled: [bool]   TF5: [timeframe]
└── TF6 Enabled: [bool]   TF6: [timeframe]

1.2 Confluence Group Settings
├── OR1 Minimum Required: [int 1-20, default 1]
├── OR2 Minimum Required: [int 1-20, default 1]
└── OR3 Minimum Required: [int 1-20, default 1]
```

**Inputs: ~16**

---

### Section 2: Entry/Exit Configuration (Centralized)

```
2. Entry/Exit Configuration
├── Entry Trigger: [Master dropdown - all triggers from enabled libraries]
├── Exit Trigger: [Master dropdown - all triggers from enabled libraries]
├── Show Raw Entry Marks: [bool]
└── Show Raw Exit Marks: [bool]
```

**Entry/Exit Trigger Dropdown Options** (populated based on enabled libraries):
```
- None
─── EMA Stack ───
- EMA: S > M Cross
- EMA: S < M Cross
- EMA: S > L Cross
- EMA: S < L Cross
- EMA: M > L Cross
- EMA: M < L Cross
─── UT Bot ───
- UT Bot: Buy
- UT Bot: Sell
─── RVOL ───
- RVOL: Volume Spike
- RVOL: Volume Extreme
- RVOL: Volume Fade
─── Swing 123 ───
- Swing: BC2 (Bullish Candle 2)
- Swing: BC3 (Bullish Candle 3)
- Swing: XC2 (Bearish Candle 2)
- Swing: XC3 (Bearish Candle 3)
─── MACD Line ───
- MACD Line: Bullish Cross
- MACD Line: Bearish Cross
- MACD Line: Zero Cross Up
- MACD Line: Zero Cross Down
─── MACD Histogram ───
- MACD Hist: Flip Bullish
- MACD Hist: Flip Bearish
- MACD Hist: Shift Up
- MACD Hist: Shift Down
─── Simple MACD ───
- Simple MACD: Bullish Cross
- Simple MACD: Bearish Cross
```

**Inputs: ~4**

---

### Section 3: Position Sizing Module

```
3. Position Sizing Module
├── Show Module: [bool]
└── Custom Name: [string]

3.1 Safety Stop Loss
├── SL Method: [ATR | Fixed $ | Percentage | Candle Wicks | Library Custom]
├── Show SL on Chart: [bool]
├── ATR Period: [int]
├── ATR Multiplier: [float]
├── Fixed Dollar Stop: [float]
├── Percentage Stop: [float]
├── Lo/Hi Lookback: [int]
└── Lo/Hi Padding: [float]

3.2 Safety Take Profit
├── TP Method: [R:R Target | ATR | Fixed $ | Percentage | Candle Wicks]
├── Show TP on Chart: [bool]
├── R:R Target: [float]
├── ATR Period: [int]
├── ATR Multiplier: [float]
├── Fixed Dollar Target: [float]
├── Percentage Target: [float]
├── Lo/Hi Lookback: [int]
└── Lo/Hi Padding: [float]

3.3 Account & Risk
├── Account Size: [float]
├── Risk Mode: [Share Qty | Fixed $ Risk | Percentage Risk]
├── Default Share Qty: [int]
├── Default Risk ($): [float]
└── Default Risk (%): [float]

3.4 Position Labels
├── Show Trigger Label: [bool]
├── Show Qty: [bool]
└── Show $Risk: [bool]
```

**Note:** Removed grade-based risk multipliers and grade display options.

**Inputs: ~22**

---

### Section 4: Backtesting Module

```
4. Backtesting Module
├── Show Backtest KPI Module: [bool]
└── Custom Name: [string]

4.1 Data & Lookback
├── Backtest Window: [Entire Chart | Fixed Bars | Date Range]
├── Window (Bars): [int]
├── Start Date: [timestamp]
├── Use End Date: [bool]
└── End Date: [timestamp]

4.2 Filters
├── Time of Day: [session]
└── Day of Week: [M] [T] [W] [Th] [F] [Sa] [Su]
```

**Inputs: ~14**

---

### Section 5: Top Table Interpreters

#### 5-1. VWAP

Each zone condition has its own group assignment row.

```
5-1. VWAP
├── Enable: [bool]
└── Display Name: [string]

5-1.1 VWAP Parameters
├── Band 1 Multiplier (σ): [float, default 1.0]
└── Band 2 Multiplier (σ): [float, default 2.0]

5-1.2 Daily VWAP Conditions (Sigma-Based Zones)
│   (Each zone → Group assignment, 7 distinct zones)
├── >+2σ (Extreme High):  [None | AND | OR1 | OR2 | OR3]
├── +1σ to +2σ:           [None | AND | OR1 | OR2 | OR3]
├── 0 to +1σ:             [None | AND | OR1 | OR2 | OR3]
├── @V (At VWAP):         [None | AND | OR1 | OR2 | OR3]
├── -1σ to 0:             [None | AND | OR1 | OR2 | OR3]
├── -2σ to -1σ:           [None | AND | OR1 | OR2 | OR3]
└── <-2σ (Extreme Low):   [None | AND | OR1 | OR2 | OR3]

5-1.3 Weekly VWAP Conditions
│   (Same 7 sigma zones as Daily)
└── ... [7 group assignment dropdowns]

5-1.4 Monthly VWAP Conditions
│   (Same 7 sigma zones as Daily)
└── ... [7 group assignment dropdowns]
```

**Inputs: ~25** (2 enable/name + 2 params + 21 conditions)

**Zone Labels (from `_kb_vwapZoneLabel`):**
- `>+2σ` = Price above +2 standard deviations
- `>+1σ` = Price between +1σ and +2σ
- `>V` = Price between VWAP and +1σ
- `@V` = Price at VWAP (within ±0.5σ)
- `<V` = Price between -1σ and VWAP
- `<-1σ` = Price between -2σ and -1σ
- `<-2σ` = Price below -2 standard deviations

---

### Section 6: Side Table Interpreters

#### 6-1. EMA Stack

Each condition × each TF = separate group assignment.

```
6-1. EMA Stack
├── Enable: [bool]
└── Display Name: [string]

6-1.1 EMA Stack Parameters
├── Short EMA Length: [int, default 10]
├── Medium EMA Length: [int, default 20]
└── Long EMA Length: [int, default 50]

6-1.2 EMA Stack – TF1 Conditions
├── SML (Bull Stack): [None | AND | OR1 | OR2 | OR3]
├── LMS (Bear Stack): [None | AND | OR1 | OR2 | OR3]
├── SLM:              [None | AND | OR1 | OR2 | OR3]
├── MSL:              [None | AND | OR1 | OR2 | OR3]
├── MLS:              [None | AND | OR1 | OR2 | OR3]
└── LSM:              [None | AND | OR1 | OR2 | OR3]

6-1.3 EMA Stack – TF2 Conditions
└── ... [same 6 conditions]

6-1.4 EMA Stack – TF3 Conditions
└── ... [same 6 conditions]

6-1.5 EMA Stack – TF4 Conditions
└── ... [same 6 conditions]

6-1.6 EMA Stack – TF5 Conditions
└── ... [same 6 conditions]

6-1.7 EMA Stack – TF6 Conditions
└── ... [same 6 conditions]
```

**Inputs: ~41** (2 enable/name + 3 params + 36 conditions)

---

#### 6-2. RVOL

```
6-2. RVOL
├── Enable: [bool]
└── Display Name: [string]

6-2.1 RVOL Parameters
├── Lookback Period: [int, default 20]
├── High Threshold: [float, default 1.5]
├── Very High Threshold: [float, default 2.0]
└── Extreme Threshold: [float, default 3.0]

6-2.2 RVOL – TF1 Conditions
├── RV! (Extreme):    [None | AND | OR1 | OR2 | OR3]
├── RV++ (Very High): [None | AND | OR1 | OR2 | OR3]
├── RV+ (Elevated):   [None | AND | OR1 | OR2 | OR3]
├── RV= (Normal):     [None | AND | OR1 | OR2 | OR3]
└── RV- (Low):        [None | AND | OR1 | OR2 | OR3]

6-2.3 through 6-2.7: TF2-TF6
└── ... [same 5 conditions per TF]
```

**Inputs: ~36** (2 enable/name + 4 params + 30 conditions)

---

#### 6-3. UT Bot

```
6-3. UT Bot
├── Enable: [bool]
└── Display Name: [string]

6-3.1 UT Bot Parameters
├── Key Value (ATR Mult): [float, default 1.0]
├── ATR Period: [int, default 10]
└── Use Heikin Ashi: [bool]

6-3.2 UT Bot – TF1 Conditions
├── Bull (Above Stop): [None | AND | OR1 | OR2 | OR3]
└── Bear (Below Stop): [None | AND | OR1 | OR2 | OR3]

6-3.3 through 6-3.7: TF2-TF6
└── ... [same 2 conditions per TF]
```

**Inputs: ~17** (2 enable/name + 3 params + 12 conditions)

---

#### 6-4. Swing 123

```
6-4. Swing 123
├── Enable: [bool]
└── Display Name: [string]

(No parameters - pure price action)

6-4.1 Swing 123 – TF1 Conditions
├── BC2 (Bullish Candle 2):    [None | AND | OR1 | OR2 | OR3]
├── BC3 (Bullish Candle 3):    [None | AND | OR1 | OR2 | OR3]
├── XC2 (Bearish Candle 2):    [None | AND | OR1 | OR2 | OR3]
├── XC3 (Bearish Candle 3):    [None | AND | OR1 | OR2 | OR3]
├── B↑ (Recent Bullish Setup): [None | AND | OR1 | OR2 | OR3]
└── X↓ (Recent Bearish Setup): [None | AND | OR1 | OR2 | OR3]

6-4.2 through 6-4.6: TF2-TF6
└── ... [same 6 conditions per TF]
```

**Inputs: ~38** (2 enable/name + 36 conditions)

---

#### 6-5. MACD Line

```
6-5. MACD Line
├── Enable: [bool]
└── Display Name: [string]

6-5.1 MACD Line Parameters
├── Fast Length: [int, default 12]
├── Slow Length: [int, default 26]
└── Signal Length: [int, default 9]

6-5.2 MACD Line – TF1 Conditions
├── M>S+ (MACD > Signal, Rising):  [None | AND | OR1 | OR2 | OR3]
├── M>S- (MACD > Signal, Falling): [None | AND | OR1 | OR2 | OR3]
├── M<S- (MACD < Signal, Falling): [None | AND | OR1 | OR2 | OR3]
└── M<S+ (MACD < Signal, Rising):  [None | AND | OR1 | OR2 | OR3]

6-5.3 through 6-5.7: TF2-TF6
└── ... [same 4 conditions per TF]
```

**Inputs: ~29** (2 enable/name + 3 params + 24 conditions)

---

#### 6-6. MACD Histogram

```
6-6. MACD Histogram
├── Enable: [bool]
└── Display Name: [string]

6-6.1 MACD Histogram Parameters
├── Fast Length: [int, default 12]
├── Slow Length: [int, default 26]
└── Signal Length: [int, default 9]

6-6.2 MACD Histogram – TF1 Conditions
├── H+↑ (Positive, Rising):  [None | AND | OR1 | OR2 | OR3]
├── H+↓ (Positive, Falling): [None | AND | OR1 | OR2 | OR3]
├── H-↓ (Negative, Falling): [None | AND | OR1 | OR2 | OR3]
└── H-↑ (Negative, Rising):  [None | AND | OR1 | OR2 | OR3]

6-6.3 through 6-6.7: TF2-TF6
└── ... [same 4 conditions per TF]
```

**Inputs: ~29** (2 enable/name + 3 params + 24 conditions)

---

#### 6-7. Simple MACD

```
6-7. Simple MACD
├── Enable: [bool]
└── Display Name: [string]

6-7.1 Simple MACD Parameters
├── Fast Length: [int, default 12]
├── Slow Length: [int, default 26]
└── Signal Length: [int, default 9]

6-7.2 Simple MACD – TF1 Conditions
├── M>S↑ (Above Signal, Crossed Up):   [None | AND | OR1 | OR2 | OR3]
├── M>S↓ (Above Signal, Crossed Down): [None | AND | OR1 | OR2 | OR3]
├── M<S↓ (Below Signal, Crossed Down): [None | AND | OR1 | OR2 | OR3]
└── M<S↑ (Below Signal, Crossed Up):   [None | AND | OR1 | OR2 | OR3]

6-7.3 through 6-7.7: TF2-TF6
└── ... [same 4 conditions per TF]
```

**Inputs: ~29** (2 enable/name + 3 params + 24 conditions)

---

### Input Count Summary

| Section | Inputs |
|---------|--------|
| 1. Global Settings | 16 |
| 2. Entry/Exit Config | 4 |
| 3. Position Sizing | 22 |
| 4. Backtesting | 14 |
| 5-1. VWAP | 31 |
| 6-1. EMA Stack | 41 |
| 6-2. RVOL | 36 |
| 6-3. UT Bot | 17 |
| 6-4. Swing 123 | 38 |
| 6-5. MACD Line | 29 |
| 6-6. MACD Histogram | 29 |
| 6-7. Simple MACD | 29 |
| **TOTAL** | **~306** |

**Well under the ~500 TradingView limit.** Room for future growth (additional interpreters, import/export features, etc.).

---

### Top Table Display (v3.0)

Replace the old confluence score/grade display with AND/OR group status:

```
┌─────────────┬───────────┬─────────┬─────────┬─────────┬─────────┐
│ Module      │ Value     │ AND     │ OR1     │ OR2     │ OR3     │
├─────────────┼───────────┼─────────┼─────────┼─────────┼─────────┤
│ Confluence  │ VALID     │ ✓ 2/2   │ ✓ 3/2   │ ✓ 1/1   │ — 0/0   │
│ VWAP Daily  │ >V        │ ✓       │ —       │ —       │ —       │
│ VWAP Weekly │ >+1σ      │ —       │ ✓       │ —       │ —       │
│ VWAP Monthly│ @V        │ —       │ —       │ —       │ —       │
└─────────────┴───────────┴─────────┴─────────┴─────────┴─────────┘
```

Where:
- `✓ 2/3` = 2 of 3 conditions in this group are true
- `— 0/0` = No conditions assigned to this group (auto-passes)
- `VALID` / `INVALID` = Final entry validity based on all groups passing

---

### Side Table Display (v3.0)

Side table shows current state per TF. Cell color indicates if the condition(s) assigned to that TF are passing.

```
┌────────────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ Interpreter│ TF1  │ TF2  │ TF3  │ TF4  │ TF5  │ TF6  │
├────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EMA Stack  │ SML  │ SLM  │ MSL  │ SML  │ LMS  │ SML  │
│ RVOL       │ RV+  │ RV=  │ RV++ │ RV+  │ RV=  │ RV-  │
│ UT Bot     │ Bull │ Bull │ Bear │ Bull │ Bull │ Bear │
└────────────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

Cell coloring:
- **Green** = At least one assigned condition in this cell is TRUE
- **Gray** = No conditions assigned, or all assigned conditions are FALSE
- **Yellow** = Condition is being monitored but not currently matching

---

## Next Steps

1. [x] Review and finalize this vision document
2. [x] Decide on open questions above
3. [x] Create `src/main/KevBot Toolkit v3.0.txt` as new working file
4. [x] Implement centralized trigger system (Section 10: `_evalTrigger()`)
5. [x] Implement AND/OR confluence engine (Sections 8-9)
6. [x] Update input structure for all interpreters (Sections 1-7)
7. [x] Update table renderers for new display format (Section 11)
8. [ ] Test with sample strategies (QA testing in progress)
9. [ ] Design Trade Analyzer config string generator (future)

---

## Implementation Notes (February 2, 2026)

The following sections were implemented in `src/main/KevBot Toolkit v3.0.txt`:

### Section 10: Centralized Trigger Routing
- `_evalTrigger(string triggerSel)` function maps 26 trigger types to library signals
- Supported triggers from: EMA Stack, UT Bot, RVOL, Swing 123, MACD Line, MACD Histogram, Simple MACD
- Single Entry Trigger and Exit Trigger dropdowns instead of per-library triggers

### Section 11: Table Renderers

**Top Table:**
- Module-based layout matching v2.0 pattern (2 columns × 5 rows per module)
- `renderModule()`, `renderModuleConf()`, `renderModuleVWAP()` helper functions
- Modules: Position Sizing, Backtest KPI, Confluence Summary, VWAP D/W/M

**Side Table:**
- 7 libraries: EMA Stack, RVOL, UT Bot, Swing 123, MACD Line, MACD Histogram, Simple MACD
- TF formatting via TimeUtils `tu.tu_fmt()` function
- Color coding for all libraries:
  - **Green:** Condition assigned AND true
  - **Yellow:** Condition assigned AND false
  - **Gray:** Condition not assigned to any group

### Known Issues Resolved
- Fixed `KB_TF_Out_V2` type definition (one field per line for Pine Script v6)
- Corrected library import version (`KevBot_TF_EMA_Stack/7`)
- Fixed variable names: `positionDirection`, `lib_vwap_D_zone`, `defShares`, etc.
- Extended color coding from EMA Stack to all 7 side table libraries

---

*Last Updated: February 2, 2026*
*v3.0 Implementation Complete - QA Testing Phase*
