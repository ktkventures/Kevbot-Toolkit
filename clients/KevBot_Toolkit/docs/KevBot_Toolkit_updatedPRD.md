# KevBot Toolkit - Product Requirements Document (PRD)

**Version:** 1.4
**Date:** January 30, 2026
**Target Platform:** TradingView (PineScript v6)
**Development Tool:** Claude Code

---

## Executive Summary

The KevBot Toolkit is a modular TradingView indicator designed to automate trade journaling and enable multivariate analysis of trading strategies. It replaces manual data entry (80+ questions per trade) with automatic context capture across multiple timeframes and indicators, displaying results in two dynamic tables.

**Current Development State:**
- v1.1 Hybrid Architecture implemented and working
- 2 Side Modules functional (proof of concept for multi-module support)
- EMA Stack library working correctly with hybrid pattern
- Parameters (paramA-F) now fully optimizable by third-party tools
- Side table displays both modules with correct EMA stack labels
- Chart plotting system: raw trigger marks (cross) and position signals (triangles/xcross)
- Label system: trigger name labels for raw signals, full metadata labels for positions
- Data export plots: invisible plots for CSV export of confluence data, position sizing, TF states
- All 10 triggers (A-J) and 10 conditions (A-J) exposed in UI dropdowns
- Sensible default input values for immediate signal display on load

**Goal of This PRD:** Enable Claude Code to accelerate development with full codebase awareness, documenting current state and architecture decisions accurately.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [System Architecture](#3-system-architecture)
4. [Technical Specifications](#4-technical-specifications)
5. [Module Specifications](#5-module-specifications)
6. [Library System](#6-library-system)
7. [Data Flow](#7-data-flow)
8. [User Interface](#8-user-interface)
9. [Development Priorities](#9-development-priorities)
10. [Testing & Validation](#10-testing--validation)
11. [Known Issues & Blockers](#11-known-issues--blockers)
12. [Success Criteria](#12-success-criteria)

---

## 1. Problem Statement

### 1.1 Current Manual Process Pain Points
- Traders must manually answer ~80 questions per trade for proper journaling
- Tracking complex conditions across 6 timeframes is error-prone
- No systematic way to identify which variables drive trading success
- Missing trade context prevents effective multivariate analysis

### 1.2 Development Blockers (Pre-Claude Code)
- ChatGPT's 500-line context window insufficient for ~1,700-line Indicator file
- Copy-paste workflow causes AI to lose context between edits
- High error rate when modifying existing features
- Demoralizing development pace: 1 small feature per full day of work

---

## 2. Solution Overview

### 2.1 Product Vision
A modular TradingView indicator that:
1. Automatically captures all trade conditions at signal generation
2. Displays data in two real-time tables (Top Table + Side Table)
3. Enables CSV export for multivariate backtesting analysis
4. Supports swappable indicator modules via library system

### 2.2 Key Features
- **Top Table:** Non-timeframe-specific variables (news, position sizing, custom indicators)
- **Side Table:** Multi-timeframe analysis (6 configurable timeframes)
- **Modular Libraries:** Plug-and-play indicator calculations
- **Confluence Scoring:** Grade-based trade quality assessment (A/B/C grades)
- **Position Engine:** Position tracking with safety stops
- **Backtest KPIs:** Win rate, profit factor, max drawdown tracking

---

## 3. System Architecture

### 3.1 Core Components

```
┌─────────────────────────────────────────────────────────┐
│                  KevBot Toolkit Main                    │
│              (Indicator File ~1,700 lines)              │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌──────────────┐
│  Top Libraries│  │  TF Libraries  │  │ Utility Libs │
│   (4 outputs) │  │  (6 TF + evt)  │  │  (TimeUtils) │
└───────────────┘  └────────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
        ┌──────────────────────────────────────┐
        │      Confluence Engine               │
        │  (TH Scoring + Grade Calculation)    │
        └──────────────────────────────────────┘
                            ▼
        ┌──────────────────────────────────────┐
        │      Position Engine                 │
        │  (Entry/Exit Logic + State Machine)  │
        └──────────────────────────────────────┘
                            ▼
        ┌──────────────────────────────────────┐
        │      Table Rendering                 │
        │  (Top Table + Side Table Display)    │
        └──────────────────────────────────────┘
```

### 3.2 File Structure

**Main Files:**
- `LEGACY - Kevbot Toolkit v1.0 - Input Skeleton.txt` — Legacy core indicator (reference only)
- `KevBot Toolkit v1.1 - Hybrid Architecture.txt` — **Current** main toolkit with hybrid pattern (~2,400 lines)
- `LEGACY - KevBot_Toolkit_EMAStack_Integration.txt` — Reference template for hybrid integration (superseded)

**Library Files:**
- `KevBot_TimeUtils.txt` — Timeframe formatting utilities
- `KevBot_Top_Minimal.txt` — Top module for MA crossover (4 outputs: A/B/C/D)
- `KevBot_TF_Placeholder.txt` — Placeholder TF module (10 conditions A-J × 6 TFs + 10 triggers A-J)
- `KevBot_TF_EMA Stack.txt` — EMA Stack with `buildOutput()` hybrid function (v7)
- `KevBot_Types.txt` — Shared type definitions (TFModuleOutput, EMAStackData, etc.)
- `KevBot_Indicators.txt` — Pure processing functions (orderEMA, isBullStack, etc.)

**Documentation Files:**
- `KevBot_Architecture_Notes.md` — PineScript limitations and architecture decisions
- `KevBot_Library_Definitions.md` — User-facing library documentation (params, triggers, conditions)
- `KevBot_Toolkit_updatedPRD.md` — This PRD

### 3.3 Architecture Principles

1. **Separation of Concerns:**
   - Libraries: Complex calculations
   - Indicator: Aggregation & display
   
2. **Standardized Outputs:**
   - All libraries output using consistent variable names
   - Enables generic table population without library-specific logic
   
3. **Modularity:**
   - Libraries are swappable
   - Indicator remains library-agnostic

---

## 4. Technical Specifications

### 4.1 Platform & Language
- **Platform:** TradingView
- **Language:** PineScript v6
- **Execution:** Browser-based, real-time chart overlay

### 4.2 Performance Constraints
- **Max Script Size:** ~60,000 characters (TradingView limit)
- **Max Computation:** Limited by TradingView's server-side execution model
- **Max Table Cells:** No hard limit, but rendering performance degrades with large tables
- **Security Requests:** Must be `simple` type for multi-timeframe data

### 4.3 Data Types & Structures

**TopModuleOutput (Library Standard):**
```pinescript
type TopModuleOutput
    float outA  // Fast MA
    float outB  // Slow MA
    float outC  // Difference (fast - slow)
    float outD  // Cross direction (1 long, -1 short, 0 none)
```

**TFModuleOutput (Library Standard - V2 Expanded):**
```pinescript
type TFModuleOutput
    // Per-TF labels (state description string for display)
    string tf1_label, tf2_label, tf3_label, tf4_label, tf5_label, tf6_label

    // 10 Conditions (A-J) per TF (60 total boolean fields)
    bool condA_tf1, condA_tf2, condA_tf3, condA_tf4, condA_tf5, condA_tf6
    bool condB_tf1, condB_tf2, condB_tf3, condB_tf4, condB_tf5, condB_tf6
    bool condC_tf1, condC_tf2, condC_tf3, condC_tf4, condC_tf5, condC_tf6
    bool condD_tf1, condD_tf2, condD_tf3, condD_tf4, condD_tf5, condD_tf6
    bool condE_tf1, condE_tf2, condE_tf3, condE_tf4, condE_tf5, condE_tf6
    bool condF_tf1, condF_tf2, condF_tf3, condF_tf4, condF_tf5, condF_tf6
    bool condG_tf1, condG_tf2, condG_tf3, condG_tf4, condG_tf5, condG_tf6  // Reserved
    bool condH_tf1, condH_tf2, condH_tf3, condH_tf4, condH_tf5, condH_tf6  // Reserved
    bool condI_tf1, condI_tf2, condI_tf3, condI_tf4, condI_tf5, condI_tf6  // Reserved
    bool condJ_tf1, condJ_tf2, condJ_tf3, condJ_tf4, condJ_tf5, condJ_tf6  // Reserved

    // 10 Triggers (A-J) - chart timeframe events
    bool trigA, trigB, trigC, trigD, trigE, trigF, trigG, trigH, trigI, trigJ

    // Trigger metadata
    float  trigger_price
    string trigger_label
```

**KB_TF_Out_V2 (Internal Normalized Structure):**
```pinescript
// Toolkit-internal normalized output (NOT exported)
// Used to standardize all library outputs for downstream processing
type KB_TF_Out_V2
    // TF state labels
    string tf1_label, tf2_label, tf3_label, tf4_label, tf5_label, tf6_label

    // 10 Conditions (A-J) per TF - mirrors TFModuleOutput structure
    // Allows generic condition evaluation without library-specific logic
    bool condA_tf1...condJ_tf6  // (60 total fields)

    // 10 Triggers (A-J)
    bool trigA...trigJ

    // Trigger metadata
    float trigger_price
    string trigger_label
```

**ConfluenceResult (Internal):**
```pinescript
type ConfluenceResult
    float  longScore, shortScore
    bool   longReqOk, shortReqOk
    string longGrade, shortGrade  // "A"/"B"/"C"/"None"/"Fail"
```

### 4.4 Naming Conventions
- **Input Groups:** `GRP_MODULENAME` (e.g., `GRP_TOP1`, `GRP_SIDE1`)
- **Internal State:** `_kb_variableName` (e.g., `_kb_posDir`, `_kb_longScore`)
- **Module Outputs:** `moduleN_outputName` (e.g., `top1_raw`, `side1_raw`)
- **Timeframes:** `tf1` through `tf6` (user-configurable)

---

## 5. Module Specifications

### 5.1 Global Toolkit Settings (Module 1)

**Purpose:** Core configuration affecting entire toolkit

**Inputs:**
- `enableTopTable` (bool): Show/hide Top Table
- `enableSideTable` (bool): Show/hide Side Table
- `useDarkTheme` (bool): Color theme toggle

**Timeframe Configuration (1.1):**
- 6 timeframe slots (TF1–TF6), each with:
  - `tfN_enabled` (bool): Include in scoring
  - `tfN` (timeframe string): Resolution (e.g., "1", "5", "15", "D")

**Confluence Grades (1.2):**
- Long thresholds: `gradeC_long`, `gradeB_long`, `gradeA_long` (int, TH Score minimums)
- Short thresholds: `gradeC_short`, `gradeB_short`, `gradeA_short`

**Example:**
```
Grade C Long: 200 TH Score
Grade B Long: 250 TH Score
Grade A Long: 300 TH Score
```

---

### 5.2 Confluence Summary Modules (Module 2)

**Purpose:** Display aggregated confluence scores in Top Table

**Inputs:**
- `showCSM_long` (bool): Show Long Confluence Module
- `csmLongName` (string): Custom display name
- `showCSM_short` (bool): Show Short Confluence Module
- `csmShortName` (string): Custom display name

**Behavior:**
- Visual only (does not affect trading logic)
- Displays current `_kb_longScore` / `_kb_shortScore` and grades

---

### 5.3 Position Sizing Module (Module 3)

**Purpose:** Configure position entry/exit logic and risk management

**Entry Configuration (3):**
- `sizer_entry` (string dropdown):
  - "Any Entry Trigger"
  - "Entry Trigger + Grade C/B/A"
  - "Library Custom"
- `sizerAllowMulti` (bool): Allow multiple entries per trade
- `sizerMaxEntries` (int): Max stacked entries

**Exit Configuration (3):**
- `sizer_exit` (string dropdown):
  - "First Exit Trigger"
  - "Exit Trigger + Grade C/B/A"
  - "Library Custom"
- `sizer_exitEOD` (bool): Force exit at end of day

**Safety Stop Loss (3.1):**
- `sl_method` (dropdown): ATR / Fixed Dollar / Percentage / Candle Wicks / Library Custom
- `sl_show` (bool): Plot on chart
- `sl_atrPer` (int): ATR period
- `sl_atrMult` (float): ATR multiplier
- `sl_fixed` (float): Fixed dollar stop
- `sl_pct` (float): Percentage stop
- `sl_lookback` (int): Candle wick lookback bars
- `sl_pad` (float): Padding for wick stops

**Safety Take Profit (3.2):**
- `tp_method` (dropdown): Risk:Reward Target / ATR / Fixed Dollar / Percentage / Candle Wicks
- `tp_show` (bool): Plot on chart
- `tp_rr` (float): Risk-reward ratio
- (Same ATR/Fixed/Pct/Lookback parameters as SL)

**Account & Risk (3.3):**
- `acctSize` (float): Total account size
- `riskMode` (dropdown): Share Qty / Fixed $ Risk / Percentage Risk
- `defShares` (int): Default share quantity
- `defRisk` (float): Fixed dollar risk
- `defRiskPct` (float): Percentage risk (% of account)

**Risk Multipliers (3.4):**
- `mult_CL`, `mult_BL`, `mult_AL` (float): Grade C/B/A Long multipliers
- `mult_CS`, `mult_BS`, `mult_AS` (float): Grade C/B/A Short multipliers

**Position Labels (3.5):**
- `lbl_warn` (bool): Show bullish/warning label text
- `lbl_grade` (bool): Show confluence grade
- `lbl_type` (bool): Show trade type (Long/Short)
- `lbl_qty` (bool): Show position quantity
- `lbl_risk` (bool): Show dollar risk

---

### 5.4 Backtesting Module (Module 4)

**Purpose:** Track KPIs and filter backtesting period

**KPI Display (4):**
- `showBack` (bool): Show Backtest KPI Module in Top Table
- `backName` (string): Custom name

**Data & Lookback (4.1):**
- `backWindow` (dropdown): Entire Chart / Fixed Bars / Date Range
- `backBars` (int): Window in bars (if Fixed Bars)
- `backStartDate` (timestamp): Start date (if Date Range)
- `backUseEnd` (bool): Enable end date filter
- `backEndDate` (timestamp): End date

**Filters (4.2):**
- `backSession` (session string): Time-of-day filter
- `backUseDOW_M` through `backUseDOW_Su` (bool): Day-of-week toggles

**Tracked Metrics:**
- Total trades
- Win count / Loss count
- Win rate (%)
- Total PNL
- Profit factor
- Max drawdown (%)

---

### 5.5 Top Module 1 (Module 5)

**Purpose:** Generic module for non-timeframe-specific indicators

**Module Configuration (5):**
- `top1_enabled` (bool): Enable module
- `top1_library` (dropdown): None / Placeholder Top Library / (future libraries)
- `top1_name` (string): Custom display name
- `top1_paramA` through `top1_paramF` (float): 6 configurable parameters

**Output Configuration (5.1–5.4):**
Each of 4 outputs (O1–O4) has:
- `top1_oN_enabled` (bool): Enable this output
- `top1_oN_source` (dropdown): None / Output A / Output B / Output C / Output D
- `top1_oN_required` (bool): Make this a mandatory condition
- `top1_oN_invLbl` (bool): Invisible label (for journaling data only)
- `top1_oN_dir` (dropdown): Long / Short / Both
- `top1_oN_mode` (dropdown): Boolean / Greater Than / Less Than / Between / Equals / Library Default
- `top1_oN_valA`, `top1_oN_valB` (float): Comparison values
- `top1_oN_score` (int): TH Score contribution

**Example Use Case:**
- Library: Moving Average Crossover
- Output A: Fast MA value
- Output B: Slow MA value
- Output C: Difference (fast - slow)
- Output D: Cross direction (1 = bullish cross, -1 = bearish cross)

**O1 Configuration:**
- Source: Output D
- Mode: Greater Than
- Value A: 0
- Direction: Long
- TH Score: 50
- **Result:** Adds 50 points to Long score when bullish cross occurs

---

### 5.6 Side Module 1 (Module 6)

**Purpose:** Multi-timeframe indicator analysis (6 TFs)

**Module Configuration (6):**
- `side1_enabled` (bool): Enable module
- `side1_library` (dropdown): None / KevBot_TF_Placeholder / EMA Stack (S/M/L)
- `side1_name` (string): Custom display name
- `side1_paramA` through `side1_paramF` (float): 6 configurable parameters

**Semantic Mapping (6 - Comments Only):**
Maps library-specific conditions to generic Condition A/B and Trigger A/B:

**EMA Stack Library:**
- Condition A → S > M > L (Bull Stack)
- Condition B → L > M > S (Bear Stack)
- Trigger A → S crosses above M
- Trigger B → S crosses below M

**Placeholder Library:**
- Condition A → Placeholder Condition A
- Condition B → Placeholder Condition B
- Trigger A → Placeholder Trigger A
- Trigger B → Placeholder Trigger B

**Directional Groups (6.1–6.4):**
Each direction (LE/LX/SE/SX) has its own subsection.

**Long Entry (6.1):**
- `side1_LE_enable` (bool): Enable Long Entry signals
- `side1_LE_index` (int): Trigger slot (A/B index)
- `side1_LE_sigType` (dropdown): Chart Mark Only / Position Only / Both
- `side1_LE_bull` (bool): Use bullish label

**Long Entry Confluence (6.1.1):**
- `side1_LEC_enable` (bool): Enable LE confluence scoring
- `side1_LEC_func` (dropdown): None / Cond A / Cond B
- `side1_LEC_bull` (bool): Bullish label

**Per-TF Settings (6.1.1):**
For each TF1–TF6:
- `side1_LE_tfN` (int): TH Score
- `side1_LE_tfN_req` (bool): Required flag
- `side1_LE_tfN_inv` (bool): Invisible label

**Long Exit (6.2), Short Entry (6.3), Short Exit (6.4):**
Same structure as Long Entry, with:
- `side1_LX_*`, `side1_LXC_*` (Long Exit)
- `side1_SE_*`, `side1_SEC_*` (Short Entry)
- `side1_SX_*`, `side1_SXC_*` (Short Exit)

---

## 6. Library System

### 6.1 Library Types

**Top Libraries:**
- Return: `TopModuleOutput` (4 float outputs: A/B/C/D)
- Function: `getTopModule(paramA, paramB, paramC, paramD, paramE, paramF)`
- Examples: MA crossover, RSI levels, custom indicators

**TF Libraries:**
- Return: `TFModuleOutput` (6 TF conditions + trigger event)
- Function: `getTFConfluence(tf1_res...tf6_res, paramA...paramF)`
- Examples: EMA Stack, MACD alignment, volume profile

**Utility Libraries:**
- Return: Helper functions (e.g., `tu_fmt(tf)` for timeframe formatting)
- Examples: TimeUtils

### 6.2 Library Interface Contract

**Top Library Requirements:**
```pinescript
export type TopModuleOutput
    float outA
    float outB
    float outC
    float outD

export getTopModule(
    simple float pA, simple float pB, simple float pC,
    simple float pD, simple float pE, simple float pF
) => TopModuleOutput.new(...)
```

**TF Library Requirements:**
```pinescript
export type TFModuleOutput
    bool   tf1_cond, string tf1_label
    bool   tf2_cond, string tf2_label
    bool   tf3_cond, string tf3_label
    bool   tf4_cond, string tf4_label
    bool   tf5_cond, string tf5_label
    bool   tf6_cond, string tf6_label
    bool   trigger_event
    float  trigger_price
    string trigger_label

export getTFConfluence(
    simple string tf1_res...tf6_res,
    simple float paramA...paramF
) => TFModuleOutput.new(...)
```

### 6.3 Existing Libraries

**KevBot_TimeUtils:**
- Function: `tu_fmt(tf)` — Format timeframe strings
- Handles: Chart / Seconds / Minutes / Hours / Days / Weeks / Months
- Example: `tu_fmt("60")` → `"1H"`

**KevBot_Top_Minimal:**
- Placeholder Top module
- Params: Fast/slow lengths, adjustments, scale/bias
- Outputs: Fast MA, Slow MA, Difference, Cross direction

**KevBot_TF_Placeholder:**
- Placeholder TF module (all conditions false)
- Uses paramA–F in labels to keep compiler happy
- Trigger event: `paramA > 0`

**KevBot_TF_EMA_Stack:**
- Implements 3-EMA stack analysis (Short/Medium/Long EMAs) across 6 timeframes
- **Params:**
  - A: Short EMA length (default: 10)
  - B: Medium EMA length (default: 20)
  - C: Long EMA length (default: 50)
  - D-F: Reserved (unused)
- **Conditions (per TF):**
  - Cond A: SML (Bull Stack - S > M > L)
  - Cond B: LMS (Bear Stack - L > M > S)
  - Cond C: SLM
  - Cond D: MSL
  - Cond E: MLS
  - Cond F: LSM
  - Cond G-J: Reserved (always false)
- **Triggers (chart TF only):**
  - Trig A: Short crosses above Medium
  - Trig B: Short crosses below Medium
  - Trig C: Short crosses above Long
  - Trig D: Short crosses below Long
  - Trig E: Medium crosses above Long
  - Trig F: Medium crosses below Long
  - Trig G-J: Reserved (always false)
- **Outputs:**
  - `tfN_label`: Current EMA ordering string (e.g., "SML", "LMS")
  - `condX_tfN`: TRUE if that TF matches the specific EMA ordering
  - `trigger_price`, `trigger_label`: Metadata for active trigger

**Architecture:** Hybrid pattern - toolkit fetches HTF data, library processes via `buildOutput()`
**Status:** Working correctly in v1.1 with hybrid architecture

---

## 7. Data Flow

### 7.1 Execution Sequence

```
1. INPUT COLLECTION
   ├─ Global settings (timeframes, grades)
   ├─ Module configurations (Top1, Side1)
   └─ Position sizing & backtest settings

2. LIBRARY EXECUTION
   ├─ Top Module 1: getTopModule() → top1_raw
   └─ Side Module 1: getTFConfluence() → side1_raw

3. CONFLUENCE ENGINE (Module 9)
   ├─ Top Confluence: calcTopConfluence() → topLongScore, topShortScore
   ├─ Side Confluence: Per-TF scoring loops → sideLongScore, sideShortScore
   └─ Global Confluence: calcGlobalConfluence() → _kb_conf (grades A/B/C)

4. POSITION ENGINE (Module 10)
   ├─ Entry Method Resolution → _kb_longEntryOK, _kb_shortEntryOK
   ├─ Exit Method Resolution → _kb_longExitOK, _kb_shortExitOK
   └─ State Machine → positionLongEntry, positionShortEntry, etc.

5. BACKTEST ENGINE (Module 11)
   ├─ Track entries/exits → bt_positionActive, bt_entryPrice
   ├─ Calculate PNL → bt_realizedPNL, bt_lastTradePNL
   └─ Update KPIs → bt_winCount, bt_lossCount, bt_totalTrades

6. TABLE RENDERING (Modules 12–13)
   ├─ Top Table: Confluence Summary, Position Sizing, Backtest KPIs, Top Modules
   └─ Side Table: Per-TF conditions for Side Modules
```

### 7.2 Confluence Scoring Flow

**Step 1: Top Module Contribution**
```
For each Top Output (O1–O4):
  IF output enabled AND source selected:
    value = getTopSourceValue(source, outA, outB, outC, outD)
    passes = evalTopCondition(mode, value, valA, valB)
    
    IF passes:
      IF direction includes Long: longScore += thScore
      IF direction includes Short: shortScore += thScore
    
    IF required AND NOT passes:
      IF direction includes Long: longReqOk = false
      IF direction includes Short: shortReqOk = false
```

**Step 2: Side Module Contribution**
```
For each TF (TF1–TF6):
  IF TF enabled AND has TH Score > 0:
    condition = library's tfN_cond (from side1_raw)
    effective = invLabel ? NOT condition : condition
    
    IF effective:
      score += TH Score
    ELSE IF required:
      reqOk = false
```

**Step 3: Grade Calculation**
```
totalLongScore = topLongScore + sideLongScore
totalShortScore = topShortScore + sideShortScore

IF NOT longReqOk:
  longGrade = "Fail"
ELSE IF totalLongScore >= gradeA_long:
  longGrade = "A"
ELSE IF totalLongScore >= gradeB_long:
  longGrade = "B"
ELSE IF totalLongScore >= gradeC_long:
  longGrade = "C"
ELSE:
  longGrade = "None"
```

### 7.3 Position State Machine

```
States: FLAT (0), LONG (1), SHORT (-1)
Entry Counter: _kb_posEntries

LONG ENTRY:
  IF _kb_longEntryOK AND can_add_long:
    IF state != LONG:
      state = LONG
      entries = 1
    ELSE:
      entries += 1 (if multi-entry allowed)
    positionLongEntry = TRUE

SHORT ENTRY:
  (Similar logic, opposite direction)

LONG EXIT:
  IF state == LONG AND _kb_longExitOK:
    state = FLAT
    entries = 0
    positionLongExit = TRUE

SHORT EXIT:
  (Similar logic, opposite direction)
```

---

## 8. User Interface

### 8.1 Top Table Layout

**Location:** Top of chart (position configurable)

**Columns:**
Inherit similar structure that is currently in the code

**Color Coding:**
- Green: If the confluence state is true for a long entry
- Red: if the confluence state is true for a short entry
- Gray: Inactive for confluence calculations
- Dark theme: Configurable via `useDarkTheme`
- Blue: If the confluence state is true for both long and short entry
- Yellow: If the confluence is being monitored but its condition is not yet meeting the criteria for a long or short entry

### 8.2 Side Table Layout

**Location:** Right side of chart (position configurable)

**Rows:** One per active Side Module
**Columns:** Indicator Name + TF1 through TF6 (dynamic based on enabled TFs)

**Example:**
| Indicator | 1m | 5m | 15m | 1H | 4H | 1D |
|-----------|----|----|-----|----|----|-----|
| EMA Stack | SML(R) | MSL | SML | LSM | MLS(R) | SML |

**Cell States:**
- **Green:** Condition TRUE, contributes to Long score
- **Red:** Condition TRUE, contributes to Short score
- **Gray:** Condition FALSE or TF disabled
- **(R):** Required condition for that directional group
- **Blue:** Condition TRUE, contributes to Short AND Long score
- **Yellow:** Condition is FALSE

### 8.3 Chart Elements

#### 8.3.1 Signal Types Overview

The toolkit distinguishes between two categories of chart signals:

**Raw Triggers** - Informational markers indicating "something interesting happened here"
- Simply indicates that a library trigger fired (e.g., EMA crossover occurred)
- Does NOT imply a trade was taken
- Useful for debugging and understanding market context
- Displayed when Signal Type = "Raw Only" or "Both"

**Position Signals** - Actual trade entry/exit markers
- Indicates a position was actually entered or exited
- Requires: Raw trigger + all required conditions met + grade threshold (if configured)
- Displayed when Signal Type = "Position Only" or "Both"

#### 8.3.2 Signal Shape & Color Specifications

**Raw Trigger Marks (Entry Only):**

| Signal | Shape | Color | PineScript Shape |
|--------|-------|-------|------------------|
| Long Entry Raw | Cross (+) | Green | `shape.cross` |
| Short Entry Raw | Cross (+) | Red | `shape.cross` |

**IMPORTANT:** Raw exit triggers do NOT plot shapes. Exit marks ONLY apply to position exits. Raw exit triggers are purely informational events that may trigger labels (if enabled) but do not render chart shapes.

**Position Signal Marks:**

| Signal | Shape | Color | PineScript Shape |
|--------|-------|-------|------------------|
| Long Position Entry | Triangle Up | Green | `shape.triangleup` |
| Short Position Entry | Triangle Down | Red | `shape.triangledown` |
| Long Position Exit | XCross (X) | Green | `shape.xcross` |
| Short Position Exit | XCross (X) | Red | `shape.xcross` |

**Exit Mark Rule:** XCross (X) shapes are ONLY plotted for actual position exits. There is no visual mark for raw exit triggers since exits only have meaning in the context of an open position.

#### 8.3.3 Label System

Labels can be displayed alongside any signal (raw or position) when enabled via the Bullish/Bearish Label toggles.

**Raw Trigger Labels:**
- Display the **trigger name** for context (e.g., "S>M" for Short EMA crossed above Medium)
- Small size (`size.small`)
- Helps user understand what event occurred at that bar

**Position Signal Labels:**
- Display **full trade metadata** including:
  - Trigger name/reason
  - Confluence grade (A/B/C) - if `lbl_grade` enabled
  - Trade type (Long/Short) - if `lbl_type` enabled
  - Position quantity - if `lbl_qty` enabled
  - Dollar risk amount - if `lbl_risk` enabled
- Normal size (`size.normal`)
- Provides complete context for the trade entry/exit

**Label Color & Position Specifications:**

| Signal Type | Color | Transparency | Position | Style |
|-------------|-------|--------------|----------|-------|
| Long Entry (Raw & Position) | Green | Solid (0%) | Below candle | `label.style_label_up` |
| Long Exit (Raw & Position) | Red | 50% transparent | Above candle | `label.style_label_down` |
| Short Entry (Raw & Position) | Red | Solid (0%) | Above candle | `label.style_label_down` |
| Short Exit (Raw & Position) | Green | 50% transparent | Below candle | `label.style_label_up` |

**Color Logic Rationale:**
- Entry labels use bold/solid colors (green for bullish entry, red for bearish entry)
- Exit labels use transparent colors representing the direction the exit is TOWARD (Long Exit = transparent red because exiting means bearish action, Short Exit = transparent green because exiting means bullish action)

**Label Suppression Logic:**
To prevent duplicate labels when the same trigger fires for multiple signal types:
1. Position labels suppress corresponding raw labels (if position fires, raw label is hidden)
2. Entry labels suppress exit labels when same trigger fires for both (entry takes priority)
3. Uses `max_labels_count = 500` in indicator declaration for extended history

#### 8.3.4 Other Chart Elements

**Plotted When Enabled:**
- Safety stop loss lines (`sl_show = true`)
- Safety take profit lines (`tp_show = true`)

### 8.4 Data Export System

The toolkit includes invisible plots that capture confluence state, position sizing, and timeframe conditions for CSV export via TradingView's data export feature.

#### 8.4.1 Export Plot Specifications

All export plots use `display = display.data_window` to remain invisible on chart but appear in exported data.

**Confluence State Exports:**
| Plot Name | Description | Values |
|-----------|-------------|--------|
| Export: Long Score | Current long confluence score | 0-999 |
| Export: Long Grade | Encoded grade | 0=None, 1=C, 2=B, 3=A |
| Export: Short Score | Current short confluence score | 0-999 |
| Export: Short Grade | Encoded grade | 0=None, 1=C, 2=B, 3=A |

**Position Sizing Exports:**
| Plot Name | Description | Values |
|-----------|-------------|--------|
| Export: Long Qty | Calculated long position quantity | Float |
| Export: Short Qty | Calculated short position quantity | Float |
| Export: Risk Amount ($) | Dollar risk based on risk mode | Float |

**Timeframe Condition State Exports (TF1-TF6):**
| Plot Name | Description | Encoded Values |
|-----------|-------------|----------------|
| Export: TFn EMA State | Current EMA stack ordering | 1=SML, 2=SLM, 3=MSL, 4=MLS, 5=LSM, 6=LMS, 0=Unknown |

**Signal Marker Exports:**
| Plot Name | Description | Values |
|-----------|-------------|--------|
| Export: Pos Long Entry | Position long entry fired | 0 or 1 |
| Export: Pos Short Entry | Position short entry fired | 0 or 1 |
| Export: Pos Long Exit | Position long exit fired | 0 or 1 |
| Export: Pos Short Exit | Position short exit fired | 0 or 1 |
| Export: Long Entry Trigger Idx | Which trigger fired for long entry | 0-9 (A-J) or -1 |
| Export: Short Entry Trigger Idx | Which trigger fired for short entry | 0-9 (A-J) or -1 |

#### 8.4.2 Encoding Helper Functions

The toolkit uses helper functions to convert string values to numeric for export:

```pinescript
_kb_encodeGrade(string g) => g == "A" ? 3 : g == "B" ? 2 : g == "C" ? 1 : 0
_kb_encodeEMALabel(string lbl) => lbl == "SML" ? 1 : lbl == "SLM" ? 2 : lbl == "MSL" ? 3 : lbl == "MLS" ? 4 : lbl == "LSM" ? 5 : lbl == "LMS" ? 6 : 0
```

#### 8.4.3 CSV Export Workflow

1. Apply indicator to chart with desired configuration
2. Use TradingView's "Export Chart Data" feature (Settings > Export data)
3. Select "Indicator data" to include all plot values
4. Import CSV into Google Sheets, Trader Sync, or analysis tool
5. Decode numeric values using encoding tables above

---

## 9. Development Priorities

### 9.1 Phase 1: Core Functionality (Current)
**Status:** 90% Complete - Hybrid architecture working, 2 Side Modules functional

**Completed:**
- [x] Input skeleton (all 6 modules, ~2,006 lines)
- [x] Library loader system
- [x] Confluence Engine
- [x] Position Engine
- [x] Basic Backtest Engine
- [x] Side Table rendering (supports 2 modules)
- [x] TimeUtils library
- [x] TF_Placeholder library (V2 structure with 10 conditions/triggers)
- [x] TF_EMA_Stack library with `buildOutput()` hybrid function (v7)
- [x] Top_Minimal library
- [x] KB_TF_Out_V2 internal normalization structure
- [x] Condition/Trigger helper functions (_kb_getCondTF1-6, _kb_getTrigger)
- [x] **✅ Hybrid Architecture (v1.1)** - Toolkit owns security calls, libraries process data
- [x] **✅ Side Module 2** - Proof of concept for multi-module support
- [x] **✅ EMA Stack working correctly** - Labels display proper EMA ordering per TF
- [x] **✅ Parameters optimizable** - paramA-F work with third-party optimizers
- [x] KevBot_Types library (shared type definitions)
- [x] KevBot_Indicators library (pure processing functions)
- [x] KevBot_Library_Definitions.md (user-facing documentation)

- [x] **✅ Expand UI input options to expose all 10 triggers (A-J) and 10 conditions (A-J)** - All dropdown options now available in inputs
- [x] **✅ Label plotting system** - Raw trigger labels and position labels with full metadata support
- [x] **✅ Chart plotting system** - Raw trigger marks (cross) and position signal marks (triangles/xcross)
- [x] **✅ Data export plots** - Invisible plots for CSV export (confluence scores, grades, position sizing, TF states)
- [x] **✅ Default input values** - Sensible defaults so signals display on indicator load
- [x] **✅ Side table color fix** - Independent evaluation of Long/Short confluence conditions

**Remaining:**
- [ ] Complete backtest KPI calculations
- [ ] CSV export functionality (data export plots ready, need export mechanism)
- [ ] Add Side Modules 3-10 (follow existing pattern)

### 9.2 Phase 2: Enhancement & Testing
**Status:** Not Started

**Planned:**
- [ ] Multiple Top Modules (Top2, Top3, Top4)
- [ ] Multiple Side Modules (Side2, Side3, Side4)
- [ ] Advanced entry/exit conditions such as supporting multiple buys within the same trade
- [ ] Alert system
- [ ] Additional libraries (MACD, RSI, Volume, etc.)

### 9.3 Phase 3: Advanced Features
**Status:** Future

**Planned:**
- [ ] Machine learning integration
- [ ] Strategy optimizer
- [ ] Risk management automation
- [ ] Portfolio-level analytics

### 9.4 Companion Tools

#### Trade Analyzer (Python/Streamlit) - POC Complete
**Location:** `tools/trade_analyzer/`
**Status:** ✅ Proof of concept complete, ready for toolkit integration

A web-based analysis tool that processes CSV exports from the toolkit to find optimal confluence combinations.

**Key Features:**
- **Confluence Record Analysis**: Atomic units combining Timeframe + Evaluator + State (e.g., "1M-EMA-SML")
- **Drill-Down Mode**: Iteratively explore which factors pair well together
- **Auto-Search Mode**: Combinatorial search for best N-factor combinations
- **Financial Modeling**: Fixed vs Compounding risk modes, starting balance, daily P&L in both R and $
- **Export TradingView Parameters**: Placeholder for auto-configuring indicator settings based on analysis results

**Integration Points:**
- Reads CSV data matching toolkit's export format (Section 8.4)
- Future: Export button will generate toolkit input parameters to match discovered confluence combinations

**Tech Stack:** Python, Streamlit, Pandas, Plotly (~800 lines)

---

## 10. Testing & Validation

### 10.1 Unit Testing

**Library Testing:**
- Verify TFModuleOutput structure matches spec
- Validate parameter handling (A–F)
- Test timeframe resolution logic
- Confirm trigger event accuracy

**Confluence Engine Testing:**
- Verify TH Score accumulation
- Test required condition logic
- Validate grade thresholds
- Check Long/Short separation

**Position Engine Testing:**
- Verify state transitions (FLAT → LONG → FLAT)
- Test multi-entry stacking
- Validate entry/exit method resolution

### 10.2 Integration Testing

**Full Workflow:**
1. Load indicator on 1-minute chart
2. Configure 6 timeframes (1m, 5m, 15m, 1H, 4H, 1D)
3. Enable EMA Stack library with params (10, 20, 50)
4. Set Grade thresholds (200, 250, 300)
5. Configure position sizing (Fixed $100 risk)
6. Generate signals and verify:
   - Confluence scores update correctly
   - Grades calculate properly
   - Position entries/exits trigger
   - Backtest KPIs accumulate
   - Tables render accurately

### 10.3 Validation Criteria

**Table Rendering:**
- All enabled modules appear in correct order
- Cell colors match condition states
- Labels display correct TF formatting
- Required flags show "(R)" suffix

**Confluence Accuracy:**
- Manual score calculation matches engine output
- Grade boundaries respected
- Required conditions properly block entries

**Position Logic:**
- No simultaneous Long/Short positions
- Entry counts respect `sizerMaxEntries`
- Exits only occur when position active

---

## 11. Known Issues & Blockers

### 11.1 Critical Issues

**1. ✅ RESOLVED: EMA Stack Library - Incorrect Label Output**
- **Original Issue:** Side Table displayed "LMS" for all timeframes regardless of actual EMA ordering
- **Root Cause:** PineScript library limitations prevented dynamic `request.security()` calls
- **Solution Implemented:** Hybrid Architecture in v1.1
  - Toolkit owns all `request.security()` calls with user-configurable params
  - Library provides `buildOutput()` function to process pre-fetched EMA data
  - Parameters (paramA = Short EMA, paramB = Medium EMA, paramC = Long EMA) are now fully optimizable
- **Status:** ✅ **RESOLVED** in v1.1 Hybrid Architecture
- **Verification:** Side Table now correctly displays EMA stack labels (SML, LMS, MSL, etc.) per timeframe

**2. ✅ RESOLVED: Label Plotting Incomplete**
- **Original Issue:** Entry/exit signals not visible on chart
- **Solution Implemented:** Complete label system with raw trigger labels and position labels
  - Raw labels show trigger names (e.g., "S>M")
  - Position labels show full metadata (grade, qty, risk, trade type)
  - Label colors/positions follow directional semantics
  - Suppression logic prevents duplicate labels
  - `max_labels_count = 500` for extended history
- **Status:** ✅ **RESOLVED** in v1.1

**3. Partial: CSV Export**
- **Impact:** Cannot fully analyze data in external tools
- **Current Status:** Data export plots implemented (Section 15 in code)
  - Invisible plots capture confluence scores, grades, position sizing, TF states
  - Signal markers for all entry/exit events
  - Encoding functions convert strings to numeric values
- **Remaining:** TradingView's export mechanism needed for actual CSV download
- **Workaround:** Use TradingView's native "Export Chart Data" feature

**4. ✅ RESOLVED: UI Trigger/Condition Options Limited**
- **Original Issue:** Input settings only showed subset of available trigger/condition options
- **Solution Implemented:** All 10 triggers (A-J) and 10 conditions (A-J) now available in UI dropdowns
- **Status:** ✅ **RESOLVED** in v1.1

### 11.2 Non-Critical Issues

**1. Limited Module Count**
- Currently supports Top Module 1 and Side Modules 1-2
- Side Module 2 added as proof of concept for multi-module support
- Future: Expand to Top2–Top4, Side3–Side10 (following same hybrid pattern)


### 11.3 Technical Debt

**1. Code Duplication**
- Per-TF scoring loops repeat similar logic
- **Solution:** Refactor into reusable functions (may hit PineScript limitations)

**2. Magic Numbers**
- Grade thresholds and multipliers scattered across code
- **Solution:** Centralize configuration constants

**3. Limited Error Handling**
- No validation for invalid parameter combinations
- **Solution:** Add runtime checks with user warnings

---

## 12. Success Criteria

### 12.1 Minimum Viable Product (MVP)

**Must Have:**
- [x] 6 configurable timeframes with enable/disable toggles
- [x] Confluence scoring with A/B/C grades (Long + Short)
- [x] Position state machine (entries/exits/multi-entry)
- [ ] Top Table rendering (Confluence Summary + Position Sizing + Backtest KPIs)
- [x] Side Table rendering (1 Side Module across 6 TFs)
- [x] At least 1 functional TF library (EMA Stack ✓)
- [x] Chart plotting for entries/exits (raw trigger marks + position signal marks)
- [x] Label system with trigger names and full position metadata
- [x] Data export plots for CSV analysis (confluence, position sizing, TF states)
- [ ] Basic backtest KPIs (win rate, PF, total trades) using single entry trades instead of multientries
- [ ] CSV file export mechanism (data plots ready, need TradingView export)

### 12.2 Full Feature Set

**Should Have:**
- [ ] Multiple Top Modules (4 total)
- [ ] Multiple Side Modules (4 total)
- [ ] Alert system for signals
- [ ] Complete backtest filters (date range, session, DOW)
- [ ] Safety stop loss / take profit plotting
- [ ] Multientry capability for trades (don't implement without authorization)


### 12.3 Performance Benchmarks

**Execution:**
- Script compilation: < 5 seconds
- Real-time updates: < 500ms per bar
- Table rendering: < 1 second for 6 TFs

**Accuracy:**
- Confluence scores: 100% match manual calculation
- Position tracking: Zero state errors over 10,000 bars
- Backtest PNL: Within 0.01% of manual calculation

### 12.4 User Experience

**Setup Time:**
- New user can configure basic strategy in < 10 minutes
- Library swap takes < 2 minutes

**Visual Clarity:**
- All table cells readable without zooming
- Color coding intuitive (green = bullish, red = bearish)
- Required conditions clearly marked with "(R)"

---

## Appendix A: Code Snippets

### A.1 Confluence Engine Entry Point

```pinescript
// 9.3 Global confluence
ConfluenceResult _kb_conf = calcGlobalConfluence(
    topLongScore, topShortScore,
    topReqLongOk, topReqShortOk,
    sideLongScore, sideShortScore,
    sideReqLongOk, sideReqShortOk,
    gradeC_long, gradeB_long, gradeA_long,
    gradeC_short, gradeB_short, gradeA_short
)

// 9.4 Expose for other modules
float  _kb_longScore   = _kb_conf.longScore
float  _kb_shortScore  = _kb_conf.shortScore
bool   _kb_longReqOk   = _kb_conf.longReqOk
bool   _kb_shortReqOk  = _kb_conf.shortReqOk
string _kb_longGrade   = _kb_conf.longGrade
string _kb_shortGrade  = _kb_conf.shortGrade
```

### A.2 Position State Machine

```pinescript
// Position state
var int _kb_posDir     = 0  // 0=flat, 1=long, -1=short
var int _kb_posEntries = 0

// Long entry
if _kb_longEntryOK and _kb_canAddLong
    positionLongEntry := true
    if _kb_posDir != 1
        _kb_posDir := 1
        _kb_posEntries := 1
    else
        _kb_posEntries += 1

// Long exit
if _kb_posDir == 1 and _kb_longExitOK
    positionLongExit := true
    _kb_posDir := 0
    _kb_posEntries := 0
```

### A.3 Side Table Cell Population

```pinescript
// Per-TF scoring loop
for i = 0 to _kb_side_tfCount - 1
    int tfIdx = array.get(_kb_side_tfIndex, i)
    int col   = 1 + i
    
    int  leScore   = tfIdx == 1 ? side1_LE_tf1 : ... // (ternary chain)
    bool tfCond    = tfIdx == 1 ? _side1_tf1_cond : ...
    string tfLabel = tfIdx == 1 ? side1_raw.tf1_label : ...
    bool reqLong   = tfIdx == 1 ? side1_LE_tf1_req : ...
    
    color cellBg = f_sideTF_color(tfCond, leScope, seScope, leScore, seScore)
    string cellText = reqLong ? tfLabel + " (R)" : tfLabel
    
    f_sideTable_cell(_kb_sideTable, col, _kb_moduleRow, cellText, color.white, cellBg)
```

---

## Appendix B: Library Development Guide

### B.1 Creating a New TF Library

**Step 1: Define Output Type (V2 Structure - MUST MATCH)**
```pinescript
//@version=6
library("MyCustomTFLibrary", overlay = false)

export type TFModuleOutput
    // Per-TF labels (state description)
    string tf1_label, tf2_label, tf3_label, tf4_label, tf5_label, tf6_label

    // Condition A per TF (define your primary bullish condition)
    bool condA_tf1, condA_tf2, condA_tf3, condA_tf4, condA_tf5, condA_tf6

    // Condition B per TF (define your primary bearish condition)
    bool condB_tf1, condB_tf2, condB_tf3, condB_tf4, condB_tf5, condB_tf6

    // Conditions C-F per TF (additional conditions as needed)
    bool condC_tf1, condC_tf2, condC_tf3, condC_tf4, condC_tf5, condC_tf6
    bool condD_tf1, condD_tf2, condD_tf3, condD_tf4, condD_tf5, condD_tf6
    bool condE_tf1, condE_tf2, condE_tf3, condE_tf4, condE_tf5, condE_tf6
    bool condF_tf1, condF_tf2, condF_tf3, condF_tf4, condF_tf5, condF_tf6

    // Conditions G-J per TF (reserved, set to false)
    bool condG_tf1, condG_tf2, condG_tf3, condG_tf4, condG_tf5, condG_tf6
    bool condH_tf1, condH_tf2, condH_tf3, condH_tf4, condH_tf5, condH_tf6
    bool condI_tf1, condI_tf2, condI_tf3, condI_tf4, condI_tf5, condI_tf6
    bool condJ_tf1, condJ_tf2, condJ_tf3, condJ_tf4, condJ_tf5, condJ_tf6

    // Triggers A-J (chart timeframe events)
    bool trigA, trigB, trigC, trigD, trigE, trigF, trigG, trigH, trigI, trigJ

    // Trigger metadata
    float trigger_price
    string trigger_label
```

**IMPORTANT:** The TFModuleOutput structure MUST match exactly between all TF libraries. Once the EMA Stack library bug is fixed, `KevBot_TF_EMA Stack.txt` will serve as the canonical reference template for new TF libraries. The main toolkit casts library outputs to a common type for normalization.

**Step 2: Implement Core Logic**
```pinescript
export getTFConfluence(
    simple string tf1_res, simple string tf2_res, simple string tf3_res,
    simple string tf4_res, simple string tf5_res, simple string tf6_res,
    simple float  paramA,  simple float paramB,  simple float paramC,
    simple float  paramD,  simple float paramE,  simple float paramF
) =>
    // Per-TF analysis
    float c1 = request.security(syminfo.tickerid, tf1_res, close)
    // ... compute conditions ...
    
    TFModuleOutput.new(
        cond1, label1,
        cond2, label2,
        cond3, label3,
        cond4, label4,
        cond5, label5,
        cond6, label6,
        trigger, price, trigLabel
    )
```

**Step 3: Publish & Import**
```pinescript
// In main Toolkit script:
import username/MyCustomTFLibrary/1 as myLib

// In module loader:
if side1_library == "My Custom Library"
    myLib.TFModuleOutput _out = myLib.getTFConfluence(...)
    side1_raw := side1Lib.TFModuleOutput.new(
        _out.tf1_cond, _out.tf1_label,
        // ... map all fields ...
    )
```

### B.2 Parameter Usage Guidelines

**Recommended Mapping:**
- **Param A:** Primary length/period (e.g., fast EMA)
- **Param B:** Secondary length/period (e.g., slow EMA)
- **Param C:** Tertiary length/period (e.g., longest EMA)
- **Param D:** Mode/trigger selector (enum-like: 0, 1, 2...)
- **Param E:** Condition selector (enum-like)
- **Param F:** Multiplier/threshold (continuous value)

**Example (EMA Stack):**
- A: Short EMA length (default: 10)
- B: Medium EMA length (default: 20)
- C: Long EMA length (default: 50)
- D: Trigger selector (0=None, 1=SxM Up, 2=SxM Down...)
- E: Confluence selector (0=SML, 1=SLM, 2=MSL...)
- F: Reserved for future use

---

## Appendix C: Troubleshooting Guide

### C.1 Common Issues

**Issue:** Table not appearing
- **Check:** `enableTopTable` / `enableSideTable` inputs
- **Check:** Module enabled flags
- **Check:** TradingView chart scale (tables may render off-screen)

**Issue:** Confluence score always zero
- **Check:** At least one module enabled
- **Check:** Output sources selected (not "None")
- **Check:** TH Scores > 0
- **Check:** Library returning valid conditions

**Issue:** Position never enters
- **Check:** `sizer_entry` method matches confluence grade
- **Check:** Required conditions not blocking entry
- **Check:** Raw trigger events firing (use `plot()` to debug)

**Issue:** Multi-entry not stacking
- **Check:** `sizerAllowMulti = true`
- **Check:** `sizerMaxEntries > 1`
- **Check:** Position already in correct direction

### C.2 Debugging Techniques

**1. Inspect Raw Library Outputs:**
```pinescript
// Add temporary plots
plot(side1_raw.tf1_cond ? 1 : 0, "TF1 Cond", display = display.data_window)
plot(side1_raw.trigger_event ? 1 : 0, "Trigger", display = display.data_window)
```

**2. Trace Confluence Scores:**
```pinescript
// Add to script
plot(_kb_longScore, "Long Score", color = color.green)
plot(_kb_shortScore, "Short Score", color = color.red)
```

**3. Validate Position State:**
```pinescript
// Add label on every bar
label.new(bar_index, high, 
    "Pos: " + str.tostring(_kb_posDir) + 
    " | Entries: " + str.tostring(_kb_posEntries),
    style = label.style_label_down)
```


---

## Appendix E: Glossary

**Confluence:** Agreement between multiple indicators/timeframes supporting a trade direction

**TH Score (Threshold Score):** Point value assigned to each condition; accumulates to determine grades

**Grade:** Quality rating (A/B/C) based on total TH Score meeting defined thresholds

**Required Condition:** Mandatory condition that must be TRUE; blocks entries/exits if FALSE

**Inv Label (Invisible Label):** Data captured for export but not plotted visibly on chart

**Top Module:** Non-timeframe-specific indicator module (e.g., RSI, news events)

**Side Module:** Multi-timeframe indicator module (e.g., EMA alignment across 6 TFs)

**Position Engine:** State machine managing entry/exit logic and position tracking

**Multi-Entry:** Ability to stack multiple entries into a single trade (pyramiding)

**Safety Stop:** Protective stop-loss independent of strategy exit signals

**Backtest KPI:** Key Performance Indicator tracked during backtesting (win rate, PF, etc.)

**Library:** Self-contained calculation module with standardized output structure

**Signal Type:** Classification of signal visibility (Chart Mark Only, Position Only, Both)

**Directional Group:** Category of signals (Long Entry, Long Exit, Short Entry, Short Exit)

---

## Appendix F: Development Workflow for Claude Code

### F.1 Recommended Approach

**Phase 1: Familiarization (Day 1)**
1. Read entire PRD (this document) and Claude Transition Document
2. Review all 5 uploaded codebase files:
   - `Kevbot_Toolkit_v1_0_-_Input_Skeleton.txt` (main indicator)
   - `KevBot_TimeUtils.txt`
   - `KevBot_Top_Minimal.txt`
   - `KevBot_TF_Placeholder.txt`
   - `KevBot_TF_EMA_Stack.txt`
3. Understand module dependencies and data flow
4. Identify incomplete sections (Trigger and Confluence Options)

**Phase 2: Critical Path (Days 2-3)**
1. **Label Plotting System**
   - Plot entry labels with metadata (grade, qty, risk)
   - Plot exit labels with PNL
   - Respect `lbl_*` input toggles
   - Handle invisible labels for data export

2. **Complete Backtest KPIs**
   - Profit factor calculation
   - Max drawdown tracking
   - Average win/loss
   - Expectancy

**Phase 3: Testing & Validation (Day 4)**
1. Create test cases for each module
2. Verify confluence calculations manually
3. Test position state machine edge cases
4. Validate table rendering across different chart sizes
5. Check backtest accuracy vs. manual calculation

**Phase 4: Documentation & Polish (Day 5)**
1. Add inline comments for complex sections
2. Create user guide (separate document)
3. Document any deviations from PRD
4. List known limitations
5. Prepare example configurations

### F.2 Code Quality Standards

**Naming Conventions:**
- Inputs: descriptive, grouped logically
- Internal state: `_kb_` prefix
- Functions: camelCase starting with lowercase
- Types: PascalCase

**Comments:**
- Section headers: `//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
- Subsections: `//──────────────────────────────────────────────────────────────────────────────`
- Inline: Single-line `//` for logic clarification
- Block: `/* ... */` for multi-line explanations

**Structure:**
- Keep functions under 50 lines when possible
- Use helper functions to avoid repetition
- Group related logic into clearly marked sections
- Maintain consistent indentation (4 spaces)

**PineScript-Specific:**
- Declare `var` for persistent state
- Use `simple` qualifier for security() calls
- Avoid forward references (declare before use)
- Minimize use of `if` inside loops (use arrays/ternaries)

### F.3 Testing Checklist

**Before Each Commit:**
- [ ] Script compiles without errors
- [ ] No runtime errors on 1-minute chart (AAPL)
- [ ] Tables render correctly
- [ ] Position state machine behaves as expected
- [ ] Confluence scores match manual calculation

**Before Milestone Completion:**
- [ ] Test on multiple symbols (stocks, forex, crypto)
- [ ] Test on multiple timeframes (1m, 1H, 1D)
- [ ] Verify backtest matches TradingView Strategy Tester
- [ ] Check performance with all modules enabled
- [ ] Validate library swapping (Placeholder → EMA Stack)

---

## Appendix G: Contact & Support

**Project Owner:** Kevin (Matthew as intermediary)  
**Development Tool:** Claude Code  
**Communication Channel:** This PRD + codebase files  

**For Questions:**
- Refer to PRD first
- Check codebase comments
- Review library SKILL.md files (if applicable)
- Document assumptions made during development

**Deliverables:**
1. Updated main indicator file (`.pine` or `.txt`)
2. Any new library files
3. Test results summary
4. Known issues / limitations list
5. User guide (basic setup instructions)

---

## Document Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-20 | Initial PRD creation | Claude (Anthropic) |
| 1.1 | 2026-01-26 | Updated to reflect current state: corrected file names/line counts, documented V2 expanded TFModuleOutput structure (10 conditions/triggers), added KB_TF_Out_V2 internal structure, documented critical EMA Stack label bug with root cause analysis, updated development status | Claude Code (Opus 4.5) |
| 1.2 | 2026-01-26 | **Major Update:** Hybrid Architecture v1.1 implemented and working. EMA Stack bug resolved via hybrid pattern (toolkit owns security calls, library processes data). Side Module 2 added as proof of concept. Created KevBot_Library_Definitions.md for user-facing documentation. Renamed legacy files with LEGACY prefix. Updated all sections to reflect current working state. | Claude Code (Opus 4.5) |
| 1.3 | 2026-01-27 | **Feature Complete Update:** (1) Chart plotting system: raw trigger marks (cross +) for entries, position signal marks (triangles for entries, xcross for exits). (2) Label system: raw labels with trigger names, position labels with full metadata, color/position semantics, suppression logic, max_labels_count=500. (3) Data export plots: invisible plots for CSV export including confluence scores, grades, position sizing, TF EMA states, signal markers. (4) UI expansion: all 10 triggers (A-J) and 10 conditions (A-J) exposed in dropdowns. (5) Default input values: sensible defaults so signals display on load. (6) Side table color fix: independent Long/Short confluence condition evaluation. Updated Section 8.3, added Section 8.4 Data Export, updated Section 9.1 completed items, marked resolved issues in Section 11. | Claude Code (Opus 4.5) |
| 1.4 | 2026-01-30 | Added Section 9.4 Companion Tools documenting the Trade Analyzer POC (Python/Streamlit). Tool analyzes CSV exports to find optimal confluence combinations using "confluence records" (TF-Evaluator-State). Features: drill-down analysis, auto-search, financial modeling (fixed/compounding risk), placeholder for TradingView parameter export. | Claude Code (Opus 4.5) |

---

**END OF DOCUMENT**

Total Pages: ~35 (estimated)  
Total Words: ~8,500  
Sections: 12 + 7 Appendices