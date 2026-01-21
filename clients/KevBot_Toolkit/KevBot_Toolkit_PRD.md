# KevBot Toolkit - Product Requirements Document (PRD)

**Version:** 1.0  
**Date:** January 20, 2026  
**Target Platform:** TradingView (PineScript v6)  
**Development Tool:** Claude Code  

---

## Executive Summary

The KevBot Toolkit is a modular TradingView indicator designed to automate trade journaling and enable multivariate analysis of trading strategies. It replaces manual data entry (80+ questions per trade) with automatic context capture across multiple timeframes and indicators, displaying results in two dynamic tables.

**Current Development State:** Input skeleton complete (~1,700 lines), 4 library files functional, stalled due to AI context window limitations.

**Goal of This PRD:** Enable Claude Code to accelerate development with full codebase awareness, resolving the context window blocker that has limited progress.

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
- **Position Engine:** Multi-entry position tracking with safety stops
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
- `Kevbot_Toolkit_v1_0_Main.txt` — Core indicator (1,881 lines currently)

**Library Files:**
- `KevBot_TimeUtils.txt` — Timeframe formatting utilities
- `KevBot_Top_Minimal.txt` — Placeholder Top module (4 outputs: A/B/C/D)
- `KevBot_TF_Placeholder.txt` — Placeholder TF module (6 TF conditions + trigger)
- `KevBot_TF_EMA_Stack.txt` — EMA Stack implementation (S/M/L ordering)

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

**TFModuleOutput (Library Standard):**
```pinescript
type TFModuleOutput
    bool   tf1_cond, tf2_cond, tf3_cond, tf4_cond, tf5_cond, tf6_cond
    string tf1_label, tf2_label, tf3_label, tf4_label, tf5_label, tf6_label
    bool   trigger_event
    float  trigger_price
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
- Implements 3-EMA stack analysis (Short/Medium/Long EMAs)
- Params:
  - A: Short EMA length
  - B: Medium EMA length
  - C: Long EMA length
  - D: Trigger selector (0=None, 1=SxM Up, 2=SxM Down, 3=SxL Up, etc.)
  - E: Confluence selector (0=SML, 1=SLM, 2=MSL, 3=MLS, 4=LSM, 5=LMS)
  - F: (unused, reserved)
- Outputs:
  - `tfN_cond`: TRUE if EMA order matches selected confluence
  - `tfN_label`: Current order (e.g., "SML", "LMS")
  - `trigger_event`: TRUE on selected EMA cross

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
| Module Name | Value/Status | Additional Info |
|-------------|-------------|-----------------|
| Long Confluence | Score: 250 | Grade: B |
| Short Confluence | Score: 150 | Grade: C |
| Position Sizing | Active: LONG | Qty: 100, Risk: $200 |
| Backtest KPIs | Win Rate: 62% | PF: 1.8, Trades: 45 |
| Top Module 1 | [Custom display] | [Library outputs] |

**Color Coding:**
- Green: Bullish/Long signals
- Red: Bearish/Short signals
- Gray: Neutral/Inactive
- Dark theme: Configurable via `useDarkTheme`

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

### 8.3 Chart Elements

**Plotted When Enabled:**
- Position labels (entry/exit markers)
- Safety stop loss lines (`sl_show = true`)
- Safety take profit lines (`tp_show = true`)
- Raw chart marks (when Signal Type = "Chart Mark Only" or "Both")

---

## 9. Development Priorities

### 9.1 Phase 1: Core Functionality (Current)
**Status:** 90% Complete

**Completed:**
- [x] Input skeleton (all 6 modules)
- [x] Library loader system
- [x] Confluence Engine
- [x] Position Engine
- [x] Basic Backtest Engine
- [x] Side Table rendering
- [x] TimeUtils library
- [x] TF_Placeholder library
- [x] TF_EMA_Stack library
- [x] Top_Minimal library

**Remaining:**
- [ ] Top Table rendering (Module 12)
- [ ] Complete backtest KPI calculations
- [ ] Label plotting system
- [ ] CSV export functionality

### 9.2 Phase 2: Enhancement & Testing
**Status:** Not Started

**Planned:**
- [ ] Multiple Top Modules (Top2, Top3, Top4)
- [ ] Multiple Side Modules (Side2, Side3, Side4)
- [ ] Advanced entry/exit conditions
- [ ] Alert system
- [ ] Additional libraries (MACD, RSI, Volume, etc.)

### 9.3 Phase 3: Advanced Features
**Status:** Future

**Planned:**
- [ ] Machine learning integration
- [ ] Strategy optimizer
- [ ] Risk management automation
- [ ] Portfolio-level analytics

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

**1. Top Table Not Yet Implemented**
- **Impact:** Cannot display confluence summary, position sizing, or backtest KPIs
- **Status:** Module 12 exists as stub, needs full implementation
- **Blocker for:** User-facing functionality, CSV export

**2. Label Plotting Incomplete**
- **Impact:** Entry/exit signals not visible on chart
- **Status:** Position labels defined but not plotted
- **Blocker for:** Visual confirmation of signals

**3. CSV Export Not Implemented**
- **Impact:** Cannot analyze data in external tools
- **Status:** No export mechanism exists yet
- **Blocker for:** Primary use case (multivariate analysis)

### 11.2 Non-Critical Issues

**1. Single Module Limitation**
- Currently only supports Top Module 1 and Side Module 1
- Future: Expand to Top2–Top4, Side2–Side4

**2. Library Selection Hardcoded**
- Library dropdown options manually maintained
- Future: Auto-discover published libraries

**3. Limited Backtest Filters**
- Only time-of-day and day-of-week filters
- Future: Add session templates, holiday filters

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
- [ ] Chart label plotting for entries/exits
- [ ] Basic backtest KPIs (win rate, PF, total trades)

### 12.2 Full Feature Set

**Should Have:**
- [ ] Multiple Top Modules (4 total)
- [ ] Multiple Side Modules (4 total)
- [ ] CSV export functionality
- [ ] Alert system for signals
- [ ] Complete backtest filters (date range, session, DOW)
- [ ] Safety stop loss / take profit plotting
- [ ] Position labels with full metadata

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

**Step 1: Define Output Type**
```pinescript
//@version=6
library("MyCustomTFLibrary", overlay = false)

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
```

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

## Appendix D: Future Enhancements

### D.1 Planned Features (Post-MVP)

**Advanced Confluence:**
- Exit-specific grades (separate from entry)
- Multiple confluence profiles (aggressive, conservative, balanced)
- Dynamic grade thresholds based on volatility
- Weighted TF scoring (closer TFs = higher weight)

**Library Ecosystem:**
- Auto-discovery of published libraries
- Library marketplace integration
- Version management for library updates
- Library testing framework

**Analytics & Export:**
- Multi-trade correlation analysis
- Parameter optimization suggestions
- Real-time performance metrics
- Integration with external journaling tools (Edgewonk, TraderSync)

**Risk Management:**
- Dynamic position sizing based on volatility
- Portfolio heat tracking
- Correlation-based exposure limits
- Drawdown-based position reduction

### D.2 Community Requests

**High Priority:**
- [ ] Multiple chart mark styles (shapes, colors, sizes)
- [ ] Customizable table positions per module
- [ ] Conditional alerts (only when Grade A, etc.)
- [ ] Replay mode compatibility

**Medium Priority:**
- [ ] Dark/light theme auto-detection
- [ ] Mobile-optimized table layouts
- [ ] Multi-symbol backtesting
- [ ] Strategy templates (pre-configured setups)

**Low Priority:**
- [ ] Custom color palettes
- [ ] Sound alerts for specific grades
- [ ] Integration with Pine v6 strategies
- [ ] Historical heatmaps

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
1. Read entire PRD (this document)
2. Review all 5 uploaded codebase files:
   - `Kevbot_Toolkit_v1_0_-_Input_Skeleton.txt` (main indicator)
   - `KevBot_TimeUtils.txt`
   - `KevBot_Top_Minimal.txt`
   - `KevBot_TF_Placeholder.txt`
   - `KevBot_TF_EMA_Stack.txt`
3. Understand module dependencies and data flow
4. Identify incomplete sections (Module 12 - Top Table)

**Phase 2: Critical Path (Days 2-3)**
1. **Top Table Implementation (Module 12)**
   - Create table structure (similar to Side Table)
   - Add Confluence Summary row (Long + Short scores/grades)
   - Add Position Sizing row (current position state)
   - Add Backtest KPI row (win rate, PF, trades)
   - Add Top Module 1 row (if enabled)

2. **Label Plotting System**
   - Plot entry labels with metadata (grade, qty, risk)
   - Plot exit labels with PNL
   - Respect `lbl_*` input toggles
   - Handle invisible labels for data export

3. **Complete Backtest KPIs**
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

---

**END OF DOCUMENT**

Total Pages: ~35 (estimated)  
Total Words: ~8,500  
Sections: 12 + 7 Appendices