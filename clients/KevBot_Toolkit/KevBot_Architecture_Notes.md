# KevBot Toolkit - Architecture Notes & Decisions

**Created:** January 26, 2026
**Purpose:** Document architectural decisions, PineScript limitations, and implementation approaches

---

## Table of Contents

1. [Critical PineScript Limitation](#1-critical-pinescript-limitation)
2. [Architecture Options Evaluated](#2-architecture-options-evaluated)
3. [Option A Failed: Additional PineScript Limitation](#3-option-a-failed-additional-pinescript-limitation)
4. [Revised Architecture: Hybrid Inline Approach](#4-revised-architecture-hybrid-inline-approach)
5. [File Size Strategy for 10 Side Modules](#5-file-size-strategy-for-10-side-modules)
6. [Revised Decision: Hybrid Inline + Library Processing](#6-revised-decision-hybrid-inline--library-processing)
7. [**v1.1 Implementation Status (CURRENT)**](#7-v11-implementation-status-current)
8. [Original Section Preserved (OUTDATED)](#8-original-section-preserved-outdated)
9. [Change Log](#9-change-log)
10. [Appendix: Error Messages Reference](#appendix-error-messages-reference)
11. [Appendix: What Libraries CAN and CANNOT Do](#appendix-what-libraries-can-and-cannot-do)

---

## 1. Critical PineScript Limitation

### The Problem

PineScript libraries have a fundamental restriction: **`request.security()` expressions inside exported functions CANNOT depend on function arguments**, even if those arguments are declared as `simple` type.

```pinescript
// THIS DOES NOT WORK
export getTFConfluence(simple float paramA, ...) =>
    request.security(syminfo.tickerid, tf, ta.ema(close, int(paramA)), ...)
    // ERROR: expression cannot depend on arguments of exported function
```

### Why This Happens

1. TradingView compiles libraries **separately** from the importing script
2. At library compile time, argument values are unknown
3. TradingView must pre-register ALL `request.security` calls at compile time
4. Therefore, security expressions must be determinable at library compile time

### Additional Discovery: `var` Variables Don't Persist

During debugging, we also discovered that `var` variables inside library exported functions **do not persist between bars**. This means manual indicator calculations (like EMA with persistent state) cannot be implemented inside library functions.

```pinescript
// THIS DOES NOT WORK AS EXPECTED
export myFunction() =>
    var float ema = na  // Re-initialized every bar, doesn't persist!
    ema := na(ema) ? close : alpha * close + (1 - alpha) * ema
```

---

## 2. Architecture Options Evaluated

### Option A: Library-Level Inputs (CHOSEN)

Each library declares its own `input.*()` at script level (outside functions).

**Pros:**
- Full continuous parameter optimization
- True modularity - each library is self-contained
- No toolkit changes needed for new libraries
- Works with third-party optimizers

**Cons:**
- Multiple input sections in TradingView UI
- paramA-F from toolkit become unused/redundant for some libraries
- UI slightly more complex

### Option B: Preset-Based Selection

Libraries pre-calculate multiple configurations, paramA selects which preset.

**Pros:**
- Clean unified UI with paramA-F
- Works within PineScript limitations
- Optimizer can sweep discrete presets

**Cons:**
- Limited to predefined presets
- Not continuous optimization
- Adding presets requires library updates

### Option C: Toolkit Handles Data Fetching

Move `request.security` calls to toolkit, library just processes values.

**Pros:**
- paramA-F work as intended

**Cons:**
- Breaks modularity - toolkit needs library-specific code
- Adding libraries requires toolkit changes
- Not scalable

### Option D: All Calculation in Toolkit

Eliminate libraries for TF modules entirely.

**Cons:**
- No modularity
- Toolkit becomes monolithic
- Not maintainable

---

## 3. Option A Failed: Additional PineScript Limitation

### The Third Limitation

After implementing Option A (Library-Level Inputs), we discovered a **third PineScript limitation**:

```pinescript
// THIS ALSO DOES NOT WORK
int EMA_SHORT = input.int(10, "Short EMA", group="EMA Stack Settings")

export getTFConfluence(...) =>
    // ERROR: exported function depends on input variable
    ta.ema(close, EMA_SHORT)
```

**Error Message:** `The exported function 'getTFConfluence' depends on the 'EMA_SHORT' input variable, which is not allowed.`

### Summary of All PineScript Library Limitations

| Limitation | Impact |
|------------|--------|
| `request.security()` can't use function args | Can't pass EMA lengths from toolkit |
| `var` doesn't persist in library functions | Can't do manual EMA calculations |
| Exported functions can't use `input.*()` | Can't use library-level inputs for indicator params |

**Conclusion:** PineScript libraries fundamentally **cannot** provide configurable, HTF-aware indicator modules. The only workaround is using fixed constants.

---

## 4. Revised Architecture: Hybrid Inline Approach

### File Size Analysis

| Metric | Value |
|--------|-------|
| Current toolkit (total) | 122,301 chars |
| Without comments/whitespace | ~78,446 chars |
| **TradingView limit** | ~60,000 chars |
| **Status** | **30% OVER LIMIT** |

### What Libraries CAN Still Do

Despite security-related limitations, libraries can:
- **Define shared types** (TFModuleOutput, ConfluenceResult)
- **Process already-fetched data** (evaluate conditions, compare values)
- **Provide utility functions** (formatting, color helpers)
- **Encapsulate non-security logic** (scoring, grading)

### Recommended Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN TOOLKIT                             │
│  • All request.security() calls (toolkit owns HTF fetching) │
│  • All input.*() declarations (toolkit owns all params)     │
│  • Table rendering, alerts, position logic                  │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│KevBot_Types     │  │KevBot_Indicators│  │KevBot_Utils     │
│• TFModuleOutput │  │• orderEMA()     │  │• tu_fmt()       │
│• ConfluenceRes  │  │• evalRSI()      │  │• color helpers  │
│• KB_TF_Out_V2   │  │• evalMACD()     │  │• string helpers │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### How It Works

1. **Toolkit fetches raw HTF data:**
   ```pinescript
   [eS1, eM1, eL1] = request.security(sym, tf1, [ta.ema(close, emaShort), ...])
   ```

2. **Library processes the data:**
   ```pinescript
   string label = indicators.orderEMA(eS1, eM1, eL1)  // Returns "SML", "LMS", etc.
   bool condA = indicators.isBullStack(eS1, eM1, eL1) // Returns true/false
   ```

3. **Toolkit aggregates and displays:**
   ```pinescript
   KB_TF_Out_V2 side1_v2 = KB_TF_Out_V2.new(label, condA, condB, ...)
   ```

### Benefits

- **Optimizable:** All indicator params are toolkit inputs (paramA-F work!)
- **Modular processing:** Add new indicator logic via library functions
- **Under size limit:** Security calls + helper function calls < full inline logic
- **Maintainable:** Clear separation of concerns

---

## 5. File Size Strategy for 10 Side Modules

### Current Breakdown (Estimated)

| Component | Chars | Notes |
|-----------|-------|-------|
| Inputs & Config | ~15,000 | Groups 1-6, relatively fixed |
| Top Table Engine | ~10,000 | Confluence, scoring |
| Side Module Template | ~8,000 | Per module (currently 1) |
| Table Rendering | ~8,000 | Top + Side tables |
| Helpers & Utils | ~5,000 | Various functions |
| **Current Total** | ~78,000 | Over limit |

### Size Reduction Strategies

1. **Move types to library** (~5,000 char savings)
2. **Move helper functions to library** (~8,000 char savings)
3. **Use loop-based TF processing** instead of copy-paste per TF
4. **Strip comments for production** (but keep documented version)
5. **Condense input declarations** (inline more aggressively)

### Projected Size for 10 Modules

| Scenario | Est. Size | Status |
|----------|-----------|--------|
| Current approach × 10 | ~150,000 | **Far over limit** |
| Hybrid + libraries | ~55,000 | **Under limit** |
| Hybrid + aggressive optimization | ~45,000 | **Comfortable** |

---

## 6. Revised Decision: Hybrid Inline + Library Processing

### Final Architecture Choice

**Use Hybrid Approach (Option C modified):**
- Toolkit owns ALL `request.security()` and `input.*()` calls
- Libraries provide type definitions + processing functions
- paramA-F control indicator parameters as originally intended

### Implementation Priority

1. Create `KevBot_Types` library (shared type definitions)
2. Create `KevBot_Indicators` library (processing functions)
3. Refactor toolkit to use libraries for processing
4. Implement EMA Stack inline with library helpers
5. Test optimizer compatibility

---

## 7. v1.1 Implementation Status (CURRENT)

### Successful Hybrid Architecture Implementation

The v1.1 toolkit successfully implements the hybrid architecture with:

#### Files Created/Modified

| File | Purpose | Status |
|------|---------|--------|
| `KevBot Toolkit v1.1 - Hybrid Architecture.txt` | Main toolkit with hybrid pattern | **Working** |
| `KevBot_TF_EMA_Stack.txt` | EMA Stack library with `buildOutput()` | **Published v7** |
| `KevBot_Types.txt` | Shared type definitions | **Published v1** |
| `KevBot_Indicators.txt` | Pure processing functions | **Published v1** |
| `KevBot_Library_Definitions.md` | User-facing library documentation | **Created** |

#### Key Implementation Details

**Side Module 1 & 2:**
- Both modules can select "EMA Stack (S/M/L)" library
- Each uses its own params (e.g., Side1: 10/20/50, Side2: 5/13/34)
- Both display independently in the side table
- Triggers from both modules are OR'd together for entry/exit signals

**Hybrid Pattern Code Structure:**
```pinescript
// 1. Toolkit owns inputs (optimizable)
int _ema_short = side1_paramA > 0 ? int(side1_paramA) : 10

// 2. Toolkit fetches HTF data
[_eS1, _eM1, _eL1] = request.security(syminfo.tickerid, tf1,
    [ta.ema(close, _ema_short), ta.ema(close, _ema_medium), ta.ema(close, _ema_long)])

// 3. Library processes pre-fetched data
tfEMAStack.TFModuleOutput _emaOut = tfEMAStack.buildOutput(
    _eS1, _eM1, _eL1, _eS2, _eM2, _eL2, ...)

// 4. Normalize to internal V2 structure
KB_TF_Out_V2 side1_v2 = _kb_mapLibToV2(side1_raw)
```

**Lines of Code Added per Module:**
- ~60 lines for hybrid EMA Stack implementation per side module
- Reasonable overhead for full optimizer compatibility

### Proof of Concept Validated

| Test | Result |
|------|--------|
| paramA controls Short EMA length | **Pass** |
| Different params per side module | **Pass** |
| Side table shows correct labels | **Pass** |
| Triggers fire correctly | **Pass** |
| Library selection still works | **Pass** |

### Next Steps

1. Add Side Modules 3-10 (following same pattern)
2. Implement additional library types (RSI, MACD, etc.)
3. File size optimization if needed
4. Third-party optimizer testing

---

## 8. Original Section Preserved: Chosen Approach (OUTDATED)

> **Note:** This section documents our original decision before discovering the third limitation.

### Original Decision (Now Invalid)

Proceed with **Option A: Library-Level Inputs** because:

1. **Optimization Priority:** Third-party optimizers (like DavidDTech) can sweep all parameters
2. **True Modularity:** Libraries are fully self-contained
3. **Scalability:** New libraries don't require toolkit changes
4. **Clean Separation:** Each library controls its own configuration

### Parameter Strategy (OUTDATED - see Section 4)

> **Note:** This approach was invalidated by the third PineScript limitation.

---

## 9. Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2026-01-26 | Initial architecture analysis | EMA Stack showing "LMS" for all TFs |
| 2026-01-26 | Discovered `var` persistence issue | EMAs showing "EQ" (equal) |
| 2026-01-26 | Confirmed `request.security` parameter limitation | Can't use function args in security |
| 2026-01-26 | Chose Option A: Library-Level Inputs | Best for optimization + modularity |
| 2026-01-26 | Implementing EMA Stack with library inputs | Testing user experience |
| 2026-01-26 | **Option A FAILED** - Third limitation discovered | Exported functions can't use input variables |
| 2026-01-26 | File size analysis: 78K chars (limit: 60K) | Already over TradingView limit |
| 2026-01-26 | **New decision: Hybrid Inline + Library Processing** | Toolkit owns security/inputs, libraries process data |
| 2026-01-26 | **v1.1 Hybrid Architecture implemented** | Created KevBot Toolkit v1.1 with hybrid EMA Stack |
| 2026-01-26 | **Side Module 2 added** | Proof of concept - two modules with different EMA params |
| 2026-01-26 | Created KevBot_Library_Definitions.md | User-facing documentation for library params/triggers |

---

## Appendix: Error Messages Reference

**Error 1:** `The "request.*()" call's "expression" cannot depend on the arguments of the exported function`
- **Cause:** Using function parameters in `request.security` expression
- **Solution:** ~~Use library-level `input.*()` instead~~ **This doesn't work either - see Error 3**

**Error 2 (Symptom):** All timeframes showing same label (e.g., "EQ" or "LMS")
- **Cause:** `var` variables not persisting in library function
- **Solution:** Use `request.security` with `ta.ema()` instead of manual calculation

**Error 3:** `The exported function 'X' depends on the 'Y' input variable, which is not allowed`
- **Cause:** Using library-level `input.*()` variables inside exported functions
- **Solution:** **None within library architecture** - must move logic to importing script

---

## Appendix: What Libraries CAN and CANNOT Do

### CANNOT Do (In Exported Functions)
- Use `request.security()` with dynamic parameters
- Use `input.*()` variables
- Maintain `var` state between bars
- Anything that requires compile-time evaluation of runtime values

### CAN Do
- Define and export `type` structures
- Export pure functions that process passed-in values
- Use constants (literals, not inputs)
- Return calculated results based on function arguments (non-security)

### Example of Valid Library Pattern

```pinescript
// Library: KevBot_Indicators
//@version=6
library("KevBot_Indicators", overlay=false)

// VALID: Export a type
export type EMAResult
    string label
    bool bullStack
    bool bearStack

// VALID: Export a pure processing function (no security, no inputs)
export orderEMA(float eS, float eM, float eL) =>
    string lab = "na"
    if na(eS) or na(eM) or na(eL)
        lab := "na"
    else if eS > eM and eM > eL
        lab := "SML"
    // ... etc
    lab

// VALID: Export a condition evaluator
export isBullStack(float eS, float eM, float eL) =>
    not na(eS) and not na(eM) and not na(eL) and eS > eM and eM > eL
```

### How Toolkit Uses This

```pinescript
// Main Toolkit Script
import yamigushi/KevBot_Indicators/1 as ind

// Toolkit owns inputs
int emaShort = input.int(10, "Short EMA")
int emaMed   = input.int(20, "Medium EMA")
int emaLong  = input.int(50, "Long EMA")

// Toolkit fetches HTF data
[eS1, eM1, eL1] = request.security(syminfo.tickerid, tf1,
    [ta.ema(close, emaShort), ta.ema(close, emaMed), ta.ema(close, emaLong)])

// Library processes the data
string label1 = ind.orderEMA(eS1, eM1, eL1)
bool bull1 = ind.isBullStack(eS1, eM1, eL1)
```
