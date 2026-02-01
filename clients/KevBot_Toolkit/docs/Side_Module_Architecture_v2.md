# KevBot Toolkit - Side Module Architecture v2

## Overview

This document outlines a new architecture for the Side Module system that addresses:
1. Pine Script `request.security()` call limits (~40-50 per script)
2. Cleaner UX with library-specific parameters
3. Scalability for adding more libraries
4. Future data export to Trade Analyzer
5. Custom toolkit generation (web app that builds user-selected library bundles)

---

## Current Architecture (v1) - Problems

```
6. Side Module 1
   └─ Library: [Dropdown]
   └─ Parameter A-F: [Generic floats]
   └─ Entry/Exit config...

7. Side Module 2
   └─ (Same structure)
```

**Issues:**
- Generic "Parameter A/B/C" is confusing
- Scaling to 10 modules hits security call limits
- Each module duplicates the entire library loader (~780 lines)
- Can't easily disable unused libraries to save security calls

---

## Proposed Architecture (v2) - Library-Centric

```
6. Side Modules (Parent Group)
   └─ 6-1. EMA Stack
        └─ Enable: [✓]
        └─ Side Table Slot: [1] (or "Hidden")
        └─ Short EMA: 10
        └─ Medium EMA: 20
        └─ Long EMA: 50
        └─ Entry/Exit config...

   └─ 6-2. VWAP
        └─ Enable: [✓]
        └─ Side Table Slot: [2]
        └─ Band 1 Mult: 1.0
        └─ Band 2 Mult: 2.0
        └─ Entry/Exit config...

   └─ 6-3. UT Bot
        └─ Enable: [ ]  ← Disabled = 0 security calls!
        └─ Side Table Slot: [Hidden]
        └─ Key Value: 1.0
        └─ ATR Period: 10
        └─ Use Heikin Ashi: No
        └─ Entry/Exit config...

   └─ 6-4. RVOL
   └─ 6-5. Swing 123
   └─ 6-6. MACD Line
   └─ 6-7. MACD Histogram
   └─ 6-8. MACD Divergence
   └─ 6-9. Simple MACD Line
   └─ 6-10. [Future libraries...]
```

**Benefits:**
- Only enabled libraries consume security calls
- Clear, library-specific parameter names
- User controls resource usage
- Add new libraries without touching existing code
- Side Table Slot selector allows flexible table positioning

---

## Key Design Decisions

### 1. Library Code - Keep As-Is (No Updates Required)

**Recommendation: Do NOT update the side libraries.**

The current libraries (tfEMAStack, tfMACDLine, etc.) work well and return their own TFModuleOutput types. The toolkit already handles the type casting to a normalized format.

**Rationale:**
- Avoids manual TradingView publish work for you
- Libraries remain independent and testable
- Toolkit is the integration layer that normalizes outputs
- Easier to add third-party libraries in the future
- If we later want libraries to return a common type, we can do that incrementally

**How it works:**
```
Library (tfEMAStack)          Toolkit                      Normalized Output
       │                         │                              │
       └─► tfEMAStack.          └─► _kb_castToNormalized()     └─► KB_TF_Out_V2
           TFModuleOutput            (handles all types)
```

### 2. Modular Code Structure for Custom Toolkit Generator

The code will be organized into clear, extractable blocks:

```pinescript
//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ██████ SIDE LIBRARY BLOCK: EMA_STACK ██████
// GENERATOR_ID: EMA_STACK
// SECURITY_CALLS: 6
// DEPENDENCIES: yamigushi/KevBot_TF_EMA_Stack/1
//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// --- EMA_STACK: IMPORTS ---
import yamigushi/KevBot_TF_EMA_Stack/1 as tfEMAStack

// --- EMA_STACK: INPUTS ---
string GRP_LIB_EMA = "6-1. EMA Stack"
bool   lib_ema_enabled = input.bool(true, "Enable", group = GRP_LIB_EMA)
int    lib_ema_slot    = input.int(1, "Side Table Slot", minval=0, maxval=10, group = GRP_LIB_EMA, tooltip="0 = Hidden")
int    lib_ema_short   = input.int(10, "Short EMA", group = GRP_LIB_EMA)
int    lib_ema_medium  = input.int(20, "Medium EMA", group = GRP_LIB_EMA)
int    lib_ema_long    = input.int(50, "Long EMA", group = GRP_LIB_EMA)
// ... entry/exit config inputs ...

// --- EMA_STACK: LOADER ---
KB_TF_Out_V2 lib_ema_output = KB_TF_Out_V2.new()  // Empty default
if lib_ema_enabled
    // Security calls + builder + normalization
    [_eS1, _eM1, _eL1] = request.security(...)
    // ... (loader logic) ...
    lib_ema_output := _kb_normalizeOutput(_emaOut)

// --- EMA_STACK: EXPORT DATA ---
// (Handled in central export section using lib_ema_output)

//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ██████ END BLOCK: EMA_STACK ██████
//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**For Custom Toolkit Generator:**
- A web app parses the "mega block" file containing all libraries
- User selects which libraries to include
- Generator extracts only those blocks + shared infrastructure
- Outputs a custom Pine Script file
- Respects security call budget (shows running total)

### 3. Data Export for Trade Analyzer

Each library will export its data in a consistent format:

```pinescript
// --- EXPORT SECTION ---
// Format: plot() calls with display.data_window for CSV export

// Library state exports (one per enabled library)
plot(lib_ema_enabled ? _encodeLibState(lib_ema_output, "EMA") : na, "EMA State", display=display.data_window)
plot(lib_vwap_enabled ? _encodeLibState(lib_vwap_output, "VWAP") : na, "VWAP State", display=display.data_window)
// ...

// Trigger exports (for backtesting)
plot(lib_ema_triggerFired ? 1 : 0, "EMA Trigger", display=display.data_window)
plot(lib_vwap_triggerFired ? 1 : 0, "VWAP Trigger", display=display.data_window)
// ...
```

**Trade Analyzer Integration:**
- TradingView CSV export captures these plot values
- Trade Analyzer parses the data
- Each library's TF states become "confluence records"
- Enables drill-down and auto-search analysis

### 4. Side Table Rendering

The side table becomes dynamic based on slot assignments:

```pinescript
// Collect all enabled libraries with their slot assignments
array<LibrarySlot> _slots = array.new<LibrarySlot>()

if lib_ema_enabled and lib_ema_slot > 0
    array.push(_slots, LibrarySlot.new(lib_ema_slot, "EMA", lib_ema_output))
if lib_vwap_enabled and lib_vwap_slot > 0
    array.push(_slots, LibrarySlot.new(lib_vwap_slot, "VWAP", lib_vwap_output))
// ... etc

// Sort by slot number and render
array.sort(_slots, order.asc)
for _slot in _slots
    _renderSideTableRow(_slot)
```

---

## Security Call Budget

| Library | Calls | Cumulative (if all enabled) |
|---------|-------|----------------------------|
| EMA Stack | 6 | 6 |
| Simple MACD Line | 6 | 12 |
| MACD Line | 6 | 18 |
| MACD Histogram | 6 | 24 |
| MACD Divergence | 0 | 24 |
| UT Bot | 7 | 31 |
| VWAP | 12 | 43 |
| RVOL | 6 | 49 |
| Swing 123 | 24 | **73** ❌ |

**User Guidance:**
- Can't enable all libraries simultaneously
- Typical usage: 3-5 libraries = ~25-35 calls ✅
- Heavy usage: 6-7 lighter libraries = ~40 calls ✅
- Display warning in settings if approaching limit

**Future: Multiple Toolkit Variants**
- KevBot Toolkit - Trend (EMA, MACD variants) - max 24 calls
- KevBot Toolkit - Volume (VWAP, RVOL) - max 18 calls
- KevBot Toolkit - Price Action (Swing 123, future patterns) - max 30 calls
- KevBot Toolkit - Full (all libraries, user manages budget)

---

## Implementation Phases

### Phase 0: File Setup ✅ COMPLETE
- [x] Move v1.1 from `src/main/` to `legacy/` folder (existing folder at root)
- [x] Create `src/main/KevBot Toolkit v2.0.txt` (copy of v1.1 as starting point)
- [x] Create `src/interpreters/side/` folder
- [x] Create `src/interpreters/top/` folder
- [x] Move side interpreter files from `src/libraries/` to `src/interpreters/side/`
- [x] Move `src/core/KevBot_Top_Minimal.txt` to `src/interpreters/top/`
- [x] Remove empty `src/libraries/` folder after migration

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Define normalized output type (KB_TF_Out_V2) - already exists
- [x] Create shared helper functions (_tf_res_v2, reusing existing _kb_mapCondSource, _kb_getCondTF*)
- [x] Design the slot-based side table renderer (Slot 1 = Row 1, etc.)

### Phase 2: Convert Side Table Libraries to New Pattern ✅ COMPLETE
- [x] EMA Stack - inputs, loader, renderer (6 security calls)
- [x] RVOL - inputs, loader, renderer (6 security calls)
- [x] UT Bot - inputs, loader, renderer (7 security calls)
- [x] Swing 123 - inputs, loader, renderer (6 security calls)
- [x] MACD Line - inputs, loader, renderer (6 security calls)
- [x] MACD Histogram - inputs, loader, renderer (6 security calls)
- [x] Simple MACD Line - inputs, loader, renderer (6 security calls)
- [x] Full LE/SE/LX/SX condition/trigger configuration per library
- **Deferred to Top Table:** VWAP, MACD Divergence

### Phase 3: Trigger/Confluence System
- [ ] Update entry/exit config to work with library-specific inputs
- [ ] Consolidate trigger routing from all enabled libraries
- [ ] Test confluence scoring across libraries

### Phase 4: Data Export
- [ ] Design export format compatible with Trade Analyzer
- [ ] Add plot() exports for each library
- [ ] Test CSV export → Trade Analyzer import

### Phase 5: Custom Toolkit Generator (Future)
- [ ] Create "mega block" file with all libraries
- [ ] Add generator markers/comments
- [ ] Build web app for library selection
- [ ] Generate custom Pine Script output

---

## File Structure

```
KevBot_Toolkit/
│
├── legacy/                                   # Archived versions (at root level)
│   ├── LEGACY - Kevbot Toolkit v1.0 - Input Skeleton.txt
│   ├── LEGACY - KevBot_Toolkit_EMAStack_Integration.txt
│   └── KevBot Toolkit v1.1 - Hybrid Architecture.txt  # Will move here
│
├── src/
│   ├── main/
│   │   └── KevBot Toolkit v2.0.txt           # New toolkit (v2 architecture)
│   │
│   ├── interpreters/
│   │   ├── side/                             # Side Table / TF Confluence Interpreters
│   │   │   ├── KevBot_TF_EMA_Stack.txt       # No changes needed
│   │   │   ├── KevBot_TF_VWAP.txt            # No changes needed
│   │   │   ├── KevBot_TF_UTBot.txt
│   │   │   ├── KevBot_TF_RVOL.txt
│   │   │   ├── KevBot_TF_Swing123.txt
│   │   │   ├── KevBot_TF_MACD_Line.txt
│   │   │   ├── KevBot_TF_MACD_Histogram.txt
│   │   │   ├── KevBot_TF_MACD_Divergence.txt
│   │   │   ├── KevBot_TF_MACD_Simple.txt
│   │   │   └── KevBot_TF_Placeholder.txt
│   │   │
│   │   └── top/                              # Top Table Module Interpreters
│   │       ├── KevBot_Top_Minimal.txt        # Existing top module
│   │       └── (future top modules...)
│   │
│   ├── core/                                 # Shared utilities (unchanged)
│   │   ├── KevBot_TimeUtils.txt
│   │   └── KevBot_Types.txt
│   │
│   └── blocks/                               # For custom toolkit generator (future)
│       ├── _infrastructure.pine              # Shared types, helpers
│       ├── _block_ema_stack.pine             # EMA Stack block (extractable)
│       ├── _block_vwap.pine                  # VWAP block
│       └── ...                               # One block per interpreter
│
├── reference-indicators/                     # Standalone reference indicators (unchanged)
│   └── (existing reference indicators)
│
├── docs/                                     # Documentation
│   └── ...
│
└── tools/
    ├── trade_analyzer/                       # Existing Python app
    └── toolkit_generator/                    # Future web app for custom toolkit builds
```

### Migration Notes

**Before starting v2 development:**
1. Move `src/main/KevBot Toolkit v1.1 - Hybrid Architecture.txt` to `legacy/`
2. Create new file `src/main/KevBot Toolkit v2.0.txt` (copy of v1.1 as starting point)
3. Create `src/interpreters/side/` and `src/interpreters/top/` folders
4. Move files from `src/libraries/` to `src/interpreters/side/`
5. Move `src/core/KevBot_Top_Minimal.txt` to `src/interpreters/top/`

---

## Answers to Your Questions

### Q1: Update Side Libraries?

**No, keep them as-is.** The toolkit handles normalization. Benefits:
- No manual TradingView publish work
- Libraries remain independent
- Easier to add third-party libraries
- Can update libraries independently if needed

### Q2: Trade Analyzer Export?

**Fully supported.** The architecture includes:
- Consistent export format per library
- TF state encoding for confluence records
- Trigger event logging
- Easy CSV export → Trade Analyzer import

### Q3: Custom Toolkit Generator?

**Architecturally ready.** The block-based structure enables:
- Clear extraction markers
- Self-contained library blocks
- Web app can parse and select
- Security call budget tracking
- Custom Pine Script output

---

## Next Steps

1. **Review this document** - Does this architecture align with your vision?
2. **Decide on Phase 1 scope** - Start with infrastructure + 1-2 libraries?
3. **Create the new toolkit file** - Begin fresh with v2 architecture
4. **Incremental testing** - Add libraries one at a time, test each

---

## Next Steps

1. **Execute Phase 0** - Set up folder structure and create v2.0 working file
2. **Begin Phase 1** - Core infrastructure in v2.0
3. **Test incrementally** - Add interpreters one at a time

---

## Future Concepts (Exploratory)

### AND/OR Condition Groups

**Status:** Concept only - not committed, needs further exploration

**Current System (Threshold-based):**
The current system uses point-based scoring where each TF/condition contributes a score, and grades are determined by reaching thresholds (e.g., 200 for Grade C). The "Required" checkbox provides basic AND logic.

**Concept: Logic-based Condition Groups:**
Instead of threshold scoring, use explicit AND/OR groupings that mirror how traders naturally think about confluence:

```
Example Setup:

AND Group 1 (all must be true):
  - EMA Stack: SML on 1m
  - EMA Stack: SML on 5m

OR Group 2 (any one is sufficient):
  - MACD: Bullish crossover
  - MACD: Histogram positive
  - MACD: Line above zero

Entry = AND_Group_1 AND OR_Group_2
```

**Potential Benefits:**
- More intuitive for traders ("I need X AND Y, but for Z I'm flexible")
- Clearer expression of strategy logic
- Potentially easier to export/replicate in Trade Analyzer
- Could simplify the UI (no point values to configure)

**Open Questions:**
- How many AND/OR groups to support?
- How do groups combine? (all ANDs must pass + at least one OR group?)
- How does this affect data export format?
- Does this replace or supplement the threshold system?
- UI complexity vs. flexibility tradeoff

**Decision:** To be revisited after v2.0 library conversions are complete. The current threshold + required system is functional and shouldn't block progress.

---

*Created: January 31, 2026*
*Updated: January 31, 2026*
*Status: Phase 2 COMPLETE - Side Table interpreters converted to v2 pattern*
