# KevBot Toolkit - TODO / Session Notes

## Current Session: v3.0 QA Testing & VWAP Refactor (February 3, 2026)

### Session Goals
- [x] Port Position Engine from v2.0 to v3.0 (adapted for AND/OR system)
- [x] Port Backtest Engine from v2.0 to v3.0 (KPI tracking)
- [x] Add PNL display on exit labels (user-toggleable)
- [x] Refactor VWAP from descriptive names to sigma-based zones
- [x] Add color coding to VWAP top table module
- [x] Fix script body size limit error (OR2/OR3 trimmed)

### VWAP Sigma-Based Zones (Refactored)
**Old (9 overlapping conditions):** Above VWAP, At/Above VWAP, At VWAP, etc.
**New (7 distinct sigma zones per anchor):**
- `>+2σ` (Extreme High)
- `>+1σ` (+1σ to +2σ)
- `>V` (0 to +1σ)
- `@V` (At VWAP, ±0.5σ)
- `<V` (-1σ to 0)
- `<-1σ` (-2σ to -1σ)
- `<-2σ` (Extreme Low)

### OR2/OR3 Implementation Status
- **AND group:** Fully implemented for all 8 libraries
- **OR1 group:** Fully implemented for all 8 libraries
- **OR2/OR3 groups:** VWAP only (other libraries deferred due to PineScript body size limit)
- **Note:** Users can use AND + OR1 for most strategies; OR2/OR3 available for VWAP zones

### Position Engine (v3.0)
- State machine: flat ↔ in_position
- Entry/Exit based on `validEntry` and `validExit` from confluence engine
- Entry labels with position size, exit labels with optional PNL

### Backtest Engine (v3.0)
- Tracks: Win Rate, Profit Factor, Max Drawdown, Total Trades
- Gross Win/Loss accumulation for PF calculation
- Equity curve tracking for drawdown

---

## Previous Session: v3.0 AND/OR Confluence Implementation ✅ COMPLETE

### v3.0 Implementation (February 2, 2026) ✅ COMPLETE
- [x] Create `src/main/KevBot Toolkit v3.0.txt` as new working file
- [x] Implement AND/OR confluence engine (replaces threshold-based scoring)
- [x] Implement direction-specific toolkit (Long OR Short, not both)
- [x] Implement centralized trigger routing (`_evalTrigger()` function)
- [x] Update table renderers (Top Table + Side Table)
- [x] Add color coding to all 7 side table libraries

### v3.0 Key Changes from v2.0
- **Removed:** TH Scores, Grade thresholds (C/B/A), Required checkbox, Per-library triggers
- **Added:** AND/OR group assignments (None/AND/OR1/OR2/OR3), Position Direction selector, Centralized Entry/Exit triggers

### Previous: v2.0 Architecture Transition ✅ COMPLETE

### Phase 2: Convert Side Table Interpreters to v2 Pattern ✅ COMPLETE
- [x] EMA Stack (template) - inputs, loader, renderer implemented
- [x] RVOL - inputs, loader, renderer implemented
- [x] UT Bot - inputs, loader, renderer implemented
- [x] Swing 123 - inputs, loader, renderer implemented
- [x] MACD Line - inputs, loader, renderer implemented
- [x] MACD Histogram - inputs, loader, renderer implemented
- [x] Simple MACD Line - inputs, loader, renderer implemented

### Deferred to Future Implementation
- [x] VWAP - implemented as Top Table confluence interpreter (February 1, 2026)
- [ ] MACD Divergence - needs design for valuable outputs

### Testing Notes (February 3, 2026)
- v3.0 compiles and runs correctly
- Top Table renders with module-based layout (Position Sizing, Backtest KPI, Confluence, VWAP)
- Side Table renders with TF formatting and color coding
- Color coding works on all 7 side table libraries (Green=assigned+true, Yellow=assigned+false, Gray=unassigned)
- VWAP top table module now has per-cell color coding for sigma zones
- Position Engine tracks entry/exit state with labels on chart
- Backtest Engine tracks KPIs (Win Rate, Profit Factor, Max Drawdown)
- Exit labels can optionally show PNL (user-toggleable)
- **Security call limit**: Still applies when enabling 7+ libraries simultaneously
- **Script body limit**: OR2/OR3 limited to VWAP only; full implementation would exceed PineScript limits

---

## Completed Items

### Phase 0: File Setup (January 31, 2026)
- [x] Move v1.1 from `src/main/` to `legacy/` folder
- [x] Create `src/main/KevBot Toolkit v2.0.txt`
- [x] Create `src/interpreters/side/` folder
- [x] Create `src/interpreters/top/` folder
- [x] Move interpreter files from `src/libraries/` to `src/interpreters/side/`
- [x] Move `KevBot_Top_Minimal.txt` to `src/interpreters/top/`

### Phase 1: Core Infrastructure (January 31, 2026)
- [x] Define normalized output type (KB_TF_Out_V2) - already existed
- [x] Create shared helper functions (_tf_res_v2, _kb_mapCondSource, _kb_getCondTF*)
- [x] Design slot-based side table renderer (Slot 1 = Row 1, etc.)
- [x] EMA Stack inputs with library-specific parameters
- [x] EMA Stack v2 loader with conditional security calls
- [x] EMA Stack side table row renderer

### Phase 2: Side Table Interpreters (January 31, 2026)
- [x] Removed legacy Side Module 1/2 blank row rendering code
- [x] Updated slot-to-row mapping (Slot N = Row N)
- [x] Converted 7 interpreters to v2 pattern with library-specific inputs:
  - EMA Stack (6 security calls)
  - RVOL (6 security calls)
  - UT Bot (7 security calls)
  - Swing 123 (6 security calls)
  - MACD Line (6 security calls)
  - MACD Histogram (6 security calls)
  - Simple MACD Line (6 security calls)
- [x] Full LE/SE/LX/SX condition/trigger configuration per library

---

## v2.0 Architecture Benefits

**What Changed:**
- Each interpreter has its own input section with clear parameter names
- Only enabled interpreters consume `request.security()` calls
- Slot-based side table (1-10 slots, user-assignable)
- Designed for future custom toolkit generator

**Key Files:**
- `src/main/KevBot Toolkit v2.0.txt` - Main toolkit (working file)
- `docs/Side_Module_Architecture_v2.md` - Architecture documentation
- `src/interpreters/side/` - All side table interpreters
- `legacy/KevBot Toolkit v1.1 - Hybrid Architecture.txt` - Previous version

---

## Next Phases

### Phase 2B: Top Table Interpreters ✅ COMPLETE
- [x] VWAP interpreter for Top Table ✅ COMPLETE (February 1, 2026)
  - Daily, Weekly, Monthly anchored VWAPs
  - **Refactored (Feb 3):** Sigma-based zones (7 per anchor: >+2σ, >+1σ, >V, @V, <V, <-1σ, <-2σ)
  - Per-cell color coding (Green=assigned+true, Yellow=assigned+false, Gray=unassigned)
- [ ] MACD Divergence interpreter for Top Table (needs design)

### Phase 3: Trigger/Confluence System ✅ COMPLETE (v3.0)
- [x] Centralized trigger routing via `_evalTrigger()` function
- [x] AND/OR confluence engine replaces threshold scoring
- [x] All 7 side table libraries integrated with color coding

### Phase 3B: QA Testing ← CURRENT
- [ ] Verify AND group logic (all assigned must be true)
- [ ] Verify OR1 group logic (minimum N of M must be true)
- [ ] Verify OR2/OR3 with VWAP sigma zones
- [ ] Test Entry/Exit trigger selection across libraries
- [ ] Validate position direction toggle (Long vs Short)
- [ ] Check side table color coding on all libraries
- [ ] Check VWAP top table color coding on sigma zones
- [ ] Verify Position Engine entry/exit state machine
- [ ] Verify Backtest KPI calculations (WR, PF, DD)
- [ ] Test PNL display on exit labels
- [ ] Test with sample trading strategies

### Phase 4: Data Export
- [ ] Design export format compatible with Trade Analyzer
- [ ] Update plot() exports for AND/OR group data
- [ ] Add import string parsing for Trade Analyzer integration

### Phase 5: Custom Toolkit Generator (Future)
- [ ] Create "mega block" file with all libraries
- [ ] Add generator markers/comments
- [ ] Build web app for library selection

---

## Other Pending Items
- [x] Complete backtest KPI calculations in toolkit ✅ (February 3, 2026)
- [ ] Publish new libraries to TradingView (VWAP, RVOL, Swing 123)
- [ ] Manually verify MACD Divergence behavior on charts

---

## Future Concepts (Exploratory - Not Committed)

### ~~AND/OR Condition Groups~~ ✅ IMPLEMENTED (v3.0)
**Status:** Implemented in v3.0 (February 2, 2026)

Replaced threshold-based scoring with explicit AND/OR logic groups:
- **AND group:** ALL assigned conditions must be true
- **OR groups (1-3):** User specifies minimum # of conditions that must be true
- More intuitive for expressing strategy logic

See `docs/KevBot_Toolkit_v3_Vision.md` for full specification.

### Trade Analyzer Integration
**Status:** Planned for future

- Import string parsing for quick configuration
- Config export from Trade Analyzer to TradingView
- What-if analysis for confluence combinations

---

*Last Updated: February 3, 2026 (v3.0 QA Testing - VWAP Sigma Zones & Position/Backtest Engine)*
