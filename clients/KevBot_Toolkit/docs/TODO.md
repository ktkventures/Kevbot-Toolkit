# KevBot Toolkit - TODO / Session Notes

## Current Session: v2.0 Architecture Transition

### Phase 2: Convert Side Table Interpreters to v2 Pattern ✅ COMPLETE
- [x] EMA Stack (template) - inputs, loader, renderer implemented
- [x] RVOL - inputs, loader, renderer implemented
- [x] UT Bot - inputs, loader, renderer implemented
- [x] Swing 123 - inputs, loader, renderer implemented
- [x] MACD Line - inputs, loader, renderer implemented
- [x] MACD Histogram - inputs, loader, renderer implemented
- [x] Simple MACD Line - inputs, loader, renderer implemented

### Deferred to Top Table Implementation
- [ ] VWAP - better suited for Top Table interpreter
- [ ] MACD Divergence - better suited for Top Table interpreter

### Testing Notes (January 31, 2026)
- All 7 side table interpreters compile and load correctly
- Slot-based side table rendering works (Slot 1 = Row 1, etc.)
- Enable/disable toggles correctly prevent security calls when disabled
- **Security call limit**: When enabling 7+ libraries simultaneously, TradingView may crash due to ~40-50 request.security() call limit. Users must be selective about which libraries to enable at once.

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

### Phase 2B: Top Table Interpreters (Future)
- [ ] VWAP interpreter for Top Table
- [ ] MACD Divergence interpreter for Top Table

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

---

## Other Pending Items
- [ ] Complete backtest KPI calculations in toolkit
- [ ] Publish new libraries to TradingView (VWAP, RVOL, Swing 123)
- [ ] Manually verify MACD Divergence behavior on charts

---

## Future Concepts (Exploratory - Not Committed)

### AND/OR Condition Groups
**Status:** Idea to explore later

Replace threshold-based scoring with explicit AND/OR logic groups:
- AND groups: All conditions must be true
- OR groups: Any one condition is sufficient
- More intuitive for expressing strategy logic

See `docs/Side_Module_Architecture_v2.md` for detailed concept notes.

**Decision:** Revisit after v2.0 library conversions are complete.

---

*Last Updated: January 31, 2026 (Phase 2 Complete)*
