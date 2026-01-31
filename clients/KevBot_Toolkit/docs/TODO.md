# KevBot Toolkit - TODO / Session Notes

## Next Session Priority

### Decision Point: Architecture Direction

Choose between two development paths:

**Option A: Top Table Module System**
- Build infrastructure for custom top table modules
- Good candidates: SR Channel, Multi-Anchor VWAP
- These modules don't need per-TF evaluation (chart TF only works fine)
- Would enable "institutional levels" view (SR zones, multiple VWAPs)

**Option B: Expand Side Modules (2 → 10)**
- Currently have 2 side module slots
- Expand to 10 side module slots for more confluence analysis
- Allows more indicator combinations per setup
- More code duplication but follows established pattern

**Recommendation:** Discuss with Kevin which provides more immediate value for his trading workflow.

### Pending Toolkit Work
- [ ] Manually verify MACD Divergence behavior against chart interpretation
- [ ] Complete backtest KPI calculations in toolkit
- [ ] Manually verify VWAP, RVOL, Swing 123 behavior on charts
- [ ] Publish new libraries to TradingView (VWAP, RVOL, Swing 123)

---

## Completed Items

### ✅ UT Bot Library - Complete & Verified (January 30, 2026)

**Location:** `src/libraries/KevBot_TF_UTBot.txt`

**What was built:**
- ATR-based trailing stop with ratcheting behavior
- Per-TF evaluation working correctly via hybrid architecture
- Key technical achievement: Uses self-referential series (not var) in `_kb_calcUTBot()` helper so `request.security()` properly evaluates on each timeframe

**Parameters:**
- A: Key Value / ATR multiplier (default: 1)
- B: ATR Period (default: 10)
- C: Use Heikin Ashi (0=No, 1=Yes)

**Conditions:** Bull (price > stop), Bear (price <= stop)
**Triggers:** Buy (cross above), Sell (cross below)

**Status:** Library complete, integrated into toolkit (Side Module 1 & 2), manually verified working

---

### ✅ Trade Analysis Tool (Python/Streamlit) - POC Complete

**Location:** `tools/trade_analyzer/`

**What was built:**
- Streamlit web app for analyzing confluence combinations
- Mock data generator mimicking TradingView CSV export format
- Two analysis modes:
  - **Drill-Down**: Start with best single factor, iteratively add confluences
  - **Auto-Search**: Combinatorial search for best N-factor combinations
- "Confluence Record" concept: atomic units combining Timeframe + Evaluator + State (e.g., "1M-EMA-SML")
- KPI Dashboard: Trades, Win Rate, Profit Factor, Daily P&L (R and $)
- P&L Settings: Fixed vs Compounding risk modes, starting balance, risk per trade
- Interactive equity curve with high water mark
- Export TradingView Parameters button (placeholder for future integration)

**Tech Stack:** Python, Streamlit, Pandas, Plotly (~800 lines)

**Status:** POC complete. Ready to integrate with real toolkit export data once format is finalized.

---

### ✅ VWAP Library - Complete (January 31, 2026)

**Location:** `src/libraries/KevBot_TF_VWAP.txt`

**What was built:**
- Session-anchored VWAP with standard deviation bands
- 7 mutually exclusive zone conditions: >+2σ, >+1σ, >V, @V, <V, <-1σ, <-2σ
- Per-TF evaluation via hybrid architecture with `_kb_calcVWAP()` helper

**Parameters:**
- A: Band 1 Multiplier (default: 1.0)
- B: Band 2 Multiplier (default: 2.0)
- C: Band 3 Multiplier (default: 3.0)

**Status:** Library complete, integrated into toolkit, needs TradingView publish

---

### ✅ RVOL Library - Complete (January 31, 2026)

**Location:** `src/libraries/KevBot_TF_RVOL.txt`

**What was built:**
- Relative volume analysis (current volume vs historical average)
- 5 zone conditions: RV! (extreme), RV++ (very high), RV+ (elevated), RV= (normal), RV- (low)
- Uses zone labels (not numeric values like "1.8x") as requested

**Parameters:**
- A: Lookback period (default: 20)
- B: High threshold (default: 1.5)
- C: Very high threshold (default: 2.0)
- D: Extreme threshold (default: 3.0)

**Status:** Library complete, integrated into toolkit, needs TradingView publish

---

### ✅ Swing 123 Library - Complete (January 31, 2026)

**Location:** `src/libraries/KevBot_TF_Swing123.txt`

**What was built:**
- 1-2-3 reversal pattern recognition
- No parameters needed (fixed pattern logic)
- Conditions: BC2, BC3, XC2, XC3, B↑ (recent bullish), X↓ (recent bearish)
- Per-TF evaluation (no cumulative state needed)

**Pattern Logic:**
- BC2: Lower low but closes above prior close (rejection)
- BC3: Close above prior high after BC2 (confirmation)
- XC2: Higher high but closes below prior close (rejection)
- XC3: Close below prior low after XC2 (confirmation)

**Status:** Library complete, integrated into toolkit, needs TradingView publish

---

### 📋 SR Channel - Deferred to Top Table Module

**Reason for deferral:**
- SR Channel requires `var` state tracking for pivot accumulation
- `request.security()` cannot preserve `var` state across timeframes
- Would only work on chart TF (same state across all TF columns)
- This defeats the purpose of a side table module
- Better suited as a Top Table module where single chart-TF view is appropriate

**Future implementation:**
- Will implement as part of Top Table module system
- Candidates for top table: SR Channel, Multi-Anchor VWAP
- These modules provide consolidated views rather than per-TF comparison

---

## Other Pending Items

- [ ] Manually verify MACD Divergence behavior against chart interpretation
- [x] Build UT Bot library (preferred trigger module)
- [x] Integrate UT Bot library into main toolkit
- [x] Manually verify UT Bot behavior against chart interpretation
- [x] Build VWAP library
- [x] Build RVOL library
- [x] Build Swing 123 library
- [ ] Build SR Channel library → **DEFERRED** (moving to top table module)
- [ ] Publish VWAP, RVOL, Swing 123 to TradingView

---

*Last Updated: January 31, 2026*
