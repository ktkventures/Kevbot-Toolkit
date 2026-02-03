# KevBot Toolkit v3.0 - QA Checklist

**Version:** 3.0
**Date:** February 3, 2026 (Updated)
**Purpose:** Verify v3.0 AND/OR confluence system behaviors

---

## Pre-Testing Setup

Before testing, configure the following baseline settings:

1. Apply `KevBot Toolkit v3.0` indicator to a chart (recommended: 5-minute chart on a liquid stock like AAPL or SPY)
2. Set Position Direction to **Long**
3. Enable at least 2 side table libraries (e.g., EMA Stack and UT Bot)
4. Enable VWAP
5. Set OR1/OR2/OR3 minimum required to 1 (default)

---

## Section 1: Global Settings

### 1.1 Position Direction Toggle
- [ ] Setting "Long" shows only Long-relevant conditions in tables
- [ ] Setting "Short" shows only Short-relevant conditions in tables
- [ ] Changing direction updates all table displays correctly
- [ ] Entry/Exit triggers fire appropriately for selected direction

### 1.2 Table Visibility
- [ ] "Enable Top Table" toggle shows/hides Top Table
- [ ] "Enable Side Table" toggle shows/hides Side Table
- [ ] Both tables can be enabled simultaneously
- [ ] Dark Theme toggle changes table colors appropriately

### 1.3 Timeframe Configuration
- [ ] TF1-TF6 enable/disable toggles work correctly
- [ ] Disabled TFs do not appear in Side Table columns
- [ ] TF labels display correctly (e.g., "1m", "5m", "1H", "D")
- [ ] Higher TFs (D, W, M) load data correctly via request.security()

---

## Section 2: Confluence Group Settings

### 2.1 OR Group Minimums
- [ ] OR1 Minimum Required input accepts values 1-20
- [ ] OR2 Minimum Required input accepts values 1-20
- [ ] OR3 Minimum Required input accepts values 1-20
- [ ] Setting minimum > assigned conditions shows appropriate status

### 2.2 AND Group Behavior
- [ ] When 0 conditions assigned to AND: group passes automatically
- [ ] When 1+ conditions assigned to AND: ALL must be true to pass
- [ ] Single false condition in AND group causes entire AND group to fail
- [ ] Confluence Summary shows "AND: X/Y" with correct counts

### 2.3 OR Group Behavior
- [ ] When 0 conditions assigned to OR1/OR2/OR3: group passes automatically
- [ ] When conditions assigned: at least minimum must be true to pass
- [ ] OR group passes when true_count >= minimum_required
- [ ] OR group fails when true_count < minimum_required
- [ ] Confluence Summary shows "OR1: X/Y" etc. with correct counts

---

## Section 3: Entry/Exit Triggers

### 3.1 Entry Trigger Dropdown
- [ ] Dropdown shows all trigger options from enabled libraries only
- [ ] EMA triggers appear when EMA Stack enabled (S>M Cross, S<M Cross, etc.)
- [ ] UT Bot triggers appear when UT Bot enabled (Buy, Sell)
- [ ] RVOL triggers appear when RVOL enabled (Spike, Extreme, Fade)
- [ ] Swing 123 triggers appear when Swing 123 enabled (BC2, BC3, XC2, XC3)
- [ ] MACD triggers appear when MACD libraries enabled
- [ ] Selecting "None" disables entry signals

### 3.2 Exit Trigger Dropdown
- [ ] Same trigger options as Entry Trigger
- [ ] Entry and Exit can use same trigger (valid for scalping strategies)
- [ ] Entry and Exit can use different triggers
- [ ] Selecting "None" disables exit signals

### 3.3 Trigger Firing
- [ ] Entry trigger fires only when all confluence groups pass
- [ ] Entry trigger fires only when entry trigger condition becomes true
- [ ] Exit trigger fires when in position and exit condition becomes true
- [ ] Chart marks appear at correct trigger points

---

## Section 4: Top Table Display

### 4.1 Module Layout
- [ ] Each module displays in 2-column × 5-row format
- [ ] Module headers show correct names (Position Sizing, Backtest KPI, etc.)
- [ ] Modules align properly in table grid

### 4.2 Position Sizing Module
- [ ] Shows when "Show Module" is enabled
- [ ] Displays Account Size correctly
- [ ] Displays Risk Mode (Share Qty / Fixed $ / Percentage)
- [ ] Displays calculated position size
- [ ] Displays calculated risk amount

### 4.3 Backtest KPI Module
- [ ] Shows when "Show Backtest KPI Module" is enabled
- [ ] Displays Total Trades count
- [ ] Displays Win/Loss counts
- [ ] Displays Win Rate percentage
- [ ] Displays Profit Factor (or "N/A" if no losing trades)

### 4.4 Confluence Summary Module
- [ ] Shows AND group status (passed/failed, count)
- [ ] Shows OR1 group status with numerator/denominator
- [ ] Shows OR2 group status with numerator/denominator
- [ ] Shows OR3 group status with numerator/denominator
- [ ] Shows overall Entry Valid status (VALID/INVALID)
- [ ] Colors update based on group pass/fail status

### 4.5 VWAP Module (D/W/M)
- [ ] Daily VWAP zone displays correctly
- [ ] Weekly VWAP zone displays correctly
- [ ] Monthly VWAP zone displays correctly
- [ ] Zone labels correct (>V, @V, <V, Extended, etc.)
- [ ] Colors reflect group assignment and condition state

---

## Section 5: Side Table Display

### 5.1 Library Rows
- [ ] Only enabled libraries appear in Side Table
- [ ] Libraries appear in enable order (first enabled = row 1)
- [ ] Library names display in first column
- [ ] Disabling a library removes its row immediately

### 5.2 TF Columns
- [ ] TF headers show formatted timeframe (e.g., "1m", "5m", "1H", "D")
- [ ] Only enabled TFs appear as columns
- [ ] TF1 is leftmost, TF6 is rightmost
- [ ] Header row aligns with data rows

### 5.3 Color Coding
For each library (EMA Stack, RVOL, UT Bot, Swing 123, MACD Line, MACD Histogram, Simple MACD):
- [ ] **Green**: Cell shows green when condition is assigned to a group AND condition is TRUE
- [ ] **Yellow**: Cell shows yellow when condition is assigned to a group AND condition is FALSE
- [ ] **Gray**: Cell shows gray when condition is NOT assigned to any group (None)
- [ ] Colors update in real-time as conditions change

### 5.4 Cell Content
- [ ] EMA Stack: Shows ordering labels (SML, LMS, SLM, MSL, MLS, LSM)
- [ ] RVOL: Shows volume zone (RV!, RV++, RV+, RV=, RV-)
- [ ] UT Bot: Shows direction (Bull, Bear)
- [ ] Swing 123: Shows pattern state
- [ ] MACD variants: Show momentum state

---

## Section 6: Library-Specific Checks

### 6.1 EMA Stack Library
- [ ] Short/Medium/Long EMA lengths configurable
- [ ] 6 conditions available per TF (SML, LMS, SLM, MSL, MLS, LSM)
- [ ] Each condition × TF can be assigned to None/AND/OR1/OR2/OR3
- [ ] Labels update correctly as price crosses EMAs

### 6.2 RVOL Library
- [ ] Lookback period configurable
- [ ] High/Very High/Extreme thresholds configurable
- [ ] 5 conditions available per TF (RV!, RV++, RV+, RV=, RV-)
- [ ] Volume zones update correctly with market activity

### 6.3 UT Bot Library
- [ ] Key Value (ATR multiplier) configurable
- [ ] ATR Period configurable
- [ ] Heikin Ashi toggle works
- [ ] Bull/Bear conditions track trailing stop correctly

### 6.4 Swing 123 Library
- [ ] No parameters (pure price action)
- [ ] 6 conditions per TF (BC2, BC3, XC2, XC3, B↑, X↓)
- [ ] Pattern detection fires on correct candle formations

### 6.5 MACD Line Library
- [ ] Fast/Slow/Signal lengths configurable
- [ ] 4 conditions per TF (M>S+, M>S-, M<S-, M<S+)
- [ ] Conditions update correctly with MACD changes

### 6.6 MACD Histogram Library
- [ ] Fast/Slow/Signal lengths configurable
- [ ] 4 conditions per TF (H+↑, H+↓, H-↓, H-↑)
- [ ] Conditions track histogram correctly

### 6.7 Simple MACD Library
- [ ] Fast/Slow/Signal lengths configurable
- [ ] 4 conditions per TF (M>S↑, M>S↓, M<S↓, M<S↑)
- [ ] Cross detection works correctly

---

## Section 7: Edge Cases

### 7.1 All Groups Empty
- [ ] When no conditions assigned to any group: Entry Valid = TRUE (all groups auto-pass)
- [ ] Entry still requires trigger to fire

### 7.2 Only AND Group Used
- [ ] OR1/OR2/OR3 auto-pass when empty
- [ ] Entry Valid based solely on AND group conditions

### 7.3 High OR Minimums
- [ ] Setting OR1 minimum = 5 with only 3 conditions assigned shows failure
- [ ] Confluence Summary shows correct "0/5 ✗" or similar

### 7.4 Rapid Direction Change
- [ ] Switching Long↔Short doesn't cause errors
- [ ] Tables update correctly after direction change
- [ ] No lingering state from previous direction

### 7.5 Library Enable/Disable
- [ ] Enabling library adds row to Side Table
- [ ] Disabling library removes row from Side Table
- [ ] Triggers from disabled library don't fire
- [ ] Conditions from disabled library don't affect confluence

---

## Section 8: Performance

### 8.1 Load Time
- [ ] Indicator loads within acceptable time (< 5 seconds)
- [ ] No "Script taking too long" errors on initial load

### 8.2 Real-Time Updates
- [ ] Tables update smoothly with each bar
- [ ] No flickering or lag in table rendering
- [ ] Color transitions are immediate

### 8.3 Multi-Library Load
- [ ] Enabling 3+ libraries simultaneously works
- [ ] Enabling 5+ libraries works (may be slower)
- [ ] Enabling 7 libraries works but may hit security() limits
- [ ] Error message appears if security() limit exceeded

---

## Issue Tracking

| # | Section | Description | Status | Notes |
|---|---------|-------------|--------|-------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

---

## Sign-Off

- [ ] All critical items verified
- [ ] All major items verified
- [ ] Known issues documented
- [ ] Ready for production use

**Tested By:** _______________
**Date:** _______________
**v3.0 Status:** _______________

---

*This checklist covers the core v3.0 AND/OR confluence system behaviors. Additional testing may be needed for edge cases specific to your trading strategies.*
