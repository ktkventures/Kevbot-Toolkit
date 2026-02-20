# Phase 20: General & Risk Management Pack Audit — Implementation Spec

## Overview

Phase 20 audited, fixed, and extended the General Packs and Risk Management Packs subsystems across five sub-phases.

## Sub-Phases Completed

### 20A: General Packs Bug Fixes & Code Quality
**File: `src/general_packs.py`**
- **avoid_opex**: Implemented as 3rd-Friday-of-month proxy (day 15-21, Friday). Previously defined in schema but never evaluated.
- **buffer_minutes**: Added help text clarifying proxy mode uses full-day blocks; actual buffer requires future event calendar API.
- **DRY time eval**: Extracted `_eval_time_window()` shared helper; `_eval_time_of_day` and `_eval_trading_session` are now thin wrappers.
- **Weekend handling**: `_eval_day_of_week()` now returns `BLOCKED_DAY` for Saturday/Sunday instead of implicitly allowing them.
- **UNKNOWN fallback**: `evaluate_condition()` now returns the template's last output state instead of `"UNKNOWN"` (which wasn't in any template's outputs list).
- **validate_parameters()**: New function that clamps all params to schema min/max ranges.

### 20B: General Packs Triggers
**Files: `src/general_packs.py`, `src/app.py`**
- Added trigger definitions to templates:
  - `time_of_day`: window_open, window_close
  - `trading_session`: session_open, session_close
  - `calendar_filter`: event_block_start, event_clear
  - `day_of_week`: no triggers (day-level transitions aren't bar-level)
- New `detect_triggers()` function: detects state transitions from GP_ condition columns, returns boolean Series per trigger.
- GP Preview: trigger markers (blue arrows) rendered at state transitions on the chart.
- Note: GP triggers are NOT wired as strategy entry/exit triggers. They are for visualization/audit. The GEN- confluence record system remains the primary integration.

### 20C: Risk Management Bug Fixes & Validation
**Files: `src/triggers.py`, `src/risk_management_packs.py`**
- **Swing lookback**: Changed `df.iloc[start_idx:bar_index + 1]` to `df.iloc[start_idx:bar_index]` in both `calculate_stop_price()` and `calculate_target_price()`. More conservative and realistic for live trading. Added ATR fallback when lookback slice is empty.
- **ATR NaN logging**: Added `logger.debug()` when ATR falls back to 1% of entry price.
- **Builder clamping**: All RM builder functions now use `_clamp()` helper to enforce schema min/max ranges. Invalid `stop_method` in rr_ratio defaults to "atr".
- **validate_parameters()**: New function mirroring GP's implementation.
- **Docstring update**: `get_target_config()` documents that None means "stop/signal only; no profit target."

### 20D: Trailing Stops & Breakeven Stops
**Files: `src/triggers.py`, `src/risk_management_packs.py`, `src/app.py`**

#### Architecture
Trailing and breakeven are **modifier keys** on the stop_config dict:
```python
{
    "method": "atr", "atr_mult": 1.5,
    "trailing":  {"enabled": True, "method": "atr", "atr_mult": 1.0, "activation_r": 0.5},
    "breakeven": {"enabled": True, "activation_r": 1.0, "offset": 0.0}
}
```

#### Key Implementation
- `update_stop_price()` in triggers.py: Called each bar while in position. Checks breakeven first, then trailing. Stop only ratchets in favorable direction (LONG: up, SHORT: down).
- Wired into `generate_trades()` state machine: inserted before exit condition checks.
- `initial_stop_price` field added to trade records for visualization of stop movement.
- Two new RM templates: `atr_trailing` (ATR initial + ATR trailing) and `breakeven_stop` (ATR initial + breakeven at R threshold).
- Default packs (disabled): "ATR Trailing (Default)" and "Breakeven at 1R".
- Strategy Builder: trailing/breakeven checkboxes with method/param controls below the stop method selector.
- **R-multiple bug fix**: `risk = abs(entry_price - stop_price)` used the final (trailed) stop, shrinking the risk denominator and inflating R-multiples. Fixed to `risk = abs(entry_price - initial_stop)` so R is always measured against the original risk.

### 20E: Preview & UI Polish + Documentation
**File: `src/app.py`**
- RM Preview: stop level markers (red) and target level markers (green) on trade entry candles. Initial stop marker shown separately when trailing moved the stop.
- RM Preview: exit reason breakdown metrics (stop_loss %, target %, signal_exit %, etc.).
- GP Preview: empty-data warning when session filter results in zero bars.
- PRD: Phase 20 marked complete, version bumped to 0.45.

## Deferred Items
- **Partial profit taking**: Requires multi-leg trade tracking. Major refactor of generate_trades().
- **Pyramiding**: Requires multi-entry position model with blended cost basis.
- **Dynamic position sizing**: Breaks fixed risk_per_trade assumption throughout analytics.
- **Event calendar API**: Would replace proxy FOMC/NFP/OpEx logic and activate buffer_minutes.
- **GP triggers as entry/exit triggers**: Would require new trigger column infrastructure in strategy builder.
- **New GP templates**: Market regime (VIX), volatility filter (ATR), news blackout.
- **RM intra-bar exits**: Stops/targets evaluate tick-by-tick. Requires position tracking in streaming engine.

## Files Modified

| File | Changes |
|------|---------|
| `src/general_packs.py` | 20A: opex, DRY eval, weekend, UNKNOWN, validate; 20B: trigger defs, detect_triggers() |
| `src/risk_management_packs.py` | 20C: _clamp, builder clamping, validate; 20D: trailing/breakeven builders, templates, defaults |
| `src/triggers.py` | 20C: swing lookback, ATR logging; 20D: update_stop_price(), state machine wire, initial_stop |
| `src/app.py` | 20B: GP preview triggers; 20D: SB trailing/breakeven UI; 20E: RM preview markers, exit reasons, GP empty warning |
| `docs/RoR_Trader_PRD.md` | Phase 20 marked complete, v0.45 |
| `docs/Implementation_Spec_Phase_20.md` | Created |
