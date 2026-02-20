# RoR Trader — Implementation Spec: Phase 19

**Version:** 1.0 — COMPLETE
**Date:** February 19, 2026
**Purpose:** Detailed, autonomous-implementation-ready spec for Phase 19: Intra-Bar Trigger Evaluation. Designed for a "Ralph Wiggum" loop — each sub-phase can be implemented without user validation between steps.
**Status:** All sub-phases (19A–19F) implemented. 18 intra-bar trigger companions created across 6 indicator groups. TriggerLevelCache, streaming tick evaluation, backtest level-fill pricing, chart price arrows, and alert badges all complete.

---

## Table of Contents

1. [Overview](#overview)
2. [Sub-Phase 19A: Level Map, TriggerLevelCache, Dual Trigger Definitions](#sub-phase-19a)
3. [Sub-Phase 19B: Wire Dual Triggers Through Detection & Drill-Down](#sub-phase-19b)
4. [Sub-Phase 19C: Streaming Engine Tick-Level Evaluation](#sub-phase-19c)
5. [Sub-Phase 19D: Backtest Approximation](#sub-phase-19d)
6. [Sub-Phase 19E: Entry/Exit Price Arrows on Charts](#sub-phase-19e)
7. [Sub-Phase 19F: Polish — Alert Display, Cooldown, PRD](#sub-phase-19f)
8. [Verification Checklist](#verification-checklist)

---

## Overview

**Goal:** Enable select triggers to evaluate tick-by-tick against pre-computed indicator levels instead of waiting for bar close. For each eligible trigger, create a companion `[I]` (intra-bar) version alongside the existing `[C]` (bar-close) version. Both appear in drill-down results so users can compare performance.

**Key principle:** Levels are extracted from indicators on each bar close. Between bar closes, ticks are compared against those cached levels. No full indicator re-run per tick — O(1) per tick per trigger.

**Execution order:** `19A → 19B → 19C → 19D → 19E → 19F` (19D and 19E can run in parallel after 19C)

### Level Extraction Map

Each intra-bar trigger maps to an indicator column and crossing direction:

| Trigger Base ID | Column | Cross Dir | Notes |
|---|---|---|---|
| `vwap_cross_above_ib` | `vwap` | above | |
| `vwap_cross_below_ib` | `vwap` | below | |
| `vwap_enter_upper_extreme_ib` | `vwap_sd2_upper` | above | |
| `vwap_enter_lower_extreme_ib` | `vwap_sd2_lower` | below | |
| `utbot_buy_ib` | `utbot_stop` | above | |
| `utbot_sell_ib` | `utbot_stop` | below | |
| `st_bull_flip_ib` | `st_line` | above | User pack |
| `st_bear_flip_ib` | `st_line` | below | User pack |
| `bb_cross_upper_ib` | `bb_upper` | above | User pack |
| `bb_cross_lower_ib` | `bb_lower` | below | User pack |
| `bb_cross_basis_up_ib` | `bb_basis` | above | User pack |
| `bb_cross_basis_down_ib` | `bb_basis` | below | User pack |
| `src_resistance_broken_ib` | `src_nearest_top` | above | User pack |
| `src_support_broken_ib` | `src_nearest_bot` | below | User pack |
| `ema_pp_cross_short_up_ib` | `ema_{short_period}` | above | Dynamic column |
| `ema_pp_cross_short_down_ib` | `ema_{short_period}` | below | Dynamic column |
| `ema_pp_cross_mid_up_ib` | `ema_{mid_period}` | above | Dynamic column |
| `ema_pp_cross_mid_down_ib` | `ema_{mid_period}` | below | Dynamic column |

**Excluded** (remain bar-close only): `vwap_return_to_vwap`, `rvol_spike`, `rvol_extreme`, all EMA stack triggers, all MACD triggers, `bb_squeeze_on/off`, `src_enter/exit_sr_zone`.

---

## Sub-Phase 19A

### Goal
Add `[I]` companion triggers alongside existing `[C]` triggers. Define the level extraction map. Rewrite `TriggerLevelCache` with proper crossing detection.

### Files to Modify

**1. `src/realtime_engine.py`** — TriggerLevelCache rewrite + INTRABAR_LEVEL_MAP

Add `INTRABAR_LEVEL_MAP` dict near the top of the file:
```python
INTRABAR_LEVEL_MAP: Dict[str, Dict[str, str]] = {
    "vwap_cross_above":          {"column": "vwap",            "cross": "above"},
    "vwap_cross_below":          {"column": "vwap",            "cross": "below"},
    "vwap_enter_upper_extreme":  {"column": "vwap_sd2_upper",  "cross": "above"},
    "vwap_enter_lower_extreme":  {"column": "vwap_sd2_lower",  "cross": "below"},
    "utbot_buy":                 {"column": "utbot_stop",      "cross": "above"},
    "utbot_sell":                {"column": "utbot_stop",      "cross": "below"},
    "st_bull_flip":              {"column": "st_line",         "cross": "above"},
    "st_bear_flip":              {"column": "st_line",         "cross": "below"},
    "bb_cross_upper":            {"column": "bb_upper",        "cross": "above"},
    "bb_cross_lower":            {"column": "bb_lower",        "cross": "below"},
    "bb_cross_basis_up":         {"column": "bb_basis",        "cross": "above"},
    "bb_cross_basis_down":       {"column": "bb_basis",        "cross": "below"},
    "src_resistance_broken":     {"column": "src_nearest_top", "cross": "above"},
    "src_support_broken":        {"column": "src_nearest_bot", "cross": "below"},
    "ema_pp_cross_short_up":     {"column": "ema_8",           "cross": "above", "param_key": "short_period"},
    "ema_pp_cross_short_down":   {"column": "ema_8",           "cross": "below", "param_key": "short_period"},
    "ema_pp_cross_mid_up":       {"column": "ema_21",          "cross": "above", "param_key": "mid_period"},
    "ema_pp_cross_mid_down":     {"column": "ema_21",          "cross": "below", "param_key": "mid_period"},
}
```

Note: For EMA triggers, `column` is the default. When `param_key` is present, `update_from_indicators()` resolves the actual column from the group's parameters (e.g., `ema_{short_period}` where `short_period` comes from the group config).

The map is keyed by the **base trigger name** (without `_ib` suffix). The `_ib` triggers share the same level spec — strip `_ib` to look up.

Rewrite `TriggerLevelCache` class (replace the existing stub at ~line 227):
```python
class TriggerLevelCache:
    """Cache of trigger levels for O(1) intra-bar price comparisons.

    On each bar close, update_from_indicators() extracts the latest indicator
    value for each active intra-bar trigger. The check() method compares
    tick prices against cached levels with crossing-direction tracking.
    """

    def __init__(self):
        self._levels: Dict[str, float] = {}            # key → level price
        self._cross_dir: Dict[str, str] = {}            # key → "above"/"below"
        self._prev_side: Dict[str, Optional[str]] = {}  # key → last side of level

    def update_from_indicators(self, strategy_id: int, trigger_base: str,
                                df: pd.DataFrame, group_params: Optional[dict] = None):
        """Extract the trigger level from the last bar's indicators."""
        base = trigger_base.removesuffix("_ib")
        spec = INTRABAR_LEVEL_MAP.get(base)
        if spec is None or len(df) == 0:
            return

        # Resolve column name (dynamic for EMA)
        col = spec["column"]
        if "param_key" in spec and group_params:
            period = group_params.get(spec["param_key"])
            if period is not None:
                col = f"ema_{period}"

        last_bar = df.iloc[-1]
        if col not in last_bar.index or pd.isna(last_bar[col]):
            return

        key = f"{strategy_id}:{trigger_base}"
        level = float(last_bar[col])
        close = float(last_bar["close"])

        self._levels[key] = level
        self._cross_dir[key] = spec["cross"]
        # Initialize prev_side from bar close relative to new level
        self._prev_side[key] = "above" if close > level else "below" if close < level else None

    def check(self, key: str, price: float) -> bool:
        """O(1) crossing detection. Returns True on first crossing in the required direction."""
        level = self._levels.get(key)
        if level is None:
            return False

        cross_dir = self._cross_dir.get(key)
        current_side = "above" if price > level else "below" if price < level else None
        prev = self._prev_side.get(key)
        self._prev_side[key] = current_side

        if prev is None or current_side is None:
            return False

        if cross_dir == "above" and prev == "below" and current_side == "above":
            return True
        if cross_dir == "below" and prev == "above" and current_side == "below":
            return True

        return False

    def clear(self):
        self._levels.clear()
        self._cross_dir.clear()
        self._prev_side.clear()
```

**2. `src/confluence_groups.py`** — Add `_ib` companion triggers

In each TEMPLATE dict that has intra-bar-capable triggers, add a companion entry. The companion has:
- `"base": "{original_base}_ib"` — suffix to distinguish
- `"name"`: same as original (the UI adds `[C]`/`[I]` tag automatically)
- `"execution": "intra_bar"`
- `"column_base": "{original_base}"` — NEW field, tells the system to share the boolean column
- Same `"direction"`, `"type"` as original

Example for the `vwap` template triggers:
```python
# Original (keep as-is):
{"base": "cross_above", "name": "Cross Above @V", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
# NEW companion:
{"base": "cross_above_ib", "name": "Cross Above @V", "direction": "LONG", "type": "ENTRY", "execution": "intra_bar", "column_base": "cross_above"},
```

Apply this pattern to ALL triggers listed in the Level Extraction Map above. Templates to modify:
- `vwap` template: `cross_above`, `cross_below`, `enter_upper_extreme`, `enter_lower_extreme`
- `utbot` template: `buy`, `sell`
- `ema_price_position` template: `cross_short_up`, `cross_short_down`, `cross_mid_up`, `cross_mid_down`

For user packs (SuperTrend, Bollinger Bands, SR Channels), update their `manifest.json` files to add companion triggers with `_ib` suffix, same pattern.

**3. `src/app.py`** — Update `_INTRABAR_CANDIDATE_TRIGGERS`

Remove from the set:
- All trigger bases that now have `[I]` companions (they're no longer "candidates", they're promoted)
- `rvol_spike`, `rvol_extreme` (not viable)
- `vwap_return_to_vwap` (too complex for simple level cross)

The set should be empty or contain only remaining future candidates.

### Verification 19A
- [x] Trigger lists in Outputs & Triggers show `[I]` tag for companion triggers (not `[I?]`)
- [x] Original triggers show `[C]` tag
- [x] RVOL triggers remain `[C]` (no companion)
- [x] `TriggerLevelCache` methods don't crash (tested via engine startup)

---

## Sub-Phase 19B

### Goal
Ensure `_ib` triggers share boolean columns with their `[C]` counterparts. Both versions appear in drill-down results and trigger dropdowns.

### Files to Modify

**1. `src/triggers.py`** — Handle `column_base` in trigger detection

In `detect_all_triggers()` (or wherever trigger boolean columns are created), when a trigger definition has a `column_base` field, do NOT create a separate boolean column. Instead, map the `_ib` trigger CID to the same column as the base trigger.

When `generate_trades()` looks up a trigger's boolean column, check for `column_base`:
```python
# When resolving the column for a trigger CID:
trigger_def = all_triggers[trigger_cid]
base = getattr(trigger_def, 'column_base', None) or trigger_def.base
column_name = f"trig_{group_id}_{base}_{trigger_type}"
```

**2. `src/app.py`** — Drill-down shows both `[C]` and `[I]` variants

In `analyze_entry_triggers()` and `analyze_exit_triggers()`, both trigger CIDs (original and `_ib`) are already in `all_trigger_map` (they were added in 19A). They should naturally appear in drill-down results since the analysis iterates all trigger CIDs.

The `[I]` variant will produce different KPIs than `[C]` because `generate_trades()` will use level-fill pricing for `[I]` triggers (implemented in 19D). Until 19D is done, both variants will show identical results (both use close price).

### Verification 19B
- [x] Entry trigger dropdown shows both `[C]` and `[I]` versions for VWAP Cross Above
- [x] MACD Cross Bull only shows `[C]` version
- [x] Drill-down lists both variants
- [x] Saving a strategy with an `[I]` trigger persists correctly
- [x] Loading old strategies (no `_ib` triggers) works fine

---

## Sub-Phase 19C

### Goal
Implement tick-level trigger evaluation in `SymbolHub.on_tick()`. Fire alerts mid-bar when price crosses a cached level.

### Files to Modify

**1. `src/realtime_engine.py`**

**Pass TriggerLevelCache to SymbolHub:**
Add `trigger_cache` parameter to `SymbolHub.__init__()`. In `start()`, pass `self._trigger_cache`.

**Modify `on_tick()`:**
```python
def on_tick(self, price, volume, timestamp):
    self.tick_count += 1
    self.last_tick_time = timestamp
    for tf_seconds, builder in self.builders.items():
        completed = builder.process_tick(price, volume, timestamp)
        if completed is not None:
            self._on_bar_close(tf_seconds, builder, timestamp)
    # Intra-bar evaluation on every tick
    self._check_intrabar_triggers(price, timestamp)
```

**Add `_check_intrabar_triggers()` method:**
- Iterate strategies on this symbol
- For each strategy, check if entry/exit trigger has `execution == "intra_bar"`
- For each intra-bar trigger: call `self._trigger_cache.check(key, price)`
- Session gate: skip if outside trading session
- Confluence gate: only fire if `self._confluence_met.get(strat_id, True)` is True
- Cooldown: use `min(tf_seconds / 2, 30)` seconds
- Dedup: track fired triggers in `self._intrabar_fired` set
- On fire: build signal dict with `"source": "intra_bar"`, `"price": price`
- Save alert + deliver via callback

**Modify `_on_bar_close()`:**
1. After `detect_signals()`, call `self._trigger_cache.update_from_indicators(...)` for each intra-bar trigger in the strategy
2. Store `self._confluence_met[strat_id]` = whether confluence was satisfied
3. Clear `self._intrabar_fired` entries for this strategy/TF
4. When checking bar-close triggers, skip triggers that already fired intra-bar this bar (dedup)

**Add state dicts to SymbolHub:**
```python
self._confluence_met: Dict[int, bool] = {}        # strategy_id → bool
self._intrabar_fired: Dict[str, Set[str]] = {}    # "strat_id:tf" → {trigger_bases}
```

### Verification 19C
- [x] Engine starts with an `[I]` trigger strategy without error
- [x] `streaming_engine.log` shows "Intra-bar alert" messages on level crossings
- [x] No duplicate alert at bar close for same trigger
- [x] `[C]` trigger strategies still alert normally at bar close
- [x] Rapid price oscillation around level doesn't spam alerts (cooldown works)

---

## Sub-Phase 19D

### Goal
Modify `generate_trades()` to use level-fill pricing for `[I]` triggers in backtests.

### Files to Modify

**1. `src/triggers.py`** — Level-fill pricing in `generate_trades()`

When the entry trigger's execution type is `"intra_bar"`:
```python
# Look up execution type from trigger definition
trigger_def = all_triggers.get(entry_trigger_cid)
is_intrabar = getattr(trigger_def, 'execution', 'bar_close') == 'intra_bar'

if is_intrabar:
    base = getattr(trigger_def, 'column_base', trigger_def.base) if hasattr(trigger_def, 'column_base') else trigger_def.base
    level_spec = INTRABAR_LEVEL_MAP.get(base)
    if level_spec and level_spec['column'] in row.index:
        level = row[level_spec['column']]
        if pd.notna(level):
            cross_dir = level_spec['cross']
            if cross_dir == "above" and row['high'] >= level:
                entry_price = float(level)
            elif cross_dir == "below" and row['low'] <= level:
                entry_price = float(level)
            else:
                continue  # level not reached within bar
        else:
            entry_price = row['close']
    else:
        entry_price = row['close']
else:
    entry_price = row['close']
```

Same pattern for exit triggers with `execution == "intra_bar"`.

**Import note:** `INTRABAR_LEVEL_MAP` should be importable from a shared location. If circular import issues arise, extract the map to `data_loader.py` or a new small module.

**2. `src/alerts.py`** — Same logic in `detect_signals()`

Ensure `detect_signals()` passes trigger definitions through so `generate_trades()` can resolve execution types.

### Verification 19D
- [x] Strategy with `[I]` VWAP trigger: entries fill at VWAP level, not close
- [x] Strategy with `[C]` VWAP trigger: entries fill at close (unchanged)
- [x] Drill-down now shows different KPIs for `[C]` vs `[I]` variants
- [x] If bar H/L doesn't reach level, `[I]` entry skipped

---

## Sub-Phase 19E

### Goal
Add entry/exit price markers on price chart candle bodies. Left ▸ arrow (target price) always shown. Right ◂ arrow (alert price) only shown when real alerts are recorded.

### Files to Modify

**1. `src/app.py`** — Chart rendering (~line 2494)

**Left ▸ arrow (target price):** Add a Line series overlay with data points at `(entry_time, entry_price)` and `(exit_time, exit_price)` from the trades DataFrame. Style with `lineWidth: 0` and point markers (or small line segments if point markers not supported). Blue for entries, green/red for exits.

**Right ◂ arrow (alert price):** Load recorded alerts from `alerts.json` for the current strategy. Match alerts to trades by strategy ID + approximate timestamp. Add a second Line series with data points at `(alert_time, alert_price)`. Only render when alert tracking is active and matching alerts exist.

**Matching logic:** An alert within a small time window (e.g., 2 bars) of a trade's entry_time, with the same strategy_id, corresponds to that trade's entry.

### Verification 19E
- [x] Left ▸ arrow appears at entry_price on entry candles
- [x] Left ▸ appears for ALL strategies (backtest and live)
- [x] Right ◂ only appears when alert tracking active with recorded alerts
- [x] Gap between arrows visible for `[I]` triggers with real alerts
- [x] Existing below/above bar arrows unchanged

---

## Sub-Phase 19F

### Goal
Show "Intra-bar" badge on alerts, tune cooldown, update PRD.

### Files to Modify

**1. `src/app.py`** — Alert display: show "Intra-bar" badge next to alerts where `source == "intra_bar"`.

**2. `src/realtime_engine.py`** — Tune cooldown for intra-bar triggers if needed based on testing.

**3. `docs/RoR_Trader_PRD.md`** — Mark Phase 19 complete, update version.

**4. `docs/Implementation_Spec_Phase_19.md`** — Final status update.

### Verification 19F
- [x] Alerts page shows "Intra-bar" badge on intra-bar alerts
- [x] PRD updated with Phase 19 completion
- [x] Implementation spec marked complete

---

## Verification Checklist (Full Phase)

### 19A — Foundation
- [x] `[I]` tags appear for companion triggers
- [x] `[C]` tags remain on original triggers
- [x] RVOL/vwap_return_to_vwap stay `[C]`
- [x] TriggerLevelCache doesn't crash on engine start

### 19B — Drill-Down
- [x] Both `[C]` and `[I]` in trigger dropdowns
- [x] MACD has only `[C]`
- [x] Drill-down lists both variants
- [x] Save/load with `[I]` triggers works
- [x] Old strategies backward compatible

### 19C — Streaming
- [x] Engine starts with `[I]` strategy
- [x] Intra-bar alerts in log on level crossings
- [x] No bar-close duplicate
- [x] `[C]` strategies unaffected
- [x] Cooldown prevents spam

### 19D — Backtest
- [x] `[I]` entries at level price
- [x] `[C]` entries at close (unchanged)
- [x] Different KPIs between `[C]` and `[I]` in drill-down
- [x] Level not reached → entry skipped

### 19E — Chart Arrows
- [x] Left ▸ at entry_price, always
- [x] Right ◂ only with real alerts
- [x] Slippage gap visible
- [x] Existing markers unchanged

### 19F — Polish
- [x] "Intra-bar" badge on alerts
- [x] PRD + spec updated
