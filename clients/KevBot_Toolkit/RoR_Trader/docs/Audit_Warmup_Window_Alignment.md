# Audit — Warmup & Visible-Window Alignment

**Date:** 2026-05-26
**Trigger:** sid 248 showed 10 trades in Mass Builder preview, ~19 (display) /
31 (internal) after Update All Data. Diagnosis: not a cache-replay bug
(§8.5 is bit-perfect — verified locally). Root cause is **data-window
mismatch** between paths, plus inconsistent trimming.

## The unified principle Kevin wants

A strategy has a **frozen visible window** anchored at
`backtest_start_date` (the start of the data Mass Builder loaded when the
strategy was saved). Every read path should:

1. **Load** `[backtest_start_date − warmup, now]`
2. **Run** the engine over the full loaded window
3. **Trim** trades / bars / equity-curve / KPI inputs to
   `>= backtest_start_date` BEFORE returning to the caller

Currently only **one** of the eight paths trims (path #2 below) — the
rest leak warmup into user-visible output.

## Matrix — 8 data-loading paths

| Path | Anchor | Warmup | Trim to visible? | Engine |
|---|---|---|---|---|
| **1. `services.get_strategy_trades`** :744 | `lookback_start_date` or `forward_test_start` :778-779 | None for lookback; `*2` for FT | ❌ None | `run_unified_backtest` :794 |
| **2. `services.prepare_forward_test_data`** :800 | `forward_test_start_dt − data_days*2` :834 | **`*2` multiplier on `data_days`** :834 | ✅ `>= backtest_start_date` :861-866 | `run_unified_backtest` :856 |
| **3. `mass_builder.py`** ~805 | `data_days` raw (default 30) | **None — raw `data_days` only** | ❌ None | `run_unified_backtest` per combo |
| **4. `forward_test_service.append_new_backtest_trades`** :1256 | `since_dt = max(entry_fill_ts)` from DB :1335 | **100-bar fixed OR snapshot resume** :514, 584-622 | ❌ Caller filters by `entry_fill_ts > latest_ts_iso` :1443 | `get_strategy_trades_for_window` |
| **5. `forward_test_service._do_recompute`** :119 | Inherits path #1 :138 | Inherits path #1 | ❌ None | path #1 |
| **6. `strategies.get_strategy_chart_data`** :2152 | `data_days`; **inflates to 365d if any 1Day secondary** :2180-2184 | None explicit | ❌ None | `prepare_data_with_indicators` :2190 |
| **7. `data_worker_engine.tick_strategy`** :449 | Snapshot's `state.last_bar_ts` :463, 507 | **None — snapshot replaces warmup** :344, 362-369 | ❌ Filters by `until_dt` only :309, 317 | `run_store_fed_window` + snapshot resume |
| **8. `coarse_bar_store` + `data_worker.py`** | **Rolling 150-day**, no per-strategy awareness :37-39 | Fixed 150 calendar days | ❌ `_trim()` by window only :86 | Feeds path #7 via facade |

## Five inconsistencies

1. **Path #2 is the only path that trims to `backtest_start_date`.** All
   others surface warmup data as user-visible trades / bars / equity-curve
   points.

2. **Warmup magnitudes diverge wildly:**
   - `*2` multiplier (path #2)
   - 100-bar fixed (path #4 cold-start)
   - 365-day inflate-if-any-1Day-secondary (path #6 chart)
   - 150-day rolling (path #8 Tier-2)
   - Snapshot-only / zero new warmup (path #7 streaming)
   - Raw `data_days` (path #3 Mass Builder)

3. **Cross-TF cold-start cliff (the "live looks dead" pattern).** Path #4's
   100-bar warmup is fine for 1Min strategies but produces ~0.06 trading
   days of 1Day-secondary data — daily MACD undefined → confluence gates
   everything → zero trades for days while it converges. Same strategy in
   path #2 gets `*2 * data_days` which usually has enough.

4. **No path reads `backtest_start_date` at load time** — only at
   trim-time (path #2 only). Means every reload re-extends back to
   `now − data_days*2`, drifting the *loaded* window forward as wall
   time moves.

5. **Chart inflates differently than KPIs.** Path #6 jumps to 365 days if
   a 1Day secondary is present; path #2 multiplies `data_days * 2`. For a
   90-day strategy with 1Day MTF, the chart shows 365 days, KPI numerator
   uses ~180 days. Same strategy → two different lookback "facts."

## Live ↔ backtest divergence risks

- **Streaming engine (#7) survives on snapshots**, no warmup issue *while
  snapshot exists*. On snapshot invalidation (config edit / store_gap /
  container restart), falls through to path #4's 100-bar cold-start —
  cross-TF strategies can be dark for days.
- **Mass Builder ↔ Update All Data** divergence is exactly the sid 248
  case. Same root cause.
- **Chart-vs-KPI** isn't catastrophic but lets the user visually misread a
  strategy.

## Proposed unified helper

`services.load_data_for_strategy(strat, *, include_warmup=True)`:

- **Visible window** = `[strat.backtest_start_date, now]`
- **Warmup** = `max(data_days * 2, longest_indicator_lookback_bars)`
  - Keep `*2` as the safety floor (Kevin's preference: "rather too much
    than too little")
  - Extend if a particular indicator (e.g. 200-bar EMA on 1Day) demands
    more
- **Load** = `[visible_start − warmup_days, now]`
- **Returns** the loaded df, plus a `_visible_start` attribute the caller
  uses to trim trades / bars / equity curve before returning to UI.

Each of paths #1, #3, #4, #5, #6 gets refactored to call the helper. Paths
#7 (streaming snapshot) and #8 (rolling Tier-2 pool) intentionally don't
need it — they're internal infrastructure — but they'd surface their
non-conformance through a documented carve-out.

## Specific file locations for action

| Finding | File:Line |
|---|---|
| Warmup `*2` multiplier | `src/services.py:834` |
| Backtest trim (path #2 only) | `src/services.py:861-866` |
| Under-provisioned 100-bar cold-start | `src/services.py:514, 610` |
| Chart window dynamic scaling | `src/api/routers/strategies.py:2180-2184` |
| Streaming snapshot resume | `src/data_worker_engine.py:615-622, 642-652` |
| Tier-2 rolling 150d | `src/coarse_bar_store.py:37-39` |
| Append trim filter | `src/api/services/forward_test_service.py:1443` |
| Mass Builder raw `data_days` | `src/mass_builder.py:805` |

## Open questions to resolve before implementation

1. **`backtest_start_date` defaults for legacy strategies that don't have
   it set.** Fallback to `now − data_days` or to `forward_test_start`?
   Affects pre-§8.5 strategies that need to keep displaying consistently.

2. **Warmup-derivation rule precision.** Stick with `*2` floor + indicator
   lookback max, or compute exactly from the indicator parameter list?
   The latter is cleaner but means parsing every strategy's
   `confluence_groups` to extract longest period.

3. **Chart inflation policy** — keep the "inflate to 365d if 1Day
   secondary" behavior (path #6) or replace it with the unified helper?
   The chart's 365d is for visual context, not engine fidelity. May want
   to keep but make it explicit.
