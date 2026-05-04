# Plan — Saturday 2026-05-02: Rebroadcast Fix + Lab Phase 2 + Models + Engine Truth Capture

## Context

Friday EOD we discovered that `BarBuilder.accept_bar` doesn't check for duplicate timestamps. When Polygon's WebSocket rebroadcasts a corrected version of the same bar within their 15-min FINRA late-print window, our worker treated each rebroadcast as a NEW bar — appending a duplicate row to history and double-applying the indicator update. This corrupts the engine's view of the world and likely contributes meaningfully to the algo-vs-live divergence Kevin's been chasing for months.

Saturday work splits into 6 milestones, M1-M3 already shipped earlier today, M4-M6 pending:

1. **M1 — Duplicate-bar bug fix** (✅ shipped, commit `a15461f`)
2. **M2 — Lab Phase 2 (cache-derived Alert Lens)** (✅ shipped, commit `7bd71e7`)
3. **M3 — Models-as-strategy-variable placeholder** (✅ shipped, commit `8b40eff`)
4. **M4 — Alert indicator snapshot (Option A)** — capture engine state at alert-fire moment to a new JSONB field on alerts. Time-sensitive: state exists only at fire moment, can't be backfilled.
5. **M5 — Lab tab replay mode (Option B)** — embed existing ReplayableChart + ReplayControls into a new `ChartReplayCard` consuming chart-data-cache. Lets user scrub bar-by-bar through the cache and watch indicator/heatmap evolution.
6. **M6 — Engine truth state capture** — new `bar_engine_states` table + writer that snapshots `monitor.indicators.state.current` at every bar close. Pure data-capture today (no read/visualization); collection now means we can build engine-truth replay later if cache-reconstruction precision turns out insufficient.

Investigation Saturday confirmed:
- Bug exists in 3 code paths, but the underlying buggy method is shared — fixing `accept_bar` (line 348) + `accept_second_bar` (line 237) covers all three. ✅ shipped.
- `IncrementalIndicatorEngine.warmup()` accumulates state (doesn't reset). New `recompute_from_history` method handles this. ✅ shipped.
- The `prepare_data_with_indicators` pipeline is 80% reusable for the cache path — only the load-from-REST step needs replacement. ✅ shipped.
- `_alert_to_row` (db.py:248) auto-routes any non-column field into the `data` JSONB blob. **No alerts table migration needed for M4** — just attach `indicator_snapshot` to the alert dict before save.
- Existing replay components (`ReplayableChart.tsx`, `ReplayControls.tsx`, `useScenarioReplay.ts`) are used on Execution Types pages. Path A (slim `ChartReplayCard` consuming chart-data-cache shape directly) is cleaner than adapting `ScenarioReplayCard` (which assumes trade scenarios + workflow steps).
- Storage estimate for M6: ~13 MB/day across 27 strategies (state JSONB is bigger than OHLCV); ~4.7 GB/year. Manageable.

## Approach overview

**M1-M3 already shipped earlier today. M4-M6 pending:**

- **M4 (~1.5 hours)**: Alert indicator snapshot — worker captures state at fire time, frontend displays in alert history table tooltip
- **M5 (~2 hours)**: ChartReplayCard component + Lab tab Replay toggle
- **M6 (~1 hour)**: bar_engine_states table + writer module (data capture only, no viz today)

Each independently shippable. Total ~4.5 hours.

---

## M1 — Duplicate-bar bug fix

### M1.1 — Backup branch

```bash
git checkout dev && git pull
git branch dev-backup-pre-rebroadcast-fix-2026-05-02 dev
git push origin dev-backup-pre-rebroadcast-fix-2026-05-02
```

Rollback target if anything goes sideways.

### M1.2 — Fix `accept_bar` (line 348)

File: `src/ralph_engine.py`. The fix detects when the incoming bar's `bar_start` matches the last history row, and replaces instead of appending. Returns a tuple `(bar_dict, was_duplicate: bool)` so the caller knows what happened.

```python
def accept_bar(self, bar_dict: dict) -> tuple[dict, bool]:
    """Accept a pre-built bar (from Polygon WebSocket).

    Returns (bar_dict, was_duplicate).
    - was_duplicate=False: new bar, history appended, bar_count incremented
    - was_duplicate=True: Polygon rebroadcast for the most recent bar, history
                          row replaced in-place, bar_count NOT incremented
    """
    bar_ts = pd.Timestamp(bar_dict['timestamp'])
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.tz_localize('UTC')
    bar_start = self._align_to_period(bar_ts.to_pydatetime())

    # Detect duplicate: same bar_start as the last history row
    if len(self.history) > 0:
        last_ts = self.history.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')
        if last_ts == pd.Timestamp(bar_start, tz='UTC'):
            self.history.iloc[-1] = pd.Series({
                'open': float(bar_dict['open']),
                'high': float(bar_dict['high']),
                'low': float(bar_dict['low']),
                'close': float(bar_dict['close']),
                'volume': float(bar_dict.get('volume', 0)),
            })
            return bar_dict, True

    # Existing path: gap-fill + append
    # ... (unchanged from current implementation)
    self._append_to_history(bar_dict)
    self._bar_count += 1
    self._partial = None
    return bar_dict, False
```

### M1.3 — Fix `accept_second_bar` (line 237)

Same pattern. The trickier method because it has the partial-bar state machine. The duplicate case here is when `bar_start` is `<=` the last history row's timestamp AND `close_on_boundary=True`:

```python
def accept_second_bar(self, bar_dict: dict,
                      close_on_boundary: bool = True) -> tuple[Optional[dict], bool]:
    """... existing docstring ...

    Returns (completed_bar_or_None, was_duplicate).
    """
    # ... existing parsing of ts, period_start ...

    # NEW: duplicate detection only matters for close_on_boundary=True;
    # for chart-visual mode (False) the partial gets discarded anyway.
    if close_on_boundary and len(self.history) > 0:
        last_ts = self.history.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')
        if last_ts == pd.Timestamp(period_start, tz='UTC'):
            # Polygon rebroadcast: replace the last completed bar with
            # corrected aggregation. We have to RECOMPUTE the bar from
            # the per-second input though — the duplicate check is
            # really "is this period_start the same as our last close?"
            # NOTE: per-second corrections may aggregate into the same
            # period over multiple events. For simplicity, treat each
            # close_on_boundary call to a duplicate period as a full
            # replace of OHLCV using the per-second values — the caller
            # is expected to feed corrections one at a time.
            self.history.iloc[-1] = pd.Series({
                'open': float(bar_dict['open']),
                'high': float(bar_dict['high']),
                'low': float(bar_dict['low']),
                'close': float(bar_dict['close']),
                'volume': float(bar_dict.get('volume', 0)),
            })
            return bar_dict, True

    # ... existing partial-bar logic, returning (completed, False)
```

### M1.4 — Add `recompute_from_history` to IncrementalIndicatorEngine

Search `class IncrementalIndicatorEngine` in `src/ralph_engine.py`. Add:

```python
def recompute_from_history(self, df: pd.DataFrame) -> None:
    """Reset all indicator state and replay every bar in df incrementally.

    Used when a bar in history was corrected (Polygon rebroadcast).
    Equivalent to calling warmup() on a fresh engine instance.
    """
    # Reset all stateful fields
    self.state = {}              # ema, macd_ema, atr, vwap accumulators
    self._initialized = False
    # ... whatever else accumulates state in this class ...
    # Then replay
    self.warmup(df)
```

Read the existing class to identify ALL state fields that need clearing (the explore agent identified `state` dict + `_initialized` but there may be more).

### M1.5 — Wire callers

In `on_polygon_bar` (line 1506) and `on_second_bar` (line 1588):
- Capture the new `was_duplicate` return value
- If duplicate: do NOT call `_run_monitor_pipeline_for_completed_bar` incrementally; instead call a new method that recomputes indicators from history and re-evaluates triggers on the corrected state
- If new: existing behavior

The trigger re-evaluation question: do we re-fire alerts on a corrected bar? **No.** Once an alert is saved, it's fact. The indicator state recomputes silently for FUTURE bar evaluations; the corrected bar's own trigger evaluation just updates internal state without firing alerts.

### M1.6 — Replay test

New file: `src/test_rebroadcast_recompute.py`. Pattern follows `test_unified_parity.py`:

```python
def test_rebroadcast_correction_yields_clean_recomputation():
    """Feed (bar X uncorrected, bar X+1, bar X corrected) and verify the
    engine's final state matches what we'd get from a fresh engine fed
    only (bar X corrected, bar X+1)."""
    
    bars_v1 = make_test_bars(...)  # uncorrected sequence
    bars_v2 = make_test_bars(...)  # bar X has corrected close
    
    # Engine A: fresh feed of corrected sequence
    engine_a = IncrementalIndicatorEngine(...)
    engine_a.warmup(bars_v2)
    state_a = engine_a.state.copy()
    
    # Engine B: feed v1 sequence then a duplicate-bar correction for X
    engine_b = IncrementalIndicatorEngine(...)
    engine_b.warmup(bars_v1)
    # Simulate the rebroadcast correction
    history_b = builder_b.history.copy()  # has duplicate bar X
    # ... apply M1.2 fix path: replace last bar, recompute_from_history
    history_b.iloc[-N] = corrected_bar_x
    engine_b.recompute_from_history(history_b)
    state_b = engine_b.state.copy()
    
    # Assert state_a ≈ state_b (within float tolerance)
    assert_state_equivalent(state_a, state_b)
```

Validates Option B is mathematically correct.

### M1.7 — Deploy + worker restart

After M1.1–M1.6 pass tests + tsc + parity regressions:
- Commit + push to `dev`
- Watch Railway deploy via CLI (worker SUCCESS)
- After deploy, worker restarts naturally — in-memory history rebuilds clean from REST warmup
- Confirm worker logs show normal operation

### M1.8 — Offline validation (this weekend)

Markets are closed all weekend, so live cache writes won't happen. M1 validation today is purely offline:

- **Replay test passes** (M1.6) — proves the math is correct
- **Existing regression tests still pass** — `pytest src/test_parity_regression.py`, `pytest src/test_unified_parity.py`
- **Worker boots cleanly** — pull `railway logs --service Worker` after deploy; confirm engine starts, warmup completes against REST data, periodic-tasks loop runs without errors. (No live bars to process, but the worker should sit happily idle waiting for Monday's market open.)
- **Run existing offline test on yesterday's cached data** — `_validate_live_bars_cache.py` against Friday's data should still show the same numbers as before (the fix doesn't retroactively change historical cache rows; it only changes future write behavior)

### M1.9 — Live validation (Monday during RTH)

Defer to Monday morning when markets reopen:
- After ~30 min of recording: cache should show `first_close` ≠ `close` on multiple 1Min bars (proving Polygon rebroadcasts now correctly update `close` while preserving `first_close`)
- For comparison, Friday's data showed AAPL 1Min ~13% of bars had corrections, AMD ~18% — Monday should show similar or higher rates now that the writer is correctly capturing them
- Document Monday's drift numbers as the post-fix baseline
- Spot-check: pick 5-10 1Min bars where `first_close ≠ close` and confirm the delta is small ($0.01–$0.10 typical) — outliers >$0.50 deserve a look but are not necessarily bugs (could be real late-print events)

---

## M2 — Lab Phase 2: chart-data-from-cache

### M2.1 — Refactor `prepare_data_with_indicators` to accept pre-loaded DF

File: `src/services.py`. Add an optional `df` parameter:

```python
def prepare_data_with_indicators(
    symbol: str, days: int = 30, ...,
    df: Optional[pd.DataFrame] = None,  # NEW
) -> pd.DataFrame:
    """... existing docstring ...

    If `df` is provided, skip the load_market_data call and use the
    provided DataFrame as the primary OHLCV source. The indicator/
    interpreter/trigger pipeline runs on it identically.
    """
    if df is None:
        df = load_market_data(symbol, days=days, seed=seed,
                              start_date=start_date, end_date=end_date,
                              timeframe=timeframe, data_feed=data_feed,
                              session=session)
    # ... existing pipeline runs on df from here ...
```

For secondary TFs, currently the function resamples primary 1Min to coarser TFs. With cache mode, we instead want to FETCH cache rows for each secondary TF directly (avoids the WS-aggregation-vs-resample mismatch).

Add a parallel parameter `secondary_tf_dfs: Optional[dict[str, pd.DataFrame]] = None`. When provided, use these instead of resampling.

### M2.2 — New helper `fetch_cache_as_df`

In `src/api/routers/strategies.py` or a new helper module:

```python
def fetch_cache_as_df(
    symbol: str,
    tf_seconds: int,
    start: datetime,
    end: datetime,
    value_type: str = 'latest',  # 'latest' or 'first'
) -> pd.DataFrame:
    """Pull live_bars rows for (symbol, tf, time range), return as
    DataFrame with UTC DatetimeIndex and OHLCV columns. Picks first_*
    or close depending on value_type."""
    # paginate over c.table('live_bars').select(...)
    # build DataFrame with index = bar_start
    # if value_type='first': use first_open/high/low/close/volume cols
    # else: use open/high/low/close/volume
    # return df
```

Reuses the fetch pattern from existing `/cache-bars` endpoint at `src/api/routers/strategies.py:1666–1760`.

### M2.3 — Factor `_build_chart_response` helper

The current `get_strategy_chart_data` endpoint (lines 1201–1452) has three stages:
1. Data prep (calls prepare_data_with_indicators)
2. Indicator classification (overlay/oscillator/heatmap)
3. Serialization

Factor stages 2+3 into a helper that takes `(df, strat)` and returns `(chart_data, overlay_indicators, oscillator_indicators, heatmap_conditions, candle_color_column)`. Reuse from both endpoints.

### M2.4 — New endpoint `/strategies/{id}/chart-data-cache`

```python
@router.get("/{strategy_id}/chart-data-cache")
def get_strategy_chart_data_from_cache(
    strategy_id: int,
    value_type: str = Query("latest", description="'latest' or 'first'"),
    days: int = Query(None),
    user=Depends(get_current_user),
):
    strat = _get_or_404(strategy_id, user)
    
    # Determine primary + secondary TFs from strategy
    sec_tfs = ...  # same logic as existing chart-data
    
    # Fetch primary cache as DF
    primary_df = fetch_cache_as_df(strat['symbol'], primary_tf_seconds,
                                    start, end, value_type)
    
    # Fetch each secondary TF cache as DF
    sec_dfs = {tf_label: fetch_cache_as_df(strat['symbol'], tf_seconds,
                                            start, end, value_type)
               for tf_label, tf_seconds in sec_tf_map.items()}
    
    # Run pipeline with injected DFs
    df = svc.prepare_data_with_indicators(
        strat['symbol'], df=primary_df,
        secondary_tf_dfs=sec_dfs,
        timeframe=strat['timeframe'],
        session=strat['trading_session'],
    )
    
    # Build response via shared helper
    return _build_chart_response(df, strat)
```

### M2.5 — Frontend hook + Lab tab wiring

In `frontend/src/hooks/queries/useStrategies.ts`, add:

```typescript
export function useStrategyChartDataCache(
  id: number | null,
  valueType: 'latest' | 'first',
  enabled: boolean = true,
) {
  return useQuery({
    queryKey: ['strategy-chart-data-cache', id, valueType],
    queryFn: () => apiFetch<ChartDataResponse>(
      `/api/strategies/${id}/chart-data-cache?value_type=${valueType}`
    ),
    enabled: enabled && id !== null,
    staleTime: 30000,
    retry: 1,
  });
}
```

In `frontend/src/views/StrategyDetailPage.tsx`, update the Lab tab's `labChartTabData` useMemo to consume `chartDataResp` from the NEW hook (when in WS modes) instead of the parent `chartDataResp`. The right-side chart now has indicators and heatmap derived from cache, properly representing the engine's view.

Remove the orange "Phase 1 caveat" warning on the Alert Lens since the limitation is fixed.

### M2.6 — Visual validation

After deploy, open a strategy with known recent alerts. Pick an alert from `alerts` table that has populated `confluence_records`. On the Lab tab, verify:
- Right-side heatmap at the alert's `fill_ts` shows the same confluence states recorded in `alert.confluence_records`
- Right-side EMA/MACD lines pass through reasonable values
- Side-by-side comparison with Algo Lens (REST) shows visible drift on knife-edge bars

---

## M3 — Models-as-strategy-variable placeholder

### M3.1 — Schema (config JSONB extension)

No table migration. Strategies already have a `config` JSONB column. Add two fields under config:
- `backtest_model: str` (default `'rest_only'`)
- `live_model: str` (default `'ws_with_corrections'` — i.e., Option B post-fix)

The fields are stored in the existing config blob. Old strategies missing them are treated as defaults via a get-or-default pattern in the API.

### M3.2 — Backend exposure

In `src/api/routers/strategies.py`:
- `get_strategy()` — return `backtest_model` and `live_model` (default-fill if missing)
- `update_strategy()` / `update_strategy_config()` — accept these fields in PATCH bodies
- Don't actually USE the values yet — they're recorded but the engine ignores them

Document allowed values in a constants module (e.g., `src/strategy_models.py`):

```python
BACKTEST_MODELS = {
    'rest_only':                {'label': 'REST only',                   'available': True,  'description': 'Polygon REST settled bars (current default)'},
    'rest_hifi':                {'label': 'REST + Hi-Fi Pass 2',         'available': True,  'description': 'REST bars with Hi-Fi 1-second refinement'},
    'rest_with_cache_overlay':  {'label': 'REST + WS cache overlay',     'available': False, 'description': 'REST for pre-cache periods, WS cache for post-cache (coming soon)'},
    'cache_only':               {'label': 'WS cache only',               'available': False, 'description': 'Only periods since cache started (coming soon)'},
    'cache_first':              {'label': 'WS cache (decision-time)',    'available': False, 'description': 'first_close — what live engine actually saw (coming soon)'},
}
LIVE_MODELS = {
    'ws_with_corrections':      {'label': 'WS + Polygon corrections',     'available': True,  'description': 'Apply Polygon WS rebroadcast corrections within 15-min window (Option B, default after 2026-05-02 fix)'},
    'ws_first_lock':            {'label': 'WS first-write (locked)',      'available': False, 'description': 'Lock the bar at first WS write, ignore subsequent rebroadcasts (Option C — coming soon, awaiting Monday TV stability test)'},
}
```

Expose these via a new endpoint `GET /api/strategies/models` so the frontend can populate the dropdown without hardcoding.

### M3.3 — Frontend selector + badge

In the strategy edit / create form (find the relevant page in `frontend/src/views/`):
- Two dropdowns: "Backtest Model" and "Live Model"
- Populate from `/api/strategies/models`
- Disable options marked `available: false` with a "Coming soon" tooltip
- Default values: `rest_only` and `ws_with_corrections`

In the strategy detail page header:
- Add two small badges showing the active models
- Click → tooltip with description from the models endpoint

### M3.4 — Documentation

Add to `docs/Strategy_Models_2026-05-02.md`:
- Why models exist (the algo-vs-live divergence issue)
- What each model means
- When changing matters (live_model has trading impact; backtest_model is safe)
- The roadmap from placeholder to behaviorally-active

---

---

## M4 — Alert indicator snapshot (Option A)

### M4.1 — Worker: capture state at alert-fire moment

File: `src/ralph_engine.py`, `AlertDispatcher.dispatch()` around line 851. After the alert dict is initially built (with type, trigger, price, bar_time, etc.) and BEFORE `save_alert()` is called, attach the indicator snapshot:

```python
# M8.7 M4 (2026-05-02): snapshot engine indicator state at alert-fire moment.
# Flows into alerts.data JSONB via _alert_to_row's pass-through (any field
# not in ALERT_COLUMN_FIELDS is auto-routed into the JSONB blob).
try:
    if monitor and hasattr(monitor, 'indicators') and \
            hasattr(monitor.indicators, 'state'):
        snapshot = dict(monitor.indicators.state.current or {})
        # Coerce all values to JSON-safe scalars (state.current is
        # already Dict[str, float|bool] but be defensive).
        alert['indicator_snapshot'] = {
            k: (float(v) if isinstance(v, (int, float)) else
                bool(v) if isinstance(v, bool) else
                str(v))
            for k, v in snapshot.items()
        }
except Exception as e:
    logger.warning("Failed to snapshot indicator state for alert: %s", e)
```

No schema migration needed — `_alert_to_row` (db.py:248) splits the alert dict into column fields + JSONB. Anything not in `ALERT_COLUMN_FIELDS` lands in `data`.

### M4.2 — Frontend: display in alert history table

File: `frontend/src/views/StrategyDetailPage.tsx`. The alert history table on Chart & Trades and Lab tabs uses `recentAlerts` (built from raw alerts). Each alert row already supports tooltips (price column has `priceTooltip` at line 3013-3022). Extend the row with:

- Add a small "📊" icon next to the entry trigger that, on hover, shows a tooltip listing the snapshot key/value pairs (e.g., `ema_9: 722.41, macd_line: 0.07, ...`)
- Pull `indicator_snapshot` from the raw alert's `data` blob: in `recentAlertEvents` map (line 972), expose `(a.data?.indicator_snapshot ?? null)` as `indicatorSnapshot` field.

Shape on display: ~10-30 lines depending on strategy's required indicators. Format with dollar prefix for prices, raw for ratios/booleans.

### M4.3 — Validation

After deploy:
- Wait for next live alert (Monday RTH)
- Query `SELECT id, data->'indicator_snapshot' FROM alerts ORDER BY id DESC LIMIT 5;` — confirm new alerts have populated snapshot
- Open Strategy Detail → hover an alert in the history table → tooltip renders snapshot

Older alerts (pre-deploy) won't have snapshots; the icon is conditionally shown only when `data.indicator_snapshot` exists.

---

## M5 — Lab tab replay mode (Option B)

### M5.1 — Build `ChartReplayCard` component

New file: `frontend/src/components/ChartReplayCard.tsx`. Slim variant of `ScenarioReplayCard` that:

- Takes the chart-data-cache response shape directly: `{ chart_data, overlay_indicators, oscillator_indicators, heatmap_conditions }` plus the strategy's `direction`/`tfMs`/`chartPrefs`
- Manages `currentTime` state internally (starts at end = fully revealed; user clicks Reset to scrub from start)
- Slices `chart_data` and overlay/oscillator data by index based on currentTime
- Renders via existing `ReplayableChart` (the prop-driven LWC wrapper) + `ReplayControls`
- Skips workflow trace (no scenario/trade-specific concept)
- Skips entry/exit drill-down 1-sec charts (Lab tab is window-level, not single-trade)

Reuse `useScenarioReplay`-style truncation logic (lines 215-257 of `useScenarioReplay.ts`) but for a generic chart, not a single-trade scenario.

### M5.2 — Integrate into Lab tab

File: `frontend/src/views/StrategyDetailPage.tsx`. On the Chart & Trades (Lab) tab, add a "Replay" toggle button next to the First-write/Latest sub-toggle on the Alert Lens header. When toggled on, replace the static `SyncedChartPane` rendering with `<ChartReplayCard>` consuming the same `labChartTabData.chartPanes` source data. When toggled off, revert to static.

(Optional enhancement, defer if time-tight: also add a Replay toggle on the Algo Lens left side. Same logic, but consuming chartTabData/REST chart-data response.)

### M5.3 — Validation

- Open a strategy with Friday-or-later cache data on the Lab tab
- Toggle Replay on Alert Lens
- Confirm: chart starts fully revealed (matches static state)
- Click Reset → chart goes empty
- Click step-forward → bars appear one at a time
- Indicators / heatmap evolve as bars appear
- Scrub the progress bar → chart jumps to that bar

---

## M6 — Engine truth state capture (data-only foundation)

### M6.1 — Schema migration

New file: `src/migrations/bar_engine_states_table.sql`. Mirrors the `live_bars` patterns:

```sql
CREATE TABLE IF NOT EXISTS bar_engine_states (
  strategy_id        INT              NOT NULL,
  bar_start          TIMESTAMPTZ      NOT NULL,
  indicator_state    JSONB            NOT NULL,
  source             TEXT             NOT NULL DEFAULT 'ws',
  written_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
  PRIMARY KEY (strategy_id, bar_start)
);
CREATE INDEX IF NOT EXISTS bar_engine_states_lookup_idx
  ON bar_engine_states (strategy_id, bar_start);
ALTER TABLE bar_engine_states DISABLE ROW LEVEL SECURITY;
```

Kevin applies in Supabase SQL editor.

### M6.2 — Writer module

New file: `src/bar_engine_state_writer.py`. Exact pattern of `live_bars_writer.py`:

- `is_enabled()` — env var `BAR_ENGINE_STATE_WRITE_ENABLED` (default off)
- `_executor` — small ThreadPoolExecutor (2 workers, lazy init)
- `write_state(strategy_id, bar_start, indicator_state, source='ws')` — submit to executor, fire-and-forget, never raise to caller
- Internal `_write_sync` — upsert via supabase-py with `on_conflict='strategy_id,bar_start'`
- `shutdown(wait=True)` — drain pool on engine stop

### M6.3 — Wire into bar-close pipeline

File: `src/ralph_engine.py`, `_run_monitor_pipeline_for_completed_bar` after `monitor.on_bar_close()` succeeds (around line 1509) and before alert dispatch:

```python
# M8.7 M6 (2026-05-02): capture engine state at bar close for future
# engine-truth replay/analysis. Fire-and-forget, env-flag gated.
try:
    from bar_engine_state_writer import write_state
    write_state(
        strategy_id=monitor.strat_id,
        bar_start=bar_dict.get('timestamp'),
        indicator_state=dict(monitor.indicators.state.current or {}),
        source='ws',
    )
except Exception:
    pass  # never block bar processing on state capture
```

Important: only fire when `was_duplicate=False` for the parent accept_bar call. On duplicate (Polygon rebroadcast), the state was already snapshotted on the original bar; we don't want a second row.

### M6.4 — Worker shutdown drain

In `RalphEngine.stop()` (line 1959), add a shutdown call alongside `live_bars_writer.shutdown(wait=True)`:

```python
try:
    from bar_engine_state_writer import shutdown as _bes_shutdown
    _bes_shutdown(wait=True)
except Exception:
    pass
```

### M6.5 — Railway env flag

After deploy, set `BAR_ENGINE_STATE_WRITE_ENABLED=true` on the Worker service via:
```
railway variables --service Worker --skip-deploys --set "BAR_ENGINE_STATE_WRITE_ENABLED=true"
```

### M6.6 — Validation (Monday)

- Run after market open: `SELECT count(*), min(written_at), max(written_at) FROM bar_engine_states WHERE bar_start > '2026-05-04'::date;`
- Confirm rows are accumulating (~14k rows/day expected)
- Spot-check a row: `SELECT strategy_id, bar_start, indicator_state FROM bar_engine_states ORDER BY written_at DESC LIMIT 5;`
- Confirm `indicator_state` JSONB has the expected keys (ema_*, macd_*, atr, etc.)

NOTE: M6 is data-capture only today. No read path, no UI integration. The point is to start capturing now so we have data to draw from when/if engine-truth-precision replay becomes a real feature.

---

## Monday follow-ups (live validation + TV stability test)

After markets open Monday RTH:

### Mon-1 — Validate M1 fix in production
- After 30+ min of trading: run `_validate_live_bars_cache.py --hours-back 1` on representative symbols
- Look for `first_close ≠ close` on 1Min bars (expected ~10-20% rate per Friday's correction patterns)
- Spot-check correction magnitudes — typical $0.01-$0.10, outliers >$0.50 worth a look
- Compare worker log entries at T+0 vs T+15min for the same bar — should see indicator values shift if a correction landed

### Mon-2 — Validate M4 alert snapshots
- After first live alerts fire: `SELECT id, type, strategy_id, data->'indicator_snapshot' FROM alerts WHERE timestamp > '2026-05-04 13:30:00+00' LIMIT 5;`
- Confirm new alerts have populated `indicator_snapshot`
- Hover an alert in Strategy Detail history table — tooltip renders correctly

### Mon-3 — Validate M6 engine state capture
- After 1 hour of trading: `SELECT strategy_id, count(*) FROM bar_engine_states WHERE bar_start > now() - interval '1 hour' GROUP BY strategy_id ORDER BY 2 DESC;`
- Confirm row counts roughly match expected per-strategy bar production rates
- Spot-check JSONB content has expected indicator keys

### Mon-4 — TV stability test (deferred from Friday)
**Method (refined 2026-05-04):** Set a 5-minute repeating timer. Each interval, download the SPY 1Min bar export from TV as a CSV (the "Save Chart Data" / export feature on TV's chart toolbar). Naming convention: `tv_spy_1min_NN_HH-MM.csv` where NN is sequential (01, 02, 03, …) and HH-MM is the snapshot time (UTC or ET, just be consistent).
- Drop CSVs into `docs/reference_images/tv_stability_2026-05-04/` (or wherever Kevin chooses)
- Continue for at least 90 minutes — covers any 15-min late-print window plus tail
- Once collected, diff a fixed bar (e.g. 9:35 ET close) across all CSVs; any cell change between snapshots = TV revises silently. Identical = TV locks.

**Why CSV instead of screenshots:** more precise (digits not pixels), grep-able diff, easier to scale across many bars instead of just one.

**Outcome informs:** whether to flip `live_model` default to `ws_first_lock` (Option C in the rebroadcast handling design). If TV locks at first close, that's evidence that "decision-time WS values" are the right anchor. If TV revises, then the FINRA late-print correction is the industry norm and we should keep `latest` as default.

### Mon-5 — Sanity-check Lab tab replay
- Open Lab tab on a strategy with cache data through Monday morning
- Toggle Replay on Alert Lens, scrub through the morning
- Confirm indicators/heatmap evolve smoothly (no jumps, no missing data)

**2026-05-04 update:** smoke test surfaced that V1 was rendering candles only — no overlays, no markers, and the time window ignored `candleCount` so it loaded all ~5 days of cache (~70k 10Sec bars). Fix shipped (commit TBD): caller now slices to `candleCount`, builds entry/exit arrow markers from the same trade list the static Alert Lens uses, and forwards `rightOffset`. ChartReplayCard's chart `id` keys on overlay count too, so seriesSetup changes after initial render trigger chart re-creation. Re-test on next Railway redeploy.

---

## Open questions resolved during planning

1. **Q: Does the bug affect secondary-TF aggregation?** **A: Yes.** Same `accept_second_bar` method is shared by sub-minute primary AND secondary TF aggregation paths. One fix in `accept_second_bar` covers both.

2. **Q: Does `warmup()` reset state?** **A: No.** Need a new `recompute_from_history` method that resets state then replays.

3. **Q: What about the corrupted in-memory state currently in worker?** **A: Worker restart on deploy rebuilds from REST warmup (clean state).** No special cleanup needed.

4. **Q: Re-fire alerts on corrected bars?** **A: No.** Once saved, alerts are fact. Corrections affect future indicator state silently; the corrected bar itself doesn't re-evaluate triggers.

---

## Critical files reference

| Concern | File | Anchor |
|---|---|---|
| accept_bar duplicate handling | `src/ralph_engine.py` | line 348 |
| accept_second_bar duplicate handling | `src/ralph_engine.py` | line 237 |
| _append_to_history | `src/ralph_engine.py` | line 335 |
| _close_bar | `src/ralph_engine.py` | line 329 |
| on_polygon_bar caller | `src/ralph_engine.py` | line 1506 |
| on_second_bar caller | `src/ralph_engine.py` | line 1588 |
| IncrementalIndicatorEngine class | `src/ralph_engine.py` (or unified_engine.py) | search "class IncrementalIndicatorEngine" |
| _run_monitor_pipeline_for_completed_bar | `src/ralph_engine.py` | line ~1392 |
| prepare_data_with_indicators | `src/services.py` | line 117 |
| chart-data endpoint | `src/api/routers/strategies.py` | line 1201 |
| cache-bars endpoint (fetch pattern) | `src/api/routers/strategies.py` | line ~1666 |
| _serialize_chart_data | `src/api/services/backtest_service.py` | line 516 |
| Lab tab Phase 1 | `frontend/src/views/StrategyDetailPage.tsx` | search "Chart & Trades (Lab)" |
| labChartTabData useMemo | `frontend/src/views/StrategyDetailPage.tsx` | search "labChartTabData" |
| useStrategyCacheBars hook | `frontend/src/hooks/queries/useStrategies.ts` | search "useStrategyCacheBars" |
| Test pattern reference | `src/test_unified_parity.py` | full file |

---

## Verification (split: offline this weekend vs live Monday)

### Offline (verifiable today / weekend, markets closed)

After M1 lands:
1. `pytest src/test_rebroadcast_recompute.py` — passes
2. `pytest src/test_parity_regression.py` — 7/7 still pass
3. `pytest src/test_unified_parity.py` — still passes
4. Worker restarts cleanly post-deploy; Railway logs show normal startup (warmup against REST, periodic-tasks loop running, no errors). Worker idles happily waiting for Monday open.

After M2 lands:
5. Open Strategy Detail Lab tab on any strategy with cached data from Friday; right-side chart shows indicators AND heatmap populated (no longer empty/REST-derived).
6. Pick a Friday alert (we have several in the alerts table); verify right-side heatmap at the alert's `fill_ts` matches the recorded `alert.confluence_records` for that bar.
7. Side-by-side visual: candles AND indicators visibly differ between Algo Lens (REST) and Alert Lens (cache) on Friday's knife-edge bars.

After M3 lands:
8. Open a strategy edit page; backtest_model and live_model dropdowns appear, populated from /api/strategies/models.
9. Strategy detail page header shows two model badges.
10. Create a test strategy with non-default model values; reload the page; verify the values persist in config.
11. Existing strategies show default values (`rest_only` and `ws_with_corrections`).

### Live (deferred to Monday morning during RTH)

12. After ~30 min of Monday market activity: `_validate_live_bars_cache.py` should show `first_close ≠ close` on a meaningful fraction of 1Min bars (~10-20% based on Friday's correction rates). This is the "fix is actually working in production" check.
13. Spot-check 5-10 corrections: deltas should be $0.01-$0.10 typical. Outliers >$0.50 deserve inspection but aren't necessarily bugs.
14. Indicator state in worker memory should track Polygon corrections — compare worker logs at T+0 vs T+15min for the same bar's `BAR_CLOSE` indicator values; should see updates if a correction came in.

---

## Risk + rollback

**Highest risk: M1.** Engine change affects every alert. Mitigations:
- Backup branch (M1.1) ✅ created
- Replay test must pass before deploy (M1.6) ✅ 5/5 pass
- Markets closed → no trading impact during validation
- Worker restart on deploy = clean state, no in-memory corruption carryover ✅ verified

**Medium risk: M2.** New endpoint, no engine change, but routes through prepare_data_with_indicators which is shared. Mitigations:
- New variant via optional `df` parameter — existing callers unaffected
- New endpoint, separate from /chart-data
- Frontend toggle isolated to Lab tab

**Low risk: M3.** Pure UI + config write. No behavioral change.

**Low-medium risk: M4 (alert snapshot).** Worker-side change at alert dispatch path. Mitigations:
- Try/except wrap around snapshot capture — if anything goes wrong, alert still fires without the snapshot field (just logs warning)
- No schema migration → no DB blast radius
- Frontend display is purely additive — older alerts without snapshots don't break

**Low risk: M5 (Lab tab replay).** Pure frontend addition. New component, opt-in via toggle. Existing chart unchanged. Doesn't touch engine.

**Low-medium risk: M6 (engine state capture).** Worker writes to a new table. Mitigations:
- Env-flag gated (default off in code; flip true on Railway after deploy)
- Fire-and-forget thread pool — never blocks bar processing
- Try/except wrap on the write path — failures log but don't crash worker
- Storage cost ~13 MB/day; budget for ~5 GB/year before retention policy revisit

**Rollback path**: revert to backup branch, force-push to dev, redeploy. Worker restarts clean. Total recovery: ~5 min. M4/M6 also support disable-via-env-flag for instant kill.
