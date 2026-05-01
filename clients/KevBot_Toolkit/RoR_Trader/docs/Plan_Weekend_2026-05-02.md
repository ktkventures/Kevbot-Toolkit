# Plan — Weekend of 2026-05-02 (Sat/Sun)

**Created:** 2026-05-01 EOD (Fri)
**Why weekend:** markets closed → can restart Worker freely without
affecting live trading; fix is invasive enough that controlled
validation is safer than mid-session.

## Context — what we discovered Friday

While building the Strategy Detail Lab tab, two important findings
surfaced that change priority:

### Finding 1 — The alert system doesn't handle Polygon's WS rebroadcasts correctly

Polygon's WebSocket rebroadcasts corrected versions of the same 1Min
bar within 15 min of close as late prints arrive (per their docs).
A correctly-implemented consumer should **overwrite** the prior bar.

What we actually do, traced in `ralph_engine.BarBuilder.accept_bar`
(line 348):

```python
def accept_bar(self, bar_dict: dict) -> dict:
    bar_ts = pd.Timestamp(bar_dict['timestamp'])
    bar_start = self._align_to_period(bar_ts.to_pydatetime())
    # Gap-fill if there are missing bars
    ...
    # Append the actual bar
    self._append_to_history(bar_dict)
    self._bar_count += 1
    self._partial = None
    return bar_dict
```

There's NO duplicate-timestamp check. When Polygon rebroadcasts a
correction:
1. `on_polygon_bar` fires AGAIN with the corrected bar (same `bar_start`)
2. `accept_bar` calls `_append_to_history` which CONCATs (doesn't replace)
3. `_bar_count += 1` (count drifts off by however many corrections came in)
4. `_run_monitor_pipeline_for_completed_bar` runs AGAIN, incrementally
   updating EMAs/MACD as if this were a brand-new bar

**Net effect**: each rebroadcast is processed as a duplicate bar.
Indicators get re-applied against the same time period, drifting
the engine's state away from what a clean recomputation would produce.

This is structural — not a one-off bug. It has been there since the
Polygon migration (Phase 31). Likely contributes meaningfully to
algo-vs-live divergence Kevin's been observing.

### Finding 2 — Phase 2 of the Lab tab needs the bug fixed first

The Lab tab's right side ("Alert Lens") shows WS cache bars but
indicators still computed from REST. To do "engine-state replay over
cache" properly, we need the cache to actually represent what the
engine SHOULD have seen — not what the buggy duplicate-processing
produced. Fix bug first, build replay second.

## Saturday work — root-cause fix

### Task 1 — `accept_bar` duplicate-timestamp handling

The fix is technically simple but design choice matters. Three options,
in order of correctness:

**Option A: Snapshot/restore pattern (correct, complex)**
- Before applying any bar, snapshot the IncrementalIndicatorEngine state
- On duplicate timestamp arrival, restore snapshot, apply corrected bar
- Pros: indicators truly reflect "this is what we'd see after corrections settle"
- Cons: requires snapshot capability on every indicator. Worth doing if we want clean state.

**Option B: Recompute from history on duplicate (correct, slower)**
- On duplicate timestamp, replace the bar in `history`, then re-run
  the incremental pipeline from N bars back (N = max EMA period)
- Pros: simpler than snapshot/restore; correct enough
- Cons: O(N) work per correction; possible for sub-minute strategies
  to lag if many corrections arrive in close succession

**Option C: Ignore rebroadcasts (simple, throws away info)**
- On duplicate timestamp, just SKIP the second call
- Pros: trivial fix, no design surface
- Cons: we keep the FIRST version of the bar even if Polygon corrects it.
  The cache's `close` column would never differ from `first_close`
  for 1Min bars (defeats the purpose of the first/close split).

**Recommendation: Option B for v1.** Get correctness, defer optimization.
If perf becomes an issue, upgrade to Option A.

**Implementation sketch:**
```python
def accept_bar(self, bar_dict: dict) -> dict:
    bar_ts = pd.Timestamp(bar_dict['timestamp'])
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.tz_localize('UTC')
    bar_start = self._align_to_period(bar_ts.to_pydatetime())

    # Detect duplicate (Polygon rebroadcast within 15 min)
    if len(self.history) > 0:
        last_ts = self.history.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')
        if last_ts == pd.Timestamp(bar_start, tz='UTC'):
            # Replace, don't append
            self.history.iloc[-1] = pd.Series({
                'open': float(bar_dict['open']),
                'high': float(bar_dict['high']),
                'low': float(bar_dict['low']),
                'close': float(bar_dict['close']),
                'volume': float(bar_dict.get('volume', 0)),
            })
            return bar_dict  # caller decides whether to re-run monitor pipeline
        # Also handle the case where bar_start is between last_period
        # and "1 period ago" — could be a Polygon rebroadcast for a
        # not-quite-most-recent bar. Need to think about this case.

    # ... existing gap-fill + append logic
```

Plus a corresponding change in the on_polygon_bar caller to handle
"this was a duplicate, recompute indicators from history rather than
incrementally apply this bar."

### Task 2 — IncrementalIndicatorEngine recompute path

For Option B, we need a helper:
```python
def recompute_from_history(self, df: pd.DataFrame) -> None:
    """Reset state and replay all bars in df. Used when a bar in the
    middle of history was corrected."""
```

May be sufficient to call existing `warmup(df)` again with the latest
N bars where N = longest indicator lookback. Need to verify this
correctly resets state.

### Task 3 — Validation

After fix, run on Saturday with worker NOT subscribed to live data:
1. Use existing `_validate_live_bars_cache.py` against pre-fix data
   to establish baseline drift
2. Build a small replay test: feed a known sequence of (bar, then
   bar-corrected) into a test engine, verify final indicator state
   matches a clean recomputation of the corrected bar
3. Restart worker Monday morning (markets reopen) with fix in place
4. Compare indicator values pre/post fix on the live engine after
   first hour of trading

## Sunday work — Phase 2 of Lab tab

After the bug fix lands, build the proper "Alert Lens" indicator path:

### Task 4 — Compute indicators from cache bars

New service function or extension to `prepare_data_with_indicators`:
- Input: `data_source: 'rest' | 'cache_latest' | 'cache_first'`
- When cache: pull bars from `live_bars` instead of REST; same indicator
  pipeline
- Need to also handle secondary TFs (cross-TF confluence) — pull cache
  for those too

### Task 5 — New chart-data-from-cache endpoint

`/api/strategies/{id}/chart-data-cache?value_type=...`
- Returns same shape as `/chart-data` (chart_data + overlay_indicators
  + oscillator_indicators + heatmap_conditions)
- Computed entirely from cache bars

### Task 6 — Wire into Alert Lens

Replace the current Lab tab right-side render so its indicators and
heatmap come from chart-data-cache instead of chart-data. Now the
right-side heatmap reflects what the engine actually saw at decision
time (with the fixed bug logic).

### Task 7 — Visual validation

Pick 3-5 strategies, screenshot the Lab tab. Verify:
- Right side's heatmap state at known alert-fire moments matches
  the recorded alert.confluence_records (which captures what live
  saw at the moment)
- Right side's EMA/MACD lines pass through reasonable values

## Open questions to think about over the weekend

1. **What do we do about the corrupted live state currently in the
   worker's BarBuilder.history?** Each rebroadcast added a duplicate row.
   When we deploy the fix, the in-memory history will still have these
   duplicates from this morning's session. Options:
   - Restart worker (history rebuilds from REST warmup, clean)
   - Just leave it — duplicates only affect indicators while still in
     window; they age out as MAX_HISTORY clips the buffer
   - Force `_warmup_all` to re-run on deploy
2. **Should we try to backfill the cache with what the live engine
   ACTUALLY saw, including corrections?** Probably no — the cache
   already has `first_close` (decision time) which is the most
   useful "what did the engine see" snapshot.
3. **Does the bug also affect the secondary-TF aggregation path?**
   Need to check — when on_polygon_bar fans out to secondary TF
   builders via `accept_second_bar(close_on_boundary=True)`, does that
   handle duplicates correctly?
4. **What about `on_second_bar` for sub-minute primaries?** Same
   `close_on_boundary=True` path. Need to verify duplicate handling.

## What we are NOT doing this weekend

- Read path for backtest reading from cache (M8.7d) — defer to next week
- More TF coverage via research strategies — defer
- Models architecture (backtest model / alert model as variables) —
  defer until empirical data justifies it
- Ultraview / parity sweep — defer

## Critical files reference

| Concern | File | Anchor |
|---|---|---|
| accept_bar duplicate handling | `src/ralph_engine.py` | line 348 |
| _append_to_history | `src/ralph_engine.py` | line 335 |
| on_polygon_bar | `src/ralph_engine.py` | line 1506 |
| IncrementalIndicatorEngine | `src/ralph_engine.py` | search "class IncrementalIndicatorEngine" |
| Lab tab Phase 1 (built Friday) | `frontend/src/views/StrategyDetailPage.tsx` | search "Chart & Trades (Lab)" |
| cache-bars endpoint | `src/api/routers/strategies.py` | search "/cache-bars" |
| live_bars writer | `src/live_bars_writer.py` | full file |

## Backup branch

Before Saturday work: create `dev-backup-pre-rebroadcast-fix-2026-05-02`
from `dev` HEAD so we have a clean rollback target if the fix causes
unexpected regressions.

---

## Summary in one paragraph (for handoff)

We discovered Friday EOD that the live alert engine processes Polygon's
WS rebroadcasts as DUPLICATE BARS rather than as corrections. Each
rebroadcast incrementally re-applies to the EMA/MACD state, drifting
the engine's view of the world. Saturday: fix `accept_bar` to detect
and handle duplicate timestamps (Option B = recompute-from-history on
duplicate). Sunday: build Phase 2 of the Lab tab so the right side's
indicators/heatmap come from cache bars (now correctly representing
what the engine sees post-fix). Markets closed both days = safe window.
