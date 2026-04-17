# Alert Recovery Plan — 2026-04-17

## Context

Over the last ~6 days of M8.5 live-chart work on `dev`, three separate issues
accumulated that together look like "the alerting / charting system fell apart."
In fact each issue lives on a distinct commit and can be fixed independently
without reverting M8.5 or touching the core engine.

This doc captures the diagnosis, the architecture reality, the parity picture,
and the 4-phase plan. Reload it if context is lost.

**Branches:**
- Current working branch: `dev` (HEAD `30a52ca` — WIP plot-snap commit)
- Safety backup: `dev-backup-dirty-live` (same HEAD as `dev` when created)
- Pre-M8.5 reference: `dev-backup-pre-m8.5` (HEAD `5046c97`)

---

## Symptoms observed

1. **~22 second alert latency.** Example: plot marker anchored to bar close at
   `11:09:10`, but alert row lands in DB at `11:09:32`. Pre-M8.5 this was
   sub-second.
2. **Algo history out of sync with alerts.** Algo history (`stored_trades`
   column) doesn't populate until sometimes 10-30s after the alert fires, and
   occasionally never populates for a bar close. Before, algo history and the
   alert appeared together.
3. **Markers plotted where no candle exists.** On 10s charts, exit `+` / `x`
   markers sometimes render on a timestamp with no bar beneath them.
4. **General feeling of "slippage" and wonkiness** that wasn't there before.

---

## Architecture reality — Ralph is Pattern B, not a separate engine

Critical finding that reframes the problem:

```python
# ralph_engine.py, top of file (both dev and dev-backup-pre-m8.5)
from unified_engine import (
    IndicatorState, IncrementalIndicatorEngine,
    TriggerEvaluator, PositionState, PositionStateMachine,
    ...
)
```

Ralph's `StrategyMonitor` is a **streaming orchestration wrapper** around
`unified_engine`'s actual logic classes. The Strategy Builder backtest
(`run_unified_backtest`) uses the **same four classes**. This is not a
parity-by-test relationship. It's literal code-sharing.

**What's shared (core logic):**
- `IncrementalIndicatorEngine` — O(1) incremental indicator updates
- `TriggerEvaluator` — trigger condition evaluation
- `PositionState` / `PositionStateMachine` — entry/exit state, trade records

**What's Ralph-only (data ingestion & multi-strategy orchestration):**
- `BarBuilder` / `PartialBar` / `accept_second_bar` — per-second → TF aggregation
- `StrategyMonitor` — per-strategy wrapper (but calls shared classes inside)
- `_ShadowIndicatorEngine` — cross-TF confluence (reconciled in Phase 30L)
- `SymbolHub` — multi-strategy-per-symbol routing
- `RalphEngine` — WebSocket subscription + asyncio loop

Implication: **"two engines" divergence is not structurally possible for the
core trade logic.** The only places parity can fray are the three listed below
(narrow, tested, mitigated).

---

## Root cause analysis — three independent issues on three independent commits

### Issue 1: Algo history detached from alert write (b83629b, 2026-04-16)

**Your own commit, two days ago.** Before `b83629b`, `DBAlertDispatcher.dispatch()`
in `worker.py` did two writes per exit alert in one function call:

1. `save_alert_admin(alert)` → `alerts` table
2. Append `trade_record` (carried on the signal dict by Ralph's
   `PositionStateMachine.get_trade_record()`) to the strategy's
   `stored_trades` column

Both writes used the same `trade_record` that the position state machine
produced on exit. One dispatch, two tables, atomic-ish (same function, same
thread, same signal). Algo history and the alert arrived **together**.

`b83629b` removed the second write and relocated it behind a new path:

- Bar close fires an async `on_bar_close_hook`
- Hook schedules `forward_test_service.recompute_and_persist_stored_trades()`
  on a `ThreadPoolExecutor`
- Recompute runs `run_unified_backtest` over full Polygon history, overwrites
  `stored_trades`
- A `_PerStrategyThrottle` may **drop** this recompute if another is in flight

Net effect: `stored_trades` now trails the alert by a full recompute (seconds to
tens of seconds), and sometimes gets dropped entirely.

**Intent of `b83629b` was sound but mismatched to the actual risk.** The idea
was "make live-vs-backtest divergence observable." But because Ralph IS Pattern
B, the divergence surface is narrow (see parity analysis below) — not large
enough to justify a second writer racing the first.

### Issue 2: Event-loop saturation → 22s alert latency (M8.5 B+ live chart work)

Two commits together produced this:

- **`feaef8f`** — "subscribe A.{ticker} unconditionally." Pre-M8.5, only
  strategies with L-type intrabar triggers subscribed to Polygon's per-second
  `A.{ticker}` channel (conditional on `has_ltype`). Post-M8.5, every stock
  symbol subscribes unconditionally.
- **`5f8a84f`** + **`102744d`** — "Ralph publishes tentative indicators +
  states on forming bars" / "intra-bar live updates for heatmap / oscillator /
  overlay panes." For every per-second bar, for every TF, `on_second_bar` now:
  - Aggregates into `accept_second_bar` (cheap)
  - Runs `_gather_tentative_state` — computes tentative EMAs / MACD / interpreter
    states (not cheap)
  - Schedules `_publish_completed_bar` → `httpx.AsyncClient.post()` to Supabase
    Realtime (task on the loop)
  - On sub-minute TF boundaries: runs full monitor pipeline synchronously →
    fires alert → **blocking synchronous `save_alert_admin` HTTP POST** on the
    event loop thread

The WebSocket consumer in `_stream_data_polygon` is async, but it calls
`hub.on_second_bar(...)` **synchronously** inside the coroutine
(`ralph_engine.py:2055-2059`). When `save_alert_admin` blocks 200-500ms on a
Supabase round-trip, per-second bars queue in the WS buffer (`max_queue=1024`).
Add `_gather_tentative_state` running for every tick × N symbols × K TFs, plus
httpx broadcast tasks queueing on the same loop, and event-loop lag compounds.

A bar that logically closes at `11:09:10` doesn't get its `save_alert` call
completed until `11:09:32`. **That's the 22s.**

### Issue 3: Marker-without-candle artifact (dfbc7b9 + follow-ons)

`/api/strategies/{id}/chart-data` returns completed historical bars. Live
sub-minute forming bars arrive via a separate Supabase Realtime channel
(`useLiveBar` hook). These merge client-side.

When Realtime drops a frame (httpx 500ms timeout, broadcast quota, transient
network), the bar never lands on the chart — but the alert's `bar_time`
still snaps the marker to that timestamp. Marker renders on an empty time slot.

Cosmetic, but symptom of a split data path for what should be one chart.

---

## Parity analysis — does Ralph produce the same trades as backtest?

**Short answer: yes, with three narrow, tested, mitigated caveats.**

### Why parity is strong by construction

Core logic (indicators, triggers, position state) = same Python classes,
imported from `unified_engine`. When Ralph advances a bar:

```
BarBuilder closes bar
  → StrategyMonitor.on_bar_close(completed, bar_count, mtf_confluence)
    → IncrementalIndicatorEngine.update(bar)           # shared with backtest
    → TriggerEvaluator.evaluate(row, ...)               # shared with backtest
    → PositionStateMachine.step(row, triggers, ...)     # shared with backtest
    → returns signals + trade_record
```

When backtest runs:

```
run_unified_backtest(df, strategy)
  → UnifiedStrategy iterates bars
    → IncrementalIndicatorEngine.update(bar)           # same class
    → TriggerEvaluator.evaluate(row, ...)               # same class
    → PositionStateMachine.step(row, triggers, ...)     # same class
    → accumulates trades_df
```

Given identical bar data, identical warmup state, identical strategy config,
the two paths produce identical trades. This is not a parity-test guarantee —
it's a shared-code guarantee.

### The three narrow divergence points

1. **Bar aggregation.** Ralph's `BarBuilder.accept_second_bar` aggregates
   per-second Polygon bars into primary-TF bars. Backtest uses
   `data_loader.resample_to_timeframe` on pre-fetched 1Min or 1Sec bars. These
   must produce identical OHLCV per bar.
   - **Mitigation:** `test_ralph_subminute_parity.py` locks this contract.
   - **Residual risk:** trade conditions filtered differently between WS and
     REST (Polygon WS applies EXCLUDED_TRADE_CONDITIONS; REST applies its own).
     Fence-sitter ticks can nudge an OHLC by a cent.

2. **Cross-TF confluence.** `_ShadowIndicatorEngine` (Ralph) computes
   secondary-TF indicators from streaming bars. Backtest resamples historical
   bars for secondary TFs and runs indicators on those.
   - **Mitigation:** Phase 30L reconciled these paths. The 2026-03-16 cross-TF
     look-ahead fix made **backtest** match **Ralph's** live behavior (Ralph was
     the reference).
   - **Residual risk:** edge cases where secondary-TF boundary alignment
     differs between live and backtest.

3. **Warmup + data source.** Ralph cold-starts with whatever history is loaded
   into `IncrementalIndicatorEngine` state. Backtest sees full N-day history.
   Polygon WS bars may differ from REST bars by milliseconds or trade-condition
   filtering.
   - **Mitigation:** Ralph loads sufficient warmup at start (per
     `feedback_indicator_warmup.md`). For bars after warmup, no difference.
   - **Residual risk:** narrow, bounded to the first few bars after cold start.

### Conclusion

**Ralph and backtest share the core trade-generation code.** The three
divergence points are narrow, tested, and bounded. `b83629b`'s premise
("we need to make divergence observable") treated these narrow risks as
systemic. The right response is **explicit reconciliation** (Refresh button
triggers `run_unified_backtest` on fresh data and lets user spot drift), **not
a second async writer** that inverts the live write ordering.

---

## The plan — 4 phases, ~1 day total

Each phase is independently testable and revertable. Order matters (phase 1 is
the safety-net fix; phases 2-4 are latency fixes that compound).

### Phase 1 — Revert `b83629b`'s write inversion (restore atomic write)

**Target:** `clients/KevBot_Toolkit/RoR_Trader/src/worker.py`

**Change:** Bring back the atomic append-on-exit in
`DBAlertDispatcher.dispatch()`. On `exit_signal`:

```python
# pseudo:
save_alert_admin(alert)                            # alerts table
append_trade_record_to_stored_trades(              # stored_trades column
    strategy_id, trade_record, user_id)
```

Both writes use the same `trade_record` carried by the signal. Happens in the
same function, same thread.

**Also:** disable the `on_bar_close_hook` path in
`SymbolHub._run_monitor_pipeline_for_completed_bar` so the throttled recompute
doesn't fire automatically. Keep `forward_test_service.recompute_and_persist_stored_trades()`
as the implementation behind `/api/strategies/{id}/refresh` — user-initiated
reconciliation remains available as a button.

**Validation:**
- Start engine. Fire a test alert on a live strategy (or mock).
- Confirm `alerts` row and `stored_trades` updated in the same ~second.
- Confirm `/api/strategies/{id}/refresh` still recomputes correctly when
  clicked.

**Rollback:** `git revert` the Phase 1 commit; the hook path comes back.

**Reference:**
- Pre-`b83629b` implementation:
  `git show ffc6c1d:clients/KevBot_Toolkit/RoR_Trader/src/worker.py` — see the
  `# M8.5 B+: persist algo trade record into stored_trades on exit` block
  (~line 151-224).

### Phase 2 — Conditional `A.{ticker}` subscription

**Target:** `clients/KevBot_Toolkit/RoR_Trader/src/ralph_engine.py:1544-1562`
(commit `feaef8f` area).

**Change:** Restore the `has_ltype` gate that was removed in M8.5. Only subscribe
to per-second `A.{ticker}` if the symbol has at least one strategy whose
required triggers include an intrabar L-type trigger **or** has a sub-minute
primary TF.

Reference code from `dev-backup-pre-m8.5:ralph_engine.py:1555-1569`:

```python
stock_channels = [f"AM.{s}" for s in stock_symbols]
crypto_channels = [f"XA.X:{s.replace('/', '')}" for s in crypto_symbols]

for sym in stock_symbols:
    hub = self.hubs.get(sym)
    if hub:
        has_ltype = any(
            any(t in _IB_L_TYPE_TRIGGERS
                for t in getattr(m, '_intrabar_triggers', set()))
            for m in hub.monitors.values()
        )
        has_subminute = any(
            m.tf_seconds < 60 for m in hub.monitors.values()
        )
        if has_ltype or has_subminute:
            stock_channels.append(f"A.{sym}")
```

(The `has_subminute` clause is new — pre-M8.5 had no sub-minute TFs, so we
need to include them now.)

**Validation:**
- Strategy with 1Min TF + no L-type triggers: confirm only `AM.{ticker}`
  subscribed (log line `Polygon subscribed to N channels`).
- Strategy with 10Sec TF: confirm both `AM.{ticker}` and `A.{ticker}`
  subscribed.
- Existing L-type strategies: unchanged behavior.

**Rollback:** revert the gate, back to unconditional subscription.

### Phase 3 — Move `save_alert_admin` off the event loop

**Target:** `clients/KevBot_Toolkit/RoR_Trader/src/worker.py`,
`DBAlertDispatcher.dispatch()`.

**Change:** Wrap the DB write + webhook delivery in a
`loop.run_in_executor(executor, ...)` call using the existing worker thread
pool. The dispatch function still fires "on bar close" logically, but the
Supabase HTTP POST runs on a worker thread, freeing the event loop.

Error handling: log failures with strategy ID + bar time. On failure, the
alert is lost (same as today — no retry queue). This is acceptable for now;
revisit with a retry queue in a future phase if needed.

Pseudo-diff:

```python
# before (blocks event loop):
alert = save_alert_admin(alert, self.user_id)

# after (fire-and-forget on executor):
loop = asyncio.get_running_loop()
loop.run_in_executor(
    self._executor,
    lambda: save_alert_admin(alert, self.user_id))
```

**Caveat:** Phase 1's atomic `stored_trades` append must also move to the
executor so it stays atomic with the alert save. Both writes happen on the
same executor task.

**Validation:**
- Time between Ralph's `BAR_CLOSE` log line and `ALERT SAVED` log line should
  drop to <200ms (vs ~1-3s currently per alert, and far worse under load).
- Per-second bar processing should no longer stall on alert dispatch — grep
  logs for any long gaps in `on_second_bar` cadence.

**Rollback:** unwrap the executor call. Synchronous save restored.

### Phase 4 — Throttle / gate forming-bar broadcasts

**Target:** `clients/KevBot_Toolkit/RoR_Trader/src/ralph_engine.py:_publish_completed_bar`
+ `_gather_tentative_state`.

**Change:** Two options, pick one or both:

- **(A)** Coarsen the forming-bar throttle from 250ms to 1000ms. Most users
  won't notice; event-loop tasks per second drop 4×.
- **(B)** Gate forming-bar publishing on "active chart presence" — only publish
  if a user has recently fetched `/api/strategies/{id}`. Presence tracked via
  Supabase Realtime subscription count or a short-TTL cache on the API side.
  More work, bigger win for multi-strategy users.

Start with (A) as the cheap win. Consider (B) in a follow-up if per-second
load is still a problem.

**Validation:**
- WebSocket consumer lag metric (time from Polygon bar receive to processing
  completion) should drop.
- Chart forming-bar visual on Strategy Detail should still animate smoothly
  (1-sec granularity is fine for visual).

**Rollback:** restore 250ms throttle.

---

## Validation — how we'll know the plan worked

After all 4 phases:

1. **Alert latency** (end-to-end): pick a live strategy on a 1Min TF. Fire a
   trigger. Time between bar close (per Ralph `BAR_CLOSE` log) and alert row
   `created_at` in Supabase should be **<1 second**.
2. **Algo history sync:** new exit alert lands. `stored_trades` column updated
   within the same second. Chart & Trades tab shows the trade immediately,
   not after a refresh / reload.
3. **Sub-minute still works:** 10Sec strategy fires alerts at sub-second
   latency after bar close.
4. **Live chart still works:** forming bar animates on Strategy Detail,
   1-second granularity is acceptable.
5. **Backtest parity:** user clicks Refresh. `run_unified_backtest` re-runs.
   Trades match what Ralph produced live (within the 3 narrow divergence
   points). If they don't match, the mismatch is a real bug worth chasing —
   not a systemic architecture problem.

---

## Rollback — how to bail

Each phase is a separate commit. Phases 1-4 can be independently reverted via
`git revert <sha>`.

If all 4 phases together break something fundamental, the full rollback is:

```bash
git reset --hard dev-backup-dirty-live
```

(This is the backup branch created before any changes.)

Worst-case nuclear option: reset `dev` to `dev-backup-pre-m8.5` and cherry-pick
the 7 non-live commits forward:

```
7ac9189 Indicator param resolution: strict contract drives engine from user groups
fea6e4a Confluence Analysis chart: filter non-plottable columns + memoize panes
86b7c9d SyncedChartPane: honor user display timezone on axis + crosshair
b83629b Algo history architecture: stored_trades = backtest truth, only
a0e9586 Refresh endpoint: user is a dict, not an object
0344617 L-type fills: clamp to realistic bar prices
0e768be Ghost line for v2 L-type triggers
```

(We'd omit `b83629b` in the cherry-pick since we've just established it's the
root cause of the algo-history-detachment symptom.)

Loses the live chart, keeps everything else. Only go here if phases 1-4 reveal
a systemic problem we haven't anticipated.

---

## Divergence monitoring done right (future work, not in scope now)

Ideas to preserve `b83629b`'s good intent without its cost:

1. **Refresh button = explicit reconciliation.** Keep current implementation.
   User clicks, sees trade list diff if any.
2. **Parity audit log.** On every live exit, optionally queue a background
   task (low priority, high debounce) that runs
   `run_unified_backtest` on the last N bars and emits a **divergence report**
   to a new `engine_divergence_log` table. Never overwrites `stored_trades`.
   User can view drift history in a future UI panel.
3. **Extend parity tests.** If any real divergence is caught, pin it as a new
   parity test case in `test_unified_parity.py` or `test_ralph_subminute_parity.py`.

These are orthogonal to the recovery plan and can be tackled after the core
symptoms are fixed.

---

## Key references

- **Backup branches:**
  - `dev-backup-dirty-live` — safety net created 2026-04-17 before any recovery
    work
  - `dev-backup-pre-m8.5` (HEAD `5046c97`) — clean pre-M8.5 state
  - `dev-backup-pre-scenario-expansion` — reference for scenario expansion era

- **Key commits to revert / reference:**
  - `b83629b` — stored_trades = backtest truth, only (**revert in Phase 1**)
  - `ffc6c1d` — persist algo trades to stored_trades on exit alert (reference
    implementation for Phase 1)
  - `feaef8f` — M8.5 B+ step 3: subscribe A.{ticker} unconditionally (**gate
    in Phase 2**)
  - `5f8a84f` — Ralph publishes tentative indicators + states on forming bars
    (throttle in Phase 4)
  - `102744d` — M8.5: intra-bar live updates for heatmap / oscillator /
    overlay panes (throttle in Phase 4)

- **Session context:**
  - `docs/Observations_2026-04-16.md` — prior session EOD notes
  - `project_algo_history_session_2026-04-16.md` (memory) — memory of prior
    session context

- **Parity tests:**
  - `src/test_unified_parity.py` — 27 tests locking unified_engine correctness
  - `src/test_ralph_subminute_parity.py` — locks Ralph sub-minute aggregation
    matches `resample_to_timeframe`

---

## Status tracking

- [ ] Phase 1 — Revert `b83629b`'s write inversion (atomic write restored)
- [ ] Phase 2 — Conditional `A.{ticker}` subscription
- [ ] Phase 3 — Move `save_alert_admin` off the event loop
- [ ] Phase 4 — Throttle / gate forming-bar broadcasts
- [ ] Validation: all five validation criteria met
