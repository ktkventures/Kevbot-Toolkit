# Parity Simulator — Implementation Plan (2026-04-24)

## Why this exists

Today's audit found 9 strategies producing forward algo trades through the unified-engine backtest path but firing zero live alerts. Most likely cause: **the unified engine (batch path) and ralph engine (incremental, live path) compute indicator state and trigger fires differently for some pack/trigger combinations.** State-machine triggers (e.g. `_detected` suffix on swing_123) are particularly suspicious because they depend on previous-bar state being initialized correctly — a class of bug that's invisible in batch and silently kills alerts in live.

Without a way to validate parity, every new confluence pack Kevin builds is a coin flip on whether it'll fire alerts in production. That paralyzes pack development.

The Parity Simulator scaffolding already exists in the UI (UserPacksPage Parity Simulator tab + PackBuilderPage Parity Simulator tab), but the body is a `{{ai_response}}` placeholder. This plan implements the engine behind it.

## Backup

Branch: `dev-backup-pre-parity-simulator-2026-04-24` at commit `be78ad3`. Safe revert point.

## V1 scope (this weekend)

Single pack at a time, single timeframe, default config, configurable symbol + days. Result: parity score + divergence list + chart visualization.

**Out of scope for V1**: cross-TF packs, custom pack params, mid-stream-restart simulation, strategy-detail mode (multiple packs composed), automatic parity gate on pack save. All deferred to V2.

## Architecture

### New module: `src/parity_simulator.py`

```python
def run_pack_parity_test(
    pack_id: str,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    user_id: str | None = None,
    feed: str = 'sip',
) -> dict:
    """Run a backtest-vs-live parity test for a single user pack.
    
    Returns a dict with backtest_fires, live_fires, matched, backtest_only,
    live_only, parity_score (0-1), and verdict (PASS/WARN/FAIL).
    """
```

### Pipeline (5 stages)

#### Stage 1 — Bar source
Load N days of bars via existing `data_loader.load_market_data(symbol, days=N, timeframe=tf, feed=feed)`. Already production-tested. Returns a pandas DataFrame of OHLCV.

#### Stage 2 — Backtest path
Run all bars at once through the unified engine:
```python
from unified_engine import run_unified_backtest
strategy_proxy = _build_strategy_proxy_for_pack(pack_id)
trades_df, enriched_df = run_unified_backtest(df, strategy_proxy, general_packs=[])
backtest_fires = _extract_trigger_fires(enriched_df, pack_id)
```

`_build_strategy_proxy_for_pack` constructs a minimal "strategy"-shaped dict with the pack as the entry trigger, no exit, no stop, no target — i.e. the simplest possible config that makes unified_engine willing to evaluate the pack's triggers.

`_extract_trigger_fires` reads the trigger boolean columns from `enriched_df` (the unified engine writes one column per trigger named like `trigger_{pack_id}_{trigger_name}`) and produces a list of `(bar_idx, timestamp, trigger_name, indicator_values)` tuples for every True row.

#### Stage 3 — Live replay
Spin up a fresh `ralph_engine.IncrementalIndicatorEngine` and `TriggerEvaluator` for the pack. Iterate bars chronologically, simulating WebSocket arrival:

```python
from ralph_engine import IncrementalIndicatorEngine, TriggerEvaluator
ind_engine = IncrementalIndicatorEngine(pack_template=pack_id, params=pack_params)
trig_evaluator = TriggerEvaluator(pack_id, pack_triggers)

live_fires = []
for idx, row in df.iterrows():
    # Mirror exactly what worker.py does on bar close
    ind_engine.update(row)  # or whatever the actual incremental method is
    interp_states = ind_engine.compute_interpreter_states()
    fires = trig_evaluator.evaluate(interp_states, prev_states=prev)
    for trig_name, value in fires:
        live_fires.append({'bar_idx': idx, 'timestamp': row['timestamp'], 'trigger': trig_name, 'value': value})
    prev = interp_states
```

**Critical**: do not warm up the incremental engine with the full DF. Feed it bar-by-bar from index 0. The point of the simulator is to detect drift caused by sequential vs batch processing — pre-warming defeats the purpose.

The replay must **mirror the worker's exact init order** (look at `worker.py:_run` lines 950+ for reference). If the worker calls `monitor.warmup(df)` with the first M bars and then processes the rest one-at-a-time, the simulator should do the same — that's how production behaves.

#### Stage 4 — Diff
Compare the two fire lists. For each `(bar_idx, trigger_name)` pair:
- Both lists have it → **matched**
- Only backtest → **backtest_only** (the broken case — pack fires in batch but not in live)
- Only live → **live_only** (rare — usually means backtest missed something)

For divergences, capture context:
- Bar timestamp + OHLC values
- Backtest indicator state at that bar (from `enriched_df`)
- Live indicator state at that bar (from `ind_engine` at that step)
- Diff between the two

#### Stage 5 — Verdict
- **PASS**: 100% match (or matched outside warmup zone)
- **WARN**: <5% divergence, all in early bars (likely warmup edge cases)
- **FAIL**: >5% divergence OR any divergences after the warmup zone

### Warmup handling

Indicator packs need N prior bars before their state is meaningful. Hardcoded per pack (e.g. swing_123 needs ~10 bars for pivot detection, ema_stack needs `max(periods)` bars for moving averages).

Two options:
- **Option A (simpler)**: Skip the first N bars from the diff entirely. Configurable per pack.
- **Option B (more rigorous)**: Compute warmup dynamically by checking when the indicator's `_initialized` flag flips True.

V1: Option A. Hardcode warmup per pack template (lookup table). V2: Option B for accuracy.

### Cross-TF gotcha

For V1, **only allow packs that operate on a single timeframe.** If the pack's confluence list references higher TFs (e.g. a 10Sec pack with 1Min confluence dependency), the test results would be misleading — live mode would have access to the higher-TF state via the SymbolHub buffer, but our simplified replay wouldn't.

**V1 detection**: at start of test, scan the pack's config. If any cross-TF deps detected, abort with a clear error message: `"Cross-TF parity testing not supported in V1 — this pack uses {N} secondary timeframes."`

V2 will support cross-TF by also feeding the higher-TF bars through their own indicator engines and merging via the same `_mtf_confluence` buffer pattern that `SymbolHub` uses.

### Determinism + caching

The same input bars + pack should always produce the same fires. If they don't on a re-run, the indicator code has a non-determinism bug (rare, but worth catching).

For caching: don't cache parity results yet. Each test pulls fresh Polygon bars and runs both paths. We can add caching in V2 once the engine's correct.

## API endpoint

```
POST /api/packs/{pack_id}/parity-test
Body: {
  "symbol": "SPY",
  "timeframe": "1Min",
  "days": 7
}

Response: {
  "pack_id": "swing_123_test_default",
  "symbol": "SPY",
  "timeframe": "1Min",
  "bars_loaded": 1620,
  "warmup_bars": 10,
  "backtest_fires": [{ "bar_idx": 11, "timestamp": "2026-04-23T...", "trigger": "swing_123_test_default_bullish_c2_detected", "value": true }, ...],
  "live_fires": [...],
  "matched": [...],
  "backtest_only": [...],   // <-- the smoking gun
  "live_only": [...],
  "parity_score": 0.62,
  "verdict": "FAIL",
  "summary": "20 fires backtest-only, 0 live-only, 33 matched. State-machine drift suspected."
}
```

File: `src/api/routers/packs.py` (add to existing user-packs router; if it doesn't have one, create alongside `webhook_groups.py`).

## Frontend wiring

File: `frontend/src/views/UserPacksPage.tsx` (lines ~1465-1500 — the Parity Simulator tab body).

Replace the `{{ai_response}}` placeholder with:

1. **Form**: symbol input, timeframe dropdown, days slider (default 7)
2. **"Run Parity Test" button** — calls the new endpoint
3. **Results panel** (renders after success):
   - **Parity score badge**: large number, color-coded (green ≥ 95%, yellow 80-95%, red < 80%)
   - **Verdict + summary**: one-line plain-English description
   - **Chart**: candles for the test window, with markers — green dots = matched fires, red Xs = backtest-only (the broken ones), yellow triangles = live-only
   - **Divergence table**: scroll list of every backtest_only fire with bar timestamp, indicator state at that bar in BOTH paths, and a side-by-side diff (e.g. "live: ema_short=140.2, batch: ema_short=140.5")

Mutation hook in `useUserPackMutations.ts` (or wherever appropriate):
```typescript
export function useRunPackParityTest() {
  return useMutation({
    mutationFn: ({ packId, symbol, timeframe, days }: ...) =>
      apiFetch(`/api/packs/${packId}/parity-test`, {
        method: 'POST',
        body: JSON.stringify({ symbol, timeframe, days }),
      }),
  });
}
```

## Implementation order (commits)

1. **Commit 1** — `parity_simulator.py` skeleton + bar loading + backtest path. No live replay yet. Returns `backtest_fires` only. Sanity check via Python script that we can extract trigger fires from unified_engine.
2. **Commit 2** — Live replay path. Returns both fire lists, no diff yet. Sanity check that live replay produces SOME fires for a known-working pack (e.g. utbot_v2, used by strategy 117).
3. **Commit 3** — Diff engine + verdict logic + warmup handling. Returns full result dict.
4. **Commit 4** — API endpoint + tests for the engine.
5. **Commit 5** — Frontend wiring on UserPacksPage Parity Simulator tab.
6. **Commit 6 (optional)** — Same wiring on PackBuilderPage Parity Simulator tab.

Each commit is independently revertable. Sanity checks at each step rather than building everything then debugging.

## Testing strategy

### Smoke test packs (run once each before declaring V1 done)

| Pack | Expected outcome | Reasoning |
|---|---|---|
| `utbot_v2` (strategy 117 uses it, fires 2900+ alerts/day) | Should PASS (≥95%) | Production-validated |
| `swing_123_test_default` (strategy 129 uses it, 0 alerts) | Should FAIL with backtest_only fires | The known broken case — proves the simulator is detecting real drift |
| `ema_price_position_v2` (strategy 124 uses it, 0 alerts) | Likely WARN or FAIL | Recent strategy, unclear if it's working — simulator should reveal |

If `utbot_v2` doesn't pass, the simulator has a bug — the engine is reporting drift that doesn't exist in production. Stop and debug.

If `swing_123_test_default` doesn't fail, the simulator isn't catching the real bug — also stop and debug.

### Edge cases to manually verify

1. **Empty test window** (no bars returned): graceful error, not a crash.
2. **Pack not found**: 404 with clear message.
3. **Pack with no triggers**: edge case — returns 100% parity (vacuous truth).
4. **Bars with NaN OHLC** (unusual but happens on illiquid tickers): both paths handle the same way.

## Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `IncrementalIndicatorEngine` init differs from worker's init order | Medium | Mirror `worker.py:_run` line-by-line; cross-check by running worker on the same data and comparing trigger fires |
| Backtest path's trigger column extraction has edge cases (renamed column, missing trigger) | Medium | Defensive parsing; fall back to "no fires for this trigger" with warning |
| Polygon rate limit if many tests run in quick succession | Low | data_loader already caches; add explicit rate guard if needed |
| The `_detected` triggers turn out to use a custom evaluator path I don't replicate correctly | Medium-high | This IS the bug class. If we can't replicate, we can't detect. Need to study `TriggerEvaluator` for state-machine handling carefully |
| Live replay state diverges from production due to subtle worker scaffolding | Medium | After V1, run the simulator against a strategy that's actively firing in production and check parity_score = 100% (sanity that the simulator agrees with reality) |

## Effort estimate

- **Commits 1-3 (engine)**: ~5-6 hours
- **Commit 4 (API)**: ~1-2 hours
- **Commit 5 (UI)**: ~2-3 hours
- **Commit 6 (PackBuilder UI)**: ~30 min — copy from UserPacks
- **Smoke tests + debugging**: ~2-3 hours
- **Total V1**: ~12-15 hours

Realistic for a focused weekend session. Will check in with progress notes after each commit.

## What V2 looks like (deferred from V1)

- **Cross-TF support**: feed multiple timeframe streams, merge via `_mtf_confluence`-pattern
- **Mid-stream restart simulation**: split the bar window in two, replay first half, simulate worker restart (clear state, re-warmup), replay second half. Compare to ungap version. Catches state-machine init bugs that only surface after worker bounces.
- **Strategy Detail mode**: re-uses the engine but feeds a strategy's full config (multiple packs composed). Adds PositionStateMachine, stop logic, target logic, time exits to the live replay.
- **Auto parity gate on pack save**: when a pack is saved/edited, queue a background parity test against a known good ticker. Result shown next time the pack is opened. Prevents silently broken packs.
- **Custom params**: let users run parity tests with non-default pack params (e.g. swing lookback=10 instead of 5)

## What I'll do next session (after creating this plan)

Following the order above, starting with commit 1. Will pause + check in after each major step (especially after commit 3 — that's the "engine works" milestone).

If anything in this plan looks off or you'd rather scope V1 differently, course-correct here:
- ✅ → proceed with the plan as-is
- ⚠️ → drop me a note in the chat; I'll adjust the plan before coding
