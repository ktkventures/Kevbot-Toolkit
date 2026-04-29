# Mass Builder & Update Data — Performance Notes
2026-04-29

Captured to consolidate what we know about the current state of mass-search performance and the perceived slowdown vs. earlier in the project. Roadmap follow-ups at the bottom.

---

## 1. How the rapid backtest was supposed to work (and your mental model)

Your description was correct. Two clarifications on wording:

> "It will go through and construct all those trades with no confluences, and then it will layer on the confluences"

What actually gets cached is **per-bar STATE**, not pre-built trades. For every bar in the dataset, the cache stores:
- The bar OHLCV
- All indicator values at that bar (EMA, MACD, ATR, RVOL, etc.)
- Trigger booleans (which triggers fired on this bar — c_triggers)
- L-type fill levels (intra-bar level crosses)
- Confluence records (state values for every interpreter on this bar)

This is `precompute_bar_cache()` in `unified_engine.py:3182`. It runs the indicator + trigger + interpreter pipeline **once per (symbol, timeframe)**.

Then for each combo (entry × exit × confluence set), `run_trades_from_cache()` (line 3267) walks the cached bars and only runs the **position state machine**:
- "Was my entry trigger boolean true on this bar?"
- "Did this bar's confluence records satisfy my required set?"
- "Run entry/exit/stop/target logic"

No indicator math. No trigger math. Just position-state-machine + lookup. The docstring claims **10–50× speedup** for analyzers that test many configs.

So your mental model is right — the heavy lifting (indicators + triggers + interpreters) runs once over the bars, and combos are layered on as cheap filters that gate which bars actually trigger trades.

---

## 2. What's actually happening now

**Mass Builder no longer uses the rapid path** as of commit `467985a` (2026-03-16):
> "Fix Mass Builder zero results: use full backtest instead of broken bar_cache"

The bug: `precompute_bar_cache()` builds its `TriggerEvaluator` from `strategy.required_triggers` — but the original "mega_config" only listed ONE entry trigger (the first selected one). When subsequent combos tested a different entry, the cache lookup raised `ValueError: Trigger 'X' not in cache`, and the combo was skipped → 0 trades for those combos.

The fix at the time was to abandon the rapid path entirely and call `run_unified_backtest()` (the full backtest) for every base config. **This is still the path today.**

**Today's mass builder loop:**

```
for symbol, tf in (sym, tf) groups:
    df = prepare_data_with_indicators(...)        # data load + indicators (once per group)
    for entry, exit_combo, dir, ... in base_configs:
        run_unified_backtest(df, config, ...)     # FULL bar-by-bar engine pass per combo
        # then confluence search layered on the trades from this run
```

So each base config does the entire indicator + trigger + position-state-machine pipeline from scratch. Indicators get **recomputed** by the engine's `IncrementalIndicatorEngine.update_bar()` per combo even though the input df already has the indicator columns — incremental engines maintain their own internal state and don't read pre-computed columns.

**The "rapid" infrastructure still exists** in `unified_engine.py` (functions `precompute_bar_cache` + `run_trades_from_cache`). It's currently used by the parity simulator, not by mass builder.

---

## 3. The estimator is wrong, and it has been since `467985a`

`mass_builder.py:53-93` constants are from Phase 33A skeleton (`b790db1`), never updated:

```python
# Rough time estimates:
# - ~200ms per base config (bar_cache replay)        ← rapid-path assumption
# - ~2ms per confluence filter
est_data_load = data_groups * 3.0       # ~3s per data load
est_base = base_configs * 0.2           # ~200ms per backtest
est_conf = total * 0.002                # ~2ms per confluence filter
```

The `200ms per base config` was the old rapid-path number. Reality from your TTP V2 run this morning:

| Component | Estimator | Actual |
|---|---|---|
| Trigger backtest | 200ms each | **328,219ms each** (1640× off) |
| Confluence filter | 2ms each | 1.265ms each ✓ |
| Data load | 3s each | 8.9 min for AGIO/1Min × 365d |

**Why the estimator felt accurate before**: the dominant term in the prediction is `total × 0.002` (confluence search). That part is genuinely close. The trigger-backtest term is rounded to nothing (~8 seconds total for your search). At small `data_days` and small base-config counts, the trigger cost was small enough to hide. At `data_days=365` × 5 symbols × 2 TFs, each backtest balloons to 5+ minutes and the prediction collapses.

---

## 4. Update Data path — actually unchanged in spirit

When you click Update Data, the frontend POSTs `/api/strategies/{id}/refresh` for each selected strategy sequentially:

```
POST /refresh
  → recompute_and_persist_stored_trades(sid, user_id)
    → services.get_strategy_trades(strat)
      → services.prepare_data_with_indicators(...)
      → services.unified_trades(df, strat)
        → unified_engine.run_unified_backtest(df, config, ...)
```

This is the same engine code as Mass Builder's per-combo backtest — `run_unified_backtest`. Per-strategy cost scales with `bars × engine_work_per_bar`.

For a 30-day × 1Min strategy in isolation: ~16s end-to-end (measured yesterday in `_profile_backtest.py`). For 10 strategies bulk-clicked: ~2-3 min total. Sub-minute timeframes (10Sec, 5Sec) scale linearly with bar count, so they're proportionally slower.

**What changed in the last few days that affected Update Data perceived speed:**

1. ✅ **`9ee42f1` + `c85dc91` (Apr 28)**: 30s → 15min in-process TTL cache for `prepare_data_with_indicators`. **Faster** for bulk Update Data on the same symbol/timeframe. No regression.

2. ⚠️ **`dbe0cb3` → `dc8abd9` → `1dc14bf` (Apr 28 evening)**: Auto-parity got bolted onto refresh briefly, then made async, then split out into a separate "Run Parity" button. During the brief sync window (a few hours yesterday), Update Data was running parity in-line and was 10-15× slower. **Already reverted** as of `1dc14bf` last night.

3. ✅ **`2914f20` (Apr 28)**: Strategy Detail chart-data state-injection vectorized. Sid 127 chart-data went from 10-20 min to 5-6s. Improved `/chart-data`, doesn't affect Update Data directly.

So as of right now (post-`1dc14bf`), Update Data should be the same speed as several days ago, plus a small *improvement* from the longer in-process cache. If you're observing it as slow, the likely causes are (a) sub-minute strategies, (b) cold cache after a Railway redeploy, or (c) Polygon rate limits if hammered.

---

## 5. Per-bar pandas overhead — independent issue, both paths

Both `run_unified_backtest` (mass builder) and the parity replay loop have a `row = df.iloc[i]` in their bar loop. Per-bar pandas Series materialization adds ~6 seconds of pure overhead per 47k bars on top of the engine work.

Yesterday I fixed this in `parity_service` (numpy column extraction outside the loop) and saw 722 → 1500 bars/s = ~2× speedup. Not done yet on `unified_engine.run_unified_backtest`. Same change, same scale of improvement.

This is independent of the rapid-path question — even if the rapid path were re-wired today, the cached-bar replay is fast enough that this doesn't matter much there. The vectorization help is for the *full* backtest path used by Update Data and any first-time-through-the-cache call.

---

## 6. Summary by Kevin's questions

**"Is mass builder still doing the rapid backtest?"**
No. Hasn't been since 2026-03-16 (`467985a`). Each base config now runs the full bar-by-bar engine.

**"Is it currently recalculating entry/exit triggers? Re-drawing candles every combo?"**
Yes — kind of. Bars themselves come from a cached `df` that's loaded once per (symbol, tf) group, but every combo re-runs the full `IncrementalIndicatorEngine.update_bar()` + `TriggerEvaluator.evaluate_bar_for_backtest()` over those bars. So bars are reused; indicator/trigger MATH is repeated.

**"Is there any semblance of the old rapid behavior in the code?"**
Yes — the functions still exist (`precompute_bar_cache`, `run_trades_from_cache`). They're just no longer called by mass builder. Parity simulator uses them.

**"How does Update Data compare to a few days ago?"**
Same engine code path. The only intentional change in Update Data flow this week was the briefly-bolted-on auto-parity, already removed. Performance should match pre-Apr-28 levels.

**"What slowed things down compared to historical fast performance?"**
- Mass Builder rapid path removal (2026-03-16) is the big regression — ~10–50× slower per combo, dominant when `data_days` is large.
- Many small engine-correctness changes since M7 (timestamps spec, exec types, L-fill clamping, ExecTypeManifest, etc.) — each adds a few percent to per-bar cost. Cumulative drift, but small relative to the rapid-path loss.
- Estimator never recalibrated, so what *should* take a few hours looks like 60 minutes in the UI prediction.

---

## 7. Roadmap follow-ups (when we come back to this)

**P1 — Restore the rapid backtest path in mass builder.** Fix the broken integration: build the precompute mega-config to declare ALL selected entry/exit triggers so the cache stores booleans for every one. Each combo's `run_trades_from_cache()` then succeeds because all its triggers are in the cache. Estimated ~10–50× speedup on mass searches per the docstring claim.

**P1a — Surface "Rapid vs Standard" as a user-facing toggle in the Mass Builder UI** (Kevin's idea, 2026-04-29). Mirror the existing pattern of Trigger Hi-Fi / Confluence Hi-Fi toggles: let the user decide between speed (rapid path, indicator math runs once per `(symbol, tf)` group) and highest fidelity (standard path, full bar-by-bar engine recomputed per combo). Sketch:
- Default: Rapid (matches the speed users had pre-2026-03-16; usually what people want for exploratory searches)
- Opt-in: Standard (when you specifically want the unified engine to re-evaluate per combo — useful as a parity sanity check or when a strategy uses an exec type that the rapid cache doesn't yet support)
- Storage: `mass_searches.config_data.execution_mode` = `"rapid" | "standard"` so the choice is recorded with each search and can be filtered in Mass Results
- Mass Builder execution path branches on this flag: rapid → call `precompute_bar_cache()` + loop combos through `run_trades_from_cache()`; standard → existing `run_unified_backtest()` per combo
- Estimator recalibrates against the chosen mode (rapid uses ~5ms/combo, standard uses bar-loop cost)

This generalizes nicely: when we eventually add new execution types or pack semantics that the rapid cache can't replay correctly, users can fall back to Standard for those searches without us having to gate it server-side.

**P2 — Vectorize the bar loop in `run_unified_backtest`.** Pre-extract numpy O/H/L/C/V arrays outside the loop, iterate by index. Same pattern applied to `parity_service` yesterday. Independent of P1; benefits Update Data, mass builder Standard mode, parity, anywhere `run_unified_backtest` is called. ~2× per-call speedup expected.

**P3 — Recalibrate the estimator.** After P1 lands, separate Rapid-mode constants from Standard-mode constants — Rapid: `~5ms per combo replay`; Standard: `bars_per_strategy × per_bar_cost × n_base_configs`. Honor the toggle from P1a in the prediction so the UI stops lying.

**P4 — Strategy-level parity workflow finalization.** This was mid-flight last night. Run Parity button shipped (`1dc14bf`). Need to: click it on the 10 Tier-A mirrors, read the verdict matrix, run user-pack 4Q parity on the v2 packs to identify any pack-level batch↔live drift. See `docs/Migration_Tracking_2026-04-28.md` and `memory/project_session_2026-04-28.md`.

P4 lands before P1/P1a/P2/P3. Migration finish + trustworthy parity = the foundation we need before optimizing Mass Builder. Otherwise we'll be optimizing strategies we can't trust.
