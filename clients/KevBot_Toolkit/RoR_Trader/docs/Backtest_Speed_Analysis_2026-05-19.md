# Backtest "Update Data" Speed — Analysis & Options (2026-05-19)

Captured after the Worker lag-spiral fix. The lag fix's core lesson —
**don't replay history you've already processed; persist incremental
state and resume** — applies directly to backtest data updates. This
doc is a discussion record, not a committed plan.

## How it works today

Four related concepts:

- **Full backtest** — `get_strategy_trades` → `_unified_trades` →
  `run_unified_backtest`. Loads the full history, warms indicators from
  scratch, replays every bar through the bar-by-bar engine.
- **Rapid backtest** — Mass Builder only. Short window + simplified
  stops, stamped `data_source='rapid'`. Already fast; lower fidelity.
- **Incremental refresh** — `refresh_strategy_data` →
  `_generate_incremental_trades`. *Already windowed*: loads
  `[last_trade_entry − ~warmup, now]`, runs the engine on that window,
  appends trades with `entry_time > last`.
- **Update All Data** — `bulk_refresh_all_strategies`: a **sequential**
  `for` loop calling `refresh_strategy_data` for every strategy.

## Why "Update All Data" is slow — the real costs

It is **not** mainly the per-strategy backtest. Three structural costs:

1. **Sequential.** `bulk_refresh_all_strategies` processes strategies
   one at a time. N strategies = N × per-strategy cost, zero overlap.

2. **O(N²) collection I/O.** `refresh_strategy_data` calls
   `load_strategies()` (the *entire* strategies collection) on entry and
   writes the whole collection back on exit — *per strategy*. Bulk
   refresh of N strategies = N full loads + N full saves of the whole
   collection. This alone can dominate at 100+ strategies.

3. **Re-warm on every update.** Even the incremental path re-loads a
   data window and re-warms indicators from scratch over it each time.
   It never *resumes* — there is no persisted engine state to continue
   from. Cheap per strategy, but ×N and ×(every refresh).

The crashing cron job is almost certainly (1)+(2): a sequential job
whose wall-time and peak memory grow with strategy count until it times
out or OOMs.

## The lesson from the lag fix

`recompute_from_history` was O(N) — it replayed the full intraday
history on every bar correction. The fix: snapshot engine state before
each bar, and on an update *restore the snapshot + apply only the new
bar(s)* — O(Δ), not O(N). The indicator engine is already a pure
bar-by-bar incremental machine; the only thing missing for backtests is
**persisting that state** so an update can resume instead of restart.

## Kevin's position-continuity insight — and why it matters

A backtest needs history for two reasons:

- **Indicator warmup** — *bounded*. ~100–200 bars for EMA/MACD/ATR.
  (VWAP is the exception — session-cumulative, needs session-start.)
- **Position-state continuity** — *unbounded in principle*. To know you
  are "in a trade" you may have entered weeks ago. This is the only
  thing forcing a replay from the true beginning.

Kevin's observation is the key unlock: **if a strategy always starts
flat at a window boundary — never inheriting an open position — then
position continuity is no longer unbounded, and an update only needs
the bounded warmup window.** That is exactly what makes the "rapid"
windowed approach legitimate rather than an approximation.

Today the incremental path already half-does this (and has a real bug
because of the half-measure): it filters `entry_time > last_entry`, so a
trade that was *open* at the last refresh can never have its exit
updated — the update can't "continue" an open trade. That mismatch
between full and incremental is the open-vs-closed discrepancy Kevin
described.

Pairing it with **"don't fire/count a live alert until a fresh
FLAT→entry transition is observed"** makes live and backtest agree:
both start flat at any boundary, neither inherits a position. TradingView
is effectively in this camp — it recomputes over a bounded window and
does not carry indefinite cross-reload position state.

## Options (tiered by effort)

### Tier 1 — quick, low-risk, big bulk-refresh win
- **Parallelize `bulk_refresh_all_strategies`** with a thread/process
  pool (the Worker already uses pools for DB writes). Strategies are
  independent; expect a 4–8× wall-time cut.
- **Fix the O(N²) I/O**: load the strategies collection *once*, refresh
  all in memory, save *once*. Refactor `refresh_strategy_data` to accept
  an already-loaded strat (no internal `load_strategies`/save).
- These two together likely make "Update All Data" tolerable and let
  the cron job stop crashing — without touching engine semantics.

### Tier 2 — persist engine state for true incremental updates
- After a backtest/refresh, serialize the `IncrementalIndicatorEngine`
  snapshot (`snapshot_state()` already exists from the lag fix — it
  deep-copies IndicatorState + user-pack engines) alongside
  `stored_trades`.
- "Update New Data" then = restore snapshot + `update_bar` per genuinely
  new bar. O(new bars), no warmup reload. The position state machine is
  part of the snapshot, so continuity is preserved *correctly* — which
  also fixes the open-trade-exit bug.
- Cost: state must be serializable (deep-copy works; JSON/pickle needs
  care for user packs). Invalidate the snapshot on any strategy-config
  edit.

### Tier 3 — unify full/rapid around "always start flat"
- Make "start flat at the forward-test boundary" the contract for both
  backtest and live (live: gate alert counting on a fresh FLAT→entry).
- Then *every* update is a bounded-warmup windowed run — "full" and
  "rapid" converge; the full-history replay is only ever needed for the
  very first backtest of a strategy.
- Biggest change (touches alert semantics + KPI continuity); needs its
  own spec. But it is the end state that makes updates structurally
  cheap and removes the full/rapid divergence entirely.

## Recommendation

Do **Tier 1 now** — it is small, safe, and directly kills the
"Update All Data" pain + cron crashes. Treat **Tier 2** as the next
real project (it reuses `snapshot_state` from the lag fix and fixes a
correctness bug). **Tier 3** is the principled end state — worth a
dedicated spec, not an incidental change.
