# Candle-Time Parity Review

**Application:** RoR Trader / Ralph alert engine

**Review scope:** Live candle construction, multi-timeframe state, backtest alignment, worker deployment, and operational reliability

**Review status:** Proposed changes for review; not a production-readiness certification

## Executive summary

The deployment platform is not the main cause of the observed behavior. Railway can run this application as a web service plus an always-on worker, but the original live-data path did not construct or release candles according to the same rules as the historical backtest path.

The highest-impact defect was in the Polygon/Massive WebSocket route. The provider supplies completed one-minute aggregate events, but the live hub only tried to deliver those events to a 60-second builder. Strategies configured for 2-minute, 5-minute, 15-minute, hourly, or secondary multi-timeframe analysis therefore did not reliably receive completed live candles. A hub containing only a 5-minute strategy could receive one-minute events continuously without ever evaluating the strategy at a five-minute close.

This branch introduces a deterministic fixed-duration candle contract for the intraday path, adds regression coverage, and corrects several related sources of live/backtest drift. It intentionally does not claim complete parity for calendar bars, cross-session multi-timeframe state, or reconnect recovery. Those remaining items are documented below.

## Candle-time contract used by these changes

The changes adopt the following rules for fixed-duration intraday candles:

1. A candle timestamp identifies the **start** of its interval in UTC.
2. A candle becomes available to a strategy only after its interval has closed.
3. One-minute provider aggregates are rolled into registered coarser timeframes while preserving open, high, low, close, and volume.
4. Missing provider intervals remain missing. The engine does not invent zero-volume candles.
5. At a shared boundary, shorter timeframes close before longer timeframes. A longer-timeframe strategy can therefore see the lower-timeframe candle that just closed, while a lower-timeframe strategy cannot see a higher-timeframe candle that has not yet been published.
6. Trading-session eligibility is determined from the completed candle timestamp, not the timestamp of the tick or event that caused it to close.
7. Historical fixed-duration resampling uses the same UTC Unix-epoch boundaries as live aggregation.

Massive/Polygon documents that an aggregate is not emitted when no qualifying trade occurs in an interval. Preserving absent intervals is therefore necessary if provider history and live aggregation are expected to produce the same indicator and bar-count behavior. See the [Massive stock WebSocket overview](https://massive.com/docs/websocket/stocks/overview?assetClass=stocks&license=personal&name=stocks_starter).

## Issues discovered and changes made

| Priority | Issue | Why it matters | Status in this branch |
| --- | --- | --- | --- |
| Critical | One-minute WebSocket bars were not routed into 2m/5m/15m/hour builders. | Most configured live timeframes and MTF shadow engines could fail to close or evaluate. | Fixed by rolling each completed source aggregate into every compatible registered builder. |
| Critical | Intrabar feed detection checked `_intrabar_triggers`, but `StrategyMonitor` never populated it. | Polygon's per-second channel was not subscribed for strategies that needed intrabar execution. | Fixed by deriving the set from resolved trigger execution types. |
| High | The live builder generated flat, zero-volume candles for missing intervals. | Synthetic rows change EMA, ATR, VWAP, interpreter state, and bar-count exits relative to provider history. | Removed from tick, aggregate, accepted-bar, and stale-close paths. |
| High | Session gating used the arriving tick timestamp in tick and stale-close paths. | The 09:29 ET premarket candle could be evaluated as RTH when a 09:30 tick closed it. | Fixed by evaluating the completed candle's timestamp. |
| High | REST warmup could include the currently forming candle. | The same logical candle could be processed once during warmup and again after WebSocket completion. | Forming candles are excluded before indicator warmup. |
| High | Live-chart MTF values were forward-filled before the secondary candle closed. | Backtest-like live charts could use future higher-timeframe state and display signals that the live engine could not yet know. | Secondary state is shifted to its availability time before forward-fill. |
| Medium | Historical resampling relied on implicit pandas timezone and boundary behavior. | Local timezone, unsorted data, or duplicates could change candle membership. | Resampling now normalizes UTC, sorts, deduplicates, and specifies epoch/left-closed boundaries. |
| Medium | Database audit rows were buffered but not periodically flushed. | Fidelity evidence could remain only in process memory and disappear on restart. | Generic auditors with a `flush()` method are flushed periodically and at shutdown. |
| Medium | Worker status was written every two seconds per active user. | This creates unnecessary database load and cost without improving candle correctness. | Status writes are throttled to 30 seconds. |
| Medium | Webhook backtests referenced an undefined `strat` variable inside a broad exception handler. | Market-data loading silently failed, preventing price and stop/target evaluation. | The trading session is now passed explicitly by both callers. |

## Purpose of the code changes

### `src/ralph_engine.py`

The main purpose is to make live fixed-duration candle construction deterministic and compatible with historical aggregation:

- `BarBuilder.process_aggregate_bar()` rolls a completed provider candle into a compatible target timeframe without losing intrabar high/low or volume.
- Sparse intervals remain absent instead of being represented by artificial flat candles.
- History is normalized to UTC, sorted, and deduplicated before warmup.
- Duplicate and out-of-order live events are ignored.
- Polygon one-minute events are dispatched to every equal or coarser registered timeframe.
- Polygon per-second events drive sub-minute builders and intrabar execution requirements.
- Shorter timeframes are processed first at shared candle boundaries.
- Session checks use completed candle timestamps.
- Currently forming REST candles are removed from warmup.
- Audit flushing and heartbeat throttling improve the worker's observability and database behavior.

### `src/data_loader.py`

The historical resampler now makes its alignment rules explicit. It requires a `DatetimeIndex`, converts it to UTC, sorts it, removes duplicates, and resamples with a Unix-epoch origin using left-labeled, left-closed intervals. This matches the fixed-duration alignment used by the live `BarBuilder`.

### `src/app.py`

The live-chart MTF preparation now delays secondary interpreter state until the secondary candle has actually closed. The webhook backtest path also receives its trading session explicitly instead of attempting to read an undefined local variable.

### `src/test_ralph_fidelity.py`

Regression tests cover:

- sparse periods without synthetic candles;
- exact five-minute OHLCV rollup from five one-minute aggregates;
- routing one-minute Polygon events into 1m, 5m, and 15m builders;
- the 09:29/09:30 ET session boundary;
- declaration of per-second feed requirements for intrabar strategies.

## Validation performed

The following checks passed on this branch:

- Python compilation for the complete `src` directory;
- 27 unified-engine parity tests;
- 19 Ralph fidelity tests;
- a focused sparse-data and UTC five-minute resampling check;
- `git diff --check`.

The tests emit warnings that the `utbot_default` and `ema_price_position_default` groups refer to templates that are not present in the test environment. These warnings predate this branch but should be reconciled so that a removed or unavailable user pack cannot silently change a production strategy.

## Remaining parity and production risks

These items are not resolved by this branch and should be treated as follow-up work before relying on the engine for unattended production alerts.

### 1. User-specific module monkey-patching

`src/worker.py` replaces module-global `ralph_engine.load_engine_state` with a user-specific closure from multiple user threads. Two engines starting concurrently can race and load the wrong user's position state. Database persistence should be injected as an engine dependency rather than installed as a module-global function.

### 2. One market-data connection per user

The worker creates a complete `RalphEngine` and provider WebSocket set for every active user, even though all instances use the same system-level provider credentials. This consumes connections unnecessarily and can hit provider limits. A safer design is one shared feed per asset class, with normalized events multiplexed to user strategy engines.

### 3. MTF shadow state is not session-specific

Shadow indicator state is shared by symbol and timeframe. It is not keyed by trading session. An RTH strategy and an extended-hours strategy can therefore observe secondary-timeframe state built from different session expectations. MTF state should be keyed by at least symbol, timeframe, and session.

### 4. Finer secondary timeframes in coarse-primary backtests

The backtest preparation derives secondary timeframes from the primary dataset. A 1-minute secondary timeframe cannot be reconstructed from a 5-minute primary candle. These configurations can currently be skipped by broad exception handling. The backtest should load a common lowest-resolution source or fetch the finer timeframe directly and align state by availability time.

### 5. Calendar timeframe semantics

The fixed-duration alignment in this branch is appropriate for intraday seconds, minutes, and hours. It is not an exchange-calendar implementation:

- a seven-day Unix-epoch bucket does not represent a normal trading week boundary;
- `1Month` is represented as a fixed 30-day duration;
- daily candles need explicit session and exchange-calendar semantics.

Daily, weekly, and monthly strategies should use provider-native calendar bars or an exchange-calendar-aware builder.

### 6. Reconnect reconciliation

After a WebSocket interruption, the first coarser candle may be built from only the source aggregates received after reconnection. Before resuming alerts, the worker should fetch and replay all provider aggregates after the last committed source timestamp, then mark the stream caught up.

### 7. Operational health reporting

`Dockerfile.worker` checks only whether `/tmp/worker_alive` exists. Once created, that file does not prove that it is recent, that the database is reachable, that WebSockets are authenticated, or that expected symbols are receiving candles. Health reporting should expose at least:

- manager-loop heartbeat age;
- provider connection state;
- last source event per symbol;
- last completed candle per timeframe;
- database write success age;
- reconnect count and current replay/caught-up status.

## Railway assessment

Railway is suitable for reviewing and running this MVP as separate web and worker services. Migrating hosts would not correct candle construction or worker races. The current worker should be budgeted as a continuously running service because it polls the database and maintains outbound WebSockets; Railway serverless sleep will not behave like a market-hours scheduler in that configuration.

Relevant Railway documentation:

- [Background workers](https://docs.railway.com/guides/cron-workers-queues)
- [Serverless behavior](https://docs.railway.com/deployments/serverless)
- [Restart policies](https://docs.railway.com/deployments/restart-policy)
- [Healthchecks](https://docs.railway.com/deployments/healthchecks)
- [Pricing](https://docs.railway.com/pricing)

For an MVP deployment, retain Railway but use a paid always-restart policy, external freshness monitoring, and a staging worker that cannot send production webhooks. Reassess the platform after the shared-feed worker design is understood; connection topology and observability are more urgent than changing hosting providers.

## Safe review and acceptance plan

This work should remain on its feature branch and in a **draft pull request** until the following review is complete:

1. Review the candle contract and confirm that all provider timestamps are interval-start timestamps.
2. Deploy the branch as a separate staging worker with webhook delivery disabled or redirected to a test endpoint.
3. Record a full session of normalized source events for several representative symbols.
4. Replay the exact same events through the live builders and the backtest path.
5. Compare candle timestamps and OHLCV, indicator values, interpreter states, trigger booleans, and position transitions at every close.
6. Test sparse trading, premarket-to-RTH boundaries, daylight-saving transitions, a WebSocket reconnect, and strategies using both lower and higher secondary timeframes.
7. Resolve the user-state race and session-specific MTF state before enabling multi-user production alerts.

Because the pull request is a draft, simply pushing updates to the branch does not modify `main` or the Railway production deployment. If the proposal is rejected, the branch can be closed or deleted without reverting production code.
