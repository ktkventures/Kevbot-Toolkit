# Current `dev` Candle-Time Parity Audit

**Audit date:** 2026-07-15

**Audited branch:** `origin/dev`

**Pinned commit:** `3cdb29fc79201efda67d2901dff8dda70d6e9532`

**Commit date:** 2026-07-14 16:54:28 -0600

**Scope:** diagnosis and regression reproductions only; no production fixes

## Purpose

This is the requested current-code follow-up to the earlier audit that was based
on a March snapshot. The goal is to evaluate the parity frontier that exists on
the current `dev` branch, particularly:

- WebSocket/per-second events through candle construction;
- primary and multi-timeframe candle-close ordering;
- live-model source routing;
- partial/grace versus final-candle decisions;
- reconnect/replay observability; and
- whether the new replay and health tooling can validate those paths.

The companion test file is
`src/test_current_dev_candle_parity_audit.py`. Its tests intentionally fail on
the pinned commit. They are evidence, not proposed fixes.

## Executive conclusion

Current `dev` is materially stronger than the old audited snapshot. The
force-full MTF re-true correction is well targeted, the liveness work is an
improvement, and the existing gap/correction suites cover meaningful historical
failures.

It is not yet safe to conclude that live and backtest candle decisions are
aligned. The highest-impact current defect is that an explicitly chart-only,
incomplete `>=60s` partial can be force-closed through the real strategy
pipeline before its final constituent data arrives. Additional current defects
exist in default-model routing and grace-fire suppression. These are
application-level issues; changing Railway hosts would not correct them.

## Findings

### P0 — A chart-only `>=60s` partial can drive a real strategy close

`on_second_bar` feeds per-second values into `>=60s` primary builders with
`close_on_boundary=False` and describes those partials as chart visualization
only. `flush_stale_bars`, however, force-closes the same builder with zero grace
and then calls `monitor.on_bar_close` for every primary monitor on that
timeframe.

Relevant locations:

- `src/ralph_engine.py:5335-5339` — declares the per-second `>=60s` partial
  chart-only and non-canonical.
- `src/ralph_engine.py:5677-5684` — updates that chart partial.
- `src/ralph_engine.py:3950-3961` — explicitly says the partial contains
  incomplete data and must not overwrite canonical cache rows.
- `src/ralph_engine.py:3997-4010` — nevertheless commits it through every
  matching strategy monitor.
- `src/ralph_engine.py:4555-4576` — later canonical `>60s` completion is handled
  as a correction and does not refire alerts.

Reproduction: after seconds `:00` through `:54`, a wall-clock flush at `+61s`
sent close `100.54` to the strategy. When seconds `:55` through `:59` arrived,
the complete value was `100.59`; the earlier indicator/position transition was
already committed and could not be retracted.

Required acceptance test: model the production ordering
`per-second visual partial -> flush wins -> completed 1Min fan-out arrives` for
both 1Min and 5Min primaries. The monitor must run exactly once and must receive
the complete canonical OHLCV.

### P1 — `ws_rest_spliced` is the default but is missing from routing gates

`strategy_models.py` declares `ws_rest_spliced` as the default live model, but
three upstream gates enumerate only `ws_agg_locked` and
`ws_agg_with_rest_backfill`:

- A-channel subscription: `src/ralph_engine.py:6436-6445`
- completed 1Min dispatch: `src/ralph_engine.py:5422-5427`
- `>60s` primary fan-out: `src/ralph_engine.py:4522-4528`

For a standalone 1Min+ strategy, the default model can receive neither the
per-second subscription nor the canonical aggregate dispatch. A co-located
strategy on the same symbol using one of the older models can mask the problem
by opening the subscription and causing the shared pipeline to run.

Required acceptance tests:

1. One symbol, one 1Min monitor, no intrabar triggers, default live model:
   `A.<symbol>` must be subscribed and one completed minute must dispatch.
2. One symbol, one 5Min default-model monitor: five completed minute aggregates
   must produce exactly one 5Min close.

### P1 — No signal at grace can suppress a valid final-candle signal

`_check_strategy_grace_fires` sets `monitor._fired_bucket` unconditionally after
evaluating a partial candle. The final close path suppresses signals whenever
that marker matches, even if the grace evaluation emitted no alert.

Relevant locations:

- `src/ralph_engine.py:3097-3115`
- `src/ralph_engine.py:4189-4208`

This contradicts the nearby fallback comments. A trigger that becomes true only
in the final seconds of a candle is silently discarded.

Required acceptance test: grace snapshot emits zero signals; late seconds turn
the trigger true; final close must dispatch once.

### P1 — Railway DB hot reload can create a different warmup/topology than boot

`worker.py` replaces the engine's native hot-reloader with `db_hot_reload`.
That replacement uses a flat seven-day primary and shadow load for newly added
strategies. The source itself notes that this leaves a 1Day shadow with roughly
five bars. Existing strategy configuration changes are re-instantiated from
whatever builder history already exists and do not run
`finalize_shadow_engines`, so a newly introduced secondary gate may have no
shadow engine until restart.

Relevant locations:

- `src/worker.py:934-1001`
- `src/worker.py:1009-1063`
- `src/worker.py:1101`
- production TF-scaled warmup: `src/ralph_engine.py:1872-1901` and
  `src/ralph_engine.py:2000-2034`

Consequently, behavior can depend on whether the same strategy existed at boot,
was added during the session, or was edited in the UI.

Required acceptance test: hot-add or edit a strategy with a 1Day gate, then
compare its initialized shadow state and confluence records to a clean boot of
the identical configuration.

### P1 — Replay v1 is an engine/bar replay, not a provider-event replay

`replay_harness.py` reads independently stored, already-aggregated
`live_bars.first_*` rows for each primary and secondary timeframe, then calls
`StrategyMonitor.on_bar_close` and shadow engines directly. Its intrabar series
comes from settled historical `1Sec` REST data.

It bypasses:

- WebSocket parsing and A-channel subscription selection;
- `WsAggMinuteBuilder` and `BarBuilder` construction;
- reconnect ordering and replayed events;
- duplicate/correction behavior;
- flush-versus-fan-out races; and
- strategy grace firing.

Therefore it cannot detect the three live-path findings above and cannot yet
serve as the requested WebSocket-to-builder replay oracle.

Relevant locations:

- `src/replay_harness.py:133-151`
- `src/replay_harness.py:198-212`
- `src/replay_harness.py:239-241`
- `src/replay_harness.py:264-327`

The file header at `src/replay_harness.py:45-53` also still warns that the
primary lane is not faithful, while current rollout documentation calls the
harness working. The clean-day replay-versus-live validation described in the
rollout plan had not occurred at the pinned commit.

### P1 — Replay scoring can understate or mis-window parity

Two independent scoring defects were reproduced:

1. `pair()` uses greedy nearest-first matching. For replay times `[4, 8]`,
   reference times `[0, 5]`, and tolerance 5 seconds, it reports one match
   (50% combined) even though a valid one-to-one assignment contains two
   matches (100%). See `src/replay_harness.py:332-345`.
2. `lanes()` selects backtest rows by `entry_fill_ts` and then takes all exits
   from those rows. An exit inside the requested window is excluded if its
   entry preceded the window, while an exit after the window is included if its
   entry was inside. Live exits are independently filtered. See
   `src/replay_harness.py:354-373`.

These defects can change the reported replay ceiling without changing engine
behavior.

### P2 — Healthcheck startup grace is unbounded and feed-blind

If the engine heartbeat does not exist, `engine_health.check` trusts a fresh
manager heartbeat. Because the manager continuously refreshes that file, there
is no actual elapsed boot deadline. The test reproduced a healthy result after
24 hours with an engine heartbeat that never appeared.

The engine pulse is also updated by a periodic loop that explicitly runs
regardless of tick arrival. A disconnected WebSocket, stale candle stream, or
failed DB writer can therefore remain Docker-healthy.

Relevant locations:

- `src/engine_health.py:38-62`
- `src/ralph_engine.py:6925-6969`
- `Dockerfile.worker:24-25`

The implementation does not yet satisfy the rollout plan's stated
"last-candle freshness" check. With a 300-second maximum age, 60-second Docker
interval, and three retries, an event-loop stall may take roughly seven to eight
minutes to reach restart eligibility.

### P2 — The July 14 primary-resync SEV path remains environment-reachable

Rollout documentation says primary resync stays permanently deleted. The
implementation remains present and can be re-enabled with
`RORT_PRIMARY_STATE_RESYNC_S>0`; `RORT_PRIMARY_STATE_RESYNC_APPLY=1` additionally
applies rebuilt state.

Relevant locations:

- `src/worker.py:1207-1265`
- `src/ralph_engine.py:2172-2203`
- `src/ralph_engine.py:3614-3660`

It is default-off, so this is not an active parity defect. It is an operational
regression risk because the same path already caused repeated multi-minute
engine starvation.

## Verification performed

Against the pinned commit:

- targeted candle/gap/rebroadcast/shadow/liveness suite: **61 passed**
- unified engine parity script: **31 passed**
- parity regression script: **7 passed**
- clobber/late-fan-out suite: **7 passed**
- Ralph fidelity script: **13 passed, 1 failed**

The Ralph fidelity failure is an old assertion for `ema_8`; its fixture now
configures EMA 9/21/200. It appears to be stale test maintenance rather than a
candle defect, but it prevents the broad fidelity script from being a clean
gate.

The companion audit tests are expected to fail on current `dev`. Each failure
should turn green only after the corresponding behavior is corrected.

## What appears sound in the latest update

- The `RORT_SHADOW_RETRUE_FORCE_FULL` change addresses the demonstrated
  fast-path lineage problem directly and has meaningful flag-off/flag-on/cold
  regression coverage.
- The engine heartbeat and alert-lag warning are materially better than the old
  existence-only healthcheck for detecting event-loop starvation.
- Current gap-fill, rebroadcast, volume-integrity, and late-constituent merge
  tests cover real historical defects and pass on the pinned commit.

## Recommended evaluation order

1. Run the companion failing audit tests unchanged against current `dev`.
2. Fix and validate the `>=60s` provisional-close ownership first.
3. Make live-model capability/routing sets single-sourced and include the
   default in subscription, 1Min dispatch, and primary fan-out tests.
4. Separate "grace evaluated" from "alert dispatched" semantics.
5. Make DB hot reload use the same TF-scaled warmup and topology finalization as
   clean startup.
6. Upgrade replay fixtures to raw provider events and route them through the
   real builders, including reconnect, duplicate, late-event, and flush races.
7. Correct replay matching/windowing before using percentages as an oracle.
8. Add an explicit boot deadline plus market/session-aware feed and candle
   freshness to the healthcheck.

No production implementation changes are proposed in this branch. That keeps
the review aligned with the requested standard: current-`dev` diagnosis backed
by failing tests, with fixes left to the project's normal validation gates.
