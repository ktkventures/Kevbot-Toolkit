# Plan — `>=60s` Primary Close/Dispatch Fix (Brandon audit P0 + P1-routing)

**Status:** DRAFT for Kevin review — no engine code changed yet. Plan-mode item (live
decision path). **Author:** Claude (2026-07-15), grounded in Brandon Armstrong's audit
(`docs/audits/Current_Dev_Candle_Parity_Audit_2026-07-15.md`, cherry-picked to dev).

## Why P0 and P1-routing are ONE seam (the key realization)
Reading current dev, the two findings share a root: the canonical close path for a
`>60s` PRIMARY only serves two live models, so the default model falls through to a
garbage close.

- **Canonical close (good path):** `ralph_engine.py:4518-4587` fans a completed 1Min
  bar into the `>60s` primary builder (`accept_second_bar(close_on_boundary=True)`) and
  drives the strategy via `_run_monitor_pipeline_for_completed_bar`. **But its eligibility
  gate (`:4522-4528`) requires `live_model in ('ws_agg_locked',
  'ws_agg_with_rest_backfill')`.** The DEFAULT model `ws_rest_spliced`
  (`strategy_models.py`) is NOT in the set → a standalone default-model `>60s` primary
  gets **no canonical close here** (Brandon P1).
- **Flush close (garbage path):** `flush_stale_bars` (`:3927`) force-closes the stale
  builder. The WRITE is correctly guarded — primary `>=60s` partials are NOT written to
  `live_bars` because they're incomplete per-second chart-visual data (`:3950-3988`,
  hotfix #2). **But the STRATEGY DECISION is NOT guarded:** `:3997-4010` calls
  `monitor.on_bar_close(completed, …)` for every matching monitor, including primary
  `>=60s`, on that same incomplete partial (Brandon P0). The later canonical completion
  is treated as a correction and does not re-fire (`:4555-4576`).

**Net:** a `ws_agg_*` `>60s` primary gets BOTH a canonical close and a spurious
flush-partial close (P0 double/early decision). A **default-model** `>60s` primary gets
ONLY the flush-partial close — it decides on incomplete data **every minute** (P0+P1
combined). This is a prime suspect for the operational gap between live and the replay
ceiling on 1Min+ strategies, and it intersects the known "5Min+ primary never closes"
item (memory `project_5min_plus_primary_never_closes`, 137).

## Reproductions (already in-tree, currently FAILING)
`src/test_current_dev_candle_parity_audit.py`:
- `test_gte_60_chart_partial_must_not_drive_a_strategy_close` — after 55 per-second
  chart partials, `flush_stale_bars(+61s)` must leave `monitor.close_calls == []`.
- `test_default_live_model_receives_the_completed_one_minute_ws_bar` — a default-model
  1Min monitor must get `A.<symbol>` subscribed and exactly one completed-minute dispatch;
  a default-model 5Min monitor: five completed minutes → exactly one 5Min close.

## Proposed change (flag-gated, default OFF: `RORT_CANONICAL_PRIMARY_CLOSE`)
Two coordinated edits, shipped together because fixing one without the other regresses:

1. **Single-source the live-model capability set (P1).** Replace the three hard-coded
   `in ('ws_agg_locked','ws_agg_with_rest_backfill')` gates with one helper, e.g.
   `_model_gets_ws_agg_dispatch(live_model)` that INCLUDES the default `ws_rest_spliced`.
   Sites: `:4522-4528` (>60s fan-out), the 1Min dispatch gate (`~:5422`), and the
   A-channel subscription gate (`~:6436`). This gives default-model `>=60s` primaries a
   real canonical close/dispatch.
2. **Stop flush from driving strategy decisions on incomplete primary `>=60s` partials
   (P0).** In `flush_stale_bars`'s monitor loop (`:3997-4010`), SKIP monitors with
   `tf_seconds >= 60` whose `completed` came from a chart-visual partial (mirror the exact
   condition that already guards the WRITE at `:3962-3988`). Sub-minute (`<60`) primaries
   KEEP flush-close (they legitimately rely on it — per-second A events are too sparse to
   always cross `close_on_boundary=True`). Pure-secondary shadow closes are untouched.

Because edit #1 makes the canonical close reach ALL models, edit #2 is safe: removing the
flush-partial decision no longer risks "never closes" — the completed-minute fan-out owns
the close for every `>=60s` primary.

## Acceptance gate
- Brandon's two tests above turn GREEN (and stay green flag-ON).
- Flag-OFF: both tests still fail (proves the flag is the switch, not a test edit).
- Existing suites stay green: `test_ralph_*` fidelity/gap/rebroadcast, `fidelity_parity_suite.py`
  (canary 267), the clobber/late-fan-out suite. Run BEFORE arming (memory `feedback_fidelity_parity_gate`).
- New canonical-close test: one default-model 5Min monitor, feed 5 completed minutes →
  exactly ONE 5Min `on_bar_close`, receiving the COMPLETE canonical OHLCV (not the partial).

## Risks / watch-items
- **"5Min+ never closes" regression (137):** the whole point of edit #1 is to prevent it;
  verify the fan-out fires for the affected symbol/tf on a live canary BEFORE arming fleet.
  Thin-symbol edge: `>=60s` primaries still get a completed minute every RTH minute, so the
  fan-out has input; sub-minute unaffected.
- **Double-volume caveat (`:4507-4516`):** the per-second loop + fan-out both add volume to
  the forming period → closed `>60s` bars can carry ~2x volume (VWAP_V2/RVOL_V2 only).
  Pre-existing; this plan does not fix it but must not worsen it. Note for a follow-up
  (`skip_volume` on the forming-loop accept).
- **Below the replay harness:** the `/replay-check` ceiling will NOT move from this fix
  (the harness consumes recorded `first_*`, bypassing construction — Brandon's scope point).
  Validate this fix by measuring LIVE before/after on a 1Min+ default-model canary vs the
  settled lane, NOT by the ceiling.

## Rollout
1. Land behind `RORT_CANONICAL_PRIMARY_CLOSE=0`. Green all suites + the 3 acceptance tests.
2. Arm on ONE default-model 1Min+ canary (announce + Deploy_Log + monitor alert-lag).
3. Measure that canary's live vs settled lane for a session; if it rises toward the ceiling
   and no over-fire, widen to the fleet.
4. Then P1-grace, P1-hot-reload, P2-healthcheck (each its own flag + Brandon test).

## Open questions for Kevin
- OK to treat P0 + P1-routing as one PR (they share the root)? Recommended: yes.
- Any `>=60s` primary you want as the first canary? (a default-model 1Min or 5Min sid).
- Do we also want the `skip_volume` forming-loop cleanup in the same PR, or a follow-up?
