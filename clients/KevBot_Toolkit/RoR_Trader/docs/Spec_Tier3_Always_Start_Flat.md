# Spec — Tier 3: Always Start Flat

**Status:** APPROVED (2026-05-20) — §6 decisions all resolved, ready
for implementation. Implementation-level spec for the "always start
flat" unification of full and rapid backtest semantics.

Companion docs:
- `Backtest_Speed_Implementation_Plan.md` — the original framing of
  Tier 3 in §"TIER 3 — Unify full/rapid around 'always start flat'".
- `Spec_Live_Execution_Fidelity.md` — Phases 1–5 (now shipped); this
  spec is independent but inherits LEF's snapshot/restore primitives.
- `Spec_LEF_Phase2_Implementation.md` — the snapshot-state contract
  Tier 2 (2c) added; Tier 3 builds on it.

This spec sits at the top of the "what's left" stack — §6 decisions
all locked 2026-05-20, implementation may begin per §8 sequencing.

## 1. Why

Three failure modes the current system can't cleanly address:

1. **Full backtest vs rapid backtest divergence.** Full = "replay all
   bars from data_seed start." Rapid = "windowed warmup + go." On an
   incremental refresh, the two can produce slightly different trade
   lists when a position straddles the warmup window boundary. The
   existing `_generate_incremental_trades` filters by
   `entry_time > last_entry` to dedupe — which is a *band-aid*, not a
   contract. The Tier 2 snapshot fixes the boundary case for the engine
   but leaves the semantic question unresolved: *if a strategy enters
   on 2026-03-01 and is still in-position on 2026-05-01, what counts
   as that strategy's first trade for "the last 30 days"?*

2. **Live restart anchoring.** When the live worker restarts (Railway
   deploy, hot reload, crash recovery), the engine warms up from
   recent history. Today the engine treats a pre-restart position
   state heuristically: if the warmup window starts with a
   `FLAT → entry` transition, it'll re-fire that entry. We've patched
   around this several times (the `_position_quantity` tracker, the
   alert recovery system, sid 132 / utbot_v4 fix). None of those
   solve the root problem: **the position machine should never inherit
   open state across a restart boundary.** A fresh FLAT→entry
   transition observed after the boundary is the only legitimate
   firing event.

3. **KPI distortion.** Strategies with very long holds (multi-day
   swing systems) can show wildly different KPIs depending on whether
   the data window happens to include their entry bar or only a
   slice of the open position. Today this is masked by
   `forward_test_start` boundary filtering, but that's a presentation
   layer — the underlying trade list is still ambiguous.

The unification target is a single contract that resolves all three:
**a strategy starts FLAT at every backtest/refresh boundary, and the
live engine never fires on a pre-existing position.**

## 2. The contract — verbatim

> At every boundary — backtest window start, live engine restart,
> incremental refresh start — a strategy's position state is FLAT.
> The engine fires an entry alert only on a fresh `FLAT → entry`
> transition observed *strictly after* the boundary. Open positions
> that span a boundary are visible only to the side of the boundary
> on which they were opened.

Two corollaries:

- **Backtest is windowed, always.** Full-history replay is needed only
  for a strategy's first backtest. Every subsequent refresh is a
  windowed run starting flat at the warmup boundary. The KPI baseline
  is the cumulative sum across all windowed runs the strategy has
  seen — same shape as today's `stored_trades`.
- **Live is restart-flat, always.** After any (re)start, the engine
  treats the first observed entry trigger as that strategy's first
  trade, regardless of whether the same trigger fired before the
  restart. The position-tracking guard from sid 132's fix
  (`_position_quantity`) becomes the *primary* mechanism, not a
  defensive fallback.

## 3. Three jobs, decoupled

Like LEF §5.1, "always start flat" has three independent jobs that
this spec must separately resolve:

### 3.1 Backtest side — windowed always

`run_unified_backtest` already accepts `include_open_position`. Today
it's a chart-rendering toggle: when True, an open trade at the end of
the data window gets a synthetic row so its entry marker shows. This
spec **redirects** that flag to mean: "include the open trade as a
trade record, even though it has no exit." That's an output choice.

The **input** change is harder: the engine must not synthesize an
inherited position from pre-window history. Today, when a windowed
backtest starts on bar N and bar N was part of an open position in
the full history, the rapid path silently ignores the open trade
(since it filters trades by `entry_time > last_entry`). Tier 3 makes
this **explicit**: the engine starts FLAT at bar 0 of the window,
and any unfinished pre-window position is "owned" by the previous
window's results.

### 3.2 Live side — restart-flat always

After a worker restart, the engine warms up indicators from history,
then enters its normal per-second loop. Today the position state
machine **reinstates** any open position it observes in the warmup
history (e.g., if the last bar before restart had an active entry,
the machine resumes with `status=IN_POSITION`). Tier 3 flips this:
**every strategy starts the restart with `status=FLAT`**. The first
entry trigger observed after the restart is treated as a fresh
entry — even if the underlying market state has not changed.

Concretely:
- `StrategyMonitor.__init__` no longer inspects warmup history for
  open-position synthesis. The position state machine starts FLAT.
- The `_position_quantity` tracker (already in place per
  `project_session_2026-04-23.md`) becomes the source of truth for
  "do we still have exposure?" — used for risk-mgmt accounting, NOT
  for entry/exit firing decisions.
- A restart that happens *during* an open position results in: the
  position remains in the broker (manual close required) but the
  engine does not fire another entry. The next entry trigger, when
  it comes, fires fresh.

### 3.3 KPI continuity

Cross-boundary trades present an accounting question: a trade that
entered before the boundary and exited after — does it count toward
the post-boundary stats?

**Decision (proposed):** *Only trades opened at or after the
boundary count toward post-boundary KPIs.* The "open trade
straddling the boundary" disappears from the post-boundary view —
its entry was already counted in the pre-boundary view and its
exit (if reached) is purely a real-money accounting concern, not a
strategy-performance question.

This decision keeps `forward_test_start` immutable
(`feedback_forward_test_start_immutable`) and makes "rolling 30 day
KPIs" mean something concrete: every trade whose entry timestamp
falls in the window.

## 4. Migration

Existing strategies have stored trade lists going back months/years.
Tier 3 should not retroactively change historical KPIs.

### 4.1 Existing trade lists

`stored_trades` stays canonical. The "start flat" contract applies
only to **new** trades generated after Tier 3 ships. The KPI
calculator already filters by `entry_time` relative to
`forward_test_start` — no migration needed.

### 4.2 Strategies in an open position at rollout

For every monitored strategy with `position.status == IN_POSITION`
at the moment Tier 3 ships, the worker:

1. Records a final `position_carryover` entry in the strategy's
   `live_executions` log.
2. Sets `position.status = FLAT` (engine state only).
3. Logs a one-line `MIGRATION` warning so we can audit cases where
   the user might be holding real-money exposure that the engine
   has now forgotten about.

Step 3 is the only operational concern — surface it in the Data
Fidelity tab as an amber banner with a "Resolve" button that confirms
the user is aware. After confirmation, the banner clears.

### 4.3 Webhook-origin strategies

These have a different position lifecycle — signals arrive from
TradingView/Scanz, not from the engine's trigger evaluator. The
contract still holds: when a webhook arrives, the engine fires on
it only if it represents a fresh FLAT→entry. A duplicate webhook
that doesn't match a fresh transition is logged as a dedupe event
and not fired.

### 4.4 Hi-Fi Pass 2

`reference_hifi_pass2.md` documents that Pass 2 refines entry/exit
timestamps. Tier 3 changes nothing about Pass 2's semantics — it
operates on already-recorded trades. The only edge case is when
Pass 2's refinement moves an entry timestamp **across** a boundary
(e.g., from 09:29:58 to 09:30:02). Decision: the entry's window
membership is determined by the **post-refinement** timestamp.

### 4.5 Mass Builder strategies (no re-run needed)

Confirmed 2026-05-20: strategies generated by Mass Builder do **not**
need re-backtesting when Tier 3 ships. The Mass Builder runs a full
backtest at creation time (it has to — no prior trades exist), so
the resulting `stored_trades` are complete under the old contract
and stay canonical per §4.1. Future "Update Data" refreshes on those
strategies use the new windowed-flat contract for new bars only;
historical trades are untouched.

The only exception is the §4.2 in-position-at-rollout case — a Mass
Builder strategy that happens to be `IN_POSITION` when Tier 3
deploys gets the same migration banner as any other strategy. New
Mass Builder runs between now and the Tier 3 deploy are safe;
nothing in that pipeline conflicts with the future contract.

## 5. Architectural changes — at a glance

| Component | Today | Tier 3 |
|---|---|---|
| `run_unified_backtest` | Optional `include_open_position` for chart rendering | Same flag; semantic = "emit the open trade in trades_df" |
| `PositionStateMachine` constructor | Accepts inherited state via `state` param | Always starts FLAT regardless of `state` param when called from a restart context |
| `StrategyMonitor.__init__` | Inspects warmup history for open-position synthesis | Starts FLAT; emits a `MIGRATION` log entry if pre-restart history suggests an open position |
| `_generate_incremental_trades` | Filters by `entry_time > last_entry` (band-aid) | Filters by `entry_time >= window_start` (contract) |
| KPI calculator | Filters by `forward_test_start` after-the-fact | Same; the contract makes the filter authoritative |
| Worker hot-reload | Re-instantiates monitor preserving position state (Tier 2c) | Re-instantiates monitor preserving position state UNLESS the live_model changed (then start flat) |

## 6. Decisions — all resolved 2026-05-20

All five decisions resolved with Kevin in the Tier 3 review session.
Resolutions stamped inline below; implementation may proceed.

### 6.1 The webhook dedupe rule

When two identical-payload webhooks arrive 100ms apart from the same
strategy, today both fire. Tier 3 makes the second a dedupe — same
trigger ID + same FLAT→entry resolution.

**Resolved 2026-05-20: DEDUPE.** A duplicate webhook that doesn't
match a fresh FLAT→entry transition is logged as a dedupe event and
not fired. Affects only webhook-origin strategies; legitimate
re-entry signals (FLAT→entry after an interim exit) are NOT
deduped.

### 6.2 The carryover position log

§4.2 proposes recording a `position_carryover` entry per
strategy-in-open-position at rollout. Engine audit, or user-facing?

**Resolved 2026-05-20: SURFACE TO USER.** A migration banner in the
Data Fidelity tab with a "Resolve" button. The banner only clears
after the user acknowledges, so we have an auditable trail of who
saw which carryover. This affects real money (the broker may still
hold the position even though the engine has forgotten it); the
silent-audit alternative was rejected.

### 6.3 The KPI accounting boundary

§3.3 proposes "trades opened at or after the boundary count" —
discarding cross-boundary holds from the post-boundary view. The
alternative is "include the post-boundary slice as a partial trade."

**Resolved 2026-05-20: DISCARD CROSS-BOUNDARY HOLDS.** Trades whose
entry timestamp falls in the post-boundary window count; everything
else does not. This makes "rolling 30 day KPIs" mean something
concrete: every trade whose entry timestamp falls in the window.
The "partial trade" alternative was rejected for its semantic
ambiguity (what's the entry price of a "partial trade" — the
boundary bar's close, or the original entry?).

### 6.4 The "rapid backtest" path

After Tier 3, rapid (windowed) and full (replay) backtests differ
only in **window length + indicator-convergence fidelity**.

**Resolved 2026-05-20: COLLAPSE.** Single path with a `fidelity`
parameter. The UI exposes a "Recompute from scratch" button that
sets fidelity to maximum (= full history replay). Default is
windowed (~100-bar warmup).

#### What "indicator-convergence fidelity" actually means

The original spec called this "stop fidelity" — overloaded term.
What actually differs between the two paths today:

| Indicator | Convergence horizon | Effect on 100-bar warmup |
|---|---|---|
| EMA-50 | ~250 bars to ~99.5% | Slight (0.1–0.5%) drift on EMA value for first ~30 bars after warmup; trivial after |
| MACD (12/26/9) | ~80 bars | Tail of warmup is enough; no observable drift |
| ATR-14 | ~50 bars | Fully converged inside the 100-bar warmup |
| Swing stop (lookback ≤ 20) | `lookback` bars | Buffer fills inside warmup — no drift |
| User-pack engines | Pack-defined; usually ≤ 50 bars | Pack-dependent; UT Bot v4 fine, longer packs may drift |

ATR drift propagates to stop/target sizing (since stops are computed
as `entry_price ± k·ATR`), so the surviving divergence is on the
**first ~30 trades after a refresh window starts**, and only on
strategies that use slow-converging indicators (EMA-50, MACD).

**Tier 2 (LEF 2c, shipped 2026-05-20) already eliminates this for
the common case** — when a valid engine snapshot exists, the
indicator state carries forward byte-identically, no re-warmup. The
remaining sources of drift after Tier 2 are:

1. Snapshot invalidated by a config change (fingerprint mismatch →
   falls back to windowed warmup)
2. First refresh after a strategy is created (no snapshot to
   resume from yet)
3. A user-initiated "Recompute from scratch" — intentional, expected

So Tier 3's `fidelity` parameter is a UI affordance over an already-
small divergence; the engine-side correctness work was done by Tier 2.

### 6.5 The chart rendering boundary

When the chart shows a date range that includes a strategy's
boundary, the rendered trade markers come from `stored_trades`.
After Tier 3, the pre-boundary slice of a cross-boundary hold is
"owned" by the pre-boundary window. Should the chart render it
anyway, or drop it?

**Resolved 2026-05-20: RENDER.** The chart is a historical record,
not a strategy-performance view. The entry marker (`+` symbol)
plots at its original entry timestamp regardless of window
ownership; the exit marker (`x` symbol) plots only when the alert
actually fired (per existing chart behavior, alert-driven). KPIs
and equity curves remain the performance-view surfaces and continue
to filter by §6.3's rule.

## 7. Validation

With §6 decisions locked (2026-05-20), three integration tests gate
the rollout when code lands:

1. **Backtest vs live parity across a restart.** Pick a strategy
   that had an open position at restart time. Run a full backtest
   ending after the restart. Compare the resulting trade list against
   the live `algo_history` lane. They must agree trade-for-trade
   for trades that opened **after** the restart. Trades that
   straddled the restart are counted only on the pre-restart side.
2. **First-signal-after-restart liveness.** Verify that a strategy
   which fired entry-trigger T before restart, and again 10s after
   restart, fires the post-restart instance as a fresh entry alert
   (not a dedupe). Test against the sid 132 historical signature.
3. **No legitimate signal dropped.** Replay one full RTH day for a
   representative strategy across a synthetic mid-session restart.
   Sum of (pre-restart alerts) + (post-restart alerts) must equal
   the alert count from an uninterrupted run, ± the
   intentionally-dropped open-position synthesis.

## 8. Sequencing

Each sub-phase ships as its own commit. Sub-phases 8.2–8.4 can
ship in parallel.

### 8.1 Lock decisions (no code) — ✅ COMPLETE 2026-05-20

Resolved §6.1–6.5 with Kevin. This spec is the authoritative
reference for the implementation; any deviation requires updating
§6 first.

### 8.2 Backtest contract

- Add the explicit FLAT-start invariant to `run_unified_backtest`
  (documentation + unit test).
- Update `_generate_incremental_trades` filter from `entry_time >
  last_entry` to `entry_time >= window_start` and verify zero
  regression on the existing strategy set.

### 8.3 Live restart contract

- Modify `StrategyMonitor.__init__` to start `position.status = FLAT`
  regardless of warmup-history inference.
- Emit the `MIGRATION` log on any strategy where the pre-restart
  position suggested IN_POSITION.
- Surface the migration banner in Data Fidelity tab per §4.2.

### 8.4 KPI accountancy

- Verify the KPI calc filter is `entry_time >= forward_test_start`
  (not `>`). Patch if not.
- Add an "open trade carryover" line item to the Configuration tab
  showing any pre-boundary holds that are still open in the broker.

### 8.5 Collapse rapid + full

§6.4 resolved to COLLAPSE — once 8.2–8.4 are stable on production,
consolidate the two paths into one with a `fidelity` parameter
(default = windowed, max = full replay). The "Recompute from
scratch" UI button sets fidelity to max. Not optional after Tier 3
lands — this is the end-state §6.4 specifies.

## 9. Risks

- **Behavioral change to alerts.** The restart-flat rule changes
  when a strategy fires after a restart. This is the *intended*
  change but must not silently drop legitimate signals. Validation
  test 7.3 gates this.
- **User confusion on KPI shifts.** Existing strategies' KPIs may
  shift slightly as the boundary contract gets enforced
  uniformly. Document loudly in release notes; show a one-time
  banner per strategy explaining the change.
- **Webhook-origin edge case.** Strategies that fire on every
  webhook (no dedupe) will see a behavior change if §6.1 lands
  "dedupe." Audit existing webhook-origin strategies before
  rollout.
- **Cross-boundary hold accounting.** §3.3's decision is clean but
  may surprise users who expect to see their "still-open" trade in
  the post-boundary view. The Configuration tab carryover line
  item (§8.4) mitigates this.

## 10. Out of scope

These are explicitly NOT Tier 3's job; defer to future work:

- Position size scaling across boundaries (deferred to portfolio
  rebalance work).
- Cross-strategy correlation accounting (separate analytics
  feature).
- Tax-lot tracking for the broker side (out of trading-system
  scope).
- Performance benchmarking ("did the contract change improve
  metrics?") — Tier 3 is a correctness contract, not a performance
  feature.
