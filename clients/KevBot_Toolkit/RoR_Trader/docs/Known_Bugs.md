# Known Bugs

A living log of bugs we've identified but not yet fixed. Each entry lives
here until either (a) shipped + verified, or (b) explicitly deprecated.

**Status legend:** `OPEN` (known, not started) · `WIP` (being worked on) ·
`FIXED-PENDING-VERIFY` (shipped, awaiting prod confirmation) · `CLOSED`
(verified fixed) · `WONTFIX` (deliberate, with rationale)

> Convention: when a bug is closed, move it to the bottom of this file
> under "Closed" — keep history but stop scrolling the active list.

---

## Active

### mlv2_cross_bear walker reclassification
- **Status:** OPEN
- **Discovered:** 2026-05-27 (Railway logs, multiple strategies)
- **Severity:** Medium — silently rewrites correct exit reasons on a
  subset of trades
- **Location:** `src/api/services/backtest_service.py:_hifi_resolve_trades`
  around line 806-828 (the stop/target walker)
- **Symptom:** When a trade's `exit_reason` is a signal name like
  `mlv2_cross_bear` (an indicator-vs-indicator cross with no L-type spec)
  AND the trade also has a defined `stop_price`/`target_price` from the
  strategy, the walker still runs over the bar's 1-sec bars looking for
  stop/target level crosses. If price happens to touch the stop or
  target level during that bar, the walker rewrites `exit_reason` to
  `stop_loss` or `target`. The original signal-cross reason is lost.
- **Observation:** Railway `[HIFI] Resolved … outcomes changed` counters
  showed 1-11 trades flipped per strategy on the 2026-05-27 17:10-17:13
  passes. sid 172 specifically didn't see flips, but the pattern is
  symbol/volatility-dependent.
- **Root cause:** The walker's bail-out at line 806-809 only skips
  `exit_reason in ('stop_loss', 'stop')` with `stop_et != 'L'`. It does
  NOT skip non-stop/target exit reasons that happen to coexist with L-type
  stop/target levels.
- **Fix sketch:** Add a third bail-out: `if exit_reason not in
  ('stop_loss', 'stop', 'target') and not is_ltype_signal_exit: continue`.
  Treats "signal exit without per-second walker support" as a
  non-candidate, preserves original reason.
- **Risk:** Need to confirm no other walker pathway relies on the
  fall-through. Tests in `test_hifi_exit_timestamp.py` should catch
  regressions.

---

### sid 150 has 5043 trades stuck at `hifi_resolved=False`
- **Status:** WONTFIX (cosmetic)
- **Discovered:** 2026-05-27 (immediately after reverting commit
  `5023be6`)
- **Severity:** Low — no functional impact; just confusing in the DB
- **Location:** `trades` table, `strategy_id=150`, `data.hifi_resolved=False`
- **Symptom:** ~5043 trades on sid 150 have `hifi_resolved=False` in
  their `data` JSONB, written by the short-lived "persist False"
  branch in `5023be6`. The current Hi-Fi pass (post-revert in `723141b`)
  only skips on `True`, so these trades still get walked every pass —
  same throughput cost as if they were `None`. The `False` marker just
  sits there confusing future debugging.
- **Rationale for WONTFIX:** A one-time SQL `UPDATE trades SET data =
  data - 'hifi_resolved' WHERE strategy_id=150 AND
  data->>'hifi_resolved' = 'false'` would clean it up. Not worth a
  migration; the new incremental mode (`51835f3`) makes it irrelevant.

---

### Volume match ~54% on rest-1s-aggs vs cache+backfill bars
- **Status:** OPEN
- **Discovered:** Kevin flagged 2026-05-27 — pre-existing known
  limitation
- **Severity:** Low for entry/exit timing (most user-pack indicators use
  candle close); could be higher for volume-based filters or VWAP
- **Location:** Bars Comparison page (parity comparison view)
- **Symptom:** When comparing the REST 1-second aggregate bars to the
  cache+backfill bars on the parity comparison page, the volume match
  rate is ~54% (vs ~100% for OHLC). Close prices line up cleanly; the
  volume column on the same timestamped bar doesn't.
- **Hypothesis:** Late prints / trade reporting that doesn't make it
  into the WS-aggregated cache stream but does appear in the REST 1-sec
  aggregates. Polygon-side data lineage difference, not a bug in our code.
- **Why we tolerate it:** Indicators we care about (EMAs, MACD, RSI,
  stochastic) operate on OHLC. Volume only affects RVOL and explicit
  volume-based triggers. None of the active user packs are
  volume-gated as of 2026-05-27. If a pack starts using volume, revisit.

---

### Ralph WS pipeline drops SPY sub-minute bars during active hours
- **Status:** MITIGATED — root cause not fixed, but impact reduced by `ws_rest_spliced`
- **Discovered:** 2026-05-27 during REST-vs-WS investigation
- **Mitigated:** 2026-05-28 via `ws_rest_spliced` rollout
- **Severity:** High while WS was the sole source; now reduced — `ws_rest_spliced` verifies WS bar closes against REST and splices REST values into indicator history when drift is caught, so indicator state converges to backtest-aligned values even when WS aggregation is faulty.
- **Location:** Ralph's WebSocket bar aggregation pipeline
  (`ralph_engine.py` BarBuilder / on_second_bar)
- **Symptom:** During the 18:00-19:30 UTC window on 2026-05-27, 26% of SPY
  10Sec bars (140/541) were missing from `live_bars`. The missing bars
  had real volume (7K-40K each), so not quiet-period artifacts.
  Confirmed not deploy-related: during the largest gap (18:06:00-18:22:10,
  16 minutes), Ralph was actively writing 22 other symbol/TF combos
  (TSLA, TSLL, SPY 120s/180s, etc).
- **Hypothesis:** Polygon per-ticker WS subscription stuck after a
  reconnect, OR race condition in BarBuilder for SPY's high tick rate,
  OR Polygon A-channel throttling on SPY specifically.
- **Why not fully fixed:** Pure REST polling proved too slow on RTH probe
  (2026-05-28, p90 8-23s) to drive sub-minute alerts directly. Chosen
  direction was the hybrid `ws_rest_spliced` (see
  `/home/kevin/.claude/plans/breezy-dreaming-umbrella.md` and memory
  `project_ws_rest_spliced_canary.md`). The hybrid surfaces drift
  events as `verification_status='corrected'` or `'drift_uncorrected'`
  on the alerts table, so the WS coverage issue is now observable +
  partially compensated rather than silent. On sub-minute TFs, the
  `apply_rest_correction` engine path is structurally unable to splice
  REST into a bar that's already 1-2 bars stale, so REAL drift on
  sub-minute strategies still shows up as `drift_uncorrected`. Tracking
  rate over a week to decide whether per-second splice escalation
  (`project_per_second_splice_idea`) is warranted.

---

### Sub-minute drift_uncorrected_cascade: live engine entirely misses multi-trade clusters
- **Status:** Known, partial mitigation in place via `ws_rest_spliced`; structural fix pending (per-second splice)
- **Discovered:** 2026-05-28 during M8 fleet observation + divergence backlog walkthrough
- **Severity:** High on sub-minute (10Sec) strategies — drives 34-56% phantom rate per strategy
- **Location:** Interaction between `unified_engine` indicator state (cumulative) + `apply_rest_correction` (latest-bar only)
- **Symptom:** When a sub-minute bar's WS close differs from REST close by ≥$0.01 and the bar correction is rejected (`drift_uncorrected` — common on 10Sec because REST settles 6-15s after bar close, by which time 1-2 newer bars are in history), the indicator state on that strategy is offset from REST-canonical. Subsequent trigger evaluations diverge between live and backtest for many bars after, EVEN ON BARS WHERE THE LIVE ALERT VERIFIES AT Δ=0.0. The cumulative state — not just the bar value — is what's offset.
- **Concrete example (sid 170, 2026-05-28 20:14-20:19Z):** backtest produced 5 trades in 5 minutes with 1-13 bar holds; live fired ONE entry alert in the same window. All 5 backtest trades show up as `missed` in the divergence backlog. Bar verification on the 1 live alert was Δ=0.0.
- **Why it's not a regression:** Pre-`ws_rest_spliced` (running `ws_agg_reconciled`), the same phantom pattern existed — there was no verifier surfacing the drift events, so the cascade was silent. `ws_rest_spliced` makes the cause visible (the `drift_uncorrected` stamps) but the engine still can't catch up to backtest.
- **Why current mitigation is partial:** `apply_rest_correction` only accepts splices on the LATEST bar in BarBuilder history. On 10Sec TF this almost never lands because REST settles after 1-2 newer bars have arrived.
- **Structural fix path:** Per-second REST splice — retain per-second bar history, splice REST 1-sec values when they settle (anywhere from 2s to 15min), re-aggregate the bar from corrected per-second data, replay indicator state forward from that bar. See `project_per_second_splice_idea` memory. Replay-from-N indicator path is the largest engineering lift; estimated 1-2 focused sessions.
- **Workaround pre-fix:** Either accept the cascade rate on sub-minute strategies, OR migrate sub-minute strategies to 1Min+ TFs where REST settle (~10s) is well within bar duration (60s+).

---

### Supabase 522 timeout 2026-05-28 ~21:26Z — origin connection
- **Status:** Resolved by Supabase project restart (~21:32Z)
- **Discovered:** 2026-05-28 during divergence investigation
- **Severity:** Medium — blocked all DB writes for ~6 min; alerts queued and eventually drained
- **Symptom:** Cloudflare returned 522 ("Connection timed out") for all Supabase REST calls. Worker writes failed; investigation queries failed.
- **Suspected cause:** Unknown. Possible candidates: (a) Supabase project-level issue (Cloudflare layer between client and Postgres), (b) Postgres overload, (c) coincidence. The bulk strategy config UPDATE I ran at M8 rollout (~18:40Z, 39 strategies) was 3 hours earlier so unlikely contributor.
- **What to watch for next time:** Worker log error patterns during the outage window (correlate with alert dispatch failures), Supabase project health dashboard.

---

## Closed

(none yet)
