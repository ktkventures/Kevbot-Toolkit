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

## Closed

(none yet)
