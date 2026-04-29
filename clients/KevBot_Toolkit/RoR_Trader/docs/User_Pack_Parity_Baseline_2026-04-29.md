# User Pack Parity Baseline — 2026-04-29

Purpose: establish a trusted baseline of which user packs are batch↔live
parity-clean before drilling into strategy-level parity failures. Per
`/home/kevin/.claude/plans/eager-wiggling-twilight.md` Phase 1, the 10
Tier-A mirror strategies' parity scores are unreliable until we know
the underlying packs themselves don't have batch/live drift.

## Tested packs

The 8 v2 packs the Tier-A mirrors actually use, plus 2 packs (`swing_123`,
`ut_bot_v4`) that previously had FAIL status to verify post-fix.

| pack | TF | session | days | secondary_tf | warmup_bars |
|---|---|---|---|---|---|
| (all) | 1Min | RTH | 7 | 15Min | 200 |

Default 4Q config from `parity_simulator.run_pack_parity_test_4q`.

## Verdict matrix — RUN COMPLETE 2026-04-29 ~10:50 UTC

| pack | overall | Q1 | Q2 | Q3 | Q4 | duration | notes |
|---|---|---|---|---|---|---|---|
| macd_line_v2 | **PASS** | PASS | PASS | PASS | SKIP | 43s | clean |
| macd_histogram_v2 | **PASS** | PASS | PASS | PASS | SKIP | 61s | clean |
| vwap_v2 | **PASS** | PASS | PASS | PASS | SKIP | 35s | ⚠ runtime warnings: `'str' object has no attribute 'timestamp'` thrown ~20× per replay during incremental updates. Doesn't affect parity (PASS), but indicates an exception path the live engine swallows. Worth a follow-up. |
| rvol_v2 | **PASS** | PASS | PASS | PASS | SKIP | 41s | clean |
| ema_stack_v2 | **PASS** | PASS | PASS | PASS | SKIP | 39s | clean |
| ema_pp_v3 | **PASS** | PASS | PASS | PASS | SKIP | 38s | clean |
| ema_pp_v4 | **PASS** | PASS | PASS | PASS | SKIP | 40s | clean |
| swing_123_test | **PASS** | PASS | PASS | PASS | SKIP | 40s | clean |
| swing_123 (recheck) | **PASS** | PASS | PASS | PASS | SKIP | 36s | ⭐ previously FAIL Q3; fix landed |
| ut_bot_v4 (recheck) | **PASS** | PASS | PASS | PASS | SKIP | 37s | ⭐ previously FAIL Q2; fix landed |

**All 10 packs PASS Q1+Q2+Q3.** Q4 (data fidelity) is intentionally SKIP across the board — that quadrant is deferred per `parity_simulator.py:1129`.

## Bucketing rules

- **PASS on all 4Q (Q4=SKIP allowed)** → trustworthy. Strategy-level failures involving this pack are NOT pack-caused; look at strategy config or engine.
- **WARN/FAIL on Q2 only** → column-contract gap (the ut_bot_v4 class — `__`-prefix or live-emit miss). Fix in `indicator_incremental.update_bar()` return dict. `pack_spec.validate_column_contract` (file `src/pack_spec.py:556–771`) names the missing columns.
- **FAIL on Q3 only** → cross-TF shadow-engine drift (the original swing_123 class). Inspect `divergent_examples` in the Q3 result.
- **FAIL on Q1** → trigger fires differently in batch vs live. Rare for v2 packs; needs trigger evaluator audit.

## Buckets (post-run)

### Trustworthy (PASS) — all 10 packs
- macd_line_v2, macd_histogram_v2, vwap_v2, rvol_v2, ema_stack_v2, ema_pp_v3, ema_pp_v4, swing_123_test, swing_123, ut_bot_v4

### Fix Q2 / Q3 / Q1 — empty
No pack-level batch↔live drift detected on the default 1Min × 15Min × 7-day config.

## Implications for Phase 2

Strategy-level parity failures (when we run Run Parity on the Tier-A mirrors in Phase 2) are NOT pack-caused — every underlying pack is parity-clean. So failures will trace to one of:

- **Strategy-config drift** (cross-TF setup, stop_config, time_exit_config, missing migration step)
- **Engine-level bug** (parity service timestamp matching, replay window sizing, etc.)
- **Indicator-level design flaw** (lookahead bias, repaints — but these would have surfaced in pack-level Q1/Q2 if the indicator had them)

Most likely: strategy-config drift. Phase 5 routing should weight pack-level fixes lower for now (since none currently fail), and weight strategy-level investigation higher.

## Open follow-ups

1. **vwap_v2 timestamp-handling warnings** — `'str' object has no attribute 'timestamp'` thrown silently inside its incremental update path. Test passed despite this (parity is unaffected) but the live engine is catching exceptions it shouldn't be. Investigate the `_user_pack_vwap_v2` engine instantiation and bar-feeding code in `unified_engine.py` / live shadow dispatch. Not blocking.

2. **Cross-TF coverage extends only to the default 15Min secondary**. The 4Q test fixed `secondary_tf=15Min`. Some mirror strategies use other secondary TFs (1d, 5m, 30s, 10m) — those cross-TF combinations aren't directly tested. The 1Min/15Min PASS implies the shadow-engine fan-out is correct in general, but a hidden TF-specific bug remains theoretically possible. Phase 2 will surface any such gaps.

## Strategy-level cross-reference (used in Phase 2)

After the baseline, every Tier-A mirror's strategy-level parity verdict will be cross-referenced against its underlying packs' bucket. Pattern:

- mirror PASS + pack PASS → fully trustworthy
- mirror FAIL + pack PASS → strategy-level bug (config / engine / matching logic)
- mirror FAIL + pack FAIL → fix the pack first, then re-run mirror parity
- mirror PASS + pack FAIL → may be coincidence (the failing quadrant didn't intersect with the strategy's actual confluence usage). Document but don't act.

## Phase 2 strategy-level results (after Phase 5 fixes)

Two engine-integration bugs were found and fixed during this sweep:

1. **`f68b410`** — `TriggerEvaluator.evaluate_bar_close` user-pack interpreter dispatch was building a single-row DataFrame from `current`, which broke any interpreter that uses `df.shift(1)` (e.g., MACD_HISTOGRAM_V2's H+up/H-dn classification). Fix: build a 2-row `[prev, current]` frame so shift produces a meaningful prev value. Effect: live shadow engines went from emitting only 2 of 4 possible MACD_HISTOGRAM_V2 states to all 4.

2. **`3aae5bc`** — `parity_service._replay_strategy` was reading `sig.get('time')` (which doesn't exist on entry signals) and falling back to bar_start `ts`. `stored_trades.entry_fill_ts` is bar_close per Trade Timestamps Spec. After `ts[:16]` minute truncation, bar_start vs bar_close land in different buckets — so matched_count was 0 by construction. Fix: read `sig.get('entry_fill_ts')` first.

| sid | name | verdict | score | matched/stored | replay_only | notes |
|---|---|---|---|---|---|---|
| 135 | META 1Min | PARTIAL | 0.24 | 65/200 | 73 | both fixes contributed |
| 136 | SPY 1Min | **PARTIAL** | **0.68** | 145/200 | 13 | best result; close to PASS-territory |
| 137 | TSLA 5Min | PARTIAL | 0.18 | 33/141 | 46 | both fixes contributed |
| 138 | TSLA 5Min | FAIL_LIVE_BLOCKED | 0.00 | 0/26 | 0 | live still fires nothing — different root cause |
| 139 | TSLA 10Sec | FAIL_LIVE_BLOCKED | 0.00 | 0/200 | 0 | live still fires nothing (after retry — first run errored on Polygon `Server disconnected`) |
| 140 | SPY 1Min | **PARTIAL** | **0.55** | 139/200 | 52 | strong improvement |
| 141 | TSLL 1Min | FAIL_LIVE_BLOCKED | 0.00 | 0/200 | 0 | live still fires nothing |
| 142 | TSLA 10Sec | NO_TRADES | – | 0/0 | 0 | no stored trades to compare |
| 143 | TSLA 10Sec | FAIL_LIVE_BLOCKED | 0.00 | 0/122 | 0 | live still fires nothing |
| 144 | AMD 1Min | PARTIAL | 0.30 | 35/50 | 65 | matched=35 (was 0 pre-fix) |

**Pattern:** five PARTIAL verdicts with non-zero matches confirm the fixes work end-to-end. **Four strategies remain FAIL_LIVE_BLOCKED with `replay_only=0`** (sids 138, 139, 141, 143) — live engine is still producing zero output. Common factor isn't TF (5Min, 10Sec, 1Min) or symbol (TSLA, TSLL) cleanly, but ALL FOUR use `swing_123_default_bull_c2` as their entry trigger AND have cross-TF confluence gates. That's a distinct bug (likely cross-TF setup specific to swing_123-based mirrors) — separate Phase 5 follow-up.

## Open Phase 5 follow-ups

1. **Three strategies still emit zero live trades** (sids 138, 141, 143). All have `replay_only=0` after the fixes. Common factor isn't TF (5Min, 1Min, 10Sec) or symbol (TSLA, TSLL). Need a probe targeted at one of these to see what's blocking the trigger or confluence.

2. **PARTIAL → PASS gap.** Even on the best result (sid 136 at 68%), there's still 55 stored_only + 13 replay_only that don't match. Possible explanations:
   - Cross-TF buffer's "warmup gap" — early replay bars don't have a populated 1Day shadow record because the shadow consumed those bars during its own warmup. Backtest's pre-computed cross-TF column uses ffill from before, so the first primary bars get a value.
   - Race condition where shadow updates lag the primary by one bar at TF boundaries.
   - Position state machine differences (e.g., bar_count cooldown).
   - Some indicators converge to slightly different values between batch (full df pass) vs incremental (bar-by-bar).

3. **4Q simulator's default config (1Min × 15Min × 7 days) was insufficient** to surface the dispatch bug fixed in `f68b410`. macd_histogram_v2 PASSED Q3 on those defaults despite the bug. Need to either raise the default test scope OR add a per-pack regression test that explicitly exercises rising/falling-state interpreters across longer windows.

4. **`_kick_stuck_parities.py` startup-recovery hook** (`a8923b7`) marks PENDING > 10 min stale as ERROR. This protected against deploy-killed threads. The 30-min timeout in `_rerun_all_parities.py` was also too short for sub-minute strategies' large data windows — bumped to 2 hours. Consider auto-extending based on strategy TF.
