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

## Next steps after this doc fills in

Phase 2 (sids 135–144 strategy-level Run Parity sweep) → expand this doc with a per-strategy table:

```
strategy | mirror_sid | verdict | most_common_failing_gate | underlying_packs | suspected_root_cause
```
