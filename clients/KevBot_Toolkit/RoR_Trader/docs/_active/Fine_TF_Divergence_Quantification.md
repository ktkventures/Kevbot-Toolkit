# Fine-TF Bar-Construction Divergence Quantification (sids 327/328/329/333/340)

**Date:** 2026-07-11 (analysis over the last 10 trading days ending 2026-07-10)
**Weekend Sprint Item 2** — Plan_MRS2_P2_Rollout.md Phase 2.2
**Question:** does the WAY each lane constructs the fine-TF (2m–15m) gate bars — WS-lane
aggregation vs backtest resample-from-primary — diverge enough from canonical to explain
these strategies' 22–63% combined paired-%?

**Answer (headline): NO.** Both lanes' fine-TF *bar construction* sits within ~1% of
canonical on prices, and the gate-state flips that induces are **0.0–1.2% of bars** —
nowhere near enough to explain 22–63% combined. The divergence must live elsewhere
(evaluation timing, gate staleness, forming-bar evaluation, trigger-lane 30Sec
divergence), not in fine-TF gate-bar construction.

---

## 1. Methodology

- **Symbol/session/window:** TSLA, RTH (13:30–20:00Z), 10 trading days:
  06-26, 06-29, 06-30, 07-01, 07-02, 07-06, 07-07, 07-08, 07-09, 07-10 (07-03 holiday).
  All data settled (run on a Saturday).
- **Canonical yardstick:** `resampled_bar_cache` (session='RTH'), tf_seconds ∈
  {120, 180, 300, 900} — proven byte-identical to `resample(session-filtered REST 1Min)`.
  Bar counts per day: 195/130/78/26 (2/3/5/15Min) — full coverage all 10 days.
- **WS-lane construction (primary method, per task):** `live_bars` rows,
  `timeframe_seconds=60`, sources = `ENGINE_CONSUMED_SOURCES`
  (`ws`,`ws_agg`,`rest_correction`,`rest_insert`,`warmup_seed`; excludes cosmetic
  `rest_backfill`) → aggregated with the real `data_loader.resample_to_timeframe`
  (left-labeled bins anchored at midnight UTC — bin-identical to canonical since
  13:30Z is aligned for all four TFs). Row mix actually used: **ws_agg 3768,
  rest_insert 76, warmup_seed 50** (3,894 of 3,900 possible 1Min rows; the 6 missing
  minutes exist only on the REST side). Two views computed: `latest`
  (post-correction columns) and `first_*` (decision-time columns).
- **WS-lane construction (bonus, "what the engine consumed"):** direct `live_bars`
  fine-TF rows (tf_seconds 120/180/300/900, engine-consumed sources) — the persisted
  fan-out output. Full 10-day coverage for 2Min (1950), 3Min (1300), 15Min (260);
  **5Min only partial (409/780 rows; ws_agg persistence started 07-07)** — its stats
  are a biased sample and marked as such.
- **Backtest construction:** for sub-minute primaries the backtest resamples fine TFs
  **from the 30Sec primary** (`services.prepare_data_with_indicators` →
  `resample_to_timeframe(primary_df)`; the store/1Min injection only applies to
  ≥1Hour "coarse" TFs, `COARSE_SECONDARY_SECONDS=3600`). Emulated with
  `load_market_data('TSLA', timeframe='30Sec', session='RTH', no_backfill=True)`
  (read-only bar_cache) for 07-08/07-09/07-10 (3 representative days per plan), then
  `resample_to_timeframe` → 4 TFs. For sid 340 (1Min primary) backtest construction
  ≡ canonical *by definition* (same resample of the same REST 1Min).
- **Gate-state probe:** the REAL pipeline — `pack_registry.scan_and_load_all()` +
  Kevin's enabled confluence groups (fetched from `confluence_groups` for the
  sids' user) + `run_all_indicators` / `run_indicators_for_group` /
  `run_all_interpreters` — run identically on canonical vs each WS-constructed frame
  over the concatenated 10-day RTH series; compared per bar: (a) interpreter state
  string differs, (b) the sid's specific gate-open boolean (state == target) flips.
- Comparison tolerance 1e-9 (i.e., exact for prices). Price cells = n_common × 4 (OHLC).
  Volume reported separately (WS vs REST volume divergence is structural).

Gate map verified from `strategies.config.confluence`:

| sid | primary | gates |
|-----|---------|-------|
| 327 | 30Sec | `2m-SWING_123-NEUTRAL`, `5m-SWING_123-BULL_C2` |
| 328 | 30Sec | `5m-SWING_123-BULL_C2` |
| 329 | 30Sec | `1M-SWING_123-BULL_C2`, `5m-SWING_123-BULL_C2` |
| 333 | 30Sec | `3m-STRAT_ASSISTANT-TWO_DOWN`, `15m-UT_BOT_V4-BEAR_TREND` |
| 340 | 1Min  | `5m-BOLLINGER_BANDS-SQUEEZE_MID`, `3m-VWAP_V2->+1σ` |

All five: live_model `ws_rest_spliced`, algo `cache_locked`, backtest `rest_hifi`, RTH.

---

## 2. WS-lane vs canonical (bar level, 10 days)

**Bar sets are IDENTICAL** — 0 WS-only and 0 canonical-only bars, every TF, every day
(the 1Min gaps in the WS lane are already healed by rest_insert/warmup_seed rows the
engine consumed).

### resample(live_bars 1Min) vs canonical — the task's construction emulation

| TF | n bars | price diff cells (latest) | price diff rate | diff magnitude p50 / p90 / max | close-col diffs | volume diff rate |
|------|------|-----|--------|--------------------------|----|--------|
| 2Min | 1950 | 52/7800 | **0.67%** | $0.06 / $0.25 / $0.37 | 9 | 98.3% |
| 3Min | 1300 | 31/5200 | **0.60%** | $0.05 / $0.25 / $0.67 | 6 | 98.7% |
| 5Min | 780  | 17/3120 | **0.54%** | $0.04 / $0.42 / $0.97 | **0** | 99.7% |
| 15Min| 260  | 4/1040  | **0.38%** | $0.24 / $0.55 / $0.67 | **0** | 100% |
| (1Min base) | 3894 | 92/15576 | 0.59% | $0.07 / – / $0.97 | 18 | 96.9% |

Decision-time (`first_*`) view is marginally worse: 0.64–0.71% per TF.
Diffs concentrate in **opens** (a late/early first-tick difference between WS and REST),
then highs/lows; **closes almost never differ** — 5Min and 15Min had *zero* close
diffs in 10 days. Volume differs on ~97–100% of bars (structural WS vs REST; max
single-bar gap 1.5M shares at 15Min).

### Direct live_bars fine-TF rows (persisted fan-out = closest to what the engine consumed)

| TF | coverage | price diff rate | magnitude p50 / p90 / max | volume diff rate |
|------|--------------|-----------|---------------------|------|
| 2Min | 1950/1950 | **1.03%** | $0.14 / $0.43 / $1.26 | 98.2% |
| 3Min | 1300/1300 | **1.33%** | $0.15 / $0.74 / $4.28 | 98.3% |
| 5Min | 409/780 (partial, mixed sources — biased) | 15.2%* | $0.38 / $1.21 / $3.42 | 66.3% |
| 15Min| 260/260 | **1.35%** | $0.19 / $1.46 / $2.15 | 98.1% |

\* 5Min direct-row stats are NOT comparable: persistence only started 07-07 and 45% of
the rows are warmup_seed/rest_correction/rest_insert — treat as anecdote, not measurement.

**Observation:** the engine's actual fan-out bars (built live from the WS stream)
diverge **~2× more** than a clean resample of the WS-lane 1Min history would
(1.0–1.4% vs 0.4–0.7% of price cells) — the fan-out path itself adds construction
noise beyond the WS-vs-REST feed difference.

---

## 3. Backtest construction vs canonical (resample REST 30Sec → fine TFs, 3 days)

| TF | days | price diff cells | volume diff rate | max vol diff |
|------|-----------------|-------|--------|---------|
| 2Min | 07-08/09/10 | **0 / 0 / 0** | 87–91% | ~100k sh |
| 3Min | 07-08/09/10 | **0 / 0 / 0** | 89–94% | ~100k sh |
| 5Min | 07-08/09/10 | **0 / 0 / 0** | 92–99% | ~100k sh |
| 15Min| 07-08/09/10 | **0 / 0 / 0** | 100%   | ~98k sh |

**resample(REST 30Sec) is PRICE-PERFECT vs canonical** — zero OHLC diff cells in all
12 (tf, day) combos. Bar sets identical. The backtest lane's fine-TF gate **prices**
are indistinguishable from canonical for these 30Sec-primary strategies.

**BUT volume differs on ~90% of bars** (Polygon 30Sec bars do not sum to Polygon 1Min
volume; up to ~100k sh/bar on 07-10). Price-based gates are untouched; a
**volume-dependent gate (VWAP_V2, RVOL) on a sub-minute-primary strategy would be
structurally divergent in backtest** — none of these five sids has that combination
(340's VWAP gate rides a 1Min primary = canonical-exact), but it's a lurking trap for
future strategies.

---

## 4. Gate-state impact (the money number)

Real interpreters (built-in SWING_123 + user packs STRAT_ASSISTANT, UT_BOT_V4,
BOLLINGER_BANDS, VWAP_V2 via pack registry), identical pipeline both sides, 10 days.

### Per-sid gate-open flip rate (canonical vs WS-lane construction)

| sid | gate | vs resample(live 1Min) latest | vs decision-time (first) | vs direct fan-out rows |
|-----|------|------|------|------|
| 327 | 2m-SWING_123-NEUTRAL | **0.15%** (3/1950) | 0.21% | 0.72% (14) |
| 327 | 5m-SWING_123-BULL_C2 | **0.00%** (0/780) | 0.00% | n/a (partial) |
| 328 | 5m-SWING_123-BULL_C2 | **0.00%** | 0.00% | n/a |
| 329 | 1M-SWING_123-BULL_C2 | **0.18%** (7/3894) | – | – |
| 329 | 5m-SWING_123-BULL_C2 | **0.00%** | 0.00% | n/a |
| 333 | 3m-STRAT_ASSISTANT-TWO_DOWN | **0.15%** (2/1300) | 0.23% | 0.38% (5) |
| 333 | 15m-UT_BOT_V4-BEAR_TREND | **0.00%** (0/260) | 0.38% | 0.38% (1) |
| 340 | 5m-BOLLINGER_BANDS-SQUEEZE_MID | **0.00%** (0/780) | 0.13% | n/a |
| 340 | 3m-VWAP_V2->+1σ | **0.92%** (12/1300) | 0.92% | 1.15% (15) |

Full state-string diff rates (any state change, not just the gate's target) are the
same order: worst is VWAP_V2 at 1.8–2.5% (volume-driven, expected), SWING_123 at
0.0–1.0%, everything else ≤0.5%. Interpreter warmup (ATR/BB/VWAP cumulative state)
was seeded identically on both sides from the same 10-day RTH series, so the
comparison is internally fair; absolute states near day 1 may differ from production
(which warms deeper) — flips are if anything *overstated* here because a single bar
diff can propagate through cumulative indicators.

### Backtest side

Price-perfection (§3) makes SWING_123 / STRAT_ASSISTANT / UT_BOT_V4 /
BOLLINGER_BANDS states on backtest-constructed fine TFs **identical to canonical by
construction** (all are price-only functions of OHLC). VWAP_V2 would diverge via the
30Sec-volume mismatch, but the only VWAP gate (sid 340) sits on a 1Min primary →
exact. **Backtest fine-TF gate-state divergence vs canonical: 0.0% for every gate on
every one of the five sids.**

---

## 5. Conclusions per sid

Combined-lane ceiling = (1 − live flip) × (1 − backtest flip) per gate, all gates.

| sid | combined % | bar-construction ceiling on gate agreement | verdict |
|-----|-----------|--------------------------------|---------|
| 327 | ~22–63% band | ≥99.6% (2m 0.15–0.72%, 5m 0.0%) | **construction explains ~0.4% at most — NOT the cause** |
| 328 | " | ≥99.9% (5m 0.0%) | **construction explains nothing** |
| 329 | " | ≥99.8% (1M 0.18%, 5m 0.0%) | **NOT the cause** |
| 333 | " | ≥99.2% (3m 0.15–0.38%, 15m 0.0–0.38%) | **NOT the cause** |
| 340 | " | ≥98.8% (5m 0.0–0.13%, 3m VWAP 0.92–1.15%) | **NOT the cause** (VWAP volume-noise is the worst single gate, still ~1%) |

**The hypothesis is falsified for these five strategies.** Both lanes construct fine-TF
gate bars that agree with canonical on ≥98.8% of gate states (mostly ≥99.6%). To lose
37–78% of pairs, the divergence must be in **when/what the lanes evaluate**, not what
the bars are:
- forming-bar / just-closed-bar evaluation timing on the live side (the 1-period
  cross-TF shift, gate evaluated mid-bar vs on settled bar);
- gate staleness / fossilized-state classes (cf. the 325/338 archetypes);
- live fan-out gaps at decision time (bars that arrived late or not at all when the
  trigger fired — the settled cache heals them afterward, so this analysis can't see them);
- the **trigger lane itself** (30Sec primary WS vs REST divergence — untested here,
  and sub-minute WS bars are far noisier than 2m+ aggregates).

## 6. Honest approximations / caveats

1. This measures **settled** bar construction. Real-time gate evaluation sees forming
   bars and late-arriving 1Min corrections; the `first_*` decision-time view partially
   captures that (0.64–0.71% price cells vs 0.54–0.67% settled) but bars that were
   *absent* at decision time are invisible here.
2. The gate-state probe warms indicators from the 10-day RTH series only (both sides
   identically), not production-depth warmup; cumulative interpreters (UT_BOT ATR,
   BB, VWAP) may sit in different absolute states early in the window. Flips are, if
   anything, overstated.
3. Backtest leg used 3 days (07-08/09/10) per plan; bar_cache 30Sec is the
   hifi-settled series as it exists today, not necessarily the exact bytes of any
   historical recompute.
4. 5Min direct fan-out rows are a partial, source-biased sample (persistence began
   07-07) — excluded from the gate-state probe.
5. sid 329's 1M gate compared live_bars 1Min vs REST 1Min directly (native 1Min is
   the canonical for a 1Min gate; `resampled_bar_cache` has no 60s tier).
6. Cross-TF gate evaluation applies a 1-period shift in both lanes; comparing state
   series per bar is shift-invariant, so the shift doesn't affect these rates.

## 7. Reproduction

Scratch scripts (session scratchpad, not committed):
`leg1_ws_vs_canon.py` (WS agg + direct rows), `leg2_bt_vs_canon.py` (30Sec resample),
`leg3_gatestate3.py` (real-pipeline gate states), `leg4_1min.py` (1Min/sid-329 gate).
All DB access read-only, day-chunked; groups loaded via direct PG select (PostgREST
`maybe_single()` 204s were biting; the empty-groups default-save path was bypassed).
