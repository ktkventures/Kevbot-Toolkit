# Phase H — Trade-Channel Bar Findings (EOD 2026-05-15)

**Status (updated 2026-05-18):** CONCLUDED — Option C. Trade-channel self-built bars are shelved. Monday RTH diagnosis found an 86-second ingestion lag that makes self-built T-channel bars unviable for live alert gating. Separately, the parity tool showed the existing decision-time cache already matches a correct REST baseline at 93-100% — the problem Phase H set out to solve is, in practice, near-zero. See [Phase H Conclusion](#phase-h-conclusion-2026-05-18).

---

## Phase H Conclusion (2026-05-18)

### What Monday's diagnosis found
Trade-channel close-match vs REST 1Min measured **2-3%** during Monday RTH (SPY/TSLA) — far worse than Friday's 48% vs REST 1Sec. Instrumentation (`trade_bar_builder.py`, commit `d70fdf2`) gave the verdict:

- **WebSocket delivery is fine** — 43,354 T events ingested in the sample window, zero parse failures, zero wrong-symbol.
- **86-second average ingestion lag** (range 32-151s). The worker's single asyncio event loop runs WS-read + the full alert engine + synchronous DB writes; it cannot also drain the raw T firehose in real time.
- The builder closes buckets on **wall-clock** time. With trades arriving ~86s after their SIP timestamp, every bucket closes ~86s before its trades arrive → **60%+ of trades dropped as `_dropped_late`**, bars built from a sparse early-minute trickle (which is why bar *opens* matched REST but nothing else did).
- Secondary: the `size < 1` filter drops **25% (SPY) / 53% (TSLA)** of trades — fractional-share prints, a large unintended volume sink.

### Why Phase H's premise is falsified
Phase H bet that self-built T-channel bars would be **fresher** than Polygon's AM/A rebroadcast (the 20-100s rebroadcast lag we were trying to escape). But our own ingestion lag (**86s**) *exceeds* that rebroadcast lag. Self-built T-channel bars cannot gate live alerts faster or more accurately than the AM/A first-emission cache on the current architecture. The 0/200/500 ms wait variants tuned milliseconds against a problem measured in minutes.

### The encouraging counter-finding
While diagnosing, the parity tool's Bars Comparison gave the result that actually matters: the **decision-time cache** (`live_bars.first_*` = Polygon's first WS emission, what the engine gated on) matched a *correct* REST baseline at:

- **93.1%** vs REST 1Sec (image copy 74)
- **100%** vs REST 1Sec (image copy 77)
- **100%** vs REST 1Min (image copy 78)

The ~16% first-vs-settled correction rate Phase G feared is, across the windows observed, frequently **0%**. The decision-time cache is already tracking settled reality.

### Decision: Option C
**Shelve the trade-channel self-built-bar approach.** The decision-time cache (AM/A first emission) remains the production alert source — Phase H never replaced it; the trade builder was always a shadow. No production change is required to "conclude" Phase H.

Recommended cleanup:
- **Disable the shadow on the worker**: `POLYGON_TRADE_SHADOW_ENABLED=false`. Beyond saving resources, the `T.SPY`/`T.TSLA` subscription is *itself adding load* to the overloaded event loop — disabling it may slightly *reduce* the production engine's own latency. (Railway env-var change — apply manually.)
- `trade_bar_builder.py`, the `live_bars_trades` table, and the `/trade-bars` endpoint can stay in place (disabled) as a record, or be removed in a later cleanup.
- Instrumentation commit `d70fdf2` is harmless; it can stay.

### Open caveats — validate before fully trusting
1. **Stress-test the 100% match** across more windows / days / strategies — a single clean window is not proof. Confirm it is not a tool artifact (e.g., both panes accidentally sourcing the same series, or TF-misalignment hiding mismatches).
2. **Verify backtest ↔ cache decision-time parity.** If the backtest engine also lines up with the decision-time cache, that closes the loop on backtest/live fidelity — the real prize.
3. **REST 1Min UI source is buggy** — shows ~$480 (≈ SPY's Jan-2024 price; suspected date-window bug) and partially-loading charts (TF-alignment: REST 1Min is fixed 60s while REST 1Sec is "any TF"). Frontend-side; the backend `/rest-bars` minute path is verified correct. Needs the frontend repo to fix.

### Superseded
The "Monday plan" and "Iteration cycle" sections below are superseded by this conclusion. The 20-min hypothesis-iteration cadence was abandoned once the 86s lag was found — it was a single-cause architectural problem, not a tuning problem.

---

## Quick navigation

- [What was built](#what-was-built)
- [Pipeline health snapshot](#pipeline-health-snapshot)
- [Diagnostic findings](#diagnostic-findings)
- [Root-cause hypotheses](#root-cause-hypotheses)
- [Open questions](#open-questions)
- [Monday plan](#monday-plan)
- [Reference: critical files & commits](#reference-critical-files--commits)

---

## What was built

### Phase H.1 — Shadow trade-channel bar builder
Polygon's AM/A WebSocket channels emit pre-aggregated bars where the first emission disagrees with REST-settled bars ~16-20% of the time at bar close (Phase G measurement) — because Polygon rebroadcasts corrected bars 20-100s later, too late to gate alerts on. H.1's hypothesis: build bars from the T (individual trades) channel ourselves, applying the same canonical Polygon condition filter that `flat_file_ingestion.py` uses for observable bars, with a configurable wait window past period end to capture in-flight late prints.

Three wait-time variants (0ms / 200ms / 500ms) run simultaneously per (symbol, tf) so we pick the best latency-vs-accuracy tradeoff empirically. Outputs land in a new `live_bars_trades` table — production alert path is unchanged.

**Configuration (env-gated, defaults shown):**
- `POLYGON_TRADE_SHADOW_ENABLED=true`
- `POLYGON_TRADE_SYMBOLS=SPY,TSLA`
- `POLYGON_TRADE_TFS_SECONDS=10,60`
- `POLYGON_TRADE_WAIT_VARIANTS=0,200,500`

**Commits in order:**
- `f915d78` Phase H.1: shadow trade-channel bar builder
- `d6e97da` Phase H.2: three trade-channel sources in Bars Comparison UI
- `0a5167e` Phase H.1 fix: Polygon WS T `t` field is **milliseconds**, not nanoseconds (flat-file CSV uses ns; live WS uses ms — distinct sources, distinct units)
- `0c639c0` Phase H.1 hotfix: ETH vol-only OHLC fallback (Form T-flagged trades during ETH would otherwise drop every bar silently)

**Schema** (`src/migrations/live_bars_trades.sql`):
PK is `(symbol, timeframe_seconds, bar_start, source)`. `trade_count` field = OHLC-eligible trades that contributed. `filtered_trades` = canonical-filter-dropped trades that arrived in the bucket window. **Built-in marker:** `trade_count == 0 AND volume > 0` flags a bar that emitted via the ETH vol-only fallback path.

### Phase H.2 — Bars Comparison UI sources
Added three new dropdown options to Admin > Parity > Bars Comparison: `Trade ch. (wait 0ms)`, `(wait 200ms)`, `(wait 500ms)`. Backend endpoint `/api/admin/parity/trade-bars` reads `live_bars_trades` with auto-aggregation-up when the requested TF isn't directly shadowed (e.g., 5Min derived from stored 60s rows). Per-pane lazy fetch — only the variant actually selected is queried.

---

## Pipeline health snapshot

**At EOD 2026-05-15 21:44 UTC:**
- 3,504 rows total in `live_bars_trades`
- All 12 (symbol × source × tf) combinations writing in lockstep
- Latest write 16s old — fully fresh
- ~42 rows/min during ETH = ~3.5/min per combo (matches expected 6 bars/min × 12 combos / 2-symbol scaling)

**RTH coverage:** 19:54-20:00 UTC window only (Phase H went live ~6 min before close). 36-37 10Sec bars per combo, 7 1Min bars per combo. **Significant deploy churn during this window** — multiple redeploys for the ms-fix + ETH-fallback hotfix. Not the cleanest data.

**ETH coverage:** 20:00 UTC onward, continuing through Friday evening. ~430+ 10Sec bars per combo, 80+ 1Min bars per combo. Plenty of data, but qualitatively different from RTH for reasons covered below.

---

## Diagnostic findings

### Finding 1: Close-match against REST 1Sec is ~48% RTH, ~72% ETH

| Source | Window | Bars | REST present | REST null | Close-match (\|Δ\|≤$0.01) |
|---|---|---|---|---|---|
| t_wait0 SPY 10s | RTH 19:54-20:00 | 35 | 35 | 0 | **17/35 = 48.6%** |
| t_wait200 SPY 10s | RTH 19:54-20:00 | 36 | 36 | 0 | **17/36 = 47.2%** |
| t_wait500 SPY 10s | RTH 19:54-20:00 | 36 | 36 | 0 | **17/36 = 47.2%** |
| t_wait0 SPY 10s | last hour (ETH) | 322 | 200 | 122 | **144/200 = 72.0%** |

All three RTH variants showed **identical largest diffs**: $0.285, -$0.265, -$0.15, -$0.14, -$0.10. The wait window changes which trades land in volume/trade_count totals but does **not** change close price (late prints arrive at-or-near current price, so they don't move the close).

### Finding 2: Wait variants do capture more trades (as designed)
At SPY 10Sec bucket `19:56:40 UTC`:
- t_wait0: vol=320,393, n=518 trades
- t_wait200: vol=325,423, n=536 trades **(+18 trades, +5K vol)**
- t_wait500: vol=325,423, n=536 trades (identical to wait200 — diminishing returns)

So the wait window is doing its volume-completeness job. It just isn't moving close prices.

### Finding 3: REST nulls during ETH are real, not a UI bug
Polygon's 1-sec aggregates endpoint emits no row for seconds where the consolidated tape's OHLC-eligible trades = 0. During ETH, many 10s buckets have zero consolidated OHLC trades. Our trade-bar still emits via the vol-only fallback (capturing Form T prints). Result: trade-bar present, REST absent, KPI denominator shrinks. **Match-KPI is computed only over bars where both sides have data** — this is correct behavior, but means the headline % reflects fewer bars than the bar count suggests.

### Finding 4: ETH vol-only fallback admits outlier prints
6 bars in the last-hour ETH window had high-low ranges >$1 (one was $21.70 with high=$759.35 vs the actual tape near $737). All had `trade_count=0` (= vol-only fallback bars). These are individual Form T or odd-lot prints that print far off the bid/ask during thin-liquidity ETH periods. The canonical filter would have excluded them during RTH; the ETH fallback currently does not.

### Finding 5: Trade-channel raw is potentially WORSE than AM/A first-emission for close-match
Phase G measured cache (AM/A first-emission) ≈ **84%** close-match vs REST. Our trade-channel measures **~48%** close-match vs REST during RTH. **Going from pre-aggregated to raw made close-match worse, not better.**

This is the most important finding of the day. Phase H's hypothesis was that T-channel would be better because it avoids Polygon's late rebroadcast cycle. But Polygon's server-side aggregator applies settlement rules that our naive "last T event wins" implementation does not replicate. Going to raw gave us latency but lost us aggregation correctness.

---

## Root-cause hypotheses

### Primary hypothesis: Polygon's REST aggregator uses different price-selection rules than naive last-trade
Our `_close_bucket` writes `close = last trade observed in bucket`. Polygon's REST 1Sec aggregator may:
- Pick last consolidated trade by exchange priority (e.g., NYSE > ARCA > BATS)
- Preserve the official close print of the second
- Use SIP-tape last-trade rather than participant-tape last-trade
- Apply additional condition exclusions beyond the OHLC/vol split we're already doing

We don't currently know which. **Monday's first 30 min is reading Polygon's aggregator docs and looking for the rule.**

### Secondary hypothesis: Our condition filter and Polygon's REST condition filter differ subtly
`polygon_conditions._load_polygon_condition_rules()` fetches from `/v3/reference/conditions` and applies `updates_high_low=False OR updates_open_close=False` as the OHLC exclusion. But REST may apply *additional* filters at the aggregator level (e.g., volume thresholds, exchange-specific exclusions).

### Tertiary hypothesis: Timing-window mismatch
Polygon's 1Sec aggregates use the trade's SIP timestamp for bucketing. We do too (`sip_ms // 1000`). But subtle differences: do we use the START of the bucket as the timestamp, or the END? Off-by-one between us and Polygon would shift every comparison by 1 second.

---

## Open questions

1. **What price-selection rule does Polygon's REST 1Sec aggregator use for close?** (#1 priority Monday)
2. **Does Polygon's 1Sec aggregator apply any condition filters beyond the OHLC-update flags?**
3. **What ETH condition codes should be INCLUDED in our vol-only fallback OHLC?** Currently we accept any vol-eligible trade. Code 37 (odd lot) at thin ETH liquidity is the main culprit for outlier OHLC.
4. **Is there a "consolidated" vs "participant" tape distinction in T events that we're not handling?**
5. **Should we abandon the wait-variant approach if wait doesn't move close?** Or repurpose it as volume-completeness tooling?
6. **Are the RTH measurements contaminated by deploy churn?** Window was only 6 min and we redeployed twice during it. May want to discard and rely on Monday RTH for the real baseline.

---

## Monday plan

### Setup (before market open)
1. Confirm Friday's flat-file ingestion ran overnight → `polygon_observable_bars` has Friday's SPY+TSLA data.
2. Pull a Monday-morning sanity dump: trade-channel vs observable for Friday's RTH portion. If they don't match closely (≥95%), there's a bug in one of them — fix before iterating.
3. Add `engine_version` or `engine_sha` to status reporting so each iteration cycle is traceable to a commit.
4. Build `/api/admin/parity/match-summary` endpoint + refactor `ParityBarComparison.tsx` to consume it (single source of truth for match-KPI computation; CLI script then becomes a thin wrapper).

### Iteration cycle (during RTH 13:30-20:00 UTC)
**Cadence:** 20 min per iteration. Each iteration:
1. Pick ONE hypothesis (e.g., "use consolidated tape only", "exclude code 37 from OHLC track", "use bid-side price for close").
2. Implement, commit, push.
3. Wait ~3 min for Railway redeploy.
4. Collect 15-17 min of data with that variant live.
5. Run the CLI parity summary: close-match% trade-channel vs REST 1Sec **and** vs observable (when day-N data is available — for Monday this is Friday's observable only, not Monday's same-day; same-day observable lands Tuesday).
6. Look at 10Sec **and** 1Min charts to see directional change.
7. Decide: keep, revert, try variant.

### Success criteria for Monday EOD
- **Primary:** trade-channel close-match ≥95% vs REST 1Sec during Monday RTH
- **Spot-check:** trade-channel close-match ≥95% vs Friday's observable on Friday RTH segment
- **Secondary:** identify which wait variant (if any) provides meaningful close-price improvement; if none, document and choose one as default

### Priorities ordered
1. **Read Polygon's aggregator docs** for price-selection rule (~30 min, may resolve close-match in one shot)
2. **Verify observable ≈ trade-channel on Friday data** — sanity check our own filter consistency
3. **Iterate on price-selection logic** — test 3-5 hypotheses in 20-min cycles
4. **Refine ETH fallback (Phase H.4)** — exclude code 37 odd-lot, possibly codes 13/32/33; only after RTH close-match is in good shape
5. **Decide on wait-variant default** — based on volume-completeness, since close doesn't move

### Out of scope for Monday
- Cutover to alert production path (Phase H.3) — needs ≥95% match first
- Crypto trade-channel support — only stocks for now
- Latency optimization — current ~17s lag is fine

---

## Reference: critical files & commits

### Files (current state)
- `src/trade_bar_builder.py` — manager + per-variant builders + writer pool
- `src/migrations/live_bars_trades.sql` — schema (PK includes `source`)
- `src/ralph_engine.py` lines ~2680 (manager init), ~3098 (T.* subscription), ~3162 (T-event router), ~3567 (periodic flush)
- `src/api/routers/admin_parity.py` lines ~388+ (`/trade-bars` endpoint)
- `src/polygon_conditions.py` — shared canonical filter (`_classify_eligibility`)
- `frontend/src/charts/ParityBarComparison.tsx` — UI consumer with three new sources
- `frontend/src/hooks/queries/useAdminParity.ts` — `useTradeBars` hook

### Commits (chronological)
| SHA | Description |
|---|---|
| `f915d78` | Phase H.1: shadow trade-channel bar builder |
| `d6e97da` | Phase H.2: three trade-channel sources in Bars Comparison |
| `0a5167e` | Phase H.1 fix: Polygon WS `t` is ms not ns |
| `0c639c0` | Phase H.1 hotfix: ETH vol-only OHLC fallback |

### Backup branches
- `dev-backup-pre-phaseH-2026-05-15` — snapshot of `dev` immediately before Phase H started

### Rollback
```bash
# Disable shadow entirely (worker auto-redeploys, no code change needed)
railway variables --service Worker --set POLYGON_TRADE_SHADOW_ENABLED=false

# Or revert all H commits at once
git checkout dev-backup-pre-phaseH-2026-05-15 -- src/trade_bar_builder.py src/migrations/live_bars_trades.sql
git checkout dev-backup-pre-phaseH-2026-05-15 -- src/ralph_engine.py src/api/routers/admin_parity.py
git checkout dev-backup-pre-phaseH-2026-05-15 -- frontend/src/charts/ParityBarComparison.tsx frontend/src/hooks/queries/useAdminParity.ts
```
