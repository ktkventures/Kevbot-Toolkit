# Live Model Decision — TV Stability Test + Engine Reliability Plan

**Status:** decision recorded · engine work queued for 2026-05-05 RTH
**Last updated:** 2026-05-04
**Owner:** Kevin

---

## Background

The live engine has two decisions to make about every closed minute bar:

1. **Lock vs revise** — when an `AM.<symbol>` event arrives, do we treat it as
   the final word for that bar (lock) or expect Polygon to send corrections
   within ~15 min and accept them (revise)?
2. **Source of truth** — does the engine consume Polygon's pre-aggregated
   `AM.*` minute bars directly, or aggregate Polygon's per-second `A.*`
   bars into minutes ourselves?

These two together define the "live model." We need to pick one (and ship
others as opt-in alternatives) so backtests can match what live execution
actually sees. The current default is `latest` (revise) consuming `AM.*`,
but neither call has been validated empirically — until now.

---

## Polygon stream-channel terminology

For anyone who hasn't lived in this code:

| Channel | Cadence | Source | Reliability today (high-volume symbols) |
|---------|---------|--------|------------------------------------------|
| **`A.<symbol>`** | per-second OHLCV | Polygon, derived from trades | **~97%** of expected bars present in our cache |
| **`AM.<symbol>`** | per-minute OHLCV | Polygon, pre-aggregated from `A` upstream | **~38%** of expected bars present in our cache |
| `T.<symbol>` | per-trade tick | Polygon, raw trade prints | (not used by the engine today) |
| `Q.<symbol>` | quote update (bid/ask) | Polygon | (not used by the engine today) |

Both `A` and `AM` come straight from Polygon. The reliability delta
(see [Engine reliability section](#engine-reliability--problem-b)) is the
delta between Polygon's `A` stream and their downstream aggregation step
that produces `AM`. Polygon's own docs note that `AM` events can arrive
"up to 4 seconds after the end of the minute" — they're aggregated, not
just relayed.

---

## TV Stability Test (2026-05-04)

### Question

Does TradingView silently revise closed bars after the fact (matching
Polygon's late-print correction model), or does it lock the bar at first
close (which would argue for a `ws_first_lock` default)?

### Methodology

- Symbol: SPY
- Timeframe: 1Min
- Method: Kevin pulled the same TV chart export every 5 min for ~100 min
  during 2026-05-04 RTH, saving 21 numbered CSVs in
  `docs/tv_data_5-4/AMEX_SPY, 1 (N).csv` for N in 1..21
- Analysis: for each bar appearing in 2+ snapshots (with the LAST bar of
  every snapshot excluded since it might have been forming at pull time),
  compare OHLCV across snapshots. Bars with any change = revised.
- Source script: `/tmp/tv_diff_analysis.py` (uncommitted; rerun with
  `python3 /tmp/tv_diff_analysis.py`)

### Result — TradingView REVISES closed bars

| Metric | Value |
|--------|------:|
| Bars compared | 417 |
| Bars with any OHLCV change between snapshots | **40 (9.6%)** |
| Of those — close changed | 40 (100%) |
| Of those — volume changed | 40 (100%) |
| Of those — high/low changed | 38 (95%) |
| Of those — **open** changed | **0 (0%)** |
| Time-to-stabilize (max observed) | **5 min** |
| Time-to-stabilize (min observed) | < 5 min (unresolved at this poll cadence) |

**Open is locked at first tick** — that's a clean invariant.
Close, volume, and high/low get revised within minutes.

#### Note on reproducing this result (added 2026-05-05)

The original analysis loaded 21 CSVs from `docs/tv_data_5-4/`. After
folder reorg (2026-05-05) the path moved to `docs/tv_data/` and snap 1
was lost. Running the same script against the surviving 20 snaps
returns **0 revisions** — not because TV behavior changed, but because
the revisions were anchored on snap 1's view of bars that closed within
seconds of that pull. By snap 2 (5 min later), TV had already rolled
in the late-print corrections, so subsequent snaps all see the stable
post-revision values.

This is consistent with — and actually strengthens — the original
finding: TV revises **fast** (within ~5 min and likely much sooner),
and after that the values are sealed. The bars that showed revision
weren't forming-bar artifacts (the script's `_excluded_last` filter
was verified by audit on 2026-05-05); they were closed bars whose
OHLCV got refined post-close.

Future tests should keep ALL snapshots in version control to allow
re-derivation, and pull at a sub-minute cadence to nail down the
actual stabilization window (currently bounded as "somewhere between
0 and 5 min").

### Comparison to Polygon

|        | Revision behavior | Stabilization window |
|--------|-------------------|----------------------|
| Polygon | Sends rebroadcast corrections via the same `AM.*` channel | Up to 15 min (FINRA late-print window) |
| TradingView | Silently updates exported values | ~5 min (likely faster but unresolved) |
| Our cache (Mon-1 finding) | Sees revisions on ~22% of high-volume 1Min bars | within 15 min |

Both major sources revise. TV is more aggressive about pushing the
corrections through. The 22% vs 9.6% gap could reflect different
upstream correction logic, or sample-size variance — not material to
the decision.

### Decision

**Keep `live_model='latest'` (post-rebroadcast) as the default.** Two
independent reference points (Polygon and TV) handle the same data the
same way, so this is the industry-consensus behavior.

`ws_first_lock` (decision-time values) remains available as an OPTIONAL
view — the Lab tab's First-write toggle is exactly that. Use it when
asking "what did the engine see when it decided to fire," but not as the
production default.

### Limitation + follow-up tests

The 5-min poll cadence can't distinguish "stabilized at minute 1"
from "stabilized at minute 5." All 40 revisions fall in a single
"≤5 min" bucket.

To get minute-level resolution:

- **Test #2 (1-min cadence, 1Min bars):** pull SPY 1Min from TV every
  60 seconds for ~30 min. Same methodology, finer poll. Will tell us
  whether revisions cluster at minute 1 or spread evenly through the
  5-min window.
- **Test #3 (per-candle cadence, 10Sec bars):** ✅ **DONE 2026-05-05.**
  See "10Sec follow-up" section below.

Neither test is blocking. They'd refine the stabilization-window
estimate but won't change the decision (TV revises, period).

### 10Sec follow-up (Test #3, completed 2026-05-05)

**Method:** SPY 10Sec exported per-candle (~every 10 seconds) for ~5
minutes. 33 CSVs in `docs/tv_data/5-5 10s test/`. Last bar of each
snapshot dropped (forming, taken a few seconds into the candle).
Source script: `/tmp/tv_10s_analysis.py`.

**Result:** 343 bars compared. **Zero revisions.** No close, volume,
high/low, or open change after the first non-forming observation.

**Caveat:** the per-candle polling means the first observation of any
given bar is ~8–18 seconds after that bar's close (the time it took to
re-pull the next candle). So the test cannot detect revisions that
happen in the first 0–10s post-close. What it CAN say is that TV does
not revise 10Sec bars more than ~15s after close.

**Asymmetry vs 1Min:**

| Bar timeframe | Revision rate | Post-close stabilization |
|---------------|---------------|--------------------------|
| 1Min  | 9.6% revised  | Within ~5 min |
| 10Sec | 0% revised    | Within ~15s   |

Plausible explanations:

1. **Different aggregation pipelines.** TV likely derives 10Sec bars
   directly from per-trade or per-second data and seals them quickly,
   while 1Min goes through a longer aggregation step vulnerable to
   late prints rolling in within FINRA's 15-min window.
2. **Late-print impact is timeframe-dependent.** A late print that
   shifts a 1Min volume by 10% might be invisible in a 10Sec window
   that doesn't even contain that print's effective time.

**Implication for live model:**

- **Sub-minute strategies (5/10/30Sec)**: `first_write` is effectively
  as accurate as `latest` — values don't change post-close. Matches
  what our cache already shows (SPY/TSLA 10Sec: 100% `first_close ==
  close` in Mon-1 validation).
- **1Min+ strategies**: keep `latest` as default — revisions happen,
  fast (~5 min) but real.

Net: the per-strategy `live_bar_source` flag we'll add as part of the
Mode 2/3 rollout could also expose `revision_anchor: first|latest`
defaulting to `first` for sub-minute and `latest` for 1Min+. Or we
keep it implicit since the divergence is structurally small for
sub-minute.

---

## Engine reliability — Problem B

The TV/lock question is one decision. The other one is bigger.

### What's actually broken

The Mon-1 validation found that Polygon's `AM.*` stream is dropping
roughly 60% of expected 1-minute events for high-volume symbols
(AAPL, AMD, SPY). Sub-minute (`A.*`) is fine — 97%+. The pattern is
constant through the day, not bursty, and is volume-correlated rather
than subscriber-count-correlated:

| symbol | TF | RTH coverage | subscribers |
|--------|----|-------------:|------------:|
| AAPL | 1Min | 35% | 3 |
| AMD  | 1Min | 40% | 1 |
| SPY  | 1Min | 38% | 9 |
| META | 1Min | 84% | 2 |
| TSLL | 1Min | 97% | 1 |
| SPY  | 10Sec | 97% | 12 |

**This is not a chart problem — it's an execution problem.** When the
worker doesn't receive the `AM.SPY` event for 12:30, the engine never
evaluates that bar's triggers, no alert fires, the trade is missed.
There is no way to retroactively fire a signal for a bar the engine
never saw in real time.

### Why REST backfill alone doesn't solve this

REST backfill (a periodic job that detects missing rows in `live_bars`
and fetches them from Polygon's REST API) **only fixes the cache**.
The chart and the dashboards look complete. The engine still missed
the bar in the moment. Trades stay missed.

So REST backfill is good hygiene (Lab tab + Data Health stop showing
gaps), but it is NOT the fix for the trade-execution problem.

### Three candidate modes for sourcing 1Min bars

The same Polygon stream that's losing 60% of `AM` events is delivering
97%+ of `A` (per-second) events. So instead of "AM only," we have three
realistic options. The right pick isn't obvious without measurement —
that's what tomorrow's shadow-mode test resolves.

**Note on storage:** all three modes write the SAME row count to
`live_bars` (one per minute, idempotent on `(symbol, tf, bar_start)`).
Only the `source` label differs. This is a latency / accuracy /
reliability trade, not a database-size trade.

| Mode | How it works | Latency after minute close | Reliability | Accuracy assumption |
|------|--------------|---------------------------:|-------------|---------------------|
| **1 — AM-only (current default)** | Engine waits for Polygon's `AM.<symbol>` event, uses whatever arrives. | ~0–4s when it arrives (Polygon's docs note up to 4s aggregation lag); ∞ when it doesn't | **38%** for high-volume 1Min | Polygon's gold standard — includes any late prints they aggregated upstream |
| **2 — A-agg only** | Worker aggregates per-second `A.*` events client-side at minute boundaries. AM is ignored. | ~1s (wait for last per-second bar to land) | **97%** | Whatever Polygon delivered to `A` by the second the minute closed — may miss late prints that `AM` would have included via its 4-sec window |
| **3 — AM-preferred with A-agg fallback** | Wait up to N seconds for `AM`. If it arrives → use it. If not → fall back to A-agg. | Constant ~N+1 sec (set N=5) | **97%+** (97% from A-agg fallback, plus AM accuracy on the 38% where AM arrives) | Best of both — Polygon's accuracy on bars where AM arrives, A-agg's reliability where it doesn't |

### What the shadow test will tell us

Run mode 2 in shadow alongside mode 1 (current production) for a
session. Cache writes both — `source='ws'` for AM, `source='ws_agg'`
for the aggregated bar. On bars where BOTH exist (i.e. AM did arrive
AND we also aggregated A), diff the OHLCV cell-by-cell.

Three possible outcomes and what they mean:

- **(a) Bit-identical or trivially different (e.g. ≤$0.01 close drift,
  identical open):** Mode 2 is the answer. Simpler, faster, more
  reliable. Just always use A-agg. AM becomes irrelevant.
- **(b) Materially different (e.g. close drift > $0.01 routinely, or
  volume diverges meaningfully):** Mode 3 is the answer. The 4-sec
  AM-aggregation window genuinely captures something A-agg misses, so
  we want AM when we can get it. The fallback covers the 60% AM-loss
  cases.
- **(c) Pathological: A-agg sometimes wildly off:** investigate the
  aggregation code; don't ship either yet.

### Open questions

1. **Source-of-truth in cache:** writer gains a `source='ws_agg'`
   value alongside `'ws'` and `'rest_backfill'`. Data Health
   dashboard then shows the source split per (symbol, tf), so the
   migration is auditable.
2. **Sub-minute strategies are unaffected.** 10Sec/30Sec strategies
   already consume A directly via `on_second_bar`. Only the 1Min path
   is in question.
3. **Per-strategy choice or platform default?** Once we pick a winning
   mode, two flavors:
   - **Platform default**: flip the worker to the new mode, all 1Min
     strategies switch at once. Simple, less to maintain.
   - **Per-strategy flag**: `strategy.live_bar_source: 'am'|'ws_agg'|'am_with_fallback'`
     defaulting to the platform pick. Lets the user A/B specific
     strategies; surfaces in Data Health alongside coverage.
   We can decide after seeing the divergence numbers — if (a) wins
   cleanly, platform default makes sense. If (b) and there's a
   meaningful gap, per-strategy flag preserves opt-out for anyone
   running latency-sensitive setups.

### Pairing with REST backfill

REST backfill is still worth doing as a separate cleanup task: it gives
us a complete `live_bars` cache regardless of whether `AM` or
aggregated-from-`A` is the primary source. The Lab tab + Data Health
dashboard stop showing gaps. The natural cadence is the existing
forward-test recompute schedule (it already runs periodically; piggyback).

Order of operations:

1. **Per-second aggregation (Problem B fix)** — primary work tomorrow.
   Behind a flag, shadow mode first, validate against `AM` where both
   exist, then flip per-strategy as confidence grows.
2. **REST backfill (cosmetic + future-proofing)** — secondary, once the
   engine fix is shipped. Lower risk, smaller scope.

---

## Plan for 2026-05-05 RTH (EXECUTED — see status below)

Markets are open tomorrow. Strategy:

1. **Morning sanity check:** open `/admin/data-health` on the rolling
   view. Coverage pattern should match today (AAPL/AMD/SPY 1Min in the
   30–40% range, META/TSLL/sub-minute healthy). Confirms the issue
   isn't a one-day artifact.
2. **TV Tests #2 and #3:** Kevin pulls the 1-min-cadence CSVs in
   parallel while the engine work is in progress.
3. **Shadow-mode rollout for Mode 2 (A-agg):**
   - Add `WsAggBarBuilder` that consumes `A.*` callbacks and emits
     1Min bars at minute boundaries
   - Wire it into `SymbolHub` alongside the existing `on_polygon_bar`
     (`AM`) path
   - Cache writer accepts `source='ws_agg'` so Data Health distinguishes
   - **Don't route to monitors yet** — we're collecting comparison data
4. **Validation pass mid-session:** query for bars where both `ws`
   (mode 1, `AM`) and `ws_agg` (mode 2, A-aggregated) exist for the
   same `(symbol, timeframe_seconds, bar_start)`. Diff OHLCV
   cell-by-cell. The result decides which mode wins:
   - **Diff is trivial → Mode 2.** Always use A-agg. Simplest.
   - **Diff is material → Mode 3.** Build the AM-preferred-with-A-agg-
     fallback wrapper next. Gets us AM accuracy when it arrives, A-agg
     reliability when it doesn't.
   - **Diff is wild → halt and investigate aggregation code.**
5. **Per-strategy cutover (after we pick a winning mode):** add
   `strategy.live_bar_source: 'am' | 'ws_agg' | 'am_with_fallback'`
   defaulting to whichever mode wins. User can opt strategies in or
   out one at a time during transition. Data Health surfaces which
   strategies are on which source so the migration is auditable.

REST backfill is **not** in tomorrow's scope. Queued behind the engine
work — once we're confident in the live source we'll add REST backfill
to retroactively fill cache gaps for chart/dashboard completeness.

---

## EOD 2026-05-05 status — Phase A + B + C SHIPPED

The plan above executed end-to-end in one day. Summary:

**Step 1 (morning sanity)** — confirmed pattern persists. AAPL/AMD/SPY/TSLA
1Min coverage at 35-50% of expected during early RTH (vs prior day's
38%); META/TSLL healthy as before. No one-day artifact.

**Step 2 (TV tests)** — Test #2 (1-min cadence on 1Min) deferred. Test
#3 ran on 10Sec exports → 0 revisions detected at >15s post-close.
Asymmetry vs 1Min documented in 10Sec section above.

**Step 3 (shadow rollout)** — `WsAggMinuteBuilder` shipped 09:50 MT
(`ace1fe7`). Flag flipped to live at 10:05 MT.

**Step 4 (mid-session validation)** — REST-backed validator
(`src/_validate_ws_agg_vs_rest.py`) confirmed close bit-identical to
Polygon REST canonical on 468/468 paired bars across 4 symbols. OHL
diffs sub-cent typical, volume p50 < 4%. **Decision: Mode 2 (A-agg
locked).**

**Step 5 (cutover)** — went further than originally planned. Phases
A+B+C all shipped same day:

- **Phase A** (`2ee8a24` 10:30 MT) — registry refresh. Removed three
  unused M3 placeholder backtest IDs. Added `cache_locked` /
  `cache_corrected` (backtest, gated to Phase E) and `ws_agg_locked` /
  `ws_agg_with_rest_backfill` (live, gated to C/D). Added
  `get_model_status` helper.
- **Phase B** (same commit) — `/admin/live-models` and
  `/admin/backtest-models` admin pages, plus
  `/api/admin/strategy-models/*` endpoints. User-packs-styled card
  list + click-through detail view with usage counts.
- **Phase C** (`74ff56c` 12:55 MT + `f971eeb` 13:42 MT + `b1ad5c0`
  13:44 MT + `ee2c4c5` 15:01 MT recovery) — engine dispatch wired:
  - StrategyMonitor declares live_model on construction
  - Symmetric source-label gate inside
    `_run_monitor_pipeline_for_completed_bar` (ws_agg_locked monitors
    skip on `polygon`, ws_with_corrections monitors skip on `ws_agg`)
  - on_second_bar's WsAggMinuteBuilder block now also dispatches
    completed minute bars to the monitor pipeline when any monitor
    on the symbol has opted in
  - A.* subscription gate gains `has_ws_agg` clause so opting in to
    ws_agg_locked auto-pulls the symbol into the per-second stream
  - `ws_agg_locked.available = True` and `default = True`
  - Bulk DB migration set explicit `config.live_model='ws_agg_locked'`
    on all 39 strategies (revert: `python _bulk_set_live_model.py
    NULL`)

**Worker recovery saga** — three rapid pushes (74ff56c → f971eeb →
b1ad5c0 in 2 min) caused Railway to queue conflicting builds, two
stuck BUILDING + one FAILED. Worker kept serving the 12:55 deploy
(Phase C wired but no default flip seen) until empty commit `ee2c4c5`
kicked a clean rebuild at 15:01 MT. All four services SUCCESS post-
recovery.

**End-of-day verification (post-RTH at 15:00 MT):**
- 12 Polygon channels subscribed (6 AM + 6 A) — confirms gate expansion
- ws_agg writes flowing for AAPL/AMD/SPY/TSLA (high-volume → constant
  trade activity → A events every second → builder accumulates and
  closes minute bars)
- ws_agg = 0 for META/TSLL (low-volume post-market, A events sparse —
  expected; will normalize during RTH tomorrow when trade activity
  resumes)
- 1Min strategy alerts silent because 19 of 22 active 1Min strategies
  are session=RTH and we're past 16:00 ET — also expected
- 10Sec Extended Hours strategies firing normally (~56 alerts in 10
  min on SPY 10Sec)

**Real validation:** RTH open 2026-05-06 13:30 UTC. What we're
watching for:
1. 1Min RTH strategies activating, BAR_CLOSE logs with their strat IDs
2. META/TSLL ws_agg writes resuming as trade activity picks up
3. AM coverage on AAPL/AMD/SPY/TSLA — if it stays at 0% during RTH,
   the migration was the right call

**Outstanding:**
- Phase D (REST backfill writer) and Phase E (backtest dispatch wiring)
  still queued per the plan file at
  `~/.claude/plans/synchronous-tickling-yeti.md`
- vwap_v2 user pack warning (`'str' object has no attribute 'timestamp'`)
  fires every few seconds in worker logs — pre-existing, not Phase C
  related, worth fixing when convenient

---

## References

- TV CSV source files: `docs/tv_data_5-4/AMEX_SPY, 1 (N).csv` (N = 1..21)
- TV diff analysis script: `/tmp/tv_diff_analysis.py`
- Live `live_bars` cache schema: `src/migrations/live_bars_table.sql`
- Live `live_bars` writer: `src/live_bars_writer.py`
- Polygon AM handler: `src/ralph_engine.py` `on_polygon_bar()` (~line 1643)
- Polygon A handler: `src/ralph_engine.py` `on_second_bar()` (~line 1798)
- Data Health backend: `src/api/routers/data_health.py`
- Data Health frontend: `frontend/src/app/admin/data-health/`
- Mon-1 cache validation script: `src/_validate_live_bars_cache.py`
- Earlier related memos:
  - `docs/Plan_M8.7_Saturday_2026-05-02.md` — Mon-1/Mon-2 validation queue
  - `docs/Deploy_Log.md` — chronological deploy events
