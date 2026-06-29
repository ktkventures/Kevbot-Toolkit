# Design — Continuous bar_cache Write-Through (REST stream unification) — 2026-06-28

**Status:** Steps 1–3 SHIPPED + write-through ENABLED in prod (2026-06-29). Burn-in live during RTH.
Steps 4–5 (consumer flip + supply-page coverage) are NEXT and still design-stage. See §10 for live state.
**Parent:** `Plan_M-RS3_RealTime_REST_Shadow.md` §7.7 (the reframe). Related: `Design_Recompute_Bar_Cache.md` (2026-06-23).

## 1. Problem — three REST/bar copies, not unified
| System | Storage | Update | Fresh? | Consumers |
|---|---|---|---|---|
| **bar_cache** | Postgres (persistent) | pull-on-read + maintain cron (default OFF) + <45min re-fetch | **DELAYED** | backtests, charts, recompute, Mass Builder |
| **SymbolBarStore (1s) + CoarseBarStore (1min)** | in-memory (data-worker) | **continuous** — 1s ingest + 60s recon, last-write-wins | real-time (~6s) | **only** streaming engines |
| **live_bars** | Postgres (immutable) | WS event-driven | real-time | forensics / parity |

The data-worker already produces a **continuous, self-correcting REST stream** — but it lives only in
memory and feeds only the streaming engines. `bar_cache` (what everything else reads) is maintained
separately and goes stale. Two REST copies, two mechanisms, no link. This sprawl is the root of:
append fossilization/missing-bars, confluence-gate blowups, and the snapshot/update-new "messiness."

## 2. Goal — one continuously-fresh source
**Have the data-worker's reconciled stream WRITE THROUGH to `bar_cache`,** so `bar_cache` is
continuously fresh + gap-reconciled and **every consumer reads one live source.** Collapses the
in-memory store + on-demand cache into a single layer. `live_bars` stays separate (WS forensic,
immutable — different goal; out of scope).

**Non-goal:** changing the live (WS) decision path, or merging `live_bars`. This is purely the
REST/backtest supply.

## 3. The write-through contract
- **Where:** in the data-worker ingest/recon cycle (`data_worker_ingest.py` `run_ingest_cycle` 1s /
  `run_recon_cycle` 60s / `run_coarse_ingest_cycle` 30s), AFTER `upsert_bars()` to the in-memory
  store, ALSO upsert the same reconciled bars into the `bar_cache` table.
- **What granularity:** **native 1Sec (`SymbolBarStore`) + native 1Min (`CoarseBarStore`)** — the two
  source layers (cache stores 1Sec+1Min, derives the rest; `feedback_fidelity_parity_gate`). Coarse
  (>=1Hour) stays `materialize_derived` from 1Min; sub-minute derives from 1Sec.
  **MUST persist NATIVE 1Min, NOT `SymbolBarStore.get_timeframe('1Min')` (1s-resample)** — Step 1
  (2026-06-28) proved resampled-1Min OHLC is byte-identical to native but **volume diverges** (29/60
  bars; 1s-volume-sum ≠ native-1Min-volume, a Polygon aggregation quirk). Native 1Min keeps volume
  faithful (RVOL/VWAP).
- **Write pattern (churn-safe):** idempotent **upsert on `(symbol, timeframe, ts)`**, last-write-wins.
  The unsettled tail is overwritten each recon (bounded rows, **no growth**); settled bars written
  once. Respects the 2026-06-28 bloat lessons (#6 autovacuum / #7 leak) — upsert-on-tail, never
  insert-per-poll.
- **Coverage:** driven by the Bar Cache Supply registry — see §3.5 (RESOLVED: stream EVERY symbol in
  the supply; auto-enroll new strategy symbols).

## 3.5 Coverage model — Bar Cache Supply as single source of truth (RESOLVED, Kevin 2026-06-28)
The **Bar Cache Supply page is the registry of capture targets** and the one control surface. No dual
path.
- **Registry = `(symbol, lookback)`.** Lookback default **1 year**, **UI-adjustable** (extend → triggers
  a deeper one-time `backfill_symbol`; this is Kevin's "expand the backtest data" lever). Can extend
  beyond 1y later.
  - **Lookback must be PER-RESOLUTION (not one flat number).** 1 year of **1Sec** ≈ ~5.8M rows/symbol
    (~23k bars/RTH-day) — fleet-wide that is enormous. Want **deep 1Min (e.g. 1y)** but a **shallow
    1Sec window** (days–weeks, only what sub-minute strategies + provisional detection need). Model
    lookback as `{'1Min': 365d, '1Sec': Nd}`, not a single value.
- **Stream-maintain everything in the registry:** the data-worker streams write-through 1Sec+1Min for
  **every symbol in the supply** (not just symbols with eligible streaming strategies) — continuous
  freshness on the recent edge; backfill provides the historical depth. Same `bar_cache`, one store.
  - **CODE CHANGE (not yet done):** today the data-worker provisions stores only for symbols with
    streaming-eligible strategies (`_load_streaming_strategies` / `_ensure_symbol_stores` in
    `data_worker.py`). Streaming ALL supply-registry symbols requires changing that discovery source
    to the supply registry (`get_capture_targets` / `bar_cache_config`, `bar_cache_admin.py`). Name
    this as an explicit Step-5 task.
- **Auto-enroll:** creating a strategy with a symbol not in the supply **auto-adds** a capture target
  (kick off backfill + ongoing stream). Manual add/extend also available in the UI. New strategies are
  never missing data; coverage stays automatic AND user-controllable ("both ways").
  - **HOOK POINT (not yet wired):** the auto-add belongs in the strategy-create path
    (`api/routers/strategies.py`), writing into the EXISTING capture-targets registry
    (`get_capture_targets` / `bar_cache_config`, `bar_cache_admin.py`) — extend it, don't build a new
    registry. Also ensure `idx_bar_cache_unsettled` exists before/at first enrollment (see §4).
- **Retire the maintain cron** (`BAR_CACHE_MAINTAIN_ENABLED`, currently OFF anyway): the stream
  write-through replaces it. The supply registry + stream is the only freshness mechanism. Coarse
  (>=1Hour) still `materialize_derived` from the fresh 1Min.

## 4. Settled / in-flux marker (schema change)
`bar_cache` today: `(symbol, timeframe, ts, open, high, low, close, volume, cached_at)` — no settled flag.
Add:
- `settled boolean NOT NULL DEFAULT false` — true once `ts < now - RORT_SHADOW_SETTLE_MIN`
  (**default 16 min**, tunable Monday; tighter than the 45min read-horizon pad).
- `revised_at timestamptz` — last overwrite time, for observability of how the tail moved (CONFIRMED —
  useful for future features / debugging).

**Consumer rule:** only `settled=true` bars are permanent backtest truth. The unsettled tail is
provisional (overwritten each recon). Divergence detection counts on settled bars; provisional
divergence is a separate, softer signal.

**REQUIRED INDEX for the settle sweep (2026-06-29 burn-in lesson).** `settle_sweep()` runs
`UPDATE bar_cache SET settled=true WHERE symbol=$1 AND settled=false AND ts<cut` every recon. With
only the PK `(symbol,timeframe,ts)`, that scans ALL rows for a symbol to find the small unsettled
tail → `57014` statement-timeout on million-row symbols (observed TSLA/SPY/TSLL the moment
write-through was enabled; data stayed correct, only the `settled` flag lagged). FIX = a **partial
index** `idx_bar_cache_unsettled ON bar_cache(symbol, ts) WHERE settled=false` (migration
`src/migrations/bar_cache_unsettled_index.sql`, applied to prod via `CREATE INDEX CONCURRENTLY`).
**This index is a hard prerequisite for §3.5 scale** — every symbol added to the supply enlarges
the table; without it the sweep cannot keep up. Step 5 enrollment MUST ensure it exists.

## 5. Consumer unification
Once write-through keeps `bar_cache` continuously fresh for covered symbols:
- Flip **`BAR_CACHE_ENABLED`** on (currently default OFF) so `data_loader.load_market_data`
  (data_loader.py:714) reads `bar_cache` first. Then backtests / charts / recompute / Mass Builder
  all read the same live source.
- The **maintain cron** (`BAR_CACHE_MAINTAIN_ENABLED`, `maintain_symbol`) becomes redundant for
  covered symbols (the stream maintains them); keep it for uncovered symbols, or retire once coverage
  is fleet-wide.

**ORDERING DEPENDENCY (Step 5 partly precedes Step 4).** The `BAR_CACHE_ENABLED` flip is only safe
once `bar_cache` has correct, fresh **historical depth** for covered symbols — not just the stream's
rolling-window prime. The enable-time backfill only seeds ~150d of 1Min and ~minutes of 1Sec; a
backtest needing a full year (or deep 1Sec) would still hit stale/missing history (it lazily
delta-fetches on read, but that defeats the "fresh unified source" intent and re-introduces the
on-read latency). So **do the full-depth backfill (a Step-5 concern) FIRST, or scope the Step-4 flip
to only fully-backfilled symbols.** The §6 #3 trade-level gate catches a bad flip, but the ordering
must be explicit so we don't flip a half-backfilled cache under live consumers.

## 6. Validation gates (the write-through must NOT become a new divergence source)
1. **Byte-identical settled-bar gate:** the stream's written `bar_cache` rows must equal a fresh
   Polygon REST pull on **settled** bars — extend `fidelity_parity_suite.py` cache-parity tests to
   validate the write-through output (1Sec + 1Min, RTH + 24/7). Reuse `_ws_rest_empirical.py` shape.
2. **`fidelity_parity_suite.py` 18/18** before flag flip (the standing gate).
3. **Trade-level gate:** a backtest reading the write-through `bar_cache` must produce trades
   byte-identical (via `trade_snapshot.py`) to one reading the current on-demand path, over a settled
   window, for the canary set.
4. **Shadow burn-in:** run write-through default-OFF in prod (write to a side column / dry-run log
   diff) for a session before flipping consumers onto it.

## 7. Safety / rollback
- **`RORT_BARCACHE_WRITETHROUGH` (default OFF):** when off, data-worker doesn't write `bar_cache`;
  consumers keep current behavior. Instant rollback by unsetting.
- **`BAR_CACHE_ENABLED`** stays independently gated (consumer read switch).
- **No churn:** upsert-on-tail bounded; covered by #6 autovacuum tuning.
- Touches `bar_cache.py` (formerly the departed backend agent's file — now owned here) +
  `data_worker_ingest.py` (ours). Flag-gated, byte-identical-gated.

## 8. Divergence payoff (maps to Kevin's pain points)
| Class | Verdict | Why |
|---|---|---|
| Append missing-bars / fossilization | **Strong** | continuous reconciled write-through — no stale edge, no insert-only dedup fossils |
| Confluence gates maxed | **Strong** | aligned, continuously-fresh bars + secondary-TF states |
| Snapshot / update-new "messiness" | **Strong** | a fresh cache removes the windowed re-fetch that the messy snapshot path compensates for |
| WS vs REST | **Partial/Strong** | removes the REST-supply seam; ~1% inherent WS/REST timing floor remains |
| Hi-Fi exit-ts jitter | **Orthogonal** | sub-second timing, not bar-source |

## 9. Decisions (RESOLVED — Kevin 2026-06-28)
1. **Settle threshold** `RORT_SHADOW_SETTLE_MIN` = **16 min** to start; tune Monday against live obs.
2. **Coverage** = **Bar Cache Supply registry is the single source of truth** (§3.5). Stream-maintain
   EVERY symbol in the supply; **auto-enroll** new strategy symbols; UI-adjustable lookback (default
   1y). One store, no dual path.
3. **Maintain cron** = **retired** (replaced by stream write-through; the supply registry is the
   control surface).
4. **settled column** = `settled` **+ `revised_at`** (both; revised_at for observability).
5. **Dedicated service** = keep write-through inside the data-worker for now (already its own
   service); split later if the 10k-strategy scale needs it.

### UI implication (Bar Cache Supply page)
The page may need: per-symbol **lookback control** (extend → deeper backfill), an **auto-enrolled**
indicator (symbol added by a new strategy vs manual), and stream/settled status. Scope the UI changes
alongside the backend.

## 9.5 Step 1 VALIDATION RESULT (2026-06-28) — `src/_validate_stream_writethrough.py`
TSLA, settled 60-min RTH window: ingest byte-perfect (store 1Sec == Polygon 1Sec, 0 diffs);
resampled-1Min **OHLC byte-identical** to native, **volume diverges** (29/60 bars) → persist native
1Min. bar_cache (TSLA) already byte-identical to Polygon. **Write-through is OHLC-safe; use native 1Min
for volume fidelity.** Side-finding: streaming engine feeds 1Min strategies the resampled (wrong-volume)
series → likely RVOL/VWAP divergence lead (separate). TODO: run SPY (known-stale) to show write-through
fixes staleness; run a sub-minute strategy window for trade-level confirmation.

## 10. Sequencing
1. ✅ **DONE** — Step 1 validation (`_validate_stream_writethrough.py`): stream byte-identical to
   Polygon on settled bars; OHLC perfect, persist native 1Min for volume.
2. ✅ **DONE** (committed `feat/barcache-writethrough` 4bebb43, gated default-OFF) — `settled`/
   `revised_at` migration applied; `write_through_bars` + `settle_sweep` in bar_cache.py; hooks in
   data_worker_ingest (ingest+coarse+primes+recon). Validated: OFF=no-op, ON=native writes,
   settled+revised_at correct, OHLCV byte-identical (`_test_writethrough.py`).
3. ✅ **SHIPPED + ENABLED (2026-06-29)** — PR #5 (bar-leak) + PR #6 (write-through) merged to `dev`;
   all 7 Railway dev services redeployed; `RORT_BARCACHE_WRITETHROUGH=1` set on the Data Worker.
   Burn-in live during RTH: writing native 1Sec+1Min for DIA/SPY/KO/TSLA/TSLL, settled/revised_at
   correct, churn-safe, stream fresh (~2s lag). **Burn-in caught + fixed** the settle-sweep `57014`
   timeout via partial index `idx_bar_cache_unsettled` (§4).
   - ✅ **Write-through parity gate ADDED** to `fidelity_parity_suite.py` (`--writethrough`): reads
     the bar_cache table directly (no backfill mask) vs Polygon on settled bars; splits WARN
     (faithful-but-incomplete-history → backfill) from FAIL (holes/diffs within coverage → bug).
     2026-06-29 burn-in run: **within-coverage OHLCVdiffs=0 rowMM=0 for ALL 5 symbols × {1Sec,1Min}**
     (TSLA/SPY PASS; DIA/KO/TSLL 1Sec WARN = pre-enable history gap, ~6.1k/4.9k/8.0k bars). Full
     suite green: 25 pass / 3 warn / 0 fail, incl. canary 267 `BAR_CACHE_ENABLED` ON==OFF tol=0.
   - ✅ **Trade-level diff DONE** (`src/_writethrough_trade_diff.py`): canary 267 closed trades are
     byte-identical (full fields: r/prices/reason/pnl/direction/exec_type) reading write-through cache
     vs native — native=327 cache=327, 0 changed. NOTE: must exclude OPEN positions (exit=NaT) — their
     r is mark-to-market vs current price, so two RTH runs seconds apart differ benignly.
   - STILL TODO under this step: tune `RORT_SHADOW_SETTLE_MIN` against live obs; (optional) run the
     trade diff on more canaries post-close.
4. Flip `BAR_CACHE_ENABLED` on so consumers read the unified source; retire maintain cron for covered.
   **PRE-REQ (see §5 ORDERING):** full-depth backfill first, or scope the flip to fully-backfilled
   symbols; trade-level gate (§6 #3) must stay clean before/after.
5. Supply-page coverage model (§3.5): auto-enroll, UI lookback control, stream all registry symbols.
   Concrete tasks now named in §3.5: **per-resolution lookback** (deep 1Min / shallow 1Sec); change
   data-worker symbol discovery to the supply registry (`data_worker.py`); auto-enroll hook in
   `api/routers/strategies.py` extending `get_capture_targets`/`bar_cache_config`; ensure
   `idx_bar_cache_unsettled` exists at enrollment.
6. (Later) provisional/divergence-detection layer on the unsettled tail (the per-second goal).

## Pointers
- `data_worker_ingest.py` (ingest/recon cycles, upsert_bars), `bar_store.py` (in-memory store),
  `bar_cache.py` (persistent table, cached_load_market_data, maintain_symbol, materialize_derived,
  read_bars), `data_loader.py:714` (load_market_data / BAR_CACHE_ENABLED), `fidelity_parity_suite.py`
  (gate), `trade_snapshot.py` (trade-level gate). Memory: `project_realtime_rest_shadow`,
  `reference_db_health_settings`, `feedback_fidelity_parity_gate`.
