# Design — Continuous bar_cache Write-Through (REST stream unification) — 2026-06-28

**Status:** DRAFT for Kevin's review. No code yet.
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
- **Stream-maintain everything in the registry:** the data-worker streams write-through 1Sec+1Min for
  **every symbol in the supply** (not just symbols with eligible streaming strategies) — continuous
  freshness on the recent edge; backfill provides the historical depth. Same `bar_cache`, one store.
- **Auto-enroll:** creating a strategy with a symbol not in the supply **auto-adds** a capture target
  (kick off backfill + ongoing stream). Manual add/extend also available in the UI. New strategies are
  never missing data; coverage stays automatic AND user-controllable ("both ways").
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

## 5. Consumer unification
Once write-through keeps `bar_cache` continuously fresh for covered symbols:
- Flip **`BAR_CACHE_ENABLED`** on (currently default OFF) so `data_loader.load_market_data`
  (data_loader.py:714) reads `bar_cache` first. Then backtests / charts / recompute / Mass Builder
  all read the same live source.
- The **maintain cron** (`BAR_CACHE_MAINTAIN_ENABLED`, `maintain_symbol`) becomes redundant for
  covered symbols (the stream maintains them); keep it for uncovered symbols, or retire once coverage
  is fleet-wide.

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
1. Add `settled` (+`revised_at`) column to `bar_cache` (migration).
2. Build write-through in `data_worker_ingest` (gated OFF) — upsert reconciled 1Sec/1Min to bar_cache.
3. Validation: extend `fidelity_parity_suite` write-through gate; dry-run burn-in; trade-level diff.
4. Flip `RORT_BARCACHE_WRITETHROUGH` on for covered symbols; observe.
5. Flip `BAR_CACHE_ENABLED` on so consumers read the unified source; retire maintain cron for covered.
6. (Later) provisional/divergence-detection layer on the unsettled tail (the per-second goal).

## Pointers
- `data_worker_ingest.py` (ingest/recon cycles, upsert_bars), `bar_store.py` (in-memory store),
  `bar_cache.py` (persistent table, cached_load_market_data, maintain_symbol, materialize_derived,
  read_bars), `data_loader.py:714` (load_market_data / BAR_CACHE_ENABLED), `fidelity_parity_suite.py`
  (gate), `trade_snapshot.py` (trade-level gate). Memory: `project_realtime_rest_shadow`,
  `reference_db_health_settings`, `feedback_fidelity_parity_gate`.
