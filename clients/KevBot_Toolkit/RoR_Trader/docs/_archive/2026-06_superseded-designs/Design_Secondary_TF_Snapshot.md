# Design — Secondary-TF snapshot for fast Update-New-Data — 2026-06-22

## ⚠️ SCOPE NOTE (2026-06-23) — snapshot currently only covers 1Hour+ secondaries
The seed + the guard-lift are both gated on `_has_long_cycle_secondary_tf` = **1Hour or
coarser** secondary (1h/2h/4h/1d/1w/1mo). Confirmed live 2026-06-23: every 1Hour+ strategy
(310/311/312/313/314/325/330/331) is seeded; sub-1Hour (2m/5m/10m/30m/3m, e.g. 327/328/329/309)
is NOT — those rely on the existing windowed resume.
- **Why it's probably fine now:** sub-1Hour secondaries warm in a short window, so the windowed
  resume is fast for them already (the guard threshold reflects this).
- **POTENTIAL GAP to verify (Kevin's flag):** on a SHORT resume window, a 10m/30m secondary may
  still be under-warmed (e.g. 1 day of 30m ≈ 48 bars < a 100-bar indicator). The 1Hour threshold
  is a heuristic; sub-1Hour under-warm on resume is unverified. NOT urgent, but as strategies run
  for months and we move to a cron, verify and — if needed — **lower the snapshot threshold so
  ALL gated strategies get the snapshot treatment** (the snapshot is cheap, ~4KB). Track under the
  bug-hunt loop (accuracy + scalability goals).

# (original title) Design — Secondary-TF cache for fast Update-New-Data (FOR REVIEW) — 2026-06-22

## Where the cost actually is (important — it reframes the fix)
Traced the backtest path (`services.prepare_data_with_indicators`, lines 349–384). The
secondary-TF work itself is **cheap**: resample is in-memory, and `run_all_indicators` /
interpreters run on a **small** coarse series (1d over a year ≈ 365 rows; 4h ≈ ~1500). The
expensive part is the **primary data LOAD** (`load_market_data`) over a **long warmup window** —
and that window is long *only because the coarse 1d/4h gates need a long history to warm up*.
The primary's own indicators (15Sec) warm in hours, not months.

So the 700s isn't "computing secondary indicators" — it's "loading months of bars so the coarse
secondary series can be built." The primary snapshot (`engine_snapshot_b64`) resumes the
**primary** incremental engine but does NOT cover the secondary series, so that long load recurs
every append.

## The fix (and the fidelity-sensitive part)
**Cache the computed secondary series** (resampled OHLCV + indicators + interpreter states) so an
append no longer needs the long primary load to rebuild it. Concretely, on Update-New-Data:
1. **Shorten the primary load window** to just the primary's own warmup + the visible/new range
   (short — 15Sec warms fast), INSTEAD of the long coarse-gate warmup.
2. **Load the cached secondary series** and **extend** it with only the few new coarse bars (a
   short recent load forms the new 1d/4h bars), then **inject via the existing `secondary_tf_dfs`
   hook** (services.py:357 already supports this — skips the resample).
3. Result: no months-long load; primary load shrinks dramatically → the speedup.

**The fidelity-sensitive step is #1** — shortening the primary warmup window. It MUST produce
byte-identical primary indicator values vs the long-warmup run, and the cached+extended secondary
series MUST be byte-identical to a fresh full resample. **This is the whole risk** (it's why the
primary snapshot had scarring — warmup edges). The parity guard gates everything.

## Caching scope (answer to Kevin's question)
- **Update-All-Data**: stays a full recompute (source of truth). It computes the full secondary
  series anyway → **writes/refreshes the cache** as a near-free side effect. Not itself sped up.
- **Update-New-Data**: **reads** the cache + extends it (the fast path) + writes back. Cold-seeds
  once if absent. **The speedup is realized here** — the periodic/cron path.
- Cache is **fingerprint-keyed** (same `compute_backtest_fingerprint`): any config change
  invalidates it → next Update-All rebuilds fresh. Per-`model_id` too (rest_hifi vs cache lanes).

## Cache storage
Payload is **small** (coarse series ≈ tens of KB). Options: a `config` field
(`secondary_series_b64`, like `engine_snapshot_b64`) or Supabase Storage (like Tier-2 dough).
Leaning config-field for simplicity given the small size; revisit if a strategy has many/large
secondaries. api is ephemeral on Railway, so it MUST be DB/Storage, not local disk.

## Parity guard (the gate — build FIRST, before enabling)
Offline test on the 1d/4h strategies (**313, 314**) asserting:
- **Secondary series byte-identical**: cached+extended vs fresh full-resample (every column,
  every coarse bar).
- **Primary indicators byte-identical** at the short-warmup boundary vs long-warmup.
- **Trades identical**: the resulting `unified_trades` output matches the full-recompute trades
  (entry/exit ts, price, exec_type) over the overlap window.
Run on: primary+secondary, a 1d gate (313 has 1d+4h), a 4h gate, and an ungated control.
**Enable only when byte-identical. Kill-switch env flag default OFF. Canary on 313/314 first.**

## MEASURED RESULTS (2026-06-22 EOD — real UND append path, sid 313, force=True)
| lane | cache | time | notes |
|------|-------|------|-------|
| BT   | seeded  | **3.8–4.6s** | uses get_strategy_trades_for_window (wired) |
| ALGO | not seeded | **390.9–391.8s** | uses the separate `full` recompute path (NOT wired) |

Lanes intact afterward (455/455, no corruption). Fidelity byte-identical (4 offline proofs).
**~85–100× speedup where the cache is present.** Flag ON on api (RORT_SECONDARY_TF_CACHE=1).

### ⚠️ REMAINING PIECE — algo-lane / full-recompute cache seeding (NOT done)
The cache is read+injected+persisted ONLY in `services.get_strategy_trades_for_window`
(the windowed path). The BT lane always routes through it → benefits + self-seeds on a cold
(envelope=None) full compute. BUT:
1. The **algo lane** uses a separate `full` recompute path (forward_test_service else-branch,
   when use_windowed=False) that does NOT call get_strategy_trades_for_window → never seeds its
   cache → stays 391s. Guard-lift can't help because the cache never gets written.
2. **Auto-seed bootstrap:** persist only fires on a full compute (`envelope is None` — so the
   secondary is fully warmed, never caching an under-warmed short-window secondary). Established
   strategies resume (envelope present) → never hit the persist → cache never seeds on its own.

**TO FINISH:** wire the cache WRITE into the full-recompute path (Update-All:
`recompute_and_persist_stored_trades`, + the algo `full` branch) so coarse strategies get seeded
fleet-wide. Then both lanes take the fast windowed path. The READ/INJECT/short-window side is
done + proven; this is the seeding side.

NOTE: my change also appears to FIX a latent correctness issue — without the cache, a coarse-gate
strategy resuming via the primary snapshot (short window) under-warms the secondary → wrong
gates; the guard currently prevents that by forcing `full`. The cache provides a warmed secondary
so the short resume becomes correct AND fast. Worth confirming when finishing the seeding.

## CORRECTED FRAMING (2026-06-22, after reading the code)
We ALREADY snapshot the primary, and strategies WITHOUT a long-cycle secondary already get the
fast windowed resume. The gap is an explicit guard — `_has_long_cycle_secondary_tf`
(forward_test_service.py:1244-1248) — that routes any strategy with a **1Hour+ secondary**
(1d/4h, e.g. 313/314) to the **`full` warmup path**, because the snapshot only captures the
PRIMARY engine state (`serialize_backtest_snapshot` = `strat.indicators` only) and a coarse
secondary would resume into undefined indicators. So there is really ONE fix, not
aggressive-vs-conservative: **extend the snapshot to also cover the secondary series, then lift
the guard for strategies that have a valid secondary snapshot.** Same mechanism as the primary,
one level out.

Note: in the BACKTEST path the secondary isn't a stateful incremental engine — it's a vectorized
resample (`resample_to_timeframe`) + `run_all_indicators` over the small coarse series
(services.py:349-384). So the "secondary snapshot" = the **cached resampled secondary OHLCV**
(indicators recompute deterministically + cheaply on it). Cache the OHLCV (the
expensive-to-build-from-1min part); recompute indicators on resume.

## Open decisions for Kevin
1. **Warmup-shortening is the crux + the risk.** Are you comfortable with the approach of
   shrinking the primary load when the secondary is cached (gated by byte-identical parity), or
   would you prefer a more conservative variant (e.g., cache the secondary but keep the full
   primary warmup — smaller speedup, lower risk)? The conservative variant saves the secondary
   *recompute* but not the long *load*, so the win is smaller.
2. Cache storage: config field vs Supabase Storage.
3. Scope: implement on 313/314 (1d/4h) first as canaries, measure, then widen.

## Recommendation
Because the speedup *requires* the fidelity-sensitive warmup-shortening, I recommend we align on
decision #1 before I implement — then I build the parity harness, implement behind the kill-switch,
prove byte-identical on 313/314, and bring you the numbers + parity results for review before any
enablement. I will NOT enable anything that isn't byte-identical.
