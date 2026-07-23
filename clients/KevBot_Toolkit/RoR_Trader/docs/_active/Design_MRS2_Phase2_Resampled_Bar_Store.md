# Design — M-RS2 Phase 2: Canonical Resampled Bar Store

**Author handoff for a parallel build session (2026-07-09).** Frames Kevin's "pre-resampled
bar-cache" idea as the next phase of M-RS2 (the shared bar store). Phase 1 = the 1Sec/1Min
`bar_cache`. Phase 2 = a canonical, session-keyed store of the RESAMPLED coarse bars (2Min…1Day)
that BOTH the live and backtest lanes read, so they can no longer construct the same gate bar
differently. Read `Design_Gate_Fidelity_Hardening.md` first (the "why").

> **STATUS (2026-07-10):** BUILT + MERGED (PRs #52/#53/#54) with a stronger-than-spec safety model
> (verify-first: flag ON runs the same output code as OFF; the store is byte-verified against the
> engine's own builds on the settled/aligned zone). Flags armed Fri 21:46Z. **The living rollout
> tracker is `docs/_archive/2026-07_completed-rollouts/Plan_MRS2_P2_Rollout.md`** (archived complete, V4.4 sweep 07-23) — phases, evidence gates, serve-cutover criteria.

---

## 1. The problem it retires (measured, 2026-07-06→09)
Nearly every divergence bug this month is a CONFLUENCE-GATE bug whose root is: **live and backtest
construct/serve the same secondary-TF bar via DIFFERENT code paths**, so the two can disagree.
- **313** (fixed via `RORT_MTF_COARSE_RTH_RELOAD`): the LIVE 4H shadow ingested session-unfiltered
  after-hours 4H buckets and carried a stale `INSIDE` state across the overnight boundary; backtest
  resampled RTH-only 1Min→4H (`TWO_DOWN`). Two constructions, two answers.
- **272** (state-freeze), **325/330/331** (contaminated 4h-SWING), **338** (session mismatch) — all
  the same family: the live coarse-bar construction ≠ the backtest coarse-bar construction.

**The structural cure:** gates evaluate on the *previous COMPLETED* coarse bar, which is settled.
If BOTH lanes read that completed bar from ONE canonical store (per symbol, tf, **session**), the
gate state is identical *by construction* — 313's contamination becomes IMPOSSIBLE (there is no
separately-built live 4H bar to disagree). This future-proofs the class: a new coarse-gated
strategy cannot reintroduce it.

## 2. What it does — and does NOT — fix (be honest)
- ✅ **Fidelity:** retires the gate-CONSTRUCTION divergence class (the biggest class this month).
- ✅ **Speed:** the coarse-resample hot paths — the parity ribbon (measured 62-day window = 82k
  rows / 32s on-the-fly) and the backtest's repeated 1Min→coarse resampling — become a cache read.
- ✅ **Future-proofing / ordering:** one definition of "the 5m/4h/1d bar," everywhere.
- ❌ **Does NOT fix** the WS-vs-REST decision-timing floor (308/266): live fires the PRIMARY trigger
  on the real-time forming bar; the store is about COMPLETED coarse bars (gates), not the forming
  primary. Don't expect it to move the primary-timing floor.
- ⚠️ **It is a hardening investment, not a blocker** — today's targeted fixes got the fleet
  tradeable-close without it. Priority: after the bug-hunt closeout; frame as advancing M-RS2, and
  keep it from colliding with M-RS5 (resident window) — the store is a natural DATA source the
  resident window can read.

## 3. The canonical definition (the single source of truth the store must reproduce)
There must be ONE function that defines "the coarse bar," and the store must be BYTE-IDENTICAL to it:
```
canonical(symbol, tf, session, window) =
    resample_to_timeframe( load_1min(symbol, window, session-filtered), tf )
```
- Resample ALWAYS from 1Min, NEVER native coarse (CLAUDE.md — Polygon native daily has split bugs).
  `data_loader.resample_to_timeframe` + `_RESAMPLE_RULES` is the canonical resampler. **NOTE the
  `'1M'`/1Min gotcha (bug #50):** `_RESAMPLE_RULES` has no `1Min` key (1Min is the base, not a
  resample target), and `'1M'` uppercase is the primary-TF sentinel in the record namespace while
  `'1m'` lowercase is a real 1-minute secondary. The store's TF set is the COARSE bars (2Min…1Day);
  1Min itself is served from `bar_cache` directly (see `strategy_data._build_native_1min_secondary`,
  added in #50, for the 1-minute-secondary case).
- **Session is a first-class key** (this is the 313 lesson — do NOT skip it). Sessions: `RTH`,
  `Extended Hours`, `24/7` (crypto). The session filter is applied to the 1Min bars BEFORE resample
  (see `ralph_engine._is_in_session` / `_SESSION_HOURS`, and how backtest filters). A store row is
  `(symbol, tf_seconds, session, bar_start)` → OHLCV.
- **The cross-TF shift (`secondary_tf_shift`) stays at GATE-EVAL time, NOT in the store.** The store
  holds the raw completed coarse bars; consumers apply the +1-period shift when reading for a gate
  (as today). Baking the shift into the store would couple it to one consumer.

## 4. Schema (extend the bar_cache model)
`resampled_bar_cache(symbol text, timeframe_seconds int, session text, ts timestamptz,
  open/high/low/close/volume, settled bool, revised_at timestamptz, source_1min_span_hash text,
  PRIMARY KEY(symbol, timeframe_seconds, session, ts))`
- `source_1min_span_hash` = hash of the 1Min bars that produced this coarse bar → detect when the
  underlying 1Min changed (T+2 revision / settle sweeper correction) → re-resample that bucket.
- `settled`/`revised_at` mirror `bar_cache` semantics (settled at ~15-min lag; but see §5 on
  the write path having to WINDOW-REPLACE, not upsert, for retracted bars).

## 5. Write / build path (the maintenance loop)
- A builder that watches the 1Min layer (`bar_cache` 1Min + the settle sweeper) and, when a 1Min
  bar in a coarse bucket settles OR revises, re-resamples the AFFECTED coarse bucket(s) per session
  and upserts. Runs alongside `settle_sweeper` (same cadence/service).
- **WINDOW-REPLACE, not upsert** (the retracted-bar lesson from the sweeper): if a 1Min bar is
  retracted (in cache but not in current Polygon), the coarse bucket must be REPLACED (delete+insert
  the bucket's rows), because an upsert can't remove a bar that no longer exists. Scope the replace
  to `(symbol, tf, session, bucket)`.
- Reuse the `bar_cache.write_through_bars` gating pattern (flag + write-through) and the sweeper's
  day-chunked backfill CLI for historical seed.

## 6. Read path — cut consumers over ONE AT A TIME (each byte-identity-gated)
The store only pays off when consumers READ it instead of resampling. Cut over in this order, each
behind its own flag + the comparator (§7) + the fidelity parity suite:
1. **Backtest coarse-secondary build** — `strategy_data._build_coarse_secondary_from_1min`
   (`RORT_COARSE_SECONDARY_FROM_1MIN`). Lowest risk (offline, already gated). Read the store instead
   of resampling.
2. **Parity ribbon / chart-data-cache** — `services.prepare_strategy_window_df` /
   `get_strategy_chart_data_from_cache`. Kills the 32s on-the-fly resample. Read-only observability.
3. **LIVE coarse shadow** — `ralph_engine._close_shadow_with_bar` (the 313 fix site; where
   `RORT_MTF_COARSE_RTH_RELOAD` currently reloads via `_load_warmup_df`). **This cutover is the one
   that KILLS the 313 class** — the live shadow stops building its own coarse bar and reads the
   canonical store. Highest value, highest care.
- The `_build_native_1min_secondary` (#50) path for 1-minute-as-secondary is analogous; keep it
  consistent (native 1Min from `bar_cache`).

## 7. ⚖️ THE fidelity gate (this is what makes it succeed vs backfire)
This is "the append problem rebuilt in a store" — the exact class M-RS5's comparator exists for.
Without a byte-identity gate, the store INTRODUCES divergence instead of removing it.
- **Comparator mode** (`RORT_RESAMPLED_STORE_COMPARE=1`, default OFF): every store READ also computes
  the canonical resample on-the-fly and byte-compares (OHLCV exact, per bar). Log + alarm on ANY
  diff, per (symbol, tf, session). Ship the store with the comparator ON in shadow (read canonical,
  compare store) BEFORE any consumer trusts the store.
- **Promotion gates (each blocks the next):**
  1. Offline: store-built bars byte-identical to `resample_to_timeframe` across ALL tf × session ×
     the active symbols, over a multi-day + a Monday-after-holiday + a session-boundary window.
  2. Comparator mode green (zero diffs) for N trading days across the fleet.
  3. Consumer cutover #1 (backtest secondary): fidelity_parity_suite 18/18 (canary 267) with the
     store-read flag both OFF and ON; byte-identical trades.
  4. Consumer cutover #3 (live shadow): 313 (and 325/330/331/338) stay fixed reading from the store;
     the coarse-gate probes become the regression test. Live gate telemetry matches the store.
- **Default-OFF, kill-switch, reversible** throughout (the bug-hunt reversibility gate).

## 8. Admin page (Kevin's "another admin page")
Mirror the Bar Cache admin view: per `(symbol, tf, session)` show coverage span, freshness
(`revised_at` of the edge bar), last re-resample, and — the money column — the **comparator diff
rate** (store vs canonical) so drift is visible before it bites. This IS the observability that
proves the store is honest.

## 9. Expected impact
- Fidelity: the gate-CONSTRUCTION divergence class (313/272/325/338 family) becomes structurally
  impossible → fewer bug-hunts, new coarse-gated strategies "just work."
- Speed: ribbon coarse windows instant; backtest coarse-secondary build cached; recompute lighter.
- NOT the WS-vs-REST primary-timing floor (unchanged — by design).

## 10. Effort & sequencing
Weeks, not days. Build store schema + builder (window-REPLACE) → comparator → seed/backfill → cut
consumers one at a time behind flags + comparator + parity suite (backtest → ribbon → live shadow).
The live-shadow cutover retires 313's class and is the milestone. Keep the coarse-fix flag
(`RORT_MTF_COARSE_RTH_RELOAD`) as the interim guard until the store-read cutover proves out — then
the store SUBSUMES it (the reload becomes "read the canonical store," which is the same answer).

## Integration points (concrete, for the build session)
- `data_loader.resample_to_timeframe`, `_RESAMPLE_RULES`, `TF_LABELS`, `get_tf_label`/`get_tf_from_label`
  (canonical resampler + label namespace; mind the `'1M'` sentinel / `'1m'` secondary distinction).
- `bar_cache.write_through_bars`, `settle_sweeper` (Phase-1 store + maintenance model to extend).
- `strategy_data._build_coarse_secondary_from_1min`, `_build_native_1min_secondary` (backtest coarse
  + 1-min-secondary build = read-path consumer #1).
- `services.prepare_strategy_window_df`, `get_strategy_chart_data_from_cache` (ribbon = consumer #2).
- `ralph_engine._close_shadow_with_bar`, `_load_warmup_df`, `_is_in_session`, `_SESSION_HOURS`,
  `RORT_MTF_COARSE_RTH_RELOAD` (live coarse shadow = consumer #3, the 313 site).
- `fidelity_parity_suite.py` (canary 267) — the promotion gate. Note the 2 pre-existing SPY/10Sec
  cache-parity fails are orthogonal; isolate them, don't block on them.
