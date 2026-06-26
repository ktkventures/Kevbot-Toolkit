# M-RS2 Phase 3 — Algo / Live Bars lane load-once read (design)

**Status:** DESIGN (2026-06-25). Implements board #43. Mirrors the REST Bars
read-path win, applied to the **algo lane** (which reads **Live Bars**, not REST).

## The opportunity (Kevin's idea)
Phase 2 sped up the **backtest lane** (reads REST Bars / `bar_cache` via the new
`read_bars()` direct-PG primitive). The **algo lane** — the lane that reproduces
what the engine saw *live*, for divergence/forensics — still reads **Live Bars**
(`live_bars`) the slow way, **once per strategy**. Multiple strategies on the same
symbol (TSLA=27, SPY=36) each re-read the *same* `live_bars` rows independently in
a full Update-All. Dedup that → faster Update-All.

## How the algo lane reads bars today (verified by recon, re-verify before coding)
- Entry: `append_new_trades_for_strategy()` (`forward_test_service.py:~1040`) →
  `algo_model = cfg.algo_model or cfg.backtest_model` (~1261) → engine run with
  `model_override=algo_model`, `diagnostics_source='algo'` (~1299–1328).
- → `get_strategy_trades[_for_window]` → `prepare_data_with_indicators` →
  `_resolve_primary_df_for_backtest_model` (`services.py:~120–187`): for a
  cache-backed model (`cache_locked`/`cache_corrected`) it calls
  **`fetch_cache_as_df(symbol, tf_seconds, start, end, sources=ENGINE_CONSUMED_SOURCES)`**.
- `fetch_cache_as_df` (`data_loader.py:~963–1051`) queries **`live_bars`** via
  **PostgREST** (supabase-py), paginated 1000 rows/page, filtered by
  `source IN ENGINE_CONSUMED_SOURCES`, returns a **UTC-indexed OHLCV** DataFrame
  (`['open','high','low','close','volume']`, float64) — **same shape as
  `bar_cache.read_bars()`**.
- Secondary TFs: another independent `fetch_cache_as_df` call each.

## The change (two parts, both low-risk if gated + byte-identical-gated)
1. **`read_live_bars()` — direct-PG reader for Live Bars.** A `read_bars()` twin
   that hits **`live_bars`** instead of `bar_cache`: direct psycopg + the SAME
   `source IN ENGINE_CONSUMED_SOURCES` filter + the SAME `(symbol, timeframe_seconds,
   bar_start)` predicate, returning the SAME UTC-indexed OHLCV. Drop-in for
   `fetch_cache_as_df`'s read. (Live `live_bars` schema uses `timeframe_seconds`
   + `source`; `bar_cache` uses `timeframe` text + no source — so this is a
   distinct query, not a parameterization of `read_bars`.)
2. **Cross-strategy share in the Update-All loop.** Before iterating strategies,
   collect the unique `(symbol, tf_seconds, window)` set and preload each once into
   a process-scoped dict; each strategy slices its window from the shared frame.
   (Same "prime once, slice many" shape as the Hi-Fi load-once.)

## HARD constraints (the fidelity red lines)
- **Reads Live Bars, never REST.** The algo lane's whole value is decision-time
  fidelity (WS-visibility forensics). The reader must NOT substitute Polygon/REST
  data. (See `Two_Bar_Caches_DEFINITIONS.md`.)
- **Same source filter.** Must apply `ENGINE_CONSUMED_SOURCES` identically, or the
  algo lane would consume `rest_backfill` rows it currently excludes → divergence.
- **Coverage fallback unchanged.** `live_bars` only covers the live-recording era;
  a window predating coverage returns empty today → falls back to REST. Keep that
  exact behavior (the shared preload returns empty for the old portion; fallback
  still fires per-strategy).

## Validation gate (same rigor as Phase 2 — non-negotiable)
- Byte-identical A/B: `read_live_bars()` (direct-PG) vs `fetch_cache_as_df()`
  (PostgREST) over the same `(symbol, tf, window, sources)` — exact row count +
  OHLCV to a sane threshold, on RTH + extended, primary + secondary TFs.
- End-to-end: algo-lane recompute of a canary (e.g. 321) with the reader ON vs OFF
  → byte-identical algo trades (entry/exit ts, price, reason, pnl), like the 321
  Hi-Fi A/B. Then a multi-strategy same-symbol run to confirm the shared frame
  matches per-strategy reads.
- Feature flag (e.g. `LIVE_BARS_DIRECT_READ`) + transparent PostgREST fallback.

## Tonight feasibility — RECOMMENDATION: design now, build + validate as its own watched change; do NOT activate under tonight's unattended overnight run
- **Tonight's overnight Update-All is already safe without this:** it uses the
  *validated* REST Bars backtest path; the algo lane stays on its *current*
  PostgREST path (unchanged). So there's **no stacking** of two new paths tonight.
- The reader is low-risk in isolation, **but activating it changes the live
  decision lane's read path** — exactly the fidelity-critical surface to validate
  watched, not flip on under an unattended overnight run (today's lesson: don't
  stack unvalidated fidelity-critical changes).
- **Plan:** (a) build `read_live_bars()` + the share + the byte-identical A/B in a
  focused block; (b) gate behind `LIVE_BARS_DIRECT_READ`; (c) validate on 321; (d)
  activate watched (markets-closed window is fine, but with eyes on it). Could be
  tonight *if* we're watching it — just not silently under the unattended run.

## Open question to resolve during build
- Confirm `fetch_cache_as_df` is the ONLY algo-lane bar read (no second path), and
  that `model_override='cache_locked'` is what the fleet's `algo_model` resolves to
  (some strategies may have a different `algo_model`). Re-verify the recon claims
  against the code before coding (my recon had a couple of self-corrections).
