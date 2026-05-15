# Phase G — Live Engine Canonical Filter (2026-05-15)

## Context

Phase E shipped the Admin > Parity > Ticks tab + observable-bar ingestion. Last night Kevin used it for the diagnostic comparison we'd been building toward: **Cache ≠ Observable, but REST ≈ Observable (≥99%)**. That isolates the root cause of the live↔algo↔backtest divergence we've been chasing for weeks:

- Observable bars (rebuilt from Polygon's flat-file trades using Polygon's canonical condition rules) ≈ REST (settled aggregates). Confirms our flat-file pipeline is correct.
- Cache bars (what `ralph_engine.py` writes to `live_bars` from WebSocket trades) **disagree drastically with both**.

The live engine is the source of the drift. Algo lane (`cache_locked`) faithfully mirrors what live engine wrote, so it inherits the same bad bars. Backtest (`rest_hifi`) uses settled REST data so it's cleaner — and that's why algo vs backtest has looked broken all this time.

**Suspect ranking (set yesterday)**:
1. **Condition-code translation gap** — `ralph_engine.py:136` defines `EXCLUDED_TRADE_CONDITIONS` as CTA/UTP **letter codes** (`'B','C','I','T'`). Polygon's WS trade events emit **numeric** condition codes. If they never translate, the filter never matches → cache includes every trade, including Form T / Odd Lot / Cash Sale / Out-of-Sequence prints that Polygon REST and Observable both exclude from OHLC. This single bug is enough to explain the divergence pattern.
2. **Filter scope wrong** — even with translated codes, the live engine's exclusion list doesn't match Polygon's canonical OHLC-eligibility list.
3. WS message loss, bar-boundary timing, etc. — deferred until we rule out #1+#2.

**Intended outcome**: replace ralph_engine's filter with the same canonical Polygon rules `flat_file_ingestion.py` uses (Phase F.2/F.4). After deploy, Cache should converge toward Observable. We verify via the Bars Comparison tab.

## Strategy

- **G.0 (Frontend, no risk)**: add a TF selector to Bars Comparison so we can compare at 10Sec, 30Sec, 1Min, 5Min, 15Min — not just minute. Lets us see exactly which timeframes the live engine breaks at. Sub-minute TFs disable REST as a source (Polygon REST aggs are 1Min minimum).
- **G.1 (Backend, the actual fix)**: replace `ralph_engine.py`'s letter-code filter with a canonical Polygon numeric-code filter. Gated by `RALPH_USE_POLYGON_CANONICAL_FILTER` env var, **default ON** per Kevin's request — start gathering insights immediately. Rollback is env-var flip (~3 min worker restart, no redeploy needed).
- **Verification immediately after deploy**: open Bars Comparison tab, pair Cache vs REST at multiple TFs, check close-match % moves from low → high.
- **Tomorrow's cron** ingests 5-15 observable. Then verify Cache vs Observable at multiple TFs as the next-day authoritative check.

## Standing Practices (continued)

- **Backup branch**: `dev-backup-pre-phaseG-2026-05-15` before any code edits.
- **Single commit per phase boundary** (G.0 ships first, then G.1).
- **Worker restart consequences**: setting the env var via `railway variables --service Worker --set ...` (without `--skip-deploys`) auto-redeploys the Worker. Kevin OK with this since no active test is running.
- **Fractional-share filter**: deferred indefinitely. Per-minute REST/Observable parity is 98-99.8% already; day-aggregate gap is ~18% from fractional shares which only matter for daily volume reporting. Documented as "low priority follow-up" — not Phase G work.

## Phase G.0 — Bars Comparison TF Selector

**Goal**: let user pick any of {10Sec, 30Sec, 1Min, 5Min, 15Min} for the Bars Comparison tab. Sub-minute TFs auto-disable REST as a source (Polygon REST aggs API only goes to 1Min).

**Files (edited)**:
- `frontend/src/charts/ParityBarComparison.tsx` — single file change

**Changes**:
- Add `tfSeconds` state, default 60 (1Min, current behavior)
- Add TF dropdown alongside left/right source dropdowns
- Generalize `bucketByMinute(bars)` → `bucketByTf(bars, tfSeconds)` — same logic, parameterized
- Update `buildRows()` to call `bucketByTf` with selected TF
- When `tfSeconds < 60` and either left or right source is REST: show inline note "REST not available below 1Min" and force the dropdown back to a valid source
- Diff-table column headers + minute-display use selected TF
- Charts still render at native granularity (cache 10Sec, observable 1Sec, REST 1Min) — the bucketing is for diff-table apples-to-apples comparison

**Reuse**: existing `useStrategyCacheBars`, `useObservableBars`, `useBars`, `TradingChart` primitives.

**Verification**: navigate to `/admin/parity > Bars Comparison`. Pick sid 169 + 5-13 13:30-14:30 window. Cycle TF dropdown through 10Sec, 30Sec, 1Min, 5Min. For sub-minute TFs: REST option disabled with note. For 1Min+: all three pairs available. Close-match % should remain high for Observable vs REST across all TFs ≥ 1Min.

## Phase G.1 — Live Engine Canonical Filter

**Goal**: replace ralph_engine's letter-code condition filter with Polygon's canonical numeric-code filter, mirroring `flat_file_ingestion.py`'s logic.

**Step 1 — Extract shared module**:
- New file: `src/polygon_conditions.py` — houses `_load_polygon_condition_rules()`, `_parse_conditions()`, `_classify_eligibility()`, and the fallback exclusion sets
- Update `src/flat_file_ingestion.py` to import from the new module (delete its local copies)
- This keeps the SAME logic in BOTH places by construction. Bug fixes in one apply to both.

**Step 2 — Replace ralph_engine filter**:
- `src/ralph_engine.py:136-155` — keep `EXCLUDED_TRADE_CONDITIONS` (the letter-code set) for now as a fallback, but mark it deprecated
- `src/ralph_engine.py:~3290-3293` — replace the filter logic. New logic:
  - Import `_classify_eligibility` from `polygon_conditions`
  - At the trade-handler, read `trade.conditions` (the actual format will be discovered at runtime — could be list of letters, list of ints, or string)
  - **Defensive normalization step**: convert whatever format we get into a set of ints. If conditions arrive as letters (likely a legacy/Alpaca-style stream), fall back to old letter-code filter. If conditions arrive as ints (Polygon canonical), use new filter.
  - Apply two-tier filter: `ohlc_ok, vol_ok = _classify_eligibility(conditions)`
  - If `not ohlc_ok and not vol_ok`: skip entirely (rare)
  - If `not ohlc_ok and vol_ok`: pass to bar builder with volume-only flag
  - If `ohlc_ok`: pass through to bar builder normally
- `BarBuilder.process_tick` or `PartialBar` needs a `volume_only=True` parameter — currently it only takes (price, volume, timestamp) and always updates both. Mirror the `_Bar.update(update_ohlc, update_volume)` pattern from `flat_file_ingestion.py:_Bar`.

**Step 3 — Feature-flag gate**:
- New helper `_polygon_canonical_filter_enabled()` in ralph_engine.py near the existing flag helpers (line ~993-1034)
- Reads env var `RALPH_USE_POLYGON_CANONICAL_FILTER`
- **Default ON** per Kevin's request — `os.environ.get('RALPH_USE_POLYGON_CANONICAL_FILTER', 'true')`
- When ON: use new canonical filter + split eligibility
- When OFF: use existing letter-code filter (unchanged behavior, rollback path)

**Step 4 — Deploy + verify**:
- Push to dev → Railway auto-redeploys api + Worker (since they share the same image and Worker reads env vars from the same source)
- Wait ~3 min for Worker to restart
- Bars Comparison tab: pick a SPY strategy, recent window (last 30 min of post-deploy bars), Cache vs REST
- Close-match % should be markedly higher than what we saw last night
- If breaks (no alerts firing, errors in logs): set `RALPH_USE_POLYGON_CANONICAL_FILTER=false` on Worker service → restart → original behavior restored

**Files (new)**:
- `src/polygon_conditions.py` — shared canonical filter module

**Files (edited)**:
- `src/flat_file_ingestion.py` — import from `polygon_conditions` instead of inline
- `src/ralph_engine.py` — new flag helper, replace filter in trade handler, add `volume_only` flag to bar builder

## Critical Files to Read First

| Phase | Read first |
|---|---|
| G.0 | `frontend/src/charts/ParityBarComparison.tsx` (full file — current bucketByMinute pattern) |
| G.1 | `src/ralph_engine.py:130-156` (current filter), `src/ralph_engine.py:3260-3310` (apply site + on_polygon_trade flow), `src/ralph_engine.py:990-1040` (existing feature-flag patterns), `src/flat_file_ingestion.py:60-210` (the canonical filter logic we're sharing), `src/ralph_engine.py:270-310` (BarBuilder.process_tick to identify where the volume_only flag plugs in) |

## Verification Plan (end-to-end)

**Immediately post-deploy** (no observable data needed):
1. Bars Comparison tab, sid 169, window = last 30 min of bars after Worker restart
2. Cache vs REST at 1Min → close-match % (was ~3% in last night's read)
3. Cycle to 5Min, 15Min — confirm match % holds at higher TFs

**Tomorrow morning** (after 5-15 observable cron runs):
1. Bars Comparison tab, sid 169, window = 5-15 13:30-14:30 UTC
2. Cache vs Observable at 1Min, 5Min, 10Sec (the most diagnostic pair)
3. Close-match % should be ≥95%
4. If still low: log inspection on ralph_engine to confirm canonical filter is engaging, possibly debug specific divergent bars

**Rollback criteria**:
- Live alerts stop firing (check `alerts` table count over a 5-min window)
- Errors in Worker logs related to bar building
- Cache match % gets WORSE not better

**Rollback procedure**:
```bash
railway variables --service Worker --set RALPH_USE_POLYGON_CANONICAL_FILTER=false
# Worker auto-redeploys; original behavior restored ~3 min
```

## Out of Scope (this plan)

- Fractional-share volume contribution (Phase F.5, documented, deferred)
- X / + / hollow-circle entry markers (Phase F.2, deferred — needs canvas overlay)
- WS message-loss investigation (suspect #3, deferred until G.1 outcome assessed)
- Bar-boundary timing investigation (suspect #4, same as above)
- Phase G.2: any follow-up fixes if G.1's filter change alone doesn't close the gap
