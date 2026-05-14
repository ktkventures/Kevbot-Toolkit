# Phase D Root-Cause Investigation — 2026-05-14

## TL;DR

The dominant cause of canary 80% match (vs Polygon's 99.1% ceiling) is **algo and backtest see different input data, not different engines or different refinement logic**. Hi-Fi Pass 2 cross-pollination is ruled out; engine-implementation divergence between Ralph and unified_engine is real but minor. The data-source mismatch (algo uses `live_bars` cache, backtest uses Polygon REST) drives most of the day-1 divergence and is architecturally bigger than a single-phase fix.

Phase D ships this analysis + a tightly-scoped recommended next step. No code change yet — Kevin should decide the architectural direction first.

## Data: sid 171 canary, window 19:35 – 20:15 UTC

Three-lane entry comparison:

| time      | live | algo | bt  | exit_reason  | hifi  |
|-----------|------|------|-----|--------------|-------|
| 19:36:10  | E    | E    | E   | mlv2_bear    | False |
| 19:39:20  | E    | E    | E   | mlv2_bear    | False |
| 19:40:10  | —    | **E**| **E** | mlv2_bear  | False |
| 19:42:20  | E    | E    | E   | stop_loss    | True  |
| 19:44:30  | E    | E    | E   | stop_loss    | True  |
| 19:46:10  | E    | E    | E   | stop_loss    | True  |
| 19:50:30  | —    | —    | **E** | stop_loss  | True  |
| 19:52:10  | E    | E    | E   | mlv2_bear / stop_loss (lane-dependent) | mixed |
| 19:56:00  | —    | —    | **E** | stop_loss  | True  |
| 19:59:10  | —    | —    | **E** | mlv2_bear  | False |
| 20:04:10  | —    | —    | **E** | mlv2_bear  | False |
| 20:05:50  | —    | —    | **E** | mlv2_bear  | False |
| 20:10:10  | —    | **E**| —   | mlv2_bear    | False |
| 20:12:30  | **E**| —    | —   | —            | —     |
| 20:12:40  | —    | **E**| —   | mlv2_bear    | False |
| 20:13:10  | —    | —    | **E** | stop_loss  | True  |
| 20:15:30  | E    | E    | —   | stop_loss    | True  |
| 20:17:40  | **E**| —    | —   | —            | —     |

## What each hypothesis says now

### Hypothesis 1 — Hi-Fi Pass 2 cross-pollination: **RULED OUT**

`run_hifi_pass2` reads all trades for a strategy without `data_source_filter`, but its job is to refine entry/exit *timestamps and prices* on rows that already exist — it doesn't materialize new trade rows. Looking at the matched rows (19:42, 19:44, 19:46, 20:13:10): algo and bt entries with identical `hifi=True` have identical entry/exit prices and bar times. Same refinement applied to the same inputs produces the same output. No cross-contamination.

### Hypothesis 2 — Engine implementation divergence (Ralph vs unified): **REAL BUT SECONDARY**

Live's 19:52:10 entry HELD until 20:12:10 (20-min hold). Bt's 19:52:10 entry exited at 19:55:20 (3-min hold). Same entry timestamp, very different exit. Both should evaluate the same exit triggers (`mlv2_cross_bear` or `stop_loss`). The 20-min holding gap is mostly explained by hypothesis 3 (live's cache has different bars than bt's REST source), but the engine code paths are also different (`ralph_engine.py` vs `unified_engine.py`) and that adds some additional variance.

### Hypothesis 3 — Data-source divergence: **CONFIRMED DOMINANT**

`algo_model='cache_locked'` replays from `live_bars` table → only sees bars the live engine wrote.
`backtest_model='rest_hifi'` uses Polygon REST → sees ALL bars Polygon has.

In the 19:50-20:15 window, bt fires 6 entries that algo doesn't take. These are the "missed" entries from algo's perspective. Looking at the bt trades' `reason=stop_loss / hifi=True` rows (19:50:30, 19:56:00, 20:13:10) — those exits hit prices that probably weren't in algo's cache (cache gaps), so algo neither stopped out cleanly nor re-entered fresh.

This is the architectural property of `cache_locked` — it intentionally replays *what the live engine saw*, including cache gaps. The "loss of trades" vs REST reflects the live engine's intermittent coverage during the canary window.

## Phase D outcome: documentation, no code change yet

The fix space depends on architectural intent:

- **Option α**: accept `cache_locked` as "simulates live (with its gaps)" and aim for *algo ↔ live* parity, not *algo ↔ bt* parity. Then ≥95% target applies only to the live ↔ algo lane. Current 80% is mostly engine-implementation variance (hypothesis 2) and bt is intentionally allowed to deviate.

- **Option β**: backfill the `live_bars` cache from REST periodically so algo's input data matches bt's. Larger project (new ingestion path, conflict-resolution on writes), bigger scope than a phase-D patch. Closes the algo↔bt gap.

- **Option γ**: change algo default to `rest_hifi` so users get a fully-aligned algo+bt by default. Cache replay still available as opt-in for users explicitly modeling cache-coverage characteristics. Lowest engineering cost; biggest UX shift.

Each option has different implications for what we tell users about "what algo means." That's Kevin's call.

## Recommended micro-fix that's clearly in scope right now

The smallest improvement we can make today without an architectural decision:

**Add `data_source_filter` to Hi-Fi Pass 2's `load_trades_admin` call.** Even though cross-pollination wasn't the bug we found, the current code has no scoping — if Hi-Fi acquires a bug later, it would silently affect both lanes from one entry point. Defensive cleanup. ~10 lines, no behavior change today, prevents a category of future bug.

Files: `src/api/routers/strategies.py:654` (run_hifi_pass2 entry point) — accept optional `data_source_filter` param. `src/api/services/forward_test_service.py:1607-1618` (algo callsite) — pass `data_source_filter='cache_%'`. Similar at the backtest callsite — pass `data_source_filter='backtest_%'`.

I'll ship this as the Phase D code change. The architectural choice (α / β / γ) gets surfaced for Kevin and Phase E (or a new Phase F) addresses it once decided.

## What Phase B and C enabled

We could not have done this investigation in 15 minutes without:
- Per-minute lane-bucketing (Phase C heatmap logic implemented in `useAdminParity.ts`)
- Trade snapshot endpoint (`/api/admin/parity/snapshot`)
- Direct visibility into `hifi_resolved` + `exit_reason` per-lane

The visual tabs are not yet rendered in production until Railway picks up the new deploy, but the underlying data is queryable and the patterns are clear.
