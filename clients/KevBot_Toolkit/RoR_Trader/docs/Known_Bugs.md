# Known Bugs

A living log of bugs we've identified but not yet fixed. Each entry lives
here until either (a) shipped + verified, or (b) explicitly deprecated.

**Status legend:** `OPEN` (known, not started) · `WIP` (being worked on) ·
`FIXED-PENDING-VERIFY` (shipped, awaiting prod confirmation) · `CLOSED`
(verified fixed) · `WONTFIX` (deliberate, with rationale)

> Convention: when a bug is closed, move it to the bottom of this file
> under "Closed" — keep history but stop scrolling the active list.

---

## Active

### REST Bars cache UPSAMPLED 1Min→sub-minute → wrecked 10Sec/15Sec backtests (SEV)
- **Status:** ✅ FIXED-PENDING-VERIFY (2026-06-26, commit b8bac23). Cache disabled in prod (`BAR_CACHE_ENABLED=0`) while verifying; fix is deployed + load-level byte-identical. Re-enable after a final check.
- **Discovered:** 2026-06-26 — Kevin re-ran UAD on sid 275 (TSLA 10Sec) as a divergence test; backtest collapsed **193→29 trades** (~6.7× = the 1Min→10Sec upsample factor) and divergence deltas blew from seconds to minutes.
- **Root cause:** `bar_cache.cached_load_market_data` chose `cache_tf="1Min"` for ANY TF not natively cached, then `resample_to_timeframe(1Min → 10Sec)`. That's **upsampling** (coarse→fine), which `resample_to_timeframe` (downsample-only) can't do — it yields sparse/garbage bars, so most of a sub-minute strategy's trades vanish. Hit any **sub-minute TF not captured natively**: TSLA 10Sec/15Sec, all SPY sub-minute. TSLA **30Sec survived** only because it's captured natively (read directly, no resample) — which is why 325 looked fine and masked the bug.
- **Also re-explains:** 321 (15Sec) "597→198" earlier attributed to its canary window was actually THIS bug (cache-off=799, cache-on=198 ≈ 799/4, the 15Sec upsample factor). Mis-attributed.
- **Fix:** choose `cache_tf` **finer-or-equal** to the request, never coarser. Sub-minute → downsample from native **1Sec** (validated byte-identical to Polygon native 10/15Sec **including volume**, 06-12/16/22). 1Min-or-coarser → native 1Min. Hard guard bails to Polygon if the chosen layer is absent or coarser. (1Min is NOT derived from 1Sec — its volume diverges 822/944 bars, so it stays a native layer; that part of the M-RS2 design was correct.)
- **Validation:** load-level A/B `cached_load_market_data` vs native `load_from_polygon` — 10/15/30Sec/1Min all byte-identical (OHLCV, RTH + 24/7).
- **Cleanup:** only **275 + 321** were recomputed while the cache was on → re-run them (cache off, or after re-enable) to restore correct backtests. 325 (30Sec) was unaffected.
- **Why my validation missed it:** the 325 byte-identical check used a *natively-cached* TF (30Sec); the 321 A/B was cache-*off*. So the uncached-sub-minute upsample path was never exercised. **CI parity must cover cache-on==off for EVERY (symbol, TF) — especially uncached sub-minute.**
- **Follow-up consideration:** sub-minute primary loads now read 1Sec (~6× more rows than native) → heavier memory/time; the cache's clear win stays the 1Min/coarse loads + Hi-Fi. Consider capturing native 10/15Sec, or bailing sub-minute primaries to Polygon, if memory/latency bites.

### Strategy KPIs double-counted (BT+ALGO) on append (`mode='new'`) recomputes
- **Status:** ✅ FIXED-PENDING-VERIFY (2026-06-26, commit 6df2eb1) — verify by re-running an affected strategy and confirming KPIs match the single-lane (backtest_%) trade count.
- **Discovered:** 2026-06-26 — Kevin re-ran UAD on sid 325; `total_trades`/Daily R/TPD/Max DD all **halved** (884→442, TPD 9.8→4.9, Daily R 4.13→2.14, Max DD -41.2→-20.4R) while **Win Rate (35.5%) + Profit Factor (1.73) stayed put**. That ratio-stable/count-halved pattern = textbook double-count.
- **Root cause:** `append_new_trades_for_strategy` (the **algo-append** lane, `forward_test_service.py` ~1510) recomputed the strategy's headline `kpis` via `load_trades_kpi_fields_admin(strategy_id, user_id)` with **no `data_source_filter`** → loaded BOTH lanes (`backtest_%` + `cache_%`). Once the algo lane began mirroring the backtest lane, every trade was counted **twice**. Since `mode='new'` runs BT-append (correct, `backtest_%`) *then* algo-append (this, all-lanes), the doubled value **overwrote** the correct KPIs. A code comment even asserted "no data_source filter — all lanes … KPIs byte-identical" — true once, false after the lanes converged.
- **Fleet evidence (2026-06-26):** un-rerun strategies showed `kpis.total_trades` = BT+ALGO (313: 978=484+494; 314: 660=330+330); strategies re-run via `mode='all'` showed the correct single lane (325: 442=BT; 321: 198=BT). `mode='all'` (line 252, `calculate_kpis(all_trades)` over the freshly-computed backtest set) was never affected.
- **Fix:** scope that call to `data_source_filter='backtest_%'` (headline KPIs are "backtest truth" per the module docstring; matches the BT-append path's KPI source).
- **Cleanup expectation:** every strategy last touched by an append (`mode='new'`) run still shows ~2x-inflated count-based KPIs until re-run. As the fleet gets re-run (or appended) on the fixed code, those will correct to true values (ratios unchanged). **Not a regression** — pre-existing display bug being cleaned up.
- **NOT caused by the M-RS2 cache / M-RS1 warmup work** — surfaced by a re-run, but the bug predates today.

### Newer strategies appear to not trigger live (alerts) — investigate user-pack registry on the live worker
- **Status:** OPEN — observation, not yet root-caused (logged 2026-06-25 from Kevin's report + corroborating context)
- **Symptom:** Some of the newer strategies (the 308–337 cohort, mostly 15Sec/30Sec) seem to not be firing live alerts / not triggering as expected. Kevin has independently noticed this.
- **Leading hypothesis — user-pack registry:** the newer strategies disproportionately use **user packs** (e.g. `ut_bot_v4`, `stoch`, multipack — see the TEST-P2-* and Mass cohorts), whereas the older fleet uses built-in triggers. If the **live worker's** pack registry fails to fully load at startup (`pack_registry.scan_and_load_all()` — api/main.py:~103 / worker startup), or a specific pack isn't registered, then user-pack **entry triggers silently fire ZERO** → 0 alerts, no error. This is the exact failure mode proven locally (see below) — just on the live worker instead of a script.
  - **Check:** grep the live worker startup logs for `scan_and_load_all() -> N packs loaded` and confirm N is the full expected count; confirm the specific packs the silent strategies use (`ut_bot_v4` etc.) are in the registry; cross-check which silent sids are user-pack-based vs built-in.
- **⚠️ Do NOT conflate with a local-harness artifact (2026-06-25):** during the M-RS2 Hi-Fi work, `get_strategy_trades()` returned 0 for 321/308/309/325 in an ad-hoc script — that was the SAME registry trap but in a *local* script that forgot `scan_and_load_all()`. Once the script loaded the registry (and used `get_strategy_by_id_admin` for the right strat shape), 321 produced **799 trades**. So the local 0 was a harness bug, NOT evidence the engine is broken. It IS, however, a live re-confirmation that "registry not loaded → user-pack triggers fire 0" — which is why it's the leading hypothesis for the live symptom. See `src/local_update.py` header (the 0-trades trap) + `feedback_local_script_pack_registry`.
- **Other angles to rule out:** (a) `forward_test_start` recency — newer strategies have recent starts (e.g. 321 = 2026-06-19), so a short live window naturally has fewer alerts; (b) confluence too tight (the strategy genuinely rarely triggers); (c) session/TOD gating (the TZ-TEST sids 334–337 are deliberately session-gated). Distinguish "correctly quiet" from "silently broken" before fixing.
- **Related:** existing silent-strategies finding (`project_silent_strategies_2026-05-06` — 6 MB mirrors fired 0/7d), `feedback_disabled_groups_kill_worker` (a disabled referenced group crashes the worker → whole-fleet silence).

### Manual Window Backfill over-produces ~17% extra trades vs UAD on same window
- **Status:** WIP — feature shipped, marked "do not trust" on the admin page; needs root-cause + fix
- **Discovered:** 2026-06-05 (parity test on sid 296 canary)
- **Symptom:** `append_windowed_backtest_trades_for_strategy` (mode='window' in `/api/update-jobs/run`) produces more BT trades than `recompute_and_persist_stored_trades` (UAD) does for the same `[window_start, window_end]` range.
- **Parity-test result on sid 296, window = [13:30, 20:00) UTC:**
  - Window-backfill inserted 433 in-window trades
  - UAD's clean recompute produced 359 in-window trades
  - Shared (identical entry/exit): 358
  - UAD-only: 1 (window-start edge case)
  - Window-backfill-only: **75** (the over-production)
- **Pattern in over-produced trades:** rapid-fire re-entry clusters, e.g.
  - 14:36:20, 14:37:50, 14:39:10, 14:40:30, 14:41:50, 14:42:50, 14:43:40 → 7 entries in 7 minutes
  - 15:07:30 → 15:14:30 → 8 entries in 7 minutes
- **Suspected root cause:** Tier 1 snapshot resume restores INDICATOR state from the envelope but forces POSITION state to FLAT (per Tier 3 §8.2 — `apply_backtest_snapshot` in `unified_engine.py` intentionally drops `envelope['position']`). What also fails to carry over is the post-exit cooldown counter (`last_exit_bar_count` from Phase 30I — "1-bar cooldown"). When the resumed engine exits a trade and the next entry signal fires immediately, it re-enters because cooldown count = 0 (cold-started), whereas UAD's continuous backtest carries the correct cooldown forward and suppresses the re-entry. Hypothesis — not yet root-cause-verified.
- **The same root cause likely affects Tier 2 (UAD-matched cold warmup):** the engine starts FLAT regardless of which warmup tier is used, so cooldown state is also cold there. (Wasn't exercised in the canary test — Tier 1 fired for both sid 263 and sid 296.)
- **Operator-visible warning:** WindowBackfillCard in `UpdateJobsAdminPage.tsx` displays a red "WIP — do not trust" banner explaining the parity gap.
- **Proposed fixes (ranked):**
  1. **Investigate + fix cooldown serialization** — walk PositionStateMachine state in `unified_engine.py`, find any bar-count-based state that isn't in the snapshot envelope, add it to `serialize_backtest_snapshot`. This preserves the perf win (snapshot resume = near-zero warmup).
  2. **Route through UAD path** — call `get_strategy_trades(strat)` (the full UAD code path) on the natural visible window, filter output to `[window_start, window_end]`. Guaranteed UAD parity by construction. Cost: full UAD load time (~minutes per strategy) — defeats the "fast targeted backfill" purpose. Use only if #1 proves too complex.
- **Until fixed, use UAD (Update All Data) to fill gaps.** UAD on sid 296 took 629s (~10.5min) tonight — slow but trustworthy.

### `alerts.live_model` field NULL on default-model strategies → rest_verifier silently skips them
- **Status:** ✅ FIXED-VERIFIED (2026-06-18) — both dispatchers now fall back to
  `get_default_live_model()` (shipped 2026-06-03, commit b2387ff): ralph_engine.py ~1836-1843,
  worker.py ~234-242. Verified live: sid 308's 257 alerts (24h) all carry
  `live_model='ws_rest_spliced'`, not NULL → new strategies 308-314 are correctly attributed +
  verified. (The 1572-1575 / 228-231 line refs below are stale pre-fix locations.)
- **Status (historical):** OPEN — root cause located; design intent ambiguous
- **Discovered:** 2026-06-04 (Kevin pushed back on Layer 1 N/A claim — pointed out `live_model=None` falls back to default)
- **Symptom:** Layer 1 (bar fidelity) in SOP is unusable for 89.7% of the fleet because `alerts.verification_status=NULL` everywhere.
- **Fleet impact (24h sample, 5000 alerts):**
  - `alerts.live_model` field values: `None: 4484 (89.7%)`, `ws_rest_spliced: 516 (10.3%)`
  - `verification_status` values: `None: 4495`, `verified: 339`, `corrected: 32`, `rest_unavailable: 115`, `gap_fill_unverified: 15`, `drift_uncorrected: 4`
  - **Only 11 sids ever get verified** (174, 194, 263, 265-273, 275 — Kevin's older strategies that have `config.live_model='ws_rest_spliced'` explicitly set, likely via `_bulk_set_live_model.py`)
  - All 30 PACKTEST canaries (276-305) miss verification entirely
- **Root cause located in BOTH dispatchers:**
  - `src/ralph_engine.py:1572-1575` (Ralph dispatcher)
  - `src/worker.py:228-231` (DBAlertDispatcher)
  ```python
  live_model_at_fire = (
      (strategy.get('config') or {}).get('live_model')
      or strategy.get('live_model')
  )  # ← Returns None when neither field set, even though engine resolves to default
  ```
  Both paths read `config.live_model` but DON'T fall back to `strategy_models.get_default_live_model()` (which returns `'ws_rest_spliced'` per the registry at `strategy_models.py:213-216`).
- **Downstream consequence:** `rest_verifier.py:401` short-circuits with `if alert.get("live_model") != "ws_rest_spliced": return`. Result: REST verification never runs for any alert with NULL live_model.
- **Design intent ambiguity:** worker.py:226-227 comment says "historical alerts stay null and render as 'unknown' per Kevin's 'honesty over speed' guidance". That makes sense for handling live_model UPGRADES (don't retroactively re-attribute), but it conflates "user hasn't decided yet" with "actually running default". For strategies that NEVER explicitly set live_model and rely on the default resolution at engine init, the alert SHOULD carry the resolved default.
- **Proposed fix:**
  ```python
  from strategy_models import get_default_live_model
  live_model_at_fire = (
      (strategy.get('config') or {}).get('live_model')
      or strategy.get('live_model')
      or get_default_live_model()
  )
  ```
  in both `ralph_engine.py:1572` and `worker.py:228`. Caveat: needs Kevin's sign-off because it changes alert attribution for ~4500 historical-from-this-point-forward alerts.
- **Alternative fix:** explicitly assign `live_model='ws_rest_spliced'` to all strategies via the `_bulk_set_live_model.py` script. More transparent (config explicitly carries the value) but requires the script to be re-run any time the default changes.
- **Impact on SOP:** until fixed, Layer 1 is uninformative for almost the entire fleet. Layer 3 (fill delta) and Layer 2 (pair rate) are the only usable layers for PACKTEST canaries.

---

### Backtest session filter NOT applied on `primary_df` injection path
- **Status:** OPEN — root cause located; fix not yet shipped
- **Discovered:** 2026-06-03 EOD via sid 268 (SPY-CANARY-10s-NoConf, `trading_session=RTH`)
- **Symptom:** strategy configured for RTH; **alerts respect RTH** (live engine filters correctly) but `backtest_*` rows in the `trades` table contain entries during extended hours (e.g. sid 268 last 30 backtest trades: 0 RTH, 30 extended — entries between 19:23-19:37 ET, well past 16:00 ET close).
- **Fleet audit (sample of last 10 backtest trades per strategy):** 13 of 14 `trading_session=RTH` strategies show extended-hour backtest trades. Only sid 266 was clean (5-min TF, few trades, not enough sample).
  - Worst offenders: sids 263, 267, 268, 270, 273, 275 — 10/10 backtest trades are extended-hours
- **Root cause located in `src/services.py:251-271`:**
  ```python
  use_injected = primary_df is not None
  ...
  if use_injected:
      df = primary_df.copy()         # ← NO session filter applied
  else:
      df = load_market_data(symbol, ..., session=session)  # REST path filters via _filter_session
  ```
  When `prepare_data_with_indicators` is called with an injected `primary_df` (the typical path for `data_worker_engine.run_store_fed_window`, which feeds bars from the live-bars cache), the `session` parameter is accepted but never applied. The REST fallback path correctly calls `load_from_polygon` which applies `_filter_session(df, session)` at `data_loader.py:563`.
- **Why live alerts are correct:** `ralph_engine.StrategyMonitor._is_in_session` filters per-bar before evaluating triggers, independent of the data source.
- **Proposed fix:** in `prepare_data_with_indicators`, when `use_injected` is True, also apply `_filter_session(df, session)` after the copy. The filter is idempotent (filtering an already-RTH df returns the same df), so safe even if a caller pre-filtered.
- **Backfill consideration:** existing extended-hour `backtest_rest_hifi` rows for RTH strategies need to be DELETEd from the `trades` table — the fix only affects forward writes.
- **Side effect on divergence metrics:** the extended-hour backtest trades inflate the "missed" count (backtest-only events with no matching alert) in `_remeasure_pair_rates_5s.py` and the strategy_health endpoint. Until the bug is fixed + backfill complete, RTH-strategy pair rates are pessimistic.

---

### Snapshot lag — algo cron not processing all canary strategies
- **Status:** OPEN — root cause hypothesized; needs verification
- **Discovered:** 2026-06-03 EOD via Kevin's UI observation (sid 302 last snapshot 3h 45min old) + fleet audit
- **Snapshot age distribution** (current as of 2026-06-04 00:13 UTC):
  - sids 276-295 (most PACKTEST canaries): 8-13 min old — fresh ✓
  - **sids 297-305 (Strat Assistant gate through VWAP v2 gate): 225-232 min old — STALE** (~3h 50min)
  - `last_recompute_until_ts` for the stale group ranges 356-358 min ago — i.e. that's when the cron last touched them
- **Smoking gun:** sids 297-305 ALL have engine_snapshot_at within 7 seconds of each other (~232.5 min ago). This is consistent with a single batch run that completed for 297-305 then stopped — the next run never reached them.
- **Hypothesis:** algo-history cron has a per-cycle bar/strategy budget and processes strategies in a fixed order (numeric sid?). When the cycle exhausts budget, later sids don't get touched. They wait until the next cycle, where the early sids consume the budget again. Higher-numbered sids never get reached.
- **Alternative hypothesis:** alerts-gate optimization in `append_new_trades_for_strategy` (line 789 of `forward_test_service.py`) skips strategies with no recent exit alerts. Many gate canaries (odd-numbered sids 297, 299, etc.) have few/no exit alerts → skipped indefinitely.
- **Impact on divergence metrics:** stale-snapshot canaries show 0% pair rates with 100% phantoms in the divergence report — alerts fire but backtest hasn't produced corresponding trades. Not a real divergence; just a measurement artifact.
- **Investigation steps:**
  1. Check the cron's `append_new_trades_for_strategy` code path for budget/timeout caps
  2. Run a manual `recompute_and_persist_stored_trades` on sid 302 and time it — if it takes >5 min, that's why the cron skips it
  3. Look at worker logs for `[ALGO-APPEND]` lines on sids 297-305 — if they're absent or only in `recently_processed` skip path, that confirms the skip pattern

---

### 15 swing-stop strategies silent in live engine since 2026-06-03T19:58 UTC
- **Status:** OPEN — root cause unknown; manual worker restart does NOT heal
- **Discovered:** 2026-06-03 EOD via Kevin's report that sid 270 had no alerts since 13:58 MDT
- **Affected sids (15):** 136, 174, 194, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275 (sid 274 never fired — separate case)
- **100% correlation:** every silent strategy uses `stop_config.method='swing'`. Every still-firing PACKTEST canary uses `method='atr'`. No exceptions either way (15-of-15 vs 19-of-19 in sampled fleet).
- **Symptoms in `monitor_status.engine_state.positions[sid]`:**
  - `status='FLAT'`, `entry_trigger=''`, `entry_bar_count=0`, `last_exit_bar_count=0`
  - i.e. the PositionState is at its default — NO entries OR exits since the manual restart at 21:10 UTC
  - For comparison, sid 302 (PACKTEST UT Bot V4 trigger, also SPY 10sec): `entry_bar_count=14297`, firing continuously
- **What's been ruled out:**
  - Worker deploy churn: silence started at 19:58 UTC in the deploy-free clean window (18:08-20:27 UTC)
  - Watchdog `os._exit(2)`: no position_carryover record at 19:58; full restart didn't happen
  - Today's commits: none touch swing-stop, StrategyMonitor instantiation, or bar feeding
  - Polygon WS subscription drop: PACKTEST canaries on SAME symbol (SPY) keep firing
  - Manual `railway redeploy --service "Data Worker"` at 21:10 UTC: did NOT heal the affected sids
- **What's NOT ruled out:**
  - Phase 2 codec snapshot-restore failing silently for these sids → some user-pack engine starts in non-firing state but fresh enough to not error
  - Bar feed reaches sid 270's monitor but its process_bar entry path is rejecting silently somewhere we haven't logged
  - Some accumulated state in the worker process that survives across deploys via something other than monitor_status (unlikely but possible)
- **Defense added 2026-06-03 (`69c3f7f`):** `SwingStop.compute` no longer falls back to `ATR × 1.5` on empty buffer; raises `StopComputationError` instead, which the engine's `check_entry` catches and logs as `[stop-comp] sid=X entry rejected: ...`. Going forward any silent stop-comp failure will leave a clear log trail. Does NOT directly fix the 15 silent strategies (they don't appear to be hitting the stop-comp path at all — they're stuck upstream).
- **Backup branch:** `dev-backup-2026-06-03-eod` at `69c3f7f` — last known state with sid 278/280 EMA PP fix, watchdog, opposite_signal fix, and ATR-fallback removal all in place.
- **Next steps for tomorrow:**
  1. Try toggling sid 270's monitor Disable→Enable from the UI. If it starts firing alerts, the bug is in strategy reload (not init).
  2. Look at the data_worker engine reload path for any code that treats swing-stop strategies differently (a code path that bails on `method='swing'` would be the smoking gun).
  3. Consider whether the Phase 2 codec persists `_high_low_buffer` correctly — if not, swing strategies restored from a snapshot would have empty buffer, which would now LOUDLY error in logs (`[stop-comp]` lines).
  4. If 1-3 don't pan out, more aggressive: full worker restart in pre-market hours (before 8 AM UTC tomorrow) so Tier 3 §8.2 carryover is clean.
- **Operational risk:** these affected strategies include Kevin's primary test bench (TEST-P2-*, TSLA-CANARY-*, SPY-CANARY-*). PACKTEST canaries are unaffected so divergence remeasurement work can still proceed.

---

### ralph_engine watchdog — needs different semantics than data_worker's
- **Status:** OPEN — design constraint, not a bug. Documented to prevent the mistake.
- **Discovered:** 2026-06-03 (Kevin flagged the risk while we shipped data_worker watchdog)
- **Why this matters:** The same silent stale-state bug class affects ralph_engine too (yesterday's gate-mode dead canaries). The instinct is to copy the data_worker watchdog (3 STALE ticks → `os._exit(2)`) onto ralph_engine. **Do not do that.**
- **Why ralph_engine restart is high-stakes:**
  - Live engine owns active position state. Restarting mid-RTH causes per-Tier-3-§8.2 always-start-flat to wipe in-memory open positions. The next bar may re-fire an entry on a strategy that should have an OPEN position, OR miss a stop/target evaluation because the position state was lost.
  - Pre-restart positions ARE owned by the pre-restart window per Tier 3 §4.1, but only if the carryover mechanism caught them. Mid-bar transitions are fragile.
  - 30-60s of alert silence during restart — alerts that should have fired during boot are dropped.
- **Safer restart windows:**
  - **Best:** before extended hours start (< 4 AM ET / < 8 AM UTC)
  - **OK:** after extended hours close (> 8 PM ET / > 24:00 UTC)
  - **Risky:** any RTH or extended hours session
  - **Emergency only:** mid-RTH (e.g., circuit OPEN or alert engine genuinely dead for >30 min)
- **Right design for ralph_engine auto-heal:**
  - Detect stale similarly (log + alert)
  - DO NOT auto-exit. Instead: schedule the auto-exit at the next safe restart window (e.g., extended-hours close at 24:00 UTC) and only execute if still stale at that time.
  - For mid-day genuinely-dead scenarios, surface to operator (Kevin) for explicit decision instead of auto-acting.
- **Action for now:** keep ralph_engine watchdog OUT of today's scope. Detection-only (log STALE) might be safe to add later. Auto-heal logic needs Tier 3 carryover review first.
- **The hard-to-notice case:** if ralph_engine degrades silently during market hours, alerts stop firing or fire wrong, and Kevin doesn't see it visually until the divergence accumulates. Mitigation: a separate dashboard alarm on "no alerts fired in last N minutes during market hours" — passive detection, no auto-action.

---

### User-pack confluence gates silently fail on secondary timeframes (Phase 2 fleet)
- **Status:** RECHARACTERIZED 2026-06-02 evening — NOT a code defect, IS a silent live-engine stale-state bug. See "Update" section below.
- **Discovered:** 2026-06-02 — verified across all 15 user packs via PACKTEST canaries
- **Severity:** **HIGHEST** — affects every production strategy that uses any user pack as a cross-TF confluence gate. Confluence requirements that should restrict entries are evaluating as "never satisfied" (or never blocking, depending on the gate semantics). The strategy fires entries that don't conform to the user's intended gate logic. **Today's quantified empirical impact: 14 of 15 PACKTEST gate canaries fired 0 alerts over a full trading day** while their trigger-mode siblings fired 3-367 alerts each.
- **Quantified scope (2026-06-02 13:30-20:30 UTC):**
  - TRIGGER canaries (pack as the entry signal): 15/15 fired alerts (range 3-367)
  - GATE canaries (pack as a 2-minute confluence requirement): **14/15 fired 0 alerts**
  - Only exception: Swing 1-2-3 gate (sid 301) fired 204 alerts. Hypothesis: its `NEUTRAL` state is the engine's default-when-unset value, so the gate passes by default regardless of whether the pack computed anything on the secondary TF.
- **Affected packs (gate canary sid in parens, all 0 alerts unless noted):**
  - Bollinger Bands (sid 277), EMA Price Position v3 (279), EMA Price Position v4 (281), EMA Stack v2 (283), MACD Histogram v2 (285), MACD Line v2 (287), RSI Zones 2 (289), Relative Volume v2 (291), Support Resistance Channels (293), Stochastic Oscillator (295), Strat Assistant (297), SuperTrend (299), UT Bot V4 (303), VWAP v2 (305)
- **Hypothesis (high-confidence, code review pending):** The live engine's `_ShadowIndicatorEngine` for secondary timeframes (the layer that computes cross-TF confluence — Phase 30L) **does not instantiate user-pack incremental engines** for the secondary TF. Only the PRIMARY-TF `_user_pack_engines` exist and get fed bars. When a gate references `2m-MACD_LINE_V2-M>S+`, the engine looks up the 2m MACD_LINE_V2 state — which was never computed because no 2m MACD_LINE_V2 engine exists — and returns "no state." The "no state" return then evaluates falsy against any specific state requirement.
- **Why Swing 1-2-3 passes:** the swing pattern state machine defaults to `NEUTRAL` between confirmed patterns. Even with no computation, requesting `NEUTRAL` returns true. Other packs require an actively-computed state name that's only produced after engine ingestion.
- **Why this wasn't caught earlier:** the Phase 2 codec fix this morning addressed *primary-TF* user-pack state persistence in the persistent snapshot. It did not address the secondary-TF engine instantiation gap. The bug was masked previously by:
  - The fleet phantom-rate noise floor (24-95%) under the original snapshot bug — secondary-TF gate failures were a small contribution to a much bigger problem
  - Most production strategies were using SWING_123 NEUTRAL or similar default-passing states, hiding the issue
  - Today's PACKTEST canaries deliberately exercise a wide variety of specific-state gates → finally exposed the gap
- **Investigation TODO:**
  - Read `unified_engine.py` around the `_ShadowIndicatorEngine` class (also see memory `feedback_polygon_xtf_live_regression.md` for prior Phase-30L bug context)
  - Confirm: does the secondary engine instantiate `_user_pack_engines` from `pack_registry`?
  - Confirm: does the secondary engine receive bar feeds on the secondary TF cadence?
  - Check the cross-TF gate evaluator: how does it resolve a state value for a secondary-TF user-pack interpreter? Does it call into the pack's interpreter function, or read a pre-computed column?
- **Repro:** open My Strategies page, look at any PACKTEST `· gate` canary other than Swing 1-2-3 — alert count = 0 over many hours. Compare to its trigger sibling which fires actively.
- **Workaround:** none safe. Users currently relying on user-pack confluence gates are not getting the gate behavior they expect. Strategies fire on raw triggers regardless of the gate's state. This is the case for most of the user's production confluence configurations (e.g., `5m-MACD_LINE_V2-M<S+`, `10m-STOCHASTIC_OSCILLATOR-BULLISH_MIDRANGE`).
- **Implication for past phantom-rate readings:** strategies appearing healthy (low phantom rate) with cross-TF user-pack gates may have been "healthy" only because both lanes were producing the same gate-bypass behavior. The backtest and live engine likely BOTH skip the gate — making them agree even though the gate is broken. Real verification requires built-in confluence (EMA_STACK, MACD_LINE which are BUILT-IN, not user-pack) on the secondary TF.

#### Update 2026-06-02 evening — rediagnosed via GATE-DIAG diagnostic deploy

Deployed `logger.info` instrumentation in `_ShadowIndicatorEngine.__init__` and `on_bar_close` (commit `734e03e`). Within 4 minutes of the worker restart that the deploy triggered, **8 of 14 previously-silent gate canaries started firing alerts** (sids 277, 283, 285, 287, 293, 299, 301, 305 — 2 alerts each). The GATE-DIAG output showed shadow engines were correctly:
- Instantiating `_user_pack_<slug>` engines from req_ind (e.g., the 2m shadow had all 15 pack engines)
- Producing pack indicator columns into `current` after `update_bar` (mhv2_hist, mlv2_line, etc.)
- Dispatching user-pack interpreters and emitting state into `interps`
- Building confluence records in the correct format (`2M-MACD_HISTOGRAM_V2-H+up`, etc.) that match the strategy's normalized confluence_set

**The bug is NOT a code defect in the gate-evaluation path.** The instrumented code paths work correctly when freshly booted. Some state in the long-running ralph_engine worker gets stale or silently fails over time, and the symptom is "cross-TF user-pack confluence stops contributing to the gate evaluation." A restart heals it.

This is in the same class as the data-worker silent stale-roster bug (see entry below). The watchdog task (#21) was scoped to data_worker only; **it should be expanded to cover ralph_engine too** — periodically self-test that secondary-TF user-pack shadow engines are still producing the expected confluence records, and force a re-init if they're not.

**Status updated to RECHARACTERIZED.** Will not be fixed by the user-pack-engine wiring change I originally proposed. Investigation should pivot to:
1. What state degrades in the ralph_engine over time that breaks shadow user-pack flow?
2. Why does the existing 5-min `STRATEGY_REFRESH_INTERVAL` reload not heal it?
3. Add observability + watchdog so we don't need a restart to recover.

The 6 gate canaries that didn't fire post-restart (RSI Zones 2, RVOL v2, Stochastic, Strat Assistant, UT Bot V4, Swing 1-2-3) all have restrictive state requirements (EXTREME_OVERBOUGHT, EXTREME, OVERBOUGHT_BEARISH, INSIDE, BULL_TREND, NEUTRAL) that may legitimately not have been met in the post-deploy window. Need more market hours to confirm those work too.

---

### Data-worker silently keeps a stale strategy roster — only fixable by restart
- **Status:** OPEN
- **Discovered:** 2026-06-02 (Kevin deleted ~50 strategies from the DB via UI; worker continued reporting `engines=64` for 23+ hours through multiple 5-min reload cycles)
- **Severity:** **Medium-high** — operationally invisible. Worker keeps ticking "ghost" engines (snapshot flush attempts to deleted strategy IDs trip the circuit breaker → real ticks get skipped → lag grows linearly). Only signal a user sees is "snapshots aren't refreshing" on the strategy-health UI; no error log is emitted that points at the root cause.
- **Location:** `src/data_worker.py:328-409` — `_load_streaming_strategies()` is supposed to add/remove engines on a `STRATEGY_RELOAD_INTERVAL` cadence (5 min). For 23h+, the engines dict held 64 entries despite the DB only having ~15. No `[stream] tracking N strategies` log line appeared in 2000+ lines of worker output during that period — the reload's success-path logger.info at line 404 wasn't firing, but the metrics thread was still emitting the stale `engines=64` count from `self._engines`.
- **Symptom on the worker metrics:**
  - `engines=N` is stuck at the count from the last successful boot/reload
  - `circuit=OPEN(opens 1) skips=40+` — flushes/inserts against deleted strategy IDs trip the breaker
  - `lag_max` grows linearly (no real ticks happening)
  - `trades_written` near zero
- **Symptom on the strategy-health UI:**
  - Strategies show snapshot ages of 16-17 hours (matching the time of the last bulk UAD) instead of refreshing on the normal 5-min cadence
  - Combined with the `strategy_health` pagination bug (fixed `d4e5dc8`), looks like 100% phantom rate even on healthy strategies
- **Root-cause hypotheses (not narrowed further):**
  - The streaming thread died on an uncaught `BaseException` (KeyboardInterrupt-class or in-process error). The `try/except Exception` at `data_worker.py:422-424` doesn't catch these. Metrics thread keeps reading the dict.
  - `load_all_desired_states()` HTTP call hung indefinitely (no explicit timeout) during a Supabase 522 window, blocking the reload. We hit several 522s on 2026-06-02 morning.
  - Some lock contention or generator state in `_load_streaming_strategies` left the reload short-circuiting silently.
- **Fix:** force redeploy via `railway variable set RESTART_ANCHOR=...` — `engines=64` snapped to `engines=15` immediately on restart, snapshots resumed refreshing within ~14 min.
- **Mitigations to ship:**
  - Add a heartbeat `logger.info("[stream] pass tick, last_reload_age=%ss")` at the top of `_streaming_pass()` so next occurrence shows whether the thread is even running.
  - Wrap `_load_streaming_strategies()` calls in a watchdog that escalates to `logger.error` if more than 2× `STRATEGY_RELOAD_INTERVAL` elapses with no reload completion.
  - Add HTTP timeouts on `load_all_desired_states()` and `load_strategies_monitoring_admin()` (currently rely on PostgREST/Supabase defaults).
- **NOT to confuse with:** the `strategy_health` 50k-row pagination bug (fixed in `d4e5dc8`). Both surfaced as "endpoint reports wrong numbers" but the root causes are independent.

---

### Streaming-tick backtest path under-fires ~7% vs warm recompute
- **Status:** OPEN
- **Discovered:** 2026-06-01 (during Phase 2 verification, sid 263 + 268 snapshot diff)
- **Severity:** Medium — accounts for the residual ~5-8% phantom rate that persists after the Phase 2 user-pack codec fix. Pre-existing structural behavior that Phase 2 didn't introduce but now sits at the top of the noise floor.
- **Location:** Most likely `src/unified_engine.py:3371-3395` (`apply_backtest_snapshot` — Tier 3 §8.2 always-start-flat). Possibly also `src/data_worker_engine.py:run_store_fed_window` boundary handling.
- **Symptom:** The data-worker streaming-tick path produces a strict SUBSET of warm-recompute trades over the same window — every streaming-tick trade is a true positive (matches warm-recompute on `entry_fill_ts` exactly + `entry_price` within $0.01), but the streaming-tick path misses ~7% of flips that warm-recompute catches. Missed trades are scattered across the window (not clustered around any obvious boundary).
- **Hard evidence (sid 263 + 268, 2026-06-01 18:36 → 20:00 UTC):**
  - sid 263: 30 streaming-tick trades / 33 warm-recompute trades — 30/30 (100%) match on timestamp + sub-cent price; 3 only-in-recompute.
  - sid 268: 39 streaming-tick trades / 43 warm-recompute trades — 39/39 (100%) match; 4 only-in-recompute.
  - Combined: 100% precision, ~93% recall.
  - Examples of "only in recompute" trades: sid 263 entry @ 2026-06-01T18:44:30, 19:55:30; sid 268 entry @ 19:25:50, 19:52:30. All on bar boundaries. None have a corresponding streaming-tick trade within ±60s.
- **Hypothesis (not yet confirmed):** Tier 3 §8.2 always-start-flat. Every streaming tick resumes with `position=FLAT` regardless of what the prior tick had open. If a flip happens early in the new tick window where the warm-recompute would have been `IN_POSITION` (and would have exited normally before entering the next), the FLAT-start streaming-tick takes a different path through the position state machine. Some entries fall in the gap. Tier 3 §4.1 says "pre-boundary window owns any open trade" — but if a strategy's previous backtest run stopped *before* the gap-trade's entry (which is the case for newly-deployed Phase 2 strategies that haven't been recomputed yet), the trade is nobody's.
- **Why this is separate from the Phase 2 codec bug:** Phase 2 fixed *user-pack state being dropped* on persist. This bug is about *position state* being deliberately reset to FLAT on resume (Tier 3 design choice). The two are independent — Phase 2 unmasked this residual because the codec removed the dominant noise source (24-95% phantom rates from cold packs).
- **Investigation plan:**
  - Run Layer 4 walkthrough on a sample of "only in recompute" trades — pull surrounding bars, position state, prior-tick boundaries.
  - Quantify: what % of UAD-vs-streaming diff falls exactly on a tick boundary (where Tier 3 §8.2 would manifest) vs. scattered (which would point elsewhere)?
  - Test hypothesis: would carrying position state across the snapshot boundary (Tier 3 §8.5 collapse) eliminate the gap? See [[project-tier3-85-engine-unification]] memory — task #25 is the long-form fix.
- **NOT a Phase 2 regression.** Pre-existing; Phase 2 just made it visible by removing the bigger bug on top. Quantify, classify, then decide if it warrants the §8.5 collapse work or a lighter targeted fix.

---

### Backtest snapshot drops ALL user-pack state — fleet-wide phantoms
- **Status:** FIXED-PENDING-VERIFY (Phase 2 codec shipped 2026-06-01 at `f006937`; verified working on sid 263/268 same day — 100% precision on streaming-tick trades vs warm recompute. 24h fleet-wide stability check still owed.)
- **Discovered:** 2026-05-29 (investigating TSLA-263 phantom-rate root cause; bug existed since Phase A snapshot redesign on 2026-05-28)
- **Severity:** **HIGHEST** — pollutes Layer 2 phantom counts and Layer 3 fill deltas across every strategy in the fleet that runs in snapshot-resume mode (i.e., everything past its `last_recompute_until_ts`)
- **Location:**
  - `src/unified_engine.py:3287-3319` — `serialize_backtest_snapshot` calls `snapshot_state(persistent=True)`
  - `src/unified_engine.py:856-871` — the `persistent=True` branch runs `pickle.dumps(deep)` as a probe; importlib-loaded user packs fail this probe and get silently dropped from `packs_out`
  - `src/unified_engine.py:890-896` — `restore_state` skips assignment when `packs` is empty (`if packs:`), so `_user_pack_engines` keeps the fresh `__init__` values
  - `src/data_worker_engine.py:323-390` — `run_store_fed_window` calls the above on every tick
- **Symptom:** Strategy's stored backtest has multi-minute gaps where live alerts fired. Each backtest snapshot-resume tick has a COLD UT Bot v4 (and all other user packs), so the indicator's `_atr` / `_trail_stop` / `_position` never accumulate across ticks. Tick processes too few bars per cycle (gated by `LAG_MINUTES=15`) for cold state to converge. Result: backtest misses entry/exit flips that live (continuously warm) engine detects.
- **Hard evidence (sid 263):**
  - From-scratch `UtBotV4Incremental` on Polygon REST 1Sec→10Sec (full 2h warmup) matches live `bar_engine_states` to 0.001 on `trail_stop` at every critical bar. All 5 flips fire on same bars. Indicator code is correct.
  - Bar OHLC values are bit-identical between live and REST (verified at 18:54:20: o=436.392 h=436.5 l=436.36 c=436.4899). Bar data is consistent.
  - Decoded `strategies.config.engine_snapshot_b64` for sid 263 (last_bar_ts=19:48:20): `packs in snapshot (0 entries): []`. UT Bot v4 is NOT in the saved snapshot.
- **Why Phase A authors knew about it:** The comment at `unified_engine.py:869` says explicitly: *"on resume this indicator will warmup from scratch on the new data window."* They consciously deferred the backtest-side fix when shipping Phase A's live-side fixes on 2026-05-28.
- **Why fresh canaries look clean:** A fresh canary's FIRST backtest is a full recompute (no snapshot resume), so user packs warm up across hours of history. Phantoms only appear after the first snapshot is saved and tick-mode kicks in. Sid 263 was created today; its phantom rate grew over the day as snapshot-resume took over.
- **Why built-ins are unaffected:** Built-in indicator state (`atr_value`, `ema`, `macd_*`, etc.) lives on the `IndicatorState` class, which pickles cleanly. The pickle-probe drop only hits the `_user_pack_engines` dict.
- **What this explains across the fleet:**
  - Sid 263's gap pattern (18:51:10 → 18:57:40 with no backtest trade despite live alerts at 18:54:30 and 18:55:10).
  - The TSLA-263 24% vs SPY-268 14% phantom-rate split — noise within the broader bug, not a real symbol-specific effect.
  - The Layer 3 entry-delta outliers on the SPY 10Sec cohort — older strategies have more accumulated snapshot-resume cycles, more drift.
  - The mlv2 missed-entries we saw in the walkthrough — same root cause for any user-pack-based strategy, not just UT Bot v4.
- **Fix options (see session_2026-05-29 memory for full pros/cons):**
  - **(A) `__getstate__`/`__setstate__` per pack** — pickle-based, ~10 LOC/pack × 16 packs
  - **(B) Custom `serialize_state()` / `restore_state()` protocol** — pickle-free, JSON-compatible state dicts, ~150 LOC + protocol contract. Recommended.
  - **(C) cloudpickle** — single-line library swap; adds dependency risk
  - **(D) Warmup-window replay** — interim patch, no snapshot change; replay last 20 bars cold before processing new bars
- **Backup branch covering pre-fix state:** `backup-2026-05-29-layer1-solid` at d8ba188
- **DO NOT confuse this with user-pack determinism.** UT Bot v4 just passed a rigorous determinism test (0.001 trail_stop agreement). The bug is in the SNAPSHOT path, not in the pack itself. Other packs may still have legitimate determinism issues — but those are separate from this one.
- **Empirical impact (sid 263, 2026-05-29 ~21:33):** Triggered "Update All Data" (full recompute via `/api/strategies/{id}/refresh`). Same-window before/after comparison on the 3h window [17:47:27 → 20:47:27]:
  - **Phantom rate (raw): 95 → 3 (-97%)**
  - **Missed rate (raw, 1h): 26 → 3 (-88%)**
  - **Paired count (raw): 2 → 94 (47×)**
  - Unpaired alerts: 20 → 3 (-85%)
  - Unpaired backtest edges: 26 → 4 (-85%)
  - Layer 3 entry avg delta: 8.5s → 3.6s; entry ≤5s: 82.1% → 91.5%
  - **KPI swing: −34.56R → +37.98R (+72.5R) on the same trade window** — proves the bug isn't just dropping/missing trades, it's producing actively WRONG ones from cold-state user packs
  - Residual ~3-5% phantoms after recompute are likely the known `phase2_signal_exit` cases + cross-event mispairings, not this bug
  - Recompute saved a NEW snapshot at 21:24 — STILL with 0 user packs. Bug recurs from this moment forward; strategy will drift back to phantom-heavy over the next 24-48h unless re-recomputed or fixed.
- **Manual workaround until Phase 2 shipped:** Click "Update All Data" on a strategy → triggers `recompute_and_persist_stored_trades` which runs the full warm engine, replaces all backtest trades. Buys clean state for ~24h before drift sets in again. Use this on canaries to keep weekend data clean.
- **Fix shipped 2026-06-01 (`f006937`) — Phase 2 user-pack state codec:**
  - New module `src/user_pack_state_codec.py` extracts `dict(eng.__dict__)` (deep-copied) per user-pack engine and stores it under the snapshot's `packs` field, sidestepping the importlib synthetic-class pickle issue that previously dropped them.
  - `unified_engine.snapshot_state(persistent=True)` and `restore_state` route through the codec.
  - Opt-in override: any pack class can define `serialize_state` + `restore_state` to take control of its own state representation.
  - `pack_spec.validate_state_protocol` + `pack_registry` lenient v1 contract check (errors on inconsistent override; warnings on exotic types).
  - Tests: 15/15 packs round-trip cleanly via `_test_userpack_state_codec.py`; `_test_lef_2c_snapshot.py` extended with a user-pack envelope round-trip case (passes).
  - Backup branch: `dev-backup-pre-phase2-codec`.
  - Phase 1 (warmup-window replay, `684265c`) was superseded; its dead code remains gated behind `BACKTEST_SNAPSHOT_WARMUP_BARS=0` Railway env and will be removed after 24h Phase 2 stability.
  - Empirical verification 2026-06-01 (sid 263 + 268): post-fix decoded snapshots have `packs=['ut_bot_v4']` (was `packs=[]`); streaming-tick trades match warm-recompute on entry_fill_ts + entry_price within $0.01 (100% precision); residual ~7% recall gap is tracked as a separate bug above.

---

### mlv2_cross_bear walker reclassification
- **Status:** OPEN
- **Discovered:** 2026-05-27 (Railway logs, multiple strategies)
- **Severity:** Medium — silently rewrites correct exit reasons on a
  subset of trades
- **Location:** `src/api/services/backtest_service.py:_hifi_resolve_trades`
  around line 806-828 (the stop/target walker)
- **Symptom:** When a trade's `exit_reason` is a signal name like
  `mlv2_cross_bear` (an indicator-vs-indicator cross with no L-type spec)
  AND the trade also has a defined `stop_price`/`target_price` from the
  strategy, the walker still runs over the bar's 1-sec bars looking for
  stop/target level crosses. If price happens to touch the stop or
  target level during that bar, the walker rewrites `exit_reason` to
  `stop_loss` or `target`. The original signal-cross reason is lost.
- **Observation:** Railway `[HIFI] Resolved … outcomes changed` counters
  showed 1-11 trades flipped per strategy on the 2026-05-27 17:10-17:13
  passes. sid 172 specifically didn't see flips, but the pattern is
  symbol/volatility-dependent.
- **Root cause:** The walker's bail-out at line 806-809 only skips
  `exit_reason in ('stop_loss', 'stop')` with `stop_et != 'L'`. It does
  NOT skip non-stop/target exit reasons that happen to coexist with L-type
  stop/target levels.
- **Fix sketch:** Add a third bail-out: `if exit_reason not in
  ('stop_loss', 'stop', 'target') and not is_ltype_signal_exit: continue`.
  Treats "signal exit without per-second walker support" as a
  non-candidate, preserves original reason.
- **Risk:** Need to confirm no other walker pathway relies on the
  fall-through. Tests in `test_hifi_exit_timestamp.py` should catch
  regressions.

---

### sid 150 has 5043 trades stuck at `hifi_resolved=False`
- **Status:** WONTFIX (cosmetic)
- **Discovered:** 2026-05-27 (immediately after reverting commit
  `5023be6`)
- **Severity:** Low — no functional impact; just confusing in the DB
- **Location:** `trades` table, `strategy_id=150`, `data.hifi_resolved=False`
- **Symptom:** ~5043 trades on sid 150 have `hifi_resolved=False` in
  their `data` JSONB, written by the short-lived "persist False"
  branch in `5023be6`. The current Hi-Fi pass (post-revert in `723141b`)
  only skips on `True`, so these trades still get walked every pass —
  same throughput cost as if they were `None`. The `False` marker just
  sits there confusing future debugging.
- **Rationale for WONTFIX:** A one-time SQL `UPDATE trades SET data =
  data - 'hifi_resolved' WHERE strategy_id=150 AND
  data->>'hifi_resolved' = 'false'` would clean it up. Not worth a
  migration; the new incremental mode (`51835f3`) makes it irrelevant.

---

### Volume match ~54% on rest-1s-aggs vs cache+backfill bars
- **Status:** OPEN
- **Discovered:** Kevin flagged 2026-05-27 — pre-existing known
  limitation
- **Severity:** Low for entry/exit timing (most user-pack indicators use
  candle close); could be higher for volume-based filters or VWAP
- **Location:** Bars Comparison page (parity comparison view)
- **Symptom:** When comparing the REST 1-second aggregate bars to the
  cache+backfill bars on the parity comparison page, the volume match
  rate is ~54% (vs ~100% for OHLC). Close prices line up cleanly; the
  volume column on the same timestamped bar doesn't.
- **Hypothesis:** Late prints / trade reporting that doesn't make it
  into the WS-aggregated cache stream but does appear in the REST 1-sec
  aggregates. Polygon-side data lineage difference, not a bug in our code.
- **Why we tolerate it:** Indicators we care about (EMAs, MACD, RSI,
  stochastic) operate on OHLC. Volume only affects RVOL and explicit
  volume-based triggers. None of the active user packs are
  volume-gated as of 2026-05-27. If a pack starts using volume, revisit.

---

### Ralph WS pipeline drops SPY sub-minute bars during active hours
- **Status:** MITIGATED — root cause not fixed, but impact reduced by `ws_rest_spliced`
- **Discovered:** 2026-05-27 during REST-vs-WS investigation
- **Mitigated:** 2026-05-28 via `ws_rest_spliced` rollout
- **Severity:** High while WS was the sole source; now reduced — `ws_rest_spliced` verifies WS bar closes against REST and splices REST values into indicator history when drift is caught, so indicator state converges to backtest-aligned values even when WS aggregation is faulty.
- **Location:** Ralph's WebSocket bar aggregation pipeline
  (`ralph_engine.py` BarBuilder / on_second_bar)
- **Symptom:** During the 18:00-19:30 UTC window on 2026-05-27, 26% of SPY
  10Sec bars (140/541) were missing from `live_bars`. The missing bars
  had real volume (7K-40K each), so not quiet-period artifacts.
  Confirmed not deploy-related: during the largest gap (18:06:00-18:22:10,
  16 minutes), Ralph was actively writing 22 other symbol/TF combos
  (TSLA, TSLL, SPY 120s/180s, etc).
- **Hypothesis:** Polygon per-ticker WS subscription stuck after a
  reconnect, OR race condition in BarBuilder for SPY's high tick rate,
  OR Polygon A-channel throttling on SPY specifically.
- **Why not fully fixed:** Pure REST polling proved too slow on RTH probe
  (2026-05-28, p90 8-23s) to drive sub-minute alerts directly. Chosen
  direction was the hybrid `ws_rest_spliced` (see
  `/home/kevin/.claude/plans/breezy-dreaming-umbrella.md` and memory
  `project_ws_rest_spliced_canary.md`). The hybrid surfaces drift
  events as `verification_status='corrected'` or `'drift_uncorrected'`
  on the alerts table, so the WS coverage issue is now observable +
  partially compensated rather than silent. On sub-minute TFs, the
  `apply_rest_correction` engine path is structurally unable to splice
  REST into a bar that's already 1-2 bars stale, so REAL drift on
  sub-minute strategies still shows up as `drift_uncorrected`. Tracking
  rate over a week to decide whether per-second splice escalation
  (`project_per_second_splice_idea`) is warranted.

---

### Sub-minute drift_uncorrected_cascade: live engine entirely misses multi-trade clusters
- **Status:** Known, partial mitigation in place via `ws_rest_spliced`; structural fix pending (per-second splice)
- **Discovered:** 2026-05-28 during M8 fleet observation + divergence backlog walkthrough
- **Severity:** High on sub-minute (10Sec) strategies — drives 34-56% phantom rate per strategy
- **Location:** Interaction between `unified_engine` indicator state (cumulative) + `apply_rest_correction` (latest-bar only)
- **Symptom:** When a sub-minute bar's WS close differs from REST close by ≥$0.01 and the bar correction is rejected (`drift_uncorrected` — common on 10Sec because REST settles 6-15s after bar close, by which time 1-2 newer bars are in history), the indicator state on that strategy is offset from REST-canonical. Subsequent trigger evaluations diverge between live and backtest for many bars after, EVEN ON BARS WHERE THE LIVE ALERT VERIFIES AT Δ=0.0. The cumulative state — not just the bar value — is what's offset.
- **Concrete example (sid 170, 2026-05-28 20:14-20:19Z):** backtest produced 5 trades in 5 minutes with 1-13 bar holds; live fired ONE entry alert in the same window. All 5 backtest trades show up as `missed` in the divergence backlog. Bar verification on the 1 live alert was Δ=0.0.
- **Why it's not a regression:** Pre-`ws_rest_spliced` (running `ws_agg_reconciled`), the same phantom pattern existed — there was no verifier surfacing the drift events, so the cascade was silent. `ws_rest_spliced` makes the cause visible (the `drift_uncorrected` stamps) but the engine still can't catch up to backtest.
- **Why current mitigation is partial:** `apply_rest_correction` only accepts splices on the LATEST bar in BarBuilder history. On 10Sec TF this almost never lands because REST settles after 1-2 newer bars have arrived.
- **Structural fix path:** Per-second REST splice — retain per-second bar history, splice REST 1-sec values when they settle (anywhere from 2s to 15min), re-aggregate the bar from corrected per-second data, replay indicator state forward from that bar. See `project_per_second_splice_idea` memory. Replay-from-N indicator path is the largest engineering lift; estimated 1-2 focused sessions.
- **Workaround pre-fix:** Either accept the cascade rate on sub-minute strategies, OR migrate sub-minute strategies to 1Min+ TFs where REST settle (~10s) is well within bar duration (60s+).

---

### Live ↔ backtest divergence on gap-fill bars during EH/AH
- **Status:** Known structural mismatch, no immediate code fix shipped. Tonight (2026-05-28) added `gap_fill_unverified` distinct status to make the EH/AH signal interpretable; structural fix pending decision.
- **Discovered:** 2026-05-28 during ws_rest_spliced Phase A+B+C extended-hours validation
- **Severity:** High for any strategy that trades during EH/pre-market. RTH effectively unaffected because trades are dense.
- **Location:** `ralph_engine.BarBuilder.accept_bar` (gap-fills empty windows with prev_close + volume=0) vs `data_loader.resample_to_timeframe` (drops empty periods via dropna)
- **Symptom:** In EH/AH, sub-minute bar windows often have zero trades. Live engine gap-fills those windows with `(prev_close, prev_close, prev_close, prev_close, vol=0)` and runs indicators on them. Backtest drops them entirely. Result: indicator state evolves differently, `max_hold_bars` counts bars differently, triggers fire on bars backtest never sees, and the divergence backlog shows phantoms that have no real cause (gap-fill artifact). Confirmed via direct Polygon probe: 22:30-22:40Z windows have 0 SPY trades for many 10-sec buckets, while the live engine continued processing gap-fill bars.
- **Why live gap-fills:** Calendar-bar semantics — `max_hold_bars=4` is intended as "4 bars (40s on 10Sec)", not "4 trades-happened bars (which could be many minutes in EH)". Removing gap-fill changes strategy meaning.
- **Three potential fixes:**
  1. **Backtest gap-fills too (preferred):** match live's calendar-bar semantics. Backtest behavior changes; historical backtests' numbers shift. Cleanest semantic fix.
  2. **Live stops gap-filling:** match backtest/REST. Changes `max_hold_bars` meaning (4 bars in EH could mean minutes). Likely breaks current strategies.
  3. **Live keeps gap-fill internally but doesn't fire triggers on gap-fill bars:** Subtle behavior change; doesn't fix indicator-state divergence.
- **Temporary mitigation (shipped 2026-05-28):** `verification_status = 'gap_fill_unverified'` distinguishes gap-fill bars (where REST has no data because no trades) from genuine `rest_unavailable` (where REST should have data but doesn't). Doesn't fix the underlying divergence but makes the verification dashboard honest in EH.
- **Validation note:** Verify this in RTH where gap-fills are rare. EH/AH phantom rates will stay elevated until structural fix.

---

### Live engine fires through confluence whose shadow engine wasn't created
- **Status:** Known bug, no fix shipped. Workaround: avoid declaring a secondary TF on a hub that already has a strategy using that TF as its primary.
- **Discovered:** 2026-05-29 while creating TSLA canary sid 264 with `confluence: ['1m-SWING_123-NEUTRAL']`. TSLA hub already had multiple 1Min PRIMARY strategies (sid 174 and the 257-260 cohort), so no shadow engine for (TSLA, 1m) was created. Result was asymmetric:
  - **Backtest correctly blocks:** the confluence record never populates → gate evaluates as FALSE always → 0 trades produced
  - **Live incorrectly fires:** sid 264 fired the SAME 28 alerts as its no-confluence twin sid 263 in the same window, as if the gate weren't there
- **Severity:** Medium-High. As the strategy fleet grows, every additional primary TF rules out using that TF as a secondary across the same hub. The fail-open behavior live means a strategy will produce alerts that backtest disagrees with, masquerading as real divergence.
- **Location:** Shadow engine creation in `ralph_engine.py` `finalize_shadow_engines` (skips when `has_real=True`); confluence gate evaluation downstream that should respect the "shadow not available" condition rather than fail open.
- **Two ways to fix:**
  1. **Live: explicit fail-closed when shadow missing** — if a strategy declares a secondary TF confluence and no shadow exists for that (symbol, TF), live should NOT fire entries that depend on the gate. Matches backtest semantics. Minor code change in the per-bar gate evaluation path.
  2. **Always create shadow engines for any declared secondary TF**, even when a primary monitor uses the same TF. The shadow's job (computing interpreter states) is distinct from the primary monitor's job (position management). Slightly more memory/CPU but eliminates the asymmetry entirely. Cleaner architectural answer.
- **Suspected fix path:** Option 2 (always create the shadow) — primary monitors already compute their own interpreter states for trigger evaluation; the shadow's `_mtf_confluence` buffer is a separate concern from the primary's position machine. Splitting them cleanly removes the conflict.
- **Workaround until fixed:** Use "odd" secondary TFs (2Min, 3Min) that don't conflict with any primary on the hub. Documented in `docs/SOP_Test_Strategy_Creation.md`.

---

### Mass-builder strategies don't auto-snapshot for 10Sec and 5Min timeframes
- **Status:** Known UX issue, no immediate code fix shipped. Workaround: manually click "Refresh Data" on each strategy after creation.
- **Discovered:** 2026-05-29 — 12 TSLA canary strategies (sids 251-262) created via mass builder yesterday; only the four 1Min strategies (257-260) got auto-snapshots at ~13:45-13:51 UTC. The six 10Sec (251-256) and two 5Min (261-262) ended up with `config.engine_snapshot_b64 = False` and `data_refreshed_at = None`, meaning the live engine could never pick them up.
- **Severity:** Medium UX. Strategies created via mass builder need an extra manual step to be eligible for live tracking. Particularly annoying because the recent Mass Builder ↔ Update Data alignment work was supposed to make them equivalent — Kevin's expectation was the strategies would "just kick in."
- **Suspected cause:** Some auto-snapshot path (possibly in `WorkerManager` or a data-worker cron) handles 1Min strategies but not 10Sec / 5Min. Possibly TF-specific routing in `forward_test_service.append_new_trades_for_strategy` or the snapshot-subscribe mechanism. Needs investigation.
- **Workaround:** From the UI, click "Update Data" / "Refresh Data" on each affected strategy. After the snapshot is created, the live engine's 5-min hot-reload cycle will pick them up.
- **Quick triage script:**
  ```python
  # Find strategies needing refresh
  c.table('strategies').select('id,symbol,timeframe,config').gte('id', N).execute()
  # For each: bool((s.get('config') or {}).get('engine_snapshot_b64')) == False → needs refresh
  ```

---

### Supabase 522 timeout 2026-05-28 ~21:26Z — origin connection
- **Status:** Resolved by Supabase project restart (~21:32Z)
- **Discovered:** 2026-05-28 during divergence investigation
- **Severity:** Medium — blocked all DB writes for ~6 min; alerts queued and eventually drained
- **Symptom:** Cloudflare returned 522 ("Connection timed out") for all Supabase REST calls. Worker writes failed; investigation queries failed.
- **Suspected cause:** Unknown. Possible candidates: (a) Supabase project-level issue (Cloudflare layer between client and Postgres), (b) Postgres overload, (c) coincidence. The bulk strategy config UPDATE I ran at M8 rollout (~18:40Z, 39 strategies) was 3 hours earlier so unlikely contributor.
- **What to watch for next time:** Worker log error patterns during the outage window (correlate with alert dispatch failures), Supabase project health dashboard.

---

## Closed

(none yet)
