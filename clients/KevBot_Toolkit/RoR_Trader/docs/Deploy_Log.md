# Deploy Log — `dev` branch / Railway Worker

Running log of pushes to `dev` that trigger a Railway redeploy. Each push
restarts the Worker container, which causes a 1–3 minute gap in bar
recording (warmup re-runs, no live alerts during restart). Use this log
to correlate cache gaps, alert misses, or WebSocket reconnect events
with deploy windows rather than chasing them as data bugs.

Times are recorded in **UTC** (chart timestamps) and **MT** (Kevin's
local). Worker container restart time = ~30s build + ~30–60s warmup =
~1–2 min unavailable per deploy.

## Format

```
- **HH:MM UTC (HH:MM MT)** — `<short_sha>` <commit subject>
  - Service(s) redeployed: Worker / api / frontend / streamlit
  - Observed cache gap: <bar_start times affected, or "none observed">
  - Notes: anything unusual (stuck deploys, reverts, etc.)
```

## 2026-05-11

- **~18:30 UTC (12:30 MT)** — `8b74ddd` + `5fbfe0d` Phase 41 — backtest trades → trades table migration
  - Service(s) redeployed: api (writer + reader changes), worker (trades_store filter wiring)
  - Required manual steps: **Run `src/migrations/phase41_backtest_trades_relax_unique.sql` in Supabase SQL Editor BEFORE the backfill script**. Then run `python -m _backfill_stored_trades_to_table --apply` from `src/` to migrate existing stored_trades JSONB content into trades table with `data_source='backtest_<model>'`.
  - Observed cache gap: TBD
  - Notes: Completes the storage unification started in Phase 40 (2026-04-24). Backtest data now lives in the trades table alongside algo data, distinguished by `data_source` LIKE pattern (`backtest_%` vs `cache_%`). This unlocks REAL REST↔CACHE divergence in the Divergence tab — previously both lanes silently read from the same trades-table rows, producing fake "perfect alignment." Unique constraint relaxed to include `data_source` so REST + CACHE rows at identical timestamps coexist. Existing NULL data_source rows tagged as 'cache_locked' (preserves cron-written history). Backup: `dev-backup-pre-backtest-trades-migration-2026-05-11`.

## 2026-05-08

- **18:45 UTC (12:45 MT)** — `e7bf21a` + `0073dbb` + `0c208de` M8.7 Builder UIs + admin rename + REST-vs-CACHE design doc
  - Service(s) redeployed: api (mass_builder.py threads model kwargs through build_strategy_config), frontend (Strategy Builder + Mass Builder gain 3-dropdown picker; admin page + sidebar relabeled)
  - Observed cache gap: SUPABASE OUTAGE in progress (US-East-C network) — Railway/Vercel redeploys may queue; verification deferred
  - Notes: Pure frontend + safe backend additions during the Supabase outage. Strategy Builder + Mass Builder now expose `backtest_model` / `algo_model` / `live_model` selectors at creation time. Builder UIs are isolated — won't affect live engine. mass_builder.py:build_strategy_config now accepts model kwargs (None → GET-enrichment default). Admin "Backtest Models" page relabeled "Backtest / Algo Models" reflecting the shared registry. Design doc for v2 dual-storage divergence committed at `docs/Design_REST_vs_CACHE_Divergence_v2.md` — recommendation locks in Option B (JSONB map keyed by model). Backup: `dev-backup-pre-builder-uis-2026-05-08`.

- **16:30 UTC (10:30 MT)** — `d867e63` + `5fddf86` M8.7 lane×mode matrix complete + button uniformity
  - Service(s) redeployed: api (2 new helpers + JWT-safe /update endpoint), recompute_jobs worker (bulk fans to both lanes), frontend (Strategy Detail header buttons unified)
  - Observed cache gap: TBD
  - Notes: All 4 lane×mode combinations now wired in /update endpoint. Adds `append_new_backtest_trades_for_strategy` (forward append → stored_trades JSONB under backtest_model) and `recompute_and_persist_algo_trades` (full recompute → trades table under algo_model). JWT expiration on long backtest runs fixed via `set_admin_user_context` wrapper. Bulk page job-worker also fans out to both lanes per Option A. Strategy Detail header now has Update New Data + Update All Data instead of Refresh. Live model labeling on alerts verified working — 20/20 alerts since 2026-05-07 22:00 UTC carry `live_model='ws_agg_locked'`. Backup: `dev-backup-pre-lane-mode-matrix-2026-05-08`.

## 2026-05-07

- **23:10 UTC (17:10 MT)** — `3bf6911` + `a6d0bb8` M8.7 algo_model split + live_model labeling + Divergence buttons
  - Service(s) redeployed: api (algo_model field, /update endpoint, divergence response), Worker (DBAlertDispatcher + AlertDispatcher stamp live_model on alert fire), frontend (Update buttons + lane badges + Live model column)
  - Observed cache gap: TBD
  - Notes: TWO migrations need to run on Supabase: `alerts_live_model_column.sql` (adds live_model TEXT column to alerts) and previously-applied `algo_history_cron_cycles.sql`. Bulk migration `_bulk_set_algo_model.py --apply` already ran — all 20 strategies now backtest_model=rest_hifi + algo_model=cache_locked. Sid 152 flipped from cache_locked-backtest. Live alerts going forward carry live_model in payload; legacy alerts render "unknown" (no backfill). Backup: `dev-backup-pre-algo-model-2026-05-07-evening`. Two of four lane×mode update combinations deferred (forward backtest append + full algo recompute).

- **21:25 UTC (15:25 MT)** — `cd58026` M8.7 Divergence tab: 3-lane comparison on Strategy Detail
  - Service(s) redeployed: api (new endpoint + service), frontend (new tab + hook + types)
  - Observed cache gap: TBD
  - Notes: Adds GET /api/strategies/{id}/divergence-data + new "Divergence" tab on Strategy Detail. Three lanes today: Backtest (stored_trades JSONB) / Algo (trades table from cron) / Live (alerts). KPIs + drift stats (median/p95/max) per lane pair; color-coded ≤2s green / ≤30s yellow / >30s red. Real REST-vs-CACHE comparison deferred to v2 (needs dual-storage backend). Backup: `dev-backup-pre-divergence-tab-2026-05-07`.

- **19:47 UTC (13:47 MT)** — `1dbf334` + `57ec22a` + `2245cc7` M8.7 cron stats panel + pack manifest cleanup + EOD docs
  - Service(s) redeployed: Worker (new DB writes), api (new endpoint + DB helpers), frontend (Jobs page CronStatsPanel)
  - Observed cache gap: TBD — pushed at 19:47 UTC
  - Notes: Three commits in one push. (1) `2245cc7` adds `trigger_levels_phase2` markers to 7 cross-style packs (cleanup; no engine path change). (2) `57ec22a` is the cron-cycle stats panel — Worker now writes one row per user per cycle to `algo_history_cron_cycles` (best-effort, swallows failures), prunes to last 200 rows every 10th cycle. New API endpoint `GET /api/jobs/cron-stats/algo-history`. Frontend: CronStatsPanel above Jobs list with Fresh/Normal/Stale buckets, last-5-cycle insertion totals, Starvation flag when oldest stamp >3x cycle interval. Migration `algo_history_cron_cycles.sql` already applied on Supabase by Kevin. (3) `1dbf334` is docs/EOD writeup + Roadmap milestones.

- **18:01 UTC (12:01 MT)** — `508cb89` M8.7 AI pack-creation guardrail: trigger_levels enforcement
  - Service(s) redeployed: api (pack_spec/pack_registry/pack_builder updates)
  - Observed cache gap: none expected (additive validation)
  - Notes: Three-layer guard — `pack_spec.audit_trigger_levels()` regex, install-time warnings via `pack_registry.scan_and_load_all`, AI builder treats audit warnings as errors in `validate_parsed_response`. New manifest schema field `trigger_levels_phase2` for intentional non-static markers.

- **17:51 UTC (11:51 MT)** — `169df09` M8.7 Hi-Fi Phase 1: signal-exit refinement via user-pack trigger_levels
  - Service(s) redeployed: api (Hi-Fi Pass 2 endpoints)
  - Observed cache gap: none expected (refinement is endpoint-driven, not cron-fired automatically yet)
  - Notes: New `_walk_1s_for_level_cross` walker + `pack_registry.get_trigger_level_spec()` resolver + `bar_df` plumbed into `_hifi_resolve_trades`. Covers `eppv3`/`eppv4`/`utv4` packs (declare static-level cross semantics). Phase 2 (indicator-vs-indicator / value-vs-threshold / dynamic) deferred — design seed includes new exec_type X discussion.

## 2026-05-04

## 2026-05-05

- **16:05 UTC (10:05 MT)** — env flag `WS_AGG_SHADOW_ENABLED=true` set on Worker (no commit)
  - Service(s) redeployed: Worker (auto-triggered by env-var change)
  - Observed cache gap: ~1-2 min during Worker restart
  - Notes: Activates the WsAggMinuteBuilder shipped at `ace1fe7`. After warmup, expect `source='ws_agg'` 1Min rows to appear in live_bars for SPY (10Sec subs already trigger A.* subscription) and TSLA. Backup branch `dev-backup-pre-wsagg-flag-on-2026-05-05` pushed pre-flip.

- **21:01 UTC (15:01 MT)** — `ee2c4c5` empty commit to recover from stuck/failed builds at 19:42-19:44
  - Service(s) redeployed: Worker (clean rebuild), frontend (auto-recovered)
  - Observed cache gap: ~3 min during Worker restart
  - Notes: 3 rapid pushes earlier (Phase C wire + default flip + bulk script) caused Railway to queue 3 conflicting builds — 2 stuck BUILDING + 1 FAILED. Worker kept serving the 12:52 deploy (Phase C code, no default flip) until this empty-commit kicked a clean rebuild. Post-recovery verification: all 27 monitored strategies have explicit `config.live_model='ws_agg_locked'`, 12 Polygon channels subscribed (6 AM + 6 A), Phase C dispatch confirmed working for AAPL/AMD/SPY/TSLA. META/TSLL ws_agg=0 due to A.* events only firing on actual trade activity (low post-market volume) — expected behavior, will normalize during RTH. 1Min alerts silent post-market because most are session=RTH and we're past 16:00 ET — also expected.

- **19:45 UTC (13:45 MT)** — `f971eeb` + `b1ad5c0` M8.7 Phase C: live_model default flipped to ws_agg_locked + bulk DB migration
  - Service(s) redeployed: Worker (auto-rebuilds on default flip code change)
  - Observed cache gap: ~1-2 min during Worker restart
  - Notes: Default live_model flipped from ws_with_corrections → ws_agg_locked. Bulk update via _bulk_set_live_model.py set explicit `config.live_model='ws_agg_locked'` on all 39 strategies. End state: every monitor reports ws_agg_locked, A.* subscribes universally on next reconnect (4 → ~9 symbols, 2.25× per-second forming-bar load — within budget). Backup: `dev-backup-pre-phase-c-2026-05-05` + snapshot `/tmp/strategies_live_model_snapshot_20260505T194258Z.json`. Revert path: re-run `_bulk_set_live_model.py NULL` then revert default in code (~3 min). Supabase had a 522/525 outage during prep window; user restarted to recover.

- **18:55 UTC (12:55 MT)** — `74ff56c` M8.7 Phase C: live engine dispatch + universal A.* subscription
  - Service(s) redeployed: Worker
  - Observed cache gap: ~1-2 min during Worker restart (RTH session active)
  - Notes: Wires live_model into StrategyMonitor + bar-close gate + ws_agg dispatch path. A.* subscription gate gains has_ws_agg condition (selecting ws_agg_locked on a strategy auto-subscribes A.<symbol>). Flipped ws_agg_locked.available=True so ModelsCard offers it. Default UNCHANGED at ws_with_corrections to avoid re-triggering M8.5 forming-bar latency regression. Backup branch `dev-backup-pre-phase-c-2026-05-05` pushed pre-change.

- **16:30 UTC (10:30 MT)** — `2ee8a24` M8.7 Phase A+B: strategy model registry + admin pages
  - Service(s) redeployed: api (new strategy_models_admin router), frontend (two new admin pages + sidebar entries)
  - Observed cache gap: N/A (additive, no engine code changes)
  - Notes: Refreshes strategy_models.py with ws_agg_locked / ws_agg_with_rest_backfill (live) and cache_locked / cache_corrected (backtest), all `available=False` until Phase C/D/E ship. Removes 3 unused M3 placeholder backtest IDs. New admin routes /admin/live-models and /admin/backtest-models surface usage counts + per-model strategy lists. Backup branch `dev-backup-pre-strategy-models-2026-05-05` pushed pre-change. Plan archived at `~/.claude/plans/synchronous-tickling-yeti.md`.

- **15:50 UTC (09:50 MT)** — `ace1fe7` WsAggMinuteBuilder shadow (disabled by default)
  - Service(s) redeployed: Worker (no-op until `WS_AGG_SHADOW_ENABLED=true`)
  - Observed cache gap: N/A (additive code path, env flag off)
  - Notes: Lands the Mode-2 candidate aggregator + validator. To activate: `WS_AGG_SHADOW_ENABLED=true` on Railway Worker, restart, wait ~30 min for paired bars, run `src/_validate_ws_agg.py` from inside `src/`. Backup branch `dev-backup-pre-wsagg-2026-05-05` pushed pre-change. Phase 1 scope: only writes for symbols where A.* is already subscribed (SPY, TSLA today).

## 2026-05-04

- **23:35 UTC (17:35 MT)** — `e30bfc6` M8.7 admin/data-health: custom date range filter
  - Service(s) redeployed: api (data_health router accepts start/end), frontend (V1 mode toggle + datetime picker)
  - Observed cache gap: N/A (read-only diagnostic)
  - Notes: Rolling mode (1h/4h/RTH/24h) is default; Custom mode reveals datetime-local inputs in user's chartPrefs.timezone with sec precision. Useful for "show me coverage since the last deploy without rolling-window noise." TV CSV stability test result also recorded — TV revises closed bars within ~5 min, confirming `live_model='latest'` should remain default.

- **23:30 UTC (17:30 MT)** — `6291a9a` M8.7 admin/data-health: per-(symbol, tf) cache coverage dashboard
  - Service(s) redeployed: api (new router), frontend (new page + sidebar entry)
  - Observed cache gap: N/A (read-only diagnostic page)
  - Notes: Backs the "is data being collected" question. Surfaces AM-stream loss pattern (AAPL/AMD/SPY 1Min ~35-40% coverage vs 10Sec 97%) plus subscribed-but-empty entries (GME, INTC, NVDA, TSLA/1Min). Backup branch `dev-backup-pre-data-health-2026-05-04` pushed.

- **23:15 UTC (17:15 MT)** — `1cca741` M8.7 M5: Custom picker uses user's tz + Apply forces refit
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Custom datetime inputs now format/parse using `chartPrefs.timezone` (matches chart axis 1:1) with DST-aware two-pass refinement. Apply button now triggers chart refit by remounting via React `key` keyed on `${windowStart}-${windowEnd}` — fixes the silent-no-update bug where setData preserved the previous visible range across window commits.

- **22:50 UTC (16:50 MT)** — `2d6ea98` M8.7 M5: Custom timestamp picker + bar-count diagnostic
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Adds explicit start/end datetime-local inputs (sec precision) so trader can target a specific trade window for forensic diagnosis. Plus robust empty/inverted-intersection handling and per-lens bar counts in header — strat 149 first-write empty case now shows "Algo X / Alert 0 bars" with diagnostic message instead of blank card. Backup branch `dev-backup-pre-timestamp-picker-2026-05-04` pushed pre-change.

- **22:25 UTC (16:25 MT)** — `0c368e8` M8.7 M5: lens-extent intersection + trigger_prefix relevance match
  - Service(s) redeployed: frontend (next.js), api (strategies router)
  - Observed cache gap: N/A (frontend + API logic only)
  - Notes: Two fixes from v2 smoke test. (a) LabReplayPanel uses INTERSECTION of both lenses' candle extents — Algo and Alert now share start AND end candles, eliminating the REST-trailing-WS visual mismatch. (b) `_build_chart_response_from_df` relevance check gains a third path: match via template's `trigger_prefix`. Fixes EPP v4 (and any user pack where group.id ≠ trigger_prefix) — affects both Chart & Trades and Lab tab.

- **22:00 UTC (16:00 MT)** — `14e211a` M8.7 M5 v2: unified Lab Replay panel
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Architectural pivot after smoke-test feedback. SyncedChartPane gets a `currentTime` prop for scrub mode (additive); new LabReplayPanel wraps two SyncedChartPanes (Algo REST + Alert cache) sharing one scrub head + window picker (Last 1h / 4h / Today / All). Replaces V1 ChartReplayCard's parallel renderer — indicators, oscillators, heatmap and markers all render identically on both lenses now. Net -290 lines on StrategyDetailPage.

- **19:40 UTC (13:40 MT)** — `dc91ba9` M8.7 M5: Replay marker parity (algo + and alert × price-level crosses)
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Round 2 of M5 smoke-test feedback. Replay now mirrors Chart & Trades' price-level cross markers (4 invisible Line series with shape='cross' for algo, 'xcross' for alert; filtered by replay scrub time). Indicator overlays still empty for strats whose user packs don't expose chart columns (e.g. ema_pp_v4) — separate backend issue.

- **18:20 UTC (12:20 MT)** — `8232289` M8.7 M5 fix: Replay parity (candleCount slice + trade markers + rightOffset)
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Monday smoke test found V1 Replay rendering candles only. Fix slices to candleCount, builds entry/exit markers, forwards rightOffset, and keys chart id on overlay count. Plan + memory updated. Backup branch `dev-backup-pre-replay-fix-2026-05-04` pushed.

- **15:13 UTC (09:13 MT)** — `53cac54` M8.7 M4 fix: carry indicator_snapshot through DBAlertDispatcher
  - Service(s) redeployed: Worker
  - Observed cache gap: TBD (RTH push — expect ~1-2 min reconnect window)
  - Notes: Monday RTH validation found 0/15 recent alerts had `data.indicator_snapshot`. Saturday's M4 only patched `AlertDispatcher.dispatch` in ralph_engine.py, but worker.py overrides `engine.dispatcher` with `DBAlertDispatcher`. Mirror added — same 3-line pattern. Validate post-restart by checking `data->indicator_snapshot` on next RTH alerts.

## 2026-05-02

- **18:30 UTC (12:30 MT)** — `5515401` Archive M8.7 Saturday plan + update Roadmap_To_Scale with M1-M6 status
  - Service(s) redeployed: all (docs-only push)
  - Observed cache gap: ~1-2 min during Worker restart (markets closed)
  - Notes: pure documentation. Plan archived in repo. Roadmap reflects shipped vs remaining work.

- **17:57 UTC (11:57 MT)** — `064f9a3` M8.7 M4+M5+M6: alert snapshots, Lab tab replay, engine-state capture
  - Service(s) redeployed: Worker (M4 snapshot, M6 writer hook), api (no functional change), frontend (M4 tooltip, M5 replay)
  - Observed cache gap: ~1-2 min during Worker restart (markets closed)
  - Notes: M4-M6 of weekend plan. Required manual steps: (1) apply src/migrations/bar_engine_states_table.sql in Supabase, (2) set BAR_ENGINE_STATE_WRITE_ENABLED=true on Worker.

- **16:49 UTC (10:49 MT)** — `8b40eff` M8.7 M3: backtest_model + live_model placeholder (schema + UI)
  - Service(s) redeployed: api (new endpoint + default fill), frontend (Models card + badges), Worker (no functional change)
  - Observed cache gap: ~1-2 min during Worker restart
  - Notes: M3 of weekend plan. Models abstraction recorded on each strategy; engine dispatch comes later. UI: ModelsCard on Config tab, read-only badges in page header.

- **16:44 UTC (10:44 MT)** — `7bd71e7` M8.7 Phase 2: Lab tab Alert Lens uses cache-derived indicators + heatmap
  - Service(s) redeployed: api (new endpoint + refactored helper), frontend (new hook + Lab tab wiring), Worker (no functional change)
  - Observed cache gap: ~1-2 min during Worker container restart (markets closed)
  - Notes: M2 of weekend plan. Closes the Phase 1 caveat — Alert Lens indicators/heatmap now computed from live_bars cache. New endpoint /chart-data-cache; refactored _build_chart_response_from_df shared with /chart-data.

- **16:36 UTC (10:36 MT)** — `a15461f` M8.7 fix: handle Polygon WS rebroadcasts as corrections, not duplicates
  - Service(s) redeployed: Worker (engine code change), api/frontend rebuilt
  - Observed cache gap: ~1-2 min during Worker container restart (markets closed → no live trading impact)
  - Notes: M1 of weekend plan. Fixes the duplicate-bar bug discovered Friday EOD. accept_bar/accept_second_bar now detect rebroadcasts and replace history rows instead of appending. IncrementalIndicatorEngine gets new recompute_from_history method. Worker restart rebuilds in-memory state from REST warmup (clean). Live validation deferred to Monday RTH.

## 2026-05-01

- **20:13 UTC (14:13 MT)** — `671dd6b` EOD checkpoint: weekend plan, deploy log, M8.7 findings + TV-cache compare data
  - Service(s) redeployed: all (docs-only push but Railway watchPatterns=[])
  - Observed cache gap: ~1 min during Worker container restart
  - Notes: pure documentation commit. No code/engine changes. Captures Friday's findings (duplicate-bar bug, TV≈REST, TF drift scaling) and Saturday's plan.

- **19:41 UTC (13:41 MT)** — `5ba2dd7` Lab tab: Phase 1 — side-by-side Algo Lens / Alert Lens
  - Service(s) redeployed: frontend (Worker rebuilt but no functional change)
  - Observed cache gap: ~1 min during Worker restart
  - Notes: replaces single-chart toggle with always-visible side-by-side layout. Right side has its own First/Latest sub-toggle. Phase 1 ships candle differences only — indicators/heatmap on right are still REST-derived (Phase 2 will fix).

- **19:23 UTC (13:23 MT)** — `106f1ca` Lab tab: data-source toggle for live-WS chart view (M8.7)
  - Service(s) redeployed: api (new endpoint), frontend (new hook + UI). Worker rebuilt but no code change there.
  - Observed cache gap: ~1 min during Worker restart. None expected from api/frontend redeploy.
  - Notes: ships the read-only side of M8.7. Backend `/cache-bars` endpoint reads from live_bars; frontend toggle on Lab tab swaps `labChartTabData.bars` between REST / WS-latest / WS-first. Indicators/heatmap stay REST-derived (full read path = M8.7d, not in scope).

- **19:06 UTC (13:06 MT)** — `ef28f0b` Strategy Detail: add 'Chart & Trades (Lab)' tab for divergence visualization
  - Service(s) redeployed: frontend (api/Worker/streamlit also rebuilt — Railway redeploys all on dev push, but no functional change to those)
  - Observed cache gap: minimal — frontend redeploy doesn't affect Worker bar recording. Worker container also rebuilt; ~1-2 min restart gap if affected.
  - Notes: pure frontend addition. New tab parallel to existing Chart & Trades. Reuses existing chart data and trade-to-alert matching logic. Adds a Price Divergence panel surfacing algo vs alert price gap per matched pair.

- **18:30 UTC (12:30 MT)** — `21cd636` M8.7 hotfix #2: don't overwrite 1Min canonical bars with stale partials
  - Service(s) redeployed: Worker (deploy SUCCESS); api, frontend, streamlit too — every dev push redeploys all
  - Observed cache gap: 1-2 min around 18:30 UTC during Worker container restart
  - Notes: yesterday's hotfix added a flush_stale_bars write hook that fired correctly for sub-minute primaries but also fired incorrectly for 60s+ builders, overwriting canonical AM bars with chart-visual partial data. This fix gates the flush write to `tf_seconds < 60`. Cleanup of 346 corrupted 1Min rows (volume <50% of first_volume) executed inline post-deploy — restored from `first_*` columns. Going forward any `first_close ≠ close` in the 1Min cache reflects a genuine Polygon WS rebroadcast correction, not the bug.

- **15:21 UTC (09:21 MT)** — `760b64c` trigger Worker redeploy (empty commit)
  - Service(s) redeployed: Worker (api unaffected)
  - Observed cache gap: `15:23:00` 1Min bar missing across AAPL/AMD/META/SPY/TSLL — confirmed by `_validate_live_bars_cache.py`
  - Notes: needed because the prior `2dd409b` hotfix deploy was stuck in BUILDING with `deploymentStopped: true` (Railway state inconsistency). Empty commit forced a fresh build that succeeded ~15:24 UTC.

- **15:11 UTC (09:11 MT)** — `2dd409b` M8.7 hotfix: add live_bars write hook to flush_stale_bars path
  - Service(s) redeployed: Worker (build cancelled — never went live)
  - Observed cache gap: none (deploy never completed, old code kept running)
  - Notes: build entered `BUILDING` then got marked `deploymentStopped` without progressing. Required the empty-commit retrigger above.

## 2026-04-30

- **18:46 MT (00:46 UTC, 2026-05-01)** — `e6d946f` M8.7: live_bars cache write path (fire-and-forget, flag-gated)
  - Service(s) redeployed: Worker, api
  - Observed cache gap: n/a (this was the first deploy that introduced live_bars; nothing to gap before it)
  - Notes: initial M8.7 ship. `LIVE_BAR_CACHE_WRITE_ENABLED=true` set on Worker via `railway variables --skip-deploys` immediately before. Deploy succeeded cleanly in ~50s.

---

## How to use this log

When investigating a cache gap, alert miss, or WebSocket reconnect:

1. Note the affected timestamp window from the data.
2. Check this log for any entry within ±5 min of that window.
3. If a deploy lines up, the gap is expected and you can move on.
4. If no deploy lines up, dig deeper — it's a real issue.

When you (Claude) push code that triggers a Worker / api / frontend
redeploy, append a new entry at the **top** of the most recent date
section before reporting back to Kevin. Pull the commit SHA and time
from `git log` after pushing. Notes about observed gaps can be added
later once validated.
