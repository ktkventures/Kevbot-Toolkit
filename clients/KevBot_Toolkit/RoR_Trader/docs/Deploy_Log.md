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

## 2026-05-04

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
