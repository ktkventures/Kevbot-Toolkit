# RoR Trader — STATUS (read me first)

**Updated: 2026-06-27.** This is the single doc to open for "what are we doing and why."
It's the live priority list + current state. Deeper detail lives in the linked docs (kept
in their current locations for now — see Doc Map). New observations get logged here (and in
`Roadmap_Divergence_Hunting.md` for divergence specifics). Work board = the **dev_tasks
admin page** (`/admin/tasks`); next-batch items are **#45–#53** (phase 3).

## North star
Backtest and live produce the **same trades within ~5s**, so we can **trade reliably**.
Current focus: **M-RS3 parallelization (#46)** so a full trustworthy fleet recompute is fast/routine.

## 🟢 2026-07-01 — M-RS4 continuous engine PROVEN LIVE; Fix 1a + Fix 2 shipped (gated) (READ FIRST)
Big session. **The M-RS4 Phase 3 continuous backtest engine works in production** — armed the `shadow-worker`
on 8 SPY canaries (276–283); it wrote real `backtest_%` trades that paired with live alerts → **93–100%
combined on the Strategy Health page**. The architecture is validated. Details + Railway flags + re-arm
procedure: memory `project_mrs4_shadow_arming_2026-07-01`.

**Shipped to dev (all gated OFF by default):**
- **Fix 1a — incremental Hi-Fi load** (PR #18): `db.load_trades_admin(created_at_gte=...)` +
  `run_hifi_pass2` pushes the `last_hifi_pass_at` watermark into SQL under `RORT_HIFI_INCREMENTAL_LOAD`.
  Byte-identical (SQL==PY, walked-set identical on 5 canaries; load −49% median/−88% p90).
  `docs/_active/Scope_Fix1_Hifi_Incremental_Load.md`.
- **Fix 2 — shadow anti-starvation** (PR #19): `RORT_SHADOW_FAIR_ORDER` (oldest-polled first),
  `RORT_SHADOW_MAX_ADVANCE_S` (bounded warm advance), `RORT_SHADOW_KPI_ASYNC` (KPI/Hi-Fi off the poll
  thread). **Also brought the anchor-hydrate fix (b14657c) to dev — it was never merged** (without it a
  cold bootstrap 409-floods on every existing trade). Byte-identical by construction + unit-tested.
- Offline **Hi-Fi gate** in `_shadow_manager_validate.py` (`VALIDATE_HIFI`) — refined stop/target exits
  intra-candle + byte-identical (green on 280/288).

**KEY correction — Strategy Health = backtest ↔ ALERTS, NOT the algo lane.** The morning's all-TBD wasn't a
crash: `data_worker_streaming_enabled` has been **FALSE since 06-05**, so nothing produces intraday backtest
trades (only the overnight recompute, last ~01:41 UTC) → intraday Health goes TBD. The shadow is the
continuous replacement. TBD literally = the backtest lane hasn't reached the alert's time yet.

**OPEN (next session):** the Fix 2 re-arm wrote 0 trades in ~22 min (with flags on AND off) — NOT the flags,
NOT a `poll()` bug (local repro: runs clean, no hang). Likely the anchor now sits at ~16:26 from the earlier
successful arm (little new to write) OR container bootstrap slowness. **Shadow left INERT (button)**; the 8
lanes are fresh to ~16:50. First check next time: did 276–283 actually produce alerts/trades 16:26→now?
Then re-arm (lane_mode=shadow + dry_run=0 + Fix2 flags), confirm 8/8 incl 276, widen to 34, Step F.

**CI note:** the "Fidelity Gate (synthetic parity)" check is chronically RED on every dev merge (missing
`pytest` in the runner, #52) — not a real failure; the 31 fidelity assertions pass. Safe to merge past.

## 🟢 2026-06-30 night — M-RS4 Phase 3 SAFE PREP done (gate + scope, no fidelity code) (READ FIRST)
Session split per `Plan_M-RS4_Phase3_Scheduling_Fixes.md` §0: tonight = safe prep, no fidelity-critical
code; the Hi-Fi change + live validation are a **morning market-hours** job. Shipped tonight (nothing
touches `run_hifi_pass2`/KPI load paths):
- **Offline Hi-Fi gate BUILT** — extended `src/_shadow_manager_validate.py` (`VALIDATE_HIFI=1`, default
  on). After the existing bar-resolution byte-identity check it refines BOTH lanes with the real
  `_hifi_resolve_trades` and asserts (1) still byte-identical post-Hi-Fi, (2) every eligible L-type
  stop/target exit landed **intra-candle** (off the primary-TF bar boundary). Degrades to **AMBER** when
  no 1-sec bars (off-hours) — never false-greens, never changes the Step C verdict. Smoke-run sid 280:
  Step C green (unchanged), Hi-Fi AMBER (0 eligible in the dead off-hours window). Turns green in the
  morning on live 1-sec data. **This is the gate the morning Fix 1 builds against.**
- **Fix 1 (Hi-Fi half) SCOPED** — `docs/_active/Scope_Fix1_Hifi_Incremental_Load.md`. Recommendation:
  push the existing `last_hifi_pass_at` watermark into SQL (`load_trades_admin(created_at_gte=...)`),
  which is **byte-identical to today's incremental filter** and kills the reload-all. Explicit gotcha:
  do NOT window by `entry_fill_ts` (backfills write old-entry/new-created trades → would strand them
  bar-aligned). KPI/equity reload-all is a separate mechanism (in-memory `kpi_series`, parent §2).
- **Emblem BACKEND — already satisfied, no edit.** The emblem's data source is `/algo-trades` →
  `load_trades_admin` → `_row_to_trade`, which already merges `provisional` (column) + `hifi_resolved`
  (`data` JSONB). Both flow today; `hifi_resolved` is already in the frontend `TradeDTO`. Remaining work
  = frontend emblem column + Health TBD gate (morning, §0b).

**MORNING pickup:** implement Fix 1 against the ready gate (turn Hi-Fi gate green on ≥3 canaries with
live 1-sec data) → arm live → confirm intra-candle exits + Health pairing → F/G. Plus emblem frontend +
Health TBD gate.

## 🟢 2026-06-27 — coarse + truncation fixes shipped, full fleet recompute, divergence honest (READ FIRST)
**Shipped + deployed to dev (all validated, fidelity suite 18/18):**
- **Coarse-secondary-from-1Min fix** (`RORT_COARSE_SECONDARY_FROM_1MIN=1` on api) — a sub-min-primary +
  coarse(≥1H) gate strategy no longer loads ~363d of the primary to build the secondary; builds it from
  1Min (cache-accelerated). Fixed the UAD **hang** (sid 338: hang→76s). Byte-identical for price gates;
  daily-volume shifts only volume-daily gates (e.g. sid 311 VWAP +5). `strategy_data.load_strategy_data`.
- **Truncation preserve-range fix** (`RORT_UAD_PRESERVE_RANGE`, default ON) — full recompute no longer
  truncates backtest history to `data_days` when a strategy lacks `backtest_start_date`. `forward_test_service._do_recompute`.
- **TV export** (TV agent, PRs #2+#3 merged): emitter coverage 14→64/68, readiness badge, + EOD
  entry-suppression in `unified_engine.py` **gated OFF** (`RORT_SUPPRESS_EOD_REENTRY`, inert until flipped — task #53).
- **Full-fleet UAD pass (overnight):** 67/67 in **215 min SEQUENTIAL** (~193s/strat) — clean, zero data loss,
  backtest now trustworthy fleet-wide. This is the **M-RS3 baseline** (parallelize → ~7-8×).

**Divergence note (don't panic):** Strategy-Health combined % dropped <90% on many strategies AFTER the
recompute. NOT a regression — the old append lane under-produced (~50% recent), inflating the score; the
full recompute revealed the true ~20-40% backtest-vs-live gap (the divergence-hunting target). Backtest is
correct (suite 18/18, coarse byte-identical). A day-specific spike on 06-26 (clean canaries ~4%→13%) ties
to our heavy deploy-churn that day (RTH Worker restarts), not code — **WATCH #45: confirm it returns to
~4% on a clean Monday RTH.** By-Deploy health tab is blank because `deploy_history.json` is stale since
06-12 (#50).

**THIS WEEKEND = M-RS3 (#46)** parallel recompute + dedicated update service (+ `compute_parity=False`
bulk lever #47, folds in algo-lane #43). Logic/divergence-sensitive changes deferred to Monday (deploy+monitor).
Next-batch board: #45 (watch), #46/#47 (M-RS3), #48 chart-coarse, #49 health-window scope, #50 deploy_history,
#51 gap-healer, #52 CI pytest, #53 EOD flip.

## 🟢 2026-06-25 — M-RS2 Phase 2 read path COMPLETE + Hi-Fi load-once (READ FIRST)
- **Two bar caches now have canonical names** (Kevin): **"Live Bars"** (`live_bars`, WS,
  immutable decision-time) vs **"REST Bars"** (`bar_cache`, Polygon REST, revisable). New
  canonical doc `docs/_active/Two_Bar_Caches_DEFINITIONS.md` + red-line header banners on
  `bar_cache.py` / `live_bars_writer.py`. The 2-3× recurring conflation risk is now codified.
- **Read path wired + byte-identical** (`bar_cache.read_bars()` direct-PG + `cached_load_market_data`
  native-TF): get_strategy_trades(325) cache-on = **442 trades, byte-identical, 1.8×** (382s vs 700s).
- **Hi-Fi Pass 2 load-once** (`prime_1s_cache_from_rest_bars`): ONE direct-PG read over the whole
  trade span feeds the per-day `_1s_cache` (Kevin's "pull big swaths once"). Validated end-to-end
  on **sid 321 (799 trades): byte-identical across 9 cols, 4.26× (13.9s→3.3s)**.
- **Composite full-Update-All estimate: ~1.5–2× faster** (Hi-Fi 4.3× + load 1.8×; engine/persist/
  secondary unchanged = M-RS3 territory). On top of M-RS1 already shipped.
- **⚠️ Two scary findings DEBUNKED as test-harness artifacts:** (1) "Polygon revises 1s over days"
  = a `1e-9` threshold catching **sub-penny** float values; a settled day (06-16) is **literally
  zero-diff**, and Polygon docs confirm intraday locks after 15min + EOD. (2) "REST Bars 1s ~50%
  incomplete" = my baseline `load_from_polygon(end_date=d+1)` pulled **2 days** (inclusive range)
  vs the cache's 1; clean per-day row counts match **exactly**. REST Bars is sound + complete + `float8`.
- **All behind `BAR_CACHE_ENABLED` + a DSN; transparent Polygon fallback when off/uncached.**
- **DEPLOY (option A):** branch `feat/m-rs2-phase2-readpath` → dev tonight (markets closed = safest:
  no live-capture/divergence disruption). Needs `psycopg[binary]` in requirements (added) +
  Kevin sets `SUPABASE_CONNECTION_STRING` on **api + worker**, then `BAR_CACHE_ENABLED=1`. Inert until set.
- **NEXT (new, Kevin's idea = M-RS2 Phase 3):** extend the load-once direct-PG read + cross-strategy
  sharing to the **algo / Live Bars lane** (dedup per-strategy `live_bars` reads). Live Bars supply
  already exists (WS writer) — win is a fast *reader* + sharing, NOT a new supply, and it must read
  **Live Bars, never REST** (keep decision-time fidelity). Board #42.

## 🟢 2026-06-24 PM — M-RS2 Phase 1 DEPLOYED + speed test (read first)
- **Deployed to dev** (deb96e5 M-RS1 + dc053b3 M-RS2 Phase1 + e6b9221 deploy-log; backup
  branch `dev-backup-2026-06-24-pre-mrs-deploy`). M-RS1 now ACTIVE in prod
  (`RORT_RIGHTSIZE_WARMUP=1`). Admin page **/admin/bar-cache** live. Tasks board got an **ID
  column** (#33–#40 = today).
- **Bar-cache supply:** TSLA **1Sec (~7.0M/yr) + 1Min (235k/yr) backfilled**, kept current by
  the worker maintain cron (`BAR_CACHE_MAINTAIN_ENABLED`). **Next: backfill SPY** (36
  strategies — the largest cohort; TSLA=27). TSLL/DIA/KO = 1 each, low priority.
- **Speed test (CORRECTED — read this):** direct-Postgres is the fastest reader; cache 1-second
  speedup **scales with window: ~1.3× (1 day) → ~4.2× (1 week)**, more beyond. Whole-period
  single read works (year = 7M rows in one read). **Phase 2 read path = direct Postgres**
  (psycopg + `SUPABASE_CONNECTION_STRING`), loading the whole backtest period in a few big reads
  (native 1Min + native sub-minute for volume fidelity + all 1Sec for Hi-Fi), NOT many small
  fetches (Kevin's idea — validated).
  - **⚠️ Correction:** earlier "Polygon 1-second is unreliable/aging/throttled" claims were a
    test-harness bug (positional arg: `load_from_polygon('TSLA','1Sec',…)` put `'1Sec'` in the
    `days` param → timeframe defaulted to 1Min). **Polygon serves full 1-second; the cache
    matches it exactly.** No fidelity/retention issue. Always pass `timeframe=` as a keyword.
- **Calibrated expectation:** single-Update-All cache win is **moderate** (M-RS1 already shrank
  the load); the big wins are fleet-scale (one capture serves all strategies), Hi-Fi (one big
  read vs many fetches), and large windows.
- **Remaining for the definitive Update-All time-saved number (next session, ~1 focused block):**
  (1) baseline a current Update-All; (2) build `bar_cache.read_bars()` (direct-PG, raw cursor,
  whole-period); (3) wire into the data-load path behind a flag + byte-identical gate; (4) A/B
  old-vs-new. Mass Builder / Update-New / charts then inherit it. Doc:
  `Design_M-RS2_Shared_Bar_Store.md`. Board: #30/#38.

## 🟢 2026-06-24 EOD — recompute-scalability day (READ FIRST)
Three big wins, all behind kill-switches / validated:

**1. M-RS1 — recompute warmup right-sized (SHIPPED + ENABLED).** Full recompute used a flat
`visible_days×2` warmup (no TF awareness) → loaded ~1yr of bars for a coarse-gate strategy.
Replaced with per-TF warmup (`PRIMARY_WARMUP_BARS=1200` + `SECONDARY=250`, matching the live
engine's 250-bar standard) in `strategy_data.compute_warmup_days`, behind
`RORT_RIGHTSIZE_WARMUP`. **Validated byte-identical** (308/309/325 local + **Railway 174 = 188
trades = local baseline**), **~3× faster** on the common cases. **ENABLED in prod.** Net win
for the fleet majority; ~neutral for normal-window coarse gates. Docs:
`Plan_M-RS1_Warmup_Rightsizing.md`, `Recompute_Scalability_Findings.md`. (dev_tasks #33)

**2. Inherit-position append fix — VALIDATED on the real path.** `RORT_RESUME_INHERIT_POSITION`
(shipped 2026-06-23) enabled today; **Update-New on 321 → combined% 45%→84.7%@5s** (near the
~90% recompute truth), fill timing median 0s. The append under-production (a primary phantom
source) is largely solved. Consider flipping default ON fleet-wide (dev_tasks #34). Detail:
memory `project_append_underproduces_recent_window`.

**3. M-RS2 — REST canonical bar store, Phase 1 backend COMPLETE.** The fleet-wide lever:
fetch each ticker's native bars ONCE, reuse everywhere (stop re-pulling months of bars per
strategy). **Design aligned with Kevin: supply-first.**
- **Phase 1 (the supply) — built + validated, behind kill-switches:** `1Sec` Polygon mapping
  (was a silent fallback to 1-minute — latent bug), revision-horizon guard (settled/unsettled
  split, `bar_cache.py`), `backfill_symbol` (seed) + `maintain_symbol` (keep-updated +
  revision-refresh) — both validated on TSLA — `bar_cache_config` table (migration applied) +
  CRUD + `maintain_all_enabled` + worker cron (`BAR_CACHE_MAINTAIN_ENABLED`, OFF). All inert
  (`BAR_CACHE_ENABLED` off).
- **⚠️ KEY FIDELITY FINDING:** 1s→1Min OHLC is **byte-identical** but **VOLUME diverges**
  (auction/condition-trade aggregation) → "build every TF from 1s" breaks volume. **Pivoted to
  cache NATIVE bars per (symbol, timeframe)** — store native 1s + native 1Min; sub-minute
  materializes from 1s, coarse from 1Min, all matched 1:1 in Phase 2 (serving).
- **Remaining Phase 1:** admin page (frontend CRUD + status + backfill-now) + deploy the worker
  cron (push ask-first; OFF by default so safe). dev_tasks #35/#36/#37.
- **Phase 2 (later):** wire backtest/append reads to the cache 1:1, validated byte-identical
  (handle volume per-feature). dev_tasks #30/#38. Doc: `Design_M-RS2_Shared_Bar_Store.md`.

**Parked / loose ends:** TV export — reverted today (generator drifted from validated
Checkpoint C: adds a corner table C lacks; ut_bot has only a gate emitter, needs an
entry/exit-trigger emitter). Treat as its own focused session (dev_tasks #39). Layer-1 health
tool `_divergence_walkthrough --denom` crashes (tolerance_seconds Query, commit 1269d46) —
one-line fix (dev_tasks #40).

**Tomorrow's moves:** (a) build the M-RS2 admin page + deploy the worker cron, then backfill a
couple tickers' 1s+1min supply; (b) decide on flipping the inherit-position default ON; (c)
optionally start M-RS2 Phase 2 (read-path 1:1 wiring) on one strategy.

## 🔴 2026-06-23 EOD — MAJOR root cause + fix (READ FIRST tomorrow)

**Found + fixed the dominant phantom / low-combined% source.** The backtest **append**
(Update-New) was under-producing the recent window by ~50% (321: snapshot-resume lane 45%
live-alert match vs cold/full-recompute 92%). Root cause: **Tier 3 §8.2 "always-start-flat"
was applied at EVERY incremental append resume boundary**, dropping the trade open at each
boundary. ~1 lost/append compounds over ~34 session appends → ~half the trades vanish.
Cold/full **recompute = truth** (matches live alerts ~92% @5s, identical local↔Railway).
NOT local-vs-Railway, NOT user-pack warmup. Full detail: memory
`project_append_underproduces_recent_window`.

**FIX shipped (inert):** `inherit_position` flag — restore the boundary-open position on
resume so the trade continues. Commits `bb33332` + `64547cf` (the 2nd fixes a real bug the
first introduced: restoring a FLAT position zeroed entries via cooldown fields → now only
OPEN positions are inherited). **Kill-switch `RORT_RESUME_INHERIT_POSITION`, default OFF,
BACKTEST-LANE-ONLY** (live/algo/chart keep §8.2 always-flat — Kevin's TradingView/restart
intent preserved). Validated: unit (boundary trade recovered, ON=cold), FLAT-window ON==OFF,
Tier-3 contract 5/5 with default OFF, 97% on a cumulative harness (off-by-2 = harness omits
the edge-band-replace the real append does).

**TOMORROW'S FIRST MOVES (RTH):**
1. **Kevin sets `RORT_RESUME_INHERIT_POSITION=1` on api + Worker** (CLI set was permission-
   blocked tonight). Then real RTH append cron uses the fix.
2. **Compare 321 + 308 lanes (built by flagged live appends) to cold recomputes** = the
   real-path byte-identical proof. Watch combined% @5s on Strategy Health — should climb.
3. Green → flip default ON fleet-wide. Problem → flag OFF (instant kill-switch).

**Also shipped today:** hardened local-update tool `src/local_update.py` + skill
`local-update` (registry-safe, smoke-gated — every local backtest/append script MUST call
`scan_and_load_all()` or user-pack strategies silently return 0; memory
`feedback_local_script_pack_registry`). Strategy-health `tolerance_seconds` param (5s/10s).

**OPEN BLOCKER (the scale lever) — cProfile'd 2026-06-24, 3-milestone plan locked.** Full
recompute of a coarse-gate strategy (sid 325) = ~29% I/O + ~70% CPU, **both inflated by an
oversized warmup**: the full path loads + processes ~1yr of 1-min (`visible×2` flat
multiplier, no secondary-TF awareness) when the 4h gate needs only ~250 bars (~125 days).
cProfile split: load 353s / engine 378s / user-pack indicators 299s / interpreters 97s.
**The earlier "1-min bar-cache" framing was a mismeasurement** (95-day isolated load = 3s;
the real in-pipeline load is ~1yr). NEW PLAN — 3 milestones, cheapest-first, each
byte-identical + kill-switched (full detail: `_active/Recompute_Scalability_Findings.md`):
- **M-RS1 — Right-size warmup** (do FIRST, biggest bang/least risk): size warmup to the
  indicator/binding-TF requirement, tying into the standard that already exists
  (`ralph_engine._secondary_warmup_days`, 250-bar target; append path binding-bpd). Cuts the
  ~1yr load AND ~2/3 of engine+indicator bars. There's a TODO at `strategy_data.py:134`.
- **M-RS2 — Shared symbol-level 1-SECOND bar store** ("super dough", Kevin's idea): fetch
  once per ticker, reused by every strategy/backtest/append/live. 1s base (sub-minute
  strategies can't build from 1-min) + materialized 1-min layer. Settled-immutable +
  recent-always-refetch guard. ~0.5–0.7 GB/ticker/yr (~$6/mo for 50 tickers). Agent page to
  checkbox tickers + capture range. Speeds I/O + kills cross-strategy redundancy (NOT CPU).
- **M-RS3 — Parallel recompute + dedicated Railway update service** (throughput multiplier,
  vertical + horizontal); maybe decouple Hi-Fi (~8min) if still on the critical path.

Backup branch: `dev-backup-2026-06-23-pre-inherit-fix`. (Superseded: `Design_Recompute_Bar_Cache.md`.)

**Catalog Update-All decision (pending Kevin):** considering a full-catalog local Update-All
tonight to establish clean faithful baselines (recompute=truth, independent of the append
fix) — see my recommendation below / in chat. Risk = the 8 coarse-gate strategies may OOM.

## 🟢 2026-06-22 EOD — DONE TODAY + 2026-06-23 PLAN (read this first)

**Done today (all deployed):**
1. **Gen-gate live ROOT FIX (`880bdb3`) — CONFIRMED LIVE.** Root cause was NOT tz/timestamp:
   `RalphEngine.start` loaded general packs with no user context → 0 packs → all gated monitors
   empty → gate always failed. Fix loads via `load_general_packs_admin(user_id)`. Confirmed: sid
   337 (after-hours TOD gate) fired bull_flip ENTRIES in-window. See `feedback_gengate_live_string_ts`.
2. **Secondary-TF SNAPSHOT — speed work, VALIDATED + deployed (`e0b1e4a`, flag `RORT_SECONDARY_TF_SNAPSHOT=1` on api).**
   Update-All seeds the snapshot → Update-New on coarse-gate strategies drops from **~780s → ~18s
   (~43×)**, both lanes, byte-identical, lanes intact. (NOT the algo/cache_% lane — it's an extension
   of `engine_snapshot_b64`.) Detail: `Design_Secondary_TF_Snapshot.md`.
3. **XTF_BLOCK_DIAG deployed (`3da94eb`)** — captures cross-TF gate-blocks (327/328/329) at RTH open tomorrow.

**Tonight:** Kevin runs **Update-All Data** (after the `3da94eb` deploy settles) → seeds snapshots →
Update-New fast all day tomorrow.

**2026-06-23 PLAN (execution order):**
1. **Verify the speed work landed in prod** (~10 min): Update-All seeded the coarse strategies +
   a production Update-New runs in seconds.
2. **Confirm gen-gating in RTH:** 334 (session, open), 335 (11:30–14:00 ET), 336 (14:00–16:00 ET)
   fire in-window. Closes the loop (337 was after-hours).
3. **Bug-hunt divergences (main work, gate to live trading):**
   a. **Cross-TF live gating 327/328/329** (TOP) — confirmed live bug: live blocks entries the
      backtest takes. Use XTF_BLOCK_DIAG to pin, then fix.
   b. **313 full recompute = 0 trades vs lane = 455** — investigate the discrepancy (pre-existing).
   c. **Re-assess phantom/missed** (325/330/331/333…) with fresh lanes + gen-gate fixed.
4. **As fidelity solidifies:** pick first live-trading strategy; revisit service-split + cron for scale.

**PRE-STEP-3 (discuss first):** set up a **bug-hunt SKILL** — a standard, more-autonomous
implement+test procedure for divergence fixes (we do this a lot; codify it). Review with Kevin
before starting the divergence bug-hunt.

---

## ⚠️ PENDING MONDAY (2026-06-22) — ✅ RESOLVED (gen-gate root-fixed + confirmed live; see above)
Jun-19 was Juneteenth (market closed) → no live data, so the live-engine changes
below could NOT be validated. ONLY these touch live firing; everything else this
session is offline-validated (Tier 2 byte-identical) or display-only (Bug 1/equity).
- **Bug 5 — TF-scaled coarse-gate warmup** (`ralph_engine._load_warmup_df`).
  VALIDATE Mon: 313/312 (1d/4h gates) fire live + no live↔backtest divergence on
  the 1d/2m/5m-gated cohort. **ROLLBACK = flip `RORT_TF_SCALED_WARMUP=0` on Worker**
  (instant; reverts to pre-Bug-5 flat days=7). Commits: `0e19838`, `3159d96`.
- **1Min support fix** (`<=60s` native, `8c2fdcb`). VALIDATE Mon: a 1Min strategy
  warms + fires (no "Cannot resample to 1Min"). Covered by the same kill-switch.
- **310 RVOL** live sub-bug (forming-bar volume 2×) — NOT yet fixed; verify Mon.
Selective rollback: each feature is its own commit → `git revert <commit>` leaves
the rest intact. We are NOT in "roll back everything" territory.

## 2026-06-19/20 update — Mass Builder hardening (full log: `Session_2026-06-19_MassBuilder.md`)
All shipped to `dev` + deployed. Mass Builder now: won't OOM on multi-trigger/long runs
(**1.15** scoping, `RORT_MASS_SCOPE_CONFLUENCE` default on, byte-identical, RSS −55%);
persists results mid-run (**1.16**, `RORT_MASS_PARTIAL_FLUSH_SEC` 30s); nested per-**trigger-set**
drill-down on Mass Results (**1.17**) with fail-loud `backtests_failed` badge.
**P0 cross-pack silent-drop FIXED** (`7fb84ae`): pre-existing bug (NOT #21 scoping) — the HiFi
dough only tracked the first combo's exit because `_resolve_trigger_ids` reads
`exit_trigger_confluence_ids` before `exit_triggers`; mixed-pack searches silently kept only the
first pack. Engine-accurate inline re-run **pulled** (no warmup via `/api/backtest/run`) → task 1.18.
**RESUME: P1b** — drill-in button deep-linking to the Mass Builder edit view pre-filtered to a
trigger-set + add a trigger-set filter there (reuse `MassBuilderPage.tsx`, don't rebuild). Then P2 perf.

## Current state (2026-06-18 EOD)
**Working / shipped:**
- 308 (ungated) fires live + has a correct backtest lane (the loader silent-truncation that
  zeroed it was fixed + the lane rebuilt). It's the ungated live tester.
- High-TF gates supported live (#23); Fidelity Gate shipped (`Fidelity_Gate_Guide.md`);
  div-data OOM fixed (#28); `alerts.live_model` NULL = NON-ISSUE (fixed 2026-06-03).
- **#21** prep scoping (`1bbad56`) + **#29** chart scoping+trim (`13c4786`) live: ~2× prep,
  OOM relief, byte-identical. (Dropped: interp/trigger scoping = marginal; #29 decouple =
  empirically breaks parity. Residual: 5.7M-bar sub-minute+daily chart load deferred.)

**Root-caused TODAY — the 309–314 "missed/phantom" picture** (full detail:
`Mass_Builder_Forward_Test_Bugs.md`). It is NOT PB/CB gate timing. It decomposes into:
- **Display artifacts** (not trading bugs): stale per-load Fwd counts; Fwd count
  double-counts backtest+cache; cache lane is a backtest seed-copy; `forward_test_start`
  anchored to created_at not the OOS start (`in_sample_end`).
- **Bug 5 (real, FIXED + shipped `0e19838`, live-confirm pending):** live warmup used a flat
  `days=7` for every gate → a 1Day gate got ~5 bars → never warmed → gated strategies never
  entered live (310/313 = 0 live vs 33/103 backtest). Fix: TF-scaled resample-from-1min
  warmup. Offline-validated (313 daily 0/12 mismatch vs backtest + BULL_TREND on fire days;
  no regression 309/311). Backup: `dev-backup-2026-06-18-pre-shadow-warmup`.
- **310 RVOL gate (real, OPEN):** forming >60s bar gets ~2× volume (per-second + ws_agg
  fan-out both add) → live RVOL reads EXTREME → 5m-RVOL-HIGH gate never matches. Separate
  from Bug 5 (volume, not warmup). Price-based gates unaffected.

## Priorities

### P0 — get the gated real-money strategies executing live
1. **Bug 5 LIVE CONFIRMATION (tomorrow at open).** Verify the worker ran the fix
   (`railway logs --service Worker | grep '[Bug5]'` → 1Day gate now warms ~245 bars not 5)
   and that 313/312 produce live alerts during market. ALSO verify engine health: post-market
   tonight monitor_status showed `connected=False` + an odd `started_at` (Apr-06 vs 19:05
   earlier) — likely a rolling-deploy/stale-replica artifact; confirm the engine is cleanly
   connected at open. (Service is **"Worker"**, capital W — not "worker".)
2. **310 RVOL volume-double-count** (live-gate sub-bug) — suppress one volume source on the
   forming >60s bar so live RVOL matches backtest. The other reason a gated strategy is blocked.
3. **#16** — guard full-UAD so a transient fetch failure can't silently zero a lane.

### P1 — Stage-2 pipeline cleanups (so Mass Builder scales to thousands, clean by construction)
4. **Bug 1** — Fwd count = backtest-lane only (`data_source='backtest_%'`, post-divider). Per
   Kevin: forward-test = backtest model split at the divider, NOT cache, NOT combined.
5. **OOS anchor DECISION** — should `forward_test_start` = `in_sample_end` (the OOS start the
   trader expects) vs `created_at` (when live monitoring began)? Three timelines currently
   collapsed into one divider. Needs Kevin's call; changes what "forward test" displays.
6. **Bug 3** — one source of truth for `forward_test_start` (top-level col vs config=None).
7. **Bug 4** — stamp the cache lane's model (`cache_None` → `cache_<algo_model>`).
8. **Bug 2** — distinguish the backtest-seed forward equity from real live fills (viz).
   → bake 4–8 into MB save + the cold-seed path (relates to #25 / #26).

### P2 — known divergence draggers (Roadmap_Divergence_Hunting / Known_Bugs)
9. **B4 1Min early-fire (#5)** — the "1-minute bars and above" timing item (265/269).
10. 292 SR-channels pack (WIP), 275 dead-live (0 alerts/22 BT), 271 rare 5m-gate under-pass;
    streaming-tick backtest under-fires ~7%; ralph_engine watchdog for shadow engines.

### P3 — larger builds (discuss scope first) + cleanup
11. **#30** pin Fidelity Gate fixture window; **#4** Live-Mode ground-truth telemetry; **#31**
    Tier-3 post-deploy monitor; trade-engine throughput (the 76% recompute lever, risky).
12. **Docs cleanup phase 2** — move active docs into `_active/`, archive done specs.

## Doc map (current locations; move = phase 2)
**Active (read these):**
- `Roadmap_Divergence_Hunting.md` — divergence root causes + draggers (has some stale 6/11
  entries; freshness pass needed).
- `Known_Bugs.md` — ~18 active documented bugs (2 weeks old; needs triage).
- `First_Live_Money_Test_2026-06-17.md` — the 308–314 launch log.
- `Fidelity_Gate_Guide.md` (reader) + `Fidelity_Gate.md` (spec) — the regression gate.
- `Append_Edge_Fossilization.md` — UND/append mechanics.
- `Fleet_UAD_Report_2026-06-12.md` — historical UAD log (reference).

**Outdated / big-picture (don't drive daily work):** `Roadmap_To_Scale.md`, the PRD, older
Implementation_Spec_Phase_* docs (candidates for a `done/` or `archive/` folder in phase 2).

## Convention
- New observation that needs action → add to the right Priority bucket above (and to
  `Roadmap_Divergence_Hunting.md` if it's a divergence specifically).
- This doc is the entry point; keep it current at each session's end.
