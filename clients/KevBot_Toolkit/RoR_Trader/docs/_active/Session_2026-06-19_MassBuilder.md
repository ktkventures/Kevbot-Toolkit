# Session 2026-06-19 → 06-20 — Mass Builder hardening + nested results UI

Multi-day push on the Mass Builder. All work on `dev` (auto-deploys all Railway
services: api, Worker, data-worker, frontend). Juneteenth (Jun 19) was a market
holiday → deploys safe, no live trading.

## Shipped + live (commits on dev, newest last)
- `a55cf95` **1.15** — confluence-group **prep scoping** for Mass Builder. Root cause
  of multi-trigger/long-window OOM crash: prep computed ALL 15 confluence groups
  (851 cols) regardless of strategy → dough copied wide per-bar state → 2.3 GB on a
  7-day 15Sec run → OS OOM-kill (silent). Fix reuses the #21 scoping infra behind a
  new `force_scope` param so it's mass-builder-only (live/single-strategy untouched).
  Kill-switch `RORT_MASS_SCOPE_CONFLUENCE` (default ON). Validated byte-identical
  (scoped vs unscoped = identical results), peak RSS 1192→532 MB (−55%).
- `cdcb841` **1.16** — **incremental mid-flight persistence**. Results now flush to the
  DB `results` field as each combo completes (throttle `RORT_MASS_PARTIAL_FLUSH_SEC`,
  default 30s), so a long run is crash-visible + viewable live. Fixes the
  single-(symbol,tf)-group case where the old checkpoint never fired until the end.
  Independent of `checkpoint` (resume) → no duplication.
- `adcc96d` + `17d4801` **1.17** — **nested per-trigger-set drill-down** on Mass Results.
  Each search card gets a "▶ Results" expander → groups results by the full
  **trigger-set** (entry × exit × stop × target) with readable labels
  (`entry → exit · stop · target`); rows differ by confluence. Live-updating
  (`useMassResult` polls 3s while running). Inline preview KPIs + equity sparkline.
  Fail-loud red badge for `diagnostics.backtests_failed`. Big searches start collapsed.
- `9968af9` — **engine-accurate fetch-on-demand: PULLED.** Built a "Confirm
  engine-accurate KPIs" button (re-run via `/api/backtest/run`), but it returns 0
  trades even for ungated base results: the generic endpoint loads the visible window
  with **no indicator warmup** (cold EMA/ATR) and has no warmup-vs-visible trim. Can't
  faithfully reproduce a mass result → pulled rather than show a wrong number. The
  removed frontend block is preserved at `d7f5983` for re-use.
- `7fb84ae` **P0 cross-pack silent-drop FIX** (the big catch from Kevin's first real
  multi-pack run). HiFi multi-trigger builds one per-(symbol,tf) "dough" with a SUPERSET
  of all entry+exit bases stuffed into `_superset['exit_triggers']`. But
  `_resolve_trigger_ids` (unified_engine.py:262-270) reads **`exit_trigger_confluence_ids`
  FIRST** and only falls back to `exit_triggers` when empty — and the superset inherited
  the FIRST combo's exit CIDs. So the dough tracked only the first combo's entry+exit;
  every OTHER pack's trigger (rvol_v2, supertrend) was never cached → those combos failed
  "trigger not in cache", caught per-combo, **silently skipped**. Kevin's run: 54 base
  BTs, 36 failed, all 252 results were swing_123-only. Fix: clear
  `exit_trigger_confluence_ids`/`_id` on the superset. Validated: before 9 run/6 fail/1
  result → after 9 run/0 fail/9 results. Additive to dough → single-pack byte-identical.
  PRE-EXISTING (fails identically scope on/off) — NOT the #21 scoping.

Offline batch earlier in session: 1.6 verified single-source (no change), 1.7 re-scoped
(safe as-is, deprioritized), 1.9 deferred (entangled config json), 2.4 Fidelity Gate
re-verified 3/3 CLEAN on dev (308/309/313 byte-identical) — confirms Bug-5 work didn't
drift backtest prep.

## OPEN — resume here
- **P1b (next): the drill-in.** Kevin's approach (avoid rebuild): add a button on each
  trigger-set in the Mass Results nested panel that **deep-links to the existing Mass
  Builder edit view** (`/mass-builder?edit={searchId}`) **pre-filtered to that
  trigger-set**, AND add a trigger-set filter to that edit view. The edit view
  (`frontend/src/views/MassBuilderPage.tsx`) already has the rich per-result cards
  (equity, KPIs, OOS, Save Strategy) + metric filters (Sort by, Min WR/PF/Trades/R²,
  Trade Qual). Add: a trigger-set filter (entry/exit/stop) + accept a deep-link param to
  pre-apply it. Reuse, don't rebuild.
- **P2: deeper perf** if Mass Results still sluggish on big searches (252 results made the
  page crawl). Partly addressed (collapsed groups). Consider: trimming polling overlap
  (`useMassResults` list 3s + `useMassResult` 3s + `useMassProgress` 250ms), virtualizing.
- **Verify** the trigger-set regrouping renders on the deployed site (Playwright) once the
  frontend deploy lands.
- **1.18 follow-up:** warmup-aware rerun endpoint (e.g. `/api/mass-builder/results/{id}/
  rerun`) using `load_strategy_data` (warmup + `trim_trades_to_visible`) → faithful inline
  engine-accurate KPIs; then re-add the pulled button.
- **Monday live validation (tasks 1.1/1.2):** Bug-5 coarse gates (313/312, 1d/4h) fire live
  + no live↔backtest divergence; 1Min warms + fires. Rollback `RORT_TF_SCALED_WARMUP=0`.

## Key facts
- **Mass Builder modes:** Rapid = post-filter approximation (the "Preview KPIs" banner —
  optimistic, confirm with Update All Data). HighFidelity = real engine per confluence
  (cache-replay, true gating + warmup) → engine-accurate, ~matches UAD. Kevin uses HiFi.
- **Faithful engine-accurate path that EXISTS today:** save a mass result as a strategy →
  Update All Data → view KPIs (warmup-correct; UAD ~40 min per sub-minute strategy).
- **Env kill-switches:** `RORT_MASS_SCOPE_CONFLUENCE` (mass scoping, default 1),
  `RORT_MASS_PARTIAL_FLUSH_SEC` (default 30), `RORT_SCOPE_CONFLUENCE_GROUPS` (global #21,
  default 0 — leave off until validated globally), `RORT_TF_SCALED_WARMUP` (Bug-5, default 1).
- **Deployed dev:** frontend `https://frontend-dev-e01a.up.railway.app`, api
  `https://api-dev-2c9d.up.railway.app`. QA account `kevin-migrate@rortrader.dev` /
  `MigrateTest123!` (docs/QA_Agent_Briefing.md). Agent CAN drive the UI via Playwright
  (proven); session-token harvesting for direct API is correctly blocked.
- **Board:** in-app Dev Task Tracker at `/admin/tasks` (DB `dev_tasks`/`dev_task_comments`,
  admin client). Mass-builder tasks: 1.15-1.19, 1.18 (rerun), 2.10 (agent QA), 2.20 (agent API).
