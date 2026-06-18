# Roadmap — Divergence Hunting (Live ↔ Backtest Pair-Rate to 95%+)

**Last updated:** 2026-06-15 (Monday harvest verdict — gate fix VERIFIED live)
**Goal:** Drive fleet-wide live↔backtest pair rate to 95%+ across the canary cohort.
**Status (2026-06-15 — HARVEST DAY):** ✅ **GATE DIVERGENCE SOLVED for the price cohort.**
On clean liquid RTH tape (14:45–16:15Z, clipped inside BT coverage to exclude TBD-lag),
the **gated 10s cohort averages 91.2%** (was 30–73% last week) — sitting right on top of
the **ungated 94.8%**. Worst Friday offender 303 = 100% (62/0/0, zero phantoms). The
own-records pollution fix (711e72b) held under live multi-strategy SPY fire. Every
remaining dragger is a KNOWN class: B4 1Min early-fire (265/269), SR Channels WIP pack
(292), dead-live 275, rare 5m-gate under-pass (271). VWAP v2 + RVOL v2 (last week's
volume laggards) RECOVERED. Full tiered breakdown: `docs/Fleet_UAD_Report_2026-06-12.md`
(Monday section). Next: B4 fix → 275 dead-live → strategy-health dashboard TBD column.
**Status (2026-06-08 late):** Phase 1 (sid 174) executed. Surfaced the real "gating
off-and-on" root cause: **confluence gates fail OPEN when records are empty**, and
**2m cross-TF confluence records have not been produced in the live worker since ~06-03**,
so every 2m-gated strategy floods ungated live while 15m/1M gates work. Fix scheduled for
2026-06-09 during trading hours (so it can be validated live). See "Update 2026-06-08 (late)".
**Status (2026-06-08 EOD):** Battle Plan window completed; trigger-mode recovered to 70.2%
(vs Friday 47.4%); 9 strategies hit 100% in ≥1 hour today. Engine capability proven.
**Status (2026-06-05):** 96.7% achieved on sid 302 immediately post-UAD. Architectural
work delivered. Issue diagnosed as OPERATIONAL (BT-lane backfill), not engine divergence.

This doc is the single source of truth for what's done, what's open, and where the
related artifacts live. Update at each session's end.

---

**Note (2026-06-18):** the live priority list now lives in `docs/_active/STATUS.md` (read
that first). This doc remains the divergence-specific detail log; some entries below
(6/11–6/15) are historical — verify before acting.

## Update 2026-06-18 — first real-money strategies surfaced two divergences

Live-money set 308–314 (TSLA 15Sec) after overnight UAD/UND:

- **308 backtest path BROKEN (P0).** Update-All-Data persisted **0** trades; a direct
  `get_strategy_trades(308)` returns **1,896 trades but only spans Mar-19 → Apr-13** then
  stops (18 min run); yet the direct `prepare_data_with_indicators` fixture path got **2,593**
  over a recent 30-day window. So the windowed/UAD path truncates + fails to persist for 308
  (309–314, same entry/exit, got full Mar→Jun lanes). Effect: 308 = 0-paired / all-phantom on
  Strategy Health (no lane to pair). Real bug in the windowed path, not a transient re-UAD.

- **309/310/313 missed trades = the documented gate decision-timing divergence (H1).** Revises
  yesterday's "gates weren't open" theory: UND-extended backtests show entries live didn't
  fire → live evaluates intra-bar/at-grace while backtest is bar-close. Diagnose with the
  **Gate Parity tab**. Contributing classes (Known_Bugs): sub-minute drift-cascade, user-pack
  gates failing on secondary TFs, EH/AH session-filter inflating "missed."

- **`alerts.live_model` NULL on new strategies** → rest-verifier silently skips 308–314, so
  they may be unmeasured/mispaired (own phantom/TBD noise). See Known_Bugs.

## Update 2026-06-10 — Gate Parity tab polish + Per-Bar Parity Drift (v1 + v2)

Shipped to dev (`c0116ea`, pushed live; backup branch `dev-backup-2026-06-10-gate-parity`):
- **Polish:** fixed the 1s-pane drag "snap-back" (out-of-bounds re-fit now gated on a real
  data-content change, not every re-render); a "Hide labels" toggle that clears the wordy
  indicator-name labels overlaying candles (keeps numeric price labels); taller 1s panes (240px).
- **Per-Bar Parity Drift (v1)** — new module between Gate Replay and the analysis card. Joins the
  two enriched bar arrays (backtest REST `/chart-data` vs live cache `/chart-data-cache?latest`)
  by timestamp and shows one ribbon per metric (bar-exists / OHLC / each indicator / each gate
  state): green match · yellow minor · red major · gray n-a, with per-row match%. Frontend-only.
  On sid 303 it instantly surfaced **gate state 71.6%**, **bar-exists 89.3%** (32 bt-only),
  **Low 99.6%** — the drift we're hunting, quantified per bar.
- **Per-Bar Parity Drift v2** — sibling module with a **`first` (decision-time WS) vs `latest`
  (REST-corrected)** toggle on the live source (`live_bars.first_close` vs `close`, both already
  persisted — no backend work). v1 left untouched as the `latest` reference.
- **Replay lens:** added a tip-source `first`/`latest` toggle (tip-only; splice stays faithful).

**Faithfulness note (verified):** both lenses run the IDENTICAL `prepare_data_with_indicators`
pipeline; only the input bars differ (REST vs WS-cache). So prices/indicators/bar-exists are true
parity. Caveats: the chart uses the BATCH pipeline while the live engine (`ralph_engine`) computes
INCREMENTALLY — a bug living only in the incremental engine won't show on the chart; and the
cross-TF gate `[PB]` is reconstructed (deterministic shift) vs ralph's emergent real-time PB.
Authoritative gate truth remains the `alerts` table + the Gate Parity Analysis card (real engine).

### FUTURE (discuss before building) — Live-Mode Ground-Truth Embedding
Goal: a real-time "Live mode" on the Gate Parity tab that embeds the ACTUAL backtest and live
engines with **no chart-side reconstruction**, so it's fully reliable.
- **Why it's currently reconstructed, not embedded:** for any PAST moment the live engine's
  real-time in-memory state (indicator values, gate decision, fidelity) was never persisted —
  there's nothing to read back, so the chart re-derives it via the batch pipeline over the bars
  the engine saw. `live_bars` stores OHLCV INPUTS (incl. `first_*` decision-time + corrected
  `close`) — confirmed — but NOT the engine's per-bar COMPUTED state. `save_engine_state_db` is
  positions/runtime recovery, not per-bar snapshots.
- **What it needs:** a per-bar state-emission / telemetry layer on `ralph_engine` (the long-flagged
  "explicit fidelity field both engines honor") that writes, each bar, what the live engine actually
  computed + decided. Then: Live mode reads that ground truth directly; and any replay of periods
  recorded after that point can use logged truth instead of reconstruction. Backtest side: read the
  actual `unified_engine` per-bar state (instrument it to emit a state log) rather than the batch
  recompute. Not impossible — just unbuilt plumbing. **Kevin wants to discuss scope/design first.**

The 06-08 "fail-open / 2m records missing" framing was **wrong** (corrected with live data):
the 2m secondary builder IS alive; records were FROZEN by a rebroadcast-cascade bug (fixed,
`recompute_confluence`, commit 93aebb4). But unfreezing didn't stop the gate-cohort flood, so a
deeper dig (cross-validated via the crossed trigger/gate experiment + the correct `trades`-table
lane) re-framed the **gate-mode break (H1)** as a **live↔backtest gate-state / decision-timing
divergence — NOT fail-open**. The trigger pairs fine alone (65–70%); adding a 2m gate collapses
it. Leading mechanism: live evaluates intra-bar/forming at grace while backtest is bar-close, and
the cross-TF (2m) gate state the live engine sees diverges from backtest's PB-shifted state.

**Built the diagnostic tool to nail this systematically — the "Gate Parity" tab** (strategy
detail page), a thin renderer over engine-truth: dual-lens **Backtest (rest_hifi · bar-close)**
vs **Alert (ws_rest_spliced)** replay, time-aligned, with 3 layers per lens (gate ribbon ·
primary candles · 1s candles), a Replay/Live(soon) toggle, an engine-truth analysis card
(PB/CB ribbons + theoretical-BT vs live entries + phantom classification), and a bar-set
divergence note (REST `dropna` drops empty bars; live cache keeps them — itself a divergence).
Backend: `src/gate_parity_harness.py` (`build_gate_parity_view`, `splice_alert_lens`) + routes
`/gate-parity` and `/window-1s`. Frontend: `LabReplayPanel` extensions, `spliceAlertLens.ts`,
`GateParityOneSec.tsx`, `useGateParity`/`useWindow1s`. All on dev, **squish fixed + verified
live via Playwright** (see memory `project_gate_parity_diagnosis` + `reference_playwright_dev_access`).

**Plan:** 2026-06-10 first thing during RTH — perfect the tool (visual polish + cross-lens
time-lock, with dense market-hours data + live verification), THEN use it to drive the
remaining autonomous divergence fixes (gate-timing alignment, Phase 4 pack cleanup, fail-closed
guard defense). Plan file: `~/.claude/plans/partitioned-sniffing-pizza.md`.

---

## Update 2026-06-08 (late) — Phase 1 dig: gate fail-open + 2m cross-TF live-record gap

Phase 1 (fix Mass Builder legacy sids 174/194) was executed as the first autonomous test.
The roadmap's **H3 hypothesis was wrong** — it is NOT a config schema mismatch, and the
`_backfill_strategy_configs.py` path would have changed nothing. The dig instead found the
mechanism behind the gate-mode break (H1) and the "gating works off and on" symptom.

### What sid 174 actually was

- 174 (TSLA LONG 1Min, gate `2m-SWING_123-BULL_C2` + `2m-SR_CHANNELS-BELOW_SUPPORT`) was
  built 2026-05-15 on the **custom** pack `swing_123_test`.
- Commit `32092e5` (2026-05-21) deleted `swing_123_test` from git, wrongly claiming "0
  strategy users." 174 (+194) were never migrated — orphaned.
- The standard replacement group `swing_123_default` (base_template `swing_123`) was left
  **`enabled: false`**. Both `services.py:291` (backtest) and `ralph_engine.py:4807` (live)
  run indicators only for `get_enabled_groups()`, so `calculate_swing_123` never ran →
  the SWING_123 interpreter defaulted every bar to NEUTRAL → `BULL_C2` was unsatisfiable →
  **0 backtest trades** (a from-scratch recompute confirmed 0). The indicator itself is fine
  (direct run on TSLA 2m: BULL_C2 ×319/2535).
- **Fix applied (DB only, reversible; backup `/tmp/cg_backup_19d47e46.json`):** enabled
  `swing_123_default`, removed the dead `swing_123_test_default` group. Validated: UAD on 174
  went **0 → 159 trades** (win 48.1%, daily_r 0.82).

### THE BUG — gates fail OPEN when confluence records are empty

The gate guard in `unified_engine.check_entry` (`unified_engine.py:2311` C-type; `:2729`
L-type) — **one implementation, used by BOTH backtest and live** (the live worker imports
`run_unified_backtest` from `unified_engine`):

```python
if self.confluence_set and confluence_records:   # ← short-circuits when records empty
    if not self.confluence_set.issubset(confluence_records):
        return None
```

When a bar has **zero** confluence records, `... and confluence_records` is False, the
subset check is skipped, and the entry fires **as if ungated**. A gate with missing records
does not block — it disappears.

Why it manifests asymmetrically (live vs backtest):
- **Backtest** always has records — `prepare_data_with_indicators` runs the interpreter even
  when the indicator didn't, emitting a default state (e.g. `2m-SWING_123-NEUTRAL`). Records
  present → gate evaluates → blocks correctly.
- **Live** for **2m** secondary-TF gates produces **no records** → guard short-circuits →
  ungated flood. (15m/1M cross-TF records DO flow live.)

### Empirical proof — gate cohort splits by secondary timeframe

| sid | gate | gate TF | BT/day | live entry/day | verdict |
|---|---|---|---|---|---|
| 136 | MACD H/L v2 | **15m / 1M** | ~9 | ~6 | ✅ gate works live |
| 277 | Bollinger | 2m | ~14 | ~260 | ❌ fail-open (~18×) |
| 293 | SR Channels | 2m | ~29 | ~296 | ❌ fail-open |
| 299 | SuperTrend | 2m | ~35 | ~350 | ❌ fail-open (~10×) |
| 303 | UT Bot v4 | 2m | ~23 | ~270 | ❌ fail-open |

Daily entry-alert inflection on the 2m gates: 06-02 ~17/day → **06-03 ~180-230/day** and
never recovered (per-hour rate jumped ~3.5× allowing for the 06-02 partial first day, created
17:35 UTC mid-RTH). **136 (15m/1M) gates correctly live; every 2m gate floods.** This is the
"gating off and on" Kevin remembered — it is **live-side and TF-specific**, since ~06-03.

NOTE: this 2m fail-open is **separate from** the 174 swing issue — 277/293/299 use *enabled*
packs and still fail open. So it is NOT a disabled-group problem; it is a **2m cross-TF live
record-production gap** in the worker, cohort-wide.

### Three distinct issues + fix plan (ORDER MATTERS)

1. **174 BT** — disabled swing group → ✅ FIXED via migration (this session).
2. **2m cross-TF live records not produced** (cohort-wide, since ~06-03) — the real "gating
   broke" cause. **PRIMARY FIX.** Worker-side; needs market-hours instrumentation to see why
   the 2m secondary-TF records aren't entering the live confluence buffer while 15m/1M are.
   Suspects: cross-TF live regression (cf. `feedback_polygon_xtf_live_regression`, fixed
   `ca57cac` for a similar gap), 2m TF not subscribed/aggregated in the worker, or required-TF
   resolution dropping 2m.
3. **Fail-open guard** (`unified_engine:2311`, `:2729`) — defense-in-depth. Change to fail
   **closed**: `if self.confluence_set: if not self.confluence_set.issubset(confluence_records
   or set()): return None`. **Do this ONLY AFTER #2** — applied first it silences all 2m gates
   live (0 alerts) instead of flooding.

### Plan for 2026-06-09 (during trading hours, to validate live)

1. Pre-market / early: map the live 2m cross-TF record path in `data_worker_engine.py` /
   `ralph_engine.py` (`_mtf_confluence` buffer, secondary-TF builder fan-out). Identify why
   2m records aren't produced.
2. During RTH: instrument one 2m gate (e.g. sid 277) — log `confluence_records` per entry in
   the live worker; confirm they're empty for 2m and populated for a 15m/1M gate.
3. Fix #2 (produce 2m records), deploy to dev, watch live: 2m gate entry rate should drop to
   match BT (~10-35/day, not ~260).
4. Then ship #3 (fail-closed guard) as the safety net.
5. **Post-deploy cleanup (confirmed by Kevin 2026-06-08):**
   - **Delete sid 174's alerts** AFTER the 2m fix deploys (so the clean slate isn't
     re-polluted by ungated firing). Its BT is already correct (159 trades).
   - **sids 267 & 301:** UAD their backtest to reflect the now-real `SWING_123-NEUTRAL` gate,
     then decide alert/baseline reset — handle together with the live fix for consistency.

### Data caveat from the migration (alerts/BT now on a different config basis)

The migration (enable `swing_123_default`) affected the 3 SWING_123 referencers: **174, 267,
301**.
- **174** alerts (401) are MISLEADING — generated ungated (live fail-open) on the broken-gate
  config. Post-fix pairing is 5.0% combined / **94.8% phantom** (380/401 unpaired). Recommend
  **deleting 174's alerts**; its BT is freshly correct (159 trades). (Live will keep firing
  ungated until fix #2 ships, so cleanest after the deploy.)
- **267, 301** gate on `SWING_123-NEUTRAL`, which used to be always-true (every bar defaulted
  NEUTRAL) → they were silently ungated. Their stored BT is now stale (overcounted); UAD needed
  to reflect the real gate (~66% NEUTRAL). Their prior "100%" and the cohort "Swing gate works
  77.9%" were ARTIFACTS of the broken gate. Canary-baseline decision pending Kevin.

---

## Update 2026-06-08 EOD — Battle Plan Results + Pack-Quality Path Forward

### Battle Plan verdict: recovery confirmed, no revert

Today's RTH (13:30-20:00 UTC) was the controlled measurement window — no deploys,
MANUAL streaming, clean observation. Cohort numbers vs Friday baseline:

| Mode | n | Friday combined % | Today combined % | Δ |
|---|---|---|---|---|
| Trigger | 26 | 47.4% | **70.2%** | +22.8 pts |
| Gate | 15 | 15.2% | 20.3% | +5.1 pts |
| Cohort total | 46 | ~35% | 48.8% | +13.8 pts |

**Both modes improved.** Friday's degradation was deploy churn, not engine drift —
hypothesis confirmed.

**9 strategies hit 100% combined % in ≥1 hour today** with healthy alert volume:

| sid | name | # 100% hours | best |
|---|---|---|---|
| 294 | PACKTEST · Stochastic Oscillator · trigger | 3 | 100% (31 alerts) |
| 263, 267, 273 | TSLA UT Bot variants | 2 each | 100% (52 alerts) |
| 282 | PACKTEST · EMA Stack v2 · trigger | 3 | 100% (20 alerts) |
| 269 | SPY-CANARY-1m-Control | 2 | 100% |
| 276, 298, 136 | Bollinger / SuperTrend / Mass mirror | 1 each | 100% |

**The engine IS capable of 100% pairing.** Multiple packs proved it across 4 different
creation paths (PACKTEST, CANARY, TEST-P2, mass-builder migration). Variance is in
*which* strategies and *which* hours, not in the engine's fundamental capability.

**Bail-out criteria NOT met** (per the Battle Plan early-exit table):
- Asymmetry is phantom-heavy (live > BT), not the BT>>live direction the table linked
  to #57H revert
- Pattern is recovery, not degradation

**Verdict recorded: do not revert. Battle Plan window closes.**

### Pack-quality pattern (the new discovery)

Cross-referenced today's combined % with `git log` on each `user_packs/` directory.
Clean correlation:

| Pack (trigger) | Today % | Last meaningful commit | Status |
|---|---|---|---|
| Stochastic | **93.6%** | 2026-05-07 phase2 markers | Clean migration, no hotfixes |
| SuperTrend | 81.2% | 2026-04-27 FLIP fix | Clean |
| Bollinger Bands | 79.4% | 2026-04-27 FLIP fix | Clean |
| Swing 1-2-3 (both!) | 77.1% / 77.9% | 2026-04-27 | Clean (caveat: gate is permissive) |
| EMA Stack v2 | 77.5% | 2026-05-07 | Clean |
| UT Bot V4 (sid 302) | 73.7% | 2026-06-04 #57 manifest contract | Recently touched |
| EMA PP v3 / v4 | 73.7% / 71.2% | 2026-06-03 phase2 migration | Recently touched |
| RelVol v2 | 71.6% | 2026-04-27 | Clean but mid-tier |
| RSI Zones 2 | 70.9% | 2026-05-07 | Clean |
| Strat Assistant | 69.8% | 2026-04-27 | Clean |
| MACD Histogram v2 | 68.9% | 2026-05-07 | Clean but mid-tier |
| MACD Line v2 | 67.7% | 2026-05-07 | Clean but mid-tier |
| **VWAP v2** | **43.1%** 🛑 | 2026-05-19 (TWO hotfixes) | "live exception storm" + "isinstance bug" |
| **SR Channels** | **20.4%** 🛑 | 2026-04-26 | Migration commit says **"97% parity"** — never finished |

**Working theory (Kevin's, validated by data):** packs that migrated cleanly to the
modular pattern (single migration commit, optionally a follow-up FLIP/phase2 fix)
perform well. Packs that needed live-side hotfixes (VWAP) or were marked incomplete
at migration time (SR Channels) underperform — the migration WAS the validation,
and incomplete migrations leave bugs in the field.

**Swing 1-2-3 gate caveat:** the only working gate (77.9%) is likely working because
Swing's gate logic is permissive most of the time (candles rarely in a Swing setup).
Do NOT use Swing's gate as a template for fixing other gate packs — its 77.9% reflects
"gate barely runs," not "gate logic is well-implemented."

### Root cause hypothesis hierarchy

| # | Hypothesis | Bug likely lives in | Test bucket | Confidence |
|---|---|---|---|---|
| H1 | Gate-mode systemic break — live emits alerts BT doesn't reproduce on 14/15 gates | Mostly BT model; possibly live emitting transient signals | A + C | **HIGH** — phantom-heavy across cohort |
| H2 | Pack-specific bugs in VWAP v2 + SR Channels triggers | Pack code (BT + live both use it) | A | **HIGH** — git history shows incomplete migrations |
| H3 | Mass Builder legacy config schema mismatch (sids 174, 194 at 1.5%) | BT model + legacy config fields | A | **HIGH** — clear lookup-failure shape |
| H4 | Pack quality variance — why Stochastic 93% vs MACD Line 67% same batch | Pack code (interpreter cleanliness, state-transition shape) | A or C | Medium — needs pack-by-pack analysis |
| H5 | Live engine over-firing on partial bars (transient ticks → alerts) | Live model | B | Medium — would explain phantom direction |
| H6 | Afternoon engine drift (14-16 → 17-19 combined % gap) | Live engine state OR market behavior | B | Low-Medium — could be symptom not cause |
| H7 | Hardware / Railway limits | Plumbing | B (instrumentation) | Low — 4000s UAD reflects engine compute, not infra |
| H8 | Update New Data lag mechanics | UAD process | A | Already understood; addressed by 6/05 + today |

### Validation buckets (testing methodology)

**Bucket A — Update All Data-testable (fast, ~5-10 min per strategy):**

For changes that affect BT trade generation. Workflow: change code → click UAD on
canary → re-run hourly analysis script → compare before/after combined %.

Covers: BT engine code, pack code, manifest fixes, interpreter changes, config schema
fixes, anything in `src/` that touches BT lane.

**Bucket B — Live-observation-required (slow, hours-to-day):**

For changes that affect LIVE alert generation only. Workflow: deploy → wait for fresh
alerts under new code → UAD BT lane → measure post-deploy window via By Deploy view.

Covers: `worker.py` / `ralph_engine.py` changes, WebSocket handling, stream catchup,
snapshot resume logic, anything that only runs in the live worker.

**Bucket C — Faster alternatives (need to build harness first):**

1. **Bar-replay harness (HIGHEST LEVERAGE):** record a representative RTH bar stream
   (e.g., Monday 06-08 RTH), then run both engines through it offline. Deterministic,
   ~minutes per iteration. Once built, converts Bucket B problems into Bucket A speed.
2. **Pack Validation Sweep:** fixed golden window (e.g., Thursday 06-04 15:00-19:00
   on SPY) → run every pack's trigger + gate through it → compare to stored reference.
   Pre-deploy regression gate.
3. **Cross-pack structural diff:** compare working pack code (Stochastic, Swing) vs
   broken (VWAP, SR Channels) line-by-line in interpreter chain. Pure code review,
   no execution.

### Phase plan — Path to 95%+ across canary cohort

Compressed timeline. Goal: work through these in **days, not weeks**.

**Phase 1 (1-2 days) — Quick wins, Bucket A:**
- ✅ **DONE (2026-06-08)** — sid 174 fixed (0→159 trades). Root cause was NOT a config
  schema mismatch (H3 wrong); it was an orphaned custom pack + disabled replacement group.
  See "Update 2026-06-08 (late)". sid 194 = separate live-sparsity issue, DEFERRED.
- 🟡 **IN PROGRESS** — Stochastic structural lead. **Preliminary (2026-06-08, full-window
  ±5s pairing — NOISY, includes broken/deploy-churn periods; needs stability-clipped hourly
  view to confirm):**
  - **Trigger frequency inversely tracks pairing.** Sparse/compound discrete-event triggers
    pair best — Stochastic `k_cross_above_d_oversold` (crossover AND oversold zone → 1742
    trades, 47.8%), SuperTrend `bull_flip` (1366, 48.1%), Bollinger `cross_upper` (983, 48.4%).
    High-frequency single-condition triggers pair worse — MACD Hist `momentum_shift_up`
    (7123, 40.9%), StratAssistant `bull_c2` (7837, 39.8%), RVOL `spike` (6595, 43.0%). More
    events = more timing-jitter mismatch opportunities.
  - **VWAP v2 is separately broken** — 26.8%, and BT(416) < alerts(513) = phantom-heavy → a
    real pack bug, not just sparsity. Feeds Phase 4 (VWAP re-audit).
  - **Hypothesis (golden child):** the best-pairing packs emit *infrequent, unambiguous,
    bar-close-deterministic* events. To push SuperTrend/Bollinger past 90%, the lever is
    likely timing determinism, not the trigger logic. CONFIRM via stability-clipped hourly
    next, then apply to weaker packs (Bucket B, tomorrow).
  - NOTE: manifest `type` field is "BOTH" for all (entry/exit flag, not a structural class) —
    structural distinction comes from firing semantics + frequency, not metadata.
- ⏳ **OPEN (needs live validation = tomorrow)** — Apply Stochastic's pattern to SuperTrend
  (81.2% → past 90%) and Bollinger (79.4% → past 90%). UAD changes the BT lane, but
  confirming the combined-% lift needs fresh live pairing → Bucket B.

**Phase 2 (2-3 days) — Bar-replay harness, Bucket C-1:**
- 🟡 **MUCH OF THIS ALREADY EXISTS** (reconnaissance 2026-06-08). `src/parity_simulator.py`
  already replays batch (BT) vs live paths and compares firing (`_replay_engine_bars`,
  `_run_live_replay_path`). It is **V1**: single pack, single TF, **no confluence gating, no
  cross-TF, no position-state composition**. The "V2 deferred" list in its docstring
  (cross-TF, strategy-detail mode = multiple packs + PSM, mid-stream restart) is EXACTLY
  what Phase 2/3 need. So Phase 2 = **extend parity_simulator to V2**, not build from scratch.
- Also available: drill/probe scripts (`_probe_full_replay.py`, `_drill_parity_full.py`,
  `_compare_cache_replay.py`) and frontend replay UI (`ChartReplayCard`, `ReplayableChart`,
  `ScenarioReplayCard`, `TradeReplayModal`, `useScenarioReplay`, on StrategyDetailPage).
- Remaining: V2 cross-TF + gating support; optionally a recorded RTH fixture for determinism.

**Phase 3 (Gate-mode systemic break, H1) — DIAGNOSIS DONE 2026-06-08, ahead of the harness:**
- The planned harness step ("find the asymmetric step — likely confluence-record
  interpretation between live tick handling and BT bar-close") is **already found** without
  the harness: it's the **fail-open gate guard** + **missing 2m cross-TF live records**.
  Full writeup in "Update 2026-06-08 (late)".
- What remains is the FIX, scheduled 2026-06-09 (live hours): produce 2m live records →
  deploy → validate sid 277 entry rate drops to BT rate → then fail-closed guard.
- The V2 parity_simulator (Phase 2) is still worth building as the regression gate so this
  class of bug can't silently return.

**Phase 4 (1-2 days, parallelizable with Phase 3) — Pack-specific cleanup, H2:**
- VWAP v2: re-audit the 2026-05-19 hotfix code, look for residual issues
- SR Channels: complete the "97% parity" migration
- Both validated via UAD

**Phase 5 (1 day, can defer) — Pack Validation Sweep, Bucket C-2:**
- Define the golden window
- Wire up sweep tool — runs all packs through golden window, compares to stored reference
- Use as pre-deploy gate

**Estimated total:** 5-8 days of focused work to get cohort to 90%+ on most strategies.
Faster if Phases 3 and 4 can run in parallel (Kevin reviewing one while Claude works
on the other).

### Autonomous operations (RoR mode) — to discuss separately

Kevin's request: define an autonomous protocol where Claude can iterate through these
phases independently. Make changes → test via UAD → analyze results → declare success
or pivot → move to next hypothesis.

**To discuss in a follow-up session:**
- Which actions Claude is authorized to execute autonomously (e.g., feature-branch
  commits, UAD runs on non-canary strategies, doc updates)
- Which require human-in-loop (push to dev during RTH, restart worker, modify pack
  manifests, revert prior commits)
- Standard "did this help" determination criteria — what numeric threshold + cohort
  size constitutes "shipped fix" vs "needs more validation"
- Escalation triggers — when to stop autonomous iteration and request review

This will be its own document/SOP once defined. For now: status quo — Claude works
on Kevin's explicit requests, surfaces analysis and proposed fixes for approval before
touching code on dev branch.

---

## TL;DR — The Big Discovery (2026-06-05)

After a full Update All Data on sid 302 (UT Bot V4 canary), pair rate jumped
from ~32% to **96.7% (321 paired / 332 events)** across the entire trading day.

| hour | before recompute | after recompute |
|---|---|---|
| 13:30-14:30 | 34.7% | **98.0%** |
| 14:30-15:30 | 26.9% | **100.0%** |
| 15:30-16:30 | 39.5% | **100.0%** |
| 16:30-17:30 | 96.7% | **100.0%** |
| 17:30-18:30 | 32.6% | **100.0%** (gap filled!) |
| 18:30-19:30 | 58.3% | **100.0%** |
| 19:30-20:30 | 0.0% → 72.2% (still streaming) | **96.3%** |

**The "low pair rate" we measured all day was a MEASUREMENT ARTIFACT.** When the
backtest data is fresh, live and backtest agree on essentially every trade. The
architecture is trustworthy.

**Why the measurement was off:** BT-lane writes happen via Data Worker's streaming
pipeline. Each Data Worker restart (from a deploy) causes a 5-10 min warmup window
during which the engine processes bars but rejects entries because stateful stops
(swing/ATR) need ~30 bars to populate their high/low buffers. During warmup: bars
processed, no trades fire. Those bars get NO BT trade record. Today's 5 deploys
in 52 minutes (17:31-18:23 UTC) created cumulative BT-lane gaps that made the
pair-rate metric look ~50% when it was actually 100% — the trades just weren't
in the database to pair against.

---

## Today's deliverables (2026-06-05)

### Architectural fixes shipped (all in production)

| commit | what | impact |
|---|---|---|
| `c1cfb4b` | #57C engine write hooks for bar_diagnostics | Per-bar indicator state capture (live/BT/algo) |
| `8f745a5` | `_probe_bar_diagnostics.py` CLI tool | Side-by-side live↔BT divergence inspection |
| `5d634ae` | #57E+F per-bar REST verification (4-pass settle) | Layer 1 drift_uncorrected 0.7% → 0.1% (**7× reduction**) |
| `b5f40c7` + `3c5861c` | #57G `live_corrected` source rows | Post-splice state captured for analysis |
| `0901baf` + `c256900` | #57H phantom_pre_correction reclassification | Structural WS noise labeled out of needs_investigation |
| `3429dce` | #42 session filter on injected primary_df (services.py:264) | Sid 268-class bug fixed (RTH/Extended Hours respected on cache_locked path) |

### Migrations applied (you ran these via Supabase SQL editor)

- `bar_diagnostics_table.sql` — per-bar indicator state ✓
- `bar_diagnostics_live_corrected.sql` — `live_corrected` source value ✓
- `alerts_would_fire_post_correction.sql` — phantom reclass flag ✓

### Validation results

- **Layer 1 — Bar fidelity:** GREEN (drift_uncorrected 0.1%, verified 87%, corrected 11%)
- **Layer 2 — Pair rate on sid 302 with current BT data:** **96.7%** (321/332)
- **Per-bar verify firing:** confirmed via worker logs (multiple SPY/TSLA corrections per minute)
- **live_corrected rows landing:** 86+ rows accumulating since deploy
- **phantom_pre_correction reclass:** working (2 alerts flagged in first window post-migration)

---

## Open work (in priority order)

### P0 — Smart gap-detection cron (today/tomorrow's work)

**State:** Designed but not built. This is the operational missing piece to make
the dashboard pair-rate metric trustworthy without manual intervention.

**Why we need it:** BT-lane gaps from deploy churn or any transient Data Worker
restart create false "phantom" alerts in the dashboard. Manual Update All Data
backfills them but doesn't scale to tens of thousands of strategies.

**Design:**
- Detection: each strategy gets a 2h lookback; bucket alerts + BT trades into 5-min
  windows; flag bucket as gap if `alerts ≥ 2 AND bt_trades < alerts/3`
- Action: queue targeted `update_jobs` over `[gap_start - 30min warmup, gap_end + 5min buffer]`
- Schedule: hourly cron; rate-limit (skip strategy if recomputed in last 30 min)

**Cost at scale:**
- Today (46 strategies): ~25s detection + ~5min recomputes if gaps exist
- 10K strategies (your future): detection parallelizable to 5-10 min; recomputes
  via existing `update_jobs` queue with multiple workers

**Verification approach (Kevin's idea, 2026-06-05):**
Run detector on one strategy → backfill identified gaps → then click manual
Update All Data → compare results. If detector's targeted backfill ≈ full
recompute, validated.

### P1 — Algo-lane cache_* trades dead (task #60)

**State:** Last `cache_cache_locked` trade written 2026-06-04T21:38 UTC. 23h silent.
Independent of today's architectural work.

**Investigation needed:** Why algo-history cron writes nothing (despite running —
last_recompute_until_ts stamps are fresh). Possible regression from yesterday's
`3322298` algo-lane incremental fix.

**Impact:** Doesn't affect live↔BT pair rate (we filter to `backtest_%`), but it's
a latent bug. Algo lane was intended to mirror "what live actually did" — without
it we lose that secondary check.

### P2 — Operational: MANUAL streaming during active dev

**Pattern (your insight, 2026-06-05):**
- Active dev session → flip streaming to MANUAL via /admin/update-jobs
- Deploy freely without causing BT-lane gaps from warmups
- After deploys settle → flip to AUTO + run Update All Data on touched strategies
- Avoids today's gap pattern entirely

No code change needed — toggle exists from yesterday.

### P3 — Investigation cleanups

- #44 swing-stop cluster: REFRAMED — only sid 274 is genuinely silent across 5 days
  (other "silent" strategies fire normally per-week, just market-conditional gates).
  Investigate sid 274 specifically when there's time.
- #55 split BT/Algo UAD buttons — deferred until scale strain
- #56 cross_exec_type_mismatch root cause investigation — classifier ships; underlying
  state-drift mechanism understood (same as #53 in different indicator)

---

## Linked artifacts

| Doc / Tool | Purpose |
|---|---|
| `docs/SOP_Strategy_Health_Check.md` | 4-layer methodology + active classifier buckets |
| `docs/Known_Bugs.md` | Active bug log |
| `src/_remeasure_pair_rates_5s.py` | Macro pair-rate measurement |
| `src/_divergence_walkthrough.py` | Layer 1 + 2B walkthrough |
| `src/_fill_delta_analysis.py` | Layer 3 fill_ts delta |
| `src/_probe_bar_diagnostics.py` | Live vs BT vs algo state at any bar |
| `src/_verify_streaming_toggle.py` | Streaming-toggle observability |
| Backup branches | `dev-backup-pre-streaming-toggle-test`, `dev-backup-2026-06-03-eod` |

---

## Measurement methodology (post 2026-06-05 update)

**Critical:** Before measuring pair rate, ensure BT data is current. If you measure
against stale BT data, "phantoms" will be measurement artifacts, not engine divergence.

**Correct workflow:**
1. Click Update All Data on the strategies you're measuring (or run gap-detection cron once it exists)
2. Wait for completion (~3-5 min per strategy)
3. THEN run pair-rate analysis
4. Result is the true engine divergence

**Incorrect (what we did most of today):**
- Measure pair rate against live-streaming BT data
- See low numbers
- Mistake operational artifacts for engine bugs
- Spend hours on "fixes" that aren't actually needed

This methodology update should be added to the SOP.

---

## Run-order for next session

**Updated 2026-06-09 EOD — supersedes earlier run-orders.** See "Update 2026-06-09 EOD" above.

### TOMORROW 2026-06-10 — first thing during RTH (Kevin's plan)
1. **Perfect the Gate Parity tool** (with dense market-hours data + live Playwright verify):
   the precise **cross-lens time-lock** (both lenses identical start/end), main-lens clean
   full-fit (left-crop), and any visual polish Kevin flags. Tool is WORKING on dev; squish
   fixed. Plan file `~/.claude/plans/partitioned-sniffing-pizza.md`; state in memory
   `project_gate_parity_diagnosis`.
2. **Use the tool to localize the gate-timing divergence** (theoretical-BT vs alert, PB/CB
   ribbons, phantom classification) on the gate cohort (277/293/299/303 etc.) and fix the
   live↔BT gate alignment.
3. **Then autonomous fixes** per SOP where possible (Phase 4 pack cleanup: VWAP v2 / SR
   Channels; fail-closed guard as defense-in-depth AFTER the timing fix).
4. **Pending cleanup (carried):** delete sid 174 alerts; UAD 267 & 301 + baseline reset.

### Earlier run-order (2026-06-08 late) — carried, lower priority now

### TOMORROW 2026-06-09 — DURING TRADING HOURS (the live-validatable work)

1. **[PRIMARY] Fix the 2m cross-TF live-record gap** (Phase 3 fix / H1). Pre-market: map the
   live 2m secondary-TF record path (`data_worker_engine.py` / `ralph_engine.py` `_mtf_confluence`
   buffer + secondary-TF builder fan-out). RTH: instrument sid 277 — confirm `confluence_records`
   empty for 2m, populated for a 15m/1M gate. Fix → deploy to dev → watch 2m gate entry rate
   drop from ~260/day to ~BT rate. **Likely needs plan mode** (engine/worker change).
2. **Then ship the fail-closed guard** (`unified_engine:2311`, `:2729`) as the safety net —
   ONLY after #1, else 2m gates go silent.
3. **Post-deploy cleanup:** delete sid 174's alerts (ungated/old-config, 94.8% phantom);
   UAD sids 267 & 301 to reflect the now-real `SWING_123-NEUTRAL` gate + decide baseline reset.
4. **Confirm + apply the Stochastic golden-child pattern** to SuperTrend/Bollinger (Bucket B —
   live pairing validates the lift). See preliminary theory in the Phase 1 block above.

### After the live fix (Bucket A / anytime)

5. **Phase 2 — extend `parity_simulator.py` to V2** (cross-TF + gating + position-state
   composition). NOT a from-scratch build. Becomes the pre-deploy regression gate so the
   fail-open class can't silently return. Reuse existing frontend replay UI where useful.
6. **Phase 4 — pack cleanup:** VWAP v2 (26.8%, phantom-heavy — confirmed laggard tonight) +
   SR Channels migration completion.
7. **Phase 5 — Pack Validation Sweep** (golden-window pre-deploy gate).
8. **sid 274** (P3 carry-over) — genuinely silent live (BT 268 / 0 alerts) — investigate.
9. **Define autonomous operations protocol** — separate discussion; own SOP doc.

### Run-order from 2026-06-05 (carried over, lower priority)

1. **Build smart gap-detection cron** — code the detector + action + cron wiring (~2 hrs)
2. **Verify on one strategy** — run detector → backfill → compare to manual Update All Data
3. **Deploy as hourly cron** — once verified accurate
4. **Re-measure pair rate fleet-wide** — should be 95%+ across all canaries with gap-detection running
5. **Investigate #60 dead algo lane** — separate latent bug, ~30 min
6. **Update SOP** with measurement methodology + new operational patterns

## 2026-06-11 EOD — post-gap-healer additions

### NEW TOOL (Kevin-approved, build ~2026-06-12): Backtest-reference toggle on Gate Parity

Add a selector on the Gate Parity tab choosing what the "backtest" side of the
Per-Bar Parity Drift ribbon / Gate Replay lens / Analysis card compares against:

- **"Fresh REST replay" (current behavior):** recompute on demand via
  `prepare_data_with_indicators` + `unified_trades` — answers "what SHOULD the
  backtest say with current code + data".
- **"Stored backtest lane" (new):** the `trades` table rows the UAD/append lane
  last WROTE — answers "what does the system actually believe".

WHY: the two are synonymous only right after a fresh full UAD. Daylight between
them = stale or fossilized stored lane — exactly the 2026-06-11 snapshot-lineage
bug (sid 302: ribbon 100% green while stored trades carried wrong stop exits).
The toggle turns that whole bug class into a glance-level diagnostic.

Implementation sketch: backend — extend the gate-parity/ribbon endpoints to
optionally source backtest edges/series from the trades table (entry/exit
fill_ts + exit_reason markers; no indicator series for stored mode, so the
ribbon's stored mode compares TRADE EDGES not per-bar values — a third
"trade-edge parity" row group). Frontend — toggle like the first/latest one on
ParityDriftRibbonV2.

### Bug landscape after iter 0611a/b (for UAD-timing decisions)

- B1 snapshot lineage fossilization — FIXED by guard 5b6f7a9 (deploy pending
  2026-06-11 night). Appends self-heal post-deploy; full UAD only needed to
  rewrite HISTORICAL stored trades.
- B2 live-edge writes: full UAD writes trades up to "now" with unsettled REST
  (sid 303's 15:26:30 trade diverged at the data edge during a deploy window);
  appends are lag-protected (15 min) but never REWRITE a wrong edge trade once
  stored. Candidate fix: clip full-UAD writes at now-15min (mirror append lag)
  and/or re-verify the last N stored trades on each append. SMALL, not urgent.
- B3 decision-time vs settled marginal crosses (~8% corrected bars;
  would_fire_post_correction stamps these) — irreducible at sub-5s latency;
  measure, don't chase.
- B4 forming-bar/grace C-type evaluation — possible small residual; re-measure
  post-guard before investing.
- B5 gated-cohort phantom round-trips / position-desync stints — likely mostly
  B1's downstream; re-measure on healed lanes before treating as separate.
- B6 parked small items: GATE_DIAG log flood strip, apply_rest_correction
  shadow branch doesn't re-derive confluence records, warmup-vs-update_bar
  `*_prev` one-bar transient, full-history flat-row cleanup (159k), 267 TSLA
  flats not cleaned.

---

## 2026-06-18 — Speed batch (what we were trying to accomplish) + pivot to gate-timing

**Goal of the speed work:** the divergence investigation, charts, UND/UAD and
Mass Builder were all bottlenecked on `prepare_data_with_indicators` recomputing
ALL 15 enabled confluence groups over the full bar set (OOM + 5-7× slowness on
sub-minute strategies). The aim was to make investigation *fast* WITHOUT touching
the fidelity (backtest↔live) parity — so we could iterate on real divergences
instead of waiting on data.

**What shipped (all behind the byte-identical Fidelity Gate):**
- **#21** (`1bbad56`) — scope the group loop in prepare to only the groups a
  strategy reads (derive-from-strat; kill-switch `RORT_SCOPE_CONFLUENCE_GROUPS`,
  now =1 on api; live engine left FULL on purpose). Result: ~2× prep + much
  narrower df (OOM relief), byte-identical (golden 3/3 + parity guard, gate
  states REAL-diff=0).
- **#29** (`13c4786`) — chart-data: pass `strat=` so #21 scoping applies to
  charts + trim the returned df to the visible window (full-history warmup kept
  for correctness; only never-rendered bars dropped).

**What we measured and DROPPED (kept the parity, not the marginal speed):**
- Profiling a scoped recompute: **trade engine = 76%**, prepare = 24%. So
  scoping interpreters/triggers (the rest of #21) is ~7-10s marginal on the
  shared path → dropped. The real recompute lever is trade-engine throughput
  (separate, risky — explicit decision required, not folded into a speed batch).
- #29 "decouple" (build daily/4h secondaries from a cheap 1Min load instead of
  the sub-minute primary) — **empirically rejected**: daily/4h OHLC+indicators
  DIFFER between 1Min-source and 15Sec-source (close Δ0.055, volume Δ527k). Since
  backtest resamples secondaries from the primary, a 1Min-sourced chart heatmap
  would disagree with the backtest → divergence. Residual: 365d×15Sec ≈ 5.7M-bar
  *load* for sub-minute+daily charts has no fidelity-safe quick fix (deferred).

**Pivot:** speed is banked; back to the actual divergence. Leading hypothesis on
the new gated real-money strategies (309/310/313): backtest gates on the
PREVIOUS secondary bar (`<interp>__<tf>`, shifted) while live gates on the
CURRENT closed bar (`_spec_<interp>__<tf>`). Diagnosing read-only via
`gate_parity_harness.py` (PB vs CB ribbon distributions + theoretical-BT vs
live-actual + CBpass&PBfail) before any fix.
