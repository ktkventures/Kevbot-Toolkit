# Roadmap — Divergence Hunting (Live ↔ Backtest Pair-Rate to 95%+)

**Last updated:** 2026-06-08 EOD
**Goal:** Drive fleet-wide live↔backtest pair rate to 95%+ across the canary cohort.
**Status (2026-06-08):** Battle Plan window completed; trigger-mode recovered to 70.2%
(vs Friday 47.4%); 9 strategies hit 100% in ≥1 hour today. Engine capability proven.
Next: pack-quality investigation + replay harness. See "Update 2026-06-08" below.
**Status (2026-06-05):** 96.7% achieved on sid 302 immediately post-UAD. Architectural
work delivered. Issue diagnosed as OPERATIONAL (BT-lane backfill), not engine divergence.

This doc is the single source of truth for what's done, what's open, and where the
related artifacts live. Update at each session's end.

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
- Fix Mass Builder legacy configs (sids 174, 194) — read DB configs, diff vs
  strategy_factory schema, update, UAD-validate. ~1 hour total.
- Investigate Stochastic's structural lead — pick a 100% hour, trace why every alert
  paired, document the pattern. Use as "golden child" template.
- Apply Stochastic's pattern to SuperTrend (81.2% → push past 90%) and Bollinger
  (79.4% → push past 90%). UAD-test each fix.

**Phase 2 (2-3 days) — Build bar-replay harness, Bucket C-1:**
- Record Monday RTH bar stream into a fixture file (`fixtures/replay_2026-06-08_rth.jsonl`)
- Write harness `src/_replay_engines.py` that runs unified_engine + ralph_engine on
  the fixture and emits trade lists + alert lists
- Validate harness reproduces today's measured combined % per strategy within ~2 pts
- This is the unlock: every subsequent hypothesis becomes minutes-iterable

**Phase 3 (2-3 days, post-harness) — Gate-mode systemic break, H1:**
- Pick Bollinger Bands gate (sid 277) as the test case
- Take 10 phantom alerts from today's RTH
- Replay each bar's context through both engines via the harness
- Find the asymmetric step (likely confluence-record interpretation between live tick
  handling and BT bar-close)
- Fix → harness-test → UAD-validate on sid 277 → live-validate during next RTH
- Apply same fix to other broken gates if the diagnosis is structural

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

**Updated 2026-06-08 — supersedes the 2026-06-05 run-order below.** See "Phase plan"
in the 2026-06-08 update above for full detail.

1. **Phase 1 quick wins** — fix Mass Builder legacy configs (sids 174, 194), document
   Stochastic's structural lead, apply pattern to SuperTrend + Bollinger
2. **Phase 2 bar-replay harness** — build the offline replay tool (`src/_replay_engines.py`
   + Monday RTH fixture). This is the unlock for fast iteration on every subsequent fix.
3. **Phase 3 gate-mode break** — diagnose Bollinger gate (sid 277) via harness, fix,
   propagate to other broken gates
4. **Phase 4 pack-specific cleanup** — VWAP v2 hotfix audit + SR Channels migration
   completion (parallelizable with Phase 3)
5. **Phase 5 Pack Validation Sweep** — fixed golden window regression test as pre-deploy
   gate
6. **Define autonomous operations protocol** — separate discussion with Kevin; capture
   in own SOP doc

### Run-order from 2026-06-05 (carried over, lower priority)

1. **Build smart gap-detection cron** — code the detector + action + cron wiring (~2 hrs)
2. **Verify on one strategy** — run detector → backfill → compare to manual Update All Data
3. **Deploy as hourly cron** — once verified accurate
4. **Re-measure pair rate fleet-wide** — should be 95%+ across all canaries with gap-detection running
5. **Investigate #60 dead algo lane** — separate latent bug, ~30 min
6. **Update SOP** with measurement methodology + new operational patterns
