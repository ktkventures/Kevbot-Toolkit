# Plan — M-RS2 Phase 2 Rollout: Resampled Bar Store → Both Lanes Serve From It

---
## ▶ ACTIVE WORK QUEUE (agreed with Kevin 07-14 — updated ~23:10Z)
Context: the gate-state class is fixed+proven (#61/#62/#65); the live engine was being
STARVED by PRIMARY-RESYNC (disabled 20:11Z — see the 🚨 section below), which invalidated
today's live paired-% numbers. Neither 07-13 (gate bugs live) nor 07-14 (stalls) is a clean
day; **07-15 is the first clean read.**

**✅ STATUS ~23:10Z — items 1, 2, and 4 are DONE; canonical-edge build DELETED.**
- **1. Liveness watchdogs** — SHIPPED+LIVE (PR #66). Alert lag median 6s; watchdogs silent.
- **2. Replay harness** — WORKS (`src/replay_harness.py`; memory `project_replay_harness`).
  Ceiling 327=95% 328=88% 333=83% @10s vs live 48/64/63% → stalls cost ~25-47pp.
- **4. Primary-drift check** — DONE: **no trade-affecting primary drift exists** (all
  in-session resync checks were misaligned-endpoint artifacts; aligned ones = volume/tip/
  float noise). ⇒ **RESYNC STAYS DELETED PERMANENTLY** (code-removal cleanup PR later). The
  340-class was fixed by #65 + boot-warmup, not by rebuilding primary state.

**⏳ TOMORROW (07-15) — the plan Kevin can commit to (~2h, not a full day):**
Kevin will hold page edits until ~**15:30Z (09:30 MT)** = ~2h of RTH + pre-market data, then
we do BOTH in one pass:
  (a) **Measure the Five** on the clean window (stalls gone, gate fixes in) — the real verdict
      vs the ≥90%@10s bar; and
  (b) **VALIDATE THE REPLAY HARNESS**: run it over the same clean window and confirm
      **REPLAY-vs-LIVE @10s ≥ ~90%**. That agreement is what promotes the harness from
      "strong hypothesis" to "proven oracle" — and it doubles as the permanent primary-drift
      monitor (live ≈ fresh warm ⇒ no drift). Once validated, the harness graduates to its own
      skill (and, later, a possible admin-page "Replay" tab). This is the payoff: validated,
      it makes every future logic edit testable offline in MINUTES instead of an overnight soak.

These items made the next read trustworthy AND make future test-and-learn fast.

**1. Latency tripwire + worker healthcheck freshness** (small, prevents recurrence)
   - WARN when alert dispatch lag (`alerts.timestamp − fill_ts`) p95 > 30s (normal 3–8s).
     Would have caught today's stall in MINUTES instead of after a contaminated day.
   - `Dockerfile.worker` HEALTHCHECK currently only does `test -f /tmp/worker_alive` (file
     EXISTENCE — a hung worker passes forever). Make it validate file AGE + last-candle
     freshness. (Credit: the Sol audit — this one survived contact with current dev.)

**STATUS 07-14 ~22:20Z — items 1 & 2 DONE; item 1's canonical-edge build DELETED.**
- ✅ **1. Liveness watchdogs SHIPPED+LIVE** (PR #66, `5fdaeaa`): `[ENGINE-STALL]`,
  `[ALERT-LAG]`, engine-heartbeat healthcheck. Post-deploy silent; alert lag median 6s.
- ✅ **2. Replay harness WORKS** (`src/replay_harness.py`, memory `project_replay_harness`).
  **ACHIEVABLE CEILING (07-14 RTH, armed stack, no stalls): 327=95%, 328=88%, 333=83%
  (93% @60s) @10s — while LIVE scored 48/64/63%.** ⇒ the resync stalls cost **~25-47pp**
  today; with resync off, tomorrow should land near the ceiling. Entries are ~92%
  EXACT-SECOND matches vs backtest; the residual is a couple of trades + exit timing in the
  10-60s band (the next hunt, now diagnosable in minutes instead of overnight).
- ❌ **CANONICAL-EDGE BUILD DROPPED.** The harness A/B'd it (`REPLAY` vs `REPLAY+FIX`) and the
  results are IDENTICAL — because the engine's fan-out secondary bar is byte-identical to the
  canonical resample-from-1Min AND to REST. The "fan-out bars are noisy" theory is dead. A
  whole build avoided by measuring first.

**2. Replay harness v1** (the accelerator + the counterfactual answer)
   - Replay recorded DECISION-TIME bars (`live_bars.first_*` — the WS view at the moment the
     engine decided; seed recipe in `src/_retro_replay_gate_states.py` header) through the
     fixed engine → the trades live WOULD have taken with no stalls/dispatch loss.
   - Pair replay-vs-backtest at **5s / 10s / 60s** → the achievable combined-% ceiling (the
     number Kevin actually reads on Strategy Health). live-vs-replay gap = operational cost
     (stalls, lost alerts); replay-vs-backtest gap = REAL remaining logic bugs.
   - Permanently replaces overnight soaks for validating a fix (soak stays as final check).

**3. Fleet divergence scan at 5s/10s/60s** (read-only; "are there other bugs?")
   - `strategy_health.get_strategy_health(user=None, tolerance_seconds=…)` offline across the
     catalog, gated strategies first; cross-read with #2's replay to separate logic from ops.

**4. PRIMARY-RESYNC: MEASURE before redesigning** (likely: delete, don't rebuild)
   - Kevin's architectural read (endorsed): resync = the OLD "reload history + rebuild" way,
     smuggled back in. The streaming architecture already corrects state EVENT-DRIVEN
     (rebroadcast correction, gap-heal insert-replay, settle sweeper). We only built the
     periodic sledgehammer because a correction primitive was silently broken — and that
     primitive is exactly what #65 fixed.
   - So: re-measure primary-state drift NOW that #65 is armed. If drift is gone → resync stays
     OFF permanently (whole risk class deleted). If real drift remains → fix it event-driven,
     never on a timer inside the trading process.
   - Note why it was so expensive: the store starts at 2Min, so sub-minute PRIMARIES
     (10s/15s/30s) fell through to a native 7-day load (~180k bars) + full replay PER
     STRATEGY, ~4s each × ~50 = ~9.5-min GIL-held cycles every 900s.
   - Clarification for the record: Update-All / nightly recompute runs on **batch-worker**, a
     SEPARATE service — it never stalls the live engine. No need to trade away intraday lane
     freshness; the rule is simply: no heavy rebuilds inside the live Worker process.
---


**The living rollout tracker** (created Fri 2026-07-10 ~22:00Z, main session). Design/why:
`Design_MRS2_Phase2_Resampled_Bar_Store.md` · hardening context: `Design_Gate_Fidelity_Hardening.md`.
North star: gated strategies (the tradable ones) reach **≥90% combined @ ≤10s tolerance** measured
from their most-recent-activation window; the store retires the gate-construction divergence class
so residual divergence = real logic bugs.

## Shipped state (all in dev mainline, Fri 07-10)
- **PR #52** — store (`resampled_bar_cache`, session-keyed, 2Min…1Day × TSLA × RTH/Extended),
  seeded 106,296 rows (TF-adaptive depth), consumers #1 (offline backtest secondary, v3) and
  #2 (ribbon swap). Suite 16/18 == baseline (2 known SPY/10Sec pre-existing fails).
- **PR #53** — consumer #3 verify-first: the LIVE warmup/reload chokepoint
  (`ralph_engine._load_warmup_df`) verifies the store against what it just built. Coarse + fine
  TFs, session-keyed. Log-only.
- **PR #54** — store maintain wired to the settle-sweeper cadence (after each 1Sec/1Min sweep;
  throttle 900s; breaker-guarded).
- **Design invariant everywhere: verify-first.** Flag ON runs the SAME output code as OFF (a trade
  cannot change even in theory); the armed flag only adds byte-verification of the store vs what
  the engine actually built, on the settled/bin-aligned comparable zone. `[ResampledStore#1]` =
  offline, `[ResampledStore#3-live]` = live. DRIFT warnings = real divergence, alarm-worthy.
- **Flags ARMED Fri 21:46Z** (by main session): `RORT_RESAMPLED_STORE_WRITE=1` on Data Worker;
  `RORT_RESAMPLED_STORE_READ=1` on Data Worker + Worker + api + batch-worker.
  **shadow-worker deliberately NOT touched** (var-set reverts its pinned snapshot; it gets the
  new code via a deliberate `railway up` + fingerprint later). Rollback = unset the var.

## Phase 1 — Arm & observe (Fri evening) ✅ COMPLETE (00:30Z Sat)
- **Live verify: 100+ GREEN sweeps** across the fleet (TSLA full board ×5+ independent boots;
  SPY/KO/DIA/TSLL as seeded/maintained). ONE true drift caught → diagnosed (late-print settle
  revision on the newest bar) → maintain re-trued → GREEN. **Full detect→diagnose→heal loop
  proven in production in under an hour.**
- **Store COMPLETE: ~700k rows** — all 7 capture symbols × 10 gate-eligible TFs × 2 sessions,
  every chunk comparator-clean at write. Targets fully config-derived (Timeframes page × Bar
  Cache capture authority) — new TFs/symbols auto-join.
- **00:00Z close check: ALL GREEN** — post-close maintain (33.6k rows, 0 errors) captured the
  settled close; sentinel settled-shadows (TSLA 1Day both sessions, SPY 2Min Ext) match=True.
- **8 silent-fail catches** during rollout (window-alignment #5, CLI TF-default shrink, silent
  maintain skip, error-dicts-as-success, bar-set-vs-edge-lag conflation, +3 earlier) — each now
  logs loudly or is structurally impossible.
- DSN (`SUPABASE_CONNECTION_STRING`) set on Data Worker (Kevin-approved 23:59Z). Sweeper idles
  outside market window → **first on-service credentialed maintain = Monday pre-market ~08:00Z**
  (first item on Monday's checklist). Weekend staleness impossible (no new bars form).
- What Friday evening could NOT give: RTH open/close boundary under full live WS load → Monday's
  observation, gating ONLY the live serve flip (Phase 4), not the weekend work.

## Phase 2 — Weekend sprint ✅ items 1/2/4 DONE (Sat ~06:30Z); item 3 remaining
1. ✅ **Fleet ledger** (`Weekend_Sprint_Ledger.md`): 41 gated sids, 18 targets — **ZERO drift
   fleet-wide**; store byte-identical to canonical everywhere the fleet gates. Store-integrity
   evidence for the compute-skip gate = MET. (1Day head margin = 0 bars: reseed head before any
   warmup-depth change.)
2. ✅ **Fine-TF quantification** (`Fine_TF_Divergence_Quantification.md`): **construction
   FALSIFIED as the cause** — ≤1% gate-flip ceiling vs the 37-78% gap. Traps found: volume gates
   (VWAP/RVOL) on sub-minute primaries structurally divergent in backtest; fan-out persisted bars
   2× noisier than clean 1Min aggregation.
3. ⏳ **Offline compute-skip** build + fleet byte-proof (main session, next).
4. ✅ **Autopsy** (`Gated_Five_Divergence_Autopsy.md`, method anchor: theoretical engine == recorded
   backtest 164/164): ranked causes of the five's divergence —
   (1) 30% interp-blind shadow = **DEAD** (PR #38, 07-07): clean-window re-baselines 327→67%,
   328→64%, 333→61%, 340→33%, 329→29%;
   (2) 30% live shadow interpreter-state divergence (stale/wrong/lagged; 333's UT_BOT wrong-side
   lock survives reboots — hysteretic warmup, needs deep-anchor warmup + #32 refresher extended
   to fine TFs + the Phase-4 store cutover);
   (3) 27% **cross-TF boundary race** — ordering SEMANTICS (live gates on the just-closed
   secondary at coinciding closes; backtest one primary bar later). Store cutover will NOT fix;
   needs a semantics-alignment DECISION (live-side one-bar defer is parity-preserving) — Kevin
   input wanted;
   (4) 6% WS≠REST primary floor (accepted per the 90%@10s bar) + 340's exit-side MACD-cross
   timing (only genuine exit-side case);
   (5) 6% **gate-pass anomaly** — 7 fires (327/328, 07-09 18:04:30Z) with truth AND telemetry
   saying gate CLOSED; check `GATE_DIAG` logs (ralph_engine.py:1357) + `check_entry` empty-set
   fail-open (unified_engine.py:~2527) — Monday (CLI log retention insufficient Saturday).

## Phase 3 — Monday: boundary + live day #1
- Weekend-boundary observation (Fri close → Mon open maintain/verify — the gap class).
- Full RTH session of live verify evidence under WS load.
- If green: arm the OFFLINE compute-skip (already proven). Watch one recompute cycle.

## Phase 4 — Live serve cutover (mid-week, evidence-gated)
- The live engine serves its gate bars from the store (the structural 313-class kill; subsumes
  `RORT_MTF_COARSE_RTH_RELOAD`). Gated on: Phase 2 ledger green + Monday green + offline
  compute-skip stable. Kill-switched; shadow-worker gets code via deliberate `railway up`.

## Phase 5 — Measure against the bar
- Re-score the gated fleet at 90%@≤10s on clean post-cutover recent windows. Remaining shortfalls
  = real logic bugs → targeted hunts with the construction fog gone.

## Open items / risks
- SPY/10Sec cache re-true (2 known suite fails, orthogonal — queued).
- `test_ralph_fidelity::test_strategy_monitor_warmup_and_bar_close` pre-existing fail (on clean
  dev too) — diagnose separately.
- Sub-minute store TFs (resample-from-1Sec) = deferred Phase 2b; `base_layer` seam reserved.
- Store coverage: TSLA-only targets today; config-derived (Timeframes page) — new TFs auto-add,
  new SYMBOLS need targets (extend when non-TSLA gated strategies matter).
- Fine-TF volume nuance: store = resample(1Min); backtest fine-TF = resample(primary). For
  sub-minute primaries the volumes legitimately differ (1Min ≠ Σsub-minute) — price-only gates
  unaffected; volume-sensitive packs (RVOL) need the quantification in Phase 2.2 before any
  fine-TF serve decision.

## Phase 3+ EXECUTION LOG (Mon 2026-07-13) — for session handoff
**Armed today (all validated before arming; deploy log has exact times):**
- 15:35Z `RORT_MTF_PB_DEFER` (PR #55) — PB gates honor last bar closed BEFORE the primary bar
  (Kevin's PB/CB ruling = memory `feedback_pb_boundary_semantics_ruling`); replay 254/254 == backtest.
- 15:41Z `RORT_GATE_FAIL_CLOSED` (PR #56) — empty record set can no longer fire ungated (autopsy E).
- 16:42Z `RORT_RESAMPLED_STORE_SERVE` on api/batch/DataWorker/Worker (Kevin-named) — OFFLINE lane
  serves settled coarse bars FROM the store (compute-skip). Canary: first run fell back on head
  coverage (seed 182d < full-recompute window) → coarse TFs re-seeded to 400d (12k rows, clean) →
  canary v2 pending SERVED confirmation.
- 17:07Z `RORT_MTF_STATE_REFRESH_S=600` — the REAL 333-class healer (close-feed FREEZE, see autopsy
  CORRECTION 1; hysteresis falsified; PR #57 deep-anchor = default-OFF insurance).
- 18:25Z maintain cadence 900s in-session (store hugs the settle boundary).
**Also:** PR #58 admin page `/admin/resampled-store`; autopsy CORRECTION 2 = 340 exits reclassified
D→C (in-memory MACD state drift; boot-warmup injection + wander; NOT the WS floor).
**Kevin rulings on file:** PB/CB boundary (backtest = spec); primary-state re-sync approved
INCLUDING mid-position ("indicator states can change mid-position — exits depend on them").
**~19:30Z — item 1 BUILT + validated (PR pending merge/arm):** `RORT_RESAMPLED_STORE_SERVE_LIVE`
(default OFF) in `_load_warmup_df`: for store TFs (2Min+), warmup serves [store settled whole-days
+ fresh 1Min head/edge] via the SAME `_coarse_secondary_serve_from_store` splice the offline lane
armed at 16:42Z; ANY doubt → unchanged deep path (== OFF bytes). Validation ALL GREEN (23/23):
11 e2e OFF-vs-ON `_load_warmup_df` cases byte-identical (TSLA all store TFs × RTH/Ext + SPY/KO);
10 fixed-window serve-vs-deep byte-proofs incl. Fri-holiday/Sat head days; 2 graceful fallbacks
(uncovered symbol/session). Bycatch fixed in the SHARED splice: head-coverage tolerance was
calendar `+2d` → any Fri/Sat-head window (2/7 of days!) + long weekends fell back needlessly
(that's why offline 5Min "fell back on head coverage" this morning — window head = Fri Jul-3
holiday weekend, NOT short store coverage) → now `+5d`, still catches real under-coverage by
months. Store depth probe: fine TFs (2Min-30Min) head 2026-04-13 (~90d) vs ≤19d warmup need;
1Hour 2025-06-09; 4Hour/1Day-RTH 2025-03-20; 1Day-Ext 2025-06-09 vs 355d need — ALL live warmup
windows covered. Ralph fidelity 13 pass (+1 known pre-existing fail); parity suite run logged
below. NOTE: 1Min PRIMARY warmup (340's boot injection) is NOT store-covered (store starts at
2Min) — that class needs item 2's re-sync (Kevin-approved) or the interim settled-only tail.
**REMAINING (ranked):**
1. ✅ **SHIPPED + ARMED 20:59Z** — Live-lane serve (Phase 4): PR #59 merged (`d48c884`),
   `RORT_RESAMPLED_STORE_SERVE_LIVE=1` on Worker (Claude-armed per Kevin's standing
   authorization). Evidence: 23/23 byte-validation; suite effectively 18/18 (267 post-settle
   PASS 308==308); 338 sandwich OFF^ON=0 across 3,323 trades with SERVED confirmed (also
   closes the 16:42Z "canary v2" pending item — the morning fallback was the +2d weekend
   tolerance, fixed to +5d). Boot watch: `[ResampledStore#4-live] SERVED` lines.
   Canary-script lessons: set_admin_user_context + pack_registry.scan_and_load_all needed in
   ANY standalone engine script; run canaries OFF/ON/OFF when a session is live.
2. ✅ **SHIPPED (PR #60 f3b297b) + FULLY ARMED** (`RESYNC_S=900` 21:14Z; **APPLY=1 ~21:45Z** —
   Kevin's tomorrow-opens-healed directive; overnight no-op, intraday drift heals ≤900s)
   — Primary-TF pack state re-sync (Kevin-approved,
   mid-position OK): `SymbolHub.resync_primary_states` rebuilds each monitor's primary
   indicator/pack state from a fresh `_load_warmup_df` (store-served for 2Min+) and diffs it
   against live in-memory state. Two flags: `RORT_PRIMARY_STATE_RESYNC_S` (cadence; dry-run
   drift METER — `[PRIMARY-RESYNC]` lines) and `RORT_PRIMARY_STATE_RESYNC_APPLY=1` (stage swap,
   consumed at the monitor's next bar close ONLY when the rebuild ends exactly at its last
   committed bar; grace-fire suppressed; pre-bar rewind tokens invalidated on swap). Position
   machine untouched. Replay validation 6/6: no-op byte-identity (swap+next-commit == never-
   swapped), injected-MACD-drift detect+heal (the 340 mode), misalignment drop, grace-fire
   suppression, hub dry/apply staging. Ralph fidelity 13 pass. ARMING PLAN: dry-run meter
   first (RESYNC_S=900, no APPLY) → review fleet drift lines → flip APPLY.
3. ❌ **RETRACTED 07-14 ~00:20Z (overnight loop): the "MTF publish CLOBBER" was an ANALYSIS
   ARTIFACT.** The 07-13 last-writer analysis grouped `bar_diagnostics` live_gate rows by
   (tf, session) **across symbol hubs** — but each hub has its own `_mtf_confluence`; rows
   are only comparable per strategy_id. The "narrow sets landing last" were other hubs'
   legitimately-narrow keys interleaving: tf=900 MACD-only = sid 136 (SPY — the ONLY
   15m-MACD_HISTOGRAM gate in the fleet), tf=120 EMA_PP-only = sid 194 (TSLL), tf=300
   MACD_LINE-only = sid 271 (SPY). **Corrected per-hub scan (15:41→24:00Z, all 41 gated
   sids): needed interpreters present in ~100% of each sid's own change-events** — 327/328/329
   tf=300 SWING 47/47, 327 tf=120 SWING 116/116, 333 tf=900 UT_BOT 4/4, 333 tf=180 86/86.
   No merge fix shipped (validation-gate rule: don't ship a fix whose target is disproven).
   The Five's residuals are NOT gate-key starvation → weight shifts to the T2 counterfactual
   audit. Old 3b ("900s starvation") stays closed — the 15m key is healthy per-hub.
3'. 🔥 **REAL bug the corrected scan exposed (T1', building now): warmup/recompute drop the
   prev-value chain → shift-based pack interpreters emit NOTHING on derive paths.**
   Evidence: sid 285 (SPY 2m Extended gate) `MACD_HISTOGRAM_V2` in **0/204** events all day;
   sid 136 (SPY 15m) key **EMPTY 5–20 min after every deploy** (7 boot-gaps 07-13 — the
   class-E ungated fires pre-15:41Z, silent blocks post-FAIL_CLOSED). Mechanism (code):
   `IncrementalIndicatorEngine.warmup()` never populates `state.prev_values`/`prev_macd_hist`/
   `prev2_macd_hist` (only `update_bar` does); `recompute_from_history` full path = reset +
   warmup → same. So `_derive_confluence_records` (warmup seed, non-RTH reload EVERY close,
   600s refresher, rebroadcast recompute, gap-heal) hands prev={} to the user-pack dispatch →
   1-row df → shift(1)=NaN → no record. Same family as the 04-29 sid-137 fix, one layer deeper.
   FIX: `RORT_WARMUP_PREV_CHAIN=1` maintains the update_bar prelude inside warmup (flag OFF =
   byte-identical). Acceptance = the fleet presence scan at 100% incl. 285/136 + no post-boot
   empty windows.
4. Grace-fire `_fired_bucket` suppression hole (ralph 2838/3846) — latent alert-drop, ticket.
5. EOD + next clean day: re-measure the five @90%@10s on post-17:07Z windows.
6. Housekeeping: SPY 10Sec re-true; 3 pre-existing test_unified_parity fails; frontend '1M'→'1m';
   span-hash incremental maintain.
## 🌙 OVERNIGHT LOOP HANDOFF (written 2026-07-13 ~23:40Z for the /loop session)
**Armed flag state (Worker):** RORT_RESAMPLED_STORE_SERVE_LIVE=1 · RORT_PRIMARY_STATE_RESYNC_S=900
· RORT_PRIMARY_STATE_RESYNC_APPLY=1 · RORT_MTF_STATE_REFRESH_S=600 · RORT_MTF_PB_DEFER=1 ·
RORT_GATE_FAIL_CLOSED=1 (+ api/batch: RORT_RESAMPLED_STORE_SERVE=1, READ=1). All kill-switch =
unset the var. Deploy log has times.
**The Five's lanes:** settled-recomputed 23:06Z under prod-parity flags (329 repaired −787).
**NIGHTLY COORDINATION:** nightly fleet recompute starts 00:20Z on batch-worker (~couple hrs).
While its jobs run: NO dev pushes / PR merges (redeploys orphan jobs), NO local lane writes,
lane reads may be mid-rewrite. Build + replay-validate + prep PRs locally meanwhile; detect
completion (recompute job records / batch-worker logs quiet 10+ min) → then ship the queue.
**OVERNIGHT LOOP EXECUTION (07-14, this session):**
- ~00:20Z clobber RETRACTED (item 3) → T1' prev-chain bug found + built (PR #61).
- ~00:45Z 🔥 **NEW finding while auditing T2 events: fine-TF shadow CASCADE-RECOMPUTE
  POISONING** (memory `project_shadow_cascade_recompute_poisoning`) — the ≥60s fan-out
  duplicate branch recomputes fine-TF RTH shadow records from `sec_builder.history` (shallow
  anchor, fan-out-flavored bars) every ~2-3 min and republishes; path-dependent SWING counts
  land off-by-one → the wrong-family state OWNS the gate key between boundary closes (proof:
  GATE_DIAG 18:41:03 sid-328 fire with 5M SWING BULL_C2 == settled truth, vs live_gate
  telemetry steady BULL_C3 18:25→19:15; cascade log re-poisoned the 2m key 11s after the
  18:42:00 boundary state fired sid 327). THE divergence engine for 327/328/329's 5m/2m SWING
  gates; revises autopsy class B; explains why REFRESH_S=600 never stuck. Fix = PR #62
  `RORT_MTF_FINE_INCREMENTAL_AUTHORITY` (skip duplicate recompute+republish for fine RTH
  shadows; refresher does canonical re-trues; sub-minute discipline mirrored).
- 02:35Z nightly quiet (69 refreshed) → 02:36Z merged PR #61 (`c1e8875`) + PR #62 (`eb4ae5b`)
  (backup `backup/dev-pre-prevchain-fineauth-2026-07-14`) → 02:38Z ARMED on Worker:
  `RORT_WARMUP_PREV_CHAIN=1` + `RORT_MTF_FINE_INCREMENTAL_AUTHORITY=1` +
  `RORT_MTF_STATE_REFRESH_S=120`. Deploy log has details. The two fixes COMPOSE: the 120s
  refresher re-true is now both clean-sourced (store) and faithful-derived (prev chain).
- T2 audit tooling ready (`scratchpad t2_audit.py` pattern); pre-fix window baseline:
  327=33% 328=25% 329=0% 333=65% 340=quiet. Most unpaired events attribute to the cascade
  class (bt entries sat inside settled gate-open windows the poisoned live key missed).
  Post-nightly re-audit + tomorrow's live telemetry-vs-settled comparison = acceptance.
- **T2 COMPLETE (~02:50Z, post-nightly lanes == pre-nightly, window final):** every unpaired
  17:07-20:00Z event on 327/328 (12 events: live 18:22/18:41/18:42 fired in boundary flashes;
  bt 18:38/19:19/19:45:30 inside settled gate-open windows the poisoned key missed) +
  329 (2: same flash mechanism on its tf=60 key — also fine-TF, #62 covers) attributes to the
  cascade class → resolved-by-stack. 333 (65%) = same class on 15m/3m keys + accepted WS
  floor residue. ≥90% attributed ✓. LIVE acceptance = T3: telemetry-vs-settled gate series
  match on the armed stack + the Five ≥90% paired in the 1h/3h window during RTH.

- **T3 PRE-MARKET BATTERY ALL GREEN (08:13Z):** fleet presence scan since 08:00Z = **32/32
  keys, zero missing/empty** (the T1 acceptance metric, met on live data); sid 285's
  `2M-MACD_HISTOGRAM_V2` present in EVERY event with evolving state (0/204 all yesterday —
  the #61 live proof); `[FINE-DUP-SKIP]` firing on TSLA/TSLL/SPY (#62 live);
  `[MTF-REFRESH] cycle: hubs=5 refreshed=18 failed=0` at the new 120s cadence;
  `[PRIMARY-RESYNC] APPLIED` healing real drift (287/288/289); 0 errors, no GATE-FREEZE.
  09:00Z pre-market paired-%: 338=90% (9E+9X), 290=87%, 285 firing again (was structurally
  blocked), 337/291 legitimately quiet. RTH goal check (the Five ≥90% @1h/3h) after 14:30Z.

- **15:30Z GOAL CHECK (loop close-out; window 13:30→15:04Z, bt fresh to 14:39Z):**
  328 = **75%** (3/4 E + 3/4 X; was 25% yesterday), 327 = **25%** (bt fired 4×, live 2×),
  329/333 quiet on BOTH lanes (agreement, no opportunities), 340 = 1 bt-only trade.
  ≥90% NOT reached for 327/328 → 3-probe run on every unpaired event:
  **RESIDUAL CLASS PINNED (new precision): live fine-TF SWING states drift from settled
  truth; the BACKTEST was RIGHT today.** At 14:39 both bt-only entries sat on gates where
  settled-1Min truth said 5m=BULL_C2 + 2m=NEUTRAL (== backtest) while the LIVE keys held
  5m=NEUTRAL + 2m=BULL_C2 (both wrong). Mechanism: (a) the boundary incremental state is
  path-dependent and drifts from its boot lineage; (b) the 120s refresher CANNOT heal the
  newest bar — its clean source (store settled + 1Min head) lags ~1 fine-TF bar (settle
  physics), so it re-trues to the one-bar-old state and CONFIRMS the boundary error instead
  of correcting it. Also one live-only entry (13:41:30) where live gates were genuinely open
  and one primary-trigger timing miss (13:51:30, gates open live, 30s trigger differed) —
  WS/REST floor class. Three-lane arbitration: live ≠ (bt ≈ settled) → LIVE-side bug.
  **⇒ KEVIN DECISION NEEDED (day session): canonical fine-TF gate STATE serve** — extend the
  store cutover so the live shadow's newest fine-TF bar/state is built from the same 1Min
  head the offline lane uses (M-RS5 resident-window / Phase 2b family; price-based packs
  safe, volume caveat only for RVOL/VWAP). Until then 327/328's ceiling is set by how often
  SWING flips near boundaries. Poisoning class (yesterday's engine) = confirmed DEAD live:
  keys stable between boundaries all morning, refresher-backed, [FINE-DUP-SKIP] active.

## 🚨 07-14 20:11Z — PRIMARY-RESYNC DISABLED (engine starvation; today's data invalidated)
Post-close three-lane arbitration of 333's 18:44:30 miss (algo == backtest == fired, live
absent, gates open on both lanes, primary bars byte-clean) traced to the LIVE ENGINE BEING
STALLED: `resync_primary_states` full-warmup-replays EVERY monitor (~50 × ~4s = **~9.5-min
cycle**) every 900s, GIL-held → no `BAR_CLOSE` processing for most of each cycle. Fleet-wide
**alert dispatch lag 5–10 min** (fill 18:54:30 → saved 18:59:43 across ~13 sids) and missed
entries. `RESYNC_S=0` + `APPLY=0` at 20:11Z → **median lag recovered 320s → 4s**.
**Implications:** (1) ALL of today's live paired-% data is invalid for judging #61/#62/#65 —
misses may be stalls, not divergence; the clean verdict comes from 07-15 with resync OFF.
(2) The gate-state fixes remain independently proven (retro-replay + `changed=5` refresher
cycles + correct keys). (3) The resync feature (340-class primary drift healer) must be
redesigned before re-arming: shared engine per (tf, session, indicator-signature) instead of
per-strategy replay, chunked cycles with a wall-clock cap, yields between strategies; add a
tripwire on alert `timestamp − fill_ts` p95 > 30s. Memory:
`project_primary_resync_engine_starvation`. **Meta-lesson: correctness validation (replay
6/6) is NOT latency validation — cost at fleet scale must be measured before arming.**

## POST-ARM STATUS (07-14 ~19:50Z) — partial evidence, clean verdict deferred to 07-15
**What the armed stack showed (17:52→19:05Z, the uncontaminated slice):**
- Refresher is genuinely re-truing now: `[MTF-REFRESH] cycle … changed=5` (pre-#65 cycles
  logged `changed=0` while states were demonstrably wrong — the fast-path bug in one line).
- Gate keys held CORRECT states all window; 333's first three entries paired EXACTLY
  (18:02:30, 18:17:00, 18:33:30) and 327's single post-arm event paired 1/1.
- **RETRO-PROOF (src/_retro_replay_gate_states.py):** replaying 14:30→15:08 under force-full
  semantics opens BOTH gates at the 14:39 events (5M BULL_C2 ✓, 2M NEUTRAL ✓ vs the actual
  NEUTRAL / BULL_C2) — i.e. this morning's goal-check misses WOULD have paired.
**Why today is not the verdict:** (a) tiny N — the Five fired 3–5 events each post-arm;
(b) **ops incident 19:20–19:33Z contaminated the tail** (Supabase broken-pipe burst from
heavy LOCAL analysis during RTH → 14 alert saves lost + 10–12 min dispatch lag; see Deploy
Log + memory `feedback_local_analysis_starves_live_worker`).
**Genuine open residual (pre-incident, worth the next hunt):** 333 bt-only entry 18:44:30
(and exit 18:48:30) with BOTH gate keys open on both lanes → not a gate-state divergence; it
is PRIMARY-side (30Sec trigger) divergence. Algo lane is empty in prod
(`RORT_UPDATE_ALL_SKIP_ALGO=1`) → compute it offline AFTER close for three-lane arbitration.
**Plan for 07-15:** full clean session under the complete stack (#61+#62+#65+PB defer+
fail-closed+store serve+resync) → measure the Five on 1h/3h windows; keep local DB load off
during RTH; then decide whether boundary-inline canonical recompute is needed at all.

## NEXT PHASE — EXECUTION UPDATE (07-14 ~17:50Z): plan item 1 SUPERSEDED by a sharper root cause
Pre-build verification of item 1's premise falsified it step by step (fresh-tail ✓ present,
anchor-dependence ✗ none, incremental==vectorized ✓, data revisions ✗ byte-identical,
write-through quality ✗ identical, refresher running ✓) and converged on the REAL mechanism:
**`recompute_confluence` → `recompute_from_history(df)` takes the snapshot FAST-PATH whenever
a pre-bar snapshot exists (NO alignment check) — restoring the LIVE lineage + re-applying only
the df's last bar. Every shadow re-true since snapshots were enabled (05-19) was
lineage-preserving, not clean.** Live proof: TSLA 5m SWING held NEUTRAL 14:35→15:06+ across
refresher cycles (changed=0) while the identical df full-replays to BULL_C2 (== vectorized ==
settled at 2/10/30d anchors; 1Min inputs byte-identical across live_bars first-writes /
bar_cache / settled). Retro-explains "shadow drift survives reloads/reboots" (autopsy class B
residue, 333's UT_BOT lock). **FIX SHIPPED: PR #65 `RORT_SHADOW_RETRUE_FORCE_FULL` — re-trues
force the full clean replay; armed on Worker ~17:50Z.** With this, the 120s refresher is
genuinely canonical: wrong boundary states heal ≤2min (the 14:39-class events pair). The
boundary-inline canonical recompute (original item 1) stays on the shelf pending post-arm
measurement — only needed if the ≤120s heal window still costs pairs. Items 2 (replay
harness) + 3 (healthcheck) unchanged.

## NEXT PHASE (agreed with Kevin 07-14 ~17:00Z) — Fine-TF Canonical Edge
Target: close the pinned residual (live fine-TF SWING state drift on the NEWEST bar) →
327/328 join 338 at ≥90%; the 5m-SWING Gate Parity row goes green.
1. 🔨 **Canonical newest-bar state serve for fine-TF shadows** (the main build). At each
   fine-TF boundary close, build the just-closed bar by AGGREGATING LIVE 1MIN BARS (same
   resample-from-1Min construction as backtest/store) on top of the refresher's clean t-1
   state — live state(t) = canonical state(t-1) + canonically-built bar(t). Residual after
   this = WS-vs-settled tip on the 1Min bars = the accepted 90%@10s floor. Also improves
   volume packs (1Min AM volume is canonical; fan-out volume is the noisy path).
   **Design choice (Kevin: inline-first approved):** inline at boundary (~100-300ms per
   fine-TF close, canonical before the next primary close) behind a new flag, measure
   hot-path cost; fall back to thread-side boundary-triggered if latency bites (sub-second
   alert latency stays the hard requirement). Same validation discipline: flag-gated,
   OFF=byte-identical, lock tests, replay-predict, parity suite + ralph fidelity.
2. 🔨 **Recorded-events replay harness (in-house).** Replay a recorded session (live_bars as
   the recorded WS stream) through live aggregation AND backtest construction; diff every
   bar, gate state, and decision. The validation tool for #1 and the permanent
   which-bar-diverged lookup. (Idea credited to the Sol audit's closing recommendation;
   building in-house — needs engine-internal hooks.)
3. 🔨 **Worker healthcheck freshness** (30-min quick win, from the Sol audit — verified
   still true): Dockerfile.worker HEALTHCHECK only does `test -f /tmp/worker_alive`;
   make it validate file age + last-candle freshness so a hung worker actually restarts.
4. External-audit protocol (brother-in-law/Sol): their branch reviewed on request; standard
   for adopting anything = failing test against current origin/dev, one small flag-gated PR
   per finding. Their uncommitted aggregation rewrite = do NOT apply (predates months of
   shipped edge-case fixes; see 07-14 review notes in session log).

**T1 — SUPERSEDED 07-14 00:20Z:** clobber RETRACTED (see item 3 above — cross-hub aggregation
artifact; per-hub scan = needed interps ~100% present; memory `project_mtf_publish_clobber`
rewritten). T1' replacement = the warmup prev-chain fix (item 3'): `RORT_WARMUP_PREV_CHAIN`,
same acceptance metric (fleet presence scan 100%, incl. sid 285 MACD_HISTOGRAM + sid 136
post-boot), no-op byte-identity OFF, parity suite + ralph fidelity.
**T2 — counterfactual audit:** today's 17:07-20:00Z unpaired events on 327/328/329/333 (16 events,
listed by `bar_diagnostics` source='live_gate' analysis): ≥90% must be resolved-by-stack;
any event in a supposedly-fixed class = live bug, fix tonight.
**T3 — pre-market live (08:00Z+):** telemetry last-writer acceptance ≥99%; [PRIMARY-RESYNC]
APPLIED lines healing real drift; `src/_measure_five.py --sids ... --since ... --until ...`
(rolling paired-%; run from src/ with .venv python; alerts pairing = side entry/exit ×
data->>'entry_fill_ts'/'exit_fill_ts').
**Gotchas for the loop:** standalone scripts need suite-main() env (admin ctx + pack registry +
USE_DB — bug-hunt skill SOP); local lane writes need prod RORT flag mirror (local-update skill
hard rule 0); canaries during live sessions need OFF/ON/OFF sandwich; health page = use 1h/3h
window buttons (24h drags today's pre-fix history); freeze deploys 13:15-13:45Z (RTH open);
real-money sids 308-314 are 15Sec RTH — no deploys during their live trading without checking.

**Fresh-session note:** this file + `Gated_Five_Divergence_Autopsy.md` (with corrections) +
memory (`project_mrs2_phase2_resampled_store`, `project_mtf_publish_clobber`,
`feedback_pb_boundary_semantics_ruling`,
`feedback_trading_target_90pct_gated_focus`) = the complete state. WS-tip clarification: live
gating is INSTANT (WS closes); the store is the SETTLED layer (~15min = settle physics, same as
bar_cache) — congruent with the primary's multi-stage treatment by design.
