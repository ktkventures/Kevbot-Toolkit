# Plan — M-RS2 Phase 2 Rollout: Resampled Bar Store → Both Lanes Serve From It

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
3. 🔥 **NEW #1 HUNT (found 07-13 ~23:45Z, telemetry-proven): MTF publish CLOBBER.** Multiple
   interp-aware gate shadows (W2-5/PR #38 class) share one `_mtf_confluence[(tf, session)]`
   key and REPLACE-publish it — last writer wins, and the clobbered set persists the whole
   window. Measured 15:41→20:00Z RTH: narrow sets landed LAST in 33/33 tf=180 windows,
   11/11 tf=900 (9/11 = MACD-only, 333's UT_BOT_V4 ABSENT; 1 EMPTY SET — the E-class
   fail-open mechanism, seen in the wild), 43/83 tf=120 (EMA_PP-only, 327's SWING absent),
   7/25 tf=300 (MACD-only, 327/328's SWING absent). Consequence: gates unsatisfiable in
   those windows → fail-closed silent blocks now (tonight's bt-only residuals), fail-open
   phantom fires before 15:41Z. UNIFIES much of autopsy class C + all of class E. NOTE:
   unsetting FAIL_CLOSED does NOT help (non-empty narrow set fails the subset check either
   way). FIX: `_publish_mtf` must MERGE per-interpreter (publisher replaces only records
   whose interpreter it computes), preserving PB effective_from bookkeeping; also covers
   refresher + seed paths (same chokepoint). Likely also the real face of old item 3
   ("900s starvation" = the 15m key being clobbered, not starved).
3b. (superseded by 3) 900s fan-out starvation root-cause — re-examine only if the merge fix
   leaves a residue.
4. Grace-fire `_fired_bucket` suppression hole (ralph 2838/3846) — latent alert-drop, ticket.
5. EOD + next clean day: re-measure the five @90%@10s on post-17:07Z windows.
6. Housekeeping: SPY 10Sec re-true; 3 pre-existing test_unified_parity fails; frontend '1M'→'1m';
   span-hash incremental maintain.
**Fresh-session note:** this file + `Gated_Five_Divergence_Autopsy.md` (with corrections) +
memory (`project_mrs2_phase2_resampled_store`, `feedback_pb_boundary_semantics_ruling`,
`feedback_trading_target_90pct_gated_focus`) = the complete state. WS-tip clarification: live
gating is INSTANT (WS closes); the store is the SETTLED layer (~15min = settle physics, same as
bar_cache) — congruent with the primary's multi-stage treatment by design.
