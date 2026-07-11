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
