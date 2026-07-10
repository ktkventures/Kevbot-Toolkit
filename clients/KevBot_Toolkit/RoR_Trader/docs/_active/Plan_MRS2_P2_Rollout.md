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

## Phase 1 — Arm & observe (Fri evening, ~2h extended hours) ✅ armed / 🔄 observing
- First maintain pass on cadence; first `#3-live` verifies (Worker boot warm-ups cover every
  (sym, tf, session) the fleet gates on; extended-hours fine-TF closes add intraday events).
- **00:00Z close check:** maintain captured the settled session close (the edge-staleness lesson:
  an RTH-time maintain leaves a forming last bucket; post-close it must settle).
- What 2h extended CANNOT give: RTH open/close boundary under full live WS load → that is
  Monday's observation, and it gates ONLY the live serve flip (Phase 4), not the weekend work.

## Phase 2 — Weekend sprint (offline; no live data needed — the lanes' history is recorded)
1. **Fleet-wide historical verify replay**: run every gated strategy's REAL engine windows with
   verify armed → GREEN/DRIFT ledger per (sid, tf, session). The "N days of evidence,"
   retroactively.
2. **Fine-TF divergence quantification (327/328/329/333/340 class)**: backtest builds 2m–30m
   secondaries by resampling the PRIMARY; live builds from WS-aggregated bars (`live_bars`
   history, recorded since 2026-04-30). Compare BOTH against the store canonical → quantify
   exactly where/how much each lane's construction diverges. Also: theoretical gate replay vs
   the live engine's recorded per-close gate states (`bar_diagnostics` source=live_gate).
3. **Build + prove the OFFLINE compute-skip** (backtest lane serves settled bars FROM the store,
   skips the resample): deterministic → provable byte-identical fleet-wide on historical data
   this weekend. Arrives Monday proven, not experimental. Kill-switched, separate flag.
4. Triage any DRIFT the replays surface (finding them in a Saturday replay >> Monday's alerts).

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
