# Morning Handoff — 2026-07-16 RTH open

**Written:** 2026-07-15 ~22:40Z (16:40 MT) by Claude, end of the tonight-arm /loop.
**Goal for today:** a clean RTH session where live-vs-backtest can actually be measured at
≥90%, with all tested fixes already in place from the first bar.

**This file is UNCOMMITTED** (writing it avoided a 5th Worker redeploy during EH). Commit it
with today's work, or just read it in place.

---

## (a) Flags left ON overnight — and why we trust them

Everything armed is flag-gated + reversible in one command. Two NEW flags armed tonight, on
top of the existing armed stack.

### NEW tonight
- **`RORT_CANONICAL_PRIMARY_CLOSE=1`** (P0+P1, commit `206002b`) — a default-model
  (`ws_rest_spliced`) `>=60s` primary was getting no canonical dispatch and deciding on an
  incomplete flush partial every minute. Now single-sourced eligibility + flush no longer
  drives `>=60s` closes on partials. **Blast radius: 8 live 1Min+ default-model strategies:
  340, 265, 269, 194, 174, 136, 266, 7** (incl. **340 — a member of the Five, which IS 1Min,
  not sub-minute**). Trust: flag-OFF proven byte-identical to baseline (stash-verified ×2);
  Brandon's 2 live-path tests pass flag-ON; 10 companion tests; zero new regressions; Worker
  came up healthy armed. Armed during dead EH so there's **no redeploy bar-gap at the open**.
- **`RORT_GRACE_FINAL_CLOSE_ELIGIBLE=1`** (P1-grace, commit `e758e31`) — sub-minute grace was
  marking a bucket "fired" even when it emitted no signal, suppressing a trigger that only
  becomes true in the final seconds. Now marks fired only on a real dispatch. **Narrow:** only
  sub-minute `ws_rest_spliced`/`ws_agg_reconciled` strategies in that edge case. Trust: same
  test rigor (Brandon test flag-ON + 4 companion tests; flag-OFF==baseline; no regressions).

### Pre-existing armed stack (unchanged, verified intact)
`RORT_WARMUP_PREV_CHAIN=1`, `RORT_MTF_FINE_INCREMENTAL_AUTHORITY=1`,
`RORT_SHADOW_RETRUE_FORCE_FULL=1`, `RORT_MTF_PB_DEFER=1`, `RORT_GATE_FAIL_CLOSED=1`,
`RORT_PRIMARY_STATE_RESYNC_S=0`, `RORT_PRIMARY_STATE_RESYNC_APPLY=0`.

## (b) Built-but-OFF / not-built + flip commands

- **P2-healthcheck (Brandon P2) — NOT built tonight, deferred to today.** On inspection it
  gates Docker *restarts* (a bad bounded-boot-deadline = restart loop), it's edge-case
  hardening (engine *never* starts — rare; the common *stall* case is already covered by the
  shipped heartbeat-age-out), so it's a daylight build, not a pre-clean-day rush. Design ready:
  worker writes a **write-once boot marker** at startup + `engine_health.check` grants
  **bounded** boot grace from that marker's age (flag `RORT_HEALTHCHECK_BOUNDED_BOOT`), so a
  fresh manager file can't grant unlimited grace. Brandon's `test_engine_never_started_boot_
  grace_is_bounded` is the acceptance gate.
- **P1-hot-reload (Brandon P1) — NOT built, today's daylight build.** `db_hot_reload` uses flat
  7-day warmup + skips `finalize_shadow_engines`, so a hot-added/edited strategy differs from a
  clean boot. Only bites on config hot-reload (didn't happen overnight; a fresh morning boot is
  clean), hence low urgency.

**Flip-OFF (if the open shows a problem):**
```
railway variables --service Worker --set RORT_CANONICAL_PRIMARY_CLOSE=0     # P0/P1
railway variables --service Worker --set RORT_GRACE_FINAL_CLOSE_ELIGIBLE=0  # grace
```
Backup branch if a code revert is ever needed: `backup/pre-p0p1-20260715`.

## (c) Anomalies tonight — how handled

- **None trading-affecting.** 4 Worker redeploys (P0/P1 push, P0/P1 arm, grace push, grace arm)
  each restarted the engine → normal `always-start-flat` carryover warnings (e.g. sid 276/278
  SPY-10s were mid-EH) and gap-heal backfill of the restart window ("no alerts re-fired").
  Benign — trading is dormant, so EH position churn costs nothing. Verified healthy after every
  deploy (SUCCESS + engine warming/gap-healing, no Traceback/stall/lag/crash).
- Stopped redeploying after the grace arm to stop churning EH state.

## (d) TOMORROW (today) plan — ordered

1. **At/just before RTH open (13:30Z):** confirm the Worker is healthy and both flags still set
   (`railway variables --service Worker --kv | grep RORT`). No pre-open redeploy needed.
2. **First ~30 min of RTH — watch the P0/P1 canaries: 265 (TSLA-1m-Control), 269 (SPY-1m-
   Control), 266 (TSLA-5m-Control), and 340.** THE key risk to watch: a `>=60s` strategy going
   **SILENT** (flush-close removed → must fire via the canonical fan-out). Brandon's P1 test
   proves the fan-out close fires, but verify LIVE that these strategies still produce entries
   and dispatch on time (alert-lag normal 3-8s). If any goes silent or over-fires → flip
   `RORT_CANONICAL_PRIMARY_CLOSE=0` and investigate.
3. **Definitive forward-check** (the `/replay-check` skill) vs the 00:20Z **nightly-settled**
   lane, on this higher-activity RTH day: measure DASHBOARD-live vs backtest + ceiling for the
   Five (327/328/329/333/340) and the 8 default-model 1Min+ strategies. Today's live should now
   track the ceiling if the fixes work. (Recall 308 already did: 89.5% ≈ 93.3% ceiling.)
4. **If clean, the armed fixes are validated live.** Keep ON. If a specific fix looks bad,
   bisect by flipping one flag.
5. **Build P2-healthcheck + P1-hot-reload** (designs above), each flag-gated + Brandon-test-
   gated + flag-off==baseline, ship flag-OFF, arm with monitoring.
6. **Fix the smoke-canary limitation** — `local_update.py`'s canary 321 is live-appended so its
   smoke gate races even post-close (aborted the fresh-recompute forward-check tonight). Set/
   choose a fixed-window NON-live-trading user-pack canary so `--mode all` works for the
   definitive forward-check.
7. **Goal check:** the Five at ≥90% live @≤10s on the settled window. If 328 is the straggler,
   its residual is the characterized ~1-trade/day (session-open convergence + rapid re-entry) —
   a fine-tuning item, not a systemic bug.

## Context pointers
- Diagnosis SSOT: three investigations agree divergence is operational/plumbing, not logic.
- `docs/_active/Plan_P0_P1_Primary_Close_Dispatch.md` — the shipped fix's design.
- `docs/audits/Current_Dev_Candle_Parity_Audit_2026-07-15.md` — Brandon's audit (P0-P2 + more).
- `docs/_active/Plan_Measurement_Trust.md` — the harness/measurement methodology.
- Memory: `project_brandon_candle_parity_audit`, `project_replay_harness`.
