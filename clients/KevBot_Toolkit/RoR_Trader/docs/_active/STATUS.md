# RoR Trader — STATUS (read me first)

**Updated: 2026-06-18.** This is the single doc to open for "what are we doing and why."
It's the live priority list + current state. Deeper detail lives in the linked docs (kept
in their current locations for now — see Doc Map). New observations get logged here (and in
`Roadmap_Divergence_Hunting.md` for divergence specifics).

## North star
Backtest and live produce the **same trades within ~5s**, so we can **trade reliably**.
Current focus: the first real-money strategies (308–314, TSLA 15Sec).

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
