---
name: bug-hunt
description: Autonomous batch divergence bug-hunt for RoR Trader. Scans Strategy Health for low paired-% strategies, diagnoses live↔backtest divergences, fixes + validates (byte-identical / kill-switched), publishes (reversibility-gated), monitors for regressions, and loops until all monitored GATED strategies are ≥90% paired (@≤10s, recent-activation window — Kevin's tradable bar). Use when the user wants to hunt/squash divergence bugs, drive paired-% up, or run a batch fidelity pass. Three modes — check-in (default), aggressive, and nightly (unattended daily routine: waits for the nightly recompute to settle, then hunts + auto-arms the fixes it can rigorously prove behind kill-switch flags + writes a morning brief).
---

# RoR Trader — Bug-Hunt Loop (divergence hunting)

## North stars (every fix serves these)
1. **Accurate** — backtest and live produce the same trades (paired-% high). Fidelity is paramount; never trade speed for fidelity. **BUT: "same trades" ≠ "backtest is truth."** When the lanes disagree, you MUST arbitrate *which lane is correct* before fixing (see Three-Lane Arbitration). Fixing live to match a broken backtest corrupts the correct lane — the worst outcome.
2. **Fast** — Update-New stays quick enough to run periodically / on a cron.
3. **Scalable** — fix the *class* of bug, not just the instance, so new user packs or new timeframes don't reintroduce it. **After a class fix, run a fleet-wide audit** to confirm the class is contained (find every instance of the signature) and add a **tripwire** (a warn/log when the class signature reappears) so a future strategy can't reintroduce it silently.

## ⚖️ THREE-LANE ARBITRATION — which lane is CORRECT? (the 2026-07-09 lesson)
The system has THREE lanes, not two. Use all three to decide which is *right* — do NOT assume backtest is ground truth (it silently over/under-counted real gates; live was the correct lane).
- **backtest** (`trades.data_source='backtest_rest_hifi'`): full REST hi-fi settled recompute.
- **algo** (`trades.data_source='cache_cache_locked'`): the engine's DECISION-TIME computation on cache-locked bars — deterministic replica of what live decided. NOT always stored; COMPUTE it offline where absent (same engine path, algo/cache-locked model).
- **live** (`alerts` table, `side='entry'`, `fill_ts`): real-time WS dispatch.

**Arbitration rules (and the trap):**
- **algo ≈ live, both ≠ backtest** → decision-time-vs-settled-bar = **WS-vs-REST TIMING FLOOR** (irreducible; live/algo decide on the non-final bar, backtest on the settled bar). Do NOT "fix" this — NOTE it. Confirm a claimed floor THIS way; never hand-wave "it's WS-vs-REST" without the algo check.
- **algo ≈ backtest, live ≠ both** → USUALLY the live real-time path is the bug — **BUT NOT ALWAYS.** The algo lane runs the *same offline engine* as backtest, so if the bug is in the shared offline path (e.g. a gate the offline engine silently drops), **algo≈backtest is the SHARED bug, not proof live is wrong.** Ask: was the strategy configured to do what LIVE does? If yes and the offline lanes ignore that config → **backtest+algo are OVER/UNDER-counting; LIVE is correct.** (329: offline silently dropped a `1M` gate the config declared; live enforced it; backtest's stored KPIs were invalid.)
- When unsure which lane is right, ask "which lane honors the strategy AS CONFIGURED?" — that lane is truth; the divergence is the other lane's bug.

## Goal / done criterion
Metric = the **"combined %"** on the Strategy Health dashboard, at a **tolerance** (±5s / ±10s alignment between live and backtest fills), measured in a **RECENT window from the most-recent-activation/deploy — NOT a multi-day window.** A wide window drags in stale + deploy-transition noise and *understates* a strategy: 2026-07-10 a 3-day scan showed 94% where the page's recent-3h read was 100%, which led to a wrong "~90% floor" conclusion. Always read the recent post-activation window (and cross-check the live page's own recent number).
- **Primary target: combined % ≥90% at the ±10-SECOND tolerance** for every monitored strategy, from its most-recent-activation window, without regressing any healthy strategy. **90%@≤10s = TRADABLE** (Kevin's bar, 2026-07-10) — perfect it later; do NOT keep grinding a strategy already ≥90%@10s. The old ≥95%@5s is an aspiration, not the gate.
- **Prioritize GATED strategies.** Strategies with real confluence gates (→ real profit factor) are the ones worth trading AND the divergent ones. Non-gated strategies can be 95-100% parity but are economically useless (thin/negative PF, "gates" that are just always-true in-session flags) — do NOT spend the loop chasing their last few %.
- **WS-vs-REST tip disagreement is accepted** ("they'll disagree on the tip, that's fine") — small in a clean window; confirm any *claimed* floor via Three-Lane Arbitration, then NOTE and move on.
- Stop when: all monitored **gated** strategies ≥90%@10s from their activation window, OR no further visible bugs / no progress (then report). Honor an explicit WIP exclusion list if named.
- Use the dashboard's **by-hour** and **by-deploy** breakdowns to isolate divergence — filter out deploy-window noise (a strategy can dip right after a deploy and recover; don't chase that).
- **Structural cure over whack-a-mole:** the dominant divergence class is gate-CONSTRUCTION (live vs backtest build the same secondary/coarse gate bar differently — warmup depth, gap-counting, session-keying). The M-RS2-P2 resampled bar store retires this class by making both lanes read ONE canonical bar; once it serves, residual divergence = real logic bugs. See [[project_mrs2_phase2_resampled_store]] and [[feedback_trading_target_90pct_gated_focus]].

## 🔬 THE SETTLE-CONFOUND TRAP — read this before comparing ANY two days (the 2026-07-27 lesson)

**Never compare a T+0 (today, unsettled) reading against a prior day's reading.** The dashboard's number for today is computed from the intraday APPEND lane; the number for any prior day is computed from the nightly SETTLE (a full delete→reinsert recompute over finalized bars, 00:20Z). Those are two different computations over two different bar states. Today will ALWAYS read worse. This is not a regression — it is the instrument.

2026-07-27 cost an entire trading day to this. Kevin correctly built matched 6h windows (same clock positions, Friday vs Monday) and today still read ~10pt lower fleet-wide. Two agents and M produced three confident, wrong explanations before the right test was run. sid 321 read **69.1% persisted → 94.8% settled** — a 25-point artifact.

### The correct instrument: score ONE recompute against EACH day's live alerts
Run a single full recompute per strategy — same code, same flags, same moment — then score that one reference against each day's `alerts` separately. The settle confound cancels because every day is measured against an identically-produced backtest reference.

```
sid 321        PERSISTED (dashboard)   SETTLED (what tonight will show)
WED 07-22      100.0%   93/93          97.9%  92/94
THU 07-23       96.5%  111/115         91.5% 108/118
FRI 07-24       96.2%  102/106         94.4% 101/107
MON 07-27       69.1%   94/136         94.8% 109/115   <- artifact, not regression
```
Read the SETTLED column across days. If today lands inside the prior days' band, there is no regression — full stop, do not go hunting.

### Corollaries
- **"The lane has rows / TBD=0 / a recompute reproduces it" proves COMPLETENESS, not CORRECTNESS.** The nightly settle exists *because* intraday appends produce errors. A complete-looking lane can still be wrong.
- **Do not use `trades.created_at` to date-stamp coverage.** The settle rewrites rows, so `created_at` is a REWRITE stamp — every prior day's rows carry last night's timestamp. It cannot distinguish "written live" from "written nightly."
- **Scope by `data_source` or the counts lie.** `cache_cache_locked` (algo lane) is not always stored and is typically absent at T+0; `backtest_rest_hifi` is. Counting rows without scoping compares two lanes against one and manufactures a fake coverage gap.
- **Equal alert VOLUME proves nothing.** A timing shift moves *which* bars fire while preserving *how many*. Compare edge timestamps, not counts.
- **Fastest settle path is the UI button, not a local script.** `/admin/update-jobs` (or the "Update all" button on the Strategies page, `mode:'all'`) runs the same recompute as the nightly, on Railway, with prod's flags and prod's dependencies **by construction**. `local_update.py` must hand-mirror the environment and once silently drifted (2026-07-27: 6 of 26 flags mirrored, pandas 2.3.3 vs prod 3.0.3, smoke test green through all of it). As of board #161 that drift now **fails closed** — `local_update.py` asserts flag parity (all 26 batch-worker RORT_ flags) and dependency parity (pandas/numpy/pyarrow major versions) before any write, and its smoke-test docstring no longer over-promises. But the manifest is a captured prod snapshot that can go stale vs Railway, and the button carries zero mirror-maintenance burden — so still **prefer the button** for settles; reach for the local script only for speed when you've confirmed the gates pass.

## Modes
- **Mode 1 — check-in (DEFAULT):** run SCAN + DIAGNOSE, then STOP and present findings + proposed fixes for the user's pushback before changing anything.
- **Mode 2 — aggressive:** run the full loop autonomously, publishing fixes that pass the **validation gates** AND clear the **reversibility gate**, looping to the done criterion. Still checks in for irreversible changes (see gate).

## ⚖️ Reversibility gate — may Mode 2 publish on its own?
PUBLISH AUTONOMOUSLY only if ALL hold:
- **Code-only** and a **clean `git revert <sha>`** away from the current state.
- Ideally behind a **kill-switch env flag** (instant rollback, no redeploy).
- **Deletes/overwrites NO data** that can't be rebuilt — no lane wipes, no destructive DB writes, no schema/column/table changes, no Storage deletions.
- Passes the validation gates below.

CHECK IN WITH THE USER FIRST (even in Mode 2) if the change is **irreversible**: deletes/overwrites unrebuildable data, alters schema, or otherwise "can't go back." **When unsure whether a revert cleanly restores → treat as irreversible → check in.**

## ✅ Validation gates — a fix must pass ALL before publish
1. **Byte-identical parity** when the change touches the shared backtest/live prep or engine: prove byte-identical trades/columns vs baseline on affected strategies (same discipline used for the secondary-TF snapshot + gen-gate fix). No byte-identical proof → no auto-publish.
2. **Fidelity parity suite** for ANY cache/warmup/load/engine-path change — run `fidelity_parity_suite.py` yourself (canary 267), 18/18 or "same as baseline." **RUN IT INDEPENDENTLY even if a sub-agent reports it passed** (the 2026-07-06 #46 lesson: merged an "output byte-identical" change without running the gate → runtime regression). If the suite shows FAILs, ISOLATE them against clean `origin/dev` before trusting or blocking — some fails are pre-existing/orthogonal (e.g. a stale SPY/10Sec cache-parity drift), not your change. Don't blindly ship past a fail; don't blindly block on a pre-existing one.
3. **PROVE THE FIX RESOLVES THE DIVERGENCE — and be willing to STOP.** The fix must be shown locally to actually fix it (not just "the code runs"). Build a HARD validation into the fix build: e.g. "the shadow must match backtest at the fire times, else STOP." If validation DISPROVES the diagnosis (the fix is inert / the mechanism is wrong), **do NOT ship — report the disproof.** (2026-07-09: two 329 fixes were built then disproven by their own validation before shipping — shipping an inert flag muddies the picture and wastes a deploy.)
4. **Replay-predict before ship.** Compute the *predicted* post-fix paired-% offline (what backtest/live WOULD produce with the fix) BEFORE deploying — so you know it'll work, not just hope. (No waiting days: the divergence is usually in data you already have.)
5. **Kill-switch** present (default OFF, byte-identical when off) for any fidelity-touching change. Prove flag-OFF byte-identical explicitly.

## 🌙 Nightly unattended mode (Kevin's daily routine) — Mode 3
**Purpose:** Kevin launches this as he leaves for the day; it babysits itself. It **waits for the nightly Update-All to finish**, then runs the aggressive loop, **auto-arms only the fixes it can rigorously prove**, and leaves a **morning brief**. It is **Mode 2 (aggressive) + a front-loaded wait phase + the auto-ARM extension + a time-box**. Run `/bug-hunt nightly` for a single night, or wrap in `/loop` to repeat day-after-day. The two goal tiers: **Goal 1 = every monitored gated strategy ≥90%@10s** (the tradable bar); **Goal 2 = fills within ≤5s** (the ideal). Chase Goal 2 only once Goal 1 holds.

### Phase 0 — WAIT FOR THE NIGHTLY, then verify it's clean (never hunt against a mid-recompute fleet)
The reference the whole hunt trusts is the freshly-settled lane — start only once it's real.
- **Expected window:** the batch-worker `RORT_NIGHTLY_RECOMPUTE` fires **00:20Z**; Kevin may tell you a different/expected time — honor it. The full-fleet recompute then runs for a while.
- **Monitor for COMPLETION, not a clock.** ⚠️ `ScheduleWakeup` is unreliable in this env (dead across multiple sessions) — **drive the wait off a background `Bash` poll** (`run_in_background`, an `until`-loop that exits when the signal trips → its task-notification re-invokes you). NEVER foreground-sleep for hours.
  - **Completion signal (any sufficient; prefer ≥2):** fleet `last_recompute_until_ts` / `data_refreshed_at` (query the strategy_health snapshot or `strategies`) advanced **past today's RTH close**; the shadow `bt_current_through` heartbeat advanced to today; the batch-worker log prints its nightly-complete line. Poll every ~10–15 min.
  - **Verify it didn't ERROR:** confirm the nightly job SUCCEEDED (no partial/aborted recompute) before trusting the lanes — a half-finished recompute is worse than a stale one. If it errored → **diagnose-only, arm NOTHING, hold for Kevin.**
- **Wait guardrails:** **max-wait cap** (~3h past expected) — if the nightly never lands, do NOT treat stale lanes as fresh: drop to **diagnose-only** with a LOUD caveat in the brief, or hold. **RTH safety:** the replay harness loads 1Sec data and can starve the live worker ([[feedback_local_analysis_starves_live_worker]]); the nightly runs post-close so this is naturally satisfied — but ASSERT you're outside RTH before any harness run.

### Time-box
Honor a budget (default ~3h wall-clock of hunting after Phase 0, or a token budget if Kevin passed one). Near the cutoff, STOP cleanly: finish/await any in-flight arm+validate, then write the brief. **Never leave a fix half-armed** (pushed but unproven) at cutoff — if you can't finish proving it, don't arm it.

### Auto-ARM (flip the flag ON so the fix is live next session) — the 6 rails
Normal Mode 2 pushes a fix **flag-OFF** (inert) and leaves arming to a human. Nightly mode **also flips the flag ON** — but ONLY under ALL six rails:
1. **Default-OFF + byte-identical when OFF** already proven (validation gate #5). A wrong fix at OFF = zero change. Non-negotiable.
2. **PROVEN, not plausible.** The fix must *demonstrably* raise the target's paired-% on the freshly-settled/replay data (gates #3 + #4), AND not regress canary 267 or any healthy strategy (re-scan), AND pass 18/18 (gate #2). **If the harness CAN'T prove it — coarse / sub-primary / short-window-fine gate classes, the known secondary-TF gap — do NOT auto-arm: diagnose + hold for Kevin.**
3. **Per-fix kill switch** — each fix its own env flag, one-command revert; record that command in the brief.
4. **Cap the arming:** at most **3 flags/night**, and **never more than ONE thing unproven-by-replay** (ideally zero — rail 2). A bad night must not arm a pile of untested flags.
5. **Shadow-worker fixes need the FINGERPRINT SOP** (`railway up`, not var-set — see Deploy/ops). If you can't complete the fingerprint verification unattended, **hold the shadow-worker arm for Kevin**; auto-arm only api/batch/Worker (var-set) fixes.
6. **Irreversible = never auto-arm** (schema / data-delete — check in). The live-trading pause makes now the safe window to build trust in this routine (armed fixes touch algo/alert lanes, not real money) — do NOT let that make the rails lax.

### Morning brief (the artifact Kevin reads first)
Prepend a dated section to `docs/_active/Divergence_Hunt_Log.md`. Keep it scannable:
- **Fleet metric:** N/M monitored gated ≥90%@10s (Goal 1) and @5s (Goal 2); delta vs last night.
- **Bugs found + CLASSIFIED** (LOGIC / PLUMBING / INFRA — see DIAGNOSE): sid, one-line symptom, class, root cause.
- **ARMED last night:** per fix — flag, sid(s), before→after (predicted + observed paired-%), service(s) armed on, and the **exact one-line revert command**.
- **Held for Kevin:** irreversible / unprovable (harness-gap classes) / ambiguous — with why.
- **Outliers:** any individual trade >30s late even where the aggregate passed.
- **Open / next.**

## The loop
1. **SCAN** — get the **screen-faithful** combined % by calling the dashboard's exact code, NOT a replication and NOT the archived `compute_three_way_divergence`. Call `api.routers.strategy_health.get_strategy_health(user=None, window_hours=N, tolerance_seconds=T)` **offline** — it uses the admin client and queries ALL strategies (the `user` arg is just an auth guard), so it returns the whole fleet with the dashboard's real fields per row: **`combined_pct`** (THE metric), `phantom_count` / `missed_count` / `paired_count`, the lag-excluded **`*_fair`** / **`*_cov`** variants, and **`tbd_count`**. (The live endpoint is user-scoped — the QA JWT ≠ Kevin's user — so the offline function call is the reliable path.) It's a **2-way alert↔backtest match** (greedy 1:1 pair of alert `fill_ts` to backtest trade edges within ±tolerance). **Phantom** = alert-only, **Missed** = backtest-only (Kevin's locked terms). Dashboard default ±60s; also compute **5s and 10s** (Kevin's target) by passing `tolerance_seconds`. Rank by lowest `combined_pct` @5s.
   - **TBD ≠ phantom:** alerts that fired but whose Update-New hasn't populated the backtest counterpart yet land in `tbd_count` — do NOT count them as phantom. Use the `*_fair`/`*_cov` counts (lag-excluded) for the real divergence.
   - **Strip deploy-transition noise:** every deploy/Update-All churns the window. Use a **custom window starting AFTER the most recent deploy** (`get_strategy_health(start=<post-deploy ISO>, end=<now>)`) and/or the **by-deploy** view. If today has many deploys, today's aggregate is unreliable — prefer a clean no-deploy stretch (or the drill in step 3).
2. **DOCUMENT** — append each candidate to the standard record `docs/_active/Divergence_Hunt_Log.md`: sid, symbol/tf, symptom, combined %, Phantom/Missed/TBD, and a **THEORY**. This is the durable, long-standing record (update it over time so we can see how a strategy's divergence shifts run-to-run — the "four-tier" standing-doc idea).
3. **DIAGNOSE** — pin the root cause. Complementary tools:
   - **⭐ REPLAY HARNESS (`src/replay_harness.py`, built 2026-07-14; memory `project_replay_harness`)** — the FASTEST way to separate a real logic bug from an operational loss, in MINUTES instead of an overnight soak. Replays recorded DECISION-TIME bars (`live_bars.first_*`) through the REAL engine under the ARMED flag stack, and prints, per sid at 5s/10s/60s: **BACKTEST** (reference), **LIVE** (what happened), **REPLAY** (the achievable ceiling with no stalls/lost alerts), plus a **REPLAY-vs-LIVE @10s** self-check. Read it as: `REPLAY≈BACKTEST` but `LIVE<REPLAY` ⇒ **operational** (stalls/lost alerts — not a logic bug); `REPLAY<BACKTEST` ⇒ a **real remaining bug** the engine reproduces offline (now debuggable with `RH_DEBUG=1` per-bar traces, no market wait). **TRUST GATE (do this before believing its ceiling numbers): on a CLEAN, un-stalled day confirm `REPLAY vs LIVE @10s ≥ ~90%`** — that agreement is what validates the harness as faithful; until it's been shown once, treat the ceiling as a strong hypothesis, not proof. Run from `src/`, OUTSIDE RTH (it loads 1Sec data — see `feedback_local_analysis_starves_live_worker`). The 7 fidelity rules that make it faithful are in its header + the memory (bar_count must increment; feed 1Sec `on_tick` or exits fake-diverge; strict-`>` refresh; no look-ahead; mirror armed flags; …).
   - **Targeted diagnostic log** (pattern: `GATE_BLOCK_DIAG` / `XTF_BLOCK_DIAG` — log needed-vs-present state when an entry is suppressed). FIRST rule out **stale-lane noise** (run Update-New + re-check before calling it a bug).
   - **⭐ TRADE-BY-TRADE DRILL (don't trust only the aggregate %):** open the **Strategy Detail page** (or load the alert lane + backtest lane and walk them in time order) and compare trades **one by one** to find *where* it falls off. Look for: fills off by a few–10s; **gaps** (a run where live just stops pairing); **regime patterns** ("7 trades in a row way off, then 5–6 not firing, then it snaps back to accurate"). The *shape* of the divergence (which trades, which window, what magnitude) usually reveals the cause faster than the headline %. **Pull the ALGO lane too** (three-lane, above) — it arbitrates floor-vs-live-bug-vs-backtest-bug. **The Gate Parity panel** (Strategy Detail) shows per-gate open% and the divergent gate directly — use it; it was fixed 2026-07-09 to read the trigger from `entry_trigger_confluence_id` and iterate ALL gates.
   - **Gate-class specifics (recurring):** most divergences are CONFLUENCE-GATE bugs where live and backtest construct/serve a secondary-TF gate state via different paths. Known sub-classes: (a) COARSE-gate session contamination — the live coarse (≥1H) shadow eats session-unfiltered after-hours buckets and carries a stale state across the overnight boundary (fix: `RORT_MTF_COARSE_RTH_RELOAD`, reload session-correct); (b) interp-blind / incidental-coverage topology (a real monitor "covers" an interp only incidentally → no shadow / wrong-fidelity own_records); (c) TF-LABEL silent-drop (`'1M'` uppercase = primary sentinel offline but 1-minute live → gate silently dropped; fix: `RORT_ENFORCE_1MIN_GATE` primary-aware); (d) secondary-state FREEZE (shadow record stops updating; fix: state refresher). See `Design_Gate_Fidelity_Hardening.md` for the structural end-state (single canonical bar/gate source) that retires the whole class.
   - **CLASSIFY every confirmed bug (Kevin's taxonomy) — from the replay / three-lane triangulation:** **LOGIC** (`REPLAY<BACKTEST` / corrected-ceiling low — engine reproduces the miss offline → a code fix); **PLUMBING** (`LIVE<REPLAY` — stalls, dispatch lag, lost alerts, and any individual trade >30s late even when the aggregate passes → operational/routing fix); **INFRA/FLOOR** (`algo≈live≠backtest` / decision-time<corrected only — WS≠REST, irreducible without infra like the canonical bar store → NOTE, don't "fix"). Put the class in the record + the brief.
   - **Hunt OUTLIERS, not just the aggregate:** even at ≥90%, walk the trade-by-trade drill for individual fills >30s late. A handful of 30s-late fills in an otherwise-passing strategy is a PLUMBING signal worth a brief line — don't let the headline % hide them.
   Update the record with the confirmed cause.
4. **BATCH FIX** — implement the confirmed batch (multiple at once is fine and more efficient than one-at-a-time, given the publish→UND→validate cycle). Kill-switch any fidelity-touching change.
5. **VALIDATE** — run the validation gates on the batch.
6. **PUBLISH (reversibility-gated)**:
   a. Check for a running update-job. A `git push` redeploys the api → **orphans a running job**. If a job is running and likely to take **>~10 min** (long full-recompute), **cancel it**; if short / nearly done, **wait**.
   b. Push to dev — auto for reversible fixes in Mode 2; check in first for irreversible. **Log the deploy time** (for divergence-noise filtering).
   c. After deploy, run/await the appropriate **Update-New** so lanes reflect the fix (avoid Update-All unless seeding/source-of-truth requires it).
7. **MONITOR** — confirm the fix took (live + lane) AND that **no previously-healthy strategy regressed** (re-scan their paired-%). Any regression → **revert immediately** (kill-switch or `git revert`).
8. **RE-SCAN + LOOP** — back to step 1. Continue to the done criterion or no-progress.

## User-pack changes (special handling — scalability goal)
If a fix needs a user-pack change (pack not firing / misfiring / bars/indicators moving wrong):
- Allowed — but **document it** (rationale + before/after) and **verify the pack's bars/indicators move correctly** after.
- **Update BOTH:** (a) the AI **user-pack-creation prompt documentation**, and (b) the **new-user-pack validation checks** — so the same class of bug can't be reintroduced by a future pack. Fix the class, not just the instance.

## Deploy / ops SOPs (RoR-specific — learned the hard way)
- **`dev` merge auto-deploys api + Worker + batch-worker + frontend** (flag-OFF changes are inert/byte-identical, safe to merge). **`shadow-worker` deploys ONLY via `railway up`** from a verified checkout — a var-set on it reverts to a stale pinned image. **FINGERPRINT SOP:** before arming any shadow-worker flag, compute `cat src/shadow_worker.py src/shadow_manager.py src/services.py | sha256sum | cut -c1-12` in the UPLOAD tree and confirm the boot log prints the SAME `code fingerprint=` — only then arm. (2026-07-07: a wrong-tree `railway up` ran stale code for ~90 min undetected.)
- **Backup branch before every merge** (`backup/dev-pre-<name>-<date>`).
- **Which services need a flag armed:** offline-lane fixes (data_loader/strategy_data/services/data_worker_engine) → arm on **api + batch-worker + shadow-worker**; live-engine fixes (ralph_engine) → arm on **Worker**. The LIVE engine is often already correct — it's the offline lanes that need to catch up.
- **Confirm the fix live by DB effect, not log tails** (`railway logs` truncates ~500 lines — a 5-min-old line is gone; query the gate telemetry / trade counts / heartbeats instead, or pull a narrow deployment-log time window).
- **Standalone engine-script environment (2026-07-13, cost 3h of false alarms):** ANY local
  script that drives the engine (canaries, replays, measures) must replicate
  `fidelity_parity_suite.main()` setup — `load_dotenv('.env')`, `USE_DB=true`,
  `set_admin_user_context(ADMIN)` (else user-context config loads run as user_id=None and
  postgrest `maybe_single()` raises APIError 204 — looks exactly like the known transient
  Supabase 204s but is deterministic), and `pack_registry.scan_and_load_all()` (else packs=0
  → silent 0-trades; always assert n_trades > 0). Run from `src/` with ABSOLUTE venv paths
  (background shells reset cwd).
- **Canary during a LIVE session (RTH, or Extended before 00:00Z): use an OFF→ON→OFF
  sandwich.** The strategy window is now-anchored; legs minutes apart shift ~1 trade at the
  moving edge and read as a false flag-diff. OFF1^OFF2 ≠ 0 ⇒ edge noise (compare OFF1^ON
  instead); OFF1==OFF2 but ON differs ⇒ the flag is genuinely implicated.

## Safety rails (always, both modes)
- Never push while a needed update-job is mid-run (orphan risk) — cancel-if-long or wait. (Enqueuing a manual recompute to PROVE a fix end-to-end is a good pattern — e.g. recompute the fixed sid and confirm its lane count changes as replay-predicted.)
- Never auto-publish an irreversible / data-deleting change — check in.
- Backup branch before a risky batch / milestone.
- Report deploy times; filter divergence around deploys (don't chase deploy-window noise). Use a **post-ARM window** (after the specific fix's arm time) to confirm a fix — the rolling window averages in pre-fix data and hides the win. Beware small-N noise in short windows.
- **Report faithfully:** if a fix didn't fully work, say so with evidence; never claim ≥95% without the dashboard confirming it. If you asserted a conclusion (e.g. "irreducible floor") without the arbitrating check, SAY it's unvalidated and go run the check.

## Invocation
- `/bug-hunt` → **Mode 1** (scan + diagnose, then check in) over all monitored strategies.
- `/bug-hunt aggressive` → **Mode 2** (reversibility-gated auto-publish), loop to the done criterion.
- `/bug-hunt nightly` → **Mode 3** (unattended daily routine): Phase-0 wait-for-nightly → aggressive loop with **auto-ARM** (6 rails) → morning brief. Kevin launches it as he leaves. Optional args: expected nightly time, time-box / token budget, WIP-exclude list. Wrap in `/loop` to repeat day-after-day. See `docs/_active/Nightly_Bughunt_SOP.md`.
- Optional args: a strategy list and/or threshold to scope a single run; a WIP-exclude list.

## First-run setup notes (verify once, then bake in)
- Confirm the exact source of the dashboard **"combined %"** (5s + 10s tolerances; likely `parity_status` JSON + `discrepancies` on the strategy row, or `compute_three_way_divergence`). Wire SCAN to read both tolerances directly, plus the by-hour / by-deploy breakdowns.
- Confirm `docs/_active/Divergence_Hunt_Log.md` is the agreed standard record (create it on first run; cross-link `Roadmap_Divergence_Hunting.md`).
