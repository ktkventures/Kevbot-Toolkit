---
name: bug-hunt
description: Autonomous batch divergence bug-hunt for RoR Trader. Scans Strategy Health for low paired-% strategies, diagnoses live↔backtest divergences, fixes + validates (byte-identical / kill-switched), publishes (reversibility-gated), monitors for regressions, and loops until all monitored strategies are ≥95% paired. Use when the user wants to hunt/squash divergence bugs, drive paired-% up, or run a batch fidelity pass. Two modes — check-in (default) and aggressive.
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
Metric = the **"combined %"** on the Strategy Health dashboard, at a **tolerance** (5-second / 10-second alignment between live and backtest fills).
- **Primary target: combined % ≥95% at the 5-SECOND tolerance** for every monitored strategy, without regressing any healthy strategy.
- **Acceptable fallback:** if a strategy reaches **≥95% at the 10-second tolerance but not 5s**, AND further investigation reveals no additional root cause → **NOTE it in the record and leave it for Kevin's review** (do NOT block the loop on it; it's likely WebSocket-vs-REST fill timing, not a logic bug).
- Stop when: all monitored ≥95% @5s (or @10s-with-note), OR no further visible bugs / no progress (then report). Honor an explicit WIP exclusion list if named.
- Use the dashboard's **by-hour** and **by-deploy** breakdowns to isolate divergence — filter out deploy-window noise (a strategy can dip right after a deploy and recover; don't chase that).

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

## The loop
1. **SCAN** — get the **screen-faithful** combined % by calling the dashboard's exact code, NOT a replication and NOT the archived `compute_three_way_divergence`. Call `api.routers.strategy_health.get_strategy_health(user=None, window_hours=N, tolerance_seconds=T)` **offline** — it uses the admin client and queries ALL strategies (the `user` arg is just an auth guard), so it returns the whole fleet with the dashboard's real fields per row: **`combined_pct`** (THE metric), `phantom_count` / `missed_count` / `paired_count`, the lag-excluded **`*_fair`** / **`*_cov`** variants, and **`tbd_count`**. (The live endpoint is user-scoped — the QA JWT ≠ Kevin's user — so the offline function call is the reliable path.) It's a **2-way alert↔backtest match** (greedy 1:1 pair of alert `fill_ts` to backtest trade edges within ±tolerance). **Phantom** = alert-only, **Missed** = backtest-only (Kevin's locked terms). Dashboard default ±60s; also compute **5s and 10s** (Kevin's target) by passing `tolerance_seconds`. Rank by lowest `combined_pct` @5s.
   - **TBD ≠ phantom:** alerts that fired but whose Update-New hasn't populated the backtest counterpart yet land in `tbd_count` — do NOT count them as phantom. Use the `*_fair`/`*_cov` counts (lag-excluded) for the real divergence.
   - **Strip deploy-transition noise:** every deploy/Update-All churns the window. Use a **custom window starting AFTER the most recent deploy** (`get_strategy_health(start=<post-deploy ISO>, end=<now>)`) and/or the **by-deploy** view. If today has many deploys, today's aggregate is unreliable — prefer a clean no-deploy stretch (or the drill in step 3).
2. **DOCUMENT** — append each candidate to the standard record `docs/_active/Divergence_Hunt_Log.md`: sid, symbol/tf, symptom, combined %, Phantom/Missed/TBD, and a **THEORY**. This is the durable, long-standing record (update it over time so we can see how a strategy's divergence shifts run-to-run — the "four-tier" standing-doc idea).
3. **DIAGNOSE** — pin the root cause. Two complementary tools:
   - **Targeted diagnostic log** (pattern: `GATE_BLOCK_DIAG` / `XTF_BLOCK_DIAG` — log needed-vs-present state when an entry is suppressed). FIRST rule out **stale-lane noise** (run Update-New + re-check before calling it a bug).
   - **⭐ TRADE-BY-TRADE DRILL (don't trust only the aggregate %):** open the **Strategy Detail page** (or load the alert lane + backtest lane and walk them in time order) and compare trades **one by one** to find *where* it falls off. Look for: fills off by a few–10s; **gaps** (a run where live just stops pairing); **regime patterns** ("7 trades in a row way off, then 5–6 not firing, then it snaps back to accurate"). The *shape* of the divergence (which trades, which window, what magnitude) usually reveals the cause faster than the headline %. **Pull the ALGO lane too** (three-lane, above) — it arbitrates floor-vs-live-bug-vs-backtest-bug. **The Gate Parity panel** (Strategy Detail) shows per-gate open% and the divergent gate directly — use it; it was fixed 2026-07-09 to read the trigger from `entry_trigger_confluence_id` and iterate ALL gates.
   - **Gate-class specifics (recurring):** most divergences are CONFLUENCE-GATE bugs where live and backtest construct/serve a secondary-TF gate state via different paths. Known sub-classes: (a) COARSE-gate session contamination — the live coarse (≥1H) shadow eats session-unfiltered after-hours buckets and carries a stale state across the overnight boundary (fix: `RORT_MTF_COARSE_RTH_RELOAD`, reload session-correct); (b) interp-blind / incidental-coverage topology (a real monitor "covers" an interp only incidentally → no shadow / wrong-fidelity own_records); (c) TF-LABEL silent-drop (`'1M'` uppercase = primary sentinel offline but 1-minute live → gate silently dropped; fix: `RORT_ENFORCE_1MIN_GATE` primary-aware); (d) secondary-state FREEZE (shadow record stops updating; fix: state refresher). See `Design_Gate_Fidelity_Hardening.md` for the structural end-state (single canonical bar/gate source) that retires the whole class.
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

## Safety rails (always, both modes)
- Never push while a needed update-job is mid-run (orphan risk) — cancel-if-long or wait. (Enqueuing a manual recompute to PROVE a fix end-to-end is a good pattern — e.g. recompute the fixed sid and confirm its lane count changes as replay-predicted.)
- Never auto-publish an irreversible / data-deleting change — check in.
- Backup branch before a risky batch / milestone.
- Report deploy times; filter divergence around deploys (don't chase deploy-window noise). Use a **post-ARM window** (after the specific fix's arm time) to confirm a fix — the rolling window averages in pre-fix data and hides the win. Beware small-N noise in short windows.
- **Report faithfully:** if a fix didn't fully work, say so with evidence; never claim ≥95% without the dashboard confirming it. If you asserted a conclusion (e.g. "irreducible floor") without the arbitrating check, SAY it's unvalidated and go run the check.

## Invocation
- `/bug-hunt` → **Mode 1** (scan + diagnose, then check in) over all monitored strategies.
- `/bug-hunt aggressive` → **Mode 2** (reversibility-gated auto-publish), loop to ≥95% paired.
- Optional args: a strategy list and/or threshold to scope a single run; a WIP-exclude list.

## First-run setup notes (verify once, then bake in)
- Confirm the exact source of the dashboard **"combined %"** (5s + 10s tolerances; likely `parity_status` JSON + `discrepancies` on the strategy row, or `compute_three_way_divergence`). Wire SCAN to read both tolerances directly, plus the by-hour / by-deploy breakdowns.
- Confirm `docs/_active/Divergence_Hunt_Log.md` is the agreed standard record (create it on first run; cross-link `Roadmap_Divergence_Hunting.md`).
