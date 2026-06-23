---
name: bug-hunt
description: Autonomous batch divergence bug-hunt for RoR Trader. Scans Strategy Health for low paired-% strategies, diagnoses live↔backtest divergences, fixes + validates (byte-identical / kill-switched), publishes (reversibility-gated), monitors for regressions, and loops until all monitored strategies are ≥95% paired. Use when the user wants to hunt/squash divergence bugs, drive paired-% up, or run a batch fidelity pass. Two modes — check-in (default) and aggressive.
---

# RoR Trader — Bug-Hunt Loop (divergence hunting)

## North stars (every fix serves these)
1. **Accurate** — backtest and live produce the same trades (paired-% high). Fidelity is paramount; never trade speed for fidelity.
2. **Fast** — Update-New stays quick enough to run periodically / on a cron.
3. **Scalable** — fix the *class* of bug, not just the instance, so new user packs or new timeframes don't reintroduce it.

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
2. **Tests pass** — relevant suite (e.g. `test_unified_parity.py`, run from `src/`).
3. **Local repro** — the fix is shown locally to actually resolve the diagnosed divergence.
4. **Kill-switch** present (default OFF until validated) for any fidelity-touching change.

## The loop
1. **SCAN** — read **combined %** for all monitored strategies using the SAME method as the `/admin/strategy-health` dashboard: `api/routers/strategy_health.py` (`GET /api/admin/strategy-health`). This is a **2-way alert↔backtest match** (NOT the archived `compute_three_way_divergence`): per strategy, over a `window_hours` window (default 24h), collect backtest **trade edges** (entry_fill_ts + exit_fill_ts) and **alert fill_ts**, **greedy 1:1 pair within ±tolerance**. `paired` = matches; **Phantom** = unpaired alert (alert-only); **Missed** = unpaired edge (backtest-only) — Kevin's locked terms. combined % = paired / (paired + phantom + missed). The dashboard tolerance is ±60s (`_DIVERGENCE_TOLERANCE_SEC`); compute also at **5s and 10s** (Kevin's stricter target). Rank by lowest combined % @5s. Also pull the **by-hour** and **by-deploy** views (`useStrategyHealthByHour` / `...ByDeploy`) to see if divergence is deploy-window noise vs spread (real). Prefer calling the live endpoint (auth) for screen-faithful numbers; replicate the greedy pairing for 5s/10s.
2. **DOCUMENT** — append each candidate to the standard record `docs/_active/Divergence_Hunt_Log.md`: sid, symbol/tf, symptom, paired-%, Phantom/Missed, and a **THEORY** (root-cause hypothesis). This is the durable record.
3. **DIAGNOSE** — pin the root cause. Deploy a targeted diagnostic log if needed (pattern: `GATE_BLOCK_DIAG` / `XTF_BLOCK_DIAG` — log needed vs present state when an entry is suppressed). FIRST rule out **stale-lane noise**: if the lane is stale, run Update-New and re-check before calling it a bug. Update the record with the confirmed cause.
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

## Safety rails (always, both modes)
- Never push while a needed update-job is mid-run (orphan risk) — cancel-if-long or wait.
- Never auto-publish an irreversible / data-deleting change — check in.
- Backup branch before a risky batch / milestone.
- Report deploy times; filter divergence around deploys (don't chase deploy-window noise).
- **Report faithfully:** if a fix didn't fully work, say so with evidence; never claim ≥95% without the dashboard confirming it.

## Invocation
- `/bug-hunt` → **Mode 1** (scan + diagnose, then check in) over all monitored strategies.
- `/bug-hunt aggressive` → **Mode 2** (reversibility-gated auto-publish), loop to ≥95% paired.
- Optional args: a strategy list and/or threshold to scope a single run; a WIP-exclude list.

## First-run setup notes (verify once, then bake in)
- Confirm the exact source of the dashboard **"combined %"** (5s + 10s tolerances; likely `parity_status` JSON + `discrepancies` on the strategy row, or `compute_three_way_divergence`). Wire SCAN to read both tolerances directly, plus the by-hour / by-deploy breakdowns.
- Confirm `docs/_active/Divergence_Hunt_Log.md` is the agreed standard record (create it on first run; cross-link `Roadmap_Divergence_Hunting.md`).
